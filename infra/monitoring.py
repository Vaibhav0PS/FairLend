"""Drift check, alarms and dashboard for the deployed API.

The drift-check Lambda is packaged with the reference profile of the model being
deployed, so it always compares traffic with the right training data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
from constructs import Construct

NAMESPACE = "CreditRisk/Monitoring"
PSI_THRESHOLD = 0.25


def stage_drift_code(project_root: Path, model_dir: Path) -> Path:
    profile = model_dir / "reference_profile.json"
    if not profile.exists():
        raise SystemExit(f"{profile} is missing; train or fetch a model that has a reference profile")
    staging = project_root / "build" / "drift_lambda"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy(project_root / "monitoring" / "drift_check.py", staging)
    shutil.copy(profile, staging)
    return staging


class ApiMonitoring(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        project_root: Path,
        model_dir: Path,
        api_function: lambda_.IFunction,
        api_log_group: logs.ILogGroup,
        bucket: s3.IBucket,
        topic: sns.ITopic,
        drift_alarm_name: str,
    ) -> None:
        super().__init__(scope, construct_id)
        stack = Stack.of(self)

        log_group = logs.LogGroup(
            self, "DriftLogs",
            log_group_name="/aws/lambda/credit-risk-drift-check",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )
        fn = lambda_.Function(
            self, "DriftCheck",
            function_name="credit-risk-drift-check",
            description="Daily PSI drift check of API inputs against the training data",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="drift_check.handler",
            code=lambda_.Code.from_asset(str(stage_drift_code(project_root, model_dir))),
            memory_size=256,
            timeout=Duration.minutes(2),
            log_group=log_group,
            environment={
                "LOG_GROUP": api_log_group.log_group_name,
                "BUCKET": bucket.bucket_name,
                "METRIC_NAMESPACE": NAMESPACE,
                "LOOKBACK_HOURS": "24",
                "MIN_SAMPLES": "50",
            },
        )
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["logs:FilterLogEvents"], resources=[api_log_group.log_group_arn]))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"], resources=["*"],
            conditions={"StringEquals": {"cloudwatch:namespace": NAMESPACE}}))
        bucket.grant_put(fn, "monitoring/*")

        events.Rule(self, "Daily", schedule=events.Schedule.rate(Duration.days(1)),
                    targets=[targets.LambdaFunction(fn)])

        def metric(name: str, statistic: str = "Maximum", **dims) -> cw.Metric:
            return cw.Metric(namespace=NAMESPACE, metric_name=name, statistic=statistic,
                             period=Duration.hours(1), dimensions_map=dims or None)

        drift_alarm = cw.Alarm(
            self, "DriftAlarm",
            alarm_name=drift_alarm_name,
            alarm_description=f"An API input's PSI is above {PSI_THRESHOLD}. Starts the retraining pipeline.",
            metric=metric("MaxFeaturePSI"),
            threshold=PSI_THRESHOLD,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        errors_alarm = cw.Alarm(
            self, "ApiErrorsAlarm",
            alarm_name="credit-risk-api-errors",
            alarm_description="The API Lambda crashed or timed out",
            metric=api_function.metric_errors(period=Duration.minutes(5), statistic="Sum"),
            threshold=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        for alarm in (drift_alarm, errors_alarm):
            alarm.add_alarm_action(cw_actions.SnsAction(topic))

        features = ["person_income", "loan_int_rate", "loan_percent_income", "loan_amnt", "person_age",
                    "person_emp_length", "cb_person_cred_hist_length", "loan_grade", "person_home_ownership",
                    "loan_intent", "cb_person_default_on_file"]
        dashboard = cw.Dashboard(self, "Dashboard", dashboard_name="credit-risk",
                                 default_interval=Duration.days(7))
        dashboard.add_widgets(
            cw.GraphWidget(title="API requests and errors", width=12,
                           left=[api_function.metric_invocations(period=Duration.hours(1))],
                           right=[api_function.metric_errors(period=Duration.hours(1))]),
            cw.GraphWidget(title="API latency (ms)", width=12,
                           left=[api_function.metric_duration(period=Duration.hours(1), statistic="p50"),
                                 api_function.metric_duration(period=Duration.hours(1), statistic="p95")]),
        )
        dashboard.add_widgets(
            cw.GraphWidget(title="Drift: max feature PSI and prediction PSI", width=12,
                           left=[metric("MaxFeaturePSI"), metric("PredictionPSI")],
                           left_annotations=[cw.HorizontalAnnotation(value=PSI_THRESHOLD, label="alarm")]),
            cw.GraphWidget(title="PSI per feature", width=12,
                           left=[metric("FeaturePSI", Feature=f) for f in features]),
        )
        dashboard.add_widgets(
            cw.GraphWidget(title="Predictions checked / high-risk share", width=12,
                           left=[metric("PredictionCount", "Sum")], right=[metric("HighRiskRate")]),
            cw.AlarmStatusWidget(title="Alarms", width=12, alarms=[drift_alarm, errors_alarm]),
        )
        self.drift_function = fn
        self.dashboard_url = (f"https://{stack.region}.console.aws.amazon.com/cloudwatch/home"
                              f"?region={stack.region}#dashboards/dashboard/credit-risk")

"""Retraining pipeline with a manual approval gate (SageMaker Pipelines).

    Train  ->  Evaluate (vs champion)  ->  CandidateBetter?
                                             yes: Register (alias "challenger",
                                                  PendingManualApproval) + approval email
                                             no:  "not better" email, nothing registered

Started by hand (scripts/start_retraining.py) or by the drift alarm.
Nothing here deploys: a person approves with scripts/promote_model.py.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import Stack
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_sns as sns
from constructs import Construct

PIPELINE_NAME = "credit-risk-retraining"
DATA_KEY = "data/raw/credit_risk_dataset.csv"


def get(path: str) -> dict:
    return {"Get": path}


def join(*values) -> dict:
    return {"Std:Join": {"On": "/", "Values": list(values)}}


class RetrainingPipeline(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        project_root: Path,
        bucket: s3.IBucket,
        job_role: iam.Role,
        topic: sns.ITopic,
        mlflow_app_arn: str,
        model_name: str,
        drift_alarm_name: str,
    ) -> None:
        super().__init__(scope, construct_id)
        stack = Stack.of(self)

        # Training image, built by CDK from the current code on every deploy.
        image = ecr_assets.DockerImageAsset(
            self,
            "TrainImage",
            directory=str(project_root),
            file="Dockerfile.train",
            platform=ecr_assets.Platform.LINUX_AMD64,
            # Only code and requirements matter; a newly fetched model must not trigger a rebuild.
            exclude=["build", "artifacts", "static", "monitoring"],
        )
        image.repository.grant_pull(job_role)

        pipeline_arn = stack.format_arn(service="sagemaker", resource="pipeline", resource_name=PIPELINE_NAME)
        processing_job_arn = stack.format_arn(service="sagemaker", resource="processing-job", resource_name="*")

        # Role SageMaker Pipelines uses to start the steps' jobs.
        pipeline_role = iam.Role(
            self,
            "PipelineRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Runs the credit risk retraining pipeline",
        )
        pipeline_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob",
                     "sagemaker:StopProcessingJob", "sagemaker:AddTags", "sagemaker:ListTags"],
            resources=[processing_job_arn],
        ))
        pipeline_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[job_role.role_arn],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
        ))
        bucket.grant_read(pipeline_role, "models/pipeline/*")  # reads evaluation.json for the condition

        bucket_root = f"s3://{bucket.bucket_name}"
        out_root = join(f"{bucket_root}/models/pipeline", get("Execution.PipelineExecutionId"))
        common_env = {
            "AWS_DEFAULT_REGION": stack.region,
            "MLFLOW_TRACKING_URI": mlflow_app_arn,
            "MLFLOW_EXPERIMENT_NAME": "credit-risk",
            "MODEL_NAME": model_name,
            "SNS_TOPIC_ARN": topic.topic_arn,
            "TRIGGER": get("Parameters.Trigger"),
            "PIPELINE_EXECUTION_ARN": get("Execution.PipelineExecutionArn"),
            "IMAGE_URI": image.image_uri,
            "GIT_PYTHON_REFRESH": "quiet",
            "MLFLOW_DISABLE_AGENT_HINT": "1",
        }

        def s3_input(name: str, uri, local: str) -> dict:
            return {"InputName": name, "S3Input": {
                "S3Uri": uri, "LocalPath": local, "S3DataType": "S3Prefix", "S3InputMode": "File"}}

        def s3_output(name: str, local: str) -> dict:
            return {"OutputName": name, "S3Output": {
                "S3Uri": join(out_root, name), "LocalPath": local, "S3UploadMode": "EndOfJob"}}

        def step(name, module, args, inputs, outputs, instance_type, env, max_runtime=3600, **extra) -> dict:
            return {
                "Name": name,
                "Type": "Processing",
                "Arguments": {
                    "ProcessingResources": {"ClusterConfig": {
                        "InstanceCount": 1, "InstanceType": instance_type, "VolumeSizeInGB": 10}},
                    "AppSpecification": {
                        "ImageUri": image.image_uri,
                        "ContainerEntrypoint": ["python", "-m", module],
                        "ContainerArguments": args,
                    },
                    "RoleArn": job_role.role_arn,
                    "ProcessingInputs": inputs,
                    # SageMaker rejects an empty Outputs list, so steps without outputs omit the block.
                    **({"ProcessingOutputConfig": {"Outputs": outputs}} if outputs else {}),
                    "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime},
                    "Environment": {**common_env, **env},
                },
                **extra,
            }

        data_dir = "/opt/ml/processing/input/data"
        data_csv = f"{data_dir}/credit_risk_dataset.csv"
        candidate_uri = get("Steps.Train.ProcessingOutputConfig.Outputs['model'].S3Output.S3Uri")
        evaluation_uri = get("Steps.Evaluate.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri")

        train = step(
            "Train", "credit_risk.train",
            ["--data", data_csv, "--out", "/opt/ml/processing/output/model", "--n-iter", get("Parameters.NIter")],
            [s3_input("data", get("Parameters.DataS3Uri"), data_dir)],
            [s3_output("model", "/opt/ml/processing/output/model")],
            "ml.t3.large",
            {"MLFLOW_REGISTERED_MODEL": "none", "DATA_S3_URI": get("Parameters.DataS3Uri")},
        )
        evaluate = step(
            "Evaluate", "credit_risk.evaluate",
            ["--candidate", "/opt/ml/processing/input/candidate", "--data", data_csv,
             "--out", "/opt/ml/processing/output/evaluation"],
            [s3_input("candidate", candidate_uri, "/opt/ml/processing/input/candidate"),
             s3_input("data", get("Parameters.DataS3Uri"), data_dir)],
            [s3_output("evaluation", "/opt/ml/processing/output/evaluation")],
            "ml.t3.medium",
            {"MIN_PR_AUC_GAIN": get("Parameters.MinPrAucGain")},
            max_runtime=1200,
            PropertyFiles=[{"PropertyFileName": "Evaluation", "OutputName": "evaluation",
                            "FilePath": "evaluation.json"}],
        )
        register = step(
            "Register", "credit_risk.register",
            ["--candidate", "/opt/ml/processing/input/candidate",
             "--evaluation", "/opt/ml/processing/input/evaluation"],
            [s3_input("candidate", candidate_uri, "/opt/ml/processing/input/candidate"),
             s3_input("evaluation", evaluation_uri, "/opt/ml/processing/input/evaluation")],
            [],
            "ml.t3.medium",
            {},
            max_runtime=900,
        )
        condition = {
            "Name": "CandidateBetter",
            "Type": "Condition",
            "Arguments": {
                "Conditions": [{
                    "Type": "Equals",
                    "LeftValue": {"Std:JsonGet": {
                        "PropertyFile": get("Steps.Evaluate.PropertyFiles.Evaluation"),
                        "Path": "candidate_better_int",
                    }},
                    "RightValue": 1,
                }],
                "IfSteps": [register],
                "ElseSteps": [],
            },
        }
        definition = {
            "Version": "2020-12-01",
            "Parameters": [
                {"Name": "DataS3Uri", "Type": "String", "DefaultValue": f"{bucket_root}/{DATA_KEY}"},
                {"Name": "NIter", "Type": "String", "DefaultValue": "60"},
                {"Name": "MinPrAucGain", "Type": "String", "DefaultValue": "0"},
                {"Name": "Trigger", "Type": "String", "DefaultValue": "manual"},
            ],
            "Steps": [train, evaluate, condition],
        }

        self.pipeline = sagemaker.CfnPipeline(
            self,
            "Pipeline",
            pipeline_name=PIPELINE_NAME,
            pipeline_description="Retrain, compare with the champion, register for manual approval",
            role_arn=pipeline_role.role_arn,
            pipeline_definition={"PipelineDefinitionBody": stack.to_json_string(definition)},
        )
        self.pipeline.node.add_dependency(pipeline_role)
        self.pipeline_arn = pipeline_arn

        # Drift alarm -> start the pipeline (the result still waits for approval).
        events_role = iam.Role(self, "StartPipelineRole", assumed_by=iam.ServicePrincipal("events.amazonaws.com"))
        events_role.add_to_policy(iam.PolicyStatement(actions=["sagemaker:StartPipelineExecution"],
                                                      resources=[pipeline_arn]))
        events.CfnRule(
            self,
            "DriftStartsRetraining",
            description="Start retraining when the drift alarm fires",
            event_pattern={
                "source": ["aws.cloudwatch"],
                "detail-type": ["CloudWatch Alarm State Change"],
                "detail": {"alarmName": [drift_alarm_name], "state": {"value": ["ALARM"]}},
            },
            targets=[events.CfnRule.TargetProperty(
                id="RetrainingPipeline",
                arn=pipeline_arn,
                role_arn=events_role.role_arn,
                sage_maker_pipeline_parameters=events.CfnRule.SageMakerPipelineParametersProperty(
                    pipeline_parameter_list=[events.CfnRule.SageMakerPipelineParameterProperty(
                        name="Trigger", value="drift-alarm")],
                ),
            )],
        )

        # Email when a pipeline run fails (success paths send their own email).
        events.Rule(
            self,
            "PipelineFailed",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Building Pipeline Execution Status Change"],
                detail={"pipelineArn": [pipeline_arn], "currentPipelineExecutionStatus": ["Failed"]},
            ),
            targets=[targets.SnsTopic(topic, message=events.RuleTargetInput.from_text(
                f"Credit risk retraining pipeline failed: {events.EventField.from_path('$.detail.pipelineExecutionArn')}. "
                "Open SageMaker AI Studio -> Pipelines to see which step failed."
            ))],
        )

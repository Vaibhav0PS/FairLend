#!/usr/bin/env python3
"""AWS infrastructure for the credit risk MLOps project (AWS CDK, Python).

Stacks:
    CreditRiskMlops    S3, ECR, IAM roles, MLflow App, GitHub OIDC deploy role,
                       alerts topic, retraining pipeline (retraining.py)
    CreditRiskServing  the API on Lambda with a Function URL, drift check,
                       alarms and dashboard (monitoring.py)

Deploy:   cdk deploy --all     (from this folder, with the project venv active)
          cdk deploy CreditRiskServing -c model_src=build/model   (serve a fetched model)
Teardown: cdk destroy --all    (deletes the bucket contents, images and MLflow data too)
"""

import os
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_sns as sns
from constructs import Construct

from monitoring import ApiMonitoring
from retraining import PIPELINE_NAME, RetrainingPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = "credit-risk"
GITHUB_REPO = "Vaibhav0PS/FairLend"
# GitHub's OIDC subject includes the owner and repository IDs (seen in CloudTrail).
# Matching the IDs means a deleted-and-recreated repo with the same name can't deploy.
GITHUB_SUBJECT_REPO = "Vaibhav0PS@184383907/FairLend@1403473338"
MLFLOW_APP_NAME = f"{PROJECT}-mlflow"
REGISTERED_MODEL = f"{PROJECT}-xgboost"
# Fixed names so the two stacks can refer to each other without CloudFormation exports.
ALERTS_TOPIC_NAME = f"{PROJECT}-alerts"
DRIFT_ALARM_NAME = f"{PROJECT}-drift"


def mlops_bucket_name(stack: Stack) -> str:
    return f"{PROJECT}-mlops-{stack.account}-{stack.region}"


class CreditRiskMlopsStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------- Storage: data, model artifacts, MLflow artifacts ----------
        # Versioning keeps every uploaded dataset version; jobs record the version ID they read.
        bucket = s3.Bucket(
            self,
            "MlopsBucket",
            bucket_name=mlops_bucket_name(self),
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    noncurrent_version_expiration=Duration.days(90),
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                )
            ],
        )

        # ---------- Training image registry ----------
        train_repo = ecr.Repository(
            self,
            "TrainRepo",
            repository_name=f"{PROJECT}-train",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=10, description="Keep the last 10 images")],
        )

        mlflow_app_arn = self.format_arn(
            service="sagemaker", resource="mlflow-app", resource_name="*"
        )

        # ---------- Role used by the MLflow App itself ----------
        mlflow_role = iam.Role(
            self,
            "MlflowAppRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Lets the MLflow App store artifacts and sync models to SageMaker Model Registry",
        )
        bucket.grant_read_write(mlflow_role)
        mlflow_role.add_to_policy(iam.PolicyStatement(
            sid="ModelRegistrySync",
            actions=[
                "sagemaker:CreateModelPackageGroup",
                "sagemaker:DescribeModelPackageGroup",
                "sagemaker:CreateModelPackage",
                "sagemaker:UpdateModelPackage",
                "sagemaker:AddTags",
            ],
            resources=[
                self.format_arn(service="sagemaker", resource="model-package-group", resource_name="*"),
                self.format_arn(service="sagemaker", resource="model-package", resource_name="*"),
            ],
        ))
        mlflow_role.add_to_policy(iam.PolicyStatement(
            sid="LineageTracking",
            actions=["sagemaker:CreateAction", "sagemaker:AddAssociation"],
            resources=["*"],
        ))

        mlflow_app = sagemaker.CfnMlflowApp(
            self,
            "MlflowApp",
            name=MLFLOW_APP_NAME,
            artifact_store_uri=bucket.s3_url_for_object("mlflow-artifacts"),
            role_arn=mlflow_role.role_arn,
            model_registration_mode="AutoModelRegistrationEnabled",
        )
        mlflow_app.node.add_dependency(mlflow_role)

        # ---------- Role assumed by SageMaker training / processing jobs ----------
        job_role = iam.Role(
            self,
            "SageMakerJobRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for credit risk SageMaker jobs",
        )
        bucket.grant_read(job_role)
        bucket.grant_write(job_role, "models/*")
        # The MLflow client uploads run artifacts straight to S3 with the job's credentials.
        bucket.grant_write(job_role, "mlflow-artifacts/*")
        train_repo.grant_pull(job_role)
        job_role.add_to_policy(iam.PolicyStatement(
            sid="JobLogs",
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
            resources=[self.format_arn(
                service="logs", resource="log-group", resource_name="/aws/sagemaker/*",
                arn_format=cdk.ArnFormat.COLON_RESOURCE_NAME,
            )],
        ))
        job_role.add_to_policy(iam.PolicyStatement(
            sid="JobMetrics",
            actions=["cloudwatch:PutMetricData"],
            resources=["*"],
            conditions={"StringLike": {"cloudwatch:namespace": "/aws/sagemaker/*"}},
        ))
        job_role.add_to_policy(iam.PolicyStatement(
            sid="UseMlflowApp",
            actions=["sagemaker:CallMlflowAppApi", "sagemaker-mlflow:*", "sagemaker:DescribeMlflowApp"],
            resources=[mlflow_app_arn],
        ))

        # ---------- GitHub Actions: keyless deploys through OIDC ----------
        github_oidc = iam.OidcProviderNative(
            self,
            "GitHubOidc",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
        )
        github_role = iam.Role(
            self,
            "GitHubDeployRole",
            role_name=f"{PROJECT}-github-deploy",
            description=f"Assumed by GitHub Actions in {GITHUB_REPO} (main branch only)",
            max_session_duration=Duration.hours(1),
            assumed_by=iam.WebIdentityPrincipal(
                github_oidc.oidc_provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                        "token.actions.githubusercontent.com:sub": f"repo:{GITHUB_SUBJECT_REPO}:ref:refs/heads/main",
                    }
                },
            ),
        )
        # cdk deploy works by assuming the bootstrap roles; the GitHub role needs nothing else to deploy.
        github_role.add_to_policy(iam.PolicyStatement(
            sid="AssumeCdkBootstrapRoles",
            actions=["sts:AssumeRole"],
            resources=[f"arn:aws:iam::{self.account}:role/cdk-hnb659fds-*-{self.account}-{self.region}"],
        ))
        github_role.add_to_policy(iam.PolicyStatement(
            sid="ReadStackOutputs",
            actions=["cloudformation:DescribeStacks"],
            resources=[
                self.format_arn(service="cloudformation", resource="stack", resource_name=f"{name}/*")
                for name in ("CreditRiskMlops", "CreditRiskServing")
            ],
        ))
        # Download the champion model from MLflow.
        github_role.add_to_policy(iam.PolicyStatement(
            sid="ReadMlflow",
            actions=["sagemaker:CallMlflowAppApi", "sagemaker-mlflow:*", "sagemaker:DescribeMlflowApp"],
            resources=[mlflow_app_arn],
        ))
        bucket.grant_read(github_role, "mlflow-artifacts/*")

        # ---------- Alerts: drift, API errors, approval requests ----------
        # The email subscription is added with scripts/subscribe_alerts.py, not here,
        # so the address stays out of this public repository.
        topic = sns.Topic(self, "Alerts", topic_name=ALERTS_TOPIC_NAME, display_name="Credit risk MLOps")
        topic.add_to_resource_policy(iam.PolicyStatement(
            sid="CloudWatchAlarms",
            principals=[iam.ServicePrincipal("cloudwatch.amazonaws.com")],
            actions=["sns:Publish"],
            resources=[topic.topic_arn],
            conditions={"StringEquals": {"aws:SourceAccount": self.account}},
        ))
        topic.grant_publish(job_role)  # evaluate / register steps send result emails
        job_role.add_to_policy(iam.PolicyStatement(
            sid="SetApprovalStatus",
            actions=["sagemaker:UpdateModelPackage", "sagemaker:DescribeModelPackage"],
            resources=[self.format_arn(service="sagemaker", resource="model-package", resource_name="*")],
        ))

        retraining = RetrainingPipeline(
            self,
            "Retraining",
            project_root=PROJECT_ROOT,
            bucket=bucket,
            job_role=job_role,
            topic=topic,
            mlflow_app_arn=mlflow_app.attr_arn,
            model_name=REGISTERED_MODEL,
            drift_alarm_name=DRIFT_ALARM_NAME,
        )

        # ---------- Outputs used by the scripts/ folder ----------
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "TrainRepoUri", value=train_repo.repository_uri)
        CfnOutput(self, "SageMakerJobRoleArn", value=job_role.role_arn)
        CfnOutput(self, "MlflowAppArn", value=mlflow_app.attr_arn)
        CfnOutput(self, "RegisteredModelName", value=REGISTERED_MODEL)
        CfnOutput(self, "GitHubDeployRoleArn", value=github_role.role_arn)
        CfnOutput(self, "AlertsTopicArn", value=topic.topic_arn)
        CfnOutput(self, "RetrainingPipelineName", value=PIPELINE_NAME)


class CreditRiskServingStack(Stack):
    """The API and web form on Lambda, behind a public Function URL.

    The image is built from the project Dockerfile with the model folder given by
    the `model_src` context value (default: the committed `artifacts/`).
    """

    def __init__(self, scope: Construct, construct_id: str, *, model_src: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        log_group = logs.LogGroup(
            self,
            "ApiLogs",
            log_group_name=f"/aws/lambda/{PROJECT}-api",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        fn = lambda_.DockerImageFunction(
            self,
            "ApiFunction",
            function_name=f"{PROJECT}-api",
            description="Credit Ledger API (FastAPI + XGBoost + SHAP)",
            code=lambda_.DockerImageCode.from_image_asset(
                str(PROJECT_ROOT),
                file="Dockerfile",
                build_args={"MODEL_SRC": model_src},
                platform=ecr_assets.Platform.LINUX_AMD64,
                exclude=["build/drift_lambda"],
            ),
            architecture=lambda_.Architecture.X86_64,
            memory_size=2048,  # more memory also means more CPU, so faster cold starts
            timeout=Duration.seconds(30),
            # Caps cost if the public URL gets hammered: at most 5 requests at once.
            reserved_concurrent_executions=5,
            log_group=log_group,
        )

        url = fn.add_function_url(auth_type=lambda_.FunctionUrlAuthType.NONE)

        monitoring = ApiMonitoring(
            self,
            "Monitoring",
            project_root=PROJECT_ROOT,
            model_dir=PROJECT_ROOT / model_src,
            api_function=fn,
            api_log_group=log_group,
            bucket=s3.Bucket.from_bucket_name(self, "MlopsBucket", mlops_bucket_name(self)),
            topic=sns.Topic.from_topic_arn(
                self, "Alerts", self.format_arn(service="sns", resource=ALERTS_TOPIC_NAME)),
            drift_alarm_name=DRIFT_ALARM_NAME,
        )

        CfnOutput(self, "ApiUrl", value=url.url)
        CfnOutput(self, "FunctionName", value=fn.function_name)
        CfnOutput(self, "DriftCheckFunctionName", value=monitoring.drift_function.function_name)
        CfnOutput(self, "DashboardUrl", value=monitoring.dashboard_url)


app = cdk.App()
env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "ap-south-1"),
)
CreditRiskMlopsStack(
    app,
    "CreditRiskMlops",
    env=env,
    description="Credit risk MLOps: S3, ECR, IAM roles and SageMaker serverless MLflow",
)
CreditRiskServingStack(
    app,
    "CreditRiskServing",
    model_src=app.node.try_get_context("model_src") or "artifacts",
    env=env,
    description="Credit risk API on Lambda with a Function URL",
)
cdk.Tags.of(app).add("project", PROJECT)
app.synth()

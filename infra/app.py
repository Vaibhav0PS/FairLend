#!/usr/bin/env python3
"""AWS infrastructure for the credit risk MLOps project (AWS CDK, Python).

Deploy:   cdk deploy     (from this folder, with the project venv active)
Teardown: cdk destroy    (deletes the bucket contents, images and MLflow data too)
"""

import os

import aws_cdk as cdk
from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

PROJECT = "credit-risk"
MLFLOW_APP_NAME = f"{PROJECT}-mlflow"
REGISTERED_MODEL = f"{PROJECT}-xgboost"


class CreditRiskMlopsStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------- Storage: data, model artifacts, MLflow artifacts ----------
        # Versioning keeps every uploaded dataset version; jobs record the version ID they read.
        bucket = s3.Bucket(
            self,
            "MlopsBucket",
            bucket_name=f"{PROJECT}-mlops-{self.account}-{self.region}",
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

        # ---------- Outputs used by scripts/launch_training.py ----------
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "TrainRepoUri", value=train_repo.repository_uri)
        CfnOutput(self, "SageMakerJobRoleArn", value=job_role.role_arn)
        CfnOutput(self, "MlflowAppArn", value=mlflow_app.attr_arn)
        CfnOutput(self, "RegisteredModelName", value=REGISTERED_MODEL)


app = cdk.App()
CreditRiskMlopsStack(
    app,
    "CreditRiskMlops",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "ap-south-1"),
    ),
    description="Credit risk MLOps: S3, ECR, IAM roles and SageMaker serverless MLflow",
)
cdk.Tags.of(app).add("project", PROJECT)
app.synth()

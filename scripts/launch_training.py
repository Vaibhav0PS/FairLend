"""Build the training image, upload the dataset, and run training on SageMaker.

Usage (from the project root, with the venv active and AWS credentials set):

    python scripts/launch_training.py                    # processing job on ml.t3.large
    python scripts/launch_training.py --job-type training --instance-type ml.m5.large
    python scripts/launch_training.py --skip-build       # reuse the newest pushed image

Steps:
 1. Read bucket, ECR repo, role and MLflow App ARN from the CreditRiskMlops stack outputs.
 2. Upload credit_risk_dataset.csv to s3://<bucket>/data/raw/ if its SHA-256 changed.
    Bucket versioning keeps old versions; the job records the version ID it trained on.
 3. Build Dockerfile.train, tag it with the git commit, push it to ECR.
 4. Start a SageMaker Processing job (default) or Training job, and wait for it.
    The job logs to MLflow and registers a new model version.

Why a Processing job by default: new AWS accounts start with a quota of 0 for
every training-job instance type, while ml.t3 processing instances are allowed.
Both job types run the same image and the same `python -m credit_risk.train`.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from _aws import isolated_docker_env

ROOT = Path(__file__).resolve().parents[1]
STACK_NAME = "CreditRiskMlops"
DATA_KEY = "data/raw/credit_risk_dataset.csv"
EXPERIMENT = "credit-risk"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=True, cwd=ROOT, **kwargs)


def stack_outputs(session: boto3.Session) -> dict[str, str]:
    stack = session.client("cloudformation").describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack["Outputs"]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_dataset(s3, bucket: str, local: Path) -> str:
    """Upload only if the content changed. Returns the S3 version ID in use."""
    local_hash = sha256(local)
    try:
        head = s3.head_object(Bucket=bucket, Key=DATA_KEY)
        if head.get("Metadata", {}).get("sha256") == local_hash:
            print(f"Dataset unchanged in S3 (version {head['VersionId']})")
            return head["VersionId"]
    except ClientError as err:
        if err.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
    with open(local, "rb") as fh:
        resp = s3.put_object(Bucket=bucket, Key=DATA_KEY, Body=fh, Metadata={"sha256": local_hash})
    print(f"Uploaded dataset to s3://{bucket}/{DATA_KEY} (version {resp['VersionId']})")
    return resp["VersionId"]


def git_commit() -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return sha or "nogit", dirty


def build_and_push(session: boto3.Session, repo_uri: str, tag: str) -> str:
    ecr = session.client("ecr")
    repo_name = repo_uri.split("/", 1)[1]
    image_uri = f"{repo_uri}:{tag}"
    try:
        ecr.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])
        print(f"Image {image_uri} already in ECR, skipping build")
        return image_uri
    except ClientError as err:
        if err.response["Error"]["Code"] != "ImageNotFoundException":
            raise

    run(["docker", "build", "--platform", "linux/amd64", "-f", "Dockerfile.train", "-t", image_uri, "."])
    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    with isolated_docker_env() as env:
        run([env["DOCKER_EXE"], "login", "--username", user, "--password-stdin", auth["proxyEndpoint"]],
            input=password.encode(), stdout=subprocess.DEVNULL, env=env)
        run([env["DOCKER_EXE"], "push", image_uri], env=env)
    return image_uri


def latest_image(session: boto3.Session, repo_uri: str) -> str:
    repo_name = repo_uri.split("/", 1)[1]
    images = session.client("ecr").describe_images(repositoryName=repo_name)["imageDetails"]
    tagged = [i for i in images if i.get("imageTags")]
    if not tagged:
        raise SystemExit("No images in ECR yet; run without --skip-build first.")
    newest = max(tagged, key=lambda i: i["imagePushedAt"])
    return f"{repo_uri}:{newest['imageTags'][0]}"


def start_processing_job(sm, *, name, image_uri, role, bucket, instance_type, max_runtime, env, train_args):
    data_dir = "/opt/ml/processing/input/data"
    out_dir = "/opt/ml/processing/output/model"
    sm.create_processing_job(
        ProcessingJobName=name,
        RoleArn=role,
        AppSpecification={
            "ImageUri": image_uri,
            "ContainerEntrypoint": ["python", "-m", "credit_risk.train"],
            "ContainerArguments": ["--data", f"{data_dir}/credit_risk_dataset.csv", "--out", out_dir, *train_args],
        },
        ProcessingInputs=[{
            "InputName": "data",
            "S3Input": {
                "S3Uri": f"s3://{bucket}/{DATA_KEY}",
                "LocalPath": data_dir,
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        }],
        ProcessingOutputConfig={"Outputs": [{
            "OutputName": "model",
            "S3Output": {"S3Uri": f"s3://{bucket}/models/{name}", "LocalPath": out_dir, "S3UploadMode": "EndOfJob"},
        }]},
        ProcessingResources={"ClusterConfig": {"InstanceCount": 1, "InstanceType": instance_type, "VolumeSizeInGB": 10}},
        StoppingCondition={"MaxRuntimeInSeconds": max_runtime},
        Environment=env,
        Tags=[{"Key": "project", "Value": "credit-risk"}],
    )
    return f"s3://{bucket}/models/{name}/"


def start_training_job(sm, *, name, image_uri, role, bucket, instance_type, max_runtime, env, train_args):
    sm.create_training_job(
        TrainingJobName=name,
        RoleArn=role,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "credit_risk.train"],
            "ContainerArguments": [
                "--data", "/opt/ml/input/data/training/credit_risk_dataset.csv", "--out", "/opt/ml/model", *train_args,
            ],
        },
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/{DATA_KEY}", "S3DataDistributionType": "FullyReplicated",
            }},
        }],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/models/"},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": max_runtime},
        Environment=env,
        Tags=[{"Key": "project", "Value": "credit-risk"}],
    )
    return f"s3://{bucket}/models/{name}/output/model.tar.gz"


def wait(sm, job_type: str, name: str) -> dict:
    describe = sm.describe_processing_job if job_type == "processing" else sm.describe_training_job
    key = "ProcessingJobName" if job_type == "processing" else "TrainingJobName"
    status_key = "ProcessingJobStatus" if job_type == "processing" else "TrainingJobStatus"
    last = None
    started = time.time()
    while True:
        desc = describe(**{key: name})
        status = desc[status_key]
        if status != last:
            print(f"[{time.time() - started:6.0f}s] {status}")
            last = status
        if status in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-type", choices=["processing", "training"], default="processing")
    parser.add_argument("--instance-type", default=None, help="default ml.t3.large (processing) / ml.m5.large (training)")
    parser.add_argument("--max-runtime", type=int, default=3600, help="hard stop in seconds (caps the cost)")
    parser.add_argument("--n-iter", type=int, default=60)
    parser.add_argument("--data", default=str(ROOT / "credit_risk_dataset.csv"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    region = session.region_name
    outputs = stack_outputs(session)
    bucket = outputs["BucketName"]

    version_id = upload_dataset(session.client("s3"), bucket, Path(args.data))

    sha, dirty = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    tag = f"{sha}-dirty-{stamp}" if dirty else sha
    if args.skip_build:
        image_uri = latest_image(session, outputs["TrainRepoUri"])
    else:
        image_uri = build_and_push(session, outputs["TrainRepoUri"], tag)
    print(f"Image: {image_uri}")

    instance_type = args.instance_type or ("ml.t3.large" if args.job_type == "processing" else "ml.m5.large")
    name = f"credit-risk-train-{stamp}"
    env = {
        "AWS_DEFAULT_REGION": region,
        "MLFLOW_TRACKING_URI": outputs["MlflowAppArn"],
        "MLFLOW_EXPERIMENT_NAME": EXPERIMENT,
        "MLFLOW_REGISTERED_MODEL": outputs["RegisteredModelName"],
        "GIT_COMMIT": f"{sha}{'-dirty' if dirty else ''}",
        "IMAGE_URI": image_uri,
        "JOB_NAME": name,
        "DATA_S3_URI": f"s3://{bucket}/{DATA_KEY}",
        "DATA_S3_VERSION_ID": version_id,
    }
    job_args = dict(
        name=name, image_uri=image_uri, role=outputs["SageMakerJobRoleArn"], bucket=bucket,
        instance_type=instance_type, max_runtime=args.max_runtime, env=env, train_args=["--n-iter", str(args.n_iter)],
    )
    sm = session.client("sagemaker")
    starter = start_processing_job if args.job_type == "processing" else start_training_job
    output_uri = starter(sm, **job_args)

    kind = "processing-jobs" if args.job_type == "processing" else "jobs"
    print(f"Started {args.job_type} job {name} on {instance_type}")
    print(f"Console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/{kind}/{name}")
    if args.no_wait:
        return

    desc = wait(sm, args.job_type, name)
    if desc.get("ProcessingJobStatus", desc.get("TrainingJobStatus")) != "Completed":
        print(f"Job failed: {desc.get('FailureReason') or desc.get('ExitMessage', 'see CloudWatch logs')}")
        sys.exit(1)
    print(f"Model artifacts: {output_uri}")

    if args.job_type == "processing":
        s3 = session.client("s3")
        prefix = f"models/{name}"
        for fname in ("metrics.json", "mlflow_run.json"):
            body = s3.get_object(Bucket=bucket, Key=f"{prefix}/{fname}")["Body"].read()
            print(f"--- {fname}\n{json.dumps(json.loads(body), indent=2)[:1500]}")


if __name__ == "__main__":
    main()

"""Promote a registered model version to "champion" (the version that gets deployed).

Usage:
    python scripts/promote_model.py --version 1 --profile dev-aws

Sets the MLflow alias "champion" on that version and marks the matching
SageMaker Model Registry package as Approved, so both registries agree.
The next deploy (GitHub Actions or scripts/deploy_serving.py) serves it.
"""

from __future__ import annotations

import argparse
import os

import boto3

from _aws import stack_outputs

ALIAS = "champion"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile
    os.environ.setdefault("AWS_DEFAULT_REGION", session.region_name)
    outputs = stack_outputs(session)
    os.environ["MLFLOW_TRACKING_URI"] = outputs["MlflowAppArn"]
    name = outputs["RegisteredModelName"]

    from mlflow import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version(name, args.version)
    run = client.get_run(mv.run_id)
    test_metrics = {k: round(v, 4) for k, v in run.data.metrics.items() if k in ("test_roc_auc", "test_pr_auc", "test_f1")}
    print(f"{name} v{mv.version} from run {mv.run_id}: {test_metrics}")

    client.set_registered_model_alias(name, ALIAS, mv.version)
    print(f"MLflow alias '{ALIAS}' -> v{mv.version}")
    if "challenger" in (mv.aliases or []):
        client.delete_registered_model_alias(name, "challenger")
        print("Removed alias 'challenger' (it is the champion now)")

    package_arn = mv.tags.get("sagemaker.model_package_arn")
    if package_arn:
        session.client("sagemaker").update_model_package(
            ModelPackageArn=package_arn,
            ModelApprovalStatus="Approved",
            ApprovalDescription=f"Promoted to {ALIAS}",
        )
        print(f"SageMaker model package approved: {package_arn}")
    else:
        print("No SageMaker model package linked to this version (registry sync may still be running).")


if __name__ == "__main__":
    main()

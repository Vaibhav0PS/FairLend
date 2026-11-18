"""Download the model files of a registered model version for the serving image.

Usage:
    python scripts/fetch_model.py --out build/model                 # the "champion" alias
    python scripts/fetch_model.py --out build/model --version 3     # a specific version

Writes the files from the training run (preprocessor, booster, metadata) plus
registry.json with the registered name, version and run ID. The API reports
these in /health and in its prediction logs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import boto3

from _aws import stack_outputs

REQUIRED = ("preprocessor.joblib", "xgb_model.json", "model_meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--version", default=None, help="overrides --alias")
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

    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version(name, args.version) if args.version else client.get_model_version_by_alias(name, args.alias)
    print(f"Fetching {name} v{mv.version} (run {mv.run_id})")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="model_files", dst_path=tmp))
        missing = [f for f in REQUIRED if not (local / f).exists()]
        if missing:
            raise SystemExit(f"Run {mv.run_id} is missing model files: {missing}")
        shutil.copytree(local, out)

    (out / "registry.json").write_text(json.dumps({
        "name": name,
        "version": str(mv.version),
        "alias": None if args.version else args.alias,
        "run_id": mv.run_id,
        "model_package_arn": mv.tags.get("sagemaker.model_package_arn"),
    }, indent=2))
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()

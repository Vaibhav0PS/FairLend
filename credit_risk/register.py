"""Pipeline step: register a better candidate and ask for approval.

Usage:
    python -m credit_risk.register --candidate DIR --evaluation DIR

Registers the candidate's MLflow run as a new version of MODEL_NAME, points the
"challenger" alias at it, marks its SageMaker Model Registry package as
PendingManualApproval, and emails approval instructions. Nothing is deployed:
a person promotes it with scripts/promote_model.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .notify import publish

CHALLENGER_ALIAS = "challenger"


def wait_for_package_arn(client, name: str, version: str, timeout: int = 180) -> str | None:
    """MLflow copies new versions to the SageMaker Model Registry in the background."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        arn = client.get_model_version(name, version).tags.get("sagemaker.model_package_arn")
        if arn:
            return arn
        time.sleep(10)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()

    import mlflow
    from mlflow import MlflowClient

    name = os.environ["MODEL_NAME"]
    run_id = json.loads(Path(args.candidate, "mlflow_run.json").read_text())["run_id"]
    evaluation = json.loads(Path(args.evaluation, "evaluation.json").read_text())

    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client = MlflowClient()
    version = str(mv.version)
    client.set_registered_model_alias(name, CHALLENGER_ALIAS, version)
    client.set_model_version_tag(name, version, "pr_auc_gain", str(evaluation["pr_auc_gain"]))
    client.set_model_version_tag(name, version, "trigger", evaluation.get("trigger") or "manual")
    print(f"Registered {name} v{version} (alias {CHALLENGER_ALIAS})")

    package_arn = wait_for_package_arn(client, name, version)
    if package_arn:
        import boto3

        boto3.client("sagemaker").update_model_package(
            ModelPackageArn=package_arn,
            ModelApprovalStatus="PendingManualApproval",
            ApprovalDescription=f"PR-AUC gain {evaluation['pr_auc_gain']:+.4f} over champion",
        )
        print(f"SageMaker package {package_arn} set to PendingManualApproval")

    cand, champ = evaluation["candidate"], evaluation["champion"]
    lines = [
        f"Trigger: {evaluation.get('trigger') or 'manual'}",
        f"New model: {name} v{version}",
        f"  PR-AUC {cand['pr_auc']:.4f}  ROC-AUC {cand['roc_auc']:.4f}  recall {cand['recall']:.3f}  precision {cand['precision']:.3f}",
    ]
    if champ:
        lines.append(
            f"Champion v{evaluation['champion_version']}: PR-AUC {champ['pr_auc']:.4f}  ROC-AUC {champ['roc_auc']:.4f}  "
            f"recall {champ['recall']:.3f}  precision {champ['precision']:.3f}"
        )
    lines += [
        f"PR-AUC gain: {evaluation['pr_auc_gain']:+.4f} on the same {evaluation['test_rows']} test rows.",
        "",
        "To approve and deploy:",
        f"  1. python scripts/promote_model.py --version {version} --profile dev-aws",
        "  2. GitHub -> Actions -> Deploy API -> Run workflow",
        "To reject: do nothing. The champion keeps serving.",
    ]
    publish(f"Credit risk: approve model v{version}?", "\n".join(lines))


if __name__ == "__main__":
    main()

"""Pipeline step: compare a newly trained model with the current champion.

Usage:
    python -m credit_risk.evaluate --candidate DIR --data CSV --out DIR

Both models are scored on the same held-out test set, rebuilt from the data
with the candidate's seed and test size. Each model uses its own threshold.
Writes evaluation.json; the pipeline's condition step reads `pr_auc_gain`.

Environment:
    MLFLOW_TRACKING_URI, MODEL_NAME   to download the champion (alias "champion")
    MIN_PR_AUC_GAIN                   gain the candidate needs to count as better (default 0)
    SNS_TOPIC_ARN                     email when the candidate is not better
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from sklearn.model_selection import train_test_split

from .data import clean, load_raw, split_xy
from .model import CreditRiskModel
from .notify import publish
from .train import classification_metrics

CHAMPION_ALIAS = "champion"
KEYS = ("pr_auc", "roc_auc", "brier", "precision", "recall", "f1", "threshold")


def load_champion(name: str, workdir: str):
    """Returns (version, model) for the champion alias, or (None, None) if there is none."""
    if not (os.environ.get("MLFLOW_TRACKING_URI") and name):
        return None, None
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    try:
        mv = MlflowClient().get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException as err:
        print(f"No champion yet: {err}")
        return None, None
    local = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="model_files", dst_path=workdir)
    return str(mv.version), CreditRiskModel.load(local)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    candidate = CreditRiskModel.load(args.candidate)
    training = candidate.meta["training"]
    X, y = split_xy(clean(load_raw(args.data)))
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=training["test_size"], random_state=training["seed"], stratify=y
    )

    def score(model) -> dict:
        m = classification_metrics(y_test.to_numpy(), model.predict_proba(X_test), model.threshold)
        return {k: round(m[k], 5) for k in KEYS}

    result = {"test_rows": int(len(X_test)), "candidate": score(candidate),
              "candidate_model_version": candidate.meta.get("model_version")}
    with tempfile.TemporaryDirectory() as tmp:
        champion_version, champion = load_champion(os.environ.get("MODEL_NAME", ""), tmp)
        result["champion_version"] = champion_version
        result["champion"] = score(champion) if champion else None

    min_gain = float(os.environ.get("MIN_PR_AUC_GAIN", "0") or 0)
    baseline = result["champion"]["pr_auc"] if result["champion"] else 0.0
    result["pr_auc_gain"] = round(result["candidate"]["pr_auc"] - baseline, 5)
    result["min_pr_auc_gain"] = min_gain
    result["candidate_better"] = result["pr_auc_gain"] > min_gain
    result["candidate_better_int"] = int(result["candidate_better"])  # read by the pipeline condition
    result["trigger"] = os.environ.get("TRIGGER", "")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if not result["candidate_better"]:
        publish(
            "Credit risk retraining: new model not better, nothing to approve",
            f"Trigger: {result['trigger'] or 'manual'}\n"
            f"Candidate PR-AUC {result['candidate']['pr_auc']:.4f} vs champion "
            f"v{champion_version} {baseline:.4f} (gain {result['pr_auc_gain']:+.4f}, "
            f"needed > {min_gain:+.4f}).\n"
            "The candidate was not registered. The champion keeps serving.\n"
            f"Execution: {os.environ.get('PIPELINE_EXECUTION_ARN', 'n/a')}",
        )


if __name__ == "__main__":
    main()

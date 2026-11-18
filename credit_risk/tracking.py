"""Optional MLflow experiment tracking and model registration.

Enabled when MLFLOW_TRACKING_URI is set. For SageMaker serverless MLflow, set it
to the MLflow App ARN; the sagemaker-mlflow plugin signs requests with the
current AWS credentials.

Environment variables:
    MLFLOW_TRACKING_URI       MLflow App ARN (or any MLflow server URI)
    MLFLOW_EXPERIMENT_NAME    default "credit-risk"
    MLFLOW_REGISTERED_MODEL   default "credit-risk-xgboost"; "none" (or "") skips registration
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
SERVING_REQUIREMENTS = PACKAGE_DIR.parent / "requirements.txt"


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _flatten(prefix: str, data: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(name, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def log_run(
    *,
    model,
    model_dir: Path,
    metrics: dict,
    figures: dict,
    input_example: pd.DataFrame,
) -> dict:
    import mlflow

    from .mlflow_model import CreditRiskPyfunc

    meta = model.meta
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit-risk"))
    registered_name = os.environ.get("MLFLOW_REGISTERED_MODEL", "credit-risk-xgboost")
    if registered_name.strip().lower() in ("", "none"):
        registered_name = None  # log the run only; the pipeline's register step decides

    with mlflow.start_run(run_name=f"train-{meta['model_version']}") as run:
        mlflow.set_tags({
            "model_version": meta["model_version"],
            "data_sha256": meta["data"]["sha256"],
            **{f"source.{k}": str(v) for k, v in meta.get("source", {}).items() if v},
        })
        params = {
            **{f"best.{k}": v for k, v in meta["training"]["best_params"].items()},
            **{k: v for k, v in meta["training"].items() if k not in ("best_params", "duration_seconds")},
            "rows_train": meta["data"]["rows_train"],
            "rows_test": meta["data"]["rows_test"],
            "calibration_a": model.calibration_a,
            "calibration_b": model.calibration_b,
            "threshold": model.threshold,
            "medium_threshold": model.medium_threshold,
        }
        mlflow.log_params(params)
        mlflow.log_metrics(_flatten("", metrics))
        mlflow.log_metric("training_duration_seconds", meta["training"]["duration_seconds"])

        for name, fig in figures.items():
            mlflow.log_figure(fig, f"plots/{name}.png")
        mlflow.log_artifacts(str(model_dir), "model_files")

        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=CreditRiskPyfunc(),
            artifacts={"model_dir": str(model_dir)},
            code_paths=[str(PACKAGE_DIR)],
            pip_requirements=str(SERVING_REQUIREMENTS) if SERVING_REQUIREMENTS.exists() else None,
            input_example=input_example,
            registered_model_name=registered_name,
        )
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "registered_model": registered_name,
            "registered_version": getattr(info, "registered_model_version", None),
        }

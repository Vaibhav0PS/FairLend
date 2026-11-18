"""Train, calibrate, evaluate and save the credit risk model.

Usage:
    python -m credit_risk.train --data credit_risk_dataset.csv --out artifacts

Steps:
 1. Clean the data and hold out a stratified 20% test set.
 2. Tune XGBoost with RandomizedSearchCV on the training set (PR-AUC).
 3. Get out-of-fold log-odds on the training set and fit Platt calibration on them.
 4. Pick the F1-optimal threshold on the calibrated out-of-fold probabilities.
    The test set is never used for tuning, calibration or the threshold.
 5. Report metrics on the test set and save the artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import tracking
from .config import CATEGORICAL_COLS, NUMERIC_COLS
from .data import clean, file_sha256, load_raw, split_xy
from .model import CreditRiskModel, logit, sigmoid
from .monitoring import build_reference_profile

REFERENCE_PROFILE_FILE = "reference_profile.json"


def s3_version_id(uri: str) -> str:
    """S3 version ID of the training data, for lineage (best effort)."""
    if not uri.startswith("s3://"):
        return ""
    try:
        import boto3

        bucket, key = uri[len("s3://"):].split("/", 1)
        return boto3.client("s3").head_object(Bucket=bucket, Key=key).get("VersionId", "")
    except Exception as err:  # lineage is nice to have, never a reason to fail training
        print(f"Could not read S3 version of {uri}: {err}")
        return ""


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERIC_COLS),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                CATEGORICAL_COLS,
            ),
        ]
    )


PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": randint(150, 600),
    "classifier__max_depth": randint(3, 9),
    "classifier__learning_rate": uniform(0.01, 0.3),
    "classifier__subsample": uniform(0.6, 0.4),
    "classifier__colsample_bytree": uniform(0.6, 0.4),
    "classifier__min_child_weight": randint(1, 10),
    "classifier__gamma": uniform(0, 5),
}


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best])


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="credit_risk_dataset.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--n-iter", type=int, default=60, help="RandomizedSearchCV trials")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    raw = load_raw(args.data)
    df = clean(raw)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Rows: raw={len(raw)} clean={len(df)} train={len(X_train)} test={len(X_test)}")

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # Baseline for comparison
    baseline = Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = classification_metrics(y_test.to_numpy(), baseline_proba, 0.5)

    # Hyperparameter search. xgboost uses 1 thread per fit so the search can run fits in parallel.
    xgb_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric=False)),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=1,
            random_state=args.seed,
        )),
    ])
    search = RandomizedSearchCV(
        xgb_pipeline,
        PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        cv=cv,
        scoring="average_precision",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")

    # Out-of-fold log-odds for calibration and threshold selection
    oof_proba_raw = cross_val_predict(
        clone(search.best_estimator_), X_train, y_train, cv=cv, method="predict_proba", n_jobs=args.n_jobs
    )[:, 1]
    oof_margin = logit(oof_proba_raw).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof_margin, y_train)
    cal_a = float(platt.coef_[0, 0])
    cal_b = float(platt.intercept_[0])
    oof_proba = sigmoid(cal_a * oof_margin[:, 0] + cal_b)

    threshold, oof_f1 = best_f1_threshold(y_train.to_numpy(), oof_proba)
    base_rate = float(y_train.mean())
    medium_threshold = min(base_rate, threshold)
    print(f"Calibration a={cal_a:.4f} b={cal_b:.4f}; threshold={threshold:.4f} (OOF F1={oof_f1:.4f})")

    final_pipeline = search.best_estimator_  # refit on the full training set
    model = CreditRiskModel(
        preprocessor=final_pipeline.named_steps["preprocessor"],
        classifier=final_pipeline.named_steps["classifier"],
        calibration_a=cal_a,
        calibration_b=cal_b,
        threshold=threshold,
        medium_threshold=medium_threshold,
    )

    test_proba = model.predict_proba(X_test)
    test_raw = sigmoid(model.margin(X_test))
    test_metrics = classification_metrics(y_test.to_numpy(), test_proba, threshold)
    test_metrics["brier_uncalibrated"] = float(brier_score_loss(y_test, test_raw))

    metrics = {
        "cv_best_pr_auc": float(search.best_score_),
        "oof_f1_at_threshold": oof_f1,
        "test": test_metrics,
        "baseline_logistic_regression_test": baseline_metrics,
    }
    model.meta = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "rows_raw": int(len(raw)),
            "rows_clean": int(len(df)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "train_default_rate": base_rate,
        },
        # Set by the SageMaker job launcher; empty for local runs.
        "source": {
            "git_commit": os.environ.get("GIT_COMMIT", ""),
            "image_uri": os.environ.get("IMAGE_URI", ""),
            "job_name": os.environ.get("JOB_NAME", os.environ.get("TRAINING_JOB_NAME", "")),
            "data_s3_uri": os.environ.get("DATA_S3_URI", ""),
            "data_s3_version_id": os.environ.get("DATA_S3_VERSION_ID") or s3_version_id(os.environ.get("DATA_S3_URI", "")),
            "pipeline_execution": os.environ.get("PIPELINE_EXECUTION_ARN", ""),
        },
        "training": {
            "seed": args.seed,
            "n_iter": args.n_iter,
            "cv_folds": args.cv_folds,
            "test_size": args.test_size,
            "scale_pos_weight": float(scale_pos_weight),
            "best_params": {
                k.replace("classifier__", ""): (v.item() if hasattr(v, "item") else v)
                for k, v in search.best_params_.items()
            },
            "duration_seconds": round(time.time() - started, 1),
        },
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    model.save(args.out)
    Path(args.out, "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    profile = build_reference_profile(model, X_train)
    profile["model_version"] = model.meta["model_version"]
    Path(args.out, REFERENCE_PROFILE_FILE).write_text(json.dumps(profile, indent=2))

    if tracking.enabled():
        from . import plots

        figures = {
            "calibration": plots.calibration_plot(y_test.to_numpy(), test_proba, test_raw),
            "precision_recall": plots.pr_curve_plot(y_test.to_numpy(), test_proba, threshold),
            "shap_importance": plots.shap_importance_plot(model.explain_many(X_test)),
        }
        run_info = tracking.log_run(
            model=model,
            model_dir=Path(args.out),
            metrics=metrics,
            figures=figures,
            input_example=X_test.head(3),
        )
        Path(args.out, "mlflow_run.json").write_text(json.dumps(run_info, indent=2))
        if run_info["registered_model"]:
            print(f"MLflow run {run_info['run_id']} registered as "
                  f"{run_info['registered_model']} v{run_info['registered_version']}")
        else:
            print(f"MLflow run {run_info['run_id']} logged (not registered)")

    t = test_metrics
    print(
        f"Test: ROC-AUC={t['roc_auc']:.4f} PR-AUC={t['pr_auc']:.4f} Brier={t['brier']:.4f} "
        f"(uncalibrated {t['brier_uncalibrated']:.4f}) P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f}"
    )
    print(f"Baseline LR test: ROC-AUC={baseline_metrics['roc_auc']:.4f} PR-AUC={baseline_metrics['pr_auc']:.4f}")
    print(f"Saved model to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

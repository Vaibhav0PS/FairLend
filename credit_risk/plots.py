"""Evaluation plots saved with each training run (needs matplotlib, training only)."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def calibration_plot(y_true: np.ndarray, proba_calibrated: np.ndarray, proba_raw: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for label, proba in [("Uncalibrated XGBoost", proba_raw), ("Platt calibrated", proba_calibrated)]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (test set)")
    ax.legend()
    fig.tight_layout()
    return fig


def pr_curve_plot(y_true: np.ndarray, proba: np.ndarray, threshold: float):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision)
    ax.scatter([recall[idx]], [precision[idx]], color="red", zorder=3, label=f"Threshold {threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend()
    fig.tight_layout()
    return fig


def shap_importance_plot(contributions: pd.DataFrame):
    importance = contributions.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(importance.index, importance.to_numpy())
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Global feature importance (test set)")
    fig.tight_layout()
    return fig

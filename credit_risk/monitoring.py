"""Reference profile of the training data, used to detect drift in live traffic.

The profile stores, for every model input and for the predicted probability:
- numeric: decile bin edges and the share of training rows in each bin
- categorical: the share of training rows per category

monitoring/drift_check.py (the scheduled Lambda) bins live requests the same
way and computes the population stability index (PSI) per feature:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Common reading: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
Binning uses bisect_left semantics (np.searchsorted side="left") on both sides.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLS, NUMERIC_COLS

EPS = 1e-4
N_BINS = 10


def _numeric_profile(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    bins = np.searchsorted(edges, values, side="left")
    counts = np.bincount(bins, minlength=len(edges) + 1)
    return {
        "type": "numeric",
        "edges": [float(e) for e in edges],
        "expected": [float(c) for c in counts / counts.sum()],
    }


def _categorical_profile(values: pd.Series) -> dict:
    shares = values.dropna().astype(str).value_counts(normalize=True)
    return {"type": "categorical", "expected": {k: float(v) for k, v in shares.items()}}


def build_reference_profile(model, X_train: pd.DataFrame) -> dict:
    features = {col: _numeric_profile(X_train[col].to_numpy(dtype=float)) for col in NUMERIC_COLS}
    features.update({col: _categorical_profile(X_train[col]) for col in CATEGORICAL_COLS})
    proba = model.predict_proba(X_train)
    return {
        "rows": int(len(X_train)),
        "features": features,
        "prediction": _numeric_profile(np.asarray(proba, dtype=float)),
        "high_risk_rate": float(np.mean(proba >= model.threshold)),
    }


def psi(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, EPS), max(a, EPS)
        total += (a - e) * math.log(a / e)
    return total

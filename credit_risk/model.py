"""Model artifact: preprocessing + XGBoost + Platt calibration + SHAP.

One XGBoost model makes every prediction, and TreeSHAP explains that same model.
Calibration is a monotonic sigmoid on the XGBoost log-odds, so

    calibrated_probability = sigmoid(a * (base_value + sum(shap_values)) + b)

holds exactly, and the factors shown to the user add up to the score.

Artifact directory layout:
    preprocessor.joblib   fitted sklearn ColumnTransformer
    xgb_model.json        XGBoost booster in its version-stable JSON format
    model_meta.json       calibration, thresholds, feature groups, metrics, versions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from .config import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS

PREPROCESSOR_FILE = "preprocessor.joblib"
BOOSTER_FILE = "xgb_model.json"
META_FILE = "model_meta.json"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def feature_groups(transformed_names: list[str]) -> dict[str, list[int]]:
    """Map each original column to the indices of its transformed columns.

    Numeric columns map 1:1. A categorical column maps to all of its one-hot
    columns, so their SHAP values can be summed into one factor.
    """
    groups: dict[str, list[int]] = {col: [] for col in FEATURE_COLS}
    for idx, name in enumerate(transformed_names):
        if name.startswith("num__"):
            col = name[len("num__"):]
            if col not in NUMERIC_COLS:
                raise ValueError(f"Unexpected numeric feature {name}")
        elif name.startswith("cat__"):
            rest = name[len("cat__"):]
            matches = [c for c in CATEGORICAL_COLS if rest.startswith(c + "_")]
            if not matches:
                raise ValueError(f"Unexpected categorical feature {name}")
            col = max(matches, key=len)
        else:
            raise ValueError(f"Unexpected feature {name}")
        groups[col].append(idx)
    empty = [c for c, idxs in groups.items() if not idxs]
    if empty:
        raise ValueError(f"No transformed columns for {empty}")
    return groups


@dataclass
class CreditRiskModel:
    preprocessor: Any
    classifier: XGBClassifier
    calibration_a: float
    calibration_b: float
    threshold: float
    medium_threshold: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.groups = feature_groups(self.feature_names)
        # The bias column of pred_contribs is the same for every row.
        dummy = np.zeros((1, len(self.feature_names)))
        self.base_value = float(self._contributions(pd.DataFrame(dummy, columns=self.feature_names))[0, -1])

    # ---------- persistence ----------
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, out / PREPROCESSOR_FILE)
        self.classifier.save_model(out / BOOSTER_FILE)
        meta = {
            **self.meta,
            "calibration": {"method": "platt", "a": self.calibration_a, "b": self.calibration_b},
            "threshold": self.threshold,
            "medium_threshold": self.medium_threshold,
            "feature_names": self.feature_names,
            "feature_groups": self.groups,
        }
        (out / META_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: str | Path) -> "CreditRiskModel":
        d = Path(model_dir)
        meta = json.loads((d / META_FILE).read_text())
        classifier = XGBClassifier()
        classifier.load_model(d / BOOSTER_FILE)
        return cls(
            preprocessor=joblib.load(d / PREPROCESSOR_FILE),
            classifier=classifier,
            calibration_a=float(meta["calibration"]["a"]),
            calibration_b=float(meta["calibration"]["b"]),
            threshold=float(meta["threshold"]),
            medium_threshold=float(meta["medium_threshold"]),
            meta=meta,
        )

    # ---------- inference ----------
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self.preprocessor.transform(df[FEATURE_COLS])
        return pd.DataFrame(np.asarray(transformed, dtype=float), columns=self.feature_names)

    def margin(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated XGBoost log-odds."""
        return self.classifier.predict(self._transform(df), output_margin=True)

    def calibrate(self, margin: np.ndarray) -> np.ndarray:
        return sigmoid(self.calibration_a * np.asarray(margin) + self.calibration_b)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrate(self.margin(df))

    def risk_band(self, probability: float) -> str:
        if probability >= self.threshold:
            return "High Risk"
        if probability >= self.medium_threshold:
            return "Medium Risk"
        return "Low Risk"

    def _contributions(self, transformed: pd.DataFrame) -> np.ndarray:
        """XGBoost's built-in TreeSHAP: one column per feature plus a bias column.

        Used instead of shap.TreeExplainer, whose expected_value does not match
        XGBoost 3.x's base score, so its factors didn't add up to the score.
        """
        dmatrix = xgb.DMatrix(transformed.to_numpy(dtype=float), feature_names=self.feature_names)
        return self.classifier.get_booster().predict(dmatrix, pred_contribs=True)

    def explain_many(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP values per original column for every row (log-odds units)."""
        values = self._contributions(self._transform(df))[:, :-1]
        return pd.DataFrame(
            {col: values[:, idxs].sum(axis=1) for col, idxs in self.groups.items()},
            index=df.index,
        )

    def explain(self, df: pd.DataFrame) -> dict[str, float]:
        """SHAP value per original column for a single row (log-odds units)."""
        if len(df) != 1:
            raise ValueError("explain() expects exactly one row")
        return {col: float(v) for col, v in self.explain_many(df).iloc[0].items()}

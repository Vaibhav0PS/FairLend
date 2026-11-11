import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk.config import (
    DEFAULT_FLAGS,
    HOME_OWNERSHIP,
    LOAN_GRADES,
    LOAN_INTENTS,
    MAX_AGE,
    MAX_EMP_LENGTH,
    MIN_AGE,
)
from credit_risk.model import CreditRiskModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("credit-risk")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "artifacts"))
model = CreditRiskModel.load(MODEL_DIR)

app = FastAPI(title="Credit Ledger", version=model.meta.get("model_version", "unknown"))


class LoanApplication(BaseModel):
    # loan_percent_income is computed on the server; extra fields such as a
    # client-sent ratio are ignored.
    model_config = ConfigDict(extra="ignore")

    person_age: int = Field(ge=MIN_AGE, le=MAX_AGE)
    person_income: float = Field(gt=0, le=10_000_000)
    person_home_ownership: Literal[HOME_OWNERSHIP]
    person_emp_length: float = Field(ge=0, le=MAX_EMP_LENGTH)
    loan_intent: Literal[LOAN_INTENTS]
    loan_grade: Literal[LOAN_GRADES]
    loan_amnt: float = Field(gt=0, le=1_000_000)
    loan_int_rate: float = Field(gt=0, le=40)
    cb_person_default_on_file: Literal[DEFAULT_FLAGS]
    cb_person_cred_hist_length: int = Field(ge=0, le=MAX_AGE)

    @model_validator(mode="after")
    def check_consistency(self) -> "LoanApplication":
        if self.person_emp_length > self.person_age:
            raise ValueError("person_emp_length cannot exceed person_age")
        if self.cb_person_cred_hist_length > self.person_age:
            raise ValueError("cb_person_cred_hist_length cannot exceed person_age")
        return self

    def to_features(self) -> dict:
        features = self.model_dump()
        # Same 2-decimal rounding as the training data.
        features["loan_percent_income"] = round(self.loan_amnt / self.person_income, 2)
        return features


HOME_LABELS = {"RENT": "Renting", "MORTGAGE": "Mortgage", "OWN": "Owns Outright", "OTHER": "Other"}
INTENT_LABELS = {
    "PERSONAL": "Personal",
    "EDUCATION": "Education",
    "MEDICAL": "Medical",
    "VENTURE": "Venture",
    "HOMEIMPROVEMENT": "Home Improvement",
    "DEBTCONSOLIDATION": "Debt Consolidation",
}


def format_feature(col: str, f: dict) -> tuple[str, str]:
    """Human-readable label and applicant value for an original input column."""
    formatters = {
        "person_income": lambda: ("Annual Income", f"${f['person_income']:,.0f}"),
        "loan_percent_income": lambda: (
            "Loan-to-income ratio",
            f"{f['loan_percent_income']:.2f} ({f['loan_percent_income'] * 100:.0f}%)",
        ),
        "loan_int_rate": lambda: ("Interest Rate", f"{f['loan_int_rate']:.2f}%"),
        "loan_amnt": lambda: ("Loan Amount", f"${f['loan_amnt']:,.0f}"),
        "person_emp_length": lambda: ("Employment Length", f"{f['person_emp_length']:.1f} yrs"),
        "person_age": lambda: ("Applicant Age", f"{f['person_age']} yrs"),
        "cb_person_cred_hist_length": lambda: ("Credit History Length", f"{f['cb_person_cred_hist_length']} yrs"),
        "loan_grade": lambda: ("Loan Grade", f"Grade {f['loan_grade']}"),
        "person_home_ownership": lambda: ("Home Ownership", HOME_LABELS[f["person_home_ownership"]]),
        "loan_intent": lambda: ("Loan Purpose", INTENT_LABELS[f["loan_intent"]]),
        "cb_person_default_on_file": lambda: (
            "Prior Default Record",
            "Yes (Prior Default)" if f["cb_person_default_on_file"] == "Y" else "No Prior Defaults",
        ),
    }
    return formatters[col]() if col in formatters else (col, "")


@app.get("/health")
def health():
    return {
        "status": "ready",
        "service": "credit-risk-ledger",
        "model_loaded": True,
        "explainer_loaded": True,
        "model_version": model.meta.get("model_version"),
        "threshold": model.threshold,
        "medium_threshold": model.medium_threshold,
    }


@app.post("/predict")
def predict(data: LoanApplication):
    started = time.perf_counter()
    features = data.to_features()
    input_df = pd.DataFrame([features])

    margin = float(model.margin(input_df)[0])
    probability = float(model.calibrate(margin))
    band = model.risk_band(probability)

    shap_by_feature = model.explain(input_df)
    factors = []
    for col, s_val in shap_by_feature.items():
        label, display_val = format_feature(col, features)
        factors.append({
            "feature": label,
            "raw_feature": col,
            "value": display_val,
            "shap_value": round(s_val, 4),
            "abs_impact": round(abs(s_val), 4),
            "direction": "increase" if s_val > 0 else "decrease",
            "effect": "Increases Risk" if s_val > 0 else "Reduces Risk",
        })
    factors.sort(key=lambda x: x["abs_impact"], reverse=True)

    risk_increasing = [f for f in factors if f["direction"] == "increase"]
    risk_decreasing = [f for f in factors if f["direction"] == "decrease"]

    logger.info(json.dumps({
        "event": "prediction",
        "model_version": model.meta.get("model_version"),
        "probability": round(probability, 4),
        "risk_band": band,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }))

    return {
        "default_probability": probability,
        "default_prediction": int(probability >= model.threshold),
        "threshold": model.threshold,
        "medium_threshold": model.medium_threshold,
        "Result": band,
        "model_version": model.meta.get("model_version"),
        "loan_percent_income": features["loan_percent_income"],
        "shap_analysis": {
            "base_value": model.base_value,
            "model_log_odds": margin,
            "factors": factors,
            "risk_increasing_count": len(risk_increasing),
            "risk_decreasing_count": len(risk_decreasing),
            "top_risk_increasing": risk_increasing[:3],
            "top_risk_decreasing": risk_decreasing[:3],
        },
    }


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

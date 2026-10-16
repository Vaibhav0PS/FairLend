import os
import sys

# Ensure matplotlib uses the project local cache to prevent font-cache rebuilds and permission errors
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
import shap

# Load model, threshold, and SHAP explainer at startup
ml_model = {}
ml_model['model'] = joblib.load('credit_risk_model.pkl')
ml_model['threshold'] = float(joblib.load('best_threshold.pkl'))

pipeline = ml_model['model'].calibrated_classifiers_[0].estimator
ml_model['preprocessor'] = pipeline.named_steps['preprocessor']
ml_model['classifier'] = pipeline.named_steps['classifier']
ml_model['explainer'] = shap.TreeExplainer(ml_model['classifier'])

base_val = ml_model['explainer'].expected_value
if isinstance(base_val, (list, tuple, np.ndarray)):
    base_val = base_val[1] if len(base_val) > 1 else base_val[0]
ml_model['base_value'] = float(base_val)
ml_model['feature_names'] = ml_model['preprocessor'].get_feature_names_out()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/health')
def health():
    """Real healthcheck verifying service and model readiness."""
    is_ready = bool(ml_model.get('model') and ml_model.get('explainer'))
    return {
        "status": "ready" if is_ready else "degraded",
        "service": "credit-risk-ledger",
        "model_loaded": bool(ml_model.get('model')),
        "explainer_loaded": bool(ml_model.get('explainer')),
        "threshold": ml_model.get("threshold", None)
    }


def format_feature(feat_name: str, input_dict: dict) -> tuple[str, str]:
    """Map encoded feature names to human-readable names and applicant values."""
    if feat_name == 'num__person_income':
        return 'Annual Income', f"${input_dict['person_income']:,.0f}"
    elif feat_name == 'num__loan_percent_income':
        pct = input_dict['loan_percent_income'] * 100
        return 'Loan-to-income ratio', f"{input_dict['loan_percent_income']:.2f} ({pct:.0f}%)"
    elif feat_name == 'num__loan_int_rate':
        return 'Interest Rate', f"{input_dict['loan_int_rate']:.2f}%"
    elif feat_name == 'num__loan_amnt':
        return 'Loan Amount', f"${input_dict['loan_amnt']:,.0f}"
    elif feat_name == 'num__person_emp_length':
        return 'Employment Length', f"{input_dict['person_emp_length']:.1f} yrs"
    elif feat_name == 'num__person_age':
        return 'Applicant Age', f"{input_dict['person_age']} yrs"
    elif feat_name == 'num__cb_person_cred_hist_length':
        return 'Credit History Length', f"{input_dict['cb_person_cred_hist_length']} yrs"
    elif feat_name.startswith('cat__loan_grade_'):
        grade = feat_name.replace('cat__loan_grade_', '')
        return 'Loan Grade', f"Grade {grade}"
    elif feat_name.startswith('cat__person_home_ownership_'):
        home = feat_name.replace('cat__person_home_ownership_', '')
        home_label = {'RENT': 'Renting', 'MORTGAGE': 'Mortgage', 'OWN': 'Owns Outright', 'OTHER': 'Other'}.get(home, home)
        return 'Home Ownership', home_label
    elif feat_name.startswith('cat__loan_intent_'):
        intent = feat_name.replace('cat__loan_intent_', '')
        intent_label = {
            'PERSONAL': 'Personal',
            'EDUCATION': 'Education',
            'MEDICAL': 'Medical',
            'VENTURE': 'Venture',
            'HOMEIMPROVEMENT': 'Home Improvement',
            'DEBTCONSOLIDATION': 'Debt Consolidation'
        }.get(intent, intent)
        return 'Loan Purpose', intent_label
    elif feat_name.startswith('cat__cb_person_default_on_file_'):
        default = feat_name.replace('cat__cb_person_default_on_file_', '')
        return 'Prior Default Record', 'Yes (Prior Default)' if default == 'Y' else 'No Prior Defaults'
    return feat_name, ''


# Pydantic input schema
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post('/predict')
def predict(data: LoanApplication):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    probability = float(ml_model['model'].predict_proba(input_df)[:, 1][0])
    threshold = float(ml_model["threshold"])
    prediction = int(probability >= threshold)

    # 3-tier risk band for realistic underwriting decisioning
    if probability >= threshold:
        result_band = "High Risk"
    elif probability >= 0.25:
        result_band = "Medium Risk"
    else:
        result_band = "Low Risk"

    # Compute SHAP explanation
    preprocessor = ml_model['preprocessor']
    explainer = ml_model['explainer']
    feature_names = ml_model['feature_names']

    transformed = preprocessor.transform(input_df)
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    raw_shap_values = explainer.shap_values(transformed_df)
    shap_vals = raw_shap_values[0] if hasattr(raw_shap_values, 'ndim') and raw_shap_values.ndim == 2 else raw_shap_values

    factors = []
    for feat_name, feat_val, s_val in zip(feature_names, transformed[0], shap_vals):
        # Skip inactive one-hot encoded categories
        if feat_name.startswith('cat__') and feat_val == 0.0:
            continue

        label, display_val = format_feature(feat_name, input_dict)
        s_float = float(s_val)
        factors.append({
            "feature": label,
            "raw_feature": feat_name,
            "value": display_val,
            "shap_value": round(s_float, 4),
            "abs_impact": round(abs(s_float), 4),
            "direction": "increase" if s_float > 0 else "decrease",
            "effect": "Increases Risk" if s_float > 0 else "Reduces Risk"
        })

    # Sort factors by impact magnitude
    factors.sort(key=lambda x: x['abs_impact'], reverse=True)

    risk_increasing = [f for f in factors if f['direction'] == 'increase']
    risk_decreasing = [f for f in factors if f['direction'] == 'decrease']

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "medium_threshold": 0.25,
        "Result": result_band,
        "shap_analysis": {
            "base_value": ml_model.get('base_value', 0.0),
            "factors": factors,
            "risk_increasing_count": len(risk_increasing),
            "risk_decreasing_count": len(risk_decreasing),
            "top_risk_increasing": risk_increasing[:3],
            "top_risk_decreasing": risk_decreasing[:3]
        }
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")

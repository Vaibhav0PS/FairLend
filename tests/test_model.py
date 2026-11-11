import numpy as np
import pandas as pd
import pytest

from credit_risk.config import FEATURE_COLS
from credit_risk.data import clean, load_raw
from credit_risk.model import CreditRiskModel


@pytest.fixture(scope="module")
def model():
    return CreditRiskModel.load("artifacts")


@pytest.fixture(scope="module")
def sample():
    df = clean(load_raw("credit_risk_dataset.csv"))
    return df[FEATURE_COLS].sample(25, random_state=0).reset_index(drop=True)


def test_feature_groups_cover_every_column_once(model):
    indices = sorted(i for idxs in model.groups.values() for i in idxs)
    assert indices == list(range(len(model.feature_names)))
    assert set(model.groups) == set(FEATURE_COLS)


def test_shap_factors_add_up_to_model_log_odds(model, sample):
    margins = model.margin(sample)
    for i in range(len(sample)):
        contributions = model.explain(sample.iloc[[i]])
        total = model.base_value + sum(contributions.values())
        assert total == pytest.approx(float(margins[i]), abs=1e-3)


def test_explain_many_matches_single_row_explain(model, sample):
    batch = model.explain_many(sample)
    single = model.explain(sample.iloc[[3]])
    assert batch.iloc[3].to_dict() == pytest.approx(single)


def test_probabilities_and_thresholds_are_valid(model, sample):
    proba = model.predict_proba(sample)
    assert np.all((proba > 0) & (proba < 1))
    assert 0 < model.medium_threshold <= model.threshold < 1


def test_risk_bands(model):
    assert model.risk_band(model.threshold) == "High Risk"
    assert model.risk_band(0.0) == "Low Risk"
    if model.medium_threshold < model.threshold:
        mid = (model.medium_threshold + model.threshold) / 2
        assert model.risk_band(mid) == "Medium Risk"


def test_clean_keeps_rows_with_missing_employment_length():
    raw = pd.DataFrame({
        "person_age": [30, 30, 150, 25],
        "person_income": [50000, 50000, 50000, 40000],
        "person_home_ownership": ["RENT"] * 4,
        "person_emp_length": [np.nan, 5.0, 5.0, 40.0],
        "loan_intent": ["PERSONAL"] * 4,
        "loan_grade": ["A"] * 4,
        "loan_amnt": [1000, 2000, 1000, 1000],
        "loan_int_rate": [10.0] * 4,
        "loan_status": [0, 1, 0, 0],
        "loan_percent_income": [0.02, 0.04, 0.02, 0.03],
        "cb_person_default_on_file": ["N"] * 4,
        "cb_person_cred_hist_length": [3] * 4,
    })
    cleaned = clean(raw)
    # Keeps the missing emp_length row; drops age 150 and emp_length > age.
    assert len(cleaned) == 2
    assert cleaned["person_emp_length"].isna().sum() == 1

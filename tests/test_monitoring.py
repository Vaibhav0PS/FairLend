import json

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from credit_risk.config import FEATURE_COLS
from credit_risk.data import clean, load_raw, split_xy
from credit_risk.model import CreditRiskModel
from credit_risk.monitoring import psi
from monitoring import drift_check


@pytest.fixture(scope="module")
def model():
    return CreditRiskModel.load("artifacts")


@pytest.fixture(scope="module")
def profile():
    return json.loads(open("artifacts/reference_profile.json").read())


@pytest.fixture(scope="module")
def test_rows(model):
    X, y = split_xy(clean(load_raw("credit_risk_dataset.csv")))
    t = model.meta["training"]
    _, X_test, _, _ = train_test_split(X, y, test_size=t["test_size"], random_state=t["seed"], stratify=y)
    # The API never sends missing values; fill like the model's imputer would.
    return X_test.fillna(X_test.median(numeric_only=True)).sample(400, random_state=0)


def to_records(model, rows):
    proba = model.predict_proba(rows)
    return [
        {
            "features": {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in FEATURE_COLS},
            "probability": float(p),
            "risk_band": model.risk_band(float(p)),
        }
        for (_, row), p in zip(rows.iterrows(), proba)
    ]


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25


def test_profile_shares_sum_to_one(profile):
    for spec in [*profile["features"].values(), profile["prediction"]]:
        shares = spec["expected"].values() if spec["type"] == "categorical" else spec["expected"]
        assert sum(shares) == pytest.approx(1.0)


def test_held_out_traffic_shows_no_major_drift(model, profile, test_rows):
    report = drift_check.compute_report(to_records(model, test_rows), profile)
    assert report["enough_data"]
    assert report["max_feature_psi"] < 0.1


def test_shifted_traffic_is_flagged(model, profile, test_rows):
    shifted = test_rows.copy()
    shifted["person_income"] = shifted["person_income"] * 0.5
    shifted["loan_int_rate"] = np.minimum(shifted["loan_int_rate"] + 5, 40)
    report = drift_check.compute_report(to_records(model, shifted), profile)
    assert report["max_feature_psi"] > 0.25
    top_two = list(report["feature_psi"])[:2]
    assert set(top_two) == {"person_income", "loan_int_rate"}


def test_too_few_requests_skip_the_drift_calculation(model, profile, test_rows):
    report = drift_check.compute_report(to_records(model, test_rows.head(10)), profile)
    assert report == {"prediction_count": 10, "min_samples": drift_check.MIN_SAMPLES, "enough_data": False}

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID = {
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "loan_amnt": 10000,
    "loan_int_rate": 7.5,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6,
}


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert 0 < body["medium_threshold"] <= body["threshold"] < 1


def test_index_is_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Credit Ledger" in res.text


def test_predict_returns_consistent_explanation():
    res = client.post("/predict", json=VALID)
    assert res.status_code == 200
    body = res.json()
    assert 0 < body["default_probability"] < 1
    assert body["Result"] in {"Low Risk", "Medium Risk", "High Risk"}

    shap = body["shap_analysis"]
    assert len(shap["factors"]) == 11  # one factor per input column
    total = shap["base_value"] + sum(f["shap_value"] for f in shap["factors"])
    assert total == pytest.approx(shap["model_log_odds"], abs=0.01)


def test_loan_percent_income_is_computed_on_server():
    res = client.post("/predict", json={**VALID, "loan_percent_income": 0.99})
    assert res.status_code == 200
    assert res.json()["loan_percent_income"] == pytest.approx(0.17)


def test_high_risk_profile_scores_higher_than_low_risk():
    low = client.post("/predict", json=VALID).json()["default_probability"]
    risky = {
        **VALID,
        "person_income": 15000,
        "loan_amnt": 9000,
        "loan_grade": "F",
        "loan_int_rate": 20.0,
        "cb_person_default_on_file": "Y",
    }
    high = client.post("/predict", json=risky).json()["default_probability"]
    assert high > low


@pytest.mark.parametrize(
    "override",
    [
        {"loan_grade": "Z"},
        {"person_home_ownership": "CASTLE"},
        {"person_income": -100},
        {"person_age": 500},
        {"loan_int_rate": 0},
        {"person_emp_length": 31},  # longer than age 30
        {"cb_person_cred_hist_length": 31},
    ],
)
def test_invalid_input_is_rejected(override):
    res = client.post("/predict", json={**VALID, **override})
    assert res.status_code == 422


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID.items() if k != "loan_amnt"}
    assert client.post("/predict", json=payload).status_code == 422

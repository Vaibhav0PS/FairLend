"""Send loan applications to the live API, optionally with drift, for monitoring demos.

Usage:
    python scripts/simulate_traffic.py --n 200                 # traffic like the training data
    python scripts/simulate_traffic.py --n 200 --drift         # poorer applicants, higher rates
    python scripts/simulate_traffic.py --n 200 --drift --check # then run the drift check now

Applications are sampled from credit_risk_dataset.csv. With --drift, income is
halved and interest rates rise by 5 points, which the drift check should flag.
--check invokes the drift-check Lambda right away with a 1-hour lookback,
instead of waiting for the daily schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from credit_risk.config import FEATURE_COLS, MAX_AGE  # noqa: E402
from credit_risk.data import clean, load_raw  # noqa: E402

from _aws import stack_outputs  # noqa: E402


def applications(n: int, drift: bool, seed: int) -> list[dict]:
    df = clean(load_raw(ROOT / "credit_risk_dataset.csv"))[FEATURE_COLS]
    df = df.fillna(df.median(numeric_only=True)).sample(n, random_state=seed)
    if drift:
        df["person_income"] = (df["person_income"] * 0.5).round()
        df["loan_int_rate"] = (df["loan_int_rate"] + 5).clip(upper=40)
    df["person_emp_length"] = df[["person_emp_length", "person_age"]].min(axis=1)
    df["cb_person_cred_hist_length"] = df[["cb_person_cred_hist_length", "person_age"]].min(axis=1)
    df = df[df["person_age"] <= MAX_AGE]
    rows = df.drop(columns=["loan_percent_income"]).to_dict(orient="records")
    for r in rows:
        r["person_age"] = int(r["person_age"])
        r["cb_person_cred_hist_length"] = int(r["cb_person_cred_hist_length"])
    return rows


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-south-1")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    serving = stack_outputs(session, "CreditRiskServing")
    url = serving["ApiUrl"].rstrip("/") + "/predict"

    statuses: dict[int, int] = {}
    rows = applications(args.n, args.drift, args.seed)
    for i, row in enumerate(rows, 1):
        code = post(url, row)
        statuses[code] = statuses.get(code, 0) + 1
        if i % 50 == 0:
            print(f"  sent {i}/{len(rows)}")
    print(f"Sent {len(rows)} {'drifted ' if args.drift else ''}applications: HTTP status counts {statuses}")

    if args.check:
        resp = session.client("lambda").invoke(
            FunctionName=serving["DriftCheckFunctionName"],
            Payload=json.dumps({"lookback_hours": 1}).encode(),
        )
        report = json.loads(resp["Payload"].read())
        print(json.dumps({k: report.get(k) for k in (
            "prediction_count", "max_feature_psi", "prediction_psi", "high_risk_rate",
            "reference_high_risk_rate", "feature_psi")}, indent=2))
        if report.get("max_feature_psi", 0) > 0.25:
            print("Drift above 0.25: the credit-risk-drift alarm should fire within a few minutes, "
                  "email you, and start the retraining pipeline.")


if __name__ == "__main__":
    main()

"""Scheduled drift check for the credit risk API (AWS Lambda, Python standard library + boto3).

1. Reads the API's prediction log lines from the last LOOKBACK_HOURS
   (each line holds the model inputs and the predicted probability).
2. Bins them with the reference profile of the deployed model's training data
   (reference_profile.json, packaged with this function at deploy time).
3. Computes PSI per feature and for the predicted probability.
4. Publishes CloudWatch metrics and saves a JSON report plus the raw
   predictions to S3 under monitoring/.

Invoke by hand with an optional event, e.g. {"lookback_hours": 1}.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3

EPS = 1e-4
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "CreditRisk/Monitoring")
LOG_GROUP = os.environ.get("LOG_GROUP", "/aws/lambda/credit-risk-api")
BUCKET = os.environ.get("BUCKET", "")
LOOKBACK_HOURS = float(os.environ.get("LOOKBACK_HOURS", "24"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "50"))

# Copied next to this file at deploy time (infra/app.py). Absent in unit tests.
_PROFILE_FILE = Path(__file__).parent / "reference_profile.json"
PROFILE = json.loads(_PROFILE_FILE.read_text()) if _PROFILE_FILE.exists() else {}


def psi(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, EPS), max(a, EPS)
        total += (a - e) * math.log(a / e)
    return total


def numeric_shares(edges: list[float], values: list[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect.bisect_left(edges, v)] += 1
    n = len(values)
    return [c / n for c in counts]


def feature_psi(spec: dict, values: list) -> float:
    if spec["type"] == "numeric":
        return psi(spec["expected"], numeric_shares(spec["edges"], [float(v) for v in values]))
    counts = Counter(str(v) for v in values)
    categories = sorted(set(spec["expected"]) | set(counts))
    n = len(values)
    return psi([spec["expected"].get(c, 0.0) for c in categories], [counts.get(c, 0) / n for c in categories])


def compute_report(records: list[dict], profile: dict = PROFILE) -> dict:
    """Drift report for a list of logged predictions (pure function, unit-tested)."""
    report = {"prediction_count": len(records), "min_samples": MIN_SAMPLES, "enough_data": len(records) >= MIN_SAMPLES}
    if not report["enough_data"]:
        return report
    features = {
        name: round(feature_psi(spec, [r["features"][name] for r in records]), 4)
        for name, spec in profile["features"].items()
    }
    probabilities = [r["probability"] for r in records]
    report.update({
        "feature_psi": dict(sorted(features.items(), key=lambda kv: -kv[1])),
        "max_feature_psi": max(features.values()),
        "prediction_psi": round(feature_psi(profile["prediction"], probabilities), 4),
        "high_risk_rate": round(sum(r["risk_band"] == "High Risk" for r in records) / len(records), 4),
        "reference_high_risk_rate": round(profile.get("high_risk_rate", 0.0), 4),
    })
    return report


def fetch_records(logs, start_ms: int, end_ms: int) -> list[dict]:
    records, token = [], None
    while True:
        kwargs = {
            "logGroupName": LOG_GROUP,
            "filterPattern": '{ $.event = "prediction" }',
            "startTime": start_ms,
            "endTime": end_ms,
        }
        if token:
            kwargs["nextToken"] = token
        resp = logs.filter_log_events(**kwargs)
        for event in resp.get("events", []):
            try:
                record = json.loads(event["message"])
            except json.JSONDecodeError:
                continue
            if "features" in record:  # lines written before inputs were logged are skipped
                records.append(record)
        token = resp.get("nextToken")
        if not token:
            return records


def put_metrics(cloudwatch, report: dict) -> None:
    data = [{"MetricName": "PredictionCount", "Value": report["prediction_count"], "Unit": "Count"}]
    if report["enough_data"]:
        data += [
            {"MetricName": "MaxFeaturePSI", "Value": report["max_feature_psi"]},
            {"MetricName": "PredictionPSI", "Value": report["prediction_psi"]},
            {"MetricName": "HighRiskRate", "Value": report["high_risk_rate"]},
        ]
        data += [
            {"MetricName": "FeaturePSI", "Dimensions": [{"Name": "Feature", "Value": name}], "Value": value}
            for name, value in report["feature_psi"].items()
        ]
    for i in range(0, len(data), 20):
        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=data[i:i + 20])


def handler(event, context):
    hours = float((event or {}).get("lookback_hours", LOOKBACK_HOURS)) if isinstance(event, dict) else LOOKBACK_HOURS
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(hours * 3600 * 1000)

    records = fetch_records(boto3.client("logs"), start_ms, end_ms)
    report = compute_report(records)
    now = datetime.now(timezone.utc)
    report.update({
        "checked_at": now.isoformat(),
        "lookback_hours": hours,
        "reference_model_version": PROFILE.get("model_version"),
        "served_versions": sorted({str(r.get("registered_version")) for r in records}),
    })

    put_metrics(boto3.client("cloudwatch"), report)
    if BUCKET:
        s3 = boto3.client("s3")
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        day = now.strftime("%Y-%m-%d")
        s3.put_object(Bucket=BUCKET, Key=f"monitoring/drift/date={day}/{stamp}.json",
                      Body=json.dumps(report, indent=2).encode(), ContentType="application/json")
        if records:
            body = "\n".join(json.dumps(r) for r in records).encode()
            s3.put_object(Bucket=BUCKET, Key=f"monitoring/predictions/date={day}/{stamp}.jsonl", Body=body)

    print(json.dumps(report))
    return report

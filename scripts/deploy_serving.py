"""Deploy the API to Lambda with the registry's champion model, then smoke-test it.

Usage (from the project root):
    python scripts/deploy_serving.py --profile dev-aws      # laptop
    python scripts/deploy_serving.py                        # GitHub Actions (OIDC credentials)
    python scripts/deploy_serving.py --version 2            # serve a specific version

Steps:
 1. Download the champion model files into build/model (scripts/fetch_model.py).
 2. `cdk deploy CreditRiskServing -c model_src=build/model`: CDK builds the
    Dockerfile with that model baked in, pushes it to ECR and updates Lambda.
 3. Call /health and /predict on the Function URL and check the answers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import boto3

from _aws import isolated_docker_env, stack_outputs

ROOT = Path(__file__).resolve().parents[1]
CDK = ["npx", "--yes", "aws-cdk@2.1148.1"]
SAMPLE = {
    "person_age": 30, "person_income": 60000, "person_home_ownership": "RENT", "person_emp_length": 5,
    "loan_intent": "PERSONAL", "loan_grade": "A", "loan_amnt": 10000, "loan_int_rate": 7.5,
    "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6,
}


def call(url: str, payload: dict | None = None, attempts: int = 5) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception as err:  # cold start right after a deploy can time out once
            if i == attempts - 1:
                raise
            print(f"  retrying {url} ({err})")
            time.sleep(5)


def smoke_test(api_url: str, expected_version: str) -> None:
    base = api_url.rstrip("/")
    started = time.time()
    health = call(f"{base}/health")
    print(f"/health ({time.time() - started:.1f}s incl. cold start): {health}")
    if health.get("registered_version") != expected_version:
        raise SystemExit(f"Expected model v{expected_version}, Lambda serves v{health.get('registered_version')}")
    result = call(f"{base}/predict", SAMPLE)
    shap = result["shap_analysis"]
    total = shap["base_value"] + sum(f["shap_value"] for f in shap["factors"])
    if abs(total - shap["model_log_odds"]) > 0.01:
        raise SystemExit("SHAP factors do not add up to the model's log-odds")
    print(f"/predict: probability={result['default_probability']:.4f} band={result['Result']} "
          f"factors={len(shap['factors'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=None, help="registered version (default: champion alias)")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-south-1")
    args = parser.parse_args()

    model_dir = ROOT / "build" / "model"
    fetch = [sys.executable, str(ROOT / "scripts" / "fetch_model.py"), "--out", str(model_dir), "--region", args.region]
    if args.profile:
        fetch += ["--profile", args.profile]
    if args.version:
        fetch += ["--version", args.version]
    subprocess.run(fetch, check=True, cwd=ROOT)
    version = json.loads((model_dir / "registry.json").read_text())["version"]

    cmd = [*CDK, "deploy", "CreditRiskServing", "--require-approval", "never",
           "-c", "model_src=build/model"]
    if args.profile:
        cmd += ["--profile", args.profile]
    with isolated_docker_env() as env:
        env["CDK_DEFAULT_REGION"] = args.region
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT / "infra", env=env, shell=sys.platform == "win32")

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    api_url = stack_outputs(session, "CreditRiskServing")["ApiUrl"]
    print(f"API: {api_url}")
    smoke_test(api_url, version)
    print("Deploy OK")


if __name__ == "__main__":
    main()

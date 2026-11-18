"""Start the retraining pipeline by hand and follow its steps.

Usage:
    python scripts/start_retraining.py --profile dev-aws
    python scripts/start_retraining.py --profile dev-aws --min-gain -1   # demo: force the approval path
    python scripts/start_retraining.py --profile dev-aws --watch <execution-arn>

--min-gain is the PR-AUC improvement over the champion that a new model needs
(default 0). A negative value lets an equal or slightly worse model through to
the approval email, which is useful to demonstrate the approval flow.
"""

from __future__ import annotations

import argparse
import time

import boto3

PIPELINE = "credit-risk-retraining"


def watch(sm, arn: str) -> str:
    seen: dict[str, str] = {}
    started = time.time()
    while True:
        status = sm.describe_pipeline_execution(PipelineExecutionArn=arn)["PipelineExecutionStatus"]
        steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=arn, SortOrder="Ascending")["PipelineExecutionSteps"]
        for s in steps:
            key, value = s["StepName"], s["StepStatus"]
            if seen.get(key) != value:
                reason = f" ({s['FailureReason']})" if s.get("FailureReason") else ""
                print(f"[{time.time() - started:5.0f}s] {key}: {value}{reason}")
                seen[key] = value
        if status in ("Succeeded", "Failed", "Stopped"):
            print(f"Pipeline {status}")
            return status
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-gain", default="0")
    parser.add_argument("--n-iter", default="60")
    parser.add_argument("--watch", default=None, help="follow an existing execution instead of starting one")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-south-1")
    args = parser.parse_args()

    sm = boto3.Session(profile_name=args.profile, region_name=args.region).client("sagemaker")
    arn = args.watch
    if not arn:
        arn = sm.start_pipeline_execution(
            PipelineName=PIPELINE,
            PipelineExecutionDisplayName=f"manual-{time.strftime('%Y%m%d-%H%M%S')}",
            PipelineParameters=[
                {"Name": "Trigger", "Value": "manual"},
                {"Name": "MinPrAucGain", "Value": str(args.min_gain)},
                {"Name": "NIter", "Value": str(args.n_iter)},
            ],
        )["PipelineExecutionArn"]
        print(f"Started {arn}")
    if not args.no_wait:
        watch(sm, arn)


if __name__ == "__main__":
    main()

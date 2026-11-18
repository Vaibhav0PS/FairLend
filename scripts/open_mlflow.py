"""Open the MLflow UI (experiments, runs, plots, registered models) in the browser.

Usage:
    python scripts/open_mlflow.py --profile dev-aws

The account has no SageMaker Studio domain, so this uses a presigned URL instead.
The link must be opened within a few minutes; run the script again for a new one.
"""

import argparse
import webbrowser

import boto3

from _aws import stack_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    arn = stack_outputs(session)["MlflowAppArn"]
    url = session.client("sagemaker").create_presigned_mlflow_app_url(Arn=arn)["AuthorizedUrl"]
    if args.print_only:
        print(url)
    else:
        webbrowser.open(url)
        print("Opened the MLflow UI in your browser.")


if __name__ == "__main__":
    main()

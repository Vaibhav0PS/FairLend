"""Subscribe an email address to the project's alerts (drift, API errors, approval requests).

Usage:
    python scripts/subscribe_alerts.py --email you@example.com --profile dev-aws

AWS sends a confirmation email; click the link in it before alerts arrive.
Kept out of CDK so the address is not committed to the public repository.
"""

import argparse

import boto3

from _aws import stack_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default="ap-south-1")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    topic_arn = stack_outputs(session)["AlertsTopicArn"]
    sns = session.client("sns")

    for sub in sns.list_subscriptions_by_topic(TopicArn=topic_arn)["Subscriptions"]:
        if sub["Protocol"] == "email" and sub["Endpoint"].lower() == args.email.lower():
            status = "pending confirmation" if sub["SubscriptionArn"] == "PendingConfirmation" else "confirmed"
            print(f"{args.email} is already subscribed ({status})")
            return

    sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=args.email)
    print(f"Confirmation email sent to {args.email}. Click the link in it to start receiving alerts.")


if __name__ == "__main__":
    main()

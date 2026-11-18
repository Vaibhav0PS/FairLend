"""Email notifications through the project's SNS topic (no-op when SNS_TOPIC_ARN is unset)."""

import os


def publish(subject: str, message: str) -> None:
    topic = os.environ.get("SNS_TOPIC_ARN")
    print(f"[notify] {subject}\n{message}")
    if not topic:
        return
    import boto3

    boto3.client("sns").publish(TopicArn=topic, Subject=subject[:100], Message=message)

"""Helpers shared by the scripts in this folder."""

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3

STACK_NAME = "CreditRiskMlops"


def stack_outputs(session: boto3.Session, stack_name: str = STACK_NAME) -> dict[str, str]:
    stack = session.client("cloudformation").describe_stacks(StackName=stack_name)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


@contextlib.contextmanager
def isolated_docker_env():
    """Environment for Docker commands that log in to ECR.

    On Windows, the ECR token is too long for the Windows credential store
    ("The stub received bad data"). A throwaway DOCKER_CONFIG stores the token
    in a temp file instead and leaves the user's own Docker config untouched.
    Docker picks the OS credential helper automatically if it finds one on PATH,
    so on Windows the Docker bin folder is left off PATH. DOCKER_EXE (and
    CDK_DOCKER, for the CDK CLI) point to docker.exe by its full path.
    """
    docker_exe = shutil.which("docker")
    if not docker_exe:
        raise SystemExit("docker not found on PATH")
    host = subprocess.run(
        [docker_exe, "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as docker_config:
        Path(docker_config, "config.json").write_text("{}")
        env = {
            **os.environ,
            "DOCKER_CONFIG": docker_config,
            "DOCKER_HOST": host,
            "DOCKER_EXE": docker_exe,
            "CDK_DOCKER": docker_exe,
        }
        if sys.platform == "win32":
            docker_dir = Path(docker_exe).parent.resolve()
            env["PATH"] = os.pathsep.join(
                p for p in os.environ.get("PATH", "").split(os.pathsep) if p and Path(p).resolve() != docker_dir
            )
        yield env

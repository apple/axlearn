# Copyright © 2023 Apple Inc.

"""Utilities to invoke docker commands."""

import pathlib
import subprocess
import sys
from typing import Dict, Optional

from absl import logging

from axlearn.gcp.utils import concat_cmd_list, handle_popen, infer_cli_name


def auth(registry: str):
    _run(f"gcloud auth configure-docker {registry}")


# TODO(markblee): Consider using docker-py.
def build(
    *,
    dockerfile: str,
    image: str,
    context: str,
    args: Optional[Dict[str, str]] = None,
    target: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Builds a Dockerfile.

    Args:
        dockerfile: Path to Dockerfile. Should be a file contained within the context.
        image: The name to assign to the built image.
        context: Build context.
        args: Optional build args to pass to `docker build`.
        target: Optional build target.
            https://docs.docker.com/engine/reference/commandline/build/#specifying-target-build-stage---target
        labels: Optional image labels. Can be viewed with `docker inspect`.

    Returns:
        The image name.
    """
    args = args or {}
    labels = labels or {}

    # Build command.
    cli_args = [f"docker build -t {image} -f {dockerfile}"]
    for k, v in args.items():
        cli_args.append(f"--build-arg {k.strip().upper()}={v.strip()}")
    for k, v in labels.items():
        cli_args.append(f"--label {k.strip()}={v.strip()}")
    if target:
        cli_args.append(f"--target {target}")
    cli_args.append(context)

    # Execute command.
    _run(concat_cmd_list(cli_args))
    return image


def push(image: str) -> str:
    """Pushes the given image to repo.

    Args:
        image: Docker image to push.

    Returns:
        The image name.
    """
    _run(f"docker push {image}")
    return image


def pull(image: str) -> str:
    """Pulls the given image from repo.

    Args:
        image: Docker image to pull.

    Returns:
        The image name.
    """
    _run(f"docker pull {image}")
    return image


def _run(*args):
    try:
        with subprocess.Popen(
            *args,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as proc:
            for line in proc.stdout:
                logging.info(line.decode("utf-8").strip())
            handle_popen(proc)
    except RuntimeError as e:
        logging.error(e)
        cli_name = infer_cli_name()
        logging.error(
            "If the error relates to authentication, please run `%s gcp config activate` "
            "and '%s gcp auth' before this script.",
            cli_name,
            cli_name,
        )
        sys.exit(1)


def registry_from_repo(repo: str) -> str:
    """Parse docker registry from repo."""
    return pathlib.Path(repo).parts[0]

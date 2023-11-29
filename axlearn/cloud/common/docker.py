# Copyright © 2023 Apple Inc.

"""Utilities to invoke docker commands."""

import os
import pathlib
import subprocess
from typing import Dict, Optional

from absl import logging

from axlearn.cloud.common.utils import handle_popen


# NOTE: docker-py doesn't seem to work well with Docker Desktop.
def build(
    *,
    dockerfile: str,
    image: str,
    context: str,
    args: Optional[Dict[str, str]] = None,
    target: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
    platform: Optional[str] = None,
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
        platform: Optional target platform for the build output.
            https://docs.docker.com/build/building/multi-platform/

    Returns:
        The image name.
    """
    args = args or {}
    labels = labels or {}

    # Build command.
    cli_args = ["docker", "build", "-t", image, "-f", dockerfile]
    for k, v in args.items():
        cli_args.extend(["--build-arg", f"{k.strip().upper()}={v.strip()}"])
    for k, v in labels.items():
        cli_args.extend(["--label", f"{k.strip()}={v.strip()}"])
    if target:
        cli_args.extend(["--target", target])
    if platform:
        cli_args.extend(["--platform", platform])
    cli_args.append(context)

    # Execute command.
    env_copy = os.environ.copy()
    env_copy["DOCKER_BUILDKIT"] = "1"
    _run(cli_args, env=env_copy)
    return image


def push(image: str) -> str:
    """Pushes the given image to repo.

    Args:
        image: Docker image to push.

    Returns:
        The image name.
    """
    _run(["docker", "push", image])
    return image


def _run(*args, **kwargs):
    with subprocess.Popen(
        *args, **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as proc:
        for line in proc.stdout:
            logging.info(line.decode("utf-8").strip())
        handle_popen(proc)


def registry_from_repo(repo: str) -> str:
    """Parse docker registry from repo."""
    return pathlib.Path(repo).parts[0]

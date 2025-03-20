# Copyright © 2023 Apple Inc.

"""Utilities specific to TPUs."""

import re
from typing import Any

from absl import logging

from axlearn.cloud.common.types import ResourceMap
from axlearn.common.compiler_options import infer_tpu_type, infer_tpu_version


def get_default_env(*, tpu_type: str, num_tpu_slices: int, job_name: str) -> dict[str, Any]:
    """Gets the default environment for TPU pods."""
    return dict(
        # Use a large refresh to mitigate DNS timeout issues until tf>2.12 upgrade.
        GCS_RESOLVE_REFRESH_SECS=600,
        TPU_TYPE=tpu_type,
        NUM_TPU_SLICES=num_tpu_slices,
        XLA_FLAGS=f"--xla_dump_to=/output/{job_name}/xla",
        TF_CPP_MIN_LOG_LEVEL=0,
        # Necessary for surfacing FATAL TPU errors.
        TPU_STDERR_LOG_LEVEL=0,
        # Default; see https://cloud.google.com/tpu/docs/troubleshooting/trouble-tf#debug_logs
        TPU_MIN_LOG_LEVEL=0,
        # Forces TensorStore to retry failed requests.
        TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS=60,
        TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES=256,
        LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4",
    )


def infer_tpu_cores(tpu_type: str) -> int:
    """Infer the number of TPU cores from the TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred number of TPU cores.
    """
    return int(tpu_type.rsplit("-", 1)[1])


def infer_tpu_workers(tpu_type: str) -> int:
    """Infer the number of worker processes for the given TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred number of TPU workers.
    """
    tpu_pattern = r"(.+)*-(\d+)"
    match = re.search(tpu_pattern, tpu_type)
    try:
        if match is not None:
            tpu_version, tpu_cores = match.groups()
            if tpu_version in {"v3", "v4", "v5p"}:
                return int(tpu_cores) // 8
            if tpu_version in {"v5litepod", "v6e"}:
                return int(tpu_cores) // 4
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Failed to parse tpu_type %s: %s", tpu_type, e)
    raise NotImplementedError(tpu_type)


def infer_tpu_resources(instance_type: str, num_replicas: int) -> ResourceMap[int]:
    """Infers resources required by the given instance type and number of replicas."""
    tpu_type = infer_tpu_type(instance_type)
    return {infer_tpu_version(tpu_type): infer_tpu_cores(tpu_type) * num_replicas}

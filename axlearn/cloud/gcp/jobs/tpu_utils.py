# Copyright © 2025 Apple Inc.
"""Utils of TPU pods."""

from typing import Any


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

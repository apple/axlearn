# Copyright © 2024 Apple Inc.

"""Run health check on TPU slices to catch potential hardware failures early.

The health check program tests TPU's matrix unit, scalar unit, HBM and ICI functionality. If
running in a multi-slice environment, DCN connectivity is also checked. Currently, only GKE launch
API is supported.

The type of health check performed depends on the health check spec passed into each of the health
check function. Health check spec is a comma separated list of check_type=timeout, where check_type
is one of ["single", "pairwise", "global"] and timeout is a float in seconds. Example:
single=300,pairwise=1800,global=180

Global health check runs after initializing the distributed coordinator, and should have the
shortest timeout. Pairwise health check should have the longest timeout since different slices may
bring up their container at different times.

The main API is the `setup` function, which is commonly enabled via context manager:
```
with setup(spec):
    # Initialize jax distributed.
```
"""

import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Literal, Optional, Union

import tensorflow as tf
from absl import flags, logging

from axlearn.cloud.gcp import tpu_health_check_main

CheckType = Literal["single", "pairwise", "global"]


def _parse_spec_and_check_if_should_skip(
    check_spec: str,
    *,
    check_type: CheckType,
    num_slices_lower_bound: int = 1,
) -> Optional[float]:
    """Parses health check spec and returns timeout if the `check_type` is in the check_spec.

    Also checks if environment variables satisfy health check requirements. Returns None if
    `check_type` is not in the spec or environment does not satisfy requirements.
    """
    assert check_spec, "Health check spec should not be empty"
    for check in check_spec.split(","):
        check_split = check.split("=")
        assert (
            len(check_split) == 2
        ), f"Expect spec to be be specified as type=timeout, e.g. single=180, but got {check}"
        if check_split[0] == check_type:
            timeout = float(check_split[1])
            break
    else:
        logging.info(
            "Skipping %s slice health check because check spec is %s.", check_type, check_spec
        )
        return None

    # These environment variables are set by GKE.
    if "MEGASCALE_NUM_SLICES" not in os.environ or "NODE_NAME" not in os.environ:
        raise RuntimeError(
            "TPU health check is enabled but MEGASCALE_NUM_SLICES and NODE_NAME are absent in env."
        )

    total_slices = int(os.environ["MEGASCALE_NUM_SLICES"])
    if total_slices < num_slices_lower_bound:
        logging.info(
            "Skipping %s slice health check since num_slices < %d.",
            check_type,
            num_slices_lower_bound,
        )
        return None
    return timeout


def _run_health_check_program(
    env: dict[str, str],
    *,
    output_dir: str,
    timeout: float,
    cur_slice_id: Union[str, int, tuple[int]],
):
    """Runs the health check program in a subprocess and handles failure."""
    start_t = time.perf_counter()
    if isinstance(cur_slice_id, tuple):
        slice_id_str = "-".join(str(x) for x in cur_slice_id)
    else:
        slice_id_str = str(cur_slice_id)
    logging.info("Starting slice %s health check...", slice_id_str)
    try:
        subprocess.run(
            ["python3", "-m", "axlearn.cloud.gcp.tpu_health_check_main"],
            timeout=timeout,
            check=True,
            env=env,
        )
        logging.info(
            "Slice %s health check passed in %fs!", slice_id_str, time.perf_counter() - start_t
        )
        return
    except subprocess.TimeoutExpired:
        # Not using logging.fatal because it will cause a core dump.
        logging.error("Slice %s health check failed due to timeout.", slice_id_str)
        error_type = "timeout"
    except subprocess.CalledProcessError:
        logging.error("Slice %s health check failed due to program error.", slice_id_str)
        error_type = "program error"
    except:  # pylint: disable=bare-except
        logging.error("Slice %s health check failed due to unknown error.", slice_id_str)
        error_type = "unknown"
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    fname = (
        f"{timestamp}-slice-{slice_id_str}-{os.environ['HOSTNAME']}-{os.environ['NODE_NAME']}.txt"
    )
    with tf.io.gfile.GFile(os.path.join(output_dir, fname), "w") as f:
        f.write(error_type)
    sys.exit(-1)


@contextmanager
def setup(check_spec: str):
    _pre_init_health_check(check_spec, output_dir=flags.FLAGS.trainer_dir)
    yield
    # Skip global health check if there's an exception.
    global_health_check(check_spec, output_dir=flags.FLAGS.trainer_dir)


def _pre_init_health_check(check_spec: str, *, output_dir: str):
    """Runs health checks that must run before jax.distributed.initialize."""
    single_slice_health_check(check_spec, output_dir=output_dir)
    pairwise_slice_health_check(check_spec, output_dir=output_dir)


def single_slice_health_check(check_spec: str, *, output_dir: str):
    """Runs health check on each TPU slices independent of each other."""
    timeout = _parse_spec_and_check_if_should_skip(
        check_spec, check_type="single", num_slices_lower_bound=2
    )
    if timeout is None:
        return

    cur_slice_id = os.environ["MEGASCALE_SLICE_ID"]
    new_env = os.environ.copy()
    # TPU_WORKER_HOSTNAMES is a sorted list of hostname separated by comma.
    new_env["MEGASCALE_COORDINATOR_ADDRESS"] = new_env["TPU_WORKER_HOSTNAMES"].split(",")[0]
    new_env["MEGASCALE_NUM_SLICES"] = "1"
    new_env["NUM_TPU_SLICES"] = "1"
    new_env["MEGASCALE_SLICE_ID"] = "0"
    _run_health_check_program(
        new_env,
        output_dir=output_dir,
        timeout=timeout,
        cur_slice_id=cur_slice_id,
    )


def pairwise_slice_health_check(check_spec: str, *, output_dir: str):
    """Runs health check on slice pairs.

    For example, if we have 5 slices, health check will run on slice pair (0, 1) and slice triplet
    (2, 3, 4).
    """
    timeout = _parse_spec_and_check_if_should_skip(
        check_spec, check_type="pairwise", num_slices_lower_bound=4
    )
    if timeout is None:
        return

    total_slices = int(os.environ["MEGASCALE_NUM_SLICES"])
    cur_slice_id = int(os.environ["MEGASCALE_SLICE_ID"])
    new_env = os.environ.copy()
    # Hostname's format is prefix-job-SLICE_ID-WORKER_ID.
    # Coordinator address has format prefix-job-SLICE_ID-WORKER_ID.prefix
    # We tweak slice id to construct coordinator's address.
    prefix = new_env["HOSTNAME"].rsplit("-", maxsplit=3)[0]

    # If number of slices is odd, last 3 slices test together.
    if cur_slice_id >= total_slices - 3 and total_slices % 2 != 0:
        primary_slice_id = total_slices - 3
        new_env["MEGASCALE_COORDINATOR_ADDRESS"] = f"{prefix}-job-{primary_slice_id}-0.{prefix}"
        new_env["MEGASCALE_NUM_SLICES"] = "3"
        new_env["NUM_TPU_SLICES"] = "3"
        new_env["MEGASCALE_SLICE_ID"] = f"{3 - (total_slices - cur_slice_id)}"
        slice_pair_ids = tuple(range(primary_slice_id, primary_slice_id + 3))
    else:
        primary_slice_id = cur_slice_id - cur_slice_id % 2
        new_env["MEGASCALE_COORDINATOR_ADDRESS"] = f"{prefix}-job-{primary_slice_id}-0.{prefix}"
        new_env["MEGASCALE_NUM_SLICES"] = "2"
        new_env["NUM_TPU_SLICES"] = "2"
        new_env["MEGASCALE_SLICE_ID"] = f"{cur_slice_id % 2}"
        slice_pair_ids = (primary_slice_id, primary_slice_id + 1)
    _run_health_check_program(
        new_env,
        output_dir=output_dir,
        timeout=timeout,
        cur_slice_id=slice_pair_ids,
    )


def global_health_check(check_spec: str, *, output_dir: str):
    """Runs global health check.

    This function does not run in a subprocess to reuse the global coordinator. Therefore, it must
    be called after `jax.distributed.initialize()`.
    """
    timeout = _parse_spec_and_check_if_should_skip(check_spec, check_type="global")
    if timeout is None:
        return

    start_t = time.perf_counter()
    logging.info("Starting multi-slice (global) health check...")

    def get_return_value(fn, return_val):
        return_val[0] = fn()

    return_val = [None]
    th = threading.Thread(target=get_return_value, args=(tpu_health_check_main.main, return_val))
    th.start()
    th.join(timeout=timeout)
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    fname = f"{timestamp}-global-{os.environ['HOSTNAME']}-{os.environ['NODE_NAME']}.txt"
    if th.is_alive():
        # Join timed out.
        logging.error("Multi-slice (global) health check failed due to timeout!")

        with tf.io.gfile.GFile(os.path.join(output_dir, fname), "w") as f:
            f.write("timeout")
        # Normal exit via sys.exit may not work when health check program hanged.
        os.kill(os.getpid(), signal.SIGKILL)
    else:
        if not return_val[0]:
            logging.error("Multi-slice (global) health check failed due to program error!")
            with tf.io.gfile.GFile(os.path.join(output_dir, fname), "w") as f:
                f.write("program error")
            os.kill(os.getpid(), signal.SIGKILL)
        logging.info("Global health check passed in %fs.", time.perf_counter() - start_t)

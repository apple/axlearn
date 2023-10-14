# Copyright © 2023 Apple Inc.

"""SPMD related utils."""

import socket
from typing import Optional

import jax
import jax.numpy as jnp
import portpicker
from jax.experimental import multihost_utils

_jax_distributed_initialized = False


def setup(
    *,
    distributed_coordinator: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
    jax_backend: Optional[str] = None,
):
    """Sets up the Jax environment for SPMD/pjit.

    Args:
        distributed_coordinator: The distributed coordinator address (in the form of <host>:<port>).
            Needed only if not running on TPU *and* jax.process_count() > 1. Otherwise the
            coordinator will be configured automatically.
        num_processes: The number of processes (GPU backend: total number of gpus). Needed only
            if not running on TPU *and* jax.process_count() > 1. Otherwise the coordinator will
            be configured automatically.
        process_id: The process id (GPU backend: the GPU rank). Needed only if not running
            on TPU *and* jax.process_count() > 1. Otherwise the coordinator will be
            configured automatically.
        jax_backend: The distributed backend, which can be "cpu", "gpu", or "tpu".
            By default, it would be configured automatically.

    Raises:
        ValueError: If distributed_coordinator, num_processes, or process_id are not None when
            jax_backend is "tpu", or if distributed_coordinator is unsupported.
    """
    # Use a GSPMD-friendly PRNG implementation.
    jax.config.update("jax_default_prng_impl", "rbg")
    # This allows replicated jax.Arrays to be used for computation on the host.
    jax.config.update("jax_spmd_mode", "allow_all")

    global _jax_distributed_initialized  # pylint: disable=global-statement
    if not _jax_distributed_initialized:
        # (jax issue): do not call jax.default_backend for gpu environment
        # which would only pick one process's gpus
        jax_backend = jax_backend or jax.default_backend()
        if jax_backend == "tpu":
            assert (
                distributed_coordinator is None and num_processes is None and process_id is None
            ), ValueError(
                "distributed_coordinator, num_processes, process_id "
                "should all be None for tpu backend"
            )
            jax.distributed.initialize(
                coordinator_address=_infer_tpu_coordinator_address(),
                num_processes=jax.process_count(),
                process_id=jax.process_index(),
            )
        else:
            num_processes = num_processes if num_processes is not None else jax.process_count()
            process_id = process_id if process_id is not None else jax.process_index()
            if not distributed_coordinator:
                if num_processes == 1:
                    distributed_coordinator = f"localhost:{portpicker.pick_unused_port()}"
                else:
                    raise ValueError(f"Unknown distributed_coordinator: {distributed_coordinator}")
            jax.distributed.initialize(
                distributed_coordinator,
                num_processes=num_processes,
                process_id=process_id,
            )
        _jax_distributed_initialized = True


def _infer_tpu_coordinator_address() -> str:
    """Infers a viable JAX coordination address on TPU (including over multiple TPU slices).

    TODO(markblee,tom_gunter): Delete this when multi-slice init is fully supported by JAX.

    Returns:
        A coordinator address string as "ip:port".
    """
    slice_local_coordinator_ip = socket.gethostbyname(socket.gethostname())
    # E.g. "172.31.4.83".
    slice_local_coordinator_ip_as_nums = [int(num) for num in slice_local_coordinator_ip.split(".")]
    # E.g. [172, 31, 4, 83].
    global_coordinator_ip_as_nums = multihost_utils.broadcast_one_to_all(
        jnp.asarray(slice_local_coordinator_ip_as_nums)
    )
    global_coordinator_ip = ".".join([str(num) for num in global_coordinator_ip_as_nums])
    # E.g. "172.31.4.83" on all hosts on all slices.
    global_coordinator_port = multihost_utils.broadcast_one_to_all(
        jnp.asarray(portpicker.pick_unused_port())
    )
    global_coordinator_address = f"{global_coordinator_ip}:{global_coordinator_port}"
    return global_coordinator_address

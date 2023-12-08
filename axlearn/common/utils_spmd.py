# Copyright © 2023 Apple Inc.

"""SPMD related utils."""

import logging
import socket
from typing import Optional

import jax
import jax.numpy as jnp
import portpicker
from jax.experimental import multihost_utils

_jax_distributed_initialized = False


def setup(
    *,
    jax_backend: str,
    distributed_coordinator: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
):
    """Sets up the JAX environment for SPMD.

    Args:
        jax_backend: The distributed backend, which can be "cpu", "gpu", or "tpu".
        distributed_coordinator: The distributed coordinator address (in the form of <host>:<port>).
            Needed only for `jax_backend != "tpu"` and `num_processes > 1`. Otherwise, the
            coordinator will be configured automatically when `num_processes` and `process_id` are
            provided.
        num_processes: The number of processes. Needed only if distributed initialization is desired
            for `jax_backend != "tpu"`.
        process_id: The process ID (the process rank). Needed only if distributed initialization is
            desired for `jax_backend != "tpu"`.

    Raises:
        ValueError: If any of the following conditions are met:
            * distributed_coordinator, num_processes, or process_id are not None when
                jax_backend is "tpu";
            * one of num_processes or process_id is None when jax_backend is not "tpu";
            * distributed_coordinator is None when jax_backend is not "tpu" and num_processes > 1.
    """
    # Use a GSPMD-friendly PRNG implementation.
    jax.config.update("jax_default_prng_impl", "rbg")
    # This allows replicated jax.Arrays to be used for computation on the host.
    jax.config.update("jax_spmd_mode", "allow_all")

    global _jax_distributed_initialized  # pylint: disable=global-statement
    if not _jax_distributed_initialized:
        if jax_backend == "tpu":
            if not (
                distributed_coordinator is None and num_processes is None and process_id is None
            ):
                raise ValueError(
                    "distributed_coordinator, num_processes, and process_id "
                    "should all be None for tpu backend."
                )
            jax.distributed.initialize(
                coordinator_address=_infer_tpu_coordinator_address(),
                num_processes=jax.process_count(),
                process_id=jax.process_index(),
            )
        else:
            if distributed_coordinator is None and num_processes is None and process_id is None:
                logging.info(
                    "Skipping distributed initialization for %s backend, "
                    "since distributed_coordinator, num_processes, and process_id are all None.",
                    jax_backend,
                )
                return

            if num_processes is None or process_id is None:
                raise ValueError(
                    "num_processes and process_id should be provided together "
                    f"if distributed initialization is desired for backend {jax_backend}. "
                    f"Instead, got num_processes={num_processes}, process_id={process_id}."
                )

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

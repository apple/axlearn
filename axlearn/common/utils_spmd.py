# Copyright © 2023 Apple Inc.

"""SPMD related utils."""

import logging
from typing import Optional

import jax
import portpicker

_jax_distributed_initialized = False


def setup(
    *,
    jax_backend: str,
    distributed_coordinator: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
    initialization_timeout: Optional[int] = None,
):
    """Sets up the JAX environment for SPMD.

    Args:
        jax_backend: The distributed backend. Can be "cpu", "gpu", "tpu", or "proxy".
        distributed_coordinator: The distributed coordinator address (e.g., "<host>:<port>").
            If jax_backend is "tpu", this may be automatically inferred by JAX.
            If jax_backend is "proxy", this is ignored.
        num_processes: The number of processes.
            If jax_backend is "tpu", this may be automatically inferred by JAX.
            If jax_backend is "proxy", this is ignored.
        process_id: The process ID (the process rank).
            If jax_backend is "tpu", this may be automatically inferred by JAX.
            If jax_backend is "proxy", this is ignored.
        initialization_timeout: The jax distributed initialization timeout in seconds.
            If None, uses jax default.
            If jax_backend is "proxy", this is ignored.

    Raises:
        ValueError: If any of the following conditions are met:
            * `jax_backend` not in ("tpu", "proxy") and (`num_processes` is None or `process_id` is
                None).
            * `jax_backend` not in ("tpu", "proxy"), `num_processes` > 1, and
                `distributed_coordinator` is None.
    """
    # Use a GSPMD-friendly PRNG implementation.
    jax.config.update("jax_default_prng_impl", "rbg")

    if jax_backend == "proxy":
        # pylint: disable-next=import-error,import-outside-toplevel
        import pathwaysutils  # pytype: disable=import-error

        pathwaysutils.initialize()
        return

    global _jax_distributed_initialized  # pylint: disable=global-statement
    if not _jax_distributed_initialized:
        init_kwargs = {}
        if initialization_timeout is not None:
            init_kwargs["initialization_timeout"] = initialization_timeout

        if jax_backend == "tpu":
            if (distributed_coordinator is None) ^ (process_id is None):
                raise ValueError(
                    "distributed_coordinator and process_id should be both None or both "
                    f"not-None, but got {distributed_coordinator=}, {process_id=}"
                )
            init_kwargs.update(
                coordinator_address=distributed_coordinator,
                process_id=process_id,
                # This is optional.
                num_processes=num_processes,
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

            init_kwargs.update(
                coordinator_address=distributed_coordinator,
                num_processes=num_processes,
                process_id=process_id,
            )
            if jax_backend == "gpu":
                # jax 0.4.34 introduced a change to cluster auto-detection behavior, supplying
                # local_device_ids arg allows us to maintain expected behavior
                init_kwargs["local_device_ids"] = list(range(8))

        jax.distributed.initialize(**init_kwargs)
        _jax_distributed_initialized = True

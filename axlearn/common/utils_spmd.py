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
            jax.distributed.initialize()
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
                initialization_timeout=900,
            )
        _jax_distributed_initialized = True

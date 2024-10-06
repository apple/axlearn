# Copyright © 2024 Apple Inc.
"""Runtime and compiler options for JAX/XLA."""
# This module must not depend on any jax/axlearn modules so that
# importing this module does not result in initializing jax.
import re
from typing import Union


def default_xla_options(
    *, instance_type: str, num_slices: int, backend: str
) -> dict[str, Union[str, bool]]:
    """Return the default flags for the given instance type and backend.

    These options can be passed to `jitted_fn.lower(...).compile(compiler_options=...)`
    or converted to flags using `xla_flags_from_options` and passed to
    `LIBTPU_INIT_ARGS` (only works on TPU) or `XLA_FLAGS` (works on any platform including TPU)
    before importing jax.

    Args:
        instance_type: A specifier for the ML accelerator. E.g., "tpu-v5p-2048".
        num_slices: The number of slices of the given instance type.
        backend: The jax backend. E.g., "tpu".

    Returns:
        A dictionary with the XLA flags and values.

    Raises:
        NotImplementedError if the instance type and backend combination is not supported.
    """
    if backend != "tpu":
        raise NotImplementedError(backend)
    version = infer_tpu_version(infer_tpu_type(instance_type))
    options = dict(
        xla_tpu_spmd_rng_bit_generator_unsafe=True,  # SPMD partition-aware RngBitGenerator.
        xla_tpu_enable_latency_hiding_scheduler="true",  # Try to schedule ops efficiently.
        xla_tpu_perform_spmd_cse_prevention="false",
        # b/229655601: prevent OOM on gpt2-small-repeat.
    )
    if version == "v4":
        options.update(
            # Per maggioni@google.com, the following flags are not supported by V3.
            # These flags are enabled by default starting on v5.
            xla_enable_async_all_gather="true",  # Allow async all-gather.
            xla_enable_async_collective_permute="true",  # Allow async collective permute.
        )
    if num_slices > 1:
        # Support multiple TPU slices connected over a data center network.
        options.update(
            # For collectives across multiple slices.
            xla_tpu_enable_megascale_barrier="true",
            # Per rwitten@google.com the following two flags allow gradient all-reduce to happen
            # concurrently with gradient computation for the following layer.
            xla_tpu_enable_data_parallel_all_reduce_opt="true",
            xla_tpu_data_parallel_opt_different_sized_ops="true",
        )

    # Validate options. Will never fail if this function is implemented correctly.
    for k, v in options.items():
        assert v in [True, False, "true", "false"], (k, v)

    return options


def xla_flags_from_options(xla_options: dict[str, Union[str, bool]]) -> str:
    """Convert an XLA options dict suitable for
    `jitted_fn.lower(...).compile(compiler_options=xla_options)`
    to XLA flags suitable for the `XLA_FLAGS` environment variable.
    """
    flags = []
    for k, v in xla_options.items():
        if isinstance(v, bool):
            v = "1" if v else "0"
        flags.append(f"--{k}={v}")
    return " ".join(flags)


class NotTpuError(ValueError):
    pass


def infer_tpu_type(instance_type: str) -> str:
    """Infers tpu type (e.g. v4-8) from instance type (e.g. tpu-v4-8 or v4-8)."""
    if not (instance_type and re.fullmatch(r"(tpu-)?v.+-\d+", instance_type)):
        raise NotTpuError(f"Invalid TPU instance: {instance_type}")
    return instance_type.replace("tpu-", "")


def infer_tpu_version(tpu_type: str) -> str:
    """Infer TPU version from the TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred TPU version string.

    Raises:
        ValueError: if the TPU version string is unknown.
    """
    tpu_type = infer_tpu_type(tpu_type)
    tpu_version = tpu_type.rsplit("-", 1)[0]  # split from the last occurrence of '-'
    if tpu_version not in _TPU_VERSIONS:
        raise ValueError(f"Unknown TPU version {tpu_version}. Expected one of {_TPU_VERSIONS}")
    return tpu_version


_TPU_VERSIONS = ("v3", "v4", "v5litepod", "v5p")

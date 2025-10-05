# Copyright © 2024 Apple Inc.
"""Runtime and compiler options for JAX/XLA."""

import math
import os

# This module must not depend on any jax/axlearn modules so that
# importing this module does not result in initializing jax.
import re
import typing
from typing import Any, Dict, Sequence, Union

from absl import logging

if typing.TYPE_CHECKING:
    from axlearn.common.utils import MeshShape


def default_xla_options(
    *, instance_type: str, num_slices: int, backend: str
) -> dict[str, Union[str, bool, int]]:
    """Return the default flags for the given instance type and backend.

    These options can be passed to `jitted_fn.lower(...).compile(compiler_options=...)`
    or converted to flags using `xla_flags_from_options` and passed to
    `LIBTPU_INIT_ARGS` (only works on TPU) or `XLA_FLAGS` (works on any platform including TPU)
    before importing jax.

    Environment variable overrides:
        XLA options can be overridden using the XLA_OPTIONS_OVERRIDE environment variable.
        The format is a comma-separated list of key=value pairs:

        Example: XLA_OPTIONS_OVERRIDE="xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction
        =false, megascale_grpc_premap_memory_bytes=34359738368"

        Values are used as-is (strings) and validated by the existing validation logic.

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
    options: Dict[str, Union[str, bool, int]] = dict(
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
    if version == "v6e":
        options.update(
            # Flag controlling the maximum number of overlapping host offloadings.
            xla_tpu_host_transfer_overlap_limit=24,
            # Flag controlling the maximum number of overlapping cross-DCN send/recv.
            xla_max_concurrent_host_send_recv=100,
            # Flag controlling the number of times the scheduler is run if the scheduled
            # peak memory usage exceeds the initial memory limit, by setting memory limit
            # to 90% of the previous memory limit each time. Default value is 1. Sometimes
            # when the scheduler thinks it goes out memory, it may not actually happen due
            # to other factors controlled by other compiler passes, or the initial memory
            # limit is already set too low. Cutting the memory limit to 90% of previous one
            # though, may make the scheduler weighting too much on the memory usage instead
            # of latency side.
            xla_latency_hiding_scheduler_rerun=2,
            # Improved performance for v6e.
            xla_tpu_scoped_vmem_limit_kib=98304,
            # For megascale performance.
            xla_jf_crs_combiner_threshold_count=10,
            # TODO(hanzhi-zhou): temporary workaround to avoid PCIe overload when using multi-slice
            # v6e training caused by allreduce over DCN. This flag doesn't impact performance.
            xla_tpu_iova_dma_chunk_size_bytes=1048576,
            # Disable collective matmul. Collective matmul could negatively affect performance in
            # some cases. Even in cases where collective matmul provides gains, the gains are
            # marginal on v6e due to the high arithmetic intensity.
            xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000,
        )

        # These flags enable SparseCore (SC).
        options.update(
            xla_tpu_use_tc_device_shape_on_sc="true",
            xla_sc_enable_instruction_fusion="false",
            xla_sc_disable_megacore_partitioning="true",
        )

        # Collective fusions are mutually exclusive with SparseCore offloading. We enable allgather
        # fusion and allreduce SC offloading by default.
        options.update(
            xla_tpu_enable_async_collective_fusion_fuse_all_gather="true",
            # Always enable SparseCore offloading for allreduce.
            xla_tpu_enable_sparse_core_collective_offload_all_reduce="true",
        )

        options.update(
            # Improved performance for v6e.
            xla_tpu_enable_async_collective_fusion="true",
            xla_tpu_enable_async_collective_fusion_multiple_steps="true",
            xla_tpu_overlap_compute_collective_tc="true",
            xla_enable_async_all_gather="true",
            # Host offloading flags
            xla_tpu_enable_all_experimental_scheduler_features="true",
            # Flag to enable memory tracking scheduling. The default AUTO only enables
            # it in some situations. Not needed if
            # xla_tpu_enable_all_experimental_scheduler_features is set to true already.
            xla_tpu_enable_scheduler_memory_pressure_tracking="true",
            # Flag to enable the aggressive removal of opt-barriers.
            xla_tpu_aggressive_opt_barrier_removal="true",
            # Flag to enable more aggressive scheduling for async ops, such as pushing
            # the async start to the beginning of the loop body.
            xla_lhs_prioritize_async_depth_over_stall="true",
            # Flag to enable pipelining of cross-DCN all-gathers.
            xla_tpu_enable_ag_backward_pipelining="true",
            xla_should_allow_loop_variant_parameter_in_chain="true",
            xla_should_add_loop_invariant_op_in_chain="true",
            xla_tpu_use_enhanced_launch_barrier="true",
        )

    if num_slices > 1:
        # Support multiple TPU slices connected over a data center network.
        # pytype: disable=wrong-arg-types
        options.update(
            # For collectives across multiple slices.
            xla_tpu_enable_megascale_barrier="true",
            # Per rwitten@google.com the following two flags allow gradient all-reduce to happen
            # concurrently with gradient computation for the following layer.
            xla_tpu_enable_data_parallel_all_reduce_opt="true",
            xla_tpu_data_parallel_opt_different_sized_ops="true",
            # Group non-blocking DCN collectives into as few stages as possible.
            xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction="true",
            # Limit the size of tensors for DCN collectives sinking to 64MB, to avoid allocating
            # unecessary extra HBM which causes OOM.
            xla_tpu_host_dcn_reduce_max_sinking_limit=64 * 1024 * 1024,
            # Change to 16GB. The default is 4GB which is too small for larger models. This
            # cause the step time to be double. You should increase this
            # further if you see "Allocator failed to allocate". A feature
            # to dynamically allocate may come later: b/380514965
            megascale_grpc_premap_memory_bytes=16 * 1024 * 1024 * 1024,
            # Aborting the coordinator after collecting errors from all workers.
            # All workers will also abort after they detect the coordinator is shutdown.
            megascale_error_reporter_abort_on_hang="true",
            # Similar to megascale_error_reporter_abort_on_hang but for unrecoverable errors.
            megascale_error_reporter_abort_on_error="true",
            # Increase the timeout at which a hang is detected/reported, default is 5m.
            megascale_graph_hang_threshold="10m",
            # Similar to megascale_graph_hang_threshold but specific to within a launch_id.
            # Default is 1m.
            megascale_graph_within_launch_hang_threshold="10m",
            # TODO(ethanli): temporary workaround to avoid memory leak in megascale.
            megascale_grpc_enable_xor_tracer="false",
            megascale_debug_port="8081",
            # Observed a lot of deadline exceeded error. Default is 45s.
            megascale_send_rpc_timeout="5m",
        )
        # pytype: enable=wrong-arg-types

    # Apply environment variable overrides
    options = _apply_overrides_from_env(options)

    # Validate options. Will never fail if this function is implemented correctly.
    for k, v in options.items():
        if isinstance(v, (int, bool)):
            continue
        elif isinstance(v, str):
            # Allow numeric strings, time-based strings (e.g., "10m", "30s", "60m"), and bool str
            if v.isdigit() or re.match(r"^\d+[ms]$", v.strip()) or v.strip() in ["true", "false"]:
                continue
            else:
                raise ValueError(f"Invalid string value for option {k}: {v}")
        else:
            raise ValueError(f"Invalid type for option {k}: {type(v).__name__} (value: {v})")

    return options


def _apply_overrides_from_env(
    options: dict[str, Union[str, bool, int]]
) -> dict[str, Union[str, bool, int]]:
    """Apply environment variable overrides to XLA options.

    Reads the XLA_OPTIONS_OVERRIDE environment variable and parses it as a comma-separated
    list of key=value pairs to override XLA options.

    Args:
        options: The original XLA options dictionary.

    Returns:
        A new dictionary with environment variable overrides applied.
    """
    override_env = os.environ.get("XLA_OPTIONS_OVERRIDE")
    if not override_env:
        return options

    overridden_options = options.copy()
    try:
        # Parse comma-separated key=value pairs
        for pair in override_env.split(","):
            if not pair:
                continue

            if "=" not in pair:
                logging.warning("Invalid XLA option override format (missing '='): %s", pair)
                continue

            key, value = pair.split("=", 1)  # Split only on first '=' in case value contains '='
            key = key.strip()
            value = value.strip()

            if not key:
                logging.warning("Empty key in XLA option override: %s", pair)
                continue

            # Log the override
            if key in overridden_options:
                logging.info(
                    "Overriding XLA option %s: %s -> %s",
                    key,
                    overridden_options[key],
                    value,
                )
            else:
                logging.info(
                    "Adding new XLA option %s: %s",
                    key,
                    value,
                )

            overridden_options[key] = value

    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        logging.error("Error parsing XLA_OPTIONS_OVERRIDE: %s", e)
        # Return original options on parse error
        return options

    return overridden_options


def xla_flags_from_options(xla_options: dict[str, Union[str, bool, int]]) -> str:
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


# TODO(markblee): Generalize to other accelerators.
def infer_tpu_type(instance_type: str) -> str:
    """Infers tpu type (e.g. v4-8 or v6e-8-1) from instance type
    (e.g. tpu-v4-8, v4-8, tpu-v6e-8-1 or v6e-8-1)."""
    if not (instance_type and re.fullmatch(r"(tpu-)?v.+-\d+", instance_type)):
        raise NotTpuError(f"Invalid TPU instance: {instance_type}")
    return instance_type.replace("tpu-", "")


# TODO(markblee): Generalize to other accelerators.
def infer_tpu_version(tpu_type: str) -> str:
    """Infer TPU version from the TPU type.

    Args:
        tpu_type: A string of the format {version}-{cores}.

    Returns:
        Inferred TPU version string.

    Raises:
        ValueError: if the TPU version string is unknown.
    """
    if tpu_type.count("-") == 2:
        tpu_type = tpu_type[: tpu_type.rfind("-")]
    tpu_type = infer_tpu_type(tpu_type)
    tpu_version = tpu_type.rsplit("-", 1)[0]  # split from the last occurrence of '-'
    # Resolve aliases like v5e to v5litepod, since in some cases (e.g. aot compilation) v5e is
    # expected.
    tpu_version = _TPU_VERSION_ALIASES.get(tpu_version, tpu_version)
    if tpu_version not in _TPU_VERSIONS:
        raise ValueError(f"Unknown TPU version {tpu_version}. Expected one of {_TPU_VERSIONS}")
    return tpu_version


def infer_xsc_compiler_options(
    *,
    halt_on_detection: bool = True,
    repeat_count: int = 1,
    device_kind: str,
) -> dict[str, Union[str, Any]]:
    """Infers compiler options for running compiled function with XLA SDC check enabled.

    Defaults are as advised by: <andig@google.com>.

    To see additional XSC logging, enable the following environment variables at start time:
    ```bash
    export TPU_MIN_LOG_LEVEL=0
    export TPU_VMODULE=tpu_configuration_ops_impl=3
    export TF_CPP_MIN_LOG_LEVEL=0
    ```

    TODO(tom_gunter): Update with link to documentation once public.

    Args:
        halt_on_detection: Whether to halt the program and raise a Python exception on detection.
        repeat_count: Number of times to repeatedly call the program and validate outputs.
        device_kind: Device kind obtained from `jax.devices()[0].device_kind`.

    Returns:
        A dictionary of compiler options that enable SDC checks.
    """
    options = dict(
        # XLA SDC Checker flags:
        # Enable the SDC checker.
        xla_tpu_enable_sdc_checker=True,
        # Number of times to repeat the function call.
        xla_tpu_sdc_check_repeat_count=repeat_count,
        # Raise Python exception on error.
        xla_tpu_sdc_check_halt_on_detection=halt_on_detection,
        # Duplicate LLO sequences. Required for single-core-per-chip device kind.
        xla_tpu_sdc_replicate_llo=device_kind
        in ["TPU v5e", "TPU v5 lite", "TPU v6e", "TPU v6 lite"],
        # Alternate primary/secondary core for each re-run for platforms with 2 cores per device.
        xla_tpu_sdc_checker_alternate_megacore_cores=True,
        # XLA ICI SDC Checker flags:
        # N.B. ICI checker only runs once after first program compilation.
        # Disable the interconnect checker by default as it is not meant for production run.
        # In a large scale job, disabling it reduced compilation time from 18mins to 15s.
        xla_tpu_ici_sdc_test_run_on_program_start=False,
        # Max distance between send/recv neighbours.
        xla_tpu_ici_sdc_test_max_distance=1,
        # Number of repeated send/recv before checking for equivalence.
        xla_tpu_ici_sdc_test_pipeline_depth=4,
        # Size of the random+checksum buffer to send/recv in 4KiB chunks.
        xla_tpu_ici_sdc_test_buffer_size_chunks=32,
        # Number of packets to split buffer into.
        xla_tpu_ici_sdc_test_packet_size_chunks=4,
        # Number of times to repeat the create-buffer/send/recv/verify loop.
        xla_tpu_ici_sdc_test_iterations=10,
        # Enable LLO log recording which will print performance (bandwith/latency) stats.
        xla_tpu_enable_log_recorder=False,
    )
    return options


_TPU_VERSION_ALIASES = {"v5e": "v5litepod"}
_TPU_VERSIONS = ("v3", "v4", "v5litepod", "v5p", "v6e")


def infer_xla_performance_flags(
    *, mesh_shape: "MeshShape", mesh_axis_names: Sequence[str], device_kind: str
) -> dict[str, str]:
    """Performs automatic XLA flag tuning based on mesh shape and device kind."""
    if device_kind not in ["TPU v6e", "TPU v6 lite"]:
        return {}
    # Sparse core offloading all collectives can improve performance of model parallelism on
    # v6e. However, it negative impacts the performance of some pure FSDP runs by about 6%.
    # Therefore, we enable them selectively on mesh shapes that have model parallelism and are
    # verified to have improved performance with sparse core offloading.
    # TODO(hanzhi-zhou): Check if these flags also improve performance on fsdp=16, model=16.
    mesh_configurations_for_sparse_core_offloading = []
    for a, b in [(32, 8), (64, 4), (16, 8)]:
        mesh_configurations_for_sparse_core_offloading.append(dict(fsdp=a, track=b))
        mesh_configurations_for_sparse_core_offloading.append(dict(fsdp=a, model=b))
    current_configuration = {}
    for name, size in zip(mesh_axis_names, mesh_shape):
        if name in ("fsdp", "track", "model") and size != 1:
            current_configuration[name] = size
    if current_configuration in mesh_configurations_for_sparse_core_offloading:
        flags = dict(
            # Must disable continuation fusion to enable sparse core offloading.
            xla_tpu_enable_async_collective_fusion_fuse_all_gather="false",
            xla_tpu_enable_async_collective_fusion_fuse_all_reduce="false",
            xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter="false",
            xla_tpu_enable_sparse_core_collective_offload_all_gather="true",
            xla_tpu_enable_sparse_core_collective_offload_reduce_scatter="true",
            xla_tpu_enable_sparse_core_collective_offload_all_reduce="true",
            xla_tpu_enable_all_gather_offload_tracing="true",
            xla_tpu_enable_reduce_scatter_offload_tracing="true",
            xla_tpu_enable_all_reduce_offload_tracing="true",
        )
        # 64x4 and 32x8 are non-native mesh shapes for v6e-256. The only native native for v6e-256
        # is 16x16 (or 256). The available bandwidth of non-native mesh shapes is half of that
        # compared to the native mesh shape. Specify the latency modifier so that the latency
        # hiding scheduler can model the actual latency better.
        if math.prod(current_configuration.values()) == 256:
            flags.update(xla_tpu_sparse_core_all_gather_latency_multiplier="2")
        logging.log_first_n(
            logging.INFO,
            "Adding new XLA flags for %s:\n%s",
            1,
            str(current_configuration),
            str(flags),
        )
        return flags
    return {}

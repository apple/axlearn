# Copyright © 2024 Apple Inc.
"""Runtime and compiler options for JAX/XLA."""

# This module must not depend on any jax/axlearn modules so that
# importing this module does not result in initializing jax.
import re
from typing import Any, Dict, Union


def default_xla_options(
    *, instance_type: str, num_slices: int, backend: str
) -> dict[str, Union[str, bool, int]]:
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
            # Change to 16GB. The default is 4GB which is too small for larger models. This
            # cause the step time to be double. You should increase this
            # further if you see "Allocator failed to allocate". A feature
            # to dynamically allocate may come later: b/380514965
            megascale_grpc_premap_memory_bytes=17179869184,
            # Flag controlling the maximum number of overlapping host offloadings.
            xla_tpu_host_transfer_overlap_limit=24,
            # Flag controlling the maximum number of overlapping cross-DCN send/recv.
            xla_max_concurrent_host_send_recv=100,
            # Flag controlling the HBM memory limit as a percentage of the total HBM size.
            # Default value is 95. Can tune up or down to give more or less memory for the
            # scheduler. The scheduler favors more on less memory usage when it's under
            # memory pressure, instead of hiding latency by overlapping more computations
            # and communications.
            xla_tpu_scheduler_percent_shared_memory_limit=90,
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
        )
        options.update(
            # Improved performance for v6e.
            xla_tpu_enable_async_collective_fusion="true",
            xla_tpu_enable_async_collective_fusion_fuse_all_gather="true",
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
            # Sparsecore offloading for all reduce.
            # Uncomment below flags to enable it.
            # xla_sc_disable_megacore_partitioning="true",
            # xla_tpu_use_tc_device_shape_on_sc="true",
            # tpu_use_continuations="true",
            # xla_jf_crs_combiner_threshold_count=10,
            # xla_sc_enable_instruction_fusion="false",
            # xla_sc_disjoint_spmem="false",
            # xla_tpu_enable_sparse_core_collective_offload_all_reduce="true",
        )
        # This flag can be removed after upgrading to Jax 0.4.38.
        # Uncomment for sparsecore offloading.
        # options["2a886c8_chip_config_name"] = "megachip_tccontrol"
    if num_slices > 1:
        # Support multiple TPU slices connected over a data center network.
        options.update(
            # For collectives across multiple slices.
            xla_tpu_enable_megascale_barrier="true",
            # Per rwitten@google.com the following two flags allow gradient all-reduce to happen
            # concurrently with gradient computation for the following layer.
            xla_tpu_enable_data_parallel_all_reduce_opt="true",
            xla_tpu_data_parallel_opt_different_sized_ops="true",
            # Group non-blocking DCN collectives into as few stages as possible.
            xla_tpu_enable_sunk_dcn_allreduce_done_with_host_reduction="true",
        )

    # Validate options. Will never fail if this function is implemented correctly.
    for k, v in options.items():
        try:
            int(v)
            continue
        except ValueError:
            assert v in [True, False, "true", "false", "megachip_tccontrol"], (k, v)

    return options


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


def infer_xsc_compiler_options(
    *,
    halt_on_detection: bool = True,
    repeat_count: int = 1,
    replicate_llo: bool = False,
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
        replicate_llo: LLO sequence duplication, useful for single-core chips (e.g. v5e, v6e).

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
        # Duplicate LLO sequences.
        xla_tpu_sdc_replicate_llo=replicate_llo,
        # Alternate primary/secondary core for each re-run for platforms with 2 cores per device.
        xla_tpu_sdc_checker_alternate_megacore_cores=True,
        # XLA ICI SDC Checker flags:
        # N.B. ICI checker only runs once after first program compilation.
        # Enable the interconnect checker on first program call.
        xla_tpu_ici_sdc_test_run_on_program_start=True,
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


_TPU_VERSIONS = ("v3", "v4", "v5litepod", "v5p", "v6e")

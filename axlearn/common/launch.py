# Copyright © 2023 Apple Inc.

"""A library with common flags to launch a trainer."""
# pylint: disable=wrong-import-position,wrong-import-order
import os
import sys

tpu_type = os.environ.get("TPU_TYPE", "none")

# Set LIBTPU_INIT_ARGS before importing jax!
libtpu_init_args = [
    "--xla_tpu_spmd_rng_bit_generator_unsafe=1",  # SPMD partition-aware RngBitGenerator.
    "--xla_tpu_enable_latency_hiding_scheduler=true",  # Try to schedule ops efficiently.
    "--xla_tpu_perform_spmd_cse_prevention=false",  # b/229655601: prevent OOM on gpt2-small-repeat.
]
if tpu_type.startswith("v4-"):
    libtpu_init_args += [
        # Per maggioni@google.com, the following flags are not supported by V3.
        "--xla_enable_async_all_gather=true",  # Allow async all-gather.
        "--xla_enable_async_collective_permute=true",  # Allow async collective permute.
    ]

num_tpu_slices = int(os.environ.get("NUM_TPU_SLICES", 1))

if num_tpu_slices > 1:
    # Support multiple TPU slices connected over a data center network.
    libtpu_init_args += [
        # For collectives across multiple slices.
        "--xla_tpu_enable_megascale_barrier=true",
        # Per rwitten@google.com the following two flags allow gradient all-reduce to happen
        # concurrently with gradient computation for the following layer.
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
        "--xla_tpu_data_parallel_opt_different_sized_ops=true",
    ]

os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "true"

# Set TF_CPP_MIN_LOG_LEVEL to ignore msg like  "PNG warning: iCCP: known incorrect sRGB profile"
# Reference: https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
# Note: this will disable other TF_CPP info and warnnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import jax before tensorflow else to avoid problems such as:
# tpu_library_init_fns.inc:98] TpuEmbeddingEngine_ExecutePartitioner not available in this library.
import jax  # jax must be imported before tensorflow!

print(
    f"jax version={jax.__version__} tpu_type={tpu_type} num_tpu_slices={num_tpu_slices}",
    file=sys.stderr,
)

import logging as pylogging

from absl import flags, logging

from axlearn.common.utils import get_data_dir
from axlearn.common.utils_spmd import setup as setup_spmd

# pylint: enable=wrong-import-position

flags.DEFINE_string(
    "module",
    None,
    "The trainer config module. "
    "Only configs from the module will be loaded to avoid dependency on other modules.",
    required=True,
)
flags.DEFINE_alias("config_module", "module")
flags.DEFINE_string("config", None, "The trainer config name.", required=True)
flags.DEFINE_string(
    "data_dir",
    None,
    "The tfds directory. "
    "If None, uses env variable DATA_DIR if set, otherwise ~/tensorflow_datasets. "
    "If 'FAKE', uses fake inputs.",
)
flags.DEFINE_integer("jax_profiler_port", None, "If not None, the profiler port.")
flags.DEFINE_string(
    "jax_backend", None, "If not None, ensures that trainer runs on the specified XLA backend."
)
flags.DEFINE_string(
    "distributed_coordinator",
    None,
    "Set this None for tpu backend but it is required for multi-gpu environment",
)
flags.DEFINE_integer(
    "num_processes", None, "Total number of hosts (nodes). Set this None for tpu backend."
)
flags.DEFINE_integer("process_id", None, "Host process id. Set this None for tpu backend.")
# TODO(markblee): Remove this flag.
flags.DEFINE_boolean(
    "filter_info_logs",
    None,
    "If None (default), info log only on process 0 on TPUs, and on all processes on GPUs. "
    "If True, info log only on process 0. "
    "If False, info log on all processes.",
)


FLAGS = flags.FLAGS


def setup():
    # Decide whether to filter logs.
    if FLAGS.filter_info_logs is not None:
        filter_info_logs = FLAGS.filter_info_logs
    else:
        # Infer from platform. For multi-node multi-gpu environment, filtering makes it so that only
        # one process' devices are visible, so we disable it by default.
        filter_info_logs = FLAGS.jax_backend is None or FLAGS.jax_backend != "gpu"

    if filter_info_logs:
        logging.get_absl_handler().addFilter(InfoLogOnlyOnMaster())

    setup_spmd(
        distributed_coordinator=FLAGS.distributed_coordinator,
        num_processes=FLAGS.num_processes,
        process_id=FLAGS.process_id,
        jax_backend=FLAGS.jax_backend,
    )

    if FLAGS.jax_profiler_port is not None:
        # Start jax.profiler for Tensorboard and profiling in open source.
        jax.profiler.start_server(FLAGS.jax_profiler_port)

    devices = jax.devices()
    logging.info("Devices: %s", devices)
    local_devices = jax.local_devices()
    logging.info("Local Devices: %s", local_devices)
    if FLAGS.jax_backend is not None:
        if not devices or not all(device.platform == FLAGS.jax_backend for device in devices):
            raise RuntimeError(f"Expected backend {FLAGS.jax_backend}. Got {devices}.")
    if FLAGS.data_dir:
        # TODO(ruoming): Get rid of --data_dir and use only env var DATA_DIR.
        os.environ["DATA_DIR"] = FLAGS.data_dir
    logging.info("DATA_DIR=%s", get_data_dir())


class InfoLogOnlyOnMaster(pylogging.Filter):
    """Filter to only log levels >= logging.INFO if on master process."""

    def __init__(self, name=""):
        super().__init__(name=name)
        self._jax_pid = jax.process_index()

    def filter(self, record):
        if self._jax_pid != 0:
            return record.levelno < logging.INFO
        return True

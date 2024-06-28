# Copyright © 2023 Apple Inc.

"""A library with common flags to launch a trainer."""

# pylint: disable=wrong-import-position,wrong-import-order
import os
import sys

instance_type = os.environ.get("TPU_TYPE", "none")

# Set LIBTPU_INIT_ARGS before importing jax!
libtpu_init_args = [
    "--xla_tpu_spmd_rng_bit_generator_unsafe=1",  # SPMD partition-aware RngBitGenerator.
    "--xla_tpu_enable_latency_hiding_scheduler=true",  # Try to schedule ops efficiently.
    "--xla_tpu_perform_spmd_cse_prevention=false",  # b/229655601: prevent OOM on gpt2-small-repeat.
]
if instance_type.startswith("v4-"):
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

# Set TF_CPP_MIN_LOG_LEVEL to ignore msg like  "PNG warning: iCCP: known incorrect sRGB profile"
# Reference: https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
# Note: this will disable other TF_CPP info and warnnings.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Import jax before tensorflow else to avoid problems such as:
# tpu_library_init_fns.inc:98] TpuEmbeddingEngine_ExecutePartitioner not available in this library.
import jax  # jax must be imported before tensorflow!

print(f"jax version={jax.__version__}", file=sys.stderr)
if instance_type != "none":
    print(f"instance_type={instance_type} num_slices={num_tpu_slices}", file=sys.stderr)

from absl import flags, logging

from axlearn.common.status_server import StatusHTTPServer
from axlearn.common.utils import get_data_dir
from axlearn.common.utils_spmd import setup as setup_spmd

# pylint: enable=wrong-import-position

flags.DEFINE_string(
    "data_dir",
    None,
    "The tfds directory. "
    "If None, uses env variable DATA_DIR if set, otherwise ~/tensorflow_datasets. "
    "If 'FAKE', uses fake inputs.",
)
flags.DEFINE_integer("jax_profiler_port", None, "If not None, the profiler port.")
flags.DEFINE_integer("status_port", None, "If not None, the status server port.")
flags.DEFINE_string("jax_backend", None, "Specifies the XLA backend to use.", required=True)
flags.DEFINE_string(
    "distributed_coordinator",
    os.environ.get("DISTRIBUTED_COORDINATOR", None),
    "Distributed coordinator IP address. Must be None on tpu, otherwise required.",
)
flags.DEFINE_integer(
    "initialization_timeout",
    None,
    "Distributed initialization timeout in seconds. If None, uses jax default.",
)
flags.DEFINE_integer(
    "num_processes",
    os.environ.get("NUM_PROCESSES", None),
    "Total number of hosts (nodes). Must be None on tpu, otherwise required.",
)
flags.DEFINE_integer(
    "process_id",
    os.environ.get("PROCESS_ID", None),
    "Rank of the current process. Must be None on tpu, otherwise required.",
)


FLAGS = flags.FLAGS


def setup():
    setup_spmd(
        distributed_coordinator=FLAGS.distributed_coordinator,
        num_processes=FLAGS.num_processes,
        process_id=FLAGS.process_id,
        jax_backend=FLAGS.jax_backend,
        initialization_timeout=FLAGS.initialization_timeout,
    )

    if FLAGS.jax_profiler_port is not None:
        # Start jax.profiler for Tensorboard and profiling in open source.
        jax.profiler.start_server(FLAGS.jax_profiler_port)

    if FLAGS.status_port is not None:
        status_server = StatusHTTPServer(FLAGS.status_port)
        status_server.start()

    devices = jax.devices()
    logging.info("Devices: %s", devices)
    local_devices = jax.local_devices()
    logging.info("Local Devices: %s", local_devices)
    if not devices or not all(device.platform == FLAGS.jax_backend for device in devices):
        raise RuntimeError(f"Expected backend {FLAGS.jax_backend}. Got {devices}.")
    if FLAGS.data_dir:
        # TODO(ruoming): Get rid of --data_dir and use only env var DATA_DIR.
        os.environ["DATA_DIR"] = FLAGS.data_dir
    logging.info("DATA_DIR=%s", get_data_dir())

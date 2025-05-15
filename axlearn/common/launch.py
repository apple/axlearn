# Copyright © 2023 Apple Inc.

"""A library with common flags to launch a trainer."""

import contextlib
import importlib
import os
import sys

# pylint: disable-next=ungrouped-imports
from axlearn.common import compiler_options

# pylint: disable=wrong-import-position,wrong-import-order


instance_type = os.environ.get("TPU_TYPE", "none")
num_tpu_slices = int(os.environ.get("NUM_TPU_SLICES", 1))

# Set LIBTPU_INIT_ARGS before importing jax!
tpu_flags_exc = None
try:
    libtpu_init_options = compiler_options.default_xla_options(
        instance_type=instance_type, num_slices=num_tpu_slices, backend="tpu"
    )
    os.environ["LIBTPU_INIT_ARGS"] = compiler_options.xla_flags_from_options(libtpu_init_options)
except compiler_options.NotTpuError as e:
    # Log this when setup() is called.
    tpu_flags_exc = e

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
flags.DEFINE_multi_string(
    "init_module",
    [],
    "Zero or more init modules to import prior to setting up JAX distributed. "
    "Each flag value should be a string containing 'module_path' or 'module_path:spec', e.g. "
    "'axlearn.cloud.gcp.tpu_health_check' or 'axlearn.cloud.gcp.tpu_health_check:output_dir=...'.\n"
    "The module should expose a public function `setup`, a context manager exposing pre- and post-"
    "SPMD setup logic which is entered prior to `setup_spmd` and exited immediately afterwards.\n"
    "The spec (if provided) will be provided to `module.setup(spec)` and therefore can be "
    "implementation dependent. Not specifying a spec is equivalent to passing `None` to `setup`.\n"
    "If specifying multiple modules, each `setup` context is entered in the given order.",
)

FLAGS = flags.FLAGS


# Kept separate for easier testing.
@contextlib.contextmanager
def _init_context(fv: flags.FlagValues = FLAGS):
    with contextlib.ExitStack() as ctx:
        for module_spec in fv.init_module:
            parts = module_spec.split(":", maxsplit=1) + [None]
            module, spec = parts[:2]
            ctx.enter_context(importlib.import_module(module).setup(spec))
        yield


def setup():
    if tpu_flags_exc is not None:
        logging.info("LIBTPU_INIT_FLAGS was not set. Reason: %s", tpu_flags_exc)
    else:
        logging.info("LIBTPU_INIT_ARGS='%s'", os.environ["LIBTPU_INIT_ARGS"])

    with _init_context():
        if FLAGS.jax_backend == "proxy":
            # pylint: disable-next=import-error,import-outside-toplevel
            import pathwaysutils  # pytype: disable=import-error

            pathwaysutils.initialize()
        else:
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

    if FLAGS.jax_backend != "proxy":
        if not devices or not all(device.platform == FLAGS.jax_backend for device in devices):
            raise RuntimeError(f"Expected backend {FLAGS.jax_backend}. Got {devices}.")

    if FLAGS.data_dir:
        # TODO(ruoming): Get rid of --data_dir and use only env var DATA_DIR.
        os.environ["DATA_DIR"] = FLAGS.data_dir
    logging.info("DATA_DIR=%s", get_data_dir())

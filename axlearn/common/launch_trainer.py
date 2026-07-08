# Copyright © 2023 Apple Inc.

"""Utilities to launch a trainer."""

import json
import os
import time
from typing import Any, Optional

import jax
from absl import flags, logging

from axlearn.common import file_system as fs
from axlearn.common import measurement
from axlearn.common.config import TrainerConfigFn, get_named_trainer_config
from axlearn.common.trainer import SpmdTrainer, select_mesh_config, sync_restore_class_vars, sync_store_class_vars
from axlearn.common.utils import MeshShape, get_data_dir, infer_mesh_shape, live_devices, set_elastic_manager
from pathwaysutils.elastic import manager, elastic
from pathwaysutils.debug import watchdog


# Trainer-specific flags.
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
    "trainer_dir",
    None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True,
)
flags.DEFINE_integer(
    "trainer_prng_seed",
    0,
    "The seed for jax.random.PRNGKey(). "
    "Used for initializing model parameters and pseudo-random number generation during training.",
)
flags.DEFINE_list("trace_at_steps", [], "Step numbers to start a 3-step profile at.")
flags.DEFINE_integer(
    "n_steps_for_each_trace",
    None,
    "Number of consecutive steps covered by each trace. If None, defaults to 3.",
)
flags.DEFINE_enum(
    "tpu_trace_mode",
    None,
    ["TRACE_ONLY_HOST", "TRACE_ONLY_XLA", "TRACE_COMPUTE", "TRACE_COMPUTE_AND_SYNC"],
    "TPU trace mode. If None, defaults to TRACE_ONLY_XLA. "
    "See https://docs.jax.dev/en/latest/profiling.html#tpu-options. ",
)
flags.DEFINE_enum(
    "host_tracer_level",
    None,
    ["0", "1", "2", "3"],
    "Host tracer level. Higher levels capture more host-side activity. "
    "If None, defaults to 2. See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_enum(
    "device_tracer_level",
    None,
    ["0", "1"],
    "Device tracer level. If None, defaults to 1. "
    "See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_enum(
    "python_tracer_level",
    None,
    ["0", "1"],
    "Python tracer level. If None, defaults to 0. "
    "See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_list(
    "eval_trace_at_iters",
    [],
    "Evaluation iters to trace with the profiler each time the evaler is run. "
    "Each trace covers one eval batch. "
    "Traces will run for at most 3 unique steps.",
)
flags.DEFINE_integer(
    "trainer_watchdog_timeout_seconds",
    3600,
    "Timeout for the trainer watchdog in seconds. "
    "If the trainer.step does not increment within this interval, "
    "the watchdog will log the stack traces of all threads.",
)
flags.DEFINE_integer(
    "trainer_crash_on_hang_timeout_seconds",
    7200,
    "Timeout for crashing the trainer on hang in seconds. "
    "If the trainer hangs for longer than this interval, "
    "the trainer will crash to prevent indefinite hanging.",
)
flags.DEFINE_integer(
    "trainer_log_every_n_steps",
    None,
    "Logging frequency for the loss value during training. If None, defaults to every 100 steps.",
)
flags.DEFINE_enum(
    "device_monitor",
    "none",
    ["none", "tpu", "gpu"],
    "Whether to enable the device monitor. "
    "The device monitor collects the system metrics and logs them periodically. "
    "The device monitor also logs the idle status of the devices on the host, "
    "and trigger a watchdog if the devices are idle for 10 minutes.",
)
flags.DEFINE_string(
    "mesh_selector",
    None,
    "The mesh selector string. See `SpmdTrainer.Config.mesh_rules` for details.",
)

FLAGS = flags.FLAGS

elastic_snapshotting_enabled = True


def get_trainer_config(
    trainer_config_fn: Optional[TrainerConfigFn] = None,
    *,
    flag_values: flags.FlagValues = FLAGS,
) -> SpmdTrainer.Config:
    if trainer_config_fn is None:
        # Attempt a direct import. This is a common case for launching from pip package.
        try:
            trainer_config_fn = get_named_trainer_config(
                flag_values.config,
                config_module=flag_values.config_module,
            )
        except (ImportError, AttributeError, KeyError):
            logging.info(
                "Did not find config '%s' or module '%s' -- will continue searching.",
                flag_values.config,
                flag_values.config_module,
            )
            # Fallback to original strategy of importing from axlearn.experiments below.
            trainer_config_fn = None

    if trainer_config_fn is None:
        trainer_config_fn = get_named_trainer_config(
            flag_values.config,
            config_module=f"axlearn.experiments.{flag_values.config_module}",
        )
    trainer_config: SpmdTrainer.Config = trainer_config_fn()
    trainer_config.dir = trainer_config.dir or flag_values.trainer_dir
    if flag_values.mesh_selector is not None:
        select_mesh_config(trainer_config, mesh_selector=flag_values.mesh_selector)
    trainer_config.mesh_axis_names = trainer_config.mesh_axis_names or ("data", "model")
    #trainer_config.mesh_shape = trainer_config.mesh_shape or (len(jax.devices()), 1)
    trainer_config.mesh_shape = trainer_config.mesh_shape or (len(live_devices()), 1)
    if isinstance(trainer_config.mesh_shape, MeshShape):
        trainer_config.mesh_shape = infer_mesh_shape(trainer_config.mesh_shape)
    trainer_config.start_trace_steps = [int(el) for el in flag_values.trace_at_steps]
    if flag_values["n_steps_for_each_trace"].present:
        trainer_config.n_steps_for_each_trace = int(flag_values.n_steps_for_each_trace)
    if flag_values["tpu_trace_mode"].present:
        trainer_config.tpu_trace_mode = flag_values.tpu_trace_mode
    if flag_values["host_tracer_level"].present:
        trainer_config.host_tracer_level = int(flag_values.host_tracer_level)
    if flag_values["device_tracer_level"].present:
        trainer_config.device_tracer_level = int(flag_values.device_tracer_level)
    if flag_values["python_tracer_level"].present:
        trainer_config.python_tracer_level = int(flag_values.python_tracer_level)
    if trainer_config.watchdog_timeout_seconds is None:
        trainer_config.watchdog_timeout_seconds = flag_values.trainer_watchdog_timeout_seconds
    if trainer_config.crash_on_hang_timeout_seconds is None:
        trainer_config.crash_on_hang_timeout_seconds = (
            flag_values.trainer_crash_on_hang_timeout_seconds
        )
    if trainer_config.log_every_n_steps is None:
        trainer_config.log_every_n_steps = flag_values.trainer_log_every_n_steps
    for eval_cfg in trainer_config.evalers.values():
        eval_cfg.trace_at_iters = [int(el) for el in flag_values.eval_trace_at_iters]
    if flag_values.device_monitor == "tpu":
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from axlearn.cloud.gcp.monitoring.tpu_device_monitor import create_tpu_monitor

        trainer_config.device_monitor = create_tpu_monitor()
    elif flag_values.device_monitor == "gpu":
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from axlearn.common.monitoring.gpu_device_monitor import create_gpu_monitor

        trainer_config.device_monitor = create_gpu_monitor()
    if hasattr(trainer_config.checkpointer, "trainer_dir"):
        # Set trainer_dir if not already set.
        if not isinstance(trainer_config.checkpointer.trainer_dir, str):
            trainer_config.checkpointer.trainer_dir = trainer_config.dir
    return trainer_config


def is_retryable_error(e: Exception) -> bool:
    if isinstance(e, jax.errors.JaxRuntimeError):
        err_str = str(e)
        if elastic.is_error_due_to_slice_down(e):
            return True
        if "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            return True
    return False


def run_trainer(trainer_config: SpmdTrainer.Config) -> Any:
    measurement.record_event(measurement.Event.START_JOB)
    trainer_config_debug_string = trainer_config.debug_string()
    logging.info("Trainer config:\n%s", trainer_config_debug_string)
    if jax.process_index() == 0:
        trainer_config_file = os.path.join(trainer_config.dir, "trainer_config")
        with fs.open(trainer_config_file, "w") as f:
            f.write(trainer_config_debug_string)

        config_file = os.path.join(trainer_config.dir, "launch_trainer_flags")
        with fs.open(config_file, "w") as f:
            json.dump(  # pytype: disable=wrong-arg-types
                {
                    **FLAGS.flag_values_dict(),
                    "data_dir": get_data_dir(),
                },
                f,
            )
    
    elastic_manager = None
    elastic_manager_initialized = False

    output = None
    jax_device_state = {}
    python_vars = {}
    immutable_data = {}
    trainer = None
    while True:
        try:
            if not elastic_manager_initialized:
                if elastic_snapshotting_enabled:
                    logging.info("[Elastic] Initializing elastic manager...")
                    elastic_manager = manager.Manager()
                    set_elastic_manager(elastic_manager)
                    logging.info("[Elastic] Elastic manager initialized.")
                else:
                    logging.info("[Elastic] Elastic snapshotting disabled or not supported (no slice_index).")
                elastic_manager_initialized = True

            clean_trainer: SpmdTrainer = trainer_config.instantiate(parent=None)

            if elastic_manager and elastic_manager.new_slice_event.is_set():
                logging.info("[Elastic] New slice event is set. Restoring from snapshot...")
                elastic_manager.new_slice_event.clear()
                trainer, prng_key = sync_restore_class_vars(clean_trainer, jax_device_state, python_vars, immutable_data)
            else:
                logging.info("[Elastic] Starting fresh trainer (no elastic recovery triggered).")
                trainer = clean_trainer
                prng_key = jax.random.PRNGKey(seed=FLAGS.trainer_prng_seed)

            output = trainer.run(prng_key)
            measurement.record_event(measurement.Event.END_JOB)
            break
            
        except jax.errors.JaxRuntimeError as e:
            if is_retryable_error(e):
                logging.warning("Caught retryable error: %s. Retrying...", e)
                if trainer is not None:
                    jax_device_state = getattr(trainer, "_jax_device_state", {})
                    python_vars = getattr(trainer, "_python_vars", {})
                    immutable_data = getattr(trainer, "_immutable_data", {})
                if elastic_manager:
                    elastic_manager.new_slice_event.set()
                time.sleep(10)
                continue
            else:
                raise e
    return output

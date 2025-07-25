# Copyright © 2023 Apple Inc.

"""Utilities to launch a trainer."""

import contextlib
import json
import os
from typing import Any, Optional

import jax
from absl import flags, logging

from axlearn.common import file_system as fs
from axlearn.common import measurement
from axlearn.common.trainer import SpmdTrainer, select_mesh_config
from axlearn.common.utils import MeshShape, get_data_dir, infer_mesh_shape
from axlearn.experiments import TrainerConfigFn, get_named_trainer_config

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
    trainer_config.mesh_shape = trainer_config.mesh_shape or (len(jax.devices()), 1)
    if isinstance(trainer_config.mesh_shape, MeshShape):
        trainer_config.mesh_shape = infer_mesh_shape(trainer_config.mesh_shape)
    trainer_config.start_trace_steps = [int(el) for el in flag_values.trace_at_steps]
    if trainer_config.watchdog_timeout_seconds is None:
        trainer_config.watchdog_timeout_seconds = flag_values.trainer_watchdog_timeout_seconds
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


def _run_trainer_impl(trainer_config: SpmdTrainer.Config) -> Any:
    """Instantiates and runs the trainer."""
    trainer_config_debug_string = trainer_config.debug_string()
    logging.info("Trainer config:\n%s", trainer_config_debug_string)
    if jax.process_index() == 0:
        trainer_config_file = os.path.join(trainer_config.dir, "trainer_config")
        with fs.open(trainer_config_file, "w") as f:
            f.write(trainer_config_debug_string)

        config_file = os.path.join(trainer_config.dir, "launch_trainer_flags")
        with fs.open(config_file, "w") as f:
            json.dump(
                {
                    **FLAGS.flag_values_dict(),
                    "data_dir": get_data_dir(),
                },
                f,
            )

    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(seed=FLAGS.trainer_prng_seed)
    return trainer.run(prng_key)


def run_trainer(trainer_config: SpmdTrainer.Config) -> Any:
    recorder = measurement.global_recorder
    job_events_manager = (
        recorder.record_event(measurement.Event.JOB) if recorder else contextlib.nullcontext()
    )
    with job_events_manager:
        return _run_trainer_impl(trainer_config)

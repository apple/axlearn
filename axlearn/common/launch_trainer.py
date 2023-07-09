"""Utilities to launch a trainer."""
import os
from typing import Any, Optional

import jax
import tensorflow as tf
from absl import flags, logging

from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments import TrainerConfigFn, get_named_trainer_config

# Trainer-specific flags.
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

FLAGS = flags.FLAGS


def get_trainer_config(trainer_config_fn: Optional[TrainerConfigFn] = None) -> SpmdTrainer.Config:
    if trainer_config_fn is None:
        # Load from known experiments and provided flags.
        trainer_config_fn: TrainerConfigFn = get_named_trainer_config(
            FLAGS.config,
            config_module=FLAGS.config_module,
            root_module="axlearn",
        )
    trainer_config: SpmdTrainer.Config = trainer_config_fn()
    trainer_config.dir = trainer_config.dir or FLAGS.trainer_dir
    trainer_config.mesh_axis_names = trainer_config.mesh_axis_names or ("data", "model")
    trainer_config.mesh_shape = trainer_config.mesh_shape or (len(jax.devices()), 1)
    trainer_config.start_trace_steps = [int(el) for el in FLAGS.trace_at_steps]

    for eval_cfg in trainer_config.evalers.values():
        eval_cfg.trace_at_iters = [int(el) for el in FLAGS.eval_trace_at_iters]

    return trainer_config


def run_trainer(trainer_config: SpmdTrainer.Config) -> Any:
    trainer_config_debug_string = trainer_config.debug_string()
    logging.info("Trainer config:\n%s", trainer_config_debug_string)
    if jax.process_index() == 0:
        trainer_config_file = os.path.join(trainer_config.dir, "trainer_config")
        with tf.io.gfile.GFile(trainer_config_file, "w") as f:  # type: ignore
            f.write(trainer_config_debug_string)

    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(seed=FLAGS.trainer_prng_seed)
    return trainer.run(prng_key)

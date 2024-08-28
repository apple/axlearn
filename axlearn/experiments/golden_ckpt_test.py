# Copyright © 2023 Apple Inc.

"""A unittest for checkpoint backwards-compatibility."""
# pylint: disable=no-self-use
import dataclasses
import os.path
import resource
import shutil
from collections.abc import Sequence
from typing import Optional

import jax.random
from absl import flags
from absl.testing import absltest

from axlearn.common.inference import InferenceRunner
from axlearn.common.summary_writer import NoOpWriter
from axlearn.common.test_utils import TestCase
from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments import get_named_trainer_config

flags.DEFINE_boolean("update_golden_checkpoints", False, "If true, update golden config files.")

FLAGS = flags.FLAGS


@dataclasses.dataclass
class TrainerConfigSpec:
    module: str
    name: str
    trainer_dir: Optional[str] = None


def _update_golden_checkpoints():
    try:
        return FLAGS.update_golden_checkpoints
    except flags.UnparsedFlagAccessError:
        return False


def named_parameters(
    trainer_configs: Sequence[TrainerConfigSpec],
) -> list[tuple[str, TrainerConfigSpec]]:
    return [(config_spec.name, config_spec) for config_spec in trainer_configs]


class GoldenCheckpointTest(TestCase):
    """Tests against golden checkpoints."""

    @property
    def root_module(self):
        return "axlearn.experiments"

    def _test_golden_checkpoint(self, config_spec: TrainerConfigSpec):
        # Temporarily raises the soft limit of open files to the hard limit.
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

        cfg_fn = get_named_trainer_config(
            config_spec.name, config_module=f"{self.root_module}.{config_spec.module}"
        )
        cfg: SpmdTrainer.Config = cfg_fn()
        cfg.summary_writer = NoOpWriter.default_config()
        for eval_cfg in cfg.evalers.values():
            eval_cfg.summary_writer = NoOpWriter.default_config()
        mesh_axis_names = cfg.mesh_axis_names or ("data", "model")
        cfg.mesh_axis_names = mesh_axis_names
        mesh_shape = cfg.mesh_shape or (len(jax.devices()), 1)
        cfg.mesh_shape = mesh_shape
        cfg.dir = config_spec.trainer_dir or os.path.join(
            os.path.dirname(__file__), "testdata", "checkpoints", config_spec.name
        )
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        step = 0
        with trainer.mesh():
            if _update_golden_checkpoints():
                trainer.init(jax.random.PRNGKey(0))
                # Remove the existing checkpoint directory.
                if os.path.isdir(cfg.dir):
                    shutil.rmtree(cfg.dir)
                trainer.checkpointer.save(step=step, state=trainer.trainer_state)
                trainer.checkpointer.stop()
            else:
                trainer.restore_checkpoint(step)
                # Check that we can also restore into an inference runner.
                inference_runner_cfg = InferenceRunner.default_config().set(
                    name=cfg.name + "_inference_runner",
                    mesh_shape=mesh_shape,
                    mesh_axis_names=mesh_axis_names,
                    model=cfg.model,
                    inference_dtype=cfg.model.dtype,
                )
                inference_runner_cfg.init_state_builder.set(
                    dir=os.path.join(cfg.dir, "checkpoints", f"step_{step:08d}")
                )
                runner = inference_runner_cfg.instantiate(parent=None)
                self.assertNestedEqual(
                    runner.inference_runner_state.prng_key, trainer.trainer_state.prng_key
                )
                self.assertNestedEqual(
                    runner.inference_runner_state.model, trainer.trainer_state.model
                )


if __name__ == "__main__":
    absltest.main()

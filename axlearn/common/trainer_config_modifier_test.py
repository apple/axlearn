# Copyright © 2024 Apple Inc.

"""Test various ConfigModifier classes in trainer_config_modifier.py."""

import jax
from absl.testing import absltest

from axlearn.common import test_utils
from axlearn.common.base_layer import RematSpec
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    GradientAccumulationModifier,
    MeshShapeModifier,
    RematSpecModifier,
)
from axlearn.common.trainer_test import DummyModel


class GradientAccumulationModifierTest(test_utils.TestCase):
    def test_gradient_accumulation_override(self):
        cfg = SpmdTrainer.default_config().set(model=DummyModel.default_config())
        cfg_modifier = (
            GradientAccumulationModifier.default_config().set(grad_acc_steps=4).instantiate()
        )
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.learner.forward_fn_transformation.steps, 4)


class RematSpecModifierTest(test_utils.TestCase):
    def test_remat_policy_override(self):
        cfg = SpmdTrainer.default_config().set(model=DummyModel.default_config())
        cfg_modifier = (
            RematSpecModifier.default_config()
            .set(
                remat_policies={
                    "model.linear": RematSpec(
                        prevent_cse=True,
                        policy=jax.ad_checkpoint.checkpoint_policies.dots_saveable,
                    ),
                }
            )
            .instantiate()
        )
        cfg = cfg_modifier(cfg)
        self.assertRegex(str(cfg.model.linear), "dots_saveable")
        cfg_modifier = (
            RematSpecModifier.default_config()
            .set(
                remat_policies={
                    "model.linear": RematSpec(
                        prevent_cse=True,
                        policy=jax.ad_checkpoint.checkpoint_policies.dots_saveable,
                    ),
                    "model.unknown": RematSpec(
                        prevent_cse=True,
                        policy=jax.ad_checkpoint.checkpoint_policies.dots_saveable,
                    ),
                }
            )
            .instantiate()
        )
        # Ensure that the exception is working.
        with self.assertRaisesRegex(ValueError, "unknown is not found in.*"):
            _ = cfg_modifier(cfg)


class MeshShapeModifierTest(test_utils.TestCase):
    def test_mesh_shape_update(self):
        cfg = SpmdTrainer.default_config().set(model=DummyModel.default_config())
        cfg_modifier = MeshShapeModifier.default_config().set(mesh_shape=(4, 1, 8, 1)).instantiate()
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.mesh_shape, (4, 1, 8, 1))


class ChainConfigModifierTest(test_utils.TestCase):
    def test_chain_config_modifier(self):
        cfg = SpmdTrainer.default_config().set(model=DummyModel.default_config())
        cfg_modifier = (
            ChainConfigModifier.default_config()
            .set(
                config_modifiers=[
                    GradientAccumulationModifier.default_config().set(grad_acc_steps=4),
                    MeshShapeModifier.default_config().set(mesh_shape=(4, 1, 8, 1)),
                ]
            )
            .instantiate()
        )
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.mesh_shape, (4, 1, 8, 1))
        self.assertEqual(cfg.learner.forward_fn_transformation.steps, 4)


if __name__ == "__main__":
    absltest.main()

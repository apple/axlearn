# Copyright © 2024 Apple Inc.

"""Test various ConfigModifier classes in trainer_config_modifier.py."""

import jax
from absl.testing import absltest, parameterized

from axlearn.common import causal_lm, test_utils
from axlearn.common.attention import RepeatedTransformerLayer, StackedTransformerLayer
from axlearn.common.base_layer import RematSpec
from axlearn.common.config import config_for_function
from axlearn.common.optimizers import sgd_optimizer
from axlearn.common.quantized_dot_general.layers import get_all_fp8_param_names
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    FP8ConfigModifier,
    GradientAccumulationModifier,
    MeshShapeModifier,
    ModuleConfigModifier,
    OverrideInplaceUpdateTransformation,
    PartitionSpecModifier,
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
        with self.assertRaisesRegex(AttributeError, r"unknown \(keys are *"):
            _ = cfg_modifier(cfg)


class ModuleConfigModifierTest(test_utils.TestCase):
    def test_model_config_override(self):
        cfg = SpmdTrainer.default_config().set(model=causal_lm.Model.default_config())
        self.assertTrue(
            str(cfg.model.decoder.transformer) == str(StackedTransformerLayer.default_config())
        )

        cfg_modifier = (
            ModuleConfigModifier.default_config()
            .set(
                target_config="model.decoder.transformer",
                modification=RepeatedTransformerLayer.default_config(),
            )
            .instantiate()
        )

        cfg = cfg_modifier(cfg)
        # The default StackedTransformerLayer should have changed to RepeatedTransformerLayer
        self.assertTrue(
            str(cfg.model.decoder.transformer) == str(RepeatedTransformerLayer.default_config())
        )
        cfg_modifier = (
            ModuleConfigModifier.default_config()
            .set(
                target_config="model.decoder.unknown",
                modification=RepeatedTransformerLayer.default_config(),
            )
            .instantiate()
        )
        # Ensure that the exception is working.
        with self.assertRaisesRegex(AttributeError, r"unknown \(keys are *"):
            _ = cfg_modifier(cfg)


class PartitionSpecModifierTest(test_utils.TestCase):
    def test_partition_spec_override(self):
        cfg = SpmdTrainer.default_config().set(model=DummyModel.default_config())
        cfg_modifier = (
            PartitionSpecModifier.default_config()
            .set(
                partition_specs={
                    "model.linear": {"param_partition_spec": ("model", ("expert", "fsdp", "seq"))},
                },
            )
            .instantiate()
        )
        cfg = cfg_modifier(cfg)
        self.assertTrue(
            str(cfg.model.linear.param_partition_spec), """("model", ("expert", "fsdp", "seq")"""
        )
        cfg_modifier = (
            PartitionSpecModifier.default_config()
            .set(
                partition_specs={
                    "model.linear": {"param_partition_spec": ("model", ("expert", "fsdp", "seq"))},
                    "model.unknown": {"param_partition_spec": ("model", ("expert", "fsdp", "seq"))},
                },
            )
            .instantiate()
        )
        # Ensure that the exception is working.
        with self.assertRaisesRegex(AttributeError, r"unknown \(keys are *"):
            _ = cfg_modifier(cfg)

        cfg_modifier = (
            PartitionSpecModifier.default_config()
            .set(
                partition_specs={
                    "model.linear": {
                        "param_partition_spec": ("model", ("expert", "fsdp", "seq")),
                        "unknown_partition_spec": ("model", ("expert", "fsdp", "seq")),
                    },
                },
            )
            .instantiate()
        )
        with self.assertRaisesRegex(AttributeError, "unknown_partition_spec *"):
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


class FP8ConfigModifierTest(test_utils.TestCase):
    @parameterized.parameters([True, False])
    def test_fp8_config_modifier(self, use_config_fn):
        cfg: SpmdTrainer.Config = SpmdTrainer.default_config().set(
            model=DummyModel.default_config()
        )
        if use_config_fn:
            cfg.learner.optimizer = config_for_function(sgd_optimizer).set(
                learning_rate=0.5,
                decouple_weight_decay=True,
            )
        else:
            cfg.learner.optimizer = sgd_optimizer(
                learning_rate=0.5,
                decouple_weight_decay=True,
            )

        cfg_modifier = (
            FP8ConfigModifier.default_config().set(fp8_amax_history_length=1).instantiate()
        )
        cfg = cfg_modifier(cfg)

        self.assertIsInstance(cfg.learner.optimizer, OverrideInplaceUpdateTransformation.Config)
        self.assertEqual(
            cfg.learner.optimizer.rules,
            [f".*/{x}" for x in get_all_fp8_param_names()],
        )
        self.assertEqual(cfg.model.linear.quantized_dot_general.fp8_amax_history_length, 1)


if __name__ == "__main__":
    absltest.main()

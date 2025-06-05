# Copyright © 2023 Apple Inc.

"""Tests trainer config utilities."""
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.flash_attention.layer import FlashBlockSizeModifier
from axlearn.common.flash_attention.layer_test import DummyModel as FlashDummyModel
from axlearn.common.flash_attention.layer_test import FlashAttention
from axlearn.common.input_fake import FakeLmInput
from axlearn.common.test_utils import mock_trainer_config
from axlearn.common.trainer_test import DummyModel
from axlearn.experiments import TrainerConfigFn
from axlearn.experiments.trainer_config_utils import V6eFlashConfigModifier, with_overrides


def _create_fake_trainer_config_fn() -> TrainerConfigFn:
    def fn():
        return mock_trainer_config(
            input_config=FakeLmInput.default_config().set(
                global_batch_size=8,
                source_length=16,
            ),
            model_config=DummyModel.default_config().set(dtype=jnp.float32),
        )

    return fn


class TrainerConfigUtilsTest(parameterized.TestCase):
    """Tests trainer config utils."""

    @parameterized.parameters(
        {"dir": "abc", "mesh_shape": (8, 16)},
        {"name": "new name"},
        {"model": DummyModel.default_config().set(dtype=jnp.bfloat16)},
    )
    def test_with_overrides(self, **kwargs):
        dummy_trainer_config_fn = _create_fake_trainer_config_fn()
        new_trainer_config_fn = with_overrides(dummy_trainer_config_fn, **kwargs)
        new_trainer_config = new_trainer_config_fn()
        for k, v in kwargs.items():
            self.assertEqual(getattr(new_trainer_config, k), v)

    def test_flash_config_modifier(self):
        cfg: FlashDummyModel.Config = FlashDummyModel.default_config()
        cfg.layer = FlashAttention.default_config()
        cfg_modifier = V6eFlashConfigModifier.default_config().instantiate()
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.layer.tpu_block_size, 1024)

    def test_gpu_flash_config_modifier(self):
        cfg: FlashDummyModel.Config = FlashDummyModel.default_config()
        cfg.layer = FlashAttention.default_config()
        cfg_modifier = FlashBlockSizeModifier.default_config().set(gpu_block_size=64).instantiate()
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.layer.gpu_block_size, 64)


if __name__ == "__main__":
    absltest.main()

# Copyright © 2023 Apple Inc.

"""Tests trainer config utilities."""
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.input_fake import FakeLmInput
from axlearn.common.test_utils import mock_trainer_config
from axlearn.common.trainer_test import DummyModel
from axlearn.experiments import TrainerConfigFn
from axlearn.experiments.trainer_config_utils import with_overrides


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


if __name__ == "__main__":
    absltest.main()

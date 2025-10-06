# Copyright © 2023 Apple Inc.

"""Tests trainer config utilities."""

import threading
from typing import Optional

import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    TrainerConfigFn,
    config_class,
    config_for_function,
    with_overrides,
)
from axlearn.common.flash_attention.layer import FlashBlockSizeModifier
from axlearn.common.flash_attention.layer_test import DummyModel as FlashDummyModel
from axlearn.common.flash_attention.layer_test import FlashAttention
from axlearn.common.input_fake import FakeLmInput
from axlearn.common.test_utils import mock_trainer_config
from axlearn.common.trainer_test import DummyModel
from axlearn.experiments.trainer_config_utils import (
    SplashAttentionConfigModifier,
    V6eFlashConfigModifier,
    V7xFlashConfigModifier,
    _DeepCopyWithClosureFnWrapper,
    _wrap_with_deep_copy_with_closure,
    config_map_cache,
)


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


@config_class
class _ConfigA(ConfigBase):
    """A dummy config class."""

    data: list[str] = []


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

    def test_v6e_flash_config_modifier(self):
        cfg: FlashDummyModel.Config = FlashDummyModel.default_config()
        cfg.layer = FlashAttention.default_config()
        cfg_modifier = V6eFlashConfigModifier.default_config().instantiate()
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.layer.tpu_block_size, 1024)

    def test_v7x_flash_config_modifier(self):
        cfg: FlashDummyModel.Config = FlashDummyModel.default_config()
        cfg.layer = FlashAttention.default_config()
        cfg_modifier = V7xFlashConfigModifier.default_config().instantiate()
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.layer.tpu_block_size, 2048)

    def test_gpu_flash_config_modifier(self):
        cfg: FlashDummyModel.Config = FlashDummyModel.default_config()
        cfg.layer = FlashAttention.default_config()
        cfg_modifier = FlashBlockSizeModifier.default_config().set(gpu_block_size=64).instantiate()
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.layer.gpu_block_size, 64)

    def test_splash_attention_config_modifier(self):
        cfg: FlashDummyModel.Config = FlashDummyModel.default_config()
        cfg.layer = FlashAttention.default_config()
        cfg_modifier = (
            SplashAttentionConfigModifier.default_config().set(splash_block_q=4096).instantiate()
        )
        cfg = cfg_modifier(cfg)
        self.assertEqual(cfg.layer.backend_overrides["splash_block_q"], 4096)


class DeepCopyWithClosureFnWrapperTest(parameterized.TestCase):
    """Test that the custom deepcopy with closure work as expected."""

    def test_serialization_with_closure(self):
        """Test deepcopy with closures will not have shared mutable states."""
        # Create a chain function where the inner function modifies a mutable object from its
        # closure.
        shared_state = []

        def f():
            shared_state.append(1)

        def g():
            f()

        _DeepCopyWithClosureFnWrapper(g)()
        self.assertEqual(shared_state, [])

    def test_serialization_with_required(self):
        """Test that deepcopy will return the same REQUIRED instance."""

        def f():
            return {"a": REQUIRED}

        def g():
            config = f()
            config["b"] = REQUIRED
            return config

        result = _DeepCopyWithClosureFnWrapper(g)()
        # Without the custom unpickler, this would fail since a new RequiredFiedlValue object will
        # be created, but we use "is" to check for this in several places.
        self.assertIs(result["a"], REQUIRED)
        self.assertIs(result["b"], REQUIRED)

    def test_unserializable(self):
        """Make sure TypeError is raised when data is unserializable."""
        t = threading.Thread(target=lambda: 1)

        def f():
            t.start()
            return 1

        with self.assertRaises(TypeError):
            _DeepCopyWithClosureFnWrapper(f)()


class TestConfigMapCache(parameterized.TestCase):
    """Test that config_map_cache works as intended."""

    def test_wrap_function(self):
        """Test that wrapping of a function works as intended."""

        def f():
            return {"a": 1}

        wrapped_f = _wrap_with_deep_copy_with_closure(f)

        # Sometimes Python strip attributes of certain functions for code like below.
        d = {}
        e = {"f": wrapped_f}
        d.update(e)
        wrapped_f = d["f"]

        # Make the marked attribute is there and the name is what we want.
        # pylint: disable-next=protected-access
        self.assertTrue(wrapped_f._is_wrapped_with_deepcopy)  # type: ignore
        self.assertEqual(wrapped_f.__module__, "axlearn.experiments.trainer_config_utils_test")
        self.assertEqual(wrapped_f.__name__, "f")
        self.assertEqual(wrapped_f(), f())

    def test_wrap_with_config(self):
        """Test with an actual ConfigBase object.

        This test also illustrates issue in how bugs could arise in the way people generate Configs
        without using any deepcopy.
        """
        shared_data = ["a"]

        def config_gen():
            return _ConfigA().set(data=shared_data)

        wrapped_config_gen = _wrap_with_deep_copy_with_closure(config_gen)

        self.assertEqual(config_gen().data, ["a"])

        # Modify shared state.
        shared_data.append("b")

        # This is not desirable.
        self.assertEqual(config_gen().data, ["a", "b"])

        # This is also not desirable either, but expected as we don't serialize until a call is
        # made in the current design so shared_data has already been changed.
        # TODO(willsong): Should we change this behavior?
        self.assertEqual(wrapped_config_gen().data, ["a", "b"])

        # Modify shared state again.
        shared_data.append("c")

        # This is not desirable.
        self.assertEqual(config_gen().data, ["a", "b", "c"])

        # This should generate consistent result from now on.
        self.assertEqual(wrapped_config_gen().data, ["a", "b"])

    def test_config_map_cache(self):
        """Test the actual config_map_cache.

        This test case also shows the benefit of using @config_map_cache, which could protect
        against closure state corruptions arise during config generation.
        """

        def get_config(name: str):
            shared_state = []

            def config_gen():
                shared_state.append(name)
                return _ConfigA().set(data=shared_state)

            return config_gen

        @config_map_cache
        def named_trainer_configs():
            return {"a": get_config("a")}

        config_map = named_trainer_configs()
        self.assertEqual(config_map["a"]().data, ["a"])
        # Repeated calls should not change the result.
        # Without @config_map_cache, current code will actually fail. Unfortunately, there's not
        # much can be done within the ConfigBase class to prevent it.
        self.assertEqual(config_map["a"]().data, ["a"])

    def test_call_with_args(self):
        """Test calling config gen with arguments."""

        def get_config():
            def config_gen(data_dir: Optional[str] = None, **overrides):
                return _ConfigA().set(data=[data_dir, overrides["k"]])

            return config_gen

        @config_map_cache
        def named_trainer_configs():
            return {"a": get_config()}

        config_map = named_trainer_configs()
        self.assertEqual(config_map["a"]("FAKE", k="data").data, ["FAKE", "data"])  # type: ignore

    def test_debug_string_when_config_gen_is_value(self):
        """Test the that debug string prints properly when config gen is used in config.

        This is taken from an actual use case.
        """

        def init_config():
            return _ConfigA()

        def gen_init_config(init_function):
            return init_function()

        def get_trainer_config_gen():
            def config_gen():
                return config_for_function(gen_init_config).set(init_function=init_config)

            return config_gen

        @config_map_cache
        def named_trainer_configs():
            return {"a": get_trainer_config_gen()}

        config = named_trainer_configs()["a"]()
        self.assertEqual(
            config.debug_string(),
            "fn: 'axlearn.experiments.trainer_config_utils_test.gen_init_config'\n"
            "init_function: 'axlearn.experiments.trainer_config_utils_test.init_config'",
        )


if __name__ == "__main__":
    pytest.main()

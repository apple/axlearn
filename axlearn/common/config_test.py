# Copyright © 2023 Apple Inc.

"""Unittests for config.py."""
# pylint: disable=too-many-public-methods
import collections
import copy
import dataclasses
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import attr
import numpy as np
import tensorflow_datasets as tfds
import wrapt
from absl.testing import absltest
from jax import numpy as jnp

from axlearn.common import config
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    Configurable,
    InstantiableConfig,
    Required,
    config_class,
    maybe_set_config,
)


class ConfigTest(absltest.TestCase):
    def test_missing_decorator(self):
        """Tests config class without the @config_class decorator."""
        with self.assertRaisesRegex(config.MissingConfigClassDecoratorError, "@config_class"):

            class Config(ConfigBase):
                num_layers: int = 10

            _ = Config()

    def test_invalid_base_class(self):
        """Tests config class without inheriting from ConfigBase."""
        with self.assertRaisesRegex(config.InvalidConfigClassError, "ConfigBase"):

            @config_class
            class Config:
                num_layers: int = 10

            _ = Config()

    def test_invalid_field_name(self):
        # Field name cannot start with "_".
        with self.assertRaisesRegex(config.InvalidConfigNameError, "_num_layers"):

            @config_class
            class Config(ConfigBase):
                _num_layers: int = 10

            _ = Config()

    def test_non_config_field(self):
        # Config fields must be of form "<field>: <type>".
        with self.assertRaisesRegex(config.NonConfigFieldError, "foo"):

            @config_class
            class Config(ConfigBase):
                foo = 10

            _ = Config()

    def test_definition(self):
        @config_class
        class EmptyConfig(ConfigBase):
            pass

        cfg = EmptyConfig()
        self.assertEqual(0, len(cfg))
        self.assertEmpty(cfg.keys())
        self.assertEmpty(cfg.items())
        self.assertNotIn("num_layers", cfg)
        self.assertEqual(
            "<class 'axlearn.common.config_test.config_class("
            f"{ConfigTest.__module__}.ConfigTest.test_definition.<locals>.EmptyConfig"
            ")'>",
            str(type(cfg)),
        )

        @config_class
        class NumLayersConfig(ConfigBase):
            num_layers: int = 10

        cfg = NumLayersConfig()
        self.assertEqual(1, len(cfg))
        self.assertEqual(["num_layers"], cfg.keys())
        self.assertEqual([("num_layers", 10)], cfg.items())
        self.assertIn("num_layers", cfg)
        self.assertEqual(10, cfg.num_layers)

    def test_mutable_values(self):
        @config_class
        class Config(ConfigBase):
            list_value: List = []

        # Different Config instances do not share the default value instance.
        cfg1 = Config()
        cfg2 = Config()
        self.assertIsNot(cfg1.list_value, cfg2.list_value)
        cfg1.list_value.append(123)
        self.assertSequenceEqual(cfg1.list_value, [123])
        self.assertSequenceEqual(cfg2.list_value, [])

        mutable_list: List = [1, 2, [3]]
        cfg2.list_value = mutable_list
        # Assignment to a config field is by value.
        self.assertSequenceEqual(cfg2.list_value, [1, 2, [3]])
        self.assertIsNot(cfg2.list_value, mutable_list)
        # Therefore changes to mutable_list do not affect the config value.
        mutable_list.append(4)
        mutable_list[2].append(3.5)
        self.assertSequenceEqual(cfg2.list_value, [1, 2, [3]])

    def test_config_inheritance(self):
        @config_class
        class BaseLayerConfig(ConfigBase):
            dtype: jnp.dtype = jnp.float32

        @config_class
        class LayerNormConfig(BaseLayerConfig):
            dim: int

        cfg: LayerNormConfig = LayerNormConfig(dim=16)  # pylint: disable=unexpected-keyword-arg
        self.assertIsInstance(cfg, BaseLayerConfig)
        self.assertEqual(16, cfg.dim)
        self.assertEqual(jnp.float32, cfg.dtype)

    def test_config_inheritance_field_redefinition(self):
        @config_class
        class BaseLayerConfig(ConfigBase):
            scale: int = 1

        scale_attr = attr.fields_dict(BaseLayerConfig)["scale"]
        self.assertIs(scale_attr.type, int)
        self.assertEqual(scale_attr.default, 1)

        @config_class
        class LayerConfig(BaseLayerConfig):
            # Redefines `scale` from BaseLayerConfig.
            scale: float = 0.1

        scale_attr = attr.fields_dict(LayerConfig)["scale"]
        self.assertIs(scale_attr.type, float)
        self.assertEqual(scale_attr.default, 0.1)

    def test_set(self):
        @config_class
        class TestConfig(ConfigBase):
            num_layers: int = 10
            hidden_dim: int = 16

        # Set via setattr.
        cfg = TestConfig()
        cfg.num_layers = 6
        self.assertEqual(6, cfg.num_layers)
        self.assertEqual(16, cfg.hidden_dim)
        # Set() can update multiple fields.
        self.assertIs(cfg.set(num_layers=8, hidden_dim=128), cfg)
        self.assertEqual(8, cfg.num_layers)
        self.assertEqual(128, cfg.hidden_dim)
        self.assertEqual([("hidden_dim", 128), ("num_layers", 8)], cfg.items())
        self.assertEqual("\n".join(["hidden_dim: 128", "num_layers: 8"]), cfg.debug_string())
        # UnknownFieldError.
        with self.assertRaisesRegex(
            config.UnknownFieldError, r"keys are \['hidden_dim', 'num_layers'\].*"
        ):
            cfg.vocab_size = 1024  # pylint: disable=attribute-defined-outside-init
        # When the unknown field is close enough to a defined field.
        with self.assertRaisesRegex(config.UnknownFieldError, r".*did you mean: \[num_layers\].*"):
            cfg.num_layer = 5  # pylint: disable=attribute-defined-outside-init

    def _non_copyable_proxy(self, fn_or_cls) -> wrapt.ObjectProxy:
        """Returns a proxy which cannot be copied."""

        @wrapt.decorator
        def decorator(fn_or_cls, instance, args, kwargs):
            del instance
            return fn_or_cls(*args, **kwargs)

        fn_or_cls = decorator(fn_or_cls)  # pylint: disable=no-value-for-parameter

        self.assertIsInstance(fn_or_cls, wrapt.ObjectProxy)
        with self.assertRaises(NotImplementedError):
            copy.deepcopy(fn_or_cls)

        return fn_or_cls

    def test_clone(self):
        @config_class
        class TestConfig(ConfigBase):
            num_layers: int = 10
            hidden_dim: int = 16

        cfg = TestConfig()

        # Clone() can update multiple fields.
        cfg_clone = cfg.clone(num_layers=8, hidden_dim=128)
        self.assertIsNot(cfg_clone, cfg)
        self.assertEqual(8, cfg_clone.num_layers)
        self.assertEqual(128, cfg_clone.hidden_dim)
        self.assertEqual([("hidden_dim", 128), ("num_layers", 8)], cfg_clone.items())
        self.assertEqual("\n".join(["hidden_dim: 128", "num_layers: 8"]), cfg_clone.debug_string())
        self.assertEqual(10, cfg.num_layers)
        self.assertEqual(16, cfg.hidden_dim)
        self.assertEqual([("hidden_dim", 16), ("num_layers", 10)], cfg.items())
        self.assertEqual("\n".join(["hidden_dim: 16", "num_layers: 10"]), cfg.debug_string())

        # Test with a function config.
        def fn(x: str):
            return x

        cfg = config.config_for_function(self._non_copyable_proxy(fn))
        cfg_clone = cfg.clone(x="test")
        self.assertIsNot(cfg_clone, cfg)
        self.assertIsInstance(cfg.x, config.RequiredFieldValue)
        self.assertEqual("test", cfg_clone.x)

    def test_value_types(self):
        @config_class
        class TestConfig(ConfigBase):
            sub: Optional[Any] = None

        cfg = TestConfig()
        # Config field values can be a list.
        cfg.sub = [None, 123, "str", np.float64]
        self.assertEqual(
            "\n".join(
                [
                    "sub[0]: None",
                    "sub[1]: 123",
                    "sub[2]: 'str'",
                    "sub[3]: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )
        # Config field values can be a tuple.
        cfg.sub = (None, 123, "str", np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub[0]: None",
                    "sub[1]: 123",
                    "sub[2]: 'str'",
                    "sub[3]: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )
        # Config field values can be a dict.
        cfg.sub = dict(none=None, int=123, str="str", type=np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub['none']: None",
                    "sub['int']: 123",
                    "sub['str']: 'str'",
                    "sub['type']: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )
        # Config field values can be a named tuple.
        ntuple = collections.namedtuple("ntuple", ("none", "int", "str", "type"))
        cfg.sub = ntuple(none=None, int=123, str="str", type=np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub['none']: None",
                    "sub['int']: 123",
                    "sub['str']: 'str'",
                    "sub['type']: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )

        # Config field values can be a dataclass.
        @dataclasses.dataclass
        class DataClass:
            int_val: int
            str_val: "str"
            type_val: np.dtype

        cfg.sub = DataClass(int_val=123, str_val="str", type_val=np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub['int_val']: 123",
                    "sub['str_val']: 'str'",
                    "sub['type_val']: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )
        cfg.sub = DataClass
        self.assertEqual(
            "sub: 'axlearn.common.config_test.DataClass'",
            cfg.debug_string(),
        )

    def test_nested_configs(self):
        class Encoder(Configurable):
            @config_class
            class Config(Configurable.Config):
                num_layers: int = 12

        class Decoder(Configurable):
            @config_class
            class Config(Configurable.Config):
                num_layers: int = 8
                vocab_size: int = 256

        class Model(Configurable):
            @config_class
            class Config(Configurable.Config):
                encoder: Encoder.Config = Encoder.default_config()
                decoder: Decoder.Config = Decoder.default_config()

        model_cfg = Model.default_config()
        self.assertEqual(8, model_cfg.decoder.num_layers)
        self.assertContainsSubset(
            [
                "decoder.num_layers: 8",
                "decoder.vocab_size: 256",
                "encoder.num_layers: 12",
            ],
            model_cfg.debug_string().split("\n"),
        )

        cfg2 = copy.deepcopy(model_cfg)
        self.assertEqual(8, cfg2.decoder.num_layers)
        cfg2.decoder.num_layers = 16
        self.assertEqual(16, cfg2.decoder.num_layers)
        # The original model_cfg remain unchanged.
        self.assertEqual(8, model_cfg.decoder.num_layers)

        cfg3 = Model.default_config()
        cfg3.decoder.num_layers = 16
        self.assertEqual(16, cfg3.decoder.num_layers)
        # The original model_cfg remain unchanged.
        self.assertEqual(8, model_cfg.decoder.num_layers)

    def test_required_values(self):
        class Layer(Configurable):
            @config_class
            class Config(Configurable.Config):
                input_dim: Required[int] = REQUIRED
                output_dim: Optional[int] = None

        cfg: Layer.Config = Layer.default_config()
        with self.assertRaisesRegex(config.RequiredFieldMissingError, "input_dim"):
            cfg.instantiate()
        cfg.input_dim = 8
        cfg.instantiate()

    def test_instantiable_config_for_configurable(self):
        class Layer(Configurable):
            @config_class
            class Config(Configurable.Config):
                input_dim: int = 8
                output_dim: int = 16

        cfg = Layer.default_config()
        self.assertIsInstance(cfg, config.InstantiableConfig)
        layer1 = cfg.instantiate()
        cfg2 = copy.deepcopy(cfg)
        layer2 = cfg2.instantiate()
        self.assertEqual(layer1.config.debug_string(), layer2.config.debug_string())

    def test_instantiable_config_from_init_signature(self):
        # Generate the config from the signature of Layer.__init__().
        class Layer:
            def __init__(self, in_features: int, out_features: int, bias: bool = True):
                self.params = {}
                self.params["weight"] = np.random.normal(size=(in_features, out_features))
                if bias:
                    self.params["bias"] = np.zeros(shape=(out_features,))

            def named_parameters(self):
                return self.params.items()

        cfg = config.config_for_class(Layer)
        self.assertEqual(
            f"config_for_class({Layer.__module__}.{Layer.__qualname__})", type(cfg).__name__
        )
        self.assertIsInstance(cfg, config.InstantiableConfig)
        self.assertContainsSubset({"klass", "in_features", "out_features", "bias"}, cfg.keys())
        self.assertEqual(cfg.klass, Layer)
        self.assertIsInstance(cfg.in_features, config.RequiredFieldValue)
        self.assertIsInstance(cfg.out_features, config.RequiredFieldValue)
        self.assertTrue(
            cfg.bias
        )  # the config default value is the same as the __init__ argument default value.
        with self.assertRaises(config.UnknownFieldError):
            cfg.unknown_field = 1
        with self.assertRaises(config.RequiredFieldMissingError):
            cfg.instantiate()
        cfg.in_features = 8
        cfg.out_features = 16
        layer1 = cfg.instantiate()
        cfg2 = copy.deepcopy(cfg)
        layer2 = cfg2.instantiate()

        def param_shapes(layer):
            return [(name, param.shape) for name, param in layer.named_parameters()]

        self.assertEqual(param_shapes(layer1), param_shapes(layer2))

    def test_instantiable_config_from_function_signature(self):
        cfg = config.config_for_function(tfds.load)
        self.assertIsInstance(cfg, config.InstantiableConfig)
        self.assertContainsSubset({"fn", "name", "split", "download"}, cfg.keys())
        self.assertIsInstance(cfg.name, config.RequiredFieldValue)
        self.assertIsNone(cfg.split)
        self.assertTrue(cfg.download)
        with self.assertRaises(config.UnknownFieldError):
            cfg.unknown_field = 1
        with self.assertRaises(config.RequiredFieldMissingError):
            cfg.instantiate()

        def fn_with_args(*var_args):
            return list(var_args)

        cfg = config.config_for_function(fn_with_args)
        self.assertEqual(
            f"config_for_function({fn_with_args.__module__}.{fn_with_args.__qualname__})",
            type(cfg).__name__,
        )
        cfg.var_args = [1, 2, 3]
        self.assertEqual(cfg.var_args, cfg.instantiate())

        def fn_with_kwargs(**var_kwargs):
            return dict(var_kwargs)

        cfg = config.config_for_function(fn_with_kwargs)
        cfg.var_kwargs = {"a": 1, "b": 2}
        self.assertEqual(cfg.var_kwargs, cfg.instantiate())

        with self.assertRaisesRegex(ValueError, "already specified"):
            self.assertEqual(cfg.var_kwargs, cfg.instantiate(a=3))

    def test_to_dict(self):
        def fn_with_args(*args):
            return list(args)

        @dataclasses.dataclass
        class Person:
            name: str
            age: int

        @config_class
        class TestConfigA(ConfigBase):
            num_layers: int = 10
            extra: Dict[str, int] = {"alpha": 1, "beta": 2}
            required_int: Required[int] = REQUIRED
            fn: InstantiableConfig = config.config_for_function(fn_with_args).set(args=[1, 2, 3])
            person: Person = Person("Johnny Appleseed", 30)  # pytype: disable=invalid-annotation
            person_cls: type = Person

        @config_class
        class TestConfigB(ConfigBase):
            count: int = 5

        @config_class
        class TestConfigC(ConfigBase):
            foo: str = "hello world"
            bar: List[str] = ["a", "b", "c"]
            my_config: TestConfigA = TestConfigA()  # pytype: disable=invalid-annotation
            config_dict: Dict[str, ConfigBase] = {"config_b": TestConfigB()}
            config_list: List[ConfigBase] = [TestConfigB(), TestConfigB().set(count=1)]
            config_type: type = ConfigTest
            config_func: Callable = config.similar_names

        cfg = TestConfigC()
        out = cfg.to_dict()
        assert isinstance(out["my_config"].pop("required_int"), config.RequiredFieldValue)
        self.assertCountEqual(
            out,
            {
                "foo": "hello world",
                "bar": ["a", "b", "c"],
                "my_config": {
                    "num_layers": 10,
                    "extra": {"alpha": 1, "beta": 2},
                    "fn": {"args": [1, 2, 3], "fn": None},
                    "person": {"name": "Johnny Appleseed", "age": 30},
                    "person_cls": "axlearn.common.config_test.Person",
                },
                "config_dict": {"config_b": {"count": 5}},
                "config_list": [{"count": 5}, {"count": 1}],
                "config_type": "axlearn.common.config_test.ConfigTest",
                "config_func": "axlearn.common.utils.similar_names",
            },
        )

    def test_to_dict_with_defaultdict(self):
        """Ensure ConfigBase.to_dict can handle encountering defaultdicts."""

        @config_class
        class TestConfigWithDefaultDict(ConfigBase):
            something: Dict[str, Any] = defaultdict(lambda: 1)

        cfg = TestConfigWithDefaultDict()
        out = cfg.to_dict()
        self.assertDictEqual(out, {"something": {}})

    def test_config_for_noncopyable_function(self):
        def fn(x: str):
            return x

        cfg = config.config_for_function(self._non_copyable_proxy(fn)).set(x="test")
        cfg = cfg.clone()
        self.assertEqual("test", cfg.instantiate())

    def test_config_for_noncopyable_class(self):
        class Dummy:
            def __init__(self, x, **kwargs):
                del kwargs
                self.x = x

        cfg = config.config_for_class(self._non_copyable_proxy(Dummy)).set(args=["test"], kwargs={})
        cfg = cfg.clone()
        self.assertEqual("test", cfg.instantiate().x)

    def test_maybe_set_config(self):
        cfg = BaseLayer.default_config()

        # Set the value if the key exists.
        self.assertEqual(cfg.vlog, 0)
        maybe_set_config(cfg, "vlog", 1)
        self.assertEqual(cfg.vlog, 1)

        # Do nothing if the key does not exist.
        not_exist_key = "not_exist_field"
        self.assertFalse(hasattr(cfg, not_exist_key))
        maybe_set_config(cfg, not_exist_key, 3)
        self.assertFalse(hasattr(cfg, not_exist_key))


if __name__ == "__main__":
    absltest.main()

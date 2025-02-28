# Copyright © 2023 Apple Inc.

"""Unittests for config.py."""

# pylint: disable=too-many-public-methods,protected-access
import collections
import copy
import dataclasses
import math
import typing
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import attr
import attrs
import numpy as np
import wrapt
from absl.testing import absltest, parameterized

from axlearn.common import config
from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    Configurable,
    InstantiableConfig,
    Required,
    RequiredFieldMissingError,
    config_class,
    maybe_set_config,
    validate_config_field_value,
)


class ConfigTest(parameterized.TestCase):
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

            del Config

    def test_non_config_field_with_inheritance(self):
        """Repeat test_non_config_field test but with inherited class.

        We need to make sure that overriding attributes (including functions) without typehints
        result in failures.
        """

        @config_class
        class PConfig(ConfigBase):
            foo: int = 10

        # Check overriding attribute without typehints result in failure.
        with self.assertRaisesRegex(config.NonConfigFieldError, "foo"):

            @config_class
            class CConfig1(PConfig):
                foo = 10

            del CConfig1

        # Check that no new __annotation__ doesn't result in failures.
        @config_class
        class CConfig2(PConfig):
            pass

        _ = CConfig2()

        # Check that overriding with a normal function without typehints results in failure.
        with self.assertRaisesRegex(config.NonConfigFieldError, "foo"):

            def f():
                pass

            @config_class
            class CConfig3(PConfig):
                foo = f

            del CConfig3

        # Check that overriding with a fake class instance method without typehints raises an error.
        with self.assertRaisesRegex(config.NonConfigFieldError, "foo"):

            def fake_foo(self):
                print(self)

            @config_class
            class CConfig4(PConfig):
                foo = fake_foo

            del CConfig4

        # Check that callable classes with `self` are caught.
        with self.assertRaisesRegex(config.NonConfigFieldError, "foo"):

            @dataclasses.dataclass
            class CallableClass:
                def my_fn(self):
                    del self

            @config_class
            class CConfig5(PConfig):
                foo = CallableClass()

            del CConfig5

        # Use lambda defined in the class to fake a class instance method.
        with self.assertRaisesRegex(config.NonConfigFieldError, "foo"):

            @config_class
            class CConfig6(PConfig):
                foo = lambda self: self

            del CConfig6

        # Check that overriding existing class instance methods are fine.
        @config_class
        class CConfig7(ConfigBase):
            def set(self, **kwargs):
                pass

        _ = CConfig7()

        # Check that attributes set after-the-fact are still caught.
        with self.assertRaisesRegex(config.NonConfigFieldError, "other_field"):
            CConfig7.other_field = 1

            _ = CConfig7()

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
        cfg.num_layers = 12
        self.assertEqual(12, cfg.num_layers)
        self.assertEqual(10, attr.fields(type(cfg)).num_layers.default)

    def test_mutable_values(self):
        @config_class
        class Config(ConfigBase):
            list_value: list = []

        # Different Config instances do not share the default value instance.
        cfg1 = Config()
        cfg2 = Config()
        self.assertIsNot(cfg1.list_value, cfg2.list_value)
        cfg1.list_value.append(123)
        self.assertSequenceEqual(cfg1.list_value, [123])
        self.assertSequenceEqual(cfg2.list_value, [])

        mutable_list: list = [1, 2, [3]]
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
            dtype: np.dtype = np.float32

        @config_class
        class LayerNormConfig(BaseLayerConfig):
            dim: int

        cfg: LayerNormConfig = LayerNormConfig(dim=16)  # pylint: disable=unexpected-keyword-arg
        self.assertIsInstance(cfg, BaseLayerConfig)
        self.assertEqual(16, cfg.dim)
        self.assertEqual(np.float32, cfg.dtype)

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
        self.assertEqual("REQUIRED", str(REQUIRED))
        self.assertEqual("REQUIRED", repr(REQUIRED))
        self.assertFalse(REQUIRED)
        self.assertIs(REQUIRED, REQUIRED)
        self.assertEqual(REQUIRED, REQUIRED)

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
            f"config_for_class({Layer.__module__}.{Layer.__qualname__})",
            type(cfg).__name__,
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

    def test_class_with_tensor_fields(self):
        @dataclasses.dataclass
        class Bias:
            shape: tuple[int, ...]
            target_positions: np.ndarray
            source_positions: Optional[np.ndarray] = None

        cfg = config.config_for_class(Bias).set(shape=[1, 2])
        target_positions = np.asarray([0, 1])
        source_positions = np.asarray([2, 3])
        bias = cfg.instantiate(target_positions=target_positions, source_positions=source_positions)
        np.testing.assert_array_equal(bias.target_positions, target_positions)
        np.testing.assert_array_equal(bias.source_positions, source_positions)
        with self.assertRaisesRegex(RequiredFieldMissingError, "target_positions"):
            cfg.instantiate(source_positions=source_positions)

    def test_instantiable_config_from_function_signature(self):
        def load(name: str, *, split: Optional[str] = None, download: bool = True):
            del name, split, download

        cfg = config.config_for_function(load)
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

        # Override the value of 'a' during instantiate().
        self.assertEqual({"a": 3, "b": 2}, cfg.instantiate(a=3))

    @parameterized.parameters(
        # Test some basic cases.
        dict(kwargs=dict(x="x", y="y", z="z"), expected=("x", "y", "z")),
        # Test configuring a builtin function.
        dict(fn=pow, kwargs=dict(base=2, exp=3), expected=8),
        # Test raising when attempting to configure positional-only args. They currently can't be
        # reliably configured, as ordering may be lost.
        # TODO(markblee): Add support for positional args.
        dict(
            fn=math.pow,
            kwargs=dict(x=2, y=3),
            expected=NotImplementedError("param kind POSITIONAL_ONLY"),
        ),
        # Not enough args, will fail when instantiating the config.
        dict(
            kwargs=dict(x="x", y="y"),
            expected=RequiredFieldMissingError("required field .* z"),
        ),
        # Test configuring a function with `fn` as an arg.
        dict(
            fn=lambda fn: fn + 1,
            kwargs=dict(fn=1),
            expected=ValueError("'fn' parameter"),
        ),
        # Test configuring a function with `_` as an arg.
        dict(fn=lambda _: 1, kwargs=dict(x=2), expected=ValueError("'_' parameter")),
    )
    def test_config_for_function(self, kwargs, fn=None, expected=None):
        if fn is None:
            fn = lambda x, y, z: (x, y, z)

        def build_and_invoke():
            cfg = config.config_for_function(fn)
            return cfg.set(**kwargs).instantiate()

        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                build_and_invoke()
        else:
            self.assertEqual(build_and_invoke(), expected)

    def test_function_with_tensor_fields(self):
        def fn(axis: int, x: np.ndarray) -> np.ndarray:
            return x.sum(axis=axis)

        cfg = config.config_for_function(fn).set(axis=0)
        x = np.asarray([[0, 1, 3], [2, 5, 7]])
        np.testing.assert_array_equal([2, 6, 10], cfg.instantiate(x=x))
        with self.assertRaisesRegex(RequiredFieldMissingError, "x"):
            cfg.instantiate()

    def test_config_for_function_has_type_information(self):
        """Tests that type information is available when using `config_for_function`."""

        # pylint: disable-next=unused-argument
        def fn(w: int, x: str, y: Union[int, str], z: Required[float]):
            pass

        cfg = config.config_for_function(fn)
        fields = attrs.fields(cfg.__class__)
        fields_dict = {field.name: field.type for field in fields}
        self.assertEqual(set(fields_dict.keys()), {"fn", "w", "x", "y", "z"})
        # collections.abc.Callable is apparently not equal to typing.Callable.
        # For forward compatibility, we just check the name.
        self.assertEqual(typing.get_origin(fields_dict["fn"]).__name__, "Callable")
        del fields_dict["fn"]
        self.assertEqual(fields_dict, dict(w=int, x=str, y=Union[int, str], z=Required[float]))

    def test_to_dict_and_debug_string(self):
        def fn_with_args(*args):
            return list(args)

        @dataclasses.dataclass
        class Person:
            name: str
            age: int

        @dataclasses.dataclass
        class Cat:
            name: str
            bleed: Optional[str] = None
            adopted: Optional[bool] = True

        @config_class
        class TestConfigA(ConfigBase):
            num_layers: int = 10
            extra: dict[str, int] = {"alpha": 1, "beta": 2}
            required_int: Required[int] = REQUIRED
            fn: InstantiableConfig = config.config_for_function(fn_with_args).set(args=[1, 2, 3])
            person: Person = Person("Johnny Appleseed", 30)  # pytype: disable=invalid-annotation
            person_cls: type = Person
            notes: Optional[str] = None
            cats: list[Cat] = [Cat(name="Ross", adopted=True)]  # pytype: disable=invalid-annotation

        @config_class
        class TestConfigB(ConfigBase):
            count: int = 5

        @config_class
        class TestConfigC(ConfigBase):
            foo: str = "hello world"
            bar: list[str] = ["a", "b", "c"]
            my_config: TestConfigA = TestConfigA()  # pytype: disable=invalid-annotation
            config_dict: dict[str, ConfigBase] = {"config_b": TestConfigB()}
            config_list: list[ConfigBase] = [TestConfigB(), TestConfigB().set(count=1)]
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
                    "cats": [{"name": "Ross"}],
                },
                "config_dict": {"config_b": {"count": 5}},
                "config_list": [{"count": 5}, {"count": 1}],
                "config_type": "axlearn.common.config_test.ConfigTest",
                "config_func": "axlearn.common.config.similar_names",
            },
        )
        self.assertCountEqual(
            cfg.to_flat_dict(omit_default_values=set()),
            {
                "bar[0]": "a",
                "bar[1]": "b",
                "bar[2]": "c",
                "config_dict['config_b'].count": 5,
                "config_func": config.similar_names,
                "config_list[0].count": 5,
                "config_list[1].count": 1,
                "config_type": ConfigTest,
                "foo": "hello world",
                "my_config.extra['alpha']": 1,
                "my_config.extra['beta']": 2,
                "my_config.fn.args[0]": 1,
                "my_config.fn.args[1]": 2,
                "my_config.fn.args[2]": 3,
                "my_config.fn.fn": fn_with_args,
                "my_config.num_layers": 10,
                "my_config.person['name']": "Johnny Appleseed",
                "my_config.person['age']": 30,
                "my_config.person_cls": Person,
                "my_config.required_int": REQUIRED,
                "my_config.notes": None,
                "my_config.cats[0]['name']": "Ross",
                "my_config.cats[0]['bleed']": None,
                "my_config.cats[0]['adopted']": True,
            },
        )
        self.assertCountEqual(
            cfg.to_flat_dict(omit_default_values={None, REQUIRED}),
            {
                "bar[0]": "a",
                "bar[1]": "b",
                "bar[2]": "c",
                "config_dict['config_b'].count": 5,
                "config_func": config.similar_names,
                "config_list[0].count": 5,
                "config_list[1].count": 1,
                "config_type": ConfigTest,
                "foo": "hello world",
                "my_config.extra['alpha']": 1,
                "my_config.extra['beta']": 2,
                "my_config.fn.args[0]": 1,
                "my_config.fn.args[1]": 2,
                "my_config.fn.args[2]": 3,
                "my_config.fn.fn": fn_with_args,
                "my_config.num_layers": 10,
                "my_config.person['name']": "Johnny Appleseed",
                "my_config.person['age']": 30,
                "my_config.person_cls": Person,
                "my_config.cats[0]['name']": "Ross",
                "my_config.cats[0]['adopted']": True,
                # REQUIRED/None are trivial default values and so are omitted.
                # "my_config.required_int": REQUIRED,
                # "my_config.notes": None,
                # "my_config.cats[0]['bleed']": None,
            },
        )
        self.assertEqual(
            (
                "bar[0]: 'a'\n"
                "bar[1]: 'b'\n"
                "bar[2]: 'c'\n"
                "config_dict['config_b'].count: 5\n"
                "config_func: 'axlearn.common.config.similar_names'\n"
                "config_list[0].count: 5\n"
                "config_list[1].count: 1\n"
                "config_type: 'axlearn.common.config_test.ConfigTest'\n"
                "foo: 'hello world'\n"
                "my_config.cats[0]['name']: 'Ross'\n"
                "my_config.cats[0]['adopted']: True\n"
                "my_config.extra['alpha']: 1\n"
                "my_config.extra['beta']: 2\n"
                "my_config.fn.args[0]: 1\n"
                "my_config.fn.args[1]: 2\n"
                "my_config.fn.args[2]: 3\n"
                "my_config.fn.fn: 'axlearn.common.config_test.fn_with_args'\n"
                "my_config.num_layers: 10\n"
                "my_config.person['name']: 'Johnny Appleseed'\n"
                "my_config.person['age']: 30\n"
                "my_config.person_cls: 'axlearn.common.config_test.Person'"
            ),
            cfg.debug_string(),
        )

    @parameterized.product(
        omit_default_values=({None, REQUIRED}, {}),
        default_value=(REQUIRED, None, False, 0, 0.0, ""),
        set_value=(REQUIRED, None, False, 0, 0.0, ""),
    )
    def test_to_flat_dict_omit_default_values(
        self, *, omit_default_values, default_value, set_value
    ):
        """Tests ConfigBase.to_flat_dict with omit_trivial_default_values=True."""

        @config_class
        class TestConfig(ConfigBase):
            value: Any = default_value

        cfg = TestConfig().set(value=set_value)
        omit = default_value is set_value and default_value in omit_default_values
        self.assertCountEqual(
            cfg.to_flat_dict(omit_default_values=omit_default_values),
            {} if omit else {"value": set_value},
            msg=f"{default_value} vs. {set_value}",
        )
        # debug_string() omits trivial default values.
        self.assertEqual(
            cfg.debug_string(omit_default_values=omit_default_values),
            "" if omit else f"value: {repr(set_value)}",
            msg=f"{default_value} vs. {set_value}",
        )

    def test_to_dict_with_defaultdict(self):
        """Ensure ConfigBase.to_dict can handle encountering defaultdicts."""

        @config_class
        class TestConfigWithDefaultDict(ConfigBase):
            something: dict[str, Any] = defaultdict(lambda: 1)

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
        @config_class
        class ModuleConfig(ConfigBase):
            vlog: Optional[int] = None

        @config_class
        class BaseLayerConfig(ModuleConfig):
            dtype: np.dtype = np.float16

        cfg = BaseLayerConfig()

        # Set the value if the key exists.
        self.assertIsNone(cfg.vlog)
        maybe_set_config(cfg, vlog=1)
        self.assertEqual(cfg.vlog, 1)

        # Do nothing if the key does not exist.
        not_exist_key = "not_exist_field"
        self.assertFalse(hasattr(cfg, not_exist_key))
        maybe_set_config(cfg, **{not_exist_key: 3})
        self.assertFalse(hasattr(cfg, not_exist_key))

        # Set multiple keys.
        maybe_set_config(cfg, vlog=2, dtype=np.float32, not_exist_field=4)
        self.assertEqual(cfg.vlog, 2)
        self.assertEqual(cfg.dtype, np.float32)
        self.assertFalse(hasattr(cfg, not_exist_key))

    def test_validate_hf(self):
        class HFLike:
            def from_pretrained(self, *args):
                del args

            def to_dict(self, *args):
                del args

        # Note that hasattr(HFLike, "from_pretrained") == True.
        validate_config_field_value(HFLike)
        validate_config_field_value(HFLike())

    def test_register_validator(self):
        # Test validating a custom type.
        class MyCustomType:
            value: int = 0

        @config_class
        class MyConfig(ConfigBase):
            custom: Any = MyCustomType()

        # Validate that custom type initially is not accepted.
        with self.assertRaises(config.InvalidConfigValueError):
            validate_config_field_value(MyCustomType())
        with self.assertRaises(config.InvalidConfigValueError):
            MyConfig()

        def validate_fn(v):
            if v.value <= 0:
                raise config.InvalidConfigValueError()

        # Register custom validator.
        config.register_validator(
            match_fn=lambda v: isinstance(v, MyCustomType), validate_fn=validate_fn
        )

        # Test that validate_fn is invoked.
        with self.assertRaises(config.InvalidConfigValueError):
            validate_config_field_value(MyCustomType())
        with self.assertRaises(config.InvalidConfigValueError):
            MyConfig()

        # Test that validation succeeds.
        MyCustomType.value = 1
        validate_config_field_value(MyCustomType())
        MyConfig()

        def raise_fn(v):
            del v
            raise config.InvalidConfigValueError()

        # Check that all matched validators are invoked.
        config.register_validator(
            match_fn=lambda v: isinstance(v, MyCustomType), validate_fn=raise_fn
        )

        with self.assertRaises(config.InvalidConfigValueError):
            validate_config_field_value(MyCustomType())
        with self.assertRaises(config.InvalidConfigValueError):
            MyConfig()

    def test_override_set_get(self):
        @config_class
        class MyConfig(ConfigBase):
            """A dummy config overriding setattr/getattr."""

            a: Required[int] = REQUIRED

            def __getattr__(self, name: str) -> Any:
                try:
                    return super().__getattr__(name)
                except KeyError:
                    return "default"

            def set(self, **kwargs):
                try:
                    return super().set(**kwargs)
                except config.UnknownFieldError:
                    return self

        cfg = MyConfig()
        cfg.a = 123
        self.assertEqual(123, cfg.a)
        self.assertEqual("default", cfg.b)
        cfg.set(b=345)
        self.assertEqual(123, cfg.a)
        self.assertEqual("default", cfg.b)
        cfg_clone = cfg.clone(b=345)
        self.assertEqual(123, cfg_clone.a)
        self.assertEqual("default", cfg_clone.b)

    def test_get_recursively(self):
        class Nested(Configurable):
            @config_class
            class Config(Configurable.Config):
                """A dummy config."""

                value: int = 0

        class Test(Configurable):
            @config_class
            class Config(Configurable.Config):
                """Another dummy config that has a nested config."""

                nested: Nested.Config = Nested.default_config()
                value: int = 1

        cfg = Test.default_config()

        # Test getting nested value.
        self.assertEqual(cfg.get_recursively(["nested", "value"]), 0)

        # Test getting top-level value.
        self.assertEqual(cfg.get_recursively(["value"]), 1)

        # Test getting non-existent value.
        with self.assertRaises(AttributeError):
            cfg.get_recursively(["non_existent"])

        # Test getting empty path, should return self.
        self.assertEqual(cfg.get_recursively([]), cfg)

    def test_set_recursively(self):
        class Nested(Configurable):
            @config_class
            class Config(Configurable.Config):
                """A dummy config."""

                value: int = 0

        class Test(Configurable):
            @config_class
            class Config(Configurable.Config):
                """Another dummy config that has a nested config."""

                nested: Nested.Config = Nested.default_config()
                value: int = 1

        cfg = Test.default_config()

        # Test setting nested value.
        cfg.set_recursively(["nested", "value"], value=10)
        self.assertEqual(cfg.nested.value, 10)

        # Test setting top-level value.
        cfg.set_recursively(["value"], value=5)
        self.assertEqual(cfg.value, 5)

        # Test setting non-existent value.
        with self.assertRaises(AttributeError):
            cfg.set_recursively(["non_existent"], value=20)

        # Test setting empty path.
        with self.assertRaises(ValueError):
            cfg.set_recursively([], value=20)


if __name__ == "__main__":
    absltest.main()

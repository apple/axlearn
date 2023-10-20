# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Classes to represent configs for ML layers, inputs, and models.

Adapted from https://github.com/tensorflow/lingvo/blob/master/lingvo/core/hyperparams.py.

See https://tinyurl.com/ml-api-styles for the design principles behind this config library.

Example usage for configuring a module:

    class MyModule(Module):
        @config_class
        class Config(Module.Config):
            num_layers: int = 0
            input_dim: int = 0
            output_dim: int = 0

        def __init__(self, cfg: "MyModule.Config"):
            super().__init__(cfg)
            cfg = self.config
            for layer_i in range(cfg.num_layers):
                ...

    def create_foo_module():
        cfg = MyModule.default_config()
        cfg.num_layers = 8
        cfg.set(input_dim=512, output_dim=256)
        return cfg.instantiate()

Config can also be used for third-party classes/functions with config_for_class(), for example:

    input_projection: InstantiableConfig = config_for_class(nn.Linear)
    lr_schedule: InstantiableConfig = config_for_function(cosine_decay_schedule).set(
        decay_steps=1000)

NOTE: all assignment of config field values (including the default values) are 'by value', i.e.,
the config instance will retain a copy of the value rather than keeping a reference to the value.
It is the case even if the values are mutable, e.g.,

    @config_class
    class Config(ConfigBase):
        x: List[int] = []

    value_x = [1]
    cfg = Config().set(x=value_x)
    assert cfg.x == [1]
    assert cfg.x is not value_x
    value_x.append(2)
    # cfg.x does not reference value_x, so modifications to value_x does not affect cfg.x
    assert cfg.x == [1]

Therefore, unlike with attrs/dataclass, the user does not need to define a default factory for
config fields with mutable values, including config values.
"""

# Note: config.py should not depend on jax, torch, or tf.
import copy
import dataclasses
import enum
import functools
import inspect
import re
import types
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# attr provides similar features as Python dataclass. Unlike
# dataclass, however, it provides a richer set of features to regulate
# the classes.
#
# More information about attr can be found at https://www.attrs.org/en/stable/why.html.
#
# Our config library relies on `__attrs_post_init__` and `on_setattr=_validate_and_transform_field`
# to apply validation on field names and values.
import attr
import numpy as np


def is_named_tuple(x: Any):
    """Returns whether an object is an instance of a collections.namedtuple.

    Examples::
        is_named_tuple((42, 'hi')) ==> False
        Foo = collections.namedtuple('Foo', ['a', 'b'])
        is_named_tuple(Foo(a=42, b='hi')) ==> True

    Args:
        x: The object to check.
    """
    return isinstance(x, tuple) and hasattr(x, "_fields") and hasattr(x, "_asdict")


def is_attrs(x: Any):
    """Returns whether an object is an instance of attrs class.

    Examples::
        @attr.define
        class Foo:
            a: int
            b: str
        is_attrs(Foo(a=42, b='hi')) ==> True

    Args:
        x: The object to check.
    """
    return hasattr(x, "__attrs_attrs__")


def similar_names(name: str, candidates: Iterable[str]) -> List[str]:
    """Return a sorted list of candidates that are similar to name."""

    def overlaps(name: str, key: str) -> float:
        """The fraction of 3-char substrings in <name> that appear in key."""
        matches = 0
        trials = 0
        for i in range(len(name) - 2):
            trials += 1
            if name[i : i + 3] in key:
                matches += 1
        return float(matches) / max(trials, 1)

    # Compute overlaps for each candidate.
    pairs = [(overlaps(name, key), key) for key in candidates]
    # Filter out candidates below 0.5 overlap threshold.
    pairs = [pair for pair in pairs if pair[0] > 0.5]
    # Sort by highest overlap, breaking ties alphabetically.
    pairs.sort(key=lambda pair: (-pair[0], pair[1]))
    # Return just the keys.
    return [key for _, key in pairs]


T = TypeVar("T")


class RequiredFieldValue:
    def __deepcopy__(self, memo):
        return self

    def __bool__(self):
        return False

    def __repr__(self):
        return "REQUIRED"


REQUIRED = RequiredFieldValue()
Required = Union[T, RequiredFieldValue, Any]


class MissingConfigClassDecoratorError(TypeError):
    pass


class InvalidConfigClassError(TypeError):
    pass


class InvalidConfigNameError(ValueError):
    pass


class InvalidConfigValueError(TypeError):
    pass


class NonConfigFieldError(ValueError):
    pass


class UnknownFieldError(AttributeError):
    pass


class RequiredFieldMissingError(ValueError):
    pass


# TODO(rpang): support frozen configs.
class FrozenConfigError(RuntimeError):
    pass


def validate_config_field_name(name: str) -> None:
    if not re.fullmatch("^[a-z][a-z0-9_]*$", name):
        raise InvalidConfigNameError(f'Invalid config field name "{name}"')


def validate_config_field_value(value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for x in value:
            validate_config_field_value(x)
    elif isinstance(value, dict):
        for _, v in value.items():
            validate_config_field_value(v)
    elif value is None or isinstance(
        value,
        (
            RequiredFieldValue,
            type,
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.BuiltinMethodType,
            int,
            float,
            str,
            enum.Enum,
            np.dtype,
        ),
    ):
        pass
    elif dataclasses.is_dataclass(value):
        validate_config_field_value(dataclasses.asdict(value))
    elif is_attrs(value):
        validate_config_field_value(attr.asdict(value, recurse=False))
    # Branch condition for Hugging Face transformers config object.
    elif hasattr(value, "from_pretrained"):
        validate_config_field_value(value.to_dict())
    else:
        raise InvalidConfigValueError(
            f'Invalid config value type {type(value)} for value "{value}"'
        )


def _validate_and_transform_field(instance, attribute, value):
    """Validates an attribute as a config field.

    Args:
        instance: the ConfigBase instance.
        attribute: the attrs.Attribute instance.
        value: the attribute value.

    Returns:
        The transformed attribute value.
    """
    validate_config_field_name(attribute.name)
    validate_config_field_value(value)

    # Exempt klass and fn from copying. Some packages, such as wrapt, decorate via an object proxy
    # which is not copyable. Since klass is known to be a class, and fn is known to be a function,
    # and since these attributes are generally not mutable, we skip the copy step. Other attributes
    # which are also proxies are expected to define __deepcopy__, since it's not immediately obvious
    # how to detect that an object is in fact a proxy in the general case.
    if (isinstance(instance, FunctionConfigBase) and attribute.name == "fn") or (
        isinstance(instance, ClassConfigBase) and attribute.name == "klass"
    ):
        return value

    # Copy value so that a mutable value is not shared across configs. This is especially important
    # when the default value is a mutable value, e.g.,
    #
    # @config_class
    # class Config(BaseLayer.Config):
    #     linear: Linear.Config = Linear.default_config()
    #
    # If we do not copy here, all Config instances will share the same mutable Linear.Config
    # instance.
    return copy.deepcopy(value)


_ConfigBase = TypeVar("_ConfigBase", bound="ConfigBase")


class ConfigBase:
    """The base class of config classes."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__attrs_init__(*args, **kwargs)

        attr_cls = type(self)
        for k in dir(attr_cls):
            if (
                not k.startswith("__")
                and k not in dir(InstantiableConfig)
                and k not in attr.fields_dict(attr_cls)
            ):
                raise NonConfigFieldError(f"Non-config attribute is not supported: {attr_cls}.{k}")

    def __attrs_init__(self):
        raise MissingConfigClassDecoratorError(f"{type(self)} was not decorated with @config_class")

    def __attrs_post_init__(self):
        # Call setattr to trigger _validate_and_transform_field on all keys and default values.
        for k, v in self.items():
            setattr(self, k, v)

    def __contains__(self, name: str) -> bool:
        return name in attr.fields_dict(type(self))

    def __len__(self) -> int:
        return len(attr.fields(type(self)))

    def keys(self) -> List[str]:
        return sorted(attr.fields_dict(type(self)).keys())

    def items(self) -> List[Tuple[str, Any]]:
        """Returns (key, value) pairs sorted by keys."""
        return [(key, getattr(self, key)) for key in self.keys()]

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def clone(self, **kwargs):
        """Returns a clone of the original config with the optional keyword overrides.

        Unlike, :meth:`self.set`, this function does not modify the config in-place.
        """
        return attr.evolve(self, **kwargs)

    def debug_string(
        self,
        *,
        kv_separator: str = ": ",
        field_separator: str = "\n",
        omit_default_values: Collection[Any] = (None, REQUIRED),
    ) -> str:
        """Returns a debug string for the config.

        Args:
            kv_separator: The key-value separator.
            field_separator: The field separator.
            omit_default_values: A set of default values to omit in debug string.
                See comments on `to_flat_dict`.

        Returns:
            A str separated by `field_separator` where each entry is of form
            f"{path}{kv_separator}{val}", representing path and value of a leaf config field.
        """
        flat_dict = self.to_flat_dict(omit_default_values=omit_default_values)

        def fmt(key: str, val: Any) -> Union[str, Tuple[str, str]]:
            if isinstance(val, (type, types.FunctionType)):
                val = f"{val.__module__}.{val.__name__}"
            return f"{key}{kv_separator}{repr(val)}"

        return field_separator.join([fmt(k, v) for k, v in flat_dict.items()])

    def to_flat_dict(self, *, omit_default_values: Collection[Any]) -> Dict[str, Any]:
        """Returns a flattened dict with path -> value mappings.

        Args:
            omit_default_values: Omit a field from the output dict if its value remains the
                default value of the field *and* the default value is a member of
                `omit_default_values`.

        Returns:
            A dict where each key is a `.`-separated str of path and each value represents a
            leaf config field value.
        """
        result = {}

        def enter(key: str, val: Any, default_result: Optional[List]) -> Optional[List]:
            if key and isinstance(val, ConfigBase):
                # Call `to_flat_dict` on any sub config. This allows a sub config to override
                # the behavior of `to_flat_dict`.
                val_entries = val.to_flat_dict(omit_default_values=omit_default_values)
                # For each entry from `debug_string`, prepend `<key>.` to each key.
                result.update({f"{key}.{k}": v for k, v in val_entries.items()})
                return []  # Nothing to traverse.
            # Otherwise adopt the default behavior.
            return default_result

        def process_kv(key: str, val: Any):
            field = attr.fields_dict(type(self)).get(key)
            if isinstance(field, attr.Attribute):
                default_val = field.default
                if val is default_val and default_val in omit_default_values:
                    return
            result[key] = val

        # Note that we cannot use `utils.flatten_items` to handle this because the treatment of
        # lists is different.
        self.visit(visit_fn=process_kv, enter_fn=enter)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Returns a nested dictionary of config fields."""

        # pylint: disable-next=too-many-return-statements
        def _to_dict(val: Any):
            if isinstance(val, ConfigBase):
                return val.to_dict()
            elif isinstance(val, defaultdict):
                # Special case for defaultdict here because its constructor
                # is different than `dict`, which causes a `TypeError` if
                # we try doing `type(val)({...})`.
                return dict({k: _to_dict(v) for k, v in val.items()})
            elif isinstance(val, dict):
                return type(val)({k: _to_dict(v) for k, v in val.items()})
            elif dataclasses.is_dataclass(val) and not isinstance(val, type):
                return _to_dict(dataclasses.asdict(val))
            elif is_attrs(val):
                return _to_dict(attr.asdict(val, recurse=False))
            elif is_named_tuple(val):
                return _to_dict(val._asdict())
            elif isinstance(val, (list, tuple)):
                return type(val)([_to_dict(v) for v in val])
            elif isinstance(val, range):
                return [_to_dict(v) for v in val]
            elif isinstance(val, (type, types.FunctionType)):
                return f"{val.__module__}.{val.__name__}"
            else:
                return val

        return {k: _to_dict(v) for k, v in self.items()}

    def __str__(self):
        return self.debug_string()

    def __repr__(self):
        return self.debug_string(kv_separator=":", field_separator="; ")

    def visit(
        self,
        visit_fn: Callable[[str, Any], None],
        enter_fn: Optional[Callable[[str, Any, Optional[List]], Optional[List]]] = None,
        exit_fn: Optional[Callable[[str, Any], None]] = None,
    ):
        """Recursively visits objects within this Config instance.

        Visit can traverse Config, lists, tuples, dataclasses, and namedtuples. By default, visit_fn
        is called on any object we don't know how to traverse into, like an integer or a string.
        enter_fn and exit_fn are called on objects we can traverse into, like Config, lists, tuples,
        dicts, dataclasses, and namedtuples. We call enter_fn before traversing the object, and
        exit_fn when we are finished.

        A default enter function will be used if enter_fn is None. The default function returns None
        if the value is not a Config, list, tuple, dict, dataclass, or namedtuple, otherwise a list
        of (subkey, subval) pairs to traverse.

        Each subkey returned by the default enter function has one of the following forms:
            key.subkey when traversing Config objects
            key[1] when traversing lists/tuples/ranges
            key[subkey] when traversing dicts, dataclasses, or namedtuples

        enter_fn, if not None, takes key, value, and the return value of the default enter function
        and returns either None or a list of (subkey, subval) pairs to traverse. This allows the
        user to override the entry decision or key format of the default function.

        Args:
            visit_fn: Called on every object for which enter_fn returns None.
            enter_fn: If not None, called on every object. If this function returns None, we call
                visit_fn and do not enter the object.
            exit_fn: Called after an enter-able object has been traversed.
        """
        if not enter_fn:
            enter_fn = lambda key, val, items: items
        if not exit_fn:
            exit_fn = lambda key, val: None

        def _visit(key: str, val: Any):
            val_items = enter_fn(key, val, _default_enter_fn(key, val))
            if val_items is None:
                visit_fn(key, val)
            else:
                for subkey, subval in val_items:
                    _visit(subkey, subval)
                exit_fn(key, val)

        # pylint: disable-next=too-many-return-statements
        def _default_enter_fn(key: str, val: Any):
            if isinstance(val, ConfigBase):
                return [(_sub_key(key, k), v) for k, v in val.items()]
            elif isinstance(val, dict):
                return [(f"{key}[{repr(k)}]", v) for k, v in val.items()]
            elif dataclasses.is_dataclass(val) and not isinstance(val, type):
                return _default_enter_fn(key, dataclasses.asdict(val))
            elif is_attrs(val):
                return _default_enter_fn(key, attr.asdict(val, recurse=False))
            elif is_named_tuple(val):
                return _default_enter_fn(key, val._asdict())
            elif isinstance(val, (list, tuple, range)):
                return [(f"{key}[{i}]", v) for i, v in enumerate(val)]
            else:
                return None  # do not enter 'val'

        def _sub_key(key, subkey):
            if key:
                return f"{key}.{subkey}"
            return subkey

        _visit("", self)

    def _key_error_string(self, name: str) -> str:
        similar = similar_names(name, list(self.keys()))
        if similar:
            return f'{name} (did you mean: [{", ".join(similar)}])'
        return f"{name} (keys are {self.keys()})"


def _config_class_kwargs():
    return dict(init=False, kw_only=True, slots=True, on_setattr=_validate_and_transform_field)


def _wrap_config_attr_cls(attr_cls: Type, *, name: Optional[str] = None):
    """Wraps `attr_cls` to override `__{setattr,getattr}__`."""
    # pylint: disable=protected-access

    orig_setattr = attr_cls.__setattr__

    def wrapped_setattr(self, key: str, value):
        if key.startswith("__"):
            self.__dict__[key] = value
        else:
            if key not in attr.fields_dict(attr_cls):
                raise UnknownFieldError(self._key_error_string(key))
            orig_setattr(self, key, value)

    def wrapped_getattr(self, key: str) -> Any:
        if key.startswith("__"):
            try:
                return self.__dict__[key]
            except KeyError as e:
                raise AttributeError(key) from e
        else:
            try:
                return attr.asdict(self, recurse=False)[key]
            except KeyError as e:
                raise AttributeError(self._key_error_string(key)) from e

    # Wrapping `attr_cls` with a class makes it tricky when working with generics. Instead, we
    # patch `__setattr__` and `__getattr__` directly.
    # TODO(markblee): See if there's a more clever way to use attrs to do this.
    attr_cls.__setattr__ = wrapped_setattr
    attr_cls.__getattr__ = wrapped_getattr

    name = name or f"config_class({attr_cls.__module__}.{attr_cls.__qualname__})"
    attr_cls.__name__ = name
    attr_cls.__qualname__ = name

    # pylint: enable=protected-access
    return attr_cls


def config_class(cls: Type[T], **kwargs) -> Type[T]:
    if not issubclass(cls, ConfigBase):
        raise InvalidConfigClassError(f"A config class must be a subclass of ConfigBase: {cls}")

    attr_cls = attr.define(maybe_cls=cls, **_config_class_kwargs(), **kwargs)
    # Pytype seems to infer attr_cls as a callable.
    return _wrap_config_attr_cls(attr_cls)  # pytype: disable=wrong-arg-types


def _validate_required_fields(cfg: ConfigBase):
    for k, v in cfg.items():
        if isinstance(v, RequiredFieldValue):
            raise RequiredFieldMissingError(
                f"Missing value for required field when instantiating {type(cfg)}: {k}"
            )


class InstantiableConfig(Generic[T], ConfigBase):
    def instantiate(self, **kwargs) -> T:
        raise NotImplementedError(type(self))


ConfigOr = Union[T, InstantiableConfig[T]]


def maybe_instantiate(x: ConfigOr[T]) -> T:
    if isinstance(x, InstantiableConfig):
        return x.instantiate()
    return x


C = TypeVar("C", bound="Configurable")


class Configurable:
    """The base class of objects that can be instantiated from a config.

    It's common for an object class to have a Config member class:

        class MyObject(Configurable):
            @config_class
            class Config(Configurable.Config):
                weight: float = 0.2  # Training weight.

        # Create a MyObject with config.
        cfg = MyObject.default_config().set(weight=0.9)
        obj = cfg.instantiate()

    By convention, anything that configures the behavior of your class should be stored in this
    Config object. However, your class may also use shared state objects which aren't really part of
    the config, like a shared lock. These can be passed as extra arguments to instantiate().

    Example:
        lock = threading.Lock()
        config = MyObject.default_config()
        obj_a = config.instantiate(lock=lock)
        obj_b = config.instantiate(lock=lock)
    """

    @config_class
    class Config(InstantiableConfig[C]):
        """The base config class for a Configurable object."""

        # Subclasses/users should not set `klass` explicitly.
        # It will be set by Configurable.default_config().
        # See ClassConfigBase for notes on why we name this `klass` rather than `cls`.
        klass: Type[C]

        def instantiate(self, **kwargs) -> C:
            """Instantiates a Configurable object.

            Args:
                **kwargs: Additional keyword arguments to pass to the constructor in
                    addition to this Config object.

            Returns:
                A constructed object where `type(object) == self.klass`.

            Raises:
                RequiredFieldMissingError: If a required field is missing.
            """
            try:
                _validate_required_fields(self)
            except RequiredFieldMissingError as e:
                raise RequiredFieldMissingError(
                    f"Failed to instantiate {self.klass}:\n\t{e}"
                ) from e
            return self.klass(self, **kwargs)

    @classmethod
    def default_config(cls: Type[C]) -> Config[C]:
        return cls.Config(klass=cls)

    def __init__(self, cfg):
        # Make a copy of `cfg` so that subsequent mutations to `cfg` won't affect self._config.
        self._config = copy.deepcopy(cfg)

    @property
    def config(self: C) -> Config[C]:
        return copy.deepcopy(self._config)

    def __repr__(self):
        return repr(self._config)


def _attr_field_from_signature_param(param: inspect.Parameter) -> attr.Attribute:
    default_value = param.default
    if default_value is inspect.Parameter.empty:
        default_value = REQUIRED
    return attr.field(default=default_value)


def _prepare_args_and_kwargs(
    kwargs: Dict[str, Any], *, sig: inspect.Signature, cfg: InstantiableConfig
) -> List:
    """Fills `kwargs` and `args` with values from `cfg` according to `sig` and returns `args`."""
    args = []

    def insert_to_kwargs(k, v):
        if k in kwargs:
            raise ValueError(f"{k} is already specified: {v} vs. {kwargs[k]}")
        kwargs[k] = v

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        value = getattr(cfg, name)
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            args = value
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            for k, v in value.items():
                insert_to_kwargs(k, v)
        elif param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            insert_to_kwargs(name, value)
        else:
            raise NotImplementedError(f"Unsupported param kind {param.kind}: {name}")

    return args


@config_class
class FunctionConfigBase(InstantiableConfig[T]):
    """The base class of configs constructed by `config_for_function`, which invokes `self.fn` upon
    instantiation.
    """

    fn: Callable[..., T]

    def instantiate(self, **kwargs) -> T:
        _validate_required_fields(self)
        args = _prepare_args_and_kwargs(kwargs, sig=inspect.signature(self.fn), cfg=self)
        return self.fn(*args, **kwargs)


F = TypeVar("F", bound=Callable)


def _config_class_for_function(fn: F) -> Type[FunctionConfigBase]:
    """Returns a config class."""
    init_sig = inspect.signature(fn)
    config_attrs = {
        name: _attr_field_from_signature_param(param) for name, param in init_sig.parameters.items()
    }
    return _wrap_config_attr_cls(
        attr.make_class(
            "FunctionConfig",
            bases=(FunctionConfigBase[F],),
            attrs=config_attrs,
            **_config_class_kwargs(),
        ),
        name=f"config_for_function({fn.__module__}.{fn.__qualname__})",
    )


def config_for_function(fn: Callable[..., T]) -> Union[Any, FunctionConfigBase[T]]:
    config_cls = _config_class_for_function(fn)
    return config_cls(fn=fn)


# Making fn positional only allows there to be an `fn` kwarg.
def config_for_partial_function(
    fn: Callable[..., T], /, *args, **kwargs
) -> InstantiableConfig[Callable[..., T]]:
    """Similar to `config_for_function` but returns a partial function when instantiated.

    Example:
        ```
        cfg = config_for_partial_function(pow, exp=2)
        fn = cfg.instantiate()
        assert fn(3) == 9
        ```

    Args:
        fn: The function to wrap.
        args: Positional arguments to pass to fn.
        kwargs: Keyword arguments to pass to fn.

    Returns:
        A Config that when instantiated, returns a partial version of `fn` based on any
        config fields that have been set.
    """
    cfg = config_for_class(functools.partial)
    cfg.set(args=(fn, *args), kwargs=kwargs)
    return cfg


@config_class
class ClassConfigBase(InstantiableConfig[T]):
    """The base class of configs constructed by `config_for_class`, which constructs instances of
    `self.klass` upon instantiation.
    """

    # Note: Generic classes come with a __new__(cls, *args, **kwds) method by default. Naming this
    # field `cls` (or even `_cls`, since `attr.make_class` strips leading underscores when
    # generating `__init__`) can cause conflicts.
    klass: Type[T]

    def instantiate(self, **kwargs) -> T:
        _validate_required_fields(self)
        args = _prepare_args_and_kwargs(
            kwargs, sig=inspect.signature(self.klass.__init__), cfg=self
        )
        return self.klass(*args, **kwargs)


def _config_class_for_class(cls: Type[T]) -> Type[ClassConfigBase[T]]:
    """Returns a config class."""
    init_sig = inspect.signature(cls.__init__)
    config_attrs = {
        name: _attr_field_from_signature_param(param)
        for name, param in init_sig.parameters.items()
        if name != "self"
    }
    return _wrap_config_attr_cls(
        attr.make_class(
            "ClassConfig", bases=(ClassConfigBase[T],), attrs=config_attrs, **_config_class_kwargs()
        ),
        name=f"config_for_class({cls.__module__}.{cls.__qualname__})",
    )


def config_for_class(cls: Type[T]) -> Union[Any, ClassConfigBase[T]]:
    config_cls = _config_class_for_class(cls)
    return config_cls(klass=cls)


def maybe_set_config(cfg: _ConfigBase, **kwargs) -> _ConfigBase:
    """Applies **kwargs to the given `cfg` if the keys exist."""
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg

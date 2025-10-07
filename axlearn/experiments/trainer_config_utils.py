# Copyright © 2023 Apple Inc.

"""Trainer config utilities."""

import io
import pickle
from functools import cache, wraps
from typing import Any, Callable, Optional

import cloudpickle

from axlearn.common.config import REQUIRED, TrainerConfigFn, config_class
from axlearn.common.flash_attention.layer import BackendOverrideModifier, FlashBlockSizeModifier
from axlearn.common.utils import get_data_dir


class V6eFlashConfigModifier(FlashBlockSizeModifier):
    """Modifies the tpu_block_size config for better performance on TPU v6e."""

    @config_class
    class Config(FlashBlockSizeModifier.Config):
        """Configures V6eFlashConfigModifier."""

        tpu_block_size: int = 1024


class V7xFlashConfigModifier(FlashBlockSizeModifier):
    """Modifies the tpu_block_size config for better performance on TPU v7x."""

    @config_class
    class Config(FlashBlockSizeModifier.Config):
        """Configures V7xFlashConfigModifier."""

        tpu_block_size: int = 2048


class SplashAttentionConfigModifier(BackendOverrideModifier):
    """Modifies the backend_overrides config for TPU Splash Attention."""

    @config_class
    class Config(BackendOverrideModifier.Config):
        """Configures SplashAttentionConfigModifier."""

        splash_block_q: Optional[int] = None
        splash_block_kv: Optional[int] = None
        splash_block_kv_compute: Optional[int] = None
        splash_block_q_dkv: Optional[int] = None
        splash_block_kv_dkv: Optional[int] = None
        splash_block_kv_dkv_compute: Optional[int] = None
        splash_block_q_dq: Optional[int] = None
        splash_block_kv_dq: Optional[int] = None

    def __call__(self, cfg: BackendOverrideModifier.Config) -> BackendOverrideModifier.Config:
        backend_overrides = {
            "splash_block_q": self.config.splash_block_q,
            "splash_block_kv": self.config.splash_block_kv,
            "splash_block_kv_compute": self.config.splash_block_kv_compute,
            "splash_block_q_dkv": self.config.splash_block_q_dkv,
            "splash_block_kv_dkv": self.config.splash_block_kv_dkv,
            "splash_block_kv_dkv_compute": self.config.splash_block_kv_dkv_compute,
            "splash_block_q_dq": self.config.splash_block_q_dq,
            "splash_block_kv_dq": self.config.splash_block_kv_dq,
        }
        backend_modifier: BackendOverrideModifier = (
            BackendOverrideModifier.default_config()
            .set(backend_overrides=backend_overrides)
            .instantiate()
        )

        return backend_modifier.__call__(cfg)


class _DummyRequired:
    """A dummy class to return the same REQUIRED instance upon init call."""

    def __new__(cls):
        return REQUIRED


class _CustomUnpickler(pickle.Unpickler):
    """A custom unpickler to make sure the deserialized REQUIRED is the same instance.

    TODO(willsong): There are other ways to write the RequiredFieldValue where this wouldn't be
    necessary. We should pursue those later and remove this unpickler.
    """

    def find_class(self, module: str, name: str) -> type:
        if name == "RequiredFieldValue":
            return _DummyRequired
        return super().find_class(module, name)


def _serialize_with_closure(obj: Any) -> bytes:
    """Serialize an object, and captures any objects used in closures.

    Args:
        obj: Object to be serialized

    Returns:
        Serialized bytes.
    """
    return cloudpickle.dumps(obj)


def _deserialize_with_closure(data: bytes) -> Any:
    """Inverse operation of serialize_with_closure.

    Args:
        data: Bytes to be deserialized.

    Returns:
        The original object.
    """
    return _CustomUnpickler(io.BytesIO(data)).load()


class _DeepCopyWithClosureFnWrapper:
    """A wrapper for TrainerConfigFn that tries to serialize them before calling.
    This helps prevent cases where trainer config functions share states in their closures. A slight
    optimization done here is to make sure that we don't have to serialize unless the function is
    called, and we'll only serialize it once. Each call will deserialize from the same serialized
    data.
    """

    def __init__(self, fn: TrainerConfigFn):
        self._fn = fn
        # TODO(willsong) Should we serialize immediately? This will introduce additional cost, but
        # elimnate risks where shared states get modified before calls are made. The likelihood of
        # this is small since only functions that share the same closure could corrupt the state
        # after they are called, but there could be global variables that are shared.
        self._serialized_fn = None

    def __call__(self, *args, **kwargs):
        if self._serialized_fn is None:
            try:
                self._serialized_fn = _serialize_with_closure(self._fn)
            except TypeError as e:
                raise TypeError(
                    "@config_map_cache is not compatible with functions that have unserializable "
                    "objects in its closures. Plesae consider removing them or remove "
                    "@config_map_cache."
                ) from e

        # TODO(willsong): Investigate whether we should cache the deserialized object. It seems like
        # all config generators are idempotent according to a test where we call each GC
        # verifiaction function twice, but theoretically people could still write something that
        # breaks, some of which could be mitigated by putting more guardrails when building the
        # Config.
        return _deserialize_with_closure(self._serialized_fn)(*args, **kwargs)


def _wrap_with_deep_copy_with_closure(fn: TrainerConfigFn) -> TrainerConfigFn:
    """Create a wrapper function for TrainerConfigFn that will do a deepcopy before calling fn.

    Args:
        fn: TrainerConfigFn to be wrapped.

    Returns:
        A wrapped TrainerConfigFn where the function goes through cloudpickle serialization and
        deserialization first to remove the potential side effects of having mutable objects in its
        closures being modified by other code paths.
    """
    wrapped_fn_obj = _DeepCopyWithClosureFnWrapper(fn)

    # Instead of using a class object, we return a function to be compatible with the type checking
    # in Config in case this function is used as a value.
    def wrapper_function(*args, **kwargs):
        return wrapped_fn_obj(*args, **kwargs)

    # To identify that these have been wrapped for deepcopy.
    wrapper_function._is_wrapped_with_deepcopy = True  # pylint: disable=protected-access

    # Sometimes TrainerConfigFn are used as values in config. Set these to the same as fn so that
    # debug strings print the original function, if applicable.
    if hasattr(fn, "__module__"):
        wrapper_function.__module__ = fn.__module__
    if hasattr(fn, "__name__"):
        wrapper_function.__name__ = fn.__name__

    return wrapper_function


def config_map_cache(
    func: Callable[..., dict[str, TrainerConfigFn]],
) -> Callable[..., dict[str, TrainerConfigFn]]:
    """A decorator for named_trainer_config that caches the config_map.

    Args:
        func: Function to be decorated. Usually a named_trainer_config function that returns a
            dictionary of config generator functions with string keys.

    Returns:
        Similar behavior to adding @cache to func, except we'll also track the value of
        get_data_dir().
    """

    # Add an argument reserved for data_dir, which may be different depending on when/how the config
    # gen is called.
    def capture_data_dir_wrapper(_: str, *args, **kwargs) -> dict[str, TrainerConfigFn]:
        return func(*args, **kwargs)

    cached_func = cache(capture_data_dir_wrapper)

    @wraps(func)
    def wrapper(*args, **kwargs) -> dict[str, TrainerConfigFn]:
        result = cached_func(get_data_dir(), *args, **kwargs)
        new_result = {}
        for key, fn in result.items():
            if getattr(fn, "_is_wrapped_with_deepcopy", False):
                # No need to wrap the function again. This can happen when a parent module
                # recursively combines other named_trainer_configs output from its submodules.
                new_result[key] = fn
            else:
                # Wrap it so that a true deepcopy with closures will happen first.
                new_result[key] = _wrap_with_deep_copy_with_closure(fn)
        return new_result

    return wrapper

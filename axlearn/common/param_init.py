"""Modules for configurable parameter initialization."""
import re
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import jax
from absl import logging
from jax import numpy as jnp

from axlearn.common.config import Configurable, InstantiableConfig, config_class, config_for_class

Shape = Sequence[int]

PARAM_REGEXP_WEIGHT = ".*weight$"
PARAM_REGEXP_BIAS = ".*bias$"
PARAM_REGEXP_SCALE = ".*scale$"


class FanAxes(NamedTuple):
    """FanAxes describes axis indices corresponding to input, output, and batch axes.

    Note: axes not listed in {in,out,batch}_axis are assumed to be the “receptive field”
    (convolution kernel spatial axes).

    Used for
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html.
    """

    # Input axis or sequence of axes of the fan "input" dimension.
    in_axis: Union[Tuple[int, ...], int]
    # Output axis or sequence of axes of the fan "output" dimension.
    out_axis: Union[Tuple[int, ...], int]
    # Batch axis or sequence of axes that should be ignored when computing fan.
    batch_axis: Union[Tuple[int, ...], int] = ()


class Initializer:
    """Base class for initializers."""

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(type(self))

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        raise NotImplementedError(type(self))


class ConstantInitializer(Initializer):
    """Constant initializer."""

    def __init__(self, value: Any):
        self._value = value

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        return jnp.full(shape=shape, fill_value=self._value, dtype=dtype)

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        return f"constant({self._value})"


class GaussianInitializer(Initializer):
    """Gaussian initializer."""

    def __init__(self, std: float):
        self._std = std

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        return jax.random.normal(prng_key, shape=shape, dtype=dtype) * self._std

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        return f"normal(0, {self._std}^2)"


def truncated_normal(stddev: float = 1e-2, dtype: jnp.dtype = jnp.float_):
    """Truncated normal variant of jax.nn.initializers.

    Args:
        stddev: Standard deviation of gaussian to draw from.
        dtype: Type to draw from.

    Returns:
        initializer fn.
    """

    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        # constant is stddev of standard normal truncated to (-2, 2)
        truncated_stddev = stddev / jnp.array(0.87962566103423978, dtype)
        return jax.random.truncated_normal(key, -2, 2, shape, dtype) * truncated_stddev

    return init


def uniform(scale: float = 1.0, dtype: jnp.dtype = jnp.float_):
    """Uniform initializer.

    Args:
        scale: output will be uniformly between [-scale, scale).
        dtype: sample dtype.

    Returns:
        initializer fn.
    """

    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape=shape, dtype=dtype, minval=-scale, maxval=scale)

    return init


class WeightInitializer(Configurable, Initializer):
    """Default weight initializer.

    The default config is Xavier initializer, i.e.
    uniform[-x, x) where x = sqrt(6. / (in + out)).
    """

    @config_class
    class Config(Configurable.Config):
        # The default scale for weight initialization.
        scale: float = 1.0
        # Type of fan to compute, supported values are "fan_in", "fan_out", "fan_avg" and None.
        # If None then no fan scaling factor is computed.
        fan: str = "fan_avg"
        # Weight distribution: "uniform", "normal", or "truncated_normal".
        distribution: str = "uniform"

    # pylint: disable-next=too-many-return-statements
    def initialize(
        self,
        name,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        cfg = self.config
        if cfg.fan is not None:
            if axes is None:
                raise NotImplementedError(f"axes must be provided: {name}")
            initializer = jax.nn.initializers.variance_scaling(
                cfg.scale,
                mode=cfg.fan,
                distribution=cfg.distribution,
                dtype=dtype,
                **axes._asdict(),
            )
        elif cfg.distribution == "uniform":
            # Uniform[-cfg.scale, cfg.scale).
            initializer = uniform(cfg.scale, dtype=dtype)
        elif cfg.distribution == "normal":
            initializer = jax.nn.initializers.normal(cfg.scale, dtype=dtype)
        elif cfg.distribution == "truncated_normal":
            initializer = truncated_normal(cfg.scale, dtype=dtype)
        else:
            raise NotImplementedError(
                f"Unsupported fan {cfg.fan} and distribution {cfg.distribution}."
            )
        return initializer(prng_key, shape=shape)

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        cfg = self.config
        if cfg.fan is not None:
            assert axes is not None
            assert shape is not None
            if cfg.distribution == "uniform":
                return (
                    f"uniform(-sqrt({3 * cfg.scale} / {cfg.fan}), "
                    f"sqrt({3 * cfg.scale} / {cfg.fan})), shape={shape}, axes={axes}"
                )
            elif cfg.distribution == "normal":
                return f"normal(0, {cfg.scale} / {cfg.fan}), shape={shape}, axes={axes}"
            elif cfg.distribution == "truncated_normal":
                # TODO(zhiyunlu): check what is a good way to represent truncated_normal.
                return (
                    f"truncated_normal(-2, 2) * sqrt({cfg.scale} / {cfg.fan}) / 0.88, "
                    f"shape={shape}, axes={axes}"
                )
            else:
                raise NotImplementedError(
                    f"variance_scaling does not support distribution {cfg.distribution}."
                )
        elif cfg.distribution == "uniform":
            # Uniform[-cfg.scale, cfg.scale).
            return f"uniform({-cfg.scale}, {cfg.scale})"
        elif cfg.distribution == "normal":
            return f"normal(0, {cfg.scale}^2)"
        elif cfg.distribution == "truncated_normal":
            # TODO(zhiyunlu): check what is a good way to represent truncated_normal.
            return f"truncated_normal(-2, 2) * {cfg.scale} / 0.88"
        else:
            raise NotImplementedError(
                f"Unsupported fan {cfg.fan} and distribution {cfg.distribution}."
            )


class DefaultInitializer(Configurable, Initializer):
    """The default initializer."""

    @config_class
    class Config(Configurable.Config):
        # Initializer rules dictionary.
        # Each entry is a (regexp, config) pair. The first match (the insertion order) "wins".
        # Note that `DefaultInitializer.__init__` populates the dict with some default values,
        # which will be consulted after user-specified entries.
        init_by_param_name: Dict[str, InstantiableConfig] = {}

    def __init__(self, cfg: Config):
        """Default initializers for common param names if not specified by the user."""
        # The default bias initializer is constant(0.).
        cfg.init_by_param_name.setdefault(
            PARAM_REGEXP_BIAS, config_for_class(ConstantInitializer).set(value=0.0)
        )
        # The default scale initializer is constant(1.).
        cfg.init_by_param_name.setdefault(
            PARAM_REGEXP_SCALE, config_for_class(ConstantInitializer).set(value=1.0)
        )
        # The default weight initializer is WeightInitializer.
        cfg.init_by_param_name.setdefault(PARAM_REGEXP_WEIGHT, WeightInitializer.default_config())
        super().__init__(cfg)

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        cfg = self.config
        # We first go through user specified entries, and then followed by
        # PARAM_REGEXP_BIAS, PARAM_REGEXP_SCALE, and PARAM_REGEXP_WEIGHT.
        for pattern, init in cfg.init_by_param_name.items():
            if re.fullmatch(pattern, name):
                logging.info(
                    "DefaultInitializer: %s matches %s: initializer=%s",
                    pattern,
                    name,
                    init.debug_string(field_separator=" "),
                )
                return init.instantiate().initialize(
                    name, prng_key=prng_key, shape=shape, dtype=dtype, axes=axes
                )

        raise NotImplementedError(f"Unsupported parameter name ({name})")

    # pylint: disable-next=arguments-differ
    def debug_string(
        self,
        *,
        name: str,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        cfg = self.config
        for pattern, init in cfg.init_by_param_name.items():
            if re.fullmatch(pattern, name):
                return init.instantiate().debug_string(name=name, shape=shape, axes=axes)

        raise NotImplementedError(f"Unsupported parameter name ({name})")


class PerGroupInitializer(Configurable, Initializer):
    """The per-group initializer.

    This initializer splits the last axis into a number of groups and initializes each group
    separately. The last axis needs to be divisible by the number of groups. One application
    is to initialize depthwise-separable convolution correctly, i.e., for depthwise convolution
    the fan_out computation needs to consider the number of groups in the convolution.

    See e.g., related discussions in other frameworks:
        https://github.com/pytorch/pytorch/issues/23854
        https://github.com/huggingface/pytorch-image-models/issues/84
    """

    @config_class
    class Config(Configurable.Config):
        # The initializer used to initialize weights for each group.
        initializer: Optional[InstantiableConfig] = DefaultInitializer.default_config()
        # The number of groups.
        num_groups: int = 1

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        if cfg.initializer is not None:
            self.initializer = cfg.initializer.instantiate()
        else:
            self.initializer = DefaultInitializer.default_config().instantiate()

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        """Per-group initialization.

        Divides last axis into num_groups parts and initializes each group with initializer.
        """
        cfg = self.config

        if shape[-1] % cfg.num_groups != 0:
            raise ValueError(f"{shape[-1]=} must be divisible by {cfg.num_groups=}.")
        shape_per_group = list(shape[:-1]) + [shape[-1] // cfg.num_groups]

        def init(prng_key_i: jax.random.KeyArray) -> jnp.ndarray:
            return self.initializer.initialize(
                name, prng_key=prng_key_i, shape=shape_per_group, dtype=dtype, axes=axes
            )

        # Preserve original behavior of initializing with original key.
        if cfg.num_groups == 1:
            param = init(prng_key)
        else:
            param = jax.vmap(init)(jax.random.split(prng_key, num=cfg.num_groups))
            param = jnp.concatenate(param, axis=-1)

        return param

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        initializer_debug_string = self.initializer.debug_string(name=name, shape=shape, axes=axes)
        return f"{initializer_debug_string} [PerGroupInitializer]"

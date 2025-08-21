# Copyright © 2025 Apple Inc.
"""
References on feature map:

    - The Hedgehog & the Porcupine: Expressive Linear Attentions with
Softmax Mimicry (https://arxiv.org/pdf/2402.04347)
"""
import enum
import functools
from typing import Callable, NamedTuple

import jax.nn as jnn
import jax.numpy as jnp

from axlearn.common.utils import Tensor


class FeatureMap(enum.Enum):
    SOFTMAX = "softmax"
    RELU = "relu"


class FeatureMapFn(NamedTuple):
    """A named tuple to hold the forward and backward functions of a feature map."""

    fwd: Callable
    bwd: Callable


def get_feature_map(feat_map: FeatureMap) -> FeatureMapFn:
    """Get the feature map function and its backward function."""
    if feat_map == FeatureMap.SOFTMAX:
        return FeatureMapFn(sm2_fwd, sm2_bwd)
    elif feat_map == FeatureMap.RELU:
        return FeatureMapFn(relu2_fwd, relu2_bwd)
    else:
        raise ValueError(f"Unknown feature map: {feat_map}")


def inner_float32(func):
    """Decorator to convert inputs to float32 before calling the function,
    and convert the output back to the original dtype. This is useful for ensuring numerical
    stability for feature map computation.
    """

    @functools.wraps(func)
    def wrapper(*args):
        original_dtype = args[0].dtype
        converted_args = [arg.astype(jnp.float32) for arg in args]
        result = func(*converted_args)
        return result.astype(original_dtype)

    return wrapper


@inner_float32
def sm_fwd(x: Tensor) -> Tensor:
    """Softmax feature map, forward pass"""
    max_x = jnp.max(x, axis=-1, keepdims=True)
    y = x - max_x
    exp_y = jnp.exp(y)
    sum_exp_y = jnp.sum(exp_y, axis=-1, keepdims=True)
    return exp_y / sum_exp_y


@inner_float32
def sm_bwd(y: Tensor, dy: Tensor) -> Tensor:
    """Softmax feature map, backward pass"""
    return y * (dy - jnp.sum(dy * y, axis=-1, keepdims=True))


@inner_float32
def sm2_fwd(x: Tensor) -> Tensor:
    """Softmax feature map, forward pass [softmax(x), softmax(-x)]"""
    y1 = sm_fwd(x)
    y2 = sm_fwd(-x)
    return jnp.concatenate([y1, y2], axis=-1)


@inner_float32
def sm2_bwd(y: Tensor, dy: Tensor) -> Tensor:
    """Softmax feature map, backward pass [softmax(x), softmax(-x)]"""
    y1, y2 = jnp.split(y, 2, axis=-1)
    dy1, dy2 = jnp.split(dy, 2, axis=-1)
    dx1 = sm_bwd(y1, dy1)
    dx2 = -sm_bwd(y2, dy2)
    return dx1 + dx2


@inner_float32
def relu_fwd(x: Tensor) -> Tensor:
    """ReLU feature map, forward pass"""
    # don't use jnp.maximum(x, 0.0), as its backward seems not well defined
    return jnn.relu(x)


@inner_float32
def relu_bwd(y: Tensor, dy: Tensor) -> Tensor:
    """ReLU feature map, backward pass"""
    return jnp.where(y > 0.0, dy, 0)


@inner_float32
def relu2_fwd(x: Tensor) -> Tensor:
    """ReLU feature map, forward pass [relu(x), relu(-x)]"""
    y1 = relu_fwd(x)
    y2 = relu_fwd(-x)
    return jnp.concatenate([y1, y2], axis=-1)


@inner_float32
def relu2_bwd(y: Tensor, dy: Tensor) -> Tensor:
    """ReLU feature map, backward pass [relu(x), relu(-x)]"""
    # Split the outputs and gradients
    y1, y2 = jnp.split(y, 2, axis=-1)
    dy1, dy2 = jnp.split(dy, 2, axis=-1)

    dx1 = relu_bwd(y1, dy1)
    # pylint: disable=invalid-unary-operand-type
    dx2 = -relu_bwd(y2, dy2)
    return dx1 + dx2

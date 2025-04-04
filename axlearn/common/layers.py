# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# huggingface/transformers:
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# bzhangGo/rmsnorm:
# Copyright (c) 2019, Biao Zhang. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

# pylint: disable=too-many-lines
"""Basic layers."""

import enum
import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import jax
from absl import logging
from jax import nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common import convolution
from axlearn.common.base_layer import BaseLayer, FactorizationSpec, ParameterNoise, ParameterSpec
from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    InstantiableConfig,
    Required,
    UnknownFieldError,
    config_class,
)
from axlearn.common.convolution import Conv2D
from axlearn.common.loss import binary_cross_entropy, categorical_hinge_loss, cross_entropy
from axlearn.common.metrics import WeightedScalar
from axlearn.common.metrics_classification import precision_recall_f_score
from axlearn.common.module import Module, child_context, nowrap
from axlearn.common.normalize import l2_normalize
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    DefaultInitializer,
    FanAxes,
    WeightInitializer,
    constant_initializer,
)
from axlearn.common.quantized_dot_general.layers import DenseGeneralBaseLayer
from axlearn.common.utils import (
    NestedTensor,
    Tensor,
    maybe_shard,
    partial_with_fn_metadata,
    with_sharding_constraint,
)

# TODO(dhwang2): remove them.
# DEPRECATED: Avoid using this; instead, directly import convolution.py. Aliases for convolution are
# provided for backward compatibility.
ConvPaddingType = convolution.ConvPaddingType
Conv1D = convolution.Conv1D
Conv2D = convolution.Conv2D
Conv2DTranspose = convolution.Conv2DTranspose
Conv2DWith1DPadding = convolution.Conv2DWith1DPadding
Conv3D = convolution.Conv3D


def get_activation_fn(name) -> Callable[[Tensor], Tensor]:
    if name == "linear":
        return lambda x: x
    elif name == "quick_gelu":
        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flax_utils.py#L63
        return lambda x: x * jax.nn.sigmoid(1.702 * x)
    elif name == "exact_gelu":
        # This is the exact gelu form, which is equivalent to GELUActivation in HF.
        # nn.gelu is by default approximate=True.
        # exact gelu is slower than the approximate: https://github.com/google/jax/issues/4428
        return lambda x: nn.gelu(x, approximate=False)
    elif name == "squared_relu":
        # Squared ReLU as proposed in Primer: https://arxiv.org/abs/2109.08668
        return lambda x: jnp.square(jax.nn.relu(x))
    elif name.startswith("nn."):
        return getattr(nn, name[3:])
    elif name.startswith("jnp."):
        return getattr(jnp, name[4:])
    else:
        raise NotImplementedError(f"Unsupported activation function {name}")


class RedirectToSharedModule(BaseLayer):
    """A layer that redirects methods to shared modules."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures RedirectToSharedModule."""

        # The name of the shared module.
        # This is the `shared_module_name` given to `_share_with_descendants()`.
        shared_module: Required[str] = REQUIRED
        # A mapping from redirection layer method name to the target layer method.
        # If empty, assume {"forward": "forward"}.
        method_map: dict[str, str] = {}

        def set(self, **kwargs) -> "RedirectToSharedModule.Config":
            try:
                super().set(**kwargs)
            except UnknownFieldError as e:
                logging.info("Ignoring %s", e)
                # We intentionally ignore this exception.
            return self

    def _methods_to_wrap_for_auto_child_context(self) -> dict[str, Callable]:
        cfg: RedirectToSharedModule.Config = self.config
        method_dict = {}
        for source_method, target_method in (cfg.method_map or {"forward": "forward"}).items():
            method_fn = partial_with_fn_metadata(
                getattr(type(self), "_redirect"), redirection_target_method=target_method
            )
            method_dict[source_method] = method_fn
        return method_dict

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            raise AttributeError(
                f"{', '.join(e.args)}. Should '{name}' be specified in `cfg.method_map`?"
            ) from e

    def _redirect(self, *args, redirection_target_method: str, **kwargs) -> Any:
        cfg: RedirectToSharedModule.Config = self.config
        shared_module = self.get_shared_module(cfg.shared_module)
        return getattr(shared_module.module, redirection_target_method)(*args, **kwargs)


def get_dropout_mask(shape: tuple[int, ...], *, prng_key: Tensor, rate: float):
    """Returns a bool dropout mask for the specified tensor shape where True indicates dropout."""
    return jax.random.bernoulli(prng_key, rate, shape)


def dropout(
    x: Tensor, *, rate: float, prng_key: Optional[Tensor] = None, mask: Optional[Tensor] = None
):
    """Performs dropout on `x` according to dropout rate or mask.

    After dropout, `x` will be rescaled by 1 / (1 - rate). If `mask` is provided, use `mask`.
    Otherwise, generate a dropout mask using `prng_key` and `rate`.

    Args:
        x: Input tensor.
        rate: Dropout rate.
        prng_key: PRNG key used for mask generation. Required if `mask` is None.
        mask: A boolean mask with the same shape as x. If provided, `prng_key` will be ignored.
            Any values in `x` where `mask` is True will be dropped.
    """
    if not 0 < rate < 1:
        raise ValueError(f"Dropout rate must be between 0 and 1. Got {rate=}")
    if mask is None:
        if prng_key is None:
            raise ValueError("prng_key must be provided when mask is not specified.")
        mask = get_dropout_mask(x.shape, prng_key=prng_key, rate=rate)
    return jnp.where(mask, 0, x) / (1 - rate)


class Dropout(BaseLayer):
    """The dropout layer."""

    @config_class
    class Config(BaseLayer.Config):
        rate: Optional[float] = None  # The dropout rate (i.e., 1 - keep_prob).

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if not self.is_training or cfg.rate is None or cfg.rate == 0:
            return x
        return dropout(x, prng_key=self.prng_key, rate=cfg.rate)

    def get_prng_key(self) -> Tensor:
        return self.prng_key


class DropToken(BaseLayer):
    """The drop token layer.

    Ref: https://arxiv.org/pdf/2212.00794.pdf
    """

    @config_class
    class Config(BaseLayer.Config):
        num_cls_tokens: int = 0
        rate: float = 0  # The drop rate (i.e., 1 - keep_prob).

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        assert 0 <= cfg.rate < 1

    def forward(self, x: Tensor) -> Tensor:
        """The forward function of DropTokens.

        Args:
            x: A Tensor with shape [batch_size, num_tokens, dim]

        Returns:
            The original Tensor, if rate==0 or None or is_training==False.
            If is_training==True, then return a Tensor with shape
                [batch_size, num_cls_tokens
                  + int((num_tokens - num_cls_tokens) * (1-rate)), dim],
                where the elements are randomly chosen from x.

        TODO(bwzhang@) Rewrite this when we support CLS packing.
        For the CLS packing, the CLS might not be the first several tokens
        of the sequence.
        """
        cfg = self.config
        batch_size = x.shape[0]
        tokens = x[:, cfg.num_cls_tokens :]
        num_tokens = tokens.shape[1]
        num_chosen_tokens = int(round((1 - cfg.rate) * num_tokens))
        assert 0 < num_chosen_tokens <= num_tokens
        if not self.is_training or num_chosen_tokens == num_tokens:
            return x
        patch_id = jnp.tile(jnp.arange(0, num_tokens), (batch_size, 1))
        sampled_id = jax.random.permutation(self.prng_key, patch_id, axis=1, independent=True)[
            :, :num_chosen_tokens
        ]
        sampled_id_one_hot = jax.nn.one_hot(sampled_id, num_tokens)
        sampled_patch = jnp.einsum("bnd,bkn->bkd", tokens, sampled_id_one_hot)
        if cfg.num_cls_tokens > 0:
            cls_tokens = x[:, : cfg.num_cls_tokens]
            sampled_patch = jnp.concatenate((cls_tokens, sampled_patch), 1)
        return sampled_patch


def set_dropout_rate_recursively(
    cfg: ConfigBase, dropout_rate: Optional[float], set_only_if_none: bool = False
):
    """Sets Dropout.Config.rate recursively.

    Args:
        cfg: the root config under which to look for Dropout.Config.
        dropout_rate: the target dropout rate.
        set_only_if_none: override Dropout.Config.rate to `dropout_rate` only if the original rate
            is None.
    """

    def is_dropout_config(cfg):
        return isinstance(cfg, Dropout.Config)

    def visit_fn(_, value):
        if is_dropout_config(value) and (not set_only_if_none or value.rate is None):
            value.rate = dropout_rate

    def enter_fn(_, value, default_kv):
        return None if is_dropout_config(value) else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)


def set_layer_norm_eps_recursively(cfg: ConfigBase, eps: float, set_only_if_none: bool = False):
    """Sets LayerNorm.Config.eps recursively.

    Args:
        cfg: The root config under which to look for LayerNorm.Config.
        eps: The target value.
        set_only_if_none: Override LayerNorm.Config.eps to `eps` only if the original is None.
    """

    def is_layer_norm_config(cfg):
        return isinstance(cfg, LayerNorm.Config)

    def visit_fn(_, value):
        if is_layer_norm_config(value) and (not set_only_if_none or value.eps is None):
            value.eps = eps

    def enter_fn(_, value, default_kv):
        return None if is_layer_norm_config(value) else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)


class BaseNormalizationLayer(BaseLayer):
    """The base class for normalization layers."""

    @config_class
    class Config(BaseLayer.Config):
        # Input feature dim.
        input_dim: Required[int] = REQUIRED

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        """Applies the normalization to inputs.

        Args:
            x: tensor of shape [batch_size, ...].
            paddings: optional 0/1 tensor of shape [batch_size, seq_len] for sequence paddings.

        Returns:
            Normalized tensor of the same shape as x.
        """
        raise NotImplementedError(type(self))


class LayerNormStateless(BaseNormalizationLayer):
    """A state-free LayerNorm.

    Pytorch LayerNorm has no state option by setting elementwise_affine=False.
    This is used in Diffusion Transformer.
    Ref: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    @config_class
    class Config(BaseNormalizationLayer.Config):
        # The epsilon.
        eps: float = 1e-8
        # Cast input to this dtype for the 'forward' call. If None, do not cast.
        forward_dtype: Optional[jnp.dtype] = jnp.float32

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        del paddings  # paddings do not affect LayerNorm results
        cfg = self.config
        x_dtype = x.dtype
        if cfg.forward_dtype is not None:
            x = x.astype(cfg.forward_dtype)
        x_mean = x.mean(axis=-1, keepdims=True)
        x -= x_mean
        variance = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + cfg.eps)
        x = x.astype(x_dtype)
        return x


class LayerNorm(LayerNormStateless):
    """Reference: https://arxiv.org/abs/1607.06450."""

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
            "bias": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
        }

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        x = super().forward(x, paddings=paddings)
        x = x * self.parameters["scale"] + self.parameters["bias"]
        return x


class RMSNormStateless(BaseNormalizationLayer):
    """Stateless version of https://github.com/bzhangGo/rmsnorm."""

    @config_class
    class Config(BaseNormalizationLayer.Config):
        # The epsilon.
        eps: float = 1e-8
        # Cast input to this dtype for the 'forward' call. If None, do not cast.
        forward_dtype: Optional[jnp.dtype] = jnp.float32
        # If not None, how to partition input activation values.
        input_partition_spec: Optional[tuple[Optional[str]]] = None
        # If not None, how to partition output activation values.
        output_partition_spec: Optional[tuple[Optional[str]]] = None

    def _forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x = maybe_shard(x, cfg.input_partition_spec)
        x_dtype = x.dtype
        if cfg.forward_dtype is not None:
            x = x.astype(cfg.forward_dtype)
        moment2 = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(moment2 + cfg.eps)
        x = x.astype(x_dtype)
        return x

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        del paddings  # paddings do not affect LayerNorm results
        cfg = self.config
        x = self._forward(x)
        x = maybe_shard(x, cfg.output_partition_spec)
        return x


class RMSNorm(RMSNormStateless):
    """Reference: https://github.com/bzhangGo/rmsnorm."""

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
        }

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        del paddings  # paddings do not affect LayerNorm results
        cfg = self.config
        x = self._forward(x) * self.parameters["scale"]
        x = maybe_shard(x, cfg.output_partition_spec)
        return x


def normalize_sum(x: Tensor, eps: float = 1e-8, axis=-1) -> Tensor:
    sum1 = x.sum(axis=axis, keepdims=True)
    return x / (sum1 + eps)


class L2Norm(BaseLayer):
    """L2 Norm layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures L2Norm."""

        eps: float = 1e-8
        forward_dtype: Optional[jnp.dtype] = jnp.float32

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        if cfg.forward_dtype is not None:
            x = x.astype(cfg.forward_dtype)

        x = l2_normalize(x, cfg.eps)
        x = x.astype(x_dtype)

        return x


def _compute_moments_with_paddings(
    x: Tensor,
    *,
    paddings: Tensor,
    reduction_axis: Sequence[int],
    keepdims: bool = False,
) -> tuple[Tensor, Tensor]:
    """Computes mean and variance over sequence data.

    Args:
        x: inputs tensor of shape [batch_size, seq_len, ...].
        paddings: 0/1 tensor of shape [batch_size, seq_len].
        reduction_axis: a list of axes to compute moments over.
        keepdims: If this is set to True, the reduced axes are left
            in the result as singleton dimensions.

    Returns:
        (mean, variance), with the same shape as `x` except for axes specified in `reduction_axis`,
            which will have dim of 1.
    """
    expanded_paddings = jnp.expand_dims(paddings, axis=tuple(range(2, x.ndim)))
    mask = 1 - expanded_paddings
    sum_x = jnp.sum(x * mask, axis=reduction_axis, keepdims=keepdims)
    count_x = jnp.sum(jnp.ones_like(x) * mask, axis=reduction_axis, keepdims=keepdims)
    denom_x = jnp.maximum(count_x, 1.0)
    mean = sum_x / denom_x
    sum_x2 = jnp.sum((x - mean) ** 2 * mask, axis=reduction_axis, keepdims=keepdims)
    variance = sum_x2 / denom_x
    return mean, variance


def _compute_mean_square_with_paddings(
    x: Tensor,
    *,
    paddings: Tensor,
    reduction_axis: Sequence[int],
) -> Tensor:
    """Computes root mean square moments over sequence data.

    Args:
        x: inputs tensor of shape [batch_size, seq_len, ...].
        paddings: 0/1 tensor of shape [batch_size, seq_len].
        reduction_axis: a list of axes to compute moments over.

    Returns:
        mean_square: with the same shape as `x` except for axes specified in `reduction_axis`,
            which will have dim of 1.
    """
    expanded_paddings = jnp.expand_dims(paddings, axis=tuple(range(2, x.ndim)))
    mask = 1 - expanded_paddings
    sum_x2 = jnp.sum((x * x) * mask, axis=reduction_axis, keepdims=True)
    count_x2 = jnp.sum(jnp.ones_like(x) * mask, axis=reduction_axis, keepdims=True)
    # If all elements of `padding` are 1 (i.e., jnp.all(padding == 1)), mean_square will be 0.
    # However, the computation remains stable due to max(1, count).
    mean_square = sum_x2 / jnp.maximum(count_x2, 1.0)
    return mean_square


class NormType(enum.Enum):
    """NormType defines the normalization methods for the GroupNorm class.

    Available normalization types:
    - **LAYERNORM**: Applies layernorm across all axes except for the group and batch axes.
    - **RMSNORM**: Applies rmsnorm across all axes except for the group and batch axes.
    """

    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"


class GroupNorm(BaseNormalizationLayer):
    """GroupNorm provides group-wise normalization for inputs.

    The choice of `norm_type` and `norm_axes` should be guided by the specific domain. In the vision
    domain LayerNorm is typically applied across all axes except the group and batch dimensions,
    e.g., normalization across spatial dims and feature channels (https://arxiv.org/abs/1803.08494).
    In the text domain, normalization may be performed along the feature axis alone, e.g., RMSNorm
    is applied along the last feature axis in [Mamba2](https://arxiv.org/abs/2405.21060).

    In the future, we may consider supporting sliding window or causal group norm, e.g.,
    https://github.com/tensorflow/lingvo/blob/b26149e423cd51498bd884ffd37a6b5ceb244d68/lingvo/core/bn_layers.py#L756-L764
    """

    @config_class
    class Config(BaseNormalizationLayer.Config):
        """Configures GroupNorm."""

        # The number of groups.
        num_groups: Required[int] = REQUIRED
        # The epsilon.
        eps: float = 1e-8
        # Cast input to this dtype for the 'forward' call. If None, do not cast.
        forward_dtype: Optional[jnp.dtype] = jnp.float32
        # LAYERNORM` or `RMSNORM`.
        # If None, assumes `LAYERNORM` (for backwards compatibility).
        norm_type: Optional[NormType] = None
        # Axes to apply the normalization in addition to the group size axis.
        # E.g., (1,) means normalizing normalizing along the time axis (as well as the group size
        # axis) if the input tensor has shape (batch, time, dim); (1, -1) has the same effect as
        # (1,) as the last dim (i.e., the group size axis) is automatically added.
        # If None, reduces along all dims except for the num of group and batch axes (for
        # backwards compatibility).
        norm_axes: Optional[Union[tuple[int, ...], int]] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        if cfg.norm_type is None:
            cfg.norm_type = NormType.LAYERNORM
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        if cfg.norm_type == NormType.LAYERNORM:
            return {
                "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
                "bias": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
            }
        else:
            return {
                "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
            }

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        """Applies group normalization.

        Args:
            x: inputs tensor of shape [batch_size, height, width, channel] if x.ndim = 4,
                else [batch_size, height, channel].
            paddings: optional 0/1 tensor of shape [batch_size, height]. Useful for sequence
                data where `height` is `time`.

        Returns:
            Tensor of the same shape as x.

        Raises:
            ValueError: if num_groups does not divide input_dim.
            ValueError: if num_groups axis is included in norm_axes.
        """
        cfg = self.config
        if cfg.num_groups <= 0 or cfg.input_dim % cfg.num_groups != 0:
            raise ValueError(f"num_groups ({cfg.num_groups}) must divide dim ({cfg.input_dim})")
        group_size = cfg.input_dim // cfg.num_groups
        x_dtype = x.dtype
        if cfg.forward_dtype is not None:
            x = x.astype(cfg.forward_dtype)
        # Reshape to [..., num_groups, group_size].
        y = jnp.reshape(x, list(x.shape[:-1]) + [cfg.num_groups, group_size])

        # Default norm axes: all axes except for the group and batch axes.
        if cfg.norm_axes is None:
            reduction_axis = list(range(1, y.ndim - 2)) + [y.ndim - 1]
        else:
            reduction_axis = cfg.norm_axes

        # Normalize to a list of non-negative axes.
        if isinstance(reduction_axis, int):
            reduction_axis = [reduction_axis]
        reduction_axis = [axis if axis >= 0 else y.ndim + axis for axis in reduction_axis]

        if y.ndim - 2 in reduction_axis:
            raise ValueError("GroupNorm should not normalize along the num_groups axis.")

        if 0 in reduction_axis:
            raise ValueError("GroupNorm should not normalize along the batch axis.")

        # Add the group size axis to the reduction axis.
        if y.ndim - 1 not in reduction_axis:
            reduction_axis.append(y.ndim - 1)

        if cfg.norm_type == NormType.LAYERNORM:
            if paddings is None:
                mean = jnp.mean(y, axis=reduction_axis, keepdims=True)
                variance = jnp.mean((y - mean) ** 2, axis=reduction_axis, keepdims=True)
            else:
                mean, variance = _compute_moments_with_paddings(
                    x=y,
                    paddings=paddings,
                    reduction_axis=reduction_axis,
                    keepdims=True,
                )

            y = (y - mean) * jax.lax.rsqrt(variance + cfg.eps)
            x = jnp.reshape(y, x.shape)
            x = x.astype(x_dtype)
            x = x * self.parameters["scale"] + self.parameters["bias"]
        else:
            if paddings is None:
                msquare = (y * y).mean(axis=reduction_axis, keepdims=True)
            else:
                msquare = _compute_mean_square_with_paddings(
                    x=y,
                    paddings=paddings,
                    reduction_axis=reduction_axis,
                )
            y = y * jax.lax.rsqrt(msquare + cfg.eps)
            x = jnp.reshape(y, x.shape)
            x = x.astype(x_dtype)
            x = x * self.parameters["scale"]
        return x


class BatchNorm(BaseNormalizationLayer):
    """https://arxiv.org/abs/1502.03167."""

    @config_class
    class Config(BaseNormalizationLayer.Config):
        """Configures BatchNorm."""

        # The decay for computing moving mean/variance.
        decay: float = 0.999
        # The epsilon.
        eps: float = 1e-8
        # Cast input to this dtype for the 'forward' call.  If None, do not cast.
        forward_dtype: Optional[jnp.dtype] = jnp.float32

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
            "bias": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
            "moving_mean": ParameterSpec(
                shape=[cfg.input_dim],
                dtype=jnp.float32,
                mesh_axes=(None,),
                initializer=constant_initializer(0.0),
                weight_decay_scale=0,
            ),
            "moving_variance": ParameterSpec(
                shape=[cfg.input_dim],
                dtype=jnp.float32,
                mesh_axes=(None,),
                initializer=constant_initializer(1.0),
                weight_decay_scale=0,
            ),
        }

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        if cfg.forward_dtype is not None:
            x = x.astype(cfg.forward_dtype)
        reduction_axis = tuple(range(x.ndim - 1))
        if self.is_training:
            if paddings is None:
                mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
                variance = jnp.mean((x - mean) ** 2, axis=reduction_axis, keepdims=False)
            else:
                mean, variance = _compute_moments_with_paddings(
                    x=x, paddings=paddings, reduction_axis=list(reduction_axis), keepdims=False
                )
            self.add_state_update(
                "moving_mean",
                cfg.decay * self.parameters["moving_mean"] + (1 - cfg.decay) * mean,
            )
            self.add_state_update(
                "moving_variance",
                cfg.decay * self.parameters["moving_variance"] + (1 - cfg.decay) * variance,
            )
        else:
            mean = self.parameters["moving_mean"]
            variance = self.parameters["moving_variance"]
        x = (x - mean) * jax.lax.rsqrt(variance + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters["scale"] + self.parameters["bias"]
        return x


class Linear(DenseGeneralBaseLayer):
    """The linear layer."""

    @config_class
    class Config(DenseGeneralBaseLayer.Config):
        """Configures Linear."""

        # Input feature dim.
        input_dim: Required[int] = REQUIRED
        # Output feature dim.
        output_dim: Required[int] = REQUIRED
        # Whether to add a bias.
        bias: bool = True
        # If not None, how to partition output values.
        output_partition_spec: Optional[tuple[Optional[str]]] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.output_dim),
                # A mapping from parameter axes to logical mesh axes (or None if replicating along
                # an axis).
                mesh_axes=cfg.param_partition_spec,
                # Used by optimizers that maintain factored stats, e.g., Adafactor.
                factorization=FactorizationSpec(axes=("row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim],
                # Follow the partitioning of the output dimension of the weight matrix.
                mesh_axes=(cfg.param_partition_spec[-1],),
            )
        return params

    def _maybe_shard(self, output: Tensor) -> Tensor:
        cfg = self.config
        if cfg.output_partition_spec is None:
            return output
        assert len(output.shape) == len(cfg.output_partition_spec)
        return with_sharding_constraint(output, PartitionSpec(*cfg.output_partition_spec))

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        output = self.einsum_maybe_quantized(
            "...d,dh->...h", activation=x, kernel=self.parameters["weight"]
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return self._maybe_shard(output)


class UnitNormLinear(Linear):
    """The linear layer with unit-norm weights."""

    def forward(self, x: Tensor) -> Tensor:
        params_with_normalized_weight = {
            k: (l2_normalize(v, axis=0) if k == "weight" else v) for k, v in self.state.items()
        }
        with child_context("normalized", module=self, state=params_with_normalized_weight):
            return super().forward(x)


class MaxPool2D(BaseLayer):
    """A wrapper for the 2D max pooling layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures MaxPool2D."""

        window: tuple[int, int] = (2, 2)
        strides: tuple[int, int] = (1, 1)

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, None)
        return cfg

    def forward(self, x: Tensor):
        cfg = self.config
        dims = (1,) + cfg.window + (1,)
        strides = (1,) + cfg.strides + (1,)
        # Ref: https://flax.readthedocs.io/en/latest/_modules/flax/linen/pooling.html.
        # Ref: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html.
        # Ref: https://www.tensorflow.org/xla/operation_semantics#reducewindow.
        output = jax.lax.reduce_window(
            operand=x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=dims,
            window_strides=strides,
            # Valid means uses no padding and "stops" the window once it no longer fits.
            padding="VALID",
        )
        return output

    @nowrap
    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 4:
            raise ValueError(f"We expect len(input_shape) = 4, but got {len(input_shape)}.")
        input_height, input_width = input_shape[1:3]

        if input_height is not None:
            output_height = max(input_height - cfg.window[0], 0) // cfg.strides[0] + 1
        else:
            output_height = None
        if input_width is not None:
            output_width = max(input_width - cfg.window[1], 0) // cfg.strides[1] + 1
        else:
            output_width = None
        return [input_shape[0], output_height, output_width, input_shape[3]]


class Embedding(BaseLayer):
    """Implements an embedding lookup function.

    Batched map for int in [0, <num_embeddings>) -> <dim> float vector.
    """

    class Scale(enum.Enum):
        """Defines the scale method on embedding activations.

        Available types:
        1. **UNIT**: Scale the activation components to ~1.

        The activation component should roughly have a magnitude of 1. Since the embedding tensor is
        initialized with a scale of `1/√dim`, the activation is multiplied by `√dim` to
        maintain the desired scale. e.g. Gemma [1]
        [1]
        https://github.com/google-deepmind/gemma/blob/0d6ae857591248422127ca14c027909546362e6a/gemma/modules.py#L80
        """

        UNIT = "unit"

    @config_class
    class Config(BaseLayer.Config):
        """Configures Embedding."""

        num_embeddings: Required[int] = REQUIRED  # Maximum number of embeddings in table.
        dim: Required[int] = REQUIRED  # Embedding vector dimensionality.
        # If not None, how to partition input activation values.
        input_partition_spec: Optional[tuple[Optional[str]]] = None
        # If not None, how to partition embedding table.
        embedding_partition_spec: Optional[tuple[Optional[str]]] = None
        # If not None, how to partition output activation values.
        output_partition_spec: Optional[tuple[Optional[str]]] = None
        # Optional scaling of the embedding activations.
        scale: Optional["Embedding.Scale"] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, "model")
        # By default, initialize to Gaussian with std=1/sqrt(dim), e.g., 0.036 when dim=768.
        #
        # This is the same as:
        # https://github.com/google-research/t5x/blob/f7978d63448c43bdb339ae73fa557ba472be30d6/t5x/examples/scalable_t5/layers.py#L535
        #
        # PyTorch uses normal with std=1.0, regardless of dim/size:
        # https://github.com/pytorch/pytorch/blob/febff45900e57d3e05ee72c1ecfe7d4fcbc582d9/torch/nn/modules/sparse.py#L149
        #
        # TensorFlow/Haiku use truncated normal with std=1.0
        # https://github.com/deepmind/dm-haiku/blob/220c6b02a22f1ee9bea7dc8e017f3090108f75e4/haiku/_src/embed.py#L117
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan="fan_out", distribution="normal"
                )
            }
        )
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=[cfg.num_embeddings, cfg.dim],
                mesh_axes=cfg.param_partition_spec,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x = maybe_shard(x, cfg.input_partition_spec)
        emb = self.parameters["weight"]
        emb = maybe_shard(emb, cfg.embedding_partition_spec)
        activation = emb[x]
        activation = self._scale(activation)
        activation = maybe_shard(activation, cfg.output_partition_spec)
        return activation

    def _scale(self, x: Tensor) -> Tensor:
        """Scale the activation if needed."""
        cfg = self.config
        if cfg.scale is None:
            return x

        # Unsloth [1] discovered that `sqrt(dim)` needs to be computed in float32.
        # [1] Sec 3 in https://unsloth.ai/blog/gemma-bugs.html
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        if cfg.scale == self.Scale.UNIT:
            x = x * math.sqrt(x.shape[-1])
        else:
            raise ValueError(f"Unknown scale {cfg.scale}.")
        x = x.astype(x_dtype)
        return x

    def attend(self, x: Tensor) -> Tensor:
        """Apply query array 'x' to the embedding weight array.

        Args:
            x: array where last dimension equals 'dim'.

        Returns:
            Result of batched inner product of 'x' and embedding weight.
        """
        return jnp.einsum("bld,nd->bln", x, self.parameters["weight"])

    def embeddings(self) -> Tensor:
        """Returns weights of shape [num_embeddings, dim]."""
        return self.parameters["weight"]


class BaseClassificationMetric(BaseLayer):
    """Base classification metrics layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseClassificationMetric."""

        num_classes: Required[int] = REQUIRED

    def forward(
        self, logits: Tensor, *, labels: Tensor, soft_labels: Optional[Tensor] = None
    ) -> Tensor:
        """Computes classification metrics, e.g. loss, accuracy.

        Args:
            logits: A float Tensor of shape [..., num_classes].
            labels: An int Tensor of shape [...].
                Targets should contain the ground truth token ids in the range [0, num_classes).
                Out-of-class targets are ignored in the loss calculation.
            soft_labels: Soft labels generated from data augmentation. If not None, it is already
                in one-hot form and has been smoothed.

        Returns:
            A float Tensor represents the loss.
        """
        raise NotImplementedError("Not implemented forward function for BaseClassificationMetric")


# TODO(llcao@, xianzhi_du@): rename the class to CrossEntropyMetric
class ClassificationMetric(BaseClassificationMetric):
    """Classification metrics layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ClassificationMetric."""

        num_classes: Required[int] = REQUIRED
        label_smoothing: float = 0.0

    def forward(
        self, logits: Tensor, *, labels: Tensor, soft_labels: Optional[Tensor] = None
    ) -> Tensor:
        """Computes classification metrics, e.g. loss, accuracy.

        Args:
            logits: A float Tensor of shape [..., num_classes].
            labels: An int Tensor of shape [...].
                Targets should contain the ground truth token ids in the range [0, num_classes).
                Out-of-class targets are ignored in the loss calculation.
            soft_labels: Soft labels generated from data augmentation. If not None, it is already
                in one-hot form and has been smoothed.

        Returns:
            A float Tensor represents the loss.
        """
        cfg = self.config
        live_targets = jnp.logical_and(0 <= labels, labels < cfg.num_classes)
        num_examples = live_targets.sum()

        loss, all_losses = cross_entropy(
            logits,
            target_labels=labels,
            live_targets=live_targets,
            label_smoothing=cfg.label_smoothing,
            soft_target_labels=soft_labels,
        )
        # [batch].
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.equal(predictions, labels).sum() / jnp.maximum(1, num_examples)

        self.add_summary("loss", WeightedScalar(loss, num_examples))
        self.add_summary("z_loss", WeightedScalar(all_losses["z_loss"], num_examples))
        self.add_summary(
            "cross_entropy_loss", WeightedScalar(all_losses["cross_entropy_loss"], num_examples)
        )
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_examples))
        self.add_summary("accuracy", WeightedScalar(accuracy, num_examples))

        return loss


class BinaryClassificationMetric(BaseClassificationMetric):
    """Binary classification metrics layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures BinaryClassificationMetric."""

        num_classes: int = 2
        prediction_threshold: float = 0.5

    def forward(
        self,
        logits: Tensor,
        *,
        labels: Tensor,
        soft_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes binary classification metrics, e.g. loss, accuracy.

        Args:
            logits: A float Tensor of shape [batch_size, d0, ..., dN].
            labels: A 0/1 int Tensor of shape [batch_size, d0, ..., dN].
                Out-of-range labels indicate padding examples.
            soft_labels: Soft labels are not used for BinaryClassificationMetric
                and therefore must be None.

        Returns:
            A float Tensor represents the loss.

        Raises:
            ValueError: If soft_labels is not None.
            ValueError: If num_classes != 2.
        """
        if soft_labels is not None:
            raise ValueError(
                f"soft_labels for binary cross entropy must be None, found {soft_labels}"
            )
        cfg = self.config
        if cfg.num_classes != 2:
            raise ValueError(
                f"Binary classification is only defined for two classes; "
                f"{cfg.num_classes} were provided"
            )
        live_targets = jnp.logical_and(0 <= labels, labels < cfg.num_classes)
        num_examples = live_targets.sum()
        loss, all_losses = binary_cross_entropy(
            logits,
            target_labels=labels,
            live_targets=live_targets,
        )
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), cfg.prediction_threshold), 1, 0)
        scores = precision_recall_f_score(
            y_true=(live_targets * labels).reshape(-1),
            y_pred=(live_targets * preds).reshape(-1),
        )
        self.add_summary("precision", scores["precision"])
        self.add_summary("recall", scores["recall"])
        self.add_summary("f_score", WeightedScalar(scores["f_score"], num_examples))
        self.add_summary("loss", WeightedScalar(loss, num_examples))
        self.add_summary(
            "binary_cross_entropy_loss",
            WeightedScalar(all_losses["binary_cross_entropy_loss"], num_examples),
        )

        return loss


class CategoricalHingeLossMetric(BaseClassificationMetric):
    """Creates a categorical hinge loss metric (https://en.wikipedia.org/wiki/Hinge_loss).

    Reference:
    https://github.com/keras-team/keras/blob/v2.11.0/keras/losses.py#L1833-L1865
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures CategoricalHingeLossMetric."""

        num_classes: Required[int] = REQUIRED

    def forward(
        self,
        logits: Tensor,
        *,
        labels: Tensor,
        soft_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes hinge loss metrics, e.g. loss, accuracy.

        Args:
            logits: A float Tensor of shape [batch_size, num_classes].
            labels: An int Tensor of shape [batch_size].
                Targets should contain the ground truth token ids in the range [0, num_classes).
            soft_labels: Soft labels are not used for CategoryHingelossMetric
                and therefore must be None.

        Returns:
            A float Tensor represents the loss.

        Raises:
            ValueError: If soft_labels is not None.
        """
        if soft_labels is not None:
            raise ValueError(
                f"soft_labels for category hinge loss must be None, found {soft_labels}"
            )

        cfg = self.config

        one_hot_labels = jax.nn.one_hot(labels, cfg.num_classes)

        per_target_loss = categorical_hinge_loss(logits, one_hot_labels)

        live_targets = jnp.logical_and(0 <= labels, labels < cfg.num_classes)
        live_targets = live_targets.astype(per_target_loss.dtype)
        num_live_targets = live_targets.sum()
        denominator = jnp.maximum(1, num_live_targets)
        loss = (per_target_loss * live_targets).sum() / denominator

        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.equal(predictions, labels).sum() / denominator

        self.add_summary("loss", WeightedScalar(loss, num_live_targets))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_live_targets))
        self.add_summary("accuracy", WeightedScalar(accuracy, num_live_targets))

        return loss


class BaseClassificationHead(BaseLayer):
    """Base classification head layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseClassificationHead."""

        input_dim: Required[int] = REQUIRED
        num_classes: Required[int] = REQUIRED
        metric: InstantiableConfig = ClassificationMetric.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("metric", cfg.metric.set(num_classes=cfg.num_classes))

    def forward(self, input_batch: NestedTensor) -> Tensor:
        """Produces predictions for the given inputs.

        Args:
            input_batch: A dict with the following entries:
                hidden_states: A Tensor of shape [..., input_dim] representing hidden states.
                **input_batch: Inputs from the calling layer.

        Returns:
            Logits of shape [..., num_classes].
        """
        raise NotImplementedError(type(self))

    def loss(
        self, *, logits: Tensor, target_labels: Tensor, soft_labels: Optional[Tensor] = None
    ) -> Tensor:
        """Computes classification loss.

        Args:
            logits: A float Tensor of shape [..., num_classes].
            target_labels: An int Tensor of shape [...].
                Targets should contain the ground truth token ids in the range [0, num_classes).
                Out-of-class targets are ignored in the loss calculation.
            soft_labels: Optional labels that are already smoothed/in one-hot form. If provided,
                target_labels will only be used for inferring the mask during loss calculation.

        Returns:
            A scalar loss value.
        """
        return self.metric(logits, labels=target_labels, soft_labels=soft_labels)


class StochasticDepth(BaseLayer):
    """Creates a stochastic depth layer.

    Reference:
    https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures StochasticDepth."""

        rate: Optional[float] = None  # Drop rate of this layer.
        mode: str = "row"  # One mode of ['batch', 'row'].

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if not self.is_training or cfg.rate is None or cfg.rate == 0:
            return x
        if cfg.rate < 0.0 or cfg.rate >= 1.0:
            raise ValueError(f"Drop rate needs to be in [0, 1), but got {cfg.rate}.")
        if cfg.mode not in ["batch", "row"]:
            raise ValueError(f"Mode has to be either 'batch' or 'row', but got {cfg.mode}.")
        keep_prob = 1.0 - cfg.rate
        random_tensor = keep_prob
        if cfg.mode == "row":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
        else:
            shape = [1] * x.ndim
        random_tensor += jax.random.uniform(self.prng_key, shape, dtype=x.dtype)
        binary_tensor = jnp.floor(random_tensor)
        return x * binary_tensor / keep_prob


def get_stochastic_depth_linear_rate(peak_rate: float, stage_order: int, num_stages: int):
    """Get stochastic depth rate for the ith stage.

    Reference:
    Equation (4) in paper: https://arxiv.org/pdf/1603.09382.pdf

    Args:
        peak_rate: The peak drop rate.
        stage_order: Order of the current stage.
        num_stages: Total number of stages.

    Returns:
        Drop rate of the ith stage.

    Raises:
        ValueError: If peak_rate or stage_order are out of the valid ranges.
    """
    if peak_rate is not None:
        if peak_rate < 0 or peak_rate >= 1:
            raise ValueError(f"Peak drop rate must be in [0, 1), but got {peak_rate}.")
        if stage_order < 0 or stage_order > num_stages:
            raise ValueError(
                f"Stage order has to be within [0, {num_stages}], but got {stage_order}."
            )
        rate = peak_rate * float(stage_order) / num_stages
    else:
        rate = None
    return rate


def set_bias_recursively(cfg: ConfigBase, bias: bool = False):
    """Sets config.bias to `bias` for all relevant descendant configs in `cfg`."""

    def visit_fn(_, value):
        if isinstance(value, BaseLayer.Config) and "bias" in value:
            value.bias = bias

    def enter_fn(_, value, default_kv):
        return None if isinstance(value, BaseLayer.Config) and "bias" in value else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
    return cfg


def set_norm_recursively(cfg: ConfigBase, norm_cfg: ConfigBase):
    """Sets normalization layer configs to `norm_cfg` for all relevant descendants in `cfg`."""

    def enter_fn(_, value, default_kv):
        if isinstance(value, ConfigBase):
            for subkey, subval in value.items():
                if isinstance(subval, BaseNormalizationLayer.Config):
                    value.set(**{subkey: norm_cfg})
        return default_kv

    cfg.visit(visit_fn=lambda k, v: None, enter_fn=enter_fn)
    return cfg


class SqueezeExcitation(BaseLayer):
    """A squeeze and excitation layer.

    Reference: https://arxiv.org/abs/1709.01507.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures SqueezeExcitation."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        # The squeeze-and-excitation ratio. The input is returned
        # if set to 0.
        se_ratio: float = 0.0
        activation: str = "nn.relu"  # The activation function.
        gating: str = "nn.sigmoid"  # The gating function.
        # The number of reduced filters; this overrides the
        # default setting of multiplying input_dim by se_ratio.
        num_reduced_filters: Optional[int] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, "model")
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.se_ratio > 0:
            if cfg.num_reduced_filters:
                num_reduced_filters = cfg.num_reduced_filters
            else:
                num_reduced_filters = max(1, int(cfg.input_dim * cfg.se_ratio))
            self._add_child(
                "reduce",
                Conv2D.default_config().set(
                    input_dim=cfg.input_dim,
                    output_dim=num_reduced_filters,
                    param_partition_spec=cfg.param_partition_spec,
                ),
            )
            self._add_child(
                "expand",
                Conv2D.default_config().set(
                    input_dim=num_reduced_filters,
                    output_dim=cfg.input_dim,
                    param_partition_spec=cfg.param_partition_spec,
                ),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        if cfg.se_ratio == 0:
            return inputs
        x = jnp.mean(inputs, axis=(1, 2), keepdims=True)
        x = self.reduce(x)
        x = get_activation_fn(cfg.activation)(x)
        x = self.expand(x)
        return get_activation_fn(cfg.gating)(x) * inputs


class MultiLinear(BaseLayer):
    """A linear layer with multiple outputs."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures MultiLinear."""

        input_dim: Required[int] = REQUIRED  # Feature dim.
        num_outputs: Required[int] = REQUIRED  # Number of outputs.
        output_dim: Required[int] = REQUIRED  # Dimension per output.
        bias: bool = True  # Whether the linear modules have biases.

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        # Shard the 'output_dim' axis by the 'model' dim of the mesh.
        cfg.param_partition_spec = (None, None, "model")
        return cfg

    # pylint: disable-next=duplicate-code
    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.num_outputs, cfg.output_dim),
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=("row", None, "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=(cfg.num_outputs, cfg.output_dim),
                mesh_axes=(cfg.param_partition_spec[0], cfg.param_partition_spec[2]),
            )
        return params

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "weight":
            if len(parameter_spec.shape) != 3:
                raise ValueError(f"Unexpected parameter spec {parameter_spec}")
            return FanAxes(in_axis=0, out_axis=(1, 2))
        else:
            return None

    def forward(self, inputs: Tensor) -> Tensor:
        params = self.parameters
        dims = "abcdefghjklm"  # should be enough dims.
        if inputs.ndim >= len(dims):
            raise NotImplementedError(inputs.shape)
        batch_dims = dims[: inputs.ndim - 1]
        outputs = jnp.einsum(f"{batch_dims}i,ino->{batch_dims}no", inputs, params["weight"])
        if "bias" in params:
            outputs += params["bias"]
        return outputs


class VariationalNoise(ParameterNoise):
    """Variational noise layer."""

    @config_class
    class Config(ParameterNoise.Config):
        """Configures VariationalNoise."""

        vn_std: Required[float] = REQUIRED

    def apply(self, prng_key: Tensor, params: NestedTensor) -> NestedTensor:
        cfg = self.config
        if cfg.vn_std <= 0:
            return params
        return jax.tree.map(
            lambda x: x + jax.random.normal(prng_key, x.shape, dtype=x.dtype) * cfg.vn_std, params
        )


class SeparableSpaceTimePositionalEmbedding(BaseLayer):
    """Positional embedding described in https://arxiv.org/abs/2205.09113.

    Creates separate spatial and temporal positional embeddings and adds them together.

    NOTE: this assumes input sequences have layout HWT, where HW represents
    the height and width (spatial) dimensions, and T represents the temporal dimension.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures SeparableSpaceTimePositionalEmbedding."""

        dim: Required[int] = REQUIRED
        # Note: if using a class other than `Embedding`, note that it must support the following:
        # - Implements `embeddings()` method that returns a tensor with shape [num_embeddings, dim].
        # - Has int field `dim` in its config that defines embedding dimension.
        spatial_embeddings: InstantiableConfig = Embedding.default_config()
        temporal_embeddings: InstantiableConfig = Embedding.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "spatial",
            cfg.spatial_embeddings.set(dim=cfg.dim),
        )
        self._add_child(
            "temporal",
            cfg.temporal_embeddings.set(dim=cfg.dim),
        )

    @property
    def num_spatial_embeddings(self) -> int:
        return self.spatial.embeddings().shape[0]

    @property
    def num_temporal_embeddings(self) -> int:
        return self.temporal.embeddings().shape[0]

    @property
    def max_seq_len(self) -> int:
        return self.num_spatial_embeddings * self.num_temporal_embeddings

    def embeddings(self) -> Tensor:
        """
        Returns:
            Combined space-time embeddings with shape [self.max_seq_len, self.config.dim].
        """
        cfg = self.config
        spatial_embeddings = self.spatial.embeddings()
        temporal_embeddings = self.temporal.embeddings()
        combined_embeddings = (
            spatial_embeddings[:, jnp.newaxis, :] + temporal_embeddings[jnp.newaxis, :, :]
        )
        combined_embeddings = combined_embeddings.reshape(-1, cfg.dim)
        return combined_embeddings

    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: An int/long tensor with arbitrary shape [...] whose values should
                be in [0, self.max_seq_len).

        Returns:
            Embedded positions with shape [..., dim].
        """
        emb = self.embeddings()
        return emb[positions]


class MovingAverage(BaseLayer):
    """A layer to maintain an exponential moving average stats.

    Given each value `x`, updates `moving_average` as:
        moving_average = x * weight + moving_average * (1 - weight)
        count = count + 1

    where `weight` is determined by:
        weight = max(cfg.min_weight, 1 / (count + 1))
    """

    @config_class
    class Config(BaseLayer.Config):
        shape: Sequence[int] = tuple()
        # The minimum weight for an update.
        min_weight: Required[float] = REQUIRED

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "count": ParameterSpec(
                shape=[],
                dtype=jnp.int32,
                mesh_axes=(None,),
                initializer=constant_initializer(0),
                weight_decay_scale=0,
            ),
            "value": ParameterSpec(
                shape=cfg.shape,
                dtype=jnp.float32,
                mesh_axes=(None,),
                initializer=constant_initializer(0.0),
                weight_decay_scale=0,
            ),
        }

    def forward(self, x: Tensor) -> Tensor:
        """Computes a moving average of `x`.

        The moving average updates will be set in OutputCollection.state_updates.

        Args:
            x: A Tensor of shape cfg.shape.

        Returns:
            Returns the current moving average.
        """
        cfg = self.config
        weight = jnp.maximum(cfg.min_weight, 1.0 / (1 + self.parameters["count"]))
        new_moving_average = (1 - weight) * self.parameters["value"] + weight * x
        self.add_state_update("value", new_moving_average)
        self.add_state_update("count", 1 + self.parameters["count"])
        return new_moving_average

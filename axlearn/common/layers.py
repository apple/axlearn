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

# pylint: disable=too-many-lines
"""Basic layers."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
from absl import logging
from jax import nn
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common.base_layer import BaseLayer, FactorizationSpec, ParameterNoise, ParameterSpec
from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    InstantiableConfig,
    Required,
    UnknownFieldError,
    config_class,
)
from axlearn.common.loss import binary_cross_entropy, categorical_hinge_loss, cross_entropy
from axlearn.common.metrics import WeightedScalar
from axlearn.common.metrics_classification import precision_recall_f_score
from axlearn.common.module import Module, child_context
from axlearn.common.normalize import l2_normalize
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    DefaultInitializer,
    FanAxes,
    WeightInitializer,
    constant_initializer,
)
from axlearn.common.utils import (
    NestedTensor,
    Tensor,
    partial_with_fn_metadata,
    with_sharding_constraint,
)


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
        method_map: Dict[str, str] = {}

        def set(self, **kwargs) -> "RedirectToSharedModule.Config":
            try:
                super().set(**kwargs)
            except UnknownFieldError as e:
                logging.info("Ignoring %s", e)
                # We intentionally ignore this exception.
            return self

    def _methods_to_wrap_for_auto_child_context(self) -> Dict[str, Callable]:
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
        cfg = self.config  # type: RedirectToSharedModule.Config
        shared_module = self.get_shared_module(cfg.shared_module)
        with child_context("redirect", module=shared_module.module, state=shared_module.state):
            return getattr(shared_module.module, redirection_target_method)(*args, **kwargs)


class Dropout(BaseLayer):
    """The dropout layer."""

    @config_class
    class Config(BaseLayer.Config):
        rate: Optional[float] = None  # The dropout rate (i.e., 1 - keep_prob).

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if not self.is_training or cfg.rate is None or cfg.rate == 0:
            return x
        assert 0 < cfg.rate < 1
        samples = jax.random.uniform(
            self.prng_key, shape=x.shape, dtype=x.dtype, minval=0.0, maxval=1.0
        )
        dropout = jnp.floor(1 - cfg.rate + samples)
        return x * dropout / (1.0 - cfg.rate)


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
        sampled_id = jax.random.shuffle(self.prng_key, patch_id, axis=1)[:, :num_chosen_tokens]
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

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
            "bias": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
        }

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        x = super().forward(x, paddings=paddings)
        x = x * self.parameters["scale"] + self.parameters["bias"]
        return x


class RMSNorm(BaseNormalizationLayer):
    """Reference: https://github.com/bzhangGo/rmsnorm."""

    @config_class
    class Config(BaseNormalizationLayer.Config):
        # The epsilon.
        eps: float = 1e-8
        # Cast input to this dtype for the 'forward' call. If None, do not cast.
        forward_dtype: Optional[jnp.dtype] = jnp.float32

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
        }

    def forward(self, x: Tensor, *, paddings: Optional[Tensor] = None) -> Tensor:
        del paddings  # paddings do not affect LayerNorm results
        cfg = self.config
        x_dtype = x.dtype
        if cfg.forward_dtype is not None:
            x = x.astype(cfg.forward_dtype)
        moment2 = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(moment2 + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters["scale"]
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
    reduction_axis: List[int],
    keepdims: bool = False,
) -> Tuple[Tensor, Tensor]:
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


class GroupNorm(BaseNormalizationLayer):
    """https://arxiv.org/abs/1803.08494."""

    @config_class
    class Config(BaseNormalizationLayer.Config):
        """Configures GroupNorm."""

        # The number of groups.
        num_groups: Required[int] = REQUIRED
        # The epsilon.
        eps: float = 1e-8
        # Cast input to this dtype for the 'forward' call. If None, do not cast.
        forward_dtype: Optional[jnp.dtype] = jnp.float32

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
            "bias": ParameterSpec(shape=[cfg.input_dim], mesh_axes=(None,)),
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
        # Reduce along spatial dims and group_size, but not along batch or num_groups.
        reduction_axis = list(range(1, y.ndim - 2)) + [-1]
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

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
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


class Linear(BaseLayer):
    """The linear layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures Linear."""

        # Input feature dim.
        input_dim: Required[int] = REQUIRED
        # Output feature dim.
        output_dim: Required[int] = REQUIRED
        # Whether to add a bias.
        bias: bool = True
        # If not None, how to partition output values.
        output_partition_spec: Optional[Tuple[Optional[str]]] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.output_dim),
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=("row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
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
        output = x @ self.parameters["weight"]
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


def _check_conv_cfg(padding: Union[str, Sequence[Tuple[int, int]]], strides: Sequence[int]):
    if isinstance(padding, str):
        if padding in ("SAME", "VALID"):
            if padding == "SAME" and any(s > 1 for s in strides):
                raise NotImplementedError("SAME padding does not support strides > 1")
        else:
            raise NotImplementedError(f"{padding} padding is not supported.")
    else:
        padding_flattened = (p for p_tuple in padding for p in p_tuple)
        if any(p < 0 for p in padding_flattened):
            raise NotImplementedError("Negative padding is not supported")


class MaxPool2D(BaseLayer):
    """A wrapper for the 2D max pooling layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures MaxPool2D."""

        window: Tuple[int, int] = (2, 2)
        strides: Tuple[int, int] = (1, 1)

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


# The accuracy of the output of this layer currently doesn't match that of PyTorch
# quite as closely as we would like. See layers_test.py:test_conv2d().
class Conv2D(BaseLayer):
    """The 2-D convolution layer.

    Kernel weights have the HWIO layout and in the shape of (window[0], window[1], input_dim,
    output_dim). Both inputs and outputs will be in the NHWC layout.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures Conv2D."""

        window: Tuple[int, int] = (1, 1)  # The convolution window.
        strides: Tuple[int, int] = (1, 1)  # The convolution strides.
        # Paddings: "SAME", "VALID", or ((top, bottom), (left, right)).
        padding: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]] = ((0, 0), (0, 0))
        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.
        # The number of groups in which the input is split along the channel axis.
        # input_dim and output_dim must both be divisible by num_input_dim_groups. For example,
        # - At num_input_dim_groups=1, all inputs are convolved to all outputs (the default).
        # - At num_input_dim_groups=2, the operation is equivalent to concatenating two conv layers
        #   side by side, each seeing half the input and producing half the output channels.
        # - At num_input_dim_groups=input_dim, each input channel is convolved with its own
        #   set of filters (of size output_dim / input_dim); if further output_dim == K * input_dim,
        #   where K is a positive integer, the operation is also known as a "depthwise convolution".
        num_input_dim_groups: Optional[int] = 1

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(cfg.padding, cfg.strides)
        params = dict(
            weight=ParameterSpec(
                shape=list(cfg.window)
                + [cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim],
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=cfg.strides,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            padding=cfg.padding,
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 4:
            raise ValueError(f"We expect len(input_shape) = 4, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                f"cfg.input_dim = {cfg.input_dim}."
            )
        input_height, input_width = input_shape[1:3]
        if cfg.padding == "SAME":
            if cfg.padding == "SAME" and any(s > 1 for s in cfg.strides):
                raise NotImplementedError("SAME padding does not support strides > 1")
            pad_height = cfg.window[0] - 1
            pad_width = cfg.window[1] - 1

        elif cfg.padding == "VALID":
            pad_height = pad_width = 0
        else:
            pad_height = cfg.padding[0][0] + cfg.padding[0][1]
            pad_width = cfg.padding[1][0] + cfg.padding[1][1]
        if input_height is not None:
            output_height = max(input_height + pad_height - cfg.window[0], 0) // cfg.strides[0] + 1
        else:
            output_height = None
        if input_width is not None:
            output_width = max(input_width + pad_width - cfg.window[1], 0) // cfg.strides[1] + 1
        else:
            output_width = None
        return [input_shape[0], output_height, output_width, cfg.output_dim]


def _compute_conv_output_1d_padding(
    in_paddings: Tensor, *, window: int, stride: int, conv_padding_cfg: Union[str, Tuple[int, int]]
):
    """Helper function to compute 1D paddings for 2D convolution.

    Args:
        in_paddings: A Tensor of shape [batch_size, seq_len].
        window: convolution window size of the time axis.
        stride: convolution stride size of the time axis.
        conv_padding_cfg: convolution padding along the time axis.
            Either the string "SAME", the string "VALID", or an
            integer pair (left, right) that gives the padding to
            apply before and after the time dimension. Front paddings
            are treated as valid frames and back paddings as invalid frames.

    Returns:
        out_paddings: A Tensor of shape [batch_size, seq_len].

    Raises:
        NotImplementedError: If conv_padding_cfg is SAME and strides is > 1.
    """
    if conv_padding_cfg == "SAME":
        if stride == 1:
            return in_paddings
        raise NotImplementedError("SAME padding does not support strides > 1")

    if isinstance(conv_padding_cfg, tuple):
        # Front paddings are valid frames.
        in_paddings = jnp.pad(in_paddings, ((0, 0), (conv_padding_cfg[0], 0)), constant_values=0)
        # Back paddings are invalid frames.
        in_paddings = jnp.pad(in_paddings, ((0, 0), (0, conv_padding_cfg[1])), constant_values=1)

    # Apply max pooling with "VALID" padding along the time axis.
    out_paddings = jax.lax.reduce_window(
        in_paddings,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, window),
        window_strides=(1, stride),
        padding="VALID",
    )
    return out_paddings


class Conv2DTranspose(BaseLayer):
    """The 2-D transposed convolution layer.

    Kernel weights have the HWIO layout and in the shape of (window[0], window[1], output_dim,
    input_dim). Both inputs and outputs will be in the NHWC layout.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures Conv2DTranspose."""

        window: Tuple[int, int] = (1, 1)
        strides: Tuple[int, int] = (1, 1)
        padding: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]] = ((0, 0), (0, 0))
        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(cfg.padding, cfg.strides)
        params = dict(
            weight=ParameterSpec(
                shape=list(cfg.window) + [cfg.output_dim, cfg.input_dim],
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_transpose(
            lhs=x,
            rhs=self.parameters["weight"],
            strides=cfg.strides,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            padding=cfg.padding,
            # if True flips spatial axes and swaps the input/output channel axes of the kernel.
            # This makes the output of this function identical to the gradient-derived functions
            # like keras.layers.Conv2DTranspose applied to the same kernel.
            transpose_kernel=True,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 4:
            raise ValueError(f"We expect len(input_shape) = 4, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                "cfg.input_dim = {cfg.input_dim}."
            )
        input_height, input_width = input_shape[1:3]
        output_height, output_width = None, None

        if cfg.padding == "SAME":
            if cfg.padding == "SAME" and any(s > 1 for s in cfg.strides):
                raise NotImplementedError("SAME padding does not support strides > 1")
            if input_height is not None:
                output_height = input_height * cfg.strides[0]
            if input_width is not None:
                output_width = input_width * cfg.strides[0]
        elif cfg.padding == "VALID":
            if input_height is not None:
                output_height = input_height * cfg.strides[0] + max(
                    cfg.window[0] - cfg.strides[0], 0
                )
            if input_width is not None:
                output_width = input_width * cfg.strides[1] + max(cfg.window[1] - cfg.strides[1], 0)

        return [input_shape[0], output_height, output_width, cfg.output_dim]


class Conv2DWith1DPadding(Conv2D):
    """The 2-D convolution with 1-D padding on the time axis.

    Kernel weights have the HWIO layout and in the shape of (window[0], window[1], input_dim,
    output_dim). Both inputs and outputs will be in the NHWC layout.

    For audio inputs/outputs, we assume dims correspond to [batch_size, time, frequency, input_dim].
    This layer also returns paddings along the time axis. If specifying `cfg.padding` as a tuple of
    (leading, trailing) paddings, leading padding frames are treated as valid (i.e. not masked by
    the output paddings) while trailing padding frames are invalid (i.e. masked by the output
    paddings).
    """

    Config = Conv2D.Config

    # We add a kwargs "paddings" to the forward method.
    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor, *, paddings: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes convolution outputs and paddings.

        Args:
            x: A Tensor of shape [batch_size, seq_len, frequency, input_dim].
            paddings: 0/1 Tensor of shape [batch_size, seq_len].

        Returns:
            output: A Tensor of shape [batch_size, seq_len, frequency, output_dim].
            paddings: 0/1 Tensor of shape [batch_size, seq_len].
        """
        cfg = self.config
        # Apply padding to the input.
        assert len(x.shape) == len(paddings.shape) + 2
        x = x * (1 - paddings[..., None, None])

        # Apply Conv2D.
        output = super().forward(x)
        # Compute paddings conv output.
        output_paddings = _compute_conv_output_1d_padding(
            paddings,
            window=cfg.window[0],
            stride=cfg.strides[0],
            conv_padding_cfg=cfg.padding if isinstance(cfg.padding, str) else cfg.padding[0],
        )
        # Apply padding to the outputs.
        output = output * (1 - output_paddings[..., None, None])
        return output, output_paddings


class Conv3D(BaseLayer):
    """The 3-D convolution layer.

    Kernel weights have the HWDIO layout and in the shape of (window[0], window[1],
    window[2], input_dim, output_dim). Both inputs and outputs will be in the NHWDC layout.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures Conv3D."""

        window: Tuple[int, int, int] = (1, 1, 1)  # The convolution window.
        strides: Tuple[int, int, int] = (1, 1, 1)  # The convolution strides.

        # Paddings: "SAME" or "VALID, or ((top, bottom), (left, right), (front, back))
        padding: Union[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = (
            (0, 0),
            (0, 0),
            (0, 0),
        )

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Required[int] = REQUIRED  # Output feature dim.

        bias: bool = True  # Whether to add a bias.

        # The number of groups in which the input is split along the channel axis.
        # input_dim and output_dim must both be divisible by num_input_dim_groups. For example,
        # - At num_input_dim_groups=1, all inputs are convolved to all outputs (the default).
        # - At num_input_dim_groups=2, the operation is equivalent to concatenating two conv layers
        #   side by side, each seeing half the input and producing half the output channels.
        # - At num_input_dim_groups=input_dim, each input channel is convolved with its own
        #   set of filters (of size output_dim / input_dim); if further output_dim == K * input_dim,
        #   where K is a positive integer, the operation is also known as a "depthwise convolution".
        num_input_dim_groups: Optional[int] = 1

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(cfg.padding, cfg.strides)
        params = dict(
            weight=ParameterSpec(
                shape=list(cfg.window)
                + [cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim],
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=cfg.strides,
            dimension_numbers=("NHWDC", "HWDIO", "NHWDC"),
            padding=cfg.padding,
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 5:
            raise ValueError(f"We expect len(input_shape) = 5, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                f"cfg.input_dim = {cfg.input_dim}."
            )

        if cfg.padding == "SAME":
            pad_height = cfg.window[0] - 1
            pad_width = cfg.window[1] - 1
            pad_depth = cfg.window[2] - 1
        elif cfg.padding == "VALID":
            pad_height = pad_width = pad_depth = 0
        else:
            pad_height = cfg.padding[0][0] + cfg.padding[0][1]
            pad_width = cfg.padding[1][0] + cfg.padding[1][1]
            pad_depth = cfg.padding[2][0] + cfg.padding[2][1]

        def compute_output_size(i, p, w, s):
            if i is None:
                return None
            return max(i + p - w, 0) // s + 1

        pad_shape = [pad_height, pad_width, pad_depth]
        output_shape = [
            compute_output_size(
                input_shape[idx + 1], pad_shape[idx], cfg.window[idx], cfg.strides[idx]
            )
            for idx in range(3)
        ]

        return [input_shape[0], *output_shape, cfg.output_dim]


class Conv1D(BaseLayer):
    """The 1D convolution layer.

    Kernel weights have the WIO layout and in the shape of (window, input_dim, output_dim).
    Both inputs and outputs will be in the NWC layout.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures Conv1D."""

        window: Required[int] = REQUIRED  # The convolution window.
        strides: int = 1  # The convolution strides.
        # Paddings: "SAME", "VALID", or (left, right).
        # For causal convolution, set padding to (window - 1, 0).
        padding: Union[str, Tuple[int, int]] = (0, 0)
        # Input feature dim, which is also the output dim.
        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.
        # The number of groups in which the input is split along the channel axis.
        # input_dim and output_dim must both be divisible by num_input_dim_groups. For example,
        # - At num_input_dim_groups=1, all inputs are convolved to all outputs (the default).
        # - At num_input_dim_groups=2, the operation is equivalent to concatenating two conv layers
        #   side by side, each seeing half the input and producing half the output channels.
        # - At num_input_dim_groups=input_dim, each input channel is convolved with its own
        #   set of filters (of size output_dim / input_dim); if further output_dim == K * input_dim,
        #   where K is a positive integer, the operation is also known as a "depthwise convolution".
        num_input_dim_groups: Optional[int] = 1

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        if cfg.padding in ("SAME", "VALID"):
            if cfg.padding == "SAME" and cfg.strides > 1:
                raise NotImplementedError("SAME padding does not support strides > 1")
        else:
            left, right = cfg.padding
            if any(p < 0 for p in (left, right)):
                raise NotImplementedError("Negative padding is not supported")
        params = dict(
            weight=ParameterSpec(
                shape=[cfg.window, cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim],
                mesh_axes=cfg.param_partition_spec,
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=(cfg.strides,),
            dimension_numbers=("NWC", "WIO", "NWC"),
            padding=cfg.padding if isinstance(cfg.padding, str) else (cfg.padding,),
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output


class DepthwiseConv1D(BaseLayer):
    """The 1-D depth-wise convolution layer.

    Kernel weights have the WIO layout and in the shape of (window, 1, output_dim=input_dim).
    Both inputs and outputs will be in the NWC layout.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures DepthwiseConv1D."""

        window: Required[int] = REQUIRED  # The convolution window.
        strides: int = 1  # The convolution strides.
        # Paddings: "SAME", "VALID", or (left, right).
        # For causal convolution, set padding to (window - 1, 0).
        padding: Union[str, Tuple[int, int]] = (0, 0)
        # Input feature dim, which is also the output dim.
        input_dim: Required[int] = REQUIRED
        bias: bool = True  # Whether to add a bias.

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        if cfg.padding in ("SAME", "VALID"):
            if cfg.padding == "SAME" and cfg.strides > 1:
                raise NotImplementedError("SAME padding does not support strides > 1")
        else:
            left, right = cfg.padding
            if any(p < 0 for p in (left, right)):
                raise NotImplementedError("Negative padding is not supported")
        params = dict(
            weight=ParameterSpec(
                # https://www.tensorflow.org/xla/operation_semantics#conv_convolution:
                # The input feature dimension of rhs needs to be equal to the lhs input feature
                # dimension divided by feature_group_count (so it already has the size of a group
                # of input features).
                shape=[cfg.window, 1, cfg.input_dim],
                mesh_axes=cfg.param_partition_spec,
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.input_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=(cfg.strides,),
            dimension_numbers=("NWC", "WIO", "NWC"),
            padding=cfg.padding if isinstance(cfg.padding, str) else (cfg.padding,),
            feature_group_count=cfg.input_dim,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output


class Embedding(BaseLayer):
    """Implements an embedding lookup function.

    Batched map for int in [0, <num_embeddings>) -> <dim> float vector.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures Embedding."""

        num_embeddings: Required[int] = REQUIRED  # Maximum number of embeddings in table.
        dim: Required[int] = REQUIRED  # Embedding vector dimensionality.

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

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=[cfg.num_embeddings, cfg.dim],
                mesh_axes=cfg.param_partition_spec,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        emb = self.parameters["weight"]
        return emb[x]

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
        mask = jnp.logical_and(0 <= labels, labels < cfg.num_classes)
        num_examples = mask.sum()

        loss, all_losses = cross_entropy(
            logits,
            labels,
            mask=mask,
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
        mask = jnp.logical_and(0 <= labels, labels < cfg.num_classes)
        num_examples = mask.sum()
        loss, all_losses = binary_cross_entropy(
            logits,
            labels,
            mask=mask,
        )
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), cfg.prediction_threshold), 1, 0)
        scores = precision_recall_f_score(
            y_true=(mask * labels).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
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

        pre_mask_loss = categorical_hinge_loss(logits, one_hot_labels)

        mask = jnp.logical_and(0 <= labels, labels < cfg.num_classes)
        mask = mask.astype(pre_mask_loss.dtype)
        num_unmasked = mask.sum()
        denominator = jnp.maximum(1, num_unmasked)
        loss = (pre_mask_loss * mask).sum() / denominator

        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.equal(predictions, labels).sum() / denominator

        self.add_summary("loss", WeightedScalar(loss, num_unmasked))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_unmasked))
        self.add_summary("accuracy", WeightedScalar(accuracy, num_unmasked))

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


class StackOverTime(BaseLayer):
    """Stack inputs along the time axis.

    StackOverTime behaves the same as Conv2DWith1DPadding w.r.t. paddings along the time axis.
    We treat front paddings as valid frames and back paddings as invalid frames.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures StackOverTime."""

        stride: Required[int] = REQUIRED  # Number of frames to stack.
        # Number of paddings to apply along the time axis. The two integers indicate
        # leading and trailing padding to add respectively.
        padding: Tuple[int, int] = (0, 0)

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Tuple[Tensor, Tensor]:
        """Stacks stride number of frames into one frame along the time axis.

        Args:
            inputs: Tensor of shape [batch, time, input_dim].
            paddings: 0/1 Tensor of shape [batch, time], paddings of the input sequences.

        Returns:
            stacked_inputs: Tensor of shape [batch, time // stride, input_dim * stride].
            stacked_paddings: 0/1 Tensor of shape [batch, time // stride]. An output frame
                is padding if at least one of the stacked input frames is padding.

        Raises:
            ValueError: If stride is <= 1.
        """
        cfg = self.config
        if cfg.stride <= 1:
            raise ValueError(f"stride should be greater than 1, but got {cfg.stride}.")
        inputs = jnp.pad(inputs, ((0, 0), cfg.padding, (0, 0)), constant_values=0)
        # Front paddings are valid frames.
        paddings = jnp.pad(paddings, ((0, 0), (cfg.padding[0], 0)), constant_values=0)
        # Back paddings are invalid frames.
        paddings = jnp.pad(paddings, ((0, 0), (0, cfg.padding[1])), constant_values=1)

        batch_size, seq_len, input_dim = inputs.shape
        output_length = seq_len // cfg.stride
        new_shape = [batch_size, output_length, input_dim * cfg.stride]
        # Stack inputs over the time dimension.
        stacked_inputs = jnp.reshape(inputs[:, : output_length * cfg.stride, :], new_shape)
        # An output frame is padding if at least one of the stacked input frames is padding.
        stacked_paddings = jnp.max(
            jnp.reshape(paddings[:, : output_length * cfg.stride], [-1, output_length, cfg.stride]),
            axis=-1,
        )
        stacked_inputs = stacked_inputs * (1 - stacked_paddings)[:, :, None]
        return stacked_inputs, stacked_paddings

    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        """Computes stacked output shape.

        Args:
            input_shape: The input dimensions are (batch, time, feature_dim).
                If the value of the dimension is not available, use None.

        Returns:
            The output shape. The dimensions are (batch, time, feature_dim).
        """
        cfg = self.config
        batch_size, seq_len, input_dim = input_shape
        output_length = (seq_len + sum(cfg.padding)) // cfg.stride if seq_len is not None else None
        return [batch_size, output_length, input_dim * cfg.stride]


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
    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
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
        return jax.tree_util.tree_map(
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

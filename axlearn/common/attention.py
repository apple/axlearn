# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google-research/t5x:
# Copyright 2022 The T5X Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# huggingface/transformers:
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# facebookresearch/deit:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# ofirpress/attention_with_linear_biases:
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.

"""Attention layers with pjit partition specs.

On `attention_logit_biases`:
* A biases tensor can have shape [batch, target_length, source_length] or
  [batch, num_heads, target_length, source_length].
* Each value represents a bias to be added to the attention logits
  (therefore a -inf represents a disconnected position pair).
* biases=None represents an all-zero tensor, i.e., all position pairs are connected.
"""
# pylint: disable=abstract-method,too-many-lines
import enum
import math
import typing
from enum import Enum, unique
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_policies as jax_remat_policies

from axlearn.common import ops, param_init
from axlearn.common.base_layer import (
    BaseLayer,
    FactorizationSpec,
    NestedParameterSpec,
    ParameterSpec,
    RematSpec,
)
from axlearn.common.config import (
    REQUIRED,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    config_for_partial_function,
)
from axlearn.common.layers import (
    Dropout,
    LayerNorm,
    Linear,
    StochasticDepth,
    get_activation_fn,
    get_stochastic_depth_linear_rate,
)
from axlearn.common.module import Module, child_context
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    ConstantInitializer,
    DefaultInitializer,
    FanAxes,
    WeightInitializer,
    constant_initializer,
)
from axlearn.common.pipeline import Pipeline
from axlearn.common.repeat import Repeat
from axlearn.common.utils import (
    NestedTensor,
    PartitionSpec,
    Tensor,
    VDict,
    check_numerics,
    get_or_none,
    shapes,
    split_prng_key,
)

NEG_INF = -1e15


class ForwardMode(enum.Enum):
    """ForwardMode describes the type of computation to be done in a forward pass through a layer.

    FORWARD: Used for a standard forward pass.
    INIT_STATES: Used for initializing the decoding cache. Typically means that the method signature
        matches EXTEND_STEP, possibly without an input cache state, and returning a prefilled cache
        along with the layer outputs.
    EXTEND_STEP: Used for incremental decoding. Typically means that the method signature consumes
        cache state and emits cache state along with layer outputs.
    """

    FORWARD = 0
    INIT_STATES = 1
    EXTEND_STEP = 2


class BaseTransformerLayer(BaseLayer):
    """An abstract class to define the common interface of all *TransformerLayer classes, including:

    * All subclasses must have `input_dim` in its Config;
    * The common Output structure;
    * The common method signature for `forward()`, `init_states()`, and `extend_step()`.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseTransformerLayer."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.

    class Output(NamedTuple):
        """BaseTransformerLayer output."""

        # [batch, target_length, input_dim]. The layer output.
        data: Tensor

        # The attention probabilities returned by the self-attention layer.
        # Shape: [..., target_length, target_length].
        #
        # self_attention_probs[..., i, j] represents self-attention probability on
        # input data[..., j, :] when computing output data[..., i, :].
        # self_attention_probs.sum(axis=-1) equals to all 1's.
        self_attention_probs: Tensor

        # The attention probabilities returned by the cross-attention layer.
        # Shape: [..., target_length, source_length].
        #
        # If not None, cross_attention_probs[..., i, j] represents attention probability on
        # cross_attention_data[..., j, :] when computing output data[..., i, :].
        # cross_attention_probs.sum(axis=-1) equals to all 1's.
        cross_attention_probs: Optional[Tensor]

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Output:
        """Computes transformer layer outputs given full-sequence inputs.

        For incremental computation, use init_states() and extend_step().

        See comments at the beginning of this file for semantics of *_attention_logit_biases.

        Args:
            data: a Tensor of shape [batch, target_length, input_dim].
            self_attention_logit_biases: an optional Tensor representing the self-attention biases.
            cross_attention_data: an optional Tensor representing cross-attention data of shape
                [source_batch, source_length, source_dim].
            cross_attention_logit_biases: an optional Tensor representing the cross-attention
                biases.

        Returns:
            BaseTransformerLayer.Output.
        """
        raise NotImplementedError(type(self))

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> NestedTensor:
        """Initializes cached states for incremental computation.

        Args:
            target_batch_size: The batch size for target sequences.
            target_max_len: The maximum number of tokens in a target sequence.

        Returns:
            A nested tree of Tensors, which can be used as `cached_states` for the initial call
            of `extend_step()`.
        """
        raise NotImplementedError(type(self))

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, Output]:
        """Initializes cached states for incremental computation.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            data: A Tensor of shape [batch, target_length, input_dim]. For batch index `i`, only
                `data[i, :time_step[i], ...]` will affect subsequent decoding.
            self_attention_logit_biases: An optional Tensor representing the self-attention biases.
            cross_attention_data: An optional Tensor representing cross-attention data of shape
                [batch, source_length, source_dim].
            cross_attention_logit_biases: An optional Tensor representing the cross-attention
                biases.

        Returns:
            A nested tree of Tensors, which can be used as `cached_states` for the initial call
            of `extend_step()`.
            A BaseTransformerLayer.Output instance, where .data is of the same shape as `data`,
            .self_attention_probs is of shape [batch, num_heads, target_length, target_length], and
            .cross_attention_probs is of shape [batch, num_heads, target_length, source_length].
        """
        raise NotImplementedError(type(self))

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, Output]:
        """Computes incremental outputs.

        Args:
            cached_states: a NestedTensor returned by `init_states()` or a previous invocation of
                `extend_step()`.
            data: a Tensor of shape [target_batch_size, target_step_length, input_dim], where
                `target_step_length` is usually 1. For self-attention, `data` will be used as the
                `query` sequence and will be appended to key and value sequences.
            self_attention_logit_biases: an optional Tensor of shape
                [..., target_step_length, target_max_len], where `target_step_length` must match
                the shape of `data` and `target_max_len` must match the value given for
                `init_states()`.
            cross_attention_data: an optional Tensor of shape [..., source_length, source_dim].
            cross_attention_logit_biases: an optional Tensor of shape
                [..., target_step_length, source_length], where `target_step_length` must match
                the shape of `data`.

        Returns:
            (updated_cached_states, output), where:
            `updated_cached_states` represents the new cached states incorporating `data`;
            `output` represents the output for the given input data. `output.data` will have the
            same shape as the input data.
        """
        raise NotImplementedError(type(self))


def make_causal_mask(seq_len: int) -> Tensor:
    """Generates attention logit biases for causal masking.

    Args:
        seq_len: sequence length.

    Returns:
        A float tensor of shape [seq_len, seq_len] where the value at [i, j] = -inf if i < j,
        0 otherwise.
    """
    # TODO(sneha): support batching
    indexes = jnp.arange(seq_len)
    return jax.lax.lt(indexes[:, None], indexes[None, :]) * NEG_INF


def make_segment_mask(*, source_segments: Tensor, target_segments: Tensor) -> Tensor:
    """Generates attention logit biases given the segment ids.

    ... such that positions belonging to different segments cannot attend to each other.

    Args:
        source_segments: An integer tensor of shape [batch, ..., source_length].
        target_segments: An integer tensor of shape [batch, ..., target_length].

    Returns:
        A float Tensor of shape [batch, 1, ..., target_length, source_length] where the
        value at [..., i, j] = 0 if target_segments[..., i] == source_segments[..., j], or -inf
        otherwise.
    """
    target_segments = jnp.expand_dims(target_segments, -1)
    source_segments = jnp.expand_dims(source_segments, -2)
    res = (jax.lax.ne(source_segments, target_segments) * NEG_INF)[:, None, ...]
    return res


class LearnedPositionalEmbedding(BaseLayer):
    """TODO(ruoming): Remove LearnedPositionalEmbedding. We can just use the Embedding layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures LearnedPositionalEmbedding."""

        dim: Required[int] = REQUIRED  # Input feature dim.
        shape: Required[Sequence[int]] = REQUIRED  # The sequence shape.

    # Similar initialization code for Embedding.
    # pylint: disable=duplicate-code
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, "model")
        # By default, initialize to Gaussian with std=1/sqrt(dim), e.g., 0.036 when dim=768.
        #
        # This is the same as:
        # https://github.com/pytorch/fairseq/blob/master/fairseq/modules/positional_embedding.py#L26
        #
        # BERT uses std=0.02 regardless of dim:
        # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L492-L495
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan="fan_out", distribution="normal"
                )
            }
        )
        return cfg

    # pylint: enable=duplicate-code

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=[1] + list(cfg.shape) + [cfg.dim],
                mesh_axes=cfg.param_partition_spec,
            )
        )

    def embeddings(self) -> Tensor:
        """Returns weights of shape cfg.shape + [dim]."""
        return self.parameters["weight"].squeeze(0)

    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: an integer tensor with arbitrary shape [...].

        Returns:
            Embeddings with shape [..., *cfg.dim[1:], dim].
        """
        embeddings = self.embeddings()
        return embeddings[positions]


def sinusoidal_positional_embeddings(
    positions: Tensor, *, dim: int, min_timescale: float = 1, max_timescale: float = 10000
) -> Tensor:
    """Sinusoidal positional embeddings.

    Proposed in the original Transformer paper: https://arxiv.org/abs/1706.03762.

    Reference:
    https://github.com/tensorflow/lingvo/blob/d2f1e1b3cccdac8f73ae20f86afb03560b1c176d/lingvo/core/layers.py#L2775-L2923

    The inputs to the sinusoid functions will be positions / timescale(k)
        for dimension 0 <= k < num_timescales = dim // 2, where:
    timescale(k) = geometric interpolation between min_timescale and max_timescale, i.e.,
      log(timescale(k) / min_timescale) / log(max_timescale / min_timescale) =
      k / num_timescales.
    Specifically: timescale(0) = min_timescale and timescale(num_timescales) = max_timescale.

    Args:
        positions: an integer tensor of any shape [...]. Each value represents an
            absolute or relative position.
        dim: the embedding dimension. Must be divisible by 2.
        min_timescale: the minimum timescale (used for channel 0 and dim // 2).
        max_timescale: the maximum timescale (used for channel dim // 2 - 1 and dim - 1).

    Returns:
        Embeddings of shape [..., dim].

    Raises:
        NotImplementedError: If dim is not divisible by 2.
    """
    if dim % 2 != 0:
        raise NotImplementedError(f"dim ({dim}) must be divisible by 2")
    num_timescales = dim // 2

    # To ensure results match other libraries, it is important to calculate
    # log_timescale_increment using float64 calculations. This has no
    # runtime cost.
    log_timescale_increment = math.log(max_timescale / min_timescale) / max(1, num_timescales - 1)

    # [num_timescales].
    inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales) * -log_timescale_increment)

    # [..., num_timescales].
    scaled_time = jnp.expand_dims(positions, -1) * inv_timescales

    # [..., dim].
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
    return signal


class SinusoidalPositionalEmbedding(BaseLayer):
    """Sinusoidal positional embeddings.

    See sinusoidal_positional_embeddings()'s comments.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures SinusoidalPositionalEmbedding."""

        dim: Required[int] = REQUIRED
        min_timescale: float = 1
        max_timescale: float = 10000

    def forward(self, positions: Tensor) -> Tensor:
        """Looks up positional embeddings by positions."""
        cfg: SinusoidalPositionalEmbedding.Config = self.config
        return sinusoidal_positional_embeddings(
            positions, dim=cfg.dim, min_timescale=cfg.min_timescale, max_timescale=cfg.max_timescale
        )


class BaseMultiheadLinear(BaseLayer):
    """The linear layer used for multi-head attention.

    It uses einsum for efficient computation on TPU to avoid reshaping.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseMultiheadLinear."""

        model_dim: Required[int] = REQUIRED  # Feature dim.
        num_heads: Required[int] = REQUIRED  # Number of attention heads.
        per_head_dim: Required[int] = REQUIRED  # Dimension per head.
        bias: bool = True  # Whether the linear modules have biases.

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        # Shard the 'num_heads' axis by the 'model' dim of the mesh.
        cfg.param_partition_spec = (None, "model", None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.model_dim, cfg.num_heads, cfg.per_head_dim),
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=("row", None, "col")),
            )
        )
        if cfg.bias:
            params["bias"] = self._bias_spec
        return params

    def forward(self, inputs: Tensor) -> Tensor:
        params = self.parameters
        outputs = jnp.einsum(self._einsum_expr, inputs, params["weight"])
        return outputs + params.get("bias", 0)

    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        raise NotImplementedError(type(self))


class MultiheadInputLinear(BaseMultiheadLinear):
    """Multi-head input linear layer."""

    @property
    def _einsum_expr(self):
        return "btd,dnh->btnh"

    @property
    def _bias_spec(self):
        cfg = self.config
        return ParameterSpec(
            shape=(cfg.num_heads, cfg.per_head_dim),
            mesh_axes=cfg.param_partition_spec[-2:],
        )

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "weight":
            return FanAxes(in_axis=0, out_axis=(1, 2))
        else:
            return None


class MultiheadOutputLinear(BaseMultiheadLinear):
    """Multi-head output linear layer."""

    @property
    def _einsum_expr(self):
        return "btnh,dnh->btd"

    @property
    def _bias_spec(self):
        cfg = self.config
        return ParameterSpec(
            shape=(cfg.model_dim,),
            mesh_axes=cfg.param_partition_spec[:1],
        )

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "weight":
            return FanAxes(in_axis=(1, 2), out_axis=0)
        else:
            return None


def apply_attention_logit_biases(
    logits: Tensor, attention_logit_biases: Optional[Tensor] = None
) -> Tensor:
    """Applies `attention_logit_biases` on `logits`.

    Args:
        logits: a float Tensor.
        attention_logit_biases: a float Tensor. If None, assume all zeros.

    Returns:
        logits + attention_logit_biases, in logits.dtype.
    """
    if attention_logit_biases is None:
        return logits
    return logits + attention_logit_biases.astype(logits.dtype)


def softmax_with_biases(logits: Tensor, attention_logit_biases: Optional[Tensor] = None) -> Tensor:
    """Computes softmax with optional masking.

    Args:
        logits: a Tensor of any shape.
        attention_logit_biases: a Tensor that is broadcastable with logits.
            See ``On attention logit biases`` in the file comments.

    Returns:
        A Tensor of same shape and dtype as logits.
    """
    check_numerics(logits)
    logits = apply_attention_logit_biases(logits, attention_logit_biases)
    logits_dtype = logits.dtype
    if logits_dtype in (jnp.bfloat16, jnp.float16):
        # Avoid computing softmax in 16-bit floats.
        logits = logits.astype(jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    if probs.dtype != logits_dtype:
        probs = probs.astype(logits_dtype)
    check_numerics(probs)
    return probs


class BaseQKVLinear(BaseLayer):
    """A layer that encapsulates mapping input queries, keys, and values to
    multi-headed output queries, keys, and values.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseQKVLinear."""

        # Input query feature dim.
        query_dim: Required[int] = REQUIRED
        # Input key feature dim.
        key_dim: Required[int] = REQUIRED
        # Input value feature dim.
        value_dim: Required[int] = REQUIRED
        # Number of attention heads.
        num_heads: Required[int] = REQUIRED
        # Dimension of each attention head.
        per_head_dim: Required[int] = REQUIRED
        # Autoregressive cache dtype. Should match the step dtype.
        # Needs to match the forward dtype for Repeated layers. If None, infer as config.dtype.
        cache_dtype: Optional[jnp.dtype] = None

    class Output(NamedTuple):
        # [batch, target_length, num_heads, per_head_dim].
        query: Tensor
        # [batch, source_length, num_heads, per_head_dim].
        key: Tensor
        # [batch, source_length, num_heads, per_head_dim].
        value: Tensor

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> NestedTensor:
        cfg = self.config
        # Default to base layer dtype for initialization if cache_dtype is None.
        dtype = cfg.cache_dtype or cfg.dtype

        # Following T5X, we cache key, value as [batch, num_heads, head_dim, seq_len] to take
        # advantage of TPU optimizations (see `extend_step`).
        # Reference:
        # https://github.com/google-research/t5x/blob/4d94d8bf41230d492e15e255c9888b5bfd9a5ee8/t5x/examples/t5/layers.py#L215
        cache = dict(
            key=jnp.zeros(
                shape=(target_batch_size, cfg.num_heads, cfg.per_head_dim, target_max_len),
                dtype=dtype,
            ),
            value=jnp.zeros(
                shape=(target_batch_size, cfg.num_heads, cfg.per_head_dim, target_max_len),
                dtype=dtype,
            ),
            time_step=jnp.zeros(target_batch_size, dtype=jnp.int32),
        )
        # TODO(sneha,markblee): Add sharding annotations for all elements in the cache.
        return cache

    def forward(
        self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None
    ) -> Output:
        """Computes per-head query, key, and value for the input query, key, value.

        Args:
            query: a Tensor of shape [batch, target_length, target_dim].
            key:   an optional Tensor of shape [batch, source_length, source_dim].
                   If None, will use `query`.
            value: an optional Tensor of shape [batch, source_length, source_dim].
                   If None, will use `query`.

        Returns:
            An Output instance, where query is of size
            [batch, target_length, num_heads, per_head_dim] and each of key, value are of dim
            [batch, source_length, num_heads, per_head_dim].
        """
        raise NotImplementedError(type(self))

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, Output]:
        """Initializes cache for autoregressive cached decoding.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            query: Tensor of shape [batch, target_length, target_dim] corresponding to query vector
                at `time_step` indices. For batch index `i`, only `query[i, :time_step[i], ...]`
                will affect subsequent decoding.
            key: An optional Tensor of shape [batch, source_length, source_dim]. If None, will use
                `query`.
            value: An optional Tensor of shape [batch, source_length, source_dim]. If None, will
                use `query`.

        Returns:
            A `NestedTensor` state of `key`, `value` of shape
            [batch, num_heads, per_head_dim, source_length], and `time_step` of shape [batch].
            An Output instance, where query is of size
            [batch, target_length, num_heads, per_head_dim] and each of key, value are of dim
            [batch, source_length, num_heads, per_head_dim].
        """
        cfg = self.config
        # Default to base layer dtype for initialization if cache_dtype is None.
        dtype = cfg.cache_dtype or cfg.dtype

        q_proj, k_proj, v_proj = self.forward(query, key=key, value=value)

        # Zero-out everything from time_step onwards. Being able to assume that non-filled cache
        # values are 0 allows us to do a slightly more efficient update to `cached_{key,value}` in
        # `extend_step`, by doing a simple add instead of a mask + add.
        time_step_mask = (jnp.arange(k_proj.shape[1]) < time_step[:, None])[..., None, None]
        k_proj = k_proj * time_step_mask
        v_proj = v_proj * time_step_mask

        # Following T5X, we cache key, value as [batch, num_heads, head_dim, seq_len] to take
        # advantage of TPU optimizations (see `extend_step`).
        # Reference:
        # https://github.com/google-research/t5x/blob/4d94d8bf41230d492e15e255c9888b5bfd9a5ee8/t5x/examples/t5/layers.py#L215
        init_state = dict(
            key=jnp.moveaxis(k_proj, -3, -1).astype(dtype),
            value=jnp.moveaxis(v_proj, -3, -1).astype(dtype),
            time_step=time_step,
        )
        return init_state, self.Output(query=q_proj, key=k_proj, value=v_proj)

    def extend_step(
        self,
        cached_states: NestedTensor,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, Output]:
        """Computes the value vector given the query of the current step.
        This function is used by autoregressive decoding.

        Based on:
        https://github.com/tensorflow/lingvo/blob/5754b2f840ebf0f8c52d87e5d4d76f22e372513e/lingvo/jax/layers/attentions.py#L1249

        Args:
            cached_states: A `NestedTensor` object containing tensors which are the results of
                previous attentions, and index used for fast decoding. Contains "key" and "value" of
                shape [batch, num_heads, per_head_dim, target_length], and a Tensor "time_step" of
                shape [batch].
            query: Tensor of shape [batch, 1, target_dim] corresponding to query vector at
                "time_step" indices.
            key: An optional Tensor of shape [batch, source_length, source_dim]. If None, will use
                `query`.
            value: An optional Tensor of shape [batch, source_length, source_dim]. If None, will
                use `query`.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where query is of size
            [batch, target_length, num_heads, per_head_dim] and each of key, value are of dim
            [batch, source_length, num_heads, per_head_dim].
        """
        time_step = cached_states["time_step"]
        assert time_step.ndim == 1

        # Project inputs to key, value and query. Each has shape [B, 1, N, H].
        q_proj, k_proj, v_proj = self.forward(query, key=key, value=value)

        # Move the length axis to the back. This allows us to update the cache key, value with
        # the "scatter via one-hot broadcast" trick, rather than a scatter/gather operation.
        # Profiling suggests moveaxis is competitive with tweaking einsum in `i_proj` -- it's
        # also a bit simpler, so we keep it for now.
        # [B, 1, N, H] --> [B, N, H, 1].
        k_proj = jnp.moveaxis(k_proj, -3, -1)
        v_proj = jnp.moveaxis(v_proj, -3, -1)

        # Update the cache via one-hot broadcast and addition.
        cached_key = cached_states["key"]
        cached_value = cached_states["value"]
        target_len = cached_key.shape[-1]
        oh_indices = jax.nn.one_hot(time_step, target_len, dtype=k_proj.dtype)
        # [B, 1, 1, T] to broadcast.
        oh_indices = oh_indices[:, None, None, :]
        # Ensure that we accumulate in original dtype.
        new_k_proj = cached_key + (k_proj * oh_indices).astype(cached_key.dtype)
        new_v_proj = cached_value + (v_proj * oh_indices).astype(cached_value.dtype)

        # Move back to original [B, T, N, H] layout.
        k_proj = jnp.moveaxis(new_k_proj, -1, -3)
        v_proj = jnp.moveaxis(new_v_proj, -1, -3)

        updated_state = dict(key=new_k_proj, value=new_v_proj, time_step=time_step + 1)
        return updated_state, self.Output(query=q_proj, key=k_proj, value=v_proj)


class QKVLinear(BaseQKVLinear):
    """Maps input query, key, and value to multi-headed output query, key, and value."""

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures for QKVLinear."""

        # The layer used to project.
        layer: MultiheadInputLinear.Config = MultiheadInputLinear.default_config()

    def __init__(self, cfg: BaseQKVLinear.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        for name, dim in (
            ("q", cfg.query_dim),
            ("k", cfg.key_dim),
            ("v", cfg.value_dim),
        ):
            proj_cfg = cfg.layer
            proj_cfg.model_dim = dim
            proj_cfg.num_heads = cfg.num_heads
            proj_cfg.per_head_dim = cfg.per_head_dim
            self._add_child(f"{name}_proj", proj_cfg)

    def forward(
        self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None
    ) -> BaseQKVLinear.Output:
        """Computes attention for the given query, key, value.

        If `key` or `value` are None, will use `query` in place.

        See parent class for full docstring.
        """
        key = query if key is None else key
        value = query if value is None else value
        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)
        return self.Output(query=q_proj, key=k_proj, value=v_proj)


class FusedQKVLinear(BaseQKVLinear):
    """Maps input query, key, and value to multi-headed query, key, and value
    using a fused weight.

    N.B. Only supports cases where query, key, and value all have the same shape.
    """

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures for FusedQKVLinear."""

        # The layer used to project.
        layer: MultiheadInputLinear.Config = MultiheadInputLinear.default_config()

    def __init__(self, cfg: BaseQKVLinear.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if not cfg.query_dim == cfg.key_dim == cfg.value_dim:
            raise ValueError(
                f"All projection dims must be equal for {type(self)}, saw: "
                f"query:{cfg.query_dim}, key:{cfg.key_dim}, value:{cfg.value_dim}"
            )
        proj_cfg = cfg.layer
        proj_cfg.model_dim = cfg.query_dim
        proj_cfg.num_heads = cfg.num_heads
        proj_cfg.per_head_dim = cfg.per_head_dim
        self._add_child("qkv_proj", proj_cfg)

    # Similar (but not identical) code in repeat.py.
    # pylint: disable=duplicate-code
    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        specs = VDict(**super().create_parameter_specs_recursively())

        def transform_factorization_spec(
            spec: Optional[FactorizationSpec],
        ) -> Optional[FactorizationSpec]:
            if spec is None:
                return None
            return FactorizationSpec(axes=[None] + list(spec.axes))

        return jax.tree_util.tree_map(
            lambda spec: ParameterSpec(
                dtype=spec.dtype,
                shape=(3, *spec.shape),
                mesh_axes=PartitionSpec(None, *spec.mesh_axes),
                factorization=transform_factorization_spec(spec.factorization),
                fan_axes=param_init.maybe_prepend_axis(
                    spec.fan_axes, axis_type=param_init.FanAxes.AxisType.BATCH_AXIS
                ),
            ),
            specs,
        )

    def initialize_parameters_recursively(
        self, prng_key: jax.random.KeyArray, *, prebuilt: Optional[NestedTensor] = None
    ) -> NestedTensor:
        if self._use_prebuilt_params(prebuilt):
            return prebuilt

        def init(prng_key_i):
            return VDict(qkv_proj=self.qkv_proj.initialize_parameters_recursively(prng_key_i))

        return jax.vmap(init)(split_prng_key(prng_key, 3).keys)

    # pylint: enable=duplicate-code

    def forward(
        self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None
    ) -> BaseQKVLinear.Output:
        """Computes multi-head query, key, and value for the input query, key, value
        using a fused weight.

        N.B. Only supports cases where query, key, and value all have the same shape if set.

        See parent class for full docstring.

        Raises:
            ValueError: If key and value are not both set or both None.
        """
        with child_context("qkv_proj"):
            params = self.qkv_proj.parameters
            if key is None and value is None:
                # Computing self attention.
                # N.B. this branch (with just the query inputs) is required in
                # order to get the best step time on TPU for self-attention.
                inputs = query  # [batch, target_length, target_dim].
                proj = jnp.einsum("btd,pdnh->pbtnh", inputs, params["weight"])
            elif key is not None and value is not None:
                # Compute cross attention but with same target/source shapes.
                assert (
                    query.shape == key.shape == value.shape  # pytype: disable=attribute-error
                ), f"Not supported for {type(self)}."
                inputs = jnp.stack(
                    [query, key, value], axis=0
                )  # [q/k/v, batch, target, model_dim].
                proj = jnp.einsum("pbtd,pdnh->pbtnh", inputs, params["weight"])
            else:
                raise ValueError("Key and value should be either both None or both set.")
            if self.qkv_proj.config.bias:
                bias = jnp.expand_dims(
                    params.get("bias", jnp.array([0], dtype=query.dtype)),
                    (1, 2),
                )
                proj = proj + bias
            q_proj, k_proj, v_proj = proj
        return self.Output(query=q_proj, key=k_proj, value=v_proj)


def _rotary_sinusoidal_positional_embeddings(
    *, positions: Tensor, max_len: int, dim: int, theta: float = 10000.0
) -> Tensor:
    """Generate the sin/cos positional embedding.

    Ref:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L76-L90

    Args:
        positions: A tensor representing the token position IDs with shape [seq_len].
        max_len: The max length of the input sequence.
        dim: The dimensionality of the positional embedding.
        theta: A parameter to scale the frequencies.

    Returns:
        Rotary Positional Embedding with shape [seq_len, dim].
    """
    exponents = jnp.arange(dim).astype(jnp.float32)
    pos_array = jnp.arange(max_len).astype(jnp.float32)
    exponents = jnp.power(theta, 2 * (exponents // 2) / dim)
    position_enc = jnp.expand_dims(pos_array, 1) / jnp.expand_dims(exponents, 0)

    rope_part_1 = jnp.sin(position_enc[:, 0::2])
    rope_part_2 = jnp.cos(position_enc[:, 1::2])
    rope = jnp.concatenate((rope_part_1, rope_part_2), axis=-1)
    return rope[positions]


class RoFormerSinusoidalPositionalEmbedding(BaseLayer):
    """Implementation of Rotary Position Embedding (RoPE).

    Ref:
    https://github.com/huggingface/transformers/blob/62ceb4/src/transformers/models/roformer/modeling_roformer.py
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures RoFormerSinusoidalPositionalEmbedding."""

        max_len: Required[int] = REQUIRED  # The max length of the input sequence.
        dim: Required[int] = REQUIRED  # The dimensionality of the positional embedding.
        theta: float = 10000.0  # The scale of base frequency.

    def forward(self, positions: Tensor) -> Tensor:
        """
        TODO(bwzhang): 1. add the batch support. 2. verify the performance under float32.

        Args:
            positions: A tensor representing the token position IDs.
                Currently, it doesn't support batched tensor as input.
                The shape is [seq_len].

        Returns:
            Rotary Positional Embedding. Shape is [seq_len, dim].

        Raises:
            ValueError: If positions has invalid shape.
        """
        cfg = self.config
        seq_len = positions.shape[0]
        if seq_len > cfg.max_len:
            raise ValueError(
                f"Seq. length ({seq_len}) should be less than or "
                "equal to max length ({cfg.max_len})"
            )
        return _rotary_sinusoidal_positional_embeddings(
            positions=positions, max_len=cfg.max_len, dim=cfg.dim, theta=cfg.theta
        )


def apply_rotary_position_embeddings(
    *,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sinusoidal_pos: Tensor,
    rotary_value: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """This is a jax implementation (a copy) of the RoPE apply_rotary_position_embeddings.

    Ref:
    https://github.com/huggingface/transformers/blob/v4.21.2/src/transformers/models/roformer/modeling_roformer.py#L322-L346

    Args:
        query: Query embeddings with shape [batch_size, seq_len, num_heads, dim].
        key: Key embeddings with shape [batch_size, seq_len, num_heads, dim].
        value: Value embeddings with shape [batch_size, seq_len, num_heads, dim].
        sinusoidal_pos: Rotary position embeddings with shape [1, seq_len, 1, dim].
        rotary_value: Whether to apply rotary position embeddings on value layer.

    Returns:
        A tuple of:
        Rotary position affined query embeddings with shape [batch_size, seq_len, num_heads, dim]
        Rotary position affined key embeddings with shape [batch_size, seq_len, num_heads, dim]
        Rotary position affined value embeddings with shape [batch_size, seq_len, num_heads, dim]
            if rotary_value == True, else original value embeddings
    """
    # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    sin, cos = jnp.split(sinusoidal_pos, 2, axis=-1)
    # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    sin_pos = jnp.reshape(jnp.stack([sin, sin], axis=-1), sinusoidal_pos.shape)
    # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    cos_pos = jnp.reshape(jnp.stack([cos, cos], axis=-1), sinusoidal_pos.shape)
    # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_query = jnp.reshape(
        jnp.stack([-query[..., 1::2], query[..., ::2]], axis=-1), query.shape
    )
    query = query * cos_pos + rotate_half_query * sin_pos
    # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
    rotate_half_key = jnp.reshape(jnp.stack([-key[..., 1::2], key[..., ::2]], axis=-1), key.shape)
    key = key * cos_pos + rotate_half_key * sin_pos
    if rotary_value:
        # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
        rotate_half_value = jnp.reshape(
            jnp.stack([-value[..., 1::2], value[..., ::2]], axis=-1), value.shape
        )
        value = value * cos_pos + rotate_half_value * sin_pos
    return query, key, value


class RoFormerQKVLinear(BaseQKVLinear):
    """RoFormerQKVLinear class

    This class maps the query, key, and value using the RoPE embeddings.
    """

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures RoFormerQKVLinear."""

        max_seq_length: Required[int] = REQUIRED
        rope_pos_emb_layer: InstantiableConfig = (
            RoFormerSinusoidalPositionalEmbedding.default_config()
        )
        input_linear: BaseQKVLinear.Config = QKVLinear.default_config()
        rotary_value: Required[bool] = REQUIRED

    def __init__(self, cfg: QKVLinear.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "rope_pos_emb_layer",
            cfg.rope_pos_emb_layer.set(max_len=cfg.max_seq_length, dim=cfg.per_head_dim),
        )
        self._add_child(
            "i_proj",
            cfg.input_linear.set(
                query_dim=cfg.query_dim,
                value_dim=cfg.value_dim,
                key_dim=cfg.key_dim,
                num_heads=cfg.num_heads,
                per_head_dim=cfg.per_head_dim,
            ),
        )

    def forward(
        self, query: Tensor, *, key: Optional[Tensor] = None, value: Optional[Tensor] = None
    ) -> BaseQKVLinear.Output:
        cfg = self.config
        query, key, value = self.i_proj(query, key=key, value=value)
        # Query should have shape of [batch_size, seq_len, num_heads, per_head_dim].
        # So `positions` will be in range [0, seq_len - 1).
        positions = jnp.arange(query.shape[1])
        # sinusoidal_pos_emb shape should be [1, seq_len, 1, dim]
        sinusoidal_pos_emb = jnp.expand_dims(self.rope_pos_emb_layer.forward(positions), [0, 2])
        sinusoidal_pos_emb = sinusoidal_pos_emb.astype(query.dtype)
        query, key, value = apply_rotary_position_embeddings(
            sinusoidal_pos=sinusoidal_pos_emb,
            query=query,
            key=key,
            value=value,
            rotary_value=cfg.rotary_value,
        )

        return self.Output(query, key, value)


class PerDimScale(BaseLayer):
    """A layer to scale individual dimensions of the input."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures PerDimScale."""

        dim: Required[int] = REQUIRED

    @classmethod
    def default_config(cls) -> Config:
        cfg: PerDimScale.Config = super().default_config()
        cfg.param_init = ConstantInitializer.default_config().set(value=0.0)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "param": ParameterSpec(shape=(cfg.dim,), mesh_axes=(None,)),
        }

    def forward(self, x: Tensor) -> Tensor:
        """Returns x * per_dim_scale."""
        cfg = self.config
        assert x.shape[-1] == cfg.dim
        # https://github.com/tensorflow/lingvo/blob/3d16483b749a1181330ae9ce318688e7518d63c9/lingvo/jax/layers/attentions.py#L232-L234
        # 1.0/jax.nn.softplus(0.0) = 1.442695041. Hard code this number to avoid unnecessary
        # XLA op fusion.
        r_softplus_0 = 1.442695041
        scale = jax.nn.softplus(self.parameters["param"]) * r_softplus_0
        return (x * scale).astype(x.dtype)


ScaleFn = Callable[[int], float]  # A function mapping per_head_dim to a scale.


def constant_scale_config(value: float) -> InstantiableConfig[ScaleFn]:
    """A config for a constant scale function for `MultiheadAttention`.

    Args:
        value: The value to scale by.

    Example:
        `query_scale = config_for_function(constant_scale).set(value=0.01)`

    Returns:
        A config that scales by `value`.
    """

    def constant_function(_: float, value: float) -> float:
        return value

    return config_for_partial_function(constant_function, value=value)


class MultiheadAttention(BaseLayer):
    """A basic multi-head attention layer.

    Differences from torch.nn.MultiheadAttention:
    - Use of einsum for efficient computation on TPU to avoid reshaping;
    - Separate weights for {q,k,v}_proj for proper weight initialization that depends
      on fan-out and efficient TPU execution (where split is not free).
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures MultiheadAttention."""

        query_dim: Required[int] = REQUIRED  # Input query feature dim.
        key_dim: Required[int] = REQUIRED  # Input key feature dim.
        value_dim: Required[int] = REQUIRED  # Input value feature dim.
        output_dim: Optional[int] = None  # Output feature dim. If None, use query_dim.
        hidden_dim: Optional[int] = None  # Hidden feature dim. If None, use query_dim.
        # Number of attention heads. Must divide hidden_dim evenly.
        num_heads: Required[int] = REQUIRED
        # Config used to produce Q,K,V projections.
        input_linear: BaseQKVLinear.Config = QKVLinear.default_config()
        # Config used for the output projection.
        output_linear: MultiheadOutputLinear.Config = MultiheadOutputLinear.default_config()
        # The dropout layer.
        dropout: Dropout.Config = Dropout.default_config()
        # The config for a function to compute a scale factor for the query matrix.
        # If None, then self.head_dim() ** -0.5.
        query_scale: Optional[InstantiableConfig[ScaleFn]] = None
        # The config for a function to compute a scale factor for the key matrix.
        # If None, then 1.
        key_scale: Optional[InstantiableConfig[ScaleFn]] = None
        # A vector to apply per dimension scale to the query projection.
        per_dim_scale: Optional[PerDimScale.Config] = None
        # Cap the absolute values of logits by tanh. Enabled by setting a positive value.
        atten_logit_cap: Optional[float] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # Configure inputs to multi-headed QKV projection.
        i_proj_cfg = cfg.input_linear
        i_proj_cfg.query_dim = cfg.query_dim
        i_proj_cfg.key_dim = cfg.key_dim
        i_proj_cfg.value_dim = cfg.value_dim
        i_proj_cfg.num_heads = cfg.num_heads
        i_proj_cfg.per_head_dim = self.per_head_dim()
        self._add_child("i_proj", i_proj_cfg)
        # Configure output projection.
        o_proj_cfg = cfg.output_linear
        o_proj_cfg.model_dim = self.output_dim()
        o_proj_cfg.num_heads = cfg.num_heads
        o_proj_cfg.per_head_dim = self.per_head_dim()
        self._add_child("o_proj", o_proj_cfg)
        # Add dropout layer.
        self._add_child("dropout", cfg.dropout)
        if cfg.per_dim_scale:
            self._add_child("per_dim_scale", cfg.per_dim_scale.set(dim=self.per_head_dim()))
        self._query_scale = self.default_query_scale_config()
        if cfg.query_scale is not None:
            self._query_scale = cfg.query_scale
        self._query_scale = self._query_scale.instantiate()
        self._key_scale = self.default_key_scale_config()
        if cfg.key_scale is not None:
            self._key_scale = cfg.key_scale
        self._key_scale = self._key_scale.instantiate()

    def output_dim(self):
        cfg = self.config
        return cfg.output_dim or cfg.query_dim

    def hidden_dim(self):
        cfg = self.config
        return cfg.hidden_dim or cfg.query_dim

    def per_head_dim(self):
        cfg = self.config
        hidden_dim = self.hidden_dim()
        if hidden_dim % cfg.num_heads != 0:
            raise ValueError(f"num_heads ({cfg.num_heads}) must divide hidden_dim ({hidden_dim})")
        return hidden_dim // cfg.num_heads

    class Output(NamedTuple):
        # [batch, target_length, output_dim]. The attention output.
        data: Tensor
        # [batch, num_heads, target_length, source_length]. The attention probabilities.
        probs: Tensor

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
    ) -> Tuple[Optional[NestedTensor], Output]:
        """Computes attention for the given query, key, value, and attention logit biases.

        If key and value are both None, computes self-attention using query.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            query: A Tensor of shape [batch, target_length, target_dim].
            key:   An optional Tensor of shape [batch, source_length, source_dim].
            value: An optional Tensor of shape [batch, source_length, source_dim].
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If key & value are an invalid combination.
            ValueError: If `mode` is unsupported.
        """
        # Validate key & value combination.
        if (key is None) != (value is None):
            raise ValueError(
                "key and value must be both None or both set, "
                f"key:{type(key)}, value:{type(value)}"
            )
        if mode == ForwardMode.FORWARD:
            i_proj_state, (q_proj, k_proj, v_proj) = None, self.i_proj(query, key=key, value=value)
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            i_proj_state, (q_proj, k_proj, v_proj) = self.i_proj.prefill_states(
                time_step=cached_states["i_proj"], query=query, key=key, value=value
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            i_proj_state, (q_proj, k_proj, v_proj) = self.i_proj.extend_step(
                cached_states["i_proj"], query, key=key, value=value
            )
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        q_proj = self._remat_name(q_proj, "q_proj")
        k_proj = self._remat_name(k_proj, "k_proj")
        v_proj = self._remat_name(v_proj, "v_proj")
        self.vlog(3, "atten.q_proj=%s", q_proj.sum())
        self.vlog(3, "atten.k_proj=%s", k_proj.sum())
        self.vlog(3, "atten.v_proj=%s", v_proj.sum())
        if attention_logit_biases is not None and attention_logit_biases.ndim == 3:
            # [batch, 1, target_length, source_length].
            attention_logit_biases = attention_logit_biases[:, None, :, :]
        context, probs = self._compute_attention(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            attention_logit_biases=attention_logit_biases,
        )
        self.vlog(3, "atten.prob=%s", probs[0, 0, 0, :])
        self.vlog(3, "atten.context=%s", context.sum())
        # [batch, target_length, output_dim].
        o_proj = self.o_proj(context)
        outputs = self._remat_name(o_proj, "o_proj")
        return dict(i_proj=i_proj_state), self.Output(data=outputs, probs=probs)

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Computes attention context and probs.

        Args:
            q_proj: [batch_size, target_length, num_heads, per_head_dim].
            k_proj: [batch_size, source_length, num_heads, per_head_dim].
            v_proj: [batch_size, source_length, num_heads, per_head_dim].
            attention_logit_biases: See ``On attention logit biases`` in the file comments.

        Returns:
            The context of shape [batch_size, target_length, num_heads, per_head_dim],
            and probs of shape [batch, num_heads, target_length, source_length].
        """
        logits = self._compute_logits(q_proj, k_proj)
        logits = self._cap_logits(logits)
        self.vlog(3, "atten.logits=%s", logits[0, 0, 0, :])
        probs = softmax_with_biases(logits, attention_logit_biases=attention_logit_biases)
        probs = self.dropout(probs)
        context = jnp.einsum("bnts,bsnh->btnh", probs, v_proj).astype(v_proj.dtype)
        context = self._remat_name(context, "context")
        return context, probs

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
    ) -> Output:
        """Computes attention for the given query, key, value, and attention logit biases.

        If key and value are both None, computes self-attention using query.

        Args:
            query: a Tensor of shape [batch, target_length, target_dim].
            key:   an optional Tensor of shape [batch, source_length, source_dim].
            value: an optional Tensor of shape [batch, source_length, source_dim].
            attention_logit_biases:  See ``On attention logit biases`` in the file comments.

        Returns:
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If key & value are an invalid combination.
        """
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            query=query,
            key=key,
            value=value,
            attention_logit_biases=attention_logit_biases,
        )
        return output

    T = TypeVar("T", bound=Union[float, Tensor])

    def _scale_query(self, q_proj: T) -> T:
        cfg = self.config
        if cfg.per_dim_scale is not None:
            # The Lingvo MultiheadAttention applies a per_dim_scale on q_proj:
            # https://github.com/tensorflow/lingvo/blob/41212226eac7a26491790c2bd476b78493f93ff6/lingvo/core/batch_major_attention.py#L790
            q_proj = self.per_dim_scale(q_proj)
        scale = self._query_scale(self.per_head_dim())
        q_proj = q_proj * scale
        if isinstance(q_proj, float):
            return q_proj
        # Force multiplying q_proj by scale before multiplying it by k_proj.
        # This prevents constant folding of the scale factors for q_proj
        # and k_proj, allowing increased numerical stability if the user
        # splits the scale factor between them.
        return ops.forward_optimization_barrier(q_proj)

    def _scale_key(self, k_proj: T) -> T:
        scale = self._key_scale(self.per_head_dim())
        k_proj = k_proj * scale
        if isinstance(k_proj, float):
            return k_proj
        # Force multiplying k_proj by scale before multiplying it by q_proj.
        # This prevents constant folding of the scale factors for q_proj
        # and k_proj, allowing increased numerical stability if the user
        # splits the scale factor between them.
        return ops.forward_optimization_barrier(k_proj)

    def _cap_logits(self, logits: Tensor) -> Tensor:
        """Caps the logits with tanh."""
        cfg = self.config
        if not cfg.atten_logit_cap or cfg.atten_logit_cap <= 0.0:
            return logits
        cap = jnp.array(cfg.atten_logit_cap, dtype=logits.dtype)
        return cap * jnp.tanh(logits / cap)

    def _compute_logits(self, q_proj: Tensor, k_proj: Tensor) -> Tensor:
        q_proj = self._scale_query(q_proj)
        k_proj = self._scale_key(k_proj)
        return jnp.einsum("btnh,bsnh->bnts", q_proj, k_proj)

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            target_batch_size: the batch size of the target to be decoded.
            target_max_len: the sequence length of the target to be decoded.

        Returns:
            The cache as a `NestedTensor` with key and value initialized.
        """
        return dict(
            i_proj=self.i_proj.init_states(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        query: Tensor,
        attention_logit_biases: Tensor,
    ) -> Tuple[NestedTensor, Output]:
        """Initializes cache for autoregressive cached decoding.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [B]. Each value is an index into the length dimension
                indicating where decoding will start from.
            query: Tensor of shape [B, T, D] corresponding to query vector up to `time_step`. For
                batch index `i`, only `query[i, :time_step[i], ...]` will affect subsequent
                decoding.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].
        """
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            query=query,
            cached_states=dict(i_proj=time_step),
            attention_logit_biases=attention_logit_biases,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        query: Tensor,
        *,
        attention_logit_biases: Tensor,
    ) -> Tuple[NestedTensor, Output]:
        """Computes the value vector given the query of the current step.
        This function is used by autoregressive decoding.

        Based on:
        https://github.com/tensorflow/lingvo/blob/5754b2f840ebf0f8c52d87e5d4d76f22e372513e/lingvo/jax/layers/attentions.py#L1249

        Args:
            cached_states: A `NestedTensor` object containing tensors which are the results of
                previous attentions, and index used for fast decoding. Contains "key" and "value" of
                shape [B, N, H, T], and a Tensor "time_step" of shape [B].
            query: Tensor of shape [B, 1, D] corresponding to query vector at "time_step" indices.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
                Additionally, target_length is expected to be 1 since this is per time step.
                The biases should already include causal masking for decoding, plus other biases
                if necessary.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where .data is of the same shape as query, .probs is of shape
            [batch, num_heads, 1, source_length].
        """
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            query=query,
            cached_states=cached_states,
            attention_logit_biases=attention_logit_biases,
        )

    @staticmethod
    def default_query_scale_config() -> InstantiableConfig[ScaleFn]:
        """The config for the default function used to compute the query scale."""
        pow_real = typing.cast(Callable[..., float], pow)
        return config_for_partial_function(pow_real, exp=-0.5)

    @staticmethod
    def default_key_scale_config() -> InstantiableConfig[ScaleFn]:
        """The config for the default function used to compute the key scale."""
        return constant_scale_config(1)


def rel_pos_to_abs_pos(x: Tensor) -> Tensor:
    """Converts a (T, relative_pos_offset) Tensor to a (T, abs_position) tensor.

    For example, t = 3:
    ..abc      abc
    .def.  =>  def
    ghi..      ghi

    Input shape: [t, 2t - 1]:
    ..abc
    .def.
    ghi..

    1. Reshape to [t * (2t - 1)]
    ..abc.def.ghi..

    2. Trim by [t-1:-1], producing shape [t * (2t - 2)].
    abc.def.ghi.

    3. Reshape to [t, 2t - 2]:
    abc.
    def.
    ghi.

    4. Trim by [:, :-(t-2)]
    abc
    def
    ghi

    Args:
        x: a Tensor of shape [T, 2*T - 1], where x[i, j] represents the bias between query[i] and
            absolute position k = i + j - (T - 1), if 0 <= k < T, otherwise the value is not used.

    Returns:
        y: a Tensor of shape [T, T], s.t. y[i, k] = x[i, j] where k = i + j - (T - 1),
            if 0 <= k < T.
    """
    t, offset_length = x.shape
    assert offset_length == 2 * t - 1
    # [t * (2t - 1)].
    x = x.reshape([-1])
    # [t * (2t - 2)].
    x = x[t - 1 : -1]
    # [t, 2t - 2].
    x = x.reshape([t, -1])
    # [t, t].
    x = x[:, : -(t - 2)]
    return x


class MultiheadRelativePositionLinear(BaseMultiheadLinear):
    """Multi-head relative position linear layer."""

    @property
    def _einsum_expr(self):
        return "ld,dnh->lnh"

    @property
    def _bias_spec(self):
        cfg = self.config
        return ParameterSpec(
            shape=(cfg.num_heads, cfg.per_head_dim),
            mesh_axes=cfg.param_partition_spec[-2:],
        )

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "weight":
            return FanAxes(in_axis=0, out_axis=(1, 2))
        else:
            return None


def xl_attention_logits(
    q_proj: Tensor, k_proj: Tensor, relative_pos_emb: Tensor, u: Tensor, v: Tensor
):
    """Computes Transformer XL self-attention logits.

    Note that this implementation follows XLNet implementation and is different from the lingvo
    implementation in that here the relative_pos_emb index is computed from key_i - query_i,
    while lingvo computes from query_i - key_i.

    Args:
        q_proj: a Tensor of shape [batch, target_length, num_heads, per_head_dim], representing
            projected queries.
        k_proj: a Tensor of shape [batch, target_length, num_heads, per_head_dim], representing
            projected keys.
        relative_pos_emb: a Tensor of shape [num_embeddings, num_heads, per_head_dim], representing
            projected relative positional embeddings, where num_embeddings = 2 * target_length - 1.
            relative_pos_emb[key_i - query_i + target_length - 1] represents positional
            embeddings between query[:, query_i] and key[:, key_i] and is usually computed from
            sinusoidal_positional_embeddings(query_i - key_i), i.e.,
            relative_pos_emb[0] represents query_i = target_length - 1 and key_i = 0.
            relative_pos_emb[-1] represents query_i = 0 and key_i = target_length - 1.
        u: a Tensor of shape [num_heads, per_head_dim].
            The trainable `u` in https://arxiv.org/pdf/1901.02860.pdf 3.3 used for term 'ac'.
        v: a Tensor of shape [num_heads, per_head_dim].
            The trainable `v` in https://arxiv.org/pdf/1901.02860.pdf 3.3 used for term 'bd'.

    Returns:
        A tensor of shape [batch, num_heads, target_length, target_length] representing attention
        logits. logit[:, :, i, j] represents the logit for query[i] and key[j].
    """
    term_ac = jnp.einsum("btnh,bsnh->bnts", q_proj + u, k_proj)
    term_bd = jnp.einsum("btnh,lnh->bntl", q_proj + v, relative_pos_emb)
    # Apply vmap twice to map over both `batch` and `num_heads`.
    term_bd = jax.vmap(jax.vmap(rel_pos_to_abs_pos))(term_bd)
    return term_ac + term_bd


class MultiheadAttentionXL(MultiheadAttention):
    """Multi-head self-attention with relative positional embeddings.

    The default config matches XL-Net implementation with `per_dim_scale=None` and
    `scale_position=LOGIT`.
    To match with Lingvo implementation, enable `per_dim_scale`
    and set `scale_position=QUERY`. Note the positional embeddings are in descending
    order, which is different from Lingvo's implementation.

    Reference:
    https://github.com/zihangdai/xlnet/blob/bbaa3a6fa0b3a2ee694e8cf66167434f9eca9660/modeling.py
    https://github.com/huggingface/transformers/blob/224bde91caff4ccfd12277ab5e9bf97c61e22ee9/src/transformers/models/xlnet/modeling_xlnet.py#L204
    https://github.com/tensorflow/lingvo/blob/a1326a09641a6ec7d775a51148012551756d888d/lingvo/core/batch_major_attention.py#L1345
    https://github.com/tensorflow/lingvo/blob/f02fed838836bcc513d51c95d482247b119543fb/lingvo/core/attention_util.py#L464-L513
    """

    @unique
    class ScalePosition(Enum):
        # Applies 1/sqrt(dim) scaling on the logits.
        LOGIT = 0
        # Applies 1/sqrt(dim) scaling on the queries.
        QUERY = 1

    @config_class
    class Config(MultiheadAttention.Config):
        """Configures MultiheadAttentionXL."""

        pos_emb_dim: Optional[int] = None  # Positional embedding dim. If None, use query_dim.
        # Config for computing relative position embeddings for range [-seq_len + 1, seq_len - 1].
        relative_pos_emb: SinusoidalPositionalEmbedding.Config = (
            SinusoidalPositionalEmbedding.default_config()
        )
        # Config used for the R projection.
        relative_pos_linear: MultiheadRelativePositionLinear.Config = (
            MultiheadRelativePositionLinear.default_config().set(bias=False)
        )
        scale_position: Required["MultiheadAttentionXL.ScalePosition"] = REQUIRED

    @classmethod
    def default_config(cls) -> Config:
        cfg: MultiheadAttentionXL.Config = super().default_config()
        cfg.scale_position = MultiheadAttentionXL.ScalePosition.LOGIT
        # pylint: disable=no-member
        cfg.input_linear = FusedQKVLinear.default_config()
        cfg.input_linear.layer.bias = False
        # pylint: enable=no-member
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: MultiheadAttentionXL.Config = self.config
        if not cfg.query_dim == cfg.key_dim == cfg.value_dim:
            raise ValueError(
                f"MultiheadAttentionXL requires query_dim ({cfg.query_dim}) == "
                f"key_dim ({cfg.key_dim}) == value_dim ({cfg.value_dim})"
            )
        pos_emb_dim = cfg.pos_emb_dim or cfg.query_dim
        self._add_child("relative_pos_emb", cfg.relative_pos_emb.set(dim=pos_emb_dim))
        self._add_child(
            "r_proj",
            cfg.relative_pos_linear.clone(
                model_dim=pos_emb_dim, num_heads=cfg.num_heads, per_head_dim=self.per_head_dim()
            ),
        )

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        params = super()._create_layer_parameter_specs()
        params["u_bias"] = params["v_bias"] = ParameterSpec(
            shape=(cfg.num_heads, self.per_head_dim()),
            initializer=constant_initializer(0),
            mesh_axes=cfg.relative_pos_linear.param_partition_spec[-2:],
        )
        return params

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
    ) -> MultiheadAttention.Output:
        if key is not None or value is not None:
            raise ValueError("Both key and value must be None for MultiheadAttentionXL")
        return super().forward(query, attention_logit_biases=attention_logit_biases)

    def _compute_logits(self, q_proj: Tensor, k_proj: Tensor) -> Tensor:
        cfg = self.config
        if cfg.per_dim_scale is not None:
            # Applies a per dim scale on q_proj.
            q_proj = self.per_dim_scale(q_proj)

        if cfg.scale_position == MultiheadAttentionXL.ScalePosition.QUERY:
            scale = self.per_head_dim() ** -0.5
            q_proj = q_proj * scale

        seq_len = q_proj.shape[1]
        # [2*seq_len - 1, pos_emb_dim].
        #
        # Following the XLNet implementation
        # https://github.com/zihangdai/xlnet/blob/bbaa3a6fa0b3a2ee694e8cf66167434f9eca9660/modeling.py#L215
        # https://github.com/huggingface/transformers/blob/28d0048218ad7bce69510b16024510afba0daed2/src/transformers/models/xlnet/modeling_xlnet.py#L1030
        # the positions are from descending from seq_len - 1 to -seq_len + 1.
        pos_emb = self.relative_pos_emb(jnp.arange(seq_len - 1, -seq_len, -1, dtype=jnp.int32))
        # [2*seq_len - 1, num_heads, per_head_dim].
        r_proj = self.r_proj(pos_emb)

        logits = xl_attention_logits(
            q_proj=q_proj,
            k_proj=k_proj,
            relative_pos_emb=r_proj,
            u=self.parameters["u_bias"],
            v=self.parameters["v_bias"],
        )
        if cfg.scale_position == MultiheadAttentionXL.ScalePosition.LOGIT:
            # In the original XL-Net code, it applies scale on AC + BD:
            #
            # https://github.com/zihangdai/xlnet/blob/bbaa3a6fa0b3a2ee694e8cf66167434f9eca9660/modeling.py#L148
            scale = self.per_head_dim() ** -0.5
            logits = logits * scale
        return logits

    def extend_step(
        self,
        cached_states: NestedTensor,
        query: Tensor,
        *,
        attention_logit_biases: Tensor,
    ) -> Tuple[NestedTensor, MultiheadAttention.Output]:
        raise NotImplementedError(type(self))


class TransformerAttentionLayer(BaseLayer):
    """A Transformer attention layer with normalization and a skip connection.

    Can be used for either self-attention or cross-attention.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures TransformerAttentionLayer."""

        target_dim: Required[int] = REQUIRED  # Input target feature dim.
        source_dim: Required[int] = REQUIRED  # Input source feature dim.
        norm: InstantiableConfig = LayerNorm.default_config()  # The normalization layer config.
        attention: InstantiableConfig = (
            MultiheadAttention.default_config()
        )  # The attention layer config.
        dropout: InstantiableConfig = Dropout.default_config()  # The dropout layer config.
        # The stochastic depth layer config.
        # Pytorch reference:
        # https://github.com/facebookresearch/deit/blob/main/models_v2.py#L58
        # Tensorflow reference:
        # https://github.com/tensorflow/models/blob/master/official/projects/vit/modeling/nn_blocks.py#L86-L92
        stochastic_depth: InstantiableConfig = StochasticDepth.default_config()
        # The inner structure of the layer: prenorm or postnorm. See
        # https://arxiv.org/abs/2002.04745 for background.
        # The structure also support hybridnorm, which uses two norms in the residual branch.
        # hybridnorm: TransformerAttentionLayer(x) = x + layernorm_2(attention(layernorm_1(x)))
        # Ref: https://github.com/google/praxis/blob/main/praxis/layers/transformers.py#L1129
        # TODO (bwzhang@) Adding a unittest for the hybridnorm.
        structure: str = "prenorm"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.structure in ["prenorm", "postnorm"]:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.target_dim))
        elif cfg.structure == "hybridnorm":
            self._add_child("prenorm", cfg.norm.set(input_dim=cfg.target_dim))
            self._add_child("postnorm", cfg.norm.set(input_dim=cfg.target_dim))
        else:
            raise NotImplementedError(cfg.structure)
        self._add_child(
            "attention",
            cfg.attention.set(
                query_dim=cfg.target_dim,
                key_dim=cfg.source_dim,
                value_dim=cfg.source_dim,
                output_dim=cfg.target_dim,
            ),
        )
        self._add_child("dropout", cfg.dropout)
        self._add_child("stochastic_depth", cfg.stochastic_depth)

    class Output(NamedTuple):
        # [batch, target_length, output_dim]. The attention output.
        data: Tensor
        # The attention probabilities returned by the attention layer.
        probs: Tensor

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        target: Tensor,
        source: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
    ) -> Tuple[Optional[NestedTensor], Output]:
        """Computes either self-attention or cross-attention for the given target and source.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            target: A Tensor of shape [batch, target_length, target_dim].
            source: An optional Tensor of shape [batch, source_length, source_dim].
                If None, uses norm(target) as source (self-attention).
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If `mode` is unsupported.
            NotImplementedError: If `cfg.structure` is not supported.
        """
        cfg = self.config

        def attention_thunk(target: Tensor) -> Tuple[Optional[NestedTensor], Tensor]:
            if mode == ForwardMode.FORWARD:
                atten_state, atten_output = None, self.attention(
                    query=target,
                    key=source,
                    value=source,
                    attention_logit_biases=attention_logit_biases,
                )
            elif mode == ForwardMode.INIT_STATES:
                assert cached_states is not None
                atten_state, atten_output = self.attention.prefill_states(
                    time_step=cached_states["attention"],
                    query=target,
                    attention_logit_biases=attention_logit_biases,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert cached_states is not None
                atten_state, atten_output = self.attention.extend_step(
                    cached_states["attention"],
                    target,
                    attention_logit_biases=attention_logit_biases,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")
            return atten_state, atten_output

        if cfg.structure == "prenorm":
            skip_input = target  # pre-norm: where normalization happens within the residual part.
            norm_target = self.norm(target)
            atten_state, atten_output = attention_thunk(norm_target)
            data = skip_input + self.stochastic_depth(self.dropout(atten_output.data))
        elif cfg.structure == "postnorm":
            # This is the structure used by the original Transformer, BERT, and RoBERTa.
            atten_state, atten_output = attention_thunk(target)
            # Post-norm: norm applied on the sum of input and attention output.
            data = self.norm(target + self.stochastic_depth(self.dropout(atten_output.data)))
        elif cfg.structure == "hybridnorm":
            skip_input = target  # pre-norm: where normalization happens within the residual part.
            norm_target = self.prenorm(target)
            atten_state, atten_output = attention_thunk(norm_target)
            data = skip_input + self.stochastic_depth(
                self.dropout(self.postnorm(atten_output.data))
            )
        else:
            raise NotImplementedError(cfg.structure)
        return dict(attention=atten_state), self.Output(data=data, probs=atten_output.probs)

    def forward(
        self,
        *,
        target: Tensor,
        source: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
    ) -> Output:
        """Computes attention with target as query and source as key and value.

        Args:
            target: a Tensor of shape [batch, target_length, target_dim].
            source: a Tensor of shape [batch, source_length, source_dim].
                If None, uses norm(target) as source (self-attention)
            attention_logit_biases: See ``On attention logit biases`` in the file comments.

        Returns:
            An Output instance, where .data is of the same shape as target and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            NotImplementedError: If cfg.structure is unsupported.
        """
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            target=target,
            source=source,
            attention_logit_biases=attention_logit_biases,
            cached_states=None,
        )
        return output

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            target_batch_size: the batch size of the target to be decoded.
            target_max_len: the sequence length of the target to be decoded.

        Returns:
            The cache as a `NestedTensor` with key and value initialized.
        """
        return dict(
            attention=self.attention.init_states(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def prefill_states(
        self,
        *,
        time_step: NestedTensor,
        target: Tensor,
        attention_logit_biases: Tensor,
    ) -> Tuple[NestedTensor, Output]:
        """Initializes cache for autoregressive cached decoding.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            target: Tensor of shape [batch, target_length, target_dim] corresponding to query vector
                at `time_step` indices. For batch index `i`, only `target[i, :time_step[i], ...]`
                will affect subsequent decoding.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.

        Returns:
            A `NestedTensor` state depending on the `attention` layer implementation.
            An Output instance, where .data is of the same shape as query, .probs is of shape
            [batch, num_heads, target_length, source_length].
        """
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            target=target,
            cached_states=dict(attention=time_step),
            attention_logit_biases=attention_logit_biases,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        target: Tensor,
        *,
        attention_logit_biases: Tensor,
    ) -> Tuple[NestedTensor, Output]:
        """Computes the value vector given the query of the current step.
        This function is used by autoregressive decoding.

        Args:
            cached_states: A `NestedTensor` object containing tensors which are the
                results of previous attentions, and index used for fast decoding. Contains
                "attention" cached states.
            target: Tensor of shape [B, 1, D] corresponding to query vector at index
                time_step.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
                Additionally, target_length is expected to be 1 since this is per time step.
                attention_logit_biases should have already taken care of causal masking for
                decoding, plus other maskings necessary.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where .data is of the same shape as query, .probs is of shape
            [batch, num_heads, 1, source_length].

        Raises:
            NotImplementedError: If cfg.structure is not supported.
        """
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            target=target,
            cached_states=cached_states,
            attention_logit_biases=attention_logit_biases,
        )


def scaled_hidden_dim(scale: float = 4) -> FunctionConfigBase:
    def scale_fn(input_dim: int, *, scale: float) -> int:
        return round(input_dim * scale)

    return config_for_function(scale_fn).set(scale=scale)


class TransformerFeedForwardLayer(BaseLayer):
    """A Transformer feed-forward layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures TransformerFeedForwardLayer."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        # The hidden dim.
        # It should be given either as an integer or a function config that instantiates
        # a dim-to-dim function, e.g., scaled_hidden_dim(4).
        hidden_dim: Required[Union[int, FunctionConfigBase]] = REQUIRED
        # Config for the first linear layer.
        linear1: InstantiableConfig = Linear.default_config().set(
            param_partition_spec=[None, "model"]
        )
        # Config for the second linear layer.
        linear2: InstantiableConfig = Linear.default_config().set(
            param_partition_spec=["model", None]
        )
        norm: InstantiableConfig = LayerNorm.default_config()  # The normalization layer config.

        # The activation function(s).
        #
        # If a single string, the activation applied on the output of linear1.
        #
        # If a tuple of two strings, this layer will contain separate child Linear layers, one for
        # each activation function, according to cfg.linear1 with `hidden_dim` as the output dim.
        # The activation outputs will be multiplied element-wise to produce the inputs for linear2.
        # See the implementation in _linear1_activation().
        # This supports the gated linear activations proposed by Shazeer in
        # https://arxiv.org/abs/2002.05202.
        activation: Union[str, Tuple[str, str]] = "nn.relu"

        # The dropout layer config.
        dropout: InstantiableConfig = Dropout.default_config()

        # The stochastic depth layer config.
        # Pytorch reference:
        # https://github.com/facebookresearch/deit/blob/main/models_v2.py#L59
        # Tensorflow reference:
        # https://github.com/tensorflow/models/blob/master/official/projects/vit/modeling/nn_blocks.py#L103-L119
        stochastic_depth: InstantiableConfig = StochasticDepth.default_config()

        # The inner structure of the layer: prenorm or postnorm.
        # See https://arxiv.org/abs/2002.04745 for background.
        # The structure also support hybridnorm, which uses two norms in the residual branch.
        # hybridnorm: TransformerFeedForwardLayer(x) = x + layernorm_2(feedforward(layernorm_1(x)))
        # Ref: https://github.com/google/praxis/blob/main/praxis/layers/transformers.py#L273
        structure: str = "prenorm"

        # outputs = inputs + residual_weight * x.
        residual_weight: float = 1.0

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: TransformerFeedForwardLayer.Config = self.config
        if cfg.structure in ["prenorm", "postnorm"]:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))
        elif cfg.structure == "hybridnorm":
            self._add_child("prenorm", cfg.norm.set(input_dim=cfg.input_dim))
            self._add_child("postnorm", cfg.norm.set(input_dim=cfg.input_dim))
        else:
            raise NotImplementedError(cfg.structure)

        if isinstance(cfg.hidden_dim, int):
            hidden_dim = cfg.hidden_dim
        else:
            hidden_dim = cfg.hidden_dim.set(input_dim=cfg.input_dim).instantiate()
        if isinstance(cfg.activation, tuple):
            assert len(cfg.activation) == 2, cfg.activation
            # Create a linear1 projection for each activation.
            for i in range(len(cfg.activation)):
                self._add_child(
                    f"linear1_{i}",
                    cfg.linear1.set(input_dim=cfg.input_dim, output_dim=hidden_dim),
                )
        else:
            assert isinstance(cfg.activation, str), cfg.activation
            self._add_child(
                "linear1",
                cfg.linear1.set(input_dim=cfg.input_dim, output_dim=hidden_dim),
            )
        self._add_child(
            "linear2",
            cfg.linear2.set(input_dim=hidden_dim, output_dim=cfg.input_dim),
        )
        if cfg.structure in ["prenorm", "hybridnorm"]:
            self._add_child("dropout1", cfg.dropout)
            self._add_child("dropout2", cfg.dropout)
        elif cfg.structure in ["postnorm"]:
            self._add_child("dropout", cfg.dropout)
        else:
            raise NotImplementedError(cfg.structure)

        self._add_child("stochastic_depth", cfg.stochastic_depth)

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        remat_pt1 = "activation"
        remat_pt2 = "linear2"
        if cfg.structure == "prenorm":
            x = self.norm(inputs)
            x = self._linear1_activation(x)
            x = self._remat_name(x, remat_pt1)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self._remat_name(x, remat_pt2)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x += inputs
        elif cfg.structure == "postnorm":
            x = self._linear1_activation(inputs)
            x = self._remat_name(x, remat_pt1)
            x = self.linear2(x)
            x = self._remat_name(x, remat_pt2)
            x = self.dropout(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x = self.norm(x + inputs)
        elif cfg.structure == "hybridnorm":
            x = self.prenorm(inputs)
            x = self._linear1_activation(x)
            x = self._remat_name(x, remat_pt1)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self._remat_name(x, remat_pt2)
            x = self.postnorm(x)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x += inputs
        else:
            raise NotImplementedError(cfg.structure)
        return x

    def _linear1_activation(self, x: Tensor) -> Tensor:
        cfg = self.config
        if isinstance(cfg.activation, tuple):
            activations = [
                get_activation_fn(activation)(self.children[f"linear1_{i}"](x))
                for i, activation in enumerate(cfg.activation)
            ]
            assert len(activations) == 2, cfg.activation
            return activations[0] * activations[1]
        else:
            x = self.linear1(x)
            return get_activation_fn(cfg.activation)(x)


class TransformerLayer(BaseTransformerLayer):
    """A Transformer layer.

    Unlike torch.nn.TransformerLayer, this allows components to be customized, e.g., replacing
    vanilla attention with relative positional attention from TransformerXL/DeBERTa or replacing
    feed-forward with a mixture-of-expert feed-forward layer.
    """

    @config_class
    class Config(BaseTransformerLayer.Config):
        """Configures TransformerLayer."""

        self_attention: InstantiableConfig = TransformerAttentionLayer.default_config()
        # If not None, the cross-attention layer config.
        cross_attention: Optional[InstantiableConfig] = None
        feed_forward: InstantiableConfig = TransformerFeedForwardLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: TransformerLayer.Config = self.config
        self._add_child(
            "self_attention",
            cfg.self_attention.set(target_dim=cfg.input_dim, source_dim=cfg.input_dim),
        )
        self._add_child("feed_forward", cfg.feed_forward.set(input_dim=cfg.input_dim))
        if cfg.cross_attention is not None:
            self._add_child("cross_attention", cfg.cross_attention.set(target_dim=cfg.input_dim))

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
    ) -> Tuple[Optional[NestedTensor], Tensor]:
        """Computes transformer layer outputs and self/cross-attention probabilities.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            self_attention_logit_biases: An optional Tensor representing the self-attention biases.
            cross_attention_data: An optional Tensor of shape [batch, source_length, source_dim].
            cross_attention_logit_biases: An optional Tensor representing the cross-attention
                biases.
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as `data`, .self_attention_probs is
            of shape [batch, num_heads, target_length, target_length], and .cross_attention_probs is
            of shape [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If `mode` is unsupported.
        """
        self.vlog(3, "transformer.input=%s", data.sum())
        if mode == ForwardMode.FORWARD:
            self_atten_state, self_atten_outputs = None, self.self_attention(
                target=data, attention_logit_biases=self_attention_logit_biases
            )
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            self_atten_state, self_atten_outputs = self.self_attention.prefill_states(
                time_step=cached_states["self_attention"],
                target=data,
                attention_logit_biases=self_attention_logit_biases,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            self_atten_state, self_atten_outputs = self.self_attention.extend_step(
                cached_states=cached_states["self_attention"],
                target=data,
                attention_logit_biases=self_attention_logit_biases,
            )
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        data = self_atten_outputs.data
        self.vlog(3, "self_attention.output=%s", data.sum())
        if cross_attention_data is not None:
            cross_atten_outputs = self.cross_attention(
                target=data,
                source=cross_attention_data,
                attention_logit_biases=cross_attention_logit_biases,
            )
            data = cross_atten_outputs.data
            cross_attention_probs = cross_atten_outputs.probs
        else:
            cross_attention_probs = None
        data = self.feed_forward(data)
        self.vlog(3, "transformer.output=%s", data.sum())
        return dict(self_attention=self_atten_state), BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=self_atten_outputs.probs,
            cross_attention_probs=cross_attention_probs,
        )

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> BaseTransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            cached_states=None,
        )
        return output

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> NestedTensor:
        return dict(
            self_attention=self.self_attention.init_states(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=dict(self_attention=time_step),
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            cached_states=cached_states,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )


def _next_power_of_two(n: float) -> int:
    if n <= 1:
        return 2
    return 1 << int(math.log2(n - 1)) + 1


class BottleNeckAdapterTransformerLayer(BaseTransformerLayer):
    """TransformerLayer with bottleneck adaptor for fine-tuning.
    Figure 3(a) in https://arxiv.org/pdf/2110.04366.pdf
    """

    @config_class
    class Config(BaseTransformerLayer.Config):
        """Configures BottleNeckAdapterTransformerLayer."""

        # The transformer layer to which an adapter will be added.
        layer: BaseTransformerLayer.Config = TransformerLayer.default_config()

        # The adapter, which in this case is a bottleneck layer composed of
        # a downward and an upward projection.
        adapter: TransformerFeedForwardLayer.Config = TransformerFeedForwardLayer.default_config()

        # The ratio by which the input dimension will be
        # reduced in the downward projection in the adapter.
        bottleneck_ratio: float = 0.5

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("layer", cfg.layer)
        self._add_child(
            "adapter",
            cfg.adapter.set(
                input_dim=cfg.layer.input_dim,
                hidden_dim=_next_power_of_two(cfg.layer.input_dim * cfg.bottleneck_ratio),
                structure="postnorm",
            ),
        )

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
    ) -> Tuple[Optional[NestedTensor], Tensor]:
        """Computes transformer layer outputs and self/cross-attention probabilities.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            self_attention_logit_biases: An optional Tensor representing the self-attention biases.
            cross_attention_data: An optional Tensor of shape [batch, source_length, source_dim].
            cross_attention_logit_biases: An optional Tensor representing the cross-attention
                biases.
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as `data`, .self_attention_probs is
            of shape [batch, num_heads, target_length, target_length], and .cross_attention_probs is
            of shape [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If `mode` is unsupported.
        """
        self.vlog(3, "transformer.input=%s", data.sum())
        if mode == ForwardMode.FORWARD:
            output = self.layer.forward(
                data=data,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
            )
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            cached_states, output = self.layer.prefill_states(
                time_step=cached_states["layer"],
                data=data,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            cached_states, output = self.layer.extend_step(
                cached_states=cached_states,
                data=data,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
            )
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        self_attention_probs = output.self_attention_probs
        cross_attention_probs = output.cross_attention_probs
        skip_input = output.data
        data = self.adapter(output.data)
        data += skip_input
        self.vlog(3, "adapted_transformer.output=%s", data.sum())
        return cached_states, BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=self_attention_probs,
            cross_attention_probs=cross_attention_probs,
        )

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> BaseTransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            cached_states=None,
        )
        return output

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> NestedTensor:
        return dict(
            layer=self.layer.init_states(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=dict(layer=time_step),
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            cached_states=cached_states,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )


def set_double_shard_weights_config(
    cfg: TransformerLayer.Config,
    *,
    batch_axis_names: Union[str, Sequence[str]] = "data",
    fsdp_axis_names: Union[str, Sequence[str]] = "data",
    tp_axis_names: Union[str, Sequence[str]] = "model",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    TODO(tom_gunter): Replace default batch/fsdp axis names with "fsdp".

    Args:
        cfg: Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
    """
    # pytype: disable=attribute-error
    ff_layer = cfg.feed_forward
    # Shard weights.
    ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
    ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
    # Encourage the right activation sharding.
    ff_layer.linear1.output_partition_spec = (batch_axis_names, None, tp_axis_names)
    ff_layer.linear2.output_partition_spec = (batch_axis_names, None, tp_axis_names)

    def set_attn_partition_specs(attn_layer: MultiheadAttention.Config):
        # Shard weights.
        attn_layer.input_linear.layer.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)
        attn_layer.output_linear.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)

    set_attn_partition_specs(cfg.self_attention.attention)
    if cfg.cross_attention is not None:
        set_attn_partition_specs(cfg.cross_attention.attention)
    # pytype: enable=attribute-error


class BaseStackedTransformerLayer(BaseTransformerLayer):
    """The common interface of all stacked transformer layer classes.

    Note that BaseStackedTransformerLayer is a subclass of BaseTransformerLayer and therefore
    can be used where a BaseTransformerLayer is expected.
    """

    @config_class
    class Config(BaseTransformerLayer.Config):
        """Configures BaseStackedTransformerLayer."""

        # The number of layers in the stack.
        num_layers: Required[int] = REQUIRED
        # Config for each layer in the stack.
        # The layer must be a subclass of BaseTransformerLayer.
        layer: BaseTransformerLayer.Config = TransformerLayer.default_config()
        peak_stochastic_depth_rate: Optional[float] = None


class StackedTransformerLayer(BaseStackedTransformerLayer):
    """A simple implementation of BaseStackedTransformerLayer."""

    @config_class
    class Config(BaseStackedTransformerLayer.Config):
        """Configures StackedTransformerLayer."""

        # If `layer` is a Config, it will be stacked cfg.num_layers times. If `layer` is a
        # sequence of Configs, the sequence length should match cfg.num_layers.
        layer: Union[
            BaseTransformerLayer.Config, Sequence[BaseTransformerLayer.Config]
        ] = TransformerLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if isinstance(cfg.layer, Sequence):
            layer_cfgs = cfg.layer
            if len(layer_cfgs) != cfg.num_layers:
                raise ValueError(
                    f"Number of layer configs ({len(layer_cfgs)}) must match "
                    f"cfg.num_layers ({cfg.num_layers})."
                )
        else:
            layer_cfgs = [cfg.layer] * cfg.num_layers
        self._layers = []
        for i, layer_cfg in enumerate(layer_cfgs):
            if layer_cfg.input_dim is not REQUIRED:
                raise ValueError(
                    f"Do not set Config.layer.input_dim. Set Config.input_dim instead: {layer_cfg}"
                )
            layer_cfg = layer_cfg.clone(input_dim=cfg.input_dim)
            if cfg.peak_stochastic_depth_rate:
                layer_rate = get_stochastic_depth_linear_rate(
                    cfg.peak_stochastic_depth_rate,
                    stage_order=i + 1,
                    num_stages=cfg.num_layers,
                )
                layer_cfg.self_attention.stochastic_depth.rate = layer_rate
                layer_cfg.feed_forward.stochastic_depth.rate = layer_rate
            self._layers.append(self._add_child(f"layer{i}", layer_cfg))

    def initialize_parameters_recursively(
        self, prng_key: jax.random.KeyArray, *, prebuilt: Optional[NestedTensor] = None
    ) -> NestedTensor:
        cfg = self.config  # type: StackedTransformerLayer.Config
        prng_key = split_prng_key(prng_key, cfg.num_layers)
        state = {}
        for i in range(cfg.num_layers):
            layer = self._layers[i]
            key = jax.tree_util.tree_map(lambda x, index=i: x[index], prng_key.keys)
            state[layer.name] = layer.initialize_parameters_recursively(
                key, prebuilt=get_or_none(prebuilt, layer.name)
            )
        return state

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
        **layer_kwargs,
    ) -> Tuple[List[Optional[NestedTensor]], TransformerLayer.Output]:
        """Computes transformer stack outputs.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            self_attention_logit_biases: See ``On attention logit biases`` in the file comments.
            cross_attention_data: An optional Tensor of shape [batch, source_length, source_dim].
            cross_attention_logit_biases: See ``On attention logit biases`` in the file comments.
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If `mode` is unsupported.
        """
        all_layer_outputs = []
        all_layer_states = []
        for i, layer in enumerate(self._layers):
            if mode == ForwardMode.FORWARD:
                layer_states, layer_outputs = None, layer(
                    data,
                    self_attention_logit_biases=self_attention_logit_biases,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                    **layer_kwargs,
                )
            elif mode == ForwardMode.INIT_STATES:
                assert cached_states is not None
                layer_states, layer_outputs = layer.prefill_states(
                    time_step=cached_states,
                    data=data,
                    self_attention_logit_biases=self_attention_logit_biases,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                    **layer_kwargs,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert cached_states is not None
                layer_states, layer_outputs = layer.extend_step(
                    cached_states=cached_states[i],
                    data=data,
                    self_attention_logit_biases=self_attention_logit_biases,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                    **layer_kwargs,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")
            all_layer_outputs.append(layer_outputs)
            all_layer_states.append(layer_states)
            data = layer_outputs.data
        aux_outputs = {}
        for field in TransformerLayer.Output._fields:
            if field == "data":
                continue
            values = [getattr(output, field) for output in all_layer_outputs]
            if None in values:
                assert all(v is None for v in values), f"{field}: {values}"
                aux_outputs[field] = None
            else:
                aux_outputs[field] = jnp.stack(values, axis=0)
        return all_layer_states, TransformerLayer.Output(data=data, **aux_outputs)

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        **layer_kwargs,
    ) -> TransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            cached_states=None,
            **layer_kwargs,
        )
        return output

    def init_states(self, *args: Any, **kwargs: Any) -> NestedTensor:
        # TODO(sneha): any better ds?
        return [layer.init_states(*args, **kwargs) for layer in self._layers]

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        **layer_kwargs,
    ) -> Tuple[List[NestedTensor], TransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=time_step,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            **layer_kwargs,
        )

    def extend_step(
        self,
        cached_states: List[NestedTensor],
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        **layer_kwargs,
    ) -> Tuple[List[NestedTensor], TransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            cached_states=cached_states,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            **layer_kwargs,
        )


class _TransformerRepeat(Repeat):
    """A Repeat layer with layer=TransformerLayer."""

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Tensor,
        cached_states: Optional[NestedTensor] = None,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[Optional[NestedTensor], TransformerLayer.Output]:
        """Computes transformer stack outputs.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            self_attention_logit_biases: See ``On attention logit biases`` in the file comments.
            cross_attention_data: An optional Tensor of shape [batch, source_length, source_dim].
            cross_attention_logit_biases: See ``On attention logit biases`` in the file comments.
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If `mode` is unsupported.
        """

        def layer_fn(carry, x_i):
            if mode == ForwardMode.FORWARD:
                layer_states, layer_outputs = None, self.layer(
                    carry,
                    self_attention_logit_biases=self_attention_logit_biases,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                )
            elif mode == ForwardMode.INIT_STATES:
                assert x_i is not None
                layer_states, layer_outputs = self.layer.prefill_states(
                    time_step=x_i,
                    data=carry,
                    self_attention_logit_biases=self_attention_logit_biases,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert x_i is not None
                layer_states, layer_outputs = self.layer.extend_step(
                    x_i,
                    carry,
                    self_attention_logit_biases=self_attention_logit_biases,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")

            ys = {k: v for k, v in layer_outputs._asdict().items() if k != "data"}
            if layer_states is not None:
                # Vectorize over scan axis.
                ys["cached_states"] = jax.tree_map(
                    VDict, layer_states, is_leaf=lambda v: isinstance(v, dict)
                )
            return layer_outputs.data, ys

        repeat_outputs: Repeat.Output = self._run(layer_fn, carry=data, xs=cached_states)
        ys = repeat_outputs.ys
        updated_states = ys.pop("cached_states", None)
        return updated_states, TransformerLayer.Output(data=repeat_outputs.carry, **ys)

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> TransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            cached_states=None,
        )
        return output

    def init_states(self, *args: Any, **kwargs: Any) -> NestedTensor:
        def layer_fn(_):
            return jax.tree_map(
                VDict,
                self.layer.init_states(*args, **kwargs),
                is_leaf=lambda v: isinstance(v, dict),
            )

        cfg = self.config
        return jax.vmap(layer_fn)(jnp.empty(cfg.num_layers))

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, TransformerLayer.Output]:
        cfg = self.config
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            data=data,
            cached_states=jnp.tile(time_step, [cfg.num_layers, 1]),
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[NestedTensor, TransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            data=data,
            cached_states=cached_states,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )


class RepeatedTransformerLayer(BaseStackedTransformerLayer):
    """An implementation of BaseStackedTransformerLayer with a scan loop.

    Compared with StackedTransformerLayer, the size of the XLA program for RepeatedTransformerLayer
    does not grow proportional to the number of layers. In practice, this significantly reduces
    XLA compilation overhead of large models with many layers.
    """

    @config_class
    class Config(BaseStackedTransformerLayer.Config):
        """Configures RepeatedTransformerLayer."""

        repeat: Repeat.Config = _TransformerRepeat.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config  # type: RepeatedTransformerLayer.Config
        repeat_cfg = cfg.repeat.set(
            layer=cfg.layer.set(input_dim=cfg.input_dim),
            num_layers=cfg.num_layers,
        )
        self._add_child("repeat", repeat_cfg)

    def initialize_parameters_recursively(
        self, prng_key: jax.random.KeyArray, *, prebuilt: Optional[NestedTensor] = None
    ) -> NestedTensor:
        # We need to call self.repeat.initialize_parameters_recursively() with the same prng_key
        # to ensure initialization parity with StackedTransformerLayer.
        return dict(
            repeat=self.repeat.initialize_parameters_recursively(
                prng_key, prebuilt=get_or_none(prebuilt, "repeat")
            )
        )

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> TransformerLayer.Output:
        return self.repeat(
            data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )

    def init_states(self, *args: Any, **kwargs: Any) -> NestedTensor:
        return self.repeat.init_states(*args, **kwargs)

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[List[NestedTensor], TransformerLayer.Output]:
        return self.repeat.prefill_states(
            time_step=time_step,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[List[NestedTensor], TransformerLayer.Output]:
        return self.repeat.extend_step(
            cached_states=cached_states,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )


class _TransformerPipeline(Pipeline):
    """Transformer pipeline layer."""

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> TransformerLayer.Output:
        carry_in = dict(data=data)
        # Even though attention logit biases do not change across layers, we
        # include them in the carry so that they are aligned with the microbatches.
        if self_attention_logit_biases is not None:
            carry_in["self_attention_logit_biases"] = self_attention_logit_biases
        if cross_attention_data is not None:
            carry_in["cross_attention_data"] = cross_attention_data
        if cross_attention_logit_biases is not None:
            carry_in["cross_attention_logit_biases"] = cross_attention_logit_biases

        carry_in = self._to_microbatches(carry_in)
        self.vlog(3, "carry_in=%s", shapes(carry_in))

        def layer_fn(carry, _):
            layer_outputs: TransformerLayer.Output = self.layer(**carry)
            carry.pop("data")
            return dict(**carry, data=layer_outputs.data), {
                k: v for k, v in layer_outputs._asdict().items() if k != "data"
            }

        pipeline_outputs: Pipeline.Output = self._run(layer_fn, carry_in)
        carry_out = self._from_microbatches(pipeline_outputs.carry["data"])

        ys = pipeline_outputs.ys
        self.vlog(3, "ys=%s", shapes(ys))
        return TransformerLayer.Output(data=carry_out, **ys)


class PipelinedTransformerLayer(BaseStackedTransformerLayer):
    """An implementation of BaseStackedTransformerLayer with pipeline model parallelism."""

    @config_class
    class Config(BaseStackedTransformerLayer.Config):
        """Configures PipelinedTransformerLayer."""

        # The number of pipeline stages. Must evenly divide `num_layers`.
        num_stages: Required[int] = REQUIRED
        # The number of pipeline microbatches. Must evenly divide batch size.
        num_microbatches: Required[int] = REQUIRED
        # Config for each stage in the pipeline.
        stage: BaseLayer.Config = StackedTransformerLayer.default_config().set(layer=None)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config  # type: PipelinedTransformerLayer.Config
        if cfg.num_layers % cfg.num_stages != 0:
            raise ValueError(f"num_stages {cfg.num_stages} must divide num_layers {cfg.num_layers}")
        num_layers_per_stage = cfg.num_layers // cfg.num_stages
        stage_cfg = cfg.stage.set(
            input_dim=cfg.input_dim, layer=cfg.layer, num_layers=num_layers_per_stage
        )
        pipeline_cfg = _TransformerPipeline.default_config().set(
            layer=stage_cfg, num_layers=cfg.num_stages, num_microbatches=cfg.num_microbatches
        )
        self._add_child("pipeline", pipeline_cfg)

    def initialize_parameters_recursively(
        self, prng_key: jax.random.KeyArray, *, prebuilt: Optional[NestedTensor] = None
    ) -> NestedTensor:
        cfg = self.config  # type: PipelinedTransformerLayer.Config
        # We pre-split all num_layers keys to ensure initialization parity with
        # StackedTransformerLayer.
        prng_key = split_prng_key(prng_key, (cfg.num_stages, cfg.num_layers // cfg.num_stages))
        return dict(
            pipeline=self.pipeline.initialize_parameters_recursively(
                prng_key, prebuilt=get_or_none(prebuilt, "pipeline")
            )
        )

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> TransformerLayer.Output:
        return self.pipeline(
            data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )

    # TODO(sneha): extend_step


def build_remat_spec(
    stack_cfg: Union[
        BaseStackedTransformerLayer.Config, "RepeatedConformerLayer.Config"  # type: ignore
    ]
) -> Optional[RematSpec]:
    """Configures how the Transformer or Conformer stack will save the linearization points.

    We try to save activations from the forward pass that are inefficient to recompute on the
    backward pass. We choose the linearization points in the MultiHeadAttention layer, as that
    demonstrated (empirically) the best throughput, allowing us to train with a batch size of 16 on
    gpt2-10b with adamw and full sharding across 4 TPU v4 chips and a RepeatedTransformerLayer,
    with 1.8x the step time of a stacked layer with a batch size of 8 and the same sharding config.

    For conformer model, we start from the same remat policy as language models.
    TODO(zhiyunlu): investigate Conformer model's memory/step-time tradeoffs. Possibly we
    need to save points in the LConv module.

    Args:
        stack_cfg: A transformer config.

    Returns:
        None (if no rematerialization is needed) or a RematSpec.
    """
    # TODO(markblee): Switch to using isinstance everywhere.
    if stack_cfg.klass is PipelinedTransformerLayer:
        return None
    attention_name = stack_cfg.layer.self_attention.attention.klass.__name__
    return RematSpec(
        prevent_cse=stack_cfg.klass is StackedTransformerLayer,
        # If we are running inside a jax.lax.scan (Repeated/Pipelined transformers
        # or Repeated Conformers) we can enable common subexpression elimination optimizations.
        policy=config_for_function(jax_remat_policies.save_only_these_names).set(
            names_which_can_be_saved=[
                f"{attention_name}.{el}"
                for el in ["q_proj", "k_proj", "v_proj", "context", "o_proj"]
            ]
        ),
    )


class AttentionLogitBiasLayer(BaseLayer):
    """Base attention logit bias layer.

    The attention logit bias layer should have input_ids as input.
    """

    def forward(self, *, segment_ids: Tensor, positions: Tensor) -> Tensor:
        """Produces attention logit biases.

        Args:
            segment_ids: An integer Tensor of shape [batch_size, seq_len] with values in
                [0, num_segments). Tokens are only allowed to attend to other tokens within the same
                segment. segment_ids == 0 represents paddings.
            positions: An Tensor of broadcastable shape to `input_ids` with values in [0, seq_len).
                This can be used to produce biases for packed inputs.

        Returns:
            A float attention logit biases of shape [batch_size, 1, seq_len, seq_len] or
                [batch_size, num_heads, seq_len, seq_len].
            Output[b,i,j] is -inf iff attention is disabled with query=input[b, i] and
            key=input[b, j].
        """
        raise NotImplementedError(type(self))


def compute_padding_biases(input_ids: Tensor, *, pad_token_id: Optional[int]) -> Tensor:
    """Compute the logits bias to disable attention to/from paddings.

    Args:
        input_ids: A Tensor of shape [batch_size, seq_len].
        pad_token_id: An int representing the padded token ID or None.

    Returns:
        A float logit biases of shape [batch_size, 1, seq_len, seq_len].
    """
    if pad_token_id is None:
        batch_size, seq_len = input_ids.shape
        return jnp.zeros([batch_size, 1, seq_len, seq_len])
    padding_bias = (input_ids == pad_token_id) * NEG_INF
    return padding_bias[:, None, None, :] + padding_bias[:, None, :, None]


class CausalAttentionLogitBiasLayer(AttentionLogitBiasLayer):
    """Causal attention logit bias layer."""

    def forward(self, *, segment_ids: Tensor, positions: Tensor) -> Tensor:
        """Refer to AttentionLogitBiasLayer.forward for docstring."""
        # Note: padding tokens are not explicitly masked.
        causal_bias = (positions[:, None, :, None] < positions[:, None, None, :]) * NEG_INF
        return apply_attention_logit_biases(
            causal_bias, make_segment_mask(source_segments=segment_ids, target_segments=segment_ids)
        )


class FullAttentionLogitBiasLayer(AttentionLogitBiasLayer):
    """Full attention logit bias layer."""

    def forward(self, *, segment_ids: Tensor, positions: Tensor) -> Tensor:
        """Refer to AttentionLogitBiasLayer.forward for docstring."""
        del positions
        return make_segment_mask(source_segments=segment_ids, target_segments=segment_ids)


def alibi_get_slopes(num_heads: int) -> List:
    """Get the slopes for different attention heads defined in ALiBi paper.

    This is a direct copy from ALiBi codebase.
    Ref:
    https://github.com/ofirpress/attention_with_linear_biases/tree/3b7c2eca/fairseq/models/transformer.py#L742-L752

    Args:
        num_heads: An integer for the number of attention heads.

    Returns:
        A tensor of slopes with shape of [num_heads]. Each value represents
        a slope for one attention head.
    """

    def get_slopes_power_of_2(n: int) -> List:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + alibi_get_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
        )


class ALiBiAttentionLogitBiasLayer(CausalAttentionLogitBiasLayer):
    """attention logit bias layer in ALiBi.

    Ref: https://github.com/ofirpress/attention_with_linear_biases/tree/3b7c2eca
    """

    @config_class
    class Config(CausalAttentionLogitBiasLayer.Config):
        """Configures ALiBiAttentionLogitBiasLayer."""

        num_heads: Required[int] = REQUIRED

    def forward(self, *, segment_ids: Tensor, positions: Tensor) -> Tensor:
        """Produces an attention logit biases of shape [batch_size, num_heads, seq_len, seq_len].

        The ALiBi bias is defined as below:
        1. Create a lower triangle matrix with the value of:
            bias = [-(i-1), ..., -2, -1, 0] * slopes
        2. Apply the casual biases.
            bias = apply_apply_attention_logit_biases(bias, causal_bias)

        Refer to AttentionLogitBiasLayer.forward for docstring.
        """
        cfg = self.config
        slopes = jnp.asarray(alibi_get_slopes(cfg.num_heads))
        # Create the lower triangle matrix w/ value [-(i-1), ..., -2, -1, 0] for each segment.
        alibi_bias = jnp.expand_dims(positions, [1]) - jnp.expand_dims(positions, [2])
        # Add head dim.
        alibi_bias = jnp.expand_dims(alibi_bias, [1])
        # Multiply w/ the slopes.
        alibi_bias = alibi_bias * jnp.expand_dims(slopes, [0, 2, 3])
        bias = super().forward(segment_ids=segment_ids, positions=positions)
        # Combine the biases.
        return apply_attention_logit_biases(alibi_bias, bias)


class SymmetricALiBiAttentionLogitBiasLayer(FullAttentionLogitBiasLayer):
    """Symmetric full attention version of ALiBiAttentionLogitBiasLayer.

    Main implementation differences between this one and `ALiBiAttentionLogitBiasLayer` (above):
        1. Muliplies alibi slopes by -1.
        2. Computes absolute value of relative positions.
        3. Multiplies results of steps 1 and 2 to get symmetric bias matrix.

    Originally proposed here by an author of the ALiBi paper:
    https://github.com/ofirpress/attention_with_linear_biases/issues/5
    """

    @config_class
    class Config(FullAttentionLogitBiasLayer.Config):
        """Configures SymmetricALiBiAttentionLogitBiasLayer."""

        num_heads: Required[int] = REQUIRED

    def forward(self, *, segment_ids: Tensor, positions: Tensor) -> Tensor:
        cfg = self.config

        slopes = -1 * jnp.asarray(alibi_get_slopes(cfg.num_heads))

        # Create the lower triangle matrix w/ value [-(i-1), ..., -2, -1, 0] for each segment.
        alibi_bias = jnp.abs(positions[:, jnp.newaxis, :] - positions[:, :, jnp.newaxis])

        # Add head dim.
        alibi_bias = alibi_bias[:, jnp.newaxis, :, :]

        # Multiply w/ the slopes.
        alibi_bias = alibi_bias * jnp.expand_dims(slopes, [0, 2, 3])

        bias = super().forward(segment_ids=segment_ids, positions=positions)
        # Combine the biases.
        return apply_attention_logit_biases(alibi_bias, bias)

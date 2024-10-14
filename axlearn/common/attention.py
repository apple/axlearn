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
* A biases tensor can have one of the following shapes:
  * [target_length, source_length]
  * [batch, target_length, source_length]
  * [batch, num_heads, target_length, source_length].
* Each value represents a bias to be added to the attention logits
  (therefore a -inf represents a disconnected position pair).
* biases=None represents an all-zero tensor, i.e., all position pairs are connected.

On `segment_ids`:
* A tensor of shape [batch, target_length] with values in [0, num_segments].
* Tokens are only allowed to attend to other tokens within the same segment.
* segment_ids == 0 represents paddings.
* None represents an all-one tensor, i.e. all positions are in the same segment.

"""

# pylint: disable=abstract-method,too-many-lines
import enum
import functools
import math
from collections.abc import Sequence
from enum import Enum, unique
from typing import Any, Callable, Literal, NamedTuple, Optional, Protocol, Union

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
    ConfigOr,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
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
from axlearn.common.quantized_dot_general.layers import DenseGeneralBaseLayer
from axlearn.common.repeat import Repeat
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    PartitionSpec,
    Tensor,
    VDict,
    check_numerics,
    flatten_items,
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


class KVState(NamedTuple):
    """Represents key/value projections, of shape [batch, source_length, num_kv_heads, head_dim]."""

    k_proj: Tensor
    v_proj: Tensor


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
        """BaseTransformerLayer output.

        Fields:
            data: [batch, target_length, input_dim]. The layer output. Always present.

            self_attention_probs: The attention probabilities returned by the self-attention layer.
                Shape: [..., target_length, target_length].

                self_attention_probs[..., i, j] represents self-attention probability on
                input data[..., j, :] when computing output data[..., i, :].
                self_attention_probs.sum(axis=-1) equals to all 1's.

                Present if "self_attention_probs" is in `return_aux`.

            self_attention_kv_state: The KV state used in self-attention.
                Present if "self_attention_kv_state" is in `return_aux`.

            cross_attention_probs: The attention probabilities returned by the cross-attention
                layer. Shape: [..., target_length, source_length].

                If not None, cross_attention_probs[..., i, j] represents attention probability on
                cross_attention_data[..., j, :] when computing output data[..., i, :].
                cross_attention_probs.sum(axis=-1) equals to all 1's.

                Present if "cross_attention_probs" is in `return_aux`.
        """

        data: Tensor
        self_attention_probs: Optional[Tensor] = None
        self_attention_kv_state: Optional[KVState] = None
        cross_attention_probs: Optional[Tensor] = None

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_kv_state: Optional[KVState] = None,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        target_segment_ids: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> Output:
        """Computes transformer layer outputs given full-sequence inputs.

        For incremental computation, use init_states() and extend_step().

        See comments at the beginning of this file for semantics of *_attention_logit_biases.

        Args:
            data: A Tensor of shape [batch, target_length, input_dim].
            self_attention_kv_state: An optional KVState used for self-attention.
            self_attention_logit_biases: An optional Tensor representing the self-attention biases.
            cross_attention_data: An optional Tensor representing cross-attention data of shape
                [source_batch, source_length, source_dim].
            cross_attention_logit_biases: An optional Tensor representing the cross-attention
                biases.
            target_segment_ids: See ``segment_ids`` in the file comments.
            return_aux: A set of auxiliary output fields to return. Each element must be an
                optional field of `Output`, e.g.,
                `return_aux = {"self_attention_probs", "self_attention_kv_state"}` means that
                `Output.{self_attention_probs, self_attention_kv_state}` will be populated.

        Returns:
            BaseTransformerLayer.Output.
        """
        raise NotImplementedError(type(self))

    def init_states(
        self,
        *,
        target_batch_size: int,
        target_max_len: int,
        self_attention_kv_state: Optional[KVState] = None,
    ) -> NestedTensor:
        """Initializes cached states for incremental computation.

        Args:
            target_batch_size: The batch size for target sequences.
            target_max_len: The maximum number of tokens in a target sequence.
            self_attention_kv_state: An optional KVState used for self-attention.

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
        self_attention_kv_state: Optional[KVState] = None,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, Output]:
        """Initializes cached states for incremental computation.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            data: A Tensor of shape [batch, target_length, input_dim]. For batch index `i`, only
                `data[i, :time_step[i], ...]` will affect subsequent decoding.
            self_attention_kv_state: An optional KVState used for self-attention.
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
        self_attention_kv_state: Optional[KVState] = None,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, Output]:
        """Computes incremental outputs.

        Args:
            cached_states: A NestedTensor returned by `init_states()` or a previous invocation of
                `extend_step()`.
            data: A Tensor of shape [target_batch_size, target_step_length, input_dim], where
                `target_step_length` is usually 1. For self-attention, `data` will be used as the
                `query` sequence and will be appended to key and value sequences.
            self_attention_kv_state: An optional KVState used for self-attention.
            self_attention_logit_biases: An optional Tensor of shape
                [..., target_step_length, target_max_len], where `target_step_length` must match
                the shape of `data` and `target_max_len` must match the value given for
                `init_states()`.
            cross_attention_data: An optional Tensor of shape [..., source_length, source_dim].
            cross_attention_logit_biases: An optional Tensor of shape
                [..., target_step_length, source_length], where `target_step_length` must match
                the shape of `data`.

        Returns:
            (updated_cached_states, output), where:
            `updated_cached_states` represents the new cached states incorporating `data`;
            `output` represents the output for the given input data. `output.data` will have the
            same shape as the input data.
        """
        raise NotImplementedError(type(self))


def make_causal_biases(seq_len: int) -> Tensor:
    """Generates attention logit biases for causal masking.

    Args:
        seq_len: Sequence length.

    Returns:
        A float tensor of shape [seq_len, seq_len] where the value at [i, j] = -inf if i < j,
        0 otherwise.
    """
    # TODO(sneha): support batching
    return _bool_to_bias(causal_mask(jnp.arange(seq_len)[:, None], jnp.arange(seq_len)[None, :]))


def make_sliding_window_causal_biases(seq_len: int, sliding_window_size: int) -> Tensor:
    """Generates attention logit biases for sliding window attention.

    Args:
        seq_len: Sequence length.

    Returns:
        A float tensor of shape [seq_len, seq_len] where the value at [i, j] = -inf
        if i - j > sliding_window_size or i < j, 0 otherwise.
    """
    mask_fn = sliding_window_causal_mask(sliding_window_size)
    return _bool_to_bias(mask_fn(jnp.arange(seq_len)[:, None], jnp.arange(seq_len)[None, :]))


def _bool_to_bias(mask: Tensor) -> Tensor:
    """Converts a bool mask tensor to a bias mask tensor.

    Maps:
    0 -> -NEG_INF
    1 -> 0.
    """
    return (~mask) * NEG_INF


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

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=[1] + list(cfg.shape) + [cfg.dim],
                mesh_axes=cfg.param_partition_spec,
            )
        )

    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if not name.endswith("weight"):
            return None
        if len(parameter_spec.shape) != 3:
            raise NotImplementedError(
                "_compute_fan_axes requires weight parameters to have exactly 3 axes "
                f"shape({name}) = {parameter_spec.shape}"
            )
        return FanAxes(batch_axis=0, in_axis=1, out_axis=2)

    def embeddings(self) -> Tensor:
        """Returns weights of shape cfg.shape + [dim]."""
        return self.parameters["weight"].squeeze(0)

    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: An integer tensor with arbitrary shape [...].

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
        positions: An integer tensor of any shape [...]. Each value represents an
            absolute or relative position.
        dim: the embedding dimension. Must be divisible by 2.
        min_timescale: The minimum timescale (used for channel 0 and dim // 2).
        max_timescale: The maximum timescale (used for channel dim // 2 - 1 and dim - 1).

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


class BaseMultiheadLinear(DenseGeneralBaseLayer):
    """The linear layer used for multi-head attention.

    It uses einsum for efficient computation on TPU to avoid reshaping.
    """

    @config_class
    class Config(DenseGeneralBaseLayer.Config):
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

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
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

    @property
    def _einsum_expr(self):
        raise NotImplementedError(type(self))

    def forward(self, inputs: Tensor) -> Tensor:
        params = self.parameters
        outputs = self.einsum_maybe_quantized(
            self._einsum_expr, activation=inputs, kernel=params["weight"]
        )
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
        logits: A float Tensor.
        attention_logit_biases: A float Tensor. If None, assume all zeros.

    Returns:
        logits + attention_logit_biases, in logits.dtype.
    """
    if attention_logit_biases is None:
        return logits
    return logits + attention_logit_biases.astype(logits.dtype)


def softmax_with_biases(logits: Tensor, attention_logit_biases: Optional[Tensor] = None) -> Tensor:
    """Computes softmax with optional masking.

    Args:
        logits: A Tensor of any shape.
        attention_logit_biases: A Tensor that is broadcastable with logits.
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


def sigmoid_with_biases(
    logits: Tensor,
    attention_logit_biases: Optional[Tensor] = None,
) -> Tensor:
    """Computes sigmoid with optional masking.

    Args:
        logits: A Tensor of any shape.
        attention_logit_biases: A Tensor that is broadcastable with logits.
            See ``On attention logit biases`` in the file comments.

    Returns:
        A Tensor of same shape and dtype as logits.
    """
    check_numerics(logits)
    logits = apply_attention_logit_biases(logits, attention_logit_biases)
    # Avoid computing sigmoid in 16-bit floats.
    logits_dtype = logits.dtype
    if logits_dtype in (jnp.bfloat16, jnp.float16):
        logits = logits.astype(jnp.float32)
    probs = jax.nn.sigmoid(logits)
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
        # Needs to match the forward dtype for Repeated layers. If None, infer as BaseLayer.dtype().
        cache_dtype: Optional[jnp.dtype] = None

    class Output(NamedTuple):
        # [batch, target_length, num_heads, per_head_dim].
        query: Tensor
        # [batch, source_length, num_heads, per_head_dim].
        key: Tensor
        # [batch, source_length, num_heads, per_head_dim].
        value: Tensor

    @property
    def num_kv_heads(self):
        raise NotImplementedError(type(self))

    def init_states(
        self,
        *,
        target_batch_size: int,
        target_max_len: int,
        kv_state: Optional[KVState] = None,
    ) -> NestedTensor:
        cfg = self.config
        # Default to base layer dtype for initialization if cache_dtype is None.
        dtype = cfg.cache_dtype or self.dtype()
        assert dtype is not None

        # Following T5X, we cache key, value as [batch, num_heads, head_dim, seq_len] to take
        # advantage of TPU optimizations (see `extend_step`).
        # Reference:
        # https://github.com/google-research/t5x/blob/4d94d8bf41230d492e15e255c9888b5bfd9a5ee8/t5x/examples/t5/layers.py#L215
        cache = dict(time_step=jnp.zeros(target_batch_size, dtype=jnp.int32))
        # If `kv_state` is provided externally, we do not have to maintain key/value in cache.
        if kv_state is None:
            cache.update(
                key=jnp.zeros(
                    shape=(target_batch_size, self.num_kv_heads, cfg.per_head_dim, target_max_len),
                    dtype=dtype,
                ),
                value=jnp.zeros(
                    shape=(target_batch_size, self.num_kv_heads, cfg.per_head_dim, target_max_len),
                    dtype=dtype,
                ),
            )
        # TODO(sneha,markblee): Add sharding annotations for all elements in the cache.
        return cache

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        kv_state: Optional[KVState] = None,
        time_step: Optional[Tensor] = None,
    ) -> Output:
        """Computes per-head query, key, and value for the input query, key, value.

        Args:
            query: A Tensor of shape [batch, target_length, target_dim].
            key:   an optional Tensor of shape [batch, source_length, source_dim].
                   If None, will use `query`.
            value: An optional Tensor of shape [batch, source_length, source_dim].
                   If None, will use `query`.
            kv_state: An optional KVState. If not None, both key and value must be None.
            time_step: An optional Tensor of shape [batch]. If None, will ignore.

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
        kv_state: Optional[KVState] = None,
    ) -> tuple[NestedTensor, Output]:
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
            kv_state: An optional KVState. If not None, both key and value must be None.

        Returns:
            A `NestedTensor` state of `key`, `value` of shape
            [batch, num_heads, per_head_dim, source_length], and `time_step` of shape [batch].
            An Output instance, where query is of size
            [batch, target_length, num_heads, per_head_dim] and each of key, value are of dim
            [batch, source_length, num_heads, per_head_dim].
        """
        cfg = self.config
        # Default to base layer dtype for initialization if cache_dtype is None.
        dtype = cfg.cache_dtype or self.dtype()
        assert dtype is not None

        if kv_state is not None:
            if key is not None or value is not None:
                raise ValueError("kv_state should not be specified together with key/value")
            kv_kwargs = dict(kv_state=kv_state)
        else:
            kv_kwargs = dict(key=key, value=value)
        # In the prefill state, the time_step filtering is not provided in the QKV forward function,
        # but in the time_step_mask defined below.
        # Therefore, time_step argument for the forward is set as None.
        q_proj, k_proj, v_proj = self.forward(query, **kv_kwargs)

        init_state = dict(time_step=time_step)
        # If external kv_state is provided, we don't need to maintain key/value in cached_state.
        if kv_state is None:
            # Zero-out everything from time_step onwards.
            # Being able to assume that non-filled cache values are 0 allows us to do a slightly
            # more efficient update to `cached_{key,value}` in `extend_step`, by doing a simple add
            # instead of a mask + add.
            time_step_mask = (jnp.arange(k_proj.shape[1]) < time_step[:, None])[..., None, None]
            k_proj = k_proj * time_step_mask
            v_proj = v_proj * time_step_mask

            # Following T5X, we cache key, value as [batch, num_heads, head_dim, seq_len] to take
            # advantage of TPU optimizations (see `extend_step`).
            # Reference:
            # https://github.com/google-research/t5x/blob/4d94d8bf41230d492e15e255c9888b5bfd9a5ee8/t5x/examples/t5/layers.py#L215
            init_state.update(
                key=jnp.moveaxis(k_proj, -3, -1).astype(dtype),
                value=jnp.moveaxis(v_proj, -3, -1).astype(dtype),
            )
        return init_state, self.Output(query=q_proj, key=k_proj, value=v_proj)

    def extend_step(
        self,
        cached_states: NestedTensor,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        kv_state: Optional[KVState] = None,
    ) -> tuple[NestedTensor, Output]:
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
            kv_state: An optional KVState. If not None, both key and value must be None.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where query is of size
            [batch, target_length, num_heads, per_head_dim] and each of key, value are of dim
            [batch, source_length, num_heads, per_head_dim].
        """
        time_step = cached_states["time_step"]
        assert time_step.ndim == 1

        if kv_state is not None:
            if key is not None or value is not None:
                raise ValueError("kv_state should not be specified together with key/value")
            kv_kwargs = dict(kv_state=kv_state)
        else:
            kv_kwargs = dict(key=key, value=value)
        # Project inputs to key, value and query. Each has shape [B, 1, N, H].
        q_proj, k_proj, v_proj = self.forward(query, **kv_kwargs, time_step=time_step)

        updated_state = dict(time_step=time_step + 1)
        if kv_state is None:
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

            updated_state.update(key=new_k_proj, value=new_v_proj)
        return updated_state, self.Output(query=q_proj, key=k_proj, value=v_proj)


class QKVLinear(BaseQKVLinear):
    """Maps input query, key, and value to multi-headed output query, key, and value."""

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures QKVLinear."""

        # The layer used to project.
        layer: MultiheadInputLinear.Config = MultiheadInputLinear.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        for name, dim, num_heads in (
            ("q", cfg.query_dim, cfg.num_heads),
            ("k", cfg.key_dim, self.num_kv_heads),
            ("v", cfg.value_dim, self.num_kv_heads),
        ):
            proj_cfg = cfg.layer
            proj_cfg.model_dim = dim
            proj_cfg.num_heads = num_heads
            proj_cfg.per_head_dim = cfg.per_head_dim
            self._add_child(f"{name}_proj", proj_cfg)

    @property
    def num_kv_heads(self):
        return self.config.num_heads

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        time_step: Optional[Tensor] = None,
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


class GroupedQKVLinear(QKVLinear):
    """A variant of QKVLinear that supports configuring a different number of key, value
    projections.

    Note that the number of key, value projections must evenly divide the number of query heads.
    """

    @config_class
    class Config(QKVLinear.Config):
        """Configures GroupedQKVLinear."""

        # Number of heads for key, value projections.
        # It is required that num_heads % num_kv_heads == 0.
        num_kv_heads: Required[int] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.num_heads % cfg.num_kv_heads != 0:
            raise ValueError(
                f"The number of query subgroups ({cfg.num_kv_heads}) should divide "
                f"the number of query heads ({cfg.num_heads})."
            )

    @property
    def num_kv_heads(self):
        return self.config.num_kv_heads


class QLinear(BaseQKVLinear):
    """Maps input query to multi-headed output query. Assumes external KVState."""

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures QLinear."""

        # The layer used to project.
        layer: MultiheadInputLinear.Config = MultiheadInputLinear.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        proj_cfg = cfg.layer
        proj_cfg.model_dim = cfg.query_dim
        proj_cfg.num_heads = cfg.num_heads
        proj_cfg.per_head_dim = cfg.per_head_dim
        self._add_child("q_proj", proj_cfg)

    @property
    def num_kv_heads(self):
        raise NotImplementedError(type(self))

    def forward(
        self,
        query: Tensor,
        *,
        kv_state: KVState,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        time_step: Optional[Tensor] = None,
    ) -> BaseQKVLinear.Output:
        """Computes projects for the given query. Uses {k,v}_proj from `kv_state`.

        See parent class for full docstring.
        """
        if kv_state is None or key is not None or value is not None:
            raise ValueError(
                f"Only kv_state is expected: key={key}, value={value}, kv_state={kv_state}"
            )
        q_proj = self.q_proj(query)
        return self.Output(query=q_proj, key=kv_state.k_proj, value=kv_state.v_proj)


class FusedQKVLinear(BaseQKVLinear):
    """Maps input query, key, and value to multi-headed query, key, and value using a fused weight.

    N.B. Only supports cases where query, key, and value all have the same shape.
    """

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures FusedQKVLinear."""

        # The layer used to project.
        layer: MultiheadInputLinear.Config = MultiheadInputLinear.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
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

    @property
    def num_kv_heads(self):
        return self.config.num_heads

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        specs = VDict(**super().create_parameter_specs_recursively())

        def transform_factorization_spec(
            spec: Optional[FactorizationSpec],
        ) -> Optional[FactorizationSpec]:
            if spec is None:
                return None
            return FactorizationSpec(axes=[None] + list(spec.axes))

        return jax.tree.map(
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
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
    ) -> NestedTensor:
        if self._use_prebuilt_params(prebuilt):
            return jax.tree.map(lambda _: None, prebuilt)

        def init(prng_key_i):
            return VDict(qkv_proj=self.qkv_proj.initialize_parameters_recursively(prng_key_i))

        return jax.vmap(init)(split_prng_key(prng_key, 3).keys)

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        time_step: Optional[Tensor] = None,
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
                proj = self.qkv_proj.einsum_maybe_quantized(
                    "btd,pdnh->pbtnh", activation=inputs, kernel=params["weight"]
                )
            elif key is not None and value is not None:
                # Compute cross attention but with same target/source shapes.
                assert (
                    query.shape == key.shape == value.shape  # pytype: disable=attribute-error
                ), f"Not supported for {type(self)}."
                inputs = jnp.stack(
                    [query, key, value], axis=0
                )  # [q/k/v, batch, target, model_dim].
                proj = self.qkv_proj.einsum_maybe_quantized(
                    "pbtd,pdnh->pbtnh", activation=inputs, kernel=params["weight"]
                )
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


class FusedGroupedQKVLinear(BaseQKVLinear):
    """Maps input query, key, and value to multi-headed query, key, and value using a fused weight.

    The main difference from FusedQKVLinear is supporting a different number of key, value heads
    than query heads. All of the projection weights are concatenated/fused along the `num_heads`
    axis and then split after projection.
    """

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures FusedGroupedQKVLinear."""

        # Number of heads for key, value projections.
        # It is required that num_heads % num_kv_heads == 0.
        num_kv_heads: Required[int] = REQUIRED
        # The layer used to project.
        layer: MultiheadInputLinear.Config = MultiheadInputLinear.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if not cfg.query_dim == cfg.key_dim == cfg.value_dim:
            raise ValueError(
                f"All projection dims must be equal for {type(self)}, saw: "
                f"query:{cfg.query_dim}, key:{cfg.key_dim}, value:{cfg.value_dim}"
            )
        if cfg.num_heads % cfg.num_kv_heads != 0:
            raise ValueError(
                f"The number of query subgroups {cfg.num_kv_heads} should divide "
                f"the number of query heads {cfg.num_heads}."
            )
        proj_cfg = cfg.layer
        proj_cfg.model_dim = cfg.query_dim
        proj_cfg.num_heads = cfg.num_heads + 2 * cfg.num_kv_heads
        proj_cfg.per_head_dim = cfg.per_head_dim
        self._add_child("qkv_proj", proj_cfg)

    @property
    def num_kv_heads(self):
        return self.config.num_kv_heads

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        time_step: Optional[Tensor] = None,
    ) -> FusedQKVLinear.Output:
        """See FusedQKVLinear for full docstring.

        N.B. Only supports cases where key and value are both None.
        """
        if key is not None or value is not None:
            raise ValueError("Key and value should be both None.")
        cfg = self.config
        proj = self.qkv_proj(query)
        q_proj, k_proj, v_proj = jnp.split(
            proj, [cfg.num_heads, cfg.num_heads + cfg.num_kv_heads], axis=-2
        )
        return self.Output(query=q_proj, key=k_proj, value=v_proj)


def _rotary_sinusoidal_positional_embeddings(
    *, positions: Tensor, dim: int, theta: float = 10000.0
) -> Tensor:
    """Generate the sin/cos positional embedding.

    Ref:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L76-L90

    Args:
        positions: A tensor representing the token position IDs with shape [batch_size, seq_len].
        dim: The dimensionality of the positional embedding.
        theta: A parameter to scale the frequencies.

    Returns:
        Rotary Positional Embedding with shape [batch_size, seq_len, dim].
    """
    if dim % 2 != 0:
        raise ValueError(f"dim: {dim} should be a multiplier of 2.")
    exponents = jnp.arange(dim).astype(jnp.float32)
    pos_array = positions.astype(jnp.float32)
    exponents = jnp.power(theta, 2 * (exponents // 2) / dim)
    position_enc = jnp.expand_dims(pos_array, 2) / jnp.expand_dims(exponents, [0, 1])

    rope_part_1 = jnp.sin(position_enc[:, :, 0::2])
    rope_part_2 = jnp.cos(position_enc[:, :, 1::2])
    rope = jnp.concatenate((rope_part_1, rope_part_2), axis=-1)
    return rope


class RoFormerSinusoidalPositionalEmbedding(BaseLayer):
    """Implementation of Rotary Position Embedding (RoPE).

    Ref:
    https://github.com/huggingface/transformers/blob/62ceb4/src/transformers/models/roformer/modeling_roformer.py
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures RoFormerSinusoidalPositionalEmbedding."""

        dim: Required[int] = REQUIRED  # The dimensionality of the positional embedding.
        theta: float = 10000.0  # The scale of base frequency.

    def forward(self, positions: Tensor) -> Tensor:
        """
        TODO(bwzhang): 1. verify the performance under float32.

        Args:
            positions: A tensor representing the token position IDs.
                The shape is [batch_size, seq_len].

        Returns:
            Rotary Positional Embedding. Shape is [seq_len, dim].
        """
        cfg = self.config
        return _rotary_sinusoidal_positional_embeddings(
            positions=positions, dim=cfg.dim, theta=cfg.theta
        )


def apply_rotary_position_embeddings(
    *,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sinusoidal_pos: Tensor,
    rotary_value: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """This is a jax implementation (a copy) of the RoPE apply_rotary_position_embeddings.

    Ref:
    https://github.com/huggingface/transformers/blob/v4.21.2/src/transformers/models/roformer/modeling_roformer.py#L322-L346

    Args:
        query: Query embeddings with shape [batch_size, seq_len, num_heads, dim].
        key: Key embeddings with shape [batch_size, seq_len, num_heads, dim].
        value: Value embeddings with shape [batch_size, seq_len, num_heads, dim].
        sinusoidal_pos: Rotary position embeddings with shape [batch_size, seq_len, 1, dim].
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
            cfg.rope_pos_emb_layer.set(dim=cfg.per_head_dim),
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

    @property
    def num_kv_heads(self):
        """Propagate num KV heads from input linear."""
        return self.i_proj.num_kv_heads

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        time_step: Optional[Tensor] = None,
    ) -> BaseQKVLinear.Output:
        cfg = self.config
        # Query should have shape of [batch_size, seq_len, num_heads, per_head_dim].
        query, key, value = self.i_proj(query, key=key, value=value)
        if time_step is None:
            # If time_step is None, then we set it to [batch_size, seq_len].
            # In this case, batch_size can be set as 1.
            time_step = jnp.expand_dims(jnp.arange(query.shape[1]), 0)
        else:
            # Time step shape is [batch_size]
            # The expected input shape for rope_pos_emb_layer is [batch_size, seq_len]
            # Therefore, expanding the shape of time_step to [batch_size, 1]
            time_step = jnp.expand_dims(time_step, 1)
        sinusoidal_pos_emb = self.rope_pos_emb_layer.forward(time_step).astype(query.dtype)
        # sinusoidal_pos_emb shape should be [batch_size, seq_len, 1, dim]
        sinusoidal_pos_emb = jnp.expand_dims(sinusoidal_pos_emb, 2)
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

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
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


def constant_scale_fn(value: float) -> ScaleFn:
    """A constant scale function for `MultiheadAttention`.

    Example:
        `key_scale = config_for_function(constant_scale_fn).set(value=0.01)`

    Args:
        value: The value to scale by.

    Returns:
        A `ScaleFn` that always returns `value`.
    """

    def constant_function(per_head_dim: int) -> float:
        del per_head_dim
        return value

    return constant_function


def pow_scale_fn(exp: float) -> ScaleFn:
    """A scale function for `MultiheadAttention` that computes `per_head_dim ** exp`.

    Example:
        `query_scale = config_for_function(pow_scale_fn).set(exp=-0.5)`

    Args:
        exp: The exponent.

    Returns:
        A `ScaleFn` that computes `per_head_dim ** exp`.
    """

    return functools.partial(pow, exp=exp)


class BaseScaleQK(BaseLayer):
    """Defines the common interface for scaling projected attention queries or keys.

    * All subclasses must have `per_head_dim` in their config.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseScaleQK."""

        # The per-head dimension.
        per_head_dim: Required[int] = REQUIRED

    def forward(self, proj: Tensor) -> Tensor:
        """Scales the projected queries or keys.

        Args:
            proj: The projected queries/keys.
                Shape: [batch, seq_length, num_heads, per_head_dim].

        Returns:
            A tensor with the same shape as the input.
        """
        raise NotImplementedError(type(self))


class ScaleQuery(BaseScaleQK):
    """Default implementation for scaling projected queries."""

    @config_class
    class Config(BaseScaleQK.Config):
        """Configures ScaleQuery."""

        # The config for a normalization layer applied along the per-head dim.
        # If None, no normalization is applied.
        norm: Optional[InstantiableConfig] = None
        # The config for a function to compute a query scale muliplier factor.
        # If None, then self.default_scale_fn_config.
        scale_factor: Optional[InstantiableConfig[ScaleFn]] = None
        # A vector to apply per dimension scale to the query projection.
        per_dim_scale: Optional[PerDimScale.Config] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._scale_factor = self.default_scale_factor_config()
        if cfg.scale_factor is not None:
            self._scale_factor = cfg.scale_factor
        self._scale_factor = self._scale_factor.instantiate()
        if cfg.norm is not None:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.per_head_dim))
        if cfg.per_dim_scale:
            self._add_child("per_dim_scale", cfg.per_dim_scale.set(dim=cfg.per_head_dim))

    def apply_norm(self, proj: Tensor) -> Tensor:
        """Applies the norm to projected queries if configured."""
        if "norm" in self.children:
            proj = self.norm(proj)
        return proj

    def apply_per_dim_scale(self, proj: Tensor) -> Tensor:
        """Applies the per-dim scale to projected queries if configured."""
        if "per_dim_scale" in self.children:
            # The Lingvo MultiheadAttention applies a per_dim_scale:
            # https://github.com/tensorflow/lingvo/blob/41212226eac7a26491790c2bd476b78493f93ff6/lingvo/core/batch_major_attention.py#L790
            proj = self.per_dim_scale(proj)
        return proj

    def apply_scale_factor(self, proj: Tensor) -> Tensor:
        """Applies the scale-factor to projected queries."""
        scale = self._scale_factor(self.config.per_head_dim)
        return proj * scale

    def forward(self, proj: Tensor) -> Tensor:
        """Scales the projected queries."""
        proj = self.apply_norm(proj)
        proj = self.apply_per_dim_scale(proj)
        proj = self.apply_scale_factor(proj)
        # Stop scale constant from being folded with others.
        # May increase numerical stability.
        return ops.forward_optimization_barrier(proj)

    @staticmethod
    def default_scale_factor_config() -> InstantiableConfig[ScaleFn]:
        """The config for the default function used to compute the query scale."""

        return config_for_function(pow_scale_fn).set(exp=-0.5)


class ScaleKey(BaseScaleQK):
    """Default implementation for scaling projected keys."""

    @config_class
    class Config(BaseScaleQK.Config):
        """Configures ScaleKey."""

        # The config for a normalization layer applied along the per-head dim.
        # If None, no normalization is applied.
        norm: Optional[InstantiableConfig] = None
        # The config for a function to compute a key scale muliplier factor.
        # If None, then self.default_scale_factor_config.
        scale_factor: Optional[InstantiableConfig[ScaleFn]] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._scale_factor = self.default_scale_factor_config()
        if cfg.scale_factor is not None:
            self._scale_factor = cfg.scale_factor
        self._scale_factor = self._scale_factor.instantiate()
        if cfg.norm is not None:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.per_head_dim))

    def forward(self, proj: Tensor) -> Tensor:
        """Scales the projected keys."""
        cfg = self.config
        if cfg.norm is not None:
            proj = self.norm(proj)
        scale = self._scale_factor(cfg.per_head_dim)
        proj = proj * scale
        # Stop scale constant from being folded with others.
        # May increase numerical stability.
        return ops.forward_optimization_barrier(proj)

    @staticmethod
    def default_scale_factor_config() -> InstantiableConfig[ScaleFn]:
        """The config for the default function used to compute the key scale."""

        return config_for_function(constant_scale_fn).set(value=1)


class MaskFn(Protocol):
    """A broadcastable function for computing a boolean logit mask."""

    def __call__(self, query_position: Tensor, key_position: Tensor) -> Tensor:
        """Returns a bool Tensor of whether the query token at `query_position` should attend
        to the key token at `key_position`.

        Implementations have the following contract:
        * Must support scalar arguments.
        * If given non-scalar arguments of the same shape, the result must be the same as
          applying the function elementwise over these arugments. I.e.,
          ```
          x = f(jnp.asarray([1,2]), jnp.asarray([3,4]))
          assert x[0] == f(jnp.asarray(1), jnp.asarray(3))[None]
          ```
        * If given non-scalar arguments of different shapes, the result must be the same if we
          first broadcast the arguments against each other to make them have the same shape.
        * Beyond requiring broadcastability, must not impose any constraints on the shapes of its
          arguments.

        Args:
            query_position: The index in the sequence of query vectors.
            key_position: The index in the sequence of key vectors.

        Returns:
            Whether the query and key vectors with the given index should attend to one another.
            True means they should attend. False means they should not.
            The shape is the same as the shape obtained after broadcasting the inputs against each
            other.
        """


def _composite_masks(op: Callable[[Tensor, Tensor], Tensor], *mask_fns: ConfigOr[MaskFn]):
    if len(mask_fns) == 0:
        raise RuntimeError(f"Input must not be empty: {mask_fns}")

    def mask(query_position: Tensor, key_position: Tensor):
        fns = [maybe_instantiate(arg) for arg in mask_fns]
        result = fns[0](query_position, key_position)
        for mask in fns[1:]:
            result = op(result, mask(query_position, key_position))
        return result

    return mask


def or_masks(*mask_fns: ConfigOr[MaskFn]) -> MaskFn:
    """Returns a MaskFn that's the union of provided MaskFn's."""
    return _composite_masks(jnp.logical_or, *mask_fns)


def and_masks(*mask_fns: ConfigOr[MaskFn]) -> MaskFn:
    """Returns a MaskFn that's the intersection of provided MaskFn's."""
    return _composite_masks(jnp.logical_and, *mask_fns)


def causal_mask(query_position: Tensor, key_position: Tensor) -> Tensor:
    """Returns the given entry of a causal attention mask.

    Implements the `MaskFn` protocol.
    See that and `MultiheadAttention.Config.mask`.
    """
    return query_position >= key_position


def sliding_window_causal_mask(sliding_window_size: int):
    """Returns a causal MaskFn for sliding window attentions of a given window size.

    Implements the `MaskFn` protocol.
    """

    def mask(query_position: Tensor, key_position: Tensor):
        return query_position - key_position <= sliding_window_size

    return and_masks(causal_mask, mask)


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
        # Config used to scale projected queries prior to computing logits.
        query_scale: BaseScaleQK.Config = ScaleQuery.default_config()
        # Config used to scale projected keys prior to computing logits.
        key_scale: BaseScaleQK.Config = ScaleKey.default_config()
        # Cap the absolute values of logits by tanh. Enabled by setting a positive value.
        atten_logit_cap: Optional[float] = None
        # A function to compute the boolean mask to apply when computing the attention
        # where True means "attend" and False means "do not attend".
        # Set to `causal_mask` for causal masking.
        # When used with certain flash attention implementations, more efficient
        # code paths may be used. (See the FlashAttention docstring for more details.)
        # This field may not be specified if `causal` (deprecated) is specified.
        # If `attention_logit_biases` argument is also specified, both masks are combined with AND.
        mask: ConfigOr[Optional[MaskFn]] = None
        # Deprecated. Use `mask=causal_mask` instead.
        # If True, applies causal masking. `key` and `value` must be None.
        # May not be specified if `mask` is already specified.
        # If `attention_logit_biases` argument is also specified, both masks are combined with AND.
        # TODO (apghml) Eliminate this field in favor of `mask`.
        causal: Optional[bool] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.causal and cfg.mask is not None:
            raise NotImplementedError("Cannot specify `causal` when using `mask`.")
        if cfg.causal:
            self._mask_fn = causal_mask
        else:
            self._mask_fn = maybe_instantiate(cfg.mask)
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
        # Add query scaling layer.
        self._add_child("scale_query", cfg.query_scale.set(per_head_dim=self.per_head_dim()))
        # Add key scaling layer.
        self._add_child("scale_key", cfg.key_scale.set(per_head_dim=self.per_head_dim()))

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
        """Outputs of MultiheadAttention.

        Fields:
            data: [batch, target_length, output_dim]. The attention output. Always present.
            probs: [batch, num_heads, target_length, source_length]. The attention probabilities.
                Populated if "probs" is in `return_aux`.
            kv_state: The KV state used for computing the attention outputs.
                Populated if "kv_state" is in `return_aux`.
        """

        data: Tensor
        probs: Optional[Tensor] = None
        kv_state: Optional[KVState] = None

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        kv_state: Optional[KVState] = None,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> tuple[Optional[NestedTensor], Output]:
        """Computes attention for the given query, key, value, and attention logit biases.

        If key and value are both None, computes self-attention using query.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            query: A Tensor of shape [batch, target_length, target_dim].
            key:   An optional Tensor of shape [batch, source_length, source_dim].
            value: An optional Tensor of shape [batch, source_length, source_dim].
            kv_state: An optional KVState. If specified, both `key` and `value` should be None.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            segment_ids: See ``On segment_ids`` in the file comments.
            cached_states: Optional NestedTensor as produced by `prefill_states`.
            return_aux: See comments on `Output`.

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
        if kv_state is not None:
            if key is not None or value is not None:
                raise ValueError("kv_state should not be specified together with key/value")
            kv_kwargs = dict(kv_state=kv_state)
        else:
            kv_kwargs = dict(key=key, value=value)

        if mode == ForwardMode.FORWARD:
            i_proj_state, (q_proj, k_proj, v_proj) = None, self.i_proj(query, **kv_kwargs)
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            i_proj_state, (q_proj, k_proj, v_proj) = self.i_proj.prefill_states(
                time_step=cached_states["i_proj"], query=query, **kv_kwargs
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            i_proj_state, (q_proj, k_proj, v_proj) = self.i_proj.extend_step(
                cached_states["i_proj"], query, **kv_kwargs
            )
        else:
            raise ValueError(f"Unrecognized mode {mode}.")

        kv_state = KVState(k_proj=k_proj, v_proj=v_proj)
        q_proj = self._remat_name(q_proj, "q_proj")
        k_proj = self._remat_name(k_proj, "k_proj")
        v_proj = self._remat_name(v_proj, "v_proj")
        self.vlog(3, "atten.q_proj=%s", q_proj.sum())
        self.vlog(3, "atten.k_proj=%s", k_proj.sum())
        self.vlog(3, "atten.v_proj=%s", v_proj.sum())
        if attention_logit_biases is not None:
            if attention_logit_biases.ndim == 3:
                # [batch, 1, target_length, source_length].
                attention_logit_biases = attention_logit_biases[:, None, :, :]
            elif attention_logit_biases.ndim == 2:
                # [1, 1, target_length, source_length].
                attention_logit_biases = attention_logit_biases[None, None, :, :]
            elif attention_logit_biases.ndim != 4:
                raise ValueError(
                    f"Invalid attention_logit_biases shape: {attention_logit_biases.shape}."
                )
        if self._mask_fn is not None:
            if mode == ForwardMode.EXTEND_STEP:
                seq_len = k_proj.shape[1]
                time_step = cached_states["i_proj"]["time_step"]
            else:
                seq_len = q_proj.shape[1]
                time_step = None
            mask = self._logit_biases_for_mask(mode=mode, seq_len=seq_len, time_step=time_step)
            if mask is not None:
                attention_logit_biases = apply_attention_logit_biases(
                    mask.astype(q_proj.dtype),
                    attention_logit_biases,
                )
        context, probs = self._compute_attention(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            attention_logit_biases=attention_logit_biases,
            segment_ids=segment_ids,
        )
        self.vlog(3, "atten.prob=%s", probs[0, 0, 0, :])
        self.vlog(3, "atten.context=%s", context.sum())

        # [batch, target_length, output_dim].
        o_proj = self.o_proj(context)
        outputs = self._remat_name(o_proj, "o_proj")
        self._add_tensor_stats("o_proj_outputs", outputs)
        return_aux = return_aux or set()
        output = self.Output(
            data=outputs,
            probs=probs if "probs" in return_aux else None,
            kv_state=kv_state if "kv_state" in return_aux else None,
        )
        return dict(i_proj=i_proj_state), output

    def _logit_biases_for_mask(
        self, *, mode: ForwardMode, seq_len: int, time_step: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Returns the configured attention mask in the form of logit biases.

        ... or None if the implementation of _compute_attention supports applying masks natively.

        For (ForwardMode.FORWARD, ForwardMode.INIT_STATES), the mask can be broadcast to
        [batch, num_heads, seq_len, seq_len].

        For ForwardMode.EXTEND_STEP, the mask can be broadcast to [batch, num_heads, 1, seq_len].
        """
        kv_pos = jnp.arange(seq_len)
        if mode in (ForwardMode.FORWARD, ForwardMode.INIT_STATES):
            mask = self._mask_fn(kv_pos[:, None], kv_pos[None, :])[
                None, None
            ]  # Query and key have same seq lengths.
        elif mode == ForwardMode.EXTEND_STEP:
            # [batch, 1, 1, kv_len].
            # Ex: for a causal mask, mask[b, :, :, kv_pos] = 0 if time_step[b] > kv_pos else 1.
            mask = self._mask_fn(time_step[:, None], kv_pos[None, :])
            mask = mask[:, None, None, :]
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        mask = _bool_to_bias(mask)
        return mask

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Computes attention context and probs.

        Args:
            q_proj: [batch_size, target_length, num_heads, per_head_dim].
            k_proj: [batch_size, source_length, num_heads, per_head_dim].
            v_proj: [batch_size, source_length, num_heads, per_head_dim].
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            segment_ids: See ``segment_ids`` in the file comments.

        Returns:
            The context of shape [batch_size, target_length, num_heads, per_head_dim],
            and probs of shape [batch, num_heads, target_length, source_length].
        """
        # Merge segment ids into attention_logit_biases.
        if segment_ids is not None:
            attention_logit_biases = apply_attention_logit_biases(
                make_segment_mask(source_segments=segment_ids, target_segments=segment_ids),
                attention_logit_biases,
            )

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
        kv_state: Optional[KVState] = None,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> Output:
        """Computes attention for the given query, key, value, and attention logit biases.

        If key and value are both None, computes self-attention using query.

        Args:
            query: A Tensor of shape [batch, target_length, target_dim].
            key:   An optional Tensor of shape [batch, source_length, source_dim].
            value: An optional Tensor of shape [batch, source_length, source_dim].
            kv_state: An optional KVState. If not None, both key and value must be None.
            attention_logit_biases:  See ``On attention logit biases`` in the file comments.
            segment_ids: See `On segment_ids` in the file comments.
            return_aux: See comments on `Output`.

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
            kv_state=kv_state,
            attention_logit_biases=attention_logit_biases,
            segment_ids=segment_ids,
            return_aux=return_aux,
        )
        return output

    def _cap_logits(self, logits: Tensor) -> Tensor:
        """Caps the logits with tanh."""
        cfg = self.config
        if not cfg.atten_logit_cap or cfg.atten_logit_cap <= 0.0:
            return logits
        cap = jnp.array(cfg.atten_logit_cap, dtype=logits.dtype)
        return cap * jnp.tanh(logits / cap)

    def _compute_logits(self, q_proj: Tensor, k_proj: Tensor) -> Tensor:
        q_proj = self.scale_query(q_proj)
        k_proj = self.scale_key(k_proj)
        return jnp.einsum("btnh,bsnh->bnts", q_proj, k_proj)

    def init_states(
        self,
        *,
        target_batch_size: int,
        target_max_len: int,
        kv_state: Optional[KVState] = None,
    ) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            target_batch_size: The batch size of the target to be decoded.
            target_max_len: The sequence length of the target to be decoded.
            kv_state: An optional KVState.

        Returns:
            The cache as a `NestedTensor` with key and value initialized.
        """
        return dict(
            i_proj=self.i_proj.init_states(
                target_batch_size=target_batch_size,
                target_max_len=target_max_len,
                kv_state=kv_state,
            )
        )

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        query: Tensor,
        kv_state: Optional[KVState] = None,
        attention_logit_biases: Optional[Tensor],
        return_aux: Optional[set[str]] = None,
    ) -> tuple[NestedTensor, Output]:
        """Initializes cache for autoregressive cached decoding.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [B]. Each value is an index into the length dimension
                indicating where decoding will start from.
            query: Tensor of shape [B, T, D] corresponding to query vector up to `time_step`. For
                batch index `i`, only `query[i, :time_step[i], ...]` will affect subsequent
                decoding.
            kv_state: An optional KVState.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            return_aux: See comments on `Output`.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].
        """
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            query=query,
            cached_states=dict(i_proj=time_step),
            kv_state=kv_state,
            attention_logit_biases=attention_logit_biases,
            return_aux=return_aux,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        query: Tensor,
        *,
        kv_state: Optional[KVState] = None,
        attention_logit_biases: Optional[Tensor],
        return_aux: Optional[set[str]] = None,
    ) -> tuple[NestedTensor, Output]:
        """Computes the value vector given the query of the current step.
        This function is used by autoregressive decoding.

        Based on:
        https://github.com/tensorflow/lingvo/blob/5754b2f840ebf0f8c52d87e5d4d76f22e372513e/lingvo/jax/layers/attentions.py#L1249

        Args:
            cached_states: A `NestedTensor` object containing tensors which are the results of
                previous attentions, and index used for fast decoding. Contains "key" and "value" of
                shape [B, N, H, T], and a Tensor "time_step" of shape [B].
            query: Tensor of shape [B, 1, D] corresponding to query vector at "time_step" indices.
            kv_state: An optional KVState.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
                Additionally, target_length is expected to be 1 since this is per time step.
                The biases should already include causal masking for decoding, plus other biases
                if necessary.
            return_aux: See comments on `Output`.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where .data is of the same shape as query, .probs is of shape
            [batch, num_heads, 1, source_length].
        """
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            query=query,
            cached_states=cached_states,
            kv_state=kv_state,
            attention_logit_biases=attention_logit_biases,
            return_aux=return_aux,
        )

    @staticmethod
    def default_query_scale_config() -> InstantiableConfig[ScaleFn]:
        """The config for the default function used to compute the query scale."""

        return config_for_function(pow_scale_fn).set(exp=-0.5)

    @staticmethod
    def default_key_scale_config() -> InstantiableConfig[ScaleFn]:
        """The config for the default function used to compute the key scale."""

        return config_for_function(constant_scale_fn).set(value=1)


class GroupedQueryAttention(MultiheadAttention):
    """A Grouped-Query Attention (GQA) layer.

    Query projections are divided into K groups along the `num_heads` dimension. Projections in the
    same query subgroup share one common key/value head. This reduces the size of the KV-cache by a
    factor of `num_heads/num_kv_heads`.

    When `input_linear` is a `GroupedQKVLinear` layer with `num_kv_heads=1`, GQA reduces to
    multi-query attention (MQA).
    When `input_linear` is a `QKVLinear` layer (i.e. `num_kv_heads=num_heads`), GQA is equivalent to
    multi-head attention (MHA).

    Note that in some cases fused variants `FusedQKVLinear` or `FusedGroupedQKVLinear` can be used
    as drop-in replacements for `QKVLinear` or `GroupedQKVLinear` respectively (see corresponding
    layer docstrings for details).

    Reference: https://arxiv.org/abs/2305.13245
    """

    @property
    def num_kv_heads(self):
        return self.i_proj.num_kv_heads

    def _repeat_kv_heads(self, key_or_value: Tensor) -> Tensor:
        """Repeats key or value heads dim to match the query."""
        num_head_repeats = self.config.num_heads // key_or_value.shape[-2]
        if num_head_repeats == 1:
            return key_or_value
        # Repeat along the num_heads dim: [batch, source_length, num_heads, per_head_dim].
        return jnp.repeat(key_or_value, num_head_repeats, axis=-2)

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """See `MultiheadAttention._compute_attention` for details."""
        k_proj = self._repeat_kv_heads(k_proj)
        v_proj = self._repeat_kv_heads(v_proj)
        return super()._compute_attention(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            **kwargs,
        )


class SigmoidAttention(MultiheadAttention):
    """A multi-head sigmoid-based attention layer, instead of softmax.

    TODO(floris_weers): Add paper reference.
    """

    @config_class
    class Config(MultiheadAttention.Config):
        """Configures SigmoidAttention."""

        seq_len: Required[int] = REQUIRED  #  Maximum sequence length used.

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """See `MultiheadAttention._compute_attention` for details."""
        # Merge segment ids into attention_logit_biases.
        if segment_ids is not None:
            attention_logit_biases = apply_attention_logit_biases(
                make_segment_mask(source_segments=segment_ids, target_segments=segment_ids),
                attention_logit_biases,
            )

        cfg = self.config
        logits = self._compute_logits(q_proj, k_proj)
        logits = self._cap_logits(logits)
        self.vlog(3, "atten.logits=%s", logits[0, 0, 0, :])

        if attention_logit_biases is None:
            attention_logit_biases = 0
        # To approximate softmax, we subtract a bias dependent on sequence length.
        attention_logit_biases = attention_logit_biases - jnp.log(cfg.seq_len)
        probs = sigmoid_with_biases(
            logits,
            attention_logit_biases=attention_logit_biases,
        )
        probs = self.dropout(probs)

        context = jnp.einsum("bnts,bsnh->btnh", probs, v_proj).astype(v_proj.dtype)
        context = self._remat_name(context, "context")
        return context, probs


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
            T is expected to be >= 1.

    Returns:
        y: a Tensor of shape [T, T], s.t. y[i, k] = x[i, j] where k = i + j - (T - 1),
            if 0 <= k < T.
    """
    t, offset_length = x.shape
    assert offset_length == 2 * t - 1
    if t <= 1:
        return x
    # [t * (2t - 1)].
    x = x.reshape([-1])
    # [t * (2t - 2)].
    x = x[t - 1 : -1]
    # [t, 2t - 2].
    x = x.reshape([t, -1])
    # [t, t]. When t = 2, do not trim.
    if t > 2:
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
        q_proj: A Tensor of shape [batch, target_length, num_heads, per_head_dim], representing
            projected queries.
        k_proj: A Tensor of shape [batch, target_length, num_heads, per_head_dim], representing
            projected keys.
        relative_pos_emb: A Tensor of shape [num_embeddings, num_heads, per_head_dim], representing
            projected relative positional embeddings, where num_embeddings = 2 * target_length - 1.
            relative_pos_emb[key_i - query_i + target_length - 1] represents positional
            embeddings between query[:, query_i] and key[:, key_i] and is usually computed from
            sinusoidal_positional_embeddings(query_i - key_i), i.e.,
            relative_pos_emb[0] represents query_i = target_length - 1 and key_i = 0.
            relative_pos_emb[-1] represents query_i = 0 and key_i = target_length - 1.
        u: A Tensor of shape [num_heads, per_head_dim].
            The trainable `u` in https://arxiv.org/pdf/1901.02860.pdf 3.3 used for term 'ac'.
        v: A Tensor of shape [num_heads, per_head_dim].
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
        # Applies query scale-factor to the logits.
        LOGIT = 0
        # Applies query scale-factor to the queries.
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

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
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
        **kwargs,
    ) -> MultiheadAttention.Output:
        if key is not None or value is not None:
            raise ValueError("Both key and value must be None for MultiheadAttentionXL")
        return super().forward(query, **kwargs)

    def _compute_logits(self, q_proj: Tensor, k_proj: Tensor) -> Tensor:
        cfg = self.config
        with child_context("apply_query_norm", module=self):
            # We apply the query norm (if configured) to the projection (not the logits).
            q_proj = self.scale_query.apply_norm(q_proj)

        with child_context("apply_per_dim_scale", module=self):
            q_proj = self.scale_query.apply_per_dim_scale(q_proj)

        if cfg.scale_position == MultiheadAttentionXL.ScalePosition.QUERY:
            with child_context("apply_scale_factor_queries", module=self):
                q_proj = self.scale_query.apply_scale_factor(q_proj)

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

        # Apply key scaling.
        k_proj = self.scale_key(k_proj)

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
            with child_context("apply_scale_factor_logits", module=self):
                logits = self.scale_query.apply_scale_factor(logits)
        return logits

    def extend_step(
        self,
        cached_states: NestedTensor,
        query: Tensor,
        **kwargs,
    ) -> tuple[NestedTensor, MultiheadAttention.Output]:
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
        """Outputs of TransformerAttentionLayer.

        Fields:
            data: [batch, target_length, output_dim]. The attention output. Always present.
            probs: The attention probabilities returned by the attention layer.
                Populated if "probs" is in return_aux.
            kv_state: The KV state used to compute output.
                Populated if "kv_state" is in return_aux.
        """

        data: Tensor
        probs: Optional[Tensor] = None
        kv_state: Optional[KVState] = None

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        target: Tensor,
        source: Optional[Union[Tensor, KVState]] = None,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> tuple[Optional[NestedTensor], Output]:
        """Computes either self-attention or cross-attention for the given target and source.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            target: A Tensor of shape [batch, target_length, target_dim].
            source: An optional KVState or Tensor of shape [batch, source_length, source_dim].
                If None, uses norm(target) as source (self-attention).
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            segment_ids: segment_ids: See ``On segment_ids`` in the file comments.
            cached_states: Optional NestedTensor as produced by `prefill_states`.
            return_aux: See comments on `Output`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If `mode` is unsupported.
            NotImplementedError: If `cfg.structure` is not supported.
        """
        cfg = self.config

        if source is None:
            kv_kwargs = {}
        elif isinstance(source, KVState):
            kv_kwargs = {"kv_state": source}
        elif isinstance(source, Tensor):
            kv_kwargs = {"key": source, "value": source}
        else:
            raise NotImplementedError(source)
        kv_kwargs["return_aux"] = return_aux

        def attention_thunk(target: Tensor) -> tuple[Optional[NestedTensor], Tensor]:
            if mode == ForwardMode.FORWARD:
                atten_state, atten_output = (
                    None,
                    self.attention(
                        query=target,
                        **kv_kwargs,
                        attention_logit_biases=attention_logit_biases,
                        segment_ids=segment_ids,
                    ),
                )
            elif mode == ForwardMode.INIT_STATES:
                assert cached_states is not None
                atten_state, atten_output = self.attention.prefill_states(
                    time_step=cached_states["attention"],
                    query=target,
                    **kv_kwargs,
                    attention_logit_biases=attention_logit_biases,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert cached_states is not None
                atten_state, atten_output = self.attention.extend_step(
                    cached_states["attention"],
                    target,
                    **kv_kwargs,
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
        return dict(attention=atten_state), self.Output(
            data=data, probs=atten_output.probs, kv_state=atten_output.kv_state
        )

    def forward(
        self,
        *,
        target: Tensor,
        source: Optional[Union[Tensor, KVState]] = None,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> Output:
        """Computes attention with target as query and source as key and value.

        Args:
            target: A Tensor of shape [batch, target_length, target_dim].
            source: An optional KVState or Tensor of shape [batch, source_length, source_dim].
                If None, uses norm(target) as source (self-attention)
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            segment_ids: See ``segment_ids`` in the file comments.
            return_aux: See comments on `Output`.

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
            segment_ids=segment_ids,
            cached_states=None,
            return_aux=return_aux,
        )
        return output

    def init_states(
        self,
        *,
        target_batch_size: int,
        target_max_len: int,
        kv_state: Optional[KVState] = None,
    ) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            target_batch_size: The batch size of the target to be decoded.
            target_max_len: The sequence length of the target to be decoded.
            kv_state: An optional KVState.

        Returns:
            The cache as a `NestedTensor` with key and value initialized.
        """
        return dict(
            attention=self.attention.init_states(
                target_batch_size=target_batch_size,
                target_max_len=target_max_len,
                kv_state=kv_state,
            )
        )

    def prefill_states(
        self,
        *,
        time_step: NestedTensor,
        target: Tensor,
        source: Optional[Union[Tensor, KVState]] = None,
        attention_logit_biases: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> tuple[NestedTensor, Output]:
        """Initializes cache for autoregressive cached decoding.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            target: Tensor of shape [batch, target_length, target_dim] corresponding to query vector
                at `time_step` indices. For batch index `i`, only `target[i, :time_step[i], ...]`
                will affect subsequent decoding.
            source: An optional KVState or Tensor of shape [batch, source_length, source_dim].
                If None, uses norm(target) as source (self-attention)
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            return_aux: See comments on `Output`.

        Returns:
            A `NestedTensor` state depending on the `attention` layer implementation.
            An Output instance, where .data is of the same shape as query, .probs is of shape
            [batch, num_heads, target_length, source_length].
        """
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            target=target,
            source=source,
            cached_states=dict(attention=time_step),
            attention_logit_biases=attention_logit_biases,
            return_aux=return_aux,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        target: Tensor,
        *,
        source: Optional[Union[Tensor, KVState]] = None,
        attention_logit_biases: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> tuple[NestedTensor, Output]:
        """Computes the value vector given the query of the current step.
        This function is used by autoregressive decoding.

        Args:
            cached_states: A `NestedTensor` object containing tensors which are the
                results of previous attentions, and index used for fast decoding. Contains
                "attention" cached states.
            target: Tensor of shape [B, 1, D] corresponding to query vector at index
                time_step.
            source: An optional KVState or Tensor of shape [batch, source_length, source_dim].
                If None, uses norm(target) as source (self-attention)
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
                Additionally, target_length is expected to be 1 since this is per time step.
                attention_logit_biases should have already taken care of causal masking for
                decoding, plus other maskings necessary.
            return_aux: See comments on `Output`.

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
            source=source,
            cached_states=cached_states,
            attention_logit_biases=attention_logit_biases,
            return_aux=return_aux,
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
        activation: Union[str, tuple[str, str]] = "nn.relu"

        # The dropout layer config.
        dropout: InstantiableConfig = Dropout.default_config()

        # The stochastic depth layer config.
        # Pytorch reference:
        # https://github.com/facebookresearch/deit/blob/main/models_v2.py#L59
        # Tensorflow reference:
        # https://github.com/tensorflow/models/blob/master/official/projects/vit/modeling/nn_blocks.py#L103-L119
        stochastic_depth: InstantiableConfig = StochasticDepth.default_config()

        # The inner structure of the layer: "prenorm", "postnorm", "hybridnorm", "nonorm".
        # * prenorm: y = x + feedforward(norm(x))
        # * postnorm: y = norm(x + feedforward(x))
        # * hybridnorm: y = postnorm(x + feedforward(prenorm(x)))
        # * nonorm: y = feedforward(x)   # no residual, which is usually applied externally.
        #
        # References:
        # prenorm/postnorm: https://arxiv.org/abs/2002.04745.
        # hybridnorm: https://github.com/google/praxis/blob/main/praxis/layers/transformers.py#L273
        # nonorm: see ParallelTransformerLayer.
        structure: str = "prenorm"

        # outputs = inputs + residual_weight * x.
        residual_weight: float = 1.0

        # Auxiliary stats.

        # If True, add "dead_neurons/{activation}" stats for activation functions that have
        # zones of near-zero gradients, e.g., x < 0 for ReLU.
        #
        # A "neuron" `i` is considered dead if all of x[..., i] (across batch/seq) fall within the
        # dead zone.
        #
        # Only supported for a subset of activation functions, including relu, gelu, and silu.
        add_dead_neuron_summary: Optional[bool] = None

        # Adds summary of RMS norms of the specified values. Supported value are:
        # - "inputs": inputs of the layer.
        # - "linear1_outputs": outputs of linear1.
        # - "linear2_outputs": outputs of linear2.
        # TODO(tlei3): deprecate this feature since we use TensorStats.
        add_value_rms_norm_summary: Sequence[str] = []

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: TransformerFeedForwardLayer.Config = self.config
        if cfg.structure in ["prenorm", "postnorm"]:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))
        elif cfg.structure == "hybridnorm":
            self._add_child("prenorm", cfg.norm.set(input_dim=cfg.input_dim))
            self._add_child("postnorm", cfg.norm.set(input_dim=cfg.input_dim))
        elif cfg.structure == "nonorm":
            pass
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
        if cfg.structure in ["prenorm", "hybridnorm", "nonorm"]:
            self._add_child("dropout1", cfg.dropout)
            self._add_child("dropout2", cfg.dropout)
        elif cfg.structure in ["postnorm"]:
            self._add_child("dropout", cfg.dropout)
        else:
            raise NotImplementedError(cfg.structure)

        self._add_child("stochastic_depth", cfg.stochastic_depth)
        # TODO(tlei3): deprecate this check since we will use TensorStats to handle what
        # tensors are logged.
        for value in cfg.add_value_rms_norm_summary:
            if value not in ["inputs", "linear1_outputs", "linear2_outputs"]:
                raise NotImplementedError(f"add_value_rms_norm_summary: {value}")

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config

        def _linear2(x):
            """Applies linear2, optionally logging RMS norm of the output."""
            x = self.linear2(x)
            self._add_tensor_stats("linear2_outputs", x)
            return x

        self._add_tensor_stats("inputs", inputs)

        remat_pt1 = "activation"
        remat_pt2 = "linear2"
        if cfg.structure == "prenorm":
            x = self.norm(inputs)
            x = self._linear1_activation(x)
            x = self._remat_name(x, remat_pt1)
            x = self.dropout1(x)
            x = _linear2(x)
            x = self._remat_name(x, remat_pt2)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x += inputs
        elif cfg.structure == "postnorm":
            x = self._linear1_activation(inputs)
            x = self._remat_name(x, remat_pt1)
            x = _linear2(x)
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
            x = _linear2(x)
            x = self._remat_name(x, remat_pt2)
            x = self.postnorm(x)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x += inputs
        elif cfg.structure == "nonorm":
            x = inputs
            x = self._linear1_activation(x)
            x = self._remat_name(x, remat_pt1)
            x = self.dropout1(x)
            x = _linear2(x)
            x = self._remat_name(x, remat_pt2)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            # We still apply `residual_weight`, since there is usually a residual link outside of
            # this layer, e.g., in ParallelTransformerLayer.
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
        else:
            raise NotImplementedError(cfg.structure)
        return x

    def _linear1_activation(self, x: Tensor) -> Tensor:
        cfg = self.config
        if isinstance(cfg.activation, tuple):
            activations = [
                self._get_activation(
                    self.children[f"linear1_{i}"](x), activation_fn_name=activation
                )
                for i, activation in enumerate(cfg.activation)
            ]
            assert len(activations) == 2, cfg.activation
            outputs = activations[0] * activations[1]
            self._add_tensor_stats("linear1_0_outputs", activations[0])
            self._add_tensor_stats("linear1_1_outputs", activations[1])
            self._add_tensor_stats("linear1_outputs", outputs)
            return outputs
        else:
            x = self.linear1(x)
            x = self._get_activation(x, activation_fn_name=cfg.activation)
            self._add_tensor_stats("linear1_outputs", x)
            return x

    def _get_activation(self, x: Tensor, activation_fn_name: str) -> Tensor:
        """Applies activation function on 'x' and optionally counts the number of dead neurons.

        Args:
            x: A tensor of shape [B, S, H].
            activation_fn_name: The name of the activation fn.

        Returns:
            activation_fn(x).
        """
        cfg = self.config
        if cfg.add_dead_neuron_summary:
            if activation_fn_name in ["quick_gelu", "exact_gelu"]:
                # To make GELU be sufficiently small.
                threshold = -4.0
            elif activation_fn_name in ["nn.silu", "nn.sigmoid"]:
                # nn.silu(jnp.array(-10.)) = -0.00045398
                # nn.sigmoid(jnp.array(-10.)) = 4.5397872e-05
                threshold = -10.0
            elif activation_fn_name in ["nn.relu", "squared_relu"]:
                threshold = 0
            else:
                threshold = None
            if threshold is not None:
                max_hidden_units = jnp.max(x, axis=(0, 1))
                num_dead_units = jnp.count_nonzero(
                    jnp.less(max_hidden_units, threshold).astype(jnp.int32)
                )
                self.add_summary(
                    f"dead_neurons/{activation_fn_name}",
                    num_dead_units,
                )
        return get_activation_fn(activation_fn_name)(x)


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
        self_attention_kv_state: Optional[KVState] = None,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        target_segment_ids: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> tuple[Optional[NestedTensor], BaseTransformerLayer.Output]:
        """Computes transformer layer outputs and self/cross-attention probabilities.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            self_attention_kv_state: An optional KVState used for self-attention.
            self_attention_logit_biases: An optional Tensor representing the self-attention biases.
            cross_attention_data: An optional Tensor of shape [batch, source_length, source_dim].
            cross_attention_logit_biases: An optional Tensor representing the cross-attention
                biases.
            target_segment_ids: See ``segment_ids`` in the file comments.
            cached_states: Optional NestedTensor as produced by `prefill_states`.
            return_aux: See comments on BaseTransformerLayer.forward.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as `data`, .self_attention_probs is
            of shape [batch, num_heads, target_length, target_length], and .cross_attention_probs is
            of shape [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If `mode` is unsupported.
        """
        self.vlog(3, "transformer.input=%s", data.sum())
        self_attention_return_aux = set()
        cross_attention_return_aux = set()
        if return_aux:
            if "self_attention_probs" in return_aux:
                self_attention_return_aux.add("probs")
            if "self_attention_kv_state" in return_aux:
                self_attention_return_aux.add("kv_state")
            if "cross_attention_probs" in return_aux:
                cross_attention_return_aux.add("probs")
        if mode == ForwardMode.FORWARD:
            self_atten_state, self_atten_outputs = (
                None,
                self.self_attention(
                    target=data,
                    segment_ids=target_segment_ids,
                    source=self_attention_kv_state,
                    attention_logit_biases=self_attention_logit_biases,
                    return_aux=self_attention_return_aux,
                ),
            )
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            if target_segment_ids is not None:
                raise NotImplementedError("target_segment_ids is not supported in INIT_STATES.")
            self_atten_state, self_atten_outputs = self.self_attention.prefill_states(
                time_step=cached_states["self_attention"],
                target=data,
                source=self_attention_kv_state,
                attention_logit_biases=self_attention_logit_biases,
                return_aux=self_attention_return_aux,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            if target_segment_ids is not None:
                raise NotImplementedError("target_segment_ids is not supported in EXTEND_STEP.")
            self_atten_state, self_atten_outputs = self.self_attention.extend_step(
                cached_states=cached_states["self_attention"],
                target=data,
                source=self_attention_kv_state,
                attention_logit_biases=self_attention_logit_biases,
                return_aux=self_attention_return_aux,
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
                return_aux=cross_attention_return_aux,
            )
            data = cross_atten_outputs.data
            cross_attention_probs = cross_atten_outputs.probs
        else:
            cross_attention_probs = None
        data = self.feed_forward(data)
        self.vlog(3, "transformer.output=%s", data.sum())
        # TODO(markblee): Support module outputs in decoding.
        if mode == ForwardMode.FORWARD:
            self.add_module_output("output", data)
        return dict(self_attention=self_atten_state), BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=self_atten_outputs.probs,
            self_attention_kv_state=self_atten_outputs.kv_state,
            cross_attention_probs=cross_attention_probs,
        )

    def forward(
        self,
        data: Tensor,
        **kwargs,
    ) -> BaseTransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            cached_states=None,
            **kwargs,
        )
        return output

    def init_states(
        self,
        *,
        target_batch_size: int,
        target_max_len: int,
        self_attention_kv_state: Optional[KVState] = None,
    ) -> NestedTensor:
        return dict(
            self_attention=self.self_attention.init_states(
                target_batch_size=target_batch_size,
                target_max_len=target_max_len,
                kv_state=self_attention_kv_state,
            )
        )

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        **kwargs,
    ) -> tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=dict(self_attention=time_step),
            data=data,
            **kwargs,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        **kwargs,
    ) -> tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            cached_states=cached_states,
            data=data,
            **kwargs,
        )


class ParallelTransformerLayer(BaseTransformerLayer):
    """A Transformer layer with parallel self-attention and feed-forward layers:

    x = norm(inputs)
    outputs = inputs + self_atten(x) + ffn(x)

    TODO(rpang): experiment to understand whether we should use separate normalization layers
        for self_atten and ffn as in PaLM.

    References:
        https://github.com/kingoflolz/mesh-transformer-jax
        PaLM: https://arxiv.org/abs/2204.02311
    """

    @config_class
    class Config(BaseTransformerLayer.Config):
        norm: InstantiableConfig = LayerNorm.default_config()  # The normalization layer config.
        self_attention: MultiheadAttention.Config = MultiheadAttention.default_config()
        feed_forward: TransformerFeedForwardLayer.Config = (
            TransformerFeedForwardLayer.default_config().set(structure="nonorm")
        )

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: TransformerLayer.Config = self.config
        self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))
        self._add_child(
            "self_attention",
            cfg.self_attention.set(
                query_dim=cfg.input_dim,
                key_dim=cfg.input_dim,
                value_dim=cfg.input_dim,
                output_dim=cfg.input_dim,
            ),
        )
        self._add_child("feed_forward", cfg.feed_forward.set(input_dim=cfg.input_dim))

    def forward(
        self,
        *,
        data: Tensor,
        self_attention_logit_biases: Optional[Tensor] = None,
        target_segment_ids: Optional[Tensor] = None,
    ) -> BaseTransformerLayer.Output:
        """Computes transformer layer outputs and self/cross-attention probabilities.

        Args:
            data: A Tensor of shape [batch, target_length, target_dim].
            self_attention_logit_biases: An optional Tensor representing the self-attention biases.
            target_segment_ids: See ``segment_ids`` in the file comments.

        Returns:
            An Output instance, where .data is of the same shape as `data`, .self_attention_probs is
            of shape [batch, num_heads, target_length, target_length].

        Raises:
            ValueError: If `mode` is unsupported.
        """
        inputs = data
        data = self.norm(data)
        self_atten_outputs = self.self_attention(
            query=data,
            key=data,
            value=data,
            attention_logit_biases=self_attention_logit_biases,
            segment_ids=target_segment_ids,
        )
        feed_forward_outputs = self.feed_forward(data)
        outputs = inputs + self_atten_outputs.data + feed_forward_outputs
        return BaseTransformerLayer.Output(
            data=outputs,
            self_attention_probs=self_atten_outputs.probs,
            self_attention_kv_state=self_atten_outputs.kv_state,
            cross_attention_probs=None,
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
        cached_states: Optional[NestedTensor] = None,
        **kwargs,
    ) -> tuple[Optional[NestedTensor], Tensor]:
        """Computes transformer layer outputs and self/cross-attention probabilities.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
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
            output = self.layer.forward(data=data, **kwargs)
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            cached_states, output = self.layer.prefill_states(
                time_step=cached_states["layer"],
                data=data,
                **kwargs,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            cached_states, output = self.layer.extend_step(
                cached_states=cached_states,
                data=data,
                **kwargs,
            )
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        skip_input = output.data
        data = self.adapter(output.data)
        data += skip_input
        self.vlog(3, "adapted_transformer.output=%s", data.sum())
        return cached_states, output._replace(data=data)

    def forward(
        self,
        data: Tensor,
        **kwargs,
    ) -> BaseTransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            cached_states=None,
            **kwargs,
        )
        return output

    def init_states(self, **kwargs) -> NestedTensor:
        return dict(layer=self.layer.init_states(**kwargs))

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        **kwargs,
    ) -> tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=dict(layer=time_step),
            data=data,
            **kwargs,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        **kwargs,
    ) -> tuple[NestedTensor, BaseTransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            cached_states=cached_states,
            data=data,
            **kwargs,
        )


def set_double_shard_weights_config(
    cfg: Union[TransformerLayer.Config, Sequence[TransformerLayer.Config]],
    *,
    batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"),
    fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
    tp_axis_names: Union[str, Sequence[str]] = "model",
    seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    # pytype: disable=attribute-error
    def set_attn_partition_specs(attn_layer: MultiheadAttention.Config):
        # Shard weights.
        input_linear_cfg = attn_layer.input_linear
        if hasattr(input_linear_cfg, "input_linear"):
            input_linear_cfg = input_linear_cfg.input_linear
        input_linear_cfg.layer.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)
        attn_layer.output_linear.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, seq_axis_names, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, seq_axis_names, tp_axis_names)

    if not isinstance(cfg, Sequence):
        cfg = [cfg]

    for layer_cfg in cfg:
        set_attn_partition_specs(layer_cfg.self_attention.attention)
        if layer_cfg.cross_attention is not None:
            set_attn_partition_specs(layer_cfg.cross_attention.attention)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)
    # pytype: enable=attribute-error


class BaseStackedTransformerLayer(BaseTransformerLayer):
    """The common interface of all stacked transformer layer classes.

    Note that BaseStackedTransformerLayer is a subclass of BaseTransformerLayer and therefore
    can be used where a BaseTransformerLayer is expected.

    The Output returned by BaseStackedTransformerLayer has the following fields:
        * .data is of the same shape as query, from the output of the final layer;
        * .self_attention_kv_state is of shape [batch, target_length, num_heads, head_dim],
          from the self-attention KV state of the final layer;
        * .probs is of shape [num_layers, batch, num_heads, target_length, source_length],
          from all layers of the stack;
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
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
    ) -> NestedTensor:
        cfg = self.config  # type: StackedTransformerLayer.Config
        prng_key = split_prng_key(prng_key, cfg.num_layers)
        state = {}
        for i in range(cfg.num_layers):
            layer = self._layers[i]
            key = jax.tree.map(lambda x, index=i: x[index], prng_key.keys)
            state[layer.name] = layer.initialize_parameters_recursively(
                key, prebuilt=get_or_none(prebuilt, layer.name)
            )
        return state

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Tensor,
        cached_states: Optional[NestedTensor] = None,
        **layer_kwargs,
    ) -> tuple[list[Optional[NestedTensor]], TransformerLayer.Output]:
        """Computes transformer stack outputs.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            (updated_cache_states, outputs), where
            updated_cached_states is an optional NestedTensor of cache states, depending on `mode`;
            outputs is an instance of Output (see comments on BaseStackedTransformerLayer).

        Raises:
            ValueError: If `mode` is unsupported.
        """
        all_layer_outputs = []
        all_layer_states = []
        for i, layer in enumerate(self._layers):
            # Prepare inputs to the current layer.
            data = self._update_data(data, all_layer_outputs=all_layer_outputs)
            self._update_layer_kwargs(layer_kwargs, all_layer_outputs=all_layer_outputs)

            if mode == ForwardMode.FORWARD:
                layer_states, layer_outputs = None, layer(data, **layer_kwargs)
            elif mode == ForwardMode.INIT_STATES:
                assert cached_states is not None
                layer_states, layer_outputs = layer.prefill_states(
                    time_step=cached_states,
                    data=data,
                    **layer_kwargs,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert cached_states is not None
                layer_states, layer_outputs = layer.extend_step(
                    cached_states=cached_states[i],
                    data=data,
                    **layer_kwargs,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")
            all_layer_outputs.append(layer_outputs)
            all_layer_states.append(layer_states)
            data = layer_outputs.data

        return all_layer_states, self._aggregate_layer_outputs(all_layer_outputs)

    def _update_data(
        self,
        data: Tensor,
        *,
        all_layer_outputs: list[BaseTransformerLayer.Output],
    ):
        """Updates `data` using other args.

        This method is called before we invoke each layer in `self._layers`.
        The updated data will be passed to the layer invocation.

        Args:
            data: A Tensor denoting the input data to the upcoming layer.
            all_layer_outputs: A list of BaseTransformerLayer.Output that is appended with
                the output of each constituent layer in the stack.

        Returns:
            A new Tensor.
        """
        del all_layer_outputs
        return data

    def _update_layer_kwargs(
        self,
        layer_kwargs: dict[str, Any],
        *,
        all_layer_outputs: list[BaseTransformerLayer.Output],
    ):
        """Updates `layer_kwargs` using other args.

        This method is called before we invoke each layer in `self._layers`.
        The updated `layer_kwargs` will be passed to the layer invocation.

        Args:
            layer_kwargs: a dictionary of arguments that can be used by individual layers.
            all_layer_outputs: a list of BaseTransformerLayer.Output that is appended with
                the output of each constituent layer in the stack.
        """
        pass  # Do nothing by default.

    def _aggregate_layer_outputs(
        self,
        layer_outputs: Sequence[BaseTransformerLayer.Output],
    ) -> BaseTransformerLayer.Output:
        """Aggregates outputs from the stack."""
        data = layer_outputs[-1].data
        self_attention_kv_state = layer_outputs[-1].self_attention_kv_state
        aux_outputs = [
            output._replace(data=None, self_attention_kv_state=None) for output in layer_outputs
        ]
        # Stack auxiliary outputs along axis 0.
        outputs = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *aux_outputs)
        return outputs._replace(data=data, self_attention_kv_state=self_attention_kv_state)

    def forward(
        self,
        data: Tensor,
        **layer_kwargs,
    ) -> TransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
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
        **layer_kwargs,
    ) -> tuple[list[NestedTensor], TransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=time_step,
            data=data,
            **layer_kwargs,
        )

    def extend_step(
        self,
        cached_states: list[NestedTensor],
        data: Tensor,
        **layer_kwargs,
    ) -> tuple[list[NestedTensor], TransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            cached_states=cached_states,
            data=data,
            **layer_kwargs,
        )


class _TransformerRepeat(Repeat):
    """A Repeat layer with layer=TransformerLayer."""

    @config_class
    class Config(Repeat.Config):
        """Configures _TransformerRepeat."""

        # The additional fields of BaseTransformerLayer.Output that should propagate as input to
        # the next layer.
        #
        # For example, carry=("data", "self_attention_kv_state") means that both `data` and
        # `self_attention_kv_state` will propagate between layers.
        #
        # If None, only "data" is propagated.
        carry: Optional[Sequence[str]] = None

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Tensor,
        cached_states: Optional[NestedTensor] = None,
        **layer_kwargs,
    ) -> tuple[Optional[NestedTensor], TransformerLayer.Output]:
        """Computes transformer stack outputs.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            (updated_cache_states, outputs), where
            updated_cached_states is an optional NestedTensor of cache states, depending on `mode`;
            outputs is an instance of Output (see comments on BaseStackedTransformerLayer).

        Raises:
            ValueError: If `mode` is unsupported.
        """
        cfg = self.config

        if cached_states is not None:
            for path, value in flatten_items(cached_states):
                assert value.shape[0] == cfg.num_layers, f"{path}={shapes(value)}"

        def layer_fn(carry, x_i):
            if mode == ForwardMode.FORWARD:
                layer_states, layer_outputs = None, self.layer(**carry, **layer_kwargs)
            elif mode == ForwardMode.INIT_STATES:
                assert x_i is not None
                layer_states, layer_outputs = self.layer.prefill_states(
                    time_step=x_i,
                    **carry,
                    **layer_kwargs,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert x_i is not None
                layer_states, layer_outputs = self.layer.extend_step(
                    cached_states=x_i,
                    **carry,
                    **layer_kwargs,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")

            ys = {k: v for k, v in layer_outputs._asdict().items() if k not in carry}
            if layer_states is not None:
                ys["cached_states"] = layer_states
            return {k: getattr(layer_outputs, k) for k in carry}, ys

        if cfg.carry is None:
            carry = {"data": data}
        else:
            layer_kwargs["data"] = data
            carry = {k: layer_kwargs.pop(k) for k in cfg.carry}
        repeat_outputs: Repeat.Output = self._run(layer_fn, carry=carry, xs=cached_states)
        carry = repeat_outputs.carry
        ys = repeat_outputs.ys
        updated_states = ys.pop("cached_states", None)

        for k in ("data", "self_attention_kv_state"):
            if k in carry:
                continue
            v = ys.pop(k, None)
            if v is not None:
                # Take the output from the last layer.
                if isinstance(v, KVState):
                    v = KVState(k_proj=v.k_proj[-1], v_proj=v.v_proj[-1])
                else:
                    v = v[-1]
            carry[k] = v
        return updated_states, TransformerLayer.Output(**carry, **ys)

    def forward(
        self,
        data: Tensor,
        **layer_kwargs,
    ) -> TransformerLayer.Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            cached_states=None,
            **layer_kwargs,
        )
        return output

    def init_states(self, *args: Any, **kwargs: Any) -> NestedTensor:
        cfg = self.config

        def layer_fn(_):
            return self.layer.init_states(*args, **kwargs)

        return jax.vmap(layer_fn)(jnp.empty(cfg.num_layers))

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        **layer_kwargs,
    ) -> tuple[NestedTensor, TransformerLayer.Output]:
        cfg = self.config
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            data=data,
            cached_states=jnp.tile(time_step, [cfg.num_layers, 1]),
            **layer_kwargs,
        )

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        **layer_kwargs,
    ) -> tuple[NestedTensor, TransformerLayer.Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            data=data,
            cached_states=cached_states,
            **layer_kwargs,
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
        self, prng_key: Tensor, *, prebuilt: Optional[NestedTensor] = None
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
        **layer_kwargs,
    ) -> TransformerLayer.Output:
        return self.repeat(data, **layer_kwargs)

    def init_states(self, *args: Any, **kwargs: Any) -> NestedTensor:
        return VDict(repeat=self.repeat.init_states(*args, **kwargs))

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        **layer_kwargs,
    ) -> tuple[list[NestedTensor], TransformerLayer.Output]:
        repeat_cached_states, output = self.repeat.prefill_states(
            time_step=time_step, data=data, **layer_kwargs
        )
        return VDict(repeat=repeat_cached_states), output

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        **layer_kwargs,
    ) -> tuple[list[NestedTensor], TransformerLayer.Output]:
        repeat_cached_states, output = self.repeat.extend_step(
            cached_states=cached_states["repeat"],
            data=data,
            **layer_kwargs,
        )
        return VDict(repeat=repeat_cached_states), output


class _TransformerPipeline(Pipeline):
    """Transformer pipeline layer."""

    def forward(
        self,
        data: Tensor,
        *,
        return_aux: Optional[set[str]] = None,
        **kwargs,
    ) -> TransformerLayer.Output:
        carry_in = dict(data=data)
        return_aux = return_aux or set()

        # Even though attention logit biases do not change across layers, we
        # include them in the carry so that they are aligned with the microbatches.
        carry_in.update(kwargs)
        carry_in = self._to_microbatches(carry_in)
        self.vlog(3, "carry_in=%s", shapes(carry_in))

        def layer_fn(carry, _):
            layer_outputs: TransformerLayer.Output = self.layer(**carry)
            carry.pop("data")
            return dict(**carry, data=layer_outputs.data), {
                k: v if k in return_aux else None
                for k, v in layer_outputs._asdict().items()
                if k != "data"
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
        # Config for the pipeline implementation, such as pipeline schedule.
        pipeline: _TransformerPipeline.Config = _TransformerPipeline.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config  # type: PipelinedTransformerLayer.Config
        if cfg.num_layers % cfg.num_stages != 0:
            raise ValueError(f"num_stages {cfg.num_stages} must divide num_layers {cfg.num_layers}")
        num_layers_per_stage = cfg.num_layers // cfg.num_stages
        stage_cfg = cfg.stage.set(
            input_dim=cfg.input_dim, layer=cfg.layer, num_layers=num_layers_per_stage
        )
        pipeline_cfg = cfg.pipeline.set(
            layer=stage_cfg, num_layers=cfg.num_stages, num_microbatches=cfg.num_microbatches
        )
        self._add_child("pipeline", pipeline_cfg)

    def initialize_parameters_recursively(
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
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
        **kwargs,
    ) -> TransformerLayer.Output:
        return self.pipeline(data, **kwargs)

    # TODO(sneha): extend_step


def build_remat_spec(
    stack_cfg: Union[
        BaseStackedTransformerLayer.Config, "RepeatedConformerLayer.Config"  # type: ignore
    ],
    self_attention: bool = True,
    feed_forward: bool = False,
    offload_dst: Optional[Literal["pinned_host"]] = None,
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
        self_attention: Checkpoint self attention layer activations if true.
        feed_forward: Checkpoint feed-forward layer activations if true.
        offload_dst: Destination of remat checkptoing offloading. Relevant Maxtext example:
          https://github.com/google/maxtext/blob/ebd39aa64d670fa13a313b6f776e01ad9e450321/MaxText/layers/models.py#L230.

    Returns:
        None (if no rematerialization is needed) or a RematSpec.
    """
    # TODO(markblee): Switch to using isinstance everywhere.
    if stack_cfg.klass is PipelinedTransformerLayer:
        return None

    checkpoints = []
    if self_attention:
        attention_name = stack_cfg.layer.self_attention.attention.klass.__name__
        checkpoints.extend(
            [f"{attention_name}.{el}" for el in ["q_proj", "k_proj", "v_proj", "context", "o_proj"]]
        )

    if feed_forward and hasattr(stack_cfg.layer, "feed_forward"):
        ffn_name = stack_cfg.layer.feed_forward.klass.__name__
        checkpoints.extend([f"{ffn_name}.{el}" for el in ["activation", "linear2"]])

    policy = config_for_function(jax_remat_policies.save_only_these_names).set(
        names_which_can_be_saved=checkpoints
    )
    if offload_dst:
        policy = config_for_function(jax_remat_policies.save_and_offload_only_these_names).set(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=checkpoints,
            offload_src="device",
            offload_dst=offload_dst,
        )

    return RematSpec(
        prevent_cse=stack_cfg.klass is StackedTransformerLayer,
        # If we are running inside a jax.lax.scan (Repeated/Pipelined transformers
        # or Repeated Conformers) we can enable common subexpression elimination optimizations.
        policy=policy,
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


def alibi_get_slopes(num_heads: int) -> list:
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

    def get_slopes_power_of_2(n: int) -> list:
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

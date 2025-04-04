# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/mvit:
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Attention layers for ViT variant vision transformers."""

import math
from collections.abc import Sequence
from typing import NamedTuple, Optional

import jax.nn
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    KVState,
    MultiheadAttention,
    TransformerAttentionLayer,
    apply_attention_logit_biases,
    softmax_with_biases,
)
from axlearn.common.attention_bias import make_segment_mask
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, config_class
from axlearn.common.layers import get_stochastic_depth_linear_rate
from axlearn.common.module import Module, Tensor
from axlearn.common.utils import NestedTensor
from axlearn.vision.window_attention import (
    window_partition_with_window_size,
    window_unpartition_with_window_size,
)


def get_rel_pos_emb(
    q_size: int,
    k_size: int,
    rel_pos_emb: Tensor,
) -> Tensor:
    """Get relative positional embeddings according to the relative positions of query and key
    sizes.

    Ref:
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        q_size: size of query q.
        k_size: size of key k.
        rel_pos_emb: A float Tensor of relative position embeddings with shape (length, dim).

    Returns:
        A Tensor of extracted positional embeddings according to relative positions
            with shape (q_size, k_size, dim).
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos_emb.shape[0] != max_rel_dist:
        # Ref for linear interpolation:
        # https://github.com/facebookresearch/detectron2/blob/1bc3a33a71991142c2c67bc99e1559d6101fb009/detectron2/modeling/backbone/utils.py#L82
        rel_pos_emb_resized = jax.image.resize(
            image=jnp.reshape(rel_pos_emb, (1, rel_pos_emb.shape[0], -1)),
            shape=(1, max_rel_dist, rel_pos_emb.shape[1]),
            method="linear",
        )
        rel_pos_emb_resized = jnp.reshape(rel_pos_emb_resized, (max_rel_dist, -1))
    else:
        rel_pos_emb_resized = rel_pos_emb

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = jnp.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = jnp.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_emb_resized[relative_coords.astype(jnp.int32)]


def add_decomposed_rel_pos_emb(
    attn: Tensor,
    q: Tensor,
    rel_pos_emb_h: Tensor,
    rel_pos_emb_w: Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> Tensor:
    """Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

    Ref:
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn: A float Tensor of attention map with shape (B, num_heads, q_h * q_w, k_h * k_w).
        q: A float Tensor of query q in the attention layer with shape (B, q_h * q_w, num_heads, C).
        rel_pos_emb_h: A float Tensor of relative position embeddings with shape (Lh, C) for height.
        rel_pos_emb_w: A float Tensor of relative position embeddings with shape (Lw, C) for width.
        q_size: A int Tuple of spatial sequence size of query q with shape (q_h, q_w).
        k_size: A int tuple of spatial sequence size of key k with shape (k_h, k_w).

    Returns:
        attn: A Tensor of attention map with added relative positional embeddings
            with shape (B, num_heads, q_h * q_w, k_h * k_w).
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos_emb(q_h, k_h, rel_pos_emb_h)
    rw = get_rel_pos_emb(q_w, k_w, rel_pos_emb_w)

    batch, _, num_heads, dim = q.shape
    r_q = jnp.reshape(jnp.transpose(q, (0, 2, 1, 3)), (batch * num_heads, q_h, q_w, dim))
    rel_h = jnp.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = jnp.einsum("bhwc,wkc->bhwk", r_q, rw)

    attn = jnp.reshape(attn, (batch * num_heads, q_h * q_w, k_h * k_w))
    attn = (
        jnp.reshape(attn, (batch * num_heads, q_h, q_w, k_h, k_w))
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    )
    attn = jnp.reshape(attn, (batch, num_heads, q_h * q_w, k_h * k_w))
    return attn


class WindowedAttention(MultiheadAttention):
    """A basic window attention layer with relative position embeddings."""

    @config_class
    class Config(MultiheadAttention.Config):
        # Use relative positional embedding.
        use_rel_pos_emb: bool = False
        # Initialize the positional embedding as constant zero.
        use_pos_zero_init: bool = True
        # Input resolution for calculating the relative positional parameter size.
        input_size: tuple[int, int] = (64, 64)
        # Cap the absolute values of logits by tanh. Enabled by setting a positive value.
        atten_logit_cap: Optional[float] = 50.0

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: WindowedAttention.Config = self.config
        if not cfg.query_dim == cfg.key_dim == cfg.value_dim:
            raise ValueError(
                f"MultiheadAttention requires query_dim ({cfg.query_dim}) == "
                f"key_dim ({cfg.key_dim}) == value_dim ({cfg.value_dim})"
            )

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = super()._create_layer_parameter_specs()
        if cfg.use_rel_pos_emb:
            if cfg.use_pos_zero_init:
                params["rel_pos_emb_h"] = ParameterSpec(
                    shape=(2 * cfg.input_size[0] - 1, cfg.output_dim // cfg.num_heads),
                    mesh_axes=(None, "model"),
                    initializer=param_init.constant_initializer(0.0),
                )
                params["rel_pos_emb_w"] = ParameterSpec(
                    shape=(2 * cfg.input_size[1] - 1, cfg.output_dim // cfg.num_heads),
                    mesh_axes=(None, "model"),
                    initializer=param_init.constant_initializer(0.0),
                )
        return params

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        query_positions: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> MultiheadAttention.Output:
        """Computes self-windowed attention for the given query and attention logit biases.

        Both key and value need to be None to computes self-attention for WindowedAttention.

        Args:
            query: a Tensor of shape [batch, target_length, target_dim].
            key:   an optional Tensor of shape [batch, source_length, source_dim].
            value: an optional Tensor of shape [batch, source_length, source_dim].
            attention_logit_biases:  See ``On attention logit biases`` in the file comments.
            segment_ids: See `On segment_ids` in MultiheadAttention's file comments.
            query_positions: See ``positions`` in MultiheadAttention's file comments.
            return_aux: See comments in MultiheadAttention.Output.

        Returns:
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            ValueError: If key & value are an invalid combination.
        """
        # Merge segment ids into attention_logit_biases.
        if segment_ids is not None:
            attention_logit_biases = apply_attention_logit_biases(
                make_segment_mask(source_segments=segment_ids, target_segments=segment_ids),
                attention_logit_biases,
            )
        if key is not None or value is not None:
            raise ValueError("Both key and value must be None for WindowedAttention")
        cfg = self.config
        batch, height, width, _ = query.shape
        query = jnp.reshape(query, (batch, height * width, -1))
        q_proj, k_proj, v_proj = self.i_proj(
            query, key=key, value=value, query_positions=query_positions
        )
        q_proj = self._remat_name(q_proj, "q_proj")
        k_proj = self._remat_name(k_proj, "k_proj")
        v_proj = self._remat_name(v_proj, "v_proj")
        self.vlog(3, "atten.q_proj=%s", q_proj.sum())
        self.vlog(3, "atten.k_proj=%s", k_proj.sum())
        self.vlog(3, "atten.v_proj=%s", v_proj.sum())
        logits = self._compute_logits(q_proj, k_proj)

        if cfg.use_rel_pos_emb:
            logits = add_decomposed_rel_pos_emb(
                logits,
                q_proj,
                self.parameters["rel_pos_emb_h"],
                self.parameters["rel_pos_emb_w"],
                (height, width),
                (height, width),
            )

        # some safeguards for softmax 2d (soft capping, max normalization)
        logits = self._cap_logits(logits)
        self.vlog(3, "atten.logits=%s", logits[0, 0, 0, :])
        if attention_logit_biases is not None and attention_logit_biases.ndim == 3:
            # [batch, 1, target_length, source_length].
            attention_logit_biases = attention_logit_biases[:, None, :, :]
        probs = softmax_with_biases(logits, attention_logit_biases=attention_logit_biases)
        probs = self.dropout(probs)
        context = self._compute_context(probs, v_proj)
        context = self._remat_name(context, "context")
        self.vlog(3, "atten.prob=%s", probs[0, 0, 0, :])
        self.vlog(3, "atten.context=%s", context.sum())
        # [batch, target_length, output_dim].
        o_proj = self.o_proj(context)
        outputs = self._remat_name(o_proj, "o_proj")
        key_positions = jnp.arange(k_proj.shape[1])[None]
        kv_state = KVState(k_proj=k_proj, v_proj=v_proj, key_positions=key_positions)
        return_aux = return_aux or set()
        return self.Output(
            data=outputs,
            probs=probs if "probs" in return_aux else None,
            kv_state=kv_state if "kv_state" in return_aux else None,
        )


class WindowedSelfAttentionLayer(TransformerAttentionLayer):
    """A Transformer attention layer with normalization and a skip connection.
    Can be used for self windowed-attention only.
    """

    @config_class
    class Config(TransformerAttentionLayer.Config):
        # Window size for window attention blocks.
        window_size: int = 14
        attention: InstantiableConfig = (
            WindowedAttention.default_config()
        )  # The attention layer config.

    def forward(
        self,
        *,
        target: Tensor,
        source: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        target_positions: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> TransformerAttentionLayer.Output:
        """Computes attention with target as query and source as key and value.

        Args:
            target: a Tensor of shape [batch, target_length, target_dim].
            source: None, uses norm(target) as source for self-attention
            attention_logit_biases: See ``On attention logit biases`` in the file comments.
            segment_ids: See `On segment_ids` in MultiheadAttention's file comments.
            target_positions: See ``positions`` in MultiheadAttention's file comments.
            return_aux: See comments in TransformerAttentionLayer.Output.

        Returns:
            An Output instance, where .data is of the same shape as target and .probs is of shape
            [batch, num_heads, target_length, source_length].

        Raises:
            NotImplementedError: If cfg.structure is unsupported.
        """
        cfg = self.config
        if cfg.structure == "prenorm":
            skip_input = target  # pre-norm: where normalization happens within the residual part.
            x = self.norm(target)
            batch, seq_len, _ = x.shape
            height = width = int(math.sqrt(seq_len))
            x = jnp.reshape(x, (batch, height, width, -1))
            # Window Partition
            if cfg.window_size > 0:
                x, pad_hw = window_partition_with_window_size(x, cfg.window_size)

            atten_output = self.attention(
                query=x,
                key=source,
                value=source,
                attention_logit_biases=attention_logit_biases,
                segment_ids=segment_ids,
                query_positions=target_positions,
                return_aux=return_aux,
            )
            x = atten_output.data

            # Reverse window partition
            if cfg.window_size > 0:
                x = window_unpartition_with_window_size(
                    atten_output.data, cfg.window_size, pad_hw, (height, width)
                )

            x = jnp.reshape(x, (batch, height * width, -1))
            data = skip_input + self.stochastic_depth(self.dropout(x))
        else:
            raise NotImplementedError(cfg.structure)
        return self.Output(data=data, probs=atten_output.probs, kv_state=atten_output.kv_state)


class StackedWindowedTransformerLayer(BaseStackedTransformerLayer):
    """The interface of all stacked windowed-attention transformer layer classes."""

    @config_class
    class Config(BaseStackedTransformerLayer.Config):
        # Window size for window attention blocks.
        window_size: int = 14
        # Input resolution for calculating the relative positional parameter size.
        # [image_size // patch_size, image_size // patch_size].
        input_size: tuple[int, int] = (64, 64)
        # Indexes for blocks using window attention.
        window_block_indexes: Sequence[int] = (
            list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11))
        )

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.layer.input_dim is not REQUIRED:
            raise ValueError(
                f"Do not set Config.layer.input_dim. Set Config.input_dim instead: {cfg.layer}"
            )
        self._layers = []
        for i in range(cfg.num_layers):
            layer_cfg = cfg.layer.set(input_dim=cfg.input_dim)
            if cfg.peak_stochastic_depth_rate:
                layer_rate = get_stochastic_depth_linear_rate(
                    cfg.peak_stochastic_depth_rate,
                    stage_order=i + 1,
                    num_stages=cfg.num_layers,
                )
                layer_cfg.self_attention.stochastic_depth.rate = layer_rate
                layer_cfg.feed_forward.stochastic_depth.rate = layer_rate

            # Specify window_size value for window self-attention if layer index
            # is within window_block_indices. Otherwise, global self-attention
            # will apply (window_size=0).
            window_size = cfg.window_size if i in cfg.window_block_indexes else 0
            layer_cfg.self_attention.window_size = window_size
            # Input size is used to initialize the relative positional embedding
            layer_cfg.self_attention.attention.input_size = (
                cfg.input_size if window_size == 0 else (window_size, window_size)
            )

            self._layers.append(self._add_child(f"layer{i}", layer_cfg))

    class Output(NamedTuple):
        # [batch, target_length, input_dim]. The layer output.
        data: Tensor

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        **layer_kwargs,
    ) -> Output:
        all_layer_outputs = []
        for layer in self._layers:
            layer_outputs = layer(
                data,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
                **layer_kwargs,
            )
            all_layer_outputs.append(layer_outputs)
            data = layer_outputs.data
        return self.Output(data=data)

    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        *,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, MultiheadAttention.Output]:
        raise NotImplementedError(type(self))

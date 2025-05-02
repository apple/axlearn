# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# microsoft/DeBERTa:
# Copyright (c) Microsoft, Inc. 2020.
# Licensed under the MIT license.
#
# huggingface/transformers:
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A replication of DeBERTa.

References:
https://arxiv.org/pdf/2006.03654.pdf
https://github.com/microsoft/DeBERTa
"""
import math
from enum import Enum
from typing import Optional, cast

import jax.numpy as jnp

from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    MultiheadAttention,
    MultiheadInputLinear,
    QKVLinear,
    RepeatedTransformerLayer,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import Tensor
from axlearn.common.bert import BertModel, bert_embedding_config, bert_model_config
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.embedding import Embedding
from axlearn.common.encoder import Encoder
from axlearn.common.layers import BaseClassificationHead, Dropout, LayerNorm, RedirectToSharedModule
from axlearn.common.module import Module, child_context
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer


def _deberta_make_log_bucket_position(
    relative_position: Tensor, *, num_directional_buckets: int, max_distance: int
) -> Tensor:
    """Computes relative position buckets from relative positions, using logarithmically bigger bins
    up to max_distance.

    Relative positions are assigned a bucket based on a piecewise function:

        bucket(x) = {
            x if abs(x) <= alpha,
            sign(x) * min(beta, [alpha + log_{gamma / alpha}(abs(x) / alpha) * (beta - alpha)]) else
        }

    Here alpha splits the piecewise function; beta configures the output range [-beta, beta]; and
    gamma determines the base of the log.

    Specifically, we set alpha = num_directional_buckets // 2, beta = num_directional_buckets, and
    gamma = max_distance. This essentially places positions in the range
    [-num_directional_buckets // 2, num_directional_buckets // 2] into their own bucket. Positions
    outside of this range are assigned to logarithmically sized buckets up until max_distance.

    Note that this is almost identical to T5 bucketing, but has slight differences.

    Reference:
    https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/da_utils.py

    Args:
        relative_position: An int Tensor of any shape.
        num_directional_buckets: A positive integer representing how many buckets we want to have in
            either direction. Must be at least 2.
        max_distance: A positive integer representing max position in either direction.
            Must be at least 2.

    Returns:
        A Tensor of same shape as relative_position. Values in [-max_distance, max_distance] are
        mapped to [-num_directional_buckets, num_directional_buckets]. Values outside this range
        will be clipped when computing attention scores in `disentangled_attention_bias`.
    """
    assert max_distance > 1 and num_directional_buckets > 1

    # Keep track of original directions.
    sign = jnp.sign(relative_position)

    # Half of the buckets (in each direction) are for exact increments in positions.
    max_exact = num_directional_buckets // 2

    # Ensure we don't run into numerical issues with log by taking abs.
    # Positions within [-max_exact, max_exact] will remain unchanged.
    relative_position_abs = jnp.maximum(
        jnp.abs(relative_position), jnp.ones_like(relative_position) * (max_exact - 1)
    )

    # The other half of the buckets are for logarithmically bigger bins up to max_distance.
    # relative_position_abs values within [max_exact, max_distance] are mapped to
    # [max_exact, num_directional_buckets]:
    # 1) max_exact maps to max_exact
    # 2) max_distance - 1 maps to num_directional_buckets - 1
    # 3) max_distance maps to num_directional_buckets
    relative_position_if_large = max_exact + jnp.ceil(
        jnp.log(relative_position_abs.astype(jnp.float32) / max_exact)
        / math.log((max_distance - 1) / max_exact)
        * (max_exact - 1)
    ).astype(jnp.int32)

    # Note that T5 clamps to [-num_directional_buckets, num_directional_buckets] here. We can in
    # theory clip here as well to achieve the same result, since we clip anyway in
    # `disentangled_attention_bias`. In keeping with the original implementation we skip it, but
    # leave it as a note in case we want to unify with T5 bucketing.
    # relative_position_if_large = jnp.minimum(relative_position_if_large, num_directional_buckets)

    # Handle the piecewise function.
    return jnp.where(
        relative_position_abs <= max_exact, relative_position, relative_position_if_large * sign
    )


def deberta_relative_position_bucket(
    *,
    query_len: int,
    key_len: int,
    num_directional_buckets: int,
    max_distance: int,
) -> Tensor:
    """Computes relative position buckets as seen in DeBERTa.

    Reference:
    https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/da_utils.py

    Each relative position up until max_distance has a scalar embedding, for a total of
    2*num_directional_buckets+1 unique embeddings in the range [0, 2*num_directional_buckets].

    Args:
        query_len: A positive integer.
        key_len: A positive integer.
        num_directional_buckets: A positive integer representing number of buckets in either
            direction.
        max_distance: A positive integer representing max position in either direction.

    Returns:
        A Tensor with shape [query_len, key_len] and values in [0, 2*num_directional_buckets].
    """
    # [query_len, 1].
    query_positions = jnp.expand_dims(jnp.arange(query_len), -1)
    # [1, key_len].
    key_positions = jnp.expand_dims(jnp.arange(key_len), 0)
    # [query_len, key_len]. Note that this is slightly different from T5.
    relative_positions = query_positions - key_positions
    relative_positions = _deberta_make_log_bucket_position(
        relative_positions,
        num_directional_buckets=num_directional_buckets,
        max_distance=max_distance,
    )
    return relative_positions + num_directional_buckets


class DisentangledAttentionType(Enum):
    """Type of disentangled attention as described in Equation (2).

    C represents 'content' and P represents 'position'.

    Content-to-content attention is always computed, so we don't include it here.
    Position-to-position attention is not used in the original paper, so we exclude it.
    """

    C2P = 1
    P2C = 2


class DisentangledSelfAttention(MultiheadAttention):
    """Computes disentangled self attention as seen in DeBERTa.

    Adapted from:
    https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/disentangled_attention.py

    We make several simplifications:
    - Assume query, key, and value always have the same shapes, which is fine for self attention.
      See also: https://github.com/microsoft/DeBERTa/issues/33#issuecomment-774022674.
    - Removes P2P, which breaks in the reference code and is not used in the original paper (3.1)
      See also: https://github.com/huggingface/transformers/issues/14621#issuecomment-986039548
    """

    @config_class
    class Config(MultiheadAttention.Config):
        """Configures DisentangledSelfAttention.Config."""

        # Type(s) of attention to include in attention score.
        attention_type: Required[set[DisentangledAttentionType]] = REQUIRED
        # Maximum distance for bucketing.
        max_distance: Required[int] = REQUIRED
        # Number of relative position buckets in each distance. If None, defaults to max_distance.
        num_directional_buckets: Optional[int] = None
        # Positional embeddings. Typically a RedirectToSharedModule to a shared embedding, e.g:
        # https://github.com/microsoft/DeBERTa/blob/c8efdecffbd2d57a6f53742c54b20bbf52b53ad0/DeBERTa/deberta/bert.py#L192-L194
        # https://github.com/microsoft/DeBERTa/blob/c8efdecffbd2d57a6f53742c54b20bbf52b53ad0/DeBERTa/apps/models/masked_language_model.py#L66
        # Should be of shape [2 * num_directional_buckets, hidden_dim].
        pos_emb: Required[InstantiableConfig] = REQUIRED
        # Dropout applied to positional embeddings.
        pos_emb_dropout: Dropout.Config = Dropout.default_config()
        # TODO(markblee): Test shared weights with FusedQKVLinear.
        # Projection Wqr. One can share weights by redirecting to the input projection's q_proj.
        pos_q_proj: Optional[InstantiableConfig] = None
        # Projection Wkr. One can share weights by redirecting to the input projection's k_proj.
        pos_k_proj: Optional[InstantiableConfig] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        self._validate_config()
        cfg = self.config
        self._add_child("pos_emb", cfg.pos_emb)
        self._add_child("pos_emb_dropout", cfg.pos_emb_dropout)

        self._share_with_descendants(self.i_proj.q_proj, shared_module_name="q_proj")
        self._share_with_descendants(self.i_proj.k_proj, shared_module_name="k_proj")

        if cfg.pos_k_proj is None:
            cfg.pos_k_proj = MultiheadInputLinear.default_config().set(
                model_dim=self.hidden_dim(),
                num_heads=cfg.num_heads,
                per_head_dim=self.per_head_dim(),
                bias=True,
            )
        if cfg.pos_q_proj is None:
            cfg.pos_q_proj = MultiheadInputLinear.default_config().set(
                model_dim=self.hidden_dim(),
                num_heads=cfg.num_heads,
                per_head_dim=self.per_head_dim(),
                bias=False,
            )

        if DisentangledAttentionType.C2P in cfg.attention_type:
            self._add_child("pos_k_proj", cfg.pos_k_proj)
        if DisentangledAttentionType.P2C in cfg.attention_type:
            # Note: P2C attention does not have a bias:
            # https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/disentangled_attention.py#L59
            self._add_child("pos_q_proj", cfg.pos_q_proj)

    def _validate_config(self):
        cfg = self.config

        # Make sure attention types are all valid.
        valid_attention_types = set(DisentangledAttentionType)
        invalid_attention_types = set(cfg.attention_type).difference(valid_attention_types)
        if invalid_attention_types:
            raise ValueError(
                f"Got invalid attention_type(s): {invalid_attention_types}. "
                f"Valid options are: {valid_attention_types}."
            )

        # Make sure attention types are unique (length matters for scaling).
        if len(set(cfg.attention_type)) != len(cfg.attention_type):
            raise ValueError(f"Got duplicate attention_type(s): {cfg.attention_type}.")

    def num_directional_buckets(self) -> int:
        cfg = self.config
        return cfg.num_directional_buckets or cfg.max_distance

    def _compute_logits(self, q_proj: Tensor, k_proj: Tensor) -> Tensor:
        logits = self._attention_scores(q_proj, k_proj)
        # TODO(markblee): Support passing in relative_pos_emb, relative_pos via forward.
        bias = self._disentangled_attention_bias(q_proj=q_proj, k_proj=k_proj)
        return logits + bias

    def _attention_scores(self, q_proj: Tensor, k_proj: Tensor) -> Tensor:
        # Compute scaled attention scores.
        # https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/disentangled_attention.py#L85
        cfg = self.config
        q_scale = (1 + len(cfg.attention_type)) ** -0.5
        q_proj = self.scale_query(q_proj * q_scale, positions=None)
        k_proj = self.scale_key(k_proj, positions=None)
        # [batch, num_heads, target_length, source_length].
        return super()._compute_logits(q_proj, k_proj)

    def _scale_qk(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        query_positions: Tensor,
        key_positions: Tensor,
    ):
        # Do not scale q/k here as _attention_scores expects unscaled q/k.
        return q_proj, k_proj

    def _disentangled_attention_bias(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        relative_pos_emb: Optional[Tensor] = None,
        relative_pos: Optional[Tensor] = None,
    ):
        """Computes disentangled attention scores.

        Args:
            q_proj: A Tensor of shape [batch, target_length, num_heads, per_head_dim].
            k_proj: A Tensor of shape [batch, source_length, num_heads, per_head_dim].
            relative_pos_emb: An optional Tensor of shape [2 * num_directional_buckets, hidden_dim].
                If None, retrieves embeddings from the configured `pos_emb`.
            relative_pos: An optional Tensor of shape [batch, target_length, source_length] with
                values between [0, 2*num_directional_buckets]. If None, computes positions using
                `deberta_relative_position_bucket`.

        Returns:
            A Tensor of shape [batch, num_heads, query_len, key_len].

        Raises:
            NotImplementedError: If query_len != key_len.
        """
        # TODO(markblee): Support query_len != key_len.
        if q_proj.shape[1] != k_proj.shape[1]:
            raise NotImplementedError(
                f"Expected query_len == key_len, instead got "
                f"query_len={q_proj.shape[1]} and key_len={k_proj.shape[1]}."
            )

        cfg = self.config
        num_directional_buckets = self.num_directional_buckets()

        if relative_pos_emb is None:
            relative_pos_emb = self.pos_emb.embeddings()
            relative_pos_emb = self.pos_emb_dropout(relative_pos_emb)

        if relative_pos is None:
            # [query_len, key_len] with values in [0, 2 * num_directional_buckets].
            relative_pos = deberta_relative_position_bucket(
                query_len=q_proj.shape[1],
                key_len=k_proj.shape[1],
                num_directional_buckets=num_directional_buckets,
                max_distance=cfg.max_distance,
            )
            # Make broadcastable to batch: [1, query_len, key_len].
            relative_pos = jnp.expand_dims(relative_pos, axis=0)

        # Note: these are asymmetric, i.e. in the range [0, 2 * num_directional_buckets - 1],
        # as described in Equation (3). See also:
        # https://github.com/microsoft/DeBERTa/blob/771f5822798da4bef5147edfe2a4d0e82dd39bac/DeBERTa/deberta/disentangled_attention.py#L125
        relative_pos_emb = relative_pos_emb[: 2 * num_directional_buckets]
        # Make broadcastable to batch dim: [1, 2 * num_directional_buckets, hidden_dim].
        relative_pos_emb = jnp.expand_dims(relative_pos_emb, axis=0)
        # Make broadcastable to num_heads: [batch, 1, query_len, key_len].
        relative_pos = jnp.expand_dims(relative_pos, axis=1)

        score = 0

        # Content->Position.
        if DisentangledAttentionType.C2P in cfg.attention_type:
            # [1, 2 * num_directional_buckets, num_heads, per_head_dim].
            pos_k_proj = self.pos_k_proj(relative_pos_emb)
            # [batch, num_heads, query_len, 2 * num_directional_buckets].
            with child_context("c2p_attn_scores", module=self):
                c2p_score = self._attention_scores(q_proj, pos_k_proj)
            # Gather along last axis using the relative positions d(i,j), as in Equation (4).
            # [batch, 1, query_len, key_len]. Range is inclusive.
            d_ij = jnp.clip(relative_pos, 0, 2 * num_directional_buckets - 1)
            # [batch, num_heads, query_len, key_len].
            score += jnp.take_along_axis(c2p_score, d_ij, axis=-1)

        # Position->Content.
        if DisentangledAttentionType.P2C in cfg.attention_type:
            # [1, 2 * num_directional_buckets, num_heads, per_head_dim].
            pos_q_proj = self.pos_q_proj(relative_pos_emb)
            # [batch, num_heads, key_len, 2 * num_directional_buckets].
            with child_context("p2c_attn_scores", module=self):
                p2c_score = self._attention_scores(k_proj, pos_q_proj)
            # Gather along last axis using the relative positions d(j,i), as in Equation (4).
            # [batch, 1, query_len, key_len']. Range is inclusive.
            d_ji = jnp.clip(
                2 * num_directional_buckets - relative_pos, 0, 2 * num_directional_buckets - 1
            )
            # [batch, num_heads, key_len, key_len'].
            p2c_score = jnp.take_along_axis(p2c_score, d_ji, axis=-1)
            score += p2c_score.transpose(0, 1, 3, 2)

        return score


class DeBERTaV2RelativePositionalEmbedding(Embedding):
    """DeBERTa V2 Relative Embeddings.

    Reference:
    https://github.com/huggingface/transformers/blob/2d956958252617a178a68a06582c99b133fe7d3d/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L470
    """

    @config_class
    class Config(Embedding.Config):
        norm: InstantiableConfig = LayerNorm.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("norm", cfg.norm.set(input_dim=cfg.dim))

    def embeddings(self) -> Tensor:
        x = super().embeddings()
        x = self.norm(x)
        return x


class DeBERTaV2Encoder(Encoder):
    """DeBERTa V2 Encoder, a variant of BERT Encoder with a shared relative position embedding.

    Reference:
    https://github.com/microsoft/DeBERTa/blob/c558ad99373dac695128c9ec45f39869aafd374e/DeBERTa/deberta/bert.py#L129
    """

    @config_class
    class Config(Encoder.Config):
        # Relative position embedding shared across the transformer layer stack.
        relative_pos_emb: Embedding.Config = DeBERTaV2RelativePositionalEmbedding.default_config()

    @classmethod
    def default_config(cls: type["DeBERTaV2Encoder"]) -> Encoder.Config:
        cfg = super().default_config()
        cfg.transformer = RepeatedTransformerLayer.default_config()
        cfg.transformer.layer.self_attention.attention = (
            DisentangledSelfAttention.default_config().set(
                pos_emb=RedirectToSharedModule.default_config().set(
                    shared_module="relative_pos_emb",
                    method_map=dict(embeddings="embeddings"),
                ),
            )
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("relative_pos_emb", cfg.relative_pos_emb.set(dim=cfg.dim))
        self._share_with_descendants(self.relative_pos_emb, shared_module_name="relative_pos_emb")


def deberta_v2_self_attention_config(
    *,
    num_heads: int,
    max_distance: int,
    hidden_dropout: float = 0.1,
    attention_dropout: float = 0.1,
    share_projections: bool = True,
    num_directional_buckets: Optional[int] = None,
    attention_type: Optional[
        set[DisentangledAttentionType]
    ] = (  # pytype: disable=annotation-type-mismatch
        DisentangledAttentionType.P2C,
        DisentangledAttentionType.C2P,
    ),
    base_cfg: Optional[DisentangledSelfAttention.Config] = None,
) -> DisentangledSelfAttention.Config:
    """Builds configs for DeBERTaV2 Self Attention.

    Args:
        num_heads: Number of attention heads.
        max_distance: Max position in either direction for relative position bucketing.
            See also `deberta_relative_position_bucket`.
        hidden_dropout: Dropout applied to positional embeddings.
        attention_dropout: Dropout applied to attention probs.
        share_projections: Share QKV projections.
        num_directional_buckets: Number of buckets in either direction for relative position
            bucketing. See also `deberta_relative_position_bucket`.
        attention_type: Type(s) of disentangled attention to compute scores for.
            Content-to-content attention is always computed;
            position-to-position attention is not used in paper.
            Hence, default is P2C and C2P.
            References:
            https://arxiv.org/pdf/2006.03654.pdf Section 3.1
            https://huggingface.co/microsoft/deberta-v3-base/blob/main/config.json#L14

        base_cfg: Optional base config. If provided, it will be cloned.

    Returns:
        The self attention config.
    """
    cfg = base_cfg.clone() if base_cfg else DisentangledSelfAttention.default_config()
    cast(DisentangledSelfAttention.Config, cfg)
    cfg.set(
        num_heads=num_heads,
        max_distance=max_distance,
        num_directional_buckets=num_directional_buckets,
        attention_type=attention_type,
    )
    cfg.dropout.rate = attention_dropout
    cfg.pos_emb_dropout.rate = hidden_dropout
    if share_projections:
        # TODO(markblee): Test FusedQKVLinear.
        assert isinstance(cfg.input_linear, QKVLinear.Config)
        cfg.pos_q_proj = RedirectToSharedModule.default_config().set(
            shared_module="q_proj",
        )
        cfg.pos_k_proj = RedirectToSharedModule.default_config().set(
            shared_module="k_proj",
        )
    return cfg


def deberta_v2_encoder_config(
    *,
    dim: int,
    vocab_size: int,
    num_layers: int,
    max_distance: int,
    max_position_embeddings: Optional[int] = None,
    num_directional_buckets: Optional[int] = None,
    base_cfg: Optional[DeBERTaV2Encoder.Config] = None,
    stack_cls: Optional[type[BaseStackedTransformerLayer]] = None,
) -> DeBERTaV2Encoder.Config:
    """Builds configs for DeBERTaV2 Encoder.

    See also:
    https://github.com/microsoft/DeBERTa/blob/994f643ec20db09e33235496dfd0144643479dff/experiments/language_model/deberta_base.json

    Args:
        dim: Embedding dim.
        vocab_size: Vocab size.
        num_layers: Number of transformer layers.
        max_distance: Max position in either direction for relative position bucketing.
            Note that this is different from `max_position_embeddings`, which corresponds to
            sequence length. See also `deberta_relative_position_bucket`.
        max_position_embeddings: Number of positional embeddings.
            If None, position embedding is not used.
        num_directional_buckets: Number of buckets in either direction for relative position
            bucketing. See also `deberta_relative_position_bucket`.
        base_cfg: Optional base config. If provided, it will be cloned.
        stack_cls: Optional transformer stack type. Defaults to a RepeatedTransformerLayer.

    Returns:
        The encoder config.
    """
    cfg = base_cfg.clone() if base_cfg else DeBERTaV2Encoder.default_config()
    cfg.set(vocab_size=vocab_size, dim=dim)
    cfg.relative_pos_emb.set(num_embeddings=(num_directional_buckets or max_distance) * 2)

    if stack_cls:
        # Regardless of stack_cls, reuse encoder's default_config()'s layer
        # to inherit DisentangledSelfAttention.
        cfg.transformer = stack_cls.default_config().set(layer=cfg.transformer.layer)

    cfg.transformer.set(num_layers=num_layers)
    cfg.transformer.layer.self_attention.set(structure="postnorm")
    cfg.transformer.layer.feed_forward.set(
        structure="postnorm",
        activation="nn.gelu",
        hidden_dim=scaled_hidden_dim(4),
    )
    # TODO(markblee): DeBERTa supports projecting emb dim. It doesn't seem to be used by default.
    # https://github.com/microsoft/DeBERTa/blob/c558ad99373dac695128c9ec45f39869aafd374e/DeBERTa/deberta/bert.py#L236-L237
    cfg.emb = bert_embedding_config(
        max_position_embeddings=max_position_embeddings,
        layer_norm_epsilon=1e-7,
    )
    return cfg


def deberta_v2_model_config(
    *,
    encoder_cfg: DeBERTaV2Encoder.Config,
    head_cfg: Optional[BaseClassificationHead.Config] = None,
) -> BertModel.Config:
    """Builds a config for the DeBERTaV2 model.

    Args:
        encoder_cfg: Config for the DeBERTaV2 encoder.
        head_cfg: Optional head config. Defaults to a BertLMHead.Config.

    Returns:
        A config used to instantiate the DeBERTaV2 model.
    """
    cfg = bert_model_config(
        vocab_size=encoder_cfg.vocab_size,
        hidden_dim=encoder_cfg.dim,
        encoder_cfg=encoder_cfg,
        head_cfg=head_cfg,
    )
    # Initializer that is consistent with Hugging Face DeBERTaV2:
    # https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L929
    # https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/deberta_v2/configuration_deberta_v2.py#L124
    cfg.param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan=None, scale=0.02, distribution="normal"
            )
        }
    )
    return cfg

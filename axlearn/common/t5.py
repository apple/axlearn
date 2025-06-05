# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/t5x:
# Copyright 2022 The T5X Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A replication of the T5 1.1 model.

Differences:
- attend_dtype is not supported:
https://github.com/google-research/t5x/blob/03dfc44be7f9a93d34c1d7fd6f896d1c364a7d4d/t5x/examples/t5/layers.py#L518-L519
"""
import math
from typing import Optional, Union

from jax import numpy as jnp

from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    apply_attention_logit_biases,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer, Module, Tensor
from axlearn.common.config import REQUIRED, FunctionConfigBase, Required, config_class
from axlearn.common.decoder import Decoder, LmHead
from axlearn.common.encoder import Encoder
from axlearn.common.encoder_decoder import EncoderDecoderModel
from axlearn.common.layers import (
    Dropout,
    Embedding,
    RedirectToSharedModule,
    RMSNorm,
    set_bias_recursively,
    set_norm_recursively,
)
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer


def t5_relative_position_bucket(
    relative_position: Tensor,
    *,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> Tensor:
    """Computes relative position buckets with the T5 algorithm.

    Based on HuggingFace code:
    https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/t5/modeling_t5.py#L346-L392

    Translate relative position to a bucket number for relative attention. The relative position is
    defined as key_position - query_position, i.e. the distance in tokens from the attending
    position to the attended-to position.

    If bidirectional=False, then positive relative positions are invalid. We use smaller buckets
    for small absolute relative_position and larger buckets for larger absolute relative_positions.
    All relative positions >= max_distance map to the same bucket. All relative positions <=
    -max_distance map to the same bucket. This should allow for more graceful generalization to
    longer sequences than the model has been trained on.

    Args:
        relative_position: An int32 Tensor of any shape.
        bidirectional: A boolean - whether the attention is bidirectional.
        num_buckets: An integer.
        max_distance: An integer.

    Returns:
        A Tensor with the same shape as relative_position, containing int32 values in the range
        [0, num_buckets).
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(jnp.int32) * num_buckets
        relative_position = jnp.abs(relative_position)
    else:
        relative_position = -jnp.minimum(relative_position, jnp.zeros_like(relative_position))
    # Now relative_position is in the range [0, inf).

    # Half of the buckets are for exact increments in positions.
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to
    # max_distance.
    relative_position_if_large = max_exact + (
        jnp.log(relative_position.astype(jnp.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    relative_position_if_large = jnp.minimum(
        relative_position_if_large,
        num_buckets - 1,
    )

    relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class T5RelativePositionalEmbedding(BaseLayer):
    """Computes relative positional embeddings for T5.

    See
    https://github.com/google-research/t5x/blob/c6b9edfdba5dec272b82dbd2d75804324010dffd/t5x/examples/t5/layers.py#L522
    """

    @config_class
    class Config(BaseLayer.Config):
        # Number of learnable biases per relative position bucket. This is usually set to
        # number of attention heads.
        dim: Required[int] = REQUIRED
        num_buckets: int = 32
        max_distance: int = 128
        # Whether attention is bidirectional. Should be True for encoder and False for decoder.
        bidirectional: Required[bool] = REQUIRED

    @classmethod
    def default_config(cls: type["T5RelativePositionalEmbedding"]) -> Config:
        cfg = super().default_config()
        # https://github.com/google-research/t5x/blob/c6b9edfdba5dec272b82dbd2d75804324010dffd/t5x/examples/t5/network.py#L229-L236.
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan="fan_avg",
                    distribution="uniform",
                    scale=1.0,
                )
            }
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "emb", Embedding.default_config().set(num_embeddings=cfg.num_buckets, dim=cfg.dim)
        )

    def forward(self, attention_logit_biases: Tensor) -> Tensor:
        """Applies relative positional biases to 'attention_logit_biases'.

        Args:
            attention_logit_biases: [batch_size, num_heads, query_seq_len, key_seq_len].
                -inf means masked (disabled) position pairs, 0 means enabled pairs.

        Returns:
            A float bias tensor of shape [batch_size, dim, query_seq_len, key_seq_len], where -inf
            means masked pairs, other values represent positional biases.
        """
        query_seq_len, key_seq_len = attention_logit_biases.shape[-2:]
        # [query_seq_len, 1].
        query_positions = jnp.expand_dims(jnp.arange(0, query_seq_len), -1)
        # [1, key_seq_len].
        key_positions = jnp.expand_dims(jnp.arange(0, key_seq_len), 0)
        # Shape: [query_seq_len, key_seq_len].
        # relative_positions[i, j] = key_positions[j] - query_positions[i].
        relative_positions = key_positions - query_positions
        # [query_seq_len, key_seq_len] with values in [0, num_buckets).
        relative_buckets = t5_relative_position_bucket(
            relative_positions, bidirectional=self.config.bidirectional
        )
        # [query_seq_len, key_seq_len, dim].
        bias = self.emb(relative_buckets)
        # [dim, query_seq_len, key_seq_len].
        bias = bias.transpose(2, 0, 1)
        # [1, dim, query_seq_len, key_seq_len].
        bias = jnp.expand_dims(bias, 0)
        self.vlog(3, "bias=%s", bias[0, 0, 0, :])
        self.vlog(3, "mask=%s", attention_logit_biases[0, 0, 0, :])
        return apply_attention_logit_biases(bias, attention_logit_biases)


class EncoderOutputLayer(BaseLayer):
    """Performs the output transformations (final_norm + dropout) in the T5 encoder."""

    @config_class
    class Config(BaseLayer.Config):
        input_dim: Required[int] = REQUIRED
        norm: RMSNorm.Config = RMSNorm.default_config()
        dropout: Dropout.Config = Dropout.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("norm", cfg.norm.clone(input_dim=cfg.input_dim))
        self._add_child("dropout", cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.dropout(x)


def t5_transformer_stack_config(
    arch: str = "t5-v1-1",
    hidden_dim: Optional[Union[int, FunctionConfigBase]] = None,
    base_cfg: Optional[BaseStackedTransformerLayer.Config] = None,
) -> BaseStackedTransformerLayer.Config:
    cfg = base_cfg or StackedTransformerLayer.default_config()
    cfg.layer.feed_forward.set(structure="prenorm")
    if arch == "t5-v1-1":
        cfg.layer.feed_forward.set(
            activation=("nn.gelu", "linear"),  # GeGLU
            # With feed_forward.hidden_dim = input_dim * 8 / 3, the feed-forward network with GLU
            # activations have a total of (input_dim ** 2) * 8 weight parameters, equal to the
            # number of weight params of a canonical non-GLU feed-forward network where
            # feed_forward.hidden_dim = input_dim * 4.
            #
            # See section 2 of https://arxiv.org/abs/2002.05202.
            hidden_dim=scaled_hidden_dim(8.0 / 3),
        )
    elif arch == "t5-v1":
        cfg.layer.feed_forward.set(
            activation="nn.relu",
            # Reference: https://github.com/google-research/t5x/tree/main/t5x/examples/t5/t5_1_0
            hidden_dim=scaled_hidden_dim(4),
        )
    elif arch == "t5-ul2":
        cfg.layer.feed_forward.set(
            # Ref: https://huggingface.co/google/ul2/blob/main/config.json#L13.
            activation=("nn.silu", "linear"),
            # Same as t5-v1-1 hidden_dim.
            hidden_dim=scaled_hidden_dim(8.0 / 3),
        )
    else:
        raise ValueError(f"Unsupported arch {arch}.")
    # Note that not all T5 architectures follow the `scaled_hidden_dim` rule, e.g. T5 1.1 "xl" uses
    # feed_forward_dim of 5120 for a hidden_dim of 2048. In these cases, we allow the caller to set
    # the feed_forward_dim directly.
    if hidden_dim is not None:
        cfg.layer.feed_forward.set(hidden_dim=hidden_dim)
    # T5 linear layers do not use bias.
    set_bias_recursively(cfg, False)
    return cfg


class T5Encoder(Encoder):
    """A T5 Encoder.

    Reference:
    https://github.com/google-research/t5x/blob/23ab1b68e608370ba94f01191b4803bf78ad65bc/t5x/examples/t5/network.py#L174-L209
    """

    @config_class
    class Config(Encoder.Config):
        relative_pos_emb: T5RelativePositionalEmbedding.Config = (
            T5RelativePositionalEmbedding.default_config().set(bidirectional=True)
        )

    @classmethod
    def default_config(cls: type["T5Encoder"]) -> Encoder.Config:
        cfg = super().default_config()  # type: T5Encoder.Config
        cfg.transformer = t5_transformer_stack_config()
        cfg.output = EncoderOutputLayer.default_config()
        # TODO(ruoming, mark): set this to None.
        cfg.pad_token_id = 0
        # T5 uses RMSNorm instead of LayerNorm.
        set_norm_recursively(cfg, RMSNorm.default_config())
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("relative_pos_emb", cfg.relative_pos_emb)

    def compute_attention_logit_biases(
        self,
        input_ids: Tensor,
        *,
        segment_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        attention_logit_biases = super().compute_attention_logit_biases(
            input_ids, segment_ids=segment_ids, positions=positions
        )
        return self.relative_pos_emb(attention_logit_biases)


def t5_encoder_config(
    *,
    vocab_size: int,
    dim: int,
    num_layers: int,
    num_attention_heads: int,
    base_cfg: Optional[T5Encoder.Config] = None,  # Keep this at the end.
) -> T5Encoder.Config:
    cfg = base_cfg.clone() if base_cfg else T5Encoder.default_config()
    cfg.set(vocab_size=vocab_size, dim=dim)
    cfg.relative_pos_emb.dim = num_attention_heads
    cfg.transformer.set(num_layers=num_layers)
    cfg.transformer.layer.self_attention.attention.num_heads = num_attention_heads
    # T5 linear layers do not use bias.
    set_bias_recursively(cfg, False)
    # T5 uses RMSNorm instead of LayerNorm.
    set_norm_recursively(cfg, RMSNorm.default_config())
    return cfg


class T5Decoder(Decoder):
    """A T5 Decoder.

    Reference:
    https://github.com/google-research/t5x/blob/23ab1b68e608370ba94f01191b4803bf78ad65bc/t5x/examples/t5/network.py#L174-L209
    """

    @config_class
    class Config(Decoder.Config):
        relative_pos_emb: T5RelativePositionalEmbedding.Config = (
            T5RelativePositionalEmbedding.default_config().set(bidirectional=False)
        )

    @classmethod
    def default_config(cls: type["T5Decoder"]) -> Decoder.Config:
        cfg = super().default_config()  # type: T5Decoder.Config
        cfg.transformer = t5_transformer_stack_config()
        # T5 uses RMSNorm instead of LayerNorm.
        cfg.output_norm = RMSNorm.default_config()
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("relative_pos_emb", cfg.relative_pos_emb)

    def compute_attention_logit_biases(
        self,
        input_ids: Tensor,
        *,
        segment_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        attention_logit_biases = super().compute_attention_logit_biases(
            input_ids, segment_ids=segment_ids, positions=positions
        )
        return self.relative_pos_emb(attention_logit_biases)


def t5_decoder_config(
    *,
    vocab_size: int,
    dim: int,
    num_layers: int,
    num_attention_heads: int,
    cross_attention: Optional[TransformerAttentionLayer.Config],
    base_cfg: Optional[T5Decoder.Config] = None,  # Keep this at the end.
) -> T5Decoder.Config:
    cfg = base_cfg.clone() if base_cfg else T5Decoder.default_config()
    cfg.set(vocab_size=vocab_size, dim=dim)
    cfg.relative_pos_emb.dim = num_attention_heads
    cfg.transformer.set(num_layers=num_layers)
    cfg.transformer.layer.self_attention.attention.num_heads = num_attention_heads
    if cross_attention is not None:
        cfg.transformer.layer.cross_attention = cross_attention
    # logits_via_embedding = False for T5 1.1:
    # https://github.com/google-research/t5x/blob/c5fddbe512838153fec602010db736a36dcf7576/t5x/examples/t5/t5_1_1/base.gin#L55.
    cfg.lm_head = LmHead.default_config()
    # T5 linear layers do not use bias.
    set_bias_recursively(cfg, False)
    # T5 uses RMSNorm instead of LayerNorm.
    set_norm_recursively(cfg, RMSNorm.default_config())
    return cfg


class T5EncoderDecoderModel(EncoderDecoderModel):
    """A T5 Encoder-Decoder Model.

    Reference:
    https://github.com/google-research/t5x/blob/c6b9edfdba5dec272b82dbd2d75804324010dffd/t5x/examples/t5/network.py#L278
    """

    @config_class
    class Config(EncoderDecoderModel.Config):
        shared_token_emb: Embedding.Config = Embedding.default_config()

    @classmethod
    def default_config(cls: type["T5EncoderDecoderModel"]) -> Config:
        cfg = super().default_config()  # type: T5EncoderDecoderModel.Config
        cfg.encoder = T5Encoder.default_config()  # type: T5Encoder.Config
        cfg.decoder = T5Decoder.default_config()  # type: T5Decoder.Config
        redirecting_token_emb_cfg = RedirectToSharedModule.default_config().set(
            shared_module="shared_token_emb",
        )
        cfg.encoder.emb.token_emb = redirecting_token_emb_cfg
        cfg.decoder.emb.token_emb = redirecting_token_emb_cfg
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("shared_token_emb", cfg.shared_token_emb)
        self._share_with_descendants(self.shared_token_emb, shared_module_name="shared_token_emb")


def t5_encoder_decoder_config(
    *,
    vocab_size: int,
    dim: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    num_attention_heads: int,
    dropout_rate: float,
    z_loss_scale: float = 0,
    label_smoothing: float = 0,
    stack_cfg: Optional[BaseStackedTransformerLayer.Config] = None,
) -> EncoderDecoderModel.Config:
    cfg = T5EncoderDecoderModel.default_config()  # type: T5EncoderDecoderModel.Config
    cfg.z_loss_scale = z_loss_scale
    cfg.label_smoothing = label_smoothing
    cfg.shared_token_emb.set(num_embeddings=vocab_size, dim=dim)
    cfg.encoder = t5_encoder_config(
        base_cfg=cfg.encoder.set(transformer=stack_cfg or cfg.encoder.transformer),
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_encoder_layers,
        num_attention_heads=num_attention_heads,
    ).set(dropout_rate=dropout_rate)
    cross_attention = TransformerAttentionLayer.default_config()
    cross_attention.attention.num_heads = num_attention_heads
    cfg.decoder = t5_decoder_config(
        base_cfg=cfg.decoder.set(transformer=stack_cfg or cfg.decoder.transformer),
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_decoder_layers,
        num_attention_heads=num_attention_heads,
        cross_attention=cross_attention,
    ).set(dropout_rate=dropout_rate)
    # T5 linear layers do not use bias.
    set_bias_recursively(cfg, False)
    # T5 uses RMSNorm instead of LayerNorm. Paper uses an eps of 1e-6.
    # HF ref:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py#L67
    # T5X ref:
    # https://github.com/google-research/t5x/blob/main/t5x/examples/t5/layers.py#L643
    set_norm_recursively(cfg, RMSNorm.default_config().set(eps=1e-6))
    return cfg

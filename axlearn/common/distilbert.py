# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# huggingface/transformers:
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

"""DistilBert Text Encoder

The DistilBert configs a TextEmbeddingEncoder.
We can use it by configuring the TextEmbeddingStreamEncoder in text_dual_encoder.py

Ref: https://arxiv.org/pdf/1910.01108.pdf
Implementation Ref:
https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/distilbert/modeling_distilbert.py
"""

from typing import Union

from axlearn.common.attention import (
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    build_remat_spec,
    scaled_hidden_dim,
)
from axlearn.common.bert import bert_embedding_config, bert_model_config, bert_transformer_config
from axlearn.common.config import FunctionConfigBase
from axlearn.common.layers import set_dropout_rate_recursively
from axlearn.common.poolings import BasePoolingLayer, FirstNTokenPooling
from axlearn.common.text_encoder import TextEmbeddingEncoder


def set_distilbert_config(
    *,
    vocab_size: int = 30522,
    num_layers: int = 6,
    model_dim: int = 768,
    num_heads: int = 12,
    max_seq_len: int = 512,
    dropout_rate: float = 0.1,
    pad_token_id: int = 0,
    pooler_config: BasePoolingLayer.Config = FirstNTokenPooling.default_config(),
    feed_forward_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4),
    feed_forward_act: str = "nn.gelu",
    layer_norm_eps: float = 1e-12,
    remat: bool = False,
) -> TextEmbeddingEncoder.Config:
    """Configure the distilbert.

    Default hyperparams are from HuggingFace open-source implementation.

    Args:
        vocab_size: An integer for the size of the vocabulary in text tokenizer.
        num_layers: An integer indicating the number of transformer blocks.
        model_dim: An integer for the output dimension of the transformer block.
        num_heads: An integer for the number of the attention heads.
        max_seq_len: An integer for the maximum length of the tokenized text.
        dropout_rate: The dropout rate of the text encoder.
        pad_token_id: The token_id for the padded tokens.
        pooler_config: An instantiable BasePoolingLayer configuration used for embedding pooling.
        feed_forward_dim: The dimension of the feedforward layer in transformer.
             It can be set as an integer or as a scaled_hidden_dim function.
        feed_forward_act: The nonlinear function used in the feedforward layer in transformer block.
        layer_norm_eps: The eps used in the layer norm.
        remat: A boolean for enabling the gradient checkpointing.

    Returns:
        A instantiable distilbert text encoder.
    """
    text_encoder_cfg = TextEmbeddingEncoder.default_config()
    text_encoder_cfg.output_dim = model_dim
    text_encoder_cfg.pad_token_id = pad_token_id
    text_encoder_cfg.pooler = pooler_config

    if remat:
        base_transformer_cfg = RepeatedTransformerLayer.default_config()
    else:
        base_transformer_cfg = StackedTransformerLayer.default_config()

    # Config the encoder.
    text_encoder_cfg.encoder = bert_model_config(
        vocab_size=vocab_size,
        hidden_dim=model_dim,
        embedding_cfg=bert_embedding_config(
            max_position_embeddings=max_seq_len,
            layer_norm_epsilon=layer_norm_eps,
        ),
        stack_cfg=bert_transformer_config(
            base_cfg=base_transformer_cfg,
            num_layers=num_layers,
            num_heads=num_heads,
            layer_norm_epsilon=layer_norm_eps,
        ),
    ).encoder

    text_encoder_cfg.encoder.transformer.layer.feed_forward.activation = feed_forward_act
    text_encoder_cfg.encoder.transformer.layer.feed_forward.hidden_dim = feed_forward_dim

    if remat:
        text_encoder_cfg.encoder.transformer.layer.remat_spec = build_remat_spec(
            text_encoder_cfg.encoder.transformer
        )

    set_dropout_rate_recursively(text_encoder_cfg, dropout_rate)
    return text_encoder_cfg

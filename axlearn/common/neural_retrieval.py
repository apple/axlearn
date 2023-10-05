# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# huggingface/transformers:
# Copyright 2018 DPR Authors, The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Dense Passage Retrieval.

Ref: https://arxiv.org/abs/2004.04906
Implementation Ref:
    https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/dpr/modeling_dpr.py
"""

from typing import Optional

from axlearn.common.config import InstantiableConfig
from axlearn.common.layers import RedirectToSharedModule
from axlearn.common.text_dual_encoder import (
    TEXT_DUAL_ENCODER_SHARED_MODULE_NAME,
    TextEmbeddingAsymmetricContrastiveLossLayer,
    TextEmbeddingDualEncoder,
    TextEmbeddingStreamEncoder,
)
from axlearn.common.utils_text_dual_encoder import (
    HF_PARAM_INIT,
    bert_text_embedding_stream_encoder_config,
)

QUERY_ENCODER_NAME = "query_encoder"
PASSAGE_ENCODER_NAME = "passage_encoder"


def set_bert_dpr_encoder_config(
    *,
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    max_seq_len: int = 512,
    pad_token_id: int = 0,
    feed_forward_act: str = "exact_gelu",
    output_norm: Optional[InstantiableConfig] = None,
    remat: bool = False,
) -> TextEmbeddingStreamEncoder.Config:
    """Configure the Bert based DPR stream encoder.

    Default hyperparams are from HuggingFace open-source implementation.

    Args:
        vocab_size: An integer for the size of the vocabulary in text tokenizer.
        num_layers: An integer indicating the number of transformer blocks.
        model_dim: An integer for the output dimension of the transformer block.
        num_heads: An integer for the number of the attention heads.
        max_seq_len: An integer for the maximum length of the tokenized text.
        pad_token_id: The token_id for the padded tokens.
        feed_forward_act: The nonlinear function used in the feedforward layer in transformer block.
        output_norm: The normalization applied on the output embeddings. Default is None.
        remat: If True, use RepeatedTransformerLayer instead of StackedTransformerLayer and remat
            spec to save memory.

    Returns:
        An instantiable Bert based DPR text encoder.
    """
    text_encoder_cfg = bert_text_embedding_stream_encoder_config(
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=model_dim,
        output_dim=model_dim,
        output_proj=None,
        output_norm=output_norm,
        remat=remat,
    )
    text_encoder_cfg.text_encoder.encoder.transformer.layer.feed_forward.activation = (
        feed_forward_act
    )
    text_encoder_cfg.text_encoder.encoder.emb.token_emb.param_init = HF_PARAM_INIT
    text_encoder_cfg.text_encoder.encoder.emb.pos_emb.param_init = HF_PARAM_INIT
    text_encoder_cfg.text_encoder.encoder.emb.type_emb.param_init = HF_PARAM_INIT
    return text_encoder_cfg


def set_dpr_model_config(
    *,
    query_encoder_cfg: TextEmbeddingStreamEncoder.Config,
    passage_encoder_cfg: TextEmbeddingStreamEncoder.Config,
    contrastive_loss_scale_factor: float = 1.0,
) -> TextEmbeddingStreamEncoder.Config:
    """Configures the DPR dual encoder.

    Args:
        query_encoder_cfg: An instantiable config for query encoder.
        passage_encoder_cfg: An instantiable config for passage encoder.
        contrastive_loss_scale_factor: The reciprocal of the temperature for the
            contrastive learning loss.

    Returns:
        An instantiable config for DPR dual encoder.
    """

    contrastive_loss_layer_cfg = TextEmbeddingAsymmetricContrastiveLossLayer.default_config().set(
        left_encoder_name=QUERY_ENCODER_NAME,
        right_encoder_name=PASSAGE_ENCODER_NAME,
        contrastive_loss_scale_factor=contrastive_loss_scale_factor,
    )

    dpr_stream_encoder = {
        QUERY_ENCODER_NAME: query_encoder_cfg,
        PASSAGE_ENCODER_NAME: passage_encoder_cfg,
    }
    dpr_fusion_network = {"contrastive_fusion_network": contrastive_loss_layer_cfg}
    dpr_model = TextEmbeddingDualEncoder.default_config().set(
        param_init=HF_PARAM_INIT,
        stream_encoder=dpr_stream_encoder,
        fusion_network=dpr_fusion_network,
    )
    return dpr_model


def set_siamese_dpr_model_config(
    *,
    shared_encoder_cfg: TextEmbeddingStreamEncoder.Config,
    contrastive_loss_scale_factor: float = 1.0,
) -> TextEmbeddingStreamEncoder.Config:
    """Configures the siamese DPR dual encoder. By default, we consider
       passage encoder share the query encoder.

    Args:
        shared_encoder_cfg: An instantiable config for the shared encoder.
        contrastive_loss_scale_factor: The reciprocal of the temperature for the
            contrastive learning loss.

    Returns:
        An instantiable config for DPR dual encoder with shared encoder.
    """

    contrastive_loss_layer_cfg = TextEmbeddingAsymmetricContrastiveLossLayer.default_config().set(
        left_encoder_name=QUERY_ENCODER_NAME,
        right_encoder_name=PASSAGE_ENCODER_NAME,
        contrastive_loss_scale_factor=contrastive_loss_scale_factor,
    )
    passage_encoder_cfg = RedirectToSharedModule.default_config().set(
        shared_module=TEXT_DUAL_ENCODER_SHARED_MODULE_NAME
    )

    dpr_stream_encoder = {
        QUERY_ENCODER_NAME: shared_encoder_cfg,
        PASSAGE_ENCODER_NAME: passage_encoder_cfg,
    }
    dpr_fusion_network = {"contrastive_fusion_network": contrastive_loss_layer_cfg}
    dpr_model = TextEmbeddingDualEncoder.default_config().set(
        param_init=HF_PARAM_INIT,
        stream_encoder=dpr_stream_encoder,
        fusion_network=dpr_fusion_network,
        shared_encoder_name=QUERY_ENCODER_NAME,
    )
    return dpr_model

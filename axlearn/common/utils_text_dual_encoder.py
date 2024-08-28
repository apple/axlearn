# Copyright © 2023 Apple Inc.

"""Text dual encoder utils."""
from typing import Optional

from axlearn.common.attention import (
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    build_remat_spec,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.bert import bert_embedding_config, bert_model_config, bert_transformer_config
from axlearn.common.config import config_for_function
from axlearn.common.layers import L2Norm, Linear
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.common.poolings import BasePoolingLayer, FirstNTokenPooling
from axlearn.common.state_builder import (
    Builder,
    ChainBuilder,
    Converter,
    HuggingFacePreTrainedBuilder,
    RestoreAndConvertBuilder,
)
from axlearn.common.text_dual_encoder import TextEmbeddingStreamEncoder
from axlearn.common.text_encoder import TextEmbeddingEncoder
from axlearn.huggingface.hf_module import download_hf_models_from_remote

# Initializer that is consistent with Huggingface's initialization for BERT model.
HF_PARAM_INIT = DefaultInitializer.default_config().set(
    init_by_param_name={
        PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
            fan=None, scale=0.02, distribution="normal"
        )
    }
)


def bert_text_embedding_stream_encoder_config(
    *,
    pad_token_id: int,
    vocab_size: int,
    max_seq_len: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    output_dim: int,
    output_proj: Optional[Linear.Config] = Linear.default_config().set(bias=True),
    output_norm: Optional[BaseLayer.Config] = L2Norm.default_config(),
    pooler: BasePoolingLayer.Config = FirstNTokenPooling.default_config(),
    remat: bool = False,
) -> TextEmbeddingStreamEncoder.Config:
    """Constructs Config for BERT-based TextEmbeddingStreamEncoder.

    Args:
        pad_token_id: PAD id.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length after tokenization.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension from the TextEmbeddingStreamEncoder.
        output_proj: Optional linear projection layer applied on embedding from text_encoder. If
            None, embedding from text_encoder is taken as it is.
        output_norm: Optional normalization layer applied on embedding from text_encoder and after
            potential projection layer. If None, no normalization will be applied.
        pooler: Pooler of the underlying TextEmbeddingEncoder.
        remat: If True, use RepeatedTransformerLayer instead of StackedTransformerLayer and remat
            spec to save memory.

    Returns:
        BERT-based TextEmbeddingStreamEncoder Config.
    """
    text_encoder_cfg = TextEmbeddingStreamEncoder.default_config().set(
        text_encoder=TextEmbeddingEncoder.default_config().set(
            pad_token_id=pad_token_id,
            encoder=bert_model_config(
                vocab_size=vocab_size,
                dropout_rate=0.1,
                embedding_cfg=bert_embedding_config(
                    max_position_embeddings=max_seq_len,
                    type_vocab_size=2,
                    layer_norm_epsilon=1e-12,
                ),
                stack_cfg=bert_transformer_config(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    layer_norm_epsilon=1e-12,
                    base_cfg=(
                        RepeatedTransformerLayer.default_config()
                        if remat
                        else StackedTransformerLayer.default_config()
                    ),
                ),
            ).encoder,
            pooler=pooler,
        ),
        output_dim=output_dim,
        output_proj=output_proj,
        output_norm=output_norm,
        hidden_dim=hidden_dim,
    )
    if remat:
        transformer_cfg = text_encoder_cfg.text_encoder.encoder.transformer
        transformer_cfg.layer.remat_spec = build_remat_spec(transformer_cfg)

    return text_encoder_cfg


################################################################################
# init_state_builder related configs                                           #
################################################################################


def init_state_builder_config(
    *,
    left_encoder_init_state_builder: Optional[Builder.Config] = None,
    right_encoder_init_state_builder: Optional[Builder.Config] = None,
    converter: Optional[Converter.Config] = None,
) -> Optional[Builder.Config]:
    """Constructs init_state_builder config for text dual encoder model.

    Args:
        left_encoder_init_state_builder: init_state_builder config for left encoder.
        right_encoder_init_state_builder: init_state_builder config for right encoder.
        converter: Optional converter to be applied on top of left and right encoder
            init_state_builder.

    Returns:
        A final Builder config to be applied to text dual encoder model state.
    """
    if left_encoder_init_state_builder is None and right_encoder_init_state_builder is None:
        return None

    builders = []
    if left_encoder_init_state_builder is not None:
        builders.append(left_encoder_init_state_builder)

    if right_encoder_init_state_builder is not None:
        builders.append(right_encoder_init_state_builder)

    init_state_builder = ChainBuilder.default_config().set(
        builders=builders,
    )
    if converter is not None:
        return RestoreAndConvertBuilder.default_config().set(
            builder=init_state_builder,
            converter=converter,
        )
    else:
        return init_state_builder


def hf_pretrained_builder_config(  # pylint: disable=dangerous-default-value
    *,
    model_path: str,
    target_scope: list[str] = [],
    source_scope: list[str] = ["encoder"],
) -> Builder.Config:
    """Constructs HuggingFacePreTrainedBuilder to initialize from HF models.

    The builder will replace the target model's parameters under
    target_scope1->target_scope2->... to the HF model's parameters under
    source_scope1->source_scope2->...

    Args:
        model_path: Local or gs location of the model artifacts folder.
        target_scope: List of scope path of target state.
        source_scope: List of scope path of source state.

    Returns:
        HuggingFacePreTrainedBuilder config to initialize from HF models.
    """
    # Lazily import to avoid introducing a dependency otherwise.
    # pylint: disable-next=import-outside-toplevel
    from transformers import AutoModel

    def auto_from_pretrained(model_path: str):
        model_path = download_hf_models_from_remote(model_path)
        return AutoModel.from_pretrained(model_path)

    builder = HuggingFacePreTrainedBuilder.default_config().set(
        hf_layer_config=config_for_function(auto_from_pretrained).set(model_path=model_path),
        target_scope=target_scope,
        source_scope=source_scope,
    )
    return builder


def init_state_builder_siamese_model_config(
    *,
    shared_encoder_init_state_builder: Optional[Builder.Config] = None,
    converter: Optional[Converter.Config] = None,
) -> Optional[Builder.Config]:
    """Constructs init_state_builder config for siamese text dual encoder model.

    Args:
        shared_encoder_init_state_builder: init_state_builder config for the shared encoder.
        converter: Optional converter to be applied on top of the shared encoder init_state_builder.

    Returns:
        A final Builder config to be applied to text dual encoder model state.
    """
    if shared_encoder_init_state_builder is None:
        return None

    if converter is not None:
        return RestoreAndConvertBuilder.default_config().set(
            builder=shared_encoder_init_state_builder,
            converter=converter,
        )
    else:
        return shared_encoder_init_state_builder

# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# openai/CLIP:
# Copyright (c) 2021 OpenAI.
# Licensed under the MIT license.

"""CLIP architecture implementation.

Ref: https://github.com/openai/CLIP/blob/main/clip/model.py
"""
# pylint: disable=duplicate-code

from typing import Optional, Protocol, Union

import jax.numpy as jnp
import numpy as np

from axlearn.common.attention import (
    AttentionLogitBiasLayer,
    BaseStackedTransformerLayer,
    CausalAttentionLogitBiasLayer,
    LearnedPositionalEmbedding,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    build_remat_spec,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.bert import bert_embedding_config, bert_model_config, bert_transformer_config
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
    maybe_instantiate,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import (
    BaseNormalizationLayer,
    L2Norm,
    LayerNorm,
    Linear,
    set_dropout_rate_recursively,
    set_norm_recursively,
)
from axlearn.common.loss import contrastive_logits, symmetric_contrastive_loss_from_logits
from axlearn.common.module import Module
from axlearn.common.multi_stream_model import FusionNetwork, MultiStreamModel, StreamEncoder
from axlearn.common.param_init import ConstantInitializer
from axlearn.common.poolings import BasePoolingLayer, FirstNTokenPooling, LastNTokenPooling
from axlearn.common.text_encoder import TEXT_EMBEDDINGS, TextEmbeddingEncoder
from axlearn.common.utils import NestedTensor, Tensor
from axlearn.common.vision_transformer import VisionTransformer, layer_norm_config
from axlearn.vision.mobilenets import MobileNets


class CLIPTextStreamEncoder(StreamEncoder):
    """A CLIP text stream encoder module.

    This class wraps the input_batch into the TextEmbeddingEncoder format.
    """

    @config_class
    class Config(StreamEncoder.Config):
        """Configures CLIPTextStreamEncoder."""

        # The dim of the output embeddings.
        output_dim: Required[int] = REQUIRED
        # The hidden dim of `text_encoder`. If None, uses output_dim.
        hidden_dim: Optional[int] = None
        text_encoder: TextEmbeddingEncoder.Config = TextEmbeddingEncoder.default_config()
        output_proj: Optional[Linear.Config] = None
        # Coca uses LayerNorm instead.
        output_norm: Optional[BaseNormalizationLayer.Config] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        assert cfg.text_encoder.pooler.num_outputs == 1
        hidden_dim = cfg.hidden_dim or cfg.output_dim
        self._add_child("text_encoder", cfg.text_encoder.set(output_dim=hidden_dim))
        if cfg.output_proj:
            self._add_child(
                "output_proj", cfg.output_proj.set(input_dim=hidden_dim, output_dim=cfg.output_dim)
            )
        if cfg.output_norm:
            if "input_dim" in cfg.output_norm:
                cfg.output_norm.set(input_dim=cfg.output_dim)
            self._add_child("output_norm", cfg.output_norm)

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """The forward function for the CLIPTextStreamEncoder.

        Args:
            input_batch: A dictionary containing:
                *"text": A Tensor with shape
                    [batch_size, num_sentences, num_tokens]

        Returns:
            A dictionary containing:
                *"output_features": A Tensor with shape
                    [batch_size, num_sentences, dim]
        TODO(bwzhang@) Remove the "input" to simplify the logic.
        """
        text = input_batch["text"]
        batch_size = text.shape[0]
        num_sentences = text.shape[1]
        text = text.reshape(batch_size * num_sentences, *text.shape[2:])

        output_dict = self.text_encoder(text)
        x = output_dict[TEXT_EMBEDDINGS]
        x = x.squeeze(axis=1)
        if "output_proj" in self.children:
            x = self.output_proj(x)
        if "output_norm" in self.children:
            x = self.output_norm(x)
        x = x.reshape(batch_size, num_sentences, *x.shape[1:])
        assert "output_features" not in output_dict
        output_dict["output_features"] = x
        return output_dict


class CLIPImageStreamEncoder(StreamEncoder):
    """A CLIP image stream encoder module.

    This class wraps the input_batch into the VisonTransformer format.
    """

    @config_class
    class Config(StreamEncoder.Config):
        """Configures CLIPImageStreamEncoder."""

        # The dim of the output embeddings.
        output_dim: Required[int] = REQUIRED
        # The hidden dim of `image_encoder`. If None, uses output_dim.
        hidden_dim: Optional[int] = None
        image_encoder: Union[
            VisionTransformer.Config, MobileNets.Config
        ] = VisionTransformer.default_config()
        output_proj: Optional[Linear.Config] = None
        # Coca uses LayerNorm instead.
        output_norm: Optional[BaseNormalizationLayer.Config] = None
        image_field: str = "image"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if isinstance(cfg.image_encoder, VisionTransformer.Config):
            assert cfg.image_encoder.pooler.num_outputs == 1
        hidden_dim = cfg.hidden_dim or cfg.output_dim
        self._add_child("image_encoder", cfg.image_encoder.set(output_dim=hidden_dim))
        if cfg.output_proj:
            self._add_child(
                "output_proj", cfg.output_proj.set(input_dim=hidden_dim, output_dim=cfg.output_dim)
            )
        if cfg.output_norm:
            if "input_dim" in cfg.output_norm:
                cfg.output_norm.set(input_dim=cfg.output_dim)
            self._add_child("output_norm", cfg.output_norm)

    def forward(  # pylint:disable=arguments-renamed
        self, input_batch: NestedTensor
    ) -> NestedTensor:
        """The forward function for the CLIPImageStreamEncoder.

        Args:
            input_batch: A dictionary containing:
                *"image": A Tensor with shape
                    [batch_size, num_images, height, width, channel]

        Returns:
            A dictionary containing:
                *"output_features": A Tensor with shape
                    [batch_size, num_images, dim]
        """
        cfg = self.config
        image_encoder_input = input_batch[cfg.image_field]
        batch_size = image_encoder_input.shape[0]
        num_images = image_encoder_input.shape[1]

        image_encoder_input = image_encoder_input.reshape(
            batch_size * num_images, *image_encoder_input.shape[2:]
        )

        output_dict = self.image_encoder(image_encoder_input)
        x = output_dict["embedding"]
        if "output_proj" in self.children:
            x = self.output_proj(x)
        if "output_norm" in self.children:
            x = self.output_norm(x)

        x = x.reshape(batch_size, num_images, *x.shape[1:])
        assert "output_features" not in output_dict
        output_dict["output_features"] = x
        return output_dict


def set_transformer_config(
    *,
    num_layers: int,
    num_heads: int,
    feed_forward_dim: int,
    feed_forward_act: str,
    remat: bool,
) -> BaseStackedTransformerLayer.Config:
    """Configure the Transformer Layer.
    Args:
        num_layers: An integer indicating the number of transformer blocks.
        num_heads: An integer for the number of the attention heads.
        feed_forward_dim: The dimension of the feedforward layer in transformer.
            It can be set as an integer or as a scaled_hidden_dim function.
        feed_forward_act: The nonlinear function used in the feedforward layer in transformer block.
        remat: A boolean for enabling the gradient checkpointing.

    Returns:
        A instantiable BaseStackedTransformerLayer Config.
    """
    if remat:
        transformer_cfg = RepeatedTransformerLayer.default_config()
    else:
        transformer_cfg = StackedTransformerLayer.default_config()
    # Set up the Transformer.
    transformer_cfg.num_layers = num_layers
    transformer_lyr_cfg = transformer_cfg.layer
    transformer_lyr_cfg.feed_forward.hidden_dim = feed_forward_dim
    transformer_lyr_cfg.feed_forward.activation = feed_forward_act
    transformer_lyr_cfg.self_attention.attention.num_heads = num_heads

    if remat:
        transformer_lyr_cfg.remat_spec = build_remat_spec(transformer_cfg)
    return transformer_cfg


def set_vision_encoder_config(
    *,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    projection_dim: int,  # The projection dim for the shared image-text space.
    feed_forward_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4),
    image_size: tuple[int, int] = (224, 224),
    patch_size: tuple[int, int] = (16, 16),
    dropout_rate: float = 0.1,
    pooler_config: BasePoolingLayer.Config = FirstNTokenPooling.default_config(),
    feed_forward_act: str = "nn.gelu",
    layer_norm_eps: float = 1e-5,
    num_cls_tokens: int = 1,
    encoder_cls: type[BaseLayer] = CLIPImageStreamEncoder,
    atten_logit_cap: Optional[float] = None,
    remat: bool = False,
) -> CLIPImageStreamEncoder.Config:
    """Configure the CLIP image stream encoder.

    Args:
        num_layers: An integer indicating the number of transformer blocks.
        model_dim: An integer for the output dimension of the transformer block.
        num_heads: An integer for the number of the attention heads.
        projection_dim: An integer for the dimension in the shared image-text embedidng space.
        feed_forward_dim: The dimension of the feedforward layer in transformer.
            It can be set as an integer or as a scaled_hidden_dim function.
        image_size: The size of the cropped image.
        patch_size: The size of the image patch.
        dropout_rate: The dropout rate of the image encoder.
        pooler_config: An instantiable BasePoolingLayer configuration used for embedding pooling.
        feed_forward_act: The nonlinear function used in the feedforward layer in transformer block.
        layer_norm_eps: The eps used in the layer norm.
        num_cls_tokens: An integer representing the number of CLS tokens.
        encoder_cls: The image encoder.
        atten_logit_cap: A soft capping value for attention logits.
        remat: A boolean for enabling the gradient checkpointing.

    Returns:
        A instantiable CLIP image encoder.
    """
    image_encoder_cfg = encoder_cls.default_config()
    # Set up the image tokenization module.
    image_encoder_cfg.output_dim = projection_dim
    image_encoder_cfg.hidden_dim = model_dim
    image_encoder_cfg.image_encoder.visual_embed.convert_to_sequence.set(patch_size=patch_size)
    image_encoder_cfg.image_encoder.pooler = pooler_config
    image_encoder_cfg.image_encoder.visual_embed.convert_to_sequence.conv.bias = False
    image_encoder_cfg.image_encoder.num_cls_tokens = num_cls_tokens
    image_encoder_cfg.image_encoder.encoder_1d.pos_emb.shape = [
        (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        + image_encoder_cfg.image_encoder.num_cls_tokens
    ]
    # Set up the Transformer.
    image_encoder_cfg.image_encoder.encoder_1d.transformer = set_transformer_config(
        num_layers=num_layers,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        feed_forward_act=feed_forward_act,
        remat=remat,
    )
    transformer_cfg = image_encoder_cfg.image_encoder.encoder_1d.transformer.layer
    transformer_cfg.self_attention.attention.atten_logit_cap = atten_logit_cap

    set_norm_recursively(image_encoder_cfg, LayerNorm.default_config().set(eps=layer_norm_eps))
    image_encoder_cfg.image_encoder.encoder_1d.set(input_norm=layer_norm_config(eps=layer_norm_eps))
    set_dropout_rate_recursively(image_encoder_cfg, dropout_rate)
    image_encoder_cfg.output_proj = Linear.default_config().set(bias=False)
    image_encoder_cfg.output_norm = L2Norm.default_config()
    return image_encoder_cfg


def set_text_encoder_config(
    *,
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    max_seq_len: int,
    projection_dim: int,  # The projection dim for the shared image-text space.
    feed_forward_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4),
    dropout_rate: float = 0.1,
    pad_token_id: int = 0,
    pooler_config: BasePoolingLayer.Config = LastNTokenPooling.default_config(),
    attention_mask_config: AttentionLogitBiasLayer.Config = (
        CausalAttentionLogitBiasLayer.default_config()
    ),
    feed_forward_act: str = "nn.gelu",
    layer_norm_eps: float = 1e-5,
    atten_logit_cap: Optional[float] = None,
    remat: bool = False,
) -> CLIPTextStreamEncoder.Config:
    """Configure the CLIP text stream encoder.

    Args:
        vocab_size: An integer for the size of the vocabulary in text tokenizer.
        num_layers: An integer indicating the number of transformer blocks.
        model_dim: An integer for the output dimension of the transformer block.
        num_heads: An integer for the number of the attention heads.
        max_seq_len: An integer for the maximum length of the tokenized text.
        projection_dim: An integer for the dimension in the shared image-text embedidng space.
        feed_forward_dim: The dimension of the feedforward layer in transformer.
            It can be set as an integer or as a scaled_hidden_dim function.
        dropout_rate: The dropout rate of the image encoder.
        pad_token_id: The token_id for the padded tokens.
        pooler_config: An instantiable BasePoolingLayer configuration used for embedding pooling.
        attention_mask_config: An instantiable AttentionLogitBiasLayer used in transformer layer.
        feed_forward_act: The nonlinear function used in the feedforward layer in transformer block.
        layer_norm_eps: The eps used in the layer norm.
        atten_logit_cap: A soft capping value for attention logits.
        remat: A boolean for enabling the gradient checkpointing.

    Returns:
        A instantiable CLIP text encoder.
    """
    text_encoder_cfg = CLIPTextStreamEncoder.default_config()
    text_encoder_cfg.output_dim = projection_dim
    text_encoder_cfg.hidden_dim = model_dim
    text_encoder_cfg.text_encoder.pooler = pooler_config
    text_encoder_cfg.text_encoder.pad_token_id = pad_token_id
    text_encoder_cfg.text_encoder.encoder.vocab_size = vocab_size
    text_encoder_cfg.text_encoder.encoder.attention_mask = attention_mask_config

    # Set up the positional embedding.
    pos_emb_cfg = LearnedPositionalEmbedding.default_config().set(shape=[max_seq_len])
    text_encoder_cfg.text_encoder.encoder.emb = TransformerTextEmbeddings.default_config().set(
        pos_emb=pos_emb_cfg
    )

    # Set up the Transformer.
    text_encoder_cfg.text_encoder.encoder.transformer = set_transformer_config(
        num_layers=num_layers,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        feed_forward_act=feed_forward_act,
        remat=remat,
    )
    transformer_cfg = text_encoder_cfg.text_encoder.encoder.transformer.layer
    transformer_cfg.self_attention.attention.atten_logit_cap = atten_logit_cap

    text_encoder_cfg.text_encoder.encoder.output = layer_norm_config(eps=layer_norm_eps)
    set_norm_recursively(text_encoder_cfg, LayerNorm.default_config().set(eps=layer_norm_eps))
    set_dropout_rate_recursively(text_encoder_cfg, dropout_rate)
    text_encoder_cfg.output_proj = Linear.default_config().set(bias=False)
    text_encoder_cfg.output_norm = L2Norm.default_config()
    return text_encoder_cfg


def set_bert_text_encoder_config(
    *,
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    max_seq_len: int,
    projection_dim: int,  # The projection dim for the shared image-text space.
    dropout_rate: float = 0.1,
    pad_token_id: int = 0,
    feed_forward_act: str = "nn.gelu",
    layer_norm_eps: float = 1e-5,
    remat: bool = False,
) -> CLIPTextStreamEncoder.Config:
    """Configure the CLIP text stream encoder as a Bert Encoder.

    Args:
        vocab_size: An integer for the size of the vocabulary in text tokenizer.
        num_layers: An integer indicating the number of transformer blocks.
        model_dim: An integer for the output dimension of the transformer block.
        num_heads: An integer for the number of the attention heads.
        max_seq_len: An integer for the maximum length of the tokenized text.
        projection_dim: An integer for the dimension in the shared image-text embedidng space.
        dropout_rate: The dropout rate of the image encoder.
        pad_token_id: The token_id for the padded tokens.
        feed_forward_act: The nonlinear function used in the feedforward layer in transformer block.
        layer_norm_eps: The eps used in the layer norm.
        remat: A boolean for enabling the gradient checkpointing.


    Returns:
        A instantiable CLIP text encoder.

    Raises:
        ValueError: The remat=True is not supported.
    """
    text_encoder_cfg = CLIPTextStreamEncoder.default_config()
    text_encoder_cfg.output_dim = projection_dim
    text_encoder_cfg.hidden_dim = model_dim
    text_encoder_cfg.text_encoder = TextEmbeddingEncoder.default_config().set(
        pad_token_id=pad_token_id,
        encoder=bert_model_config(
            vocab_size=vocab_size,
            embedding_cfg=bert_embedding_config(
                max_position_embeddings=max_seq_len,
            ),
            stack_cfg=bert_transformer_config(
                num_layers=num_layers,
                num_heads=num_heads,
                base_cfg=(
                    RepeatedTransformerLayer.default_config()
                    if remat
                    else StackedTransformerLayer.default_config()
                ),
            ),
        ).encoder,
    )

    if remat:
        transformer_cfg = text_encoder_cfg.text_encoder.encoder.transformer
        transformer_cfg.layer.remat_spec = build_remat_spec(transformer_cfg)
    set_norm_recursively(text_encoder_cfg, LayerNorm.default_config().set(eps=layer_norm_eps))
    set_dropout_rate_recursively(text_encoder_cfg, dropout_rate)
    text_encoder_cfg.text_encoder.encoder.transformer.layer.feed_forward.activation = (
        feed_forward_act
    )
    text_encoder_cfg.output_proj = Linear.default_config().set(bias=False)
    text_encoder_cfg.output_norm = L2Norm.default_config()
    return text_encoder_cfg


class _ContrastiveLossFn(Protocol):
    def __call__(
        self, x_y_logits: Tensor, y_x_logits: Tensor, *, temperature: Union[Tensor, float]
    ) -> Tensor:
        ...


class CLIPFusionNetwork(FusionNetwork):
    """CLIP fusion network. See also CLIPModel."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures CLIPFusionNetwork."""

        # Ref: https://arxiv.org/pdf/2103.00020.pdf pp.5 Sect. 2.5
        # Ref: https://arxiv.org/pdf/1805.01978.pdf pp.3 Eq.2 and pp.5 Sect. 3.4
        log_logit_scale_init: InstantiableConfig = ConstantInitializer.default_config().set(
            value=np.log(1 / 0.07)
        )
        temperature_max_cap: float = 100
        contrastive_loss_fn: ConfigOr[_ContrastiveLossFn] = symmetric_contrastive_loss_from_logits

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._log_logit_scale_init = cfg.log_logit_scale_init.instantiate()
        self._contrastive_loss_fn = maybe_instantiate(cfg.contrastive_loss_fn)

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        param_specs = {}
        param_specs["log_logit_scale"] = ParameterSpec(
            shape=(1,),
            mesh_axes=None,
            initializer=self._log_logit_scale_init,
        )
        return param_specs

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """Forward function of CLIP.

        Args:
            input_batch: A dictionary containing:
                *"visual_encoder":
                    *"output_features": A Tensor with shape [batch_size, num_images, dim]
                *"textual_encoder":
                    *"output_features": A Tensor with shape [batch_size, num_sentences, dim]

        Returns:
            loss: A Tensor representing the loss
            A dictionary containing:
                *"similarity": A Tensor representing the similarity between
                    the text and image.
        """
        cfg = self.config
        x = input_batch["visual_encoder"]["output_features"]
        y = input_batch["textual_encoder"]["output_features"]

        batch_size = x.shape[0]
        num_images = x.shape[1]
        num_sentences = y.shape[1]

        x = x.reshape(batch_size * num_images, *x.shape[2:])
        y = y.reshape(batch_size * num_sentences, *y.shape[2:])
        # HF CLIP temperature implementation.
        # The logits is calculated as below:
        # logits = image.T * text * exp(log_logit_scale).
        log_logit_scale = self.parameters["log_logit_scale"]
        # Ref: https://arxiv.org/pdf/2103.00020.pdf pp.5 Sect. 2.5
        # Clip to prevent scaling the logits by more than 100.
        log_logit_scale = jnp.clip(log_logit_scale, a_max=jnp.log(cfg.temperature_max_cap))
        temperature = 1 / jnp.exp(log_logit_scale)
        similarity = contrastive_logits(x, y)
        loss = self._contrastive_loss_fn(similarity, similarity.T, temperature=temperature)
        self.add_summary("temperature", temperature)
        # Show the first 2048 samples. As the data is randomly sampled, this is
        # an approximation of the whole datapoints.
        self.add_summary("positive_similarity_range", jnp.diag(similarity)[:2048])
        self.add_summary("all_similarity_range", similarity[:45, :45])
        self.add_summary("positive_similarity", jnp.mean(jnp.diag(similarity)))
        self.add_summary("log_logit_scale", log_logit_scale)
        return loss, {"similarity": similarity}


class CLIPModel(MultiStreamModel):
    """A CLIP two stream model.

    This class provides a customized predict function.

    The predict is called during the inference. As we only need to calculate the
        StreamEncoder outputs.

    The unittest_predict is called for the unittest purpose.
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32
        return cfg

    def unittest_predict(self, input_batch: NestedTensor) -> NestedTensor:
        input_batch = super().predict(input_batch)
        for fusion_name in self._fusion_network:  # pylint: disable=consider-using-dict-items
            assert fusion_name not in input_batch
            _, input_batch[fusion_name] = self._fusion_network[fusion_name](input_batch)
        return input_batch

    def embed_text_batch(self, input_batch: NestedTensor) -> NestedTensor:
        """Computes embeddings for texts in `input_batch`.

        Args:
            input_batch: A dictionary supporting:
                "text": A Tensor representing the text input.
                The text shape is [batch_size, num_sentences, num_tokens]

        Returns:
            output_features: A Tensor representing the output features with shape
                [batch_size, num_sentences, dim].
        """
        output = self._stream_encoder["textual_encoder"](input_batch)
        return output["output_features"]

    def embed_image_batch(self, input_batch: NestedTensor) -> NestedTensor:
        """Computes embeddings for images in `input_batch`

        Args:
            input_batch: A dictionary supporting:
                "image": A Tensor representing the image input.
                The image shape is [batch_size, num_images, height, width, channels]

        Returns:
            output_features: A Tensor representing the output features with shape
                [batch_size, num_images, dim]
        """
        output = self._stream_encoder["visual_encoder"](input_batch)
        return output["output_features"]


def set_clip_model_config(
    *, text_encoder_cfg: InstantiableConfig, vision_encoder_cfg: InstantiableConfig
) -> CLIPModel.Config:
    clip_fusion_network_cfg = CLIPFusionNetwork.default_config()

    clip_stream_encoder = {
        "visual_encoder": vision_encoder_cfg,
        "textual_encoder": text_encoder_cfg,
    }
    clip_fusion_network = {"fusion_network": clip_fusion_network_cfg}
    clip_model = CLIPModel.default_config().set(
        stream_encoder=clip_stream_encoder, fusion_network=clip_fusion_network
    )
    return clip_model

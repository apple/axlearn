# Copyright © 2023 Apple Inc.

# pylint: disable=too-many-lines
"""CoCa architecture implementation.

Ref: https://arxiv.org/pdf/2205.01917.pdf
"""

from typing import Optional, Union

import jax.numpy as jnp

from axlearn.common.attention import (
    AttentionLogitBiasLayer,
    BaseStackedTransformerLayer,
    CausalAttentionLogitBiasLayer,
    LearnedPositionalEmbedding,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    build_remat_spec,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import (
    REQUIRED,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
)
from axlearn.common.decoder import DecodingLayer
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.encoder import CausalEncoder
from axlearn.common.layers import (
    BaseNormalizationLayer,
    Dropout,
    L2Norm,
    LayerNorm,
    Linear,
    RedirectToSharedModule,
    set_dropout_rate_recursively,
    set_norm_recursively,
)
from axlearn.common.loss import cross_entropy
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module
from axlearn.common.multi_stream_model import FusionNetwork, MultiStreamModel, StreamEncoder
from axlearn.common.poolings import AttentionPooling, BasePoolingLayer, LastNTokenPooling
from axlearn.common.utils import Nested, NestedTensor, Tensor, TensorSpec, validate_contains_paths
from axlearn.common.vision_transformer import VisionTransformer, layer_norm_config
from axlearn.vision.clip import CLIPFusionNetwork


class CoCaTextStreamEncoder(StreamEncoder):
    """A CoCa text stream encoder module.

    This class wraps the input_batch into the TextEmbeddingEncoder format.
    """

    @config_class
    class Config(StreamEncoder.Config):
        """Configures CoCaTextStreamEncoder."""

        text_encoder: CausalEncoder.Config = CausalEncoder.default_config()
        # The hidden dim of `text_encoder`. If None, uses contrastive_output_dim.
        hidden_dim: Optional[int] = None
        # Pooler for contrastive output. Use cls token if not set.
        pooler: Optional[BasePoolingLayer.Config] = None
        pad_token_id: Required[int] = REQUIRED
        # The dim of the contrastive output embeddings.
        contrastive_output_dim: Required[int] = REQUIRED
        contrastive_output_proj: Optional[Linear.Config] = None
        # Coca uses LayerNorm instead.
        contrastive_output_norm: Optional[BaseNormalizationLayer.Config] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim or cfg.contrastive_output_dim
        self._add_child(
            "text_encoder", cfg.text_encoder.set(dim=hidden_dim, pad_token_id=cfg.pad_token_id)
        )

        if cfg.pooler:
            assert cfg.pooler.num_outputs == 1
            self._add_child("pooler", cfg.pooler.set(input_dim=hidden_dim, output_dim=hidden_dim))
        else:
            assert cfg.text_encoder.num_cls_tokens == 1, "Only support 1 cls token for now!"

        if cfg.contrastive_output_dim:
            self._add_child(
                "contrastive_output_proj",
                cfg.contrastive_output_proj.set(
                    input_dim=hidden_dim, output_dim=cfg.contrastive_output_dim
                ),
            )
        else:
            assert hidden_dim == cfg.contrastive_output_dim
        if cfg.contrastive_output_norm:
            if "input_dim" in cfg.contrastive_output_norm:
                cfg.contrastive_output_norm.set(input_dim=cfg.contrastive_output_dim)
            self._add_child("contrastive_output_norm", cfg.contrastive_output_norm)

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """The forward function for the CoCaTextStreamEncoder.

        Args:
            input_batch: A dictionary containing:
                *"text": An int Tensor with shape ..., containing both BOS and EOS
                    [batch_size, num_sentences, num_tokens]

        Returns:
            A dictionary containing:
                *"output_features": A Tensor with shape
                    [batch_size, num_sentences, dim]
                *"caption_features": A Tensor with shape
                    [batch_size, num_sentences, seq_len, dim]
                *"caption_ids": A Tensor with shape
                    [batch_size, num_sentences, seq_len]
                *"caption_labels": A Tensor with shape
                    [batch_size, num_sentences, seq_len]
        """
        cfg = self.config
        text = input_batch["text"]
        batch_size = text.shape[0]
        num_sentences = text.shape[1]

        input_ids, target_ids = text[:, :, :-1], text[:, :, 1:]

        # Reshape the input for encoding.
        input_ids_reshaped = input_ids.reshape(batch_size * num_sentences, *input_ids.shape[2:])

        encoder_output = self.text_encoder(input_ids_reshaped)
        hidden_states = normalized_states = encoder_output["hidden_states"]
        if "normalized_states" in encoder_output:
            normalized_states = encoder_output["normalized_states"]
        if "pooler" in self.children:
            paddings = input_ids_reshaped == cfg.pad_token_id
            x = self.pooler(normalized_states, paddings=paddings)
            x = x.squeeze(axis=1)
        else:
            x = normalized_states[:, -1, :]
            hidden_states = hidden_states[:, :-1, :]
        if "contrastive_output_proj" in self.children:
            x = self.contrastive_output_proj(x)
        if "contrastive_output_norm" in self.children:
            x = self.contrastive_output_norm(x)

        def _reshape_to_input(value):
            return value.reshape(batch_size, num_sentences, *value.shape[1:])

        # Reshape the output back to the input shapes.
        x = _reshape_to_input(x)
        encoder_hidden_states = _reshape_to_input(hidden_states)

        # Use `output_features` for contrastive task to be consistent with CLIP model.
        output_dict = {
            "output_features": x,
            "caption_features": encoder_hidden_states,
            "caption_ids": input_ids,  # Used for creating masks in captioning task.
            "caption_labels": target_ids,
        }
        return output_dict

    def init_states(self, *, batch_size: int, max_sequence_length: int) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            batch_size: the batch size of the target to be decoded.
            max_sequence_length: the sequence length of the target to be decoded.

        Returns:
            The cache as a `NestedTensor` with key and value initialized.
        """
        encoder_init_states = self.text_encoder.init_states(
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
        )

        return {
            "transformer_state": encoder_init_states["transformer_state"],
            "input_ids": encoder_init_states["input_ids"],
        }

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        input_ids: Tensor,
    ) -> tuple[NestedTensor, NestedTensor]:
        states, encoder_output = self.text_encoder.prefill_states(
            time_step=time_step,
            input_ids=input_ids,
        )
        output_dict = dict(caption_features=encoder_output["hidden_states"])
        return states, output_dict

    def extend_step(
        self,
        *,
        cached_states: NestedTensor,
        input_ids: Tensor,
    ) -> tuple[NestedTensor, NestedTensor]:
        updated_state, encoder_output = self.text_encoder.extend_step(
            cached_states=cached_states,
            input_ids=input_ids,
        )

        output_dict = {
            "caption_features": encoder_output["hidden_states"],
        }

        return updated_state, output_dict


class CoCaImageStreamEncoder(StreamEncoder):
    """A CoCa image stream encoder module.

    This class wraps the input_batch into the VisonTransformer format.
    """

    @config_class
    class Config(StreamEncoder.Config):
        """Configures CoCaImageStreamEncoder."""

        image_encoder: VisionTransformer.Config = VisionTransformer.default_config()
        # The hidden dim of `text_encoder`. If None, uses contrastive_output_dim.
        hidden_dim: Optional[int] = None
        # The pooler mode for contrastive and captioning tasks.
        #   Options: `cascade`|`parallel`|`bottleneck`.
        # -- cascade: the contrastive pooler operates on the caption pooler outputs.
        # -- parallel: the contrastive and caption poolers both operate on the encoder outputs.
        # -- bottleneck: both 'output_features' and 'caption_features' will reduce to
        #       one global image feature vector, which can be obtained via attention pooling or
        #       a naive mean pooling of ViT features.
        pooler_mode: str = "cascade"
        caption_pooler: BasePoolingLayer.Config = AttentionPooling.default_config().set(
            num_outputs=256
        )
        contrastive_pooler: BasePoolingLayer.Config = AttentionPooling.default_config().set(
            num_outputs=1
        )
        # The dim of the contrastive output embeddings.
        contrastive_output_dim: Required[int] = REQUIRED
        contrastive_output_proj: Optional[Linear.Config] = None
        contrastive_output_norm: Optional[
            BaseNormalizationLayer.Config
        ] = None  # CoCa uses LayerNorm instead.

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        hidden_dim = cfg.hidden_dim or cfg.contrastive_output_dim
        self._add_child("image_encoder", cfg.image_encoder.set(output_dim=hidden_dim))
        assert cfg.contrastive_pooler.num_outputs == 1
        assert cfg.pooler_mode in ["parallel", "cascade", "bottleneck"]
        self._add_child(
            "contrastive_pooler",
            cfg.contrastive_pooler.set(input_dim=hidden_dim, output_dim=hidden_dim),
        )
        if cfg.pooler_mode != "bottleneck":
            # allows pooler dimension to be overridden
            output_dim = cfg.caption_pooler.output_dim or hidden_dim
            self._add_child(
                "caption_pooler",
                cfg.caption_pooler.set(input_dim=hidden_dim, output_dim=output_dim),
            )
        if cfg.contrastive_output_proj:
            self._add_child(
                "contrastive_output_proj",
                cfg.contrastive_output_proj.set(
                    input_dim=hidden_dim, output_dim=cfg.contrastive_output_dim
                ),
            )
        else:
            assert hidden_dim == cfg.contrastive_output_dim
        if cfg.contrastive_output_norm:
            if "input_dim" in cfg.contrastive_output_norm:
                cfg.contrastive_output_norm.set(input_dim=cfg.contrastive_output_dim)
            self._add_child("contrastive_output_norm", cfg.contrastive_output_norm)

    def forward(  # pylint:disable=arguments-renamed
        self, input_batch: NestedTensor
    ) -> NestedTensor:
        """The forward function for the CoCaImageStreamEncoder.

        Args:
            input_batch: A dictionary containing:
                *"image": A Tensor with shape
                    [batch_size, num_images, height, width, channel]

        Returns:
            A dictionary containing:
                *"output_features": A Tensor with shape
                    [batch_size, num_images, dim]
                *"caption_features": A Tensor with shape
                    [batch_size, num_images, num_tokens, dim]

        Raises:
            ValueError: The pooler mode is unsupported.
        """
        cfg = self.config
        image_encoder_input = input_batch["image"]
        batch_size = image_encoder_input.shape[0]
        num_images = image_encoder_input.shape[1]

        image_encoder_input = image_encoder_input.reshape(
            batch_size * num_images, *image_encoder_input.shape[2:]
        )

        encoder_output_dict = self.image_encoder(image_encoder_input)
        encoded_features = encoder_output_dict["encoded_features"]

        if cfg.pooler_mode == "bottleneck":
            # option 1: attention pooling to obtain the bottleneck feature
            caption_features = contrastive_features = self.contrastive_pooler(encoded_features)
            # option 2: simple mean pooling to obtain the bottleneck feature
            # TODO(zhe): may be added back in the future
            # caption_features = contrastive_features = jnp.mean(encoded_features, [1],
            #    keepdims=True)
        elif cfg.pooler_mode in ["cascade", "parallel"]:
            caption_features = self.caption_pooler(encoded_features)

            if cfg.pooler_mode == "cascade":
                contrastive_features = self.contrastive_pooler(caption_features)
            else:
                assert cfg.pooler_mode == "parallel"
                contrastive_features = self.contrastive_pooler(encoded_features)
        else:
            raise ValueError(f"Pooler mode {cfg.pooler_mode} not supported.")
        contrastive_features = contrastive_features.squeeze(axis=1)

        if "contrastive_output_proj" in self.children:
            contrastive_features = self.contrastive_output_proj(contrastive_features)
        if "contrastive_output_norm" in self.children:
            contrastive_features = self.contrastive_output_norm(contrastive_features)

        contrastive_features = contrastive_features.reshape(
            batch_size, num_images, *contrastive_features.shape[1:]
        )
        caption_features = caption_features.reshape(
            batch_size, num_images, *caption_features.shape[1:]
        )
        # Use `output_features` for contrastive task to be consistent with CLIP model.
        output_dict = {
            "output_features": contrastive_features,
            "caption_features": caption_features,
        }

        return output_dict


def set_coca_vision_encoder_config(
    *,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    contrastive_output_dim: int,  # The projection dim for the shared image-text space.
    feed_forward_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4),
    image_size: tuple[int, int] = (224, 224),
    patch_size: tuple[int, int] = (16, 16),
    dropout_rate: float = 0.1,
    contrastive_pooler_config: BasePoolingLayer.Config = AttentionPooling.default_config(),
    caption_pooler_config: BasePoolingLayer.Config = AttentionPooling.default_config(),
    caption_pooler_num_outputs: int = 256,  # Secs 3.2, Attentional Poolers.
    pooler_mode: str = "cascade",
    feed_forward_act: str = "nn.gelu",
    layer_norm_eps: float = 1e-5,
    num_cls_tokens: int = 0,
    atten_logit_cap: Optional[float] = None,
    remat: bool = False,
) -> CoCaImageStreamEncoder.Config:
    # pylint: disable=duplicate-code
    image_encoder_cfg = CoCaImageStreamEncoder.default_config()
    # Set up the image tokenization module.
    image_encoder_cfg.contrastive_output_dim = contrastive_output_dim
    image_encoder_cfg.hidden_dim = model_dim
    image_encoder_cfg.image_encoder.visual_embed.convert_to_sequence.set(patch_size=patch_size)
    image_encoder_cfg.image_encoder.visual_embed.convert_to_sequence.conv.bias = False
    image_encoder_cfg.image_encoder.num_cls_tokens = num_cls_tokens
    image_encoder_cfg.image_encoder.encoder_1d.pos_emb.shape = [
        (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) + num_cls_tokens
    ]
    image_encoder_cfg.caption_pooler = caption_pooler_config.set(
        num_outputs=caption_pooler_num_outputs
    )
    if caption_pooler_config.klass == AttentionPooling:
        image_encoder_cfg.caption_pooler.cross_attention.attention.num_heads = num_heads
    image_encoder_cfg.contrastive_pooler = contrastive_pooler_config.set(num_outputs=1)
    if contrastive_pooler_config.klass == AttentionPooling:
        image_encoder_cfg.contrastive_pooler.cross_attention.attention.num_heads = num_heads
    image_encoder_cfg.pooler_mode = pooler_mode
    # Set up the Transformer.
    if remat:
        image_encoder_cfg.image_encoder.encoder_1d.transformer = (
            RepeatedTransformerLayer.default_config()
        )

    image_encoder_cfg.image_encoder.encoder_1d.transformer.num_layers = num_layers
    transformer_cfg = image_encoder_cfg.image_encoder.encoder_1d.transformer.layer
    transformer_cfg.self_attention.attention.num_heads = num_heads
    transformer_cfg.self_attention.attention.atten_logit_cap = atten_logit_cap
    transformer_cfg.feed_forward.hidden_dim = feed_forward_dim
    transformer_cfg.feed_forward.activation = feed_forward_act
    if remat:
        transformer_cfg.remat_spec = build_remat_spec(
            image_encoder_cfg.image_encoder.encoder_1d.transformer
        )
    set_norm_recursively(image_encoder_cfg, LayerNorm.default_config().set(eps=layer_norm_eps))
    image_encoder_cfg.image_encoder.encoder_1d.set(input_norm=layer_norm_config(eps=layer_norm_eps))
    set_dropout_rate_recursively(image_encoder_cfg, dropout_rate)
    image_encoder_cfg.contrastive_output_proj = Linear.default_config().set(bias=False)
    image_encoder_cfg.contrastive_output_norm = L2Norm.default_config()
    return image_encoder_cfg
    # pylint: enable=duplicate-code


def set_coca_text_encoder_config(
    *,
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    max_seq_len: int,
    contrastive_output_dim: int,  # The projection dim for the shared image-text space.
    feed_forward_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4),
    dropout_rate: float = 0.1,
    pad_token_id: int = 0,
    pooler_config: Optional[BasePoolingLayer.Config] = LastNTokenPooling.default_config(),
    attention_mask_config: AttentionLogitBiasLayer.Config = (
        CausalAttentionLogitBiasLayer.default_config()
    ),
    feed_forward_act: str = "nn.gelu",
    layer_norm_eps: float = 1e-5,
    atten_logit_cap: Optional[float] = None,
    remat: bool = False,
) -> CoCaTextStreamEncoder.Config:
    # pylint: disable=duplicate-code
    text_encoder_cfg = CoCaTextStreamEncoder.default_config()
    text_encoder_cfg.contrastive_output_dim = contrastive_output_dim
    text_encoder_cfg.hidden_dim = model_dim
    text_encoder_cfg.pooler = pooler_config
    text_encoder_cfg.pad_token_id = pad_token_id
    text_encoder_cfg.text_encoder.vocab_size = vocab_size
    text_encoder_cfg.text_encoder.attention_mask = attention_mask_config
    if pooler_config is None:
        text_encoder_cfg.text_encoder.num_cls_tokens = 1
        text_encoder_cfg.text_encoder.pad_token_id = pad_token_id

    # Set up the positional embedding.
    pos_emb_cfg = LearnedPositionalEmbedding.default_config().set(shape=[max_seq_len])
    text_encoder_cfg.text_encoder.emb = TransformerTextEmbeddings.default_config().set(
        pos_emb=pos_emb_cfg
    )

    # Set up the Transformer.
    if remat:
        text_encoder_cfg.text_encoder.transformer = RepeatedTransformerLayer.default_config()
    else:
        text_encoder_cfg.text_encoder.transformer = StackedTransformerLayer.default_config()

    text_encoder_cfg.text_encoder.transformer.num_layers = num_layers
    transformer_cfg = text_encoder_cfg.text_encoder.transformer.layer
    transformer_cfg.feed_forward.activation = feed_forward_act
    transformer_cfg.self_attention.attention.num_heads = num_heads
    transformer_cfg.self_attention.attention.atten_logit_cap = atten_logit_cap
    transformer_cfg.feed_forward.hidden_dim = feed_forward_dim

    if remat:
        transformer_cfg.remat_spec = build_remat_spec(text_encoder_cfg.text_encoder.transformer)
    text_encoder_cfg.text_encoder.output = layer_norm_config(eps=layer_norm_eps)

    set_norm_recursively(text_encoder_cfg, LayerNorm.default_config().set(eps=layer_norm_eps))
    set_dropout_rate_recursively(text_encoder_cfg, dropout_rate)
    text_encoder_cfg.contrastive_output_proj = Linear.default_config().set(bias=False)
    text_encoder_cfg.contrastive_output_norm = L2Norm.default_config()
    return text_encoder_cfg
    # pylint: enable=duplicate-code


class CoCaLMHead(BaseLayer):
    """A wrapper LM head for the CoCa model.

    This combines the lm head with the normalization and dropout that usually precedes
    the head.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures CoCaLMHead."""

        head: InstantiableConfig = RedirectToSharedModule.default_config().set(
            shared_module="shared_token_emb",
            method_map={"forward": "attend"},
        )
        norm: InstantiableConfig = LayerNorm.default_config()
        dropout: Dropout.Config = Dropout.default_config()

    def __init__(self, cfg: Config, *, parent: Optional["Module"]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("head", cfg.head)
        self._add_child("norm", cfg.norm)
        self._add_child("dropout", cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Computes logits with token embedding.

        Args:
            x: an int Tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            logits: A float Tensor of shape [batch_size, seq_len, vocab_size].
            hidden_states: A float Tneosr of shape [batch_size seq_len, hidden_dim]
        """
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x), x


class CoCaCaptioningFusionNetwork(FusionNetwork):
    """CoCa captioning fusion network. See also CoCaModel."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures CoCaCaptioningFusionNetwork."""

        # Multimodal decoder.
        transformer: BaseStackedTransformerLayer.Config = StackedTransformerLayer.default_config()
        attention_mask: AttentionLogitBiasLayer.Config = (
            CausalAttentionLogitBiasLayer.default_config()
        )

        # When set to True, attend to the visual features to generate captions.
        use_cross_attention: bool = True

        lm_head: Optional[CoCaLMHead.Config] = CoCaLMHead.default_config()

        dim: Required[int] = REQUIRED
        pad_token_id: int = 0

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child("transformer", cfg.transformer.set(input_dim=cfg.dim))
        self._add_child("attention_mask", cfg.attention_mask)
        lm_head_cfg = cfg.lm_head
        lm_head_cfg.norm.set(input_dim=cfg.dim)
        self._add_child("lm_head", lm_head_cfg)

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        """Forward function of CoCa captioning task.

        Args:
            input_batch: A dictionary containing:
                *"visual_encoder":
                    *"caption_features": A Tensor with shape
                        [batch_size, num_images, num_tokens, dim]
                *"textual_encoder":
                    *"caption_features": A Tensor with shape
                        [batch_size, num_sentences, seq_len, dim]
                    *"caption_ids": A Tensor with shape
                        [batch_size, num_sentences, seq_len]
                    *"caption_labels": A Tensor with shape
                        [batch_size, num_sentences, seq_len]

        Returns:
            loss: A Tensor representing the loss
            A dictionary containing: TBA

        Raises:
            ValueError: if there is more than one input image in each input example,
                and the number of input images and captions are not consistent in each input pair.
        """
        cfg = self.config

        caption_input_features = input_batch["textual_encoder"]["caption_features"]
        caption_ids = input_batch["textual_encoder"]["caption_ids"]
        caption_labels = input_batch["textual_encoder"]["caption_labels"]

        batch_size = caption_input_features.shape[0]
        num_captions = caption_input_features.shape[1]

        caption_input_features = caption_input_features.reshape(
            batch_size * num_captions, *caption_input_features.shape[2:]
        )
        caption_ids = caption_ids.reshape(batch_size * num_captions, *caption_ids.shape[2:])
        caption_labels = caption_labels.reshape(
            batch_size * num_captions, *caption_labels.shape[2:]
        )

        self_attention_logit_biases = self.attention_mask(
            segment_ids=caption_ids != cfg.pad_token_id,
            positions=jnp.arange(caption_ids.shape[-1])[None, :],
        )
        if cfg.use_cross_attention:
            visual_features = input_batch["visual_encoder"]["caption_features"]
            num_images = visual_features.shape[1]

            if num_images == 1:
                visual_features = jnp.repeat(visual_features, num_captions, axis=1)
                num_images = num_captions
            elif num_images != num_captions:
                raise ValueError(
                    "Only 1) one image per example, or 2) same number of images and captions"
                    "are supported."
                )

            visual_features = visual_features.reshape(
                batch_size * num_images, *visual_features.shape[2:]
            )

            x = self.transformer(
                caption_input_features,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=visual_features,
            ).data
        else:
            x = self.transformer(
                caption_input_features,
                self_attention_logit_biases=self_attention_logit_biases,
            ).data
        logits, _ = self.lm_head(x)

        live_targets = (caption_ids != cfg.pad_token_id).astype(
            jnp.float32
        )  # pytype: disable=attribute-error

        # Compute prediction loss.
        loss, _ = cross_entropy(logits, target_labels=caption_labels, live_targets=live_targets)

        # Add training metrics.
        token_accuracy = jnp.equal(jnp.argmax(logits, axis=-1), caption_labels)
        num_targets = live_targets.sum()
        accuracy_micro = (token_accuracy * live_targets).sum() / jnp.maximum(1, num_targets)

        num_targets_per_seq = live_targets.sum(axis=-1)
        live_seqs = num_targets_per_seq > 0
        seq_accuracy = (token_accuracy * live_targets).sum(axis=-1) / jnp.maximum(
            1, num_targets_per_seq
        )
        num_valid_seqs = live_seqs.sum()
        accuracy_macro = (seq_accuracy * live_seqs).sum() / jnp.maximum(1, num_valid_seqs)

        self.add_summary("accuray_micro", WeightedScalar(accuracy_micro, num_targets))
        self.add_summary("accuray_macro", WeightedScalar(accuracy_macro, num_valid_seqs))
        self.add_summary("loss_weighted", WeightedScalar(loss, num_targets))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_targets))

        return loss, {"logits": logits}

    def init_states(self, *, batch_size: int, max_sequence_length: int) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            batch_size: the batch size of the target to be decoded.
            max_sequence_length: the sequence length of the target to be decoded.

        Returns:
            The cache as a `NestedTensor` with key and value initialized.
        """
        cfg = self.config
        init_state, _ = self.transformer.init_states(
            time_step=None,
            data=TensorSpec([batch_size, max_sequence_length, cfg.dim]),
        )
        return dict(transformer_state=init_state)

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        input_ids: Tensor,
        input_features: Tensor,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, NestedTensor]:
        cfg = self.config
        transformer_state, transformer_data = self.transformer.init_states(
            time_step=time_step,
            data=input_features,
            self_attention_logit_biases=self.attention_mask(
                segment_ids=input_ids != cfg.pad_token_id,
                positions=jnp.arange(input_ids.shape[-1])[None, :],
            ),
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        logits, hidden_states = self.lm_head(transformer_data.data)
        states = dict(transformer_state=transformer_state)
        return states, dict(logits=logits, hidden_states=hidden_states)

    def extend_step(
        self,
        *,
        cached_states: NestedTensor,
        input_ids: Tensor,
        input_features: Tensor,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, NestedTensor]:
        cfg = self.config
        if not cfg.use_cross_attention:
            cross_attention_data = None
            cross_attention_logit_biases = None

        cache = cached_states
        time_step = cache["time_step"]  # Note we do not update time_step here.
        assert time_step.ndim == 1

        full_mask_logit_biases = self.attention_mask(
            segment_ids=input_ids != cfg.pad_token_id,
            positions=jnp.arange(input_ids.shape[-1])[None, :],
        )
        # Select logit biases corresponding to time step [batch, num_heads, 1, source_length].
        # Note: if `time_step` exceeds `target_len`, e.g. in the case where one decode starts at a
        # later index than another, clip the indices instead of producing NaNs.
        # TODO(markblee): Update attention masks to support explicit positions, so we can skip this.
        self_attention_biases = jnp.take_along_axis(
            full_mask_logit_biases, time_step[:, None, None, None], axis=2, mode="clip"
        )

        updated_transformer_state, transformer_data = self.transformer.extend_step(
            cached_states=cache["transformer_state"],
            data=input_features,
            self_attention_logit_biases=self_attention_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        logits, hidden_states = self.lm_head(transformer_data.data)

        updated_state = dict(transformer_state=updated_transformer_state)
        return updated_state, {
            "logits": logits,
            "hidden_states": hidden_states,
        }


def set_captioning_cfg(
    num_layers: int,
    model_dim: int,
    num_heads: int,
    use_cross_attention: bool,
    cross_attention_dim: int,
    pad_token_id: int = 0,
    feed_forward_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4),
    feed_forward_act: str = "nn.gelu",
    layer_norm_eps: float = 1e-5,
    dropout_rate: float = 0.0,
    attention_mask_config: AttentionLogitBiasLayer.Config = (
        CausalAttentionLogitBiasLayer.default_config()
    ),
    atten_logit_cap: Optional[float] = None,
    remat: bool = False,
):
    captioning_cfg = CoCaCaptioningFusionNetwork.default_config()
    captioning_cfg.attention_mask = attention_mask_config
    captioning_cfg.dim = model_dim
    captioning_cfg.pad_token_id = pad_token_id
    captioning_cfg.use_cross_attention = use_cross_attention

    # Decoder config
    if remat:
        decoder_cfg = RepeatedTransformerLayer.default_config()
    else:
        decoder_cfg = StackedTransformerLayer.default_config()

    decoder_cfg.num_layers = num_layers

    transformer_cfg = decoder_cfg.layer
    transformer_cfg.self_attention.attention.num_heads = num_heads
    transformer_cfg.self_attention.attention.atten_logit_cap = atten_logit_cap
    transformer_cfg.feed_forward.hidden_dim = feed_forward_dim
    transformer_cfg.feed_forward.activation = feed_forward_act

    if remat:
        transformer_cfg.remat_spec = build_remat_spec(decoder_cfg)

    if captioning_cfg.use_cross_attention:
        # Add cross attention
        decoder_cfg.layer.cross_attention = TransformerAttentionLayer.default_config().set(
            target_dim=model_dim,
            source_dim=cross_attention_dim,
        )
        decoder_cfg.layer.cross_attention.attention.num_heads = num_heads
        decoder_cfg.layer.cross_attention.attention.atten_logit_cap = atten_logit_cap

    captioning_cfg.transformer = decoder_cfg

    set_norm_recursively(captioning_cfg, LayerNorm.default_config().set(eps=layer_norm_eps))
    set_dropout_rate_recursively(captioning_cfg, dropout_rate)

    return captioning_cfg


class CoCaModel(MultiStreamModel):
    """A CoCa two stream model.

    This class provides a customized predict function.

    The predict is called during the inference. As we only need to calculate the
        StreamEncoder outputs.

    The unittest_predict is called for the unittest purpose.
    """

    @config_class
    class Config(MultiStreamModel.Config):
        """Configures CoCaModel."""

        # Required when running beam_search_decode.
        pad_token_id: Optional[int] = None
        eos_token_id: Optional[int] = None
        decoding: DecodingLayer.Config = DecodingLayer.default_config()

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: CoCaModel.Config = self.config
        self._decoding: DecodingLayer = cfg.decoding.set(
            pad_token_id=cfg.pad_token_id, eos_token_id=cfg.eos_token_id
        ).instantiate(decoder=self)
        self._share_with_descendants(
            self._stream_encoder["textual_encoder"].text_encoder.emb.token_emb,
            shared_module_name="shared_token_emb",
        )

    def predict(self, input_batch: NestedTensor) -> NestedTensor:
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
            A Tensor representing the output features with shape
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

    def init_states(self, *, batch_size: int, max_sequence_length: int) -> NestedTensor:
        """See `BaseDecoder.init_states` for details."""
        textual_encoder_states = self._stream_encoder["textual_encoder"].init_states(
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
        )
        captioning_network_states = self._fusion_network["captioning_fusion_network"].init_states(
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
        )
        return {
            "textual_encoder_transformer_state": textual_encoder_states["transformer_state"],
            "captioning_network_transformer_state": captioning_network_states["transformer_state"],
            "input_ids": textual_encoder_states["input_ids"],
            "time_step": jnp.zeros(batch_size, dtype=jnp.int32),
        }

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        input_batch: Nested[Tensor],
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, NestedTensor]:
        """See `BaseDecoder.prefill_states` for details.

        Args:
            time_step: A Tensor of shape [batch_size]. See `BaseDecoder.prefill_states` for details.
            input_batch: A dict containing at minimum:
                * input_ids: An int Tensor of shape [batch_size, seq_len].
                    Values should be in the range [0, vocab_size), where `vocab_size` is commonly
                    configured in `textual_encoder`.
            cross_attention_data: A float Tensor of shape [batch_size, source_len, hidden_dim].
            cross_attention_logit_biases: A Tensor of shape [batch_size, target_len, source_len].
                A -inf represents a disconnected position pair.

        Returns:
            See `BaseDecoder.prefill_states` for details.
        """
        validate_contains_paths(input_batch, paths=["input_ids"])
        input_ids = input_batch["input_ids"]

        textual_encoder_state, textual_encoder_output = self._stream_encoder[
            "textual_encoder"
        ].prefill_states(
            time_step=time_step,
            input_ids=input_ids,
        )
        (
            captioning_network_state,
            captioning_network_output,
        ) = self._fusion_network["captioning_fusion_network"].prefill_states(
            time_step=time_step,
            input_ids=input_ids,
            input_features=textual_encoder_output["caption_features"],
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        states = dict(
            time_step=time_step,
            input_ids=input_ids,
            textual_encoder_transformer_state=textual_encoder_state["transformer_state"],
            captioning_network_transformer_state=captioning_network_state["transformer_state"],
        )
        return states, captioning_network_output

    def extend_step(
        self,
        *,
        cached_states: NestedTensor,
        input_ids: Tensor,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, NestedTensor]:
        """See `BaseDecoder.extend_step` for details."""
        # This structure is shared with Decoder, but is necessary to repeat in extend_step.
        time_step = cached_states["time_step"]
        assert time_step.ndim == 1

        encoder_cached_state = {
            "transformer_state": cached_states["textual_encoder_transformer_state"],
            "input_ids": cached_states["input_ids"],
            "time_step": cached_states["time_step"],
        }

        (
            updated_textual_encoder_state,
            textual_encoder_output,
        ) = self._stream_encoder["textual_encoder"].extend_step(
            cached_states=encoder_cached_state,
            input_ids=input_ids,
        )

        captioning_cached_states = {
            "transformer_state": cached_states["captioning_network_transformer_state"],
            "time_step": cached_states["time_step"],
        }
        (
            updated_captioning_network_state,
            captioning_network_output,
        ) = self._fusion_network["captioning_fusion_network"].extend_step(
            cached_states=captioning_cached_states,
            input_ids=updated_textual_encoder_state["input_ids"],
            input_features=textual_encoder_output["caption_features"],
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )

        updated_state = {
            "textual_encoder_transformer_state": updated_textual_encoder_state["transformer_state"],
            "input_ids": updated_textual_encoder_state["input_ids"],
            "captioning_network_transformer_state": updated_captioning_network_state[
                "transformer_state"
            ],
            # There are some non-greedy DFS/BFS and sliding attention algorithms that
            # recursively search through potentials.
            # They backtrace to some anchor time step after exploring for t steps.
            # This requires tracking time_step separately from the attention time_step.
            "time_step": cached_states["time_step"] + 1,
        }

        return updated_state, captioning_network_output

    def beam_search_decode(
        self,
        *,
        input_batch: Nested[Tensor],
        max_sequence_length: int,
        num_decodes: int,
        **kwargs,
    ):
        """See configured `decoding` implementation for details."""
        return self._decoding.beam_search_decode(
            input_batch=input_batch,
            max_sequence_length=max_sequence_length,
            num_decodes=num_decodes,
            **kwargs,
        )

    def sample_decode(
        self,
        *,
        input_batch: Nested[Tensor],
        max_sequence_length: int,
        num_decodes: int,
        **kwargs,
    ):
        """See configured `decoding` implementation for details."""
        return self._decoding.sample_decode(
            input_batch=input_batch,
            max_sequence_length=max_sequence_length,
            num_decodes=num_decodes,
            **kwargs,
        )

    def predict_caption(
        self,
        input_batch: NestedTensor,
        *,
        max_sequence_length: int,
        num_decodes: int,
        decode_method: str = "beam_search_decode",
    ):
        """Predict captions with selected decoding strategy.
        Args:
            input_batch: A dictionary supporting:
                "image": A Tensor representing the image input,
                    with shape [batch_size, num_images, height, width, channels]
                    num_images is set to 1 by default.
                "prefix": A Tensor representing the caption prefix,
                    with shape [batch_size, 1]
                    This can be a vector of [BOS] tokens.
            max_sequence_length: The maximum sequence length of tokens to generate.
            num_decodes: The number of decoded sqeuences to return.
            decode_method: either "beam_search_decode", or "sample_decode".
        Returns:
            output: BeamSearchOutputs if beam search is used.
                output.sequences: a tensor representing the generated captions,
                    with shape [batch_size, num_decodes, max_sequence_length]
        Raises:
            NotImplementedError: The decoding method is unsupported.
        """
        cfg = self.config
        assert cfg.pad_token_id is not None and cfg.eos_token_id is not None

        # Get visual feature, take the first image feature if there are multiple images per example.
        # with shape [batch_size, num_images, num_tokens, dim]
        visual_features = self._stream_encoder["visual_encoder"](input_batch)
        # with shape [batch_size, num_tokens, dim]
        visual_features = visual_features["caption_features"][:, 0, :, :]

        if decode_method in ("beam_search_decode", "sample_decode"):
            output = getattr(self, decode_method)(
                input_batch=input_batch,
                max_sequence_length=max_sequence_length,
                num_decodes=num_decodes,
                cross_attention_data=visual_features,
            )
        else:
            raise NotImplementedError(
                f"decode_method `{decode_method}` is not supported. Choose from "
                "[beam_search, sample]"
            )
        return output


def set_coca_config(
    *,
    contrastive_output_dim,
    text_encoder_cfg: dict,
    vision_encoder_cfg: dict,
    captioning_cfg: dict,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    coca_visual_encoder_cfg = set_coca_vision_encoder_config(
        **vision_encoder_cfg, contrastive_output_dim=contrastive_output_dim
    )
    coca_textual_encoder_cfg = set_coca_text_encoder_config(
        **text_encoder_cfg, contrastive_output_dim=contrastive_output_dim
    )

    coca_contrastive_fusion_network_cfg = CLIPFusionNetwork.default_config()
    coca_captioning_fusion_network_cfg = set_captioning_cfg(**captioning_cfg)

    coca_stream_encoder = {
        "visual_encoder": coca_visual_encoder_cfg,
        "textual_encoder": coca_textual_encoder_cfg,
    }
    coca_fusion_network = {
        "contrastive_fusion_network": coca_contrastive_fusion_network_cfg,
        "captioning_fusion_network": coca_captioning_fusion_network_cfg,
    }

    # Ref: https://arxiv.org/pdf/2205.01917.pdf Sec 4.1
    loss_weights = {
        "contrastive_fusion_network": 1.0,
        "captioning_fusion_network": 2.0,
    }

    coca_model = CoCaModel.default_config().set(
        stream_encoder=coca_stream_encoder,
        fusion_network=coca_fusion_network,
        loss_weights=loss_weights,
        # pad_token_id and eos_token_id are required when running decoding at inference time.
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    return coca_model

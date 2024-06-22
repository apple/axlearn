# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# microsoft/unilm:
# Copyright (c) Microsoft Corporation.
# Licensed under The MIT License.

# pylint: disable=duplicate-code
"""Multiway transformer layers.

References:
https://arxiv.org/pdf/2111.02358.pdf
https://github.com/microsoft/unilm/tree/master/vlmo
"""
from typing import Dict, Optional, Set, Tuple

import numpy as np
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.attention import (
    BaseTransformerLayer,
    ForwardMode,
    LearnedPositionalEmbedding,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import Dropout, Embedding, LayerNorm, set_dropout_rate_recursively
from axlearn.common.module import Module, NestedTensor, Tensor, child_context
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, GaussianInitializer
from axlearn.common.poolings import BasePoolingLayer, FirstNTokenPooling
from axlearn.common.vision_transformer import VisualEmbedding

TEXT_MODALITY = 0
IMAGE_MODALITY = 1
TEXT_IMAGE_MODALITY = 2


def layer_norm_config(eps=1e-6):
    return LayerNorm.default_config().set(eps=eps)


class MultiwayTransformerLayer(BaseTransformerLayer):
    """A Multiway Transformer layer.

    The layer consists of one shared self-attention layer and multiple FFN experts in parallel.
    The routing logic is as below:
        - For monomodal input: route the input to the corresponding monomodal FFN expert;
        - For multimodal input: if a multimodal FFN expert has been created, route the input to it.
            If not, split the input to monomodal inputs then route individual input to the
            corresponding expert.
    """

    @config_class
    class Config(BaseTransformerLayer.Config):
        self_attention: InstantiableConfig = TransformerAttentionLayer.default_config()
        # If not None, the cross-attention layer config.
        cross_attention: Optional[InstantiableConfig] = None
        feed_forward: InstantiableConfig = TransformerFeedForwardLayer.default_config()
        # The number of parallel FFNs that will be created. If 1, the layer falls back to a
        # regular transformer layer.
        num_ffn: int = 1
        # The maximum text embedding length. Only used for multimodal inputs.
        max_text_len: Optional[int] = None

    Output = BaseTransformerLayer.Output

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "self_attention",
            cfg.self_attention.set(target_dim=cfg.input_dim, source_dim=cfg.input_dim),
        )
        for i in range(cfg.num_ffn):
            self._add_child(f"feed_forward{i}", cfg.feed_forward.set(input_dim=cfg.input_dim))
        if cfg.cross_attention is not None:
            self._add_child("cross_attention", cfg.cross_attention.set(target_dim=cfg.input_dim))

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Tensor,
        feed_forward_index: int = 0,
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
        return_aux: Optional[Set[str]] = None,
    ) -> Tuple[Optional[NestedTensor], Tensor]:
        """Computes transformer layer outputs and self/cross-attention probabilities.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            data: A Tensor of shape [batch, target_length, target_dim].
            feed_forward_index: An integer indicating which expert (feed-forward layer) to use.
            self_attention_logit_biases: An optional Tensor representing the self-attention biases.
            cross_attention_data: An optional Tensor of shape [batch, source_length, source_dim].
            cross_attention_logit_biases: An optional Tensor representing the cross-attention
                biases.
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

        cfg = self.config
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
            self_atten_state, self_atten_outputs = None, self.self_attention(
                target=data,
                attention_logit_biases=self_attention_logit_biases,
                return_aux=self_attention_return_aux,
            )
        elif mode == ForwardMode.INIT_STATES:
            self_atten_state, self_atten_outputs = self.self_attention.prefill_states(
                time_step=cached_states["self_attention"],
                target=data,
                attention_logit_biases=self_attention_logit_biases,
                return_aux=self_attention_return_aux,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            self_atten_state, self_atten_outputs = self.self_attention.extend_step(
                cached_states=cached_states["self_attention"],
                target=data,
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

        if feed_forward_index <= cfg.num_ffn - 1:
            # If an expert has been created for the current data modality, route data to the
            # expert specified by `feed_forward_index`.
            data = getattr(self, f"feed_forward{feed_forward_index}")(data)
        else:
            # If no multimodal expert has been created, split the multimodal data and feed to the
            # corresponding expert then concatenate. Currently only support image and text experts.
            if not isinstance(cfg.max_text_len, int):
                raise ValueError(f"max_text_len should be an integer, but got {cfg.max_text_len}.")
            data_txt, data_img = jnp.split(data, [cfg.max_text_len], axis=1)
            data_txt = getattr(self, f"feed_forward{TEXT_MODALITY}")(data_txt)
            data_img = getattr(self, f"feed_forward{IMAGE_MODALITY}")(data_img)
            data = jnp.concatenate([data_txt, data_img], axis=1)

        self.vlog(3, "transformer.output=%s", data.sum())
        return dict(self_attention=self_atten_state), self.Output(
            data=data,
            self_attention_probs=self_atten_outputs.probs,
            self_attention_kv_state=self_atten_outputs.kv_state,
            cross_attention_probs=cross_attention_probs,
        )

    # pylint: disable-next=arguments-differ
    def forward(
        self,
        data: Tensor,
        *,
        feed_forward_index: int = 0,
        **kwargs,
    ) -> Output:
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            feed_forward_index=feed_forward_index,
            **kwargs,
        )
        return output

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> NestedTensor:
        return dict(
            self_attention=self.self_attention.init_states(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    # pylint: disable-next=arguments-differ
    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        feed_forward_index: int = 0,
        **kwargs,
    ) -> Tuple[NestedTensor, Output]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=dict(self_attention=time_step),
            data=data,
            feed_forward_index=feed_forward_index,
            **kwargs,
        )

    # pylint: disable-next=arguments-differ
    def extend_step(
        self,
        cached_states: NestedTensor,
        data: Tensor,
        *,
        feed_forward_index: int = 0,
        **kwargs,
    ) -> Tuple[NestedTensor, Output]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            cached_states=cached_states,
            data=data,
            feed_forward_index=feed_forward_index,
            **kwargs,
        )


class MultiModalEncoder(BaseLayer):
    """A multimodal encoder implementation.

    Reference: https://github.com/microsoft/unilm/blob/master/vlmo/vlmo/modules/vlmo_module.py#L667
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures MultiModalEncoder."""

        # The output dimension of visual embedding. This is the input and output dimension of
        # Transformer and Pooler layer.
        output_dim: Required[int] = REQUIRED
        # The number of prepending tokens such as a classification token.
        num_cls_tokens: Required[int] = REQUIRED
        # The config to indicate use mask tokens or not. If true, all masked tokens in the inputs
        # will be replaced by a trainable vector of shape [output_dim].
        use_mask_tokens: bool = False
        # Visual embedding.
        visual_embed: VisualEmbedding.Config = VisualEmbedding.default_config()
        # Visual positional embedding config.
        visual_pos_emb: InstantiableConfig = LearnedPositionalEmbedding.default_config()
        # Text embedding.
        text_embed: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config()
        # The modality type specific embedding config.
        modality_emb: InstantiableConfig = Embedding.default_config()
        # Input dropout config.
        input_dropout: InstantiableConfig = Dropout.default_config().set(rate=0.1)
        # The stacked multiway transformer layer config.
        transformer: InstantiableConfig = StackedTransformerLayer.default_config().set(
            layer=MultiwayTransformerLayer.default_config()
        )
        # The normalization layer config for encoder output.
        output_norm: InstantiableConfig = layer_norm_config()
        # Pooler: the default is FirstNTokenPooling.
        pooler: BasePoolingLayer.Config = FirstNTokenPooling.default_config()

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        # TODO(xianzhi): verify the model initialization method.
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: param_init.WeightInitializer.default_config().set(
                    fan="fan_avg", distribution="uniform"
                )
            }
        )
        # pylint: disable=no-member
        cfg.visual_pos_emb.param_init = GaussianInitializer.default_config().set(std=0.02)
        transformer_layer_cfg = cfg.transformer.layer
        transformer_layer_cfg.feed_forward.activation = "nn.gelu"
        transformer_layer_cfg.feed_forward.norm = layer_norm_config()
        transformer_layer_cfg.feed_forward.hidden_dim = scaled_hidden_dim(4)
        transformer_layer_cfg.self_attention.norm = layer_norm_config()
        # pylint: enable=no-member
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        param_specs = {}
        if cfg.num_cls_tokens:
            param_specs["cls_token"] = ParameterSpec(
                shape=(1, cfg.num_cls_tokens, cfg.output_dim),
                mesh_axes=(None, None, "model"),
                initializer=param_init.constant_initializer(0.0),
            )
        if cfg.use_mask_tokens:
            param_specs["mask_token"] = ParameterSpec(
                shape=(1, 1, cfg.output_dim),
                mesh_axes=(None, None, "model"),
                initializer=param_init.constant_initializer(0.0),
            )
        return param_specs

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("visual_embed", cfg.visual_embed.set(output_dim=cfg.output_dim))
        self._add_child("visual_pos_emb", cfg.visual_pos_emb.clone(dim=cfg.output_dim))
        self._add_child("text_embed", cfg.text_embed.set(dim=cfg.output_dim))
        self._add_child(
            "modality_emb",
            cfg.modality_emb.clone(dim=cfg.output_dim),
        )
        self._add_child("input_dropout", cfg.input_dropout)
        self._add_child("transformer", cfg.transformer.clone(input_dim=cfg.output_dim))
        self._add_child("output_norm", cfg.output_norm.clone(input_dim=cfg.output_dim))
        self._add_child(
            "pooler", cfg.pooler.set(input_dim=cfg.output_dim, output_dim=cfg.output_dim)
        )

    def get_visual_embed(
        self, data: Tensor, *, modality: int, is_masked: Optional[Tensor] = None
    ) -> Tensor:
        """Generates visual embeddings.

        Args:
            data: a float Tensor of shape (batch, height, width, 3) representing the input image.
            modality: the modality index.
            is_masked: an optional boolean Tensor of shape (batch, length) representing the masked
                position for the pachified image.

        Returns:
            A float Tensor of shape (batch, length, output_dim) representing the visual embedding.
        """
        cfg = self.config
        batch_size = data.shape[0]
        x = self.visual_embed(data)
        if cfg.use_mask_tokens and is_masked is not None:
            mask_tokens = jnp.tile(self.parameters["mask_token"], (batch_size, x.shape[1], 1))
            is_masked = jnp.expand_dims(is_masked, axis=-1)
            x = x * (1 - is_masked) + mask_tokens * is_masked
        # Append the classification token.
        if cfg.num_cls_tokens > 0:
            cls_tokens = jnp.tile(self.parameters["cls_token"], (batch_size, 1, 1))
            x = jnp.concatenate([cls_tokens, x], axis=1)
        # Add positional embedding.
        x = x + self.visual_pos_emb.embeddings()
        # Add modality type embedding.
        x = x + self.modality_emb(jnp.full(x.shape[:2], modality))
        return x

    def get_text_embed(self, data: Tensor, modality: int) -> Tensor:
        # Same text embeddings as Bert.
        x = self.text_embed(inputs=data)
        # Add modality type embedding.
        x = x + self.modality_emb(jnp.full(x.shape[:2], modality))
        return x

    def forward(
        self, inputs: Dict[int, Tensor], is_masked: Optional[Tensor] = None
    ) -> Dict[int, Tensor]:
        """Compute prediction on the multimodal inputs.

        Args:
            inputs: A Dict that contains modality type and modality inputs with keys:
                0 or TEXT_MODALITY: representing texts in shape [batch, seq_len].
                1 or IMAGE_MODALITY: representing images in shape [batch, height, width, channels].
                2 or TEXT_IMAGE_MODALITY: a dict that contains text-image multimodal data with the
                    same format as monomodal data.
            is_masked: a boolean Tensor in shape (batch, length), representing masked positions for
                the patchifie input sequence.

        Returns:
            A dict containing modality type (same as the inputs) and embeddings in shape
                (batch, output_dim).

        Raises:
            ValueError: If modality is not supported.
        """
        outputs = {}
        for modality, data in inputs.items():
            with child_context(f"modality{modality}", module=self):
                # Generate modality-specific embeddings.
                if modality == TEXT_MODALITY:
                    x = self.get_text_embed(data, TEXT_MODALITY)
                elif modality == IMAGE_MODALITY:
                    x = self.get_visual_embed(data, modality=IMAGE_MODALITY, is_masked=is_masked)
                elif modality == TEXT_IMAGE_MODALITY:
                    with child_context(f"modality{TEXT_MODALITY}", module=self):
                        x_txt = self.get_text_embed(data[TEXT_MODALITY], TEXT_MODALITY)
                    with child_context(f"modality{IMAGE_MODALITY}", module=self):
                        x_img = self.get_visual_embed(
                            data[IMAGE_MODALITY], modality=IMAGE_MODALITY, is_masked=is_masked
                        )
                    x = jnp.concatenate([x_txt, x_img], axis=1)
                else:
                    raise ValueError(f"Unsupported modality {modality}.")
                x = self.input_dropout(x)
                # Multiway transformer layers.
                x = self.transformer(x, feed_forward_index=modality).data
                x = self.output_norm(x)
                x = self.pooler(x)
                x = jnp.squeeze(x, axis=1)
                outputs[modality] = x
        return outputs


def _set_model_config(
    cfg,
    *,
    num_modalities: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    text_vocab_size: int,
    max_text_len: int,
    feed_forward_dim: Optional[int] = None,
    image_size: Tuple[int, int] = (224, 224),
    patch_size: Tuple[int, int] = (16, 16),
    stride: Optional[Tuple[int, int]] = None,
    num_cls_tokens: int = 1,
    dtype: jnp.dtype = jnp.float32,
    dropout_rate: float = 0.0,
    peak_stochastic_depth_rate: Optional[float] = None,
):
    cfg.dtype = dtype
    if stride is None:
        stride = patch_size
    if not all((i - p) % s == 0 for i, p, s in zip(image_size, patch_size, stride)):
        raise ValueError(
            f"stride ({stride}) must divide image_size-patch_size ({image_size}-{patch_size})"
        )

    cfg.modality_emb.num_embeddings = num_modalities

    seq_len = int(np.prod([(i - p) // s + 1 for i, p, s in zip(image_size, patch_size, stride)]))
    seq_len += num_cls_tokens
    cfg.num_cls_tokens = num_cls_tokens

    # Set visual embedding configs.
    cfg.visual_embed.convert_to_sequence.patch_size = patch_size
    cfg.visual_embed.convert_to_sequence.stride = stride
    cfg.visual_pos_emb.shape = (seq_len,)

    # Set text embedding configs.
    cfg.text_embed.vocab_size = text_vocab_size
    cfg.text_embed.pos_emb = LearnedPositionalEmbedding.default_config().set(shape=(max_text_len,))
    cfg.text_embed.norm = LayerNorm.default_config()

    # Set multimodal encoder layer configs.
    cfg.set(output_dim=model_dim)
    cfg.transformer.num_layers = num_layers
    cfg.transformer.layer.num_ffn = num_modalities
    cfg.transformer.layer.max_text_len = max_text_len
    if feed_forward_dim is not None:
        cfg.transformer.layer.feed_forward.hidden_dim = feed_forward_dim
    cfg.transformer.layer.self_attention.attention.num_heads = num_heads
    cfg.pooler = FirstNTokenPooling.default_config().set(num_outputs=num_cls_tokens)

    # Set regularization configs.
    set_dropout_rate_recursively(cfg, dropout_rate)
    if peak_stochastic_depth_rate is not None:
        cfg.transformer.peak_stochastic_depth_rate = peak_stochastic_depth_rate

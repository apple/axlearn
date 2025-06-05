# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/vision_transformer:
# Copyright 2023 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# facebookresearch/deit:
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Vision transformer layers.

References:
- https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py
"""
# Many similarities with resnet.
# pylint: disable=duplicate-code
import copy
import math
from typing import Any, Optional

import jax.nn
import numpy as np
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    LearnedPositionalEmbedding,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    build_remat_spec,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.convolution import Conv2D
from axlearn.common.layers import (
    Dropout,
    DropToken,
    L2Norm,
    LayerNorm,
    Linear,
    set_dropout_rate_recursively,
)
from axlearn.common.module import Module, Tensor
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, GaussianInitializer
from axlearn.common.poolings import AveragePooling, BasePoolingLayer, FirstNTokenPooling


def layer_norm_config(eps=1e-6):
    return LayerNorm.default_config().set(eps=eps)


def sequence_to_space_with_scaling(
    x: Tensor,
    *,
    num_cls_tokens: int,
    target_len: Optional[int] = None,
) -> dict[str, Tensor]:
    """A method to convert a 1D sequence features to 2D space features with 2D scaling.

    Inspired from https://github.com/facebookresearch/deit/blob/main/main.py#L284-L302.

    Args:
        x: a Tensor of sequence features in shape [batch, length, dim].
        num_cls_tokens: number of classification tokens prepended to the 1D sequence.
        target_len: None or a square number representing the target length after scaling. If set,
            the 2D features will be resized by `jax.image.resize`.

    Returns:
        A Dict that contains `sequence_features` in shape [batch, length, dim] and
            `space_features` in shape [batch, height, width, dim].

    Raises:
        ValueError: if the input sequence length or target_len are not square numbers.
    """
    cls_tokens, seq_feat = x[:, :num_cls_tokens, :], x[:, num_cls_tokens:, :]
    batch_size, seq_len, dim = seq_feat.shape
    space_len = int(math.sqrt(seq_len))
    if space_len**2 != seq_len:
        raise ValueError(f"seq_len has to be a square number, but got {seq_len}.")
    space_feat = jnp.reshape(seq_feat, (batch_size, space_len, space_len, dim))

    # If target_len is set, scale feat in the space domain.
    if target_len:
        new_space_len = int(math.sqrt(target_len))
        if new_space_len**2 != target_len:
            raise ValueError(f"target_len has to be a square number, but got {target_len}.")
        new_shape = (batch_size, new_space_len, new_space_len, dim)
        # Refer to https://jax.readthedocs.io/en/latest/_autosummary/jax.image.resize.html for
        # more resizing methods.
        space_feat = jax.image.resize(space_feat, new_shape, method="bicubic")

    seq_feat = jnp.reshape(space_feat, (batch_size, -1, dim))
    seq_feat = jnp.concatenate([cls_tokens, seq_feat], axis=1)
    return {"sequence_features": seq_feat, "space_features": space_feat}


class ConvertToSequence(BaseLayer):
    """A layer to flatten 2-D images to 1-D sequences."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ConvertToSequence."""

        patch_size: tuple[int, int] = (16, 16)  # The 2-D patch size.
        input_dim: Required[int] = REQUIRED  # The input feature dim.
        output_dim: Required[int] = REQUIRED  # The output feature dim.
        conv: InstantiableConfig = Conv2D.default_config().set(
            # We may not need bias when we later apply a learnable positional embedding, but we
            # use it here for consistency with the original ViT implementation.
            bias=True,
            param_partition_spec=(None, None, None, "model"),
        )
        # The stride for patching. This defaults to the patch size.
        stride: Optional[tuple[int, int]] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child(
            "conv",
            cfg.conv.clone(
                window=cfg.patch_size,
                padding=((0, 0), (0, 0)),
                strides=cfg.stride if cfg.stride else cfg.patch_size,
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Converts 2D to 1D.

        Args:
            x: a Tensor of shape [B, H, W, input_dim].

        Returns:
            A tensor of shape [B, H*W, output_dim].
        """
        x = self.conv(x)
        batch, height, width, output_dim = x.shape
        return jnp.reshape(x, (batch, height * width, output_dim))


class Encoder1D(BaseLayer):
    """A sequence encoder consisting of multiple transformer layers."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures Encoder1D."""

        # Input feature dim.
        input_dim: Required[int] = REQUIRED
        # Number of cls tokens added to the inputs.
        num_cls_tokens: int = 1
        # Input norm used in the CLIP model.
        input_norm: Optional[InstantiableConfig] = None
        # Input dropout config.
        input_dropout: InstantiableConfig = Dropout.default_config().set(rate=0.1)
        # Positional embedding config.
        pos_emb: InstantiableConfig = LearnedPositionalEmbedding.default_config()
        # The transformer layer stack config.
        transformer: BaseStackedTransformerLayer.Config = StackedTransformerLayer.default_config()
        # The normalization layer config for encoder output.
        output_norm: InstantiableConfig = layer_norm_config()
        # The DropToken layer proposed in the FLIP paper.
        # This layer basically drops x% of visual tokens during training.
        # https://arxiv.org/pdf/2212.00794.pdf
        drop_token: DropToken.Config = DropToken.default_config()
        # Use absolute positional embedding.
        use_pos_emb: bool = True

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        # pylint: disable=no-member
        # https://github.com/google-research/vision_transformer/blob/dc8ddbcdeefd281d6cc7fea0c97355495688ca9c/vit_jax/models.py#L189
        if cfg.use_pos_emb:
            cfg.pos_emb.param_init = GaussianInitializer.default_config().set(std=0.02)
        # Vision transformer uses 'gelu' and dropout=0.1 by default.
        set_dropout_rate_recursively(cfg, dropout_rate=0.1)
        transformer_layer_cfg = cfg.transformer.layer
        transformer_layer_cfg.feed_forward.activation = "nn.gelu"
        transformer_layer_cfg.feed_forward.norm = layer_norm_config()
        transformer_layer_cfg.feed_forward.hidden_dim = scaled_hidden_dim(4)
        transformer_layer_cfg.self_attention.norm = layer_norm_config()
        # pylint: enable=no-member
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("input_dropout", cfg.input_dropout)
        if cfg.input_norm is not None:
            # Layer norm used by CLIP
            self._add_child("input_norm", cfg.input_norm.clone(input_dim=cfg.input_dim))
        if cfg.use_pos_emb:
            self._add_child("pos_emb", cfg.pos_emb.clone(dim=cfg.input_dim))
        self._add_child("transformer", cfg.transformer.clone(input_dim=cfg.input_dim))
        self._add_child("output_norm", cfg.output_norm.clone(input_dim=cfg.input_dim))
        self._add_child("drop_token", cfg.drop_token.set(num_cls_tokens=cfg.num_cls_tokens))

    def forward(self, inputs: Tensor) -> Tensor:
        """Computes global features for a given sequence..

        Args:
            inputs: A Tensor of shape [B, length, input_dim].

        Returns:
            Normalized transformer features of shape [B, length, input_dim].
        """
        cfg = self.config
        if cfg.use_pos_emb:
            pos_emb = self.pos_emb.embeddings()
            inputs_len, pos_emb_len = inputs.shape[-2], pos_emb.shape[-2]
            # Scale pos_emb in 2D space to match the length of input seq.
            if inputs_len != pos_emb_len:
                pos_emb = sequence_to_space_with_scaling(
                    pos_emb[None, :],
                    num_cls_tokens=cfg.num_cls_tokens,
                    target_len=inputs_len - cfg.num_cls_tokens,
                )["sequence_features"]
            x = inputs + pos_emb
        else:
            x = inputs
        x = self.drop_token(x)
        x = self.input_dropout(x)
        if self.config.input_norm is not None:
            x = self.input_norm(x)
        x = self.transformer(x).data
        x = self.output_norm(x)
        return x


class VisualEmbedding(BaseLayer):
    """A vision embedding layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures VisualEmbedding."""

        # Output dimension of the visual embedding layer.
        output_dim: Required[int] = REQUIRED
        # An optional 2-D encoder, e.g., ResNetEncoder.
        encoder_2d: Optional[InstantiableConfig] = None
        # The layer to flatten 2-D images or features to 1-D sequences.
        convert_to_sequence: InstantiableConfig = ConvertToSequence.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.encoder_2d is not None:
            enc = self._add_child("encoder_2d", cfg.encoder_2d)  # ResNetEncoder
            feature_dim = enc.output_dim
        else:
            self.encoder_2d = lambda x: x  # identity
            feature_dim = 3
        self._add_child(
            "convert_to_sequence",
            cfg.convert_to_sequence.clone(input_dim=feature_dim, output_dim=cfg.output_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Compute prediction on an image.

        Args:
            image: The input image. Shape: (batch, height, width, channels).

        Returns:
            Tokenized visual embedding: A tensor of shape (batch, seq_len, output_dim).
        """
        x = self.encoder_2d(image)
        x = self.convert_to_sequence(x)
        return x


class VisionTransformer(BaseLayer):
    """A Vision Transformer encoder."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures VisionTransformer."""

        # The dimension for the output of visual embedding.
        # This is the input and output dimension of Encoder and Pooler layer.
        output_dim: Required[int] = REQUIRED
        # The number of prepending tokens such as a classification token.
        num_cls_tokens: Required[int] = REQUIRED
        # Visual embedding.
        visual_embed: VisualEmbedding.Config = VisualEmbedding.default_config()
        # The 1-D encoder.
        encoder_1d: Encoder1D.Config = Encoder1D.default_config()
        # Pooler: the default is FirstNTokenPooling.
        pooler: BasePoolingLayer.Config = FirstNTokenPooling.default_config()
        # The config to indicate use mask tokens or not.
        use_mask_tokens: bool = False
        # Optional projection layers after pooler.
        output_proj: Optional[InstantiableConfig] = None
        # Optional normalization layer for projection layer.
        output_proj_norm: Optional[InstantiableConfig] = None

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        # https://github.com/google-research/vision_transformer/blob/e7d87a59784503b5bd8825ec368bca17822fb959/vit_jax/models.py#L76.
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: param_init.WeightInitializer.default_config().set(
                    fan="fan_avg", distribution="uniform"
                )
            }
        )
        return cfg

    def get_output_features_dim(self) -> int:
        cfg = self.config
        if cfg.output_proj is not None:
            return cfg.output_proj.output_dim
        return cfg.output_dim

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
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
                initializer=param_init.gaussian_initializer(std=0.02),
            )
        return param_specs

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("visual_embed", cfg.visual_embed.set(output_dim=cfg.output_dim))
        self._add_child(
            "encoder_1d",
            cfg.encoder_1d.set(input_dim=cfg.output_dim, num_cls_tokens=cfg.num_cls_tokens),
        )
        self._add_child(
            "pooler", cfg.pooler.set(input_dim=cfg.output_dim, output_dim=cfg.output_dim)
        )
        if cfg.output_proj:
            self._add_child("output_proj", cfg.output_proj.clone(input_dim=cfg.output_dim))
        if cfg.output_proj_norm:
            self._add_child("output_proj_norm", cfg.output_proj_norm)

    def forward(self, image: Tensor, is_masked: Optional[Tensor] = None) -> dict[str, Tensor]:
        """Compute prediction on an image.

        Args:
            image: The input image. Shape: (batch, height, width, channels).
            is_masked: a boolean Tensor in shape (batch, length), representing masked positions
                for the patchifie input sequence.

        Returns:
            A dictionary containing:
            * "encoded_features": encoded features with shape (batch, length, output_dim);
            * "pooled_features": pooled features with shape (batch, num_outputs, output_dim);
            * "patch_features": encoded features without cls tokens with shape
                (batch, length - num_cls_tokens, output_dim). Representing patch-wise features;
            * "embedding": final model embedding with shape (batch, output_dim);
            * str(level): 2D form of "encoded_features" with shape
                (batch, height, width, output_dim), where height*width = length - num_cls_tokens.
                Will not be added if `drop_token` is enabled.
        """
        cfg = self.config
        batch_size = image.shape[0]
        x = self.visual_embed(image)
        if cfg.use_mask_tokens and is_masked is not None:
            mask_tokens = jnp.tile(self.parameters["mask_token"], (batch_size, x.shape[1], 1))
            is_masked = jnp.expand_dims(is_masked, axis=-1)
            x = x * (1 - is_masked) + mask_tokens * is_masked
        if cfg.num_cls_tokens > 0:
            cls_tokens = jnp.tile(self.parameters["cls_token"], (batch_size, 1, 1))
            x = jnp.concatenate([cls_tokens, x], axis=1)

        x = self.encoder_1d(x)
        pooled_output = self.pooler(x)

        outputs = {
            "encoded_features": x,
            "pooled_features": pooled_output[:, :1],
            "patch_features": x[:, cfg.num_cls_tokens :],
        }
        if cfg.num_cls_tokens > 1:
            outputs["distillation_features"] = pooled_output[:, 1:]

        embedding = jnp.squeeze(outputs["pooled_features"], axis=1)
        if "output_proj" in self.children:
            embedding = self.output_proj(embedding)
            if "output_proj_norm" in self.children:
                embedding = self.output_proj_norm(embedding)
        outputs["embedding"] = embedding

        # Output 2D encoded features is not compatible with drop_token.
        if cfg.encoder_1d.drop_token.rate == 0:
            level = int(math.log2(cfg.visual_embed.convert_to_sequence.patch_size[0]))
            # Add the 2D space representation of the 1D encoded features to outputs.
            outputs[str(level)] = sequence_to_space_with_scaling(
                outputs["encoded_features"], num_cls_tokens=cfg.num_cls_tokens
            )["space_features"]

        return outputs

    @property
    def endpoints_dims(self) -> dict[str, int]:
        """A dict of {str: hidden_dim} specifies dimension of intermediate and output features."""
        cfg = self.config
        patch_size = cfg.visual_embed.convert_to_sequence.patch_size[0]
        level = int(math.log2(patch_size))
        return {str(level): cfg.output_dim, "embedding": self.get_output_features_dim()}


_NAMED_VIT_MODELS = {
    "Test16": dict(num_layers=1, model_dim=8, num_heads=4),
    # Table 1 of https://arxiv.org/pdf/2010.11929.pdf.
    # Table 2 of https://arxiv.org/pdf/2106.04560.pdf.
    "Ti16": dict(num_layers=12, model_dim=192, num_heads=3),
    "S16": dict(num_layers=12, model_dim=384, num_heads=6),
    "B16": dict(num_layers=12, model_dim=768, num_heads=12),
    "B32": dict(num_layers=12, model_dim=768, num_heads=12, patch_size=(32, 32)),
    "L14": dict(num_layers=24, model_dim=1024, num_heads=16, patch_size=(14, 14)),
    "L16": dict(num_layers=24, model_dim=1024, num_heads=16),
    "L32": dict(num_layers=24, model_dim=1024, num_heads=16, patch_size=(32, 32)),
    # When patch_size=(14, 14), use global average pooling for feature extraction so that the
    # sequence length is 256 instead of 257.
    "H14": dict(
        num_layers=32,
        model_dim=1280,
        num_heads=16,
        patch_size=(14, 14),
        global_feature_extraction="gap",
    ),
    # Table 2 of https://arxiv.org/pdf/2106.04560.pdf.
    "g14-paper": dict(
        num_layers=40,
        model_dim=1408,
        num_heads=16,
        feed_forward_dim=6144,
        patch_size=(14, 14),
        global_feature_extraction="gap",
    ),
    "g14-clip": dict(
        num_layers=40,
        model_dim=1536,
        num_heads=16,
        patch_size=(14, 14),
    ),
    "G14": dict(
        num_layers=48,
        model_dim=1664,
        num_heads=16,
        feed_forward_dim=8192,
        patch_size=(14, 14),
        global_feature_extraction="gap",
        dropout_rate=0.0,
    ),
}


def _set_model_config(
    cfg: VisionTransformer.Config,
    *,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    feed_forward_dim: Optional[int] = None,
    image_size: tuple[int, int] = (224, 224),
    patch_size: tuple[int, int] = (16, 16),
    stride: Optional[tuple[int, int]] = None,
    # One of ["cls_token", "cls_distill_token", "gap"].
    global_feature_extraction: str = "cls_token",
    dtype: jnp.dtype = jnp.float32,
    dropout_rate: float = 0.1,
    transformer_stack_cfg: Optional[InstantiableConfig] = None,
    remat: bool = False,
    peak_stochastic_depth_rate: Optional[float] = None,
    output_proj_dim: Optional[int] = None,
    # Use positional embedding as the inputs to the Encoder.
    use_pos_emb: bool = True,
    # Cap the absolute values of logits by tanh.
    atten_logit_cap: Optional[float] = None,
):
    cfg.dtype = dtype
    if stride is None:
        stride = patch_size
    if not all((i - p) % s == 0 for i, p, s in zip(image_size, patch_size, stride)):
        raise ValueError(
            f"stride ({stride}) must divide image_size-patch_size ({image_size}-{patch_size})"
        )
    cfg.visual_embed.convert_to_sequence.patch_size = patch_size
    cfg.visual_embed.convert_to_sequence.stride = stride
    encoder_cfg = cfg.encoder_1d
    cfg.set(output_dim=model_dim)
    if transformer_stack_cfg is not None:
        if remat:
            transformer_stack_cfg.layer.remat_spec = build_remat_spec(
                transformer_stack_cfg
            )  # pytype: disable=attribute-error
        encoder_cfg.transformer = transformer_stack_cfg
    encoder_cfg.transformer.num_layers = num_layers
    seq_len = int(np.prod([(i - p) // s + 1 for i, p, s in zip(image_size, patch_size, stride)]))
    if global_feature_extraction in ["cls_token", "cls_distill_token"]:
        num_cls_tokens = 1 if global_feature_extraction == "cls_token" else 2
        seq_len += num_cls_tokens
        cfg.num_cls_tokens = num_cls_tokens
        encoder_cfg.num_cls_tokens = num_cls_tokens
        cfg.pooler = FirstNTokenPooling.default_config().set(num_outputs=num_cls_tokens)
    elif global_feature_extraction == "gap":
        cfg.num_cls_tokens = 0
        cfg.pooler = AveragePooling.default_config().set(num_outputs=1)
    else:
        raise ValueError(
            f"The global_feature_extraction ({global_feature_extraction}) is not supported."
        )
    encoder_cfg.use_pos_emb = use_pos_emb
    if use_pos_emb:
        encoder_cfg.pos_emb.shape = (seq_len,)

    # pylint: disable=attribute-error
    if feed_forward_dim is not None:
        encoder_cfg.transformer.layer.feed_forward.hidden_dim = feed_forward_dim
    encoder_cfg.transformer.layer.self_attention.attention.num_heads = num_heads

    if atten_logit_cap is not None:
        encoder_cfg.transformer.layer.self_attention.attention.atten_logit_cap = atten_logit_cap
    # pylint: enable=attribute-error

    set_dropout_rate_recursively(cfg, dropout_rate)
    if peak_stochastic_depth_rate is not None:
        encoder_cfg.transformer.peak_stochastic_depth_rate = peak_stochastic_depth_rate

    if output_proj_dim is not None:
        cfg.set(
            output_proj=Linear.default_config().set(output_dim=output_proj_dim, bias=False),
            output_proj_norm=L2Norm.default_config(),
        )


def build_vit_model_config(**kwargs):
    cfg = VisionTransformer.default_config()
    _set_model_config(cfg, **kwargs)
    return cfg


def named_model_configs(
    *,
    extra_settings: Optional[dict[str, Any]] = None,
    include_models: Optional[list[str]] = None,  # If set, only get models configs in the list.
) -> dict[str, InstantiableConfig]:
    # Avoid modifying `_NAMED_VIT_MODELS` in place.
    models = copy.deepcopy(_NAMED_VIT_MODELS)
    # Large ViT models with RepeatedTransformerLayer.
    for model_id in ["H14", "g14-paper", "g14-clip", "G14"]:
        models[f"{model_id}-repeat"] = dict(
            **models[model_id],
            transformer_stack_cfg=RepeatedTransformerLayer.default_config().set(
                # pylint: disable-next=no-member
                layer=Encoder1D.default_config().transformer.layer
            ),
            remat=True,
        )
    config_map = {}
    for model_name, model_settings in models.items():
        if include_models and model_name not in include_models:
            continue
        if extra_settings:
            model_settings.update(extra_settings)
        cfg = build_vit_model_config(**model_settings)
        config_map[model_name] = cfg
    return config_map

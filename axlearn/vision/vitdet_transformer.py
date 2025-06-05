# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/detectron2:
# Copyright 2019-2020, detectron2 contributors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""An AXLearn implementation of ViTDet transformer layers.

ViTDet References:
- https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
"""
import copy
import math
from collections.abc import Sequence
from typing import Any, Optional

import jax.nn
from jax import numpy as jnp

from axlearn.common.config import InstantiableConfig, config_class
from axlearn.common.layers import set_dropout_rate_recursively
from axlearn.common.module import Module, Tensor
from axlearn.common.poolings import AveragePooling
from axlearn.common.vision_transformer import Encoder1D, VisionTransformer
from axlearn.vision.attention import StackedWindowedTransformerLayer, WindowedSelfAttentionLayer


def get_abs_pos(
    abs_pos: Tensor,
    has_cls_tokens: bool,
    seq_len: int,
) -> Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos: A float Tensor of absolute positional embeddings in shape (length, dim).
        has_cls_tokens: If True, has 1 embedding in abs_pos for cls token.
        seq_len: length of input visual tokens.

    Returns:
        A Tensor of absolute positional embeddings after processing in shape (1, length, dim)

    Raises:
        ValueError: if the input sequence length or emb_len are not square numbers.
    """
    height = width = int(math.sqrt(seq_len))
    if height * width != seq_len:
        raise ValueError(f"seq_len has to be a square number, but got {seq_len}.")
    if has_cls_tokens:
        abs_pos = abs_pos[1:, :]
    emb_len, dim = abs_pos.shape
    size = int(math.sqrt(emb_len))
    if size**2 != emb_len:
        raise ValueError(f"emb_len has to be a square number, but got {emb_len}.")

    # TODO: A difference between jax.image.resize and F.interpolate when using "bicubic"
    # Ref: https://github.com/google/jax/issues/15768
    if size != height:
        new_abs_pos = jax.image.resize(
            image=jnp.reshape(abs_pos, (1, size, size, -1)),
            shape=(1, height, width, dim),
            method="bicubic",
        )
        new_abs_pos = jnp.reshape(new_abs_pos, (1, height * width, -1))
        return new_abs_pos
    else:
        return abs_pos


class ViTDetEncoder(Encoder1D):
    """ViTDet Encoder."""

    @config_class
    class Config(Encoder1D.Config):
        """Configures ViTDetEncoder."""

        # The transformer layer stack config.
        transformer: InstantiableConfig = StackedWindowedTransformerLayer.default_config()
        # If True, pretraining models use class token.
        pretrain_use_cls_token: bool = True

    @classmethod
    def default_config(cls):
        cfg = super().default_config()

        transformer_layer_cfg = cfg.transformer.layer
        transformer_layer_cfg.self_attention = WindowedSelfAttentionLayer.default_config()
        return cfg

    def __init__(self, cfg: Config, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)

    def forward(self, inputs: Tensor) -> Tensor:
        """Computes global features for a given sequence..

        Args:
            inputs: A Tensor of shape [B, length, input_dim].

        Returns:
            Normalized transformer features of shape [B, length, input_dim].
        """
        cfg = self.config
        pos_emb = self.pos_emb.embeddings()
        inputs_len, pos_emb_len = inputs.shape[-2], pos_emb.shape[-2]
        if cfg.use_pos_emb:
            if inputs_len != pos_emb_len:
                # Scale pos_emb in 2D space to match the length of inputs
                pos_emb = get_abs_pos(pos_emb, cfg.pretrain_use_cls_token, inputs_len)
                x = inputs + pos_emb
            else:
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


_NAMED_VITDET_MODELS = {
    "Test16": dict(
        num_layers=3,
        model_dim=8,
        num_heads=4,
        window_size=14,
        window_block_indexes=list(range(0, 2)),
    ),
    "B16": dict(
        num_layers=12,
        model_dim=768,
        num_heads=12,
        window_size=14,
        window_block_indexes=(
            list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11))
        ),
    ),
    "L16": dict(
        num_layers=24,
        model_dim=1024,
        num_heads=16,
        window_size=14,
        window_block_indexes=(
            list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
        ),
        peak_stochastic_depth_rate=0.4,
    ),
    "H16": dict(
        num_layers=32,
        model_dim=1280,
        num_heads=16,
        window_size=14,
        window_block_indexes=(
            list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
        ),
        peak_stochastic_depth_rate=0.5,
    ),
}


def _set_model_config(
    cfg,
    *,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    feed_forward_dim: Optional[int] = None,
    image_size: tuple[int, int] = (224, 224),
    patch_size: tuple[int, int] = (16, 16),
    stride: Optional[tuple[int, int]] = None,
    global_feature_extraction: str = "gap",
    dtype: jnp.dtype = jnp.float32,
    dropout_rate: float = 0.0,
    transformer_stack_cfg: Optional[InstantiableConfig] = None,
    peak_stochastic_depth_rate: Optional[float] = None,
    # If True, use absolute positional embeddings.
    use_pos_emb: bool = True,
    # if True, add relative positional embeddings to the attention map.
    use_rel_pos_emb: bool = False,
    # Window size for window attention blocks.
    window_size: int = 14,
    # Indexes for blocks using window attention.
    window_block_indexes: Optional[Sequence[int]] = None,
    # Input image size for pretraining models.
    pretrain_image_size: int = 224,
    # Input patch size for pretraining models.
    pretrain_patch_size: int = 16,
    # If True, class token exists in pretraining models.
    pretrain_use_cls_token: bool = True,
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
    cfg.set(output_dim=model_dim)

    # Here we specify the encoder_1d as the customized ViTDetEncoder.
    cfg.encoder_1d = ViTDetEncoder.default_config()
    encoder_cfg = cfg.encoder_1d
    encoder_cfg.use_pos_emb = use_pos_emb
    encoder_cfg.pretrain_use_cls_token = pretrain_use_cls_token

    if transformer_stack_cfg is not None:
        encoder_cfg.transformer = transformer_stack_cfg
    encoder_cfg.transformer.num_layers = num_layers
    encoder_cfg.transformer.window_size = window_size
    encoder_cfg.transformer.window_block_indexes = window_block_indexes

    if global_feature_extraction == "gap":
        cfg.num_cls_tokens = 0
        cfg.pooler = AveragePooling.default_config().set(num_outputs=1)
    else:
        raise ValueError(
            f"The global_feature_extraction ({global_feature_extraction}) is not supported."
        )

    num_patches = (pretrain_image_size // pretrain_patch_size) ** 2
    seq_len = (num_patches + 1) if pretrain_use_cls_token else num_patches
    if use_pos_emb:
        encoder_cfg.pos_emb.shape = (seq_len,)

    input_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
    encoder_cfg.transformer.input_size = input_size

    if feed_forward_dim is not None:
        encoder_cfg.transformer.layer.feed_forward.hidden_dim = feed_forward_dim
    encoder_cfg.transformer.layer.self_attention.attention.num_heads = num_heads
    encoder_cfg.transformer.layer.self_attention.attention.use_rel_pos_emb = use_rel_pos_emb
    set_dropout_rate_recursively(cfg, dropout_rate)

    if peak_stochastic_depth_rate is not None:
        encoder_cfg.transformer.peak_stochastic_depth_rate = peak_stochastic_depth_rate


def build_vitdet_model_config(**kwargs):
    cfg = VisionTransformer.default_config()
    _set_model_config(cfg, **kwargs)
    return cfg


def named_model_configs(
    *,
    extra_settings: Optional[dict[str, Any]] = None,
    include_models: Optional[list[str]] = None,  # If set, only get models configs in the list.
) -> dict[str, InstantiableConfig]:
    # Avoid modifying `_NAMED_VITDET_MODELS` in place.
    models = copy.deepcopy(_NAMED_VITDET_MODELS)
    config_map = {}
    for model_name, model_settings in models.items():
        if include_models and model_name not in include_models:
            continue
        if extra_settings:
            model_settings.update(extra_settings)
        cfg = build_vitdet_model_config(**model_settings)
        config_map[model_name] = cfg
    return config_map

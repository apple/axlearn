# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# microsoft/unilm:
# Copyright (c) 2022 Microsoft.
# Licensed under The MIT License.

"""BEiT-3 Image Tokenizer

Sect. 2.2 of https://arxiv.org/pdf/2208.10442.pdf, BEiT-3 uses BEiT-v2 image tokenizer.
In the official implementation of BEiT-v2 image tokenizer, they used EMA on KMeans weights.
Currently, it is unknown whether BEiT-3 finetuned its tokenizer.
Therefore, we directly use KMeansQuantizer.
Later, when we support image tokenizer finetuning, we might implement EMA on KMeans weights.

Ref: https://github.com/microsoft/unilm/blob/master/beit2/norm_ema_quantizer.py
"""

from typing import Optional, Union

import jax
import jax.numpy as jnp

from axlearn.common.attention import scaled_hidden_dim
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, FunctionConfigBase, Required, config_class
from axlearn.common.layers import (
    L2Norm,
    LayerNorm,
    Linear,
    set_dropout_rate_recursively,
    set_norm_recursively,
)
from axlearn.common.module import Module
from axlearn.common.param_init import ConstantInitializer
from axlearn.common.poolings import AveragePooling, BasePoolingLayer
from axlearn.common.quantizer import KmeansVectorQuantizer
from axlearn.common.utils import Tensor
from axlearn.common.vision_transformer import VisionTransformer
from axlearn.vision.clip import set_transformer_config


class BEiTStochasticDepth(BaseLayer):
    """Creates a stochastic depth layer for BEiT image tokenizer.

    The BEiT image tokenizer stochastic depth layer adds a gamma parameter.
    The forward function is:
        x = stochastic_depth(gamma * x),
    where gamma is constant initialized and with shape of (1, 1, input_dim).

    Reference:
    https://github.com/microsoft/unilm/blob/716e4726e6154b4feb95db0b4dffd925bb5f5218/beit2/modeling_finetune.py#L197
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BEiTStochasticDepth."""

        input_dim: Required[int] = REQUIRED
        rate: Optional[float] = None  # Drop rate of this layer.
        mode: str = "row"  # One mode of ['batch', 'row'].

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_init = ConstantInitializer.default_config().set(value=1)
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        param_specs = {}
        param_specs["gamma"] = ParameterSpec(
            shape=(1, 1, self.config.input_dim),
            mesh_axes=None,
        )
        return param_specs

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.parameters["gamma"]
        cfg = self.config
        if not self.is_training or cfg.rate is None or cfg.rate == 0:
            return x
        if cfg.rate < 0.0 or cfg.rate >= 1.0:
            raise ValueError(f"Drop rate needs to be in [0, 1), but got {cfg.rate}.")
        if cfg.mode == "row":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
        else:
            assert cfg.mode == "batch"
            shape = [1] * x.ndim
        keep_prob = 1.0 - cfg.rate
        random_tensor = keep_prob + jax.random.uniform(self.prng_key, shape, dtype=x.dtype)
        binary_tensor = jnp.floor(random_tensor)
        return x * binary_tensor / keep_prob


class BEiTImageTokenizerMapping(BaseLayer):
    """Output projection for BEiTImageTokenizer.

    Backbone output (model_dim) -> Linear1 (model_dim) -> tanh -> Linear2 (dim=32)

    https://github.com/microsoft/unilm/blob/716e4726e6154b4feb95db0b4dffd925bb5f5218/beit2/modeling_vqkd.py#L87-L91
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BEiTImageTokenizerMapping."""

        # The input_dim for BEiT-3 is ViT model_dim.
        input_dim: Required[int] = REQUIRED
        # The output_dim for BEiT-3 is 32.
        # https://github.com/microsoft/unilm/blob/716e4726e6154b4feb95db0b4dffd925bb5f5218/beit2/modeling_vqkd.py#L34
        output_dim: Required[int] = REQUIRED
        linear1: Linear.Config = Linear.default_config()
        linear2: Linear.Config = Linear.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # Random projection uses Xavier initialization.
        self._add_child(
            "linear1", cfg.linear1.set(input_dim=cfg.input_dim, output_dim=cfg.input_dim)
        )
        self._add_child(
            "linear2", cfg.linear2.set(input_dim=cfg.input_dim, output_dim=cfg.output_dim)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Output projection for BEiTImageTokenizer.

        Args:
            inputs: Tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            A tensor of shape [batch_size, seq_len, output_dim]
        """
        x = self.linear1(inputs)
        x = jnp.tanh(x)
        x = self.linear2(x)
        return x


class BEiTImageVQKD(BaseLayer):
    """Image encoder for BEiTImageTokenizer.

    https://github.com/microsoft/unilm/blob/716e4726e6154b4feb95db0b4dffd925bb5f5218/beit2/modeling_vqkd.py#L87-L91
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BEiTImageVQKD."""

        backbone: VisionTransformer.Config = VisionTransformer.default_config()
        output_proj: BEiTImageTokenizerMapping.Config = BEiTImageTokenizerMapping.default_config()
        quantizer: KmeansVectorQuantizer.Config = KmeansVectorQuantizer.default_config()
        l2norm: L2Norm.Config = L2Norm.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # Random projection uses Xavier initialization.
        self._add_child("backbone", cfg.backbone)
        self._add_child("quantizer", cfg.quantizer)
        self._add_child("l2norm", cfg.l2norm)
        self._add_child("output_proj", cfg.output_proj)

    def forward(self, inputs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Image encoder for BeiTImageTokenizer.

        Args:
            inputs: Tensor of shape [batch_size, height, width, channels].

        Returns:
            A tuple of (targets, aux):
            - targets: the prediction targets with shape [batch_size, seq_len];
            - aux: a nested Tensor with keys:
                *quantized_vectors: A tensor of shape [batch_size, seq_len, 1, codebook_dim];
                *quantized_codebook_ids: A tensor of shape [batch_size, seq_len, 1, codebook_size].
        """
        x = self.backbone(inputs)
        # encoded_features shape [batch_size, seq_len, output_dim]
        encoded_features = x["encoded_features"]
        # encoded_outputs shape [batch_size, seq_len, codebook_dim]
        encoded_outputs = self.output_proj(encoded_features)
        encoded_outputs = self.l2norm(encoded_outputs)
        paddings = jnp.zeros(encoded_outputs.shape[:2])
        quantized_output = self.quantizer(inputs=encoded_outputs, paddings=paddings)
        # quantized_output.quantized_vectors shape [batch_size, seq_len, 1, codebook_dim]
        # quantized_output.ids in shape [batch_size, seq_len, 1]
        onehots = jax.nn.one_hot(
            quantized_output.ids,
            num_classes=self.config.quantizer.codebook_size,
            axis=-1,
            dtype=jnp.int32,
        )
        return jnp.squeeze(quantized_output.ids, axis=-1), {
            "quantized_vectors": jnp.squeeze(quantized_output.quantized_vectors, axis=-2),
            "quantized_codebook_onehots": jnp.squeeze(onehots, axis=-2),
        }


def set_beit_image_tokenizer_encoder_config(
    *,
    num_layers: int = 12,
    model_dim: int = 768,
    codebook_size: int = 8192,
    codebook_dim: int = 32,
    num_heads: int = 12,
    feed_forward_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4),
    image_size: tuple[int, int] = (224, 224),
    patch_size: tuple[int, int] = (16, 16),
    dropout_rate: float = 0,
    pooler_config: BasePoolingLayer.Config = AveragePooling.default_config(),
    feed_forward_act: str = "exact_gelu",
    layer_norm_eps: float = 1e-6,
    num_cls_tokens: int = 0,
    beta: float = 0,
    remat: bool = False,
) -> BEiTImageVQKD.Config:
    """Configure the BEiT-3 Image Tokenizer.

    The default setting is adapted from:
    https://github.com/microsoft/unilm/blob/716e4726e6154b4feb95db0b4dffd925bb5f5218/beit2/modeling_vqkd.py#L243

    Args:
        num_layers: An integer indicating the number of transformer blocks.
        model_dim: An integer for the output dimension of the transformer block.
        codebook_dim: An integer for the dimension of the codebook.
        codebook_size: An integer for the size of the codebook.
        num_heads: An integer for the number of the attention heads.
        feed_forward_dim: The dimension of the feedforward layer in transformer.
            It can be set as an integer or as a scaled_hidden_dim function.
        image_size: The size of the cropped image.
        patch_size: The size of the image patch.
        dropout_rate: The dropout rate of the image encoder.
        pooler_config: An instantiable BasePoolingLayer configuration used for embedding pooling.
        feed_forward_act: The nonlinear function used in the feedforward layer in transformer block.
        layer_norm_eps: The eps used in the layer norm.
        num_cls_tokens: An integer representing the number of CLS tokens.
        beta: The commitment loss weight of KmeansVectorQuantizer.
        remat: A boolean for enabling the gradient checkpointing.

    Returns:
        A instantiable BEiT-3 image tokenizer.
    """
    image_encoder_cfg = BEiTImageVQKD.default_config()
    image_encoder_transformer_cfg = image_encoder_cfg.backbone
    image_encoder_transformer_cfg.output_dim = model_dim
    # Set up the image tokenization module.
    image_encoder_transformer_cfg.num_cls_tokens = num_cls_tokens
    image_encoder_transformer_cfg.visual_embed.convert_to_sequence.set(patch_size=patch_size)
    image_encoder_transformer_cfg.pooler = pooler_config
    image_encoder_transformer_cfg.visual_embed.convert_to_sequence.conv.bias = False
    image_encoder_transformer_cfg.encoder_1d.pos_emb.shape = [
        (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        + image_encoder_transformer_cfg.num_cls_tokens
    ]
    # Set up the Transformer.
    image_encoder_transformer_cfg.encoder_1d.transformer = set_transformer_config(
        num_layers=num_layers,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        feed_forward_act=feed_forward_act,
        remat=remat,
    )
    image_encoder_cfg.output_proj = BEiTImageTokenizerMapping.default_config().set(
        input_dim=model_dim, output_dim=codebook_dim
    )

    set_norm_recursively(
        image_encoder_transformer_cfg, LayerNorm.default_config().set(eps=layer_norm_eps)
    )
    set_dropout_rate_recursively(image_encoder_transformer_cfg, dropout_rate)

    image_encoder_cfg.quantizer = KmeansVectorQuantizer.default_config().set(
        codebook_dim=codebook_dim,
        codebook_size=codebook_size,
        num_codebooks=1,
        beta=beta,
    )
    return image_encoder_cfg

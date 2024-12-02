# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# pytorch/vision:
# Copyright (c) Soumith Chintala 2016. All rights reserved.
# Licensed under BSD 3-Clause License.
#
# tensorflow/tpu:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""An AXLearn implementation for ResNet and ResNet-RS.

ResNet reference:
https://arxiv.org/abs/1512.03385
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

ResNet-RS reference:
https://arxiv.org/pdf/2103.07579v1.pdf
https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py
"""
# Many similarities with vision_transformer.
# pylint: disable=duplicate-code

import math
from collections.abc import Sequence
from typing import Optional

import jax.nn
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.convolution import Conv2D
from axlearn.common.layers import (
    BatchNorm,
    SqueezeExcitation,
    StochasticDepth,
    get_activation_fn,
    get_stochastic_depth_linear_rate,
)
from axlearn.common.module import Module

Tensor = jnp.ndarray


def batch_norm():
    return BatchNorm.default_config().set(decay=0.9, eps=1e-5)


class Downsample(BaseLayer):
    """A projection layer that adjusts spatial and channel dimensions."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures Downsample."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        stride: int = 1  # The convolution stride.
        norm: InstantiableConfig = batch_norm()  # The normalization layer config.
        # The downsample operation. Can be one of ['conv', 'maxpool'].
        downsample_op: str = "conv"

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, "model")
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.stride != 1 or cfg.input_dim != cfg.output_dim:
            if cfg.downsample_op not in ["conv", "maxpool"]:
                raise ValueError(
                    f"Downsample op needs to be 'conv' or 'maxpool', but got {cfg.downsample_op}."
                )
            self._add_child(
                "conv",
                Conv2D.default_config().set(
                    strides=(cfg.stride, cfg.stride) if cfg.downsample_op == "conv" else (1, 1),
                    input_dim=cfg.input_dim,
                    output_dim=cfg.output_dim,
                    bias=False,
                    param_partition_spec=cfg.param_partition_spec,
                ),
            )
            self._add_child("norm", cfg.norm.clone(input_dim=cfg.output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        # No-op if stride is 1 and input_dim matches output_dim.
        if cfg.stride == 1 and cfg.input_dim == cfg.output_dim:
            return inputs

        if cfg.downsample_op == "maxpool" and cfg.stride > 1:
            x = jax.lax.reduce_window(
                inputs,
                init_value=-jnp.inf,
                computation=jax.lax.max,
                window_dimensions=(1, cfg.stride, cfg.stride, 1),
                window_strides=(1, cfg.stride, cfg.stride, 1),
                padding="SAME",
            )
        else:
            x = inputs
        x = self.conv(x)
        return self.norm(x)


class StemV0(BaseLayer):
    """The vanilla ResNet stem."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures StemV0."""

        hidden_dim: Required[int] = REQUIRED  # Feature dim of the conv layer.
        input_dim: int = 3
        # The convolution layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(7, 7),
            bias=False,
            padding=((3, 3), (3, 3)),
            strides=(2, 2),
            param_partition_spec=(None, None, None, "model"),
        )
        norm: InstantiableConfig = batch_norm()  # The normalization layer config.
        activation: str = "nn.relu"  # The activation function.

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim
        self._add_child(
            "conv1",
            cfg.conv.clone(input_dim=cfg.input_dim, output_dim=hidden_dim),
        )
        self._add_child("norm1", cfg.norm.clone(input_dim=hidden_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = get_activation_fn(cfg.activation)(x)
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 3, 3, 1),
            window_strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )
        return x


class StemV1(BaseLayer):
    """The ResNet-D stem. Reference: https://arxiv.org/pdf/1812.01187.pdf."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures StemV1."""

        hidden_dim: Required[int] = REQUIRED  # Feature dim of the conv layer.
        input_dim: int = 3
        # The convolution layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(3, 3),
            bias=False,
            padding=((1, 1), (1, 1)),
            param_partition_spec=(None, None, None, "model"),
        )
        norm: InstantiableConfig = batch_norm()  # The normalization layer config.
        activation: str = "nn.swish"  # The activation function.

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim
        # Use 3 consecutive 3x3 conv layers to replace the first 7x7 conv in the vanilla stem.
        self._add_child(
            "conv1_a",
            cfg.conv.clone(strides=(2, 2), input_dim=cfg.input_dim, output_dim=hidden_dim // 2),
        )
        self._add_child("norm1_a", cfg.norm.clone(input_dim=hidden_dim // 2))
        self._add_child(
            "conv1_b",
            cfg.conv.clone(input_dim=hidden_dim // 2, output_dim=hidden_dim // 2),
        )
        self._add_child("norm1_b", cfg.norm.clone(input_dim=hidden_dim // 2))
        self._add_child(
            "conv1_c",
            cfg.conv.clone(input_dim=hidden_dim // 2, output_dim=hidden_dim),
        )
        self._add_child("norm1_c", cfg.norm.clone(input_dim=hidden_dim))
        # Use a stride-2 conv layer to replace maxpool for downsampling.
        self._add_child(
            "conv1_d",
            cfg.conv.clone(strides=(2, 2), input_dim=hidden_dim, output_dim=hidden_dim),
        )
        self._add_child("norm1_d", cfg.norm.clone(input_dim=hidden_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        x = self.conv1_a(inputs)
        x = self.norm1_a(x)
        x = get_activation_fn(cfg.activation)(x)
        x = self.conv1_b(x)
        x = self.norm1_b(x)
        x = get_activation_fn(cfg.activation)(x)
        x = self.conv1_c(x)
        x = self.norm1_c(x)
        x = get_activation_fn(cfg.activation)(x)
        x = self.conv1_d(x)
        x = self.norm1_d(x)
        x = get_activation_fn(cfg.activation)(x)
        return x


class BasicBlock(BaseLayer):
    """A basic ResNet block."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures BasicBlock."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        stride: int = 1  # The convolution stride.
        # The convolution layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(3, 3),
            bias=False,
            padding=((1, 1), (1, 1)),
            param_partition_spec=(None, None, None, "model"),
        )
        norm: InstantiableConfig = batch_norm()  # The normalization layer config.
        activation: str = "nn.relu"  # The activation function.
        # The layer used to adjust spatial or channel dimension of the skip connection features.
        downsample: Downsample.Config = Downsample.default_config()
        # The squeeze-and-excitation layer config.
        squeeze_excitation: InstantiableConfig = SqueezeExcitation.default_config()
        # The stochastic depth layer config.
        stochastic_depth: InstantiableConfig = StochasticDepth.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "conv1",
            cfg.conv.clone(
                strides=(cfg.stride, cfg.stride),
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )
        self._add_child("norm1", cfg.norm.clone(input_dim=cfg.output_dim))
        self._add_child(
            "conv2",
            cfg.conv.clone(strides=(1, 1), input_dim=cfg.output_dim, output_dim=cfg.output_dim),
        )
        self._add_child("norm2", cfg.norm.clone(input_dim=cfg.output_dim))
        self._add_child(
            "squeeze_excitation", cfg.squeeze_excitation.clone(input_dim=cfg.output_dim)
        )
        self._add_child("stochastic_depth", cfg.stochastic_depth)
        self._add_child(
            "downsample",
            cfg.downsample.clone(
                stride=cfg.stride,
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = get_activation_fn(cfg.activation)(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = self.squeeze_excitation(x)
        x = self.stochastic_depth(x)

        x += self.downsample(inputs)
        x = get_activation_fn(cfg.activation)(x)
        return x


class Bottleneck(BaseLayer):
    """A bottleneck resnet block."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures Bottleneck."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        stride: int = 1  # The convolution stride.
        # The convolution layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(3, 3),
            bias=False,
            padding=((1, 1), (1, 1)),
            param_partition_spec=(None, None, None, "model"),
        )
        norm: InstantiableConfig = batch_norm()  # The normalization layer config.
        activation: str = "nn.relu"  # The activation function.
        # The layer used to adjust spatial or channel dimension of the skip connection features.
        downsample: Downsample.Config = Downsample.default_config()
        # The squeeze-and-excitation layer config.
        squeeze_excitation: InstantiableConfig = SqueezeExcitation.default_config()
        # The stochastic depth layer config.
        stochastic_depth: InstantiableConfig = StochasticDepth.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.output_dim // 4  # Channel dimension for outputs of conv1 and conv2
        self._add_child(
            "conv1",
            Conv2D.default_config().set(
                window=(1, 1),
                padding=((0, 0), (0, 0)),
                strides=(1, 1),
                bias=False,
                param_partition_spec=(None, None, None, "model"),
                input_dim=cfg.input_dim,
                output_dim=hidden_dim,
            ),
        )
        self._add_child("norm1", cfg.norm.clone(input_dim=hidden_dim))
        self._add_child(
            "conv2",
            cfg.conv.clone(
                strides=(cfg.stride, cfg.stride),
                input_dim=hidden_dim,
                output_dim=hidden_dim,
            ),
        )
        self._add_child("norm2", cfg.norm.clone(input_dim=hidden_dim))
        self._add_child(
            "conv3",
            Conv2D.default_config().set(
                window=(1, 1),
                padding=((0, 0), (0, 0)),
                strides=(1, 1),
                bias=False,
                param_partition_spec=(None, None, None, "model"),
                input_dim=hidden_dim,
                output_dim=cfg.output_dim,
            ),
        )
        self._add_child("norm3", cfg.norm.clone(input_dim=cfg.output_dim))
        self._add_child(
            "squeeze_excitation", cfg.squeeze_excitation.clone(input_dim=cfg.output_dim)
        )
        self._add_child("stochastic_depth", cfg.stochastic_depth)
        self._add_child(
            "downsample",
            cfg.downsample.clone(
                stride=cfg.stride,
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config

        x = self.conv1(inputs)
        x = self.norm1(x)
        x = get_activation_fn(cfg.activation)(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = get_activation_fn(cfg.activation)(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.squeeze_excitation(x)
        x = self.stochastic_depth(x)

        x += self.downsample(inputs)
        x = get_activation_fn(cfg.activation)(x)
        return x


class ResNetStage(BaseLayer):
    """A stage of ResNet, consisting of multiple blocks."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ResNetStage."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        stride: int = 1  # The convolution stride.
        block: InstantiableConfig = BasicBlock.default_config()  # The block config.
        num_blocks: Required[int] = REQUIRED  # Number of blocks in this stage.

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        input_dim = cfg.input_dim
        for block_i in range(cfg.num_blocks):
            if block_i == 0:
                stride = cfg.stride
            else:
                stride = 1
            self._add_child(
                f"block{block_i}",
                cfg.block.clone(
                    input_dim=input_dim,
                    output_dim=cfg.output_dim,
                    stride=stride,
                ),
            )
            input_dim = cfg.output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        x = inputs
        for block_i in range(cfg.num_blocks):
            x = getattr(self, f"block{block_i}")(x)
        self.add_module_output("forward", x)
        return x


class ResNet(BaseLayer):
    """The generic ResNet model."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ResNet."""

        hidden_dim: int = 64  # The feature dim between the stem layer and the first block.
        # The stem config. "StemV0" and "StemV1" represent vanilla stem and ResNet-D stem.
        # Reference: https://arxiv.org/abs/2103.07579, https://arxiv.org/abs/1812.01187.
        stem: InstantiableConfig = StemV0.default_config()
        stage: InstantiableConfig = ResNetStage.default_config()  # The stage config.
        # A list of integers, representing number of blocks per stage.
        num_blocks_per_stage: Required[Sequence[int]] = REQUIRED
        # The peak stochastic depth drop rate. A linear schedule where the rate increases
        # from first stage to last stage (Reference: https://arxiv.org/pdf/1603.09382.pdf).
        peak_stochastic_depth_rate: Optional[float] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            init_by_param_name={
                param_init.PARAM_REGEXP_WEIGHT: param_init.WeightInitializer.default_config().set(
                    # Equivalent to kaiming_normal_(mode='fan_out', nonlinearity='relu').
                    fan="fan_out",
                    distribution="normal",
                    scale=math.sqrt(2),
                )
            }
        )
        cfg.dtype = jnp.float32
        return cfg

    @classmethod
    def resnet18_config(cls):
        cfg = cls.default_config()
        cfg.num_blocks_per_stage = [2, 2, 2, 2]
        return cfg

    @classmethod
    def resnet34_config(cls):
        cfg = cls.default_config()
        cfg.num_blocks_per_stage = [3, 4, 6, 3]
        return cfg

    @classmethod
    def resnet50_config(cls):
        cfg = cls.default_config()
        cfg.stage.block = Bottleneck.default_config()
        cfg.num_blocks_per_stage = [3, 4, 6, 3]
        return cfg

    @classmethod
    def resnet101_config(cls):
        cfg = cls.default_config()
        cfg.stage.block = Bottleneck.default_config()
        cfg.num_blocks_per_stage = [3, 4, 23, 3]
        return cfg

    @classmethod
    def resnet152_config(cls):
        cfg = cls.default_config()
        cfg.stage.block = Bottleneck.default_config()
        cfg.num_blocks_per_stage = [3, 8, 36, 3]
        return cfg

    @classmethod
    def resnet200_config(cls):
        cfg = cls.default_config()
        cfg.stage.block = Bottleneck.default_config()
        cfg.num_blocks_per_stage = [3, 24, 36, 3]
        return cfg

    @classmethod
    def resnet270_config(cls):
        cfg = cls.default_config()
        cfg.stage.block = Bottleneck.default_config()
        cfg.num_blocks_per_stage = [4, 29, 53, 4]
        return cfg

    @classmethod
    def resnet350_config(cls):
        cfg = cls.default_config()
        cfg.stage.block = Bottleneck.default_config()
        cfg.num_blocks_per_stage = [4, 36, 72, 4]
        return cfg

    @classmethod
    def resnet420_config(cls):
        cfg = cls.default_config()
        cfg.stage.block = Bottleneck.default_config()
        cfg.num_blocks_per_stage = [4, 44, 87, 4]
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim

        # The stem conv and downsampling layers.
        self._add_child("stem", cfg.stem.clone(hidden_dim=hidden_dim))

        # The ResNet block layers.
        self._endpoints_dims = {}
        use_bottleneck = cfg.stage.block.klass == Bottleneck
        for stage_i, num_blocks in enumerate(cfg.num_blocks_per_stage):
            if use_bottleneck:
                output_dim = hidden_dim * 4 if stage_i == 0 else hidden_dim * 2
            else:
                output_dim = hidden_dim if stage_i == 0 else hidden_dim * 2
            stage_cfg = cfg.stage.clone(
                input_dim=hidden_dim,
                output_dim=output_dim,
                stride=1 if stage_i == 0 else 2,
                num_blocks=num_blocks,
            )
            # Set per stage stochastic depth rate.
            if cfg.peak_stochastic_depth_rate:
                stage_cfg.block.stochastic_depth.rate = get_stochastic_depth_linear_rate(
                    cfg.peak_stochastic_depth_rate, stage_order=stage_i + 2, num_stages=5
                )
            self._add_child(f"stage{stage_i}", stage_cfg)
            hidden_dim = output_dim
            self._endpoints_dims[str(stage_i + 2)] = hidden_dim  # Endpoints index starts from 2.

        # The embedding layer after global average pooling.
        self._endpoints_dims["embedding"] = hidden_dim

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """Computes prediction on an input batch.

        Args:
            image: A float Tensor with value of shape (batch, height, width, 3).

        Returns:
            endpoints: A dict that contains intermediate features.
        """
        cfg = self.config
        x = self.stem(image)

        endpoints = {}
        for stage_i, _ in enumerate(cfg.num_blocks_per_stage):
            x = getattr(self, f"stage{stage_i}")(x)
            endpoints[str(stage_i + 2)] = x  # Endpoints index starts from 2.

        # [batch, hidden].
        x = jnp.mean(x, axis=(1, 2))
        endpoints["embedding"] = x
        return endpoints

    @property
    def endpoints_dims(self) -> dict[str, int]:
        """A dict of {level: hidden_dim} specifies hidden dimension of intermediate features.

        2**level is the ratio between the input resolution and the current feature resolution,
        e.g. level 3 denotes the current feature resolution is 1/8 of the input image resolution.

        ResNet provides endpoints from level 2 to 5.
        """
        return self._endpoints_dims

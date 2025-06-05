# Copyright © 2023 Apple Inc.
#
# tensorflow/tpu:
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# facebookresearch/detectron2:
# Copyright 2019-2020, detectron2 contributors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# zylo117/Yet-Another-EfficientDet-Pytorch:
# Licensed under GNU Lesser General Public License Version 3.

# pylint: disable=too-many-lines
"""An implementation of feature pyramid networks.

FPN Reference: https://arxiv.org/abs/1612.03144
SimpleFPN Reference: https://arxiv.org/abs/2203.16527
BiFPN Reference: https://arxiv.org/abs/1911.09070
"""
import copy
import enum
from collections.abc import Iterable
from typing import Optional, Union

import jax
import jax.nn
import numpy as np
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.convolution import Conv2D, Conv2DTranspose
from axlearn.common.layers import BatchNorm, LayerNorm, MaxPool2D, get_activation_fn, normalize_sum
from axlearn.common.module import Module
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    ConstantInitializer,
    DefaultInitializer,
    PerGroupInitializer,
    WeightInitializer,
)
from axlearn.common.utils import Tensor


def batch_norm():
    return BatchNorm.default_config().set(decay=0.9, eps=1e-5)


def layer_norm():
    return LayerNorm.default_config().set(eps=1e-5)


class FusionMethod(str, enum.Enum):
    SUM = "sum"
    ATTENTION = "attention"
    FASTATTENTION = "fast_attention"


class LayerType(str, enum.Enum):
    DEFAULT = "default"
    FIRSTLAYER = "first_layer"


class RescaleImageMethod(str, enum.Enum):
    IDENTITY = "identity"
    DOUBLE = "double"
    HALVE = "halve"


class FPN(BaseLayer):
    """Feature pyramid network.

    Reference:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/architecture/fpn.py.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures FPN."""

        hidden_dim: Required[int] = REQUIRED  # Output dim of the conv layer.
        # Input specifications dictionary of (level, dim) pairs for input features in the pyramid.
        # Must contain consecutive levels from `min_level`.
        input_dims: Required[dict[str, int]] = REQUIRED
        min_level: int = 3  # Min level for the feature pyramid.
        max_level: int = 7  # Max level for the feature pyramid.
        # The convolution layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(1, 1),
            padding="SAME",
            param_partition_spec=(None, None, None, "model"),
        )
        # The normalization layer config. If set to None, no batch norm will be used.
        norm: Optional[InstantiableConfig] = batch_norm()
        activation: str = "nn.relu"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        backbone_max_level = self._get_backbone_max_level()

        for level in range(cfg.min_level, backbone_max_level + 1):
            self._add_child(
                f"conv{level}",
                cfg.conv.clone(input_dim=cfg.input_dims[str(level)], output_dim=cfg.hidden_dim),
            )
            self._add_child(
                f"post_hoc_conv{level}",
                cfg.conv.clone(input_dim=cfg.hidden_dim, output_dim=cfg.hidden_dim, window=(3, 3)),
            )

        # Add conv layers to generate features for level [backbone_max_level + 1, cfg.max_level].
        for level in range(backbone_max_level + 1, cfg.max_level + 1):
            self._add_child(
                f"conv{level}",
                cfg.conv.clone(
                    input_dim=cfg.hidden_dim,
                    output_dim=cfg.hidden_dim,
                    window=(3, 3),
                    strides=(2, 2),
                    padding=((1, 1), (1, 1)),
                ),
            )

        if cfg.norm is not None:
            for level in range(cfg.min_level, cfg.max_level + 1):
                self._add_child(f"norm{level}", cfg.norm.clone(input_dim=cfg.hidden_dim))

    def _get_backbone_max_level(self):
        levels = [int(l) for l in self.config.input_dims.keys() if l.isdigit()]
        return min(max(levels), self.config.max_level)

    def forward(self, inputs: dict[str, Tensor]) -> dict[int, Tensor]:
        """FPN forward pass.

        Args:
            inputs: A dictionary of {level: features} for the input feature pyramid.
            Must contain all levels in cfg.input_dims.

        Returns:
            A dictionary of {level: features} that contains the output feature pyramid for
            for [`min_level`, `max_level`].
        """
        cfg = self.config
        backbone_max_level = self._get_backbone_max_level()

        # Build lateral connections.
        x_lateral = {}
        for level in range(cfg.min_level, backbone_max_level + 1):
            x_lateral[level] = getattr(self, f"conv{level}")(inputs[str(level)])

        # Adds top-down path.
        x = {backbone_max_level: x_lateral[backbone_max_level]}
        for level in range(backbone_max_level - 1, cfg.min_level - 1, -1):
            b, h, w, c = x[level + 1].shape
            x[level] = (
                jax.image.resize(x[level + 1], [b, h * 2, w * 2, c], method="nearest")
                + x_lateral[level]
            )

        # Adds post-hoc 3x3 convolution kernel.
        for level in range(cfg.min_level, backbone_max_level + 1):
            x[level] = getattr(self, f"post_hoc_conv{level}")(x[level])

        # Adds coarser FPN levels introduced in RetinaNet.
        for level in range(backbone_max_level + 1, cfg.max_level + 1):
            x_in = x[level - 1]
            if level > backbone_max_level + 1:
                x_in = get_activation_fn(cfg.activation)(x_in)
            x[level] = getattr(self, f"conv{level}")(x_in)

        # Append norm layers.
        if cfg.norm is not None:
            for level in range(cfg.min_level, cfg.max_level + 1):
                x[level] = getattr(self, f"norm{level}")(x[level])

        return x


class SimpleFPN(BaseLayer):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.

    Reference:
    https://github.com/facebookresearch/detectron2/blob/3c7bb714795edc7a96c9a1a6dd83663ecd293e36/detectron2/modeling/backbone/vit.py#L363-L503

    It creates pyramid features built on top of the input feature map.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures SimpleFPN."""

        hidden_dim: Required[int] = REQUIRED  # Output dim of the conv layer.
        # Input specifications dictionary of (level, dim) pairs for input features in the pyramid.
        # Must contain consecutive levels from `min_level`.
        input_dims: Required[dict[int, int]] = REQUIRED
        min_level: int = 2  # Min level for the feature pyramid.
        max_level: int = 6  # Max level for the feature pyramid.
        # The convolution layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(1, 1),
            padding="VALID",
            strides=(1, 1),
            bias=False,
            param_partition_spec=(None, None, None, "model"),
            # Equivalent to kaiming_normal_(mode='fan_out', nonlinearity='relu').
            param_init=DefaultInitializer.default_config().set(
                init_by_param_name={
                    PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                        fan="fan_out",
                        scale=np.sqrt(2.0),
                        distribution="normal",
                    ),
                }
            ),
        )
        # The transposed convolution layer config.
        deconv: InstantiableConfig = Conv2DTranspose.default_config().set(
            window=(2, 2),
            padding="VALID",
            strides=(2, 2),
            transpose_kernel=True,
            param_partition_spec=(None, None, None, "model"),
            # Equivalent to kaiming_normal_(mode='fan_out', nonlinearity='relu').
            param_init=DefaultInitializer.default_config().set(
                init_by_param_name={
                    PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                        fan="fan_out",
                        scale=np.sqrt(2.0),
                        distribution="normal",
                    ),
                }
            ),
        )
        # MaxPool2d layer config.
        maxpool2d: InstantiableConfig = MaxPool2D.default_config().set(
            window=(2, 2),
            strides=(2, 2),
            param_partition_spec=(None, None, None, "model"),
        )
        # The normalization layer config.
        norm: InstantiableConfig = layer_norm()
        activation: str = "nn.gelu"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self.backbone_level = int(min(l for l in cfg.input_dims.keys() if l.isdigit()))
        assert self.backbone_level >= cfg.min_level

        backbone_dim = cfg.input_dims[str(self.backbone_level)]
        backbone_max_level = self._get_backbone_max_level(self.backbone_level)

        for level in range(cfg.min_level, backbone_max_level + 1):
            out_dim, delta = backbone_dim, self.backbone_level - level - 1
            for delta in range(self.backbone_level - level):
                self._add_child(
                    f"deconv{level}_{delta + 1}",
                    cfg.deconv.clone(
                        input_dim=backbone_dim // (2 ** (delta)),
                        output_dim=backbone_dim // (2 ** (delta + 1)),
                        window=(2, 2),
                        strides=(2, 2),
                    ),
                )
                if delta != self.backbone_level - level - 1:
                    self._add_child(
                        f"norm{level}_{delta + 1}",
                        cfg.norm.clone(
                            input_dim=backbone_dim // (2 ** (delta + 1)),
                        ),
                    )
            out_dim = backbone_dim // (2 ** (delta + 1))

            self._add_child(
                f"conv{level}",
                cfg.conv.clone(
                    input_dim=out_dim,
                    output_dim=cfg.hidden_dim,
                    window=(1, 1),
                    bias=False,
                ),
            )
            self._add_child(
                f"norm{level}",
                cfg.norm.clone(
                    input_dim=cfg.hidden_dim,
                ),
            )
            self._add_child(
                f"post_hoc_conv{level}",
                cfg.conv.clone(
                    input_dim=cfg.hidden_dim,
                    output_dim=cfg.hidden_dim,
                    window=(3, 3),
                    padding=((1, 1), (1, 1)),
                    bias=False,
                ),
            )
            self._add_child(
                f"post_hoc_norm{level}",
                cfg.norm.clone(
                    input_dim=cfg.hidden_dim,
                ),
            )

        # Add conv layers to generate features for level [backbone_max_level + 1, cfg.max_level].
        for level in range(backbone_max_level + 1, cfg.max_level + 1):
            # Generate C5 feature.
            if level <= 5:
                self._add_child(
                    f"maxpool2d{level}",
                    cfg.maxpool2d.clone(
                        window=(2, 2),
                        strides=(2, 2),
                    ),
                )
                self._add_child(
                    f"conv{level}",
                    cfg.conv.clone(
                        input_dim=out_dim,
                        output_dim=cfg.hidden_dim,
                        window=(1, 1),
                        bias=False,
                    ),
                )
                self._add_child(
                    f"norm{level}",
                    cfg.norm.clone(
                        input_dim=cfg.hidden_dim,
                    ),
                )
                self._add_child(
                    f"post_hoc_conv{level}",
                    cfg.conv.clone(
                        input_dim=cfg.hidden_dim,
                        output_dim=cfg.hidden_dim,
                        window=(3, 3),
                        padding=((1, 1), (1, 1)),
                        bias=False,
                    ),
                )
                self._add_child(
                    f"post_hoc_norm{level}",
                    cfg.norm.clone(
                        input_dim=cfg.hidden_dim,
                    ),
                )
            else:
                self._add_child(
                    f"conv{level}",
                    cfg.conv.clone(
                        input_dim=cfg.hidden_dim,
                        output_dim=cfg.hidden_dim,
                        window=(3, 3),
                        strides=(2, 2),
                        padding=((1, 1), (1, 1)),
                    ),
                )
        self.vlog(
            3,
            "The constructed layers from level=%d to level=%d in SimpleFPN are: %s",
            cfg.min_level,
            cfg.max_level,
            self.children.keys(),
        )

    def _get_backbone_max_level(self, backbone_level):
        return min(backbone_level, self.config.max_level)

    def forward(self, inputs: dict[str, Tensor]) -> dict[int, Tensor]:
        """SimpleFPN forward pass.

        Args:
            inputs: A dictionary of {level: features} for the input feature pyramid.
            Must contain all levels in cfg.input_dims.

        Returns:
            A dictionary of {level: features} that contains the output feature pyramid for
            for [`min_level`, `max_level`].
        """
        cfg = self.config
        backbone_max_level = self._get_backbone_max_level(self.backbone_level)

        x_output = {}
        for level in range(cfg.min_level, backbone_max_level + 1):
            x = inputs[str(self.backbone_level)]
            for delta in range(self.backbone_level - level):
                x = getattr(self, f"deconv{level}_{delta + 1}")(x)
                if delta != self.backbone_level - level - 1:
                    x = getattr(self, f"norm{level}_{delta + 1}")(x)
                    x = get_activation_fn(cfg.activation)(x)
            x = getattr(self, f"conv{level}")(x)
            x = getattr(self, f"norm{level}")(x)
            x = getattr(self, f"post_hoc_conv{level}")(x)
            x = getattr(self, f"post_hoc_norm{level}")(x)
            x_output[level] = x

        for level in range(backbone_max_level + 1, cfg.max_level + 1):
            x = inputs[str(self.backbone_level)]
            if level <= 5:
                x = getattr(self, f"maxpool2d{level}")(x)
                x = getattr(self, f"conv{level}")(x)
                x = getattr(self, f"norm{level}")(x)
                x = getattr(self, f"post_hoc_conv{level}")(x)
                x = getattr(self, f"post_hoc_norm{level}")(x)
                x_output[level] = x
            else:
                x = x_output[level - 1]
                # Different from the original paper, we use conv to get L6 (and L7)
                # from C5 features, original paper uses the maxpool to get L6 feature.
                if level > 6:
                    x = get_activation_fn(cfg.activation)(x)
                x = getattr(self, f"conv{level}")(x)
                x_output[level] = x

        return x_output


class WeightedFeatureFusion(BaseLayer):
    """Weighted feature fusion layer.

    Reference: Section 3.3 in https://arxiv.org/pdf/1911.09070.pdf
    Reference implementation for SUM and FASTATTENTION:
        https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py
    """

    _normalization_functions = {
        FusionMethod.SUM: lambda x: x,
        FusionMethod.ATTENTION: lambda x: jax.nn.softmax(x, axis=0),
        FusionMethod.FASTATTENTION: lambda x: normalize_sum(
            get_activation_fn("nn.relu")(x),
            axis=0,
            eps=1e-4,  # replicate eps of reference implementation
        ),
    }

    @config_class
    class Config(BaseLayer.Config):
        """Configures WeightedFeatureFusion."""

        # Number of tensors to fuse.
        num_input_tensors: Required[int] = REQUIRED
        # Method for fusing input tensors.
        method: FusionMethod = FusionMethod.SUM
        # Activation function to apply after the fusion.
        activation: str = "linear"
        # Default initializer is set to 1.0
        param_init: InstantiableConfig = ConstantInitializer.default_config().set(value=1.0)

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        if cfg.method in self._normalization_functions:
            self.normalize = self._normalization_functions[cfg.method]
        else:
            raise ValueError(f"No normalization function for FusionMethod {cfg.method=} found.")

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.num_input_tensors, 1),
            )
        )
        return params

    def forward(self, x: list[Tensor]) -> Tensor:
        """Takes a list of Tensors and returns a weighted sum

        Args:
            x: float tensors of shape [batch_size, height, width, channels]

        Returns:
            A float tensor of same shape as x
        """
        cfg = self.config
        assert len(x) == cfg.num_input_tensors

        params_with_normalized_weight = {
            k: (self.normalize(v) if k == "weight" else v) for k, v in self.state.items()
        }
        output = jnp.zeros_like(x[0])
        for i in range(cfg.num_input_tensors):
            output += x[i] * params_with_normalized_weight["weight"][i]
        output = get_activation_fn(cfg.activation)(output)
        return output


class DepthwiseSeparableConvolution(BaseLayer):
    """Depthwise separable convolution as used in the BiFPN of EfficientDet.

    Reference:
    https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/layers/convolutional/separable_conv2d.py#L34

    TODO(pdufter) merge with axlearn/common/mobilenetv3_blocks.py:DepthwiseSeparable
    and move to axlearn/common/layers.py
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures DepthwiseSeparableConvolution."""

        # Input dimension
        input_dim: Required[int] = REQUIRED
        # Output dimension
        output_dim: Required[int] = REQUIRED
        # Kernel size for depthwise convolution
        depthwise_kernel_size: int = 3
        # Padding for depthwise convolution
        padding: Union[str, tuple[tuple[int, int], tuple[int, int]]] = "SAME"
        # Whether to add a bias to the pointwise convolution
        pointwise_conv_bias: bool = False

        # Convolution operation to use for both depthwise and pointwise conv
        conv: InstantiableConfig = Conv2D.default_config().set(
            # kaiming normal initialization
            param_init=DefaultInitializer.default_config().set(
                init_by_param_name={
                    PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                        fan="fan_out",
                        scale=np.sqrt(2.0),
                        distribution="normal",
                    ),
                }
            )
        )

        # Norm operation to use after the convolution
        norm: Optional[InstantiableConfig] = None

        # Activation function
        activation: str = "linear"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "depthwise_conv",
            cfg.conv.clone(
                input_dim=cfg.input_dim,
                output_dim=cfg.input_dim,
                window=(cfg.depthwise_kernel_size, cfg.depthwise_kernel_size),
                padding=cfg.padding,
                num_input_dim_groups=cfg.input_dim,
                bias=False,
                param_init=PerGroupInitializer.default_config().set(
                    initializer=cfg.conv.param_init,
                    num_groups=cfg.input_dim,
                ),
            ),
        )
        self._add_child(
            "pointwise_conv",
            cfg.conv.clone(
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
                window=(1, 1),
                bias=cfg.pointwise_conv_bias,
            ),
        )
        if cfg.norm is not None:
            self._add_child("norm", cfg.norm.clone(input_dim=cfg.output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config

        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)

        if cfg.norm is not None:
            x = self.norm(x)
        x = get_activation_fn(cfg.activation)(x)
        return x


class ResampleFeatures(BaseLayer):
    """Resamples height and width, and projects number of channels."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ResampleFeatures."""

        # Number of input channels
        input_dim: Required[int] = REQUIRED
        # Number of output channels
        output_dim: Required[int] = REQUIRED
        # If true project channels even when input_dim == output_dim
        force_projection: bool = False
        # Whether to add a norm layer after projection
        norm: Optional[InstantiableConfig] = None
        # Whether to add a bias after projection even when normalization layer is added
        add_redundant_bias: bool = False
        # Projection using a 1x1 convolution
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(1, 1),
            # kaiming normal initialization
            param_init=DefaultInitializer.default_config().set(
                init_by_param_name={
                    PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                        fan="fan_out",
                        scale=np.sqrt(2.0),
                        distribution="normal",
                    ),
                }
            ),
        )
        # Rescale operation to adjust height/width
        rescale_op: RescaleImageMethod = RescaleImageMethod.IDENTITY

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        if cfg.input_dim != cfg.output_dim or cfg.force_projection:
            self._add_child(
                "conv",
                cfg.conv.clone(
                    input_dim=cfg.input_dim,
                    output_dim=cfg.output_dim,
                    bias=(cfg.add_redundant_bias or cfg.norm is None),
                ),
            )
            if cfg.norm is not None:
                self._add_child(
                    "norm",
                    cfg.norm.clone(input_dim=cfg.output_dim),
                )

    def forward(self, inputs: Tensor) -> Tensor:
        """Resamples height and width, and projects number of channels.

        Args:
            inputs: float tensor of shape [batch_size, height, width, input_dim]

        Returns:
            float tensor of shape [batch_size, resampled_height, resampled_width, output_dim]
        """
        cfg = self.config

        x = inputs
        if cfg.input_dim != cfg.output_dim or cfg.force_projection:
            x = self.conv(x)
            if cfg.norm is not None:
                x = self.norm(x)

        if cfg.rescale_op == RescaleImageMethod.DOUBLE:
            # double height and width using nearest neighbor sampling
            scale_factor = (2, 2)
            batch_size, height, width, channels = x.shape
            target_shape = (
                batch_size,
                round(height * scale_factor[0]),
                round(width * scale_factor[1]),
                channels,
            )
            x = jax.image.resize(x, target_shape, "nearest")
        elif cfg.rescale_op == RescaleImageMethod.HALVE:
            # halve height and width using max pooling
            window_size = (3, 3)
            stride = (2, 2)
            padding = ((1, 1), (1, 1))
            x = jax.lax.reduce_window(
                x,
                init_value=-jnp.inf,
                computation=jax.lax.max,
                window_dimensions=(1, window_size[0], window_size[1], 1),
                window_strides=(1, stride[0], stride[1], 1),
                padding=((0, 0),) + padding + ((0, 0),),
            )
        return x


# pylint: disable=too-many-branches
class BiFPNLayer(BaseLayer):
    """Bidirectional feature pyramid network layer.

    BiFPNLayer implements a single top down and bottom up path, optionally
    preceded by downchanneling and/or adding more feature maps.

    Reference Implementation:
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BiFPNLayer."""

        # Input specifications dictionary of (level, dim) pairs for input features in the pyramid.
        # Must contain consecutive levels from `min_level`.
        input_dims: Required[dict[int, int]] = REQUIRED
        # Number of channels in hidden layers
        hidden_dim: Required[int] = REQUIRED
        # Depending on layer type add additional operations, e.g., downchanneling for first layer
        layer_type: LayerType = LayerType.DEFAULT

        # Min level for the feature pyramid
        min_level: int = 3
        # Max level for the feature pyramid
        max_level: int = 7

        # If true add redundant biases in subcomponents
        add_redundant_bias: bool = False

        # Convolutional layers
        conv: InstantiableConfig = DepthwiseSeparableConvolution.default_config()
        # Fusion layers for fusing several node inputs together
        fusion: InstantiableConfig = WeightedFeatureFusion.default_config()

        # Resampling wrapper to adjust height, width and number of channels
        resample: InstantiableConfig = ResampleFeatures.default_config()
        # Normalization layer
        norm: InstantiableConfig = BatchNorm.default_config()

    def _get_input_max_level(self) -> int:
        cfg = self.config
        return max(cfg.input_dims.keys())

    def _get_levels_up(self) -> Iterable:
        cfg = self.config
        return range(cfg.min_level, cfg.max_level + 1)

    def _get_levels_down(self) -> Iterable:
        return reversed(self._get_levels_up())

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        if cfg.min_level not in cfg.input_dims:
            raise ValueError(
                "min_level of BiFPN must be contained in input_dims. "
                f"Got {cfg.min_level=}, {cfg.input_dims=}."
            )

        if cfg.layer_type == LayerType.FIRSTLAYER:
            current_input_dims = copy.deepcopy(cfg.input_dims)
            input_max_level = self._get_input_max_level()

            # Maybe add additional feature maps
            for level in self._get_levels_up():
                if level > input_max_level:
                    # Add additional feature maps
                    self._add_child(
                        f"add_level{level}",
                        cfg.resample.clone(
                            input_dim=current_input_dims[level - 1],
                            output_dim=cfg.hidden_dim,
                            rescale_op=RescaleImageMethod.HALVE,
                            norm=cfg.norm.clone(),
                            add_redundant_bias=cfg.add_redundant_bias,
                        ),
                    )
                    current_input_dims[level] = cfg.hidden_dim

            # Adjust channels
            for level in self._get_levels_up():
                # When an input dimension is coincidentally equal to cfg.hidden_dim
                # force a projection (1x1 convolution) in the first layer to be
                # consistent with the reference implementation
                force_projection = level <= input_max_level

                self._add_child(
                    f"input_downchannel_skip{level}",
                    cfg.resample.clone(
                        input_dim=current_input_dims[level],
                        output_dim=cfg.hidden_dim,
                        norm=cfg.norm.clone(),
                        add_redundant_bias=cfg.add_redundant_bias,
                        force_projection=force_projection,
                    ),
                )
                if level > cfg.min_level:
                    self._add_child(
                        f"input_downchannel{level}",
                        cfg.resample.clone(
                            input_dim=current_input_dims[level],
                            output_dim=cfg.hidden_dim,
                            norm=cfg.norm.clone(),
                            add_redundant_bias=cfg.add_redundant_bias,
                            force_projection=force_projection,
                        ),
                    )
        else:
            for level, dim in cfg.input_dims.items():
                if dim != cfg.hidden_dim:
                    raise ValueError(
                        f"all input dims must be equal to {cfg.hidden_dim=} "
                        f"when using {cfg.layer_type=}. Got {cfg.input_dims=}."
                    )

        # top down path
        for level in self._get_levels_down():
            if level in (cfg.min_level, cfg.max_level):
                continue
            self._add_child(
                f"intermediate_resample{level}",
                cfg.resample.clone(
                    input_dim=cfg.hidden_dim,
                    output_dim=cfg.hidden_dim,
                    rescale_op=RescaleImageMethod.DOUBLE,
                ),
            )
            self._add_child(
                f"intermediate_conv{level}",
                cfg.conv.clone(
                    input_dim=cfg.hidden_dim,
                    output_dim=cfg.hidden_dim,
                    pointwise_conv_bias=cfg.add_redundant_bias,
                ),
            )
            self._add_child(
                f"intermediate_fusion{level}",
                cfg.fusion.clone(num_input_tensors=2),
            )

        # bottom up path
        for level in self._get_levels_up():
            if level == cfg.min_level:
                self._add_child(
                    f"output_resample{level}",
                    cfg.resample.clone(
                        input_dim=cfg.hidden_dim,
                        output_dim=cfg.hidden_dim,
                        rescale_op=RescaleImageMethod.DOUBLE,
                    ),
                )
            else:
                self._add_child(
                    f"output_resample{level}",
                    cfg.resample.clone(
                        input_dim=cfg.hidden_dim,
                        output_dim=cfg.hidden_dim,
                        rescale_op=RescaleImageMethod.HALVE,
                    ),
                )

            self._add_child(
                f"output_conv{level}",
                cfg.conv.clone(
                    input_dim=cfg.hidden_dim,
                    output_dim=cfg.hidden_dim,
                    pointwise_conv_bias=cfg.add_redundant_bias,
                ),
            )
            if level in (cfg.min_level, cfg.max_level):
                # For the top and bottom level there are only two inputs to fuse
                self._add_child(
                    f"output_fusion{level}",
                    cfg.fusion.clone(num_input_tensors=2),
                )
            else:
                self._add_child(
                    f"output_fusion{level}",
                    cfg.fusion.clone(num_input_tensors=3),
                )

    def forward(self, inputs: dict[int, Tensor]) -> dict[int, Tensor]:
        """BiFPN layer forward pass.

        Args:
            inputs: {level: features}, features are float tensors of shape
                [batch_size, height_i, width_i, input_dims[i]]} for level=i.
                Contains same levels as input_dims.

        Returns:
            {level: features}, features are float tensor of shape
                [batch_size, height_i, width_i, hidden_dim]} for level=i.
                Level goes from range(cfg.min_level, cfg.max_level + 1).
        """
        cfg = self.config

        if cfg.layer_type == LayerType.FIRSTLAYER:
            input_max_level = self._get_input_max_level()
            # Add additional input feature maps
            extended_inputs = copy.deepcopy(inputs)
            for level in self._get_levels_up():
                if level > input_max_level:
                    add_level = getattr(self, f"add_level{level}")
                    extended_inputs[level] = add_level(extended_inputs[level - 1])

            # Adjust number of channels
            down_channelled_inputs = {}
            down_channelled_inputs_skip = {}
            for level in self._get_levels_up():
                downchannel = getattr(self, f"input_downchannel_skip{level}")
                down_channelled_inputs_skip[level] = downchannel(extended_inputs[level])
                if level > cfg.min_level:
                    downchannel = getattr(self, f"input_downchannel{level}")
                    down_channelled_inputs[level] = downchannel(extended_inputs[level])
        else:
            down_channelled_inputs = copy.deepcopy(inputs)
            down_channelled_inputs_skip = copy.deepcopy(inputs)

        # Top down path
        intermediate = {}
        for level in self._get_levels_down():
            if level in (cfg.min_level, cfg.max_level):
                continue
            conv = getattr(self, f"intermediate_conv{level}")
            fusion = getattr(self, f"intermediate_fusion{level}")
            resample = getattr(self, f"intermediate_resample{level}")

            if level + 1 == cfg.max_level:
                # Fall back to original input
                resampled_input = resample(down_channelled_inputs[level + 1])
            else:
                resampled_input = resample(intermediate[level + 1])
            intermediate[level] = conv(fusion([down_channelled_inputs[level], resampled_input]))

        # Bottom up path
        output = {}
        for level in self._get_levels_up():
            conv = getattr(self, f"output_conv{level}")
            fusion = getattr(self, f"output_fusion{level}")
            resample = getattr(self, f"output_resample{level}")

            if level == cfg.min_level:
                resampled_input = resample(intermediate[level + 1])
                fusion_input = [down_channelled_inputs_skip[level], resampled_input]
            elif level == cfg.max_level:
                resampled_input = resample(output[level - 1])
                fusion_input = [down_channelled_inputs_skip[level], resampled_input]
            else:
                resampled_input = resample(output[level - 1])
                fusion_input = [
                    down_channelled_inputs_skip[level],
                    intermediate[level],
                    resampled_input,
                ]

            output[level] = conv(fusion(fusion_input))
        return output


# pylint: enable=too-many-branches


class BiFPN(BaseLayer):
    """Bidirectional feature pyramid network (BiFPN).

    BiFPN stacks together multiple BiFPNLayers.

    Reference Implementation:
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BiFPN."""

        # Input specifications dictionary of (level, dim) pairs for input features in the pyramid.
        # Must contain consecutive levels from `min_level`.
        input_dims: Required[dict[str, int]] = REQUIRED
        # Output dim for all output levels of the pyramid
        hidden_dim: Required[int] = REQUIRED
        # Number of BiFPN layers to create
        num_bifpn_layers: int = 1
        # Actual BiFPN layers
        bifpn_layer: InstantiableConfig = BiFPNLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        current_input_dims = {int(l): dim for (l, dim) in cfg.input_dims.items() if l.isdigit()}
        for current_bifpn_layer in range(cfg.num_bifpn_layers):
            if current_bifpn_layer == 0:
                layer_type = LayerType.FIRSTLAYER
            else:
                layer_type = LayerType.DEFAULT
            self._add_child(
                f"bifpn_layer{current_bifpn_layer}",
                cfg.bifpn_layer.clone(
                    input_dims=current_input_dims,
                    hidden_dim=cfg.hidden_dim,
                    layer_type=layer_type,
                ),
            )
            current_input_dims = {level: cfg.hidden_dim for level in current_input_dims}

    def forward(self, inputs: dict[str, Tensor]) -> dict[int, Tensor]:
        cfg = self.config

        x = {int(l): f for (l, f) in inputs.items() if l.isdigit()}
        for current_bifpn_layer in range(cfg.num_bifpn_layers):
            bifpn_layer = getattr(self, f"bifpn_layer{current_bifpn_layer}")
            x = bifpn_layer(x)

        return x


def bifpn_layer_config(
    *,
    hidden_dim: int = 64,
    min_level: int = 3,
    max_level: int = 7,
    layer_type: LayerType = LayerType.DEFAULT,
    layer_cfg: Optional[BiFPNLayer.Config] = None,
    conv_cfg: Optional[InstantiableConfig] = None,
    fusion_cfg: Optional[InstantiableConfig] = None,
    resample_cfg: Optional[InstantiableConfig] = None,
    add_redundant_bias: bool = False,
    bifpn_activation: str = "nn.swish",
) -> BiFPNLayer.Config:
    """Builds configs for BiFPN Layer as used in EfficientDet (https://arxiv.org/abs/1911.09070).

    Defaults are from the EfficientDet-D0 model with FusionMethod.FASTATTENTION.
    TODO(pdufter) simplify function args (e.g., remove hidden_dim, bifpn_activation)

    Args:
        hidden_dim: Hidden dim of the BiFPN
        min_level: Min feature level of BiFPN
        max_level: Max feature level of BiFPN
        layer_type: Layer type of the bifpn layer
        layer_cfg: Config for BiFPN layer
        conv_cfg: Config for convolution operation used in BiFPN
        fusion_cfg: Config for fusion operation used in BiFPN
        resample_cfg: Config for resample operation used in BiFPN
        add_redundant_bias: If true, add redundant biases before normalization.
        bifpn_activation: Activation function used in BiFPN

    Returns:
        The BiFPNLayer config.
    """

    bifpn_layer_cfg = (
        layer_cfg.clone()
        if layer_cfg
        else BiFPNLayer.default_config().set(
            hidden_dim=hidden_dim,
            layer_type=layer_type,
            min_level=min_level,
            max_level=max_level,
            add_redundant_bias=add_redundant_bias,
        )
    )

    conv_cfg = (
        conv_cfg.clone()
        if conv_cfg
        else DepthwiseSeparableConvolution.default_config().set(
            norm=BatchNorm.default_config(),
        )
    )

    fusion_cfg = (
        fusion_cfg.clone()
        if fusion_cfg
        else WeightedFeatureFusion.default_config().set(
            activation=bifpn_activation,
            method=FusionMethod.FASTATTENTION,
        )
    )

    resample_cfg = resample_cfg.clone() if resample_cfg else ResampleFeatures.default_config()

    bifpn_layer_cfg = bifpn_layer_cfg.set(
        conv=conv_cfg,
        fusion=fusion_cfg,
        resample=resample_cfg,
    )
    return bifpn_layer_cfg


def bifpn_config(
    *,
    input_dims: Optional[dict[str, int]] = None,
    hidden_dim: int = 64,
    min_level: int = 3,
    max_level: int = 7,
    num_bifpn_layers: int = 3,
    bifpn_cfg: Optional[BiFPN.Config] = None,
    bifpn_layer_cfg: Optional[BiFPNLayer.Config] = None,
    fusion_cfg: Optional[InstantiableConfig] = None,
    add_redundant_bias: bool = False,
    bifpn_activation: str = "nn.swish",
) -> BiFPN.Config:
    """Builds configs for BiFPN as used in EfficientDet (https://arxiv.org/abs/1911.09070).

    Defaults are from the EfficientDet-D0 model with FusionMethod.FASTATTENTION.
    TODO(pdufter) num_bifpn_layers, remove input_dims, hidden_dim, add_redundant_bias,
        and bifpn_activation from args

    Args:
        input_dims: Input dimension for each feature level
        hidden_dim: Hidden dim of the BiFPN
        min_level: Min feature level of BiFPN
        max_level: Max feature level of BiFPN
        num_bifpn_layers: Number of layers
        bifpn_cfg: Config for BiFPN
        bifpn_layer_cfg: Config for BiFPN layer
        fusion_cfg: Config for fusion operation used in BiFPN
        add_redundant_bias: If true, add redundant biases before normalization.
        bifpn_activation: Activation function used in BiFPN

    Returns:
        The BiFPN config.
    """
    bifpn_cfg = (
        bifpn_cfg.clone()
        if bifpn_cfg
        else BiFPN.default_config().set(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_bifpn_layers=num_bifpn_layers,
        )
    )

    bifpn_layer_cfg = (
        bifpn_layer_cfg.clone()
        if bifpn_layer_cfg
        else bifpn_layer_config(
            hidden_dim=hidden_dim,
            min_level=min_level,
            max_level=max_level,
            fusion_cfg=fusion_cfg,
            add_redundant_bias=add_redundant_bias,
            bifpn_activation=bifpn_activation,
        )
    )

    bifpn_cfg.set(
        bifpn_layer=bifpn_layer_cfg,
    )

    return bifpn_cfg

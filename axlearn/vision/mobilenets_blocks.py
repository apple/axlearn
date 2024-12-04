# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# rwightman/efficientnet-jax:
# Copyright 2020 Ross Wightman.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# huggingface/pytorch-image-models:
# Copyright 2019 Ross Wightman.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Blocks for mobile networks (incl. MobileNetV3,
EfficientNet, EfficientNetV2 and EfficientNet-Lite).

References:
https://arxiv.org/abs/1905.02244
https://arxiv.org/abs/1905.11946
https://arxiv.org/abs/2104.00298

Implementation adapted from:
https://github.com/rwightman/efficientnet-jax/
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py
"""
import enum
import math
from typing import Optional

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.convolution import Conv2D
from axlearn.common.layers import BatchNorm, get_activation_fn
from axlearn.common.module import Module
from axlearn.common.param_init import PerGroupInitializer
from axlearn.common.utils import Tensor


class MobileBlockType(str, enum.Enum):
    """Mobile block type.

    CONV_BN_ACT: Convolution followed by BatchNorm followed by activation.
    DEPTHWISE_SEPARABLE: Depthwise separable convolution.
    INVERTED_BOTTLENECK: Inverted bottleneck block.
    FUSED_INVERTED_BOTTLENECK: Fused inverted bottleneck block.
    """

    CONV_BN_ACT = "conv_bn_act"
    DEPTHWISE_SEPARABLE = "depthwise_separable"
    INVERTED_BOTTLENECK = "inverted_bottleneck"
    FUSED_INVERTED_BOTTLENECK = "fused_inverted_bottleneck"


class SamePaddingType(str, enum.Enum):
    """Padding type for computing same padding.

    DEFAULT: Same padding is computed as outlined in:
        https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    SYMMETRIC: The padding is forced to be symmetric for top/bottom and left/right.
    """

    DEFAULT = "default"
    SYMMETRIC = "symmetric"


class SeReduceReference(str, enum.Enum):
    """Reference dimension for Squeeze Excitation layer.

    This enum determines which dimensionality is used for computing num_reduced_filters
    in the Squeeze Excitation layer when used within mobile blocks.

    INPUT_DIM: Use the input dim as a reference for computing num_reduced_filters.
    NUM_FEATURES: Use the number of features as a reference for computing num_reduced_filters.
    """

    INPUT_DIM = "input_dim"
    NUM_FEATURES = "num_features"


def _make_divisible(
    value: int,
    *,
    divisor: int = 8,
    min_value: Optional[int] = None,
    round_limit: float = 0.9,
) -> int:
    """Adapt value to make it divisble by divisor.

    Reference:
    https://github.com/rwightman/efficientnet-jax/blob/20e1b87d04f3588dbc4470701b551ae680ea6177/jeffnet/common/block_utils.py#L1-L7
    """
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-round_limit).
    if new_value < round_limit * value:
        new_value += divisor
    return new_value


def round_features(
    num_features: int,
    *,
    multiplier: float = 1.0,
    divisor: int = 8,
    min_num_features: Optional[int] = None,
    round_limit: float = 0.9,
) -> int:
    """Round number of filters based on depth multiplier.

    Reference:
    https://github.com/rwightman/efficientnet-jax/blob/20e1b87d04f3588dbc4470701b551ae680ea6177/jeffnet/common/block_utils.py#L10-L15
    """
    if not multiplier:
        return num_features
    num_features *= multiplier
    return _make_divisible(
        int(num_features), divisor=divisor, min_value=min_num_features, round_limit=round_limit
    )


def _compute_same_padding(
    *,
    kernel_size: int = 1,
    stride: int = 1,
    padding_type: SamePaddingType = SamePaddingType.DEFAULT,
) -> list[tuple[int, int]]:
    """Compute explicit SAME padding based on kernel size and stride.

    Reference:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    total_padding = kernel_size - stride
    if padding_type == SamePaddingType.DEFAULT:
        padding = (total_padding // 2, total_padding - total_padding // 2)
    elif padding_type == SamePaddingType.SYMMETRIC:
        padding = (math.ceil(total_padding / 2), math.ceil(total_padding / 2))
    return [padding, padding]


class ConvBnAct(BaseLayer):
    """Convolution followed by BatchNorm followed by activation, with optional skip connection."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ConvBnAct."""

        # Input dim.
        input_dim: Required[int] = REQUIRED
        # Output dim; number of features.
        output_dim: Required[int] = REQUIRED

        # Convolution layer.
        conv_layer: InstantiableConfig = Conv2D.default_config()
        # Normalization layer.
        norm_layer: InstantiableConfig = BatchNorm.default_config()
        # Activation function.
        activation: str = "nn.relu"

        # Kernel size of the convolution layer.
        kernel_size: int = 3
        # Stride of the convolution layer.
        stride: int = 1
        # If True use a skip connection if the input and output dimensions match
        # and we use a stride of 1.
        use_skip_connection_if_possible: bool = False
        # Padding type for computing SAME padding.
        padding_type: SamePaddingType = SamePaddingType.DEFAULT

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        cfg.conv_layer.set(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            window=(cfg.kernel_size, cfg.kernel_size),
            strides=(cfg.stride, cfg.stride),
            padding=_compute_same_padding(
                kernel_size=cfg.kernel_size, stride=cfg.stride, padding_type=cfg.padding_type
            ),
            bias=False,
        )
        self._add_child("conv", cfg.conv_layer)
        self._add_child("bn", cfg.norm_layer.set(input_dim=cfg.output_dim))

        # Use a skip connection if the input and output dimensions match and we use a stride of 1.
        self._use_skip_connection = (
            cfg.use_skip_connection_if_possible
            and cfg.stride == 1
            and cfg.input_dim == cfg.output_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        shortcut = x
        x = self.conv(x)
        x = self.bn(x)
        x = get_activation_fn(cfg.activation)(x)
        if self._use_skip_connection:
            x = x + shortcut
        return x


class MobileBlock(BaseLayer):
    """Implements different blocks for mobile architectures.

    1)  Depthwise separable convolution, which separates spatial filtering (via a depthwise
        convolution) from feature generation (via a pointwise convolution), with an optional
        squeeze-excitation block in between to adaptively weight each channel.

        References:
        https://arxiv.org/abs/1704.04861
        https://arxiv.org/abs/1905.02244

    2)  Mobile inverted bottleneck conv block (MBConv, sometimes also referred to as
        inverted residual block), which has the same structure as a depthwise separable
        block, but with an optional point-wise expansion in the beginning.

        References:
        https://arxiv.org/abs/1801.04381
        https://arxiv.org/abs/1905.02244

    3)  Fused mobile inverted bottleneck conv block (Fused-MBConv, sometimes also referred to as
        edge residual block), with expansion convolution followed by a point-wise linear projection,
        with an optional squeeze-excitation block in between to adaptively weight each channel.

        References:
        https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
        https://arxiv.org/abs/2104.00298
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures MobileBlock."""

        # Type of the mobile block.
        mobile_block_type: Required[MobileBlockType] = REQUIRED
        # Input dim.
        input_dim: Required[int] = REQUIRED
        # Output dim; number of output features.
        output_dim: Required[int] = REQUIRED

        # Convolution layer.
        conv_layer: InstantiableConfig = Conv2D.default_config()
        # Normalization layer.
        norm_layer: InstantiableConfig = BatchNorm.default_config()
        # Squeeze-excitation block will not be applied if se_layer is None.
        se_layer: Optional[InstantiableConfig] = None
        # Drop-path regularization will not be applied if drop_path is None.
        drop_path: Optional[InstantiableConfig] = None
        # Activation function.
        activation: str = "nn.relu"

        # Kernel size of the convolution layer.
        kernel_size: int = 3
        # Stride of the convolution layer.
        stride: int = 1
        # Padding type for computing SAME padding.
        padding_type: SamePaddingType = SamePaddingType.DEFAULT
        # Expansion ratio for expansion convolution.
        exp_ratio: float = 1.0
        # Reference dimension for computing num_reduced_filters in the squeeze
        # excitation layer; if None, default to input_dim for DEPTHWISE_SEPARABLE,
        # and to num_features for INVERTED_BOTTLENECK and FUSED_INVERTED_BOTTLENECK.
        se_reduce_ref: Optional[SeReduceReference] = None
        # Divisor for computing num_reduced_filters in the squeeze excitation layer.
        se_reduction_divisor: int = 8

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.mobile_block_type in {
            MobileBlockType.INVERTED_BOTTLENECK,
            MobileBlockType.FUSED_INVERTED_BOTTLENECK,
        }:
            num_features = _make_divisible(cfg.input_dim * cfg.exp_ratio)
        else:
            num_features = cfg.input_dim

        if cfg.exp_ratio > 1.0 and (
            cfg.mobile_block_type
            in {MobileBlockType.INVERTED_BOTTLENECK, MobileBlockType.FUSED_INVERTED_BOTTLENECK}
        ):
            # Use expansion convolution (point-wise for INVERTED_BOTTLENECK).
            conv_exp_args = {
                "input_dim": cfg.input_dim,
                "output_dim": num_features,
                "bias": False,
            }
            if cfg.mobile_block_type == MobileBlockType.FUSED_INVERTED_BOTTLENECK:
                conv_exp_args.update(
                    {
                        "window": (cfg.kernel_size, cfg.kernel_size),
                        "strides": (cfg.stride, cfg.stride),
                        "padding": _compute_same_padding(
                            kernel_size=cfg.kernel_size,
                            stride=cfg.stride,
                            padding_type=cfg.padding_type,
                        ),
                    }
                )

            self._add_child("conv_exp", cfg.conv_layer.clone(**conv_exp_args))
            self._add_child("bn_exp", cfg.norm_layer.clone(input_dim=num_features))

        if cfg.mobile_block_type in {
            MobileBlockType.DEPTHWISE_SEPARABLE,
            MobileBlockType.INVERTED_BOTTLENECK,
        }:
            # Use depth-wise convolution.
            self._add_child(
                "conv_dw",
                cfg.conv_layer.clone(
                    input_dim=num_features,
                    output_dim=num_features,
                    window=(cfg.kernel_size, cfg.kernel_size),
                    strides=(cfg.stride, cfg.stride),
                    padding=_compute_same_padding(
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding_type=cfg.padding_type,
                    ),
                    num_input_dim_groups=num_features,
                    bias=False,
                    param_init=PerGroupInitializer.default_config().set(
                        initializer=cfg.conv_layer.param_init,
                        num_groups=num_features,
                    ),
                ),
            )
            self._add_child("bn_dw", cfg.norm_layer.clone(input_dim=num_features))

        self._add_child(
            "conv_pw",
            cfg.conv_layer.clone(
                input_dim=num_features,
                output_dim=cfg.output_dim,
                bias=False,
            ),
        )
        self._add_child("bn_pw", cfg.norm_layer.clone(input_dim=cfg.output_dim))
        if cfg.se_layer is not None:
            if cfg.se_reduce_ref is None:
                if cfg.mobile_block_type == MobileBlockType.DEPTHWISE_SEPARABLE:
                    cfg.se_reduce_ref = SeReduceReference.INPUT_DIM
                else:
                    cfg.se_reduce_ref = SeReduceReference.NUM_FEATURES
            if cfg.se_reduce_ref == SeReduceReference.INPUT_DIM:
                se_reduce_ref_dim = cfg.input_dim
            elif cfg.se_reduce_ref == SeReduceReference.NUM_FEATURES:
                se_reduce_ref_dim = num_features
            num_reduced_filters = _make_divisible(
                se_reduce_ref_dim * cfg.se_layer.se_ratio, divisor=cfg.se_reduction_divisor
            )
            self._add_child(
                "se",
                cfg.se_layer.set(
                    input_dim=num_features,
                    num_reduced_filters=num_reduced_filters,
                ),
            )
        if cfg.drop_path is not None:
            self._add_child("drop_path", cfg.drop_path)

        # Use a skip connection if the input and output dimensions match and we use a stride of 1.
        self._use_skip_connection = cfg.input_dim == cfg.output_dim and cfg.stride == 1

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        shortcut = x

        if cfg.exp_ratio > 1.0 and (
            cfg.mobile_block_type
            in {MobileBlockType.INVERTED_BOTTLENECK, MobileBlockType.FUSED_INVERTED_BOTTLENECK}
        ):
            # Expansion convolution (point-wise for INVERTED_BOTTLENECK).
            x = self.conv_exp(x)
            x = self.bn_exp(x)
            x = get_activation_fn(cfg.activation)(x)

        if cfg.mobile_block_type in {
            MobileBlockType.DEPTHWISE_SEPARABLE,
            MobileBlockType.INVERTED_BOTTLENECK,
        }:
            # Depth-wise convolution.
            x = self.conv_dw(x)
            x = self.bn_dw(x)
            x = get_activation_fn(cfg.activation)(x)

        if cfg.se_layer is not None:
            # Squeeze-excitation.
            x = self.se(x)

        # Point-wise convolution.
        x = self.conv_pw(x)
        x = self.bn_pw(x)

        if self._use_skip_connection:
            # Skip connection.
            if cfg.drop_path is not None:
                x = self.drop_path(x)
            x = x + shortcut
        return x

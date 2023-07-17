# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# rwightman/efficientnet-jax:
# Copyright 2020 Ross Wightman.
# Licensed under the Apache License, Version 2.0 (the "License").

"""MobileNetV3 blocks.

Reference: https://arxiv.org/abs/1905.02244.
Implementation adapted from: https://github.com/rwightman/efficientnet-jax/.
"""
import enum
import math
from typing import List, Optional, Tuple

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import BatchNorm, Conv2D, get_activation_fn
from axlearn.common.module import Module
from axlearn.common.param_init import PerGroupInitializer
from axlearn.common.utils import Tensor


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
    in the Squeeze Excitation layer when used within InvertedResidual layers.

    INPUT_DIM: Use the input dim as a reference for computing num_reduced_filters.
    NUM_FEATURES: Use the number of features as a reference for computing num_reduced_filters.
    """

    INPUT_DIM = "input_dim"
    NUM_FEATURES = "num_features"


def _make_divisible(value: int, *, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """Adapt value to make it divisble by divisor.

    Reference:
    https://github.com/rwightman/efficientnet-jax/blob/20e1b87d04f3588dbc4470701b551ae680ea6177/jeffnet/common/block_utils.py#L1-L7
    """
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def round_features(
    num_features: int,
    *,
    multiplier: float = 1.0,
    divisor: int = 8,
    min_num_features: Optional[int] = None,
) -> int:
    """Round number of filters based on depth multiplier.

    Reference:
    https://github.com/rwightman/efficientnet-jax/blob/20e1b87d04f3588dbc4470701b551ae680ea6177/jeffnet/common/block_utils.py#L10-L15
    """
    if not multiplier:
        return num_features
    num_features *= multiplier
    return _make_divisible(int(num_features), divisor=divisor, min_value=min_num_features)


def _compute_same_padding(
    *,
    kernel_size: int = 1,
    stride: int = 1,
    padding_type: SamePaddingType = SamePaddingType.DEFAULT,
) -> List[Tuple[int, int]]:
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
    """Convolution followed by BatchNorm followed by activation function."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ConvBnAct."""

        conv_layer: InstantiableConfig = Conv2D.default_config()
        norm_layer: InstantiableConfig = BatchNorm.default_config()
        activation: str = "nn.relu"

        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        kernel_size: int = 3
        stride: int = 1
        # Padding type for computing SAME padding
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

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x = self.conv(x)
        x = self.bn(x)
        x = get_activation_fn(cfg.activation)(x)
        return x


class DepthwiseSeparable(BaseLayer):
    """Depthwise separable convolution, which separates spatial filtering (via a depthwise
    convolution) from feature generation (via a pointwise convolution), with an optional
    squeeze-excitation block in-between to adaptively weight each channel.

    References:
    https://arxiv.org/pdf/1704.04861.pdf
    https://arxiv.org/pdf/1905.02244.pdf
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures DepthwiseSeparable."""

        conv_layer: InstantiableConfig = Conv2D.default_config()
        norm_layer: InstantiableConfig = BatchNorm.default_config()
        # Squeeze-excitation block will not be applied if se_layer is None
        se_layer: InstantiableConfig = None
        # Drop-path regularization will not be applied if drop_path is None
        drop_path: InstantiableConfig = None
        activation: str = "nn.relu"

        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        kernel_size: int = 3
        stride: int = 1
        # Padding type for computing SAME padding
        padding_type: SamePaddingType = SamePaddingType.DEFAULT
        # Reference dimension for computing the number of reduced features
        # in the Squeeze Excitation Layer; if 0 default to input_dim
        se_reduce_ref_dim: Optional[int] = 0
        # Divisor for computing num_reduced_filters in Squeeze Excitation Layer
        se_reduction_divisor: int = 8

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "conv_dw",
            cfg.conv_layer.clone(
                input_dim=cfg.input_dim,
                output_dim=cfg.input_dim,
                window=(cfg.kernel_size, cfg.kernel_size),
                strides=(cfg.stride, cfg.stride),
                padding=_compute_same_padding(
                    kernel_size=cfg.kernel_size, stride=cfg.stride, padding_type=cfg.padding_type
                ),
                num_input_dim_groups=cfg.input_dim,
                bias=False,
                param_init=PerGroupInitializer.default_config().set(
                    initializer=cfg.conv_layer.param_init,
                    num_groups=cfg.input_dim,
                ),
            ),
        )
        self._add_child(
            "conv_pw",
            cfg.conv_layer.clone(
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
                bias=False,
            ),
        )
        self._add_child("bn_dw", cfg.norm_layer.clone(input_dim=cfg.input_dim))
        self._add_child("bn_pw", cfg.norm_layer.clone(input_dim=cfg.output_dim))
        if cfg.se_layer is not None:
            if cfg.se_reduce_ref_dim == 0:
                # TODO(pdufter) change this default behaviour
                # and always populate cfg.se_reduce_ref_dim instead
                se_reduce_ref_dim = cfg.input_dim
            else:
                se_reduce_ref_dim = cfg.se_reduce_ref_dim
            num_reduced_filters = _make_divisible(
                se_reduce_ref_dim * cfg.se_layer.se_ratio, divisor=cfg.se_reduction_divisor
            )
            self._add_child(
                "se",
                cfg.se_layer.set(
                    input_dim=cfg.input_dim,
                    num_reduced_filters=num_reduced_filters,
                ),
            )
        if cfg.drop_path is not None:
            self._add_child("drop_path", cfg.drop_path)

    def forward(self, x: Tensor, *, shortcut: Optional[Tensor] = None) -> Tensor:
        cfg = self.config
        if shortcut is None:
            shortcut = x

        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = get_activation_fn(cfg.activation)(x)

        if cfg.se_layer is not None:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn_pw(x)

        if cfg.stride == 1 and x.shape == shortcut.shape:
            if cfg.drop_path is not None:
                x = self.drop_path(x)
            x = x + shortcut
        return x


class InvertedResidual(BaseLayer):
    """Inverted residual block, which has the same structure as a depthwise separable block, but
    with an optional point-wise expansion in the beginning.

    References:
    https://arxiv.org/pdf/1801.04381.pdf
    https://arxiv.org/pdf/1905.02244.pdf
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures InvertedResidual."""

        depthwise_separable: InstantiableConfig = DepthwiseSeparable.default_config()
        conv_layer: InstantiableConfig = Conv2D.default_config()
        norm_layer: InstantiableConfig = BatchNorm.default_config()
        activation: str = "nn.relu"

        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        exp_ratio: float = 1.0  # Expansion ratio for point-wise conv expansion.
        # Reference dimension for squeeze excitation layer
        se_reduce_ref: SeReduceReference = SeReduceReference.NUM_FEATURES

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        features = _make_divisible(cfg.input_dim * cfg.exp_ratio)

        if cfg.exp_ratio > 1.0:
            self._add_child(
                "conv_exp",
                cfg.conv_layer.clone(
                    input_dim=cfg.input_dim,
                    output_dim=features,
                    bias=False,
                ),
            )
            self._add_child("bn_exp", cfg.norm_layer.clone(input_dim=features))

        if cfg.se_reduce_ref == SeReduceReference.INPUT_DIM:
            se_reduce_ref_dim = cfg.input_dim
        elif cfg.se_reduce_ref == SeReduceReference.NUM_FEATURES:
            se_reduce_ref_dim = features
        self._add_child(
            "depthwise_separable",
            cfg.depthwise_separable.set(
                input_dim=features, output_dim=cfg.output_dim, se_reduce_ref_dim=se_reduce_ref_dim
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        shortcut = x

        # Point-wise expansion
        if cfg.exp_ratio > 1.0:
            x = self.conv_exp(x)
            x = self.bn_exp(x)
            x = get_activation_fn(cfg.activation)(x)

        x = self.depthwise_separable(x, shortcut=shortcut)
        return x

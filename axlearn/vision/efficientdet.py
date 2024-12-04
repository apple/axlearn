# Copyright © 2023 Apple Inc.

"""An implementation of EfficientDet prediction heads.

Reference: https://arxiv.org/abs/1911.09070
"""
import contextlib
import enum
from collections.abc import Iterable
from typing import Optional

import numpy as np

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.convolution import Conv2D
from axlearn.common.layers import BatchNorm, get_activation_fn, set_norm_recursively
from axlearn.common.module import Module, Tensor, child_context
from axlearn.common.param_init import (
    PARAM_REGEXP_BIAS,
    PARAM_REGEXP_WEIGHT,
    ConstantInitializer,
    DefaultInitializer,
    WeightInitializer,
)
from axlearn.vision.fpn import DepthwiseSeparableConvolution, bifpn_config
from axlearn.vision.mobilenets import EndpointsMode, ModelNames, named_model_configs
from axlearn.vision.retinanet import RetinaNetModel

EFFICIENTDETVARIANTS = {
    # (image_size, backbone_variant, hidden_dim, num_bifpn_layers, num_head_layers)
    "test": (32, "test", 2, 2, 2),
    "d0": (512, "b0", 64, 3, 3),
    "d1": (640, "b1", 88, 4, 3),
    "d2": (768, "b2", 112, 5, 3),
    "d3": (896, "b3", 160, 6, 4),
    "d4": (1024, "b4", 224, 7, 4),
    "d5": (1280, "b5", 288, 7, 4),
    "d6": (1280, "b6", 384, 8, 5),
    "d7": (1536, "b6", 384, 8, 5),
}


def efficientdet_conv_head_initialization() -> DefaultInitializer:
    return DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan="fan_in",
                scale=1.0,
                distribution="normal",
            ),
        }
    )


class HeadWeightSharing(str, enum.Enum):
    NONE = "none"
    SHARELEVELS = "share_levels"


class PredictionHead(BaseLayer):
    """Prediction head for EfficientDet.

    Prediction head consisting of convolutional layers followed by a head convolution.
    Consumes features and computes class scores on multiple feature levels.
    If head_conv is None, compute only class features on multiple feature levels.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures PredictionHead."""

        # Input dim, identical across input levels
        input_dim: Required[int] = REQUIRED
        # Number of layers, if 0 add only head convolution
        num_layers: Required[int] = REQUIRED

        # Min feature level
        min_level: int = 3
        # Max feature level
        max_level: int = 7
        # The convolutional layer config
        conv: InstantiableConfig = Conv2D.default_config()
        # Weight sharing pattern for convolutions
        conv_weight_sharing: HeadWeightSharing = HeadWeightSharing.NONE
        # The normalization layer config. If set to None, no norm will be used.
        norm: InstantiableConfig = BatchNorm.default_config()
        # Weight sharing pattern for normalization
        norm_weight_sharing: HeadWeightSharing = HeadWeightSharing.NONE
        # Activation to be applied after normalization
        activation: Optional[str] = "linear"

        # Output dimension of head convolution, i.e., the number of classes
        head_conv_output_dim: Optional[int] = 1
        # The head convolution config. If set to None, no head convolution will be used.
        head_conv: Optional[InstantiableConfig] = Conv2D.default_config()

    def _get_levels_up(self) -> Iterable:
        cfg = self.config
        return range(cfg.min_level, cfg.max_level + 1)

    def _add_weight_sharing_child(
        self, *, name: str, cfg: Config, sharing_pattern: HeadWeightSharing
    ):
        if sharing_pattern == HeadWeightSharing.NONE:
            # Create layers for each level
            for level in self._get_levels_up():
                self._add_child(f"{name}_l{level}", cfg.clone())
        elif sharing_pattern == HeadWeightSharing.SHARELEVELS:
            # Share layers across level
            self._add_child(name, cfg.clone())
        else:
            raise NotImplementedError(f"{sharing_pattern=} not implemented.")

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        for layer in range(cfg.num_layers):
            if cfg.conv is not None:
                self._add_weight_sharing_child(
                    name=f"conv{layer}",
                    cfg=cfg.conv.clone(
                        input_dim=cfg.input_dim,
                        output_dim=cfg.input_dim,
                    ),
                    sharing_pattern=cfg.conv_weight_sharing,
                )
            if cfg.norm is not None:
                self._add_weight_sharing_child(
                    name=f"norm{layer}",
                    cfg=cfg.norm.clone(input_dim=cfg.input_dim),
                    sharing_pattern=cfg.norm_weight_sharing,
                )

        if cfg.head_conv is not None:
            if cfg.head_conv_output_dim is None or cfg.head_conv_output_dim < 1:
                raise ValueError(f"Invalid value {cfg.head_conv_output_dim=}.")
            self._add_child(
                "head_conv",
                cfg.head_conv.clone(
                    input_dim=cfg.input_dim,
                    output_dim=cfg.head_conv_output_dim,
                ),
            )

    def forward(self, inputs: dict[int, Tensor]) -> dict[int, Tensor]:
        """Computes class scores or class features (if cfg.head_conv is None).

        Args:
            inputs: {level: features}, features are float tensors of shape
                [batch_size, height_i, width_i, input_dim]} for level=i.
                Needs to contain all levels from range(cfg.min_level, cfg.max_level + 1).

        Returns:
            {level: features}, features are float tensor of shape
                [batch_size, height_i, width_i, channels]} for level=i.
                Level goes from range(cfg.min_level, cfg.max_level + 1).
                channels equal input_dim or head_conv_output_dim (if head_conv is not None)
        """
        cfg = self.config

        @contextlib.contextmanager
        def _get_weight_shared_module(name: str, sharing_pattern: HeadWeightSharing):
            if sharing_pattern == HeadWeightSharing.NONE:
                yield getattr(self, f"{name}_l{level}")
            elif sharing_pattern == HeadWeightSharing.SHARELEVELS:
                with child_context(f"{name}_l{level}", module=getattr(self, name)):
                    yield getattr(self, name)
            else:
                raise NotImplementedError()

        outputs = {}
        for level in range(cfg.min_level, cfg.max_level + 1):
            x = inputs[level]
            for layer in range(cfg.num_layers):
                if cfg.conv is not None:
                    with _get_weight_shared_module(f"conv{layer}", cfg.conv_weight_sharing) as conv:
                        x = conv(x)
                if cfg.norm is not None:
                    with _get_weight_shared_module(f"norm{layer}", cfg.norm_weight_sharing) as norm:
                        x = norm(x)
                x = get_activation_fn(cfg.activation)(x)
            if cfg.head_conv is not None:
                with child_context(f"head_conv_l{level}", module=getattr(self, "head_conv")):
                    head_conv = getattr(self, "head_conv")
                    x = head_conv(x)
            outputs[level] = x
        return outputs


class BoxClassHead(BaseLayer):
    """EfficientDet head with box regression and class prediction.

    Consumes features on multiple feature levels, and computes class and
    box regressions scores.

    TODO(pdufter) unify with RetinaNetHead in axlearn/common/retinanet.py
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BoxClassHead."""

        # Input dim, identical across input levels
        input_dim: Required[int] = REQUIRED
        # Number of classes
        num_classes: Required[int] = REQUIRED
        # Number of anchors per feature, e.g., when we have 8 x 8 features
        # we have a total number of anchors equal to 8 x 8 x num_anchors
        num_anchors: Required[int] = REQUIRED

        # The box regression head
        box_head: InstantiableConfig = PredictionHead.default_config()
        # The class prediction head
        class_head: InstantiableConfig = PredictionHead.default_config()
        # Temporarily adding fake configs to be able to re-use RetinaNet
        # TODO(pdufter) remove this once RetinaNetModel init function is simplified
        hidden_dim: int = 0

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child(
            "box_head",
            cfg.box_head.clone(
                input_dim=cfg.input_dim,
                head_conv_output_dim=cfg.num_anchors * 4,
            ),
        )

        self._add_child(
            "class_head",
            cfg.class_head.clone(
                input_dim=cfg.input_dim,
                head_conv_output_dim=cfg.num_anchors * cfg.num_classes,
            ),
        )

    def forward(self, inputs: dict[int, Tensor]) -> dict[str, dict[int, Tensor]]:
        """Computes class scores and box regression scores.

        Args:
            inputs: {level: features}, features are float tensors of shape
                [batch_size, height_i, width_i, input_dim]} for level=i.
                Needs to contain all levels from range(cfg.min_level, cfg.max_level + 1).

        Returns:
            For {"class_outputs": {level: class_predictions},
                "box_outputs": {level: box_regressions}}
                class_predictions are float tensors of shape
                [batch_size, height_i, width_i, num_anchors * num_classes]} for level=i.
                _prbox_regressions are float tensors of shape
                [batch_size, height_i, width_i, num_anchors * 4]} for level=i.
        """
        return {"class_outputs": self.class_head(inputs), "box_outputs": self.box_head(inputs)}


def set_efficientdet_config(
    *,
    backbone_variant: str = "b0",
    backbone_version: str = "V1",
    num_head_layers: int = 3,
    min_level: int = 3,
    max_level: int = 7,
    add_redundant_bias: bool = False,
    use_ds_conv_in_head: bool = True,
) -> RetinaNetModel.Config:
    """Returns the EfficientDet config.

    Args:
        backbone_variant: The EfficientNet backbone variant, e.g., "b0", "lite0".
        backbone_version: The EfficientNet backbone version, i.e.,
            "V1" (https://arxiv.org/abs/1905.11946) or "V2" (https://arxiv.org/abs/2104.00298).
        num_head_layers: The number of layers in the box and class head.
        min_level: The minimum feature level in the BiFPN and prediction heads.
        max_level: The maximum feature level in the BiFPN and prediction heads.
        add_redundant_bias: If true, add a redundant bias before normalization layers.
        use_ds_conv_in_head: If true, use depthwise separable convolutions in the prediction heads.

    Returns:
        The EfficientDet config.
    """
    base_cfg = RetinaNetModel.default_config()

    backbone_cfg = named_model_configs(
        ModelNames.EFFICIENTNET,
        backbone_variant,
        efficientnet_version=backbone_version,
    )
    backbone_cfg = backbone_cfg.set(
        embedding_layer=None,
        endpoints_mode=EndpointsMode.LASTBLOCKS,
        endpoints_names=("3", "4", "5"),
    )

    fpn_cfg = bifpn_config(
        min_level=min_level,
        max_level=max_level,
        add_redundant_bias=add_redundant_bias,
    )

    head_cfg = efficientdet_boxclasshead_config(
        num_layers=num_head_layers,
        min_level=min_level,
        max_level=max_level,
        add_redundant_bias=add_redundant_bias,
        use_ds_conv_in_head=use_ds_conv_in_head,
    )

    base_cfg = base_cfg.set(
        backbone=backbone_cfg,
        fpn=fpn_cfg,
        head=head_cfg,
    )

    set_norm_recursively(base_cfg, BatchNorm.default_config().set(eps=1e-3, decay=0.9))

    return base_cfg


def efficientdet_boxclasshead_config(
    *,
    input_dim: int = 64,
    num_classes: int = 91,
    num_anchors: int = 9,
    num_layers: int = 3,
    min_level: int = 3,
    max_level: int = 7,
    add_redundant_bias: bool = False,
    use_ds_conv_in_head: bool = True,
) -> BoxClassHead.Config:
    """Builds configs for BoxClassHead of EfficientDet model.

    Defaults are from EfficientDet-D0 model for COCO.
    TODO(pdufter) remove input_dim, num_classes, num_anchors, num_layers from args

    Args:
        input_dim: input dim, identical across input levels
        num_classes: number of output classes
        num_anchors: number of anchors per feature
        num_layers: number of convolutional layers for both box and class head
        min_level: min level of input to consider for prediction
        max_level: max level of input to consider for prediction
        add_redundant_bias: if true, add redundant biases before normalization
        use_ds_conv_in_head: if true, use depthwise-separable conv in the head convolution
    Returns:
        The box class head config.
    """
    if use_ds_conv_in_head:
        head_conv = DepthwiseSeparableConvolution.default_config().set(
            conv=Conv2D.default_config().set(
                param_init=efficientdet_conv_head_initialization(),
            ),
            pointwise_conv_bias=True,
        )
    else:
        head_conv = Conv2D.default_config().set(
            window=(3, 3),
            padding="SAME",
            param_init=efficientdet_conv_head_initialization(),
        )

    head_cfg = PredictionHead.default_config().set(
        num_layers=num_layers,
        min_level=min_level,
        max_level=max_level,
        activation="nn.swish",
        conv=DepthwiseSeparableConvolution.default_config().set(
            conv=Conv2D.default_config().set(
                param_init=efficientdet_conv_head_initialization(),
            ),
            pointwise_conv_bias=add_redundant_bias,
        ),
        conv_weight_sharing=HeadWeightSharing.SHARELEVELS,
        head_conv=head_conv,
    )

    box_head_cfg = head_cfg.clone()
    class_head_cfg = head_cfg.clone()

    class_head_param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_BIAS: ConstantInitializer.default_config().set(
                value=-np.log((1 - 0.01) / 0.01),
            ),
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan="fan_in",
                scale=1.0,
                distribution="normal",
            ),
        }
    )
    if use_ds_conv_in_head:
        class_head_cfg.head_conv.conv.param_init = class_head_param_init
    else:
        class_head_cfg.head_conv.param_init = class_head_param_init

    base_cfg = BoxClassHead.default_config().set(
        input_dim=input_dim,
        num_classes=num_classes,
        num_anchors=num_anchors,
        box_head=box_head_cfg,
        class_head=class_head_cfg,
    )
    return base_cfg

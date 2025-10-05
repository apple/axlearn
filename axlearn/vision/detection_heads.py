# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Object Detection prediction heads."""
import itertools
from collections.abc import Sequence
from enum import Enum

from jax import numpy as jnp

from axlearn.common import flax_struct
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.convolution import Conv2D
from axlearn.common.layers import BatchNorm, Linear, get_activation_fn
from axlearn.common.module import Module, Tensor, child_context
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer


class Detections(flax_struct.PyTreeNode):
    """A data class for detections.

    boxes: A float tensor of shape [batch, num_boxes, num_classes * 4] containing encoded box
        coordinates. For boxes shared among classes `num_classes` must be 1.
    scores: A float tensor of shape [batch, num_boxes, num_classes] containing class scores.
    """

    boxes: Tensor
    scores: Tensor


class BoxPredictionType(Enum):
    # Predict a single set of boxes shared by all classes.
    CLASS_AGNOSTIC = 1
    # Predict separate set of boxes for each class.
    CLASS_SPECIFIC = 2


class RCNNDetectionHead(BaseLayer):
    """R-CNN detection head."""

    # Config References:
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
    @config_class
    class Config(BaseLayer.Config):
        """Configures RCNNDetectionHead."""

        # Number of channels in the RoI feature tensor.
        input_dim: Required[int] = REQUIRED
        # Spatial size of the RoI feature tensor.
        roi_size: Required[int] = REQUIRED
        # Number of classes to predict scores.
        num_classes: Required[int] = REQUIRED
        # Box Prediction Type. See Enum for options.
        box_prediction_type: BoxPredictionType = BoxPredictionType.CLASS_SPECIFIC
        # `output_dim` of the conv layers to apply on RoI features.
        conv_dim: Sequence[int] = [256, 256, 256, 256]
        # `output_dim` of the fully connected layers to apply after conv layers.
        fc_dim: Sequence[int] = [1024]
        # Base conv layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(3, 3),
            bias=True,
            padding="SAME",
            param_partition_spec=(None, None, None, "model"),
        )
        # Base linear layer config.
        fc: InstantiableConfig = Linear.default_config().set(
            param_partition_spec=(None, "model"),
        )
        # Normalization layer to apply after each conv and fc layers.
        norm: InstantiableConfig = BatchNorm.default_config().set(decay=0.99, eps=1e-3)
        # Post normalization activation function.
        activation: str = "nn.relu"

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.conv.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan="fan_out", scale=2.0, distribution="normal"
                )
            }
        )
        cfg.fc.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan="fan_out", scale=1 / 3, distribution="uniform"
                )
            }
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        for i, dim in enumerate(cfg.conv_dim):
            self._add_child(
                f"conv_{i}",
                cfg.conv.clone(
                    input_dim=cfg.input_dim if i == 0 else cfg.conv_dim[i - 1],
                    output_dim=dim,
                ),
            )
            self._add_child(f"conv_norm_{i}", cfg.norm.clone(input_dim=dim))

        for i, dim in enumerate(cfg.fc_dim):
            self._add_child(
                f"fc_{i}",
                cfg.fc.clone(
                    input_dim=(
                        (cfg.conv_dim[-1] if cfg.conv_dim else cfg.input_dim) * cfg.roi_size**2
                        if i == 0
                        else cfg.fc_dim[i - 1]
                    ),
                    output_dim=dim,
                ),
            )
            self._add_child(f"fc_norm_{i}", cfg.norm.clone(input_dim=dim))

        self._add_child(
            "classifier",
            cfg.fc.clone(
                input_dim=cfg.fc_dim[-1],
                output_dim=cfg.num_classes,
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                            fan=None, scale=0.01, distribution="normal"
                        )
                    }
                ),
            ),
        )

        if cfg.box_prediction_type is BoxPredictionType.CLASS_AGNOSTIC:
            box_regressor_output_dim = 4
        elif cfg.box_prediction_type is BoxPredictionType.CLASS_SPECIFIC:
            box_regressor_output_dim = 4 * cfg.num_classes
        else:
            raise NotImplementedError(f"Unknown BoxPredictionType: {cfg.box_prediction_type}")

        self._add_child(
            "box_regressor",
            cfg.fc.clone(
                input_dim=cfg.fc_dim[-1],
                output_dim=box_regressor_output_dim,
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                            fan=None, scale=0.001, distribution="normal"
                        )
                    }
                ),
            ),
        )

    def forward(self, inputs: Tensor) -> Detections:
        """Runs forward pass and returns detections.

        Args:
            inputs: A float tensor of shape [batch, num_rois, height, width, filters] containing
                roi features.

        Returns:
            A `Detections` object containing predicted boxes and classes.
        """
        cfg = self.config
        roi_features = inputs
        _, num_rois, height, width, filters = roi_features.shape
        x = jnp.reshape(roi_features, [-1, height, width, filters])
        for i in range(len(cfg.conv_dim)):
            x = getattr(self, f"conv_{i}")(x)
            x = getattr(self, f"conv_norm_{i}")(x)
            x = get_activation_fn(cfg.activation)(x)
        _, _, _, new_filters = x.shape
        x = jnp.reshape(x, [-1, num_rois, height * width * new_filters])
        for i in range(len(cfg.fc_dim)):
            x = getattr(self, f"fc_{i}")(x)
            x = getattr(self, f"fc_norm_{i}")(x)
            x = get_activation_fn(cfg.activation)(x)

        scores = self.classifier(x)
        boxes = self.box_regressor(x)
        return Detections(boxes=boxes, scores=scores)


class BoxProposals(flax_struct.PyTreeNode):
    """A data class for bounding box proposals.

    boxes: A dictionary of float tensors of shape [batch, height_i, width_i, num_anchors * 4]
        containing encoded box coordinates from different feature levels.  Keys indicate the
        feature levels.
    scores: A dictionary of float tensors of shape [batch, height_i, width_i, num_anchors]
        containing proposal scores from different feature levels.  Keys indicate the feature levels.
    """

    boxes: dict[int, Tensor]
    scores: dict[int, Tensor]


class RPNHead(BaseLayer):
    """Region Proposal Network head."""

    # Config References:
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
    @config_class
    class Config(BaseLayer.Config):
        """Configures RPNHead."""

        # Number of channels in the input features.
        input_dim: Required[int] = REQUIRED
        # Min level of Feature pyramid network.
        min_level: int = 3
        # Max level of Feature pyramid network.
        max_level: int = 7
        # Number of anchors on the feature grid. Defaults to 1 scale with 3 aspect ratios.
        anchors_per_location: int = 3
        # `output_dim` of the conv layers to apply on input features.
        conv_dim: Sequence[int] = [256]
        # Base conv layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(3, 3),
            bias=True,
            padding="SAME",
            param_partition_spec=(None, None, None, "model"),
        )
        # Normalization layer to apply after each conv layer.
        norm: InstantiableConfig = BatchNorm.default_config().set(decay=0.99, eps=1e-3)
        # Post normalization activation function.
        activation: str = "nn.relu"

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.conv.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None, scale=0.01, distribution="normal"
                )
            }
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        # Conv layers are shared between feature pyramid network levels.
        for i, dim in enumerate(cfg.conv_dim):
            self._add_child(
                f"conv_{i}",
                cfg.conv.clone(
                    input_dim=cfg.input_dim if i == 0 else cfg.conv_dim[-1],
                    output_dim=dim,
                ),
            )
        # Norm layers are separate between feature pyramid network levels even though we share
        # conv layers.
        for i, level in itertools.product(
            range(len(cfg.conv_dim)), range(cfg.min_level, cfg.max_level + 1)
        ):
            self._add_child(f"norm_{i}_l_{level}", cfg.norm.clone(input_dim=cfg.conv_dim[i]))

        self._add_child(
            "classifier",
            cfg.conv.clone(
                input_dim=cfg.conv_dim[-1] if cfg.conv_dim else cfg.input_dim,
                output_dim=cfg.anchors_per_location,
                window=(1, 1),
                padding="VALID",
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                            fan=None, scale=1e-5, distribution="normal"
                        )
                    }
                ),
            ),
        )
        self._add_child(
            "box_regressor",
            cfg.conv.clone(
                input_dim=cfg.conv_dim[-1] if cfg.conv_dim else cfg.input_dim,
                output_dim=4 * cfg.anchors_per_location,
                window=(1, 1),
                padding="VALID",
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                            fan=None, scale=1e-5, distribution="normal"
                        )
                    }
                ),
            ),
        )

    def forward(self, inputs: dict[int, Tensor]) -> BoxProposals:
        """Runs forward pass and returns proposals.

        Args:
            inputs: A dictionary of features from the feature pyramid network. Keys indicate the
                feature levels.

        Returns:
            A `Proposals` object containing predicted boxes and classes.
        """
        cfg = self.config
        scores = {}
        boxes = {}
        for level in range(cfg.min_level, cfg.max_level + 1):
            x = inputs[level]

            for i in range(len(cfg.conv_dim)):
                with child_context(f"conv_{i}_l_{level}", module=getattr(self, f"conv_{i}")):
                    x = getattr(self, f"conv_{i}")(x)
                x = getattr(self, f"norm_{i}_l_{level}")(x)
                x = get_activation_fn(cfg.activation)(x)

            with child_context(f"classifier_l_{level}", module=self.classifier):
                scores[level] = self.classifier(x)

            with child_context(f"box_regressor_l_{level}", module=self.box_regressor):
                boxes[level] = self.box_regressor(x)

        return BoxProposals(boxes=boxes, scores=scores)

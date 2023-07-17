# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""ROI Align layer."""
import jax

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.utils import NestedTensor, Tensor
from axlearn.vision import spatial_transform_ops


class RoIAligner(BaseLayer):
    """A RoIAlign layer.

    See section 3 of https://arxiv.org/pdf/1703.06870.pdf.

    Note that this layer is only differentiable with respect to the features and uses
    stop_gradient to block gradient with respect to the boxes.
    """

    # Config References:
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
    @config_class
    class Config(BaseLayer.Config):
        """Configures RoIAligner."""

        # Size of the resized square crop.
        output_size: int = 7
        # Whether to align the corners of the box with corners of the grid.
        align_corners: bool = False
        # Min feature level of the Feature Pyramid Network.
        min_level: int = 3
        # Max feature level of the Feature Pyramid Network.
        max_level: int = 7
        # Level of the feature that maps most closely to the resolution of the pre-trained model.
        unit_scale_level: int = 4
        # Pretraining image size. This is used to map the boxes to feature maps at the correct
        # scale. Default value of 224 is a common pre-training image size. See Sec. 4 of paper
        # https://arxiv.org/pdf/1612.03144.pdf
        pretraining_image_size: int = 224

    def forward(self, *, features: NestedTensor, boxes: Tensor) -> Tensor:
        """Applies RoI Align and returns features for boxes.

        Args:
            features: A dictionary of float tensors of shape [batch_size, height_i, width_i,
                channels] containing features. Keys indicate feature levels.
            boxes: A float tensor of shape [batch_size, num_boxes, 4] containing boxes of the form
                [ymin, xmin, ymax, xmax] in image absolute coordinates.

        Returns:
            A 5D float tensor of shape [batch_size, num_boxes, output_size, output_size, channels]
            representing the RoI features.
        """
        boxes = jax.lax.stop_gradient(boxes)
        feature_list = [
            features[level] for level in range(self.config.min_level, self.config.max_level + 1)
        ]

        absolute_box_levels = spatial_transform_ops.get_box_levels(
            boxes=boxes,
            min_level=self.config.min_level,
            max_level=self.config.max_level,
            unit_scale_level=self.config.unit_scale_level,
            pretraining_image_size=self.config.pretraining_image_size,
        )
        relative_box_levels = absolute_box_levels - self.config.min_level
        # Project boxes to feature coordinates.
        boxes /= 2 ** absolute_box_levels[..., None]
        return spatial_transform_ops.roi_align(
            features=feature_list,
            boxes=boxes,
            box_levels=relative_box_levels,
            output_size=(self.config.output_size, self.config.output_size),
            align_corners=self.config.align_corners,
        )

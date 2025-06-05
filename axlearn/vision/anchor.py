# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Anchor generator and labeler definition.

Reference: https://github.com/tensorflow/models/blob/master/official/vision/ops/anchor.py.
"""
import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from axlearn.common.utils import Tensor
from axlearn.vision import matchers


@dataclass
class AnchorLabels:
    """A structure to hold groundtruth that matches anchors.

    groundtruth_boxes: A float tensor of shape [batch, num_anchors, 4] containing matching
        groundtruth boxes in the form [ymin, xmin, ymax, xmax]. Background matches are set to
        zeros.
    groundtruth_classes: An integer tensor of shape [batch, num_anchors, ...] containing classes
        for matching groundtruth boxes. Foreground classes have values in range [1, num_classes]
        while background and ignored matches are set to -1.
    box_paddings: A boolean tensor of shape [batch, num_anchors] indicating paddings in
        `groundtruth_boxes`. All non-foreground matches are marked as paddings.
    class_paddings: A boolean tensor of shape [batch, num_anchors] indicating paddings in
        `groundtruth_classes`. Ignored matches are set as paddings.
    anchor_boxes: A float tensor of shape [num_anchors, 4] containing concatenated per level
        anchor boxes in the form [ymin, xmin, ymax, xmax].
    """

    groundtruth_boxes: Tensor
    groundtruth_classes: Tensor
    box_paddings: Tensor
    class_paddings: Tensor
    anchor_boxes: Tensor


def _expand_trailing_dims(*, target_tensor, reference_tensor):
    trailing_dims = tuple(range(2, reference_tensor.ndim))
    return jnp.expand_dims(target_tensor, trailing_dims)


class AnchorLabeler:
    """Labeler for dense object detector.

    Anchor labeler finds the best matching groundtruth (or background) for each anchor based on
    their intersection over union (IoU) values with groundtruth boxes.

    Differences to the TensorFlow implementation above:
        1.  This implementation does not encode the matched groundtruth with respect to the anchors.
        2.  Also, it expands single image anchor boxes to match the batch dimension of the
            groundtruth boxes/classes and performs batch anchor labeling as opposed to single
            image anchor labeling.
    """

    def __init__(
        self, box_matcher_cfg: matchers.Matcher.Config = matchers.ArgmaxMatcher.default_config()
    ):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
            box_matcher_cfg: InstantiableConfig for matcher class.
        """
        self.box_matcher = box_matcher_cfg.instantiate()

    def _get_matches(
        self, *, per_level_anchor_boxes: dict[str, Tensor], groundtruth_boxes: Tensor
    ) -> tuple[matchers.MatchResults, Tensor, Tensor]:
        """Labels anchors with ground truth boxes.

        Matches anchor boxes with groundtruth boxes and returns match results along with
        foreground and background matches.

        Args:
            per_level_anchor_boxes: A [str, Tensor] dictionary containing a float tensor of
                shape [num_anchors, 4] containing anchors for each level. Each anchor is of the
                form [ymin, xmin, ymax, xmax].
            groundtruth_boxes: A float tensor of shape of [batch_size, num_groundtruth, 4]
                containing groundtruth boxes in the form [ymin, xmin, ymax, xmax]. They must
                be in the same coordinate system as `anchor_boxes`. Values of -1 indicate invalid
                groundtruth that will be ignored.

        Returns:
            match_results of type matchers.MatchResults, and bool tensors foreground_matches
                and background_matches of shape [batch_size, num_anchors] indicating foreground
                or background match.
        """
        match_results = self.box_matcher.match(per_level_anchor_boxes, groundtruth_boxes)

        foreground_matches = match_results.labels == matchers.FOREGROUND
        background_matches = match_results.labels == matchers.BACKGROUND

        return match_results, foreground_matches, background_matches

    @staticmethod
    def _get_matched_groundtruths(
        *,
        match_results: matchers.MatchResults,
        foreground_matches: Tensor,
        groundtruth_classes: Tensor,
        groundtruth_boxes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Returns anchor matched groundtruth boxes and classes.

        Obtains matched groundtruth boxes and groundtruth classes based on match results
        obtained from the box matcher.

        Args:
            match_results: matchers.MatchResults containing matched indices of groundtruth boxes
                with anchor boxes.
            foreground_matches: A bool tensor of shape [batch_size, num_anchors] indicating
                foreground matches of anchor boxes with matched groundtruth boxes.
            groundtruth_classes: An int32 tensor of shape [batch_size, num_groundtruth, ...]
                containing groundtruth classes. The trailing dimension, when present, represent
                multi-class labels.
            groundtruth_boxes: A float tensor of shape of [batch_size, num_groundtruth, 4]
                containing groundtruth boxes in the form [ymin, xmin, ymax, xmax]. They must
                be in the same coordinate system as `anchor_boxes`. Values of -1 indicate invalid
                groundtruth that will be ignored.
        """
        matched_groundtruth_classes = jnp.take_along_axis(
            groundtruth_classes,
            _expand_trailing_dims(
                target_tensor=match_results.matches, reference_tensor=groundtruth_classes
            ),
            axis=1,
        )
        matched_groundtruth_classes = jnp.where(
            _expand_trailing_dims(
                target_tensor=foreground_matches, reference_tensor=matched_groundtruth_classes
            ),
            matched_groundtruth_classes,
            -1 * jnp.ones_like(matched_groundtruth_classes),
        )

        matched_groundtruth_boxes = jnp.take_along_axis(
            groundtruth_boxes, match_results.matches[..., None], axis=-2
        )
        matched_groundtruth_boxes = jnp.where(
            foreground_matches[..., None],
            matched_groundtruth_boxes,
            jnp.zeros_like(matched_groundtruth_boxes),
        )
        return matched_groundtruth_classes, matched_groundtruth_boxes

    def __call__(
        self,
        *,
        per_level_anchor_boxes: dict[str, Tensor],
        groundtruth_boxes: Tensor,
        groundtruth_classes: Tensor,
    ) -> AnchorLabels:
        """Labels anchors with ground truth inputs.

        Args:
            per_level_anchor_boxes: A [str, Tensor] dictionary containing a float tensor of
                shape [num_anchors, 4] containing anchors for each level. Each anchor is of the
                form [ymin, xmin, ymax, xmax].
            groundtruth_boxes: A float tensor of shape of [batch_size, num_groundtruth, 4]
                containing groundtruth boxes in the form [ymin, xmin, ymax, xmax]. They must
                be in the same coordinate system as `anchor_boxes`. Values of -1 indicate invalid
                groundtruth that will be ignored.
            groundtruth_classes: An int32 tensor of shape [batch_size, num_groundtruth, ...]
                containing groundtruth classes. The trailing dimension, when present, represent
                multi-class labels.

        Returns:
            An AnchorLabels object.
        """
        match_results, foreground_matches, background_matches = self._get_matches(
            per_level_anchor_boxes=per_level_anchor_boxes, groundtruth_boxes=groundtruth_boxes
        )

        matched_groundtruth_classes, matched_groundtruth_boxes = self._get_matched_groundtruths(
            match_results=match_results,
            foreground_matches=foreground_matches,
            groundtruth_classes=groundtruth_classes,
            groundtruth_boxes=groundtruth_boxes,
        )

        return AnchorLabels(
            groundtruth_boxes=matched_groundtruth_boxes,
            groundtruth_classes=matched_groundtruth_classes,
            box_paddings=~foreground_matches,
            class_paddings=~(foreground_matches | background_matches),
            # Removing batch dimension here to return concatenated list of per-level anchor boxes.
            anchor_boxes=match_results.anchor_boxes[0],
        )


class AnchorGenerator:
    """Class for generating multi level anchors.

    Note that this a JAX implementation unlike the equivalent TF version above.
    """

    # Default values are based on Tensorflow Model Garden implementation for RetinaNet.
    def __init__(
        self,
        *,
        min_level: int = 3,
        max_level: int = 7,
        num_scales: int = 3,
        aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
        anchor_size: float = 4.0,
    ):
        """Constructs multiscale Anchor Generator.


        Args:
            min_level: Minimum level of the output feature pyramid.
            max_level: Maximum level of the output feature pyramid.
            num_scales: Number of intermediate scales to add on each level. For example,
                num_scales=2 adds one additional intermediate anchor scales [2^0, 2^0.5] on each
                level.
            aspect_ratios: A tuple of aspect ratio for anchors added on each level. The number
                indicates the ratio of width to height. For example, aspect_ratios=[1.0, 2.0, 0.5]
                adds three anchors at each scale.
            anchor_size: Scale of the base anchor assigned to each level.
        """
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size

    def __call__(self, image_size: tuple[int, int]) -> dict[str, Tensor]:
        """Generates multiscale anchor boxes.

        Args:
            image_size: A tuple of integer numbers or Tensors representing [height, width] of the
                input image size.The image_size should be divisible by the largest feature stride
                2^max_level.

        Returns:
            A dictionary of float tensors. Keys indicate the feature level and values are float
            tensors of shape [num_anchors_i, 4] containing anchors. Each anchor is of the form
            [ymin, xmin, ymax, xmax]. Keys are numeric strings.
        """
        boxes_all = {}
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            for scale, aspect_ratio in itertools.product(
                range(self.num_scales), self.aspect_ratios
            ):
                stride = 2**level
                intermediate_scale = 2 ** (scale / float(self.num_scales))
                base_anchor_size = self.anchor_size * stride * intermediate_scale
                aspect_x = aspect_ratio**0.5
                aspect_y = aspect_ratio**-0.5
                half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                half_anchor_size_y = base_anchor_size * aspect_y / 2.0
                y = np.arange(stride / 2, image_size[0], stride)
                x = np.arange(stride / 2, image_size[1], stride)
                xv, yv = np.meshgrid(x, y)
                xv = np.reshape(xv, [-1]).astype(np.float32)
                yv = np.reshape(yv, [-1]).astype(np.float32)
                boxes = np.stack(
                    [
                        yv - half_anchor_size_y,
                        xv - half_anchor_size_x,
                        yv + half_anchor_size_y,
                        xv + half_anchor_size_x,
                    ],
                    axis=-1,
                )
                boxes_l.append(boxes)
            boxes_l = np.stack(boxes_l, axis=-2)
            boxes_l = np.reshape(boxes_l, [-1, 4])
            boxes_all[str(level)] = jnp.asarray(boxes_l)
        return boxes_all

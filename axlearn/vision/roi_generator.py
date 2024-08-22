# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""RoI generator e.g. as seen in RCNN."""

import jax
import jax.numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.module import Module, Tensor
from axlearn.vision import utils_detection
from axlearn.vision.box_coder import BoxCoder
from axlearn.vision.detection_generator import _generate_detections


class RoIGenerator(BaseLayer):
    """Generates proposal boxes and scores for the second stage of an RCNN detector.

    The implementation is a lightweight wrapper around MultilevelDetectionGenerator that adds a
    dummy background class to the raw scores.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures RoIGenerator.

        https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
        """

        # Number of top scores proposals to be kept before applying NMS.
        pre_nms_top_k: int = 2000
        # The score threshold to apply before applying NMS. Proposals with scores are below this
        # threshold are discarded.
        pre_nms_score_threshold: float = 0.0
        # Intersection over Union threshold for Non Max Suppression (NMS).
        # To skip applying NMS set threshold to 1.0.
        nms_iou_threshold: float = 0.7
        # Maximum number of proposals to generate.
        max_num_proposals: int = 1000
        # Box coder to decode predicted boxes.
        box_coder: BoxCoder.Config = BoxCoder.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        self.box_coder = self.config.box_coder.instantiate()

    def forward(
        self,
        raw_boxes: dict[int, Tensor],
        raw_scores: dict[int, Tensor],
        anchor_boxes: dict[str, Tensor],
        image_shape: Tensor,
    ) -> dict[str, Tensor]:
        """Generates final proposals.

        Args:
            raw_boxes: A dict with keys representing FPN levels and values representing box
                tenors of shape [batch, feature_h, feature_w, num_anchors * 4].
            raw_scores: A dict with keys representing FPN levels and values representing logit
                tensors of shape [batch, feature_h, feature_w, num_anchors].
            anchor_boxes: A dict with keys representing FPN levels and values representing anchor
                tenors of shape [num_anchors_i, 4], representing the corresponding anchor boxes
                w.r.t box_outputs.
            image_shape: A tensor of shape of [batch_size, 2] with the image height and width
                that are used to clip proposal boxes that exceed the image boundaries.

        Returns:
            A dictionary with the following tensors:
                proposal_boxes: A float tensor of shape [batch, max_num_detections, 4]
                    representing top proposal boxes in [ymin, xmin, ymax, xmax].
                proposal_scores: A float tensor of shape [batch, max_num_detections] representing
                    sorted confidence scores for proposal boxes. The values are between [0, 1].
                num_proposals: An int tensor of shape [batch] only the first num_proposals
                    boxes are valid proposals.
        """
        all_boxes = []
        all_scores = []
        for per_level_raw_boxes, per_level_raw_scores, per_level_anchor_boxes in zip(
            raw_boxes.values(), raw_scores.values(), anchor_boxes.values()
        ):
            per_level_raw_scores = jnp.reshape(
                per_level_raw_scores, [per_level_raw_scores.shape[0], -1, 1]
            )
            per_level_normalized_scores = jax.nn.sigmoid(per_level_raw_scores)
            per_level_raw_boxes = jnp.reshape(
                per_level_raw_boxes, [per_level_raw_boxes.shape[0], -1, 1, 4]
            )
            per_level_boxes = self.box_coder.decode(
                encoded_boxes=per_level_raw_boxes, anchors=per_level_anchor_boxes[None, :, None, :]
            )
            per_level_boxes = utils_detection.clip_boxes_jax(
                per_level_boxes, image_shape[:, None, None, :]
            )
            (nmsed_boxes, nmsed_scores, _, _) = _generate_detections(
                boxes=per_level_boxes,
                scores=per_level_normalized_scores,
                pre_nms_top_k=self.config.pre_nms_top_k,
                pre_nms_score_threshold=self.config.pre_nms_score_threshold,
                nms_iou_threshold=self.config.nms_iou_threshold,
                max_num_detections=min(per_level_boxes.shape[1], self.config.max_num_proposals),
            )
            all_boxes.append(nmsed_boxes)
            all_scores.append(nmsed_scores)
        roi_boxes = jnp.concatenate(all_boxes, axis=1)
        roi_scores = jnp.concatenate(all_scores, axis=1)
        indices = utils_detection.sorted_top_k(
            roi_scores, k=min(roi_boxes.shape[1], self.config.max_num_proposals)
        )
        rois = {
            "proposal_scores": jax.numpy.take_along_axis(roi_scores, indices, axis=1),
            "proposal_boxes": jax.numpy.take_along_axis(roi_boxes, indices[..., None], axis=1),
        }
        rois["num_proposals"] = jnp.sum(rois["proposal_scores"] > 0, axis=-1)
        return rois

# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""RCNN losses."""
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.loss import (
    ReductionMethod,
    cross_entropy,
    huber_loss,
    sigmoid_cross_entropy_with_logits,
)
from axlearn.common.utils import NestedTensor, Tensor, safe_not


class RPNMetric(BaseLayer):
    """Region Proposal Network Losses."""

    # Default values are set based on the Tensorflow Model Garden Implementation.
    # https://github.com/tensorflow/models/blob/74209bc76c229d6beb8030ddbc777f6e6bb88d30/official/vision/configs/maskrcnn.py#L191-L201
    @config_class
    class Config(BaseLayer.Config):
        """Configures RPNMetric."""

        # Delta for box loss.
        huber_loss_delta: float = 1.0 / 9.0
        # Weight of box loss.
        box_loss_weight: float = 1.0
        # Weight of score loss.
        score_loss_weight: float = 1.0
        # Eps to avoid divide-by-zero during normalization.
        normalizer_eps: float = 1e-08

    def forward(self, *, outputs: NestedTensor, labels: NestedTensor, paddings: Tensor) -> Tensor:
        """Computes Region Proposal Network losses.

        Computes Binary Cross Entropy loss for proposal scores and Huber loss for proposal box
        coordinates.

        Args:
            outputs: A NestedTensor containing `rpn_scores` and `rpn_boxes`.
                rpn_scores: A float tensor of shape [..., num_proposals] containing proposal
                    scores.
                rpn_boxes: A float tensor of shape [..., num_proposals, 4] containing proposal
                    boxes.
            labels: a NestedTensor contains `rpn_score_targets` and `rpn_box_targets`.
                rpn_score_targets: A float tensor of shape [..., num_proposals] containing
                    proposal score targets that are in {0.0, 1.0}.
                rpn_box_targets:  A float tensor of shape [..., num_proposals, 4] containing
                    proposal box targets.

            paddings: A bool tensor of shape [..., num_proposals] indicating paddings.

        Returns:
            A Tensor represents the final loss. `None` is returned if `labels` is None.
        """
        score_loss_normalizer = jnp.sum(safe_not(paddings)) + self.config.normalizer_eps
        true_scores = labels["rpn_score_targets"]
        pred_scores = outputs["rpn_scores"]
        score_loss = (
            jnp.sum(
                sigmoid_cross_entropy_with_logits(logits=pred_scores, targets=true_scores)
                * safe_not(paddings)
            )
            / score_loss_normalizer
        )
        # Box weights to only apply loss on positive samples.
        box_weights = safe_not(paddings) & (true_scores > 0)
        box_loss_normalizer = jnp.sum(box_weights) + self.config.normalizer_eps
        true_boxes = labels["rpn_box_targets"]
        pred_boxes = outputs["rpn_boxes"]
        box_loss = huber_loss(
            predictions=pred_boxes,
            targets=true_boxes,
            delta=self.config.huber_loss_delta,
            reduce_axis=-1,
            sample_weight=box_weights / box_loss_normalizer,
            reduction=ReductionMethod.SUM,
        )
        loss = self.config.score_loss_weight * score_loss + self.config.box_loss_weight * box_loss
        return loss


class RCNNDetectionMetric(BaseLayer):
    """R-CNN 2nd stage detection losses."""

    # Default values are set based on the Tensorflow Model Garden Implementation.
    # https://github.com/tensorflow/models/blob/74209bc76c229d6beb8030ddbc777f6e6bb88d30/official/vision/configs/maskrcnn.py#L191-L201
    @config_class
    class Config(BaseLayer.Config):
        """Configures RCNNDetectionMetric."""

        # Delta for box loss.
        huber_loss_delta: float = 1.0
        # Weight of box loss.
        box_loss_weight: float = 1.0
        # Weight of score loss.
        class_loss_weight: float = 1.0
        # Number of classes including a background class.
        num_classes: Required[int] = REQUIRED
        # Eps to avoid divide-by-zero during normalization.
        normalizer_eps: float = 1e-08

    def forward(self, *, outputs: NestedTensor, labels: NestedTensor, paddings: Tensor) -> Tensor:
        """Computes R-CNN 2nd stage detection losses.

        Computes multiclass Binary Cross Entropy loss for class scores and Huber loss for box
        coordinates.

        When `outputs[boxes]` contains per-class boxes, boxes corresponding to the groundtruth
        class are used for computing box regression loss.

        Args:
            outputs: A NestedTensor containing `class_scores` and `boxes`.
                class_scores: A float tensor of shape [..., num_proposals, num_classes] containing
                    class scores.
                boxes: A float tensor of shape [..., num_proposals, num_classes * 4] containing
                    detection boxes. For boxes shared among classes `num_classes` must be 1.
            labels: a NestedTensor contains `class_targets` and `box_targets`.
                class_targets: A int32 tensor of shape [..., num_detections] containing
                    containing 1-indexed target classes.
                box_targets:  A float tensor of shape [..., num_detections, 4] containing
                    detection box targets.
            paddings: A bool tensor of shape [..., num_detections] indicating paddings.

        Returns:
            A Tensor represents the final loss. `None` is returned if `labels` is None.
        """
        score_loss = cross_entropy(
            logits=outputs["class_scores"],
            target_labels=labels["class_targets"],
            live_targets=safe_not(paddings),
        )[1]["cross_entropy_loss"]

        # Box weights to only apply loss on positive samples.
        box_weights = safe_not(paddings) & (labels["class_targets"] > 0)
        box_loss_normalizer = jnp.sum(box_weights) + self.config.normalizer_eps
        true_boxes = labels["box_targets"]
        # [batch, num_proposals, num_classes * 4]
        pred_boxes = outputs["boxes"]
        batch, num_boxes, num_coordinates = pred_boxes.shape
        num_classes_for_boxes = num_coordinates // 4
        # [batch, num_proposals, num_classes, 4]
        pred_boxes = jnp.reshape(pred_boxes, [batch, num_boxes, num_classes_for_boxes, 4])
        # [batch, num_proposals]
        # `num_classes_for_boxes - 1` ensures we take the only set of boxes available when boxes
        # are shared among classes.
        box_class_ids = jnp.minimum((labels["class_targets"]), num_classes_for_boxes - 1)
        # [batch, num_proposals, 4]
        pred_boxes = jnp.take_along_axis(
            pred_boxes,
            box_class_ids[..., None, None],
            axis=-2,
        )[..., 0, :]

        box_loss = huber_loss(
            predictions=pred_boxes,
            targets=true_boxes,
            delta=self.config.huber_loss_delta,
            reduce_axis=-1,
            sample_weight=box_weights / box_loss_normalizer,
            reduction=ReductionMethod.SUM,
        )
        loss = self.config.class_loss_weight * score_loss + self.config.box_loss_weight * box_loss
        return loss

# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Detection post-processing operations."""

import jax
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.module import Module, Tensor
from axlearn.vision import nms, utils_detection
from axlearn.vision.box_coder import BoxCoder


def _select_top_k_scores(scores_in: Tensor, pre_nms_num_detections: int) -> tuple[Tensor, ...]:
    """Selects top_k scores and indices for each class.

    Args:
        scores_in: A `Tensor` with shape `[batch_size, N, num_classes]`, which stacks class logit
            outputs on all feature levels. The N is the number of total anchors on all levels.
            The num_classes is the number of classes predicted by the model.
        pre_nms_num_detections: Number of candidates before NMS.

    Returns:
        A `Tensor` with shape `[batch_size, pre_nms_num_detections, num_classes]`.
    """
    scores_trans = jax.lax.transpose(scores_in, permutation=[0, 2, 1])
    top_k_scores, top_k_indices = jax.lax.top_k(scores_trans, k=pre_nms_num_detections)
    return jnp.transpose(top_k_scores, [0, 2, 1]), jnp.transpose(top_k_indices, [0, 2, 1])


def _generate_detections(
    boxes: Tensor,
    scores: Tensor,
    pre_nms_top_k: int = 5000,
    pre_nms_score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_num_detections: int = 100,
) -> tuple[Tensor, ...]:
    """Generates the final detections given the model outputs.

    This implementation unrolls classes dimension while using the lax.while_loop
    to implement the batched NMS, so that it can be parallelized at the batch
    dimension. It is TPU compatible.

    Note that `num_classes` dimension of the `boxes` and `scores` tensors do not include
    background class.

    Args:
        boxes: A `Tensor` with shape `[batch_size, N, num_classes, 4]` with box predictions on all
            feature levels. N is the number of total anchors. For boxes shared among classes
            `num_classes` must be 1.
        scores: A `Tensor` with shape `[batch_size, N, num_classes]`, which
            stacks class probability on all feature levels. The N is the number of
            total anchors on all levels. The num_classes is the number of classes
            predicted by the model. Note that the class_outputs here is the raw score.
        pre_nms_top_k: An `int` number of top candidate detections per class before NMS.
        pre_nms_score_threshold: A `float` representing the threshold for deciding
            when to remove boxes based on score.
        nms_iou_threshold: A `float` representing the threshold for deciding whether
            boxes overlap too much with respect to IOU.
        max_num_detections: A `scalar` representing maximum number of boxes retained
            over all classes.

    Returns:
        nms_boxes: A `float` Tensor of shape [batch_size, max_num_detections, 4]
            representing top detected boxes in [y1, x1, y2, x2].
        nms_scores: A `float` Tensor of shape [batch_size, max_num_detections] representing
            sorted confidence scores for detected boxes. The values are between [0, 1].
        nms_classes: An `int` Tensor of shape [batch_size, max_num_detections]
            representing classes for detected boxes.
        valid_detections: An `int` Tensor of shape [batch_size] only the top
            `valid_detections` boxes are valid detections. Values will be in range
            [0, max_num_detections].
    """
    nmsed_boxes = []
    nmsed_classes = []
    nmsed_scores = []
    valid_detections = []
    batch_size, _, num_classes_for_boxes, _ = boxes.shape
    _, total_anchors, num_classes = scores.shape

    # Selects top pre_nms_num scores and indices before NMS.
    scores, indices = _select_top_k_scores(scores, min(total_anchors, pre_nms_top_k))
    for i in range(num_classes):
        scores_i = scores[:, :, i]
        per_class_boxes = boxes[..., min(i, num_classes_for_boxes - 1), :]
        # Obtains pre_nms_top_k before running NMS.
        boxes_i = jnp.stack([x[y] for x, y in zip(per_class_boxes, indices[:, :, i])])
        # Filter out scores.
        boxes_i, scores_i = utils_detection.filter_boxes_by_scores(
            boxes_i, scores_i, min_score_threshold=pre_nms_score_threshold
        )
        nmsed_scores_i, nmsed_boxes_i = nms.non_max_suppression_padded(
            scores_i.astype(jnp.float32),
            boxes_i.astype(jnp.float32),
            max_num_detections,
            iou_threshold=nms_iou_threshold,
        )
        nmsed_classes_i = jnp.full([batch_size, max_num_detections], i)
        nmsed_boxes.append(nmsed_boxes_i)
        nmsed_scores.append(nmsed_scores_i)
        nmsed_classes.append(nmsed_classes_i)

    nmsed_boxes = jnp.concatenate(nmsed_boxes, axis=1)
    nmsed_scores = jnp.concatenate(nmsed_scores, axis=1)
    nmsed_classes = jnp.concatenate(nmsed_classes, axis=1)
    nmsed_scores, indices = jax.lax.top_k(nmsed_scores, k=max_num_detections)

    nmsed_boxes = jnp.stack([nmsed_boxes[i][indices[i]] for i in range(len(nmsed_boxes))])
    nmsed_classes = jnp.stack([nmsed_classes[i][indices[i]] for i in range(len(nmsed_classes))])
    valid_detections = jnp.sum(jnp.greater(nmsed_scores, 0.0).astype(jnp.int32), axis=1)
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


class MultilevelDetectionGenerator(BaseLayer):
    """Generates detected boxes with scores and classes for one-stage detector."""

    # The defaults are set based on Tensorflow Model Garden Implementation.
    # https://github.com/tensorflow/models/blob/74209bc76c229d6beb8030ddbc777f6e6bb88d30/official/vision/configs/maskrcnn.py
    @config_class
    class Config(BaseLayer.Config):
        """Configures MultilevelDetectionGenerator."""

        apply_nms: bool = True
        # Number of top scores proposals to be kept before applying NMS.
        pre_nms_top_k: int = 5000
        # The score threshold to apply before applying NMS. Proposals with scores are below this
        # threshold are discarded.
        pre_nms_score_threshold: float = 0.05
        nms_iou_threshold: float = 0.5
        # The final number of total detections to generate.
        max_num_detections: int = 100
        # Box coder to decode predicted boxes.
        box_coder: BoxCoder.Config = BoxCoder.default_config()
        # If true, ignore first class (in most cases, the background class) when decoding boxes.
        ignore_first_class: bool = True
        # If true, clip boxes to be within image boundaries.
        clip_boxes: bool = True

    def __init__(self, cfg: Config, parent: Module):
        super().__init__(cfg, parent=parent)
        self.box_coder = self.config.box_coder.instantiate()

    # pylint: disable-next=no-self-use
    def _decode_multilevel_outputs(
        self,
        raw_boxes: dict[int, Tensor],
        raw_scores: dict[int, Tensor],
        anchor_boxes: dict[str, Tensor],
        image_shape: Tensor,
    ):
        """Collects dict of multilevel boxes and scores into lists."""
        cfg = self.config
        boxes = []
        scores = []
        levels = list(raw_boxes.keys())
        min_level = int(min(levels))
        max_level = int(max(levels))
        for i in range(min_level, max_level + 1):
            raw_boxes_i = raw_boxes[i]
            raw_scores_i = raw_scores[i]
            batch_size, feat_h_i, feat_w_i, num_anchors_per_loc_4x = raw_boxes_i.shape
            num_locations = feat_h_i * feat_w_i
            num_anchors_per_loc = num_anchors_per_loc_4x // 4
            num_classes = raw_scores_i.shape[-1] // num_anchors_per_loc
            # Applies score transformation and remove the implicit background class.
            scores_i = jax.nn.sigmoid(
                jnp.reshape(
                    raw_scores_i,
                    [batch_size, num_locations * num_anchors_per_loc, num_classes],
                )
            )
            if cfg.ignore_first_class:
                scores_i = jax.lax.slice(scores_i, [0, 0, 1], scores_i.shape)
            # Box decoding. The anchor boxes are shared for all data in a batch.
            # One stage detector only supports class agnostic box regression.
            anchor_boxes_i = jnp.reshape(
                anchor_boxes[str(i)], [batch_size, num_locations * num_anchors_per_loc, 4]
            )
            raw_boxes_i = jnp.reshape(
                raw_boxes_i, [batch_size, num_locations * num_anchors_per_loc, 4]
            )
            boxes_i = self.box_coder.decode(encoded_boxes=raw_boxes_i, anchors=anchor_boxes_i)
            if cfg.clip_boxes:
                # Box clipping.
                boxes_i = utils_detection.clip_boxes_jax(
                    boxes_i, jnp.expand_dims(image_shape, axis=1)
                )
            boxes.append(boxes_i)
            scores.append(scores_i)

        boxes = jnp.concatenate(boxes, axis=1)
        scores = jnp.concatenate(scores, axis=1)
        return boxes, scores

    def forward(
        self,
        raw_boxes: dict[int, Tensor],
        raw_scores: dict[int, Tensor],
        anchor_boxes: dict[str, Tensor],
        image_shape: Tensor,
    ):
        """Generates final detections.

        Args:
            raw_boxes: A `dict` with keys representing FPN levels and values representing box
                tenors of shape `[batch, feature_h, feature_w, num_anchors * 4]`.
            raw_scores: A `dict` with keys representing FPN levels and values representing logit
                tensors of shape `[batch, feature_h, feature_w, num_anchors * num_classes]`.
            anchor_boxes: A `dict` with keys representing FPN levels and values representing anchor
                tenors of shape `[batch_size, K, 4]`, representing the corresponding anchor boxes
                w.r.t `box_outputs`.
            image_shape: A `Tensor` of shape of [batch_size, 2] storing the image height and width
                w.r.t. the scaled image, i.e. the same image space as `box_outputs` and
                `anchor_boxes`.

        Returns:
            If `apply_nms` = True, the return is a dictionary with keys:
                `detection_boxes`: A `float` Tensor of shape [batch, max_num_detections, 4]
                    representing top detected boxes in [y1, x1, y2, x2].
                `detection_scores`: A `float` Tensor of shape [batch, max_num_detections]
                    representing sorted confidence scores for detected boxes.
                    The values are between [0, 1].
                `detection_classes`: An `int` Tensor of shape [batch, max_num_detections]
                    representing classes for detected boxes.
                `num_detections`: An `int` Tensor of shape [batch] only the first `num_detections`
                    boxes are valid detections.
            If `apply_nms` = False, the return is a dictionary with keys:
                `decoded_boxes`: A `float` Tensor of shape [batch, num_raw_boxes, 4]
                    representing all the decoded boxes.
                `decoded_box_scores`: A `float` Tensor of shape [batch, num_raw_boxes]
                    representing scores of all the decoded boxes.
        """
        cfg = self.config
        boxes, scores = self._decode_multilevel_outputs(
            raw_boxes, raw_scores, anchor_boxes, image_shape
        )
        if not cfg.apply_nms:
            return {
                "decoded_boxes": boxes,
                "decoded_box_scores": scores,
            }
        boxes = boxes[..., None, :]
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = _generate_detections(
            boxes,
            scores,
            pre_nms_top_k=cfg.pre_nms_top_k,
            pre_nms_score_threshold=cfg.pre_nms_score_threshold,
            nms_iou_threshold=cfg.nms_iou_threshold,
            max_num_detections=cfg.max_num_detections,
        )
        if cfg.ignore_first_class:
            # Adds 1 to offset the background class which has index 0.
            nmsed_classes += 1
        return {
            "num_detections": valid_detections,
            "detection_boxes": nmsed_boxes,
            "detection_classes": nmsed_classes,
            "detection_scores": nmsed_scores,
        }


class DetectionGenerator(BaseLayer):
    """Generates detected boxes with scores and classes for a single level."""

    # Config References:
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/maskrcnn/r50fpn_640_coco_scratch_tpu4x4.yaml
    @config_class
    class Config(BaseLayer.Config):
        """Configures DetectionGenerator."""

        apply_nms: bool = True
        # Number of top scores detections to be kept before applying NMS.
        pre_nms_top_k: int = 1000
        # The score threshold to apply before applying NMS. Detection with scores are below this
        # threshold are discarded.
        pre_nms_score_threshold: float = 0.05
        # Intersection over Union threshold for Non Max Suppression.
        nms_iou_threshold: float = 0.5
        # Maximum number of detections to generate.
        max_num_detections: int = 100
        # Box coder to decode predicted boxes.
        box_coder: BoxCoder.Config = BoxCoder.default_config().set(weights=(10.0, 10.0, 5.0, 5.0))

    def __init__(self, cfg, parent: Module):
        super().__init__(cfg, parent=parent)
        self.box_coder = self.config.box_coder.instantiate()

    def forward(
        self,
        raw_boxes: Tensor,
        raw_scores: Tensor,
        anchor_boxes: Tensor,
        image_shape: Tensor,
    ) -> dict[str, Tensor]:
        """Generates final detections.

        Args:
            raw_boxes: A float tensor of shape [batch, num_boxes, num_classes * 4] representing box
                coordinates. For boxes shared among classes `num_classes` must be 1.
            raw_scores: A float tensor of shape [batch, num_boxes, num_classes] representing
                logits.
            anchor_boxes: A float tensor of shape [batch, num_boxes, 4] representing anchor boxes
                corresponding to the raw boxes.
            image_shape: A tensor of shape of [batch_size, 2] with the image height and width
                that are used to clip detection boxes that exceed the image boundaries.

        Returns:
            A dictionary with the following tensors:
                detection_boxes: A float tensor of shape [batch, max_num_detections, 4]
                    representing top detected boxes in [y1, x1, y2, x2].
                detection_scores: A float tensor of shape [batch, max_num_detections] representing
                    sorted confidence scores for detected boxes. The values are between [0, 1] and
                    obtained by applying softmax on the raw scores.
                detection_classes: An int Tensor of shape [batch, max_num_detections]
                    representing classes for detected boxes.
                num_detections: An int tensor of shape [batch] only the first num_detections
                    boxes are valid proposals.

            Note that if `apply_nms` is set to False, max_num_detections is equal to
            number of input raw boxes.
        """
        normalized_scores = jax.nn.softmax(raw_scores)
        batch, num_boxes, num_coordinates = raw_boxes.shape
        raw_boxes = jnp.reshape(raw_boxes, [batch, num_boxes, num_coordinates // 4, 4])
        scores_without_bg = normalized_scores[..., 1:]
        # If raw_boxes contain per class predictions, remove boxes corresponding to background
        # class.
        if raw_boxes.shape[-2] > 1:
            raw_boxes = raw_boxes[..., 1:, :]
        # Expand anchors boxes along `num_classes` dimension to be compatible with raw_boxes.
        anchor_boxes = anchor_boxes[..., None, :]
        decoded_boxes = self.box_coder.decode(encoded_boxes=raw_boxes, anchors=anchor_boxes)
        clipped_boxes = utils_detection.clip_boxes_jax(
            decoded_boxes, image_shape[..., None, None, :]
        )
        if not self.config.apply_nms:
            num_detections = jnp.tile(decoded_boxes.shape[1], decoded_boxes.shape[0])
            # [batch, num_detections]
            detection_class_ids = jnp.argmax(scores_without_bg, axis=-1)
            num_classes_for_boxes = raw_boxes.shape[-2]
            # `num_classes_for_boxes - 1` ensures we take the only set of boxes available when
            # boxes are shared among classes.
            box_class_ids = jnp.minimum(detection_class_ids, num_classes_for_boxes - 1)
            return {
                "detection_boxes": jnp.take_along_axis(
                    clipped_boxes, box_class_ids[..., None, None], axis=-2
                )[..., 0, :],
                # Scores are scalar values corresponding to the detection class.
                "detection_scores": jnp.take_along_axis(
                    scores_without_bg, detection_class_ids[..., None], axis=-1
                )[..., 0],
                "detection_classes": detection_class_ids + 1,  # Make class ids 1-indexed.
                "num_detections": num_detections,
            }
        else:
            nmsed_boxes, nmsed_scores, nmsed_classes, num_detections = _generate_detections(
                clipped_boxes,
                scores_without_bg,
                pre_nms_top_k=self.config.pre_nms_top_k,
                pre_nms_score_threshold=self.config.pre_nms_score_threshold,
                nms_iou_threshold=self.config.nms_iou_threshold,
                max_num_detections=self.config.max_num_detections,
            )
        # Adds 1 to offset the background class which has index 0.
        nmsed_classes += 1
        return {
            "detection_boxes": nmsed_boxes,
            "detection_scores": nmsed_scores,
            "detection_classes": nmsed_classes,
            "num_detections": num_detections,
        }

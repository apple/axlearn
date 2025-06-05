# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""R-CNN model implementations, e.g. as seen in https://arxiv.org/abs/1506.01497."""

import jax.numpy as jnp

from axlearn.common.base_model import BaseModel
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_class,
)
from axlearn.common.module import Module, Tensor
from axlearn.vision.anchor import AnchorGenerator
from axlearn.vision.box_coder import BoxCoder
from axlearn.vision.detection_generator import DetectionGenerator
from axlearn.vision.detection_heads import RCNNDetectionHead, RPNHead
from axlearn.vision.fpn import FPN
from axlearn.vision.rcnn_losses import RCNNDetectionMetric, RPNMetric
from axlearn.vision.rcnn_sampler import RCNNSampler
from axlearn.vision.resnet import ResNet
from axlearn.vision.roi_aligner import RoIAligner
from axlearn.vision.roi_generator import RoIGenerator
from axlearn.vision.rpn_sampler import RPNSampler
from axlearn.vision.utils_detection import multi_level_flatten


class FasterRCNN(BaseModel):
    """A Faster R-CNN model.

    This is an implementation of Faster R-CNN based on "Feature Pyramid Networks for Object
    Detection" by Lin et al.

    The model is organized into two stages. The first stage called the Region Proposal Network (
    RPN) predicts proposal boxes and object-ness scores. The second stage called the R-CNN
    resamples features for the proposal boxes and predicts detection boxes and multiclass scores.

    1.  RPN generates multi-scale features using a backbone network in conjunction with a feature
        pyramid network (FPN) and then applies a convolutional head network to predict proposal
        boxes and binary scores.

        Training targets for proposal boxes are computed by taking offsets from pre-defined anchor
        boxes to the matching groundtruth boxes.

    2.  R-CNN network resamples multiscale RPN features for each of the non-max suppressed
        proposal boxes and applies a fully connected network to predict detection boxes and
        multiclass scores.

        Training targets for detection boxes are computed by taking offsets from non-max suppressed
        proposal boxes to the matching groundtruth boxes.

    References:
    ===========
    1. Feature Pyramid Networks for Object Detection: https://arxiv.org/abs/1612.03144
    2. Faster R-CNN: https://arxiv.org/abs/1506.01497
    3. Mask R-CNN: https://arxiv.org/abs/1703.06870
    """

    # Config References:
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/maskrcnn/r50fpn_640_coco_scratch_tpu4x4.yaml
    @config_class
    class Config(BaseModel.Config):
        """Configures FasterRCNN."""

        # The number of classes.
        num_classes: Required[int] = REQUIRED
        # Backbone network to use for feature extraction.
        backbone: InstantiableConfig = ResNet.resnet50_config()
        # FPN Network to generate multi-scale features.
        fpn: InstantiableConfig = FPN.default_config()
        # Head to predict proposal boxes and binary scores.
        rpn_head: InstantiableConfig = RPNHead.default_config()
        # Generator to produce anchor boxes for RPN.
        anchor_generator: InstantiableConfig = config_for_class(AnchorGenerator)
        # RPN Box coder to encode/decode boxes.
        # Default values are based from https://arxiv.org/pdf/1506.01497.pdf.
        rpn_box_coder: BoxCoder.Config = BoxCoder.default_config()
        # RCNN Box coder to encode/decode boxes.
        # Default values are based from https://arxiv.org/pdf/1506.01497.pdf.
        rcnn_box_coder: BoxCoder.Config = BoxCoder.default_config().set(
            weights=(10.0, 10.0, 5.0, 5.0)
        )
        # Subsampler and target assigner for `rpn_metric`
        rpn_sampler: InstantiableConfig = RPNSampler.default_config()
        # Losses of RPN.
        rpn_metric: InstantiableConfig = RPNMetric.default_config()
        # Post-processor for RPN proposals (also used in training).
        roi_generator: InstantiableConfig = RoIGenerator.default_config()
        # Differentiable resampler to gather proposal box features used in R-CNN.
        roi_aligner: InstantiableConfig = RoIAligner.default_config()
        # Head to predict detection boxes and multi-class scores.
        rcnn_detection_head: InstantiableConfig = RCNNDetectionHead.default_config()
        # Subsampler and target assigner for `rcnn_metric`
        rcnn_sampler: InstantiableConfig = RCNNSampler.default_config()
        # Losses of R-CNN.
        rcnn_metric: InstantiableConfig = RCNNDetectionMetric.default_config()
        # Post-processor for R-CNN detections.
        detection_generator: InstantiableConfig = DetectionGenerator.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("backbone", cfg.backbone)
        self._add_child(
            "fpn",
            cfg.fpn.clone(
                input_dims=self.backbone.endpoints_dims,
            ),
        )
        self.anchor_generator = cfg.anchor_generator.clone(
            min_level=cfg.fpn.min_level,
            max_level=cfg.fpn.max_level,
        ).instantiate()
        self._add_child(
            "rpn_head",
            cfg.rpn_head.clone(
                input_dim=cfg.fpn.hidden_dim,
                min_level=cfg.fpn.min_level,
                max_level=cfg.fpn.max_level,
                anchors_per_location=(
                    len(cfg.anchor_generator.aspect_ratios) * cfg.anchor_generator.num_scales
                ),
            ),
        )
        self._add_child(
            "rcnn_detection_head",
            cfg.rcnn_detection_head.clone(
                input_dim=cfg.fpn.hidden_dim,
                roi_size=cfg.roi_aligner.output_size,
                num_classes=cfg.num_classes,
            ),
        )
        self.rpn_box_coder = cfg.rpn_box_coder.instantiate()
        self._add_child("roi_generator", cfg.roi_generator.set(box_coder=cfg.rpn_box_coder))
        self.rcnn_box_coder = cfg.rcnn_box_coder.instantiate()
        self._add_child(
            "detection_generator", cfg.detection_generator.set(box_coder=cfg.rcnn_box_coder)
        )
        self._add_child(
            "roi_aligner",
            cfg.roi_aligner.clone(
                min_level=cfg.fpn.min_level,
                max_level=cfg.fpn.max_level,
            ),
        )
        self._add_child("rcnn_sampler", cfg.rcnn_sampler)
        self._add_child("rpn_sampler", cfg.rpn_sampler)
        self._add_child("rcnn_metric", cfg.rcnn_metric.clone(num_classes=cfg.num_classes))
        self._add_child("rpn_metric", cfg.rpn_metric)

    def _extract_rpn_features(self, image: Tensor) -> dict[int, Tensor]:
        backbone_features = self.backbone(image)
        multilevel_features = self.fpn(backbone_features)
        return multilevel_features

    def predict(self, input_batch: dict[str, Tensor]) -> tuple[None, dict[str, Tensor]]:
        """Runs prediction on images and returns post-processed outputs.

        Args:
            input_batch: A dictionary of tensors:
                image: A float tensor of shape [batch, height, width, 3] containing input images.
                image_info: An integer [batch, 4, 2] tensor that encodes the size of original
                    image, resized image size, along with scale and shift parameters. The tensor
                    is of the form:
                    [[original_height, original_width],
                     [desired_height, desired_width],
                     [y_scale, x_scale],
                     [y_offset, x_offset]].

        Returns:
            A dictionary of tensors with:
                detection_boxes: A float tensor of shape [batch, max_num_detections, 4]
                    containing detection boxes in image coordinates and in the form [ymin, xmin,
                    ymax, xmax]
                detection_scores: A float tensor of shape [batch, max_num_detections]
                    containing confidence scores for detected boxes. `detection_boxes` are
                    sorted by `detection_scores`.
                detection_classes: An integer tensor of shape [batch, max_num_detections]
                    containing classes for the detected boxes.
                num_detections: An integer tensor of shape [batch] indicating the number of
                    valid detections in `detection_boxes`. Only the first `num_detections` boxes
                    are valid.
        """

        multilevel_features = self._extract_rpn_features(input_batch["image"])
        encoded_proposals = self.rpn_head(multilevel_features)

        # Size of the Actual image that sits at the left top of padded input image.
        true_image_shape = input_batch["image_info"][:, 1, :]
        # RoIGenerator returns decoded proposals in `rois`.
        rois = self.roi_generator(
            raw_boxes=encoded_proposals.boxes,
            raw_scores=encoded_proposals.scores,
            anchor_boxes=self.anchor_generator(input_batch["image"].shape[1:3]),
            image_shape=true_image_shape,
        )
        rcnn_features = self.roi_aligner(features=multilevel_features, boxes=rois["proposal_boxes"])
        encoded_detections = self.rcnn_detection_head(rcnn_features)

        # DetectionGenerator returns decoded detection boxes in `final_detections`.
        final_detections = self.detection_generator(
            raw_boxes=encoded_detections.boxes,
            raw_scores=encoded_detections.scores,
            anchor_boxes=rois["proposal_boxes"],
            image_shape=true_image_shape,
        )
        return final_detections

    def forward(self, input_batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Runs forward pass and computes total loss.

        Args:
            input_batch: A dictionary of tensor with
                image_data: A sub dictionary with the following tensors:
                    image: A float tensor of shape [batch, height, width, 3] containing input
                        images.
                    image_info: An integer [batch, 4, 2] tensor that encodes the size of original
                        image, resized image size, along with scale and shift parameters. The
                        tensor is
                        of the form:
                        [[original_height, original_width],
                         [desired_height, desired_width],
                         [y_scale, x_scale],
                         [y_offset, x_offset]].
                labels: A sub dictionary with the following tensors:
                    groundtruth_boxes: A float [batch, max_num_instances, 4] tensor containing
                        padded groundtruth boxes in image coordinates and in the form [ymin, xmin,
                        ymax, xmax]. Values of -1s indicate padding.
                    groundtruth_classes: An integer [batch, max_num_instances] tensor containing
                        padded groundtruth classes. Values of -1s indicate padding.

        Returns:
            A tuple of (loss, {}) where `loss` is a float scalar with total loss.
        """
        image_data = input_batch["image_data"]
        image = image_data["image"]
        labels = input_batch["labels"]
        # Size of the Actual image that sits at the left top of padded input image.
        true_image_shape = image_data["image_info"][:, 1, :]

        # ==================================== First stage (RPN) ==================================
        multilevel_features = self._extract_rpn_features(input_batch["image_data"]["image"])
        encoded_proposals = self.rpn_head(multilevel_features)
        rpn_samples = self.rpn_sampler(
            anchor_boxes=jnp.concatenate(
                list(self.anchor_generator(image.shape[1:3]).values()), axis=0
            ),
            proposal_boxes=multi_level_flatten(encoded_proposals.boxes, last_dim=4),
            proposal_scores=multi_level_flatten(encoded_proposals.scores),
            groundtruth_boxes=labels["groundtruth_boxes"],
            groundtruth_classes=labels["groundtruth_classes"],
        )
        encoded_rpn_box_targets = self.rpn_box_coder.encode(
            boxes=rpn_samples.groundtruth_boxes, anchors=rpn_samples.anchor_boxes
        )
        loss_rpn = self.rpn_metric(
            outputs={
                "rpn_scores": rpn_samples.proposal_scores,
                "rpn_boxes": rpn_samples.proposal_boxes,
            },
            labels={
                "rpn_score_targets": rpn_samples.groundtruth_classes,
                "rpn_box_targets": encoded_rpn_box_targets,
            },
            paddings=rpn_samples.paddings,
        )

        # ============================== Second stage (R-CNN) =====================================
        rois = self.roi_generator(
            raw_boxes=encoded_proposals.boxes,
            raw_scores=encoded_proposals.scores,
            anchor_boxes=self.anchor_generator(image.shape[1:3]),
            image_shape=true_image_shape,
        )
        rcnn_samples = self.rcnn_sampler(
            proposal_boxes=rois["proposal_boxes"],
            groundtruth_boxes=labels["groundtruth_boxes"],
            groundtruth_classes=labels["groundtruth_classes"],
        )
        rcnn_features = self.roi_aligner(
            features=multilevel_features, boxes=rcnn_samples.proposal_boxes
        )
        encoded_detections = self.rcnn_detection_head(rcnn_features)
        encoded_rcnn_box_targets = self.rcnn_box_coder.encode(
            boxes=rcnn_samples.groundtruth_boxes, anchors=rcnn_samples.proposal_boxes
        )
        loss_rcnn = self.rcnn_metric(
            outputs={"class_scores": encoded_detections.scores, "boxes": encoded_detections.boxes},
            labels={
                "class_targets": rcnn_samples.groundtruth_classes,
                "box_targets": encoded_rcnn_box_targets,
            },
            paddings=rcnn_samples.paddings,
        )
        # =========================================================================================

        total_loss = loss_rpn + loss_rcnn
        return total_loss, {}

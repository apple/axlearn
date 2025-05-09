# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/tpu:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""An AXLearn implementation of RetinaNet.

Reference: https://arxiv.org/abs/1708.02002.
"""

import jax
import numpy as np
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_class,
)
from axlearn.common.convolution import Conv2D
from axlearn.common.layers import BatchNorm, get_activation_fn
from axlearn.common.loss import ReductionMethod, focal_loss, huber_loss
from axlearn.common.module import Module, child_context
from axlearn.common.param_init import (
    PARAM_REGEXP_BIAS,
    PARAM_REGEXP_WEIGHT,
    ConstantInitializer,
    DefaultInitializer,
    WeightInitializer,
)
from axlearn.common.utils import NestedTensor, Tensor, safe_not
from axlearn.vision.anchor import AnchorGenerator, AnchorLabeler
from axlearn.vision.box_coder import BoxCoder
from axlearn.vision.detection_generator import MultilevelDetectionGenerator
from axlearn.vision.fpn import FPN
from axlearn.vision.resnet import ResNet
from axlearn.vision.utils_detection import multi_level_flatten


def batch_norm():
    return BatchNorm.default_config().set(decay=0.9, eps=1e-5)


class RetinaNetHead(BaseLayer):
    """RetinaNet head implementation.

    Reference:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/architecture/heads.py
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures RetinaNetHead."""

        input_dim: Required[int] = REQUIRED
        num_classes: Required[int] = REQUIRED
        min_level: int = 3
        max_level: int = 7
        anchors_per_location: int = 9  # Default to 3 anchor sizes with 3 aspect ratios.
        num_layers: int = 4  # The number of conv layers before the final classifier/regressor.
        hidden_dim: int = 256
        # The convolution layer config.
        conv: InstantiableConfig = Conv2D.default_config().set(
            window=(3, 3),
            bias=False,
            padding="SAME",
            param_partition_spec=(None, None, None, "model"),
        )
        norm: InstantiableConfig = batch_norm()  # The normalization layer config.
        activation: str = "nn.relu"  # The activation function.

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

        for i in range(cfg.num_layers):
            # Create weight-sharing conv layers across levels for the classification net.
            self._add_child(
                f"class_conv{i}",
                cfg.conv.clone(
                    input_dim=cfg.input_dim if i == 0 else cfg.hidden_dim,
                    output_dim=cfg.hidden_dim,
                ),
            )
            # Create weight-sharing conv layers across levels for the box regression net.
            self._add_child(
                f"box_conv{i}",
                cfg.conv.clone(
                    input_dim=cfg.input_dim if i == 0 else cfg.hidden_dim,
                    output_dim=cfg.hidden_dim,
                ),
            )
            # Create layer-specific norm layers.
            for level in range(cfg.min_level, cfg.max_level + 1):
                self._add_child(f"class_norm{i}_l{level}", cfg.norm.clone(input_dim=cfg.hidden_dim))
                self._add_child(f"box_norm{i}_l{level}", cfg.norm.clone(input_dim=cfg.hidden_dim))

        # Create the classification layer.
        classifier_cfg = cfg.conv.clone(
            input_dim=cfg.hidden_dim if cfg.num_layers else cfg.input_dim,
            output_dim=cfg.num_classes * cfg.anchors_per_location,
            bias=True,
        )
        classifier_cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_BIAS: ConstantInitializer.default_config().set(
                    value=-np.log((1 - 0.01) / 0.01),
                ),
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None,
                    scale=1e-5,
                    distribution="normal",
                ),
            }
        )
        self._add_child("classifier", classifier_cfg)
        # Create the box regression layer.
        box_regressor_cfg = cfg.conv.clone(
            input_dim=cfg.hidden_dim if cfg.num_layers else cfg.input_dim,
            output_dim=4 * cfg.anchors_per_location,
            bias=True,
        )
        box_regressor_cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None, scale=1e-5, distribution="normal"
                )
            }
        )
        self._add_child(
            "box_regressor",
            box_regressor_cfg,
        )

    def forward(self, inputs: dict[int, Tensor]) -> dict[str, dict[int, Tensor]]:
        """RetinaNet head forward pass.

        Args:
            inputs: A dictionary of (level, features) pairs for the input feature pyramid.
                Must contain all levels in [cfg.min_level, cfg.max_level].

        Returns:
            Classification and box regression outputs. Each is a dictionary of (level, features)
            pairs that contains the output feature pyramid for [cfg.min_level, cfg.max_level].
        """
        cfg = self.config
        class_outputs = {}
        box_outputs = {}

        for level in range(cfg.min_level, cfg.max_level + 1):
            x_class = inputs[level]
            x_box = inputs[level]

            for i in range(cfg.num_layers):
                # Classification nets.
                with child_context(
                    f"class_conv{i}_l{level}", module=getattr(self, f"class_conv{i}")
                ):
                    x_class = getattr(self, f"class_conv{i}")(x_class)
                x_class = getattr(self, f"class_norm{i}_l{level}")(x_class)
                x_class = get_activation_fn(cfg.activation)(x_class)

                # Box regression nets.
                with child_context(f"box_conv{i}_l{level}", module=getattr(self, f"box_conv{i}")):
                    x_box = getattr(self, f"box_conv{i}")(x_box)
                x_box = getattr(self, f"box_norm{i}_l{level}")(x_box)
                x_box = get_activation_fn(cfg.activation)(x_box)

            # Classifier layer.
            with child_context(f"classifier_l{level}", module=self.classifier):
                class_outputs[level] = self.classifier(x_class)

            # Box regressor layer.
            with child_context(f"box_regressor_l{level}", module=self.box_regressor):
                box_outputs[level] = self.box_regressor(x_box)

        return {"class_outputs": class_outputs, "box_outputs": box_outputs}


class RetinaNetMetric(BaseLayer):
    """RetinaNet metrics. See also RetinaNetModel."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures RetinaNetMetric."""

        num_classes: Required[int] = REQUIRED
        focal_loss_alpha: float = 0.25
        focal_loss_gamma: float = 1.5
        huber_loss_delta: float = 0.1
        box_loss_weight: int = 50
        per_category_metrics: bool = False

    # pylint: disable-next=no-self-use
    def _compute_sample_weights(self, labels: dict[str, Tensor]) -> dict[str, Tensor]:
        """Updates sample cls and box sample weights.

        Args:
            labels: See docstring of `forward`.

        Returns:
            A dictionary with the following tensors:
                cls: A float tensor with same shape as labels["class_weights"].
                box: A float tensor with same shape as labels["box_weights"].
        """
        cls_sample_weight = labels["class_weights"]
        box_sample_weight = labels["box_weights"]
        # Sums all positives in a batch for normalization and avoids zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives = jnp.sum(box_sample_weight) + 1.0
        return dict(
            cls=cls_sample_weight / num_positives,
            box=box_sample_weight / num_positives,
        )

    def _convert_to_multihot(self, class_targets: Tensor) -> Tensor:
        """Converts class targets to a multi-hot tensor.

        Args:
            class_targets: A float [batch, num_boxes, ...] tensor with targets for the
                    predicted classes. The trailing dimensions, when present, represent multi-class
                    labels. Multi-class labels are converted to multi-hot vectors before applying
                    binary cross entropy loss / focal loss.

        Returns:
            A boolean [batch, num_boxes, num_classes] tensor.
        """
        cfg = self.config
        y_true_cls_one_hot = jax.nn.one_hot(class_targets, cfg.num_classes)
        # reduce multi-class one-hot vectors into a multi-hot vector.
        y_true_cls_multi_hot = jnp.any(
            y_true_cls_one_hot, axis=tuple(range(2, y_true_cls_one_hot.ndim - 1))
        )
        return y_true_cls_multi_hot

    def _compute_losses(
        self,
        *,
        outputs: dict[str, Tensor],
        labels: dict[str, Tensor],
        sample_weights: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Computes losses.

        Args:
            outputs: See docstring of `forward`.
            labels: See docstring of `forward`.
            sample_weights: A dictionary with the following tensors:
                cls: A float [batch, num_boxes] or [batch, num_boxes, num_classes] tensor with
                    cls weights.
                box: A float [batch, num_boxes] tensor with box weights.

        Returns:
            A dictionary with the following tensors:
                cls: A scalar float representing the class loss.
                box_huber: A scalar float representing the box huber loss.
        """
        cfg = self.config
        y_true_cls_multi_hot = self._convert_to_multihot(labels["class_targets"])

        y_pred_cls = multi_level_flatten(outputs["class_outputs"], last_dim=cfg.num_classes)
        y_true_box = labels["box_targets_encoded"]
        y_pred_box = multi_level_flatten(outputs["box_outputs"], last_dim=4)

        # Compute classification, box regression and total loss.
        cls_loss = focal_loss(
            logits=y_pred_cls,
            targets=y_true_cls_multi_hot,
            alpha=cfg.focal_loss_alpha,
            gamma=cfg.focal_loss_gamma,
            sample_weight=(
                sample_weights["cls"]
                if sample_weights["cls"].ndim == y_pred_cls.ndim
                else jnp.expand_dims(sample_weights["cls"], axis=-1)
            ),
        )
        box_huber_loss = huber_loss(
            predictions=y_pred_box,
            targets=y_true_box,
            delta=cfg.huber_loss_delta,
            reduce_axis=-1,
            sample_weight=sample_weights["box"],
            reduction=ReductionMethod.SUM,
        )

        return dict(cls=cls_loss, box_huber=box_huber_loss)

    def forward(self, outputs: dict[str, Tensor], labels: dict[str, Tensor]) -> Tensor:
        """Computes RetinaNet metrics.

        Args:
            outputs: A dictionary with the following tensors:
                box_outputs: A float [batch, num_boxes, 4] tensor with predicted boxes.
                class_outputs: A float [batch, num_boxes, num_classes] tensor with predicted
                    class scores.
            labels: A dictionary with the following tensors:
                box_targets_encoded: A float [batch, num_boxes, 4] tensor with encoded targets for
                    the predicted boxes.
                class_targets: A float [batch, num_boxes, ...] tensor with targets for the
                    predicted classes. The trailing dimensions, when present, represent multi-class
                    labels. Multi-class labels are converted to multi-hot vectors before applying
                    binary cross entropy loss / focal loss.
                box_weights: A float [batch, num_boxes] tensor with weights for
                    `box_targets_encoded`.
                class_weights: A float [batch, num_boxes] or [batch, num_boxes, num_classes]
                    tesnsor with weights for `class_targets`.

        Returns:
            A Tensor represents the final loss.
        """
        cfg = self.config

        sample_weights = self._compute_sample_weights(labels)

        losses = self._compute_losses(outputs=outputs, labels=labels, sample_weights=sample_weights)

        loss = losses["cls"] + cfg.box_loss_weight * losses["box_huber"]
        self.add_summary("class_loss", losses["cls"])
        self.add_summary("box_loss", losses["box_huber"])
        self.add_summary("loss", loss)
        return loss


class RetinaNetModel(BaseLayer):
    """The RetinaNet model."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures RetinaNetModel."""

        num_classes: Required[int] = REQUIRED  # The number of classification classes.
        backbone: InstantiableConfig = ResNet.resnet50_config()
        fpn: InstantiableConfig = FPN.default_config()
        head: InstantiableConfig = RetinaNetHead.default_config()
        fpn_hidden_dim: int = 256
        head_hidden_dim: int = 256
        # Box coder to encode/decode boxes.
        box_coder: BoxCoder.Config = BoxCoder.default_config()
        detection_generator: InstantiableConfig = MultilevelDetectionGenerator.default_config()
        # Generator to produce anchor boxes.
        anchor_generator: InstantiableConfig = config_for_class(AnchorGenerator)
        # Labeler to assign groundtruth to anchors.
        anchor_labeler: InstantiableConfig = config_for_class(AnchorLabeler)
        metrics: InstantiableConfig = RetinaNetMetric.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("backbone", cfg.backbone)
        self._add_child(
            "fpn",
            cfg.fpn.clone(
                input_dims=self.backbone.endpoints_dims,
                hidden_dim=cfg.fpn_hidden_dim,
            ),
        )
        self._add_child(
            "head",
            cfg.head.clone(
                input_dim=cfg.fpn_hidden_dim,
                hidden_dim=cfg.head_hidden_dim,
                num_classes=cfg.num_classes,
            ),
        )
        self.anchor_generator = cfg.anchor_generator.instantiate()
        self.anchor_labeler = cfg.anchor_labeler.instantiate()
        self.box_coder = cfg.box_coder.instantiate()
        self._add_child("detection_generator", cfg.detection_generator.set(box_coder=cfg.box_coder))
        self._add_child("metrics", cfg.metrics.clone(num_classes=cfg.num_classes))

    def _generate_anchors(self, image_shape):
        # Tile anchors along the batch dimension to be compatible with
        # `multilevel_detection_generator`.
        batch_size = image_shape[0]
        anchor_boxes = {
            key: jnp.tile(boxes[None, ...], [batch_size, 1, 1])
            for key, boxes in self.anchor_generator(image_size=image_shape[1:3]).items()
        }
        return anchor_boxes

    def _predict_raw(self, input_batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Generate a dict of {level: feature} represents endpoints from backbone.
        x = self.backbone(input_batch["image"])
        # Generate a dict of {level: feature} represents multi-scale feature pyramid.
        x = self.fpn(x)
        # Generate a dict of {name: feature} represents classification logits and box offsets.
        outputs = self.head(x)
        return outputs

    def predict(self, input_batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Runs prediction on images and returns post-processed results.

        This method is used for inference and evaluation.

        Args:
            input_batch: A dictionary with the following tensors:
                image: A float [batch, height, width, 3] tensor containing images.
                image_info: An integer [batch, 4, 2] tensor that encodes the size of original
                    image, resized image size, along with scale and shift parameters. The tensor
                    is of the form:
                    [[original_height, original_width],
                     [desired_height, desired_width],
                     [y_scale, x_scale],
                     [y_offset, x_offset]].

        Returns:
            A dictionary with the following entries:
                `detection_boxes`: A `float` Tensor of shape [batch, max_num_detections, 4]
                    representing top detected boxes in [y1, x1, y2, x2].
                `detection_scores`: A `float` Tensor of shape [batch, max_num_detections]
                    representing sorted confidence scores for detected boxes. The values are
                    between [0, 1].
                `detection_classes`: An `int` Tensor of shape [batch, max_num_detections]
                    representing classes for detected boxes.
                `num_detections`: An `int` Tensor of shape [batch] only the first `num_detections`
                    boxes are valid detections.
        """
        outputs = self._predict_raw(input_batch)
        processed_outputs = self.detection_generator(
            raw_boxes=outputs["box_outputs"],
            raw_scores=outputs["class_outputs"],
            anchor_boxes=self._generate_anchors(image_shape=input_batch["image"].shape),
            image_shape=input_batch["image_info"][:, 1, :],
        )
        return processed_outputs

    def forward(self, input_batch: dict[str, Tensor]) -> tuple[float, NestedTensor]:
        """Runs forward pass and returns total loss.

        Args:
            input_batch: A dictionary of tensor with
                image_data: A sub-dictionary with the following tensors:
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
                labels: A sub-dictionary with the following tensors:
                    class_targets: A dictionary of float [batch, height_i, width_i,
                        num_anchors_per_location] tensors with class targets. The keys indicate the
                        feature level.
                    box_targets: A dictionary of float [batch, height_i, width_i,
                        num_anchors_per_location * 4] tensors with box targets. The keys indicate
                        the feature level.
                    anchor_boxes: A dictionary of float [batch, height_i, width_i,
                        num_anchors_per_location * 4] tensors with anchor boxes. The keys indicate
                        the feature level.
                    box_weights: A float
                        [batch, sum_{i} (height_{i} * width_{i} * num_anchors_per_location)] tensor
                        with box weights.
                    class_weights: A float
                        [batch, sum_{i} (height_{i} * width_{i} * num_anchors_per_location)] tensor
                        with class weights.

        Returns:
            A tuple of (loss, {}) where `loss` is a float scalar with total loss.
        """
        labels = input_batch["labels"]
        image_data = input_batch["image_data"]
        outputs = self._predict_raw(image_data)
        per_level_anchor_boxes = self.anchor_generator(image_data["image"].shape[1:3])
        anchor_labels = self.anchor_labeler(
            per_level_anchor_boxes=per_level_anchor_boxes,
            groundtruth_boxes=labels["groundtruth_boxes"],
            groundtruth_classes=labels["groundtruth_classes"],
        )
        loss = self.metrics(
            outputs={
                "box_outputs": outputs["box_outputs"],
                "class_outputs": outputs["class_outputs"],
            },
            labels={
                "box_targets_encoded": self.box_coder.encode(
                    boxes=anchor_labels.groundtruth_boxes,
                    anchors=anchor_labels.anchor_boxes[None, ...],
                ),
                "box_targets": anchor_labels.groundtruth_boxes,
                "class_targets": anchor_labels.groundtruth_classes,
                "box_weights": safe_not(anchor_labels.box_paddings),
                "class_weights": safe_not(anchor_labels.class_paddings),
                # TODO(pdufter) move self._generate_anchors to anchor_labeler and only pass
                # anchor_labels to metrics
                "anchor_boxes_tiled": self._generate_anchors(
                    image_shape=input_batch["image_data"]["image"].shape
                ),
                "image_shape": input_batch["image_data"]["image_info"][:, 1, :],
            },
        )
        return loss, {}


def set_retinanet_config(
    min_level: int = 3,
    max_level: int = 7,
) -> RetinaNetModel.Config:
    """Returns the RetinaNetModel config.

    Reference: https://arxiv.org/abs/1708.02002

    Args:
        min_level: Minimum level for FPN, box and class head.
        max_level: Maximum level for FPN, box and class head.

    Returns:
        The RetinaNetModel config.
    """
    fpn_cfg = FPN.default_config().set(
        min_level=min_level,
        max_level=max_level,
    )

    head_cfg = RetinaNetHead.default_config().set(
        min_level=min_level,
        max_level=max_level,
    )

    retinanet_cfg = RetinaNetModel.default_config().set(
        fpn=fpn_cfg,
        head=head_cfg,
        anchor_generator=config_for_class(AnchorGenerator).clone(
            min_level=min_level,
            max_level=max_level,
        ),
    )

    return retinanet_cfg

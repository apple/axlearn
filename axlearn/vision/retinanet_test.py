# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for RetinaNet components."""
import jax.random
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.config import config_for_function
from axlearn.common.module import functional as F
from axlearn.common.vision_transformer import build_vit_model_config
from axlearn.vision.input_detection import DetectionInput, fake_detection_dataset
from axlearn.vision.retinanet import (
    RetinaNetHead,
    RetinaNetMetric,
    RetinaNetModel,
    set_retinanet_config,
)


class RetinaNetHeadTest(parameterized.TestCase):
    @parameterized.product(
        is_training=(False, True),
        num_layers=(0, 4),
    )
    def test_forward(self, is_training, num_layers):
        batch_size = 2
        min_level = 3
        max_level = 7
        image_size = 256
        hidden_dim = 256
        input_dim = 64
        anchors_per_location = 9
        num_classes = 91

        inputs = {}
        for level in range(min_level, max_level + 1):
            inputs[level] = np.random.uniform(
                -1, 1, [batch_size, image_size // 2**level, image_size // 2**level, input_dim]
            ).astype(np.float32)

        cfg = RetinaNetHead.default_config().set(
            name="test",
            min_level=min_level,
            max_level=max_level,
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            anchors_per_location=anchors_per_location,
            num_classes=num_classes,
            num_layers=num_layers,
        )
        model: RetinaNetHead = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )

        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs["class_outputs"])
            self.assertIn(level, outputs["box_outputs"])

            class_expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                anchors_per_location * num_classes,
            )
            self.assertEqual(class_expected_shape, outputs["class_outputs"][level].shape)

            box_expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                anchors_per_location * 4,
            )
            self.assertEqual(box_expected_shape, outputs["box_outputs"][level].shape)


# Reference numbers from: https://github.com/tensorflow/models/tree/master/official/vision.
REF_BACKBONE_PARAMS = 23561152
REF_FPN_PARAMS = 3873792
REF_HEAD_PARAMS = 6732375


def _input_config(is_training: bool, batch_size: int, image_size: int):
    cfg = DetectionInput.default_config().set(name="test", is_training=is_training)
    cfg.source.set(
        dataset_name="coco/2017",
        split="validation",
        train_shuffle_buffer_size=100,
    )
    cfg.processor.set(image_size=(image_size, image_size))
    cfg.batcher.set(global_batch_size=batch_size)
    return cfg


class RetinaNetModelTest(parameterized.TestCase):
    def test_model_params(self):
        inputs = {
            "image_data": {
                "image": np.random.uniform(-1, 1, [2, 256, 256, 3]).astype(np.float32),
                "image_info": np.array(
                    [
                        [[256, 256], [256, 256], [1.0, 1.0], [0.0, 0.0]],
                        [[256, 256], [256, 256], [1.0, 1.0], [0.0, 0.0]],
                    ]
                ),
            },
            "labels": {
                "groundtruth_boxes": np.array(
                    [
                        [[10, 20, 100, 200], [33, 35, 60, 100]],
                        [[120, 140, 250, 210], [-1, -1, -1, -1]],
                    ]
                ),
                "groundtruth_classes": np.array([[2, 54], [76, -1]]),
            },
        }

        cfg = set_retinanet_config(
            min_level=3,
            max_level=7,
        )
        cfg.name = "test"
        cfg.num_classes = 91
        cfg.fpn_hidden_dim = 256
        cfg.head_hidden_dim = 256
        # Set head conv bias to True only to match the TF implementation.
        # Will verify if adding bias is necessary since batch norm layers are appended.
        cfg.head.conv.bias = True

        model: RetinaNetModel = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=inputs),
        )

        # Test backbone number of parameters against the TF implementation.
        self.assertEqual(
            REF_BACKBONE_PARAMS,
            utils.count_model_params(state["backbone"]),
        )
        # Test fpn number of parameters against the TF implementation.
        self.assertEqual(REF_FPN_PARAMS, utils.count_model_params(state["fpn"]))
        # Test head number of parameters against the TF implementation.
        self.assertEqual(REF_HEAD_PARAMS, utils.count_model_params(state["head"]))

    # pylint: disable=no-self-use
    @parameterized.product(
        is_training=(False, True),
        backbone_type=("resnet", "vit"),
    )
    def test_model_forward(self, is_training, backbone_type):
        image_size = 256
        batch_size = 8

        input_cfg = _input_config(is_training, batch_size, image_size)
        input_cfg.source = config_for_function(fake_detection_dataset)
        if not is_training:
            input_cfg.source.set(total_num_examples=batch_size)
        dataset = input_cfg.instantiate(parent=None)
        inputs = next(iter(dataset))
        cfg = set_retinanet_config(
            min_level=3,
            max_level=7,
        )
        cfg.name = "test"
        cfg.num_classes = 91
        cfg.fpn_hidden_dim = 256
        cfg.head_hidden_dim = 256
        if backbone_type == "vit":
            cfg.backbone = build_vit_model_config(
                num_layers=1,
                model_dim=8,
                num_heads=4,
                image_size=(image_size, image_size),
                patch_size=(8, 8),
            )
        model: RetinaNetModel = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=inputs if is_training else inputs["image_data"]),
            method="forward" if is_training else "predict",
        )


class RetinaNetMetricTest(parameterized.TestCase):
    # pylint: disable=no-self-use
    @parameterized.parameters(
        # Single-class labels
        ([3, 4, 0], [[0, 0, 0, 1000, 0], [0, 0, 0, 0, 1000], [1000, 0, 0, 0, 0]]),
        # Multi-class labels
        (
            [[3, -1, -1], [4, 2, -1], [0, 2, -1]],
            [[0, 0, 0, 1000, 0], [0, 0, 1000, 0, 1000], [1000, 0, 1000, 0, 0]],
        ),
    )
    def test_metric_forward(self, class_labels, class_outputs):
        outputs = {
            "box_outputs": {
                3: np.array([[[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1], [0.5, 0.3, 1, 0.5]]])
            },
            "class_outputs": {3: np.array([class_outputs])},
        }
        labels = {
            "box_targets_encoded": np.array([[[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1], [0, 0, 0, 0]]]),
            "class_targets": np.array([class_labels]),
            "box_weights": np.array([[1, 1, 0]]),
            "class_weights": np.array([[1, 1, 0]]),
        }
        cfg = RetinaNetMetric.default_config().set(
            name="test",
            num_classes=5,
            focal_loss_alpha=1,
        )
        model: RetinaNetMetric = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels),
        )
        self.assertEqual(loss, 0)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

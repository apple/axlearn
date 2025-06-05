# Copyright © 2023 Apple Inc.

"""Tests detection heads."""
import jax.random
import numpy as np
from absl.testing import parameterized

from axlearn.common import utils
from axlearn.common.module import functional as F
from axlearn.vision.detection_heads import BoxPredictionType, RCNNDetectionHead, RPNHead


class RCNNDetectionHeadTest(parameterized.TestCase):
    """Tests RCNNDetectionHead."""

    @parameterized.product(
        is_training=(False, True),
        conv_dim=([], [128] * 4),
        box_prediction_type=(BoxPredictionType.CLASS_AGNOSTIC, BoxPredictionType.CLASS_SPECIFIC),
    )
    def test_model_forward(self, is_training, conv_dim, box_prediction_type):
        batch_size = 2
        fc_dim = [256]
        input_dim = 32
        num_classes = 5
        num_rois = 2
        roi_size = 7

        inputs = np.random.uniform(
            -1, 1, [batch_size, num_rois, roi_size, roi_size, input_dim]
        ).astype(np.float32)
        cfg = RCNNDetectionHead.default_config().set(
            name="MaskRCNNDetectionHead",
            input_dim=input_dim,
            conv_dim=conv_dim,
            fc_dim=fc_dim,
            num_classes=num_classes,
            roi_size=roi_size,
            box_prediction_type=box_prediction_type,
        )

        model: RCNNDetectionHead = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        if box_prediction_type is BoxPredictionType.CLASS_AGNOSTIC:
            box_output_dim = 4
        elif box_prediction_type is BoxPredictionType.CLASS_SPECIFIC:
            box_output_dim = 4 * num_classes
        self.assertEqual((batch_size, num_rois, box_output_dim), outputs.boxes.shape)
        self.assertEqual((batch_size, num_rois, num_classes), outputs.scores.shape)

        conv_param_count = 0
        conv_norm_param_count = 0
        for in_dim, out_dim in zip([input_dim] + conv_dim, conv_dim):
            conv_param_count += 3 * 3 * in_dim * out_dim + out_dim
            conv_norm_param_count += out_dim * 4
        fc_param_count = 0
        fc_norm_param_count = 0
        feature_dim = conv_dim[-1] if conv_dim else input_dim
        for in_dim, out_dim in zip([feature_dim] + fc_dim, fc_dim):
            fc_param_count += (in_dim * roi_size**2) * out_dim + out_dim
            fc_norm_param_count += out_dim * 4
        box_regressor_param_count = fc_dim[-1] * box_output_dim + box_output_dim
        box_classifier_param_count = fc_dim[-1] * num_classes + num_classes
        total_param_count = (
            box_regressor_param_count
            + box_classifier_param_count
            + conv_param_count
            + conv_norm_param_count
            + fc_param_count
            + fc_norm_param_count
        )
        self.assertEqual(total_param_count, utils.count_model_params(state))


class RPNHeadTest(parameterized.TestCase):
    @parameterized.product(is_training=(False, True), conv_dim=([], [256, 256, 256, 256]))
    def test_model_forward(self, is_training, conv_dim):
        batch_size = 2
        min_level = 3
        max_level = 7
        image_size = 256
        input_dim = 64
        anchors_per_location = 9

        inputs = {}
        for level in range(min_level, max_level + 1):
            inputs[level] = np.random.uniform(
                -1, 1, [batch_size, image_size // 2**level, image_size // 2**level, input_dim]
            ).astype(np.float32)

        cfg = RPNHead.default_config().set(
            name="RPNHead",
            min_level=min_level,
            max_level=max_level,
            input_dim=input_dim,
            anchors_per_location=anchors_per_location,
            conv_dim=conv_dim,
        )
        model: RPNHead = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )

        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs.scores)
            self.assertIn(level, outputs.boxes)

            class_expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                anchors_per_location,
            )
            self.assertEqual(class_expected_shape, outputs.scores[level].shape)

            box_expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                anchors_per_location * 4,
            )
            self.assertEqual(box_expected_shape, outputs.boxes[level].shape)

        feature_dim = conv_dim[-1] if conv_dim else input_dim
        box_regressor_param_count = (
            feature_dim * anchors_per_location * 4 + anchors_per_location * 4
        )
        box_classifier_param_count = feature_dim * anchors_per_location + anchors_per_location
        conv_param_count = 0
        norm_param_count = 0
        for in_dim, out_dim in zip([input_dim] + conv_dim, conv_dim):
            conv_param_count += 3 * 3 * in_dim * out_dim + out_dim
            norm_param_count += out_dim * 4 * (max_level - min_level + 1)
        total_param_count = (
            box_regressor_param_count
            + box_classifier_param_count
            + conv_param_count
            + norm_param_count
        )
        self.assertEqual(total_param_count, utils.count_model_params(state))

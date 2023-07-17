# Copyright © 2023 Apple Inc.

"""Tests RCNN losses."""
import jax.random
import numpy as np
from absl.testing import absltest

from axlearn.common.module import functional as F
from axlearn.vision import rcnn_losses


# pylint: disable=no-self-use
class RPNMetricTest(absltest.TestCase):
    """Tests RPNMetric."""

    def test_perfect_rpn_loss_with_padding(self):
        paddings = np.array([[False, True, False], [False, False, True]])
        rpn_score_targets = np.array([[0, 100, 1], [1, 1, 500]])
        rpn_scores = np.array([[-100, 500, 100], [100, 100, 400]])
        rpn_box_targets = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]],
            ]
        )
        rpn_boxes = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [-1.0, -1.0, -1.0, -1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [-1.0, -1.0, -1.0, -1.0]],
            ]
        )

        outputs = {
            "rpn_scores": rpn_scores,
            "rpn_boxes": rpn_boxes,
        }
        labels = {
            "rpn_box_targets": rpn_box_targets,
            "rpn_score_targets": rpn_score_targets,
        }
        model: rcnn_losses.RPNMetric = (
            rcnn_losses.RPNMetric.default_config().set(name="rpn_loss").instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels, paddings=paddings),
        )
        np.testing.assert_almost_equal(0.0, loss)

    def test_score_loss_normalization(self):
        paddings = np.array([[False, True, False], [False, False, True]])
        rpn_score_targets = np.array([[0, 100, 1], [1, 1, 500]])
        rpn_scores = np.array([[-100, 500, 100], [100, -100, 400]])
        rpn_box_targets = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]],
            ]
        )
        rpn_boxes = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [-1.0, -1.0, -1.0, -1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [-1.0, -1.0, -1.0, -1.0]],
            ]
        )

        outputs = {
            "rpn_scores": rpn_scores,
            "rpn_boxes": rpn_boxes,
        }
        labels = {
            "rpn_box_targets": rpn_box_targets,
            "rpn_score_targets": rpn_score_targets,
        }
        model: rcnn_losses.RPNMetric = (
            rcnn_losses.RPNMetric.default_config().set(name="rpn_loss").instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels, paddings=paddings),
        )
        np.testing.assert_almost_equal(100.0 / 4, loss)

    def test_box_loss_normalization(self):
        paddings = np.array([[False, True, False], [False, False, True]])
        rpn_score_targets = np.array([[0, 100, 1], [1, 1, 500]])
        rpn_scores = np.array([[-100, 500, 100], [100, 100, 400]])
        rpn_box_targets = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]],
            ]
        )
        rpn_boxes = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [-1.0, -1.0, -1.0, -1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.0, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [-1.0, -1.0, -1.0, -1.0]],
            ]
        )

        outputs = {
            "rpn_scores": rpn_scores,
            "rpn_boxes": rpn_boxes,
        }
        labels = {
            "rpn_box_targets": rpn_box_targets,
            "rpn_score_targets": rpn_score_targets,
        }
        model: rcnn_losses.RPNMetric = (
            rcnn_losses.RPNMetric.default_config().set(name="rpn_loss").instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels, paddings=paddings),
        )
        np.testing.assert_almost_equal(0.0123 / 3, loss, decimal=4)


class RCNNDetectionMetricTest(absltest.TestCase):
    """Tests RCNNDetectionMetric."""

    def test_perfect_frcnn_loss_with_padding(self):
        paddings = np.array([[False, True, False], [False, False, True]])
        rcnn_score_targets = np.array([[0, 100, 1], [0, 2, 500]], dtype=np.int32)
        rcnn_scores = np.array(
            [
                [[-500, -1000, -1000], [-400, -400, -400], [-1000, -500, -1000]],
                [[-500, -1000, -1000], [-1000, -1000, -500], [-400, -400, -400]],
            ],
            dtype=np.float32,
        )
        rcnn_box_targets = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]],
            ]
        )
        rcnn_boxes = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [-1.0, -1.0, -1.0, -1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [-1.0, -1.0, -1.0, -1.0]],
            ]
        )

        outputs = {
            "class_scores": rcnn_scores,
            "boxes": rcnn_boxes,
        }
        labels = {
            "box_targets": rcnn_box_targets,
            "class_targets": rcnn_score_targets,
        }
        model: rcnn_losses.RCNNDetectionMetric = (
            rcnn_losses.RCNNDetectionMetric.default_config()
            .set(name="frcnn_loss", num_classes=3)
            .instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels, paddings=paddings),
        )
        np.testing.assert_almost_equal(0.0, loss)

    def test_score_loss_normalization(self):
        paddings = np.array([[False, True, False], [False, False, True]])
        rcnn_score_targets = np.array([[0, 100, 1], [0, 2, 500]], dtype=np.int32)
        rcnn_scores = np.array(
            [
                [[-500, -1000, -1000], [-400, -400, -400], [-1000, -500, -1000]],
                [[-1000, -500, -1000], [-1000, -1000, -500], [-400, -400, -400]],
            ],
            dtype=np.float32,
        )
        rcnn_box_targets = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]],
            ]
        )
        rcnn_boxes = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [-1.0, -1.0, -1.0, -1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [-1.0, -1.0, -1.0, -1.0]],
            ]
        )

        outputs = {
            "class_scores": rcnn_scores,
            "boxes": rcnn_boxes,
        }
        labels = {
            "box_targets": rcnn_box_targets,
            "class_targets": rcnn_score_targets,
        }
        model: rcnn_losses.RCNNDetectionMetric = (
            rcnn_losses.RCNNDetectionMetric.default_config()
            .set(name="frcnn_loss", num_classes=3)
            .instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels, paddings=paddings),
        )
        np.testing.assert_almost_equal(500 / 4, loss)

    def test_box_loss_normalization(self):
        paddings = np.array([[False, True, False], [False, False, True]])
        rcnn_score_targets = np.array([[0, 100, 2], [1, 3, 500]], dtype=np.int32)
        rcnn_scores = np.array(
            [
                [[100, -100, -100, -100], [-400, -400, -400, -400], [-100, -100, 100, -100]],
                [[-100, 100, -100, -100], [-100, -100, -100, 100], [-400, -400, -400, -400]],
            ],
            dtype=np.float32,
        )
        rcnn_box_targets = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]],
            ]
        )
        rcnn_boxes = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [-1.0, -1.0, -1.0, -1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.0, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [-1.0, -1.0, -1.0, -1.0]],
            ]
        )

        outputs = {
            "class_scores": rcnn_scores,
            "boxes": rcnn_boxes,
        }
        labels = {
            "box_targets": rcnn_box_targets,
            "class_targets": rcnn_score_targets,
        }
        model: rcnn_losses.RCNNDetectionMetric = (
            rcnn_losses.RCNNDetectionMetric.default_config()
            .set(name="frcnn_loss", num_classes=4)
            .instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels, paddings=paddings),
        )
        np.testing.assert_almost_equal(0.0313 / 3, loss, decimal=4)

    def test_per_class_box_predictions(self):
        paddings = np.array([[False, True, False], [False, False, True]])
        rcnn_score_targets = np.array([[0, 100, 2], [1, 3, 500]], dtype=np.int32)
        rcnn_scores = np.array(
            [
                [
                    [-500, -1000, -1000, -1000],
                    [-400, -400, -400, -400],
                    [-1000, -1000, -500, -1000],
                ],
                [
                    [-1000, -500, -1000, -1000],
                    [-1000, -1000, -1000, -500],
                    [-400, -400, -400, -400],
                ],
            ],
            dtype=np.float32,
        )
        rcnn_box_targets = np.array(
            [
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0]],
                [[0.5, 0.0, 1.0, 0.5], [0.0, 0.5, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0]],
            ]
        )
        rcnn_boxes = np.array(
            [
                [
                    [0.0, 0.0, 0.5, 0.5] + [0.0, 0.0, 0.0, 0.0] * 3,
                    [-1.0, -1.0, -1.0, -1.0] * 4,
                    [0.0, 0.0, 0.0, 0.0] * 2 + [0.5, 0.0, 1.0, 1.0] + [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 1.0, 0.5] + [0.0, 0.0, 0.0, 0.0] * 2,
                    [0.0, 0.0, 0.0, 0.0] * 3 + [0.0, 0.5, 0.5, 1.0],
                    [-1.0, -1.0, -1.0, -1.0] * 4,
                ],
            ]
        )

        outputs = {
            "class_scores": rcnn_scores,
            "boxes": rcnn_boxes,
        }
        labels = {
            "box_targets": rcnn_box_targets,
            "class_targets": rcnn_score_targets,
        }
        model: rcnn_losses.RCNNDetectionMetric = (
            rcnn_losses.RCNNDetectionMetric.default_config()
            .set(name="frcnn_loss", num_classes=4)
            .instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        loss, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(outputs=outputs, labels=labels, paddings=paddings),
        )
        np.testing.assert_almost_equal(0.0313 / 3, loss, decimal=4)

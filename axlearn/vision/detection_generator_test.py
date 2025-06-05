# Copyright © 2023 Apple Inc.

"""Tests for detection_generator.py."""
import random

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common.module import functional as F
from axlearn.vision import anchor, detection_generator


class SelectTopKScoresTest(parameterized.TestCase, tf.test.TestCase):
    """Tests select_top_k_scores."""

    def test_select_topk_scores(self):
        pre_nms_num_boxes = 2
        scores_data = [[[0.2, 0.2], [0.1, 0.9], [0.5, 0.1], [0.3, 0.5]]]
        scores_in = jnp.asarray(scores_data, dtype=jnp.float32)
        # pylint: disable-next=protected-access
        top_k_scores, top_k_indices = detection_generator._select_top_k_scores(
            scores_in, pre_nms_num_detections=pre_nms_num_boxes
        )
        expected_top_k_scores = np.array([[[0.5, 0.9], [0.3, 0.5]]], dtype=np.float32)
        expected_top_k_indices = [[[2, 1], [3, 3]]]
        self.assertAllEqual(top_k_scores, expected_top_k_scores)
        self.assertAllEqual(top_k_indices, expected_top_k_indices)


class MultilevelDetectionGeneratorTest(parameterized.TestCase, tf.test.TestCase):
    """Tests MultilevelDetectionGenerator."""

    @staticmethod
    def _get_multilevel_detection_generator_inputs():
        random.seed(123)
        np.random.seed(123)

        min_level = 5
        max_level = 6
        num_scales = 2
        aspect_ratios = (1.0, 2.0)
        anchor_scale = 2.0
        output_size = (64, 64)
        num_classes = 4

        input_anchor = anchor.AnchorGenerator(
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_scale,
        )
        anchor_boxes = input_anchor(output_size)
        anchor_boxes = {k: np.asarray(v) for k, v in anchor_boxes.items()}

        cls_outputs_all = (np.random.rand(20, num_classes) - 0.5) * 3
        box_outputs_all = np.random.rand(20, 4)

        class_output_dim = num_scales * len(aspect_ratios) * num_classes
        class_outputs = {
            5: jnp.reshape(
                jnp.asarray(cls_outputs_all[0:16], dtype=jnp.float32),
                [1, 2, 2, class_output_dim],
            ),
            6: jnp.reshape(
                jnp.asarray(cls_outputs_all[16:20], dtype=jnp.float32),
                [1, 1, 1, class_output_dim],
            ),
        }
        box_output_dim = num_scales * len(aspect_ratios) * 4
        box_outputs = {
            5: jnp.reshape(
                jnp.asarray(box_outputs_all[0:16], dtype=jnp.float32), [1, 2, 2, box_output_dim]
            ),
            6: jnp.reshape(
                jnp.asarray(box_outputs_all[16:20], dtype=jnp.float32), [1, 1, 1, box_output_dim]
            ),
        }
        image_info = jnp.asarray(
            [[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]], dtype=jnp.float32
        )
        return (box_outputs, class_outputs, anchor_boxes, image_info[:, 1, :])

    def test_detections_output_shape(self):
        max_num_detections = 10
        pre_nms_top_k = 5000
        pre_nms_score_threshold = 0.01
        batch_size = 1

        cfg = detection_generator.MultilevelDetectionGenerator.default_config().set(
            name="test",
            apply_nms=True,
            pre_nms_top_k=pre_nms_top_k,
            pre_nms_score_threshold=pre_nms_score_threshold,
            nms_iou_threshold=0.5,
            max_num_detections=max_num_detections,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=self._get_multilevel_detection_generator_inputs(),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        boxes = outputs["detection_boxes"]
        classes = outputs["detection_classes"]
        scores = outputs["detection_scores"]
        valid_detections = outputs["num_detections"]

        self.assertEqual(boxes.shape, (batch_size, max_num_detections, 4))
        self.assertEqual(scores.shape, (batch_size, max_num_detections))
        self.assertEqual(classes.shape, (batch_size, max_num_detections))
        self.assertEqual(valid_detections.shape, (batch_size,))

    @parameterized.parameters(
        (True, 0.0, 100.0),
        (False, -float("inf"), float("inf")),
    )
    def test_detections_clip_boxes(
        self, clip_boxes: bool, min_box_value: float, max_box_value: float
    ):
        cfg = detection_generator.MultilevelDetectionGenerator.default_config().set(
            name="test",
            max_num_detections=10,
            clip_boxes=clip_boxes,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=self._get_multilevel_detection_generator_inputs(),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )

        self.assertAllGreaterEqual(outputs["detection_boxes"], min_box_value)
        self.assertAllLessEqual(outputs["detection_boxes"], max_box_value)

    @parameterized.parameters(
        (True, jnp.array([[3, 2, 2, 1, 3, 1, 2, 1, 3, 2]])),
        (False, jnp.array([[3, 2, 2, 1, 3, 0, 1, 0, 2, 1]])),
    )
    def test_detections_ignore_first_class(
        self, ignore_first_class: bool, detection_classes: jnp.ndarray
    ):
        cfg = detection_generator.MultilevelDetectionGenerator.default_config().set(
            name="test",
            max_num_detections=10,
            ignore_first_class=ignore_first_class,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=self._get_multilevel_detection_generator_inputs(),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )

        self.assertAllEqual(outputs["detection_classes"], detection_classes)


def _random_logits_for_single_fg_class(batch, num_detections, num_classes):
    """Returns random logits that contain a single dominant class under softmax normalization."""
    return jnp.where(
        jax.nn.one_hot(
            np.random.randint(low=1, high=num_classes, size=(batch, num_detections)),
            num_classes=num_classes,
        ),
        -500.0,
        -1000.0,
    )


def _assert_scores_allclose(scores, valid_detections, target, rtol=1e-6, atol=1e-3):
    """Asserts valid scores are equal to target scaler."""
    for scores_i, valid_detections_i in zip(scores, valid_detections):
        np.testing.assert_allclose(scores_i[:valid_detections_i], target, rtol=rtol, atol=atol)


class DetectionGeneratorTest(absltest.TestCase):
    """Tests DetectionGenerator."""

    def test_detections_with_nms(self):
        num_detections = 20
        max_num_detections = 10
        num_classes = 4
        pre_nms_top_k = 5000
        pre_nms_score_threshold = 0.01
        batch_size = 1

        class_outputs = _random_logits_for_single_fg_class(batch_size, num_detections, num_classes)
        box_outputs = np.random.rand(1, num_detections, 4)
        anchor_boxes = np.random.rand(1, num_detections, 4)
        image_info = jnp.asarray(
            [[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]], dtype=jnp.float32
        )

        cfg = detection_generator.DetectionGenerator.default_config().set(
            name="test",
            apply_nms=True,
            pre_nms_top_k=pre_nms_top_k,
            pre_nms_score_threshold=pre_nms_score_threshold,
            nms_iou_threshold=0.5,
            max_num_detections=max_num_detections,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=(box_outputs, class_outputs, anchor_boxes, image_info[:, 1, :]),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        boxes = outputs["detection_boxes"]
        classes = outputs["detection_classes"]
        scores = outputs["detection_scores"]
        valid_detections = outputs["num_detections"]
        self.assertEqual(boxes.shape, (batch_size, max_num_detections, 4))
        self.assertEqual(scores.shape, (batch_size, max_num_detections))
        self.assertEqual(classes.shape, (batch_size, max_num_detections))
        self.assertEqual(valid_detections.shape, (batch_size,))
        _assert_scores_allclose(scores, valid_detections, 1.0)

    def test_per_class_detections_with_nms(self):
        num_detections = 20
        max_num_detections = 10
        num_classes = 4
        pre_nms_top_k = 5000
        pre_nms_score_threshold = 0.01
        batch_size = 1

        class_outputs = _random_logits_for_single_fg_class(batch_size, num_detections, num_classes)
        box_outputs = np.random.rand(1, num_detections, num_classes * 4)
        anchor_boxes = np.random.rand(1, num_detections, 4)
        image_info = jnp.asarray(
            [[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]], dtype=jnp.float32
        )

        cfg = detection_generator.DetectionGenerator.default_config().set(
            name="test",
            apply_nms=True,
            pre_nms_top_k=pre_nms_top_k,
            pre_nms_score_threshold=pre_nms_score_threshold,
            nms_iou_threshold=0.5,
            max_num_detections=max_num_detections,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=(box_outputs, class_outputs, anchor_boxes, image_info[:, 1, :]),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        boxes = outputs["detection_boxes"]
        classes = outputs["detection_classes"]
        scores = outputs["detection_scores"]
        valid_detections = outputs["num_detections"]

        self.assertEqual(boxes.shape, (batch_size, max_num_detections, 4))
        self.assertEqual(scores.shape, (batch_size, max_num_detections))
        self.assertEqual(classes.shape, (batch_size, max_num_detections))
        self.assertEqual(valid_detections.shape, (batch_size,))
        _assert_scores_allclose(scores, valid_detections, 1.0)

    def test_detections_without_nms(self):
        num_detections = 20
        num_classes = 4
        batch_size = 1

        class_outputs = _random_logits_for_single_fg_class(batch_size, num_detections, num_classes)
        box_outputs = np.random.rand(1, num_detections, 4)
        anchor_boxes = np.random.rand(1, num_detections, 4)
        image_info = jnp.asarray(
            [[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]], dtype=jnp.float32
        )

        cfg = detection_generator.DetectionGenerator.default_config().set(
            name="test",
            apply_nms=False,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=(box_outputs, class_outputs, anchor_boxes, image_info[:, 1, :]),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        boxes = outputs["detection_boxes"]
        scores = outputs["detection_scores"]
        classes = outputs["detection_classes"]
        valid_detections = outputs["num_detections"]

        self.assertEqual(boxes.shape, (batch_size, 20, 4))
        self.assertEqual(scores.shape, (batch_size, 20))
        self.assertEqual(classes.shape, (batch_size, 20))
        self.assertEqual(valid_detections.shape, (batch_size,))
        _assert_scores_allclose(scores, valid_detections, 1.0)


if __name__ == "__main__":
    absltest.main()

# Copyright © 2023 Apple Inc.

"""Tests for detection utils."""
import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.vision import utils_detection


# pylint: disable=no-self-use
class BoxUtilsTest(parameterized.TestCase, tf.test.TestCase):
    """Tests box utils."""

    @parameterized.parameters((50, [0.5, 0.5], [0, 0], [0, 0]))
    def test_resize_and_crop_boxes(self, num_boxes, image_scale, output_size, offset):
        boxes = tf.convert_to_tensor(np.random.rand(num_boxes, 4))
        processed_boxes = utils_detection.resize_and_crop_boxes(
            boxes,
            tf.cast(image_scale, tf.double),
            output_size,
            tf.cast(offset, tf.double),
        )
        processed_boxes_shape = tf.shape(processed_boxes)
        self.assertAllEqual([num_boxes, 4], processed_boxes_shape.numpy())
        self.assertAllEqual(tf.math.reduce_sum(processed_boxes), tf.convert_to_tensor(0))

    def test_clip_boxes(self):
        image_size = 224
        boxes = tf.constant([[-1.0, -1.0, 30.0, 30.0], [200.0, 200.0, 230.0, 230.0]])
        clipped_boxes = utils_detection.clip_boxes(boxes, [image_size, image_size])
        self.assertTrue(tf.math.reduce_min(clipped_boxes) >= 0.0)
        self.assertTrue(tf.math.reduce_max(clipped_boxes) <= image_size)

    def test_normalize_denormalize_boxes(self):
        image_size = 100
        boxes = tf.constant([[0.0, 0.0, 0.1, 0.1], [0.9, 0.9, 1.0, 1.0]])
        denormalized_boxes = utils_detection.denormalize_boxes_tf(boxes, [image_size, image_size])
        self.assertAllClose(denormalized_boxes, boxes * image_size)
        normalized_boxes = utils_detection.normalize_boxes_tf(
            denormalized_boxes, [image_size, image_size]
        )
        self.assertAllClose(normalized_boxes, boxes)

    def test_normalize_boxes_jax_tf_parity(self):
        image_size = 100
        boxes = tf.constant([[0.0, 0.0, 10.0, 10.0], [90.0, 90.0, 10.0, 10.0]])
        tf_normalized_boxes = utils_detection.normalize_boxes_tf(boxes, [image_size, image_size])
        jax_normalized_boxes = utils_detection.normalize_boxes(
            boxes=boxes.numpy(), image_shape=np.array([[image_size, image_size]])
        )
        np.testing.assert_allclose(jax_normalized_boxes, tf_normalized_boxes)

    def test_denormalize_boxes_jax_tf_parity(self):
        image_size = 100
        boxes = tf.constant([[0.0, 0.0, 0.1, 0.1], [0.9, 0.9, 1.0, 1.0]])
        tf_denormalized_boxes = utils_detection.denormalize_boxes_tf(
            boxes, [image_size, image_size]
        )
        jax_denormalized_boxes = utils_detection.denormalize_boxes(
            boxes=boxes.numpy(), image_shape=np.array([[image_size, image_size]])
        )
        np.testing.assert_allclose(jax_denormalized_boxes, tf_denormalized_boxes)

    def test_get_non_empty_box_indices(self):
        boxes = tf.constant(
            [[1.0, 30.0, 30.0, 1.0], [30.0, 1.0, 1.0, 30.0], [1.0, 1.0, 30.0, 30.0]]
        )
        non_empty_indices = utils_detection.get_non_empty_box_indices(boxes)
        self.assertAllEqual([2], non_empty_indices.numpy())

    @parameterized.parameters([False, True])
    def test_multi_level_flatten(self, keep_last_dim):
        batch_size = 8
        min_level = 3
        max_level = 7
        multi_level_inputs = {
            l: np.random.uniform(-1, 1, size=[batch_size, 128 // 2**l, 4])
            for l in range(min_level, max_level + 1)
        }
        flattened_inputs = utils_detection.multi_level_flatten(
            multi_level_inputs, last_dim=4 if keep_last_dim else None
        )
        if keep_last_dim:
            expected_shape = (8, sum(128 // 2**l for l in range(min_level, max_level + 1)), 4)
        else:
            expected_shape = (8, sum(128 // 2**l for l in range(min_level, max_level + 1)) * 4)
        self.assertEqual(flattened_inputs.shape, expected_shape)

    @parameterized.parameters([1, 5, 10])
    def test_clip_or_pad_to_fixed_size(self, size):
        padded_size = 30
        inputs = tf.convert_to_tensor(np.random.uniform(-1, 1, size=[size, 4]))
        padded_inputs = utils_detection.clip_or_pad_to_fixed_size(inputs, size=padded_size)
        expected_shape = [padded_size, 4]
        self.assertEqual(padded_inputs.shape, expected_shape)

    def test_yxyx_to_xywh(self):
        boxes = np.asarray([[10.0, 25.0, 20.0, 25.0], [20.0, 40.0, 30.0, 50.0]])
        outputs = utils_detection.yxyx_to_xywh(boxes)
        expected_outputs = [[25.0, 10.0, 0.0, 10.0], [40.0, 20.0, 10.0, 10.0]]
        self.assertAllClose(outputs, expected_outputs)

    @parameterized.parameters(
        (
            utils_detection.BoxFormat.YminXminYmaxXmax,
            utils_detection.BoxFormat.XminYminWH,
            [[25.0, 10.0, 0.0, 10.0], [40.0, 20.0, 10.0, 10.0]],
        ),
        (
            utils_detection.BoxFormat.YminXminYmaxXmax,
            utils_detection.BoxFormat.XminYminXmaxYmax,
            [[25.0, 10.0, 25.0, 20.0], [40.0, 20.0, 50.0, 30.0]],
        ),
        (
            utils_detection.BoxFormat.XminYminWH,
            utils_detection.BoxFormat.YminXminHW,
            [[25.0, 10.0, 25.0, 20.0], [40.0, 20.0, 50.0, 30.0]],
        ),
        (
            utils_detection.BoxFormat.YminXminYmaxXmax,
            utils_detection.BoxFormat.CxCyWH,
            [[25.0, 15.0, 0.0, 10.0], [45.0, 25.0, 10.0, 10.0]],
        ),
        (
            utils_detection.BoxFormat.CxCyWH,
            utils_detection.BoxFormat.CxCyWH,
            [[10.0, 25.0, 20.0, 25.0], [20.0, 40.0, 30.0, 50.0]],
        ),
    )
    def test_transform_boxes(self, source_format, target_format, expected_outputs):
        boxes = np.asarray([[10.0, 25.0, 20.0, 25.0], [20.0, 40.0, 30.0, 50.0]])
        outputs = utils_detection.transform_boxes(boxes, source_format, target_format)
        self.assertAllClose(outputs, expected_outputs)

    def test_reshape_box_decorator(self):
        # (batch_size, num_locations, num_anchors, box coordinates)
        orig_boxes = np.random.normal(size=(2, 12, 9, 4))

        # (batch_size, num_locations, num_anchors * box coordinates)
        boxes = orig_boxes.reshape((2, 12, -1))

        outputs = utils_detection.reshape_box_decorator(utils_detection.transform_boxes)(
            boxes,
            source_format=utils_detection.BoxFormat.YminXminYmaxXmax,
            target_format=utils_detection.BoxFormat.XminYminXmaxYmax,
        )
        a, b, c, d = np.moveaxis(orig_boxes, -1, 0)
        expected_outputs = np.stack((b, a, d, c), axis=-1).reshape((2, 12, -1))

        self.assertAllClose(outputs, expected_outputs)


class IoUSimilarityTest(tf.test.TestCase):
    """Tests IoUSimilarity."""

    def test_similarity_unbatched(self):
        boxes = tf.constant(
            [
                [0, 0, 1, 1],
                [5, 0, 10, 5],
            ],
            dtype=tf.float32,
        )
        gt_boxes = tf.constant(
            [
                [0, 0, 5, 5],
                [0, 5, 5, 10],
                [5, 0, 10, 5],
                [5, 5, 10, 10],
            ],
            dtype=tf.float32,
        )
        sim_calc = utils_detection.IouSimilarity()
        sim_matrix = sim_calc(boxes, gt_boxes)
        self.assertAllClose(sim_matrix.numpy(), [[0.04, 0, 0, 0], [0, 0, 1.0, 0]])

    def test_similarity_batched(self):
        boxes = tf.constant(
            [
                [
                    [0, 0, 1, 1],
                    [5, 0, 10, 5],
                ]
            ],
            dtype=tf.float32,
        )
        gt_boxes = tf.constant(
            [
                [
                    [0, 0, 5, 5],
                    [0, 5, 5, 10],
                    [5, 0, 10, 5],
                    [5, 5, 10, 10],
                ]
            ],
            dtype=tf.float32,
        )
        sim_calc = utils_detection.IouSimilarity()
        sim_matrix = sim_calc(boxes, gt_boxes)
        self.assertAllClose(sim_matrix.numpy(), [[[0.04, 0, 0, 0], [0, 0, 1.0, 0]]])


class TargetGatherTest(tf.test.TestCase):
    """Tests TargetGather."""

    def test_target_gather_batched(self):
        gt_boxes = tf.constant(
            [
                [
                    [0, 0, 5, 5],
                    [0, 5, 5, 10],
                    [5, 0, 10, 5],
                    [5, 5, 10, 10],
                ]
            ],
            dtype=tf.float32,
        )
        gt_classes = tf.constant([[[2], [10], [3], [-1]]], dtype=tf.int32)

        labeler = utils_detection.TargetGather()

        match_indices = tf.constant([[0, 2]], dtype=tf.int32)
        match_indicators = tf.constant([[-2, 1]])
        mask = tf.less_equal(match_indicators, 0)
        cls_mask = tf.expand_dims(mask, -1)
        matched_gt_classes = labeler(gt_classes, match_indices, cls_mask)
        box_mask = tf.tile(cls_mask, [1, 1, 4])
        matched_gt_boxes = labeler(gt_boxes, match_indices, box_mask)

        self.assertAllEqual(matched_gt_classes.numpy(), [[[0], [3]]])
        self.assertAllClose(matched_gt_boxes.numpy(), [[[0, 0, 0, 0], [5, 0, 10, 5]]])

    def test_target_gather_unbatched(self):
        gt_boxes = tf.constant(
            [
                [0, 0, 5, 5],
                [0, 5, 5, 10],
                [5, 0, 10, 5],
                [5, 5, 10, 10],
            ],
            dtype=tf.float32,
        )
        gt_classes = tf.constant([[2], [10], [3], [-1]], dtype=tf.int32)

        labeler = utils_detection.TargetGather()

        match_indices = tf.constant([0, 2], dtype=tf.int32)
        match_indicators = tf.constant([-2, 1])
        mask = tf.less_equal(match_indicators, 0)
        cls_mask = tf.expand_dims(mask, -1)
        matched_gt_classes = labeler(gt_classes, match_indices, cls_mask)
        box_mask = tf.tile(cls_mask, [1, 4])
        matched_gt_boxes = labeler(gt_boxes, match_indices, box_mask)

        self.assertAllEqual(matched_gt_classes.numpy(), [[0], [3]])
        self.assertAllClose(matched_gt_boxes.numpy(), [[0, 0, 0, 0], [5, 0, 10, 5]])


class BoxMatcherTest(tf.test.TestCase):
    """Tests BoxMatcher."""

    def test_box_matcher_unbatched(self):
        sim_matrix = tf.constant([[0.04, 0, 0, 0], [0, 0, 1.0, 0]], dtype=tf.float32)

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = utils_detection.BoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold], indicators=[-3, -2, -1, 1]
        )
        match_indices, match_indicators = matcher(sim_matrix)
        positive_matches = tf.greater_equal(match_indicators, 0)
        negative_matches = tf.equal(match_indicators, -2)

        self.assertAllEqual(positive_matches.numpy(), [False, True])
        self.assertAllEqual(negative_matches.numpy(), [True, False])
        self.assertAllEqual(match_indices.numpy(), [0, 2])
        self.assertAllEqual(match_indicators.numpy(), [-2, 1])

    def test_box_matcher_batched(self):
        sim_matrix = tf.constant([[[0.04, 0, 0, 0], [0, 0, 1.0, 0]]], dtype=tf.float32)

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = utils_detection.BoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold], indicators=[-3, -2, -1, 1]
        )
        match_indices, match_indicators = matcher(sim_matrix)
        positive_matches = tf.greater_equal(match_indicators, 0)
        negative_matches = tf.equal(match_indicators, -2)

        self.assertAllEqual(positive_matches.numpy(), [[False, True]])
        self.assertAllEqual(negative_matches.numpy(), [[True, False]])
        self.assertAllEqual(match_indices.numpy(), [[0, 2]])
        self.assertAllEqual(match_indicators.numpy(), [[-2, 1]])


# pylint: disable=no-self-use
class BoxNormalizerDenormalizerTest(parameterized.TestCase, absltest.TestCase):
    """Tests box normalize/denormalize."""

    @parameterized.named_parameters(
        {
            "testcase_name": "per_box_image_shape",
            "denormalized_boxes": np.array(
                [
                    [[25, 25, 75, 75], [0, 25, 50, 100], [50, 50, 100, 100], [50, 0, 100, 50]],
                ]
            ),
            "normalized_boxes": np.array(
                [
                    [
                        [0.25, 0.25, 0.75, 0.75],
                        [0.0, 0.25, 0.5, 1.0],
                        [0.25, 0.25, 0.5, 0.5],
                        [0.25, 0, 0.5, 0.25],
                    ],
                ]
            ),
            "image_shape": np.array([[100, 100], [100, 100], [200, 200], [200, 200]]),
        },
        {
            "testcase_name": "per_batch_image_shape",
            "denormalized_boxes": np.array(
                [
                    [
                        [25, 25, 75, 75],
                        [0, 25, 50, 100],
                    ],
                    [[50, 50, 100, 100], [50, 0, 100, 50]],
                ]
            ),
            "image_shape": np.array([[100, 100], [200, 200]])[:, None, :],
            "normalized_boxes": np.array(
                [
                    [[0.25, 0.25, 0.75, 0.75], [0.0, 0.25, 0.5, 1.0]],
                    [[0.25, 0.25, 0.5, 0.5], [0.25, 0, 0.5, 0.25]],
                ]
            ),
        },
        {
            "testcase_name": "global_image_shape",
            "denormalized_boxes": np.array(
                [
                    [
                        [25, 25, 75, 75],
                        [0, 25, 50, 100],
                    ],
                    [[50, 50, 100, 100], [50, 0, 100, 50]],
                ]
            ),
            "image_shape": np.array([100, 100])[None, None, :],
            "normalized_boxes": np.array(
                [
                    [[0.25, 0.25, 0.75, 0.75], [0.0, 0.25, 0.5, 1.0]],
                    [[0.5, 0.5, 1.0, 1.0], [0.5, 0, 1.0, 0.5]],
                ]
            ),
        },
    )
    def test_roundtrip(self, denormalized_boxes, normalized_boxes, image_shape):
        np.testing.assert_allclose(
            utils_detection.normalize_boxes(boxes=denormalized_boxes, image_shape=image_shape),
            normalized_boxes,
        )
        np.testing.assert_allclose(
            utils_detection.denormalize_boxes(boxes=normalized_boxes, image_shape=image_shape),
            denormalized_boxes,
        )


# pylint: disable=no-self-use
class SortedTopKTest(absltest.TestCase):
    """Tests sorted_top_k."""

    def test_1d(self):
        x = np.array([5.0, 1.0, 6.0, 3.0, -1.0])
        output = utils_detection.sorted_top_k(x, k=3)
        self.assertEqual(list(output), [2, 0, 3])

    def test_2d(self):
        row1 = np.arange(5)
        row2 = np.arange(5)
        np.random.shuffle(row1)
        np.random.shuffle(row2)
        x = np.stack([row1, row2])
        output = utils_detection.sorted_top_k(x, k=3)
        np.testing.assert_array_equal(
            np.take_along_axis(x, output, axis=-1), [[4, 3, 2], [4, 3, 2]]
        )


if __name__ == "__main__":
    absltest.main()

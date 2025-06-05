# Copyright © 2023 Apple Inc.

"""Tests NMS utils."""
import jax.numpy as jnp
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.vision import nms


class NMSTest(parameterized.TestCase, tf.test.TestCase):
    def test_nms(self):
        scores = jnp.asarray([0.9, 0.8, 0.7])
        boxes = jnp.asarray(
            [
                [1.0, 10.0, 2, 20.0],  # will be kept.
                [1.1, 10.1, 2.1, 20.1],  # will be suppressed by the first box.
                [30.0, 50.0, 40.0, 60.0],  # will be kept.
            ]
        )
        max_output_size = 3
        iou_threshold = 0.5

        nmsed_scores, nmsed_boxes = nms.non_max_suppression_padded(
            jnp.expand_dims(scores, axis=0),
            jnp.expand_dims(boxes, axis=0),
            max_output_size,
            iou_threshold,
        )
        expected_scores = [[0.9, 0.7, 0.0]]
        expected_boxes = [
            [
                [1.0, 10.0, 2, 20.0],
                [30.0, 50.0, 40.0, 60.0],
                [0.0, 0.0, 0.0, 0.0],  # padded box.
            ]
        ]
        self.assertAllClose(nmsed_scores, expected_scores)
        self.assertAllClose(nmsed_boxes, expected_boxes)


if __name__ == "__main__":
    absltest.main()

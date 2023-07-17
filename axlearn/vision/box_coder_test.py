# Copyright © 2023 Apple Inc.

"""Tests box coder implementations."""
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.vision import box_coder


# pylint: disable=no-self-use
class BoxCoderTest(parameterized.TestCase, absltest.TestCase):
    """Tests BoxCoder."""

    @parameterized.named_parameters(
        {"testcase_name": "default_weights", "weights": None},
        {"testcase_name": "faster_rcnn_weights", "weights": (10.0, 10.0, 5.0, 5.0)},
    )
    def test_encode_and_decode_boxes(self, weights):
        coder_cfg = box_coder.BoxCoder.default_config()
        if weights:
            coder_cfg = coder_cfg.set(weights=weights)
        coder = coder_cfg.instantiate()
        boxes = jnp.asarray([[10.0, 25.0, 20.0, 25.0], [20.0, 40.0, 30.0, 50.0]])
        anchors = jnp.asarray([[12.0, 23.0, 22.0, 27.0], [22.0, 38.0, 33.0, 47.0]])
        encoded_boxes = coder.encode(boxes=boxes, anchors=anchors)
        decoded_boxes = coder.decode(encoded_boxes=encoded_boxes, anchors=anchors)
        np.testing.assert_allclose(boxes, decoded_boxes)

    def test_encoded_zero_area_boxes_are_finite(self):
        boxes = jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-2.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 2.0],
                [-1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        anchors = jnp.asarray(
            [
                [12.0, 23.0, 22.0, 27.0],
                [22.0, 38.0, 33.0, 47.0],
                [25.0, 50.0, 35.0, 97.0],
                [22.0, 38.0, 33.0, 47.0],
                [12.0, 23.0, 22.0, 27.0],
            ]
        )
        coder = box_coder.BoxCoder.default_config().instantiate()
        encoded_boxes = coder.encode(boxes=boxes, anchors=anchors)
        self.assertTrue(jnp.isfinite(encoded_boxes).all())
        decoded_boxes = coder.decode(encoded_boxes=encoded_boxes, anchors=anchors)
        np.testing.assert_allclose(boxes, decoded_boxes, rtol=1e-6, atol=1e-6)

    @parameterized.parameters(
        (
            box_coder.BoxClipMethod.MaxHW,
            [[-195.50003, 314.5, 429.50003, 439.5], [1193.0, 130.49997, 1943.0, 755.5]],
        ),
        (
            box_coder.BoxClipMethod.MinMaxYXHW,
            [[-195.50003, 187.5, 429.50003, 312.5], [502.99997, 130.49997, 1253.0, 755.5]],
        ),
    )
    def test_clip_boxes(self, box_clip_method, expected_output):
        coder_cfg = box_coder.BoxCoder.default_config().set(clip_boxes=box_clip_method)
        coder = coder_cfg.instantiate()
        encoded_boxes = jnp.asarray([[10.0, 126.0, 20.0, 126.0], [120.0, 40.0, 130.0, 50.0]])
        anchors = jnp.asarray([[12.0, 124.0, 22.0, 126.0], [122.0, 38.0, 134.0, 48.0]])
        decoded_boxes = coder.decode(encoded_boxes=encoded_boxes, anchors=anchors)
        np.testing.assert_allclose(
            jnp.asarray(expected_output),
            decoded_boxes,
        )

    @parameterized.parameters(
        (
            0.0,
            [[-0.5, -1.0, -20.723267, -0.693147]],
        ),
        (
            1.0,
            [[-0.45, -1.0, -2.302585, -0.693147]],
        ),
    )
    def test_box_wh_min_value(self, box_wh_min_value, expected_output):
        boxes = jnp.asarray(
            [
                [12.0, 20.0, 12.0, 22.0],
            ]
        )
        anchors = jnp.asarray(
            [
                [12.0, 23.0, 22.0, 27.0],
            ]
        )
        coder = (
            box_coder.BoxCoder.default_config()
            .set(
                box_wh_min_value=box_wh_min_value,
            )
            .instantiate()
        )
        encoded_boxes = coder.encode(boxes=boxes, anchors=anchors)
        np.testing.assert_allclose(
            jnp.asarray(expected_output), encoded_boxes, rtol=1e-6, atol=1e-6
        )

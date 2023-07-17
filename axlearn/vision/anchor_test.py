# Copyright © 2023 Apple Inc.

"""Tests for anchor ops."""
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.vision import anchor, matchers


# pylint: disable=no-self-use
class AnchorLabelerTest(parameterized.TestCase, absltest.TestCase):
    @parameterized.parameters(
        # Single-label scenario
        (
            [[5, 3]],
            [[5, 5, 3, -1, -1, 3, -1, 5, -1]],
        ),
        # Multi-label scenario
        (
            [[[5, 2, -1, -1], [3, -1, -1, -1]]],
            [
                [
                    [5, 2, -1, -1],
                    [5, 2, -1, -1],
                    [3, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [3, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [5, 2, -1, -1],
                    [-1, -1, -1, -1],
                ]
            ],
        ),
    )
    def test_foreground_background_labeling(self, groundtruth_classes, expected_classes):
        anchor_level_boxes = {
            "2": np.array(
                [
                    [0.0, 0.0, 0.5, 0.5],  # 0.5 IoU,  groundtruth 0
                    [0.0, 0.5, 0.5, 1.0],  # 0.5 IoU,  groundtruth 0
                    [0.5, 0.0, 1.0, 0.5],  # 0.5 IoU,  groundtruth 1
                    [0.55, 0.0, 1.0, 0.5],  # 0.45 IoU, groundtruth 1
                    [0.5, 0.5, 1.0, 1.0],  # 0.0 IoU,  None
                ],
            ),
            "3": np.array(
                [
                    [0.0, 0.0, 1.0, 0.5],  # 1.0 IoU,  groundtruth 1
                    [0.0, 0.5, 1.0, 1.0],  # 1/3 IoU,  groundtruth 0
                    [0.0, 0.0, 0.5, 1.0],  # 1.0 IoU,  groundtruth 0
                    [0.5, 0.0, 1.0, 1.0],  # 1/3 IoU,  groundtruth 1
                ]
            ),
        }
        groundtruth_boxes = np.array([[[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 1.0, 0.5]]])

        anchor_labeler = anchor.AnchorLabeler(
            matchers.ArgmaxMatcher.default_config().set(thresholds=[0.4, 0.5])
        )
        anchor_labels = anchor_labeler(
            per_level_anchor_boxes=anchor_level_boxes,
            groundtruth_boxes=groundtruth_boxes,
            groundtruth_classes=np.array(groundtruth_classes),
        )
        np.testing.assert_array_equal(expected_classes, anchor_labels.groundtruth_classes)
        np.testing.assert_array_equal(
            [[False, False, False, True, False, False, False, False, False]],
            anchor_labels.class_paddings,
        )
        np.testing.assert_array_almost_equal(
            [
                [
                    [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ],
            anchor_labels.groundtruth_boxes,
            decimal=2,
        )
        np.testing.assert_array_equal(
            [[False, False, False, True, True, False, True, False, True]],
            anchor_labels.box_paddings,
        )


# pylint: disable=no-self-use
class AnchorGeneratorTest(parameterized.TestCase, absltest.TestCase):
    @parameterized.parameters(
        # Single scale anchor.
        (
            5,
            5,
            1,
            [1.0],
            2.0,
            [[-16, -16, 48, 48], [-16, 16, 48, 80], [16, -16, 80, 48], [16, 16, 80, 80]],
        ),
        # Multi scale anchor.
        (
            5,
            6,
            1,
            [1.0],
            2.0,
            [
                [-16, -16, 48, 48],
                [-16, 16, 48, 80],
                [16, -16, 80, 48],
                [16, 16, 80, 80],
                [-32, -32, 96, 96],
            ],
        ),
        # # Multi aspect ratio anchor.
        (
            6,
            6,
            1,
            [1.0, 4.0, 0.25],
            2.0,
            [[-32, -32, 96, 96], [-0, -96, 64, 160], [-96, -0, 160, 64]],
        ),
    )
    def testAnchorGenerationWithImageSizeAsTensor(
        self, min_level, max_level, num_scales, aspect_ratios, anchor_size, expected_boxes
    ):
        image_size = (64, 64)
        anchor_dict = anchor.AnchorGenerator(
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_size,
        )(image_size=image_size)
        anchor_boxes = np.concatenate(list(anchor_dict.values()), axis=0)
        np.testing.assert_allclose(expected_boxes, anchor_boxes)


if __name__ == "__main__":
    absltest.main()

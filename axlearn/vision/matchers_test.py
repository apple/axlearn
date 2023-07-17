# Copyright © 2023 Apple Inc.

"""Tests matchers."""
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.vision import matchers


# pylint: disable=no-self-use
class ArgmaxMatcherTest(absltest.TestCase):
    """Tests ArgmaxMatcher."""

    def test_only_high_quality_matches(self):
        per_level_anchor_boxes = {
            "all": np.array(
                [
                    [0, 0, 1, 0.3],  # 0.04 IoU, groundtruth 0; 0.001 IoU, groundtruth 1
                    [0, 0.5, 1, 1],  # 1.0 IoU, groundtruth 2; 0.42 IoU, groundtruth 3
                ]
            )
        }
        groundtruth_boxes = np.array(
            [
                [[0, 0.26, 1, 1], [0, 0, 0, 0], [0, 0.5, 1, 1], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0.02, 0.02], [0, 0, 0, 0], [0, 0.3, 1, 0.8]],
            ]
        )

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.01

        matcher_cfg = matchers.ArgmaxMatcher.default_config()
        matcher_cfg.thresholds = [bg_thresh_lo, bg_thresh_hi, fg_threshold]
        matcher_cfg.labels = [-2, -1, 0, 1]
        matcher_cfg.force_match_columns = False
        matcher = matcher_cfg.instantiate()

        match_results = matcher.match(
            per_level_anchor_boxes=per_level_anchor_boxes, groundtruth_boxes=groundtruth_boxes
        )
        np.testing.assert_array_equal([[0, 2], [1, 3]], match_results.matches)
        np.testing.assert_array_equal([[-1, 1], [-2, 0]], match_results.labels)

    def test_allow_low_quality_matches(self):
        per_level_anchor_boxes = {
            "all": np.array(
                [
                    [0, 0.5, 0.5, 0.85],  # Positive match with groundtruth 1, 0.7 IoU.
                    [0, 0, 0.4, 0.5],  # Positive match with groundtruth 0, 0.8 IoU.
                    [0.5, 0.5, 0.55, 1],  # Forced Positive match with groundtruth 3, 0.1 IoU.
                    [0, 0.5, 0.1, 1],  # Negative match with groundtruth 1, 0.2 IoU.
                    [0.5, 0, 0.75, 0.5],  # Forced Positive match with groundtruth 2, 0.5 IoU.
                    [0, 0.5, 0.2, 1],  # Ignore match with groundtruth 1, 0.4 IoU.
                    [0, 0.5, 0.1, 1],  # Negative match with groundtruth 1, 0.2 IoU.
                    [0, 0, 0.05, 0.5],  # Negative match with groundtruth 0, 0.1 IoU.
                ]
            )
        }
        groundtruth_boxes = np.array(
            [
                [[0, 0, 0.5, 0.5], [0, 0.5, 0.5, 1], [0.5, 0, 1, 0.5], [0.5, 0.5, 1, 1]],
            ]
        )

        fg_threshold = 0.6
        bg_thresh_hi = 0.3
        bg_thresh_lo = 0.0

        matcher_cfg = matchers.ArgmaxMatcher.default_config()
        matcher_cfg.thresholds = [bg_thresh_lo, bg_thresh_hi, fg_threshold]
        matcher_cfg.labels = [-2, -1, 0, 1]
        matcher = matcher_cfg.instantiate()

        match_results = matcher.match(
            per_level_anchor_boxes=per_level_anchor_boxes, groundtruth_boxes=groundtruth_boxes
        )
        np.testing.assert_array_equal([[1, 0, 3, 1, 2, 1, 1, 0]], match_results.matches)
        np.testing.assert_array_equal([[1, 1, 1, -1, 1, 0, -1, -1]], match_results.labels)


class ATSSMatcherTest(parameterized.TestCase, absltest.TestCase):
    """Tests ATSSMatcher."""

    @parameterized.parameters(matchers.DistanceType.L2, matchers.DistanceType.IOU)
    def test_matches(self, mode):
        per_level_anchor_boxes = {
            "2": np.array(
                [
                    [0, 0, 0.5, 0.5],  # Positive match with groundtruth 0
                    [0, 0.5, 0.5, 1],
                    [0.5, 0, 1, 0.5],
                    [0.5, 0.5, 1, 1],
                ]
            ),
            "3": np.array(
                [
                    [0, 0, 1, 0.5],
                    [0, 0, 0.5, 1],  # Positive match with groundtruth 1
                    [0, 0.5, 1, 1],
                    [0.5, 0, 1, 1],
                ]
            ),
        }
        groundtruth_boxes = np.array(
            [
                [
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0.5, 1],
                ],
            ]
        )

        matcher_cfg = matchers.ATSSMatcher.default_config()
        matcher_cfg.top_k = 3
        matcher_cfg.distance_type = mode
        matcher_cfg.labels = [-1, 0, 1]
        matcher = matcher_cfg.instantiate()

        match_results = matcher.match(per_level_anchor_boxes, groundtruth_boxes)
        np.testing.assert_array_equal([[0, 1, 0, 0, 0, 1, 1, 0]], match_results.matches)
        np.testing.assert_array_equal([[1, -1, -1, -1, -1, 1, -1, -1]], match_results.labels)

    def test_labels(self):
        per_level_anchor_boxes = {
            "2": np.array(
                [
                    [0.0, 0.0, 256.0, 256.0],
                    [256.0, 0.0, 512.0, 256.0],
                    [0.0, 256.0, 256.0, 512.0],
                    [256.0, 256.0, 512.0, 512.0],
                ]
            ),
            "3": np.array(
                [
                    [0.0, 0.0, 128.0, 128.0],
                    [128.0, 0.0, 256.0, 128.0],
                    [0.0, 128.0, 128.0, 256.0],
                    [128.0, 128.0, 256.0, 256.0],
                    [0.0, 256.0, 128.0, 384.0],
                    [128.0, 256.0, 256.0, 384.0],
                    [0.0, 384.0, 128.0, 512.0],
                    [128.0, 384.0, 256.0, 512.0],
                    [256.0, 0.0, 384.0, 128.0],
                    [384.0, 0.0, 512.0, 128.0],
                    [256.0, 128.0, 384.0, 256.0],
                    [384.0, 128.0, 512.0, 256.0],
                    [256.0, 256.0, 384.0, 384.0],
                    [384.0, 256.0, 512.0, 384.0],
                    [256.0, 384.0, 384.0, 512.0],
                    [384.0, 384.0, 512.0, 512.0],
                ]
            ),
        }
        groundtruth_boxes = np.array(
            [
                [
                    [26.117289, 17.156782, 406.18396, 228.53214],
                    [349.10214, 123.73946, 386.32037, 151.13103],
                    [381.43216, 133.47852, 406.7317, 157.99454],
                    [138.24228, 73.86143, 159.00032, 90.12147],
                    [125.70181, 70.05465, 143.48924, 84.57509],
                ],
            ]
        )

        matcher_cfg = matchers.ATSSMatcher.default_config()
        matcher_cfg.top_k = 3
        matcher_cfg.distance_type = matchers.DistanceType.IOU
        matcher_cfg.labels = [-1, 0, 1]
        matcher = matcher_cfg.instantiate()

        match_results = matcher.match(per_level_anchor_boxes, groundtruth_boxes)
        np.testing.assert_array_equal(
            [[1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1]],
            match_results.labels,
        )


if __name__ == "__main__":
    absltest.main()

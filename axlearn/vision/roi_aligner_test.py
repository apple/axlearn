# Copyright © 2023 Apple Inc.

"""Tests ROI Align layer."""
import jax
import numpy as np
from absl.testing import absltest

from axlearn.common.module import functional as F
from axlearn.vision.roi_aligner import RoIAligner


# pylint: disable=no-self-use
class RoIAlignerTest(absltest.TestCase):
    """Tests ROIAligner."""

    def test_multilevel_roi_align(self):
        cfg = RoIAligner.default_config().set(
            name="roi_aligner",
            output_size=2,
            align_corners=True,
            min_level=2,
            max_level=5,
            unit_scale_level=4,
            pretraining_image_size=224,
        )
        roi_aligner = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = roi_aligner.initialize_parameters_recursively(init_key)

        image_size = 640
        fpn_min_level = 2
        fpn_max_level = 5
        batch_size = 1
        num_filters = 1
        features = {}
        boxes = np.array(
            [
                [
                    [0, 0, 111, 111],  # Level 2.
                    [0, 0, 113, 113],  # Level 3.
                    [0, 0, 223, 223],  # Level 3.
                    [0, 0, 225, 225],  # Level 4.
                    [0, 0, 449, 449],  # Level 5.
                ],
            ],
            dtype=np.float32,
        )
        for level in range(fpn_min_level, fpn_max_level + 1):
            feat_size = int(image_size / 2**level)
            features[level] = np.zeros(
                [batch_size, feat_size, feat_size, num_filters], dtype=np.float32
            )
            # Set non-zero values to cover projected boxes.
            features[level][:, :30, :30, :] = float(level)

        roi_features, _ = F(
            roi_aligner,
            inputs={"features": features, "boxes": boxes},
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        np.testing.assert_allclose(2 * np.ones((2, 2, 1)), roi_features[0][0])
        np.testing.assert_allclose(3 * np.ones((2, 2, 1)), roi_features[0][1])
        np.testing.assert_allclose(3 * np.ones((2, 2, 1)), roi_features[0][2])
        np.testing.assert_allclose(4 * np.ones((2, 2, 1)), roi_features[0][3])
        np.testing.assert_allclose(5 * np.ones((2, 2, 1)), roi_features[0][4])

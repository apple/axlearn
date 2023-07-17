# Copyright © 2023 Apple Inc.

"""Tests spatial transform ops."""
import numpy as np
import pytest
from absl.testing import absltest

from axlearn.vision import spatial_transform_ops


# pylint: disable=no-self-use
class BoxGridCoordinatesTest(absltest.TestCase):
    """Tests box_grid_coordinates."""

    def test_4x4_grid(self):
        boxes = np.array([[[0.0, 0.0, 6.0, 6.0]]])
        box_grid = spatial_transform_ops.box_grid_coordinates(
            boxes, size_y=4, size_x=4, align_corners=False
        )
        expected_grid_y = np.array([[[0.75, 2.25, 3.75, 5.25]]])
        expected_grid_x = np.array([[[0.75, 2.25, 3.75, 5.25]]])
        np.testing.assert_allclose(expected_grid_y, box_grid.y)
        np.testing.assert_allclose(expected_grid_x, box_grid.x)

    def test_2x4_grid_with_aligned_corners(self):
        boxes = np.array([[[0.0, 0.0, 6.0, 6.0]]])
        box_grid = spatial_transform_ops.box_grid_coordinates(
            boxes, size_y=2, size_x=4, align_corners=True
        )
        expected_grid_y = np.array([[[0, 6]]])
        expected_grid_x = np.array([[[0, 2, 4, 6]]])
        np.testing.assert_allclose(expected_grid_y, box_grid.y)
        np.testing.assert_allclose(expected_grid_x, box_grid.x)

    def test_2x2_grid_multiple_boxes(self):
        boxes = np.array([[[0.0, 0.0, 6.0, 3.0], [0.0, 0.0, 3.0, 6.0]]])
        box_grid = spatial_transform_ops.box_grid_coordinates(
            boxes, size_y=2, size_x=2, align_corners=False
        )
        expected_grid_y = np.array([[[1.5, 4.5], [0.75, 2.25]]])
        expected_grid_x = np.array([[[0.75, 2.25], [1.5, 4.5]]])
        np.testing.assert_allclose(expected_grid_y, box_grid.y)
        np.testing.assert_allclose(expected_grid_x, box_grid.x)

    def test_2x4_grid(self):
        boxes = np.array([[[0.0, 0.0, 6.0, 6.0]]])
        box_grid = spatial_transform_ops.box_grid_coordinates(
            boxes, size_y=2, size_x=4, align_corners=False
        )
        expected_grid_y = np.array([[[1.5, 4.5]]])
        expected_grid_x = np.array([[[0.75, 2.25, 3.75, 5.25]]])
        np.testing.assert_allclose(expected_grid_y, box_grid.y)
        np.testing.assert_allclose(expected_grid_x, box_grid.x)

    def test_4x4_with_offgrid_boxes(self):
        boxes = np.array([[[1.2, 2.3, 7.2, 8.3]]], dtype=np.float32)
        box_grid = spatial_transform_ops.box_grid_coordinates(
            boxes, size_y=4, size_x=4, align_corners=False
        )
        expected_grid_y = np.array([[[0.75, 2.25, 3.75, 5.25]]]) + 1.2
        expected_grid_x = np.array([[[0.75, 2.25, 3.75, 5.25]]]) + 2.3
        np.testing.assert_allclose(expected_grid_y, box_grid.y)
        np.testing.assert_allclose(expected_grid_x, box_grid.x)


class FeatureGridCoordinateTest(absltest.TestCase):
    """Tests feature_grid_coordinates."""

    def test_snap_box_points_to_nearest_4_pixels(self):
        box_grid_y = np.array([[[1.5, 4.6]]], dtype=np.float32)
        box_grid_x = np.array([[[2.4, 5.3]]], dtype=np.float32)
        feature_grid = spatial_transform_ops.feature_grid_coordinates(
            spatial_transform_ops.BoxGrid(y=box_grid_y, x=box_grid_x),
            true_feature_shapes=np.array([[[6, 7]]]),
        )
        expected_grid_y0 = np.array([[[1, 4]]])
        expected_grid_y1 = np.array([[[2, 5]]])
        expected_grid_x0 = np.array([[[2, 5]]])
        expected_grid_x1 = np.array([[[3, 6]]])
        np.testing.assert_array_equal(expected_grid_y0, feature_grid.y0)
        np.testing.assert_array_equal(expected_grid_y1, feature_grid.y1)
        np.testing.assert_array_equal(expected_grid_x0, feature_grid.x0)
        np.testing.assert_array_equal(expected_grid_x1, feature_grid.x1)

    def test_snap_box_points_outside_pixel_grid_to_nearest_neighbor(self):
        box_grid_y = np.array([[[0.33, 1.0, 1.66]]], dtype=np.float32)
        box_grid_x = np.array([[[-0.5, 1.0, 2.66]]], dtype=np.float32)
        feature_grid = spatial_transform_ops.feature_grid_coordinates(
            spatial_transform_ops.BoxGrid(y=box_grid_y, x=box_grid_x),
            true_feature_shapes=np.array([[[4, 3]]]),
        )
        expected_grid_y0 = np.array([[[0, 1, 1]]])
        expected_grid_y1 = np.array([[[1, 2, 2]]])
        expected_grid_x0 = np.array([[[0, 1, 2]]])
        expected_grid_x1 = np.array([[[1, 2, 2]]])
        np.testing.assert_array_equal(expected_grid_y0, feature_grid.y0)
        np.testing.assert_array_equal(expected_grid_y1, feature_grid.y1)
        np.testing.assert_array_equal(expected_grid_x0, feature_grid.x0)
        np.testing.assert_array_equal(expected_grid_x1, feature_grid.x1)


class BoxLevelsTest(absltest.TestCase):
    """Tests get_box_levels."""

    def test_correct_fpn_levels(self):
        pretraining_image_size = 224
        boxes = np.array(
            [
                [
                    [0, 0, 111, 111],  # Level 0.
                    [0, 0, 113, 113],  # Level 1.
                    [0, 0, 223, 223],  # Level 1.
                    [0, 0, 225, 225],  # Level 2.
                    [0, 0, 449, 449],  # Level 3.
                ],
            ],
            dtype=np.float32,
        )

        levels = spatial_transform_ops.get_box_levels(
            boxes,
            min_level=0,
            max_level=3,
            unit_scale_level=2,
            pretraining_image_size=pretraining_image_size,
        )
        np.testing.assert_array_equal([[0, 1, 1, 2, 3]], levels)

    def test_map_illformed_boxes_to_min_level(self):
        pretraining_image_size = 224
        boxes = np.array(
            [
                [
                    [50, 60, 40, 80],  # Level 0.
                    [50, 60, 40, 30],  # Level 0.
                ],
            ],
            dtype=np.float32,
        )

        levels = spatial_transform_ops.get_box_levels(
            boxes,
            min_level=0,
            max_level=3,
            unit_scale_level=2,
            pretraining_image_size=pretraining_image_size,
        )
        np.testing.assert_array_equal([[0, 0]], levels)


class PadToMaxSizeTest(absltest.TestCase):
    """Tests pad_to_max_size."""

    def test_pad_features_to_max_size(self):
        features_a = np.random.rand(2, 10, 20, 4)
        features_b = np.random.rand(2, 20, 40, 4)
        padded_features = spatial_transform_ops.pad_to_max_size([features_a, features_b])
        # pylint: disable=unsubscriptable-object
        np.testing.assert_array_equal((2, 2, 20, 40, 4), padded_features.features.shape)
        np.testing.assert_allclose(features_a, padded_features.features[:, 0, :10, :20, :])
        np.testing.assert_allclose(features_b, padded_features.features[:, 1, :20, :40, :])
        np.testing.assert_array_equal(((10, 20), (20, 40)), padded_features.true_shapes)
        # pylint: enable=unsubscriptable-object


class RavelIndicesTest(absltest.TestCase):
    """Tests ravel_indices."""

    def test_feature_point_indices(self):
        feature_grid_y = np.array([[[1, 2, 4, 5], [2, 3, 4, 5]]], dtype=np.int32)
        feature_grid_x = np.array([[[1, 3, 4], [2, 3, 4]]], dtype=np.int32)
        num_feature_levels = 2
        feature_height = 6
        feature_width = 5
        box_levels = np.array([[0, 1]], dtype=np.int32)

        indices = spatial_transform_ops.ravel_indices(
            feature_grid_y=feature_grid_y,
            feature_grid_x=feature_grid_x,
            num_levels=num_feature_levels,
            height=feature_height,
            width=feature_width,
            box_levels=box_levels,
        )
        expected_indices = np.array(
            [
                [
                    [[6, 8, 9], [11, 13, 14], [21, 23, 24], [26, 28, 29]],
                    [[42, 43, 44], [47, 48, 49], [52, 53, 54], [57, 58, 59]],
                ]
            ]
        )
        np.testing.assert_array_equal(expected_indices.flatten(), indices)


class ValidCoordinateIndicatorTest(absltest.TestCase):
    """Tests valid_coordinates."""

    def test_valid_coordinate_indicator(self):
        feature_grid_y = np.array([[[1, 4, 2, 5]]])
        feature_grid_x = np.array([[[2, 5, 3, 6]]])
        true_feature_shapes = np.array([[[3, 4]]])
        valid_indicator = spatial_transform_ops.valid_coordinates(
            feature_grid_y=feature_grid_y,
            feature_grid_x=feature_grid_x,
            true_feature_shapes=true_feature_shapes,
        )
        expected_indicator = np.array(
            [
                [True, False, True, False],
                [False, False, False, False],
                [True, False, True, False],
                [False, False, False, False],
            ]
        )
        np.testing.assert_array_equal(valid_indicator, expected_indicator.flatten())


class GatherValidIndices(absltest.TestCase):
    """Tests gather_valid_indices."""

    def test_gather(self):
        tensor = np.random.rand(9, 4, 2)
        indices = np.array([0, 1, 8, -1, -1])
        actual_tensor = spatial_transform_ops.gather_valid_indices(tensor=tensor, indices=indices)
        expected_tensor = np.stack(
            [tensor[0], tensor[1], tensor[8], np.zeros_like(tensor[0]), np.zeros_like(tensor[0])]
        )
        np.testing.assert_allclose(expected_tensor, actual_tensor)


class RoIAlignTest(absltest.TestCase):
    """Tests gather_valid_indices."""

    def test_perfectly_aligned_cell_center_and_feature_pixels(self):
        features = np.arange(25).reshape(1, 5, 5, 1).astype(np.float32)
        boxes = np.array([[[0, 0, 4, 4]]], dtype=np.float32)
        box_levels = np.array([[0]], dtype=np.int32)
        crop_output = spatial_transform_ops.roi_align(
            features=[features], boxes=boxes, box_levels=box_levels, output_size=(2, 2)
        )
        expected_output = [[[[[6], [8]], [[16], [18]]]]]
        np.testing.assert_allclose(crop_output, expected_output)

    def test_interpolation_with_4_points_per_bin(self):
        features = np.array(
            [
                [
                    [[1], [2], [3], [4]],
                    [[5], [6], [7], [8]],
                    [[9], [10], [11], [12]],
                    [[13], [14], [15], [16]],
                ]
            ],
            dtype=np.float32,
        )
        boxes = np.array([[[1, 1, 2, 2]]], dtype=np.float32)
        box_levels = np.array([[0]], dtype=np.int32)
        crop_output = spatial_transform_ops.roi_align(
            features=[features],
            boxes=boxes,
            box_levels=box_levels,
            output_size=(1, 1),
            num_samples_per_cell_y=2,
            num_samples_per_cell_x=2,
        )
        expected_output = [[[[[(7.25 + 7.75 + 9.25 + 9.75) / 4]]]]]
        np.testing.assert_allclose(crop_output, expected_output)

    def test_1x1_crops_on_2x2_features(self):
        features = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
        boxes = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
        box_levels = np.array([[0]], dtype=np.int32)
        crop_output = spatial_transform_ops.roi_align(
            features=[features], boxes=boxes, box_levels=box_levels, output_size=(1, 1)
        )
        expected_output = [[[[[2.5]]]]]
        np.testing.assert_allclose(crop_output, expected_output)

    def test_3x3_crops_on_2x2_features(self):
        features = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
        boxes = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
        box_levels = np.array([[0]], dtype=np.int32)
        crop_output = spatial_transform_ops.roi_align(
            features=[features], boxes=boxes, box_levels=box_levels, output_size=(3, 3)
        )
        expected_output = [
            [
                [
                    [[9 / 6], [11 / 6], [13 / 6]],
                    [[13 / 6], [15 / 6], [17 / 6]],
                    [[17 / 6], [19 / 6], [21 / 6]],
                ]
            ]
        ]
        np.testing.assert_allclose(crop_output, expected_output)

    def test_2x2_crops_on_3x3_features(self):
        features = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
        boxes = np.array([[[0, 0, 2, 2], [0, 0, 1, 1]]], dtype=np.float32)
        box_levels = np.array([[0, 0]], dtype=np.int32)
        crop_output = spatial_transform_ops.roi_align(
            features=[features], boxes=boxes, box_levels=box_levels, output_size=(2, 2)
        )
        expected_output = [[[[[3], [4]], [[6], [7]]], [[[2.0], [2.5]], [[3.5], [4.0]]]]]
        np.testing.assert_allclose(crop_output, expected_output)

    def test_2x2_crops_on_4x4_features(self):
        features = np.array(
            [
                [
                    [[0], [1], [2], [3]],
                    [[4], [5], [6], [7]],
                    [[8], [9], [10], [11]],
                    [[12], [13], [14], [15]],
                ]
            ],
            dtype=np.float32,
        )
        boxes = np.array(
            [[[0, 0, 3 * 2 / 3, 3 * 2 / 3], [0, 0, 3 * 2 / 3, 3 * 1]]], dtype=np.float32
        )
        box_levels = np.array([[0, 0]], dtype=np.int32)
        crop_output = spatial_transform_ops.roi_align(
            features=[features], boxes=boxes, box_levels=box_levels, output_size=(2, 2)
        )
        expected_output = np.array(
            [[[[[2.5], [3.5]], [[6.5], [7.5]]], [[[2.75], [4.25]], [[6.75], [8.25]]]]]
        )
        np.testing.assert_allclose(crop_output, expected_output)

    def test_clipping_3x3_crops_on_2x2_features(self):
        features = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
        boxes = np.array([[[-1, -1, 2, 2]]], dtype=np.float32)
        box_levels = np.array([[0]], dtype=np.int32)
        crop_output = spatial_transform_ops.roi_align(
            features=[features], boxes=boxes, box_levels=box_levels, output_size=(3, 3)
        )
        expected_output = np.array(
            [[[[[-0.5], [0.5], [1.0]], [[1.5], [2.5], [3.0]], [[2.5], [3.5], [4.0]]]]]
        )
        np.testing.assert_allclose(crop_output, expected_output)

    def test_multilevel_roi_align(self):
        image_size = 640
        fpn_min_level = 2
        fpn_max_level = 5
        batch_size = 1
        output_size = (2, 2)
        num_filters = 1
        features = []
        for level in range(fpn_min_level, fpn_max_level + 1):
            feat_size = int(image_size / 2**level)
            features.append(
                float(level)
                * np.ones([batch_size, feat_size, feat_size, num_filters], dtype=np.float32)
            )
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
        levels = np.array([[0, 1, 1, 2, 3]], dtype=np.int32)
        roi_features = spatial_transform_ops.roi_align(
            features=features, boxes=boxes, box_levels=levels, output_size=output_size
        )
        np.testing.assert_allclose(2 * np.ones((2, 2, 1)), roi_features[0][0])
        np.testing.assert_allclose(3 * np.ones((2, 2, 1)), roi_features[0][1])
        np.testing.assert_allclose(3 * np.ones((2, 2, 1)), roi_features[0][2])
        np.testing.assert_allclose(4 * np.ones((2, 2, 1)), roi_features[0][3])
        np.testing.assert_allclose(5 * np.ones((2, 2, 1)), roi_features[0][4])

    # TODO(markblee): Re-enable in CI when we have access to a larger instance.
    @pytest.mark.high_cpu
    def test_large_input(self):
        input_size = 1408
        min_level = 2
        max_level = 6
        batch_size = 2
        num_boxes = 512
        num_filters = 256
        output_size = (7, 7)
        features = []
        for level in range(min_level, max_level + 1):
            feat_size = int(input_size / 2**level)
            features.append(
                np.reshape(
                    np.arange(batch_size * feat_size * feat_size * num_filters, dtype=np.float32),
                    [batch_size, feat_size, feat_size, num_filters],
                )
            )
        boxes = np.array(
            [
                [[0, 0, 256, 256]] * num_boxes,
            ],
            dtype=np.float32,
        )
        boxes = np.tile(boxes, [batch_size, 1, 1])
        levels = np.random.randint(5, size=[batch_size, num_boxes], dtype=np.int32)
        roi_features = spatial_transform_ops.roi_align(
            features=features, boxes=boxes, box_levels=levels, output_size=output_size
        )
        np.testing.assert_array_equal(
            (batch_size, num_boxes, output_size[0], output_size[1], num_filters), roi_features.shape
        )


if __name__ == "__main__":
    absltest.main()

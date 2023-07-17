# Copyright © 2023 Apple Inc.

"""Tests similarity ops."""
import numpy as np
from absl.testing import absltest

from axlearn.vision import similarity_ops


# pylint: disable=no-self-use
class IntersectionTest(absltest.TestCase):
    """Tests intersection ops."""

    def test_elementwise_intersection(self):
        boxes_a = np.array(
            [
                [[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                [[5.0, 6.0, 10.0, 7.0], [4.0, 3.0, 7.0, 5.0]],
            ]
        )
        boxes_b = np.array(
            [
                [[3.0, 4.0, 6.0, 8.0], [6.0, 5.0, 8.0, 7.0]],
                [[0.0, 0.0, 20.0, 20.0], [3.0, 1.0, 6.0, 4.0]],
            ]
        )
        intersection = similarity_ops.elementwise_intersection_areas(
            boxes_a=boxes_a, boxes_b=boxes_b
        )
        expected_output = [[2.0, 2.0], [5.0, 2.0]]
        np.testing.assert_allclose(expected_output, intersection)

    def test_pairwise_intersection(self):
        boxes_a = np.array(
            [
                [[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                [[5.0, 6.0, 10.0, 7.0], [4.0, 3.0, 7.0, 5.0]],
            ]
        )
        boxes_b = np.array(
            [
                [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]],
                [[0.0, 0.0, 20.0, 20.0], [14.0, 14.0, 15.0, 15.0], [3.0, 4.0, 6.0, 8.0]],
            ]
        )
        intersection = similarity_ops.pairwise_intersection_areas(boxes_a=boxes_a, boxes_b=boxes_b)
        expected_output = [[[2.0, 0.0, 6.0], [1.0, 0.0, 5.0]], [[5.0, 0.0, 1.0], [6.0, 0.0, 2.0]]]
        np.testing.assert_allclose(expected_output, intersection)


class AreaTest(absltest.TestCase):
    """Tests areas."""

    def test_area(self):
        boxes = np.array(
            [
                [[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                [[5.0, 6.0, 10.0, 7.0], [4.0, 3.0, 7.0, 5.0]],
            ]
        )
        areas = similarity_ops.areas(boxes)
        expected_output = [[6.0, 5.0], [5.0, 6.0]]
        np.testing.assert_allclose(expected_output, areas)


class IoUTest(absltest.TestCase):
    """Tests IoU ops."""

    def test_pairwise_iou(self):
        boxes_a = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        boxes_b = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]])
        iou = similarity_ops.pairwise_iou(boxes_a=boxes_a, boxes_b=boxes_b)
        expected_output = [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
        np.testing.assert_allclose(expected_output, iou)

    def test_pairwise_ioa(self):
        boxes_a = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        boxes_b = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]])
        ioa = similarity_ops.pairwise_ioa(boxes_a=boxes_a, boxes_b=boxes_b)
        expected_output = [[2.0 / 6.0, 0, 6.0 / 6.0], [1.0 / 5.0, 0.0, 5.0 / 5.0]]
        np.testing.assert_allclose(expected_output, ioa)

    def test_pairwise_iou_with_box_paddings(self):
        boxes_a = np.array([[[4.0, 3.0, 7.0, 5.0], [0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 10.0, 7.0]]])
        boxes_b = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [3.0, 4.0, 6.0, 8.0],
                    [14.0, 14.0, 15.0, 15.0],
                    [0.0, 0.0, 20.0, 20.0],
                ]
            ]
        )
        paddings_a = np.array([[False, True, False]])
        paddings_b = np.array([[True, False, False, False]])
        p_iou = similarity_ops.pairwise_iou(
            boxes_a=boxes_a, boxes_b=boxes_b, paddings_a=paddings_a, paddings_b=paddings_b
        )
        expected_output = [
            [
                [-1.0, 2.0 / 16.0, 0, 6.0 / 400.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, 1.0 / 16.0, 0.0, 5.0 / 400.0],
            ]
        ]
        np.testing.assert_allclose(expected_output, p_iou)

    def test_elementwise_iou(self):
        boxes_a = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        boxes_b = np.array([[3.0, 4.0, 6.0, 8.0], [6.0, 5.0, 8.0, 7.0]])
        iou = similarity_ops.elementwise_iou(boxes_a=boxes_a, boxes_b=boxes_b)
        expected_output = [2.0 / 16.0, 2.0 / 7.0]
        np.testing.assert_allclose(expected_output, iou)

    def test_elementwise_iou_with_box_paddings(self):
        boxes_a = np.array([[[4.0, 3.0, 7.0, 5.0], [0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 10.0, 7.0]]])
        boxes_b = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [3.0, 4.0, 6.0, 8.0],
                    [6.0, 5.0, 8.0, 7.0],
                ]
            ]
        )
        paddings_a = np.array([[False, True, False]])
        paddings_b = np.array([[True, False, False]])
        e_iou = similarity_ops.elementwise_iou(
            boxes_a=boxes_a, boxes_b=boxes_b, paddings_a=paddings_a, paddings_b=paddings_b
        )
        expected_output = [[-1.0, -1.0, 2.0 / 7.0]]
        np.testing.assert_allclose(expected_output, e_iou)


if __name__ == "__main__":
    absltest.main()

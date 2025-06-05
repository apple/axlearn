# Copyright © 2023 Apple Inc.

"""Tests for coco_utils.py."""
from typing import Any, Union

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.vision import coco_utils


class CocoToolsTest(parameterized.TestCase, tf.test.TestCase):
    """Tests COCO utils."""

    def test_export_detections_to_coco(self):
        image_ids = ["first", "second"]
        detections_boxes = [
            np.array([[100, 100, 200, 200]], float),
            np.array([[50, 50, 100, 100]], float),
        ]
        detections_scores = [np.array([0.8], float), np.array([0.7], float)]
        detections_classes = [np.array([1], np.int32), np.array([1], np.int32)]
        predictions = {
            "source_id": np.expand_dims(image_ids, axis=0),
            "detection_boxes": np.expand_dims(detections_boxes, axis=0),
            "detection_classes": np.expand_dims(detections_classes, axis=0),
            "detection_scores": np.expand_dims(detections_scores, axis=0),
            "num_detections": np.expand_dims([1, 1], axis=0),
        }
        results = coco_utils.convert_predictions_to_coco_annotations(predictions)

        expected = [
            {
                "image_id": "first",
                "category_id": 1,
                "bbox": [100.0, 100.0, 100.0, 100.0],
                "score": 0.8,
                "id": 1,
            },
            {
                "image_id": "second",
                "category_id": 1,
                "bbox": [50.0, 50.0, 50.0, 50.0],
                "score": 0.7,
                "id": 2,
            },
        ]
        for i, r in enumerate(results):
            for k, v in r.items():
                self.assertAllEqual(v, expected[i][k])

    @parameterized.parameters(
        # Single-label
        (
            [[11], [32]],
            [{"id": 32}, {"id": 11}],
            [
                {
                    "image_id": 2367,
                    "iscrowd": 0,
                    "category_id": 11,
                    "bbox": [100.0, 100.0, 100.0, 100.0],
                    "area": 10000.0,
                    "id": 1,
                },
                {
                    "image_id": 1692,
                    "iscrowd": 0,
                    "category_id": 32,
                    "bbox": [50.0, 50.0, 50.0, 50.0],
                    "area": 2500.0,
                    "id": 2,
                },
            ],
        ),
        # Multi-label
        (
            [[[11, -1, -1, -1]], [[32, 31, -1, -1]]],
            [{"id": 32}, {"id": 11}, {"id": 31}],
            [
                {
                    "image_id": 2367,
                    "iscrowd": 0,
                    "category_id": 11,
                    "bbox": [100.0, 100.0, 100.0, 100.0],
                    "area": 10000.0,
                    "id": 1,
                },
                {
                    "image_id": 1692,
                    "iscrowd": 0,
                    "category_id": 32,
                    "bbox": [50.0, 50.0, 50.0, 50.0],
                    "area": 2500.0,
                    "id": 2,
                },
                {
                    "image_id": 1692,
                    "iscrowd": 0,
                    "category_id": 31,
                    "bbox": [50.0, 50.0, 50.0, 50.0],
                    "area": 2500.0,
                    "id": 3,
                },
            ],
        ),
    )
    def test_export_groundtruths_to_coco(
        self,
        groundtruth_classes: Union[list[list[int]], list[list[list[int]]]],
        expected_categories: list[dict[str, int]],
        expected_annotations: list[dict[str, Any]],
    ):
        groundtruth_boxes = np.array(
            [[[100, 100, 200, 200]], [[50, 50, 100, 100]]], dtype=np.float32
        )
        groundtruths = {
            "source_id": np.array([np.array([2367, 1692], dtype=np.int32)]),
            "height": np.array([np.array([512, 512], dtype=np.int32)]),
            "width": np.array([np.array([512, 512], dtype=np.int32)]),
            "boxes": np.array([groundtruth_boxes]),
            "classes": np.array([np.array(groundtruth_classes)]),
            "num_detections": np.array([np.array([1, 1], dtype=np.int32)]),
        }
        results = coco_utils.convert_groundtruths_to_coco_dataset(groundtruths)

        expected = {
            "images": [
                {"id": 2367, "height": 512, "width": 512},
                {"id": 1692, "height": 512, "width": 512},
            ],
            "categories": expected_categories,
            "annotations": expected_annotations,
        }

        self.assertAllEqual(results["images"], expected["images"])
        self.assertAllEqual(
            {i["id"] for i in results["categories"]},
            {i["id"] for i in expected["categories"]},
        )
        self.assertAllEqual(results["annotations"], expected["annotations"])

    def test_calculate_per_category_metrics(self):
        categories = [1, 2, 3]
        iou_thresh = list(np.arange(0.5, 1, 0.05))
        recall_thresh = list(np.arange(0, 1.01, 0.01))
        area_ranges = ["all", "small", "medium", "large"]
        max_dets = [1, 10, 100]

        precision = np.zeros(
            [len(iou_thresh), len(recall_thresh), len(categories), len(area_ranges), len(max_dets)]
        )
        recall = np.zeros([len(iou_thresh), len(categories), len(area_ranges), len(max_dets)])

        # Assign precision for "small" object to 1
        precision[:, :, :, 1, :] = 1
        # Assign recall for maxdets=10 to 1
        recall[:, :, :, 1] = 1

        category_wise_metrics = coco_utils.calculate_per_category_metrics(
            categories, precision, recall
        )

        self.assertAllEqual(category_wise_metrics.shape, (12, 3))
        self.assertAllEqual(category_wise_metrics[0], [0, 0, 0])
        self.assertAllEqual(category_wise_metrics[1], [0, 0, 0])
        self.assertAllEqual(category_wise_metrics[2], [0, 0, 0])
        self.assertAllEqual(category_wise_metrics[3], [1, 1, 1])
        self.assertAllEqual(category_wise_metrics[4], [0, 0, 0])
        self.assertAllEqual(category_wise_metrics[5], [0, 0, 0])
        self.assertAllEqual(category_wise_metrics[6], [0, 0, 0])
        self.assertAllEqual(category_wise_metrics[7], [1, 1, 1])
        self.assertAllEqual(category_wise_metrics[8], [0, 0, 0])
        self.assertAllEqual(category_wise_metrics[9], [1 / 3, 1 / 3, 1 / 3])
        self.assertAllEqual(category_wise_metrics[10], [1 / 3, 1 / 3, 1 / 3])
        self.assertAllEqual(category_wise_metrics[11], [1 / 3, 1 / 3, 1 / 3])


if __name__ == "__main__":
    absltest.main()

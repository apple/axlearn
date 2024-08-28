# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/tpu:
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Util functions related to pycocotools and COCO eval.

The functions in this file have been borrowed from the tensorflow model garden's coco_utils.py.
Reference:
https://github.com/tensorflow/tpu/blob/119236319e51d1b575c57b99a69812a0dff90d36/models/official/detection/evaluation/coco_utils.py
"""

import copy
from typing import Any, Optional

import numpy as np
import six
import tensorflow as tf
from absl import logging
from PIL import Image
from pycocotools import coco
from pycocotools import mask as mask_api

from axlearn.vision import utils_detection


# pylint: disable=arguments-renamed, too-many-branches, too-many-statements, unused-argument, unused-variable
class COCOWrapper(coco.COCO):
    """COCO wrapper class.

    This class wraps COCO API object, which provides the following additional
    functionalities:
        1. Support string type image id.
        2. Support loading the groundtruth dataset using the external annotation dictionary.
        3. Support loading the prediction results using the external annotation dictionary.
    """

    def __init__(self, eval_type="box", annotation_file=None, gt_dataset=None):
        """Instantiates a COCO-style API object.

        Args:
            eval_type: either 'box' or 'mask'.
            annotation_file: a JSON file that stores annotations of the eval dataset.
                This is required if `gt_dataset` is not provided.
            gt_dataset: the groundtruth eval dataset in COCO API format.

        Raises:
            ValueError: If annotation_file and gt_dataset are both specified or both left
                unspecified, or if eval_type is invalid.
        """
        if (annotation_file and gt_dataset) or ((not annotation_file) and (not gt_dataset)):
            raise ValueError(
                "One and only one of `annotation_file` and `gt_dataset` " "needs to be specified."
            )

        if eval_type not in ["box", "mask"]:
            raise ValueError("The `eval_type` can only be either `box` or `mask`.")

        coco.COCO.__init__(self, annotation_file=annotation_file)
        self._eval_type = eval_type
        if gt_dataset:
            self.dataset = gt_dataset
            self.createIndex()

    def loadRes(self, predictions: list[dict[str, Any]]) -> coco.COCO:
        """Loads result file and return a result api object.

        Args:
            predictions: a list of dictionary each representing an annotation in COCO format. The
                required fields are `image_id`, `category_id`, `score`, `bbox`, `segmentation`.

        Returns:
            res: result COCO api object.

        Raises:
            ValueError: If the set of image id from predctions is not the subset of
                the set of image id of the groundtruth dataset.
        """
        res = coco.COCO()
        res.dataset["images"] = copy.deepcopy(self.dataset["images"])
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])

        image_ids = [ann["image_id"] for ann in predictions]
        if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
            raise ValueError("Results do not correspond to the current dataset!")
        for ann in predictions:
            x1, x2, y1, y2 = [
                ann["bbox"][0],
                ann["bbox"][0] + ann["bbox"][2],
                ann["bbox"][1],
                ann["bbox"][1] + ann["bbox"][3],
            ]
            if self._eval_type == "box":
                ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            elif self._eval_type == "mask":
                ann["area"] = mask_api.area(ann["segmentation"])

        res.dataset["annotations"] = copy.deepcopy(predictions)
        res.createIndex()
        return res


def calculate_per_category_metrics(
    categories: list[int], precision: np.ndarray, recall: np.ndarray
) -> np.ndarray:
    """Returns per category coco metrics.

    Returns the following per category metrics:
    (Reference: https://cocodataset.org/#detection-eval)
        - Precision mAP ByCategory
        - Precision mAP ByCategory@50IoU
        - Precision mAP ByCategory@75IoU
        - Precision mAP ByCategory (small)
        - Precision mAP ByCategory (medium)
        - Precision mAP ByCategory (large)
        - Recall AR@1 ByCategory
        - Recall AR@10 ByCategory
        - Recall AR@100 ByCategory
        - Recall AR (small) ByCategory
        - Recall AR (medium) ByCategory
        - Recall AR (large) ByCategory

    Args:
        categories: A list of integer categories.
        precision: A float ndarray of shape
            [IOUthresh x RecallThresh x CategoryID x AreaRanges x MaxDets].
        recall: A float ndarray of shape [IOUthresh x CategoryID x AreaRanges x MaxDets]
            where IOUthresh are the IOU thresholds for evaluation, RecallThresh are the recall
            thresholds for evaluation, AreaRanges are the object area ranges for evaluation, and
            MaxDets are the thresholds on max detections per image.
            More details on precision and recall array shapes can be found at
            https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Returns:
        A float ndarray of shape (12, num_categories) containing the above indicated 12 per
            category metrics for each category.
            (TODO: Surface class names in the returned metrics instead of integers)
    """
    category_stats = np.zeros([12, len(categories)])

    # precision - [IOUthresh x RecallThresh x CategoryID x AreaRanges x MaxDets]
    # recall - [IOUthresh x CategoryID x AreaRanges x MaxDets]

    for category_index, category_id in enumerate(categories):
        # Precision mAP ByCategory
        category_stats[0][category_index] = precision[:, :, category_index, 0, :].mean()
        # Precision mAP ByCategory@50IoU
        category_stats[1][category_index] = precision[0, :, category_index, 0, :].mean()
        # Precision mAP ByCategory@75IoU
        category_stats[2][category_index] = precision[5, :, category_index, 0, :].mean()
        # Precision mAP ByCategory (small)
        category_stats[3][category_index] = precision[:, :, category_index, 1, :].mean()
        # Precision mAP ByCategory (medium)
        category_stats[4][category_index] = precision[:, :, category_index, 2, :].mean()
        # Precision mAP ByCategory (large)
        category_stats[5][category_index] = precision[:, :, category_index, 3, :].mean()
        # Recall AR@1 ByCategory
        category_stats[6][category_index] = recall[:, category_index, 0, 0].mean()
        # Recall AR@10 ByCategory
        category_stats[7][category_index] = recall[:, category_index, 0, 1].mean()
        # Recall AR@100 ByCategory
        category_stats[8][category_index] = recall[:, category_index, 0, 2].mean()
        # Recall AR (small) ByCategory
        category_stats[9][category_index] = recall[:, category_index, 1, :].mean()
        # Recall AR (medium) ByCategory
        category_stats[10][category_index] = recall[:, category_index, 2, :].mean()
        # Recall AR (large) ByCategory
        category_stats[11][category_index] = recall[:, category_index, 3, :].mean()

    return category_stats


def convert_predictions_to_coco_annotations(predictions: dict[str, list[np.ndarray]]):
    """Converts a batch of predictions to annotations in COCO format.

    Args:
        predictions: a dictionary of lists of numpy arrays including the following
            fields. K below denotes the maximum number of instances per image.
            Required fields:
            - source_id: a list of numpy arrays of int or string of shape [batch_size].
            - num_detections: a list of numpy arrays of int of shape [batch_size].
            - detection_boxes: a list of numpy arrays of float of shape
                [batch_size, K, 4], where coordinates are in the original image
                space (not the scaled image space).
            - detection_classes: a list of numpy arrays of int of shape [batch_size, K].
            - detection_scores: a list of numpy arrays of float of shape [batch_size, K].

        Optional fields:
            - detection_masks: a list of numpy arrays of float of shape
                [batch_size, K, mask_height, mask_width].

    Returns:
        coco_predictions: prediction in COCO annotation format.

    Raises:
        ValueError: If attempting to convert predictions for instance segmentation.
    """
    coco_predictions = []
    num_batches = len(predictions["source_id"])
    max_num_detections = predictions["detection_classes"][0].shape[1]
    use_outer_box = "detection_outer_boxes" in predictions
    for i in range(num_batches):
        predictions["detection_boxes"][i] = utils_detection.yxyx_to_xywh(
            predictions["detection_boxes"][i]
        )
        if use_outer_box:
            predictions["detection_outer_boxes"][i] = utils_detection.yxyx_to_xywh(
                predictions["detection_outer_boxes"][i]
            )

        batch_size = predictions["source_id"][i].shape[0]
        for j in range(batch_size):
            if "detection_masks" in predictions:
                raise ValueError("Instance segmentation not supported yet.")
            for k in range(max_num_detections):
                ann = {}
                ann["image_id"] = predictions["source_id"][i][j]
                ann["category_id"] = predictions["detection_classes"][i][j, k]
                ann["bbox"] = predictions["detection_boxes"][i][j, k]
                ann["score"] = predictions["detection_scores"][i][j, k]
                if "detection_masks" in predictions:
                    raise ValueError("Instance segmentation not supported yet.")
                coco_predictions.append(ann)

    for i, ann in enumerate(coco_predictions):
        ann["id"] = i + 1

    return coco_predictions


def convert_groundtruths_to_coco_dataset(
    groundtruths: dict[str, np.ndarray], label_map: Optional[dict[int, str]] = None
):
    """Converts groundtruths to the dataset in COCO format.

    Converts groundtruths to COCO dataset format. In case of multi-label groundtruths, a separate
    annotation corresponding to each groundtruth label is created.

    Args:
        groundtruths: a dictionary of numpy arrays including the fields below.
            Note that each element in the list represent the number for a single
            example without batch dimension. K below denotes the actual number of
            instances for each image.
            Required fields:
            - source_id: a list of numpy arrays of int or string of shape [batch_size].
            - height: a list of numpy arrays of int of shape [batch_size].
            - width: a list of numpy arrays of int of shape [batch_size].
            - num_detections: a list of numpy arrays of int of shape [batch_size].
            - boxes: a list of numpy arrays of float of shape [batch_size, K, 4],
                where coordinates are in the original image space (not the
                normalized coordinates).
            - classes: a list of numpy arrays of int of shape [batch_size, K, ...], where trailing
                dimensions can exist for multi-label data. When multi-label data exists, an
                annotation is created for each of the labels.
            Optional fields:
            - is_crowds: a list of numpy arrays of int of shape [batch_size, K]. If
                th field is absent, it is assumed that this instance is not crowd.
            - areas: a list of numy arrays of float of shape [batch_size, K]. If the
                field is absent, the area is calculated using either boxes or
                masks depending on which one is available.
            - masks: a list of numpy arrays of string of shape [batch_size, K],
        label_map: (optional) a dictionary that defines items from the category id to the category
            name. If `None`, collect the category mapping from the `groundtruths`.

    Returns:
        coco_groundtruths: the groundtruth dataset in COCO format.
    """
    source_ids = np.concatenate(groundtruths["source_id"], axis=0)
    heights = np.concatenate(groundtruths["height"], axis=0)
    widths = np.concatenate(groundtruths["width"], axis=0)
    gt_images = [
        {"id": int(i), "height": int(h), "width": int(w)}
        for i, h, w in zip(source_ids, heights, widths)
    ]

    def _create_annotation(*, batch_id: int, image_id: int, box_id: int, class_label: int) -> dict:
        ann = {}
        ann["image_id"] = int(groundtruths["source_id"][batch_id][image_id])
        if "is_crowds" in groundtruths:
            ann["iscrowd"] = int(groundtruths["is_crowds"][batch_id][image_id, box_id])
        else:
            ann["iscrowd"] = 0
        ann["category_id"] = class_label
        boxes = groundtruths["boxes"][batch_id]
        ann["bbox"] = [
            float(boxes[image_id, box_id, 1]),
            float(boxes[image_id, box_id, 0]),
            float(boxes[image_id, box_id, 3] - boxes[image_id, box_id, 1]),
            float(boxes[image_id, box_id, 2] - boxes[image_id, box_id, 0]),
        ]
        if "areas" in groundtruths:
            ann["area"] = float(groundtruths["areas"][batch_id][image_id, box_id])
        else:
            ann["area"] = float(
                (boxes[image_id, box_id, 3] - boxes[image_id, box_id, 1])
                * (boxes[image_id, box_id, 2] - boxes[image_id, box_id, 0])
            )
        if "masks" in groundtruths:
            if isinstance(groundtruths["masks"][batch_id][image_id, box_id], tf.Tensor):
                mask = Image.open(
                    six.BytesIO(groundtruths["masks"][batch_id][image_id, box_id].numpy())
                )
                width, height = mask.size
                np_mask = np.array(mask.getdata()).reshape(height, width).astype(np.uint8)
            else:
                mask = Image.open(six.BytesIO(groundtruths["masks"][batch_id][image_id, box_id]))
                width, height = mask.size
                np_mask = np.array(mask.getdata()).reshape(height, width).astype(np.uint8)
            np_mask[np_mask > 0] = 255
            encoded_mask = mask_api.encode(np.asfortranarray(np_mask))
            ann["segmentation"] = encoded_mask
            # Ensure the content of `counts` is JSON serializable string.
            if "counts" in ann["segmentation"]:
                ann["segmentation"]["counts"] = six.ensure_str(ann["segmentation"]["counts"])
            if "areas" not in groundtruths:
                ann["area"] = mask_api.area(encoded_mask)
        return ann

    gt_annotations = []
    num_batches = len(groundtruths["source_id"])
    # pylint: disable=too-many-nested-blocks
    for i in range(num_batches):
        logging.debug("convert_groundtruths_to_coco_dataset: Processing annotation %d", i)
        max_num_instances = groundtruths["classes"][i].shape[1]
        batch_size = groundtruths["source_id"][i].shape[0]
        for j in range(batch_size):
            num_instances = groundtruths["num_detections"][i][j]
            if num_instances > max_num_instances:
                logging.warning(
                    "num_groundtruths is larger than max_num_instances, %d v.s. %d",
                    num_instances,
                    max_num_instances,
                )
                num_instances = max_num_instances
            for k in range(int(num_instances)):
                # For multi label data, groundtruth_classes has dimension 1.
                # For single label data, convert to 1d below.
                groundtruth_classes = np.atleast_1d(groundtruths["classes"][i][j, k])
                # Get number of labels (without padding).
                num_labels = int((groundtruth_classes > -1).sum())
                # Create annotation corresponding to each label and append to gt_annotations.
                for label_id in range(num_labels):
                    ann = _create_annotation(
                        batch_id=i, image_id=j, box_id=k, class_label=groundtruth_classes[label_id]
                    )
                    gt_annotations.append(ann)

    for i, ann in enumerate(gt_annotations):
        ann["id"] = i + 1

    if label_map:
        gt_categories = [{"id": i, "name": label_map[i]} for i in label_map]
    else:
        category_ids = [gt["category_id"] for gt in gt_annotations]
        gt_categories = [{"id": i} for i in set(category_ids)]

    gt_dataset = {
        "images": gt_images,
        "categories": gt_categories,
        "annotations": copy.deepcopy(gt_annotations),
    }
    return gt_dataset

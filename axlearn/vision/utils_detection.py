# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Utils for object detection.

Code reference:
https://github.com/tensorflow/models/blob/master/official/vision/ops/preprocess_ops.py
"""
# pylint: disable=too-many-lines
import enum
import math
from typing import Callable, Optional, Union

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from axlearn.common.utils import Tensor


def normalize_boxes(*, boxes: Tensor, image_shape: Tensor) -> Tensor:
    """Normalizes box coordinates with respect to the image shape.

    Args:
        boxes: A float [..., 4] tensor with boxes in image coordinates. Boxes must be in the form
            [ymin, xmin, ymax, xmax].
        image_shape: A [..., 2] integer tensor representing image shape of the form [height, width].

    Returns:
        A float [..., 4] tensor with boxes in normalized coordinates and in the form
            [ymin, xmin, ymax, xmax].
    """
    return boxes / jnp.concatenate([image_shape, image_shape], axis=-1)


def denormalize_boxes(*, boxes: Tensor, image_shape: Tensor) -> tf.Tensor:
    """Converts normalized boxes to image coordinates.

    Args:
        boxes: A float [..., 4] tensor with boxes in image coordinates. Boxes must be in the form
            [ymin, xmin, ymax, xmax].
        image_shape: A [..., 2] integer tensor representing image shape of the form [height, width].


    Returns:
        A float [..., 4] tensor with boxes in image coordinates and in the form
            [ymin, xmin, ymax, xmax].
    """
    return boxes * jnp.concatenate([image_shape, image_shape], axis=-1)


# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
def normalize_boxes_tf(
    boxes: tf.Tensor, image_shape: Union[tuple[int, int], tf.Tensor]
) -> tf.Tensor:
    """Converts boxes to the normalized coordinates.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        image_shape: a two-element tuple or a tensor such that all but the last
            dimensions are `broadcastable` to `boxes`. The last dimension is 2,
            which represents [height, width].

    Returns:
        normalized_boxes: a tensor whose shape is the same as `boxes` representing
            the normalized boxes.

    Raises:
        ValueError: If the last dimension of boxes is not 4.
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes.shape[-1] is {boxes.shape[-1]}, but must be 4.")

    if isinstance(image_shape, (list, tuple)):
        height, width = image_shape
    else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        height = image_shape[..., 0:1]
        width = image_shape[..., 1:2]

    ymin = boxes[..., 0:1] / height
    xmin = boxes[..., 1:2] / width
    ymax = boxes[..., 2:3] / height
    xmax = boxes[..., 3:4] / width

    normalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return normalized_boxes


def denormalize_boxes_tf(
    boxes: tf.Tensor, image_shape: Union[tuple[int, int], tf.Tensor]
) -> tf.Tensor:
    """Converts boxes normalized by [height, width] to pixel coordinates.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        image_shape: a list of two integers, a two-element vector or a tensor such
            that all but the last dimensions are `broadcastable` to `boxes`. The last
            dimension is 2, which represents [height, width].

    Returns:
        denormalized_boxes: a tensor whose shape is the same as `boxes` representing
            the denormalized boxes.

    Raises:
        ValueError: If the last dimension of boxes is not 4.
    """
    if isinstance(image_shape, (list, tuple)):
        height, width = image_shape
    else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        if image_shape.shape[-1] != 2:
            raise ValueError("Last dimension of image_shape must be 2.")
        height, width = tf.split(image_shape, 2, axis=-1)
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
    ymin = ymin * height
    xmin = xmin * width
    ymax = ymax * height
    xmax = xmax * width
    denormalized_boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)
    return denormalized_boxes


def process_source_id(source_id: tf.Tensor) -> tf.Tensor:
    """Processes source_id from tf.string to integer.

    Args:
        source_id: A `tf.Tensor` that contains the source ID. It can be empty.

    Returns:
        A formatted source ID. -1 if source_id is empty.
    """
    if source_id.dtype == tf.string:
        source_id = tf.strings.to_number(source_id, tf.int64)
    source_id = tf.cond(
        pred=tf.equal(tf.size(input=source_id), 0),
        true_fn=lambda: tf.cast(tf.constant(-1), tf.int64),
        false_fn=lambda: tf.identity(source_id),
    )
    return source_id


def clip_boxes(boxes: tf.Tensor, image_shape: list[int]):
    """Clips boxes to image boundaries.

    Args:
        boxes: a tensor whose last dimension is 4 representing the absolute coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        image_shape: a list of two integers, a two-element vector or a tensor such
            that all but the last dimensions are `broadcastable` to `boxes`. The last
            dimension is 2, which represents [height, width].

    Returns:
        clipped_boxes: a tensor whose shape is the same as `boxes` representing the
            clipped boxes.

    Raises:
        ValueError: If the last dimension of boxes is not 4.
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes.shape[-1] is {boxes.shape[-1]}, but must be 4.")

    if isinstance(image_shape, (list, tuple)):
        height, width = image_shape
        max_len = [height, width, height, width]
    else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        height, width = tf.unstack(image_shape, axis=-1)
        max_len = tf.stack([height, width, height, width], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_len), 0.0)
    return clipped_boxes


def sorted_top_k(values: Tensor, *, k: int):
    """Returns indices of top-k elements sorted in non-increasing order.

    Top-K is performed along the last axis.

    Args:
        values: A [..., c] tensor where c >= k.
        k: Number of top entries.

    Returns:
        Top-K indices along the last axis.
    """
    indices = jnp.argsort(values, axis=-1)
    return indices[..., ::-1][..., :k]


def clip_boxes_jax(boxes: Tensor, image_shape: Tensor) -> Tensor:
    """Clips boxes to image boundaries.

    Args:
        boxes: a tensor whose last dimension is 4 representing the absolute coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        image_shape: a  tensor such that all but the last dimensions are `broadcastable` to
            `boxes`. The last dimension is 2, which represents [height, width].

    Returns:
        clipped_boxes: a tensor whose shape is the same as `boxes` representing the
            clipped boxes.

    Raises:
        ValueError: If the last dimension of boxes is not 4.
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes.shape[-1] is {boxes.shape[-1]}, but must be 4.")

    if isinstance(image_shape, (list, tuple)):
        height, width = image_shape
        max_len = [height, width, height, width]
    else:
        image_shape = image_shape.astype(boxes.dtype)
        height, width = jnp.split(image_shape, 2, axis=-1)
        height = jnp.squeeze(height, axis=-1)
        width = jnp.squeeze(width, axis=-1)
        max_len = jnp.stack([height, width, height, width], axis=-1)

    clipped_boxes = jnp.maximum(jnp.minimum(boxes, max_len), 0.0)
    return clipped_boxes


def resize_and_crop_boxes(
    boxes: tf.Tensor,
    image_scale: tf.Tensor,
    output_size: tf.Tensor,
    offset: tf.Tensor,
) -> tf.Tensor:
    """Resizes boxes to output size with scale and offset.

    Args:
        boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
        image_scale: 2-length float `Tensor` representing scale factors that apply to
            [height, width] of input image.
        output_size: 2-length `Tensor` or `int` representing [height, width] of target
            output image size.
        offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
            boxes.

    Returns:
        boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
    """
    # Adjusts box coordinates based on image_scale and offset.
    boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    # Clips the boxes.
    boxes = clip_boxes(boxes, output_size)
    return boxes


def get_non_empty_box_indices(boxes: tf.Tensor) -> tf.Tensor:
    """Get indices for non-empty boxes.

    Args:
        boxes: `Tensor` of shape [N, 4] representing the input boxes.

    Returns:
        A 1-D tensor of non-empty box indices.
    """
    # Selects indices if box height or width is 0.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
    return indices[:, 0]


def compute_padded_size(desired_size: tuple[int, int], stride: int):
    """Compute the padded size given the desired size and the stride.

    The padded size will be the smallest rectangle, such that each dimension is
    the smallest multiple of the stride which is larger than the desired
    dimension. For example, if desired_size = (100, 200) and stride = 32,
    the output padded_size = (128, 224).

    Args:
        desired_size: a `Tensor` or `int` list/tuple of two elements representing
            [height, width] of the target output image size.
        stride: an integer, the stride of the backbone network.

    Returns:
        padded_size: a `Tensor` or `int` list/tuple of two elements representing
            [height, width] of the padded output image size.
    """
    if isinstance(desired_size, (list, tuple)):
        padded_size = [int(math.ceil(d * 1.0 / stride) * stride) for d in desired_size]
    else:
        padded_size = tf.cast(
            tf.math.ceil(tf.cast(desired_size, dtype=tf.float32) / stride) * stride, tf.int32
        )
    return padded_size


def area(box: tf.Tensor) -> tf.Tensor:
    """Computes area of boxes.
    B: batch_size
    N: number of boxes

    Args:
        box: a float Tensor with [N, 4], or [B, N, 4].

    Returns:
        A float Tensor with [N], or [B, N]
    """
    y_min, x_min, y_max, x_max = tf.split(value=box, num_or_size_splits=4, axis=-1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def intersection(gt_boxes: tf.Tensor, boxes: tf.Tensor) -> tf.Tensor:
    """Compute pairwise intersection areas between boxes.
    B: batch_size
    N: number of groundtruth boxes.
    M: number of anchor boxes.

    Args:
        gt_boxes: a float Tensor with [N, 4], or [B, N, 4]
        boxes: a float Tensor with [M, 4], or [B, M, 4]

    Returns:
        A float Tensor with shape [N, M] or [B, N, M] representing pairwise intersections.
    """
    y_min1, x_min1, y_max1, x_max1 = tf.split(value=gt_boxes, num_or_size_splits=4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxes, num_or_size_splits=4, axis=-1)

    boxes_rank = len(boxes.shape)
    perm = [1, 0] if boxes_rank == 2 else [0, 2, 1]
    # [N, M] or [B, N, M]
    y_min_max = tf.minimum(y_max1, tf.transpose(y_max2, perm))
    y_max_min = tf.maximum(y_min1, tf.transpose(y_min2, perm))
    x_min_max = tf.minimum(x_max1, tf.transpose(x_max2, perm))
    x_max_min = tf.maximum(x_min1, tf.transpose(x_min2, perm))

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min
    zeros_t = tf.cast(0, intersect_heights.dtype)
    intersect_heights = tf.maximum(zeros_t, intersect_heights)
    intersect_widths = tf.maximum(zeros_t, intersect_widths)
    return intersect_heights * intersect_widths


def iou(gt_boxes: tf.Tensor, boxes: tf.Tensor) -> tf.Tensor:
    """Computes pairwise intersection-over-union between box collections.

    Args:
        gt_boxes: a float Tensor with [N, 4].
        boxes: a float Tensor with [M, 4].

    Returns:
        A Tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = intersection(gt_boxes, boxes)
    gt_boxes_areas = area(gt_boxes)
    boxes_areas = area(boxes)
    boxes_rank = len(boxes_areas.shape)
    boxes_axis = 1 if (boxes_rank == 2) else 0
    gt_boxes_areas = tf.expand_dims(gt_boxes_areas, -1)
    boxes_areas = tf.expand_dims(boxes_areas, boxes_axis)
    unions = gt_boxes_areas + boxes_areas
    unions = unions - intersections
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections),
        tf.truediv(intersections, unions),
    )


class IouSimilarity:
    """Class to compute similarity based on Intersection over Union (IOU) metric."""

    def __init__(self, mask_val: int = -1):
        self.mask_val = mask_val

    def __call__(
        self,
        boxes_1: tf.Tensor,
        boxes_2: tf.Tensor,
        boxes_1_masks: tf.Tensor = None,
        boxes_2_masks: tf.Tensor = None,
    ):
        """Compute pairwise IOU similarity between ground truth boxes and anchors.
        B: batch_size
        N: Number of groundtruth boxes.
        M: Number of anchor boxes.

        Args:
            boxes_1: a float Tensor with M or B * M boxes.
            boxes_2: a float Tensor with N or B * N boxes, the rank must be less than
                or equal to rank of `boxes_1`.
            boxes_1_masks: a boolean Tensor with M or B * M boxes. Optional.
            boxes_2_masks: a boolean Tensor with N or B * N boxes. Optional.

        Returns:
            A Tensor with shape [M, N] or [B, M, N] representing pairwise
            iou scores, anchor per row and groundtruth_box per colulmn.

        Input shape:
            boxes_1: [N, 4], or [B, N, 4]
            boxes_2: [M, 4], or [B, M, 4]
            boxes_1_masks: [N, 1], or [B, N, 1]
            boxes_2_masks: [M, 1], or [B, M, 1]

        Output shape:
            [M, N], or [B, M, N]

        Raises:
            ValueError: If input shapes are invalid.
        """
        boxes_1 = tf.cast(boxes_1, tf.float32)
        boxes_2 = tf.cast(boxes_2, tf.float32)

        boxes_1_rank = len(boxes_1.shape)
        boxes_2_rank = len(boxes_2.shape)
        if boxes_1_rank < 2 or boxes_1_rank > 3:
            raise ValueError(f"`groudtruth_boxes` must be rank 2 or 3, got {boxes_1_rank}.")
        if boxes_2_rank < 2 or boxes_2_rank > 3:
            raise ValueError(f"`anchors` must be rank 2 or 3, got {boxes_2_rank}.")
        if boxes_1_rank < boxes_2_rank:
            raise ValueError(
                "`groundtruth_boxes` is unbatched while `anchors` is batched is not a valid use"
                f"case, got groundtruth_box rank {boxes_1_rank}, and anchors rank {boxes_2_rank}."
            )

        result = iou(boxes_1, boxes_2)
        if boxes_1_masks is None and boxes_2_masks is None:
            return result
        background_mask = None
        mask_val_t = tf.cast(self.mask_val, result.dtype) * tf.ones_like(result)
        perm = [1, 0] if boxes_2_rank == 2 else [0, 2, 1]
        if boxes_1_masks is not None and boxes_2_masks is not None:
            background_mask = tf.logical_or(boxes_1_masks, tf.transpose(boxes_2_masks, perm))
        elif boxes_1_masks is not None:
            background_mask = boxes_1_masks
        else:
            background_mask = tf.logical_or(
                tf.zeros(tf.shape(boxes_2)[:-1], dtype=tf.bool), tf.transpose(boxes_2_masks, perm)
            )
        return tf.where(background_mask, mask_val_t, result)


class TargetGather:
    """Target gather for dense object detector."""

    def __call__(
        self,
        labels: tf.Tensor,
        match_indices: tf.Tensor,
        mask: tf.Tensor = None,
        mask_val: int = 0,
    ):
        """Labels anchors with ground truth inputs.
        B: batch_size
        N: number of groundtruth boxes.

        Args:
            labels: An integer tensor with shape [N, dims] or [B, N, ...] representing
                groundtruth labels.
            match_indices: An integer tensor with shape [M] or [B, M] representing
                match label index.
            mask: An boolean tensor with shape [M, dims] or [B, M,...] representing
                match labels.
            mask_val: An integer to fill in for mask.

        Returns:
            target: An integer Tensor with shape [M] or [B, M]

        Raises:
            ValueError: If `labels` is higher than rank 3.
        """
        if len(labels.shape) <= 2:
            return self._gather_unbatched(labels, match_indices, mask, mask_val)
        elif len(labels.shape) == 3:
            return self._gather_batched(labels, match_indices, mask, mask_val)
        else:
            raise ValueError(
                "`TargetGather` does not support `labels` with rank larger than 3,"
                f"got {len(labels.shape)}."
            )

    # pylint: disable-next=no-self-use
    def _gather_unbatched(
        self, labels: tf.Tensor, match_indices: tf.Tensor, mask: tf.Tensor, mask_val: int
    ):
        """Gather based on unbatched labels and boxes."""
        num_gt_boxes = tf.shape(labels)[0]

        def _assign_when_rows_empty():
            if len(labels.shape) > 1:
                mask_shape = [match_indices.shape[0], labels.shape[-1]]
            else:
                mask_shape = [match_indices.shape[0]]
            return tf.cast(mask_val, labels.dtype) * tf.ones(mask_shape, dtype=labels.dtype)

        def _assign_when_rows_not_empty():
            targets = tf.gather(labels, match_indices)
            if mask is None:
                return targets
            else:
                masked_targets = tf.cast(mask_val, labels.dtype) * tf.ones_like(
                    mask, dtype=labels.dtype
                )
                return tf.where(mask, masked_targets, targets)

        return tf.cond(
            tf.greater(num_gt_boxes, 0), _assign_when_rows_not_empty, _assign_when_rows_empty
        )

    def _gather_batched(
        self, labels: tf.Tensor, match_indices: tf.Tensor, mask: tf.Tensor, mask_val: int
    ):
        """Gather based on batched labels."""
        batch_size = labels.shape[0]
        if batch_size == 1:
            if mask is not None:
                result = self._gather_unbatched(
                    tf.squeeze(labels, axis=0),
                    tf.squeeze(match_indices, axis=0),
                    tf.squeeze(mask, axis=0),
                    mask_val,
                )
            else:
                result = self._gather_unbatched(
                    tf.squeeze(labels, axis=0), tf.squeeze(match_indices, axis=0), None, mask_val
                )
            return tf.expand_dims(result, axis=0)
        else:
            indices_shape = tf.shape(match_indices)
            indices_dtype = match_indices.dtype
            batch_indices = tf.expand_dims(
                tf.range(indices_shape[0], dtype=indices_dtype), axis=-1
            ) * tf.ones([1, indices_shape[-1]], dtype=indices_dtype)
            gather_nd_indices = tf.stack([batch_indices, match_indices], axis=-1)
            targets = tf.gather_nd(labels, gather_nd_indices)
            if mask is None:
                return targets
            else:
                masked_targets = tf.cast(mask_val, labels.dtype) * tf.ones_like(
                    mask, dtype=labels.dtype
                )
                return tf.where(mask, masked_targets, targets)


class BoxMatcher:
    """Matcher based on highest value.
    This class computes matches from a similarity matrix. Each column is matched
    to a single row.
    To support object detection target assignment this class enables setting both
    positive_threshold (upper threshold) and negative_threshold (lower thresholds)
    defining three categories of similarity which define whether examples are
    positive, negative, or ignored, for example:
    (1) thresholds=[negative_threshold, positive_threshold], and
        indicators=[negative_value, ignore_value, positive_value]: The similarity
        metrics below negative_threshold will be assigned with negative_value,
        the metrics between negative_threshold and positive_threshold will be
        assigned ignore_value, and the metrics above positive_threshold will be
        assigned positive_value.
    (2) thresholds=[negative_threshold, positive_threshold], and
        indicators=[ignore_value, negative_value, positive_value]: The similarity
        metric below negative_threshold will be assigned with ignore_value,
        the metrics between negative_threshold and positive_threshold will be
        assigned negative_value, and the metrics above positive_threshold will be
        assigned positive_value.
    """

    def __init__(
        self, thresholds: list[float], indicators: list[int], force_match_for_each_col: bool = False
    ):
        """Construct BoxMatcher.

        Args:
            thresholds: A list of thresholds to classify the matches into different
                types (e.g. positive or negative or ignored match). The list needs to be
                sorted, and will be prepended with -Inf and appended with +Inf.
            indicators: A list of values representing match types (e.g. positive or
                negative or ignored match). len(`indicators`) must equal to
                len(`thresholds`) + 1.
            force_match_for_each_col: If True, ensures that each column is matched to
                at least one row (which is not guaranteed otherwise if the
                positive_threshold is high). Defaults to False. If True, all force
                matched row will be assigned to `indicators[-1]`.

        Raises:
            ValueError: If `threshold` not sorted, or len(indicators) != len(threshold) + 1.
        """
        # pylint: disable-next=use-a-generator
        if not all([lo <= hi for (lo, hi) in zip(thresholds[:-1], thresholds[1:])]):
            raise ValueError(f"`threshold` must be sorted, got {thresholds}.")
        self.indicators = indicators
        if len(indicators) != len(thresholds) + 1:
            raise ValueError(
                "len(`indicators`) must be len(`thresholds`) + 1,"
                f"got indicators {indicators}, thresholds {thresholds}."
            )
        thresholds = thresholds[:]
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        self.thresholds = thresholds
        self._force_match_for_each_col = force_match_for_each_col

    def __call__(self, similarity_matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Tries to match each column of the similarity matrix to a row.

        Args:
            similarity_matrix: A float tensor of shape [num_rows, num_cols] or
                [batch_size, num_rows, num_cols] representing any similarity metric.

        Returns:
            matched_columns: An integer tensor of shape [num_rows] or [batch_size, num_rows]
                storing the index of the matched column for each row.
            match_indicators: An integer tensor of shape [num_rows] or [batch_size, num_rows]
                storing the match type indicator (e.g. positive or negative or ignored match).
        """
        squeeze_result = False
        if len(similarity_matrix.shape) == 2:
            squeeze_result = True
            similarity_matrix = tf.expand_dims(similarity_matrix, axis=0)

        static_shape = similarity_matrix.shape.as_list()
        num_rows = static_shape[1] or tf.shape(similarity_matrix)[1]
        batch_size = static_shape[0] or tf.shape(similarity_matrix)[0]

        def _match_when_rows_are_empty():
            """Performs matching when the rows of similarity matrix are empty.
            When the rows are empty, all detections are false positives. So we return
            a tensor of -1's to indicate that the rows do not match to any columns.

            Returns:
                matched_columns: An integer tensor of shape [num_rows] or [batch_size, num_rows]
                    storing the index of the matched column for each row.
                match_indicators: An integer tensor of shape [num_rows] or [batch_size, num_rows]
                    storing the match type indicator (e.g. positive or negative or ignored match).
            """
            with tf.name_scope("empty_gt_boxes"):
                matched_columns = tf.zeros([batch_size, num_rows], dtype=tf.int32)
                match_indicators = tf.negative(tf.ones([batch_size, num_rows], dtype=tf.int32))
                return matched_columns, match_indicators

        def _match_when_rows_are_non_empty():
            """Performs matching when the rows of similarity matrix are non empty.

            Returns:
                matched_columns: An integer tensor of shape [num_rows] or [batch_size, num_rows]
                    storing the index of the matched column for each row.
                match_indicators: An integer tensor of shape [num_rows] or [batch_size, num_rows]
                    storing the match type indicator (e.g. positive or negative or ignored match).
            """
            with tf.name_scope("non_empty_gt_boxes"):
                matched_columns = tf.argmax(similarity_matrix, axis=-1, output_type=tf.int32)

                # Get logical indices of ignored and unmatched columns as tf.int64
                matched_vals = tf.reduce_max(similarity_matrix, axis=-1)
                match_indicators = tf.zeros([batch_size, num_rows], tf.int32)

                match_dtype = matched_vals.dtype
                for ind, low, high in zip(
                    self.indicators, self.thresholds[:-1], self.thresholds[1:]
                ):
                    low_threshold = tf.cast(low, match_dtype)
                    high_threshold = tf.cast(high, match_dtype)
                    mask = tf.logical_and(
                        tf.greater_equal(matched_vals, low_threshold),
                        tf.less(matched_vals, high_threshold),
                    )
                    match_indicators = self._set_values_using_indicator(match_indicators, mask, ind)

                if self._force_match_for_each_col:
                    # [batch_size, num_cols], for each column (groundtruth_box), find the
                    # best matching row (anchor).
                    matching_rows = tf.argmax(input=similarity_matrix, axis=1, output_type=tf.int32)
                    # [batch_size, num_cols, num_rows], a transposed 0-1 mapping matrix M,
                    # where M[j, i] = 1 means column j is matched to row i.
                    column_to_row_match_mapping = tf.one_hot(matching_rows, depth=num_rows)
                    # [batch_size, num_rows], for each row (anchor), find the matched
                    # column (groundtruth_box).
                    force_matched_columns = tf.argmax(
                        input=column_to_row_match_mapping, axis=1, output_type=tf.int32
                    )
                    # [batch_size, num_rows]
                    force_matched_column_mask = tf.cast(
                        tf.reduce_max(column_to_row_match_mapping, axis=1), tf.bool
                    )
                    # [batch_size, num_rows]
                    matched_columns = tf.where(
                        force_matched_column_mask, force_matched_columns, matched_columns
                    )
                    match_indicators = tf.where(
                        force_matched_column_mask,
                        self.indicators[-1] * tf.ones([batch_size, num_rows], dtype=tf.int32),
                        match_indicators,
                    )

                return matched_columns, match_indicators

        num_gt_boxes = similarity_matrix.shape.as_list()[-1] or tf.shape(similarity_matrix)[-1]
        matched_columns, match_indicators = tf.cond(
            pred=tf.greater(num_gt_boxes, 0),
            true_fn=_match_when_rows_are_non_empty,
            false_fn=_match_when_rows_are_empty,
        )

        if squeeze_result:
            matched_columns = tf.squeeze(matched_columns, axis=0)
            match_indicators = tf.squeeze(match_indicators, axis=0)

        return matched_columns, match_indicators

    # pylint: disable-next=no-self-use,invalid-name
    def _set_values_using_indicator(self, x, indicator, val):
        """Set the indicated fields of x to val.

        Args:
            x: tensor.
            indicator: boolean with same shape as x.
            val: scalar with value to set.

        Returns:
            Modified tensor.
        """
        indicator = tf.cast(indicator, x.dtype)
        return tf.add(tf.multiply(x, 1 - indicator), val * indicator)


class BoxList:
    """Box collection."""

    def __init__(self, boxes: tf.Tensor):
        """Constructs box collection.

        Args:
            boxes: a tensor of shape [N, 4] representing box corners

        Raises:
            ValueError: if invalid dimensions for bbox data or if bbox data is not in float32.
        """
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
            raise ValueError("Invalid dimensions for box data.")
        if boxes.dtype != tf.float32:
            raise ValueError("Invalid tensor type: should be tf.float32")
        self.data = {"boxes": boxes}

    def num_boxes(self):
        """Returns number of boxes held in collection.

        Returns:
            A tensor representing the number of boxes held in the collection.
        """
        return tf.shape(input=self.data["boxes"])[0]

    def num_boxes_static(self):
        """Returns number of boxes held in collection.

        This number is inferred at graph construction time rather than run-time.

        Returns:
            Number of boxes held in collection (integer) or None if this is not
            inferable at graph construction time.
        """
        return self.data["boxes"].get_shape().dims[0].value

    def get_all_fields(self):
        """Returns all fields."""
        return self.data.keys()

    def get_extra_fields(self):
        """Returns all non-box fields (i.e., everything not named 'boxes')."""
        # pylint: disable-next=consider-iterating-dictionary
        return [k for k in self.data.keys() if k != "boxes"]

    def add_field(self, field: str, field_data: tf.Tensor):
        """Add field to box list.

        This method can be used to add related box data such as
        weights/labels, etc.

        Args:
            field: a string key to access the data via `get`
            field_data: a tensor containing the data to store in the BoxList
        """
        self.data[field] = field_data

    def has_field(self, field):
        return field in self.data

    def get(self):
        """Convenience function for accessing box coordinates.

        Returns:
            A tensor with shape [N, 4] representing box coordinates.
        """
        return self.get_field("boxes")

    def set(self, boxes: tf.Tensor):
        """Convenience function for setting box coordinates.

        Args:
            boxes: a tensor of shape [N, 4] representing box corners

        Raises:
            ValueError: If invalid dimensions for bbox data.
        """
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
            raise ValueError("Invalid dimensions for box data.")
        self.data["boxes"] = boxes

    def get_field(self, field: str):
        """Accesses a box collection and associated fields.

        This function returns specified field with object; if no field is specified,
        it returns the box coordinates.

        Args:
            field: A string parameter specifying a field to be accessed.

        Returns:
            A tensor representing the box collection or an associated field.

        Raises:
            ValueError: if invalid field
        """
        if not self.has_field(field):
            raise ValueError("field " + str(field) + " does not exist")
        return self.data[field]

    def set_field(self, field: str, value: tf.Tensor):
        """Sets the value of a field.

        Updates the field of a box_list with a given value.

        Args:
            field: The name of the field to set.
            value: The value to assign to the field.

        Raises:
            ValueError: If the box_list does not have specified field.
        """
        if not self.has_field(field):
            raise ValueError(f"field {field} does not exist.")
        self.data[field] = value

    def get_center_coordinates_and_sizes(self, scope: Optional[str] = None):
        """Computes the center coordinates, height and width of the boxes.

        Args:
            scope: name scope of the function.

        Returns:
            A list of 4 1-D tensors [ycenter, xcenter, height, width].
        """
        if not scope:
            scope = "get_center_coordinates_and_sizes"
        with tf.name_scope(scope):
            box_corners = self.get()
            ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(a=box_corners))
            width = xmax - xmin
            height = ymax - ymin
            ycenter = ymin + height / 2.0
            xcenter = xmin + width / 2.0
            return [ycenter, xcenter, height, width]

    def transpose_coordinates(self, scope: Optional[str] = None):
        """Transpose the coordinate representation in a boxlist.

        Args:
            scope: name scope of the function.
        """
        if not scope:
            scope = "transpose_coordinates"
        with tf.name_scope(scope):
            y_min, x_min, y_max, x_max = tf.split(value=self.get(), num_or_size_splits=4, axis=1)
            self.set(tf.concat([x_min, y_min, x_max, y_max], 1))

    def as_tensor_dict(self, fields: Optional[list[str]] = None):
        """Retrieves specified fields as a dictionary of tensors.

        Args:
            fields: (optional) list of fields to return in the dictionary. If None
                (default), all fields are returned.

        Returns:
            tensor_dict: A dictionary of tensors specified by fields.

        Raises:
            ValueError: If specified field is not contained in boxlist.
        """
        tensor_dict = {}
        if fields is None:
            fields = self.get_all_fields()
        for field in fields:
            if not self.has_field(field):
                raise ValueError("boxlist must contain all specified fields")
            tensor_dict[field] = self.get_field(field)
        return tensor_dict


def multi_level_flatten(
    multi_level_inputs: dict[int, Tensor], last_dim: Optional[int] = None
) -> Tensor:
    """Flattens a multi-level input.

    Args:
        multi_level_inputs: Dict with {level: features}.
        last_dim: Whether the output should be [batch_size, None], or [batch_size,
            None, last_dim] if specified.

    Returns:
        Concatenated output [batch_size, None], or [batch_size, None, dim]
    """
    flattened_inputs = []
    batch_size = None
    for level in multi_level_inputs.keys():
        single_input = multi_level_inputs[level]
        if batch_size is None:
            batch_size = single_input.shape[0]
        if last_dim is not None:
            flattened_input = single_input.reshape([batch_size, -1, last_dim])
        else:
            flattened_input = single_input.reshape([batch_size, -1])
        flattened_inputs.append(flattened_input)
    return jnp.concatenate(flattened_inputs, axis=1)


def clip_or_pad_to_fixed_size(
    input_tensor: tf.Tensor, size: int, constant_values: int = 0
) -> tf.Tensor:
    """Pads data to a fixed length at the first dimension.

    Args:
        input_tensor: `Tensor` with any dimension.
        size: specifies the first dimension of output Tensor.
        constant_values: the value assigned to the paddings.

    Returns:
        `Tensor` with the first dimension padded to `size`.
    """
    input_shape = input_tensor.get_shape().as_list()
    padding_shape = []
    # Computes the padding length on the first dimension, clip input tensor if it
    # is longer than `size`.
    input_length = tf.shape(input_tensor)[0]
    input_length = tf.clip_by_value(input_length, 0, size)
    input_tensor = input_tensor[:input_length]
    padding_length = tf.maximum(0, size - input_length)
    padding_shape.append(padding_length)
    # Copies shapes of the rest of input shape dimensions.
    for i in range(1, len(input_shape)):
        padding_shape.append(tf.shape(input_tensor)[i])
    # Pads input tensor to the fixed first dimension.
    paddings = tf.cast(constant_values * tf.ones(padding_shape), input_tensor.dtype)
    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    padded_tensor = tf.concat([input_tensor, paddings], axis=0)
    output_shape = input_shape
    output_shape[0] = size
    padded_tensor.set_shape(output_shape)
    return padded_tensor


def pad_groundtruths_to_fixed_size(
    groundtruths: dict[str, tf.Tensor], size: int
) -> dict[str, tf.Tensor]:
    """Pads the first dimension of groundtruth labels to the fixed size.

    Args:
        groundtruths: a dictionary of {str: tf.Tensor} that contains groundtruth box annotations
            of `boxes`, `is_crowds`, `areas` and `classes` and other image information.
        size: specifies the expected size of the first dimension of padded tensors.

    Returns:
        A dictionary of the same keys and original tensors or padded tensors as values.
        Only box annotations are padded.
    """
    groundtruths["boxes"] = clip_or_pad_to_fixed_size(groundtruths["boxes"], size, -1)
    groundtruths["is_crowds"] = clip_or_pad_to_fixed_size(groundtruths["is_crowds"], size, 0)
    groundtruths["areas"] = clip_or_pad_to_fixed_size(groundtruths["areas"], size, -1)
    groundtruths["classes"] = clip_or_pad_to_fixed_size(groundtruths["classes"], size, -1)
    return groundtruths


def filter_boxes_by_scores(
    boxes: Tensor, scores: Tensor, min_score_threshold: float
) -> tuple[Tensor, ...]:
    """Filters and removes boxes whose scores are smaller than the threshold.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates of
            boxes in ymin, xmin, ymax, xmax order.
        scores: a tensor whose shape is the same as tf.shape(boxes)[:-1]
            representing the original scores of the boxes.
        min_score_threshold: a float representing the minimal box score threshold.
            Boxes whose score are smaller than it will be filtered out.

    Returns:
        filtered_boxes: a tensor whose shape is the same as `boxes` but with
            the position of the filtered boxes are filled with 0.
        filtered_scores: a tensor whose shape is the same as 'scores' but with
            the position of the filtered boxes are filled with -1.

    Raises:
        ValueError: If boxes has an invalid shape.
    """
    if boxes.shape[-1] != 4:
        raise ValueError("The last dim of boxes.shape but must be 4.")

    filtered_mask = jnp.greater(scores, min_score_threshold)
    filtered_scores = jnp.where(filtered_mask, scores, -1 * jnp.ones_like(scores))
    filtered_boxes = jnp.expand_dims(filtered_mask, axis=-1).astype(boxes.dtype) * boxes
    return filtered_boxes, filtered_scores


def yxyx_to_xywh(boxes: np.ndarray):
    """Converts boxes from corner representation (ymin, xmin, ymax, xmax) to corner-size
    representation (xmin, ymin, width, height).

    Args:
        boxes: a numpy array whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.

    Returns:
        boxes: a numpy array whose shape is the same as `boxes` in new format.

    Raises:
        ValueError: If the last dimension of boxes is not 4.
    """
    if boxes.shape[-1] != 4:
        raise ValueError("The last dim of boxes must be 4.")

    boxes_ymin = boxes[..., 0]
    boxes_xmin = boxes[..., 1]
    boxes_width = boxes[..., 3] - boxes[..., 1]
    boxes_height = boxes[..., 2] - boxes[..., 0]
    new_boxes = np.stack([boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=-1)

    return new_boxes


class BoxFormat(str, enum.Enum):
    # pylint: disable=invalid-name
    YminXminYmaxXmax = "YminXminYmaxXmax"  # Default box format in AXLearn detectors.
    XminYminXmaxYmax = "XminYminXmaxYmax"
    YminXminHW = "YminXminHW"  # Default for raw box scores in AXLearn detectors.
    XminYminWH = "XminYminWH"
    CxCyWH = "CxCyWH"
    # pylint: enable=invalid-name


def transform_boxes(boxes: Tensor, source_format: BoxFormat, target_format: BoxFormat) -> Tensor:
    """Transform the box format.

    Args:
        boxes: A [..., 4] float tensor with box coordinates.
        source_format: The source format of the box coordinates.
        target_format: The target format you want to the box coordinatest to be transformed into.

    Raises:
        ValueError: If boxes have an unexpected shape.
        NotImplementedError: When transformation from source_format to target_format is missing.

    Returns:
        A float tensor with same shape as boxes.
    """
    # pylint: disable=invalid-name
    if boxes.shape[-1] != 4:
        raise ValueError("The last dim of boxes must be 4.")
    if source_format == target_format:
        return boxes
    elif (source_format, target_format) == (
        BoxFormat.YminXminYmaxXmax,
        BoxFormat.XminYminXmaxYmax,
    ) or (source_format, target_format) == (BoxFormat.XminYminXmaxYmax, BoxFormat.YminXminYmaxXmax):
        a, b, c, d = jnp.moveaxis(boxes, -1, 0)
        return jnp.stack([b, a, d, c], axis=-1)
    elif (source_format, target_format) == (BoxFormat.YminXminYmaxXmax, BoxFormat.XminYminWH):
        ymin, xmin, ymax, xmax = jnp.moveaxis(boxes, -1, 0)
        return jnp.stack([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    elif (source_format, target_format) == (BoxFormat.XminYminWH, BoxFormat.YminXminHW):
        xmin, ymin, w, h = jnp.moveaxis(boxes, -1, 0)
        return jnp.stack([ymin, xmin, h, w], axis=-1)
    elif (source_format, target_format) == (BoxFormat.YminXminYmaxXmax, BoxFormat.CxCyWH):
        ymin, xmin, ymax, xmax = jnp.moveaxis(boxes, -1, 0)
        return jnp.stack(
            [
                0.5 * (xmin + xmax),
                0.5 * (ymin + ymax),
                xmax - xmin,
                ymax - ymin,
            ],
            axis=-1,
        )
    else:
        raise NotImplementedError(
            f"No box transformation implemented from {source_format=} to {target_format=}"
        )
    # pylint: enable=invalid-name


def reshape_box_decorator(func: Callable) -> Callable:
    """Reshapes boxes before and after calling func.

    Decorates func by reshaping boxes from [..., x * 4] to [..., x, 4] before calling func
    and reverting the reshape after func has been called.

    Args:
        func: Function to be wrapped with box reshapes.

    Returns:
        The wrapped function.
    """

    def reshape_wrapper(boxes: Tensor, **kwargs) -> Tensor:
        boxes_shape = boxes.shape
        boxes = jnp.reshape(boxes, [*boxes_shape[:-1], -1, 4])
        transformed_boxes = func(boxes=boxes, **kwargs)
        return jnp.reshape(transformed_boxes, boxes_shape)

    return reshape_wrapper

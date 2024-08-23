# Copyright © 2023 Apple Inc.

"""Matchers for pairing anchors with groundtruth boxes.

Matchers pair anchors with groundtruths based on similarity matrix. Object Detection models
require assigning anchors/predicted boxes to groundtruth boxes to generate targets and perform
balanced sampling.
"""
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

from axlearn.common.config import Configurable, config_class
from axlearn.common.utils import Tensor
from axlearn.vision import similarity_ops

FOREGROUND = 1
IGNORE = -2
BACKGROUND = -1


@dataclass
class MatchResults:
    """Results of matching.

    Matches for N elements to a set of M elements. In object detection, N is typically the number of
    predictions or anchors while M is the number of groundtruth.

    matches: An int32 tensor of shape [N] where matches[i] is a matched column index in [0, M).
    labels: An int32 tensor of shape [N], where labels[i] indicates the type of match. For example
        {-1, 0, 1} are commonly used labels that map to {ignore, negative, positive} matches
        respectively.
    anchor_boxes: A float32 tensor of shape [1, num_anchors, 4] which contains the list of
        concatenated per level anchor boxes, broadcasted over batch dimension.
    """

    matches: Tensor
    labels: Tensor
    anchor_boxes: Tensor


class Matcher(Configurable):
    """Matcher base class.

    Assigns a groundtruth to each anchor. Each anchor will have exactly one match while each
    groundtruth may be matched to zero or more anchors. Matching is based on the logic
    implemented by the subclass.
    """

    @config_class
    class Config(Configurable.Config):
        # A list of values representing match types (e.g. positive or negative or ignored match).
        labels: list[int] = [BACKGROUND, IGNORE, FOREGROUND]

    def match(self, per_level_anchor_boxes: dict[str, Tensor], groundtruth_boxes: Tensor):
        """Computes groundtruth assignment for anchors based on pairwise similarity.

        Args:
            per_level_anchor_boxes: A dictionary containing anchor boxes of shape [num_anchors, 4]
                per level of the feature pyramid network. The anchor boxes are broadcasted over the
                batch dimension for similarity matching with groundtruth boxes.
            groundtruth_boxes: Tensor [batch_size, num_boxes, 4] containing the groundtruth boxes
                to match with the anchor boxes.

        Returns:
            A MatchResult object containing the matching results.
        """
        raise NotImplementedError(type(self))


def _expand_anchor_dims(*, anchor_boxes: Tensor, groundtruth_boxes: Tensor):
    """Expands anchor boxes along the batch dimension if required.

    Args:
        anchor_boxes: A [batch_size, num_anchors, 4] or [num_anchors, 4] tensor containing
            proposal anchor boxes.
        groundtruth_boxes: A [batch_size, num_boxes, 4] containing the groundtruth boxes
            to match with the anchor boxes.

    Returns:
        Anchor boxes expanded along the batch dimension.

    Raises:
        ValueError: On unexpected dimension of anchor boxes.
    """
    if anchor_boxes.ndim == groundtruth_boxes.ndim:
        return anchor_boxes
    elif anchor_boxes.ndim == groundtruth_boxes.ndim - 1:
        return anchor_boxes[None, ...]
    else:
        raise ValueError(f"Expected anchor boxes dimension is 2 or 3 but got {anchor_boxes.ndim}")


# pylint: disable=unused-argument
class ArgmaxMatcher(Matcher):
    """Argmax matcher.

    Matching is based on pairwise similarities between anchors and groundtruths. Labels
    are assigned FOREGROUND, BACKGROUND or IGNORE labels based on user defined thresholds.
    """

    @config_class
    class Config(Matcher.Config):
        # A list (ascending) of similarity thresholds sorted in ascending order to stratify the
        # matches into different levels.
        thresholds: list[float] = [0.5, 0.5]
        # Ensure that each anchor is matched to at least one groundtruth even though there may be
        # no groundtruths with sufficient similarity.
        force_match_columns: bool = True

    def __init__(self, cfg: Config):
        """Constructs ArgmaxMatcher

        Raises:
            ValueError: On incompatible lengths of `thresholds` and `labels`.
            ValueError: On unsorted thresholds.
        """
        super().__init__(cfg)
        cfg = self.config

        if len(cfg.labels) != len(cfg.thresholds) + 1:
            raise ValueError(
                f"len(labels) must be len(thresholds) + 1, got {len(cfg.labels)} and "
                f"{len(cfg.thresholds)}"
            )
        if sorted(cfg.thresholds) != cfg.thresholds:
            raise ValueError("`thresholds` must be sorted.")

        self.labels = cfg.labels
        self.force_match_columns = cfg.force_match_columns
        self.thresholds = [-float("inf")] + cfg.thresholds + [float("inf")]

    def match(self, per_level_anchor_boxes: dict[str, Tensor], groundtruth_boxes: Tensor):
        """Computes groundtruth assignment for anchors based on pairwise similarity.

        Args:
            per_level_anchor_boxes: A dictionary containing anchor boxes of shape [num_anchors, 4]
                per level of the feature pyramid network. The anchor boxes are broadcasted over the
                batch dimension for similarity matching with groundtruth boxes.
            groundtruth_boxes: Tensor [batch_size, num_boxes, 4] containing the groundtruth boxes
                to match with the anchor boxes.

        Returns:
            A MatchResult object containing the matching results.
        """
        anchor_boxes = jnp.concatenate(list(per_level_anchor_boxes.values()), axis=0)
        # Expand along batch dimension if required.
        anchor_boxes = _expand_anchor_dims(
            anchor_boxes=anchor_boxes, groundtruth_boxes=groundtruth_boxes
        )

        anchor_padding = jnp.amax(anchor_boxes, axis=-1) < 0.0
        groundtruth_padding = jnp.amax(groundtruth_boxes, axis=-1) < 0.0

        # [batch, num_boxes, num_groundtruth_boxes].
        similarity_matrix = similarity_ops.pairwise_iou(
            boxes_a=anchor_boxes,
            boxes_b=groundtruth_boxes,
            paddings_a=anchor_padding,
            paddings_b=groundtruth_padding,
            fill_value=-1.0,
        )

        matches = jnp.argmax(similarity_matrix, axis=-1)
        match_vals = jnp.amax(similarity_matrix, axis=-1)
        match_labels = jnp.zeros_like(matches, dtype=jnp.int32)
        for label, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            match_loc = (match_vals >= low) & (match_vals < high)
            match_labels = jnp.where(match_loc, label, match_labels)

        if self.force_match_columns:
            # [batch_size, num_cols].
            # For each column, finds the best matching row.
            best_rows_for_each_column = jnp.argmax(similarity_matrix, axis=1)
            # [batch_size, num_cols, num_rows]
            # A binary matrix indicating the best matching row for each column.
            column_to_best_rows = jax.nn.one_hot(
                best_rows_for_each_column, num_classes=similarity_matrix.shape[-2]
            )
            # [batch_size, num_rows]
            # Inverts `best_rows_for_each_column` and assigns a column for each row if that row
            # is the column's best match.
            force_matched_column_for_rows = jnp.argmax(column_to_best_rows, axis=-2)
            # [batch_size, num_rows]
            # A binary matrix indicating if each row is the best match for some column.
            force_matched_column_mask = jnp.amax(column_to_best_rows, axis=-2) > 0
            # [batch_size, num_rows]
            # Updates `matches` and `match_labels` for rows that are best matches for some column.
            matches = jnp.where(force_matched_column_mask, force_matched_column_for_rows, matches)
            match_labels = jnp.where(
                force_matched_column_mask,
                self.labels[-1] * jnp.ones_like(match_labels),
                match_labels,
            )
        return MatchResults(matches=matches, labels=match_labels, anchor_boxes=anchor_boxes)


class DistanceType(Enum):
    L2 = 1
    IOU = 2


class ATSSMatcher(Matcher):
    """ATSS matcher.

    Matching is based on pairwise similarities between groundtruths and anchors. Labels are
    assigned based on Adaptive Training Sample Selection (ATSS). This is an implementation of the
    paper https://arxiv.org/pdf/1912.02424.pdf. Instead of user defined thresholds, ATSS
    automatically selects FOREGROUND and BACKGROUND samples according to statistical thresholds.
    """

    @config_class
    class Config(Matcher.Config):
        # Value of k to get the top-k candidate anchor matches per level.
        top_k: int = 9
        # Type of distance to use to calculate the matches.
        distance_type: DistanceType = DistanceType.IOU

    def __init__(self, cfg: Config):
        """Constructs ATSSMatcher"""
        super().__init__(cfg)
        cfg = self.config

        self.labels = cfg.labels
        self.k = cfg.top_k
        self.distance_type = cfg.distance_type

    # pylint: disable=no-self-use
    def _anchor_centers_in_groundtruth_boxes(
        self, *, groundtruth_boxes: Tensor, anchor_boxes: Tensor
    ):
        """Returns a boolean tensor if anchor center is in corresponding groundtruth box.

        Args:
            groundtruth_boxes: A float tensor of shape [batch_size, num_anchors, 4]
                containing anchor matched groundtruth boxes in the form [ymin, xmin, ymax, xmax].
            anchor_boxes: A float tensor of shape [batch_size, num_anchors, 4]
                containing groundtruth boxes in the form [ymin, xmin, ymax, xmax].

        Returns:
            A boolean tensor of shape [batch_size, num_anchors] with a True value of the anchor
                center lies inside the corresponding groundtruth box.
        """
        anchor_boxes_ctr_y = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2
        anchor_boxes_ctr_x = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2

        in_y = (anchor_boxes_ctr_y > groundtruth_boxes[..., 0]) & (
            anchor_boxes_ctr_y < groundtruth_boxes[..., 2]
        )
        in_x = (anchor_boxes_ctr_x > groundtruth_boxes[..., 1]) & (
            anchor_boxes_ctr_x < groundtruth_boxes[..., 3]
        )
        return in_x & in_y

    def _l2_distances(
        self,
        anchor_boxes_ctr_y: Tensor,
        anchor_boxes_ctr_x: Tensor,
        groundtruth_boxes_ctr_y: Tensor,
        groundtruth_boxes_ctr_x: Tensor,
    ):
        """Obtain pair-wise L2 distance between centers of anchor boxes and groundtruth boxes.

        Args:
            anchor_boxes_ctr_x: Tensor [num_anchors] with the x-coordinate of anchor boxes center.
            anchor_boxes_ctr_y: Tensor [num_anchors] with the y-coordinate of anchor boxes center.
            groundtruth_boxes_ctr_x: Tensor [batch_size, num_groundtruth_boxes] with the
                x-coordinate of the centers of groundtruth boxes.
            groundtruth_boxes_ctr_y: Tensor [batch_size, num_groundtruth_boxes] with the
                y-coordinate of the centers of groundtruth boxes.

        Returns:
            Tensor [batch_size, num_groundtruth_boxes, num_anchors] containing L2 distance between
                anchor and groundtruth box center.
        """
        distances = jnp.array(
            [
                anchor_boxes_ctr_x[..., None] - groundtruth_boxes_ctr_x[:, None, ...],
                anchor_boxes_ctr_y[..., None] - groundtruth_boxes_ctr_y[:, None, ...],
            ]
        )
        distances = -1 * jnp.linalg.norm(distances, axis=0)
        distances = jnp.swapaxes(distances, 1, 2)
        return distances

    def _iou_distances(self, pairwise_iou: Tensor, num_anchors: int, anchor_level: int):
        """Extracts the IOU between anchor boxes at given anchor level and groundtruth boxes from
        the IOU matrix.

        Args:
            pairwise_iou: A float tensor of shape [batch_size, num_groundtruth_boxes,
                total_anchors] containing pairwise IOUs between groundtruth boxes and
                anchor boxes.
            num_anchors: Number of anchors at the current anchor level.
            anchor_level: Current level of anchors on the feature pyramid network.

        Returns:
            Tensor [batch_size, num_groundtruth_boxes, num_anchors] containing IOU between
                anchor and groundtruth box center.
        """
        distances = jnp.take_along_axis(
            pairwise_iou,
            (np.array(range(num_anchors)) + anchor_level)[None, None, ...],
            axis=-1,
        )
        return distances

    def match(self, per_level_anchor_boxes: dict[str, Tensor], groundtruth_boxes: Tensor):
        """Computes groundtruth assignment for anchors based on pairwise similarity.

        Args:
            per_level_anchor_boxes: A dictionary containing anchor boxes of shape [num_anchors, 4]
                per level of the feature pyramid network. The anchor boxes are broadcasted over the
                batch dimension for similarity matching with groundtruth boxes.
            groundtruth_boxes: Tensor [batch_size, num_boxes, 4] containing the groundtruth boxes
                to match with the anchor boxes.

        Returns:
            A MatchResult object containing the matching results.

        Raises:
            NotImplementedError: If distance type is unsupported.
        """
        anchor_boxes = jnp.concatenate(list(per_level_anchor_boxes.values()), axis=0)
        # Expand along batch dimension if required.
        anchor_boxes = _expand_anchor_dims(
            anchor_boxes=anchor_boxes, groundtruth_boxes=groundtruth_boxes
        )

        anchor_padding = jnp.amax(anchor_boxes, axis=-1) < 0.0
        groundtruth_padding = jnp.amax(groundtruth_boxes, axis=-1) < 0.0

        # [batch, num_boxes, num_groundtruth_boxes].
        similarity_matrix = similarity_ops.pairwise_iou(
            boxes_a=anchor_boxes,
            boxes_b=groundtruth_boxes,
            paddings_a=anchor_padding,
            paddings_b=groundtruth_padding,
            fill_value=-1.0,
        )

        matches = jnp.argmax(similarity_matrix, axis=-1)
        matched_groundtruth_boxes = jnp.take_along_axis(groundtruth_boxes, matches[..., None], 1)

        # Assign all labels to be negative by default. Assign positive based on threshold below.
        match_labels = jnp.ones_like(matches, dtype=jnp.int32) * self.labels[0]

        groundtruth_boxes_ctr_y = (groundtruth_boxes[..., 0] + groundtruth_boxes[..., 2]) / 2
        groundtruth_boxes_ctr_x = (groundtruth_boxes[..., 1] + groundtruth_boxes[..., 3]) / 2

        anchor_levels = jnp.cumsum(
            np.array([0] + [i.shape[0] for i in per_level_anchor_boxes.values()])
        )

        similarity_matrix = jnp.swapaxes(similarity_matrix, 1, 2)

        candidate_indices = []
        candidate_ious = []
        for index, (_, level_anchor_boxes) in enumerate(per_level_anchor_boxes.items()):
            anchor_boxes_ctr_y = (level_anchor_boxes[..., 0] + level_anchor_boxes[..., 2]) / 2
            anchor_boxes_ctr_x = (level_anchor_boxes[..., 1] + level_anchor_boxes[..., 3]) / 2

            if self.distance_type == DistanceType.L2:
                distances = self._l2_distances(
                    anchor_boxes_ctr_y,
                    anchor_boxes_ctr_x,
                    groundtruth_boxes_ctr_y,
                    groundtruth_boxes_ctr_x,
                )
            elif self.distance_type == DistanceType.IOU:
                distances = self._iou_distances(
                    similarity_matrix, level_anchor_boxes.shape[0], anchor_levels[index]
                )
            else:
                raise NotImplementedError(
                    f"ATSS matcher distance type {self.distance_type} is not implemented."
                )

            _, top_k_indices = jax.lax.top_k(distances, min(self.k, distances.shape[-1]))

            top_k_indices_offset = top_k_indices + anchor_levels[index]
            top_k_ious = jnp.take_along_axis(similarity_matrix, top_k_indices_offset, -1)
            candidate_indices.append(top_k_indices_offset)
            candidate_ious.append(top_k_ious)

        candidate_indices = jnp.concatenate(candidate_indices, axis=-1)
        candidate_ious = jnp.concatenate(candidate_ious, axis=-1)

        mean = jnp.mean(candidate_ious, axis=-1)
        stddev = jnp.std(candidate_ious, axis=-1)
        iou_thresholds = mean + stddev

        anchor_centers_in_groundtruth_boxes = self._anchor_centers_in_groundtruth_boxes(
            groundtruth_boxes=matched_groundtruth_boxes, anchor_boxes=anchor_boxes
        )

        # Positive mask of shape [batch_size, num_boxes, num_candidates] calculated
        # based on IOU thresholds and groundtruth paddings.
        positive_mask = (candidate_ious >= iou_thresholds[..., None]) & ~groundtruth_padding[
            ..., None
        ]

        positive_indices = jnp.where(positive_mask, candidate_indices, -1)
        batch_size = positive_indices.shape[0]
        # Mask of shape [batch_size, num_anchors] which is calculated based on positive mask and
        # if anchor centers lie inside matched groundtruth boxes.
        foreground_mask = (
            jnp.any(
                jax.nn.one_hot(positive_indices.reshape(batch_size, -1), match_labels.shape[1]),
                axis=-2,
            )
            & anchor_centers_in_groundtruth_boxes
        )
        match_labels = jnp.where(foreground_mask, self.labels[-1], match_labels)

        return MatchResults(matches=matches, labels=match_labels, anchor_boxes=anchor_boxes)

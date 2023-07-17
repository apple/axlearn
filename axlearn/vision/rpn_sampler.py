# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""RPN sampler."""
from dataclasses import dataclass

import jax.numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.module import Module
from axlearn.common.utils import Tensor
from axlearn.vision import matchers, samplers

FOREGROUND = 1
IGNORE = -2
BACKGROUND = -1
INVALID = -3


@dataclass
class RPNSamples:
    """Anchor samples along with corresponding proposals boxes, scores and matching groundtruth.

    anchor_boxes: A float tensor of shape [batch, sample_size, 4] containing sampled anchors in the
        form [ymin, xmin, ymax, xmax].
    proposal_boxes: A float tensor of shape [batch, sample_size, 4] containing proposal boxes for
        sampled anchors.
    proposal_scores: A float tensor of shape [batch, sample_size] containing proposal scores for
        sampled anchors.
    groundtruth_boxes: A float tensor of shape [batch, sample_size, 4] containing  matching
        groundtruth boxes for anchors in the form [ymin, xmin, ymax, xmax].
    groundtruth_classes: An int32 tensor of shape [batch, sample_size] containing classes for
        matching groundtruth boxes. The values are in {0, 1} indicating background and foreground
        respectively.
    paddings: A bool tensor of shape [batch, sample_size] indicating whether the samples are
        paddings.
    """

    anchor_boxes: Tensor
    proposal_boxes: Tensor
    proposal_scores: Tensor
    groundtruth_boxes: Tensor
    groundtruth_classes: Tensor
    paddings: Tensor


class RPNSampler(BaseLayer):
    """A class to sample anchors, corresponding proposals and assign groundtruth for RPN."""

    # Config References:
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
    @config_class
    class Config(BaseLayer.Config):
        """Configures RPNSampler."""

        # Number of samples per image.
        sample_size: int = 256
        # Proportion of foreground anchors in the sample.
        foreground_fraction: float = 0.5
        # IoU threshold for an anchor box to be considered as positive, i.e
        # (if IoU >= `foreground_iou_threshold`).
        foreground_iou_threshold: float = 0.7
        # IoU threshold for an anchor box to be considered as negative, i.e
        # (if IoU is in range [`background_iou_low_threshold`, `background_iou_high_threshold`])
        background_iou_high_threshold: float = 0.3
        background_iou_low_threshold: float = 0

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        box_matcher_cfg = matchers.ArgmaxMatcher.default_config()
        box_matcher_cfg.thresholds = [
            self.config.background_iou_low_threshold,
            self.config.background_iou_high_threshold,
            self.config.foreground_iou_threshold,
        ]
        box_matcher_cfg.labels = [INVALID, BACKGROUND, IGNORE, FOREGROUND]
        self.box_matcher = box_matcher_cfg.instantiate()
        self._sampler = samplers.LabelSampler(
            size=self.config.sample_size,
            foreground_fraction=self.config.foreground_fraction,
            background_label=BACKGROUND,
            ignore_label=IGNORE,
        )

    def forward(
        self,
        *,
        anchor_boxes: Tensor,
        proposal_boxes: Tensor,
        proposal_scores: Tensor,
        groundtruth_boxes: Tensor,
        groundtruth_classes: Tensor,
    ) -> RPNSamples:
        """Assigns anchors to groundtruth and returns a subsample.

        Applies the following algorithm to sample anchors and return corresponding proposals and
        matching groundtruth.

            1. Calculates the pairwise IoU between anchor boxes and groundtruth boxes.
            2. For each anchor, assigns a matching groundtruth with highest IoU.
            3. Samples `sample_size` anchors and returns corresponding proposals and matching
               groundtruth boxes and groundtruth classes.

        Args:
            anchor_boxes: A float tensor of shape [num_boxes, 4] containing anchor boxes in the
                form [ymin, xmin, ymax, xmax].
            proposal_boxes: A float tensor of shape of [batch, num_boxes, 4] containing proposal
                boxes.
            proposal_scores: A float tensor of shape of [batch, num_boxes] containing proposal
                scores.
            groundtruth_boxes: A float tensor of shape of [batch_size, num_groundtruth, 4]
                containing groundtruth boxes in the form [ymin, xmin, ymax, xmax]. They must
                be in the same coordinate system as `anchor_boxes`. Values of -1 indicate invalid
                groundtruth that will be ignored.
            groundtruth_classes: An int32 tensor of shape [batch_size, num_groundtruth] containing
                groundtruth classes. Values of -1 indicate invalid groundtruth that will be ignored.

        Returns:
            An RPNSamples object.
        """
        # [batch, num_boxes].
        match_results = self.box_matcher.match(
            per_level_anchor_boxes={"all": anchor_boxes}, groundtruth_boxes=groundtruth_boxes
        )
        foreground_matches = match_results.labels == FOREGROUND
        background_matches = match_results.labels == BACKGROUND
        invalid_matches = match_results.labels == INVALID

        matched_groundtruth_classes = jnp.take_along_axis(
            groundtruth_classes, match_results.matches, axis=-1
        )
        matched_groundtruth_classes = jnp.where(
            ~foreground_matches,
            jnp.zeros_like(matched_groundtruth_classes),
            jnp.ones_like(matched_groundtruth_classes),
        )

        matched_groundtruth_boxes = jnp.take_along_axis(
            groundtruth_boxes, match_results.matches[..., None], axis=-2
        )
        matched_groundtruth_boxes = jnp.where(
            ~foreground_matches[..., None],
            jnp.zeros_like(matched_groundtruth_boxes),
            matched_groundtruth_boxes,
        )

        match_results.matches = jnp.where(
            background_matches | invalid_matches,
            -1 * jnp.ones_like(match_results.matches),
            match_results.matches,
        )

        sample_candidates = foreground_matches | background_matches
        samples = self._sampler(
            labels=match_results.matches, paddings=~sample_candidates, prng_key=self.prng_key
        )
        return RPNSamples(
            anchor_boxes=jnp.take_along_axis(
                match_results.anchor_boxes,
                # pylint: disable-next=unsubscriptable-object
                samples.indices[..., None],
                axis=-2,
            ),
            # pylint: disable-next=unsubscriptable-object
            proposal_boxes=jnp.take_along_axis(proposal_boxes, samples.indices[..., None], axis=-2),
            proposal_scores=jnp.take_along_axis(proposal_scores, samples.indices, axis=-1),
            groundtruth_boxes=jnp.take_along_axis(
                matched_groundtruth_boxes,
                # pylint: disable-next=unsubscriptable-object
                samples.indices[..., None],
                axis=-2,
            ),
            groundtruth_classes=jnp.take_along_axis(
                matched_groundtruth_classes, samples.indices, axis=-1
            ),
            paddings=samples.paddings,
        )

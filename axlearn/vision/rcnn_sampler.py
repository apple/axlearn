# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""RCNN sampler."""
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
class RCNNSamples:
    """Proposal samples along with corresponding scores and matching groundtruth.

    proposal_boxes: A float tensor of shape [batch, num_boxes, 4] containing sampled proposal
        boxes in the form [ymin, xmin, ymax, xmax].
    groundtruth_boxes: A float tensor of shape [batch, sample_size, 4] containing  matching
        groundtruth boxes for proposal boxes in the form [ymin, xmin, ymax, xmax].
    groundtruth_classes: An int32 tensor of shape [batch, sample_size] containing classes for
        matching groundtruth boxes. The values are in [0, num_groundtruth].
    paddings: A bool tensor of shape [batch, sample_size] indicating whether the samples are
        paddings.
    """

    proposal_boxes: Tensor
    groundtruth_boxes: Tensor
    groundtruth_classes: Tensor
    paddings: Tensor


class RCNNSampler(BaseLayer):
    """A class to sample proposals and assign groundtruth for second stage of Faster R-CNN."""

    # Config References:
    # https://github.com/tensorflow/models/blob/master/official/vision/configs/maskrcnn.py
    @config_class
    class Config(BaseLayer.Config):
        """Configures RCNNSampler."""

        # Whether to use groundtruth boxes as proposals in addition to `proposal_boxes`.
        # Enabling this option, improves the stability of Faster R-CNN training during initial
        # stages as well as the overall end performance.
        mix_groundtruth_boxes: bool = True
        # Number of samples per image.
        sample_size: int = 512
        # Proportion of foreground RoIs in the sample.
        foreground_fraction: float = 0.25
        # IoU threshold for a box to be considered as positive (if >= `foreground_iou_threshold`).
        foreground_iou_threshold: float = 0.5
        # IoU threshold for a box to be considered as negative (if overlap in
        # [`background_iou_low_threshold`, `background_iou_high_threshold`])
        background_iou_high_threshold: float = 0.5
        background_iou_low_threshold: float = 0
        # Whether to subsample foreground and background proposals.
        apply_subsampling: bool = True

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
        proposal_boxes: Tensor,
        groundtruth_boxes: Tensor,
        groundtruth_classes: Tensor,
    ) -> RCNNSamples:
        """Assigns proposals to groundtruth and returns a subsample.

        Given `proposal_boxes`, `groundtruth_boxes` and `groundtruth_classes` applies the following
        algorithm:

            1. Calculates the pairwise IoU between each proposal boxes and groundtruth boxes.
            2. For each proposal, assigns a matching groundtruth with highest IoU.
            3. Samples `sample_size` proposals and returns matching groundtruth boxes and
               groundtruth classes.

        Args:
            proposal_boxes: A float tensor of shape of [batch, num_boxes, 4] containing proposal
                boxes in the form [ymin, xmin, ymax, xmax].
            groundtruth_boxes: A float tensor of shape of [batch_size, num_groundtruth, 4]
                containing groundtruth boxes in the form [ymin, xmin, ymax, xmax]. They must
                be in the same coordinate system as `proposal_boxes`. Values of -1 indicate
                invalid groundtruth that will be ignored.
            groundtruth_classes: An int32 tensor of shape [batch_size, num_groundtruth] containing
                groundtruth classes. Values of -1 indicate invalid groundtruth that will be ignored.

        Returns:
            An RCNNSamples object.
        """
        if self.config.mix_groundtruth_boxes:
            proposal_boxes = jnp.concatenate([proposal_boxes, groundtruth_boxes], axis=1)

        # [batch, num_boxes].
        match_results = self.box_matcher.match(
            per_level_anchor_boxes={"all": proposal_boxes}, groundtruth_boxes=groundtruth_boxes
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
            matched_groundtruth_classes,
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
        if self.config.apply_subsampling:
            sample_candidates = foreground_matches | background_matches
            samples = self._sampler(
                labels=match_results.matches, paddings=~sample_candidates, prng_key=self.prng_key
            )
            # pylint: disable=unsubscriptable-object
            return RCNNSamples(
                proposal_boxes=jnp.take_along_axis(
                    proposal_boxes, samples.indices[..., None], axis=-2
                ),
                groundtruth_boxes=jnp.take_along_axis(
                    matched_groundtruth_boxes, samples.indices[..., None], axis=-2
                ),
                groundtruth_classes=jnp.take_along_axis(
                    matched_groundtruth_classes, samples.indices, axis=-1
                ),
                paddings=samples.paddings,
            )
            # pylint: enable=unsubscriptable-object
        else:
            return RCNNSamples(
                proposal_boxes=proposal_boxes,
                groundtruth_boxes=matched_groundtruth_boxes,
                groundtruth_classes=matched_groundtruth_classes,
                paddings=~(foreground_matches | background_matches),
            )

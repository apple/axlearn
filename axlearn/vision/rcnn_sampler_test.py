# Copyright © 2023 Apple Inc.

"""Tests RCNN sampler."""
import jax
import numpy as np
from absl.testing import absltest

from axlearn.common.module import functional as F
from axlearn.vision import rcnn_sampler, similarity_ops


def boxes_close_as_set(actual_boxes, expected_boxes):
    p_iou = similarity_ops.pairwise_iou(boxes_a=actual_boxes, boxes_b=expected_boxes)
    np.testing.assert_allclose(1.0, np.amax(p_iou, axis=-1))


def padded_boxes_close_as_set(actual_boxes, expected_boxes):
    p_iou = similarity_ops.pairwise_iou(boxes_a=actual_boxes, boxes_b=expected_boxes)
    ious = np.where(np.amax(actual_boxes, axis=-1) > 0, np.amax(p_iou, axis=-1), 1.0)
    np.testing.assert_allclose(1.0, ious)


# pylint: disable=no-self-use
class RCNNSamplerTest(absltest.TestCase):
    """Tests RCNNSampler."""

    def test_fg_bg_matching_and_sampling(self):
        cfg = rcnn_sampler.RCNNSampler.default_config().set(
            name="rcnn_sampler",
            mix_groundtruth_boxes=False,
            sample_size=4,
            foreground_fraction=0.25,
            foreground_iou_threshold=0.5,
            background_iou_high_threshold=0.5,
            background_iou_low_threshold=0,
            apply_subsampling=True,
        )
        sampler = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = sampler.initialize_parameters_recursively(init_key)

        proposal_boxes = np.array(
            [
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.5, 0.5, 1.0],
                    [0.5, 0.0, 1.0, 0.5],
                    [0.5, 0.5, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.5, 1.0, 1.0],
                    [0.0, 0.0, 0.5, 1.0],
                    [0.5, 0.0, 1.0, 1.0],
                ]
            ]
        )
        groundtruth_boxes = np.array([[[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 1.0, 0.5]]])
        groundtruth_classes = np.array([[1, 3]])
        rcnn_samples, _ = F(
            sampler,
            inputs={
                "proposal_boxes": proposal_boxes,
                "groundtruth_boxes": groundtruth_boxes,
                "groundtruth_classes": groundtruth_classes,
            },
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        np.testing.assert_array_equal([1], np.sum(rcnn_samples.groundtruth_classes > 0, axis=-1))
        np.testing.assert_array_equal([3], np.sum(rcnn_samples.groundtruth_classes == 0, axis=-1))
        np.testing.assert_array_equal(False, rcnn_samples.paddings)
        e_iou = similarity_ops.elementwise_iou(
            boxes_a=rcnn_samples.proposal_boxes, boxes_b=rcnn_samples.groundtruth_boxes
        )
        foreground_iou = e_iou[rcnn_samples.groundtruth_classes > 0]
        background_iou = e_iou[rcnn_samples.groundtruth_classes == 0]
        np.testing.assert_array_equal(foreground_iou >= 0.5, True)
        np.testing.assert_array_less(background_iou, 0.5)
        boxes_close_as_set(rcnn_samples.proposal_boxes, proposal_boxes)
        padded_boxes_close_as_set(rcnn_samples.groundtruth_boxes, groundtruth_boxes)

    def test_fg_bg_ignore_matching_and_sampling(self):
        cfg = rcnn_sampler.RCNNSampler.default_config().set(
            name="rcnn_sampler",
            mix_groundtruth_boxes=False,
            sample_size=4,
            foreground_fraction=0.5,
            foreground_iou_threshold=0.5,
            background_iou_high_threshold=0.4,
            background_iou_low_threshold=0,
            apply_subsampling=True,
        )
        sampler = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = sampler.initialize_parameters_recursively(init_key)
        proposal_boxes = np.array(
            [
                [
                    [0.0, 0.0, 0.5, 0.5],  # 0.5 IoU,  groundtruth 0
                    [0.0, 0.5, 0.5, 1.0],  # 0.5 IoU,  groundtruth 0
                    [0.5, 0.0, 1.0, 0.5],  # 0.5 IoU,  groundtruth 1
                    [0.55, 0.0, 1.0, 0.5],  # 0.45 IoU, groundtruth 1
                    [0.5, 0.5, 1.0, 1.0],  # 0.0 IoU,  None
                    [0.0, 0.0, 1.0, 0.5],  # 1.0 IoU,  groundtruth 1
                    [0.0, 0.5, 1.0, 1.0],  # 1/3 IoU,  groundtruth 0
                    [0.0, 0.0, 0.5, 1.0],  # 1.0 IoU,  groundtruth 0
                    [0.5, 0.0, 1.0, 1.0],  # 1/3 IoU,  groundtruth 1
                ]
            ]
        )
        groundtruth_boxes = np.array([[[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 1.0, 0.5]]])
        groundtruth_classes = np.array([[1, 3]])
        rcnn_samples, _ = F(
            sampler,
            inputs={
                "proposal_boxes": proposal_boxes,
                "groundtruth_boxes": groundtruth_boxes,
                "groundtruth_classes": groundtruth_classes,
            },
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # check 0.3
        np.testing.assert_array_equal([2], np.sum(rcnn_samples.groundtruth_classes > 0, axis=-1))
        np.testing.assert_array_equal([2], np.sum(rcnn_samples.groundtruth_classes == 0, axis=-1))
        np.testing.assert_array_equal(False, rcnn_samples.paddings)
        e_iou = similarity_ops.elementwise_iou(
            boxes_a=rcnn_samples.proposal_boxes, boxes_b=rcnn_samples.groundtruth_boxes
        )
        foreground_iou = e_iou[rcnn_samples.groundtruth_classes > 0]
        background_iou = e_iou[rcnn_samples.groundtruth_classes == 0]
        np.testing.assert_array_equal(foreground_iou >= 0.5, True)
        np.testing.assert_array_less(background_iou, 0.4)
        boxes_close_as_set(rcnn_samples.proposal_boxes, proposal_boxes)
        padded_boxes_close_as_set(rcnn_samples.groundtruth_boxes, groundtruth_boxes)

    def test_mix_groundtruth_matching_and_sampling(self):
        cfg = rcnn_sampler.RCNNSampler.default_config().set(
            name="rcnn_sampler",
            mix_groundtruth_boxes=True,
            sample_size=4,
            foreground_fraction=0.25,
            foreground_iou_threshold=0.5,
            background_iou_high_threshold=0.5,
            background_iou_low_threshold=0,
            apply_subsampling=True,
        )
        sampler = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = sampler.initialize_parameters_recursively(init_key)
        proposal_boxes = np.array(
            [
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.5, 0.5, 1.0],
                    [0.5, 0.0, 1.0, 0.5],
                ]
            ]
        )
        groundtruth_boxes = np.array([[[0.5, 0.5, 1.0, 1.0]]])
        groundtruth_classes = np.array([[1]])
        rcnn_samples, _ = F(
            sampler,
            inputs={
                "proposal_boxes": proposal_boxes,
                "groundtruth_boxes": groundtruth_boxes,
                "groundtruth_classes": groundtruth_classes,
            },
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        np.testing.assert_array_equal([1], np.sum(rcnn_samples.groundtruth_classes > 0, axis=-1))
        np.testing.assert_array_equal([3], np.sum(rcnn_samples.groundtruth_classes == 0, axis=-1))
        np.testing.assert_array_equal(False, rcnn_samples.paddings)
        e_iou = similarity_ops.elementwise_iou(
            boxes_a=rcnn_samples.proposal_boxes, boxes_b=rcnn_samples.groundtruth_boxes
        )
        foreground_iou = e_iou[rcnn_samples.groundtruth_classes > 0]
        background_iou = e_iou[rcnn_samples.groundtruth_classes == 0]
        np.testing.assert_array_equal(foreground_iou >= 0.5, True)
        np.testing.assert_array_less(background_iou, 0.5)
        boxes_close_as_set(
            rcnn_samples.proposal_boxes,
            np.concatenate([proposal_boxes, groundtruth_boxes], axis=-2),
        )
        padded_boxes_close_as_set(rcnn_samples.groundtruth_boxes, groundtruth_boxes)

    def test_skip_sampling(self):
        cfg = rcnn_sampler.RCNNSampler.default_config().set(
            name="rcnn_sampler",
            mix_groundtruth_boxes=False,
            sample_size=4,
            foreground_fraction=0.5,
            foreground_iou_threshold=0.5,
            background_iou_high_threshold=0.3,
            background_iou_low_threshold=0,
            apply_subsampling=False,
        )
        sampler = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = sampler.initialize_parameters_recursively(init_key)
        proposal_boxes = np.array(
            [
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.5, 0.5, 1.0],
                    [0.5, 0.0, 1.0, 0.5],
                    [0.5, 0.5, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.5, 1.0, 1.0],
                    [0.0, 0.0, 0.5, 1.0],
                    [0.5, 0.0, 1.0, 1.0],
                ]
            ]
        )
        groundtruth_boxes = np.array([[[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 1.0, 0.5]]])
        groundtruth_classes = np.array([[1, 3]])
        rcnn_samples, _ = F(
            sampler,
            inputs={
                "proposal_boxes": proposal_boxes,
                "groundtruth_boxes": groundtruth_boxes,
                "groundtruth_classes": groundtruth_classes,
            },
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        prng_key = jax.random.PRNGKey(123)
        np.testing.assert_allclose(proposal_boxes, rcnn_samples.proposal_boxes)
        padded_boxes_close_as_set(rcnn_samples.groundtruth_boxes, groundtruth_boxes)
        np.testing.assert_array_equal([[1, 1, 3, 0, 3, 0, 1, 0]], rcnn_samples.groundtruth_classes)
        np.testing.assert_allclose(
            [
                [
                    [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ],
            rcnn_samples.groundtruth_boxes,
        )
        np.testing.assert_array_equal(
            [[False, False, False, False, False, True, False, True]], rcnn_samples.paddings
        )

# Copyright © 2023 Apple Inc.

"""Tests RPN sampler."""
import jax
import numpy as np
from absl.testing import absltest

from axlearn.common.module import functional as F
from axlearn.vision import rpn_sampler, similarity_ops


def boxes_close_as_set(actual_boxes, expected_boxes):
    p_iou = similarity_ops.pairwise_iou(boxes_a=actual_boxes, boxes_b=expected_boxes)
    np.testing.assert_allclose(1.0, np.amax(p_iou, axis=-1))


def padded_boxes_close_as_set(actual_boxes, expected_boxes):
    p_iou = similarity_ops.pairwise_iou(boxes_a=actual_boxes, boxes_b=expected_boxes)
    ious = np.where(np.amax(actual_boxes, axis=-1) > 0, np.amax(p_iou, axis=-1), 1.0)
    np.testing.assert_allclose(1.0, ious)


# pylint: disable=no-self-use
class RPNSamplerTest(absltest.TestCase):
    """Tests RPNSampler."""

    def test_fg_bg_matching_and_sampling(self):
        cfg = rpn_sampler.RPNSampler.default_config().set(
            name="rpn_sampler",
            sample_size=4,
            foreground_fraction=0.5,
            foreground_iou_threshold=0.5,
            background_iou_high_threshold=0.4,
            background_iou_low_threshold=0,
        )
        sampler = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = sampler.initialize_parameters_recursively(init_key)

        anchor_boxes = np.array(
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
        )
        groundtruth_boxes = np.array([[[0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 1.0, 0.5]]])
        groundtruth_classes = np.array([[5, 3]])

        proposal_ymin = np.random.random((1, 9))
        proposal_xmin = np.random.random((1, 9))
        proposal_ymax = np.clip(proposal_ymin + np.random.random((1, 9)), a_min=0.0, a_max=1.0)
        proposal_xmax = np.clip(proposal_xmin + np.random.random((1, 9)), a_min=0.0, a_max=1.0)
        proposal_boxes = np.stack(
            [proposal_ymin, proposal_xmin, proposal_ymax, proposal_xmax], axis=-1
        )
        proposal_scores = np.random.random((1, 9))

        rpn_samples, _ = F(
            sampler,
            inputs={
                "anchor_boxes": anchor_boxes,
                "proposal_boxes": proposal_boxes,
                "proposal_scores": proposal_scores,
                "groundtruth_boxes": groundtruth_boxes,
                "groundtruth_classes": groundtruth_classes,
            },
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        np.testing.assert_array_equal([2], np.sum(rpn_samples.groundtruth_classes == 1, axis=-1))
        np.testing.assert_array_equal([2], np.sum(rpn_samples.groundtruth_classes == 0, axis=-1))
        np.testing.assert_array_equal(False, rpn_samples.paddings)
        e_iou = similarity_ops.elementwise_iou(
            boxes_a=rpn_samples.anchor_boxes, boxes_b=rpn_samples.groundtruth_boxes
        )
        foreground_iou = e_iou[rpn_samples.groundtruth_classes > 0]
        background_iou = e_iou[rpn_samples.groundtruth_classes == 0]
        np.testing.assert_array_equal(foreground_iou >= 0.5, True)
        np.testing.assert_array_less(background_iou, 0.5)
        boxes_close_as_set(rpn_samples.proposal_boxes, proposal_boxes)
        padded_boxes_close_as_set(rpn_samples.groundtruth_boxes, groundtruth_boxes)

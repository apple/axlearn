# Copyright © 2023 Apple Inc.

"""Tests for roi_generator.py."""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from axlearn.common.module import functional as F
from axlearn.vision import anchor
from axlearn.vision.roi_generator import RoIGenerator


class RoIGeneratorTest(absltest.TestCase):
    """Tests RoIGenerator."""

    def test_proposals_with_nms(self):
        min_level = 5
        max_level = 6
        num_scales = 2
        max_num_proposals = 10
        aspect_ratios = (1.0, 2.0)
        anchor_scale = 2.0
        output_size = (64, 64)
        pre_nms_top_k = 5000
        pre_nms_score_threshold = 0.01
        batch_size = 1

        input_anchor = anchor.AnchorGenerator(
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_scale,
        )
        anchor_boxes = input_anchor(output_size)
        anchor_boxes = {k: np.asarray(v) for k, v in anchor_boxes.items()}
        cls_outputs_all = np.random.rand(20)
        box_outputs_all = np.random.rand(20, 4)

        class_output_dim = num_scales * len(aspect_ratios)
        class_outputs = {
            5: jnp.reshape(
                jnp.asarray(cls_outputs_all[0:16], dtype=jnp.float32),
                [1, 2, 2, class_output_dim],
            ),
            6: jnp.reshape(
                jnp.asarray(cls_outputs_all[16:20], dtype=jnp.float32),
                [1, 1, 1, class_output_dim],
            ),
        }
        box_output_dim = num_scales * len(aspect_ratios) * 4
        box_outputs = {
            5: jnp.reshape(
                jnp.asarray(box_outputs_all[0:16], dtype=jnp.float32), [1, 2, 2, box_output_dim]
            ),
            6: jnp.reshape(
                jnp.asarray(box_outputs_all[16:20], dtype=jnp.float32), [1, 1, 1, box_output_dim]
            ),
        }
        image_info = jnp.asarray(
            [[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]], dtype=jnp.float32
        )

        cfg = RoIGenerator.default_config().set(
            name="roi_generator",
            pre_nms_top_k=pre_nms_top_k,
            pre_nms_score_threshold=pre_nms_score_threshold,
            nms_iou_threshold=0.5,
            max_num_proposals=max_num_proposals,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=(box_outputs, class_outputs, anchor_boxes, image_info[:, 1, :]),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        boxes = outputs["proposal_boxes"]
        scores = outputs["proposal_scores"]
        valid_detections = outputs["num_proposals"]

        self.assertEqual(boxes.shape, (batch_size, max_num_proposals, 4))
        self.assertEqual(scores.shape, (batch_size, max_num_proposals))
        self.assertEqual(valid_detections.shape, (batch_size,))

    def test_proposals_without_nms(self):
        min_level = 5
        max_level = 6
        num_scales = 2
        aspect_ratios = (1.0, 2.0)
        anchor_scale = 2.0
        output_size = (64, 64)
        batch_size = 1

        input_anchor = anchor.AnchorGenerator(
            min_level=min_level,
            max_level=max_level,
            num_scales=num_scales,
            aspect_ratios=aspect_ratios,
            anchor_size=anchor_scale,
        )
        anchor_boxes = input_anchor(output_size)
        anchor_boxes = {k: np.asarray(v) for k, v in anchor_boxes.items()}
        cls_outputs_all = np.random.rand(20)
        box_outputs_all = np.random.rand(20, 4)

        class_output_dim = num_scales * len(aspect_ratios)
        class_outputs = {
            5: jnp.reshape(
                jnp.asarray(cls_outputs_all[0:16], dtype=jnp.float32),
                [1, 2, 2, class_output_dim],
            ),
            6: jnp.reshape(
                jnp.asarray(cls_outputs_all[16:20], dtype=jnp.float32),
                [1, 1, 1, class_output_dim],
            ),
        }
        box_output_dim = num_scales * len(aspect_ratios) * 4
        box_outputs = {
            5: jnp.reshape(
                jnp.asarray(box_outputs_all[0:16], dtype=jnp.float32), [1, 2, 2, box_output_dim]
            ),
            6: jnp.reshape(
                jnp.asarray(box_outputs_all[16:20], dtype=jnp.float32), [1, 1, 1, box_output_dim]
            ),
        }
        image_info = jnp.asarray(
            [[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]], dtype=jnp.float32
        )

        cfg = RoIGenerator.default_config().set(
            name="roi_generator",
            nms_iou_threshold=1.0,
        )
        generator = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = generator.initialize_parameters_recursively(init_key)

        outputs, _ = F(
            generator,
            inputs=(box_outputs, class_outputs, anchor_boxes, image_info[:, 1, :]),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        boxes = outputs["proposal_boxes"]
        scores = outputs["proposal_scores"]
        valid_detections = outputs["num_proposals"]
        total_proposals = 20
        self.assertEqual(boxes.shape, (batch_size, total_proposals, 4))
        self.assertEqual(scores.shape, (batch_size, total_proposals))
        self.assertEqual(valid_detections.shape, (batch_size,))


if __name__ == "__main__":
    absltest.main()

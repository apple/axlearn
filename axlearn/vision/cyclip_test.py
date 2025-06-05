# Copyright © 2023 Apple Inc.

"""Tests CyCLIP implementation."""
# pylint: disable=no-self-use

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.vision.clip import CLIPFusionNetwork
from axlearn.vision.cyclip import CyCLIPFusionNetwork


class TestCyCLIPFusionNetwork(parameterized.TestCase):
    """Tests CyCLIPFusionNetwork."""

    def _compare_against_clip_model(
        self,
        cyclip_fusion_network,
        clip_fusion_network,
        batch_size,
        embedding_dim,
        num_images,
        num_texts,
        cross_modal_weight,
        in_modal_weight,
    ):
        img_emb = jnp.array(np.random.random((batch_size, num_images, embedding_dim)))
        txt_emb = jnp.array(np.random.random((batch_size, num_texts, embedding_dim)))
        input_data = {
            "input_batch": {
                "visual_encoder": {"output_features": img_emb},
                "textual_encoder": {"output_features": txt_emb},
            }
        }

        cyclip_outputs, cyclip_metrics_dict = F(
            cyclip_fusion_network,
            inputs=input_data,
            state=cyclip_fusion_network.initialize_parameters_recursively(jax.random.PRNGKey(0)),
            is_training=True,
            prng_key=None,
        )
        clip_outputs, _ = F(
            clip_fusion_network,
            inputs=input_data,
            state=clip_fusion_network.initialize_parameters_recursively(jax.random.PRNGKey(0)),
            is_training=True,
            prng_key=None,
        )
        assert_allclose(
            cyclip_outputs[1]["similarity"],
            clip_outputs[1]["similarity"],
        )
        assert_allclose(
            cyclip_outputs[0] - clip_outputs[0],
            cross_modal_weight * cyclip_metrics_dict.summaries["loss_cross_modal"]
            + in_modal_weight * cyclip_metrics_dict.summaries["loss_in_modal"],
        )

    def test_cyclip_fusion_model(self):
        # We test if the step works correctly.
        batch_size = 3
        dim = 3
        num_images = 2
        num_text = 2
        cross_modal_weight = 0
        in_modal_weight = 0

        clip_fusion_cfg = CLIPFusionNetwork.default_config().set(
            name="clip_fusion_network",
        )
        clip_fusion_net = clip_fusion_cfg.instantiate(parent=None)
        cyclip_fusion_cfg = CyCLIPFusionNetwork.default_config().set(
            name="cyclip_fusion_network",
            cross_modal_weight=cross_modal_weight,
            in_modal_weight=in_modal_weight,
            clip_fusion_network=clip_fusion_cfg,
        )
        cyclip_fusion_net = cyclip_fusion_cfg.instantiate(parent=None)

        self._compare_against_clip_model(
            cyclip_fusion_net,
            clip_fusion_net,
            batch_size,
            dim,
            num_images,
            num_text,
            cross_modal_weight,
            in_modal_weight,
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

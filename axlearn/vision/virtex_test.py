# Copyright © 2023 Apple Inc.

"""Tests VirTex implementations."""

# pylint: disable=no-self-use
from collections.abc import Sequence

import jax.numpy as jnp
import jax.random
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from axlearn.common.decoding import BeamSearchOutputs
from axlearn.common.module import functional as F
from axlearn.vision.virtex import VirTexModel


def generate_sample_captioning_inputs(
    batch_size: int = 2,
    image_size: Sequence[int] = (224, 224, 3),
    seq_len: int = 10,
    vocab_size: int = 1000,
):
    return {
        "image": np.random.uniform(-1, 1, (batch_size,) + tuple(image_size)).astype(np.float32),
        "caption_tokens": np.random.randint(0, vocab_size - 1, [batch_size, seq_len]).astype(
            np.int32
        ),
    }


class VirTexTest(parameterized.TestCase):
    """Tests VirTexModel."""

    def _add_textual_config(
        self,
        cfg,
        seq_len: int = 10,
        vocab_size: int = 1000,
        hidden_dim: int = 32,
        num_layers: int = 2,
        self_attn_num_heads: int = 2,
        cross_attn_num_heads: int = 2,
        eos_token_id: int = 1,
    ):
        cfg.textual.vocab_size = vocab_size
        cfg.textual.emb.token_emb.num_embeddings = seq_len
        cfg.textual.dim = hidden_dim
        cfg.textual.transformer.num_layers = num_layers
        cfg.textual.transformer.layer.self_attention.attention.num_heads = self_attn_num_heads
        cfg.textual.transformer.layer.cross_attention.source_dim = cfg.textual.dim
        cfg.textual.transformer.layer.cross_attention.attention.num_heads = cross_attn_num_heads
        cfg.textual.transformer.layer.feed_forward.hidden_dim = 32
        cfg.textual.eos_token_id = eos_token_id

    def _base_aux_outputs_test(self, cfg):
        """Verify that appropriate auxiliary outputs are returned.

        Implicitly this also tests the forward pass executes without error.

        The auxiliary outputs for VirTex include:
            - 'visual_features': Outputs of the visual backbone.
        """
        model: VirTexModel = cfg.set(name="test_intermediate_features").instantiate(parent=None)

        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        input_batch = generate_sample_captioning_inputs()
        input_batch["image"] = jnp.array(input_batch["image"]).astype(cfg.visual.dtype)

        (_, aux), _ = F(
            model,
            state=state,
            inputs=dict(input_batch=input_batch, return_aux=True),
            prng_key=jax.random.PRNGKey(123),
            is_training=False,
        )
        self.assertIn("visual_features", aux)

    @parameterized.parameters("resnet18", "resnet50")
    def test_aux_outputs_resnet(self, resnet_name):
        cfg = VirTexModel.resnet_backbone_config(resnet_name)
        self._add_textual_config(cfg)
        self._base_aux_outputs_test(cfg)

    @parameterized.parameters("B16", "B32", "L16", "L32", "H14", "G14")
    # TODO(markblee): Re-enable in CI when we have access to a larger instance.
    @pytest.mark.high_cpu
    def test_aux_outputs_vit(self, vit_name: str):
        cfg = VirTexModel.vit_backbone_config(vit_name)
        self._add_textual_config(cfg)
        self._base_aux_outputs_test(cfg)

    @parameterized.parameters(1, 5)
    def test_caption(self, num_decodes: int):
        batch_size, seq_len = 2, 10
        bos_id = eos_id = 1
        cfg = VirTexModel.vit_backbone_config("B16")
        self._add_textual_config(cfg, eos_token_id=eos_id)

        model: VirTexModel = cfg.set(name="test_caption").instantiate(parent=None)

        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        prefix = jnp.full((batch_size, 1), fill_value=bos_id)
        caption_outputs: BeamSearchOutputs
        caption_outputs, _ = F(
            model,
            state=state,
            inputs=dict(
                image=generate_sample_captioning_inputs(batch_size=batch_size, seq_len=seq_len)[
                    "image"
                ],
                prefix=prefix,
                max_sequence_length=seq_len,
                num_decodes=num_decodes,
            ),
            prng_key=jax.random.PRNGKey(123),
            is_training=False,
            method="caption",
        )
        assert caption_outputs.sequences.shape == (batch_size, num_decodes, seq_len)


if __name__ == "__main__":
    absltest.main()

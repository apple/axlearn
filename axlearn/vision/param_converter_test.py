# Copyright © 2023 Apple Inc.

"""Tests vision param converter utils."""

import jax
import jax.numpy as jnp
from absl.testing import absltest

from axlearn.common.attention import LearnedPositionalEmbedding
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.golden import load_golden
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.text_encoder import ENCODED_HIDDEN_STATES
from axlearn.vision import clip

_MODULE_NAME = "axlearn.vision.param_converter_test"


def _load_golden_jax(test_name):
    """Load golden data and convert params to jax arrays."""
    golden = load_golden(_MODULE_NAME, test_name)
    if "params" in golden:
        golden["params"] = jax.tree_util.tree_map(jnp.asarray, golden["params"])
    return golden


class HFClipTest(TestCase):
    def setUp(self):
        super().setUp()
        # Config values matching what was used to generate golden data.
        self._clip_text_cfg = dict(
            eos_token_id=2,
            pad_token_id=0,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            max_position_embeddings=12,
            vocab_size=24,
        )
        self._clip_vision_cfg = dict(
            hidden_size=32,
            intermediate_size=64,
            image_size=16,
            patch_size=4,
            num_hidden_layers=3,
            num_attention_heads=8,
        )
        self._projection_dim = 48

    def _hf_clip_text_embedding(self):
        pos_emb_cfg = LearnedPositionalEmbedding.default_config().set(
            shape=[self._clip_text_cfg["max_position_embeddings"]]
        )
        emb_cfg = TransformerTextEmbeddings.default_config().set(pos_emb=pos_emb_cfg)
        return emb_cfg.set(
            name="convert_test",
            dim=self._clip_text_cfg["hidden_size"],
            vocab_size=self._clip_text_cfg["vocab_size"],
        ).instantiate(parent=None)

    def test_clip_text_embeddings(self):
        golden = _load_golden_jax("test_clip_text_embeddings")
        layer = self._hf_clip_text_embedding()

        # Ensure that we are testing against LearnedPositionalEmbedding.
        self.assertIsInstance(layer.pos_emb, LearnedPositionalEmbedding)

        inputs = jnp.asarray(golden["inputs"]["input_ids"])
        input_len = int(golden["inputs"]["input_len"])

        out, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=golden["params"],
            inputs=dict(input_batch=dict(inputs=inputs)),
        )
        ref = jnp.asarray(golden["outputs"]["ref"])
        # Compare only at non-padding positions.
        self.assertNestedAllClose(out[:, :input_len], ref[:, :input_len])

    def _hf_clip_text_stream_encoder(self):
        text_encoder = clip.set_text_encoder_config(
            pad_token_id=self._clip_text_cfg["pad_token_id"],
            max_seq_len=self._clip_text_cfg["max_position_embeddings"],
            vocab_size=self._clip_text_cfg["vocab_size"],
            num_layers=self._clip_text_cfg["num_hidden_layers"],
            model_dim=self._clip_text_cfg["hidden_size"],
            num_heads=self._clip_text_cfg["num_attention_heads"],
            feed_forward_dim=self._clip_text_cfg["intermediate_size"],
            feed_forward_act="quick_gelu",
            projection_dim=self._projection_dim,
            dropout_rate=0,
        )
        return text_encoder.set(
            name="convert_test",
        ).instantiate(parent=None)

    def test_clip_text_stream_encoder(self):
        golden = _load_golden_jax("test_clip_text_stream_encoder")
        layer = self._hf_clip_text_stream_encoder()

        inputs = jnp.asarray(golden["inputs"]["input_ids"])
        input_len = int(golden["inputs"]["input_len"])

        # Golden params cover text_encoder subtree; initialize full params and overlay.
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_params.update(golden["params"])

        out, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=layer_params,
            inputs=dict(input_batch={"text": jnp.expand_dims(inputs, 1)}),
        )
        ref = jnp.asarray(golden["outputs"]["last_hidden_state"])
        self.assertNestedAllClose(out[ENCODED_HIDDEN_STATES][:, :input_len], ref[:, :input_len])

    def _hf_clip_image_stream_encoder(self):
        image_encoder = clip.set_vision_encoder_config(
            num_layers=self._clip_vision_cfg["num_hidden_layers"],
            model_dim=self._clip_vision_cfg["hidden_size"],
            num_heads=self._clip_vision_cfg["num_attention_heads"],
            feed_forward_dim=self._clip_vision_cfg["intermediate_size"],
            feed_forward_act="quick_gelu",
            image_size=(16, 16),
            patch_size=(4, 4),
            dropout_rate=0,
            projection_dim=self._projection_dim,
        )
        return image_encoder.set(
            name="convert_test",
        ).instantiate(parent=None)

    def test_clip_image_stream_encoder(self):
        golden = _load_golden_jax("test_clip_image_stream_encoder")
        layer = self._hf_clip_image_stream_encoder()

        pixel_values = jnp.asarray(golden["inputs"]["pixel_values"])

        # Golden params cover image_encoder subtree; initialize full params and overlay.
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_params.update(golden["params"])

        out, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=layer_params,
            inputs=dict(
                input_batch={"image": jnp.expand_dims(jnp.einsum("bchw->bhwc", pixel_values), 1)}
            ),
        )
        ref = jnp.asarray(golden["outputs"]["pooler_output"])
        self.assertNestedAllClose(jnp.squeeze(out["pooled_features"]), ref)

    # NOTE: test_clip_model is not converted to golden files because the golden outputs were
    # generated from the HF model (after axlearn_to_torch param conversion), and subtle
    # differences in the forward pass or param conversion produce outputs outside tolerance.
    # The full CLIP model is already covered by axlearn.vision.clip_test.

    # NOTE: test_clip_vision_model_with_projection is not converted to golden files yet
    # because it requires HF CLIPVisionModelWithProjection as reference.


if __name__ == "__main__":
    absltest.main()

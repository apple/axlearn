# Copyright © 2023 Apple Inc.

"""Tests CLIP implementations."""

# pylint: disable=no-self-use
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.golden import load_golden
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_encoder import TEXT_EMBEDDINGS
from axlearn.vision.clip import (
    set_clip_model_config,
    set_text_encoder_config,
    set_vision_encoder_config,
)

_MODULE_NAME = "axlearn.vision.clip_test"

# Our reimplementation and HF CLIP uses the same EOS token id.
EOS_TOKEN_ID = 49407
# HF CLIP uses the same token id for EOS and PAD.
HF_PAD_TOKEN_ID = 49407
# We use a separate pad_token_id for our implementation.
OUR_PAD_TOKEN_ID = 49408

OUR_VOCAB_SIZE = 49409


def _hf_to_axlearn_input_ids(input_ids):
    """Convert HF-format input_ids (PAD=49407) to AXLearn format (PAD=49408).

    The golden data stores input_ids in HF format where PAD == EOS == 49407.
    AXLearn uses a separate pad_token_id of 49408. This replaces all PAD positions
    (everything after the first EOS token) with OUR_PAD_TOKEN_ID.
    """
    ids = np.array(input_ids)
    for row_idx in range(ids.shape[0]):
        row = ids[row_idx]
        eos_positions = np.where(row == EOS_TOKEN_ID)[0]
        if len(eos_positions) > 0:
            first_eos = eos_positions[0]
            ids[row_idx, first_eos + 1 :] = OUR_PAD_TOKEN_ID
    return ids


def _load_golden_jax(test_name):
    """Load golden data and convert params to jax arrays."""
    golden = load_golden(_MODULE_NAME, test_name)
    if "params" in golden:
        golden["params"] = jax.tree_util.tree_map(jnp.asarray, golden["params"])
    return golden


class TestCLIPEncoder(parameterized.TestCase):
    """Tests CLIPEncoder."""

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_clip_visual_encoder(self, act_fn):
        act_fn_key = act_fn.replace(".", "_")
        golden = _load_golden_jax(f"test_clip_visual_encoder_{act_fn_key}")

        model_dim = 32
        ff_dim = 64
        image_size = 16
        patch_size = 4
        num_layers = 3
        num_heads = 8

        kwargs = {
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "feed_forward_dim": ff_dim,
            "feed_forward_act": act_fn,
            "image_size": (image_size, image_size),
            "patch_size": (patch_size, patch_size),
            "dropout_rate": 0,
            "projection_dim": model_dim,
        }
        layer_cfg = set_vision_encoder_config(**kwargs)
        layer_cfg.set(name="test")
        layer = layer_cfg.instantiate(parent=None)

        # Golden params cover the image_encoder subtree; initialize full params and overlay.
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_params.update(golden["params"])
        pixel_values = golden["inputs"]["pixel_values"]

        layer_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch=dict(
                    image=jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(pixel_values)), 1)
                )
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(
            jnp.squeeze(layer_outputs["pooled_features"]),
            jnp.asarray(golden["outputs"]["pooler_output"]),
        )

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_clip_textual_encoder(self, act_fn):
        act_fn_key = act_fn.replace(".", "_")
        golden = _load_golden_jax(f"test_clip_textual_encoder_{act_fn_key}")

        model_dim = 32
        ff_dim = 64
        num_layers = 3
        num_heads = 8
        max_seq_len = 12

        kwargs = {
            "pad_token_id": OUR_PAD_TOKEN_ID,
            "max_seq_len": max_seq_len,
            "vocab_size": OUR_VOCAB_SIZE,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "feed_forward_dim": ff_dim,
            "feed_forward_act": act_fn,
            "dropout_rate": 0,
            "projection_dim": model_dim,
        }

        layer_cfg = set_text_encoder_config(**kwargs)
        layer_cfg.set(name="test")
        layer = layer_cfg.instantiate(parent=None)

        # Golden params cover the text_encoder subtree; initialize full params and overlay.
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_params.update(golden["params"])
        # Convert HF-format input_ids (PAD==EOS==49407) to AXLearn format (PAD=49408).
        input_ids = _hf_to_axlearn_input_ids(golden["inputs"]["input_ids"])

        layer_outputs, _ = F(
            layer,
            inputs=dict(input_batch={"text": jnp.expand_dims(jnp.asarray(input_ids), 1)}),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(
            jnp.squeeze(layer_outputs[TEXT_EMBEDDINGS]),
            jnp.asarray(golden["outputs"]["pooler_output"]),
        )


class TestCLIPModel(parameterized.TestCase):
    """Tests CLIPModel."""

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_clip_model(self, act_fn):
        act_fn_key = act_fn.replace(".", "_")
        golden = _load_golden_jax(f"test_clip_model_{act_fn_key}")

        text_encoder_dict = {
            "pad_token_id": OUR_PAD_TOKEN_ID,
            "max_seq_len": 12,
            "vocab_size": OUR_VOCAB_SIZE,
            "num_layers": 3,
            "model_dim": 32,
            "num_heads": 8,
            "feed_forward_dim": 64,
            "feed_forward_act": act_fn,
            "dropout_rate": 0,
        }

        vision_encoder_dict = {
            "num_layers": 3,
            "model_dim": 32,
            "num_heads": 8,
            "feed_forward_dim": 64,
            "feed_forward_act": act_fn,
            "image_size": (16, 16),
            "patch_size": (4, 4),
            "dropout_rate": 0,
        }

        clip_model_cfg = set_clip_model_config(
            text_encoder_cfg=set_text_encoder_config(**text_encoder_dict, projection_dim=32),
            vision_encoder_cfg=set_vision_encoder_config(**vision_encoder_dict, projection_dim=32),
        )
        layer = clip_model_cfg.set(name="clip_model").instantiate(parent=None)

        layer_params = golden["params"]
        # Convert HF-format input_ids (PAD==EOS==49407) to AXLearn format (PAD=49408).
        input_ids = _hf_to_axlearn_input_ids(golden["inputs"]["input_ids"])
        pixel_values = golden["inputs"]["pixel_values"]
        images = jnp.asarray(pixel_values)

        layer_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch=dict(
                    input={
                        "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", images), 1),
                        "text": jnp.expand_dims(jnp.asarray(input_ids), 1),
                    }
                )
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="unittest_predict",
        )
        # HF CLIP temperature implementation.
        # The logits is calculated as below:
        # logits = image.T * text * exp(log_logit_scale).
        assert_allclose(
            layer_outputs["fusion_network"]["similarity"]
            * jnp.exp(layer_params["fusion_network"]["log_logit_scale"]),
            jnp.asarray(golden["outputs"]["logits_per_image"]),
        )
        assert_allclose(
            jnp.squeeze(layer_outputs["textual_encoder"]["output_features"]),
            jnp.asarray(golden["outputs"]["text_embeds"]),
        )
        assert_allclose(
            jnp.squeeze(layer_outputs["visual_encoder"]["output_features"]),
            jnp.asarray(golden["outputs"]["image_embeds"]),
        )

        # Unittesting the "embed_image_batch" and "embed_text_batch".
        embed_image_output, _ = F(
            layer,
            inputs=dict(
                input_batch={
                    "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", images), 1),
                }
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_image_batch",
        )
        assert_allclose(
            embed_image_output.reshape(-1, embed_image_output.shape[-1]),
            jnp.asarray(golden["outputs"]["image_embeds"]),
        )

        embed_text_output, _ = F(
            layer,
            inputs=dict(
                input_batch={
                    "text": jnp.expand_dims(jnp.asarray(input_ids), 1),
                }
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_text_batch",
        )
        assert_allclose(
            embed_text_output.reshape(-1, embed_text_output.shape[-1]),
            jnp.asarray(golden["outputs"]["text_embeds"]),
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

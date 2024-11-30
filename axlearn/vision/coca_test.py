# Copyright © 2023 Apple Inc.

"""Tests CoCa implementations."""

# pylint: disable=no-self-use
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.module import functional as F
from axlearn.common.poolings import FirstNTokenPooling
from axlearn.common.test_utils import assert_allclose
from axlearn.vision.clip import (
    set_clip_model_config,
    set_text_encoder_config,
    set_vision_encoder_config,
)
from axlearn.vision.coca import (
    set_coca_config,
    set_coca_text_encoder_config,
    set_coca_vision_encoder_config,
)

EOS_TOKEN_ID = 49407
PAD_TOKEN_ID = 49408
VOCAB_SIZE = 49409


def generate_random_tokenized_text(*, batch_size, max_seq_len):
    # Generate a random text.
    tokenized_text = np.random.randint(low=2, high=VOCAB_SIZE - 3, size=[batch_size, max_seq_len])

    # Generate a random EOS position.
    eos_position = np.random.randint(low=1, high=max_seq_len - 1)

    # Set the EOS and PAD for input to our reimplementation TextualEncoder.
    tokenized_text[:, eos_position] = EOS_TOKEN_ID
    tokenized_text[:, eos_position + 1 :] = PAD_TOKEN_ID
    return np.expand_dims(tokenized_text, 1)


class TestCoCaEncoder(parameterized.TestCase):
    """Tests CoCa encoders."""

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_coca_visual_encoder(self, act_fn):
        model_dim = 32
        image_size = 16  # dummy_image: 16x16x3
        patch_size = 4
        num_layers = 3
        num_heads = 8
        batch_size = 2
        caption_pooler_num_outputs = 4

        # Common args for CoCa and CLIP.
        kwargs = {
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "feed_forward_act": act_fn,
            "image_size": (image_size, image_size),
            "patch_size": (patch_size, patch_size),
            "dropout_rate": 0,
            "num_cls_tokens": 1,
            "atten_logit_cap": 20.0,
        }

        clip_layer_cfg = set_vision_encoder_config(**kwargs, projection_dim=model_dim)
        clip_layer_cfg.set(name="test_clip")
        clip_visual_encoder = clip_layer_cfg.instantiate(parent=None)

        coca_layer_cfg = set_coca_vision_encoder_config(
            **kwargs,
            contrastive_output_dim=model_dim,
            caption_pooler_num_outputs=caption_pooler_num_outputs,
            contrastive_pooler_config=FirstNTokenPooling.default_config(),
            pooler_mode="parallel",
        )
        coca_layer_cfg.set(name="test")
        coca_visual_encoder = coca_layer_cfg.instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        params = coca_visual_encoder.initialize_parameters_recursively(init_key)

        # Parameters required by CLIP visual encoder.
        params["output_proj"] = params["contrastive_output_proj"]
        params["output_norm"] = params["contrastive_output_norm"]

        target = (
            np.random.randint(low=0, high=255, size=[batch_size, 3, image_size, image_size]) / 128
            - 1
        )

        coca_outputs, _ = F(
            coca_visual_encoder,
            inputs=dict(
                input_batch=dict(
                    image=jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(target)), 1)
                )
            ),
            state=params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        clip_outputs, _ = F(
            clip_visual_encoder,
            inputs=dict(
                input_batch=dict(
                    image=jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(target)), 1)
                )
            ),
            state=params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        # Compare contrastive output feature against clip model.
        assert_allclose(coca_outputs["output_features"], clip_outputs["output_features"])
        # Validate captioning output shapes.
        self.assertIn("caption_features", coca_outputs)
        self.assertEqual(
            coca_outputs["caption_features"].shape,
            (batch_size, 1, caption_pooler_num_outputs, model_dim),
        )

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_coca_textual_encoder(self, act_fn):
        model_dim = 32
        num_layers = 3
        num_heads = 8
        batch_size = 2
        max_seq_len = 12

        kwargs = {
            "pad_token_id": PAD_TOKEN_ID,
            "max_seq_len": max_seq_len,
            "vocab_size": VOCAB_SIZE,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "feed_forward_act": act_fn,
            "dropout_rate": 0,
        }

        coca_layer_cfg = set_coca_text_encoder_config(**kwargs, contrastive_output_dim=model_dim)
        coca_layer_cfg.set(name="coca_test")
        coca_textual_encoder = coca_layer_cfg.instantiate(parent=None)

        clip_layer_cfg = set_text_encoder_config(**kwargs, projection_dim=model_dim)
        clip_layer_cfg.set(name="clip_test")
        clip_textual_encoder = clip_layer_cfg.instantiate(parent=None)

        tokenized_text = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=max_seq_len
        )

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        params = coca_textual_encoder.initialize_parameters_recursively(init_key)

        # Parameters required by CLIP textual encoder.
        params["output_proj"] = params["contrastive_output_proj"]
        params["output_norm"] = params["contrastive_output_norm"]
        params["text_encoder"]["encoder"] = dict(params["text_encoder"])

        coca_outputs, _ = F(
            coca_textual_encoder,
            inputs=dict(input_batch={"text": jnp.asarray(tokenized_text)}),
            state=params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        clip_outputs, _ = F(
            clip_textual_encoder,
            inputs=dict(input_batch={"text": jnp.asarray(tokenized_text[:, :, :-1])}),
            state=params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        # Compare contrastive features against clip model.
        assert_allclose(coca_outputs["output_features"], clip_outputs["output_features"])

        # Validate captioning features and output shapes.
        self.assertIn("caption_features", coca_outputs)
        self.assertIn("caption_ids", coca_outputs)
        self.assertIn("caption_labels", coca_outputs)
        # The coca textual encoder only takes (max_seq_len - 1) tokens as the input.
        self.assertEqual(
            coca_outputs["caption_features"].shape, (batch_size, 1, max_seq_len - 1, model_dim)
        )

    def test_coca_textual_encoder_with_cls_token(self):
        model_dim = 32
        num_layers = 3
        num_heads = 8
        batch_size = 2
        max_seq_len = 12

        kwargs = {
            "pad_token_id": PAD_TOKEN_ID,
            "max_seq_len": max_seq_len,
            "vocab_size": VOCAB_SIZE,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "dropout_rate": 0,
            "contrastive_output_dim": model_dim,
        }

        coca_layer_cfg = set_coca_text_encoder_config(**kwargs)
        coca_layer_cfg.set(name="coca_test")
        coca_textual_encoder = coca_layer_cfg.instantiate(parent=None)

        tokenized_text = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=max_seq_len
        )

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        params = coca_textual_encoder.initialize_parameters_recursively(init_key)

        coca_outputs, _ = F(
            coca_textual_encoder,
            inputs=dict(input_batch={"text": jnp.asarray(tokenized_text)}),
            state=params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        self.assertIn("output_features", coca_outputs)
        self.assertIn("caption_features", coca_outputs)
        self.assertIn("caption_ids", coca_outputs)
        self.assertIn("caption_labels", coca_outputs)
        # Validate cls token output and caption feature shapes.
        self.assertEqual(coca_outputs["output_features"].shape, (batch_size, 1, model_dim))
        # The coca textual encoder only takes (max_seq_len - 1) tokens as the input.
        self.assertEqual(
            coca_outputs["caption_features"].shape, (batch_size, 1, max_seq_len - 1, model_dim)
        )


class TestCoCaModel(parameterized.TestCase):
    """Tests CoCaModel."""

    def _compare_against_clip_model(
        self, coca_model, clip_model, coca_params, batch_size, image_size, max_seq_len
    ):
        images = (
            np.random.randint(low=0, high=255, size=[batch_size, 3, image_size, image_size]) / 128
            - 1
        )
        tokenized_text = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=max_seq_len
        )

        # Parameters required by CLIP visual encoder.
        clip_params = deepcopy(coca_params)
        clip_params["visual_encoder"]["output_proj"] = coca_params["visual_encoder"][
            "contrastive_output_proj"
        ]
        clip_params["visual_encoder"]["output_norm"] = coca_params["visual_encoder"][
            "contrastive_output_norm"
        ]
        clip_params["textual_encoder"]["output_proj"] = coca_params["textual_encoder"][
            "contrastive_output_proj"
        ]
        clip_params["textual_encoder"]["output_norm"] = coca_params["textual_encoder"][
            "contrastive_output_norm"
        ]
        clip_params["textual_encoder"]["text_encoder"]["encoder"] = dict(
            coca_params["textual_encoder"]["text_encoder"]
        )
        clip_params["fusion_network"] = coca_params["contrastive_fusion_network"]

        coca_outputs, _ = F(
            coca_model,
            inputs=dict(
                input_batch=dict(
                    input={
                        "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                        "text": tokenized_text,
                    }
                )
            ),
            state=coca_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="predict",
        )
        clip_outputs, _ = F(
            clip_model,
            inputs=dict(
                input_batch=dict(
                    input={
                        "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                        "text": tokenized_text[:, :, :-1],
                    }
                )
            ),
            state=clip_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="unittest_predict",
        )
        assert_allclose(
            coca_outputs["contrastive_fusion_network"]["similarity"],
            clip_outputs["fusion_network"]["similarity"],
        )
        assert_allclose(
            coca_outputs["visual_encoder"]["output_features"],
            clip_outputs["visual_encoder"]["output_features"],
        )
        assert_allclose(
            coca_outputs["textual_encoder"]["output_features"],
            clip_outputs["textual_encoder"]["output_features"],
        )

        # Unittesting the "embed_image_batch".
        coca_embed_image_output, _ = F(
            coca_model,
            inputs=dict(
                input_batch={
                    "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                }
            ),
            state=coca_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_image_batch",
        )
        clip_embed_image_output, _ = F(
            clip_model,
            inputs=dict(
                input_batch={
                    "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                }
            ),
            state=clip_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_image_batch",
        )
        assert_allclose(
            coca_embed_image_output,
            clip_embed_image_output,
        )

        # Unittesting the "embed_text_batch".
        coca_embed_text_output, _ = F(
            coca_model,
            inputs=dict(
                input_batch={
                    "text": tokenized_text,
                }
            ),
            state=coca_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_text_batch",
        )
        clip_embed_text_output, _ = F(
            clip_model,
            inputs=dict(
                input_batch={
                    "text": tokenized_text,
                }
            ),
            state=clip_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_text_batch",
        )
        assert_allclose(
            coca_embed_text_output,
            clip_embed_text_output,
        )

    @parameterized.parameters([("nn.gelu", True), ("quick_gelu", True), ("nn.gelu", False)])
    def test_coca_model(self, act_fn, use_cross_attention):
        image_size = 16
        batch_size = 2
        max_seq_len = 12

        # Shared parameters of CoCa and CLIP.
        text_encoder_dict = {
            "pad_token_id": PAD_TOKEN_ID,
            "max_seq_len": 12,
            "vocab_size": VOCAB_SIZE,
            "num_layers": 3,
            "model_dim": 32,
            "num_heads": 8,
            "feed_forward_act": act_fn,
            "dropout_rate": 0,
        }

        vision_encoder_dict = {
            "num_layers": 3,
            "model_dim": 32,
            "num_heads": 8,
            "feed_forward_act": act_fn,
            "image_size": (image_size, image_size),
            "patch_size": (4, 4),
            "dropout_rate": 0,
            "num_cls_tokens": 1,
        }

        captioning_dict = {
            "num_layers": 3,
            "model_dim": 32,
            "num_heads": 8,
            "use_cross_attention": use_cross_attention,
            "cross_attention_dim": 32,
            "feed_forward_act": act_fn,
            "dropout_rate": 0,
        }

        clip_model_cfg = set_clip_model_config(
            text_encoder_cfg=set_text_encoder_config(**text_encoder_dict, projection_dim=48),
            vision_encoder_cfg=set_vision_encoder_config(**vision_encoder_dict, projection_dim=48),
        )
        clip_model = clip_model_cfg.set(name="clip_model").instantiate(parent=None)

        # CoCa parameters
        coca_vision_encoder_dict = deepcopy(vision_encoder_dict)
        coca_vision_encoder_dict["contrastive_pooler_config"] = FirstNTokenPooling.default_config()
        coca_vision_encoder_dict["pooler_mode"] = "parallel"

        coca_model_cfg = set_coca_config(
            contrastive_output_dim=48,
            text_encoder_cfg=text_encoder_dict,
            vision_encoder_cfg=coca_vision_encoder_dict,
            captioning_cfg=captioning_dict,
        )
        coca_model = coca_model_cfg.set(name="coca_model").instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        params = coca_model.initialize_parameters_recursively(init_key)

        self._compare_against_clip_model(
            coca_model, clip_model, params, batch_size, image_size, max_seq_len
        )

    def _get_default_model_config(self, image_size, model_dim, use_cross_attention):
        text_encoder_dict = {
            "pad_token_id": PAD_TOKEN_ID,
            "max_seq_len": 512,
            "vocab_size": VOCAB_SIZE,
            "num_layers": 3,
            "model_dim": model_dim,
            "num_heads": 8,
            "feed_forward_act": "nn.gelu",
            "dropout_rate": 0,
        }

        vision_encoder_dict = {
            "num_layers": 3,
            "model_dim": model_dim,
            "num_heads": 8,
            "feed_forward_act": "nn.gelu",
            "image_size": (image_size, image_size),
            "patch_size": (4, 4),
            "dropout_rate": 0,
            "contrastive_pooler_config": FirstNTokenPooling.default_config(),
            "pooler_mode": "parallel",
        }

        captioning_dict = {
            "num_layers": 3,
            "model_dim": model_dim,
            "num_heads": 8,
            "use_cross_attention": use_cross_attention,
            "cross_attention_dim": model_dim,
            "feed_forward_act": "nn.gelu",
            "dropout_rate": 0,
        }

        coca_model_cfg = set_coca_config(
            contrastive_output_dim=48,
            text_encoder_cfg=text_encoder_dict,
            vision_encoder_cfg=vision_encoder_dict,
            captioning_cfg=captioning_dict,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
        )

        return coca_model_cfg

    @parameterized.product(
        use_cross_attention=[False, True],
        prefill_states=[False, True],
    )
    def test_extend_step(self, use_cross_attention: bool, prefill_states: bool):
        image_size = 16
        batch_size = 2
        model_dim = 32
        tgt_len = 6

        # Setup CoCa Model
        coca_model_cfg = self._get_default_model_config(image_size, model_dim, use_cross_attention)
        coca_model = coca_model_cfg.set(name="coca_model").instantiate(parent=None)
        params = coca_model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Construct test inputs
        images = (
            np.random.randint(low=0, high=255, size=[batch_size, 3, image_size, image_size]) / 128
            - 1
        )
        tokenized_text = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=tgt_len + 1
        )

        forward_outputs, _ = F(
            coca_model,
            inputs=dict(
                input_batch=dict(
                    input={
                        "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                        "text": tokenized_text,
                    }
                )
            ),
            state=params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
            method="predict",
        )

        if use_cross_attention:
            # [batch, num_tokens=hwc, model_dim].
            cross_attention_data = forward_outputs["visual_encoder"]["caption_features"].squeeze(
                axis=1
            )
        else:
            cross_attention_data = None

        # We don't need the last token when decoding with extend_step.
        tokenized_text = tokenized_text[:, :, :-1].squeeze(axis=1)

        if prefill_states:
            time_step = jnp.arange(batch_size)
            (initial_state, initial_outputs), _ = F(
                coca_model,
                state=params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=dict(
                    time_step=time_step,
                    input_batch=dict(input_ids=tokenized_text),
                    cross_attention_data=cross_attention_data,
                ),
                method="prefill_states",
            )
            # Zero-out outputs starting from initial time_step, and test that we can recover the
            # full outputs by calling extend_step starting from time_step.
            # [batch, tgt_len].
            time_step_mask = jnp.arange(tgt_len) < time_step[:, None]
            # [batch, tgt_len, num_classes].
            logits = initial_outputs["logits"] * time_step_mask[:, :, None]
        else:
            time_step = jnp.zeros(batch_size, dtype=jnp.int32)
            initial_state, _ = F(
                coca_model,
                state=params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=dict(batch_size=batch_size, max_sequence_length=tgt_len),
                method="init_states",
            )
            logits = jnp.zeros(shape=[batch_size, tgt_len, VOCAB_SIZE])

        # [batch, tgt_len, num_classes] --> [batch, num_classes, tgt_len].
        logits = jnp.moveaxis(logits, -2, -1)

        inputs = dict(cached_states=initial_state)
        while jnp.any(time_step < tgt_len):
            # [batch, tgt_len=1].
            inputs["input_ids"] = jnp.take_along_axis(
                tokenized_text, time_step[:, None], axis=1, mode="clip"
            )
            if use_cross_attention:
                inputs["cross_attention_data"] = cross_attention_data

            (updated_state, outputs), _ = F(
                coca_model,
                state=params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            inputs["cached_states"] = updated_state

            # [batch, num_classes, tgt_len=1].
            curr_logits = jnp.moveaxis(outputs["logits"], -2, -1)
            # [batch, 1, tgt_len].
            oh_indices = jax.nn.one_hot(time_step, tgt_len)[:, None, :]
            logits = logits + curr_logits * oh_indices

            time_step = time_step + 1

        # [batch, num_classes, tgt_len] --> [batch, tgt_len, num_classes].
        logits = jnp.moveaxis(logits, -1, -2)
        assert_allclose(logits, forward_outputs["captioning_fusion_network"]["logits"], atol=1e-5)

    @parameterized.parameters(["beam_search_decode", "sample_decode"])
    def test_predict_caption(self, decode_method):
        image_size = 16
        batch_size = 2
        model_dim = 32

        # Setup CoCa Model
        coca_model_cfg = self._get_default_model_config(
            image_size, model_dim, use_cross_attention=True
        )
        coca_model = coca_model_cfg.set(name="coca_model").instantiate(parent=None)
        params = coca_model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Construct test inputs
        images = (
            np.random.randint(low=0, high=255, size=[batch_size, 3, image_size, image_size]) / 128
            - 1
        )
        tokenized_text = np.random.randint(low=2, high=VOCAB_SIZE - 3, size=[batch_size, 1])

        max_sequence_length = 5
        num_decodes = 3
        outputs, _ = F(
            coca_model,
            inputs=dict(
                input_batch={
                    "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                    "prefix": tokenized_text,
                },
                max_sequence_length=max_sequence_length,
                num_decodes=num_decodes,
                decode_method=decode_method,
            ),
            state=params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
            method="predict_caption",
        )

        self.assertEqual(outputs.sequences.shape, (batch_size, num_decodes, max_sequence_length))


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

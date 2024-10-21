# Copyright © 2023 Apple Inc.

"""Tests CLIP implementations."""
# pylint: disable=no-self-use
import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest, parameterized
from transformers.models.clip import modeling_clip as hf_clip

from axlearn.common import utils
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_encoder import TEXT_EMBEDDINGS
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import as_tensor
from axlearn.vision.clip import (
    set_clip_model_config,
    set_text_encoder_config,
    set_vision_encoder_config,
)

# Our reimplementation and HF CLIP uses the same EOS token id.
EOS_TOKEN_ID = 49407
# HF CLIP uses the same token id for EOS and PAD.
HF_PAD_TOKEN_ID = 49407
# We use a separate pad_token_id for our implementation.
OUR_PAD_TOKEN_ID = 49408

HF_VOCAB_SIZE = 49408
OUR_VOCAB_SIZE = 49409


def generate_random_tokenized_text(*, batch_size, max_seq_len):
    # Generate a random text.
    tokenized_text = np.random.randint(
        low=2, high=OUR_VOCAB_SIZE - 3, size=[batch_size, max_seq_len - 1]
    )

    # Generate a random EOS position.
    eos_position = np.random.randint(low=1, high=max_seq_len - 1)

    # Set the EOS and PAD for HF CLIP input.
    tokenized_text_hf = tokenized_text.copy()
    tokenized_text_hf[:, eos_position] = EOS_TOKEN_ID
    tokenized_text_hf[:, eos_position + 1 :] = HF_PAD_TOKEN_ID

    # Set the EOS and PAD for input to our reimplementation TextualEncoder.
    tokenized_text[:, eos_position] = EOS_TOKEN_ID
    tokenized_text[:, eos_position + 1 :] = OUR_PAD_TOKEN_ID
    return tokenized_text_hf, np.expand_dims(tokenized_text, 1)


def get_hf_act(act_fn):
    hf_act = act_fn
    if hf_act == "nn.gelu":
        # The gelu_new in HF is the same as nn.gelu in AXLearn.
        hf_act = "gelu_new"
    return hf_act


class TestCLIPEncoder(parameterized.TestCase):
    """Tests CLIPEncoder."""

    def _compare_against_clip_visual_encoder(self, ref, layer, batch_size, image_size, model_dim):
        layer_params = parameters_from_torch_layer(ref)
        # A dummy weight for the output projection.
        layer_params["output_proj"] = dict(
            weight=jnp.array(np.random.random((model_dim, model_dim))).astype(jnp.float32),
            bias=jnp.array(np.random.random(model_dim)).astype(jnp.float32),
        )
        target = (
            np.random.randint(low=0, high=255, size=[batch_size, 3, image_size, image_size]) / 128
            - 1
        )

        ref_outputs = ref.forward(torch.as_tensor(target, dtype=torch.float32))

        layer_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch=dict(
                    image=jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(target)), 1)
                )
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(
            layer_outputs["pooled_features"], as_tensor(torch.unsqueeze(ref_outputs[1], 1))
        )

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_clip_visual_encoder(self, act_fn):
        model_dim = 32
        ff_dim = 64
        image_size = 16  # dummy_image: 16x16x3
        patch_size = 4
        num_layers = 3
        num_heads = 8
        batch_size = 2
        clip_config = hf_clip.CLIPVisionConfig(
            hidden_size=model_dim,
            intermediate_size=ff_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_act=get_hf_act(act_fn),
        )
        ref_visual_encoder = hf_clip.CLIPVisionTransformer(clip_config)

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
        clip_visual_encoder = layer_cfg.instantiate(parent=None)

        self._compare_against_clip_visual_encoder(
            ref_visual_encoder, clip_visual_encoder, batch_size, image_size, model_dim
        )

    def _compare_against_clip_textual_encoder(
        self,
        ref,
        layer,
        model_dim,
        tokenized_text_generator_params,
    ):
        layer_params = parameters_from_torch_layer(ref)
        # A dummy weight for the output projection.
        layer_params["output_proj"] = dict(
            weight=jnp.array(np.random.random((model_dim, model_dim))).astype(jnp.float32),
            bias=jnp.array(np.random.random(model_dim)).astype(jnp.float32),
        )

        tokenized_text_hf, tokenized_text = generate_random_tokenized_text(
            **tokenized_text_generator_params
        )

        ref_outputs = ref.forward(torch.as_tensor(tokenized_text_hf))

        layer_outputs, _ = F(
            layer,
            inputs=dict(input_batch={"text": jnp.asarray(tokenized_text)}),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_outputs[TEXT_EMBEDDINGS], as_tensor(ref_outputs[1].unsqueeze(1)))

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_clip_textual_encoder(self, act_fn):
        model_dim = 32
        ff_dim = 64
        num_layers = 3
        num_heads = 8
        batch_size = 2
        max_seq_len = 12

        clip_config = hf_clip.CLIPTextConfig(
            hidden_size=model_dim,
            intermediate_size=ff_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_seq_len,
            pad_token_id=HF_PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            vocab_size=HF_VOCAB_SIZE,
            hidden_act=get_hf_act(act_fn),
        )
        ref_textual_encoder = hf_clip.CLIPTextTransformer(clip_config)

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
        clip_textual_encoder = layer_cfg.instantiate(parent=None)

        tokenized_text_generator_params = dict(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        self._compare_against_clip_textual_encoder(
            ref_textual_encoder,
            clip_textual_encoder,
            model_dim,
            tokenized_text_generator_params,
        )


class TestCLIPModel(parameterized.TestCase):
    """Tests CLIPModel."""

    def _compare_against_clip_model(
        self,
        ref,
        layer,
        batch_size,
        image_size,
        tokenized_text_generator_params,
    ):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_param_shapes = jax.tree.map(lambda x: x.shape, layer_params)
        print(f"layer state={layer_param_shapes}")
        layer_params = parameters_from_torch_layer(ref)
        images = (
            np.random.randint(low=0, high=255, size=[batch_size, 3, image_size, image_size]) / 128
            - 1
        )

        tokenized_text_hf, tokenized_text = generate_random_tokenized_text(
            **tokenized_text_generator_params
        )

        ref_outputs = ref.forward(
            input_ids=torch.as_tensor(tokenized_text_hf),
            pixel_values=torch.as_tensor(images, dtype=torch.float32),
        )

        layer_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch=dict(
                    input={
                        "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                        "text": tokenized_text,
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
            as_tensor(ref_outputs["logits_per_image"]),
        )
        assert_allclose(
            layer_outputs["textual_encoder"][TEXT_EMBEDDINGS],
            as_tensor(ref_outputs["text_model_output"]["pooler_output"].unsqueeze(1)),
        )
        assert_allclose(
            layer_outputs["visual_encoder"]["pooled_features"],
            as_tensor(ref_outputs["vision_model_output"]["pooler_output"].unsqueeze(1)),
        )
        assert_allclose(
            layer_outputs["visual_encoder"]["output_features"],
            as_tensor(ref_outputs["image_embeds"].unsqueeze(1)),
        )
        assert_allclose(
            layer_outputs["textual_encoder"]["output_features"],
            as_tensor(ref_outputs["text_embeds"].unsqueeze(1)),
        )

        # Unittesting the "embed_image_batch" and "embed_text_batch".
        embed_image_output, _ = F(
            layer,
            inputs=dict(
                input_batch={
                    "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", jnp.asarray(images)), 1),
                }
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_image_batch",
        )
        assert_allclose(
            embed_image_output.reshape(-1, embed_image_output.shape[-1]),
            as_tensor(ref_outputs["image_embeds"]),
        )

        embed_text_output, _ = F(
            layer,
            inputs=dict(
                input_batch={
                    "text": tokenized_text,
                }
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="embed_text_batch",
        )
        assert_allclose(
            embed_text_output.reshape(-1, embed_text_output.shape[-1]),
            as_tensor(ref_outputs["text_embeds"]),
        )

    @parameterized.parameters(["nn.gelu", "quick_gelu"])
    def test_clip_model(self, act_fn):
        batch_size = 2
        max_seq_len = 12

        text_config = dict(
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=HF_PAD_TOKEN_ID,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=8,
            max_position_embeddings=12,
            vocab_size=HF_VOCAB_SIZE,
            hidden_act=get_hf_act(act_fn),
        )
        vision_config = dict(
            hidden_size=32,
            intermediate_size=64,
            image_size=16,
            patch_size=4,
            num_hidden_layers=3,
            num_attention_heads=8,
            hidden_act=get_hf_act(act_fn),
        )
        clip_config = hf_clip.CLIPConfig(
            text_config_dict=text_config, vision_config_dict=vision_config, projection_dim=48
        )
        ref_clip_model = hf_clip.CLIPModel(clip_config)

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
            "num_layers": clip_config.vision_config.num_hidden_layers,
            "model_dim": clip_config.vision_config.hidden_size,
            "num_heads": clip_config.vision_config.num_attention_heads,
            "feed_forward_dim": clip_config.vision_config.intermediate_size,
            "feed_forward_act": act_fn,
            "image_size": (16, 16),
            "patch_size": (4, 4),
            "dropout_rate": 0,
        }

        clip_model_cfg = set_clip_model_config(
            text_encoder_cfg=set_text_encoder_config(**text_encoder_dict, projection_dim=32),
            vision_encoder_cfg=set_vision_encoder_config(**vision_encoder_dict, projection_dim=32),
        )
        clip_model = clip_model_cfg.set(name="clip_model").instantiate(parent=None)

        tokenized_text_generator_params = dict(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        self._compare_against_clip_model(
            ref_clip_model,
            clip_model,
            batch_size=batch_size,
            image_size=clip_config.vision_config.image_size,
            tokenized_text_generator_params=tokenized_text_generator_params,
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

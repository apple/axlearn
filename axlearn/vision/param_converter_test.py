# Copyright © 2023 Apple Inc.

"""Tests vision param converter utils."""
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from transformers.models.clip import modeling_clip as hf_clip

from axlearn.common.attention import LearnedPositionalEmbedding
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.normalize import l2_normalize
from axlearn.common.param_converter_test import BaseParamConverterTest, torch_output_to_dict
from axlearn.common.text_encoder import ENCODED_HIDDEN_STATES
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import as_tensor
from axlearn.vision import clip
from axlearn.vision.param_converter import as_torch_tensor, axlearn_to_torch


class HFClipTest(BaseParamConverterTest):
    def setUp(self):
        super().setUp()
        clip_text_cfg = dict(
            eos_token_id=2,
            pad_token_id=0,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            max_position_embeddings=12,
            vocab_size=24,
            hidden_act="quick_gelu",
        )
        clip_vision_cfg = dict(
            hidden_size=32,
            intermediate_size=64,
            image_size=16,
            patch_size=4,
            num_hidden_layers=3,
            num_attention_heads=8,
            hidden_act="quick_gelu",
        )
        self.clip_cfg = hf_clip.CLIPConfig(
            text_config_dict=clip_text_cfg, vision_config_dict=clip_vision_cfg, projection_dim=48
        )

    def _dummy_clip_input(self, batch: int):
        image_inputs = jax.random.randint(
            jax.random.PRNGKey(111),
            shape=(batch, 3, 16, 16),
            minval=0,
            maxval=255,
        )
        image_inputs = image_inputs / 128 - 1

        # Construct a "realistic" input where padding tokens only appear near the end.
        text_input_len = jax.random.randint(
            jax.random.PRNGKey(111),
            shape=(),
            minval=2,
            maxval=self.clip_cfg.text_config.max_position_embeddings + 1,
        )
        text_inputs = jax.random.randint(
            jax.random.PRNGKey(222),
            shape=(batch, text_input_len),
            minval=1,
            maxval=self.clip_cfg.text_config.vocab_size,
        )
        text_inputs = jnp.concatenate(
            (
                text_inputs,
                jnp.ones(text_inputs[:, 0:1].shape, dtype=jnp.int32)
                * self.clip_cfg.text_config.vocab_size
                - 1,
            ),
            1,
        )
        text_inputs = jnp.pad(
            text_inputs,
            [(0, 0), (0, self.clip_cfg.text_config.max_position_embeddings - text_input_len - 1)],
        )
        return image_inputs, text_inputs, text_input_len

    def _hf_clip_text_embedding(self):
        pos_emb_cfg = LearnedPositionalEmbedding.default_config().set(
            shape=[self.clip_cfg.text_config.max_position_embeddings]
        )
        emb_cfg = TransformerTextEmbeddings.default_config().set(pos_emb=pos_emb_cfg)
        return emb_cfg.set(
            name="convert_test",
            dim=self.clip_cfg.text_config.hidden_size,
            vocab_size=self.clip_cfg.text_config.vocab_size,
        ).instantiate(parent=None)

    def test_clip_text_embeddings(self):
        batch = 3
        layer = self._hf_clip_text_embedding()
        hf_layer = hf_clip.CLIPTextEmbeddings(self.clip_cfg.text_config)

        # Ensure that we are testing against LearnedPositionalEmbedding.
        self.assertIsInstance(layer.pos_emb, LearnedPositionalEmbedding)

        _, inputs, input_len = self._dummy_clip_input(batch)

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=[inputs],
            ref_inputs=as_torch_tensor(inputs),
        )
        # Compare only at non-padding positions.
        self.assertNestedAllClose(out[:, :input_len], hf_out[:, :input_len])

    def _hf_clip_text_stream_encoder(self):
        text_encoder = clip.set_text_encoder_config(
            pad_token_id=self.clip_cfg.text_config.pad_token_id,
            max_seq_len=self.clip_cfg.text_config.max_position_embeddings,
            vocab_size=self.clip_cfg.text_config.vocab_size,
            num_layers=self.clip_cfg.text_config.num_hidden_layers,
            model_dim=self.clip_cfg.text_config.hidden_size,
            num_heads=self.clip_cfg.text_config.num_attention_heads,
            feed_forward_dim=self.clip_cfg.text_config.intermediate_size,
            feed_forward_act="quick_gelu",
            projection_dim=self.clip_cfg.projection_dim,
            dropout_rate=0,
        )
        return text_encoder.set(
            name="convert_test",
        ).instantiate(parent=None)

    def test_clip_text_stream_encoder(self):
        batch = 3
        layer = self._hf_clip_text_stream_encoder()
        hf_layer = hf_clip.CLIPTextTransformer(self.clip_cfg.text_config)

        _, inputs, input_len = self._dummy_clip_input(batch)

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=dict(input_batch={"text": jnp.expand_dims(inputs, 1)}),
            ref_inputs=as_torch_tensor(inputs),
        )
        self.assertNestedAllClose(
            out[ENCODED_HIDDEN_STATES][:, :input_len], hf_out["last_hidden_state"][:, :input_len]
        )

    def _hf_clip_image_stream_encoder(self):
        image_encoder = clip.set_vision_encoder_config(
            num_layers=self.clip_cfg.vision_config.num_hidden_layers,
            model_dim=self.clip_cfg.vision_config.hidden_size,
            num_heads=self.clip_cfg.vision_config.num_attention_heads,
            feed_forward_dim=self.clip_cfg.vision_config.intermediate_size,
            feed_forward_act="quick_gelu",
            image_size=(16, 16),
            patch_size=(4, 4),
            dropout_rate=0,
            projection_dim=self.clip_cfg.projection_dim,
        )
        return image_encoder.set(
            name="convert_test",
        ).instantiate(parent=None)

    def test_clip_image_stream_encoder(self):
        batch = 3
        layer = self._hf_clip_image_stream_encoder()
        hf_layer = hf_clip.CLIPVisionTransformer(self.clip_cfg.vision_config)

        inputs, _, _ = self._dummy_clip_input(batch)

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=dict(
                input_batch={"image": jnp.expand_dims(jnp.einsum("bchw->bhwc", inputs), 1)}
            ),
            ref_inputs=as_torch_tensor(inputs),
        )
        self.assertNestedAllClose(jnp.squeeze(out["pooled_features"]), hf_out["pooler_output"])

    def _hf_clip_model(self, remat):
        text_encoder = {
            "pad_token_id": self.clip_cfg.text_config.pad_token_id,
            "max_seq_len": self.clip_cfg.text_config.max_position_embeddings,
            "vocab_size": self.clip_cfg.text_config.vocab_size,
            "num_layers": self.clip_cfg.text_config.num_hidden_layers,
            "model_dim": self.clip_cfg.text_config.hidden_size,
            "num_heads": self.clip_cfg.text_config.num_attention_heads,
            "feed_forward_dim": self.clip_cfg.text_config.intermediate_size,
            "feed_forward_act": "quick_gelu",
            "dropout_rate": 0,
            "remat": remat,
        }
        image_encoder = {
            "num_layers": self.clip_cfg.vision_config.num_hidden_layers,
            "model_dim": self.clip_cfg.vision_config.hidden_size,
            "num_heads": self.clip_cfg.vision_config.num_attention_heads,
            "feed_forward_dim": self.clip_cfg.vision_config.intermediate_size,
            "feed_forward_act": "quick_gelu",
            "image_size": (16, 16),
            "patch_size": (4, 4),
            "dropout_rate": 0,
            "remat": remat,
        }
        clip_model = clip.set_clip_model_config(
            text_encoder_cfg=clip.set_text_encoder_config(
                **text_encoder, projection_dim=self.clip_cfg.projection_dim
            ),
            vision_encoder_cfg=clip.set_vision_encoder_config(
                **image_encoder, projection_dim=self.clip_cfg.projection_dim
            ),
        )
        return clip_model.set(
            name="convert_test",
        ).instantiate(parent=None)

    @parameterized.parameters([False, True])
    def test_clip_model(self, remat):
        batch = 3
        layer = self._hf_clip_model(remat=remat)
        hf_layer = hf_clip.CLIPModel(self.clip_cfg)

        image_inputs, inputs, _ = self._dummy_clip_input(batch)

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=dict(
                input_batch=dict(
                    input={
                        "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", image_inputs), 1),
                        "text": jnp.expand_dims(inputs, 1),
                    }
                )
            ),
            ref_inputs={
                "input_ids": as_torch_tensor(inputs),
                "pixel_values": as_torch_tensor(image_inputs),
            },
            method="unittest_predict",
        )
        self.assertNestedAllClose(
            jnp.squeeze(out["textual_encoder"]["output_features"]), hf_out["text_embeds"]
        )
        self.assertNestedAllClose(
            jnp.squeeze(out["visual_encoder"]["output_features"]), hf_out["image_embeds"]
        )

    @parameterized.parameters([False, True])
    def test_clip_vision_model_with_projection(self, remat):
        batch = 3
        layer = self._hf_clip_model(remat=remat)
        vision_config = self.clip_cfg.vision_config
        hf_layer = hf_clip.CLIPVisionModelWithProjection(vision_config)

        image_inputs, _, _ = self._dummy_clip_input(batch)

        out, hf_out = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=hf_layer,
            test_inputs=dict(
                input_batch={
                    "image": jnp.expand_dims(jnp.einsum("bchw->bhwc", image_inputs), 1),
                }
            ),
            ref_inputs={
                "pixel_values": as_torch_tensor(image_inputs),
            },
            method="embed_image_batch",
        )
        self.assertNestedAllClose(jnp.squeeze(out), l2_normalize(hf_out["image_embeds"]))

    def test_clip_roundtrip(self):
        """Test CLIP Hugging Face to AXLearn and back."""
        batch = 3
        hf_layer = hf_clip.CLIPModel(self.clip_cfg)
        hf_layer_copy = hf_clip.CLIPModel(self.clip_cfg)
        layer = self._hf_clip_model(remat=False)

        params = parameters_from_torch_layer(hf_layer)
        axlearn_to_torch(layer, params, hf_layer_copy)

        image_inputs, inputs, _ = self._dummy_clip_input(batch)

        hf_inputs = {
            "input_ids": as_torch_tensor(inputs),
            "pixel_values": as_torch_tensor(image_inputs),
        }
        expected, actual = jax.tree.map(
            as_tensor,
            (
                torch_output_to_dict(hf_layer(**hf_inputs)),
                torch_output_to_dict(hf_layer_copy(**hf_inputs)),
            ),
        )
        self.assertNestedAllClose(expected, actual)

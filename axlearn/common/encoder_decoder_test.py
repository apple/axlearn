"""Tests EncoderDecoder layers."""
from typing import Optional

import jax
import numpy as np
import pytest
import torch
from jax import numpy as jnp
from transformers import BertConfig, BertModel, EncoderDecoderConfig
from transformers import EncoderDecoderModel as HFEncoderDecoderModel
from transformers import GPT2Config, GPT2LMHeadModel

from axlearn.common import encoder_decoder, utils
from axlearn.common.attention import StackedTransformerLayer, TransformerAttentionLayer
from axlearn.common.bert import (
    bert_embedding_config,
    bert_transformer_config,
    set_layer_norm_eps_recursively,
)
from axlearn.common.bert_test import bert_encoder_config_from_hf
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.decoder import Decoder, LmHead
from axlearn.common.encoder import Encoder
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor, parameters_from_t5x_encoder_decoder
from axlearn.common.t5 import t5_encoder_decoder_config
from axlearn.common.test_utils import (
    TestCase,
    assert_allclose,
    dummy_padding_mask,
    dummy_segments_positions,
)
from axlearn.common.torch_utils import parameters_from_torch_layer

try:
    # pytype: disable=import-error
    from t5x.examples.t5 import network as t5x_network

    # pytype: enable=import-error

    _T5X_INSTALLED = True
except ImportError:
    _T5X_INSTALLED = False


def set_decoder_cross_attention_config(
    decoder_cfg: Decoder.Config,
    num_heads: int,
):
    """Add cross attention to decoder config.

    Args:
        decoder_cfg: A config of Decoder.
        num_heads: Number of attention heads per transformer layer.
    """
    layer_cfg = decoder_cfg.transformer.layer
    # Cross attention transformer layer config.
    layer_cfg.cross_attention = TransformerAttentionLayer.default_config()
    layer_cfg.cross_attention.attention.num_heads = num_heads


# TODO(bwzhang@) Set the unittest that it will use cross_attention_logit_biases.
class TestEncoderDecoder(TestCase):
    """Tests EncoderDecoder layer."""

    def test_tied_lm_head_differs_from_untied(self):
        hidden_dim = 12
        num_heads = 4
        vocab_size = 24
        source_len = 11
        target_len = 5

        encoder = Encoder.default_config().set(
            dim=hidden_dim,
            vocab_size=vocab_size,
            emb=bert_embedding_config(type_vocab_size=1, max_position_embeddings=source_len),
            transformer=bert_transformer_config(num_layers=2, num_heads=num_heads),
            pad_token_id=0,
        )
        set_layer_norm_eps_recursively(encoder, 1e-8)

        shared_model_kwargs = dict(
            encoder=encoder,
        )
        decoder = gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=target_len,
        )
        set_decoder_cross_attention_config(
            decoder_cfg=decoder,
            num_heads=num_heads,
        )
        tied_head = (
            encoder_decoder.EncoderDecoderModel.default_config()
            .set(name="test_tied", decoder=decoder, **shared_model_kwargs)
            .instantiate(parent=None)
        )
        tied_head_state = tied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        self.assertIsNone(tied_head_state.get("lm_head"))
        untied_decoder = gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=target_len,
        )
        set_decoder_cross_attention_config(
            decoder_cfg=untied_decoder,
            num_heads=num_heads,
        )
        untied_decoder.lm_head = LmHead.default_config()
        untied_head = (
            encoder_decoder.EncoderDecoderModel.default_config()
            .set(name="test_untied", decoder=untied_decoder, **shared_model_kwargs)
            .instantiate(parent=None)
        )
        untied_head_state = untied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        self.assertIsNotNone(untied_head_state.get("decoder").get("lm_head"))
        batch_size = 3
        source_ids = jax.random.randint(
            jax.random.PRNGKey(1), minval=1, maxval=vocab_size, shape=(batch_size, source_len)
        )
        target_ids = jnp.ones((batch_size, target_len), dtype=jnp.int32)
        target_labels = jax.random.randint(
            jax.random.PRNGKey(1), minval=1, maxval=vocab_size, shape=(batch_size, target_len)
        )

        # Test values.
        def layer_output(state, layer):
            return F(
                layer,
                inputs=dict(
                    input_batch=dict(
                        source_ids=source_ids, target_ids=target_ids, target_labels=target_labels
                    ),
                    return_aux=True,
                ),
                state=state,
                is_training=False,
                prng_key=jax.random.PRNGKey(2),
            )[0][1]["logits"]

        tied_logits = layer_output(tied_head_state, tied_head)
        untied_logits = layer_output(untied_head_state, untied_head)
        np.testing.assert_raises(AssertionError, assert_allclose, tied_logits, untied_logits)

        # Test grads.
        def layer_loss(state, layer):
            return layer_output(state, layer).sum()

        def check_grads(tied_state, untied_state):
            tied_head_grad = jax.grad(layer_loss)(tied_state, tied_head)["decoder"]["emb"][
                "token_emb"
            ]["weight"]
            untied_head_grad = jax.grad(layer_loss)(untied_state, untied_head)["decoder"]["emb"][
                "token_emb"
            ]["weight"]
            np.testing.assert_raises(
                AssertionError, assert_allclose, tied_head_grad, untied_head_grad
            )

        # Assert grad is different tied vs untied
        check_grads(tied_head_state, untied_head_state)
        # Set untied head weight to tied lm_head value and check again.
        untied_head_state["decoder"]["lm_head"]["weight"] = tied_head_state["decoder"]["emb"][
            "token_emb"
        ]["weight"]
        check_grads(tied_head_state, untied_head_state)


def gpt2_decoder_config_from_hf(
    hf_cfg: GPT2Config,
    vocab_size: Optional[int] = None,
    layer_norm_epsilon: Optional[float] = None,
    dropout_rate: Optional[float] = None,
) -> Decoder.Config:
    return gpt_decoder_config(
        stack_cfg=StackedTransformerLayer.default_config(),
        num_layers=hf_cfg.n_layer,
        hidden_dim=hf_cfg.n_embd,
        num_heads=hf_cfg.n_head,
        vocab_size=vocab_size,
        activation_function=f"nn.{hf_cfg.activation_function}",
        max_position_embeddings=hf_cfg.n_positions,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout_rate=dropout_rate,
    )


class TestAgainstHF(TestCase):
    """Tests EncoderDecoder layer against HF."""

    def setUp(self):
        super().setUp()
        self.hf_encoder_cfg = BertConfig(
            vocab_size=24,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=11,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout=0.0,
            layer_norm_eps=1e-5,
        )
        self.hf_decoder_cfg = GPT2Config(
            n_embd=self.hf_encoder_cfg.hidden_size,
            n_head=self.hf_encoder_cfg.num_attention_heads,
            n_layer=self.hf_encoder_cfg.num_hidden_layers,
            n_positions=4,  # seq_len.
            vocab_size=self.hf_encoder_cfg.vocab_size,
            activation_function="relu",
            bos_token_id=1,
            eos_token_id=2,
            add_cross_attention=True,
            layer_norm_epsilon=self.hf_encoder_cfg.layer_norm_eps,
            is_decoder=True,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        self.hf_encoder_decoder_cfg = EncoderDecoderConfig.from_encoder_decoder_configs(
            self.hf_encoder_cfg,
            self.hf_decoder_cfg,
        )

        # Setup dummy axlearn model.
        axlearn_encoder = bert_encoder_config_from_hf(
            self.hf_encoder_cfg,
            vocab_size=self.hf_encoder_cfg.vocab_size,
            layer_norm_epsilon=self.hf_encoder_cfg.layer_norm_eps,
            dropout_rate=self.hf_encoder_cfg.hidden_dropout_prob,
        )
        axlearn_decoder = gpt2_decoder_config_from_hf(
            self.hf_decoder_cfg,
            vocab_size=self.hf_decoder_cfg.vocab_size,
            layer_norm_epsilon=self.hf_decoder_cfg.layer_norm_epsilon,
            dropout_rate=self.hf_decoder_cfg.embd_pdrop,
        )
        set_decoder_cross_attention_config(axlearn_decoder, self.hf_decoder_cfg.n_head)
        self.axlearn_encoder_decoder = (
            encoder_decoder.EncoderDecoderModel.default_config()
            .set(name="layer_test", encoder=axlearn_encoder, decoder=axlearn_decoder)
            .instantiate(parent=None)
        )

        # Setup dummy HF model.
        hf_encoder = BertModel(self.hf_encoder_cfg, add_pooling_layer=False)
        hf_decoder = GPT2LMHeadModel(self.hf_decoder_cfg)
        hf_encoder_decoder = HFEncoderDecoderModel(
            encoder=hf_encoder, decoder=hf_decoder, config=self.hf_encoder_decoder_cfg
        )
        hf_encoder_decoder.config.pad_token_id = self.hf_encoder_cfg.pad_token_id
        hf_encoder_decoder.config.decoder_start_token_id = self.hf_decoder_cfg.bos_token_id

        self.hf_encoder_decoder = hf_encoder_decoder.eval()

    def test_basic(self):
        batch_size = 3
        vocab_size = self.hf_encoder_cfg.vocab_size
        source_len = self.hf_encoder_cfg.max_position_embeddings
        target_len = self.hf_decoder_cfg.n_positions
        type_vocab_size = self.hf_encoder_cfg.type_vocab_size
        source_ids = jax.random.randint(
            jax.random.PRNGKey(101),
            (batch_size, source_len),
            minval=0,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        source_token_type_ids = jax.random.randint(
            jax.random.PRNGKey(102),
            (batch_size, source_len),
            minval=0,
            maxval=type_vocab_size,
            dtype=jnp.int32,
        )
        target_ids = jax.random.randint(
            jax.random.PRNGKey(103),
            (batch_size, target_len),
            minval=0,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(104),
            (batch_size, target_len),
            minval=0,
            maxval=vocab_size,
            dtype=jnp.int32,
        )

        # Compute outputs.
        (loss, test_aux), ref_outputs = self._compute_layer_outputs(
            test_layer=self.axlearn_encoder_decoder,
            ref_layer=self.hf_encoder_decoder,
            test_inputs=dict(
                input_batch=dict(
                    source_ids=source_ids,
                    source_token_type_ids=source_token_type_ids,
                    target_ids=target_ids,
                    target_labels=target_labels,
                ),
                return_aux=True,
            ),
            ref_inputs=dict(
                input_ids=as_torch_tensor(source_ids),
                token_type_ids=as_torch_tensor(source_token_type_ids),
                decoder_input_ids=as_torch_tensor(target_ids),
                labels=as_torch_tensor(target_labels).to(torch.long),
                output_hidden_states=True,
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )

        # Compare outputs.
        # We occasionally observe rounding errors.
        assert_allclose(test_aux["logits"], utils.as_tensor(ref_outputs.logits), atol=5e-6)
        assert_allclose(loss, utils.as_tensor(ref_outputs.loss))


@pytest.mark.skipif(not _T5X_INSTALLED, reason="T5 is not installed.")
class TestAgainstT5X(TestCase):
    """Tests EncoderDecoder layer against T5X."""

    def setUp(self):
        super().setUp()

        # Setup dummy T5X model.
        self.t5x_config = t5x_network.T5Config(
            vocab_size=48,
            dtype="float32",  # Note: T5X uses "bfloat16" by default.
            emb_dim=16,
            num_heads=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            head_dim=4,
            mlp_dim=64,
            mlp_activations=("gelu", "linear"),
            dropout_rate=0.0,
            logits_via_embedding=False,
        )
        self.t5x_encoder_decoder = t5x_network.Transformer(config=self.t5x_config)

        # Setup dummy axlearn model.
        cfg = t5_encoder_decoder_config(
            vocab_size=self.t5x_config.vocab_size,
            dim=self.t5x_config.emb_dim,
            num_attention_heads=self.t5x_config.num_heads,
            num_encoder_layers=self.t5x_config.num_encoder_layers,
            num_decoder_layers=self.t5x_config.num_decoder_layers,
            dropout_rate=0,
            z_loss_scale=0,
        )
        self.axlearn_encoder_decoder = cfg.set(name="test").instantiate(parent=None)

    def test_segment_ids(self):
        # HF does not seem to support packing:
        # https://github.com/huggingface/transformers/issues/17726
        # Instead, we test against T5X.
        batch_size = 3
        num_segments = 4
        vocab_size = self.t5x_config.vocab_size
        # {source,target}_len should be <= vocab_size:
        # https://github.com/google-research/t5x/blob/f9bba7f6aedaea96b4a78d6d3ea1d262289890f7/t5x/examples/t5/network.py#L285
        source_len = 32
        target_len = 15

        # Generate dummy inputs.
        source_ids = jax.random.randint(
            jax.random.PRNGKey(101),
            (batch_size, source_len),
            minval=2,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        target_tokens = jax.random.randint(
            jax.random.PRNGKey(102),
            (batch_size, target_len + 1),
            minval=2,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        target_ids = target_tokens[:, :-1]
        target_labels = target_tokens[:, 1:]
        source_segment_ids, source_positions = dummy_segments_positions(
            batch_size, source_len, num_segments=num_segments
        )
        target_segment_ids, target_positions = dummy_segments_positions(
            batch_size, target_len, num_segments=num_segments
        )
        # Allow targets to have trailing padding. When packing, no padding tokens are introduced
        # between segments.
        padding_mask = dummy_padding_mask(batch_size=batch_size, max_seq_len=target_len)
        target_ids *= padding_mask
        target_labels = jnp.where(padding_mask, target_labels, -1)
        target_segment_ids = jnp.where(padding_mask, target_segment_ids, 0)
        target_positions = jnp.where(padding_mask, target_positions, target_len)

        # Compute outputs.
        ref_outputs, variables = self.t5x_encoder_decoder.init_with_output(
            jax.random.PRNGKey(104),
            source_ids,
            target_ids,
            target_labels,
            encoder_segment_ids=source_segment_ids,
            decoder_segment_ids=target_segment_ids,
            # Note: it seems that T5 does not use encoder/decoder positions.
            # https://github.com/google-research/t5x/blob/f9bba7f6aedaea96b4a78d6d3ea1d262289890f7/t5x/examples/t5/network.py#L386
            # https://github.com/google-research/t5x/blob/f9bba7f6aedaea96b4a78d6d3ea1d262289890f7/t5x/examples/t5/network.py#L221
            encoder_positions=source_positions,
            decoder_positions=target_positions,
            enable_dropout=False,
        )
        test_outputs, _ = F(
            self.axlearn_encoder_decoder,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=parameters_from_t5x_encoder_decoder(
                variables["params"],
                self.axlearn_encoder_decoder,
            ),
            inputs=dict(
                input_batch=dict(
                    source_ids=source_ids,
                    target_ids=target_ids,
                    target_labels=target_labels,
                    source_segment_ids=source_segment_ids,
                    target_segment_ids=target_segment_ids,
                    source_positions=source_positions,
                    target_positions=target_positions,
                ),
            ),
            method="predict",
        )

        # Compare.
        test_outputs = test_outputs["logits"]
        ref_outputs = utils.as_tensor(ref_outputs)
        mask = padding_mask[..., None]
        self.assertNestedAllClose(test_outputs * mask, ref_outputs * mask)

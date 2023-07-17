# Copyright © 2023 Apple Inc.

"""Tests DPR utils."""
# pylint: disable=no-self-use
import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import parameterized
from transformers import DPRContextEncoder, DPRQuestionEncoder
from transformers.models.dpr import modeling_dpr as hf_dpr

from axlearn.common.module import functional as F
from axlearn.common.neural_retrieval import set_bert_dpr_encoder_config
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_dual_encoder import POSITIVE_EMBEDDINGS, POSITIVE_INPUT_IDS
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import as_tensor


def generate_random_tokenized_text(*, batch_size, max_seq_len):
    # Generate a random text.
    tokenized_text = np.random.randint(low=3, high=30522, size=[batch_size, max_seq_len - 1])

    # Generate a random EOS position.
    eos_position = np.random.randint(low=1, high=max_seq_len - 1)

    # Set the PAD for HF CLIP input.
    tokenized_text_hf = tokenized_text.copy()
    tokenized_text_hf[:, eos_position + 1 :] = 0

    # Set the PAD for input to our reimplementation DPRTextualEncoder.
    tokenized_text[:, eos_position + 1 :] = 0
    return tokenized_text_hf, np.expand_dims(tokenized_text, 1)


class TestDPRQuestionEncoder(parameterized.TestCase):
    """Tests against DPRQuestionEncoder."""

    def test_dpr(self):
        model_dim = 32
        ff_dim = 128
        num_layers = 3
        num_heads = 8
        batch_size = 2
        max_seq_len = 12

        hf_dpr_config = hf_dpr.DPRConfig(
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0,
            hidden_size=model_dim,
            intermediate_size=ff_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_seq_len,
            pad_token_id=0,
            vocab_size=30522,
            hidden_act="gelu",
        )
        hf_dpr_question_encoder = DPRQuestionEncoder(hf_dpr_config)
        kwargs = {
            "pad_token_id": 0,
            "max_seq_len": max_seq_len,
            "vocab_size": 30522,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "feed_forward_act": "exact_gelu",
        }
        layer_cfg = set_bert_dpr_encoder_config(**kwargs)
        layer_cfg.set(name="test")
        dpr_question_encoder = layer_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(hf_dpr_question_encoder)
        tokenized_text_hf, tokenized_text = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=max_seq_len
        )

        ref_outputs = hf_dpr_question_encoder.forward(torch.as_tensor(tokenized_text_hf))

        layer_outputs, _ = F(
            dpr_question_encoder,
            inputs=dict(input_batch={POSITIVE_INPUT_IDS: jnp.asarray(tokenized_text)}),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_outputs[POSITIVE_EMBEDDINGS], as_tensor(ref_outputs[0].unsqueeze(1)))


class TestDPRContextEncoder(parameterized.TestCase):
    """Tests against DPRContextEncoder."""

    def test_dpr(self):
        model_dim = 32
        ff_dim = 128
        num_layers = 3
        num_heads = 8
        batch_size = 2
        max_seq_len = 12

        hf_dpr_config = hf_dpr.DPRConfig(
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0,
            hidden_size=model_dim,
            intermediate_size=ff_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_seq_len,
            pad_token_id=0,
            vocab_size=30522,
            hidden_act="gelu",
        )
        hf_dpr_context_encoder = DPRContextEncoder(hf_dpr_config)
        kwargs = {
            "pad_token_id": 0,
            "max_seq_len": max_seq_len,
            "vocab_size": 30522,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "feed_forward_act": "exact_gelu",
        }
        layer_cfg = set_bert_dpr_encoder_config(**kwargs)
        layer_cfg.set(name="test")
        dpr_context_encoder = layer_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(hf_dpr_context_encoder)
        tokenized_text_hf, tokenized_text = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=max_seq_len
        )

        ref_outputs = hf_dpr_context_encoder.forward(torch.as_tensor(tokenized_text_hf))

        layer_outputs, _ = F(
            dpr_context_encoder,
            inputs=dict(input_batch={POSITIVE_INPUT_IDS: jnp.asarray(tokenized_text)}),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_outputs[POSITIVE_EMBEDDINGS], as_tensor(ref_outputs[0].unsqueeze(1)))

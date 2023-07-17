# Copyright © 2023 Apple Inc.

"""Tests DistilBert layers."""
# pylint: disable=no-self-use
import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import parameterized
from transformers.models.distilbert import configuration_distilbert as hf_distilbert_config
from transformers.models.distilbert import modeling_distilbert as hf_distilbert

from axlearn.common.distilbert import set_distilbert_config
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_encoder import TEXT_EMBEDDINGS
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
    return tokenized_text_hf, tokenized_text


class TestDistilBertModel(parameterized.TestCase):
    """Tests DistilBertModel."""

    def test_distilbert(self):
        model_dim = 32
        ff_dim = 64
        num_layers = 1
        num_heads = 8
        batch_size = 2
        max_seq_len = 12

        hf_distilbert_cfg = hf_distilbert_config.DistilBertConfig(
            attention_dropout=0,
            dropout=0,
            qa_dropout=0,
            seq_classif_dropout=0,
            dim=model_dim,
            hidden_dim=ff_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            max_position_embeddings=max_seq_len,
            pad_token_id=0,
            vocab_size=30522,
            activation="gelu",
        )
        hf_distilbert_model = hf_distilbert.DistilBertModel(hf_distilbert_cfg)
        kwargs = {
            "pad_token_id": 0,
            "max_seq_len": max_seq_len,
            "vocab_size": 30522,
            "num_layers": num_layers,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "feed_forward_dim": ff_dim,
            "feed_forward_act": "exact_gelu",
            "dropout_rate": 0,
        }
        layer_cfg = set_distilbert_config(**kwargs)
        layer_cfg.set(name="test")
        axlearn_distilbert_model = layer_cfg.instantiate(parent=None)

        layer_params = parameters_from_torch_layer(hf_distilbert_model)
        tokenized_text_hf, tokenized_text = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=max_seq_len
        )

        # HF distilbert ignores the pad_token_id.
        # It is important to set the attention_mask.
        ref_outputs = hf_distilbert_model.forward(
            input_ids=torch.as_tensor(tokenized_text_hf),
            attention_mask=torch.as_tensor(tokenized_text_hf != 0).int(),
        )

        layer_outputs, _ = F(
            axlearn_distilbert_model,
            inputs=dict(input_ids=jnp.asarray(tokenized_text)),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_outputs[TEXT_EMBEDDINGS], as_tensor(ref_outputs[0][:, 0:1]))

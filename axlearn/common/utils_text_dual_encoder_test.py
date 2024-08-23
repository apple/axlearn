# Copyright © 2023 Apple Inc.

"""Tests text dual encoder utils."""

# pylint: disable=no-self-use
import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import parameterized
from transformers import BertConfig, BertModel

from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_dual_encoder import POSITIVE_EMBEDDINGS, POSITIVE_INPUT_IDS
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import as_tensor
from axlearn.common.utils_text_dual_encoder import bert_text_embedding_stream_encoder_config


def generate_random_tokenized_text(
    *, batch_size: int, max_seq_len: int
) -> tuple[np.ndarray, torch.Tensor]:
    # Generate a random text.
    input_ids = np.random.randint(low=3, high=30522, size=[batch_size, max_seq_len - 1])

    # Generate a random EOS position.
    eos_position = np.random.randint(low=1, high=max_seq_len - 1)

    input_ids[:, eos_position + 1 :] = 0
    attention_mask = np.ones(input_ids.shape)
    attention_mask[:, eos_position + 1 :] = 0
    return input_ids, torch.as_tensor(attention_mask)


class TestBertStreamEncoderConfig(parameterized.TestCase):
    """Tests against HF BertModel."""

    @parameterized.parameters(True, False)
    def test_bert_stream_encoder_config(self, remat: bool):
        model_dim = 32
        num_layers = 3
        num_heads = 8
        batch_size = 2
        max_seq_len = 12

        hf_bert_config = BertConfig(
            hidden_size=model_dim,
            intermediate_size=model_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_seq_len,
            hidden_act="gelu_new",
        )
        hf_bert_model = BertModel(hf_bert_config)
        hf_bert_model.eval()

        kwargs = {
            "pad_token_id": 0,
            "max_seq_len": max_seq_len,
            "vocab_size": 30522,
            "num_layers": num_layers,
            "hidden_dim": model_dim,
            "output_dim": model_dim,
            "num_heads": num_heads,
            "output_proj": None,
            "output_norm": None,
            "remat": remat,
        }
        text_encoder_cfg = bert_text_embedding_stream_encoder_config(**kwargs)
        text_encoder_cfg.set(name="test")

        axlearn_text_encoder = text_encoder_cfg.instantiate(parent=None)

        dst_layer = None
        if remat:
            dst_layer = axlearn_text_encoder.text_encoder.encoder

        layer_params = parameters_from_torch_layer(hf_bert_model, dst_layer=dst_layer)

        input_ids, attention_mask = generate_random_tokenized_text(
            batch_size=batch_size, max_seq_len=max_seq_len
        )

        ref_outputs = hf_bert_model.forward(torch.as_tensor(input_ids), attention_mask)
        emb = ref_outputs.last_hidden_state[:, 0, :]

        layer_outputs, _ = F(
            axlearn_text_encoder,
            inputs=dict(
                input_batch={POSITIVE_INPUT_IDS: jnp.asarray(np.expand_dims(input_ids, 1))}
            ),
            state=dict(text_encoder=layer_params),
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_outputs[POSITIVE_EMBEDDINGS], as_tensor(emb.unsqueeze(1)))

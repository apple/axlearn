# Copyright © 2023 Apple Inc.

"""Tests text dual encoder utils."""

# pylint: disable=no-self-use

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from axlearn.common.golden import load_golden
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_dual_encoder import POSITIVE_EMBEDDINGS, POSITIVE_INPUT_IDS
from axlearn.common.utils_text_dual_encoder import bert_text_embedding_stream_encoder_config


class TestBertStreamEncoderConfig(absltest.TestCase):
    """Tests bert_text_embedding_stream_encoder_config against golden HF reference data."""

    def test_bert_stream_encoder_config(self):
        model_dim = 32
        num_layers = 3
        num_heads = 8
        max_seq_len = 12

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
            "remat": False,
        }
        text_encoder_cfg = bert_text_embedding_stream_encoder_config(**kwargs)
        text_encoder_cfg.set(name="test")
        axlearn_text_encoder = text_encoder_cfg.instantiate(parent=None)

        golden = load_golden(
            "axlearn.common.utils_text_dual_encoder_test", "test_bert_stream_encoder_config"
        )

        layer_outputs, _ = F(
            axlearn_text_encoder,
            inputs=dict(
                input_batch={
                    POSITIVE_INPUT_IDS: jnp.asarray(
                        np.expand_dims(golden["inputs"]["input_ids"], 1)
                    )
                }
            ),
            state=dict(text_encoder=golden["params"]),
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(
            layer_outputs[POSITIVE_EMBEDDINGS],
            jnp.expand_dims(golden["outputs"]["emb"], 1),
        )


if __name__ == "__main__":
    absltest.main()

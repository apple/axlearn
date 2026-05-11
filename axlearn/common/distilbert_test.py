# Copyright © 2023 Apple Inc.

"""Tests DistilBert layers."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.distilbert import set_distilbert_config
from axlearn.common.golden import load_golden
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_encoder import TEXT_EMBEDDINGS

_MODULE_NAME = "axlearn.common.distilbert_test"


class TestDistilBertModel(parameterized.TestCase):
    """Tests DistilBertModel."""

    def test_distilbert(self):
        model_dim = 32
        ff_dim = 64
        num_layers = 1
        num_heads = 8
        max_seq_len = 12

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
        layer = layer_cfg.instantiate(parent=None)

        golden = load_golden(_MODULE_NAME, "test_distilbert")

        layer_outputs, _ = F(
            layer,
            inputs=dict(input_ids=jnp.asarray(golden["inputs"]["input_ids"])),
            state=golden["params"],
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(layer_outputs[TEXT_EMBEDDINGS], golden["outputs"]["text_embeddings"])


if __name__ == "__main__":
    absltest.main()

# Copyright © 2023 Apple Inc.

"""Tests DPR utils."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.golden import load_golden
from axlearn.common.module import functional as F
from axlearn.common.neural_retrieval import set_bert_dpr_encoder_config
from axlearn.common.test_utils import assert_allclose
from axlearn.common.text_dual_encoder import POSITIVE_EMBEDDINGS, POSITIVE_INPUT_IDS

_MODULE_NAME = "axlearn.common.neural_retrieval_test"


class TestDPRQuestionEncoder(parameterized.TestCase):
    """Tests against DPRQuestionEncoder."""

    def test_dpr(self):
        kwargs = {
            "pad_token_id": 0,
            "max_seq_len": 12,
            "vocab_size": 30522,
            "num_layers": 3,
            "model_dim": 32,
            "num_heads": 8,
            "feed_forward_act": "exact_gelu",
        }
        layer_cfg = set_bert_dpr_encoder_config(**kwargs)
        layer_cfg.set(name="test")
        layer = layer_cfg.instantiate(parent=None)

        golden = load_golden(_MODULE_NAME, "test_dpr_question_encoder")

        layer_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch={
                    POSITIVE_INPUT_IDS: jnp.asarray(golden["inputs"]["positive_input_ids"])
                }
            ),
            state=golden["params"],
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(
            layer_outputs[POSITIVE_EMBEDDINGS], golden["outputs"]["positive_embeddings"]
        )


class TestDPRContextEncoder(parameterized.TestCase):
    """Tests against DPRContextEncoder."""

    def test_dpr(self):
        kwargs = {
            "pad_token_id": 0,
            "max_seq_len": 12,
            "vocab_size": 30522,
            "num_layers": 3,
            "model_dim": 32,
            "num_heads": 8,
            "feed_forward_act": "exact_gelu",
        }
        layer_cfg = set_bert_dpr_encoder_config(**kwargs)
        layer_cfg.set(name="test")
        layer = layer_cfg.instantiate(parent=None)

        golden = load_golden(_MODULE_NAME, "test_dpr_context_encoder")

        layer_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch={
                    POSITIVE_INPUT_IDS: jnp.asarray(golden["inputs"]["positive_input_ids"])
                }
            ),
            state=golden["params"],
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        assert_allclose(
            layer_outputs[POSITIVE_EMBEDDINGS], golden["outputs"]["positive_embeddings"]
        )


if __name__ == "__main__":
    absltest.main()

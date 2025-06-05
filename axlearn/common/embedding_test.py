# Copyright © 2023 Apple Inc.

"""Test embedding layers."""

# pylint: disable=no-self-use
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import transformers.models.bert.modeling_bert as hf_bert
from absl.testing import absltest, parameterized
from transformers import BertConfig

from axlearn.common import module, utils
from axlearn.common.attention import LearnedPositionalEmbedding
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import Embedding, LayerNorm
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer


class TestTransformerTextEmbeddings(TestCase):
    """Tests TransformerTextEmbeddings."""

    @parameterized.parameters(
        (LearnedPositionalEmbedding, False),
        (LearnedPositionalEmbedding, True),
        (Embedding, False),
        (Embedding, True),
    )
    def test_against_hf_bert_embeddings(self, pos_emb_cls: type, use_explicit_positions: bool):
        hidden_dim = 12
        vocab_size = 24
        num_heads = 4
        num_layers = 2
        source_length = 11
        type_vocab_size = 1
        layer_norm_epsilon = 1e-05
        # Reference implementation.
        encoder_config = BertConfig(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_size=hidden_dim,
            max_position_embeddings=source_length,
            vocab_size=vocab_size,
            intermediate_size=4 * hidden_dim,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_epsilon,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        model = hf_bert.BertEmbeddings(config=encoder_config)

        ref_layer = model.eval()

        # Equivalent AXLearn implementation.
        if pos_emb_cls == LearnedPositionalEmbedding:
            pos_emb_cfg = LearnedPositionalEmbedding.default_config().set(shape=(source_length,))
        else:
            pos_emb_cfg = Embedding.default_config().set(num_embeddings=source_length)
        emb = TransformerTextEmbeddings.default_config().set(
            dim=hidden_dim,
            vocab_size=vocab_size,
            pos_emb=pos_emb_cfg,
            type_emb=Embedding.default_config().set(num_embeddings=type_vocab_size),
            norm=LayerNorm.default_config().set(eps=layer_norm_epsilon),
        )

        layer = emb.set(name="layer_test").instantiate(parent=None)
        batch_size = 3
        input_ids = np.random.randint(1, vocab_size, size=(batch_size, source_length))
        test_inputs = dict(inputs=input_ids)
        if use_explicit_positions:
            test_inputs["positions"] = np.arange(source_length)

        test_hidden_states, ref_hidden_states = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(input_batch=test_inputs),
            ref_inputs=dict(
                input_ids=as_torch_tensor(input_ids),
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )
        assert_allclose(test_hidden_states, utils.as_tensor(ref_hidden_states))

    @parameterized.parameters(itertools.product((0.0, 10.0), (True, False)))
    def test_embed_attend(self, soft_cap_logits, is_training):
        seq_len = 5
        vocab_size = 24
        hidden_dim = 12

        emb = TransformerTextEmbeddings.default_config().set(
            name="embed",
            dim=hidden_dim,
            vocab_size=vocab_size,
            soft_cap_logits=soft_cap_logits,
        )
        rng = jax.random.PRNGKey(1)
        layer = emb.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(rng)

        x = jax.random.normal(rng, shape=(3, seq_len, hidden_dim))
        actual_attends = module.functional(
            layer, rng, state=state, inputs=[x], is_training=is_training, method="attend"
        )[0]
        ref = jnp.dot(x, state["token_emb"]["weight"].T)
        if soft_cap_logits > 0:
            ref = soft_cap_logits * jnp.tanh(ref / soft_cap_logits)
        assert_allclose(ref, actual_attends)

    def test_embed_with_emb_scale(self):
        seq_len = 5
        vocab_size = 24
        hidden_dim = 256

        emb = TransformerTextEmbeddings.default_config().set(
            name="embed",
            dim=hidden_dim,
            vocab_size=vocab_size,
        )
        emb.token_emb.set(scale=emb.token_emb.klass.Scale.UNIT)
        layer = emb.instantiate(parent=None)

        prng_key = jax.random.PRNGKey(1)
        prng_key, init_key, data_key, fwd_key = jax.random.split(prng_key, num=4)
        state = layer.initialize_parameters_recursively(init_key)

        input_ids = jax.random.randint(data_key, shape=(3, seq_len), minval=1, maxval=vocab_size)
        test_inputs = dict(inputs=input_ids)
        outputs, _ = module.functional(
            layer,
            prng_key=fwd_key,
            state=state,
            inputs=dict(input_batch=test_inputs),
            is_training=False,
        )

        assert_allclose(jnp.mean(outputs), 0.0, atol=0.05)
        assert_allclose(jnp.std(outputs), 1.0, atol=0.05)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

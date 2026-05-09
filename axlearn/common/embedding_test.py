# Copyright © 2023 Apple Inc.

"""Test embedding layers."""

# pylint: disable=no-self-use
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import module, utils
from axlearn.common.attention import LearnedPositionalEmbedding
from axlearn.common.embedding import ModalityEmbedding, ModalityVocabInfo, TransformerTextEmbeddings
from axlearn.common.golden import load_golden
from axlearn.common.layers import Embedding, LayerNorm
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose


class TestTransformerTextEmbeddings(TestCase):
    """Tests TransformerTextEmbeddings."""

    @parameterized.parameters(False, True)
    def test_against_hf_bert_embeddings(self, use_explicit_positions: bool):
        hidden_dim = 12
        vocab_size = 24
        source_length = 11
        type_vocab_size = 1
        layer_norm_epsilon = 1e-05

        pos_emb_cfg = LearnedPositionalEmbedding.default_config().set(shape=(source_length,))
        emb = TransformerTextEmbeddings.default_config().set(
            dim=hidden_dim,
            vocab_size=vocab_size,
            pos_emb=pos_emb_cfg,
            type_emb=Embedding.default_config().set(num_embeddings=type_vocab_size),
            norm=LayerNorm.default_config().set(eps=layer_norm_epsilon),
        )

        layer = emb.set(name="layer_test").instantiate(parent=None)

        golden = load_golden("axlearn.common.embedding_test", "test_against_hf_bert_embeddings")

        input_ids = golden["inputs"]["input_ids"]
        test_inputs = dict(inputs=input_ids)
        if use_explicit_positions:
            test_inputs["positions"] = np.arange(source_length)

        test_hidden_states, _ = F(
            layer,
            inputs=dict(input_batch=test_inputs),
            state=golden["params"],
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(test_hidden_states, golden["outputs"]["hidden_states"])

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


class TestModalityEmbedding(TestCase):
    """Tests ModalityEmbedding."""

    @staticmethod
    def _mock_config(dim: int, modality_vocab_info: ModalityVocabInfo) -> ModalityEmbedding.Config:
        class MockedEmbedding(ModalityEmbedding):
            """Mocked ModalityEmbedding."""

            def __init__(self, cfg, *, parent):
                super().__init__(cfg, parent=parent)
                cfg = self.config
                self.id_val = self.config.modality_vocab_info.vocab_start + 1
                self.logit_val = 2.0

            def forward(self, input_batch):
                return self.Output(
                    ids=jnp.ones_like(input_batch["placeholders"]) * self.id_val,
                    embeddings=input_batch["embeddings"],
                    paddings=None,
                    batch_idx=input_batch.get("batch_idx", None),
                )

            def attend(self, x):
                if self.config.modality_vocab_info.generate_logits:
                    return (
                        jnp.ones([*x.shape[:2], self.config.modality_vocab_info.vocab_size])
                        * self.logit_val
                    )
                return None

            def lookup_modality_embeddings(self, input_ids, accum):
                del input_ids
                return accum

        return MockedEmbedding.default_config().set(
            dim=dim,
            modality_vocab_info=modality_vocab_info,
        )

    @parameterized.parameters(True, False)
    def test_forward(self, is_training: bool):
        dim = 12
        modality_vocab_info = ModalityVocabInfo(
            modality_name="fake",
            placeholder_start=10,
            placeholder_end=20,
            vocab_start=100,
            vocab_end=200,
        )
        emb = self._mock_config(dim, modality_vocab_info).set(name="test").instantiate(parent=None)
        state = emb.initialize_parameters_recursively(jax.random.PRNGKey(0))

        # Constructs a dummy batch.
        placeholders = jnp.array(
            [
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
            ]
        )
        embeddings = placeholders[..., None] * jnp.ones([dim])
        input_batch = dict(placeholders=placeholders, embeddings=embeddings)

        # Test forward.
        outputs, _ = module.functional(
            emb,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=input_batch),
            is_training=is_training,
            drop_output_collections=(),
        )
        self.assertEqual(outputs.ids.shape, placeholders.shape)
        assert_allclose(outputs.ids, emb.id_val)
        assert_allclose(outputs.embeddings, embeddings)

    @parameterized.product(
        is_training=[True, False],
        generate_logits=[True, False],
    )
    def test_attend(self, is_training: bool, generate_logits: bool):
        dim = 12
        modality_vocab_info = ModalityVocabInfo(
            modality_name="fake",
            placeholder_start=10,
            placeholder_end=20,
            vocab_start=100,
            vocab_end=200,
            generate_logits=generate_logits,
        )
        emb = self._mock_config(dim, modality_vocab_info).set(name="test").instantiate(parent=None)
        state = emb.initialize_parameters_recursively(jax.random.PRNGKey(0))

        # Test attend.
        seq_len = 4
        x = jax.random.normal(jax.random.PRNGKey(123), shape=(3, seq_len, dim))
        outputs, _ = module.functional(
            emb,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(x=x),
            is_training=is_training,
            drop_output_collections=(),
            method="attend",
        )
        if generate_logits:
            self.assertEqual(outputs.shape, (3, seq_len, modality_vocab_info.vocab_size))
            assert_allclose(outputs, emb.logit_val)
        else:
            self.assertEqual(outputs, None)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

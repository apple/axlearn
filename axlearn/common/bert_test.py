# Copyright © 2023 Apple Inc.

"""Tests BERT layers."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import bert, utils
from axlearn.common.attention import BaseStackedTransformerLayer
from axlearn.common.attention_bias import NEG_INF
from axlearn.common.golden import load_golden
from axlearn.common.layers import (
    BinaryClassificationMetric,
    Dropout,
    Embedding,
    LayerNorm,
    set_dropout_rate_recursively,
    set_layer_norm_eps_recursively,
)
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose

_MODULE_NAME = "axlearn.common.bert_test"


# TODO(markblee): Consider adding a shared util to convert HF to AXLearn config, something like a
# config_converter, to avoid the many variants of this code.


def dummy_inputs_for_mlm(
    batch_size: int,
    max_seq_len: int,
    vocab_size: int,
    type_vocab_size: int,
    mask_input_id: int,
    padding_input_id: int,
    ignored_target_id: int,
):
    """Builds dummy inputs for AXLearn and Hugging Face BERT."""
    # pylint: disable=import-outside-toplevel
    from axlearn.common.param_converter import as_torch_tensor
    from axlearn.common.test_utils import dummy_padding_mask

    # pylint: enable=import-outside-toplevel

    rng = np.random.default_rng(seed=123)
    attention_mask = 1 - dummy_padding_mask(batch_size=batch_size, max_seq_len=max_seq_len)

    targets_mask = rng.integers(0, 2, size=(batch_size, max_seq_len)).astype(bool)
    targets_mask = np.logical_and(targets_mask, np.logical_not(attention_mask))

    targets = np.full((batch_size, max_seq_len), ignored_target_id)
    targets_vals = rng.integers(1, vocab_size, size=targets.shape)
    np.putmask(targets, targets_mask, targets_vals)

    input_ids = rng.choice(
        list(range(1, mask_input_id)) + list(range(mask_input_id + 1, vocab_size)),
        size=(batch_size, max_seq_len),
    )
    input_type_ids = rng.integers(0, type_vocab_size, size=(batch_size, max_seq_len))
    np.putmask(input_ids, targets_mask, mask_input_id)

    np.putmask(input_ids, attention_mask, padding_input_id)
    hf_attention_mask = np.logical_not(attention_mask)

    test_inputs = dict(
        input_ids=input_ids,
        token_type_ids=input_type_ids,
        target_labels=targets,
    )
    ref_inputs = dict(
        input_ids=as_torch_tensor(input_ids),
        attention_mask=as_torch_tensor(hf_attention_mask),
        token_type_ids=as_torch_tensor(input_type_ids),
        labels=as_torch_tensor(targets),
    )
    return test_inputs, ref_inputs


def bert_encoder_config_from_hf(
    hf_cfg,
    vocab_size: Optional[int] = None,
    layer_norm_epsilon: Optional[float] = None,
    dropout_rate: Optional[float] = None,
    base_cfg: Optional[BaseStackedTransformerLayer.Config] = None,
) -> bert.Encoder.Config:
    encoder_cfg = bert.Encoder.default_config().set(
        dim=hf_cfg.hidden_size,
        vocab_size=vocab_size or hf_cfg.vocab_size,
        dropout_rate=hf_cfg.hidden_dropout_prob if dropout_rate is None else dropout_rate,
        emb=bert.bert_embedding_config(
            type_vocab_size=hf_cfg.type_vocab_size,
            max_position_embeddings=hf_cfg.max_position_embeddings,
        ),
        transformer=bert.bert_transformer_config(
            num_layers=hf_cfg.num_hidden_layers,
            num_heads=hf_cfg.num_attention_heads,
            base_cfg=base_cfg,
        ),
        pad_token_id=0,
    )
    set_layer_norm_eps_recursively(encoder_cfg, layer_norm_epsilon or hf_cfg.layer_norm_eps)
    return encoder_cfg


class BertTest(TestCase):
    def setUp(self):
        super().setUp()
        self.dtype = jnp.float32
        self.ignored_target_id = -100  # HF only computes loss for labels in [0, vocab_size].
        self.mask_input_id = 1  # [MASK] token
        self.pad_token_id = 0  # [PAD] token
        self.ref_cfg = SimpleNamespace(
            vocab_size=24,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=20,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            hidden_act="gelu_new",
            classifier_dropout=0.0,
            num_labels=2,
            layer_norm_eps=1e-12,
        )

    def test_attention_mask(self):
        """Test attention masking (padding tokens)."""
        encoder_cfg = bert_encoder_config_from_hf(self.ref_cfg).set(name="layer_test")
        layer = encoder_cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        # Test a "realistic" input case, where padding tokens always trail.
        input_ids = jnp.array(
            [
                [1, 2, 3, self.pad_token_id, self.pad_token_id],
                [4, 5, 6, 7, self.pad_token_id],
            ]
        )
        actual, _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(
                input_ids=input_ids,
                segment_ids=None,
            ),
            method="compute_attention_logit_biases",
        )
        # [batch_size, seq_len, 1].
        query_is_padding_bias = (input_ids == self.pad_token_id)[:, :, None] * NEG_INF
        # Remove num_heads dim and ignore padding queries.
        actual = jnp.squeeze(actual, axis=1) + query_is_padding_bias
        # fmt: off
        # Note that False indicates that we can attend to those positions.
        expected = jnp.array([
            [[False, False, False, True, True],
             [False, False, False, True, True],
             [False, False, False, True, True],
             [True,  True,  True, True, True],
             [True,  True,  True, True, True]],

            [[False, False, False, False, True],
             [False, False, False, False, True],
             [False, False, False, False, True],
             [False, False, False, False, True],
             [True,  True,  True,  True, True]],
        ])
        # fmt: on
        expected = expected * NEG_INF
        assert_allclose(jnp.exp(expected), jnp.exp(actual))

        # Test an unrealistic but more general input case, where padding tokens can appear anywhere.
        input_ids = jnp.array(
            [
                [self.pad_token_id, 2, self.pad_token_id, 4, 5],
                [4, self.pad_token_id, 6, 7, self.pad_token_id],
            ]
        )
        actual, _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(
                input_ids=input_ids,
                segment_ids=None,
            ),
            method="compute_attention_logit_biases",
        )
        # [batch_size, seq_len, 1].
        query_is_padding_bias = (input_ids == self.pad_token_id)[:, :, None] * NEG_INF
        # Remove num_heads dim and ignore padding queries.
        actual = jnp.squeeze(actual, axis=1) + query_is_padding_bias
        # fmt: off
        expected = jnp.array([
            [[True,  True, True,  True,  True],
             [True, False, True, False, False],
             [True,  True, True,  True,  True],
             [True, False, True, False, False],
             [True, False, True, False, False]],

            [[False, True, False, False, True],
             [ True, True,  True,  True, True],
             [False, True, False, False, True],
             [False, True, False, False, True],
             [ True, True,  True,  True, True]],
        ])
        # fmt: on
        expected = expected * NEG_INF
        assert_allclose(jnp.exp(expected), jnp.exp(actual))

        # A more intuitive, programmatic check.
        for i in range(input_ids.shape[0]):  # batch
            for j in range(input_ids.shape[1]):  # seq_len
                for k in range(input_ids.shape[1]):  # seq_len
                    if self.pad_token_id in (input_ids[i, j], input_ids[i, k]):
                        self.assertLessEqual(actual[i, j, k], NEG_INF)
                    else:
                        self.assertLessEqual(actual[i, j, k], 0)

    def test_dropout_rate(self):
        """Test dropout rate is properly set for all child layers."""
        dropout_rate = 0.123
        ref_cfg = self.ref_cfg
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg, dropout_rate=dropout_rate),
            head=bert.bert_lm_head_config(
                base_cfg=cfg.head,  # pylint: disable=no-member
                layer_norm_epsilon=ref_cfg.layer_norm_eps,
                ignored_target_id=self.ignored_target_id,
                vocab_size=ref_cfg.vocab_size,
            ),
            dim=ref_cfg.hidden_size,
            vocab_size=ref_cfg.vocab_size,
        )
        layer = cfg.instantiate(parent=None)
        dropout_rates = []

        def get_dropout_rate_recursively(layer: Module):
            if isinstance(layer, Dropout):
                dropout_rates.append(layer.config.rate)
                return
            for _, child in layer.children.items():
                get_dropout_rate_recursively(child)

        get_dropout_rate_recursively(layer)
        assert_allclose(dropout_rates, [dropout_rate] * len(dropout_rates))

    @parameterized.parameters(jnp.float32, jnp.float16, jnp.bfloat16)
    def test_layer_norm_cfg(self, dtype):
        """Test layer norm is properly configured based on dtype."""
        ref_cfg = self.ref_cfg
        model_cfg = bert.bert_model_config(vocab_size=ref_cfg.vocab_size, dtype=dtype)
        layer = model_cfg.set(name="test").instantiate(parent=None)
        all_layer_norm_eps = []

        def get_layer_norm_eps_recursively(layer: Module):
            if isinstance(layer, LayerNorm):
                all_layer_norm_eps.append(layer.config.eps)
                return
            for _, child in layer.children.items():
                get_layer_norm_eps_recursively(child)

        get_layer_norm_eps_recursively(layer)

        assert_allclose(
            all_layer_norm_eps,
            [bert.bert_layer_norm_epsilon(dtype=dtype)] * len(all_layer_norm_eps),
        )

    def _bert_mlm_config(self, ref_cfg=None) -> bert.BertModel.Config:
        ref_cfg = ref_cfg or self.ref_cfg
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg),
            dim=ref_cfg.hidden_size,
            head=bert.bert_lm_head_config(
                base_cfg=cfg.head,  # pylint: disable=no-member
                layer_norm_epsilon=ref_cfg.layer_norm_eps,
                ignored_target_id=self.ignored_target_id,
                vocab_size=ref_cfg.vocab_size,
            ),
            vocab_size=ref_cfg.vocab_size,
        )
        return cfg

    def test_for_mlm(self):
        """Test BertModel MLM without masking. In the MLM case, pooler output is disabled."""
        cfg = self._bert_mlm_config()
        layer = cfg.instantiate(parent=None)

        golden = load_golden(_MODULE_NAME, "test_for_mlm")
        input_ids = golden["inputs"]["input_ids"]
        token_type_ids = golden["inputs"]["token_type_ids"]
        target_labels = golden["inputs"]["target_labels"]

        (loss, aux_outputs), _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=golden["params"],
            inputs=dict(
                input_batch=dict(
                    input_ids=input_ids, token_type_ids=token_type_ids, target_labels=target_labels
                ),
                return_aux=True,
            ),
        )
        padding_mask = (np.asarray(input_ids) != self.pad_token_id)[..., None]

        assert_allclose(
            aux_outputs["sequence_output"] * padding_mask,
            golden["outputs"]["hidden_states_last"] * padding_mask,
        )
        assert_allclose(
            aux_outputs["logits"] * padding_mask,
            golden["outputs"]["logits"] * padding_mask,
        )
        assert_allclose(loss, golden["outputs"]["loss"])

    def test_for_mlm_with_padding(self):
        """Test BertModel MLM attention with padding. In the MLM case, pooler output is disabled."""
        cfg = self._bert_mlm_config()
        layer = cfg.instantiate(parent=None)

        golden = load_golden(_MODULE_NAME, "test_for_mlm_with_padding")
        input_ids = golden["inputs"]["input_ids"]
        token_type_ids = golden["inputs"]["token_type_ids"]
        target_labels = golden["inputs"]["target_labels"]

        (loss, aux_outputs), _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=golden["params"],
            inputs=dict(
                input_batch=dict(
                    input_ids=input_ids, token_type_ids=token_type_ids, target_labels=target_labels
                ),
                return_aux=True,
            ),
        )
        padding_mask = (np.asarray(input_ids) != self.pad_token_id)[..., None]

        assert_allclose(
            aux_outputs["sequence_output"] * padding_mask,
            golden["outputs"]["hidden_states_last"] * padding_mask,
        )
        assert_allclose(
            aux_outputs["logits"] * padding_mask,
            golden["outputs"]["logits"] * padding_mask,
        )
        assert_allclose(loss, golden["outputs"]["loss"])

    def test_loss_metrics(self):
        """Test loss function and metrics."""
        ref_cfg = self.ref_cfg
        cfg = bert.bert_lm_head_config(
            layer_norm_epsilon=ref_cfg.layer_norm_eps,
            ignored_target_id=self.ignored_target_id,
            vocab_size=ref_cfg.vocab_size,
        )
        cfg.set(
            name="test_loss",
            num_classes=3,
            input_dim=1,
            inner_head=Embedding.default_config().set(num_embeddings=1),
        )
        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # fmt: off
        logits = jnp.array([
            [[0., 1., 0.],   # correct
             [1., 0., 0.]],  # masked
            [[1., 0., 0.],   # incorrect
             [1., 0., 0.]],  # correct
        ])
        targets = jnp.array([
            [1, self.ignored_target_id],
            [2, 0],
        ])
        # fmt: on

        _, output_collection = F(
            layer,
            method="loss",
            inputs=dict(logits=logits, target_labels=targets),
            state=state,
            is_training=False,
            prng_key=jax.random.PRNGKey(2),
        )
        assert_allclose(output_collection.summaries["metric"]["accuracy"].mean, 2.0 / 3.0)

    @parameterized.parameters(1, 2, 3)
    def test_sequence_classification(self, num_classes: int):
        ref_cfg = self.ref_cfg

        # Construct our layer.
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg),
            head=bert.BertSequenceClassificationHead.default_config().set(
                num_classes=num_classes,
            ),
            dim=ref_cfg.hidden_size,
            vocab_size=ref_cfg.vocab_size,
        )
        set_dropout_rate_recursively(cfg.head, ref_cfg.classifier_dropout)
        layer = cfg.instantiate(parent=None)

        golden = load_golden(_MODULE_NAME, f"test_sequence_classification_{num_classes}")
        input_ids = jnp.asarray(golden["inputs"]["input_ids"])
        target_labels = jnp.asarray(golden["inputs"]["target_labels"])

        (loss, aux_outputs), _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=golden["params"],
            inputs=dict(
                input_batch=dict(
                    input_ids=input_ids,
                    target_labels=target_labels,
                ),
                return_aux=True,
            ),
        )

        # Compare.
        assert_allclose(aux_outputs["logits"], golden["outputs"]["logits"])
        assert_allclose(loss, golden["outputs"]["loss"])

    def test_multiple_choice_classification(self):
        num_classes = 2
        ref_cfg = self.ref_cfg

        # Construct our layer.
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg),
            head=bert.BertMultipleChoiceHead.default_config().set(
                num_classes=num_classes,
            ),
            dim=ref_cfg.hidden_size,
            vocab_size=ref_cfg.vocab_size,
        )
        set_dropout_rate_recursively(cfg.head, ref_cfg.classifier_dropout)
        layer = cfg.instantiate(parent=None)

        golden = load_golden(_MODULE_NAME, "test_multiple_choice_classification")
        input_ids = jnp.asarray(golden["inputs"]["input_ids"])
        token_type_ids = jnp.asarray(golden["inputs"]["token_type_ids"])
        target_labels = jnp.asarray(golden["inputs"]["target_labels"])

        batch_size = input_ids.shape[0]

        (loss, aux_outputs), _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=golden["params"],
            inputs=dict(
                input_batch=dict(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    target_labels=target_labels,
                ),
                return_aux=True,
            ),
        )

        # Compare.
        assert_allclose(
            aux_outputs["logits"].reshape((batch_size, num_classes)),
            golden["outputs"]["logits"],
        )
        assert_allclose(loss, golden["outputs"]["loss"])

    # pylint: disable=duplicate-code
    @parameterized.parameters(2, 3)
    def test_sequence_binary_classification(self, num_classes: int):
        ref_cfg = self.ref_cfg

        # Construct our layer.
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg),
            head=bert.BertSequenceClassificationHead.default_config().set(
                num_classes=num_classes,
                metric=BinaryClassificationMetric.default_config(),
            ),
            dim=ref_cfg.hidden_size,
            vocab_size=ref_cfg.vocab_size,
        )
        set_dropout_rate_recursively(cfg.head, ref_cfg.classifier_dropout)
        layer = cfg.instantiate(parent=None)

        should_raise = num_classes != 2
        ctx = nullcontext()
        if should_raise:
            ctx = self.assertRaisesRegex(ValueError, "only defined for two classes")

        with ctx:
            if not should_raise:
                golden = load_golden(_MODULE_NAME, "test_sequence_binary_classification")
                input_ids = jnp.asarray(golden["inputs"]["input_ids"])
                target_labels = jnp.asarray(golden["inputs"]["target_labels"])

                (loss, aux_outputs), _ = F(
                    layer,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    state=golden["params"],
                    inputs=dict(
                        input_batch=dict(
                            input_ids=input_ids,
                            target_labels=target_labels,
                        ),
                        return_aux=True,
                    ),
                )
                # Compare.
                assert_allclose(aux_outputs["logits"], golden["outputs"]["logits"])
                assert_allclose(loss, golden["outputs"]["loss"])
            else:
                # For num_classes=3, just verify it raises.
                batch_size = 4
                input_ids = jax.random.randint(
                    jax.random.PRNGKey(123),
                    shape=(batch_size, ref_cfg.max_position_embeddings),
                    minval=1,
                    maxval=ref_cfg.vocab_size,
                )
                target_labels = jax.random.randint(
                    jax.random.PRNGKey(321),
                    shape=(batch_size, num_classes),
                    minval=0,
                    maxval=num_classes,
                )
                params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
                F(
                    layer,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    state=params,
                    inputs=dict(
                        input_batch=dict(
                            input_ids=input_ids,
                            target_labels=target_labels,
                        ),
                        return_aux=True,
                    ),
                )

    # pylint: enable=duplicate-code
    def test_respect_custom_layer_norm_eps(self):
        expected_eps = 1e-6
        self.assertNotEqual(expected_eps, bert.bert_layer_norm_epsilon())
        transformer_cfg = bert.bert_transformer_config(
            num_layers=1, num_heads=2, layer_norm_epsilon=expected_eps
        )
        self.assertEqual(transformer_cfg.layer.self_attention.norm.eps, expected_eps)
        self.assertEqual(transformer_cfg.layer.feed_forward.norm.eps, expected_eps)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

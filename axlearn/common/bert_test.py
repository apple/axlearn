# Copyright © 2023 Apple Inc.

"""Tests BERT layers."""
from contextlib import nullcontext
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest, parameterized
from transformers.models.bert import modeling_bert as hf_bert

from axlearn.common import bert, utils
from axlearn.common.attention import NEG_INF, BaseStackedTransformerLayer
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
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose, dummy_padding_mask
from axlearn.common.torch_utils import parameters_from_torch_layer


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
    rng = np.random.default_rng(seed=123)
    attention_mask = 1 - dummy_padding_mask(batch_size=batch_size, max_seq_len=max_seq_len)

    # Build target mask (e.g. ignore non-[MASK] tokens).
    # A value of True indicates a [MASK] token, whereas False indicates non-[MASK].
    targets_mask = rng.integers(0, 2, size=(batch_size, max_seq_len)).astype(bool)
    # Target should not be [MASK] at padding positions.
    targets_mask = np.logical_and(targets_mask, np.logical_not(attention_mask))

    # Build targets.
    targets = np.full((batch_size, max_seq_len), ignored_target_id)
    targets_vals = rng.integers(1, vocab_size, size=targets.shape)
    # putmask assigns targets to targets_vals at the locations where targets_mask is True.
    # Here we assign some targets to be != ignored_target_id, so they're included in loss.
    np.putmask(targets, targets_mask, targets_vals)

    # Build inputs (avoiding [MASK] tokens for now).
    input_ids = rng.choice(
        list(range(1, mask_input_id)) + list(range(mask_input_id + 1, vocab_size)),
        size=(batch_size, max_seq_len),
    )
    input_type_ids = rng.integers(0, type_vocab_size, size=(batch_size, max_seq_len))
    # Here we assign inputs to be [MASK] where targets are != ignored_target_id.
    np.putmask(input_ids, targets_mask, mask_input_id)

    # Build input padding mask.
    # Our padding mask is encoded in the inputs directly as `padding_input_id`.
    np.putmask(input_ids, attention_mask, padding_input_id)
    # hf expects the opposite masking scheme as we do:
    # A float value of 0. represents padding and 1. represents non-padding.
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


# TODO(markblee): Consider adding a shared util to convert HF to AXLearn config, something like a
# config_converter, to avoid the many variants of this code.
def bert_encoder_config_from_hf(
    hf_cfg: hf_bert.BertConfig,
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
        self.ref_cfg = hf_bert.BertConfig(
            vocab_size=24,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=20,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            # Note: the results are slightly different between gelu and gelu_new.
            # Reference:
            # https://github.com/huggingface/transformers/blob/215e0681e4c3f6ade6e219d022a5e640b42fcb76/src/transformers/activations.py#L27-L37
            hidden_act="gelu_new",
            classifier_dropout=0.0,
            num_labels=2,
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
        (actual, _) = F(
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
        (actual, _) = F(
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

    def _bert_mlm_config(
        self, ref_cfg: Optional[hf_bert.BertConfig] = None
    ) -> bert.BertModel.Config:
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
        ref_cfg = self.ref_cfg
        ref_layer = hf_bert.BertForMaskedLM(ref_cfg).eval()

        cfg = self._bert_mlm_config()
        layer = cfg.instantiate(parent=None)

        batch_size = 3
        max_seq_len = ref_cfg.max_position_embeddings
        rng = np.random.default_rng(seed=123)

        # Build target mask (e.g. ignore non-[MASK] tokens).
        # A value of True indicates a [MASK] token, whereas False indicates non-[MASK].
        targets_mask = rng.integers(0, 2, size=(batch_size, max_seq_len)).astype(bool)

        # Build targets.
        targets = np.full((batch_size, max_seq_len), self.ignored_target_id)
        targets_vals = rng.integers(1, ref_cfg.vocab_size, size=targets.shape)
        # putmask assigns targets to targets_vals at the locations where targets_mask is True.
        # Here we assign some targets to be != ignored_target_id, so they're included in loss.
        np.putmask(targets, targets_mask, targets_vals)

        # Build inputs.
        input_ids = rng.integers(1, ref_cfg.vocab_size, size=(batch_size, max_seq_len))
        token_type_ids = rng.integers(1, ref_cfg.type_vocab_size, size=(batch_size, max_seq_len))
        # Here we assign inputs to be [MASK] where targets are != ignored_target_id.
        np.putmask(input_ids, targets_mask, self.mask_input_id)

        (loss, aux_outputs), ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(
                input_batch=dict(
                    input_ids=input_ids, token_type_ids=token_type_ids, target_labels=targets
                ),
                return_aux=True,
            ),
            ref_inputs=dict(
                input_ids=as_torch_tensor(input_ids),
                token_type_ids=as_torch_tensor(token_type_ids),
                labels=as_torch_tensor(targets),
                position_ids=None,
                output_hidden_states=True,
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )
        padding_mask = (input_ids != self.pad_token_id)[..., None]

        assert_allclose(
            aux_outputs["sequence_output"] * padding_mask,
            utils.as_tensor(ref_outputs.hidden_states[-1]) * padding_mask,
        )
        assert_allclose(
            aux_outputs["logits"] * padding_mask,
            utils.as_tensor(ref_outputs.logits) * padding_mask,
        )
        assert_allclose(loss, utils.as_tensor(ref_outputs.loss))

    def test_for_mlm_with_padding(self):
        """Test BertModel MLM attention with padding. In the MLM case, pooler output is disabled."""
        ref_cfg = self.ref_cfg
        ref_layer = hf_bert.BertForMaskedLM(ref_cfg).eval()

        cfg = self._bert_mlm_config()
        layer = cfg.instantiate(parent=None)

        test_inputs, ref_inputs = dummy_inputs_for_mlm(
            batch_size=3,
            max_seq_len=ref_cfg.max_position_embeddings,
            vocab_size=ref_cfg.vocab_size,
            type_vocab_size=ref_cfg.type_vocab_size,
            mask_input_id=self.mask_input_id,
            padding_input_id=self.pad_token_id,
            ignored_target_id=self.ignored_target_id,
        )

        (loss, aux_outputs), ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(
                input_batch=test_inputs,
                return_aux=True,
            ),
            ref_inputs=dict(
                **ref_inputs,
                output_hidden_states=True,
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )
        padding_mask = (test_inputs["input_ids"] != self.pad_token_id)[..., None]

        assert_allclose(
            aux_outputs["sequence_output"] * padding_mask,
            utils.as_tensor(ref_outputs.hidden_states[-1]) * padding_mask,
        )
        assert_allclose(
            aux_outputs["logits"] * padding_mask,
            utils.as_tensor(ref_outputs.logits) * padding_mask,
        )
        assert_allclose(loss, utils.as_tensor(ref_outputs.loss))

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
        ref_cfg = ref_cfg.from_dict(ref_cfg.to_dict())  # Copy.
        ref_cfg.num_labels = num_classes

        # Construct ref layer.
        ref_layer = hf_bert.BertForSequenceClassification(ref_cfg).eval()

        # Construct our layer.
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg),
            head=bert.BertSequenceClassificationHead.default_config().set(
                num_classes=ref_cfg.num_labels,
            ),
            dim=ref_cfg.hidden_size,
            vocab_size=ref_cfg.vocab_size,
        )
        set_dropout_rate_recursively(
            cfg.head, ref_cfg.classifier_dropout  # pylint: disable=no-member
        )
        layer = cfg.instantiate(parent=None)

        # Generate dummy inputs for sequence classification.
        batch_size = 4
        input_ids = jax.random.randint(
            jax.random.PRNGKey(123),
            shape=(batch_size, ref_cfg.max_position_embeddings),
            minval=1,
            maxval=ref_cfg.vocab_size,
        )
        padding_mask = dummy_padding_mask(
            batch_size=batch_size, max_seq_len=ref_cfg.max_position_embeddings
        )
        input_ids = input_ids * padding_mask
        if ref_cfg.num_labels == 1:
            target_labels = jax.random.uniform(
                jax.random.PRNGKey(234), shape=(batch_size,), minval=0, maxval=5
            )
            hf_labels = as_torch_tensor(target_labels)
        else:
            target_labels = jax.random.randint(
                jax.random.PRNGKey(432),
                shape=(batch_size,),
                minval=0,
                maxval=ref_cfg.num_labels,
            )
            hf_labels = as_torch_tensor(target_labels).to(torch.long)

        # Compute outputs.
        (loss, aux_outputs), ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(
                input_batch=dict(
                    input_ids=input_ids,
                    target_labels=target_labels,
                ),
                return_aux=True,
            ),
            ref_inputs=dict(
                input_ids=as_torch_tensor(input_ids),
                attention_mask=as_torch_tensor(padding_mask),
                labels=hf_labels,
                output_hidden_states=True,
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )

        # Compare.
        assert_allclose(aux_outputs["logits"], utils.as_tensor(ref_outputs.logits))
        assert_allclose(loss, utils.as_tensor(ref_outputs.loss))

    def test_multiple_choice_classification(self):
        num_classes = 2
        ref_cfg = self.ref_cfg
        ref_cfg = ref_cfg.from_dict(ref_cfg.to_dict())  # Copy.
        ref_cfg.num_labels = num_classes

        # Construct ref layer.
        ref_layer = hf_bert.BertForMultipleChoice(ref_cfg)

        # Construct our layer.
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg),
            head=bert.BertMultipleChoiceHead.default_config().set(
                num_classes=ref_cfg.num_labels,
            ),
            dim=ref_cfg.hidden_size,
            vocab_size=ref_cfg.vocab_size,
        )
        set_dropout_rate_recursively(
            cfg.head, ref_cfg.classifier_dropout  # pylint: disable=no-member
        )
        layer = cfg.instantiate(parent=None)

        # Generate dummy inputs for multiple choice classification.
        batch_size = 12
        input_ids = jax.random.randint(
            jax.random.PRNGKey(123),
            shape=(batch_size, num_classes, ref_cfg.max_position_embeddings),
            minval=1,
            maxval=ref_cfg.vocab_size,
        )
        token_type_ids = jax.random.randint(
            jax.random.PRNGKey(124),
            shape=(batch_size, num_classes, ref_cfg.max_position_embeddings),
            minval=0,
            maxval=ref_cfg.type_vocab_size,
        )
        padding_mask = dummy_padding_mask(
            batch_size=batch_size * num_classes, max_seq_len=ref_cfg.max_position_embeddings
        ).reshape((batch_size, num_classes, ref_cfg.max_position_embeddings))
        input_ids = input_ids * padding_mask
        target_labels = jax.random.randint(
            jax.random.PRNGKey(321), shape=(batch_size,), minval=0, maxval=num_classes
        )
        hf_input_ids = as_torch_tensor(input_ids)
        hf_token_type_ids = as_torch_tensor(token_type_ids)
        hf_labels = as_torch_tensor(target_labels).to(torch.long)
        hf_attention_mask = as_torch_tensor(padding_mask)

        # Compute outputs.
        (loss, aux_outputs), ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(
                input_batch=dict(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    target_labels=target_labels,
                ),
                return_aux=True,
            ),
            ref_inputs=dict(
                input_ids=hf_input_ids,
                token_type_ids=hf_token_type_ids,
                attention_mask=hf_attention_mask,
                labels=hf_labels,
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )

        # Compare.
        assert_allclose(
            aux_outputs["logits"].reshape((batch_size, num_classes)),
            utils.as_tensor(ref_outputs.logits),
        )
        assert_allclose(loss, utils.as_tensor(ref_outputs.loss))

    # pylint: disable=duplicate-code
    @parameterized.parameters(2, 3)
    def test_sequence_binary_classification(self, num_classes: int):
        ref_cfg = self.ref_cfg
        ref_cfg = ref_cfg.from_dict(ref_cfg.to_dict())  # Copy.
        ref_cfg.num_labels = num_classes
        ref_cfg.problem_type = "multi_label_classification"
        # Construct ref layer.
        ref_layer = hf_bert.BertForSequenceClassification(ref_cfg)

        # Construct our layer.
        cfg: bert.BertModel.Config = bert.BertModel.default_config()
        cfg = cfg.set(
            name="layer_test",
            encoder=bert_encoder_config_from_hf(ref_cfg),
            head=bert.BertSequenceClassificationHead.default_config().set(
                num_classes=ref_cfg.num_labels,
                metric=BinaryClassificationMetric.default_config(),
            ),
            dim=ref_cfg.hidden_size,
            vocab_size=ref_cfg.vocab_size,
        )
        set_dropout_rate_recursively(
            cfg.head, ref_cfg.classifier_dropout  # pylint: disable=no-member
        )
        layer = cfg.instantiate(parent=None)

        # Generate dummy inputs for sequence classification.
        batch_size = 4
        input_ids = jax.random.randint(
            jax.random.PRNGKey(123),
            shape=(batch_size, ref_cfg.max_position_embeddings),
            minval=1,
            maxval=ref_cfg.vocab_size,
        )
        padding_mask = dummy_padding_mask(
            batch_size=batch_size, max_seq_len=ref_cfg.max_position_embeddings
        )
        input_ids = input_ids * padding_mask
        target_labels = jax.random.randint(
            jax.random.PRNGKey(321),
            shape=(batch_size, num_classes),
            minval=0,
            maxval=num_classes,
        )
        should_raise = num_classes != 2
        ctx = nullcontext()
        if should_raise:
            ctx = self.assertRaisesRegex(ValueError, "only defined for two classes")
        with ctx:
            # Compute outputs.
            (loss, aux_outputs), ref_outputs = self._compute_layer_outputs(
                test_layer=layer,
                ref_layer=ref_layer,
                test_inputs=dict(
                    input_batch=dict(
                        input_ids=input_ids,
                        target_labels=target_labels,
                    ),
                    return_aux=True,
                ),
                ref_inputs=dict(
                    input_ids=as_torch_tensor(input_ids),
                    # HF expects float labels for multi-label classification.
                    labels=as_torch_tensor(target_labels).to(torch.float32),
                    attention_mask=as_torch_tensor(padding_mask),
                ),
                parameters_from_ref_layer=parameters_from_torch_layer,
            )
        if not should_raise:
            # Compare.
            assert_allclose(aux_outputs["logits"], utils.as_tensor(ref_outputs.logits))
            assert_allclose(loss, utils.as_tensor(ref_outputs.loss))

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

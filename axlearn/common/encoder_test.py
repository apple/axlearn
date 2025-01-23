# Copyright © 2023 Apple Inc.

"""Tests encoder layers."""

# pylint: disable=no-self-use
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from transformers import BertConfig, BertModel

from axlearn.common import utils
from axlearn.common.attention import (
    CausalAttentionLogitBiasLayer,
    MultiheadAttention,
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
    TransformerLayer,
)
from axlearn.common.bert import (
    BertSequenceClassificationHead,
    bert_embedding_config,
    bert_transformer_config,
)
from axlearn.common.bert_test import bert_encoder_config_from_hf
from axlearn.common.encoder import CausalEncoder, Encoder, EncoderModel
from axlearn.common.layers import BaseClassificationHead, set_layer_norm_eps_recursively
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.common.test_utils import (
    TestCase,
    assert_allclose,
    dummy_segments_positions,
    take_segment,
)
from axlearn.common.torch_utils import parameters_from_torch_layer


class TestEncoder(TestCase):
    """Tests Encoder layer."""

    def setUp(self):
        super().setUp()
        self.hf_encoder_cfg = BertConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=12,
            max_position_embeddings=11,
            vocab_size=24,
            intermediate_size=4 * 12,
            type_vocab_size=1,
            layer_norm_eps=1e-05,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        axlearn_encoder_cfg = bert_encoder_config_from_hf(
            self.hf_encoder_cfg,
            vocab_size=self.hf_encoder_cfg.vocab_size,
            layer_norm_epsilon=self.hf_encoder_cfg.layer_norm_eps,
            dropout_rate=self.hf_encoder_cfg.hidden_dropout_prob,
        )
        self.axlearn_encoder = axlearn_encoder_cfg.set(name="test").instantiate(parent=None)
        self.hf_encoder = BertModel(self.hf_encoder_cfg, add_pooling_layer=False).eval()

    # The configs here have similarities with encoder_decoder.py tests.
    def test_against_hf_encoder(self):
        hidden_dim = 12
        vocab_size = 24
        num_heads = 4
        num_layers = 2
        source_length = 11
        # Reference implementation.
        encoder_config = BertConfig(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_size=hidden_dim,
            max_position_embeddings=source_length,
            vocab_size=vocab_size,
            intermediate_size=4 * hidden_dim,
            type_vocab_size=1,
            layer_norm_eps=1e-05,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        model = BertModel(config=encoder_config, add_pooling_layer=False)
        ref_layer = model.eval()

        # Equivalent AXLearn implementation.
        encoder = Encoder.default_config().set(
            dim=hidden_dim,
            vocab_size=vocab_size,
            emb=bert_embedding_config(type_vocab_size=1, max_position_embeddings=source_length),
            transformer=bert_transformer_config(num_layers=num_layers, num_heads=num_heads),
            pad_token_id=0,
        )
        encoder.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None, scale=0.02, distribution="normal"
                )
            }
        )
        set_layer_norm_eps_recursively(encoder, 1e-5)

        layer = encoder.set(name="layer_test").instantiate(parent=None)
        batch_size = 3
        input_ids = np.random.randint(1, vocab_size, size=(batch_size, source_length))

        # Note: we don't use `_compute_layer_outputs` here for a specific reason:
        # The same Hugging Face model `BertModel` maps to both `encoder.Encoder` and
        # `bert.BertModel`, meaning we need to explicitly lookup `params_from_ref["encoder"]` here.
        test_hidden_states, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=parameters_from_torch_layer(ref_layer)["encoder"],
            inputs=dict(input_ids=input_ids),
        )
        ref_layer.eval()
        ref_outputs = ref_layer(input_ids=as_torch_tensor(input_ids), return_dict=True)

        assert_allclose(test_hidden_states, utils.as_tensor(ref_outputs["last_hidden_state"]))

    @parameterized.parameters(1, 2, 3)
    def test_embeddings(self, num_segments: int):
        batch_size = 3
        vocab_size = self.hf_encoder_cfg.vocab_size
        source_len = self.hf_encoder_cfg.max_position_embeddings
        type_vocab_size = self.hf_encoder_cfg.type_vocab_size
        source_ids = jax.random.randint(
            jax.random.PRNGKey(101),
            (batch_size, source_len),
            minval=0,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        source_type_ids = jax.random.randint(
            jax.random.PRNGKey(102),
            (batch_size, source_len),
            minval=0,
            maxval=type_vocab_size,
            dtype=jnp.int32,
        )
        source_segment_ids, source_positions = dummy_segments_positions(
            batch_size, source_len, num_segments=num_segments
        )

        for segment in range(num_segments):
            # Select inputs corresponding to the segment. [batch_size, source_len].
            hf_source_ids = take_segment(source_ids, source_segment_ids == segment)
            hf_source_type_ids = take_segment(source_type_ids, source_segment_ids == segment)

            # Compute ref and test embeddings. [batch_size, source_len, hidden_dim].
            test_outputs, _ = F(
                self.axlearn_encoder.emb,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                state=parameters_from_torch_layer(self.hf_encoder)["encoder"]["emb"],
                inputs=dict(
                    input_batch=dict(
                        inputs=source_ids,
                        token_type_ids=source_type_ids,
                        positions=source_positions,
                    ),
                ),
            )
            ref_outputs = self.hf_encoder.embeddings(
                input_ids=as_torch_tensor(hf_source_ids),
                token_type_ids=as_torch_tensor(hf_source_type_ids),
            )
            ref_outputs = utils.as_tensor(ref_outputs)

            # Compare only outputs corresponding to current segment, at the non-padding positions.
            # [batch_size, source_len, hidden_dim].
            test_outputs = take_segment(test_outputs, source_segment_ids == segment)
            mask = utils.as_tensor(hf_source_ids != 0)[..., None]
            assert_allclose(test_outputs * mask, ref_outputs * mask)

    def test_dropout_rate(self):
        hidden_dim = 12
        num_heads = 4
        vocab_size = 24
        source_length = 11
        dropout_rate = 0.1
        num_layers = 2
        encoder = Encoder.default_config().set(
            dim=hidden_dim,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
            emb=bert_embedding_config(type_vocab_size=1, max_position_embeddings=source_length),
            transformer=bert_transformer_config(num_layers=num_layers, num_heads=num_heads),
            pad_token_id=0,
        )
        encoder.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None, scale=0.02, distribution="normal"
                )
            }
        )
        set_layer_norm_eps_recursively(encoder, 1e-5)
        layer_test = encoder.set(name="layer_test").instantiate(parent=None)
        self.assertEqual(layer_test.emb.dropout.config.rate, dropout_rate)
        for i in range(num_layers):
            transformer_layer = getattr(layer_test.transformer, f"layer{i}")
            self.assertEqual(transformer_layer.self_attention.dropout.config.rate, dropout_rate)
            self.assertEqual(transformer_layer.feed_forward.dropout.config.rate, dropout_rate)


class TestCausalEncoder(TestCase):
    """Tests CausalEncoder layer."""

    @parameterized.product(
        prefill_states=[True, False],
        prefix_zero=[True, False],
    )
    def test_extend_step(self, prefill_states: bool, prefix_zero: bool):
        hidden_dim = 12
        vocab_size = 24
        num_heads = 4
        num_layers = 2
        source_length = 11

        encoder = CausalEncoder.default_config().set(
            dim=hidden_dim,
            vocab_size=vocab_size,
            dropout_rate=0,
            attention_mask=CausalAttentionLogitBiasLayer.default_config(),
            emb=bert_embedding_config(type_vocab_size=1, max_position_embeddings=source_length),
            transformer=bert_transformer_config(num_layers=num_layers, num_heads=num_heads),
            param_init=DefaultInitializer.default_config().set(
                init_by_param_name={
                    PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                        fan=None, scale=0.02, distribution="normal"
                    )
                }
            ),
            pad_token_id=0,
        )
        set_layer_norm_eps_recursively(encoder, 1e-5)

        layer = encoder.set(name="layer_test").instantiate(parent=None)
        batch_size = 3

        # We ignore padding ids (0) for now to simplify the mask generation process.
        if prefix_zero:
            prefix = jnp.zeros([batch_size, 1], dtype=jnp.int32)
        else:
            prefix = jax.random.randint(
                jax.random.PRNGKey(123), [batch_size, 1], minval=1, maxval=vocab_size - 1
            )
        input_ids = jax.random.randint(
            jax.random.PRNGKey(123),
            [batch_size, source_length - 1],
            minval=1,
            maxval=vocab_size - 1,
        )
        input_ids = jnp.hstack([prefix, input_ids])

        params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        ref_hidden_states, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=params,
            inputs=dict(
                input_ids=input_ids,
                input_segment_ids=input_ids != 0,
                positions=jnp.arange(input_ids.shape[-1])[None, :],
            ),
        )
        ref_hidden_states = ref_hidden_states["hidden_states"]

        if prefill_states:
            time_step = jnp.arange(batch_size)
            (initial_state, initial_outputs), _ = F(
                layer,
                inputs=dict(time_step=time_step, input_ids=input_ids),
                state=params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
                method="prefill_states",
            )
            # Zero-out outputs starting from initial time_step, and test that we can recover the
            # full outputs by calling extend_step starting from time_step.
            # [batch, tgt_len].
            time_step_mask = jnp.arange(source_length) < time_step[:, None]
            # [batch, tgt_len, hidden_dim].
            hidden_states = initial_outputs["hidden_states"] * time_step_mask[:, :, None]
        else:
            time_step = jnp.zeros(batch_size, dtype=jnp.int32)
            initial_state = layer.init_states(
                batch_size=batch_size, max_sequence_length=source_length
            )
            hidden_states = jnp.zeros(shape=[batch_size, source_length, hidden_dim])

        # [batch, source_length, hidden_dim] --> [batch, hidden_dim, source_length].
        hidden_states = jnp.moveaxis(hidden_states, -2, -1)

        inputs = dict(cached_states=initial_state)
        while jnp.any(time_step < source_length):
            # [batch, source_length=1].
            inputs["input_ids"] = jnp.take_along_axis(
                input_ids, time_step[:, None], axis=1, mode="clip"
            )
            (updated_state, outputs), _ = F(
                layer,
                state=params,
                is_training=False,
                prng_key=jax.random.PRNGKey(123),
                inputs=inputs,
                method="extend_step",
            )
            inputs["cached_states"] = updated_state

            # [batch, hidden_dim, source_length=1].
            curr_hidden_states = jnp.moveaxis(outputs["hidden_states"], -2, -1)
            # [batch, 1, source_length].
            oh_indices = jax.nn.one_hot(time_step, source_length)[:, None, :]

            hidden_states = hidden_states + curr_hidden_states * oh_indices
            time_step = time_step + 1

        # [batch, hidden_dim, source_length] --> [batch, source_length, hidden_dim].
        hidden_states = jnp.moveaxis(hidden_states, -1, -2)
        assert_allclose(hidden_states, ref_hidden_states, atol=1e-5)


class TestEncoderModel(TestCase):
    """Tests Encoder model."""

    @parameterized.parameters(
        dict(head=None),
        dict(head=BertSequenceClassificationHead.default_config().set(num_classes=2)),
    )
    def test_encoder_model_with_head_config_variation(
        self, head: Optional[BaseClassificationHead.Config]
    ):
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        hidden_dim = 64
        config = EncoderModel.default_config().set(
            dim=64,
            vocab_size=vocab_size,
            head=head,
        )
        config.encoder.set(
            pad_token_id=0,
            transformer=TransformerLayer.default_config().set(
                self_attention=TransformerAttentionLayer.default_config().set(
                    attention=MultiheadAttention.default_config().set(num_heads=1)
                ),
                feed_forward=TransformerFeedForwardLayer.default_config().set(
                    hidden_dim=hidden_dim
                ),
            ),
        )
        encoder_model: EncoderModel = config.set(name="test").instantiate(parent=None)
        params = encoder_model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        input_ids = jax.random.randint(
            jax.random.PRNGKey(111), [batch_size, seq_len], minval=0, maxval=vocab_size
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(123), [batch_size], minval=0, maxval=1
        )
        input_batch = dict(input_ids=input_ids, target_labels=target_labels)
        test_outputs, _ = F(
            encoder_model,
            inputs=[input_batch],
            state=params,
            is_training=False,
            method="forward",
            prng_key=jax.random.PRNGKey(1),
        )
        self.assertEqual(test_outputs[0] is None, head is None)

        test_outputs, _ = F(
            encoder_model,
            inputs=[input_batch],
            state=params,
            is_training=False,
            method="predict",
            prng_key=jax.random.PRNGKey(1),
        )
        self.assertIn("sequence_output", test_outputs)
        self.assertEqual(test_outputs["sequence_output"].shape, (batch_size, seq_len, hidden_dim))

        if head is None:
            self.assertNotIn("logits", test_outputs)
        else:
            self.assertIn("logits", test_outputs)
            self.assertEqual(test_outputs["logits"].shape, (batch_size, head.num_classes))

    def test_encoder_model_with_soft_labels_passed(self):
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        hidden_dim = 64
        num_classes = 2
        head = BertSequenceClassificationHead.default_config().set(num_classes=num_classes)
        config = EncoderModel.default_config().set(
            dim=64,
            vocab_size=vocab_size,
            head=head,
        )
        config.encoder.set(
            pad_token_id=0,
            transformer=TransformerLayer.default_config().set(
                self_attention=TransformerAttentionLayer.default_config().set(
                    attention=MultiheadAttention.default_config().set(num_heads=1)
                ),
                feed_forward=TransformerFeedForwardLayer.default_config().set(
                    hidden_dim=hidden_dim
                ),
            ),
        )
        encoder_model: EncoderModel = config.set(name="test").instantiate(parent=None)
        params = encoder_model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        input_ids = jax.random.randint(
            jax.random.PRNGKey(111), [batch_size, seq_len], minval=0, maxval=vocab_size
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(123), [batch_size], minval=0, maxval=1
        )

        input_batch = dict(input_ids=input_ids, target_labels=target_labels)
        output, _ = F(
            encoder_model,
            inputs=[input_batch],
            state=params,
            is_training=False,
            method="forward",
            prng_key=jax.random.PRNGKey(1),
        )

        soft_labels = jax.random.uniform(
            jax.random.PRNGKey(42),
            [batch_size, num_classes],
            minval=-1,
            maxval=1,
        )
        input_batch_with_soft_labels = dict(
            input_ids=input_ids, target_labels=target_labels, soft_labels=soft_labels
        )
        output_with_soft_labels, _ = F(
            encoder_model,
            inputs=[input_batch_with_soft_labels],
            state=params,
            is_training=False,
            method="forward",
            prng_key=jax.random.PRNGKey(1),
        )

        self.assertNotAlmostEqual(output[0].item(), output_with_soft_labels[0].item())


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

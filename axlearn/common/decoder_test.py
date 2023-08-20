# Copyright © 2023 Apple Inc.

"""Tests decoder layers."""
# pylint: disable=no-self-use,too-many-branches
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import decoding, utils
from axlearn.common.attention import (
    NEG_INF,
    ALiBiAttentionLogitBiasLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.config import InstantiableConfig
from axlearn.common.decoder import LmHead, _segment_ids_from_causal_input_ids
from axlearn.common.module import functional
from axlearn.common.test_utils import TestCase, assert_allclose


class TestDecoder(TestCase):
    """Tests decoder layers."""

    def test_tied_lm_head_differs_from_untied(self):
        hidden_dim = 12
        num_heads = 4
        vocab_size = 24
        source_length = 11

        # Similarities with encoder_decoder_test.
        # pylint: disable=duplicate-code
        decoder = gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=source_length,
        )
        # pylint: enable=duplicate-code

        tied_head = decoder.set(name="test_tied").instantiate(parent=None)
        tied_head_state = tied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        assert tied_head_state.get("lm_head") is None

        # Similarities with encoder_decoder_test.
        # pylint: disable=duplicate-code
        untied_decoder = gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=source_length,
        )
        # pylint: enable=duplicate-code

        untied_decoder.lm_head = LmHead.default_config()
        untied_head = untied_decoder.set(name="test_untied").instantiate(parent=None)
        untied_head_state = untied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        assert untied_head_state.get("lm_head") is not None

        inputs = jax.random.randint(
            jax.random.PRNGKey(1), minval=1, maxval=vocab_size, shape=(3, source_length)
        )

        # Test values.
        def layer_output(state, layer):
            return functional(
                layer,
                inputs=dict(input_ids=inputs),
                state=state,
                is_training=False,
                prng_key=jax.random.PRNGKey(2),
            )[0]["logits"]

        # Similarities with encoder_decoder_test.
        # pylint: disable=duplicate-code
        tied_logits = layer_output(tied_head_state, tied_head)
        untied_logits = layer_output(untied_head_state, untied_head)
        np.testing.assert_raises(AssertionError, assert_allclose, tied_logits, untied_logits)
        # pylint: enable=duplicate-code

        # Test grads.
        def layer_loss(state, layer):
            return layer_output(state, layer).sum()

        def check_grads(tied_state, untied_state):
            tied_head_grad = jax.grad(layer_loss)(tied_state, tied_head)["emb"]["token_emb"][
                "weight"
            ]
            untied_head_grad = jax.grad(layer_loss)(untied_state, untied_head)["emb"]["token_emb"][
                "weight"
            ]
            np.testing.assert_raises(
                AssertionError, assert_allclose, tied_head_grad, untied_head_grad
            )

        # Assert grad is different tied vs untied
        check_grads(tied_head_state, untied_head_state)
        # Set untied head weight to tied lm_head value and check again.
        untied_head_state["lm_head"]["weight"] = tied_head_state["emb"]["token_emb"]["weight"]
        check_grads(tied_head_state, untied_head_state)

    @parameterized.parameters(None, 0.0, 0.2)
    def test_dropout_rate(self, output_dropout_rate):
        hidden_dim = 12
        num_heads = 4
        vocab_size = 24
        source_length = 11
        num_layers = 2
        dropout_rate = 0.1
        decoder = gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=source_length,
            dropout_rate=dropout_rate,
        )
        self.assertIsNone(decoder.output_dropout.rate)
        if output_dropout_rate is not None:
            # Explicitly set output_dropout.rate. This decouples output_dropout.rate from
            # dropout_rate.
            decoder.output_dropout.rate = output_dropout_rate
        layer_test = decoder.set(name="layer_test").instantiate(parent=None)
        self.assertEqual(layer_test.emb.dropout.config.rate, dropout_rate)
        for i in range(num_layers):
            transformer_layer = getattr(layer_test.transformer, f"layer{i}")
            self.assertEqual(transformer_layer.self_attention.dropout.config.rate, dropout_rate)
            self.assertEqual(transformer_layer.feed_forward.dropout1.config.rate, dropout_rate)
            self.assertEqual(transformer_layer.feed_forward.dropout2.config.rate, dropout_rate)
        # Check potentially decoupled output_dropout.rate.
        self.assertEqual(
            layer_test.output_dropout.config.rate,
            dropout_rate if output_dropout_rate is None else output_dropout_rate,
        )

    @parameterized.product(
        use_cross_attention=[False, True],
        stack_cfg=[
            StackedTransformerLayer.default_config(),
            RepeatedTransformerLayer.default_config(),
        ],
        custom_attention_mask_cfg=[None, ALiBiAttentionLogitBiasLayer.default_config()],
        prefill_states=[True, False],
    )
    def test_extend_step(
        self,
        use_cross_attention: bool,
        stack_cfg: InstantiableConfig,
        custom_attention_mask_cfg: Optional[InstantiableConfig],
        prefill_states: bool,
    ):
        batch_size, src_len, tgt_len, vocab_size = 2, 11, 6, 24
        num_layers, num_heads = 2, 4
        hidden_dim, src_dim = 12, 10
        dropout_rate = 0.1

        cfg = gpt_decoder_config(
            stack_cfg=stack_cfg,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=tgt_len,
            dropout_rate=dropout_rate,
        )
        if custom_attention_mask_cfg:
            if isinstance(custom_attention_mask_cfg, ALiBiAttentionLogitBiasLayer.Config):
                # Set value for num_heads.
                custom_attention_mask_cfg.set(num_heads=num_heads)
            cfg.set(attention_mask=custom_attention_mask_cfg)
        if use_cross_attention:
            # Add cross attention
            cfg.transformer.layer.cross_attention = TransformerAttentionLayer.default_config().set(
                target_dim=hidden_dim,
                source_dim=src_dim,
            )
            cfg.transformer.layer.cross_attention.attention.num_heads = num_heads

        layer = cfg.set(name="test_extend_step").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # When prefilling, prefix can contain padding and eos.
        if prefill_states:
            input_ids = jax.random.randint(
                jax.random.PRNGKey(124),
                shape=[batch_size, tgt_len],
                minval=0,
                maxval=2,
            )
            # Prefix lengths.
            time_step = jnp.arange(batch_size)
            prefix_mask = jnp.arange(tgt_len) < time_step[:, None]
            # Explicitly fill positions >= prefix_length with pad_token_id.
            # Note that each batch example may have a different prefix length.
            # [batch_size, tgt_len].
            input_ids = input_ids * prefix_mask + cfg.pad_token_id * (1 - prefix_mask)
            # Set last token to a non-pad token, to fix the prefix length.
            oh_indices = jax.nn.one_hot(time_step, tgt_len, dtype=input_ids.dtype)
            input_ids = input_ids * (1 - oh_indices) + (cfg.pad_token_id + 1) * oh_indices
        else:
            # TODO(markblee): Remove this branch once beam search decode supports prefilling.
            time_step = jnp.zeros(batch_size, dtype=jnp.int32)
            input_ids = jnp.full([batch_size, tgt_len - 1], cfg.pad_token_id)
            input_ids = jnp.concatenate(
                [
                    input_ids,
                    jax.random.randint(
                        jax.random.PRNGKey(123), [batch_size, 1], minval=0, maxval=vocab_size - 1
                    ),
                ],
                axis=-1,
            )

        cross_attention_data = None
        cross_attention_logit_biases = None
        if use_cross_attention:
            cross_attention_data = jnp.ones([batch_size, src_len, src_dim])
            cross_attention_logit_biases = (
                jax.random.randint(
                    jax.random.PRNGKey(124),
                    shape=[batch_size, tgt_len, src_len],
                    minval=0,
                    maxval=2,
                )
                * NEG_INF
            )

        forward_outputs, _ = functional(
            layer,
            inputs=dict(
                input_ids=input_ids,
                input_segment_ids=jnp.ones_like(input_ids),
                positions=jnp.arange(input_ids.shape[-1])[None, :],
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        if prefill_states:
            (initial_state, initial_outputs), _ = functional(
                layer,
                inputs=dict(
                    time_step=time_step,
                    input_ids=input_ids,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                ),
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
                method="prefill_states",
            )
            # Zero-out outputs starting from initial time_step, and test that we can recover the
            # full outputs by calling extend_step starting from time_step.
            # [batch, tgt_len, num_classes].
            logits = initial_outputs["logits"] * prefix_mask[:, :, None]
        else:
            initial_state = layer.init_states(batch_size=batch_size, max_sequence_length=tgt_len)
            logits = jnp.zeros(shape=[batch_size, tgt_len, vocab_size])

        # [batch, tgt_len, num_classes] --> [batch, num_classes, tgt_len].
        logits = jnp.moveaxis(logits, -2, -1)

        inputs = dict(cached_states=initial_state)
        while jnp.any(time_step < tgt_len):
            # [batch, tgt_len=1].
            inputs["input_ids"] = jnp.take_along_axis(
                input_ids, time_step[:, None], axis=1, mode="clip"
            )
            if use_cross_attention:
                inputs["cross_attention_data"] = cross_attention_data
                # [batch, tgt_len=1, src_len].
                inputs["cross_attention_logit_biases"] = jnp.take_along_axis(
                    cross_attention_logit_biases,
                    time_step[:, None, None],
                    axis=1,
                    mode="clip",
                )
            (updated_state, outputs), _ = functional(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            inputs["cached_states"] = updated_state

            # [batch, num_classes, tgt_len=1].
            curr_logits = jnp.moveaxis(outputs["logits"], -2, -1)
            # [batch, 1, tgt_len].
            oh_indices = jax.nn.one_hot(time_step, tgt_len)[:, None, :]
            logits = logits + curr_logits * oh_indices
            time_step = time_step + 1

        # [batch, num_classes, tgt_len] --> [batch, tgt_len, num_classes].
        logits = jnp.moveaxis(logits, -1, -2)
        assert_allclose(logits, forward_outputs["logits"])

    @parameterized.product(
        stack_cfg=[
            StackedTransformerLayer.default_config(),
            RepeatedTransformerLayer.default_config(),
        ],
        cross_attention_mode=["none", "full", "broadcast"],
        num_decodes=[5],
        # Each is of shape [batch], representing per-example prefix lengths.
        prefix_length=[jnp.array([1, 1]), jnp.array([1, 3, 6])],
        method=["sample_decode", "beam_search_decode"],
        pad_token_id=[0, -1],
    )
    # pylint: disable-next=too-many-statements
    def test_decode(
        self,
        stack_cfg: InstantiableConfig,
        cross_attention_mode: Literal["none", "full", "broadcast"],
        num_decodes: int,
        prefix_length: utils.Tensor,
        method: Literal["sample_decode", "beam_search_decode"],
        pad_token_id: int,
    ):
        """Test beam search and sample decoding from a randomly initialized decoder."""
        with jax.checking_leaks():
            batch_size, src_len, tgt_len, vocab_size = prefix_length.shape[0], 11, 10, 6
            bos_id = eos_id = 1
            num_layers, num_heads = 3, 4
            hidden_dim, src_dim = 12, 10
            dropout_rate = 0.1

            if isinstance(stack_cfg, RepeatedTransformerLayer.Config):
                remat_spec = RematSpec(prevent_cse=False)
            else:
                remat_spec = None

            decoder = gpt_decoder_config(
                stack_cfg=stack_cfg,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                vocab_size=vocab_size,
                activation_function="nn.relu",
                max_position_embeddings=tgt_len,
                dropout_rate=dropout_rate,
                layer_remat=remat_spec,
            )
            decoder.set(pad_token_id=pad_token_id)

            cross_attention_data = None
            cross_attention_logit_biases = None
            if cross_attention_mode != "none":
                # Add cross attention
                decoder.transformer.layer.cross_attention = (
                    TransformerAttentionLayer.default_config().set(
                        target_dim=hidden_dim,
                        source_dim=src_dim,
                    )
                )
                decoder.transformer.layer.cross_attention.attention.num_heads = num_heads
                cross_attention_data = jnp.ones((batch_size, src_len, src_dim))

                if cross_attention_mode == "full":
                    cross_attention_tgt_len = tgt_len
                elif cross_attention_mode == "broadcast":
                    cross_attention_tgt_len = 1
                else:
                    raise ValueError(f"Unrecognized cross_attention_mode {cross_attention_mode}")

                cross_attention_logit_biases = (
                    jax.random.randint(
                        jax.random.PRNGKey(123),
                        shape=[batch_size, cross_attention_tgt_len, src_len],
                        minval=0,
                        maxval=2,
                    )
                    * NEG_INF
                )

            decoder_head = decoder.set(name="test_tied", eos_token_id=eos_id).instantiate(
                parent=None
            )
            decoder_head_state = decoder_head.initialize_parameters_recursively(
                jax.random.PRNGKey(0)
            )

            prefix = jax.random.randint(
                jax.random.PRNGKey(124),
                shape=[batch_size, tgt_len],
                # Prefix can consist of any tokens, including pad and eos.
                # TODO(markblee): Remove "else" once beam search decode supports prefilling.
                minval=0 if method == "sample_decode" else 2,
                maxval=vocab_size,
            )
            # Explicitly fill positions >= prefix_length with pad_token_id.
            # Note that each batch example may have a different prefix length.
            # [batch_size, tgt_len].
            prefix_mask = jnp.arange(tgt_len) < prefix_length[:, None]
            prefix = prefix * prefix_mask + pad_token_id * (1 - prefix_mask)
            # Set last token to a non-pad token, to fix the prefix length.
            oh_indices = jax.nn.one_hot(prefix_length - 1, tgt_len, dtype=prefix.dtype)
            prefix = prefix * (1 - oh_indices) + bos_id * oh_indices

            inputs = dict(
                prefix=prefix,
                max_sequence_length=tgt_len,
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
                num_decodes=num_decodes,
            )

            if method == "sample_decode":
                # Modify logits so that we will always sample the last token ID.
                inputs["logits_modifier"] = (
                    lambda logits: jnp.full_like(logits, decoding.NEG_INF).at[:, -1].set(0)
                )

            outputs, _ = functional(
                decoder_head,
                inputs=inputs,
                state=decoder_head_state,
                is_training=False,
                prng_key=jax.random.PRNGKey(2),
                method=method,
            )
            sequences = outputs.sequences
            self.assertTrue(sequences.shape == (batch_size, num_decodes, tgt_len))
            if method == "beam_search_decode":
                # Per sequence scores for beam search decode.
                self.assertTrue(outputs.scores.shape == (batch_size, num_decodes))
            elif method == "sample_decode":
                # Per token scores for sample-decoding.
                self.assertTrue(outputs.token_scores.shape == (batch_size, num_decodes, tgt_len))
            else:
                raise NotImplementedError(f"Don't know how to test method {method}.")

            # Shift prefix/mask to drop dummy BOS.
            prefix_mask = jnp.concatenate(
                [prefix_mask[:, 1:], jnp.zeros([batch_size, 1], dtype=prefix_mask.dtype)], axis=1
            )
            prefix = jnp.concatenate(
                [prefix[:, 1:], jnp.full([batch_size, 1], pad_token_id, dtype=prefix.dtype)], axis=1
            )
            # Expand num_heads dim.
            prefix_mask = prefix_mask[:, None, :]
            prefix = prefix[:, None, :]

            # Check that all hypotheses start with the prefix.
            # Note that mask excludes the dummy BOS token.
            self.assertTrue(jnp.all(sequences * prefix_mask == prefix * prefix_mask))
            # If sample-decoding, test that the remainder of the tokens are equal to
            # the last token-id (due to adding the logits modifier).
            if method == "sample_decode":
                self.assertTrue(
                    jnp.all(sequences * (1 - prefix_mask) == (vocab_size - 1) * (1 - prefix_mask))
                )


class UtilsTest(TestCase):
    @parameterized.parameters(
        dict(
            input_ids=jnp.array(
                [
                    [1, 0, 2, 3, 0, 0],
                    [1, 2, 3, 4, 0, 5],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            pad_token_id=0,
            expected_segment_ids=jnp.array(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        dict(
            input_ids=jnp.array(
                [
                    [1, -1, 2, 3, -1, -1],
                    [1, 2, 3, 4, -1, 5],
                    [0, 0, 0, -1, -1, -1],
                ]
            ),
            pad_token_id=-1,
            expected_segment_ids=jnp.array(
                [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                ]
            ),
        ),
    )
    def test_segment_ids_from_causal_input_ids(
        self,
        input_ids: utils.Tensor,
        pad_token_id: int,
        expected_segment_ids: utils.Tensor,
    ):
        self.assertTrue(
            jnp.all(
                _segment_ids_from_causal_input_ids(input_ids=input_ids, pad_token_id=pad_token_id)
                == expected_segment_ids
            )
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

# Copyright © 2023 Apple Inc.

"""Tests EncoderDecoder layers."""

from typing import Literal, Optional
from unittest import mock

import jax
import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import decoding, utils
from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.bert import bert_embedding_config, bert_transformer_config
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.decoder import Decoder, LmHead
from axlearn.common.encoder import Encoder
from axlearn.common.encoder_decoder import EncoderDecoderModel
from axlearn.common.golden import load_golden
from axlearn.common.layers import set_layer_norm_eps_recursively
from axlearn.common.module import functional as F
from axlearn.common.t5 import t5_encoder_decoder_config
from axlearn.common.t5x_param_converter import parameters_from_t5x_encoder_decoder
from axlearn.common.test_utils import TestCase, assert_allclose, dummy_padding_mask

_MODULE_NAME = "axlearn.common.encoder_decoder_test"


def set_decoder_cross_attention_config(
    decoder_cfg: Decoder.Config,
    num_heads: int,
):
    """Add cross attention to decoder config.

    Args:
        decoder_cfg: A config of Decoder.
        num_heads: Number of attention heads per transformer layer.
    """
    layer_cfg = decoder_cfg.transformer.layer
    # Cross attention transformer layer config.
    layer_cfg.cross_attention = TransformerAttentionLayer.default_config()
    layer_cfg.cross_attention.attention.num_heads = num_heads


def _model_config(
    *, vocab_size: int, source_len: int, target_len: int, remat_spec: Optional[RematSpec] = None
) -> EncoderDecoderModel.Config:
    hidden_dim = 12
    num_heads = 4
    encoder = Encoder.default_config().set(
        dim=hidden_dim,
        vocab_size=vocab_size,
        emb=bert_embedding_config(type_vocab_size=1, max_position_embeddings=source_len),
        transformer=bert_transformer_config(num_layers=2, num_heads=num_heads),
        pad_token_id=0,
    )
    set_layer_norm_eps_recursively(encoder, 1e-8)

    decoder = gpt_decoder_config(
        stack_cfg=StackedTransformerLayer.default_config(),
        num_layers=2,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        activation_function="nn.relu",
        max_position_embeddings=target_len,
        layer_remat=remat_spec,
    )
    set_decoder_cross_attention_config(decoder_cfg=decoder, num_heads=num_heads)
    return EncoderDecoderModel.default_config().set(decoder=decoder, encoder=encoder)


class TestEncoderDecoder(TestCase):
    """Tests EncoderDecoder layer."""

    def test_tied_lm_head_differs_from_untied(self):
        vocab_size = 24
        source_len = 11
        target_len = 5

        tied_cfg = _model_config(
            vocab_size=vocab_size, source_len=source_len, target_len=target_len
        )
        tied_head = tied_cfg.set(name="test_tied").instantiate(parent=None)
        tied_head_state = tied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        self.assertIsNone(tied_head_state.get("lm_head"))

        untied_cfg = tied_cfg.clone()
        untied_cfg.decoder.lm_head = LmHead.default_config()
        untied_head = untied_cfg.set(name="test_untied").instantiate(parent=None)
        untied_head_state = untied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        self.assertIsNotNone(untied_head_state.get("decoder").get("lm_head"))

        batch_size = 3
        source_ids = jax.random.randint(
            jax.random.PRNGKey(1), minval=1, maxval=vocab_size, shape=(batch_size, source_len)
        )
        target_ids = jnp.ones((batch_size, target_len), dtype=jnp.int32)
        target_labels = jax.random.randint(
            jax.random.PRNGKey(1), minval=1, maxval=vocab_size, shape=(batch_size, target_len)
        )

        # Test values.
        def layer_output(state, layer):
            input_batch = dict(
                source=dict(input_ids=source_ids),
                target=dict(input_ids=target_ids),
                target_labels=target_labels,
            )
            predictions = F(
                layer,
                inputs=dict(input_batch=input_batch),
                state=state,
                is_training=False,
                prng_key=jax.random.PRNGKey(2),
                method="predict",
            )[0]
            logits = F(
                layer.decoder,
                inputs=dict(forward_outputs=predictions),
                state=state["decoder"],
                is_training=False,
                prng_key=jax.random.PRNGKey(2),
                method="compute_logits",
            )[0]
            return logits

        tied_logits = layer_output(tied_head_state, tied_head)
        untied_logits = layer_output(untied_head_state, untied_head)
        np.testing.assert_raises(AssertionError, assert_allclose, tied_logits, untied_logits)

        # Test grads.
        def layer_loss(state, layer):
            return layer_output(state, layer).sum()

        def check_grads(tied_state, untied_state):
            tied_head_grad = jax.grad(layer_loss)(tied_state, tied_head)["decoder"]["emb"][
                "token_emb"
            ]["weight"]
            untied_head_grad = jax.grad(layer_loss)(untied_state, untied_head)["decoder"]["emb"][
                "token_emb"
            ]["weight"]
            np.testing.assert_raises(
                AssertionError, assert_allclose, tied_head_grad, untied_head_grad
            )

        # Assert grad is different tied vs untied
        check_grads(tied_head_state, untied_head_state)
        # Set untied head weight to tied lm_head value and check again.
        untied_head_state["decoder"]["lm_head"]["weight"] = tied_head_state["decoder"]["emb"][
            "token_emb"
        ]["weight"]
        check_grads(tied_head_state, untied_head_state)

    @parameterized.product(
        stack_cfg=[
            StackedTransformerLayer.default_config(),
            RepeatedTransformerLayer.default_config(),
        ],
        num_decodes=[5],
        # Each is of shape [batch], representing per-example prefix lengths.
        prefix_length=[jnp.array([1, 1]), jnp.array([1, 3, 6])],
        method=["sample_decode", "beam_search_decode"],
        pad_token_id=[0, -1],
    )
    # pylint: disable-next=too-many-statements
    def test_decode(
        self,
        stack_cfg: BaseStackedTransformerLayer.Config,
        num_decodes: int,
        prefix_length: utils.Tensor,
        method: Literal["sample_decode", "beam_search_decode"],
        pad_token_id: int,
    ):
        """Test beam search and sample decoding from a randomly initialized model."""
        with jax.checking_leaks():
            batch_size, src_len, tgt_len, vocab_size = prefix_length.shape[0], 11, 10, 6
            bos_id = 1
            init_key, prefix_key, source_key, method_key = jax.random.split(
                jax.random.PRNGKey(0), num=4
            )

            if isinstance(stack_cfg, RepeatedTransformerLayer.Config):
                remat_spec = RematSpec(prevent_cse=False)
            else:
                remat_spec = None

            cfg = _model_config(
                vocab_size=vocab_size, source_len=src_len, target_len=tgt_len, remat_spec=remat_spec
            )
            cfg.encoder.pad_token_id = pad_token_id
            cfg.decoder.pad_token_id = pad_token_id
            model = cfg.set(name="test").instantiate(parent=None)
            params = model.initialize_parameters_recursively(init_key)

            prefix = jax.random.randint(
                prefix_key,
                shape=[batch_size, tgt_len],
                # Prefix can consist of any tokens, including pad and eos.
                minval=0,
                maxval=vocab_size,
            )
            # Explicitly fill positions >= prefix_length with pad_token_id.
            # Note that each batch example may have a different prefix length.
            # [batch_size, tgt_len].
            prefix_mask = utils.sequence_mask(lengths=prefix_length, max_len=tgt_len)
            prefix = prefix * prefix_mask + pad_token_id * (1 - prefix_mask)
            # Set last token to a non-pad token, to fix the prefix length.
            oh_indices = jax.nn.one_hot(prefix_length - 1, tgt_len, dtype=prefix.dtype)
            prefix = prefix * (1 - oh_indices) + bos_id * oh_indices

            source_ids = jax.random.randint(
                source_key, minval=1, maxval=vocab_size, shape=(batch_size, src_len)
            )
            source_mask = dummy_padding_mask(batch_size=batch_size, max_seq_len=src_len)
            source_ids = source_ids * source_mask + pad_token_id * (1 - source_mask)
            inputs = dict(
                input_batch=dict(prefix=prefix, source=dict(input_ids=source_ids)),
                max_sequence_length=tgt_len,
                num_decodes=num_decodes,
            )
            if method == "sample_decode":
                # Modify logits so that we will always sample the last token ID.
                inputs["logits_modifier"] = (
                    lambda logits: jnp.full_like(logits, decoding.NEG_INF).at[..., -1].set(0)
                )

            outputs, _ = F(
                model,
                inputs=inputs,
                state=params,
                is_training=False,
                prng_key=method_key,
                method=method,
            )
            sequences = outputs.sequences
            self.assertEqual(sequences.shape, (batch_size, num_decodes, tgt_len))
            if method == "beam_search_decode":
                # Per sequence scores for beam search decode.
                self.assertEqual(outputs.scores.shape, (batch_size, num_decodes))
            elif method == "sample_decode":
                # Per token scores for sample-decoding.
                self.assertEqual(outputs.token_scores.shape, (batch_size, num_decodes, tgt_len))
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

    def test_forward_key_conflict(self):
        # pylint: disable=unused-argument
        class DummyEncoderDecoderModel(EncoderDecoderModel):
            def predict(self, *args, **kwargs):
                return dict(x=1)

            def _metrics(self, *args, **kwargs):
                return 0.1, dict(x=2)

        cfg = _model_config(vocab_size=1, source_len=1, target_len=1)
        cfg = DummyEncoderDecoderModel.default_config().set(
            encoder=cfg.encoder, decoder=cfg.decoder
        )
        model = cfg.set(name="test").instantiate(parent=None)
        with (
            mock.patch(f"{utils.__name__}.validate_contains_paths"),
            self.assertRaisesRegex(KeyError, "conflict"),
        ):  # noqa: F821
            F(
                model,
                inputs=dict(
                    input_batch=dict(
                        source=dict(input_ids=jnp.empty([1, 1], dtype=jnp.int32)),
                        target=dict(input_ids=jnp.empty([1, 1], dtype=jnp.int32)),
                        target_labels=jnp.empty([1, 1], dtype=jnp.int32),
                    )
                ),
                state=model.initialize_parameters_recursively(jax.random.PRNGKey(123)),
                is_training=False,
                prng_key=None,
            )


class TestAgainstHF(TestCase):
    """Tests EncoderDecoder layer against HF golden outputs."""

    def setUp(self):
        super().setUp()
        # Config values matching the HF model used to generate goldens.
        vocab_size = 24
        hidden_size = 16
        num_heads = 4
        num_layers = 2
        source_len = 11
        target_len = 8
        type_vocab_size = 2
        layer_norm_eps = 1e-5
        self.encoder_pad_id = 0
        self.decoder_pad_id = 0
        self.source_len = source_len
        self.target_len = target_len

        # Build AXLearn encoder config (equivalent to bert_encoder_config_from_hf).
        test_encoder = Encoder.default_config().set(
            dim=hidden_size,
            vocab_size=vocab_size,
            dropout_rate=0.0,
            emb=bert_embedding_config(
                type_vocab_size=type_vocab_size,
                max_position_embeddings=source_len,
            ),
            transformer=bert_transformer_config(num_layers=num_layers, num_heads=num_heads),
            pad_token_id=0,
        )
        set_layer_norm_eps_recursively(test_encoder, layer_norm_eps)

        # Build AXLearn decoder config (equivalent to _gpt2_decoder_config_from_hf).
        test_decoder = gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=num_layers,
            hidden_dim=hidden_size,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.relu",
            max_position_embeddings=target_len,
            layer_norm_epsilon=layer_norm_eps,
            dropout_rate=0.0,
        )
        test_decoder.pad_token_id = self.decoder_pad_id
        set_decoder_cross_attention_config(test_decoder, num_heads)

        self.test_encoder_decoder = (
            EncoderDecoderModel.default_config()
            .set(name="layer_test", encoder=test_encoder, decoder=test_decoder)
            .instantiate(parent=None)
        )

    @parameterized.product(
        # Parameterize how source padding is represented:
        # 1. none: Test no padding.
        # 2. pad_id: Allow source_ids to contain pad_id.
        # 3. segment_ids: Supply segment_ids where paddings are 0's.
        source_padding_type=["none", "pad_id", "segment_ids"],
        # Parameterize how target padding is represented:
        # 1. none: Test no padding.
        # 2. pad_id: Allow target_ids to contain pad_id.
        # 3. segment_ids: Supply segment_ids where paddings are 0's.
        target_padding_type=["none", "pad_id", "segment_ids"],
    )
    def test_forward(self, *, source_padding_type: str, target_padding_type: str):
        if (source_padding_type == "segment_ids") != (target_padding_type == "segment_ids"):
            # segment_ids on source/target should be provided together.
            return

        # Load golden data.
        test_name = f"test_forward_{source_padding_type}_{target_padding_type}"
        golden = load_golden(_MODULE_NAME, test_name)

        # Load inputs from golden.
        source_ids = jnp.array(golden["inputs"]["source_ids"])
        source_token_type_ids = jnp.array(golden["inputs"]["source_token_type_ids"])
        target_ids = jnp.array(golden["inputs"]["target_ids"])
        target_labels = jnp.array(golden["inputs"]["target_labels"])

        source_segment_ids = source_positions = None
        target_segment_ids = target_positions = None
        source_mask = golden["inputs"].get("source_mask")
        target_mask = golden["inputs"].get("target_mask")
        if source_mask is not None:
            source_mask = jnp.array(source_mask)
        if target_mask is not None:
            target_mask = jnp.array(target_mask)

        # Apply source padding masks.
        if source_padding_type == "pad_id":
            source_ids = jnp.where(source_mask, source_ids, self.encoder_pad_id)
        elif source_padding_type == "segment_ids":
            source_segment_ids = source_mask
            source_positions = jnp.arange(self.source_len)[None, :] * source_mask

        # Apply target padding masks.
        if target_padding_type == "pad_id":
            target_ids = jnp.where(target_mask, target_ids, self.decoder_pad_id)
        elif target_padding_type == "segment_ids":
            target_segment_ids = target_mask
            target_positions = jnp.arange(self.target_len)[None, :] * target_mask

        # Load params from golden.
        state = jax.tree.map(jnp.array, golden["params"])

        # Run the AXLearn model forward pass.
        input_batch = dict(
            source=dict(
                input_ids=source_ids,
                token_type_ids=source_token_type_ids,
                input_segment_ids=source_segment_ids,
                positions=source_positions,
            ),
            target=dict(
                input_ids=target_ids,
                input_segment_ids=target_segment_ids,
                positions=target_positions,
            ),
            target_labels=target_labels,
        )
        (loss, test_aux), _ = F(
            self.test_encoder_decoder,
            inputs=dict(input_batch=input_batch, return_aux=True),
            state=state,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
        )

        # Compute logits from hidden_states via decoder.compute_logits.
        test_logits = F(
            self.test_encoder_decoder.decoder,
            inputs=dict(forward_outputs=test_aux),
            state=state["decoder"],
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            method="compute_logits",
        )[0]

        # Load expected outputs from golden.
        ref_logits = jnp.array(golden["outputs"]["logits"])
        ref_loss = jnp.array(golden["outputs"]["loss"])

        # Compare outputs at non-padding positions.
        if target_mask is not None:
            test_logits *= target_mask[..., None]
            ref_logits *= target_mask[..., None]
        test_label = f"{source_padding_type}:{target_padding_type}"

        # We occasionally observe rounding errors.
        assert_allclose(test_logits, ref_logits, atol=1e-5, err_msg=test_label)
        assert_allclose(loss, ref_loss, err_msg=test_label)


class TestAgainstT5X(TestCase):
    """Tests EncoderDecoder layer against T5X."""

    @parameterized.parameters(False, True)
    def test_against_t5x(self, packing: bool):
        testcase = load_golden(_MODULE_NAME, f"test_against_t5x_{packing}")

        # Setup dummy axlearn model.
        cfg = t5_encoder_decoder_config(
            vocab_size=48,
            dim=16,
            num_attention_heads=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dropout_rate=0,
            z_loss_scale=0,
        )
        test_encoder_decoder = cfg.set(name="test").instantiate(parent=None)

        state = parameters_from_t5x_encoder_decoder(
            testcase["params"],
            test_encoder_decoder,
        )

        def _maybe_as_tensor(x):
            return jnp.asarray(x) if x is not None else None

        test_outputs, _ = F(
            test_encoder_decoder,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(
                input_batch=dict(
                    source=dict(
                        input_ids=jnp.asarray(testcase["source_ids"]),
                        input_segment_ids=_maybe_as_tensor(testcase.get("source_segment_ids")),
                        positions=_maybe_as_tensor(testcase.get("source_positions")),
                    ),
                    target=dict(
                        input_ids=jnp.asarray(testcase["target_ids"]),
                        input_segment_ids=_maybe_as_tensor(testcase.get("target_segment_ids")),
                        positions=_maybe_as_tensor(testcase.get("target_positions")),
                    ),
                    target_labels=jnp.asarray(testcase["target_labels"]),
                ),
            ),
            method="predict",
        )

        # Compute logits from hidden_states via decoder.compute_logits.
        test_logits = F(
            test_encoder_decoder.decoder,
            inputs=dict(forward_outputs=test_outputs),
            state=state["decoder"],
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            method="compute_logits",
        )[0]

        # Compare.
        ref_outputs = utils.as_tensor(testcase["outputs"])
        mask = testcase["padding_mask"][..., None]
        self.assertNestedAllClose(test_logits * mask, ref_outputs * mask)


if __name__ == "__main__":
    absltest.main()

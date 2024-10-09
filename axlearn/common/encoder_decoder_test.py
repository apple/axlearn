# Copyright © 2023 Apple Inc.

"""Tests EncoderDecoder layers."""

import os
from typing import Literal, Optional

import jax
import numpy as np
import torch
from absl.testing import parameterized
from jax import numpy as jnp
from transformers import BertConfig, BertModel, EncoderDecoderConfig
from transformers import EncoderDecoderModel as HFEncoderDecoderModel
from transformers import GPT2Config, GPT2LMHeadModel

from axlearn.common import decoding, utils
from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.bert import bert_embedding_config, bert_transformer_config
from axlearn.common.bert_test import bert_encoder_config_from_hf
from axlearn.common.causal_lm import gpt_decoder_config
from axlearn.common.decoder import Decoder, LmHead
from axlearn.common.encoder import Encoder
from axlearn.common.encoder_decoder import EncoderDecoderModel
from axlearn.common.layers import set_layer_norm_eps_recursively
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor, parameters_from_t5x_encoder_decoder
from axlearn.common.t5 import t5_encoder_decoder_config
from axlearn.common.test_utils import TestCase, assert_allclose, dummy_padding_mask
from axlearn.common.torch_utils import parameters_from_torch_layer

testdata_dir = os.path.join(os.path.dirname(__file__), "../experiments/testdata")


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
            return F(
                layer,
                inputs=dict(
                    input_batch=dict(
                        source=dict(input_ids=source_ids),
                        target=dict(input_ids=target_ids),
                        target_labels=target_labels,
                    ),
                    return_aux=True,
                ),
                state=state,
                is_training=False,
                prng_key=jax.random.PRNGKey(2),
            )[0][1]["logits"]

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
            prefix_mask = jnp.arange(tgt_len) < prefix_length[:, None]
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
                    lambda logits: jnp.full_like(logits, decoding.NEG_INF).at[:, -1].set(0)
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
            def _validate_input_batch(self, *args, **kwargs):
                pass

            def predict(self, *args, **kwargs):
                return dict(x=1)

            def _metrics(self, *args, **kwargs):
                return 0.1, dict(x=2)

        cfg = _model_config(vocab_size=1, source_len=1, target_len=1)
        cfg = DummyEncoderDecoderModel.default_config().set(
            encoder=cfg.encoder, decoder=cfg.decoder
        )
        model = cfg.set(name="test").instantiate(parent=None)
        with self.assertRaisesRegex(KeyError, "conflict"):
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


def _gpt2_decoder_config_from_hf(
    hf_cfg: GPT2Config,
    vocab_size: Optional[int] = None,
    layer_norm_epsilon: Optional[float] = None,
    dropout_rate: Optional[float] = None,
) -> Decoder.Config:
    cfg = gpt_decoder_config(
        stack_cfg=StackedTransformerLayer.default_config(),
        num_layers=hf_cfg.n_layer,
        hidden_dim=hf_cfg.n_embd,
        num_heads=hf_cfg.n_head,
        vocab_size=vocab_size,
        activation_function=f"nn.{hf_cfg.activation_function}",
        max_position_embeddings=hf_cfg.n_positions,
        layer_norm_epsilon=layer_norm_epsilon,
        dropout_rate=dropout_rate,
    )
    cfg.pad_token_id = hf_cfg.pad_token_id
    return cfg


class TestAgainstHF(TestCase):
    """Tests EncoderDecoder layer against HF."""

    def setUp(self):
        super().setUp()
        self.hf_encoder_cfg = BertConfig(
            vocab_size=24,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=11,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout=0.0,
            layer_norm_eps=1e-5,
        )
        self.hf_decoder_cfg = GPT2Config(
            n_embd=self.hf_encoder_cfg.hidden_size,
            n_head=self.hf_encoder_cfg.num_attention_heads,
            n_layer=self.hf_encoder_cfg.num_hidden_layers,
            n_positions=8,  # seq_len.
            vocab_size=self.hf_encoder_cfg.vocab_size,
            activation_function="relu",
            bos_token_id=1,
            eos_token_id=2,
            add_cross_attention=True,
            layer_norm_epsilon=self.hf_encoder_cfg.layer_norm_eps,
            is_decoder=True,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            pad_token_id=self.hf_encoder_cfg.pad_token_id,
        )
        self.hf_encoder_decoder_cfg = EncoderDecoderConfig.from_encoder_decoder_configs(
            self.hf_encoder_cfg,
            self.hf_decoder_cfg,
        )

        # Setup dummy axlearn model.
        test_encoder = bert_encoder_config_from_hf(
            self.hf_encoder_cfg,
            vocab_size=self.hf_encoder_cfg.vocab_size,
            layer_norm_epsilon=self.hf_encoder_cfg.layer_norm_eps,
            dropout_rate=self.hf_encoder_cfg.hidden_dropout_prob,
        )
        test_decoder = _gpt2_decoder_config_from_hf(
            self.hf_decoder_cfg,
            vocab_size=self.hf_decoder_cfg.vocab_size,
            layer_norm_epsilon=self.hf_decoder_cfg.layer_norm_epsilon,
            dropout_rate=self.hf_decoder_cfg.embd_pdrop,
        )
        set_decoder_cross_attention_config(test_decoder, self.hf_decoder_cfg.n_head)
        self.test_encoder_decoder = (
            EncoderDecoderModel.default_config()
            .set(name="layer_test", encoder=test_encoder, decoder=test_decoder)
            .instantiate(parent=None)
        )

        # Setup dummy HF model.
        hf_encoder = BertModel(self.hf_encoder_cfg, add_pooling_layer=False)
        hf_decoder = GPT2LMHeadModel(self.hf_decoder_cfg)
        hf_encoder_decoder = HFEncoderDecoderModel(
            encoder=hf_encoder, decoder=hf_decoder, config=self.hf_encoder_decoder_cfg
        )
        hf_encoder_decoder.config.pad_token_id = self.hf_encoder_cfg.pad_token_id
        hf_encoder_decoder.config.decoder_start_token_id = self.hf_decoder_cfg.bos_token_id

        self.hf_encoder_decoder = hf_encoder_decoder.eval()

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
        batch_size = 3
        vocab_size = self.hf_encoder_cfg.vocab_size
        source_len = self.hf_encoder_cfg.max_position_embeddings
        target_len = self.hf_decoder_cfg.n_positions
        type_vocab_size = self.hf_encoder_cfg.type_vocab_size
        encoder_pad_id = self.hf_encoder_cfg.pad_token_id
        decoder_pad_id = self.hf_decoder_cfg.pad_token_id

        # Initially generate inputs without padding.
        source_ids = jax.random.randint(
            jax.random.PRNGKey(101),
            (batch_size, source_len),
            minval=encoder_pad_id + 1,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        source_token_type_ids = jax.random.randint(
            jax.random.PRNGKey(102),
            (batch_size, source_len),
            minval=0,
            maxval=type_vocab_size,
            dtype=jnp.int32,
        )
        target_ids = jax.random.randint(
            jax.random.PRNGKey(103),
            (batch_size, target_len),
            minval=decoder_pad_id + 1,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(104),
            (batch_size, target_len),
            minval=decoder_pad_id + 1,
            maxval=vocab_size,
            dtype=jnp.int32,
        )
        hf_source_ids = source_ids
        hf_target_ids = target_ids
        source_segment_ids = source_positions = None
        target_segment_ids = target_positions = None
        source_mask = target_mask = None

        # Generate source paddings.
        if source_padding_type != "none":
            source_mask = dummy_padding_mask(batch_size=batch_size, max_seq_len=source_len)
            hf_source_ids = jnp.where(source_mask, source_ids, encoder_pad_id)

        # Generate target paddings.
        if target_padding_type != "none":
            target_mask = dummy_padding_mask(batch_size=batch_size, max_seq_len=target_len)
            target_labels = jnp.where(target_mask, target_labels, -100)  # HF expects -100.
            hf_target_ids = jnp.where(target_mask, target_ids, decoder_pad_id)

        # Apply source padding masks.
        if source_padding_type == "pad_id":
            source_ids = jnp.where(source_mask, source_ids, encoder_pad_id)
        elif source_padding_type == "segment_ids":
            source_segment_ids = source_mask
            source_positions = jnp.arange(source_len)[None, :] * source_mask
            # Make sure we aren't relying on pad_id.
            self.assertTrue(jnp.all(source_ids > encoder_pad_id))

        # Apply target padding masks.
        if target_padding_type == "pad_id":
            target_ids = jnp.where(target_mask, target_ids, decoder_pad_id)
        elif target_padding_type == "segment_ids":
            target_segment_ids = target_mask
            target_positions = jnp.arange(target_len)[None, :] * target_mask
            # Make sure we aren't relying on pad_id.
            self.assertTrue(jnp.all(target_ids > decoder_pad_id))

        # Compute outputs.
        (loss, test_aux), ref_outputs = self._compute_layer_outputs(
            test_layer=self.test_encoder_decoder,
            ref_layer=self.hf_encoder_decoder,
            test_inputs=dict(
                input_batch=dict(
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
                ),
                return_aux=True,
            ),
            ref_inputs=dict(
                input_ids=as_torch_tensor(hf_source_ids),
                attention_mask=(None if source_mask is None else as_torch_tensor(source_mask)),
                token_type_ids=as_torch_tensor(source_token_type_ids),
                decoder_input_ids=as_torch_tensor(hf_target_ids),
                decoder_attention_mask=(
                    None if target_mask is None else as_torch_tensor(target_mask)
                ),
                labels=as_torch_tensor(target_labels).to(torch.long),
                output_hidden_states=True,
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
            require_same_tree_structure=False,
        )

        # Compare outputs at non-padding positions.
        test_logits = test_aux["logits"]
        ref_logits = utils.as_tensor(ref_outputs.logits)
        if target_mask is not None:
            test_logits *= target_mask[..., None]
            ref_logits *= target_mask[..., None]
        test_name = f"{source_padding_type}:{target_padding_type}"

        # We occasionally observe rounding errors.
        assert_allclose(test_logits, ref_logits, atol=5e-6, err_msg=test_name)
        assert_allclose(loss, utils.as_tensor(ref_outputs.loss), err_msg=test_name)


class TestAgainstT5X(TestCase):
    """Tests EncoderDecoder layer against T5X."""

    @parameterized.parameters(False, True)
    def test_against_t5x(self, packing: bool):
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, f"test_against_t5x_{packing}.npy"),
            allow_pickle=True,
        ).item()

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

        test_outputs, _ = F(
            test_encoder_decoder,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=parameters_from_t5x_encoder_decoder(
                testcase["params"],
                test_encoder_decoder,
            ),
            inputs=dict(
                input_batch=dict(
                    source=dict(
                        input_ids=testcase["source_ids"],
                        input_segment_ids=testcase["source_segment_ids"],
                        positions=testcase["source_positions"],
                    ),
                    target=dict(
                        input_ids=testcase["target_ids"],
                        input_segment_ids=testcase["target_segment_ids"],
                        positions=testcase["target_positions"],
                    ),
                    target_labels=testcase["target_labels"],
                ),
            ),
            method="predict",
        )

        # Compare.
        test_outputs = test_outputs["logits"]
        ref_outputs = utils.as_tensor(testcase["outputs"])
        mask = testcase["padding_mask"][..., None]
        self.assertNestedAllClose(test_outputs * mask, ref_outputs * mask)

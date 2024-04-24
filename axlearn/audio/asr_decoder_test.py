# Copyright © 2023 Apple Inc.

"""Tests ASR decoder layers."""
# pylint: disable=no-self-use,too-many-lines

import functools

import jax.random
import numpy as np
import optax
import torch
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.audio.asr_decoder import (
    CTCDecoderModel,
    CTCPrefixMerger,
    DecodeOutputs,
    _is_valid_ctc_seq,
    _map_label_sequences,
)
from axlearn.common.config import config_for_function
from axlearn.common.decoder import _scores_from_logits
from axlearn.common.decoding import NEG_INF
from axlearn.common.logit_modifiers import top_k_logits
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import Nested, Tensor, shapes


class UtilsTest(TestCase):
    """Tests util functions."""

    @parameterized.parameters(
        dict(
            inputs=jnp.asarray(
                [
                    [1, 1, 0, 2, 2, 2, 3, 0, 0, 4],
                ]
            ),
            expected=dict(
                sequences=jnp.asarray(
                    [
                        [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
                    ]
                ),
                paddings=jnp.asarray(
                    [
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    ]
                ),
                lengths=jnp.asarray([[4]]),
            ),
            blank_id=0,
            pad_id=0,
        ),
        dict(
            inputs=jnp.asarray(
                [
                    [1, 1, 0, 2, 2, 2, 3, 0, 0, 4],
                    [0, 0, 0, 1, 2, 1, 0, 0, 3, 0],
                ]
            ),
            expected=dict(
                sequences=jnp.asarray(
                    [
                        [1, 2, 3, 4, -1, -1, -1, -1, -1, -1],
                        [1, 2, 1, 3, -1, -1, -1, -1, -1, -1],
                    ]
                ),
                paddings=jnp.asarray(
                    [
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    ]
                ),
                lengths=jnp.asarray([[4], [4]]),
            ),
            blank_id=0,
            pad_id=-1,
        ),
    )
    def test_map_label_sequences(
        self, inputs: Tensor, expected: Nested[Tensor], blank_id: int, pad_id: int
    ):
        jit_fn = jax.jit(_map_label_sequences, static_argnames=("blank_id", "pad_id"))
        self.assertNestedEqual(
            expected,
            jit_fn(inputs, blank_id=blank_id, pad_id=pad_id),
        )


class ValidCtcSeqTest(TestCase):
    def get_logits_and_labels(
        self, batch_size: int, input_lengths: int, target_lengths: int, vocab_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        prng_key = jax.random.PRNGKey(1234)
        logits = jax.random.normal(
            prng_key, (batch_size, input_lengths, vocab_size), dtype=jnp.float32
        )
        paddings = jnp.zeros((batch_size, input_lengths), dtype=np.int32)
        target_labels = jax.random.randint(
            prng_key,
            shape=(batch_size, target_lengths),
            minval=1,
            maxval=vocab_size - 1,
            dtype=jnp.int32,
        )
        target_paddings = jnp.zeros(shape=(batch_size, target_lengths), dtype=jnp.int32)
        return logits, paddings, target_labels, target_paddings

    def test_label_longer_than_input(self):
        batch_size = 4
        input_lengths = 10
        target_lengths = 11
        vocab_size = 400
        # Generate logits and labels, which has logits shorter than labels.
        logits, paddings, target_labels, target_paddings = self.get_logits_and_labels(
            batch_size, input_lengths, target_lengths, vocab_size
        )
        per_seq_loss = optax.ctc_loss(logits, paddings, target_labels, target_paddings, blank_id=0)
        for x in per_seq_loss:
            # Because these are invalid sequence loss, the optax.ctc_loss will return
            # -logeps for these sequences (but theoretically, this is not correct).
            self.assertGreater(x, 1e5)
        per_seq_validality = _is_valid_ctc_seq(
            paddings=paddings, target_labels=target_labels, target_paddings=target_paddings
        ).astype(jnp.float32)
        self.assertNestedAllClose(
            per_seq_validality, jnp.array([0.0] * batch_size, dtype=per_seq_validality.dtype)
        )

    def test_label_shorter_than_input(self):
        batch_size = 4
        input_lengths = 15
        target_lengths = 10
        vocab_size = 400
        logits, paddings, _, target_paddings = self.get_logits_and_labels(
            batch_size, input_lengths, target_lengths, vocab_size
        )
        # This is to make sure there is no duplicate in the labels.
        labels = jnp.tile(jnp.arange(target_lengths)[jnp.newaxis, :], [batch_size, 1])

        per_seq_loss = optax.ctc_loss(logits, paddings, labels, target_paddings)
        # `per_seq_loss` in this case looks normal, it should be around log(400)*15, so
        # significantly smaller than 1e5.
        for x in per_seq_loss:
            self.assertLess(x, 1e5)
        per_seq_validality = _is_valid_ctc_seq(
            paddings=paddings, target_labels=labels, target_paddings=target_paddings
        ).astype(jnp.float32)
        self.assertNestedAllClose(
            per_seq_validality, jnp.array([1.0] * batch_size, dtype=per_seq_validality.dtype)
        )

    def test_label_with_duplicates(self):
        batch_size = 5
        input_lengths = 12
        target_lengths = 10
        vocab_size = 400
        logits, paddings, _, target_paddings = self.get_logits_and_labels(
            batch_size, input_lengths, target_lengths, vocab_size
        )
        # There are 12 timesteps, and 10 labels. If the consecutive duplicates in
        # one sequence is larger than 2, then the pair become non-valid
        target_labels = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # no duplicates
                [0, 0, 1, 1, 2, 3, 4, 5, 6, 7],  # 2 consecutive duplicates
                [0, 0, 0, 1, 1, 2, 3, 4, 5, 6],  # 3 duplicates -> invalid seq
                [0, 0, 1, 1, 2, 3, 4, 5, 6, 6],  # 2 duplicates, since the last 6 is a padding
                [0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
                # "0,1,2,3" is duplicated 2 times, but they are not consecutive
            ],
            dtype=np.int32,
        )
        target_paddings = target_paddings.at[3, 9].set(1)
        per_seq_loss = optax.ctc_loss(logits, paddings, target_labels, target_paddings)
        # per_seq_loss[0:1] and per_seq_loss[3] should near log(400) * 15, while
        # per_seq_loss[2] should be around logepsilon
        self.assertLess(per_seq_loss[0], 1e5)
        self.assertLess(per_seq_loss[1], 1e5)
        self.assertLess(per_seq_loss[3], 1e5)
        self.assertLess(per_seq_loss[4], 1e5)
        self.assertGreater(per_seq_loss[2], 1e5)
        per_seq_validality = _is_valid_ctc_seq(
            paddings=paddings, target_labels=target_labels, target_paddings=target_paddings
        ).astype(jnp.float32)
        self.assertNestedAllClose(
            per_seq_validality, jnp.array([1.0, 1.0, 0.0, 1.0, 1.0], dtype=per_seq_validality.dtype)
        )


class CTCPrefixMergerTest(TestCase):
    """Tests CTCPrefixMerger."""

    def test_ctc_prefix_merger(self):
        blank_id = 0
        merger = CTCPrefixMerger(blank_id=blank_id)
        batch_size, num_decodes, max_decode_len = 2, 2, 4
        # Initialize empty state.
        state = merger.init_state(
            tokens=jnp.full([batch_size, num_decodes, max_decode_len], blank_id, dtype=jnp.int32),
        )
        tokens = jnp.asarray(
            [
                [
                    # Sequences with repeated tokens. Both will be normalized to [1, 3] eventually.
                    [1, 3, 3],
                    [1, 1, 3],
                ],
                [
                    # Sequences with blank tokens, but with different results.
                    [2, 0, 2],  # normalizes to [2, 2].
                    [0, 2, 2],  # normalizes to [2].
                ],
            ]
        )
        print(f"state0={state}")
        np.testing.assert_array_equal(state["lengths"], [[0, 0], [0, 0]])
        np.testing.assert_array_equal(
            merger.compute(state),
            [
                # The prefixes are identical.
                [[1, 1], [0, 0]],
                [[1, 1], [0, 0]],
            ],
        )

        state = merger.update(tokens=tokens[:, :, 0], state=state)
        print(f"state1={state}")
        np.testing.assert_array_equal(state["lengths"], [[1, 1], [1, 0]])
        np.testing.assert_array_equal(
            state["sequences"][:, :, :2],
            [
                [
                    [1, -1],
                    [1, -1],
                ],
                [
                    [2, -1],
                    [-1, -1],  # This prefix is [] because 0 is a blank id.
                ],
            ],
        )
        np.testing.assert_array_equal(
            merger.compute(state),
            [
                # The prefixes are identical: [1].
                [[1, 1], [0, 0]],
                # The prefixes are different: [2] vs. [].
                [[1, 0], [0, 1]],
            ],
        )

        state = merger.update(tokens=tokens[:, :, 1], state=state)
        print(f"state2={state}")
        np.testing.assert_array_equal(state["lengths"], [[2, 1], [1, 1]])
        np.testing.assert_array_equal(
            state["sequences"][:, :, :2],
            [[[1, 3], [1, -1]], [[2, -1], [2, -1]]],
        )
        np.testing.assert_array_equal(
            merger.compute(state),
            [
                # The prefixes are different [1, 3] vs. [1].
                [[1, 0], [0, 1]],
                # The prefixes are identical: [2] vs. [2].
                [[1, 1], [0, 0]],
            ],
        )

        state = merger.update(tokens=tokens[:, :, 2], state=state)
        print(f"state3={state}")
        np.testing.assert_array_equal(state["lengths"], [[2, 2], [2, 1]])
        np.testing.assert_array_equal(
            state["sequences"][:, :, :2],
            [[[1, 3], [1, 3]], [[2, 2], [2, -1]]],
        )
        np.testing.assert_array_equal(
            merger.compute(state),
            [
                # The prefixes are identical [1, 3] vs. [1, 3].
                [[1, 1], [0, 0]],
                # The prefixes are different: [2, 2] vs. [2].
                [[1, 0], [0, 1]],
            ],
        )

        # Test prefilling: initialize with all of the tokens.
        prefill_state = jax.jit(merger.init_state)(
            # Pad tokens to max length.
            tokens=jnp.concatenate(
                [
                    tokens,
                    jnp.full([batch_size, num_decodes, 1], -1, dtype=jnp.int32),
                ],
                axis=-1,
            )
        )
        jax.tree_util.tree_map(
            np.testing.assert_array_equal,
            state,
            prefill_state,
        )


class CTCDecoderModelTest(TestCase):
    """Tests CTCDecoderModel."""

    @parameterized.parameters([0, 1])
    def test_predict(self, blank_token_id):
        dim, vocab_size = 6, 8
        cfg = CTCDecoderModel.default_config().set(
            dim=dim,
            vocab_size=vocab_size,
            blank_token_id=blank_token_id,
        )
        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            {"lm_head": dict(weight=(dim, vocab_size), bias=(vocab_size,))},
            shapes(layer_params),
        )

        batch_size, max_seq_len = 2, 10
        seq_len = jnp.array([7, 5])
        # [batch_size, max_seq_len, dim] with the same data across sequences.
        inputs = jnp.tile(
            jax.random.normal(jax.random.PRNGKey(123), [1, max_seq_len, dim]), [batch_size, 1, 1]
        )
        # [batch_size, max_seq_len].
        paddings = (jnp.arange(max_seq_len) >= seq_len[:, None]).astype(inputs.dtype)

        # Generate different padding data.
        padding_data = jax.random.normal(jax.random.PRNGKey(130), [batch_size, max_seq_len, dim])
        # Generate input sequences with the same data at non-pad positions.
        inputs = jnp.where(paddings[..., None], padding_data, inputs)

        @jax.jit
        def jit_predict(input_batch):
            outputs, _ = F(
                layer,
                inputs=dict(input_batch=input_batch),
                is_training=True,
                prng_key=prng_key,
                state=layer_params,
                method="predict",
            )
            return outputs

        outputs = jit_predict(dict(inputs=inputs, paddings=paddings))
        self.assertSequenceEqual((batch_size, max_seq_len, vocab_size), outputs.shape)
        # Check that the outputs are the same in the non-padded positions.
        assert_allclose(outputs[0, : seq_len[1]], outputs[1, : seq_len[1]])
        outputs_at_padding = paddings[:, :, None] * outputs
        # Check that all padding position have 0 outputs.
        self.assertTrue(jnp.all(jnp.logical_not(outputs_at_padding)))

    @parameterized.parameters([0, 1])
    def test_forward(self, blank_token_id):
        dim, vocab_size = 16, 20
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            dim=dim,
            vocab_size=vocab_size,
            blank_token_id=blank_token_id,
        )
        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key, target_key = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_seq_len = 8, 10

        # Sample indices 2, 3 are invalid, since target_lengths exceeds input_lengths.
        input_lengths = jnp.array([10, 5, 7, 0, 6, 3, 8, 1], dtype=jnp.int32)
        target_lengths = jnp.array([6, 3, 9, 1, 6, 0, 4, 0], dtype=jnp.int32)
        per_example_weight = jnp.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=jnp.float32)
        self.assertTrue(jnp.all(input_lengths) <= max_seq_len)
        self.assertTrue(jnp.all(target_lengths) <= max_seq_len)
        self.assertEqual(len(input_lengths), len(target_lengths))
        self.assertEqual(len(input_lengths), batch_size)

        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, dim]) * 1000
        target_labels = jax.random.randint(
            target_key, [batch_size, max_seq_len], minval=0, maxval=vocab_size
        )
        # [batch_size, max_seq_len].
        paddings = (jnp.arange(max_seq_len) >= input_lengths[:, None]).astype(target_labels.dtype)
        # Map padding targets out-of-vocab.
        target_labels = jnp.where(
            jnp.arange(max_seq_len) >= target_lengths[:, None], -1, target_labels
        )

        @jax.jit
        def jit_forward(input_batch):
            (loss, aux_outputs), _ = F(
                layer,
                inputs=dict(input_batch=input_batch),
                is_training=True,
                prng_key=prng_key,
                state=layer_params,
            )
            return loss, aux_outputs

        # Compute test loss.
        loss, aux_outputs = jit_forward(
            dict(
                inputs=inputs,
                paddings=paddings,
                target_labels=target_labels,
            )
        )
        # Compute reference loss.
        logits, _ = F(
            layer,
            inputs=dict(input_batch=dict(inputs=inputs, paddings=paddings)),
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
            method="predict",
        )
        # Transpose since torch ctc_loss expects [max_seq_len, batch_size, ...].
        ref_inputs = as_torch_tensor(_scores_from_logits(logits)).transpose(0, 1)
        ref_target_labels = as_torch_tensor(target_labels)
        ref_per_example_loss = torch.nn.functional.ctc_loss(
            log_probs=ref_inputs,
            targets=ref_target_labels,
            input_lengths=as_torch_tensor(input_lengths),
            target_lengths=as_torch_tensor(target_lengths),
            blank=cfg.blank_token_id,
            reduction="none",
            zero_infinity=True,
        )
        ref_per_example_loss = (
            np.nan_to_num(ref_per_example_loss.detach().numpy()) * per_example_weight
        )
        self.assertNestedEqual(per_example_weight, aux_outputs["per_example_weight"])
        assert_allclose(ref_per_example_loss, aux_outputs["per_example_loss"] * per_example_weight)
        assert_allclose(np.sum(ref_per_example_loss) / np.sum(per_example_weight), loss)

    def _check_paddings(self, outputs: DecodeOutputs, *, blank_token_id: int):
        # Padding positions should correspond to pad_id.
        self.assertTrue(jnp.all(outputs.sequences * outputs.paddings == 0))
        # Other positions should not contain pad_id or blanks.
        self.assertTrue(jnp.all((outputs.sequences != 0) | outputs.paddings))
        if blank_token_id != 0:
            self.assertTrue(jnp.all((outputs.sequences != blank_token_id) | outputs.paddings))

    @parameterized.product(
        num_decodes=[1, 3],
        vocab_size=[5, 20],
        blank_token_id=[0, 1],
        logits_modifier=[top_k_logits(1), config_for_function(top_k_logits).set(k=1)],
    )
    def test_greedy_decode(self, num_decodes, vocab_size, blank_token_id, logits_modifier):
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            dim=6,
            vocab_size=vocab_size,
            blank_token_id=blank_token_id,
        )

        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        decode_key, predict_key, init_key, input_key = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_seq_len = 4, 10
        seq_len = jnp.array([10, 7, 5, 8])
        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, cfg.dim]) * 1000
        # [batch_size, max_seq_len].
        paddings = (jnp.arange(max_seq_len) >= seq_len[:, None]).astype(jnp.int32)

        @functools.partial(jax.jit, static_argnames=("method", "modify_logits", "num_decodes"))
        def jit_method(inputs, prng_key, method, modify_logits=False, num_decodes=None):
            if modify_logits and logits_modifier is not None:
                inputs["logits_modifier"] = logits_modifier
            if num_decodes is not None:
                inputs["num_decodes"] = num_decodes
            outputs, _ = F(
                layer,
                inputs=inputs,
                is_training=True,
                prng_key=prng_key,
                state=layer_params,
                method=method,
            )
            return outputs

        sample_decode_outputs: DecodeOutputs = jit_method(
            dict(input_batch=dict(inputs=inputs, paddings=paddings)),
            prng_key=decode_key,
            method="sample_decode",
            modify_logits=True,
            num_decodes=num_decodes,
        )

        greedy_decode_outputs: DecodeOutputs = jit_method(
            dict(input_batch=dict(inputs=inputs, paddings=paddings)),
            prng_key=decode_key,
            method="greedy_decode",
        )

        # Should be equivalent to taking the top logit of each output.
        # [batch_size, max_seq_len, vocab_size].
        logits = jit_method(
            dict(input_batch=dict(inputs=inputs, paddings=paddings)),
            prng_key=predict_key,
            method="predict",
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_probs += paddings[..., None] * NEG_INF
        log_probs = jnp.pad(log_probs, ((0, 0), (0, 1), (0, 0)), constant_values=NEG_INF)
        paddings_extended = jnp.pad(paddings, ((0, 0), (0, 1)), constant_values=1)
        eos_log_probs = (1 - paddings_extended[:, :, None]) * NEG_INF
        ref_log_probs = jnp.concatenate([log_probs, eos_log_probs], axis=-1)

        # Sequences have shape [batch_size, max_seq_len].
        ref_raw_sequences = jnp.argmax(ref_log_probs, axis=-1)
        ref_outputs = _map_label_sequences(ref_raw_sequences, blank_id=cfg.blank_token_id)
        ref_sequences, ref_paddings = ref_outputs["sequences"], ref_outputs["paddings"]

        # Mask out padding/EOS tokens.
        ref_paddings = ref_paddings | (ref_sequences == vocab_size)
        ref_sequences = ref_sequences * (1 - ref_paddings)

        # Compute reference scores [batch_size, max_seq_len + 1, 1].
        ref_scores = jnp.take_along_axis(ref_log_probs, ref_raw_sequences[..., None], axis=-1)

        # Drop dummy token [batch_size, max_seq_len, ...].
        ref_paddings = ref_paddings[..., :-1]
        ref_sequences = ref_sequences[..., :-1]
        ref_scores = ref_scores[..., :-1, :]

        # Aggregate scores [batch_size].
        ref_scores = jnp.squeeze(ref_scores, axis=-1) * (1 - paddings)
        ref_scores = jnp.sum(ref_scores, axis=-1)

        # Sample decode top decode should match.
        self.assertNestedEqual(ref_sequences, sample_decode_outputs.sequences[:, 0, :])
        self.assertNestedEqual(ref_paddings, sample_decode_outputs.paddings[:, 0, :])
        self.assertNestedEqual(ref_scores, sample_decode_outputs.scores[:, 0])

        self._check_paddings(sample_decode_outputs, blank_token_id=cfg.blank_token_id)

        # Greedy decode output should match.
        self.assertNestedEqual(ref_sequences, greedy_decode_outputs.sequences[:, 0, :])
        self.assertNestedEqual(ref_paddings, greedy_decode_outputs.paddings[:, 0, :])
        self.assertNestedEqual(ref_scores, greedy_decode_outputs.scores[:, 0])

        self._check_paddings(greedy_decode_outputs, blank_token_id=cfg.blank_token_id)

    @parameterized.product(
        num_decodes=[1, 3],
        vocab_size=[5, 20],
        blank_token_id=[0, 1],
    )
    def test_beam_search_decode(self, num_decodes, vocab_size, blank_token_id):
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            dim=6,
            vocab_size=vocab_size,
            blank_token_id=blank_token_id,
        )
        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        decode_key1, decode_key2, init_key, input_key = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_seq_len = 4, 10
        seq_len = jnp.array([10, 7, 5, 8])
        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, cfg.dim]) * 1000
        # [batch_size, max_seq_len].
        paddings = (jnp.arange(max_seq_len) >= seq_len[:, None]).astype(jnp.int32)

        @functools.partial(jax.jit, static_argnames=("method", "logits_modifier", "num_decodes"))
        def jit_method(inputs, prng_key, method, logits_modifier=None, num_decodes=None):
            if logits_modifier is not None:
                inputs["logits_modifier"] = logits_modifier
            if num_decodes is not None:
                inputs["num_decodes"] = num_decodes
            outputs, _ = F(
                layer,
                inputs=inputs,
                is_training=True,
                prng_key=prng_key,
                state=layer_params,
                method=method,
            )
            return outputs

        input_batch = dict(input_batch=dict(inputs=inputs, paddings=paddings))

        # Beam search decode.
        beam_search_outputs: DecodeOutputs = jit_method(
            input_batch,
            prng_key=decode_key1,
            method="beam_search_decode",
            num_decodes=num_decodes,
        )
        # Check that beams are sorted descending by score.
        self.assertTrue(jnp.all(jnp.diff(beam_search_outputs.scores, axis=-1) <= 0))
        self._check_paddings(beam_search_outputs, blank_token_id=cfg.blank_token_id)

        # Greedy decode.
        greedy_outputs: DecodeOutputs = jit_method(
            input_batch,
            prng_key=decode_key2,
            method="sample_decode",
            logits_modifier=top_k_logits(1),
            num_decodes=num_decodes,
        )
        # Because tokens are treated as conditionally independent, top hypothesis of
        # beam_search_outputs should match that of greedy_outputs.
        self.assertNestedEqual(
            greedy_outputs.sequences[:, 0, :],
            beam_search_outputs.sequences[:, 0, :],
        )
        self.assertNestedAllClose(greedy_outputs.scores[:, 0], beam_search_outputs.scores[:, 0])

    def test_prefix_merger(self):
        # Use a small vocab_size to encourage similar prefixes.
        dim, vocab_size, num_decodes = 6, 3, 4
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            dim=dim,
            vocab_size=vocab_size,
        )
        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        decode_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_seq_len = 4, 8
        seq_len = jnp.array([8, 5, 4, 6])
        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, dim]) * 1000
        # [batch_size, max_seq_len].
        paddings = (jnp.arange(max_seq_len) >= seq_len[:, None]).astype(jnp.int32)

        @functools.partial(jax.jit, static_argnames=("method", "prefix_merger", "num_decodes"))
        def jit_method(inputs, prng_key, method, prefix_merger=None, num_decodes=None):
            if prefix_merger is not None:
                inputs["prefix_merger"] = prefix_merger
            if num_decodes is not None:
                inputs["num_decodes"] = num_decodes
            outputs, _ = F(
                layer,
                inputs=inputs,
                is_training=True,
                prng_key=prng_key,
                state=layer_params,
                method=method,
            )
            return outputs

        input_batch = dict(input_batch=dict(inputs=inputs, paddings=paddings))

        # Decode without merging.
        beam_search_outputs: DecodeOutputs = jit_method(
            input_batch,
            prng_key=decode_key,
            method="beam_search_decode",
            num_decodes=num_decodes,
        )
        self.assertNestedEqual(
            jnp.array(
                [
                    [
                        # Without merging prefixes, sequences 0 and 1 are duplicates.
                        [1, 2, 1, 2, 1, 0, 0, 0],
                        [1, 2, 1, 2, 1, 0, 0, 0],
                        [1, 2, 1, 2, 1, 2, 0, 0],
                        [1, 2, 2, 1, 0, 0, 0, 0],
                    ],
                    [
                        # Sequence 0 and 3 are duplicates.
                        [2, 1, 1, 2, 0, 0, 0, 0],
                        [2, 1, 1, 0, 0, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0, 0, 0],
                        [2, 1, 2, 1, 0, 0, 0, 0],
                    ],
                    [
                        # Sequence 0 and 1 are duplicates.
                        [2, 0, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0, 0, 0],
                        [2, 2, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        # Sequence 0 and 3 are duplicates.
                        [2, 1, 2, 1, 0, 0, 0, 0],
                        [1, 2, 1, 0, 0, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0, 0, 0],
                        [2, 1, 2, 1, 0, 0, 0, 0],
                    ],
                ]
            ),
            beam_search_outputs.sequences,
        )
        # Check that the duplicate sequences have valid scores, i.e., we would continue to extend
        # them if decoding continued.
        self.assertNotEqual(beam_search_outputs.scores[0, 0], beam_search_outputs.scores[0, 1])
        self.assertNotEqual(beam_search_outputs.scores[1, 0], beam_search_outputs.scores[1, 3])
        self.assertNotEqual(beam_search_outputs.scores[2, 0], beam_search_outputs.scores[2, 1])
        self.assertNotEqual(beam_search_outputs.scores[3, 0], beam_search_outputs.scores[3, 3])

        # Decode with merging.
        beam_search_outputs: DecodeOutputs = jit_method(
            input_batch,
            prng_key=decode_key,
            method="beam_search_decode",
            num_decodes=num_decodes,
            prefix_merger=CTCPrefixMerger(blank_id=cfg.blank_token_id),
        )
        self.assertNestedEqual(
            jnp.array(
                [
                    [
                        # The duplicate sequences 0 and 1 were merged, allowing for new beams.
                        [1, 2, 1, 2, 1, 0, 0, 0],
                        [1, 2, 1, 2, 1, 2, 0, 0],
                        [1, 2, 2, 1, 0, 0, 0, 0],
                        [1, 2, 2, 1, 2, 0, 0, 0],
                    ],
                    [
                        # Sequences 0 and 3 were merged.
                        [2, 1, 1, 2, 0, 0, 0, 0],
                        [2, 1, 1, 0, 0, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0, 0, 0],
                        [2, 1, 2, 1, 0, 0, 0, 0],
                    ],
                    [
                        # The duplicate sequences 0 and 1 were merged.
                        # However, the new beams are also duplicates: sequence 2 is now merged with
                        # sequence 0, and 3 with 1.
                        # Both 2 and 3 should have scores close to NEG_INF.
                        [2, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0, 0, 0],
                        [2, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 0, 0, 0, 0, 0, 0],
                    ],
                    [
                        # Sequences 0 and 3 from before were merged.
                        # However, the new beam at sequence 3 is also merged with sequence 2.
                        # Sequence 3 should have a score close to NEG_INF.
                        [2, 1, 2, 1, 0, 0, 0, 0],
                        [1, 2, 1, 0, 0, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0, 0, 0],
                        [2, 1, 2, 0, 0, 0, 0, 0],
                    ],
                ]
            ),
            beam_search_outputs.sequences,
        )
        # Check that the duplicate sequences have low scores, i.e., we would not extend them if we
        # continued decoding.
        self.assertLess(beam_search_outputs.scores[2, 2], 0.5 * NEG_INF)
        self.assertLess(beam_search_outputs.scores[2, 3], 0.5 * NEG_INF)
        self.assertLess(beam_search_outputs.scores[3, 3], 0.5 * NEG_INF)

    def test_postprocess(self):
        dim, vocab_size = 3, 8
        cfg = CTCDecoderModel.default_config().set(dim=dim, vocab_size=vocab_size)

        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(123))

        sequences = jnp.array(
            [
                [[3, 0, 5, 0, 0, 6, 6, 2, 0, 0, 0], [5, 4, 4, 5, 5, 0, 0, 4, 3, 0, 0]],
                [[4, 4, 4, 0, 4, 6, 4, 4, 2, 0, 0], [3, 5, 0, 0, 0, 6, 2, 2, 2, 2, 0]],
            ],
        )
        paddings = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        scores = jnp.array(
            [
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            ]
        )

        @jax.jit
        def jit_postprocess(sequences, paddings, scores):
            outputs, _ = F(
                layer,
                inputs=dict(sequences=sequences, paddings=paddings, scores=scores),
                is_training=True,
                prng_key=jax.random.PRNGKey(456),
                state=layer_params,
                method="_postprocess_outputs",
            )
            return outputs

        outputs: DecodeOutputs = jit_postprocess(sequences, paddings, scores)
        self.assertNestedEqual(
            outputs.raw_sequences,
            jnp.array(
                [
                    [[3, 0, 5, 0, 0, 6, 6, 2, 0, 0], [5, 4, 4, 5, 5, 0, 0, 4, 0, 0]],
                    [[4, 4, 4, 0, 4, 6, 4, 4, 2, 0], [3, 5, 0, 0, 0, 6, 2, 2, 2, 0]],
                ]
            ),
        )
        self.assertNestedEqual(
            outputs.sequences,
            jnp.array(
                [
                    [[3, 5, 6, 2, 0, 0, 0, 0, 0, 0], [5, 4, 5, 4, 0, 0, 0, 0, 0, 0]],
                    [[4, 4, 6, 4, 2, 0, 0, 0, 0, 0], [3, 5, 6, 2, 0, 0, 0, 0, 0, 0]],
                ]
            ),
        )
        self.assertNestedEqual(
            outputs.paddings,
            jnp.array(
                [
                    [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
                    [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
                ]
            ),
        )
        self.assertNestedEqual(outputs.scores, jnp.array([[28, 28], [36, 36]]))

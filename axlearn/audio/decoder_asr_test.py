# Copyright © 2023 Apple Inc.

"""Tests ASR decoder layers."""
# pylint: disable=no-self-use,too-many-lines

import functools
from typing import Any, Union
from unittest.mock import patch

import jax.random
import numpy as np
import optax
import torch
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.audio.aligner import ctc_aligner_test
from axlearn.audio.decoder_asr import (
    CommonPrefixMerger,
    CTCDecoderModel,
    DecodeOutputs,
    LASDecoderModel,
    RNNPredictionNetwork,
    TransducerDecoderModel,
    _is_valid_ctc_seq,
    _map_label_sequences,
)
from axlearn.common import attention, causal_lm
from axlearn.common.config import config_for_function
from axlearn.common.decoder import _scores_from_logits
from axlearn.common.decoding import NEG_INF
from axlearn.common.logit_modifiers import top_k_logits
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.rnn import BaseRNNCell, IdentityCell, LSTMCell
from axlearn.common.test_utils import TestCase, assert_allclose, set_threefry_partitionable
from axlearn.common.utils import Nested, NestedTensor, Tensor, safe_not, shapes

_NEG_INF = -1.0e7


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
                ).astype(jnp.bool),
                lengths=jnp.asarray([[4]]),
            ),
            blank_id=0,
            pad_id=0,
            remove_repeats=True,
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
                ).astype(jnp.bool),
                lengths=jnp.asarray([[4], [4]]),
            ),
            blank_id=0,
            pad_id=-1,
            remove_repeats=True,
        ),
        dict(
            inputs=jnp.asarray(
                [
                    [[0, 3, 3, 3, 0, 0, 2, 2, 2, 3], [2, 2, 1, 0, 0, 0, 0, 0, 0, 0]],
                    [[2, 0, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                ]
            ),
            expected=dict(
                sequences=jnp.asarray(
                    [
                        [[3, 3, 3, 2, 2, 2, 3, 0, 0, 0], [2, 2, 1, 0, 0, 0, 0, 0, 0, 0]],
                        [[2, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    ],
                ),
                paddings=jnp.asarray(
                    [
                        [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]],
                        [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                    ]
                ).astype(jnp.bool),
                lengths=jnp.asarray([[[7], [3]], [[3], [0]]]),
            ),
            blank_id=0,
            pad_id=0,
            remove_repeats=False,
        ),
        dict(
            inputs=jnp.asarray(
                [
                    [[0, 3, 3, 3, 0, 0, 2, 2, 2, 3], [2, 2, 1, 0, 0, 0, -1, -1, -1, -1]],
                    [[2, 0, 2, 2, 1, 1, -1, -1, -1, -1], [3, 1, 3, 1, 3, 1, -1, -1, -1, -1]],
                ]
            ),
            expected=dict(
                sequences=jnp.asarray(
                    [
                        [[0, 0, 0, 2, 2, 2, -1, -1, -1, -1], [2, 2, 1, 0, 0, 0, -1, -1, -1, -1]],
                        [[2, 0, 2, 2, 1, 1, -1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1, -1, -1, -1]],
                    ],
                ),
                paddings=jnp.asarray(
                    [
                        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]],
                        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]],
                    ]
                ).astype(jnp.bool),
                lengths=jnp.asarray([[[6], [6]], [[6], [3]]]),
            ),
            blank_id=3,
            pad_id=-1,
            remove_repeats=False,
        ),
    )
    def test_map_label_sequences(
        self,
        inputs: Tensor,
        expected: Nested[Tensor],
        blank_id: int,
        pad_id: int,
        remove_repeats: bool,
    ):
        jit_fn = jax.jit(
            _map_label_sequences, static_argnames=("blank_id", "pad_id", "remove_repeats")
        )
        self.assertNestedEqual(
            expected,
            jit_fn(inputs, blank_id=blank_id, pad_id=pad_id, remove_repeats=remove_repeats),
        )


class ValidCtcSeqTest(TestCase):
    def get_logits_and_labels(
        self, batch_size: int, input_lengths: int, target_lengths: int, vocab_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        prng_key = jax.random.PRNGKey(1234)
        logits = jax.random.normal(
            prng_key, (batch_size, input_lengths, vocab_size), dtype=jnp.float32
        )
        paddings = jnp.zeros((batch_size, input_lengths), dtype=jnp.bool)
        target_labels = jax.random.randint(
            prng_key,
            shape=(batch_size, target_lengths),
            minval=1,
            maxval=vocab_size - 1,
            dtype=jnp.int32,
        )
        target_paddings = jnp.zeros(shape=(batch_size, target_lengths), dtype=jnp.bool)
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


class CommonPrefixMergerTest(TestCase):
    """Tests CommonPrefixMerger."""

    def test_ctc_prefix_merger(self):
        blank_id = 0
        merger = CommonPrefixMerger(blank_id=blank_id, pad_id=-1, remove_repeats=True)
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
        jax.tree.map(
            np.testing.assert_array_equal,
            state,
            prefill_state,
        )


class CTCDecoderModelTest(TestCase):
    """Tests CTCDecoderModel."""

    @parameterized.parameters([0, 1])
    def test_predict(self, blank_id):
        input_dim, vocab_size = 6, 8
        cfg = CTCDecoderModel.default_config().set(
            input_dim=input_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
        )
        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            {"lm_head": dict(weight=(input_dim, vocab_size), bias=(vocab_size,))},
            shapes(layer_params),
        )

        batch_size, max_seq_len = 2, 10
        seq_len = jnp.array([7, 5])
        # [batch_size, max_seq_len, dim] with the same data across sequences.
        inputs = jnp.tile(
            jax.random.normal(jax.random.PRNGKey(123), [1, max_seq_len, input_dim]),
            [batch_size, 1, 1],
        )
        # [batch_size, max_seq_len].
        paddings = jnp.arange(max_seq_len) >= seq_len[:, None]

        # Generate different padding data.
        padding_data = jax.random.normal(
            jax.random.PRNGKey(130), [batch_size, max_seq_len, input_dim]
        )
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
    def test_forward(self, blank_id):
        input_dim, vocab_size = 16, 20
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            input_dim=input_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
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
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, input_dim]) * 1000
        target_labels = jax.random.randint(
            target_key, [batch_size, max_seq_len], minval=0, maxval=vocab_size
        )
        # [batch_size, max_seq_len].
        paddings = jnp.arange(max_seq_len) >= input_lengths[:, None]
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
            blank=cfg.blank_id,
            reduction="none",
            zero_infinity=True,
        )
        ref_per_example_loss = (
            np.nan_to_num(ref_per_example_loss.detach().numpy()) * per_example_weight
        )
        self.assertNestedEqual(per_example_weight, aux_outputs["per_example_weight"])
        assert_allclose(ref_per_example_loss, aux_outputs["per_example_loss"] * per_example_weight)
        assert_allclose(np.sum(ref_per_example_loss) / np.sum(per_example_weight), loss)

    def _check_summary(
        self, summary_collection: dict[str, Any], name: str, value: Union[Tensor, WeightedScalar]
    ):
        self.assertIn(name, summary_collection)
        msg = f"mismatch in {name}: {summary_collection[name]} vs {value}"
        self.assertEqual(summary_collection[name], value, msg)

    @set_threefry_partitionable(False)  # TODO(yongqiang): update for threefry_partitionable True
    def test_forward_summary(self):
        input_dim, vocab_size = 16, 20
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            input_dim=input_dim,
            vocab_size=vocab_size,
            blank_id=0,
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
        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, input_dim]) * 1000
        target_labels = jax.random.randint(
            target_key, [batch_size, max_seq_len], minval=0, maxval=vocab_size
        )
        # [batch_size, max_seq_len].
        paddings = jnp.arange(max_seq_len) >= input_lengths[:, None]
        # Map padding targets out-of-vocab.
        target_labels = jnp.where(
            jnp.arange(max_seq_len) >= target_lengths[:, None], -1, target_labels
        )
        target_paddings = target_labels == -1
        input_batch = dict(inputs=inputs, paddings=paddings, target_labels=target_labels)
        _, output_collections = F(
            layer,
            inputs=dict(input_batch=input_batch),
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )
        summaries = output_collections.summaries
        # 6 out of 8 examples are valid, therefore the average example weight is 0.75
        self._check_summary(summaries, "loss/example_weight", WeightedScalar(0.75, 8))
        self._check_summary(summaries, "loss/ctc_loss", WeightedScalar(6972.135, 6))
        self._check_summary(summaries, "loss/invalid_seq_percent", 0.25)
        total_ctc_loss = summaries["loss/ctc_loss"].weight * summaries["loss/ctc_loss"].mean
        num_valid_frames = jnp.sum(safe_not(paddings) * per_example_weight[:, None])
        num_valid_labels = jnp.sum(safe_not(target_paddings) * per_example_weight[:, None])
        num_valid_examples = jnp.sum(per_example_weight)
        self._check_summary(
            summaries,
            "loss/per_frame_ctc_loss",
            WeightedScalar(total_ctc_loss / num_valid_frames, num_valid_frames),
        )
        self._check_summary(
            summaries,
            "loss/per_label_ctc_loss",
            WeightedScalar(total_ctc_loss / num_valid_labels, num_valid_labels),
        )

        self._check_summary(
            summaries,
            "input_stats/average_target_length",
            WeightedScalar(num_valid_labels / num_valid_examples, num_valid_examples),
        )
        self._check_summary(
            summaries,
            "input_stats/average_source_length",
            WeightedScalar(num_valid_frames / num_valid_examples, num_valid_examples),
        )
        self._check_summary(
            summaries,
            "input_stats/frame_packing_efficiency",
            WeightedScalar(num_valid_frames / paddings.size, paddings.size),
        )

    def _check_paddings(self, outputs: DecodeOutputs, *, blank_id: int):
        # Padding positions should correspond to pad_id.
        self.assertTrue(jnp.all(outputs.sequences * outputs.paddings == 0))
        # Other positions should not contain pad_id or blanks.
        self.assertTrue(jnp.all((outputs.sequences != 0) | outputs.paddings))
        if blank_id != 0:
            self.assertTrue(jnp.all((outputs.sequences != blank_id) | outputs.paddings))

    @parameterized.product(
        num_decodes=[1, 3],
        vocab_size=[5, 20],
        blank_id=[0, 1],
        logits_modifier=[top_k_logits(1), config_for_function(top_k_logits).set(k=1)],
    )
    def test_greedy_decode(self, num_decodes, vocab_size, blank_id, logits_modifier):
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            input_dim=6,
            vocab_size=vocab_size,
            blank_id=blank_id,
        )

        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        decode_key, predict_key, init_key, input_key = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_seq_len = 4, 10
        seq_len = jnp.array([10, 7, 5, 8])
        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, cfg.input_dim]) * 1000
        # [batch_size, max_seq_len].
        paddings = jnp.arange(max_seq_len) >= seq_len[:, None]

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
        eos_log_probs = safe_not(paddings_extended)[:, :, None] * NEG_INF
        ref_log_probs = jnp.concatenate([log_probs, eos_log_probs], axis=-1)

        # Sequences have shape [batch_size, max_seq_len].
        ref_raw_sequences = jnp.argmax(ref_log_probs, axis=-1)
        ref_outputs = _map_label_sequences(
            ref_raw_sequences, remove_repeats=True, blank_id=cfg.blank_id
        )
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
        ref_scores = jnp.squeeze(ref_scores, axis=-1) * safe_not(paddings)
        ref_scores = jnp.sum(ref_scores, axis=-1)

        # Sample decode top decode should match.
        self.assertNestedEqual(ref_sequences, sample_decode_outputs.sequences[:, 0, :])
        self.assertNestedEqual(ref_paddings, sample_decode_outputs.paddings[:, 0, :])
        self.assertNestedEqual(ref_scores, sample_decode_outputs.scores[:, 0])

        self._check_paddings(sample_decode_outputs, blank_id=cfg.blank_id)

        # Greedy decode output should match.
        self.assertNestedEqual(ref_sequences, greedy_decode_outputs.sequences[:, 0, :])
        self.assertNestedEqual(ref_paddings, greedy_decode_outputs.paddings[:, 0, :])
        self.assertNestedEqual(ref_scores, greedy_decode_outputs.scores[:, 0])

        self._check_paddings(greedy_decode_outputs, blank_id=cfg.blank_id)

    @parameterized.product(
        num_decodes=[1, 3],
        vocab_size=[5, 20],
        blank_id=[0, 1],
    )
    def test_beam_search_decode(self, num_decodes, vocab_size, blank_id):
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            input_dim=6,
            vocab_size=vocab_size,
            blank_id=blank_id,
        )
        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        decode_key1, decode_key2, init_key, input_key = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_seq_len = 4, 10
        seq_len = jnp.array([10, 7, 5, 8])
        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, cfg.input_dim]) * 1000
        # [batch_size, max_seq_len].
        paddings = jnp.arange(max_seq_len) >= seq_len[:, None]

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
        self._check_paddings(beam_search_outputs, blank_id=cfg.blank_id)

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

    @set_threefry_partitionable(False)  # TODO(markblee): update for threefry_partitionable True
    def test_prefix_merger(self):
        # Use a small vocab_size to encourage similar prefixes.
        input_dim, vocab_size, num_decodes = 6, 3, 4
        cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(
            input_dim=input_dim,
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
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, input_dim]) * 1000
        # [batch_size, max_seq_len].
        paddings = jnp.arange(max_seq_len) >= seq_len[:, None]

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
            prefix_merger=CommonPrefixMerger(blank_id=cfg.blank_id),
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
        input_dim, vocab_size = 3, 8
        cfg = CTCDecoderModel.default_config().set(input_dim=input_dim, vocab_size=vocab_size)

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
        ).astype(jnp.bool)
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
            ).astype(jnp.bool),
        )
        self.assertNestedEqual(outputs.scores, jnp.array([[28, 28], [36, 36]]))

    def test_align(self):
        (
            log_probs,
            log_prob_paddings,
            labels,
            label_paddings,
        ), expected_align = ctc_aligner_test.generate_batched_test_data(
            batch_size=2, blank_id=0, max_num_frames=32, max_num_labels=5, vocab_size=64
        )
        labels = jnp.where(label_paddings, -1, labels)
        input_batch = {"inputs": log_probs, "paddings": log_prob_paddings, "target_labels": labels}

        input_dim, vocab_size = 6, 8
        cfg = CTCDecoderModel.default_config().set(
            input_dim=input_dim,
            vocab_size=vocab_size,
            blank_id=0,
        )
        # Initialize layer parameters.
        layer: CTCDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        with (
            patch.object(layer, "predict", return_value=log_probs) as mocked_pred,
            patch("jax.nn.log_softmax", return_value=log_probs) as mocked_softmax,
        ):
            outputs, _ = F(
                layer,
                inputs=dict(input_batch=input_batch),
                is_training=False,
                prng_key=prng_key,
                state=layer_params,
                method="align",
            )
            mocked_pred.assert_called_once()
            mocked_softmax.assert_called_once()

        self.assertNestedAllClose(outputs, expected_align.asdict())


class RNNPredictionNetworkTest(TestCase):
    def test_forward(self):
        batch_size, seq_len, vocab_size, output_dim = 2, 8, 5, 3
        # Tests that out-of-range token ids (-1, -2, 6, 10) work.
        inputs = jnp.array([[3, 3, 2, 4, 3, -1, -1, -2], [2, 1, 3, 3, 3, 10, 6, -1]])
        layer: RNNPredictionNetwork = (
            RNNPredictionNetwork.default_config()
            .set(
                name="test",
                vocab_size=vocab_size,
                emb_dim=2,
                output_dim=output_dim,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
        forward_outputs, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),  # not used.
            state=layer_params,
            inputs=dict(inputs=inputs),
        )
        # Tests that `time_major_inputs` axis order is correctly handled.
        self.assertSequenceEqual(forward_outputs.shape, (batch_size, seq_len, output_dim))


class SimpleRecurrentCell(BaseRNNCell):
    """A simple recurrent cell."""

    def __init__(self, cfg: BaseRNNCell.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.output_dim and cfg.output_dim != cfg.input_dim:
            raise ValueError(
                "SimpleRecurrentCell requires input_dim = output_dim, but got "
                f"input_dim = {cfg.input_dim}, output_dim = {cfg.output_dim}."
            )

    def init_states(self, *, batch_size: int) -> NestedTensor:
        """Returns the initial step states, to be used by `extend_step`."""
        cfg = self.config
        return {
            "step": jnp.zeros((batch_size, 1)),
            "memory": jnp.zeros((batch_size, cfg.input_dim)),
        }

    def extend_step(
        self, *, data: Tensor, cached_states: NestedTensor
    ) -> tuple[NestedTensor, Tensor]:
        # [batch*beam, emb_dim].
        memory_init = data
        # Markov chain transition probability.
        transition = jnp.array([[0.5, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        memory_prob = jnp.exp(cached_states["memory"]) @ transition
        memory_update = jnp.where(memory_prob > 0, jnp.log(memory_prob), _NEG_INF)
        memory_new = jnp.where(cached_states["step"] > 0, memory_update, memory_init)
        new_states = dict(step=cached_states["step"] + 1, memory=memory_new)
        return new_states, memory_new

    def _batch_size(self, inputs: NestedTensor) -> int:
        assert isinstance(inputs, Tensor)
        assert inputs.ndim == 3, inputs.shape
        return inputs.shape[1]

    def _seq_len(self, inputs: NestedTensor) -> int:
        assert isinstance(inputs, Tensor)
        assert inputs.ndim == 3, inputs.shape
        return inputs.shape[0]


class TransducerDecoderModelTest(TestCase):
    def _set_up_transducer(
        self,
        vocab_size,
        lm_type="lstm_lm",
        blank_logit_bias=None,
    ):
        """Helper function to set up transducer."""
        am_dim = emb_dim = lm_dim = joint_dim = 4
        cfg = TransducerDecoderModel.default_config().set(
            name="transducer_decoder",
            input_dim=am_dim,
            lm_dim=lm_dim,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )
        if lm_type == "lstm_lm":
            rnn_cfg = LSTMCell.default_config().set(hidden_dim=12)
        elif lm_type == "recurrent_lm":
            rnn_cfg = SimpleRecurrentCell.default_config()
            cfg.transducer.activation_fn = "linear"
        else:
            rnn_cfg = IdentityCell.default_config()
            cfg.transducer.activation_fn = "linear"

        cfg.prediction_network.rnn_cell = rnn_cfg
        cfg.prediction_network.emb_dim = emb_dim

        if blank_logit_bias:
            cfg.transducer.logits_to_log_probs.blank_logit_bias = blank_logit_bias

        layer = cfg.instantiate(parent=None)  # type: TransducerDecoderModel
        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        if lm_type != "lstm_lm":
            # Initialize weights.
            layer_params["transducer"]["proj"]["weight"] = jnp.identity(joint_dim)
            layer_params["lm_proj"]["weight"] = jnp.identity(lm_dim)

        if lm_type == "stateless_lm":
            layer_params["am_proj"]["weight"] = jnp.identity(am_dim)
            layer_params["prediction_network"]["embedding"]["weight"] = jnp.zeros(
                (emb_dim, emb_dim)
            )
        elif lm_type == "stateful_lm":
            layer_params["am_proj"]["weight"] = jnp.zeros((am_dim, am_dim))
            # [vocab_size, emb_dim].
            layer_params["prediction_network"]["embedding"]["weight"] = jnp.array(
                [
                    # blank. This should never be used.
                    [jnp.log(0.25), jnp.log(0.25), jnp.log(0.25), jnp.log(0.25)],
                    [jnp.log(0.8), _NEG_INF, _NEG_INF, jnp.log(0.2)],  # bos.
                    [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],  # eos.
                    [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],  # token.
                ]
            )
        elif lm_type == "recurrent_lm":
            layer_params["am_proj"]["weight"] = jnp.zeros((am_dim, am_dim))
            # [vocab_size, emb_dim].
            layer_params["prediction_network"]["embedding"]["weight"] = jnp.array(
                [
                    # blank. This should never be used.
                    [jnp.log(0.25), jnp.log(0.25), jnp.log(0.25), jnp.log(0.25)],
                    # bos. Markov chain initial state.
                    [jnp.log(0.8), _NEG_INF, _NEG_INF, jnp.log(0.2)],
                    # eos.
                    [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],
                    # token.
                    [jnp.log(0.25), jnp.log(0.25), jnp.log(0.25), jnp.log(0.25)],
                ]
            )
        return layer, layer_params, prng_key

    @parameterized.parameters(
        # [batch_size, tgt_len].
        (
            jnp.array(
                [
                    [14, 8, 17, 19, 17, 2],  # length 5.
                    [17, 4, 18, 2, -1, -1],  # length 3.
                    [2, -1, -1, -1, -1, -1],  # length 0.
                ]
            ),
            False,
        ),
        (jnp.array([[9, 11, 8, 5, 2, -1, -1, -1]]), True),
    )
    @set_threefry_partitionable(False)  # TODO(Luzy): update for threefry_partitionable True
    def test_forward(self, target_labels, tile_input):
        """Tests that loss computation excludes empty sequence, and respects paddings."""
        am_dim, bos_id = 4, 1
        layer, layer_params, prng_key = self._set_up_transducer(vocab_size=20)

        # Generate inputs.
        if tile_input:
            batch_size, src_len, max_src_len = 4, 5, 10
            # [batch_size, src_len, am_dim].
            inputs = jnp.tile(
                jax.random.normal(jax.random.PRNGKey(707), [1, max_src_len, am_dim]) * 1000,
                [batch_size, 1, 1],
            )
            paddings = jnp.tile(jnp.arange(max_src_len)[None, :] >= src_len, [batch_size, 1])
            # Generate different padding data.
            pad_inputs_data = (
                jax.random.normal(jax.random.PRNGKey(124), [batch_size, max_src_len, am_dim]) * 200
            )
            # Generate inputs with the same data at non-pad positions.
            inputs = jnp.where(paddings[:, :, None], pad_inputs_data, inputs)
            assert_allclose(
                jnp.diff(inputs[:, :src_len], axis=0), jnp.zeros([batch_size - 1, src_len, am_dim])
            )
            self.assertGreater(
                jnp.abs(jnp.diff(inputs[:, src_len:], axis=0)).sum(),
                1e3,
            )
            target_labels = jnp.tile(target_labels, [batch_size, 1])
        else:
            batch_size = target_labels.shape[0]
            max_src_len = 10
            src_len = np.array([10, 0, 7])
            # [batch_size, src_len, am_dim].
            inputs = (
                jax.random.normal(jax.random.PRNGKey(311), [batch_size, max_src_len, am_dim]) * 1000
            )
            paddings = jnp.arange(max_src_len)[None, :] >= src_len[:, None]

        input_ids = jnp.concatenate(
            [jnp.full([batch_size, 1], bos_id), target_labels[:, :-1]], axis=1
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
                target=dict(input_ids=input_ids),
            )
        )
        assert_allclose(
            loss,
            (aux_outputs["per_example_loss"] * aux_outputs["per_example_weight"]).sum()
            / jnp.sum(aux_outputs["per_example_weight"]),
        )
        if tile_input:
            # The loss is the same for all examples in the batch.
            assert_allclose(jnp.diff(aux_outputs["per_example_loss"]), jnp.zeros(batch_size - 1))
            assert_allclose(aux_outputs["per_example_weight"], jnp.ones(batch_size))
            expected_loss = 25.908243
        else:
            # Empty source example has weight 0.
            expected_weight = jnp.array([1.0, 0.0, 1.0])
            self.assertNestedEqual(aux_outputs["per_example_weight"], expected_weight)
            expected_loss = 29.287447
        assert_allclose(loss, expected_loss)

    def _generate_decode_test_data(self, lm_type):
        if lm_type == "stateless_lm":
            # Generate inputs.
            max_src_len = 5
            src_len = jnp.array([2, 2, 3, 3])

            # [batch_size, max_src_len, am_dim].
            am_data = jnp.array(
                [
                    [  # Test that blank token transits to the next frame.
                        [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],
                        [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                    ],
                    [  # Test that label token stays at the same frame.
                        [_NEG_INF, _NEG_INF, _NEG_INF, jnp.log(1.0)],
                        [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                    ],
                    [  # Test that the decoding exits with 3 blank emissions.
                        [jnp.log(0.5), _NEG_INF, _NEG_INF, jnp.log(0.5)],
                        [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],
                        [jnp.log(1.0), _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                    ],
                    [  # Test that eos token is excluded until the last frame.
                        # But eos token's probability affect the beam search scores.
                        [jnp.log(0.2), _NEG_INF, jnp.log(0.6), jnp.log(0.2)],
                        [jnp.log(0.5), _NEG_INF, jnp.log(0.5), _NEG_INF],
                        [jnp.log(0.5), _NEG_INF, jnp.log(0.5), _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                        [_NEG_INF, _NEG_INF, _NEG_INF, _NEG_INF],
                    ],
                ]
            )
            am_paddings = jnp.arange(max_src_len)[None, :] >= src_len[:, None]
            expected_decodes = dict(
                raw_sequences=jnp.array(
                    [
                        [
                            [0, 0, 2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [3, 3, 3, 3, 3, 3, 2],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 2, 0, 0, 0],
                            [3, 0, 0, 0, 2, 0, 0],
                            [3, 3, 0, 0, 0, 2, 0],
                            [3, 3, 3, 0, 0, 0, 2],
                        ],
                        [
                            [0, 0, 0, 2, 0, 0, 0],
                            [3, 0, 0, 0, 2, 0, 0],
                            [3, 3, 0, 0, 0, 2, 0],
                            [3, 3, 3, 0, 0, 0, 2],
                        ],
                    ]
                ),
                paddings=jnp.array(
                    [
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                        ],
                        [
                            [0, 0, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                        ],
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1],
                        ],
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1],
                        ],
                    ]
                ).astype(jnp.bool),
                probabilities=jnp.array(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0.5, 0.25, 0.125, 0.0625],
                        [0.05, 0.01, 0.002, 0.0004],
                    ]
                ),
            )

        elif lm_type == "stateful_lm":
            batch_size, max_src_len, am_dim = 2, 5, 4
            src_len = jnp.array([3, 4])
            # [batch_size, max_src_len, am_dim].
            am_data = jnp.zeros([batch_size, max_src_len, am_dim])
            am_paddings = jnp.arange(max_src_len)[None, :] >= src_len[:, None]
            expected_decodes = dict(
                raw_sequences=jnp.array(
                    [
                        [
                            [0, 0, 0, 2, 0, 0, 0],
                            [3, 0, 0, 0, 2, 0, 0],
                            [0, 3, 0, 0, 2, 0, 0],
                            [0, 0, 3, 0, 2, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0, 2, 0, 0],
                            [3, 0, 0, 0, 0, 2, 0],
                            [0, 3, 0, 0, 0, 2, 0],
                            [0, 0, 3, 0, 0, 2, 0],
                        ],
                    ]
                ),
                paddings=jnp.array(
                    [
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                        ],
                        [
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1],
                        ],
                    ]
                ).astype(jnp.bool),
                probabilities=jnp.array(
                    [
                        [0.8**3, 0.2, 0.8 * 0.2, 0.8**2 * 0.2],
                        [0.8**4, 0.2, 0.8 * 0.2, 0.8**2 * 0.2],
                    ]
                ),
            )
        elif lm_type == "recurrent_lm":
            batch_size, max_src_len, am_dim = 2, 4, 4
            src_len = jnp.array([3, 4])
            # [batch_size, max_src_len, am_dim].
            am_data = jnp.zeros([batch_size, max_src_len, am_dim])
            am_paddings = jnp.arange(max_src_len)[None, :] >= src_len[:, None]
            expected_decodes = dict(
                raw_sequences=jnp.array(
                    [
                        [[0, 0, 0, 2, 0], [3, 3, 3, 3, 2], [0, 0, 3, 3, 2], [0, 3, 3, 3, 2]],
                        [[0, 0, 0, 0, 2], [0, 0, 0, 3, 2], [3, 3, 3, 3, 2], [0, 0, 3, 3, 2]],
                    ]
                ),
                paddings=jnp.array(
                    [
                        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]],
                        [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 1, 1, 1]],
                    ]
                ).astype(jnp.bool),
                probabilities=jnp.array(
                    [
                        [
                            0.8**3,
                            0.2 * 0.6 * 0.8 * 0.9,
                            0.8 * 0.8 * 0.2 * 0.6,
                            0.8 * 0.2 * 0.6 * 0.8,
                        ],
                        [0.8**4, 0.8**3 * 0.2, 0.2 * 0.6 * 0.8 * 0.9, 0.8 * 0.8 * 0.2 * 0.6],
                    ]
                ),
            )
        else:
            raise ValueError(f"Unrecognized lm type {lm_type}.")

        return dict(
            inputs=am_data,
            paddings=am_paddings,
            expected_decodes=expected_decodes,
        )

    @parameterized.parameters(["stateless_lm", "stateful_lm", "recurrent_lm"])
    def test_special_beam_search_decode(self, lm_type):
        """Tests beam search decode.

        In stateless_lm, prediction network is set to all 0s. We test that am input
            transits to the next frame by blank token.

        In stateful_lm, prediction network is an embedding lookup, and am_proj is set to all 0s.
            We test that lm input transits to the next token by label token.

        In recurrent_lm, prediction network is a toy markov chain. and am_proj is set to all 0s.
            We test that prediction network state transits by label token.
        """
        vocab_size, num_decodes = 4, 4
        max_decode_len = 5 if lm_type == "recurrent_lm" else 7
        layer, layer_params, prng_key = self._set_up_transducer(
            vocab_size=vocab_size, lm_type=lm_type
        )

        test_data = self._generate_decode_test_data(lm_type=lm_type)

        beam_search_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch=dict(inputs=test_data["inputs"], paddings=test_data["paddings"]),
                num_decodes=num_decodes,
                max_decode_len=max_decode_len,
            ),
            is_training=False,
            prng_key=prng_key,
            state=layer_params,
            method="beam_search_decode",
        )

        self.assertNestedEqual(
            beam_search_outputs.raw_sequences,
            test_data["expected_decodes"]["raw_sequences"],
        )
        self.assertNestedEqual(
            beam_search_outputs.paddings,
            test_data["expected_decodes"]["paddings"],
        )
        assert_allclose(
            jnp.exp(beam_search_outputs.scores),
            test_data["expected_decodes"]["probabilities"],
        )
        # Check that no blank id in the final sequence.
        self.assertTrue(
            jnp.all(
                jnp.logical_or(
                    beam_search_outputs.sequences != layer.config.blank_id,
                    beam_search_outputs.paddings,
                )
            )
        )

    def test_general_beam_search_decode(self):
        am_dim, vocab_size = 4, 8
        num_decodes, max_decode_len = 4, 10
        layer, layer_params, prng_key = self._set_up_transducer(
            vocab_size,
            blank_logit_bias=0.6,
        )

        batch_size, max_src_len = 3, 5
        src_len = jnp.array([2, 5, 0])
        # [batch_size, src_len, am_dim].
        am_data = jax.random.normal(jax.random.PRNGKey(312), [batch_size, max_src_len, am_dim])
        am_paddings = jnp.arange(max_src_len)[None, :] >= src_len[:, None]
        # Test beam_search_decode.
        beam_search_outputs, _ = F(
            layer,
            inputs=dict(
                input_batch=dict(inputs=am_data, paddings=am_paddings),
                num_decodes=num_decodes,
                max_decode_len=max_decode_len,
            ),
            is_training=False,
            prng_key=prng_key,
            state=layer_params,
            method="beam_search_decode",
        )
        # src_len = 0 for the 3rd example.
        self.assertNestedEqual(
            beam_search_outputs.paddings[2],
            jnp.ones((num_decodes, max_decode_len), dtype=jnp.bool),
        )
        # The decoding finishes with the eos token.
        self.assertNestedEqual(
            jnp.sum(beam_search_outputs.raw_sequences[:2] == 2, axis=-1),
            jnp.ones((batch_size - 1, num_decodes), dtype=jnp.int32),
        )
        # Check that token_ids does not contain blank tokens.
        self.assertTrue(
            jnp.all(
                jnp.logical_or(
                    beam_search_outputs.sequences != layer.config.blank_id,
                    beam_search_outputs.paddings,
                )
            )
        )


class LASDecoderModelTest(TestCase):
    def _set_up_las(self, encoder_dim, decoder_dim, num_heads, vocab_size, pad_id=-1):
        num_layers = 2
        cfg = LASDecoderModel.default_config().set(
            name="las_decoder",
            input_dim=encoder_dim,
            vocab_size=vocab_size,
        )
        cfg.decoder = causal_lm.gpt_decoder_config(
            stack_cfg=attention.StackedTransformerLayer.default_config(),
            num_layers=num_layers,
            hidden_dim=decoder_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.gelu",
            max_position_embeddings=10,
            layer_norm_epsilon=0.1,
            dropout_rate=0.0,
        )
        cfg.decoder.pad_token_id = pad_id
        transformer_cfg = cfg.decoder.transformer.layer
        transformer_cfg.cross_attention = attention.TransformerAttentionLayer.default_config()
        transformer_cfg.cross_attention.attention.num_heads = num_heads
        return cfg

    def test_forward(self):
        encoder_dim, decoder_dim, num_heads, vocab_size = 8, 6, 3, 20
        cfg: LASDecoderModel.Config = self._set_up_las(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
        )
        bos_id = eos_id = cfg.decoder.eos_token_id
        pad_id = cfg.decoder.pad_token_id
        # Initialize layer parameters.
        layer: LASDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, input_key = jax.random.split(prng_key, num=3)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_src_len = 3, 10
        target_labels = jnp.array(
            [
                [14, 8, 17, 19, 17, eos_id],  # length 5.
                [17, 4, 18, eos_id, pad_id, pad_id],  # length 3.
                [eos_id, pad_id, pad_id, pad_id, pad_id, pad_id],  # length 0.
            ]
        )
        input_ids = jnp.concatenate(
            [jnp.full([batch_size, 1], bos_id), target_labels[:, :-1]], axis=1
        )

        src_len = np.array([10, 0, 7])
        target_len = np.array([6, 4, 1])
        # [batch_size, src_len, am_dim].
        inputs = jax.random.normal(input_key, [batch_size, max_src_len, encoder_dim]) * 1000
        paddings = jnp.arange(max_src_len)[None, :] >= src_len[:, None]

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
                target=dict(input_ids=input_ids),
            )
        )
        # Empty source example has weight 0.
        expected_weight = target_len * (src_len > 0)
        self.assertNestedAllClose(aux_outputs["per_example_weight"], expected_weight)
        assert_allclose(
            loss,
            aux_outputs["per_example_loss"].sum() / aux_outputs["per_example_weight"].sum(),
        )
        self.assertGreater(loss, 0.0)

    def test_decode(self):
        encoder_dim, decoder_dim, num_heads, vocab_size = 5, 16, 4, 20
        num_decodes = 5
        cfg: LASDecoderModel.Config = self._set_up_las(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
        )
        pad_id = cfg.decoder.pad_token_id
        # Initialize layer parameters.
        layer: LASDecoderModel = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        decode_key, init_key, input_key, prefix_key = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        batch_size, max_src_len, max_tgt_len = 3, 10, 8
        src_len = jnp.array([10, 7, 5])
        prefix_length = jnp.array([1, 3, 6])

        # [batch_size, max_seq_len, dim].
        inputs = jax.random.normal(input_key, [batch_size, max_src_len, cfg.input_dim]) * 1000
        # [batch_size, max_seq_len].
        paddings = jnp.arange(max_src_len) >= src_len[:, None]
        prefix = jax.random.randint(
            prefix_key,
            shape=[batch_size, max_tgt_len],
            # Prefix can consist of any tokens, including pad and eos.
            minval=0,
            maxval=vocab_size,
        )
        # Explicitly fill positions >= prefix_length with pad_id (-1).
        # Note that each batch example may have a different prefix length.
        # [batch_size, max_tgt_len].
        prefix_mask = jnp.arange(max_tgt_len) < prefix_length[:, None]
        prefix = prefix * prefix_mask + pad_id * (1 - prefix_mask)

        @functools.partial(jax.jit, static_argnames=("method", "num_decodes", "logits_modifier"))
        def jit_method(inputs, prng_key, method, num_decodes, logits_modifier=None):
            if logits_modifier is not None:
                inputs["logits_modifier"] = logits_modifier
            outputs, _ = F(
                layer,
                inputs=dict(**inputs, max_decode_len=max_tgt_len, num_decodes=num_decodes),
                is_training=True,
                prng_key=prng_key,
                state=layer_params,
                method=method,
            )
            return outputs

        decode_inputs = dict(
            input_batch=dict(inputs=inputs, paddings=paddings, prefix=prefix),
        )

        # Beam search decode.
        beam_search_outputs: DecodeOutputs = jit_method(
            decode_inputs,
            prng_key=decode_key,
            method="beam_search_decode",
            num_decodes=num_decodes,
        )
        # Check that beams are sorted descending by score.
        self.assertTrue(jnp.all(jnp.diff(beam_search_outputs.scores, axis=-1) <= 0))
        self.assertTrue(jnp.all(beam_search_outputs.sequences * beam_search_outputs.paddings == 0))
        self.assertSequenceEqual(
            beam_search_outputs.sequences.shape, [batch_size, num_decodes, max_tgt_len]
        )

        # Sample decode.
        sample_outputs: DecodeOutputs = jit_method(
            decode_inputs,
            prng_key=decode_key,
            method="sample_decode",
            num_decodes=2,
        )
        self.assertSequenceEqual(sample_outputs.sequences.shape, [batch_size, 2, max_tgt_len])


if __name__ == "__main__":
    absltest.main()

# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/t5x:
# Copyright 2023 The T5X Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for beam_search.

Adapted from: https://github.com/google-research/t5x/blob/main/t5x/decoding_test.py
"""
# pylint: disable=no-self-use,too-many-lines,protected-access
import logging
import os
from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import seqio
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common import decoding, utils
from axlearn.common.utils import NestedTensor, Tensor

EOS_ID = 1
NEG_INF = decoding.NEG_INF
tokenizers_dir = os.path.join(os.path.dirname(__file__), "../data/tokenizers")
_SENTENCEPIECE_DIR = os.path.join(tokenizers_dir, "sentencepiece")
_T5_VOCAB_FILE = os.path.join(_SENTENCEPIECE_DIR, "t5-base")


class _TokenSumPrefixMerger(decoding.PrefixMerger):
    """A PrefixMerge that merges all prefixes with the same sum of ids."""

    def init_state(self, *, tokens: Tensor) -> NestedTensor:
        return dict(sum=tokens.sum(axis=-1))

    def compute(self, state: NestedTensor) -> Tensor:
        return decoding.compute_merge_matrix_by_prefix_ids(state["sum"])

    def update(self, *, tokens: Tensor, state: NestedTensor) -> NestedTensor:
        return dict(sum=state["sum"] + tokens)


class DecodeTest(parameterized.TestCase):
    def test_brevity_penalty_fn(self):
        raw_scores = jax.random.normal(jax.random.PRNGKey(0), shape=(3, 4))
        t5_brevity_penalty_score = decoding.brevity_penalty_fn()(
            length=jnp.array(1), raw_scores=raw_scores
        )
        np.testing.assert_allclose(t5_brevity_penalty_score, raw_scores / 1, atol=1e-6)

        t5_brevity_penalty_score = decoding.brevity_penalty_fn(alpha=0.5)(
            length=jnp.array(19), raw_scores=raw_scores
        )
        np.testing.assert_allclose(t5_brevity_penalty_score, raw_scores / 2, atol=1e-6)

        hf_brevity_penalty_score = decoding.brevity_penalty_fn(bp_type="hf")(
            length=jnp.array(1), raw_scores=raw_scores
        )
        np.testing.assert_allclose(hf_brevity_penalty_score, raw_scores / 1, atol=1e-6)

        hf_brevity_penalty_score = decoding.brevity_penalty_fn(alpha=0.5, bp_type="hf")(
            length=jnp.array(4), raw_scores=raw_scores
        )
        np.testing.assert_allclose(t5_brevity_penalty_score, raw_scores / 2, atol=1e-6)

        with self.assertRaises(NotImplementedError):
            fn = decoding.brevity_penalty_fn(bp_type="next")  # pytype: disable=wrong-arg-types
            fn(length=jnp.array(1), raw_scores=raw_scores)

    def test_beam_search_decode_with_brevity_penalty(self):
        # Toy problem, we have 4 states, A, B, START, END, (plus PAD).
        # Scores are given by a first-order Markov model.
        batch_size = 2
        beam_size = 3
        # PAD doesn't matter for this test, but part of the contract for beam_search
        # is giving the PAD token id 0.
        states = ["PAD", "A", "B", "START-", "-END"]
        num_states = len(states)
        decode_length = 5

        # Edge potentials (written inside edges for diagonals):
        #            END
        #              \ 1.2
        #            1  \   -1
        #         A ---- A ---- A
        #       0   \  -1  \  1   0
        # START      X      X       END
        #       0   /  -1  /  1   0
        #         B ---- B ---- B
        #            1  /   -1
        #              / 1.2
        #            END

        # put the above edge potentials in a 3-tensor
        # There are 4 time steps (edges) with 5 states (list is above).
        ab_edge_potentials = np.asarray([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]])
        # now we have to add on the START, END states
        # and PAD at 0
        edge_potentials = np.ones([4, 5, 5]) * NEG_INF
        edge_potentials[1:3, 1:3, 1:3] = ab_edge_potentials
        # START can go to either A or B for free at t0
        edge_potentials[0, 3, 1] = 0
        edge_potentials[0, 3, 2] = 0
        # either A or B can go to END for free at t3
        edge_potentials[3, 1, 4] = 0
        edge_potentials[3, 2, 4] = 0
        # either A or B can go to END at t2
        edge_potentials[2, 2, 4] = 1.2
        edge_potentials[2, 1, 4] = 1.2
        # PAD can go to anything for free (doesn't matter for this test)
        edge_potentials[:, 0, :] = 0
        # END can go to the PAD for free.
        edge_potentials[:, 4, 0] = 0

        edge_potentials = jnp.asarray(edge_potentials)

        # at time 0, we start with state=START=3
        logits0 = jnp.asarray([NEG_INF, NEG_INF, NEG_INF, 0, NEG_INF])

        # add dummy flattened batch x beam dim for broadcasting
        logits0 = jnp.expand_dims(logits0, axis=0)
        edge_potentials = jnp.expand_dims(edge_potentials, axis=0)

        def tokens_to_scores(
            token_indices: Tensor, state_cache: NestedTensor
        ) -> tuple[Tensor, NestedTensor]:
            cur_iter = state_cache["cur_iter"]
            # grab edge potentials for the current timestep
            cur_edge_potentials = jnp.take_along_axis(
                edge_potentials,
                jnp.reshape(jnp.maximum(0, cur_iter[:, 0] - 1), (batch_size * beam_size, 1, 1, 1)),
                axis=1,
            )
            cur_edge_potentials = jnp.squeeze(cur_edge_potentials, axis=1)
            # get "logits" from edge potentials for requested tokens (except at t0)
            cur_logits = jnp.matmul(
                jnp.reshape(
                    jax.nn.one_hot(token_indices, num_states, axis=1),
                    (batch_size * beam_size, 1, num_states),
                ),
                cur_edge_potentials,
            )
            cur_logits = jnp.squeeze(cur_logits, axis=1)
            # use our START-only logits for t0, otherwise use the edge potentials
            logits_for_tokens = jnp.where(cur_iter == 0, logits0, cur_logits)
            log_probs_for_tokens = jax.nn.log_softmax(logits_for_tokens)
            # update state in the cache
            new_cache = state_cache.copy()
            new_cache["cur_iter"] = cur_iter + 1
            return log_probs_for_tokens, new_cache

        init_cache = {}
        init_cache["cur_iter"] = jnp.zeros((batch_size, 1), dtype=jnp.int32)

        # alpha is zero and beam search will prefer shorter sequences.
        inputs = np.zeros([batch_size, decode_length])
        beam_search_output = decoding.beam_search_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=0),
            cache=init_cache,
            tokens_to_scores=tokens_to_scores,
            eos_id=4,
            num_decodes=beam_size,
            max_decode_len=decode_length,
            brevity_penalty=decoding.brevity_penalty_fn(bp_type="hf"),
            loop="python",
        )
        top_scoring = beam_search_output.sequences
        no_bp_scores = beam_search_output.scores

        # The three top scoring sequences should be:
        # START-AA-ENDPAD
        # START-BB-ENDPAD
        # and
        # START-AAB-END
        # Although  START-AAB-END and START-BBA-END have the same log probs, jax,lax.top_k
        # will prefer the smaller index, so we expect START-AAB-END here.
        top_scoring_strings = [
            "".join(states[tok] for tok in top_scoring[0, i, :]) for i in range(beam_size)
        ]
        expected = ["START-AA-ENDPAD", "START-BB-ENDPAD", "START-AAB-END"]
        np.testing.assert_array_equal(expected, top_scoring_strings)

        alpha = 0.9

        # Length normalization with alpha 1 and hf bp type.
        # Divides the log probs by the length of the input sequence.
        inputs = np.zeros([batch_size, decode_length])
        beam_search_output = decoding.beam_search_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=0),
            cache=init_cache,
            tokens_to_scores=tokens_to_scores,
            eos_id=4,
            num_decodes=beam_size,
            max_decode_len=decode_length,
            brevity_penalty=decoding.brevity_penalty_fn(alpha=alpha, bp_type="hf"),
        )
        top_scoring = beam_search_output.sequences

        # The three top scoring sequences should be:
        # START-AAB-END
        # START-BBA-END
        # and
        # START-AA-ENDPAD
        top_scoring_strings = [
            "".join(states[tok] for tok in top_scoring[0, i, :]) for i in range(beam_size)
        ]
        expected = ["START-AAB-END", "START-BBA-END", "START-AA-ENDPAD"]
        np.testing.assert_array_equal(expected, top_scoring_strings)
        bp_scores = beam_search_output.scores

        # no_bp_scores[0][0] and bp_scores[0][2] correspond the log probs of 'START-AA-ENDPAD'
        # with no length normalization and length normalization.
        # bp_scores[0][2] should be nobp_scores[0][0] / (len('START-AA-ENDPAD') ** alpha)
        # Here len('START-AA-ENDPAD') is 4 since PAD is ignored.
        np.testing.assert_almost_equal(
            no_bp_scores[0][0] / (4**alpha), bp_scores[0][2], decimal=5
        )
        # no_bp_scores[0][2] and bp_scores[0][0] correspond the log probs of 'START-AAB-END'
        np.testing.assert_almost_equal(
            no_bp_scores[0][2] / (5**alpha), bp_scores[0][0], decimal=5
        )

    def test_add_decoding_dim(self):
        x = np.array([[0, 5, 1, 0], [0, 8, 6, 9]], dtype=np.int32)
        y = decoding.add_decoding_dim(x, num_decodes=3)
        self.assertEqual(y.shape, (2, 3, 4))
        np.testing.assert_array_equal(
            [
                [[0, 5, 1, 0], [0, 5, 1, 0], [0, 5, 1, 0]],
                [[0, 8, 6, 9], [0, 8, 6, 9], [0, 8, 6, 9]],
            ],
            y,
        )

    def test_flatten_decoding_dim(self):
        x = np.array(
            [
                [[0, 5, 1, 0], [0, 5, 1, 0], [0, 5, 1, 0]],
                [[0, 8, 6, 9], [0, 8, 6, 9], [0, 8, 6, 9]],
            ],
            dtype=np.int32,
        )
        y = decoding.flatten_decoding_dim(x)
        self.assertEqual(y.shape, (6, 4))
        np.testing.assert_array_equal(
            [[0, 5, 1, 0], [0, 5, 1, 0], [0, 5, 1, 0], [0, 8, 6, 9], [0, 8, 6, 9], [0, 8, 6, 9]], y
        )

    @parameterized.parameters(jnp, np)
    def test_unflatten_decoding_dim(self, module):
        x = module.array(
            [[0, 5, 1, 0], [0, 5, 1, 0], [0, 5, 1, 0], [0, 8, 6, 9], [0, 8, 6, 9], [0, 8, 6, 9]],
            dtype=np.int32,
        )
        y = decoding.unflatten_decoding_dim(x, batch_size=2, num_decodes=3)
        self.assertEqual(y.shape, (2, 3, 4))
        np.testing.assert_array_equal(
            [
                [[0, 5, 1, 0], [0, 5, 1, 0], [0, 5, 1, 0]],
                [[0, 8, 6, 9], [0, 8, 6, 9], [0, 8, 6, 9]],
            ],
            y,
        )

    def test_gather_beams(self):
        batch_size = 2
        old_beam_size = 2
        new_beam_size = 4
        length = 5
        num_heads = 5
        per_head_dim = 5
        beam_indices = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        seqs = np.array(
            [
                [[73, 29, 175, 123, 70], [89, 199, 4, 63, 149]],
                [[56, 167, 83, 67, 9], [115, 187, 182, 140, 157]],
            ]
        )  # vocab_size = 200

        np_ones_array = np.ones([1, 1, num_heads, length, per_head_dim])  # batch, beam
        np_zeros_array = np.zeros([1, 1, num_heads, length, per_head_dim])

        cache = {
            "layer_0": {
                "attention": {
                    "params": np.repeat(
                        np.concatenate([np_ones_array, np_zeros_array], axis=1), 2, axis=0
                    )
                }
            }
        }

        gathered_seqs, gathered_cache = decoding._gather_beams(
            [seqs, cache], beam_indices, batch_size, old_beam_size, new_beam_size
        )
        self.assertEqual(gathered_seqs.shape, (batch_size, new_beam_size, length))
        expected_seqs = [
            [
                [73, 29, 175, 123, 70],
                [73, 29, 175, 123, 70],
                [89, 199, 4, 63, 149],
                [89, 199, 4, 63, 149],
            ],
            [
                [56, 167, 83, 67, 9],
                [115, 187, 182, 140, 157],
                [56, 167, 83, 67, 9],
                [115, 187, 182, 140, 157],
            ],
        ]
        expected_cache = {
            "layer_0": {
                "attention": {
                    "params": jnp.concatenate(
                        [
                            jnp.concatenate(
                                [np_ones_array, np_ones_array, np_zeros_array, np_zeros_array],
                                axis=1,
                            ),
                            jnp.concatenate(
                                [np_ones_array, np_zeros_array, np_ones_array, np_zeros_array],
                                axis=1,
                            ),
                        ],
                        axis=0,
                    )
                }
            }
        }
        np.testing.assert_array_equal(expected_seqs, gathered_seqs)
        jax.tree.map(np.testing.assert_array_equal, expected_cache, gathered_cache)

    def test_beam_search_decode(self):
        # Toy problem, we have 4 states, A, B, START, END, (plus PAD).
        # Scores are given by a first-order Markov model.
        batch_size = 2
        beam_size = 2
        # PAD doesn't matter for this test, but part of the contract for beam_search
        # is giving the PAD token id 0.
        states = ["PAD", "A", "B", "START-", "-END"]
        num_states = len(states)
        decode_length = 7

        # Edge potentials (written inside edges for diagonals):
        #            1      -1     1      -1
        #         A ---- A ---- A ---- A ---- A
        #       0   \  -1  \  1   \  -1  \  1   0
        # START      X      X      X      X       END
        #       0   /  -1  /  1   /  -1  / 0.5  0
        #         B ---- B ---- B ---- B ---- B
        #            1      -1     1      -1

        # put the above edge potentials in a 3-tensor
        # There are 6 time steps (edges) with 5 states (list is above).
        ab_edge_potentials = np.asarray(
            [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]], [[1, -1], [-1, 1]], [[-1, 0.5], [1, -1]]]
        )
        # now we have to add on the START, END states
        # and PAD at 0
        edge_potentials = np.ones([6, 5, 5]) * NEG_INF
        edge_potentials[1:5, 1:3, 1:3] = ab_edge_potentials
        # START can go to either A or B for free at t0
        edge_potentials[0, 3, 1] = 0
        edge_potentials[0, 3, 2] = 0
        # either A or B can go to END for free at t5
        edge_potentials[5, 1, 4] = 0
        edge_potentials[5, 2, 4] = 0
        # PAD can go to anything for free (doesn't matter for this test)
        edge_potentials[:, 0, :] = 0

        edge_potentials = jnp.asarray(edge_potentials)

        # at time 0, we start with state=START=3
        logits0 = jnp.asarray([NEG_INF, NEG_INF, NEG_INF, 0, NEG_INF])

        # add dummy flattened batch x beam dim for broadcasting
        logits0 = jnp.expand_dims(logits0, axis=0)
        edge_potentials = jnp.expand_dims(edge_potentials, axis=0)

        def tokens_to_scores(
            token_indices: Tensor, state_cache: NestedTensor
        ) -> tuple[Tensor, NestedTensor]:
            cur_iter = state_cache["cur_iter"]
            # grab edge potentials for the current timestep
            cur_edge_potentials = jnp.take_along_axis(
                edge_potentials,
                jnp.reshape(jnp.maximum(0, cur_iter[:, 0] - 1), (batch_size * beam_size, 1, 1, 1)),
                axis=1,
            )
            cur_edge_potentials = jnp.squeeze(cur_edge_potentials, axis=1)
            # get "logits" from edge potentials for requested tokens (except at t0)
            cur_logits = jnp.matmul(
                jnp.reshape(
                    jax.nn.one_hot(token_indices, num_states, axis=1),
                    (batch_size * beam_size, 1, num_states),
                ),
                cur_edge_potentials,
            )
            cur_logits = jnp.squeeze(cur_logits, axis=1)
            # use our START-only logits for t0, otherwise use the edge potentials
            logits_for_tokens = jnp.where(cur_iter == 0, logits0, cur_logits)
            log_probs_for_tokens = jax.nn.log_softmax(logits_for_tokens)
            # update state in the cache
            new_cache = state_cache.copy()
            new_cache["cur_iter"] = cur_iter + 1
            return log_probs_for_tokens, new_cache

        init_cache = {}
        init_cache["cur_iter"] = jnp.zeros((batch_size, 1), dtype=jnp.int32)

        inputs = np.zeros([batch_size, decode_length])
        beam_search_output = decoding.beam_search_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=0),
            cache=init_cache,
            tokens_to_scores=tokens_to_scores,
            eos_id=4,
            num_decodes=beam_size,
            max_decode_len=decode_length,
        )
        top_scoring = beam_search_output.sequences

        # The two top scoring sequences should be:
        # START-AABBA-END
        # and
        # START-BBAAB-END
        # (and greedy beam search will find both these with just two beams)

        top_scoring_strings = [
            "".join(states[tok] for tok in top_scoring[0, i, :]) for i in range(beam_size)
        ]

        expected = ["START-AABBA-END", "START-BBAAB-END"]
        np.testing.assert_array_equal(expected, top_scoring_strings)

    @parameterized.parameters([False, True])
    def test_beam_search_decode_force_prefix(self, use_eos_as_prefix: bool):
        beam_size = 2
        # Use id 2 then 3 for batch example 0 and id 3 then 2 for example 1.
        batch_log_probs = np.array(
            [
                [-1e5, -1e6, -0.37, -1.17, -1e4, -1e4, -1e4, -1e4],
                [-1e5, -1e6, -1.17, -0.37, -1e4, -1e4, -1e4, -1e4],
            ],
            dtype=np.float32,
        )

        def token_to_scores(ids, cache):  # pylint: disable=unused-argument
            log_probs = np.repeat(np.expand_dims(batch_log_probs, axis=1), [beam_size], axis=1)
            log_probs = decoding.flatten_decoding_dim(log_probs)
            return log_probs, {}

        # batch element 0 has length 1 and element 1 has length 2.
        start_token_id = EOS_ID if use_eos_as_prefix else 0
        # we only use the EOS for the first example in the batch to demonstrate
        # that EOS prefixes work when even a partial set of examples in a batch
        # start with an EOS token and the others do not.
        inputs = np.array([[start_token_id, 7, 0, 0, 0], [0, 4, 5, 0, 0]], dtype=np.int32)
        rolled_inputs = np.array([[7, 0, 0, 0, 0], [4, 5, 0, 0, 0]], dtype=np.int32)
        beam_search_output = decoding.beam_search_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=0),
            cache={},
            tokens_to_scores=token_to_scores,
            eos_id=EOS_ID,
            num_decodes=beam_size,
        )
        beam_search_seqs = beam_search_output.sequences
        beam_search_scores = beam_search_output.scores

        # Beam search scores are in the descending order.
        self.assertTrue(np.all(np.diff(beam_search_scores) <= 0))
        # Prefixes are forced depending on inputs.
        expected_sequences = np.array(
            [[[7, 1, 0, 0, 0], [7, 2, 1, 0, 0]], [[4, 5, 1, 0, 0], [4, 5, 3, 1, 0]]]
        )
        np.testing.assert_array_equal(expected_sequences, beam_search_seqs)

        expected_scores = []
        for batch, log_probs, prompt in zip(expected_sequences, batch_log_probs, rolled_inputs):
            beam_expected_scores = []
            for beam in batch:
                # Add them directly since they are static.
                beam_scores = []
                for token, prompt_token in zip(beam, prompt):
                    # A sequence ending with eos_id will be padded with 0. Do not include their
                    # scores.
                    if prompt_token != 0 or token == 0:
                        beam_scores.append(0)
                    else:
                        beam_scores.append(log_probs[token])
                beam_expected_scores.append(sum(beam_scores))
            expected_scores.append(beam_expected_scores)
        np.testing.assert_allclose(expected_scores, beam_search_scores, atol=1e-5)

    def test_beam_search_decode_force_prefix_eos(self):
        """Test that starting prefix with EOS changes what the next token is."""
        beam_size = 2
        # Use id 2 then 3 for batch example 0 and id 3 then 2 for example 1.
        batch_log_probs = np.array(
            [
                [-1e5, -1e6, -0.37, -1.17, -1e4, -1e4, -1e4, -1e4],
                [-1e5, -1e6, -1.17, -0.37, -1e4, -1e4, -1e4, -1e4],
            ],
            dtype=np.float32,
        )
        batch_log_probs_eos = np.array(
            [
                [-1e5, -0.37, -1e6, -1.17, -1e4, -1e4, -1e4, -1e4],
                [-1e5, -1e6, -1.17, -1e4, -0.37, -1e4, -1e4, -1e4],
            ],
            dtype=np.float32,
        )

        def token_to_scores(ids, cache):  # pylint: disable=unused-argument
            log_probs = np.repeat(np.expand_dims(batch_log_probs, axis=1), [beam_size], axis=1)
            log_probs = decoding.flatten_decoding_dim(log_probs)

            log_probs_eos = np.repeat(
                np.expand_dims(batch_log_probs_eos, axis=1), [beam_size], axis=1
            )
            log_probs_eos = decoding.flatten_decoding_dim(log_probs_eos)

            # If the id is an EOS id, pick log_prob_eos, else pick log_probs.
            log_probs = (ids == EOS_ID) * log_probs_eos + (ids != EOS_ID) + log_probs
            return log_probs, {}

        inputs_zero = np.zeros((2, 5), dtype=np.int32)
        inputs_eos_prefix = np.array([[EOS_ID, 0, 0, 0, 0], [EOS_ID, 0, 0, 0, 0]], dtype=np.int32)

        beam_search_output_zeros = decoding.beam_search_decode(
            inputs=inputs_zero,
            time_step=decoding.infer_initial_time_step(inputs_zero, pad_id=0),
            cache={},
            tokens_to_scores=token_to_scores,
            eos_id=EOS_ID,
            num_decodes=beam_size,
        )
        beam_search_output_eos_prefix = decoding.beam_search_decode(
            inputs=inputs_eos_prefix,
            time_step=decoding.infer_initial_time_step(inputs_eos_prefix, pad_id=0),
            cache={},
            tokens_to_scores=token_to_scores,
            eos_id=EOS_ID,
            num_decodes=beam_size,
        )
        self.assertTrue(
            jnp.any(beam_search_output_zeros.sequences != beam_search_output_eos_prefix.sequences)
        )
        self.assertTrue(
            jnp.any(
                beam_search_output_zeros.live_sequences
                != beam_search_output_eos_prefix.live_sequences
            )
        )

    @parameterized.parameters([0, 1])
    def test_beam_search_decode_no_prefix(self, pad_id: int):
        beam_size = 2

        def token_to_scores(ids, cache):  # pylint: disable=unused-argument
            # Use id 2 then 3 for batch element 0 and id 3 then 2 for element 1.
            log_probs = np.repeat(
                np.expand_dims(
                    np.array(
                        [[-1e5, -1e6, -0.37, -1.17], [-1e5, -1e6, -1.17, -0.37]], dtype=np.float32
                    ),
                    axis=1,
                ),
                [beam_size],
                axis=1,
            )
            log_probs = decoding.flatten_decoding_dim(log_probs)
            return log_probs, {}

        # No prefix is passed.
        inputs = np.full((2, 5), pad_id, dtype=np.int32)
        beam_search_output = decoding.beam_search_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            cache={},
            tokens_to_scores=token_to_scores,
            eos_id=EOS_ID,
            num_decodes=beam_size,
            pad_id=pad_id,
        )

        # Beam search scores are in the descending order.
        self.assertTrue(np.all(np.diff(beam_search_output.scores) <= 0))
        self.assertTrue(np.all(np.diff(beam_search_output.live_scores) <= 0))
        np.testing.assert_array_equal(
            beam_search_output.sequences,
            np.array(
                [
                    [[1, pad_id, pad_id, pad_id, pad_id], [2, 1, pad_id, pad_id, pad_id]],
                    [[1, pad_id, pad_id, pad_id, pad_id], [3, 1, pad_id, pad_id, pad_id]],
                ],
            ),
        )
        np.testing.assert_array_equal(
            beam_search_output.live_sequences,
            np.array([[[2, 2, 2, 2, 2], [2, 2, 2, 2, 3]], [[3, 3, 3, 3, 3], [3, 3, 3, 3, 2]]]),
        )

    @parameterized.product(prefix_length=[1, 3, 6, 7], low=[0, 1])
    def test_beam_init_with_inputs(self, prefix_length: int, low: int):
        batch_size = 2
        beam_size = 3
        max_decode_len = 6
        pad_id = 0

        # The user should be able to use teacher forcing on non-consecutive indices.
        # This is tested with low=0.
        inputs = jax.random.randint(
            jax.random.PRNGKey(123),
            shape=[batch_size, prefix_length],
            minval=low,
            maxval=10,
        )

        common_kwargs = dict(
            beam_size=beam_size,
            max_decode_len=max_decode_len,
            cache={},
            inputs=inputs,
            pad_id=pad_id,
        )

        if prefix_length > max_decode_len:
            with self.assertRaisesRegex(ValueError, "max_decode_len"):
                decoding._beam_init(
                    time_step=jnp.zeros([batch_size], dtype=inputs.dtype),
                    **common_kwargs,
                )
            return

        # Test that inputs and time_step must have the same batch_size dim.
        with self.assertRaisesRegex(ValueError, "time_step.shape"):
            decoding._beam_init(
                time_step=jnp.zeros([inputs.shape[0] + 1], dtype=inputs.dtype),
                **common_kwargs,
            )

        state = decoding._beam_init(
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            **common_kwargs,
        )
        # Starting index should be within valid range.
        self.assertTrue(jnp.all((0 <= state.cur_index) & (state.cur_index < max_decode_len)))
        # Starting index should always point to a non-pad token, unless all initial tokens are
        # padding.
        self.assertTrue(
            jnp.all(
                (
                    jnp.take_along_axis(state.live_seqs, state.cur_index[:, None, None], axis=-1)
                    != pad_id
                )
                | (jnp.all(state.live_seqs == pad_id, axis=-1, keepdims=True))
            )
        )
        # Everything afterwards should always point to a pad token.
        self.assertTrue(
            jnp.all(
                jnp.where(
                    jnp.arange(max_decode_len)[None, None, :] > state.cur_index[:, None, None],
                    state.live_seqs,
                    pad_id,
                )
                == pad_id
            )
        )
        np.testing.assert_array_equal(
            state.live_seqs[:, :, :prefix_length], np.tile(inputs[:, None, :], (1, beam_size, 1))
        )

    def test_compute_merge_matrix_by_prefixes(self):
        prefixes = jnp.asarray(
            [
                [
                    [1, 2, 3],
                    [1, 2, 4],
                    [1, 2, 3],
                    [1, 2, 3],
                ]
            ],
            dtype=jnp.int32,
        )
        np.testing.assert_array_equal(
            [
                [
                    # Prefix 0, 2, 3 will be merged into 0.
                    [1, 0, 1, 1],
                    # Prefix 1 will be merged into itself.
                    [0, 1, 0, 0],
                    # Nothing will be merged into prefix 2 and 3.
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ],
            decoding.compute_merge_matrix_by_prefix_ids(prefixes),
        )

    def test_merge_prefixes(self):
        merge_matrix = jnp.asarray(
            [
                [
                    # Prefix 0, 2, 3 will be merged into 0.
                    [1, 0, 1, 1],
                    # Prefix 1 will be merged into itself.
                    [0, 1, 0, 0],
                    # Nothing will be merged into prefix 2 and 3.
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ]
        )
        log_probs = jnp.log(jnp.asarray([[0.4, 0.3, 0.2, 0.1]]))
        np.testing.assert_allclose(
            jnp.asarray([[0.4 + 0.2 + 0.1, 0.3, 0.0, 0.0]]),
            jnp.exp(decoding._merge_prefixes(merge_matrix, log_probs=log_probs)),
        )

    def test_beam_search_with_path_merger(self):
        batch_size, num_decodes = 1, 4
        max_decode_len = 8
        vocab_size = 4
        num_expected_tokens = 5
        cache = dict(step=jnp.zeros([]))

        def tokens_to_scores(tokens, cache):
            del tokens
            step = cache["step"]
            has_enough_tokens = step >= num_expected_tokens
            eps = 1e-2 + 1e-3 * step
            continue_log_probs = jax.nn.log_softmax(
                # A mostly uniform distribution among tokens except for PAD(0) and EOS(1) with
                # slight preference towards lower ids.
                #
                # [NEG_INF, NEG_INF, 0, -eps, -2 * eps, ...].
                jnp.pad(
                    jnp.arange(0, vocab_size - 2, dtype=jnp.float32) * -eps,
                    ((2, 0)),
                    constant_values=NEG_INF,
                )
            )
            # Force EOS=1: [NEG_INF, 0, NEG_INF, NEG_INF, ...].
            finish_log_probs = (1 - jax.nn.one_hot(1, vocab_size)) * NEG_INF
            log_probs = (
                has_enough_tokens * finish_log_probs + (1 - has_enough_tokens) * continue_log_probs
            )
            return log_probs, dict(step=cache["step"] + 1)

        inputs = jnp.zeros([batch_size, max_decode_len], dtype=jnp.int32)
        kwargs = dict(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=0),
            cache=cache,
            tokens_to_scores=tokens_to_scores,
            eos_id=1,
            num_decodes=num_decodes,
        )
        outputs_without_merger = decoding.beam_search_decode(**kwargs)

        prefix_merger = _TokenSumPrefixMerger()
        outputs_with_merger = decoding.beam_search_decode(**kwargs, prefix_merger=prefix_merger)

        logging.info(
            "scores_without_merger=%s scores_with_merger=%s",
            outputs_without_merger.scores,
            outputs_with_merger.scores,
        )

        np.testing.assert_array_equal(
            jnp.asarray(
                [
                    [
                        # Without prefix merging, we prefer sequences with mostly 2's.
                        [2, 2, 2, 2, 2, 1, 0, 0],
                        [3, 2, 2, 2, 2, 1, 0, 0],
                        [2, 3, 2, 2, 2, 1, 0, 0],
                        [2, 2, 3, 2, 2, 1, 0, 0],
                    ]
                ]
            ),
            outputs_without_merger.sequences,
        )

        np.testing.assert_array_equal(
            jnp.asarray(
                [
                    [
                        # Two 3's and three 2's.
                        [3, 2, 2, 3, 2, 1, 0, 0],
                        # Three 3's and two 2's.
                        [3, 2, 2, 3, 3, 1, 0, 0],
                        # One 3 and four 2's.
                        [3, 2, 2, 2, 2, 1, 0, 0],
                        # This is a prefix merged and therefore has a score close to NEG_INF.
                        [3, 2, 2, 2, 3, 1, 0, 0],
                    ]
                ]
            ),
            outputs_with_merger.sequences,
        )
        # With prefix merging, the sequence has the combined scores from multiple sequences.
        self.assertLess(outputs_without_merger.scores[0, 0], outputs_with_merger.scores[0, 0])
        # The final sequence of outputs_with_merger has a score close to NEG_INF.
        self.assertLess(outputs_with_merger.scores[0, -1], NEG_INF * 0.5)

    @parameterized.product(
        [
            # All begin at index 0.
            dict(prompt_length=[1, 1, 1], num_decodes=5),
            # Each begins at a different index.
            dict(prompt_length=[5, 7, 128, 21, 38, 256], num_decodes=3),
        ],
        pad_id=[0, -1],
        prefix_merger=[None, _TokenSumPrefixMerger()],
        brevity_penalty=[None, decoding.brevity_penalty_fn(bp_type="hf", alpha=1.0)],
    )
    def test_beam_search_prefill(
        self,
        prompt_length: Sequence[int],
        num_decodes: int,
        pad_id: int,
        prefix_merger: Optional[decoding.PrefixMerger],
        brevity_penalty: Optional[decoding.BrevityPenaltyFn],
    ):
        """Tests beam search with prefilling matches decoding from scratch."""
        vocab_size = 1024
        decode_length = 256
        eos_id = vocab_size - 1
        prompt_length = jnp.minimum(jnp.asarray(prompt_length), decode_length)[:, None]
        batch_size = prompt_length.shape[0]

        # Construct prompt tokens.
        prompt_tokens = jax.random.randint(
            jax.random.PRNGKey(123),
            [batch_size, decode_length],
            minval=1,
            maxval=vocab_size - 1,
        )
        # Ensure pad_id and eos_id are not already in prompt_tokens.
        self.assertFalse(jnp.any(prompt_tokens == pad_id))
        self.assertFalse(jnp.any(prompt_tokens == eos_id))

        prompt_mask = jnp.arange(decode_length) < prompt_length
        prompt_tokens = prompt_tokens * prompt_mask + pad_id * (1 - prompt_mask)

        # Create a dummy mapping from prefix -> logits.
        dummy_emb = jax.random.uniform(jax.random.PRNGKey(123), [decode_length, vocab_size]) * -1

        def tokens_to_scores(tokens: Tensor, cache: NestedTensor) -> tuple[Tensor, NestedTensor]:
            # [batch, vocab]
            logits = jnp.einsum("ij,jk->ik", tokens + 1, dummy_emb)
            logits = logits.at[:, 0].set(NEG_INF)  # Don't emit padding.
            return logits, cache

        # [batch_size, decode_length].
        common_kwargs = dict(
            cache={},
            tokens_to_scores=tokens_to_scores,
            eos_id=eos_id,
            num_decodes=num_decodes,
            pad_id=pad_id,
            loop="lax",
            # Note: we can reuse the same prefix_merger instance as it's stateless.
            prefix_merger=prefix_merger,
            brevity_penalty=brevity_penalty,
        )

        # Decode once with prefilling (non-pad inputs and non-zero input_scores).
        prefill_output = decoding.beam_search_decode(
            inputs=prompt_tokens,
            time_step=decoding.infer_initial_time_step(prompt_tokens, pad_id=pad_id),
            **common_kwargs,
        )

        # Decode again *without* pre-filling, and ensure that outputs all match.
        from_scratch_output = decoding.beam_search_decode(
            inputs=prompt_tokens,
            time_step=jnp.zeros([batch_size], dtype=prompt_tokens.dtype),
            **common_kwargs,
        )

        # Check equivalence.
        # Note that live_{sequences,scores} are not expected to match when decoding from index 0 vs
        # decoding from the end of each prefix. This is because live_{sequences,scores} depends on
        # where cur_index is at: starting from 0 will always terminate at the same position across
        # the batch, but starting from the end of each prefix may end at different positions.
        # This should be consistent with the original T5X-based implementation.
        self.assertTrue(jnp.allclose(prefill_output.sequences, from_scratch_output.sequences))
        self.assertTrue(jnp.allclose(prefill_output.scores, from_scratch_output.scores))

    def test_stop_on_subsequence_raises_on_empty_sequence(self):
        msg = "Zero length stopping seqs are not supported. Zero length seqs at indices [1]."
        self.assertRaisesWithLiteralMatch(ValueError, msg, decoding.StopOnSubsequence, [[1], []])

    @parameterized.parameters(
        dict(
            # All in prompt, all zeros.
            stopper=decoding.StopOnSubsequence([[0]]),
            index=jnp.array(0),
            out_of_prompt=jnp.array([[False, False], [False, False]]),
            expected=jnp.array([[False, False], [False, False]]),
        ),
        dict(
            # All out of prompt.
            # Seqs (0, 0), (0, 1), (1, 0) are zero at index 0.
            stopper=decoding.StopOnSubsequence([[0]]),
            index=jnp.array(0),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[True, True], [True, False]]),
        ),
        dict(
            # Seq (1, 0) is zero at index 1, but is in prompt.
            stopper=decoding.StopOnSubsequence([[0]]),
            index=jnp.array(1),
            out_of_prompt=jnp.array([[True, True], [False, False]]),
            expected=jnp.array([[False, False], [False, False]]),
        ),
        dict(
            # Seq (1, 0) is zero at index 1, and is out of prompt.
            stopper=decoding.StopOnSubsequence([[0]]),
            index=jnp.array(1),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, False], [True, False]]),
        ),
        dict(
            # Seq (1, 1) is zero at index 3.
            stopper=decoding.StopOnSubsequence([[0]]),
            index=jnp.array(3),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, False], [False, True]]),
        ),
        dict(
            # Seq (0, 0), (1, 1) are zero at index 4.
            stopper=decoding.StopOnSubsequence([[0]]),
            index=jnp.array(4),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[True, False], [False, True]]),
        ),
        dict(
            # No matches.
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array(0),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, False], [False, False]]),
        ),
        dict(
            # No matches.
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array(1),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, False], [False, False]]),
        ),
        dict(
            # Seq (1, 1) matches [1, 2, 3] at index 2.
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array(2),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, False], [False, True]]),
        ),
        dict(
            # Seqs (0, 0) and (0, 1) match [1, 2, 3] at index 3.
            # Seq (1, 0) matches [3, 4].
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array(3),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[True, True], [True, False]]),
        ),
        dict(
            # Seq (0, 1) matches [3, 4] at index 4.
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array(4),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, True], [False, False]]),
        ),
        # Test when different batch elements are at different indices.
        dict(
            # Seqs (0, :) are at index 2, (1, :) are at 0.
            # No matches.
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array([2, 0]),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, False], [False, False]]),
        ),
        dict(
            # Seqs (0, :) are at index 3, (1, :) are at 1.
            # Seq (0, 0) and (0, 1) match [1, 2, 3].
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array([3, 1]),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[True, True], [False, False]]),
        ),
        dict(
            # Seqs (0, :) are at index 4, (1, :) are at 2.
            # Seq (0, 1) matches [3, 4].
            # Seq (1, 1) matches [1, 2, 3].
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array([4, 2]),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, True], [False, True]]),
        ),
        dict(
            # Seqs (0, :) are at index 4, (1, :) are at 3.
            # Seqs (0, 1) and (1, 0) match [3, 4].
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array([4, 3]),
            out_of_prompt=jnp.array([[True, True], [True, True]]),
            expected=jnp.array([[False, True], [True, False]]),
        ),
        dict(
            # Seqs (0, :) are at index 4, (1, :) are at 3.
            # Seqs (0, 1) and (1, 0) match [3, 4], but (1, 0) is in prompt.
            stopper=decoding.StopOnSubsequence([[1, 2, 3], [3, 4]]),
            index=jnp.array([4, 3]),
            out_of_prompt=jnp.array([[True, True], [False, False]]),
            expected=jnp.array([[False, True], [False, False]]),
        ),
    )
    def test_stop_on_subsequence(self, stopper, index, out_of_prompt, expected):
        sequences = jnp.array(
            [
                [
                    [0, 1, 2, 3, 0],
                    [0, 1, 2, 3, 4],
                ],
                [[0, 0, 3, 4, 5], [1, 2, 3, 0, 0]],
            ]
        )
        actual = stopper(index=index, sequences=sequences, out_of_prompt=out_of_prompt)
        self.assertTrue(jnp.all(expected == actual))

    def test_sample_decode_init(self):
        batch_size, num_decodes, max_decode_len = 2, 3, 7
        bos_id, pad_id = 1, -1
        cache = dict(a=jnp.array([1, 2, 3, 4]))
        common_kwargs = dict(
            num_decodes=num_decodes,
            max_decode_len=max_decode_len,
            cache=cache,
            prng_key=jax.random.PRNGKey(0),
            pad_id=pad_id,
        )

        # Test without token_scores.
        inputs = jnp.full((batch_size, max_decode_len), pad_id)
        inputs = inputs.at[:, 0].set(bos_id)
        state = decoding._decode_init(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            **common_kwargs,
        )
        self.assertTrue(jnp.all(state.cur_index == 0))
        self.assertTrue(jnp.all(state.token_scores == 0))
        self.assertTrue(jnp.all(state.sequences == inputs[:, None, :]))
        self.assertTrue(state.cache["a"].shape == (cache["a"].shape[0], num_decodes))

        # Test inputs length < max_decode_len.
        inputs = jnp.full((batch_size, max_decode_len // 2), bos_id)
        state = decoding._decode_init(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            **common_kwargs,
        )
        self.assertTrue(jnp.all(state.cur_index == (max_decode_len // 2 - 1)))
        self.assertTrue(jnp.all(state.token_scores == 0))
        self.assertTrue(jnp.all(state.sequences[:, :, : inputs.shape[1]] == inputs[:, None, :]))
        self.assertTrue(jnp.all(state.sequences[:, :, inputs.shape[1] :] == pad_id))

        # Test with time_step.shape[0] != inputs.shape[0].
        with self.assertRaisesRegex(ValueError, "time_step.shape"):
            decoding._decode_init(
                inputs=inputs,
                time_step=jnp.zeros([inputs.shape[0] + 1], dtype=inputs.dtype),
                **common_kwargs,
            )

        # Test with inputs length > max_decode_len.
        inputs = jnp.full((batch_size, max_decode_len + 1), bos_id)
        with self.assertRaisesRegex(ValueError, "Expected inputs.shape"):
            decoding._decode_init(
                inputs=inputs,
                time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
                **common_kwargs,
            )

        # Test with token_scores.
        inputs = jnp.full((batch_size, max_decode_len // 2), bos_id)
        token_scores = jax.random.uniform(
            jax.random.PRNGKey(234), (batch_size, max_decode_len // 2)
        )
        state = decoding._decode_init(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            token_scores=token_scores,
            **common_kwargs,
        )
        self.assertTrue(jnp.all(state.token_scores[:, :, 0] == 0))  # Dummy token scores are 0.
        self.assertTrue(
            jnp.allclose(
                state.token_scores[:, :, 1 : 1 + token_scores.shape[1]], token_scores[:, None, :]
            )
        )
        self.assertTrue(jnp.all(state.token_scores[:, :, 1 + token_scores.shape[1]] == 0))

        # Test with invalid token_scores shape.
        inputs = jnp.full((batch_size, max_decode_len // 2), bos_id)
        token_scores = jnp.ones((batch_size, max_decode_len))
        with self.assertRaisesRegex(ValueError, "Expected token_scores.shape"):
            decoding._decode_init(
                inputs=inputs,
                time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
                token_scores=token_scores,
                **common_kwargs,
            )

    @parameterized.product(
        [
            # All begin at index 0.
            dict(prompt_length=[1, 1, 1], num_decodes=5),
            # Each begins at a different index.
            dict(prompt_length=[3, 2, 1], num_decodes=3),
        ],
        pad_id=[0, -1],
        prefill_token_scores=[False, True],
    )
    def test_sample_decode_eos_stopping_condition(
        self,
        prompt_length: Sequence[int],
        num_decodes: int,
        pad_id: int,
        prefill_token_scores: bool,
    ):
        """Tests sample decoding with varying initial indices, using EOS as stopping condition."""
        vocab_size = 11
        decode_length = 7
        eos_id = vocab_size - 1
        prompt_length = jnp.minimum(jnp.asarray(prompt_length), decode_length)[:, None]
        initial_index = prompt_length - 1
        batch_size = prompt_length.shape[0]

        def tokens_to_scores(
            token_indices: Tensor, state_cache: NestedTensor
        ) -> tuple[Tensor, NestedTensor]:
            # Emit cur_iter + 1 unless decode index == cur_iter, in which case emit EOS.
            cur_iter = state_cache["cur_iter"]
            prompt_length = state_cache["prompt_length"]
            should_emit_eos = jnp.arange(token_indices.shape[0])[:, None] == (
                cur_iter - prompt_length + 1
            )
            tokens = (cur_iter + 1) * ~should_emit_eos + jnp.full_like(
                cur_iter, eos_id
            ) * should_emit_eos

            # Inside of prompt, want some log-probability mass on the whole vocabulary.
            # This will allow us to later tell prompted token scores apart from pad_id scores.
            inside_log_probs = jnp.full((cur_iter.shape[0], vocab_size), -10, dtype=jnp.float32)

            # Outside of prompt, want one-hot on token-to-emit.
            outside_log_probs = jnp.full(
                (cur_iter.shape[0], vocab_size), NEG_INF, dtype=jnp.float32
            )
            outside_log_probs += jnp.squeeze(jax.nn.one_hot(tokens, vocab_size)) * -0.99 * NEG_INF

            log_probs_for_tokens = jnp.where(
                cur_iter + 1 >= prompt_length, outside_log_probs, inside_log_probs
            )

            new_cache = state_cache.copy()
            new_cache["cur_iter"] = cur_iter + 1
            return log_probs_for_tokens, new_cache

        init_cache = dict(
            cur_iter=jnp.reshape(initial_index, (-1, 1)),
            prompt_length=prompt_length,
        )
        # [batch_size, decode_length].
        input_mask = jnp.arange(decode_length) < prompt_length

        # Test prefilling token scores up until prefix.
        if prefill_token_scores:
            init_scores = (
                jax.random.uniform(jax.random.PRNGKey(123), [batch_size, decode_length])
                * input_mask
            )
        else:
            init_scores = None

        inputs = input_mask * eos_id + (1 - input_mask) * pad_id
        sample_decoding_output = decoding.sample_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            cache=init_cache,
            tokens_to_scores=tokens_to_scores,
            stop_decoding_condition=decoding.StopOnSubsequence([[eos_id]]),
            pad_id=pad_id,
            num_decodes=num_decodes,
            prng_key=jax.random.PRNGKey(0),
            loop="lax",
            input_token_scores=init_scores,
        )
        # Check shapes.
        sequences = sample_decoding_output.sequences
        self.assertEqual(sequences.shape, (batch_size, num_decodes, decode_length))
        self.assertEqual(
            sample_decoding_output.token_scores.shape, (batch_size, num_decodes, decode_length)
        )

        # Check that EOS directly precedes padding for sequences that terminated within
        # decode_length. See `tokens_to_scores` above for when we emit EOS.
        flat_sequences = jnp.reshape(sequences, (-1, decode_length))
        tiled_prompt_length = jnp.repeat(prompt_length, repeats=num_decodes, axis=0)
        expected_eos_ix = tiled_prompt_length + jnp.arange(batch_size * num_decodes)[:, None] - 1
        self.assertTrue(
            jnp.all(
                (expected_eos_ix >= decode_length)
                | (jnp.take_along_axis(flat_sequences, expected_eos_ix, axis=-1) == eos_id)
            )
        )

        # Check that the sampled tokens are as expected.
        for ix, decode in enumerate(flat_sequences):
            # All tokens after emitted EOS are pad_id.
            self.assertTrue(jnp.all(decode[tiled_prompt_length[ix, 0] + ix :] == pad_id))
            # Everything before this is not the pad_id.
            self.assertTrue(jnp.all(decode[: tiled_prompt_length[ix, 0] + ix] != pad_id))
        # Check that the token scores are 0 for pad_id tokens.
        self.assertTrue(jnp.all(sample_decoding_output.token_scores[sequences == pad_id] == 0))
        # Check that the token scores are < 0 for non pad_id tokens starting from the initial index.
        mask = jnp.arange(decode_length) >= jnp.reshape(initial_index, (-1, 1, 1))
        mask = mask & (sequences != pad_id)
        self.assertTrue(jnp.all(sample_decoding_output.token_scores[mask] < 0))

        # If we prefilled token scores, check that scores match prior to the last the prefix token.
        # Note that the last prefix token is where we start decoding from.
        if prefill_token_scores:
            mask = jnp.arange(decode_length) < jnp.reshape(
                tiled_prompt_length - 1, (batch_size, num_decodes, 1)
            )
            self.assertTrue(
                jnp.all(
                    sample_decoding_output.token_scores * mask == init_scores[:, None, :] * mask
                )
            )

    @parameterized.product(
        [
            # All begin at index 0.
            dict(prompt_length=[1, 1, 1], num_decodes=5),
            # Each begins at a different index.
            dict(prompt_length=[3, 2, 1], num_decodes=3),
        ],
        pad_id=[0, -1],
    )
    def test_sample_decode_prefill(
        self,
        prompt_length: Sequence[int],
        num_decodes: int,
        pad_id: int,
    ):
        """Tests sample decoding with prefilling matches decoding from scratch."""
        vocab_size = 11
        decode_length = 7
        eos_id = vocab_size - 1
        prompt_length = jnp.minimum(jnp.asarray(prompt_length), decode_length)[:, None]
        initial_index = prompt_length - 1
        batch_size = prompt_length.shape[0]

        # Construct prompt tokens, tiled across num_decodes.
        prompt_tokens = jnp.tile(
            jax.random.randint(
                jax.random.PRNGKey(123),
                [batch_size, 1, decode_length],
                minval=1,
                maxval=vocab_size - 1,
            ),
            [1, num_decodes, 1],
        )
        prompt_mask = (jnp.arange(decode_length) < prompt_length)[:, None, :]
        # Ensure pad_id and eos_id are not already in prompt_tokens.
        # This is mainly because when decoding from scratch, padding is used to infer whether we're
        # out of prompt, whereas in the prefill case we treat intermediate padding as within prompt.
        self.assertFalse(jnp.any(prompt_tokens == pad_id))
        self.assertFalse(jnp.any(prompt_tokens == eos_id))

        # Construct reference tokens to emit in tokens_to_scores. Tokens that are part of the prompt
        # are tiled across num_decodes.
        ref_tokens = jax.random.randint(
            jax.random.PRNGKey(123),
            [batch_size, num_decodes, decode_length],
            minval=1,
            maxval=vocab_size - 1,
        )
        ref_tokens = prompt_tokens * prompt_mask + ref_tokens * (1 - prompt_mask)
        # Ensure pad_id and eos_id are not already in ref_tokens.
        self.assertFalse(jnp.any(ref_tokens == pad_id))
        self.assertFalse(jnp.any(ref_tokens == eos_id))

        def tokens_to_scores(
            token_indices: Tensor, state_cache: NestedTensor
        ) -> tuple[Tensor, NestedTensor]:
            cur_iter = state_cache["cur_iter"]
            prompt_length = state_cache["prompt_length"]

            # Emit ref_token unless decode index == cur_iter, in which case emit EOS.
            should_emit_eos = jnp.arange(token_indices.shape[0])[:, None] == (
                cur_iter - prompt_length + 1
            )
            ref_token = jnp.take_along_axis(
                jnp.reshape(ref_tokens, (-1, decode_length)), cur_iter + 1, axis=-1, mode="clip"
            )
            eos_token = jnp.full_like(cur_iter, eos_id)
            tokens = ref_token * ~should_emit_eos + eos_token * should_emit_eos

            # Produce one-hot on token-to-emit.
            log_probs = jnp.full((cur_iter.shape[0], vocab_size), NEG_INF, dtype=jnp.float32)
            log_probs += jnp.squeeze(jax.nn.one_hot(tokens, vocab_size)) * -0.99 * NEG_INF

            new_cache = state_cache.copy()
            new_cache["cur_iter"] = cur_iter + 1
            return log_probs, new_cache

        # [batch_size, decode_length].
        common_kwargs = dict(
            tokens_to_scores=tokens_to_scores,
            stop_decoding_condition=decoding.StopOnSubsequence([[eos_id]]),
            pad_id=pad_id,
            num_decodes=num_decodes,
            prng_key=jax.random.PRNGKey(0),
            loop="lax",
        )

        # Decode once with prefilling (non-pad inputs and non-zero input_scores).
        inputs = prompt_tokens * prompt_mask + pad_id * (1 - prompt_mask)
        # Since prompt is tiled across num_decodes, we pick first decode.
        inputs = inputs[:, 0, :]
        sample_decoding_output = decoding.sample_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            cache=dict(
                cur_iter=jnp.reshape(initial_index, (-1, 1)),
                prompt_length=prompt_length,
            ),
            # Emitted token scores are always fixed to 0.01 * NEG_INF; see tokens_to_scores.
            input_token_scores=(
                jnp.full((batch_size, decode_length), 0.01 * NEG_INF) * prompt_mask[:, 0, :]
            ),
            **common_kwargs,
        )

        # Decode again *without* pre-filling, and ensure that outputs all match.
        inputs = jnp.full_like(inputs, pad_id)
        inputs = inputs.at[:, 0].set(eos_id)
        from_scratch_decoding_output = decoding.sample_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=pad_id),
            cache=dict(
                cur_iter=jnp.zeros((batch_size, 1), dtype=jnp.int32),
                prompt_length=prompt_length,
            ),
            input_token_scores=None,
            **common_kwargs,
        )
        # Check equivalence.
        self.assertTrue(
            jnp.allclose(sample_decoding_output.sequences, from_scratch_decoding_output.sequences)
        )
        self.assertTrue(
            jnp.allclose(
                sample_decoding_output.token_scores, from_scratch_decoding_output.token_scores
            )
        )

    @parameterized.parameters(
        # pylint: disable=line-too-long
        # All begin at index 0. Batch size is 2 and num decodes is 3.
        dict(
            fake_decodes=[
                ["Contains subsequence one", "subsequence two at start", "neither"],
                ["subsequence one at start", "neither", "Contains subsequence two"],
            ],
            prompts=[["prompt"] * 3, ["prompt"] * 3],
            # fmt: off
            expected=[
                [
                    ["▁", "Contains", "▁sub", "s", "e", "que", "nce", "▁one", "<pad>"],
                    ["▁sub", "s", "e", "que", "nce", "▁two", "<pad>", "<pad>", "<pad>"],
                    ["▁neither", "</s>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"],
                ],
                [
                    ["▁sub", "s", "e", "que", "nce", "▁one", "<pad>", "<pad>", "<pad>"],
                    ["▁neither", "</s>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"],
                    ["▁", "Contains", "▁sub", "s", "e", "que", "nce", "▁two", "<pad>"],
                ],
            ],
            # fmt: on
        ),
        # Each example begins at a different index (due to different prompt lengths).
        # Batch size is 2 and num decodes is 3.
        dict(
            fake_decodes=[
                ["Contains subsequence one", "subsequence two at start", "neither"],
                ["subsequence one at start", "neither", "Contains subsequence two"],
            ],
            prompts=[["prompt hello"] * 3, ["prompt hello again"] * 3],
            # fmt: off
            expected=[
                [
                    ["▁hello", "▁", "Contains", "▁sub", "s", "e", "que", "nce", "▁one", "<pad>", "<pad>"],
                    ["▁hello", "▁sub", "s", "e", "que", "nce", "▁two", "<pad>", "<pad>", "<pad>", "<pad>"],
                    ["▁hello", "▁neither", "</s>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"],
                ],
                [
                    ["▁hello", "▁again", "▁sub", "s", "e", "que", "nce", "▁one", "<pad>", "<pad>", "<pad>"],
                    ["▁hello", "▁again", "▁neither", "</s>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"],
                    ["▁hello", "▁again", "▁", "Contains", "▁sub", "s", "e", "que", "nce", "▁two", "<pad>"],
                ],
            ],
            # fmt: on
        ),
        # Test a case where we decode without early stopping, letting some batch elements decode
        # longer than others (due to smaller starting index).
        # Batch size is 2 and num decodes is 2.
        dict(
            fake_decodes=[
                ["short", "also short"],
                ["no early stop longer sequence", "no early stop longer sequence"],
            ],
            prompts=[["prompt"] * 2, ["prompt hello second"] * 2],
            # fmt: off
            expected=[
                [
                    # The shorter sequences terminate early.
                    ["▁short", "</s>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"],
                    ["▁also", "▁short", "</s>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"],
                ],
                [
                    # The longer sequences continue to decode to completion.
                    ["▁hello", "▁second", "▁no", "▁early", "▁stop", "▁longer", "▁sequence", "</s>"],
                    ["▁hello", "▁second", "▁no", "▁early", "▁stop", "▁longer", "▁sequence", "</s>"],
                ],
            ],
            # fmt: on
        ),
        # pylint: enable=line-too-long
    )
    @pytest.mark.skipif(not os.path.exists(_T5_VOCAB_FILE), reason="Missing testdata.")
    def test_sample_decode_with_complex_stopping_condition(
        self,
        fake_decodes: Sequence[Sequence[str]],
        prompts: Sequence[Sequence[str]],
        expected: Sequence[Sequence[str]],
    ):
        vocab = seqio.SentencePieceVocabulary(_T5_VOCAB_FILE)
        batch_size = len(fake_decodes)
        num_decodes = len(fake_decodes[0])

        # Tokenize the test cases.
        # Ragged of shape [batch_size, num_decodes, None].
        ragged_prompts = vocab.encode_tf(prompts)
        ragged_decodes = vocab.encode_tf(fake_decodes)
        eos_ids = tf.fill([batch_size, num_decodes, 1], vocab.eos_id)

        # Construct the fake decodes.
        # [batch_size, num_decodes, max_decode_len].
        ragged_tokens = tf.concat([ragged_prompts, ragged_decodes, eos_ids], -1)
        faked_tokens = ragged_tokens.to_tensor().numpy()

        vocab_size = vocab.vocab_size
        # [batch, 1].
        prompt_length = ragged_prompts.nested_row_lengths()[-1].numpy()[::num_decodes, None]
        max_prompt_length = prompt_length.max()
        # Subtract 1 since we drop the conditioning token.
        max_decode_length = int(faked_tokens.shape[-1]) - max_prompt_length - 1

        def tokens_to_scores(
            token_indices: Tensor, state_cache: NestedTensor  # pylint: disable=unused-argument
        ) -> tuple[Tensor, NestedTensor]:
            # [batch_size * num_decodes, 1].
            cur_iter = state_cache["cur_iter"]
            prompt_length = state_cache["prompt_length"]
            next_iter = jnp.minimum(cur_iter + 1, faked_tokens.shape[-1] - 1)
            # [batch_size * num_decodes, 1].
            tokens = jnp.take_along_axis(
                jnp.reshape(faked_tokens, (-1, faked_tokens.shape[-1])), next_iter, axis=-1
            )

            # Inside of prompt, want some log-probability mass on the whole vocabulary.
            # This will allow us to later tell prompted token scores apart from pad_id scores.
            inside_log_probs = jnp.full((cur_iter.shape[0], vocab_size), -10, dtype=jnp.float32)

            # Outside of prompt, want one-hot on token-to-emit.
            outside_log_probs = jnp.full(
                (cur_iter.shape[0], vocab_size), NEG_INF, dtype=jnp.float32
            )
            outside_log_probs += jnp.squeeze(jax.nn.one_hot(tokens, vocab_size)) * -0.99 * NEG_INF

            log_probs_for_tokens = jnp.where(
                cur_iter + 1 >= prompt_length, outside_log_probs, inside_log_probs
            )

            new_cache = state_cache.copy()
            new_cache["cur_iter"] = cur_iter + 1
            return log_probs_for_tokens, new_cache

        # Each batch element starts at a different index.
        initial_index = prompt_length - 1
        init_cache = dict(
            cur_iter=jnp.reshape(initial_index, (-1, 1)),
            prompt_length=prompt_length,
        )
        inputs = ragged_prompts.to_tensor(
            shape=[batch_size, None, max_decode_length + max_prompt_length],
            default_value=vocab.pad_id,
        )
        # Select the first prompt of each batch elem.
        inputs = inputs[:, 0, :].numpy()
        sample_decoding_output = decoding.sample_decode(
            inputs=inputs,
            time_step=decoding.infer_initial_time_step(inputs, pad_id=vocab.pad_id),
            cache=init_cache,
            tokens_to_scores=tokens_to_scores,
            stop_decoding_condition=decoding.StopOnSubsequence(
                [
                    [vocab.eos_id],
                    vocab.encode("subsequence one"),
                    vocab.encode("subsequence two"),
                ]
            ),
            pad_id=vocab.pad_id,
            num_decodes=num_decodes,
            prng_key=jax.random.PRNGKey(0),
            loop="lax",
        )
        # Check shapes.
        sequences = sample_decoding_output.sequences
        token_scores = sample_decoding_output.token_scores
        self.assertEqual(
            sequences.shape, (batch_size, num_decodes, max_decode_length + max_prompt_length)
        )
        self.assertEqual(
            token_scores.shape, (batch_size, num_decodes, max_decode_length + max_prompt_length)
        )

        # Compare against expected.
        target = jnp.asarray(jax.tree_map(vocab.tokenizer.piece_to_id, expected))
        self.assertTrue(jnp.all(sequences == target))

        # Check that the token scores are 0 for pad_id tokens.
        self.assertTrue(jnp.all(token_scores[sequences == vocab.pad_id] == 0))
        # Check that the token scores are < 0 for non pad_id tokens beyond the initial index.
        # First mask out the token scores before the initial index.
        token_scores = jnp.where(
            (jnp.arange(token_scores.shape[-1]) < jnp.reshape(initial_index, (-1, 1, 1))),
            NEG_INF,
            token_scores,
        )
        # Compare the rest of the token scores.
        self.assertTrue(jnp.all(token_scores[sequences != vocab.pad_id] < 0))

    @parameterized.parameters(
        dict(
            prefix=jnp.array(
                [
                    [1, 2, 0],
                    [0, 1, 0],
                    [0, 1, 2],
                    [0, 0, 0],
                    [1, 2, 3],
                ]
            ),
            pad_id=0,
            expected=jnp.array([1, 1, 2, 0, 2]),
        ),
        dict(
            prefix=jnp.array(
                [
                    [1, 2, 4],
                    [4, 1, 4],
                    [4, 1, 2],
                    [4, 4, 4],
                    [1, 2, 3],
                ]
            ),
            pad_id=4,
            expected=jnp.array([1, 1, 2, 0, 2]),
        ),
    )
    def test_infer_initial_time_step(self, prefix: Tensor, pad_id: int, expected: Tensor):
        self.assertTrue(
            jnp.all(expected == decoding.infer_initial_time_step(prefix, pad_id=pad_id))
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

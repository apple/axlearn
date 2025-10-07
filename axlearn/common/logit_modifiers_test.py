# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/t5x:
# Copyright 2022 The T5X Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests logit modifiers."""
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common import decoding, utils
from axlearn.common.logit_modifiers import (
    _float32_binary_search,
    _int32_binary_search,
    chain,
    scale_by,
    top_k_logits,
    top_p_logits,
)
from axlearn.common.test_utils import TestCase


# pylint: disable=no-self-use
class TestLogitsTransforms(TestCase):
    """Tests logit modifiers."""

    @parameterized.parameters(0.01, 0.1, 1.0, 10.0)
    def test_scale_by_modifier(self, temperature: float):
        batch_size = 2
        vocab_size = 7
        logits = -jax.random.uniform(
            jax.random.PRNGKey(0), shape=(batch_size, vocab_size), dtype=jnp.float32
        )
        scaled_by_logits = scale_by(temperature)(logits)
        self.assertNestedAllClose(scaled_by_logits, logits / temperature)

    @parameterized.parameters(1e-6, 1e-8, 0.0)
    def test_scale_by_modifier_with_low_temperature(self, temperature: float):
        # For low temperature we expect to raise.
        with self.assertRaises(ValueError):
            scale_by(temperature, min_temperature=1e-4)

    @parameterized.parameters(1e-4, 0.1, 0.5, 0.99, 1.0)
    def test_top_p_modifier(self, p: float):
        batch_size = 3
        vocab_size = 13
        logits = -jax.random.uniform(
            jax.random.PRNGKey(1), shape=(batch_size, vocab_size), dtype=jnp.float32
        )
        top_p_modified_logits = top_p_logits(p)(logits)
        for eg, modified_eg in zip(logits, top_p_modified_logits):
            probs = jax.nn.softmax(eg, axis=-1)
            sorted_p, sorted_ix = jax.lax.top_k(probs, k=len(probs))
            cumulative_prob = 0
            for prob, ix in zip(sorted_p, sorted_ix):
                if cumulative_prob >= p:
                    # These were outside of the top-p mass, should be neg inf.
                    self.assertAlmostEqual(modified_eg[ix], decoding.NEG_INF)
                else:
                    cumulative_prob += prob
                    # Should be including this value.
                    self.assertAlmostEqual(modified_eg[ix], eg[ix], delta=1e-6)

    def test_top_p_modifier_batched_p(self):
        """Test top_p_modifier with a tensor p."""
        batch_size = 3
        vocab_size = 13
        p = jnp.array([1e-4, 0.1, 0.9])
        logits = -jax.random.uniform(
            jax.random.PRNGKey(1), shape=(batch_size, vocab_size), dtype=jnp.float32
        )
        top_p_modified_logits = top_p_logits(p)(logits)
        for idx, (eg, modified_eg) in enumerate(zip(logits, top_p_modified_logits)):
            probs = jax.nn.softmax(eg, axis=-1)
            sorted_p, sorted_ix = jax.lax.top_k(probs, k=len(probs))
            cumulative_prob = 0
            for prob, ix in zip(sorted_p, sorted_ix):
                if cumulative_prob >= p[idx]:
                    # These were outside of the top-p mass, should be neg inf.
                    self.assertAlmostEqual(modified_eg[ix], decoding.NEG_INF)
                else:
                    cumulative_prob += prob
                    # Should be including this value.
                    self.assertAlmostEqual(modified_eg[ix], eg[ix], delta=1e-6)

    @parameterized.parameters(1, 10, 17)
    def test_top_k_modifier(self, k: int):
        batch_size = 5
        vocab_size = 17
        logits = -jax.random.uniform(
            jax.random.PRNGKey(2), shape=(batch_size, vocab_size), dtype=jnp.float32
        )
        top_k_modified_logits = top_k_logits(k)(logits)
        for eg, modified_eg in zip(logits, top_k_modified_logits):
            _, sorted_ix = jax.lax.top_k(eg, k=len(eg))
            num_seen = 0
            for ix in sorted_ix:
                if num_seen < k:
                    # Should be including this value, haven't seen k items yet.
                    self.assertAlmostEqual(modified_eg[ix], eg[ix], delta=1e-6)
                    num_seen += 1
                else:
                    # These were outside of the top-k, should be neg inf.
                    self.assertAlmostEqual(modified_eg[ix], decoding.NEG_INF)

    def test_top_k_modifier_with_ties(self):
        batch_size = 2
        vocab_size = 1024
        logits = jnp.concatenate(
            (
                jnp.full((batch_size, 1), -jnp.pi),
                jnp.full((batch_size, vocab_size // 2 - 2), -2 * jnp.pi),
                jnp.full((batch_size, 1), -10.1),
            ),
            axis=-1,
        )
        top_k_modified_logits = top_k_logits(2)(logits)
        # This behavior is consistent with the T5X implementation of topk:
        # <https://github.com/google-research/t5x/blob/1f8cec78b/t5x/binary_search.py#L164-L224>
        # Check that we returned all values for the -pi and -2*pi logits.
        self.assertNestedAllClose(
            top_k_modified_logits[:, : vocab_size // 2 - 1], logits[:, : vocab_size // 2 - 1]
        )
        # Check that the rest of the array is neg inf.
        self.assertNestedAllClose(
            top_k_modified_logits[:, -1:],
            jnp.full((batch_size, 1), decoding.NEG_INF),
        )

    @parameterized.product(
        batch_size=[2, 8],
        vocab_size=[512, 1024],
        break_ties=["all", "smallest_index"],
    )
    def test_top_k_modifier_break_ties(self, batch_size, vocab_size, break_ties):
        logits = jnp.concatenate(
            (
                jnp.full((batch_size, vocab_size - 1), -2 * jnp.pi),
                jnp.full((batch_size, 1), -10.1),
            ),
            axis=-1,
        )
        top_k_modified_logits = top_k_logits(1, break_ties=break_ties)(logits)
        num_returned_logits = vocab_size - 1 if break_ties == "all" else 1
        self.assertNestedAllClose(
            top_k_modified_logits[:, :num_returned_logits], logits[:, :num_returned_logits]
        )
        # Check that the rest of the array is neg inf.
        self.assertNestedAllClose(
            top_k_modified_logits[:, num_returned_logits:],
            jnp.full((batch_size, vocab_size - num_returned_logits), decoding.NEG_INF),
        )

    @parameterized.parameters(1e-4, 0.1, 0.5, 0.99)
    def test_top_p_modifier_with_ties(self, p: float):
        batch_size = 3
        vocab_size = 512
        logits = jax.nn.log_softmax(jnp.full((batch_size, vocab_size), -1), axis=-1)
        top_p_modified_logits = top_p_logits(p)(logits)
        # All logits were tied, so we return the entire set.
        self.assertNestedAllClose(top_p_modified_logits, logits)

    def test_chain(self):
        batch_size = 13
        vocab_size = 7
        temperature = 0.1
        logits = -jax.random.uniform(
            jax.random.PRNGKey(9), shape=(batch_size, vocab_size), dtype=jnp.float32
        )
        # Apply scale by and then top-1 manually.
        scaled_logits = logits / temperature
        top_one_ix = jnp.argmax(scaled_logits, axis=-1)
        top_one_mask = jax.nn.one_hot(top_one_ix, scaled_logits.shape[-1], dtype=bool)
        expected_chain_modified_logits = jnp.where(top_one_mask, scaled_logits, decoding.NEG_INF)
        # Apply scale by and then top-1 via chain function.
        chain_modified_logits = chain(scale_by(temperature), top_k_logits(1))(logits)
        self.assertNestedAllClose(chain_modified_logits, expected_chain_modified_logits)


class BinarySearchTest(TestCase):
    INT32_MIN = jnp.iinfo(jnp.int32).min
    INT32_MAX = jnp.iinfo(jnp.int32).max

    def test_int32_binary_search(self):
        expected_solution = jnp.asarray(
            [
                1,
                42,
                72,
                2043,
                0,
                2044,
                self.INT32_MIN,
                self.INT32_MIN + 1,
                self.INT32_MAX,
                self.INT32_MAX - 1,
            ],
            dtype=jnp.int32,
        )

        def predicate(x):
            return x > expected_solution

        solution = _int32_binary_search(expected_solution.shape, predicate=predicate)
        self.assertSequenceEqual(solution.tolist(), expected_solution.tolist())

    def test_int32_binary_search_extreme_predicates(self):
        def always_false(x):
            return jnp.full_like(x, False)

        self.assertSequenceEqual(
            [self.INT32_MAX], _int32_binary_search((1,), predicate=always_false).tolist()
        )

        def always_true(x):
            return jnp.full_like(x, True)

        self.assertSequenceEqual(
            [self.INT32_MIN], _int32_binary_search((1,), predicate=always_true).tolist()
        )

    def test_float32_binary_search(self):
        expected_solution = jnp.asarray([1.23, 0.0, -0.0, 105.4, -1024, 4.3], dtype=jnp.float32)

        def predicate(x):
            return x > expected_solution

        solution = _float32_binary_search(expected_solution.shape, predicate=predicate)
        # Ensure that we use the same underlying implementation (i.e. whatever JAX is doing).
        self.assertTrue(
            jnp.all(solution == expected_solution), f"a={solution}, c={expected_solution}"
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

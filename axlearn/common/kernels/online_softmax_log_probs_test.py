# Copyright © 2025 Apple Inc.

"""Tests for online_softmax_log_probs Pallas kernel (uses interpret mode for CPU)."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.kernels.online_softmax_log_probs import online_softmax_log_probs_pallas
from axlearn.common.test_utils import assert_allclose


class TestOnlineSoftmaxLogProbsPallas(parameterized.TestCase):
    """Tests the Pallas kernel via online_softmax_log_probs_pallas(interpret=True)."""

    @parameterized.parameters(
        dict(vocab_size=24, tile_v=24),
        dict(vocab_size=24, tile_v=8),
        dict(vocab_size=100, tile_v=16),
        dict(vocab_size=100, tile_v=32),
    )
    def test_matches_naive(self, vocab_size: int, tile_v: int):
        """Pallas kernel log-normalizer matches full matmul + log_softmax."""
        batch_size, seq_len, hidden_dim = 2, 128, 8
        rng = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(rng, 3)

        x = jax.random.normal(k1, (batch_size, seq_len, hidden_dim))
        weight = jax.random.normal(k2, (vocab_size, hidden_dim))
        target_ids = jax.random.randint(k3, (batch_size, seq_len), minval=0, maxval=vocab_size)

        # Naive reference.
        full_logits = jnp.einsum("bsh,vh->bsv", x, weight)
        log_normalizer_ref = jax.scipy.special.logsumexp(full_logits, axis=-1)
        full_log_probs = jax.nn.log_softmax(full_logits, axis=-1)
        expected_target = jnp.take_along_axis(
            full_log_probs, target_ids[:, :, None], axis=-1
        ).squeeze(-1)

        log_normalizer = online_softmax_log_probs_pallas(
            x,
            weight,
            top_k=0,
            tile_s=128,
            tile_v=tile_v,
            interpret=True,
        )

        assert_allclose(log_normalizer, log_normalizer_ref, atol=1e-5, rtol=1e-5)

        # Also verify target_log_probs = target_logit - log_normalizer.
        safe_ids = jnp.clip(target_ids, 0, vocab_size - 1)
        target_logit = jnp.sum(x * weight[safe_ids], axis=-1)
        target_log_probs = target_logit - log_normalizer
        assert_allclose(target_log_probs, expected_target, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        dict(top_k=1, tile_v=24),
        dict(top_k=3, tile_v=8),
        dict(top_k=5, tile_v=16),
    )
    def test_top_k_matches_naive(self, top_k: int, tile_v: int):
        """Pallas top-k matches full log_softmax + jax.lax.top_k."""
        vocab_size, hidden_dim = 24, 8
        batch_size, seq_len = 2, 128
        rng = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(rng, 2)

        x = jax.random.normal(k1, (batch_size, seq_len, hidden_dim))
        weight = jax.random.normal(k2, (vocab_size, hidden_dim))

        # Naive reference.
        full_logits = jnp.einsum("bsh,vh->bsv", x, weight)
        full_log_probs = jax.nn.log_softmax(full_logits, axis=-1)
        expected_topk_vals, _ = jax.lax.top_k(full_log_probs, top_k)

        log_normalizer, topk_logits, _ = online_softmax_log_probs_pallas(
            x,
            weight,
            top_k=top_k,
            tile_s=128,
            tile_v=tile_v,
            interpret=True,
        )

        actual_topk_log_probs = topk_logits - log_normalizer[:, :, jnp.newaxis]
        assert_allclose(actual_topk_log_probs, expected_topk_vals, atol=1e-5, rtol=1e-5)

    def test_matches_naive_with_different_tile_v(self):
        """Two different tile_v values produce the same results."""
        vocab_size, hidden_dim = 100, 8
        batch_size, seq_len = 2, 128
        top_k = 3
        rng = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(rng, 2)

        x = jax.random.normal(k1, (batch_size, seq_len, hidden_dim))
        weight = jax.random.normal(k2, (vocab_size, hidden_dim))

        log_norm_a, topk_logits_a, _ = online_softmax_log_probs_pallas(
            x,
            weight,
            top_k=top_k,
            tile_s=128,
            tile_v=16,
            interpret=True,
        )
        log_norm_b, topk_logits_b, _ = online_softmax_log_probs_pallas(
            x,
            weight,
            top_k=top_k,
            tile_s=128,
            tile_v=32,
            interpret=True,
        )

        assert_allclose(log_norm_a, log_norm_b, atol=1e-5, rtol=1e-5)
        topk_lp_a = topk_logits_a - log_norm_a[:, :, jnp.newaxis]
        topk_lp_b = topk_logits_b - log_norm_b[:, :, jnp.newaxis]
        assert_allclose(topk_lp_a, topk_lp_b, atol=1e-5, rtol=1e-5)

    def test_seq_padding(self):
        """Handles S not divisible by tile_s."""
        vocab_size, hidden_dim = 24, 8
        batch_size, seq_len = 2, 100  # 100 not divisible by 128
        rng = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(rng, 2)

        x = jax.random.normal(k1, (batch_size, seq_len, hidden_dim))
        weight = jax.random.normal(k2, (vocab_size, hidden_dim))

        full_logits = jnp.einsum("bsh,vh->bsv", x, weight)
        expected_log_norm = jax.scipy.special.logsumexp(full_logits, axis=-1)

        actual_log_norm = online_softmax_log_probs_pallas(
            x,
            weight,
            top_k=0,
            tile_s=128,
            tile_v=8,
            interpret=True,
        )
        self.assertEqual(actual_log_norm.shape, (batch_size, seq_len))
        assert_allclose(actual_log_norm, expected_log_norm, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()

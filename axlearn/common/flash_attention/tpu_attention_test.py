# Copyright © 2023 Apple Inc.

"""Tests TPU FlashAttention kernels."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common.flash_attention.tpu_attention import flash_attention
from axlearn.common.flash_attention.utils import mha_reference
from axlearn.common.test_utils import TestCase

if jax.default_backend() != "tpu":
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer."""

    _TEST_CONFIGS = [
        dict(
            batch_size=2,
            seq_len=384,
            num_heads=4,
            per_head_dim=32,
        ),
        dict(
            batch_size=2,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
        ),
        dict(
            batch_size=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
        ),
    ]

    @parameterized.product(
        _TEST_CONFIGS,
        causal=[False, True],
        with_attention_bias=[False, True],
        with_segment_ids=[False, True],
    )
    def test_forward_and_backward(
        self,
        batch_size,
        seq_len,
        num_heads,
        per_head_dim,
        causal,
        with_attention_bias,
        with_segment_ids,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
        q = jax.random.normal(
            k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16
        )
        k = jax.random.normal(
            k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16
        )
        v = jax.random.normal(
            k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16
        )
        attention_bias = None
        if with_attention_bias:
            attention_bias = jax.random.normal(
                k4, (batch_size, num_heads, seq_len, seq_len), dtype=jnp.bfloat16
            )
        segment_ids = None
        if with_segment_ids:
            segment_ids = jax.random.bernoulli(k5, shape=(batch_size, seq_len)).astype(jnp.int32)
            segment_ids = jnp.cumsum(segment_ids, axis=1)

        softmax_scale = q.shape[-1] ** -0.5

        def ref_fn(q, k, v, bias, ids):
            return mha_reference(q, k, v, bias, ids, causal=causal, softmax_scale=softmax_scale)

        def fn(q, k, v, bias, ids):
            return flash_attention(q, k, v, bias, ids, causal=causal, softmax_scale=softmax_scale)

        # Compare outputs.
        out = fn(q, k, v, attention_bias, segment_ids)
        ref_out = ref_fn(q, k, v, attention_bias, segment_ids)
        self.assertNestedAllClose(out, ref_out, atol=0.05)

        # Compare grads.
        grad_out = jax.grad(lambda q, k, v, b, s: fn(q, k, v, b, s).mean(), argnums=(0, 1, 2))(
            q, k, v, attention_bias, segment_ids
        )
        ref_grad_out = jax.grad(
            lambda q, k, v, b, s: ref_fn(q, k, v, b, s).mean(), argnums=(0, 1, 2)
        )(q, k, v, attention_bias, segment_ids)
        self.assertNestedAllClose(grad_out, ref_grad_out, atol=0.05)

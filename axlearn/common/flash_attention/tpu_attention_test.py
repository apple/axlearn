# Copyright © 2023 Apple Inc.

"""Tests TPU FlashAttention kernels."""
from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common.attention import causal_mask
from axlearn.common.flash_attention import tpu_attention
from axlearn.common.flash_attention.tpu_attention import tpu_flash_attention
from axlearn.common.flash_attention.utils import mha_reference
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Tensor

if jax.default_backend() != "tpu":
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


def jax_fn_mask(query_position: Tensor, key_position: Tensor) -> Tensor:
    """A MaskFn that calls jax.

    The mask is the same as `causal_mask`.

    However, this implementation requires specially handling to use with
    SplashAttention since `tpu_flash-attention()` needs to wrap this function
    to return numpy values if the input is numpy. (Otherwise we get tracer errors in jit.)
    """
    return jnp.greater_equal(query_position, key_position)


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer."""

    _TEST_CONFIGS = [
        dict(
            batch_size=2,
            seq_len=384,
            num_heads=4,
        ),
        dict(
            batch_size=8,
            seq_len=2048,
            num_heads=4,
        ),
    ]

    @parameterized.product(
        _TEST_CONFIGS,
        mask=[None, causal_mask, jax_fn_mask],
        attention_bias_type=[None, "2d", "4d"],
        with_segment_ids=[False, True],
        per_head_dim=[32, 64, 128, 256],
    )
    def test_forward_and_backward(
        self,
        batch_size,
        seq_len,
        num_heads,
        per_head_dim,
        mask,
        attention_bias_type,
        with_segment_ids,
    ):
        # pylint: disable=protected-access
        causal = mask in [causal_mask, jax_fn_mask]

        fallback_to_legacy = (
            per_head_dim % 128 != 0 or (attention_bias_type is not None) or with_segment_ids
        )

        if fallback_to_legacy and mask is jax_fn_mask:
            pytest.skip("Custom masks are not supported by legacy attention.")

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
        if attention_bias_type == "2d":
            attention_bias = jax.random.normal(k4, (1, 1, seq_len, seq_len), dtype=jnp.bfloat16)
        elif attention_bias_type == "4d":
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

        legacy_flash_wrapper = unittest.mock.Mock(wraps=tpu_attention._legacy_tpu_flash_attention)

        def fn(q, k, v, bias, ids):
            record_legacy_call = unittest.mock.patch.object(
                tpu_attention, "_legacy_tpu_flash_attention", legacy_flash_wrapper
            )
            with record_legacy_call:
                return tpu_flash_attention(
                    q, k, v, bias, ids, mask=mask, softmax_scale=softmax_scale
                )

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

        # Check splash attention is was used when it should be.
        if fallback_to_legacy:
            legacy_flash_wrapper.assert_called()
        else:
            legacy_flash_wrapper.assert_not_called()

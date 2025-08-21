# Copyright © 2025 Apple Inc.

"""Tests LinearAttention Pallas kernels."""
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common.rattention.kernels.linear_attention_kernels import (
    residual_linear_attention,
    residual_linear_attention_linear_scan,
    residual_linear_attention_w_timestep,
)
from axlearn.common.rattention.kernels.utils import FeatureMap
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import Tensor

if jax.default_backend() != "tpu":
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


def _generate_test_inputs(shape: tuple, dtype: jnp.dtype, seed: int) -> tuple[Tensor, ...]:
    """
    Args:
        shape: [batch_size, num_heads, num_kv_heads, seq_len]
        dtype: float32, bfloat16
        seed: random seed

    Returns:
        q: [batch_size, num_heads, seq_len, dk]
        k: [batch_size, num_kv_heads, seq_len, dk]
        v: [batch_size, num_kv_heads, seq_len, dv]
        do: [batch_size, num_heads, seq_len, dv]
    """
    bs, nh, nkvh, l = shape
    rng = jax.random.PRNGKey(seed)
    q_key, k_key, v_key, do_key = jax.random.split(rng, 4)

    q = jax.random.uniform(q_key, (bs, nh, l, 128), dtype=dtype)
    k = jax.random.uniform(k_key, (bs, nkvh, l, 128), dtype=dtype)
    v = jax.random.uniform(v_key, (bs, nkvh, l, 128), dtype=dtype)
    h0 = jax.random.uniform(k_key, (bs, nh, 256, 128), dtype=jnp.float32)
    do = jax.random.uniform(do_key, (bs, nh, l, 128), dtype=dtype)
    return q, k, v, h0, do


class LinearAttentionPallasKernelTest(TestCase):
    @parameterized.product(
        batch_size=[2, 4],
        num_heads=[4, 8],
        num_kv_heads=[2, 4],
        seq_len=[1024, 2048],
        dtype=["float32", "bfloat16"],
        window_size=[127, 255],
        feat_map=["softmax", "relu"],
        chunk_size=[128, 256],
        seed=[0, 1],
    )
    def test_rla_forward_and_backward(
        self,
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        dtype,
        window_size,
        feat_map,
        chunk_size,
        seed,
    ):
        shape = (batch_size, num_heads, num_kv_heads, seq_len)
        # pylint: disable=invalid-name
        q, k, v, h0, do = _generate_test_inputs(shape, dtype, seed)
        if dtype == "bfloat16":
            tol = 1e-0
        else:
            tol = 1e-2

        feat_map = FeatureMap(feat_map)
        o_pallas = residual_linear_attention(
            q, k, v, h0, window_size=window_size, feat_map=feat_map, chunk_size=chunk_size
        )
        o_ref, _ = residual_linear_attention_linear_scan(
            q, k, v, h0, window_size=window_size, feat_map=feat_map, chunk_size=chunk_size
        )

        assert_allclose(o_pallas, o_ref, atol=tol, rtol=tol)

        # pylint: disable=invalid-name
        _residual_linear_attention = lambda q, k, v, h0: residual_linear_attention(
            q, k, v, h0, window_size=window_size, feat_map=feat_map, chunk_size=chunk_size
        )
        _, pallas_backward = jax.vjp(_residual_linear_attention, q, k, v, h0)
        dq_pallas, dk_pallas, dv_pallas, dh0_pallas = pallas_backward(do)
        _linear_attention_linear_scan = lambda q, k, v, h0: residual_linear_attention_linear_scan(
            q, k, v, h0, window_size=window_size, feat_map=feat_map, chunk_size=chunk_size
        )[0]
        _, reference_backward = jax.vjp(_linear_attention_linear_scan, q, k, v, h0)
        dq_ref, dk_ref, dv_ref, dh0_ref = reference_backward(do)

        self.assertEqual(dq_pallas.dtype, q.dtype)
        self.assertEqual(dk_pallas.dtype, k.dtype)
        self.assertEqual(dv_pallas.dtype, v.dtype)
        self.assertEqual(dh0_pallas.dtype, h0.dtype)

        assert_allclose(dq_pallas, dq_ref, atol=tol, rtol=tol)
        assert_allclose(dk_pallas, dk_ref, atol=tol, rtol=tol)
        assert_allclose(dv_pallas, dv_ref, atol=tol, rtol=tol)
        assert_allclose(dh0_pallas, dh0_ref, atol=tol, rtol=tol)

    @parameterized.product(
        dtype=["float32", "bfloat16"],
    )
    def test_rla_prefill(self, dtype):
        batch_size, num_heads, seq_len = 2, 4, 1024
        window_size = 255
        chunk_size = 64

        shape = (batch_size, num_heads, num_heads, seq_len)
        q, k, v, h0, _ = _generate_test_inputs(shape, dtype, 0)
        feat_map = FeatureMap("softmax")
        time_step = jnp.arange(batch_size) + 6
        tol = 1e-2

        o_ref, _ = residual_linear_attention_linear_scan(
            q, k, v, h0, window_size=window_size, feat_map=feat_map, chunk_size=chunk_size
        )
        o_prefill, _ = residual_linear_attention_w_timestep(
            q,
            k,
            v,
            h0,
            time_step,
            window_size=window_size,
            feat_map=feat_map,
            chunk_size=chunk_size,
        )

        timestep_mask = jnp.arange(seq_len)[None, :] < time_step[:, None]
        o_ref = o_ref * timestep_mask[:, None, :, None]
        o_prefill = o_prefill * timestep_mask[:, None, :, None]

        self.assertEqual(o_ref.dtype, v.dtype)
        assert_allclose(o_prefill, o_ref, atol=tol, rtol=tol)

# Copyright © 2024 Amazon Inc.
"""Tests for Flash attention on Neuron. Tested on trn1 & trn2."""
import functools

import chex
import jax
import jax.numpy as jnp
import pytest

from axlearn.common.flash_attention.neuron_attention import flash_attention
from axlearn.common.flash_attention.utils import mha_reference


if jax.default_backend() != "neuron":
    pytestmark = pytest.mark.skip(reason="Incompatible hardware, AWS Neuron only test.")


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        (1, 2048, 1, 64),
        (2, 2048, 2, 64),
        (1, 2048, 1, 128),
        (2, 2048, 2, 128),
        (1, 2048, 8, 128),
        (2, 2048, 8, 128),
    ],
)
@pytest.mark.parametrize("use_fwd", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("input_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_fwd_against_ref(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    per_head_dim: int,
    use_fwd: bool,
    causal: bool,
    input_dtype: jnp.dtype,
):
    sm_scale = 1.0 / (per_head_dim**0.5)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)

    bias = None
    segment_ids = None

    if use_fwd:

        @jax.jit
        def impl(q, k, v, bias):
            fn = functools.partial(
                flash_attention,
                causal=causal,
                softmax_scale=sm_scale,
            )
            out, _ = jax.vjp(fn, q, k, v, bias)
            return out

    else:
        impl = functools.partial(
            flash_attention,
            causal=causal,
            softmax_scale=sm_scale,
        )

    o = impl(q, k, v, bias)
    o_ref = mha_reference(q, k, v, bias, segment_ids, causal=causal, softmax_scale=sm_scale)
    chex.assert_trees_all_close(o, o_ref, atol=0.05)


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 1, 2048, 64),
        (2, 2, 2048, 64),
        (1, 1, 2048, 128),
        (2, 2, 2048, 128),
        (1, 8, 2048, 128),
        (2, 8, 2048, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("input_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_bwd_against_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    causal: bool,
    input_dtype: jnp.dtype,
):
    sm_scale = 1.0 / (per_head_dim**0.5)
    q = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype
    )

    bias = None
    segment_ids = None

    def fn(q, k, v, bias):
        return flash_attention(
            q,
            k,
            v,
            bias,
            causal=causal,
            softmax_scale=sm_scale,
        ).sum()

    def ref_fn(q, k, v, bias, segment_ids):
        return mha_reference(
            q,
            k,
            v,
            bias,
            segment_ids,
            causal=causal,
            softmax_scale=sm_scale,
        ).sum()

    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, bias)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v, bias, segment_ids)
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.07)

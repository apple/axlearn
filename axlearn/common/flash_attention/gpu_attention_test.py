# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# jax-ml/jax-triton:
# Copyright 2023 The jax_triton Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests GPU FlashAttention kernels.

Currently tested on A100.
"""
# pylint: disable=wrong-import-position
import functools
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import chex
import jax
import jax.numpy as jnp
import pytest

try:
    import jax_triton as jt  # pytype: disable=import-error  # pylint: disable=import-error

    from axlearn.common.flash_attention.gpu_attention import flash_attention
    from axlearn.common.flash_attention.utils import mha_reference

    if jt.get_compute_capability(0) < 80:
        pytest.skip(reason="Incompatible hardware.", allow_module_level=True)
except ModuleNotFoundError as e:
    # Some libraries can only be installed on GPU, so we'll skip on CI.
    pytest.skip(
        reason=f"Skipping flash_attention tests due to missing deps: {e}",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        (1, 384, 1, 64),
        (2, 384, 2, 64),
        (1, 384, 1, 64),
        (2, 384, 2, 64),
        (1, 384, 8, 64),
        (2, 384, 8, 64),
    ],
)
@pytest.mark.parametrize("block_size", [128, 64])
@pytest.mark.parametrize("use_fwd", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("sm_scale", [1.0, 0.123])
@pytest.mark.parametrize("bias_type", ["none", "matrix"])
def test_fwd_against_ref(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    per_head_dim: int,
    block_size: int,
    use_fwd: bool,
    causal: bool,
    sm_scale: float,
    bias_type: str,
):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16)

    if bias_type == "matrix":
        bias = jax.random.normal(k4, (batch_size, num_heads, seq_len, seq_len), dtype=jnp.float16)
    else:
        bias = None

    if use_fwd:

        @jax.jit
        def impl(q, k, v, bias):
            fn = functools.partial(
                flash_attention,
                block_q=block_size,
                block_k=block_size,
                causal=causal,
                softmax_scale=sm_scale,
            )
            out, _ = jax.vjp(fn, q, k, v, bias)
            return out

    else:
        impl = functools.partial(
            flash_attention,
            block_q=block_size,
            block_k=block_size,
            causal=causal,
            softmax_scale=sm_scale,
        )

    o = impl(q, k, v, bias)
    o_ref = mha_reference(q, k, v, bias, causal=causal, softmax_scale=sm_scale)
    chex.assert_trees_all_close(o, o_ref, atol=0.05)


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 1, 384, 64),
        (2, 2, 384, 64),
        (1, 1, 384, 64),
        (2, 2, 384, 64),
        (1, 8, 384, 64),
        (2, 8, 384, 64),
    ],
)
@pytest.mark.parametrize("bias_type", ["none", "matrix"])
@pytest.mark.parametrize("block_size", [128, 64])
@pytest.mark.parametrize("causal", [True, False])
def test_bwd_against_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    bias_type: str,
    block_size: int,
    causal: bool,
):
    q = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
    )

    if bias_type == "matrix":
        bias = jax.random.normal(
            jax.random.PRNGKey(3), (batch_size, num_heads, seq_len, seq_len), dtype=jnp.float16
        )
    else:
        bias = None

    assert str(q.device()) == "gpu:0"
    sm_scale = q.shape[-1] ** -0.5

    # Compare outputs.
    jax_out = flash_attention(q, k, v, bias, causal=causal, softmax_scale=sm_scale)
    jax_ref_out = mha_reference(q, k, v, bias, causal=causal, softmax_scale=sm_scale)
    chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005)

    def fn(q, k, v, bias):
        return flash_attention(
            q,
            k,
            v,
            bias,
            causal=causal,
            softmax_scale=sm_scale,
            block_q=block_size,
            block_k=block_size,
        ).sum()

    def ref_fn(q, k, v, bias):
        return mha_reference(q, k, v, bias, causal=causal, softmax_scale=sm_scale).sum()

    # Compare gradients.
    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, bias)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v, bias)
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.05)

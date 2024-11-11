# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# jax-ml/jax-triton:
# Copyright 2023 The jax_triton Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests GPU FlashAttention kernels.

Currently tested on A100/H100.
"""
import functools
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import pytest

from axlearn.common.flash_attention.gpu_attention import (
    cudnn_dot_product_attention,
    flash_attention,
)
from axlearn.common.flash_attention.utils import mha_reference

if jax.default_backend() != "gpu":
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        (1, 384, 1, 64),
        (2, 384, 2, 64),
        (1, 384, 1, 128),
        (2, 384, 2, 128),
        (1, 384, 8, 128),
        (2, 384, 8, 128),
    ],
)
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("sm_scale", [1.0, 0.123])
@pytest.mark.parametrize("attention_bias_type", [None, "2d", "4d"])
@pytest.mark.parametrize("use_segment_ids", [True, False])
@pytest.mark.parametrize("input_dtype", [jnp.float16, jnp.float32])
def test_triton_fwd_only_against_ref(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    per_head_dim: int,
    block_size: int,
    causal: bool,
    sm_scale: float,
    attention_bias_type: Literal["2d", "4d", None],
    use_segment_ids: bool,
    input_dtype: jnp.dtype,
):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)

    if attention_bias_type == "4d":
        bias = jax.random.normal(k4, (batch_size, num_heads, seq_len, seq_len), dtype=input_dtype)
    elif attention_bias_type == "2d":
        bias = jax.random.normal(k4, (1, 1, seq_len, seq_len), dtype=input_dtype)
    else:
        bias = None

    segment_left = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_right = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_ids = (
        jnp.concatenate([segment_left, segment_right], axis=-1) if use_segment_ids else None
    )

    @jax.jit
    def impl(q, k, v, bias, segment_ids):
        fn = functools.partial(
            flash_attention,
            block_q=block_size,
            block_k=block_size,
            causal=causal,
            softmax_scale=sm_scale,
        )
        out, _ = jax.vjp(fn, q, k, v, bias, segment_ids)
        return out

    o = impl(q, k, v, bias, segment_ids)
    o_ref = mha_reference(q, k, v, bias, segment_ids, causal=causal, softmax_scale=sm_scale)
    chex.assert_trees_all_close(o, o_ref, atol=0.07)


# We test the flash_attention against the reference mha_reference.
# The outputs should be close in both fp16 and fp32, with a relaxed bound due
# to the numerical difference during operations.
@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 1, 384, 64),
        (2, 2, 384, 64),
        (1, 1, 384, 128),
        (2, 2, 384, 128),
        (1, 8, 384, 128),
        (2, 8, 384, 128),
    ],
)
@pytest.mark.parametrize("attention_bias_type", [None, "2d", "4d"])
@pytest.mark.parametrize("use_segment_ids", [True, False])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("input_dtype", [jnp.float16, jnp.float32])
def test_triton_against_xla_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    attention_bias_type: Literal["2d", "4d", None],
    use_segment_ids: bool,
    block_size: int,
    causal: bool,
    input_dtype: jnp.dtype,
):
    q = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype
    )

    if attention_bias_type == "4d":
        bias = jax.random.normal(
            jax.random.PRNGKey(3), (batch_size, num_heads, seq_len, seq_len), dtype=input_dtype
        )
    elif attention_bias_type == "2d":
        bias = jax.random.normal(jax.random.PRNGKey(3), (1, 1, seq_len, seq_len), dtype=input_dtype)
    else:
        bias = None

    segment_left = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_right = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_ids = (
        jnp.concatenate([segment_left, segment_right], axis=-1) if use_segment_ids else None
    )

    sm_scale = q.shape[-1] ** -0.5

    # Compare outputs.
    jax_out = flash_attention(
        q,
        k,
        v,
        bias,
        segment_ids,
        causal=causal,
        softmax_scale=sm_scale,
        block_q=block_size,
        block_k=block_size,
    )
    jax_ref_out = mha_reference(q, k, v, bias, segment_ids, causal=causal, softmax_scale=sm_scale)
    if input_dtype == jnp.float16:
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005)
    elif input_dtype == jnp.float32:
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.05)
    else:
        raise ValueError(f"Unsupported dtype: {input_dtype}")

    def fn(q, k, v, bias, segment_ids):
        return flash_attention(
            q,
            k,
            v,
            bias,
            segment_ids,
            causal=causal,
            softmax_scale=sm_scale,
            block_q=block_size,
            block_k=block_size,
        ).sum()

    def ref_fn(q, k, v, bias, segment_ids):
        return mha_reference(
            q, k, v, bias, segment_ids, causal=causal, softmax_scale=sm_scale
        ).sum()

    # Compare gradients.
    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, bias, segment_ids)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v, bias, segment_ids)
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.05)


# We test the cudnn_dot_product_attention against the reference flash_attention.
# Due to its algorithmic equivalence, the outputs should be close in both fp16 and bfloat16.
@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 2, 2048, 128),
        (2, 2, 2048, 128),
        (1, 4, 4096, 128),
        (2, 8, 4096, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_cudnn_against_triton_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    causal: bool,
    dtype: jnp.dtype,
):
    q = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype
    )

    sm_scale = q.shape[-1] ** -0.5

    # Compare outputs.
    jax_out = cudnn_dot_product_attention(q, k, v, bias=None, causal=causal, softmax_scale=sm_scale)
    jax_ref_out = flash_attention(q, k, v, bias=None, causal=causal, softmax_scale=sm_scale)
    if dtype == jnp.bfloat16:
        # We relax the atol to support bf16 in the unit test.
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.02, rtol=1e-5)
    elif dtype == jnp.float16:
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005, rtol=1e-5)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    def fn(q, k, v):
        return cudnn_dot_product_attention(
            q, k, v, bias=None, causal=causal, softmax_scale=sm_scale
        ).sum()

    def ref_fn(q, k, v):
        return flash_attention(q, k, v, bias=None, causal=causal, softmax_scale=sm_scale).sum()

    # Compare gradients.
    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v)
    # The diff between grads are expected to be larger than the forward pass.
    if dtype == jnp.bfloat16:
        # We relax the rtol to support bf16 in the unit test.
        chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.05, rtol=1e-2)
    elif dtype == jnp.float16:
        chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.05, rtol=1e-5)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

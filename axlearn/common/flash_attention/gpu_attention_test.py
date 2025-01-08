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
from absl.testing import parameterized

from axlearn.common.attention_bias import sliding_window_causal_mask
from axlearn.common.flash_attention.gpu_attention import (
    cudnn_dot_product_attention,
    flash_attention,
)
from axlearn.common.flash_attention.gpu_decoding import NEG_INF, flash_decoding
from axlearn.common.flash_attention.utils import _repeat_kv_heads, mha_reference
from axlearn.common.test_utils import TestCase

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
@pytest.mark.parametrize("kv_seq_len", [-1, 512])
@pytest.mark.parametrize("dropout_rate", [0, 0.1])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("softmax_scale", [1.0, 0.123])
@pytest.mark.parametrize("attention_bias_type", [None, "2d", "4d"])
@pytest.mark.parametrize("use_segment_ids", [True, False])
@pytest.mark.parametrize("input_dtype", [jnp.float16, jnp.float32])
def test_triton_fwd_only_against_ref(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    per_head_dim: int,
    kv_seq_len: int,
    dropout_rate: float,
    block_size: int,
    causal: bool,
    softmax_scale: float,
    attention_bias_type: Literal["2d", "4d", None],
    use_segment_ids: bool,
    input_dtype: jnp.dtype,
):
    if kv_seq_len == -1:
        kv_seq_len = seq_len
    if kv_seq_len != seq_len and use_segment_ids:
        pytest.skip()
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_heads, per_head_dim), dtype=input_dtype)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_heads, per_head_dim), dtype=input_dtype)

    if attention_bias_type == "4d":
        bias = jax.random.normal(
            k4, (batch_size, num_heads, seq_len, kv_seq_len), dtype=input_dtype
        )
    elif attention_bias_type == "2d":
        bias = jax.random.normal(k4, (1, 1, seq_len, kv_seq_len), dtype=input_dtype)
    else:
        bias = None

    segment_left = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_right = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_ids = (
        jnp.concatenate([segment_left, segment_right], axis=-1) if use_segment_ids else None
    )

    o = flash_attention(
        q,
        k,
        v,
        bias,
        segment_ids,
        k5,
        block_q=block_size,
        block_k=block_size,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_rate=dropout_rate,
    )
    o_ref = mha_reference(
        q,
        k,
        v,
        bias,
        segment_ids,
        k5,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_rate=dropout_rate,
    )
    if input_dtype == jnp.float16:
        chex.assert_trees_all_close(o, o_ref, atol=0.07)
    elif input_dtype == jnp.float32:
        chex.assert_trees_all_close(o, o_ref, atol=0.03)


class FlashDecodingTest(TestCase):
    """Tests FlashDecoding."""

    @parameterized.product(
        [
            dict(zip(["batch_size", "seq_len", "num_heads", "per_head_dim"], args))
            for args in [
                (1, 1024, 32, 64),
                (1, 444, 16, 64),
                (8, 1596, 48, 128),
                (8, 4044, 64, 128),
            ]
        ],
        softmax_scale=[1.0, 0.83],
        attention_bias_type=["2d", "4d", None],
        input_dtype=[jnp.float32, jnp.float16],
        padding=[0, 111],
        kv_head_factor=[1, 4, 8],
        window_len=[-1, 16, 127],
    )
    def test_decode_against_ref(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        per_head_dim: int,
        softmax_scale: float,
        attention_bias_type: Literal["2d", "4d", None],
        input_dtype: jnp.dtype,
        padding: int,
        kv_head_factor: int,
        window_len: int,
    ):
        self.assertEqual(num_heads % kv_head_factor, 0)
        assert num_heads % kv_head_factor == 0
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
        q = jax.random.normal(k1, (batch_size, 1, num_heads, per_head_dim), dtype=input_dtype)
        k = jax.random.normal(
            k2,
            (batch_size, seq_len + padding, num_heads // kv_head_factor, per_head_dim),
            dtype=input_dtype,
        )
        v = jax.random.normal(
            k3,
            (batch_size, seq_len + padding, num_heads // kv_head_factor, per_head_dim),
            dtype=input_dtype,
        )

        if attention_bias_type == "4d":
            bias = jax.random.normal(
                k4, (batch_size, num_heads, 1, seq_len + padding), dtype=input_dtype
            )
        elif attention_bias_type == "2d":
            bias = jax.random.normal(k4, (1, 1, 1, seq_len + padding), dtype=input_dtype)
        else:
            bias = None

        mask_fn = None
        if window_len > 0:
            mask_fn = sliding_window_causal_mask(window_len)
        o = flash_decoding(
            q, k, v, bias=bias, softmax_scale=softmax_scale, kv_seq_len=seq_len, mask_fn=mask_fn
        )
        if bias is not None:
            bias = bias[:, :, :, :seq_len]
        if window_len > 0:
            if bias is None:
                bias = jnp.zeros((1, 1, 1, seq_len), dtype=input_dtype)
            bias = bias.at[:, :, :, : -window_len - 1].set(NEG_INF)
        o_ref = mha_reference(
            q,
            _repeat_kv_heads(num_heads, k[:, :seq_len]),
            _repeat_kv_heads(num_heads, v[:, :seq_len]),
            bias,
            None,
            causal=False,
            softmax_scale=softmax_scale,
        )
        self.assertGreaterEqual(jnp.median(jnp.abs(o_ref)).item(), 0.25)
        if input_dtype is jnp.float32:
            self.assertNestedAllClose(o, o_ref, rtol=0.01, atol=0.01)
        else:
            self.assertNestedAllClose(o, o_ref, rtol=0.05, atol=0.05)


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
@pytest.mark.parametrize("kv_seq_len", [-1, 512])
@pytest.mark.parametrize("dropout_rate", [0, 0.1])
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
    kv_seq_len: int,
    attention_bias_type: Literal["2d", "4d", None],
    use_segment_ids: bool,
    dropout_rate: float,
    block_size: int,
    causal: bool,
    input_dtype: jnp.dtype,
):
    if kv_seq_len == -1:
        kv_seq_len = seq_len
    if kv_seq_len != seq_len and use_segment_ids:
        pytest.skip()
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    k = jax.random.normal(k2, (batch_size, kv_seq_len, num_heads, per_head_dim), dtype=input_dtype)
    v = jax.random.normal(k3, (batch_size, kv_seq_len, num_heads, per_head_dim), dtype=input_dtype)

    if attention_bias_type == "4d":
        bias = jax.random.normal(
            k4, (batch_size, num_heads, seq_len, kv_seq_len), dtype=input_dtype
        )
    elif attention_bias_type == "2d":
        bias = jax.random.normal(k4, (1, 1, seq_len, kv_seq_len), dtype=input_dtype)
    else:
        bias = None

    segment_left = jnp.ones((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_right = jnp.zeros((batch_size, seq_len // 2), dtype=jnp.int32)
    segment_ids = (
        jnp.concatenate([segment_left, segment_right], axis=-1) if use_segment_ids else None
    )

    softmax_scale = q.shape[-1] ** -0.5

    # Compare outputs.
    call_flash = functools.partial(
        flash_attention,
        causal=causal,
        softmax_scale=softmax_scale,
        block_q=block_size,
        block_k=block_size,
        dropout_rate=dropout_rate,
    )
    jax_out = call_flash(
        q,
        k,
        v,
        bias,
        segment_ids,
        k5,
    )
    jax_ref_out = mha_reference(
        q,
        k,
        v,
        bias,
        segment_ids,
        k5,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_rate=dropout_rate,
    )
    if input_dtype == jnp.float16:
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005)
    elif input_dtype == jnp.float32:
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005)
    else:
        raise ValueError(f"Unsupported dtype: {input_dtype}")

    def fn(q, k, v, bias, segment_ids, k5):
        return call_flash(
            q,
            k,
            v,
            bias,
            segment_ids,
            k5,
        ).sum()

    def ref_fn(q, k, v, bias, segment_ids, k5):
        return mha_reference(
            q,
            k,
            v,
            bias,
            segment_ids,
            k5,
            causal=causal,
            softmax_scale=softmax_scale,
            dropout_rate=dropout_rate,
        ).sum()

    # Compare gradients.
    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, bias, segment_ids, k5)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v, bias, segment_ids, k5)
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
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=dtype)

    softmax_scale = q.shape[-1] ** -0.5

    # Compare outputs.
    jax_out = cudnn_dot_product_attention(
        q, k, v, bias=None, causal=causal, softmax_scale=softmax_scale
    )
    jax_ref_out = flash_attention(q, k, v, bias=None, causal=causal, softmax_scale=softmax_scale)
    if dtype == jnp.bfloat16:
        # We relax the atol to support bf16 in the unit test.
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.02, rtol=1e-5)
    elif dtype == jnp.float16:
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.005, rtol=1e-5)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    def fn(q, k, v):
        return cudnn_dot_product_attention(
            q, k, v, bias=None, causal=causal, softmax_scale=softmax_scale
        ).sum()

    def ref_fn(q, k, v):
        return flash_attention(q, k, v, bias=None, causal=causal, softmax_scale=softmax_scale).sum()

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


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 1, 128, 128),
        (2, 4, 128, 128),
        (1, 2, 64, 64),
        (2, 8, 64, 64),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("dropout_rate", [0.1, 0.25])
def test_cudnn_dropout_against_xla_dropout(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    causal: bool,
    dtype: jnp.dtype,
    dropout_rate: float,
):
    """Tests that cudnn dropout works as expected.

    Since cuDNN uses a different kind of RNG than Jax, we retrieve the mask generated by cuDNN
    by setting V to the identity matrix. However, this only works when seq_len == per_head_dim,
    i.e. when the shape of output is the same as the shape of the dropout mask.
    """
    qkv_shape = (batch_size, seq_len, num_heads, per_head_dim)
    softmax_scale = 1.0
    cudnn_attn = functools.partial(
        cudnn_dot_product_attention,
        bias=None,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_rate=dropout_rate,
    )

    dropout_mask = (
        cudnn_attn(
            jnp.zeros(qkv_shape, dtype=dtype),
            jnp.zeros(qkv_shape, dtype=dtype),
            jnp.broadcast_to(jnp.eye(per_head_dim, dtype=dtype)[None, :, None], qkv_shape),
        )
        == 0.0
    ).swapaxes(1, 2)
    # Clear the compilation cache to reset cudnn RNG offset, so the next invocation will generate
    # the same mask.
    jax.clear_caches()

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k1, qkv_shape, dtype=dtype)
    k = jax.random.normal(k2, qkv_shape, dtype=dtype)
    v = jax.random.normal(k3, qkv_shape, dtype=dtype)

    ref_attn = functools.partial(
        mha_reference,
        bias=None,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_mask=dropout_mask,
        dropout_rate=dropout_rate,
    )
    # Compare outputs.
    jax_out = cudnn_attn(q, k, v)
    jax_ref_out = ref_attn(q, k, v)
    if dtype == jnp.bfloat16:
        # We relax the atol to support bf16 in the unit test.
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.25, rtol=1e-3)
    elif dtype == jnp.float16:
        chex.assert_trees_all_close(jax_out, jax_ref_out, atol=0.05, rtol=1e-3)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    def fn(q, k, v):
        return cudnn_attn(q, k, v).mean()

    def ref_fn(q, k, v):
        return ref_attn(q, k, v).mean()

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


def test_cudnn_dropout_determinism():
    """Tests that cuDNN dropout produces identical outputs across runs."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
    q = jax.random.normal(k1, (1, 128, 2, 64), dtype=jnp.float16)
    k = jax.random.normal(k2, (1, 128, 2, 64), dtype=jnp.float16)
    v = jax.random.normal(k3, (1, 128, 2, 64), dtype=jnp.float16)
    outputs = []
    grads = []

    def fn(q, k, v):
        return cudnn_dot_product_attention(q, k, v, dropout_rate=0.1).mean()

    for i in range(10):
        outputs.append(cudnn_dot_product_attention(q, k, v, dropout_rate=0.1))
        grads.append(jax.grad(fn, argnums=(0, 1, 2))(q, k, v))

    jax.clear_caches()

    for i in range(10):
        chex.assert_trees_all_equal(
            cudnn_dot_product_attention(q, k, v, dropout_rate=0.1), outputs[i]
        )
        chex.assert_trees_all_equal(jax.grad(fn, argnums=(0, 1, 2))(q, k, v), grads[i])

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
from typing import Any, Callable, Literal

import chex
import jax
import jax.numpy as jnp
import pytest

from axlearn.common.attention_bias import (
    CausalAttentionBias,
    ZeroAttentionBias,
    causal_mask,
    sliding_window_causal_mask,
)
from axlearn.common.flash_attention.gpu_attention import (
    CuDNNGPUFlashAttention,
    PallasGPUFlashAttention,
)
from axlearn.common.flash_attention.test_utils import generate_attention_data
from axlearn.common.flash_attention.utils import ReferenceMHA
from axlearn.common.utils import Tensor

if jax.default_backend() not in ("gpu", "cpu"):
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


def _default_tol_fn(backend, dtype):
    del backend
    if dtype == jnp.bfloat16:
        return dict(atol=0.05, rtol=1e-2)
    if dtype == jnp.float16:
        return dict(atol=0.05, rtol=1e-5)
    if dtype == jnp.float32:
        return dict(atol=0.025, rtol=1e-5)
    raise ValueError(f"Unsupported dtype: {dtype}")


TestFn = Callable[[Tensor, Tensor, Tensor], Tensor]
TolFn = Callable[[str, Any], dict[str, float]]


def _test_forward_and_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias,
    *,
    ref_fn: TestFn,
    test_fn: TestFn,
    forward_tol_fn: Callable = _default_tol_fn,
    backward_tol_fn: Callable = _default_tol_fn,
):
    ref_fn = jax.jit(ref_fn)
    test_fn = jax.jit(test_fn)

    jax_out = test_fn(q, k, v, bias)
    jax_ref_out = ref_fn(q, k, v, bias)
    backend = jax.default_backend()
    chex.assert_trees_all_close(jax_out, jax_ref_out, **forward_tol_fn(backend, q.dtype))

    # Compare gradients.
    jax_grads = jax.grad(lambda q, k, v: ref_fn(q, k, v, bias).mean(), argnums=(0, 1, 2))(q, k, v)
    jax_ref_grads = jax.grad(lambda q, k, v: test_fn(q, k, v, bias).mean(), argnums=(0, 1, 2))(
        q, k, v
    )
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, **backward_tol_fn(backend, q.dtype))


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
@pytest.mark.parametrize("kv_seq_len", [None, 512])
@pytest.mark.parametrize("dropout_rate", [0, 0.1])
@pytest.mark.parametrize("attention_bias_type", [None, "2d", "4d"])
@pytest.mark.parametrize("use_segment_ids", [True, False])
@pytest.mark.parametrize("block_size", [128])  # Triton broken for block size !=128
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
    if jax.default_backend() == "cpu" and kv_seq_len >= 512:
        pytest.skip(reason="Too slow on CPU.")
    q, k, v, bias = generate_attention_data(
        batch_size,
        seq_len,
        kv_seq_len or seq_len,
        num_heads,
        per_head_dim,
        mask_fn=causal_mask if causal else None,
        attention_bias_type=attention_bias_type,
        with_segment_ids=use_segment_ids,
        dtype=input_dtype,
    )

    cfg = dict(
        softmax_scale=q.shape[-1] ** -0.5,
        interpret=jax.default_backend() == "cpu",
        dropout_rate=dropout_rate,
        gpu_block_size=block_size,
    )
    # Compare outputs.
    call_flash = PallasGPUFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()

    def forward_tol_fn(backend, dtype):
        del dtype
        # TODO(kelvin-zou): Investigate the discrepancy between CPU and GPU.
        if backend == "cpu":
            return dict(rtol=5e-2, atol=1e-2)
        return dict(atol=0.005)

    _test_forward_and_backward(
        q, k, v, bias, ref_fn=ref_fn, test_fn=call_flash, forward_tol_fn=forward_tol_fn
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [512, 2048])
@pytest.mark.parametrize("sliding_window_size", [256])
@pytest.mark.parametrize("use_segment_ids", [True, False])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("per_head_dim", [128])
def test_sliding_window_mask(
    batch_size,
    seq_len,
    num_heads,
    per_head_dim,
    sliding_window_size,
    use_segment_ids: bool,
):
    q, k, v, bias = generate_attention_data(
        batch_size,
        seq_len,
        seq_len,
        num_heads,
        per_head_dim,
        mask_fn=sliding_window_causal_mask(sliding_window_size),
        with_segment_ids=use_segment_ids,
    )

    cfg = dict(
        softmax_scale=q.shape[-1] ** -0.5,
        interpret=jax.default_backend() == "cpu",
    )
    fn = PallasGPUFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
    _test_forward_and_backward(q, k, v, bias, ref_fn=ref_fn, test_fn=fn)


# We test the cudnn_dot_product_attention against the reference flash_attention.
# Due to its algorithmic equivalence, the outputs should be close in both fp16 and bfloat16.
@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 2, 1024, 128),
        (2, 2, 1024, 128),
        (1, 4, 2048, 128),
        (2, 8, 2048, 128),
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
    if jax.default_backend() == "cpu":
        pytest.skip(reason="cudnn function needs GPU.")

    q, k, v, bias = generate_attention_data(
        batch_size,
        seq_len,
        seq_len,
        num_heads,
        per_head_dim,
        mask_fn=causal_mask if causal else None,
        dtype=dtype,
    )

    cfg = dict(
        softmax_scale=q.shape[-1] ** -0.5,
    )

    # Compare outputs.
    test_fn = CuDNNGPUFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()

    def forward_tol_fn(backend, dtype):
        del backend
        if dtype == jnp.bfloat16:
            return dict(atol=0.02, rtol=1e-5)
        if dtype == jnp.float16:
            return dict(atol=0.005, rtol=1e-5)

    _test_forward_and_backward(
        q, k, v, bias, ref_fn=ref_fn, test_fn=test_fn, forward_tol_fn=forward_tol_fn
    )


def _cudnn_xla_forward_tol_fn(backend, dtype):
    del backend
    # cuDNN has higher diff when compared to non-fused attention in XLA.
    if dtype == jnp.bfloat16:
        return dict(atol=0.25, rtol=1e-3)
    if dtype == jnp.float16:
        return dict(atol=0.05, rtol=1e-3)


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
    if jax.default_backend() == "cpu":
        pytest.skip(reason="cudnn function needs GPU.")
    qkv_shape = (batch_size, seq_len, num_heads, per_head_dim)
    cfg = dict(softmax_scale=per_head_dim**-0.5, dropout_rate=dropout_rate)
    bias = (
        CausalAttentionBias(
            target_positions=jnp.arange(seq_len)[None], source_positions=jnp.arange(seq_len)[None]
        )
        if causal
        else ZeroAttentionBias()
    )

    # Compare outputs.
    test_fn = CuDNNGPUFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()

    dropout_mask = (
        test_fn(
            jnp.zeros(qkv_shape, dtype=dtype),
            jnp.zeros(qkv_shape, dtype=dtype),
            jnp.broadcast_to(jnp.eye(per_head_dim, dtype=dtype)[None, :, None], qkv_shape),
            bias,
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

    ref_fn = functools.partial(
        ref_fn,
        dropout_mask=dropout_mask,
    )

    _test_forward_and_backward(
        q, k, v, bias, ref_fn=ref_fn, test_fn=test_fn, forward_tol_fn=_cudnn_xla_forward_tol_fn
    )


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,kv_seq_len,per_head_dim",
    [
        (1, 1, 378, 676, 128),
        (2, 4, 582, 582, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_cudnn_arbitrary_seq_len(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    kv_seq_len: int,
    per_head_dim: int,
    causal: bool,
    dtype: jnp.dtype,
):
    """Tests that cudnn supports any even sequence length."""
    if jax.default_backend() == "cpu":
        pytest.skip(reason="cudnn function needs GPU.")
    q, k, v, bias = generate_attention_data(
        batch_size,
        seq_len,
        kv_seq_len,
        num_heads,
        per_head_dim,
        mask_fn=causal_mask if causal else None,
        dtype=dtype,
    )

    cfg = dict(
        softmax_scale=q.shape[-1] ** -0.5,
    )

    # Compare outputs.
    test_fn = CuDNNGPUFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()

    _test_forward_and_backward(
        q, k, v, bias, ref_fn=ref_fn, test_fn=test_fn, forward_tol_fn=_cudnn_xla_forward_tol_fn
    )


def test_cudnn_dropout_determinism():
    """Tests that cuDNN dropout produces identical outputs across runs."""
    if jax.default_backend() == "cpu":
        pytest.skip(reason="cudnn function needs GPU.")
    q, k, v, bias = generate_attention_data(*(1, 128, 128, 2, 64))
    fn = CuDNNGPUFlashAttention.default_config().set(dropout_rate=0.1).instantiate()

    outputs = []
    grads = []

    for i in range(10):
        outputs.append(fn(q, k, v, bias))
        grads.append(jax.grad(fn, argnums=(0, 1, 2))(q, k, v))

    jax.clear_caches()

    for i in range(10):
        chex.assert_trees_all_equal(fn(q, k, v, bias), outputs[i])
        chex.assert_trees_all_equal(jax.grad(fn, argnums=(0, 1, 2))(q, k, v), grads[i])

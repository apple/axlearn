# pylint: disable=unbalanced-tuple-unpacking
# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# jax-ml/jax-triton:
# Copyright 2023 The jax_triton Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests GPU FlashAttention kernels.

Currently tested on A100/H100. To run tests in parallel on a multi-GPU machine, use this:
```
PARALLEL_GPU_TEST=1 pytest -n 8 axlearn/common/flash_attention/gpu_attention_test.py
```
"""
import functools
from typing import Any, Callable, Literal, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random
import pytest

from axlearn.common.attention_bias import (
    CausalAttentionBias,
    MaskFn,
    ZeroAttentionBias,
    causal_mask,
)
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.gpu_attention import (
    CuDNNGPUFlashAttention,
    CuDNNGPUFlashAttentionWithExplicitBias,
    PallasGPUFlashAttention,
)
from axlearn.common.flash_attention.test_utils import generate_attention_data
from axlearn.common.utils import Nested, Tensor

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


TestFn = Callable[[Nested[Tensor]], Tensor]
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
    float_batch = dict(query=q, key=k, value=v)
    aux_batch = dict(prng_key=jax.random.PRNGKey(44), bias=bias)
    input_batch = {**float_batch, **aux_batch}
    ref_fn = jax.jit(ref_fn)
    test_fn = jax.jit(test_fn)
    # pylint: disable=not-callable
    jax_out = test_fn(input_batch)
    jax_ref_out = ref_fn(input_batch)
    backend = jax.default_backend()
    chex.assert_trees_all_close(jax_out, jax_ref_out, **forward_tol_fn(backend, q.dtype))

    def grad_ref(float_inputs, aux_inputs):
        full_batch = {**float_inputs, **aux_inputs}
        return ref_fn(full_batch).mean()

    def grad_test(float_inputs, aux_inputs):
        full_batch = {**float_inputs, **aux_inputs}
        return test_fn(full_batch).mean()

    # Compare gradients.
    jax_grads = jax.grad(grad_ref, argnums=0)(float_batch, aux_batch)
    jax_ref_grads = jax.grad(grad_test, argnums=0)(float_batch, aux_batch)

    chex.assert_trees_all_close(jax_grads, jax_ref_grads, **backward_tol_fn(backend, q.dtype))


def common_attn_test_params(func):
    params = [
        pytest.mark.parametrize("kv_len", [None, 512]),
        pytest.mark.parametrize("dropout_rate", [0, 0.1]),
        pytest.mark.parametrize("attention_bias_type", [None, "2d", "4d"]),
        pytest.mark.parametrize("with_segment_ids", [True, False]),
        pytest.mark.parametrize("block_size", [128]),  # Triton broken for block size !=128.
        pytest.mark.parametrize("mask_fn", [causal_mask, None]),
        pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32]),
    ]
    # Apply in reverse order to stack correctly.
    for param in reversed(params):
        func = param(func)
    return func


@pytest.mark.parametrize(
    "batch_size,num_heads,query_len,per_head_dim",
    [
        (1, 1, 384, 64),
        (2, 2, 256, 64),
        (2, 2, 256, 72),
        (1, 1, 512, 128),
        (2, 2, 256, 128),
        (1, 8, 384, 128),
        (2, 4, 384, 128),
    ],
)
@common_attn_test_params
def test_triton_fwd_only_against_ref(
    batch_size: int,
    query_len: int,
    num_heads: int,
    per_head_dim: int,
    kv_len: int,
    dropout_rate: float,
    block_size: int,
    mask_fn: Optional[MaskFn],
    attention_bias_type: Literal["2d", "4d", None],
    with_segment_ids: bool,
    dtype: jnp.dtype,
):
    if query_len >= 384 and jax.default_backend() == "cpu":
        pytest.skip("Too slow on CPU.")
    q, k, v, bias = generate_attention_data(
        batch_size,
        query_len,
        kv_len,
        num_heads,
        per_head_dim,
        mask_fn=mask_fn,
        attention_bias_type=attention_bias_type,
        with_segment_ids=with_segment_ids,
        dtype=dtype,
    )

    cfg = dict(
        softmax_scale=q.shape[-1] ** -0.5,
        interpret=jax.default_backend() == "cpu",
        dropout_rate=dropout_rate,
        gpu_block_size=block_size,
    )
    # Compare outputs.
    test_fn = PallasGPUFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
    input_batch = dict(query=q, key=k, value=v, prng_key=jax.random.PRNGKey(43), bias=bias)
    chex.assert_equal(test_fn.is_supported(input_batch), True)
    o = test_fn(input_batch)
    o_ref = ref_fn(input_batch)

    if dtype == jnp.float16:
        chex.assert_trees_all_close(o, o_ref, atol=0.07)
    elif dtype == jnp.float32:
        chex.assert_trees_all_close(o, o_ref, atol=0.03)


@pytest.mark.parametrize(
    "batch_size,num_heads,query_len,per_head_dim",
    [
        (1, 1, 384, 64),
        (2, 2, 256, 64),
        (1, 1, 512, 128),
        (2, 2, 256, 128),
        (1, 8, 384, 128),
        (2, 4, 384, 128),
    ],
)
@common_attn_test_params
def test_triton_against_xla_ref(
    batch_size: int,
    num_heads: int,
    query_len: int,
    per_head_dim: int,
    kv_len: int,
    attention_bias_type: Literal["2d", "4d", None],
    with_segment_ids: bool,
    dropout_rate: float,
    block_size: int,
    mask_fn: Optional[MaskFn],
    dtype: jnp.dtype,
):
    if query_len >= 384 and jax.default_backend() == "cpu":
        pytest.skip("Too slow on CPU.")
    q, k, v, bias = generate_attention_data(
        batch_size,
        query_len,
        kv_len,
        num_heads,
        per_head_dim,
        mask_fn=mask_fn,
        attention_bias_type=attention_bias_type,
        with_segment_ids=with_segment_ids,
        dtype=dtype,
    )

    cfg = dict(
        softmax_scale=q.shape[-1] ** -0.5,
        interpret=jax.default_backend() == "cpu",
        dropout_rate=dropout_rate,
        # Override the gpu_block_size if running on the B200 platform
        gpu_block_size=(
            64
            if jax.default_backend() == "gpu" and "NVIDIA B200" in jax.devices("gpu")[0].device_kind
            else block_size
        ),
    )
    # Compare outputs.
    test_fn = PallasGPUFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
    input_batch = dict(query=q, key=k, value=v, bias=bias)
    chex.assert_equal(test_fn.is_supported(input_batch), True)

    def forward_tol_fn(backend, dtype):
        del dtype
        # TODO(kelvin-zou): Investigate the discrepancy between CPU and GPU.
        if backend == "cpu":
            return dict(rtol=5e-2, atol=1e-2)
        return dict(atol=0.01)

    _test_forward_and_backward(
        q, k, v, bias, ref_fn=ref_fn, test_fn=test_fn, forward_tol_fn=forward_tol_fn
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [512, 2048])
@pytest.mark.parametrize("sliding_window_size", [256])
@pytest.mark.parametrize("use_segment_ids", [True, False])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("per_head_dim", [128])
@pytest.mark.parametrize("test_cls", [PallasGPUFlashAttention, CuDNNGPUFlashAttention])
def test_sliding_window_mask(
    batch_size,
    seq_len,
    num_heads,
    per_head_dim,
    sliding_window_size,
    use_segment_ids,
    test_cls,
):
    if jax.default_backend() != "gpu" and test_cls is CuDNNGPUFlashAttention:
        pytest.skip("cuDNN requires GPU.")
    q, k, v, bias = generate_attention_data(
        batch_size,
        seq_len,
        seq_len,
        num_heads,
        per_head_dim,
        sliding_window_sz=sliding_window_size,
        with_segment_ids=use_segment_ids,
        dtype=jnp.float16,
    )

    cfg = dict(
        softmax_scale=q.shape[-1] ** -0.5,
        interpret=jax.default_backend() == "cpu",
    )
    test_fn = test_cls.default_config().set(**cfg).instantiate()
    input_batch = dict(query=q, key=k, value=v, bias=bias)
    if test_cls is CuDNNGPUFlashAttention and use_segment_ids:
        chex.assert_equal(test_fn.is_supported(input_batch), False)
        test_fn = CuDNNGPUFlashAttentionWithExplicitBias.default_config().set(**cfg).instantiate()
    chex.assert_equal(test_fn.is_supported(input_batch), True)
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
    _test_forward_and_backward(q, k, v, bias, ref_fn=ref_fn, test_fn=test_fn)


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
    input_batch = dict(query=q, key=k, value=v, bias=bias)
    chex.assert_equal(test_fn.is_supported(input_batch), True)
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
            dict(
                query=jnp.zeros(qkv_shape, dtype=dtype),
                key=jnp.zeros(qkv_shape, dtype=dtype),
                value=jnp.broadcast_to(
                    jnp.eye(per_head_dim, dtype=dtype)[None, :, None], qkv_shape
                ),
                bias=bias,
            ),
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
    input_batch = dict(query=q, key=k, value=v, bias=bias)
    chex.assert_equal(test_fn.is_supported(input_batch), True)

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
        (1, 1, 378, 676, 72),
        (2, 4, 582, 582, 56),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_cudnn_seqlen_head_support(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    kv_seq_len: int,
    per_head_dim: int,
    causal: bool,
    dtype: jnp.dtype,
):
    """Tests that cudnn supports any even sequence length and head dim % 8 == 0."""
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
    input_batch = dict(query=q, key=k, value=v, bias=bias)
    chex.assert_equal(test_fn.is_supported(input_batch), True)

    _test_forward_and_backward(
        q, k, v, bias, ref_fn=ref_fn, test_fn=test_fn, forward_tol_fn=_cudnn_xla_forward_tol_fn
    )


def test_cudnn_dropout_determinism():
    """Tests that cuDNN dropout produces identical outputs across runs."""
    if jax.default_backend() == "cpu":
        pytest.skip(reason="cudnn function needs GPU.")
    q, k, v, bias = generate_attention_data(*(1, 128, 128, 2, 64))
    input_batch = dict(
        query=q,
        key=k,
        value=v,
        bias=bias,
    )
    fn = CuDNNGPUFlashAttention.default_config().set(dropout_rate=0.1).instantiate()

    outputs = []
    grads = []

    def grad_fn(q, k, v, bias):
        input_batch = dict(
            query=q,
            key=k,
            value=v,
            bias=bias,
        )
        return fn(input_batch).mean()

    for i in range(10):
        outputs.append(fn(input_batch))
        grads.append(jax.grad(grad_fn, argnums=(0, 1, 2))(q, k, v, bias))

    jax.clear_caches()

    for i in range(10):
        chex.assert_trees_all_equal(fn(input_batch), outputs[i])
        chex.assert_trees_all_equal(jax.grad(grad_fn, argnums=(0, 1, 2))(q, k, v, bias), grads[i])

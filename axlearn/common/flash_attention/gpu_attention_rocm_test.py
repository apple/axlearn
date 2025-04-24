# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# jax-ml/jax-triton:
# Copyright 2023 The jax_triton Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests GPU FlashAttention kernels.

Currently tested on MI300. To run tests in parallel on a multi-GPU machine, use this:
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
    ROCmTransformerEngineFlashAttention,
)
from axlearn.common.flash_attention.test_utils import generate_attention_data
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
    prng_key = jax.random.PRNGKey(44)
    jax_out = test_fn(q, k, v, bias, prng_key)
    jax_ref_out = ref_fn(q, k, v, bias, prng_key)
    backend = jax.default_backend()
    chex.assert_trees_all_close(jax_out, jax_ref_out, **forward_tol_fn(backend, q.dtype))

    # Compare gradients.
    jax_grads = jax.grad(lambda *args: ref_fn(*args).mean(), argnums=(0, 1, 2))(
        q, k, v, bias, prng_key
    )
    jax_ref_grads = jax.grad(lambda *args: test_fn(*args).mean(), argnums=(0, 1, 2))(
        q, k, v, bias, prng_key
    )
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


# We test the ROCm TE DotProductAttention against the reference flash_attention.
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
def test_rocmte_against_xla_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    causal: bool,
    dtype: jnp.dtype,
):
    if jax.default_backend() == "cpu":
        pytest.skip(reason="ROCm function needs GPU.")

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
    test_fn = ROCmTransformerEngineFlashAttention.default_config().set(**cfg).instantiate()
    chex.assert_equal(test_fn.is_supported(query=q, key=k, value=v, bias=bias), True)
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
    "batch_size,num_heads,seq_len,kv_seq_len,per_head_dim",
    [
        (1, 1, 378, 676, 72),
        (2, 4, 582, 582, 56),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_rocmte_seqlen_head_support(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    kv_seq_len: int,
    per_head_dim: int,
    causal: bool,
    dtype: jnp.dtype,
):
    """Tests that ROCm TE supports any even sequence length and head dim % 8 == 0."""
    if jax.default_backend() == "cpu":
        pytest.skip(reason="ROCm function needs GPU.")
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
    test_fn = ROCmTransformerEngineFlashAttention.default_config().set(**cfg).instantiate()
    ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
    chex.assert_equal(test_fn.is_supported(query=q, key=k, value=v, bias=bias), True)

    _test_forward_and_backward(
        q, k, v, bias, ref_fn=ref_fn, test_fn=test_fn, forward_tol_fn=_cudnn_xla_forward_tol_fn
    )

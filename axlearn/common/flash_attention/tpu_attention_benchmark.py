# Copyright © 2023 Apple Inc.

"""Benchmark TPU FlashAttention kernels.

Sample outputs: (v5p)
CMD: python \
/opt/venv/lib/python3.10/site-packages/axlearn/common/flash_attention/tpu_attention_benchmark.py \
2>&1 | grep -E "Benchmarking|ref_|HBM usage"

Benchmarking attention representative of 1.2b model layer on TPU v5.
ref_fwd:0.2291s, flash_fwd:0.0014s
ref_bwd:0.0217s, flash_bwd:0.0058s
Benchmarking attention representative of 12.6b model layer on TPU v5.
ref_fwd:0.5699s, flash_fwd:0.0032s
ref_bwd:0.0524s, flash_bwd:0.0152s
Benchmarking attention representative of 29.6b model layer on TPU v5.
ref_fwd:0.7957s, flash_fwd:0.0043s
ref_bwd:0.0731s, flash_bwd:0.0204s
Benchmarking attention representative of 65.2b model layer on TPU v5.
ref_fwd:1.0225s, flash_fwd:0.0055s
ref_bwd:0.0948s, flash_bwd:0.0262s
Benchmarking attention representative of 134b model layer on TPU v5.
ref_fwd:1.2485s, flash_fwd:0.0067s
ref_bwd:0.1159s, flash_bwd:0.0313s
Benchmarking attention representative of 261.7b model layer on TPU v5.
ref_fwd:1.5577s, flash_fwd:0.0072s
ref_bwd:0.1349s, flash_bwd:0.0373s
"""
import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from axlearn.common.attention_bias import (
    CompositeAttentionBias,
    MaskFnAttentionBias,
    TensorAttentionBias,
    causal_mask,
    sliding_window_causal_mask,
)
from axlearn.common.flash_attention.utils import flash_attention_implementation, mha_reference

_BENCHMARK_CONFIGS = {
    "1.2b": dict(
        num_heads=16,
        per_head_dim=128,
    ),
    "12.6b": dict(
        num_heads=40,
        per_head_dim=128,
    ),
    "29.6b": dict(
        num_heads=56,
        per_head_dim=128,
    ),
    "65.2b": dict(
        num_heads=72,
        per_head_dim=128,
    ),
    "134b": dict(
        num_heads=88,
        per_head_dim=128,
    ),
    "261.7b": dict(
        num_heads=110,
        per_head_dim=128,
    ),
    # OOM in mha_reference.
    # "539.5b": dict(
    #     num_heads=140,
    #     per_head_dim=128,
    # ),
}


def _time_call(fn: Callable, *, num_iters: int = 5) -> float:
    """Times average execution time for fn call over num_iters after warmup."""
    fn().block_until_ready()
    tic = time.perf_counter()
    for _ in range(num_iters):
        fn().block_until_ready()
    toc = time.perf_counter()
    return (toc - tic) / num_iters


def _benchmark(
    *,
    batch_size: int,
    seq_len: int,
    block_size: int,
    num_heads: int,
    per_head_dim: int,
    causal: bool = True,
    use_bias: bool = False,
    use_segment_ids: bool = False,
    sliding_window_size: Optional[int] = None,
):
    """Benchmarks TPU FlashAttention vs reference impl."""
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)

    bias = None
    if use_bias:
        bias = jax.random.normal(k4, (batch_size, num_heads, seq_len, seq_len), dtype=jnp.bfloat16)
    segment_ids = None
    if use_segment_ids:
        segment_ids = jnp.cumsum(
            jax.random.bernoulli(k5, shape=(batch_size, seq_len)).astype(jnp.int32), axis=1
        )

    softmax_scale = per_head_dim**-0.5
    ref_fwd_time = _time_call(
        lambda: mha_reference(
            q, k, v, bias, segment_ids, causal=causal, softmax_scale=softmax_scale
        )
    )

    grad_fn = jax.jit(
        jax.grad(
            lambda q, k, v, b, s: mha_reference(
                q, k, v, b, s, causal=causal, softmax_scale=softmax_scale
            ).mean(),
            argnums=(0, 1, 2),
        )
    )
    ref_bwd_time = _time_call(lambda: grad_fn(q, k, v, bias, segment_ids)[0])

    mask = None
    if causal and sliding_window_size is None:
        mask = causal_mask
    elif causal:
        mask = sliding_window_causal_mask(sliding_window_size)
    mask = MaskFnAttentionBias(mask, shape=(seq_len, seq_len))
    if use_bias:
        bias = CompositeAttentionBias([mask, TensorAttentionBias(bias)])
    else:
        bias = CompositeAttentionBias([mask])

    # Get fwd & bwd timing information when softmax scaling applied before calling the kernel.
    mha_impl = flash_attention_implementation(
        "tpu", softmax_scale=softmax_scale, block_size=block_size
    )

    flash_fwd_time = _time_call(lambda: mha_impl(q, k, v, bias))

    flash_grad_fn = jax.jit(
        jax.grad(lambda q, k, v, b: mha_impl(q, k, v, b).mean(), argnums=(0, 1, 2))
    )
    flash_bwd_time = _time_call(lambda: flash_grad_fn(q, k, v, bias)[0])

    print(f"ref_fwd:{ref_fwd_time:.4f}s, flash_fwd:{flash_fwd_time:.4f}s")
    print(f"ref_bwd:{ref_bwd_time:.4f}s, flash_bwd:{flash_bwd_time:.4f}s\n")


if __name__ == "__main__":
    assert jax.default_backend() == "tpu", "Benchmarking requires a TPU backend."
    device_kind = jax.devices()[0].device_kind
    for name, cfg in _BENCHMARK_CONFIGS.items():
        print(f"Benchmarking attention representative of {name} model layer on {device_kind}.")
        _benchmark(
            batch_size=2,
            seq_len=1024 * 8,
            block_size=4 * 128,
            sliding_window_size=1024,
            **cfg,
        )

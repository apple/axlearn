# Copyright © 2023 Apple Inc.

"""Benchmark TPU FlashAttention kernels.

Sample outputs:

Benchmarking attention representative of 1.2b model layer on TPU v4.
ref_fwd:0.0030s, flash_fwd:0.0012s
ref_bwd:0.0075s, flash_bwd:0.0054s

Benchmarking attention representative of 12.6b model layer on TPU v4.
ref_fwd:0.0043s, flash_fwd:0.0016s
ref_bwd:0.0098s, flash_bwd:0.0071s

Benchmarking attention representative of 29.6b model layer on TPU v4.
ref_fwd:0.0060s, flash_fwd:0.0022s
ref_bwd:0.0135s, flash_bwd:0.0101s

Benchmarking attention representative of 65.2b model layer on TPU v4.
ref_fwd:0.0077s, flash_fwd:0.0028s
ref_bwd:0.0176s, flash_bwd:0.0130s

Benchmarking attention representative of 134b model layer on TPU v4.
ref_fwd:0.0094s, flash_fwd:0.0035s
ref_bwd:0.0216s, flash_bwd:0.0158s

Benchmarking attention representative of 261.7b model layer on TPU v4.
ref_fwd:0.0105s, flash_fwd:0.0035s
ref_bwd:0.0254s, flash_bwd:0.0182s

Benchmarking attention representative of 539.5b model layer on TPU v4.
ref_fwd:0.0134s, flash_fwd:0.0047s
ref_bwd:0.0324s, flash_bwd:0.0237s
"""
import time
from typing import Callable

import jax
import jax.numpy as jnp

from axlearn.common.flash_attention.utils import flash_attention_implementation, mha_reference

_BENCHMARK_CONFIGS = {
    "1.2b": dict(
        num_heads=32,
        per_head_dim=64,
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
    "539.5b": dict(
        num_heads=140,
        per_head_dim=128,
    ),
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
):
    """Benchmarks TPU FlashAttention vs reference impl."""
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
    bias = jax.random.normal(k4, (batch_size, num_heads, seq_len, seq_len), dtype=jnp.bfloat16)
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

    # Get fwd & bwd timing information when softmax scaling applied before calling the kernel.
    mha_impl = flash_attention_implementation(
        "tpu", causal=causal, softmax_scale=softmax_scale, block_size=block_size
    )

    flash_fwd_time = _time_call(lambda: mha_impl(q, k, v, bias, segment_ids))

    flash_grad_fn = jax.jit(
        jax.grad(lambda q, k, v, b, s: mha_impl(q, k, v, b, s).mean(), argnums=(0, 1, 2))
    )
    flash_bwd_time = _time_call(lambda: flash_grad_fn(q, k, v, bias, segment_ids)[0])

    print(f"ref_fwd:{ref_fwd_time:.4f}s, flash_fwd:{flash_fwd_time:.4f}s")
    print(f"ref_bwd:{ref_bwd_time:.4f}s, flash_bwd:{flash_bwd_time:.4f}s\n")


if __name__ == "__main__":
    assert jax.default_backend() == "tpu", "Benchmarking requires a TPU backend."
    device_kind = jax.devices()[0].device_kind
    for name, cfg in _BENCHMARK_CONFIGS.items():
        print(f"Benchmarking attention representative of {name} model layer on {device_kind}.")
        _benchmark(
            batch_size=2,
            seq_len=2048,
            block_size=4 * 128,
            **cfg,
        )

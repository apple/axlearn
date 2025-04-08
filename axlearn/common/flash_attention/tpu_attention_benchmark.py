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

from axlearn.common.attention_bias import causal_mask
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.test_utils import generate_attention_data
from axlearn.common.flash_attention.utils import flash_attention_implementation

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
        num_heads=8,
        num_kv_heads=1,
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
    num_kv_heads: Optional[int] = None,
    is_decoding: bool = False,
    causal: bool = True,
    use_bias: bool = False,
    sliding_window_size: Optional[int] = None,
):
    """Benchmarks TPU FlashAttention vs reference impl."""
    q, k, v, bias = generate_attention_data(
        batch_size,
        1 if is_decoding else seq_len,
        seq_len,
        num_heads,
        per_head_dim,
        num_kv_heads or num_heads,
        mask_fn=causal_mask if causal and not sliding_window_size else None,
        sliding_window_sz=sliding_window_size,
        attention_bias_type="4d" if use_bias else None,
        query_offset=seq_len - 1 if is_decoding else 0,
    )
    softmax_scale = q.shape[-1] ** 0.5
    # Get fwd & bwd timing information when softmax scaling applied before calling the kernel.
    ref_mha_impl = (
        ReferenceMHA.default_config()
        .set(softmax_scale=softmax_scale, tpu_block_size=block_size, is_decoding=is_decoding)
        .instantiate()
    )
    mha_impl = flash_attention_implementation(
        "tpu",
        query=q,
        key=k,
        value=v,
        bias=bias,
        softmax_scale=softmax_scale,
        tpu_block_size=block_size,
        is_decoding=is_decoding,
    )

    ref_fwd_time = _time_call(lambda: ref_mha_impl(q, k, v, bias))
    flash_fwd_time = _time_call(lambda: mha_impl(q, k, v, bias))

    if not is_decoding:
        flash_grad_fn = jax.jit(
            jax.grad(lambda q, k, v, b: ref_mha_impl(q, k, v, b).mean(), argnums=(0, 1, 2))
        )
        ref_bwd_time = _time_call(lambda: flash_grad_fn(q, k, v, bias)[0])
        flash_grad_fn = jax.jit(
            jax.grad(lambda q, k, v, b: mha_impl(q, k, v, b).mean(), argnums=(0, 1, 2))
        )
        flash_bwd_time = _time_call(lambda: flash_grad_fn(q, k, v, bias)[0])

    print(f"ref_fwd:{ref_fwd_time:.4f}s, flash_fwd:{flash_fwd_time:.4f}s")
    if not is_decoding:
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
            sliding_window_size=4096,
            is_decoding=False,
            **cfg,
        )

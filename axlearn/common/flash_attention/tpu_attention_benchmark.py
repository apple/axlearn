# Copyright © 2023 Apple Inc.

"""Benchmark TPU FlashAttention kernels.
For fast running kernel, python benchmark time may not be accurate.
Therefore, we tyically would enable jax profiler checking trace.
Example run:
    python tpu_attention_benchmark.py
            --config=134b \
            --run_reference=False \
            --enable_trace_profiling=True \
            --batch_size=8 \
            --seq_len=65536 \
            --page_size=128
    Traced log dir can be specified with --trace-dir.

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
from absl import app, flags

from axlearn.common.attention_bias import causal_mask
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.test_utils import (
    generate_attention_data,
    generate_paged_attention_data,
)
from axlearn.common.flash_attention.utils import flash_attention_implementation
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.kv_cache.paged_kv_cache import PagedKVCache

flags.DEFINE_boolean("enable_trace_profiling", False, "Whether to enable JAX trace profiling.")
flags.DEFINE_string("trace_dir", "/tmp/axlearn_profiler", "Directory to save profiler traces.")
flags.DEFINE_boolean("run_reference", True, "Whether to run the reference implementation.")
flags.DEFINE_string("config", None, "Configuration to run benchmark.")
flags.DEFINE_integer("batch_size", 2, "Batch size for running benchmark.")
flags.DEFINE_integer("seq_len", 8192, "Sequence length to run benchmark.")
flags.DEFINE_integer("page_size", None, "Page size for paged attention.")
flags.DEFINE_integer("sliding_window_size", None, "Sliding window size for attention.")
FLAGS = flags.FLAGS


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
    "539.5b": dict(
        num_heads=140,
        per_head_dim=128,
    ),
}


def _time_call(fn: Callable, *, num_iters: int = 5) -> float:
    """Times average execution time for fn call over num_iters after warmup."""
    fn().block_until_ready()
    tic = time.perf_counter()
    if FLAGS.enable_trace_profiling:
        jax.profiler.start_trace(FLAGS.trace_dir)
    for _ in range(num_iters):
        fn().block_until_ready()
    if FLAGS.enable_trace_profiling:
        jax.profiler.stop_trace()
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
    kv_cache_type: Optional[type[BaseKVCache]] = KVCache,
    causal: bool = True,
    use_bias: bool = False,
    sliding_window_size: Optional[int] = None,
    page_size: Optional[int] = None,
):
    """Benchmarks TPU FlashAttention vs reference impl."""
    if not (kv_cache_type is None or kv_cache_type in (KVCache, PagedKVCache)):
        raise NotImplementedError(f"This benchmark doesn't support {kv_cache_type=} yet.")
    if page_size is not None:
        kv_cache_type = PagedKVCache
        assert page_size is not None
        q, k, v, page_tables, bias = generate_paged_attention_data(
            batch_size=batch_size,
            query_len=1,
            kv_len=seq_len,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
            page_size=page_size,
            num_kv_heads=num_kv_heads or num_heads,
            mask_fn=causal_mask if causal and not sliding_window_size else None,
            sliding_window_sz=sliding_window_size,
            attention_bias_type="4d" if use_bias else None,
            query_offset=seq_len - 1,
        )
    else:
        is_decoding = kv_cache_type == KVCache
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
        page_tables = None

    softmax_scale = q.shape[-1] ** 0.5
    # Get fwd & bwd timing information when softmax scaling applied before calling the kernel.
    ref_mha_impl = (
        ReferenceMHA.default_config()
        .set(softmax_scale=softmax_scale, tpu_block_size=block_size)
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
        kv_cache_type=kv_cache_type,
        page_tables=page_tables,
    )

    input_batch = dict(query=q, key=k, value=v, bias=bias, page_tables=page_tables)
    if FLAGS.run_reference:
        ref_mha_impl = (
            ReferenceMHA.default_config()
            .set(softmax_scale=softmax_scale, tpu_block_size=block_size)
            .instantiate()
        )
        ref_fwd_time = _time_call(lambda: ref_mha_impl(input_batch))
        print(f"ref_fwd: {ref_fwd_time:.4f}s")

    flash_fwd_time = _time_call(lambda: mha_impl(input_batch))
    print(f"flash_fwd:{flash_fwd_time:.4f}s")

    if kv_cache_type is None:

        def grad_test(float_inputs, aux_inputs):
            full_batch = {**float_inputs, **aux_inputs}
            return mha_impl(full_batch).mean()

        if FLAGS.run_reference:

            def grad_ref(float_inputs, aux_inputs):
                full_batch = {**float_inputs, **aux_inputs}
                return ref_mha_impl(full_batch).mean()

            ref_grad_fn = jax.jit(jax.grad(grad_ref, argnums=0))
            ref_bwd_time = _time_call(lambda: ref_grad_fn(float_inputs, aux_inputs)["query"])
            print(f"ref_bwd:{ref_bwd_time:.4f}s")

        float_inputs = dict(query=q, key=k, value=v)
        aux_inputs = dict(bias=bias)
        flash_grad_fn = jax.jit(jax.grad(grad_test, argnums=0))
        flash_bwd_time = _time_call(lambda: flash_grad_fn(float_inputs, aux_inputs)["query"])
        print(f"flash_bwd:{flash_bwd_time:.4f}s")


def main(_):
    # Check if TPU backend is available
    if jax.default_backend() != "tpu":
        print(f"Skipping TPU benchmarks: backend is {jax.default_backend()}, not 'tpu'")
        return

    device_kind = jax.devices()[0].device_kind
    if FLAGS.config is not None:
        config_list = [(FLAGS.config, _BENCHMARK_CONFIGS[FLAGS.config])]
    else:
        # Run Sweep without benchmarking 539.5b
        config_list = [(k, v) for k, v in _BENCHMARK_CONFIGS.items() if k != "539.5b"]
    for name, cfg in config_list:
        print(f"Benchmarking attention representative of {name} model layer on {device_kind}.")
        _benchmark(
            batch_size=FLAGS.batch_size,
            seq_len=FLAGS.seq_len,
            block_size=4 * 128,
            sliding_window_size=FLAGS.sliding_window_size,
            page_size=FLAGS.page_size,
            kv_cache_type=KVCache,
            **cfg,
        )


if __name__ == "__main__":
    app.run(main)

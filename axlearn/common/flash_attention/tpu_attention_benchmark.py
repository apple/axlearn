# Copyright © 2023 Apple Inc.

"""Benchmark TPU FlashAttention kernels.
For fast running kernel, python benchmark time may not be accurate.
Therefore, we typically would enable jax profiler checking trace.
Example run:
    python tpu_attention_benchmark.py
    python tpu_attention_benchmark.py --is_decoding=true
"""
import time
from typing import Callable, Optional

import jax
from absl import app, flags
from jaxlib.xla_client import XlaRuntimeError

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
flags.DEFINE_boolean("run_reference", False, "Whether to run the reference implementation.")
flags.DEFINE_integer("batch_size", 2, "Batch size for running benchmark.")
flags.DEFINE_boolean("is_decoding", False, "Whether to run decoding (single query) benchmark.")
flags.DEFINE_integer(
    "page_size", None, "Page size for paged attention (requires is_decoding=True)."
)
flags.DEFINE_integer("sliding_window_size", None, "Sliding window size for attention.")
FLAGS = flags.FLAGS


_BENCHMARK_CONFIGS = {
    "heads=16, dim=128": dict(
        num_heads=16,
        per_head_dim=128,
    ),
    "heads=8, kv_heads=1, dim=512": dict(
        num_heads=8,
        num_kv_heads=1,
        per_head_dim=512,
    ),
}


def _time_call(fn: Callable, *, num_iters: int = 10, warmup: int = 3) -> float:
    """Times average execution time for fn call over num_iters after warmup."""
    for _ in range(warmup):
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
    """Benchmarks TPU FlashAttention vs reference impl.

    Args:
        batch_size: Batch size for the benchmark.
        seq_len: Sequence length.
        block_size: Block size for attention computation.
        num_heads: Number of attention heads.
        per_head_dim: Dimension per head.
        num_kv_heads: Number of key/value heads (for GQA). Defaults to num_heads.
        kv_cache_type: Type of KV cache to use (KVCache, PagedKVCache, or None for prefill).
        causal: Whether to use causal masking.
        use_bias: Whether to use attention bias.
        sliding_window_size: Size of sliding window for local attention.
        page_size: Page size for paged attention (only valid with PagedKVCache).

    Returns:
        A dict containing timing results with keys: 'flash_fwd', 'flash_bwd',
        and optionally 'ref_fwd', 'ref_bwd' if run_reference is True.
    """
    if not (kv_cache_type is None or kv_cache_type in (KVCache, PagedKVCache)):
        raise NotImplementedError(f"This benchmark doesn't support {kv_cache_type=} yet.")

    is_decoding = kv_cache_type in (KVCache, PagedKVCache)

    if kv_cache_type == PagedKVCache:
        if page_size is None:
            raise ValueError("page_size must be provided when using PagedKVCache")
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
    if mha_impl is None:
        raise ValueError("Attention implementation is not available.")

    input_batch = dict(
        query=q, key=k, value=v, bias=bias, page_tables=page_tables, prng_key=jax.random.PRNGKey(42)
    )
    results = {}

    if FLAGS.run_reference:
        ref_mha_impl = (
            ReferenceMHA.default_config()
            .set(softmax_scale=softmax_scale, tpu_block_size=block_size)
            .instantiate()
        )
        ref_fwd_time = _time_call(lambda: ref_mha_impl(input_batch))
        results["ref_fwd"] = ref_fwd_time

    flash_fwd_time = _time_call(lambda: mha_impl.fn(input_batch))
    results["flash_fwd"] = flash_fwd_time

    if not is_decoding:
        float_inputs = dict(query=q, key=k, value=v)
        aux_inputs = dict(bias=bias, prng_key=jax.random.PRNGKey(42))

        def grad_test(float_inputs, aux_inputs):
            full_batch = {**float_inputs, **aux_inputs}
            return mha_impl.fn(full_batch).mean()

        if FLAGS.run_reference:

            def grad_ref(grad_inputs):
                return ref_mha_impl({**grad_inputs, **aux_inputs}).mean()

            ref_grad_fn = jax.jit(jax.value_and_grad(grad_ref))
            ref_bwd_time = _time_call(lambda: ref_grad_fn(float_inputs)[1]["query"])
            results["ref_bwd"] = ref_bwd_time

        flash_grad_fn = jax.jit(jax.value_and_grad(grad_test))
        flash_bwd_time = _time_call(lambda: flash_grad_fn(float_inputs, aux_inputs)[1]["query"])
        results["flash_bwd"] = flash_bwd_time

    return results


def _print_summary(all_results: dict):
    """Prints a formatted summary of benchmark results.

    Args:
        all_results: Dictionary mapping config names to their benchmark results.
            Each result should contain 'seq_lengths', 'flash_fwd', 'flash_bwd',
            'ref_fwd', and 'ref_bwd' lists.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    for name, config_results in all_results.items():
        print(f"\nBenchmarking attention representative of {name} model layer.")
        for i, seq_len in enumerate(config_results["seq_lengths"]):
            print(f"  seq_len={seq_len}:")
            if config_results["ref_fwd"] and config_results["ref_fwd"][i] > 0:
                print(
                    f"    ref_fwd:{config_results['ref_fwd'][i]:.4f}s, "
                    f"flash_fwd:{config_results['flash_fwd'][i]:.4f}s"
                )
            else:
                print(f"    flash_fwd:{config_results['flash_fwd'][i]:.4f}s")
            if config_results["ref_bwd"] and config_results["ref_bwd"][i] > 0:
                print(
                    f"    ref_bwd:{config_results['ref_bwd'][i]:.4f}s, "
                    f"flash_bwd:{config_results['flash_bwd'][i]:.4f}s"
                )
            elif config_results["flash_bwd"] and config_results["flash_bwd"][i] > 0:
                print(f"    flash_bwd:{config_results['flash_bwd'][i]:.4f}s")

    print("\n" + "=" * 80)
    print("Benchmarking complete!")
    print("=" * 80)


def main(_):
    # Check if TPU backend is available
    if jax.default_backend() != "tpu":
        print(f"Skipping TPU benchmarks: backend is {jax.default_backend()}, not 'tpu'")
        return

    # Validate flag combinations
    if FLAGS.page_size is not None and not FLAGS.is_decoding:
        raise ValueError("page_size can only be used with is_decoding=True")

    # Determine kv_cache_type based on flags
    if FLAGS.is_decoding:
        kv_cache_type = PagedKVCache if FLAGS.page_size is not None else KVCache
    else:
        kv_cache_type = None

    config_list = list(_BENCHMARK_CONFIGS.items())

    seq_lengths = [2048, 4096, 8192, 16384, 32768]

    # Collect all results for each config
    all_results = {}
    for name, cfg in config_list:
        print(f"\nRunning benchmarks for {name} model configuration...")
        config_results = {
            "seq_lengths": [],
            "flash_fwd": [],
            "flash_bwd": [],
            "ref_fwd": [],
            "ref_bwd": [],
        }

        for seq_len in seq_lengths:
            print(f"  Testing seq_len={seq_len}...")
            try:
                results = _benchmark(
                    batch_size=FLAGS.batch_size,
                    seq_len=seq_len,
                    block_size=4 * 128,
                    sliding_window_size=FLAGS.sliding_window_size,
                    page_size=FLAGS.page_size,
                    kv_cache_type=kv_cache_type,
                    **cfg,
                )
                config_results["seq_lengths"].append(seq_len)
                config_results["flash_fwd"].append(results.get("flash_fwd", 0))
                config_results["flash_bwd"].append(results.get("flash_bwd", 0))
                config_results["ref_fwd"].append(results.get("ref_fwd", 0))
                config_results["ref_bwd"].append(results.get("ref_bwd", 0))
            # pylint: disable-next=broad-exception-caught
            except XlaRuntimeError as e:
                print(f"    Skipping seq_len={seq_len} due to error: {e}")
                continue

        all_results[name] = config_results

    # Print all results at the end
    _print_summary(all_results)


if __name__ == "__main__":
    app.run(main)

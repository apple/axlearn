# Copyright © 2024 Apple Inc.

"""Benchmark Frontend.

1) frontend_utils.frame benchmark
* TPU v4
  * frame_time: 0.0686s.
  * Note: before pull/807, frame_time: 1.7839s
* CPU
  * frame_time:0.2298s
  * Note: before pull/807, frame_time: 0.1403s
"""

import functools
import time
from typing import Callable

import jax

from axlearn.audio import frontend_utils, test_utils

_BENCHMARK_CONFIGS = {
    "b32s600": dict(
        batch_size=32,
        seq_secs=600,
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
    seq_secs: int,
):
    """Benchmarks TPU FlashAttention vs reference impl."""
    sample_rate = 16_000
    frame_size_ms = 25
    frame_step_ms = 10
    frame_size = frontend_utils.ms_to_samples(frame_size_ms, sample_rate=sample_rate)
    frame_step = frontend_utils.ms_to_samples(frame_step_ms, sample_rate=sample_rate)
    seq_len = seq_secs * sample_rate
    inputs, _ = test_utils.fake_audio(
        prng_key=jax.random.PRNGKey(123), batch_size=batch_size, seq_len=seq_len
    )

    fn = functools.partial(
        frontend_utils.frame, x=inputs, frame_size=frame_size, hop_size=frame_step
    )
    fn = jax.jit(fn)
    frame_time = _time_call(fn)

    print(f"frame_time: {frame_time:.4f}s.")


if __name__ == "__main__":
    print(f"Benchmarking on {jax.default_backend()} backend.")
    device_kind = jax.devices()[0].device_kind
    for name, cfg in _BENCHMARK_CONFIGS.items():
        print(f"Benchmarking frame() representative of {name} config on {device_kind}.")
        _benchmark(**cfg)

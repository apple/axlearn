# Copyright © 2024 Apple Inc.

"""Benchmark Frontend.

> python -m axlearn.audio.frontend_benchmark

ChunkSize1 and ChunkSize80 are important. ChunkSize80 corresponds to a frame size of 25ms and
hop size of 10ms at 16kHz (for both Logmel and STFT), while ChunkSize1 represents Logmel with
pre_emphasis=0.97 (or not None).

A notable point is that while GPU is consistently fast, TPU is more than 10 times slower than GPU.
Especially at ChunkSize1, the performance is so slow that it almost seems like a bug.

TPU v5p
ChunkSize1: frame_time 104.716ms
ChunkSize2: frame_time 91.805ms
ChunkSize4: frame_time 58.432ms
ChunkSize5: frame_time 46.348ms
ChunkSize8: frame_time 38.428ms
ChunkSize10: frame_time 35.586ms
ChunkSize16: frame_time 22.548ms
ChunkSize20: frame_time 21.454ms
ChunkSize32: frame_time 11.166ms
ChunkSize40: frame_time 11.875ms
ChunkSize80: frame_time 1.276ms
ChunkSize160: frame_time 0.976ms

GPU H100
ChunkSize1: frame_time 0.276ms
ChunkSize2: frame_time 0.249ms
ChunkSize4: frame_time 0.157ms
ChunkSize5: frame_time 0.186ms
ChunkSize8: frame_time 0.109ms
ChunkSize10: frame_time 0.142ms
ChunkSize16: frame_time 0.094ms
ChunkSize20: frame_time 0.101ms
ChunkSize32: frame_time 0.090ms
ChunkSize40: frame_time 0.096ms
ChunkSize80: frame_time 0.087ms
ChunkSize160: frame_time 0.096ms

CPU
ChunkSize1: frame_time 4.576ms
ChunkSize2: frame_time 4.437ms
ChunkSize4: frame_time 5.087ms
ChunkSize5: frame_time 4.672ms
ChunkSize8: frame_time 3.846ms
ChunkSize10: frame_time 4.572ms
ChunkSize16: frame_time 4.287ms
ChunkSize20: frame_time 4.755ms
ChunkSize32: frame_time 3.966ms
ChunkSize40: frame_time 3.988ms
ChunkSize80: frame_time 3.823ms
ChunkSize160: frame_time 4.526ms
"""

import functools
import time
from typing import Callable

import jax

from axlearn.audio import frontend_utils, test_utils

# Map frame_offset to chunk_size by `[(i, np.gcd(160, 400 + i)) for i in range(81)]`.
_BENCHMARK_CONFIGS = {
    "ChunkSize1": dict(batch_size=8, seq_secs=120, frame_offset=1),
    "ChunkSize2": dict(batch_size=8, seq_secs=120, frame_offset=2),
    "ChunkSize4": dict(batch_size=8, seq_secs=120, frame_offset=4),
    "ChunkSize5": dict(batch_size=8, seq_secs=120, frame_offset=5),
    "ChunkSize8": dict(batch_size=8, seq_secs=120, frame_offset=8),
    "ChunkSize10": dict(batch_size=8, seq_secs=120, frame_offset=10),
    "ChunkSize16": dict(batch_size=8, seq_secs=120, frame_offset=32),
    "ChunkSize20": dict(batch_size=8, seq_secs=120, frame_offset=20),
    "ChunkSize32": dict(batch_size=8, seq_secs=120, frame_offset=16),
    "ChunkSize40": dict(batch_size=8, seq_secs=120, frame_offset=40),
    "ChunkSize80": dict(batch_size=8, seq_secs=120, frame_offset=0),
    "ChunkSize160": dict(batch_size=8, seq_secs=120, frame_offset=80),
}


def _time_call(fn: Callable, *, num_iters: int = 20) -> float:
    """Times average execution time for fn call over num_iters after warmup."""
    fn().block_until_ready()
    tic = time.perf_counter()
    for _ in range(num_iters - 1):
        fn()
    fn().block_until_ready()
    toc = time.perf_counter()
    return (toc - tic) / num_iters


def _benchmark(exp: str, *, batch_size: int, seq_secs: int, frame_offset: int):
    """Benchmarks TPU FlashAttention vs reference impl."""
    sample_rate = 16_000
    frame_size_ms = 25
    hop_size_ms = 10
    frame_size = frontend_utils.ms_to_samples(frame_size_ms, sample_rate=sample_rate)
    # For Logmel, frame_offset=0 imitates pre_emphasis=None, and 1 does pre_emphasis=0.97.
    frame_size += frame_offset
    hop_size = frontend_utils.ms_to_samples(hop_size_ms, sample_rate=sample_rate)
    seq_len = seq_secs * sample_rate
    inputs, _ = test_utils.fake_audio(
        prng_key=jax.random.PRNGKey(123), batch_size=batch_size, seq_len=seq_len
    )

    fn = functools.partial(frontend_utils.frame, x=inputs, frame_size=frame_size, hop_size=hop_size)
    fn = jax.jit(fn)
    frame_time = _time_call(fn)
    print(f"{exp}: frame_time {frame_time * 1000:.3f}ms")


if __name__ == "__main__":
    print(f"Benchmarking on {jax.default_backend()} backend.")
    for name, cfg in _BENCHMARK_CONFIGS.items():
        _benchmark(name, **cfg)

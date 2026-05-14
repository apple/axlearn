"""Benchmark online_softmax_log_probs (Pallas) vs full matmul baseline.

Compares throughput (tokens/sec) between:
  - Baseline at BS_BASELINE (materializes full [B,S,V] logits)
  - Pallas kernel at BS_ONLINE_SOFTMAX (fused in VMEM, no HBM logits)

Usage:
    CPU machine:
    bazel run axlearn/common/kernels:online_softmax_log_probs_benchmark

    TPU machine:
    bazel run --define=tpu=true //axlearn/common/kernels:online_softmax_log_probs_benchmark

Note: CPU benchmarks use Pallas interpret mode (slow, not representative of TPU
performance). For accurate numbers, run this script on TPU machine with PALLAS_INTERPRET=False.

Here is the Benchmark result on TPU:
====================================================================
THROUGHPUT COMPARISON
====================================================================

Method                            Latency   Tokens      Throughput
--------------------------------------------------------------------
Baseline (BS=1)                    52.1 ms      512        9,828 tok/s
Baseline top1 (BS=1)               52.3 ms      512        9,798 tok/s
Pallas topk (BS=16)               232.8 ms     8192       35,194 tok/s
Pallas top1 (BS=16)               215.1 ms     8192       38,088 tok/s

Pallas topk vs Baseline topk: 3.58x throughput
Pallas top1 vs Baseline top1: 3.89x throughput
"""

import time
from typing import Callable

import jax
import jax.numpy as jnp
from absl import app

from axlearn.common.kernels.online_softmax_log_probs import online_softmax_log_probs_pallas

# ---------- Configuration ----------
# Use smaller dimensions for CPU. For TPU benchmarking, try larget number.
S, H, V = 512, 256, 8192
TOP_K = 8
N_ITERS = 5

# BS = 1 reflect the existing baseline setup.
BS_BASELINE = 1
BS_ONLINE_SOFTMAX = 16

# Pallas tile sizes.
TILE_S = 128
TILE_V = 1024

# Set to False on TPU (interpret mode is for CPU testing only).
PALLAS_INTERPRET = False
# ------------------------------------


def baseline(x, weight, target_ids, top_k):
    """Baseline: full matmul + log_softmax + gather + top_k."""
    logits = jnp.einsum("bsh,vh->bsv", x, weight)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_lp = jnp.take_along_axis(log_probs, target_ids[:, :, None], axis=-1).squeeze(-1)
    topk_vals, topk_idx = jax.lax.top_k(log_probs, top_k)
    return target_lp, topk_vals, topk_idx


# TODO (muyang_yu): change this to Google Benchmark
def bench(fn: Callable, n: int = N_ITERS) -> tuple[float, list[float]]:
    """Run fn n times, return (min_ms, all_ms)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(r)
        times.append((time.perf_counter() - t0) * 1000)
    return min(times), times


def make_inputs(batch_size):
    rng = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(rng, 3)
    x = jax.random.normal(k1, (batch_size, S, H), dtype=jnp.float32)
    weight = jax.random.normal(k2, (V, H), dtype=jnp.float32)
    target_ids = jax.random.randint(k3, (batch_size, S), minval=0, maxval=V)
    return x, weight, target_ids


def main(_):
    print(f"Config: S={S}, H={H}, V={V}, top_k={TOP_K}")
    print(f"Baseline BS={BS_BASELINE}, Online-softmax BS={BS_ONLINE_SOFTMAX}")
    print(f"Pallas tile_s={TILE_S}, tile_v={TILE_V}, interpret={PALLAS_INTERPRET}")
    print(f"Iterations per benchmark: {N_ITERS}")
    print()

    x_ns, weight_ns, tgt_ns = make_inputs(BS_BASELINE)
    x_st, weight_st, _ = make_inputs(BS_ONLINE_SOFTMAX)

    # --- Define benchmark cases ---
    cases = [
        {
            "name": f"Baseline (BS={BS_BASELINE})",
            "fn": lambda: baseline(x_ns, weight_ns, tgt_ns, TOP_K),
            "batch_size": BS_BASELINE,
        },
        {
            "name": f"Baseline top1 (BS={BS_BASELINE})",
            "fn": lambda: baseline(x_ns, weight_ns, tgt_ns, 1),
            "batch_size": BS_BASELINE,
        },
        {
            "name": f"Pallas topk (BS={BS_ONLINE_SOFTMAX})",
            "fn": lambda: online_softmax_log_probs_pallas(
                x_st,
                weight_st,
                top_k=TOP_K,
                tile_s=TILE_S,
                tile_v=TILE_V,
                interpret=PALLAS_INTERPRET,
            ),
            "batch_size": BS_ONLINE_SOFTMAX,
        },
        {
            "name": f"Pallas top1 (BS={BS_ONLINE_SOFTMAX})",
            "fn": lambda: online_softmax_log_probs_pallas(
                x_st,
                weight_st,
                top_k=0,
                tile_s=TILE_S,
                tile_v=TILE_V,
                interpret=PALLAS_INTERPRET,
            ),
            "batch_size": BS_ONLINE_SOFTMAX,
        },
    ]

    # --- Warmup (JIT compile all cases) ---
    print("Warming up (JIT compiling)...")
    for case in cases:
        _ = case["fn"]()
    print("Warmup done.\n")

    # --- Benchmark ---
    print("=" * 68)
    print("THROUGHPUT COMPARISON")
    print("=" * 68)
    print()

    results = []
    for case in cases:
        ms, _ = bench(case["fn"])
        tokens = case["batch_size"] * S
        tps = tokens / (ms / 1000)
        results.append((case["name"], ms, tokens, tps))

    # --- Report ---
    method, latency, tokens_hdr, throughput = "Method", "Latency", "Tokens", "Throughput"
    print(f"{method:<30} {latency:>10} {tokens_hdr:>8} {throughput:>15}")
    print("-" * 68)
    for name, ms, tokens, tps in results:
        print(f"{name:<30} {ms:>8.1f} ms {tokens:>8} {tps:>12,.0f} tok/s")

    print()
    baseline_topk_tps = results[0][3]
    baseline_top1_tps = results[1][3]
    pallas_topk_tps = results[2][3]
    pallas_top1_tps = results[3][3]
    print(
        f"Pallas topk vs Baseline topk: " f"{pallas_topk_tps / baseline_topk_tps:.2f}x throughput"
    )
    print(
        f"Pallas top1 vs Baseline top1: " f"{pallas_top1_tps / baseline_top1_tps:.2f}x throughput"
    )

    if PALLAS_INTERPRET:
        print(
            "\nNOTE: Pallas numbers use interpret mode (CPU emulation). "
            "Set PALLAS_INTERPRET=False on TPU for accurate performance."
        )


if __name__ == "__main__":
    app.run(main)

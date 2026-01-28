# Copyright © 2025 Apple Inc.
#
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""grouped matrix multiplication kernel benchmarks.

To run: python3 gmm_gpu_benchmark.py

Tested on H100 with Jax == 0.5.1:

"""

# pytype: disable=pyi-error
import functools

import jax
import jax.numpy as jnp

# pylint: disable=no-name-in-module
from jax.errors import JaxRuntimeError

from axlearn.common.flash_attention.gpu_attention_benchmark import measure
from axlearn.common.megablock.gmm_gpu import gmm
from axlearn.common.megablock.gmm_gpu_test import (
    generate_perfect_group_sizes,
    generate_random_data,
    gmm_lax_scan,
)
from axlearn.common.megablock.ops import gmm as gmm_lib

# pylint: disable=pointless-string-statement
"""
The benchmark reports summarize the time taken to run the GMM kernel benchmark using the following
call:
gmm(lhs, rhs, group_sizes, tiling=(tm, tk, tn), preferred_element_type=test_dtype)
where
- lhs.shape = (m, k)
- rhs.shape = (num_groups, k, n)
- (tm, tk, tn) represent the block sizes along the m, k, and n dimensions
- test_dtype can be either float32 or float16

Below are highlighted benchmark reports with common used parameters, where
m=4096, k=512, n=2048, num_groups=40.
According to the benchmark results, we suggest to use
tm=32, tk=32, tn=64 for training/inference tiling;
tm=64, tk=16, tn=64 for inference tiling.


Forward pass with random tiling. (each tile could maps to random number of groups)
Summary Report:
# pylint: disable=line-too-long
(m, k, n, tm, tk, tn, num_groups)             test_dtype      time_taken(ms)  baseline(ms)    X times faster (baseline/time_taken)
====================================================================================================
(32768, 512, 2048, 32, 32, 64, 40)            float32         1.066784        25.497503       23.9
(32768, 512, 2048, 32, 32, 64, 40)            bfloat16        0.615776        16.614271       27.0
(65536, 512, 2048, 32, 32, 64, 40)            float32         1.986752        50.482048       25.4
(65536, 512, 2048, 32, 32, 64, 40)            bfloat16        1.171616        31.636448       27.0
(4096, 512, 2048, 64, 16, 128, 40)            float32         0.224096        4.378464        19.5
(4096, 512, 2048, 64, 16, 128, 40)            bfloat16        0.157312        2.867776        18.2
(4096, 512, 2048, 64, 16, 64, 40)             float32         0.230528        4.395104        19.1
(4096, 512, 2048, 64, 16, 64, 40)             bfloat16        0.166272        2.881152        17.3
(4096, 512, 2048, 64, 16, 32, 40)             float32         0.259584        4.370848        16.8
(4096, 512, 2048, 64, 16, 32, 40)             bfloat16        0.195584        2.856832        14.6
(4096, 512, 2048, 64, 16, 16, 40)             float32         0.331520        4.381376        13.2
(4096, 512, 2048, 64, 16, 16, 40)             bfloat16        0.287680        2.877504        10.0
(4096, 512, 2048, 32, 32, 128, 40)            float32         0.274112        4.408064        16.1
(4096, 512, 2048, 32, 32, 128, 40)            bfloat16        0.172704        2.865248        16.6
(4096, 512, 2048, 32, 32, 64, 40) # default   float32         0.255104        4.384672        17.2
(4096, 512, 2048, 32, 32, 64, 40) # default   bfloat16        0.181376        2.879296        15.9
(4096, 512, 2048, 32, 32, 32, 40)             float32         0.293632        4.367040        14.9
(4096, 512, 2048, 32, 32, 32, 40)             bfloat16        0.211776        2.860448        13.5
(4096, 512, 2048, 32, 32, 16, 40)             float32         0.358944        4.394080        12.2
(4096, 512, 2048, 32, 32, 16, 40)             bfloat16        0.263200        2.880480        10.9
(4096, 512, 2048, 32, 16, 128, 40)            float32         0.254208        4.373568        17.2
(4096, 512, 2048, 32, 16, 128, 40)            bfloat16        0.173824        2.850016        16.4
(4096, 512, 2048, 32, 16, 64, 40)             float32         0.264704        4.381920        16.6
(4096, 512, 2048, 32, 16, 64, 40)             bfloat16        0.192256        2.866080        14.9
(4096, 512, 2048, 32, 16, 32, 40)             float32         0.307008        4.353440        14.2
(4096, 512, 2048, 32, 16, 32, 40)             bfloat16        0.219968        2.859904        13.0
(4096, 512, 2048, 32, 16, 16, 40)             float32         0.383616        4.379072        11.4
(4096, 512, 2048, 32, 16, 16, 40)             bfloat16        0.320384        2.867712        9.0
(4096, 512, 2048, 128, 16, 128, 40)           float32         8.068320        4.348256        0.5
(4096, 512, 2048, 128, 16, 128, 40)           bfloat16        3.682464        2.857824        0.8
(4096, 512, 2048, 128, 16, 64, 40)            float32         0.248992        4.376608        17.6
(4096, 512, 2048, 128, 16, 64, 40)            bfloat16        0.158560        2.863584        18.1
(4096, 512, 2048, 128, 16, 32, 40)            float32         0.258848        4.368640        16.9
(4096, 512, 2048, 128, 16, 32, 40)            bfloat16        0.189184        2.864896        15.1
(4096, 512, 2048, 128, 16, 16, 40)            float32         0.344256        4.391296        12.8
(4096, 512, 2048, 128, 16, 16, 40)            bfloat16        0.263968        2.876096        10.9


Backward pass with random tiling (each tile could maps to a random number of groups)

Summary Report:
# pylint: disable=line-too-long
(m, k, n, tm, tk, tn, num_groups)             test_dtype      time_taken(ms)  baseline(ms)    X times faster (baseline/time_taken)
====================================================================================================
(32768, 512, 2048, 32, 32, 64, 40)            float32         2.949440        CoreDump
(32768, 512, 2048, 32, 32, 64, 40)            bfloat16        1.965664        CoreDump
(65536, 512, 2048, 32, 32, 64, 40)            float32         5.816160        CoreDump
(65536, 512, 2048, 32, 32, 64, 40)            bfloat16        3.676736        CoreDump
(4096, 512, 2048, 64, 16, 128, 40)            float32         0.957152        6.655008        7.0
(4096, 512, 2048, 64, 16, 128, 40)            bfloat16        0.309760        3.274048        10.6
(4096, 512, 2048, 64, 16, 64, 40)             float32         0.672064        6.676896        9.9
(4096, 512, 2048, 64, 16, 64, 40)             bfloat16        0.334048        3.228256        9.7
(4096, 512, 2048, 64, 16, 32, 40)             float32         0.766080        6.681088        8.7
(4096, 512, 2048, 64, 16, 32, 40)             bfloat16        0.463264        3.244640        7.0
(4096, 512, 2048, 64, 16, 16, 40)             float32         1.211712        6.703008        5.5
(4096, 512, 2048, 64, 16, 16, 40)             bfloat16        0.770560        3.291488        4.3
(4096, 512, 2048, 32, 32, 128, 40)            float32         0.774464        6.730624        8.7
(4096, 512, 2048, 32, 32, 128, 40)            bfloat16        0.243712        3.274944        13.4
(4096, 512, 2048, 32, 32, 64, 40)  # default  float32         0.439168        6.737312        15.3
(4096, 512, 2048, 32, 32, 64, 40)  # default  bfloat16        0.287872        3.267136        11.3
(4096, 512, 2048, 32, 32, 32, 40)             float32         0.560736        6.732992        12.0
(4096, 512, 2048, 32, 32, 32, 40)             bfloat16        0.433888        3.285664        7.6
(4096, 512, 2048, 32, 32, 16, 40)             float32         0.842880        6.738720        8.0
(4096, 512, 2048, 32, 32, 16, 40)             bfloat16        0.711840        3.295040        4.6
(4096, 512, 2048, 32, 16, 128, 40)            float32         0.610432        6.702816        11.0
(4096, 512, 2048, 32, 16, 128, 40)            bfloat16        0.314816        3.256128        10.3
(4096, 512, 2048, 32, 16, 64, 40)             float32         0.602144        6.733344        11.2
(4096, 512, 2048, 32, 16, 64, 40)             bfloat16        0.367904        3.279168        8.9
(4096, 512, 2048, 32, 16, 32, 40)             float32         0.746240        6.712896        9.0
(4096, 512, 2048, 32, 16, 32, 40)             bfloat16        0.546432        3.265760        6.0
(4096, 512, 2048, 32, 16, 16, 40)             float32         1.142400        6.720512        5.9
(4096, 512, 2048, 32, 16, 16, 40)             bfloat16        0.908736        3.242208        3.6
(4096, 512, 2048, 128, 16, 128, 40)           float32         14.794912       6.690496        0.5
(4096, 512, 2048, 128, 16, 128, 40)           bfloat16        0.772000        3.254976        4.2
(4096, 512, 2048, 128, 16, 64, 40)            float32         2.162368        6.724064        3.1
(4096, 512, 2048, 128, 16, 64, 40)            bfloat16        0.419136        3.257600        7.8
(4096, 512, 2048, 128, 16, 32, 40)            float32         1.041344        6.709440        6.4
(4096, 512, 2048, 128, 16, 32, 40)            bfloat16        0.506496        3.272768        6.5
(4096, 512, 2048, 128, 16, 16, 40)            float32         1.493056        6.770080        4.5
(4096, 512, 2048, 128, 16, 16, 40)            bfloat16        0.815232        3.273184        4.0
"""

# List of parameter combinations: m, k, n, tm, tk, tn, num_groups.
# Note: set tk=1 because it isn't used for tiling on GPU, only apply to TPU tiling
# Pallas triton requires the tile size(tm, tn) > 16
param_combinations = [
    (4096, 512, 2048, 64, 16, 128, 40),
    (4096, 512, 2048, 64, 16, 64, 40),  # default tiling
    (4096, 512, 2048, 64, 16, 32, 40),
    (4096, 512, 2048, 64, 16, 16, 40),
    (4096, 512, 2048, 32, 32, 128, 40),
    (4096, 512, 2048, 32, 32, 64, 40),
    (4096, 512, 2048, 32, 32, 32, 40),
    (4096, 512, 2048, 32, 32, 16, 40),
    (4096, 512, 2048, 32, 16, 128, 40),
    (4096, 512, 2048, 32, 16, 64, 40),
    (4096, 512, 2048, 32, 16, 32, 40),
    (4096, 512, 2048, 32, 16, 16, 40),
    (4096, 512, 2048, 128, 16, 128, 40),
    (4096, 512, 2048, 128, 16, 64, 40),
    (4096, 512, 2048, 128, 16, 32, 40),
    (4096, 512, 2048, 128, 16, 16, 40),
    # Look for "Back up table" at the end of the file to check the results for below parameters
    # (4096, 512, 2048, 32, 16, 64, 40),
    # (4096, 512, 2048, 32, 16, 64, 40),
    #
    # (4096, 512, 2048, 32, 16, 64, 128),
    # (4096, 512, 4096, 32, 16, 64, 128),
    # (4096, 1024, 4096, 32, 32, 64, 128),
    #
    # (4096, 512, 2048, 32, 16, 64, 64),
    # (4096, 512, 4096, 32, 16, 64, 64),
    # (4096, 1024, 4096, 32, 32, 64, 64),
    #
    # (2048, 1024, 4096, 32, 32, 128, 8),
    # (2048, 1024, 4096, 32, 32, 128, 16),
    # (4096, 512, 2048, 32, 16, 128, 32),
    # (4096, 512, 4096, 32, 16, 128, 32),
    # (4096, 1024, 4096, 32, 32, 128, 32),
    #
    # (4096, 512, 2048, 32, 16, 64, 32),
    # (4096, 512, 2048, 64, 16, 64, 32),
    # (4096, 512, 4096, 32, 16, 64, 32),
    # (4096, 512, 4096, 64, 16, 64, 32),
    # (4096, 1024, 4096, 32, 32, 64, 32),
    # (4096, 1024, 4096, 64, 32, 64, 32),
    #
    # (1024, 512, 2048, 32, 16, 64, 8),
    # (4096, 512, 4096, 32, 16, 64, 8),
    # (4096, 1024, 4096, 32, 32, 64, 8),
    #
    # (1024, 512, 2048, 32, 16, 64, 16),
    # (4096, 512, 2048, 32, 16, 64, 16),
    # (4096, 512, 4096, 32, 16, 64, 16),
    # (4096, 1024, 4096, 32, 32, 64, 16),
    #
    # (4096, 512, 2048, 32, 16, 128, 16),
    # (4096, 512, 4096, 32, 16, 128, 16),
    # (4096, 1024, 4096, 32, 32, 128, 16),
    #
    # (512, 512, 2048, 32, 16, 32, 8),
    # (4096, 512, 4096, 32, 16, 32, 8),
    # (4096, 512, 4096, 32, 16, 32, 16),
    # (4096, 1024, 4096, 32, 32, 32, 16),
]
# List of test dtypes to test
test_dtypes = [jnp.float32, jnp.bfloat16]


def bench_fwd(benchmark_perfect_tiling, baseline_fn=gmm_lax_scan):
    gmm_results = []

    # Iterate over the combinations of parameters and test_dtypes
    for m, k, n, tm, tk, tn, num_groups in param_combinations:
        for test_dtype in test_dtypes:
            # Generate random data with the given parameters
            lhs, rhs, group_sizes = generate_random_data(m, k, n, num_groups, test_dtype, False)
            if benchmark_perfect_tiling:
                # Generate perfect tiling
                group_sizes = generate_perfect_group_sizes(
                    num_groups=num_groups, maxval=m, block_size=tm
                )

            # Create the gmm function with the specific tiling and test_dtype
            gmm_fn = functools.partial(
                gmm,
                tiling=(tm, tk, tn),
                preferred_element_type=test_dtype,
            )

            reference_fn = functools.partial(
                baseline_fn,
                preferred_element_type=test_dtype,
            )
            try:
                # Measure the performance
                _, t = measure(gmm_fn, lhs, rhs, group_sizes)

                # Collect the result in the results list
                gmm_result = {
                    "param_combination": (m, k, n, tm, tk, tn, num_groups),
                    "test_dtype": test_dtype,
                    "time_taken": t,
                }
            except JaxRuntimeError:
                # If an OOM error occurs, append 'OOM' to results
                gmm_result = {
                    "param_combination": (m, k, n, tm, tk, tn, num_groups),
                    "test_dtype": test_dtype,
                    "time_taken": "OOM",
                }
            try:
                # Measure the baseline
                _, baseline_t = measure(reference_fn, lhs, rhs, group_sizes)
                gmm_result["baseline_t"] = baseline_t

            except JaxRuntimeError:
                gmm_result["baseline_t"] = "OOM"
            gmm_results.append(gmm_result)
    if benchmark_perfect_tiling:
        print("Forward pass with perfect tiling. (each tile maps to a single group)")
    else:
        print("Forward pass with random tiling. (each tile could maps to random number of groups)")
    print_report(gmm_results)


def bench_bwd(benchmark_perfect_tiling, baseline_fn=gmm_lax_scan):
    gmm_results = []
    skipped_test = []
    # Iterate over the combinations of parameters and test_dtypes
    for m, k, n, tm, tk, tn, num_groups in param_combinations:
        for test_dtype in test_dtypes:
            # Generate random data with the given parameters
            try:
                cotangent = jax.random.normal(jax.random.PRNGKey(3), (m, n), dtype=test_dtype)
                lhs, rhs, group_sizes = generate_random_data(m, k, n, num_groups, test_dtype, False)
            except JaxRuntimeError:
                skipped_test.append(param_combinations)
                continue
            if benchmark_perfect_tiling:
                # Generate perfect tiling
                group_sizes = generate_perfect_group_sizes(
                    num_groups=num_groups, maxval=m, block_size=tm
                )

            try:
                _, vjpfun = jax.vjp(
                    functools.partial(
                        gmm_lib,
                        preferred_element_type=test_dtype,
                        tiling=(tm, tk, tn),
                    ),
                    lhs,
                    rhs,
                    group_sizes,
                )

                _, t = measure(vjpfun, cotangent)

                # Collect the result in the results list
                gmm_result = {
                    "param_combination": (m, k, n, tm, tk, tn, num_groups),
                    "test_dtype": test_dtype,
                    "time_taken": t,
                }
            except JaxRuntimeError:
                # If an OOM error occurs, append 'OOM' to results
                gmm_result = {
                    "param_combination": (m, k, n, tm, tk, tn, num_groups),
                    "test_dtype": test_dtype,
                    "time_taken": "OOM",
                }
            try:
                # Measure the baseline
                _, reference_vjpfun = jax.vjp(
                    functools.partial(baseline_fn, preferred_element_type=test_dtype),
                    lhs,
                    rhs,
                    group_sizes,
                )

                _, baseline_t = measure(reference_vjpfun, cotangent)
                gmm_result["baseline_t"] = baseline_t

            except JaxRuntimeError:
                gmm_result["baseline_t"] = "OOM"
            gmm_results.append(gmm_result)
    if benchmark_perfect_tiling:
        print("Backward pass with perfect tiling (each tile maps to a single group)")
    else:
        print(
            "Backward pass with random tiling (each tile could maps to a random number of"
            " groups)"
        )
    print_report(gmm_results)
    print("Skipped tests: ", skipped_test)


def print_report(results):
    print("\nSummary Report:")
    print(
        f"{"(m, k, n, tm, tk, tn, num_groups)":<45} {"test_dtype":<15} {"time_taken(ms)":<15} "
        f"{"baseline(ms)":<15} {"X times faster (baseline/time_taken)":<15}"
    )
    print("=" * 120)
    for result in results:
        param_comb_str = str(result["param_combination"])
        dtype_str = str(result["test_dtype"]).rsplit(".", maxsplit=1)[-1]
        if "OOM" not in [result["time_taken"], result["baseline_t"]]:
            scale = round(result["baseline_t"] / result["time_taken"], 1)
            print(
                f"{param_comb_str:<45} {dtype_str:<15} {result["time_taken"]:<15.6f} "
                f"{result["baseline_t"]:<15.6f} {scale:<15.1f}"
            )
        else:
            scale = "N/A"
            print(
                f"{param_comb_str:<45} {dtype_str:<15} {result["time_taken"]:<15.6} "
                f"{result["baseline_t"]:<15.6} {scale:<15}"
            )


def benchmark_with_lax_ragged_dot():
    bench_fwd(benchmark_perfect_tiling=False, baseline_fn=jax.lax.ragged_dot)
    bench_bwd(benchmark_perfect_tiling=False, baseline_fn=jax.lax.ragged_dot)


def benchmark_with_native_python():
    bench_fwd(benchmark_perfect_tiling=False)
    bench_bwd(benchmark_perfect_tiling=False)

    bench_fwd(benchmark_perfect_tiling=True)
    bench_bwd(benchmark_perfect_tiling=True)


if __name__ == "__main__":
    benchmark_with_native_python()
    benchmark_with_lax_ragged_dot()

# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# jax-ml/jax:
# Copyright 2023 The JAX Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
# pylint: disable=line-too-long

"""FlashAttention kernel benchmarks.

Tor run: python3 gpu_attention_benchmark.py > out.txt
Sample numbers on H100 SXM5 with Jax == 0.4.38:
is_decode=True, use_bwd=False, num_heads=8, num_kv_heads=8, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
bs=1,seq_len=1024                                 0.022144      0.020960      0.043520
bs=1,seq_len=4096                                 0.038880      0.026912      0.119872
bs=1,seq_len=8192                                 0.034336      0.037056      0.222784
bs=1,seq_len=131072                               0.233728      0.199520      3.113216
bs=4,seq_len=1024                                 0.025760      0.023392      0.043200
bs=4,seq_len=4096                                 0.066336      0.055488      0.119680
bs=4,seq_len=8192                                 0.078976      0.077792      0.221600
bs=4,seq_len=131072                               0.830560      0.702752      3.263360
bs=8,seq_len=1024                                 0.043392      0.033632      0.043872
bs=8,seq_len=4096                                 0.090624      0.074976      0.121024
bs=8,seq_len=8192                                 0.132480      0.116960      0.223904
bs=8,seq_len=131072                               1.628800      1.372448      3.149344
bs=16,seq_len=1024                                0.058080      0.051520      0.047520
bs=16,seq_len=4096                                0.134080      0.118848      0.124288
bs=16,seq_len=8192                                0.236288      0.202720      0.217920
bs=16,seq_len=131072                              3.194208      2.721792      3.102272
bs=32,seq_len=1024                                0.083200      0.074112      0.077760
bs=32,seq_len=4096                                0.282784      0.203488      0.219808
bs=32,seq_len=8192                                0.443488      0.371360      0.411392
bs=32,seq_len=131072                              6.384032      5.438784      6.138784
is_decode=True, use_bwd=False, num_heads=8, seq_len=32768, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
bs=1,num_kv_heads=1                               0.033344      0.061408      0.863968
bs=1,num_kv_heads=8                               0.078976      0.076832      0.800064
bs=8,num_kv_heads=1                               0.110240      0.082464      1.464544
bs=8,num_kv_heads=8                               0.431200      0.372608      0.814688
is_decode=True, use_bwd=False, num_heads=8, num_kv_heads=8, per_head_dim=128
                                                  jax           axlearn       jax-cudnn
bs=1,seq_len=131072,sw_sz=-1                      0.236064      0.202016      3.098304
bs=1,seq_len=131072,sw_sz=4096                    0.235904      0.058176      3.091552
bs=1,seq_len=131072,sw_sz=16384                   0.236960      0.066272      3.093632
bs=8,seq_len=131072,sw_sz=-1                      1.633824      1.374656      3.183424
bs=8,seq_len=131072,sw_sz=4096                    1.632896      0.196224      3.124256
bs=8,seq_len=131072,sw_sz=16384                   1.622656      0.318752      3.104640
is_decode=False, use_bwd=False, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
bs=2                                              6.151616      0.903744      0.453472
bs=4                                              11.448928     1.728224      0.868096
bs=8                                              23.125055     3.385728      1.692704
is_decode=False, use_bwd=False, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn
bs=2,dtype=<class 'jax.numpy.float32'>            9.706688      3.208640
bs=4,dtype=<class 'jax.numpy.float32'>            19.720287     6.164672
bs=8,dtype=<class 'jax.numpy.float32'>            39.786495     12.005504
is_decode=False, use_bwd=False, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
num_heads=12                                      2.344864      0.388864      0.200256
num_heads=16                                      3.104704      0.493696      0.250944
num_heads=32                                      6.151008      0.902208      0.452736
num_heads=48                                      9.222688      1.319168      0.661536
num_heads=72                                      12.914016     1.946592      0.968800
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
seq_len=256                                       0.052096      0.016224      0.015360
seq_len=512                                       0.133024      0.038944      0.027648
seq_len=1024                                      0.428480      0.095648      0.052288
seq_len=2048                                      1.448448      0.273632      0.141568
seq_len=4096                                      6.142496      0.905152      0.453632
seq_len=8192                                      19.964993     3.300000      1.638720
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
per_head_dim=16                                   5.845152      0.527168      0.309536
per_head_dim=32                                   5.905984      0.560672      0.310912
per_head_dim=64                                   5.972000      0.684672      0.313184
per_head_dim=128                                  6.147936      0.902272      0.453280
is_decode=False, use_bwd=False, num_kv_heads=None, per_head_dim=128
                                                  jax           axlearn       jax-cudnn
bs=1,num_heads=4,seq_len=8192,sw_sz=1024          1.528320      0.080000      0.052032
bs=1,num_heads=4,seq_len=8192,sw_sz=4096          1.528800      0.249504      0.138400
bs=1,num_heads=4,seq_len=16384,sw_sz=1024         5.917536      0.150784      0.094304
bs=1,num_heads=4,seq_len=16384,sw_sz=4096         5.918464      0.477888      0.263616
bs=1,num_heads=4,seq_len=32768,sw_sz=1024         24.009888     0.287040      0.174208
bs=1,num_heads=4,seq_len=32768,sw_sz=4096         23.968737     0.920320      0.498016
is_decode=False, use_bwd=True, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
bs=2                                              6.134624      0.940480      0.488192
bs=4                                              11.365568     1.791296      0.922528
bs=8                                              22.983904     3.470272      1.795264
is_decode=False, use_bwd=True, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn
bs=2,dtype=<class 'jax.numpy.float32'>            9.692192      3.243680
bs=4,dtype=<class 'jax.numpy.float32'>            19.611168     6.216032
bs=8,dtype=<class 'jax.numpy.float32'>            39.664703     12.213696
is_decode=False, use_bwd=True, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
num_heads=12                                      2.335136      0.406336      0.215584
num_heads=16                                      3.088672      0.513216      0.272192
num_heads=32                                      6.135712      0.941312      0.488288
num_heads=48                                      9.171968      1.371936      0.705792
num_heads=72                                      12.810016     2.001088      1.032000
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
seq_len=256                                       0.054560      0.026080      0.025824
seq_len=512                                       0.137696      0.052096      0.039616
seq_len=1024                                      0.424864      0.109216      0.067808
seq_len=2048                                      1.447232      0.297248      0.164768
seq_len=4096                                      6.145216      0.939520      0.486816
seq_len=8192                                      19.973057     3.357696      1.695168
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                                  jax           axlearn       jax-cudnn
per_head_dim=16                                   5.843936      0.541312      0.319744
per_head_dim=32                                   5.892192      0.563232      0.324736
per_head_dim=64                                   5.955232      0.700224      0.333440
per_head_dim=128                                  6.132384      0.941472      0.489696
is_decode=False, use_bwd=True, num_kv_heads=None, per_head_dim=128
                                                  jax           axlearn       jax-cudnn
bs=1,num_heads=4,seq_len=8192,sw_sz=1024          1.528192      0.091232      0.064576
bs=1,num_heads=4,seq_len=8192,sw_sz=4096          1.530336      0.262080      0.148960
bs=1,num_heads=4,seq_len=16384,sw_sz=1024         5.922656      0.164960      0.107584
bs=1,num_heads=4,seq_len=16384,sw_sz=4096         5.914208      0.493888      0.277248
bs=1,num_heads=4,seq_len=32768,sw_sz=1024         23.976608     0.308416      0.195104
bs=1,num_heads=4,seq_len=32768,sw_sz=4096         24.004320     0.942208      0.517792
"""
# pylint: enable=line-too-long
import itertools
from typing import Any, Optional, Protocol, Union

import jax
import jax.numpy as jnp
from jax.experimental.mosaic.gpu.profiler import _event_elapsed, _event_record, has_registrations

from axlearn.common.attention_bias import causal_mask
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.gpu_attention import (
    CuDNNGPUFlashAttentionWithExplicitBias,
    PallasGPUFlashAttention,
)
from axlearn.common.flash_attention.gpu_decoding import GPUDecoding
from axlearn.common.flash_attention.test_utils import generate_attention_data
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.utils import Tensor

X = jnp.zeros((8192, 8192))
Y = jnp.zeros((8192, 8192))

BenchFnResult = Union[tuple[Tensor], Tensor]


class BenchFn(Protocol):
    def __call__(self, *args: Tensor) -> BenchFnResult:
        ...


class SweepFn(Protocol):
    def __call__(self, library: str, *args: Any, **kwargs: Any) -> tuple[BenchFnResult, float]:
        ...


def measure(f: BenchFn, *args: Tensor) -> tuple[Tensor, float]:
    """Measures the time it takes to execute the function on the GPU.

    This function is modified from
    https://github.com/jax-ml/jax/blob/978d35f69704ce95a9d792f9ca9c7e3ee356417f/jax/experimental/mosaic/gpu/profiler.py#L72
    to support measuring fast kernels more accurately. This is done by queueing expensive GEMM
    kernels before the benchmarked function is launched to avoid including dispatch and kernel
    launch overhead to cuda event.

    Args:
        f: The function to measure. It must accept at least one argument and return
            at least one output to be measurable.
        *args: The arguments to pass to ``f``.

    Returns:
        The return value of ``f`` and the elapsed time in milliseconds.
    """
    if not has_registrations:
        raise RuntimeError("This function requires jaxlib >=0.4.36 with CUDA support.")

    if not args:
        # We require at least one argument and at least one output to ensure
        # that there is a data dependency between `_event_record` calls in
        # the resulting HLO program.
        raise ValueError("Can only measure functions with arguments")

    @jax.jit
    def run(*args):
        start_event, args = _event_record(args, copy_before=True)
        end_event, outs = _event_record(f(*args), copy_before=False)
        if jax.tree.structure(outs).num_leaves == 0:
            raise ValueError("Can only measure functions with at least one output")
        return outs, _event_elapsed(start_event, end_event)

    jax.block_until_ready(run(*args))  # Warmup.
    # Queue some expensive kernels into the stream to make events more accurate.
    for _ in range(2):
        _ = X @ Y
    outs, elapsed = run(*args)
    return outs, float(elapsed)


def bench_flash_attention(
    library: str,
    bs: int,
    num_heads: int,
    num_kv_heads: Optional[int],
    seq_len: int,
    per_head_dim: int,
    is_decode: bool,
    use_bwd: bool,
    sw_sz: int = -1,
    dtype=jnp.float16,
):
    if use_bwd and is_decode:
        raise ValueError("use_bwd and is_decode cannot both be true.")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    q_seq_len = 1 if is_decode else seq_len
    mask_fn = causal_mask
    if sw_sz != -1:
        mask_fn = None

    q, k, v, bias = generate_attention_data(
        bs,
        q_seq_len,
        seq_len,
        num_heads,
        per_head_dim,
        num_kv_heads=num_kv_heads,
        mask_fn=mask_fn,
        sliding_window_sz=sw_sz,
        dtype=dtype,
        query_offset=seq_len - 1 if is_decode else 0,
    )

    if "axlearn" in library:
        base_fn = PallasGPUFlashAttention.default_config().instantiate()
        if is_decode:
            base_fn = GPUDecoding.default_config().instantiate()
    elif "cudnn" in library:
        base_fn = CuDNNGPUFlashAttentionWithExplicitBias.default_config().instantiate()
    else:
        base_fn = ReferenceMHA.default_config().instantiate()

    kv_cache_type = KVCache if is_decode else None
    assert base_fn.is_supported(
        dict(query=q, key=k, value=v, bias=bias), kv_cache_type=kv_cache_type
    )
    if use_bwd:
        fn = jax.grad(
            lambda q, k, v, b: base_fn(dict(query=q, key=k, value=v, bias=b)).mean(),
            argnums=(0, 1, 2),
        )
    else:
        fn = lambda q, k, v, b: base_fn(dict(query=q, key=k, value=v, bias=b))
    return measure(fn, q, k, v, bias)


def _sweep(
    fn: SweepFn, libraries: list[str], common_kwargs: dict[str, Any], **_sweep_kwargs: list[Any]
):
    """Benchmarks `fn` by sweeping through combinations of parameters.

    Args:
        fn: The function to benchmark.
        libraries: Libraries to benchmark.
        common_kwargs: kwargs (k=v) that stays unchanged in the sweep.
        sweep_kwargs: kwargs (k=[...]) that will be pass in as cartesian products to `fn`.
    """
    sweep_kwargs: dict[str, list[tuple[str, Any]]] = {}
    for k, args in _sweep_kwargs.items():
        common_kwargs.pop(k, None)
        sweep_kwargs[k] = [(k, v) for v in args]

    # Simple sanity check for results.
    def check_fn(result, ref_result):
        if not jax.numpy.allclose(result, ref_result, atol=0.1):
            raise ValueError(
                f"{library} not equal to jax reference. Args: {common_kwargs} {bench_kwargs}."
                f"Diff: {result - ref_result}"
            )

    results = []
    ref_result = None
    for comb in itertools.product(*sweep_kwargs.values()):
        bench_kwargs = dict(comb)
        bench_key = ",".join(f"{k}={v}" for k, v in comb)
        lib_results = [bench_key]
        for i, library in enumerate(libraries):
            result, t = fn(library=library, **bench_kwargs, **common_kwargs)
            if i == 0:
                ref_result = result
            elif i > 0:
                jax.tree.map(check_fn, result, ref_result)
            lib_results.append(t)
        results.append(lib_results)

    # Header.
    print(", ".join(f"{k}={v}" for k, v in common_kwargs.items()))
    print(("{:<50}" + "{:<14}" * len(libraries)).format("", *libraries))
    # Result rows.
    format_str = "{:<50}" + "{:<14.6f}" * len(libraries)
    for lib_results in results:
        print(format_str.format(*lib_results))


def benchmark_sweep(libraries: list[str], common_kwargs: dict[str, Any], **sweep_args: list[Any]):
    _sweep(bench_flash_attention, libraries, common_kwargs.copy(), **sweep_args)


def benchmark_decode():
    libraries = ["jax", "axlearn", "jax-cudnn"]
    common_kwargs = dict(
        is_decode=True,
        use_bwd=False,
        bs=1,
        num_heads=8,
        num_kv_heads=8,
        seq_len=32 * 1024,
        per_head_dim=128,
        sw_sz=-1,
    )
    benchmark_sweep(
        libraries, common_kwargs, bs=[1, 4, 8, 16, 32], seq_len=[1024, 4096, 8192, 128 * 1024]
    )
    benchmark_sweep(libraries, common_kwargs, bs=[1, 8], num_kv_heads=[1, 8])
    benchmark_sweep(
        libraries, common_kwargs, bs=[1, 8], seq_len=[128 * 1024], sw_sz=[-1, 4096, 16 * 1024]
    )


def bench_flash_attention_fwd_bwd(use_bwd: bool):
    common_kwargs = dict(
        is_decode=False,
        use_bwd=use_bwd,
        bs=2,
        num_heads=32,
        num_kv_heads=None,
        seq_len=4096,
        per_head_dim=128,
        sw_sz=-1,
    )
    libraries = ["jax", "axlearn", "jax-cudnn"]
    benchmark_sweep(libraries, common_kwargs, bs=[2, 4, 8])
    # cuDNN doesn't support fp32.
    benchmark_sweep(["jax", "axlearn"], common_kwargs, bs=[2, 4, 8], dtype=[jnp.float32])
    benchmark_sweep(libraries, common_kwargs, num_heads=[12, 16, 32, 48, 72])
    # 256 to 8192.
    benchmark_sweep(libraries, common_kwargs, seq_len=[int(2**i) for i in range(8, 14)])
    benchmark_sweep(libraries, common_kwargs, per_head_dim=[16, 32, 64, 128])
    benchmark_sweep(
        libraries,
        common_kwargs,
        bs=[1],
        num_heads=[4],
        seq_len=[8192, 16384, 32768],
        sw_sz=[1024, 4096],
    )


def main():
    """Main function to run benchmarks."""
    # Check if CUDA is available
    if jax.default_backend() != "gpu":
        print(f"Skipping GPU benchmarks: backend is {jax.default_backend()}, not 'gpu'")
        return

    # Check for CUDA support in jaxlib
    if not has_registrations:
        print("Skipping GPU benchmarks: jaxlib >=0.4.36 with CUDA support required")
        return

    # Run benchmarks
    benchmark_decode()
    bench_flash_attention_fwd_bwd(False)
    bench_flash_attention_fwd_bwd(True)


if __name__ == "__main__":
    main()

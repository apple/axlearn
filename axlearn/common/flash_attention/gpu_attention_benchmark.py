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
Requires Jax >= 0.4.36. Sample numbers on H100 SXM5:
is_decode=True, use_bwd=False, num_heads=8, num_kv_heads=8, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn
bs=1,seq_len=1024                       0.020608      0.018656      0.023680
bs=1,seq_len=4096                       0.037856      0.022784      0.056704
bs=1,seq_len=8192                       0.033792      0.032768      0.104448
bs=1,seq_len=131072                     0.227808      0.198816      1.486752
bs=4,seq_len=1024                       0.021440      0.022208      0.024032
bs=4,seq_len=4096                       0.069728      0.054624      0.059584
bs=4,seq_len=8192                       0.081952      0.076064      0.105920
bs=4,seq_len=131072                     0.823104      0.705056      1.488832
bs=8,seq_len=1024                       0.032544      0.030688      0.024608
bs=8,seq_len=4096                       0.089728      0.071648      0.063584
bs=8,seq_len=8192                       0.129184      0.114944      0.109856
bs=8,seq_len=131072                     1.616800      1.376288      1.503360
bs=16,seq_len=1024                      0.050976      0.048608      0.037504
bs=16,seq_len=4096                      0.136768      0.117312      0.104224
bs=16,seq_len=8192                      0.234688      0.200128      0.190944
bs=16,seq_len=131072                    3.211200      2.727040      2.779872
bs=32,seq_len=1024                      0.078656      0.072992      0.061440
bs=32,seq_len=4096                      0.236576      0.204512      0.190752
bs=32,seq_len=8192                      0.443488      0.372352      0.361216
bs=32,seq_len=131072                    6.392320      5.453344      5.495488
is_decode=True, use_bwd=False, num_heads=8, seq_len=32768, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn
bs=1,num_kv_heads=1                     0.049280      0.059296      0.378304
bs=1,num_kv_heads=8                     0.076352      0.070912      0.377344
bs=8,num_kv_heads=1                     0.111072      0.080480      0.377696
bs=8,num_kv_heads=8                     0.425536      0.368576      0.386880
is_decode=True, use_bwd=False, num_heads=8, num_kv_heads=8, per_head_dim=128
                                        jax           axlearn       jax-cudnn
bs=1,seq_len=131072,sw_sz=-1            0.228640      0.199040      1.476928
bs=1,seq_len=131072,sw_sz=4096          0.232320      0.053824      4.441376
bs=1,seq_len=131072,sw_sz=16384         0.233696      0.061120      4.420992
bs=8,seq_len=131072,sw_sz=-1            1.621696      1.374080      1.496224
bs=8,seq_len=131072,sw_sz=4096          1.626016      0.193792      4.463296
bs=8,seq_len=131072,sw_sz=16384         1.628704      0.318176      4.451648
is_decode=False, use_bwd=False, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
bs=2                                    3.502944      0.915360      0.467744      0.845792
bs=4                                    6.969376      1.753152      0.890496      1.617280
bs=8                                    13.962816     3.415232      1.735232      3.150752
is_decode=False, use_bwd=False, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
num_heads=12                            1.262560      0.393536      0.205952      0.362304
num_heads=16                            1.786816      0.498304      0.257664      0.459936
num_heads=32                            3.507488      2.591456      0.468672      2.443296
num_heads=48                            5.246336      1.338272      0.675968      1.231328
num_heads=72                            7.866848      1.961152      0.995712      1.805376
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=128                             0.030592      0.011584      0.013024      0.012960
seq_len=256                             0.051520      0.015648      0.016640      0.015744
seq_len=512                             0.118720      0.038976      0.028224      0.037152
seq_len=1024                            0.310880      0.096256      0.054784      0.090368
seq_len=2048                            0.931072      0.277312      0.150784      0.256928
seq_len=4096                            3.516672      2.595872      0.465568      2.448128
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
per_head_dim=16                         3.220960      0.487808      0.332928      0.478720
per_head_dim=32                         3.277824      0.530240      0.334624      0.515040
per_head_dim=64                         3.345376      0.696480      0.338944      0.631296
per_head_dim=128                        3.515616      2.594208      0.465824      2.442784
is_decode=False, use_bwd=True, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
bs=2                                    10.780096     4.573344      2.080672      4.487104
bs=4                                    21.426336     9.336192      3.988224      9.159904
bs=8                                    42.808033     18.926559     7.975296      18.075487
is_decode=False, use_bwd=True, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
num_heads=12                            4.128352      1.738016      0.882976      1.696704
num_heads=16                            5.467808      2.307488      1.120608      2.247904
num_heads=32                            10.782432     4.559456      2.082592      4.488448
num_heads=48                            16.119776     6.958272      3.027808      6.858144
num_heads=72                            24.140833     10.706656     4.560288      10.279136
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=128                             0.058944      0.037824      0.039040      0.036384
seq_len=256                             0.100384      0.069024      0.052608      0.067872
seq_len=512                             0.317056      0.159904      0.111840      0.158912
seq_len=1024                            0.906400      0.431104      0.244160      0.421792
seq_len=2048                            2.861056      1.319648      0.655840      1.297728
seq_len=4096                            10.762560     4.576864      2.079904      4.489056
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
per_head_dim=16                         10.084800     1.744640      1.263264      1.711296
per_head_dim=32                         10.204480     2.098816      1.291104      2.041184
per_head_dim=64                         10.374720     2.649888      1.335200      2.510304
per_head_dim=128                        10.779680     4.568096      2.079264      4.489792
"""
# pylint: enable=line-too-long
import itertools
from functools import partial
from typing import Any, Optional, Protocol, Union

import jax
import jax.numpy as jnp
from jax.experimental.mosaic.gpu.profiler import _event_elapsed, _event_record, has_registrations
from jax.experimental.pallas.ops.gpu.attention import mha as pallas_mha

from axlearn.common.attention_bias import sliding_window_causal_mask
from axlearn.common.flash_attention.gpu_attention import (
    cudnn_dot_product_attention,
    flash_attention,
)
from axlearn.common.flash_attention.gpu_decoding import Tensor, flash_decoding
from axlearn.common.flash_attention.utils import mha_reference

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
      **kwargs: The keyword arguments to pass to ``f``.

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
):
    if use_bwd and is_decode:
        raise ValueError("use_bwd and is_decode cannot both be true.")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    q_seq_len = 1 if is_decode else seq_len
    if is_decode:
        if "pallas" in library:
            q_seq_len = 16  # min supported seq length for triton and pallas
        else:
            q_seq_len = 1
    else:
        q_seq_len = seq_len
    q = jax.random.normal(
        jax.random.PRNGKey(0),
        (bs, q_seq_len, num_heads, per_head_dim),
        dtype=jnp.float16,
    )
    k = jax.random.normal(
        jax.random.PRNGKey(1), (bs, seq_len, num_kv_heads, per_head_dim), dtype=jnp.float16
    )
    v = jax.random.normal(
        jax.random.PRNGKey(2), (bs, seq_len, num_kv_heads, per_head_dim), dtype=jnp.float16
    )
    # Bias is not supported in pallas, so we don't include it here.
    bias = None
    if sw_sz != -1:
        mask_fn = sliding_window_causal_mask(sw_sz)
        assert bias is None
        bias = jnp.zeros((1, 1, 1, seq_len), dtype=jnp.float16)
        bias = bias.at[:, :, :, :-sw_sz].set(jnp.finfo(jnp.float16).min)
    else:
        mask_fn = None
    args = (q, k, v, bias)
    if "axlearn" in library:
        if use_bwd:

            @jax.jit
            def triton_fn(q, k, v, bias):
                # Use mean rather than sum so that gradients won't overflow.
                return flash_attention(q, k, v, bias, causal=True).mean()

            fn = jax.grad(triton_fn, argnums=(0, 1, 2))
        else:
            if q_seq_len == 1:
                fn = partial(flash_decoding, kv_seq_len=None, mask_fn=mask_fn)
                args = (q, k, v)
            else:
                fn = partial(flash_attention, causal=True)
    elif "pallas" in library:
        k = k.repeat(num_heads // num_kv_heads, axis=2)
        v = v.repeat(num_heads // num_kv_heads, axis=2)
        args = (q, k, v)
        if use_bwd:

            @jax.jit
            def pallas_fn(q, k, v):
                return pallas_mha(q, k, v, segment_ids=None, causal=True).mean()

            fn = jax.grad(pallas_fn, argnums=(0, 1, 2))
        else:
            fn = partial(pallas_mha, segment_ids=None, causal=not is_decode)
    elif "cudnn" in library:
        k = k.repeat(num_heads // num_kv_heads, axis=2)
        v = v.repeat(num_heads // num_kv_heads, axis=2)
        if use_bwd:

            @jax.jit
            def cudnn_fn(q, k, v, bias):
                return cudnn_dot_product_attention(q, k, v, bias=bias, causal=True).mean()

            fn = jax.grad(cudnn_fn, argnums=(0, 1, 2))
        else:
            fn = partial(cudnn_dot_product_attention, causal=not is_decode)
    else:
        k = k.repeat(num_heads // num_kv_heads, axis=2)
        v = v.repeat(num_heads // num_kv_heads, axis=2)
        if use_bwd:

            @jax.jit
            def ref_fn(q, k, v, bias):
                return mha_reference(q, k, v, bias, causal=True).mean()

            fn = jax.grad(ref_fn, argnums=(0, 1, 2))
        else:
            fn = partial(mha_reference, causal=not is_decode)

    return measure(fn, *args)


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
            elif i > 0 and library != "jax-pallas":
                jax.tree.map(check_fn, result, ref_result)
            lib_results.append(t)
        results.append(lib_results)

    # Header.
    print(", ".join(f"{k}={v}" for k, v in common_kwargs.items()))
    print(("{:<40}" + "{:<14}" * len(libraries)).format("", *libraries))
    # Result rows.
    format_str = "{:<40}" + "{:<14.6f}" * len(libraries)
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
    libraries = ["jax", "axlearn", "jax-cudnn", "jax-pallas"]
    benchmark_sweep(libraries, common_kwargs, bs=[2, 4, 8])
    benchmark_sweep(libraries, common_kwargs, num_heads=[12, 16, 32, 48, 72])
    # 128 to 4096.
    benchmark_sweep(libraries, common_kwargs, seq_len=[int(2**i) for i in range(7, 13)])
    benchmark_sweep(libraries, common_kwargs, per_head_dim=[16, 32, 64, 128])


benchmark_decode()
bench_flash_attention_fwd_bwd(False)
bench_flash_attention_fwd_bwd(True)

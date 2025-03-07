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
Requires Jax >= 0.4.36. Sample numbers on H100 SXM5 with Jax == 0.4.36:
is_decode=True, use_bwd=False, num_heads=8, num_kv_heads=8, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn
bs=1,seq_len=1024                       0.020832      0.017536      0.024128
bs=1,seq_len=4096                       0.037472      0.021248      0.058656
bs=1,seq_len=8192                       0.034016      0.032576      0.108576
bs=1,seq_len=131072                     0.229856      0.198944      1.558464
bs=4,seq_len=1024                       0.021632      0.023296      0.024352
bs=4,seq_len=4096                       0.068064      0.055168      0.061312
bs=4,seq_len=8192                       0.080352      0.075968      0.109696
bs=4,seq_len=131072                     0.824576      0.703360      1.560768
bs=8,seq_len=1024                       0.033536      0.030304      0.024448
bs=8,seq_len=4096                       0.089056      0.071712      0.062944
bs=8,seq_len=8192                       0.128960      0.114848      0.112736
bs=8,seq_len=131072                     1.620032      1.373088      1.566208
bs=16,seq_len=1024                      0.050368      0.048064      0.036608
bs=16,seq_len=4096                      0.134816      0.116320      0.104320
bs=16,seq_len=8192                      0.234880      0.200384      0.191936
bs=16,seq_len=131072                    3.219008      2.726912      2.784768
bs=32,seq_len=1024                      0.078112      0.070816      0.061568
bs=32,seq_len=4096                      0.235648      0.203296      0.191936
bs=32,seq_len=8192                      0.442080      0.371936      0.365152
bs=32,seq_len=131072                    6.404832      5.448480      5.541504
is_decode=True, use_bwd=False, num_heads=8, seq_len=32768, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn
bs=1,num_kv_heads=1                     0.027648      0.058464      0.398816
bs=1,num_kv_heads=8                     0.076096      0.070368      0.398912
bs=8,num_kv_heads=1                     0.101696      0.078560      0.399040
bs=8,num_kv_heads=8                     0.426656      0.367616      0.403360
is_decode=True, use_bwd=False, num_heads=8, num_kv_heads=8, per_head_dim=128
                                        jax           axlearn       jax-cudnn
bs=1,seq_len=131072,sw_sz=-1            0.230336      0.199968      1.559168
bs=1,seq_len=131072,sw_sz=4096          0.235296      0.051296      4.414048
bs=1,seq_len=131072,sw_sz=16384         0.235904      0.062976      4.385216
bs=8,seq_len=131072,sw_sz=-1            1.619008      1.372768      1.570272
bs=8,seq_len=131072,sw_sz=4096          1.635424      0.194720      4.390976
bs=8,seq_len=131072,sw_sz=16384         1.632832      0.321280      4.361984
is_decode=False, use_bwd=False, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
bs=2                                    3.573152      0.899136      0.458144      0.839488
bs=4                                    7.099072      1.721024      0.870624      1.608000
bs=8                                    14.183424     3.369152      1.705248      3.126688
is_decode=False, use_bwd=False, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
num_heads=12                            1.277984      0.387104      0.198784      0.360832
num_heads=16                            1.812416      0.488224      0.251104      0.455040
num_heads=32                            3.580384      0.900160      0.453920      2.424960
num_heads=48                            5.318784      1.315072      0.662368      1.220640
num_heads=72                            7.986816      1.925024      0.973344      1.792544
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=256                             0.048288      0.015136      0.015520      0.014944
seq_len=512                             0.108736      0.038464      0.027296      0.037024
seq_len=1024                            0.299776      0.095840      0.052128      0.089120
seq_len=2048                            0.933824      0.274720      0.144736      0.254912
seq_len=4096                            3.570368      0.899776      0.452832      2.431424
seq_len=8192                            14.231360     3.270272      1.649280      3.052800
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
per_head_dim=16                         3.263456      0.523584      0.304416      0.468032
per_head_dim=32                         3.316736      0.547872      0.306144      0.514304
per_head_dim=64                         3.415104      0.691200      0.308704      0.616896
per_head_dim=128                        3.571488      0.898720      0.455296      2.428064
is_decode=False, use_bwd=True, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
bs=2                                    10.707744     3.956448      1.958240      4.454080
bs=4                                    21.305376     8.083104      3.799424      9.148736
bs=8                                    42.545246     16.388351     7.600864      18.032385
is_decode=False, use_bwd=True, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
num_heads=12                            4.081632      1.524320      0.834816      1.692224
num_heads=16                            5.411136      2.013376      1.061248      2.227840
num_heads=32                            10.724896     3.964064      1.959168      4.461600
num_heads=48                            15.999936     6.114784      2.883072      6.932064
num_heads=72                            23.977665     9.295424      4.333504      10.293408
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=256                             0.092320      0.057280      0.052928      0.064896
seq_len=512                             0.294752      0.137600      0.108864      0.156992
seq_len=1024                            0.883040      0.361856      0.237632      0.415168
seq_len=2048                            2.855808      1.125152      0.625536      1.285472
seq_len=4096                            10.700160     3.963456      1.955680      4.454624
seq_len=8192                            42.416161     15.214400     6.937056      16.611712
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
per_head_dim=16                         10.013792     1.822496      1.190880      1.670848
per_head_dim=32                         10.118080     1.984032      1.224000      2.017568
per_head_dim=64                         10.279040     2.543008      1.270880      2.492608
per_head_dim=128                        10.704416     3.942368      1.963296      4.455680

is_decode=False, use_bwd=False, bs=1, num_heads=4, num_kv_heads=None, per_head_dim=128
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=8192,sw_sz=1024                 0.882752      0.079968      0.325088      0.319456
seq_len=8192,sw_sz=4096                 0.882272      0.250880      0.325152      0.914560
seq_len=16384,sw_sz=1024                4.639008      0.150720      0.971968      0.948480
seq_len=16384,sw_sz=4096                4.635008      0.471840      0.977888      2.816384
seq_len=32768,sw_sz=1024                25.747295     0.286272      3.327232      3.198048
seq_len=32768,sw_sz=4096                25.761728     0.916800      3.305376      9.667904

is_decode=False, use_bwd=True, bs=1, num_heads=4, num_kv_heads=None, per_head_dim=128
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=8192,sw_sz=1024                 2.481056      0.313504      1.625088      1.154720
seq_len=8192,sw_sz=4096                 2.475456      0.955936      1.610944      1.152000
seq_len=16384,sw_sz=1024                14.337056     0.592704      4.390048      4.075360
seq_len=16384,sw_sz=4096                14.300768     1.845344      4.402304      4.078464
seq_len=32768,sw_sz=1024                44.028992     1.149792      16.766592     15.573600
seq_len=32768,sw_sz=4096                43.984417     3.662720      16.771521     15.582080

"""
# pylint: enable=line-too-long
import itertools
from functools import partial
from typing import Any, Optional, Protocol, Union

import jax
import jax.numpy as jnp
from jax.experimental.mosaic.gpu.profiler import _event_elapsed, _event_record, has_registrations
from jax.experimental.pallas.ops.gpu.attention import mha as pallas_mha

from axlearn.common.attention_bias import causal_mask, sliding_window_causal_mask
from axlearn.common.flash_attention.gpu_attention import (
    NEG_INF,
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
        # We convert mask into a bias tensor for jax and cudnn.
        assert bias is None
        if not is_decode:
            bias = jnp.zeros((1, 1, seq_len, seq_len), dtype=jnp.float16)
            bias = jnp.where(
                mask_fn(jnp.arange(seq_len)[:, None], jnp.arange(seq_len)[None, :]), bias, NEG_INF
            )
        else:
            bias = jnp.zeros((1, 1, 1, seq_len), dtype=jnp.float16)
            bias = bias.at[:, :, :, :-sw_sz].set(NEG_INF)
    else:
        mask_fn = causal_mask
    if "axlearn" in library:
        args = (q, k, v)
        if use_bwd:

            @jax.jit
            def triton_fn(q, k, v):
                # Use mean rather than sum so that gradients won't overflow.
                return flash_attention(q, k, v, mask_fn=mask_fn).mean()

            fn = jax.grad(triton_fn, argnums=(0, 1, 2))
        else:
            if q_seq_len == 1:
                fn = partial(flash_decoding, kv_seq_len=None, mask_fn=mask_fn)
                args = (q, k, v)
            else:
                fn = partial(flash_attention, mask_fn=mask_fn)
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
        args = (q, k, v, bias)

        if use_bwd:

            @jax.jit
            def cudnn_fn(q, k, v, bias):
                return cudnn_dot_product_attention(q, k, v, bias=bias, causal=True).mean()

            fn = jax.grad(cudnn_fn, argnums=(0, 1, 2))
        else:
            fn = partial(cudnn_dot_product_attention, causal=not is_decode)
    else:
        args = (q, k, v, bias)

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
    libraries = ["jax", "axlearn", "jax-cudnn", "jax-pallas"]
    benchmark_sweep(libraries, common_kwargs, bs=[2, 4, 8])
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


benchmark_decode()
bench_flash_attention_fwd_bwd(False)
bench_flash_attention_fwd_bwd(True)

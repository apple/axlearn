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
bs=2                                    3.583424      0.894912      0.488480      0.852960
bs=4                                    7.107168      1.712448      0.922592      1.629888
bs=8                                    14.202400     3.341568      1.801920      3.184064
is_decode=False, use_bwd=False, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
num_heads=12                            1.287712      0.383200      0.214400      0.365120
num_heads=16                            1.803232      0.485408      0.270496      0.463040
num_heads=32                            3.578208      0.896576      0.488544      2.468096
num_heads=48                            5.346112      1.305856      0.707872      1.241728
num_heads=72                            8.001568      1.915776      1.035200      1.820288
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=256                             0.049184      0.015360      0.016352      0.015488
seq_len=512                             0.110400      0.038624      0.028480      0.037760
seq_len=1024                            0.302304      0.094560      0.056736      0.090464
seq_len=2048                            0.936832      0.269856      0.154304      0.258944
seq_len=4096                            3.584800      0.895776      0.487104      2.462560
seq_len=8192                            14.260608     3.268320      1.742048      3.104640
is_decode=False, use_bwd=False, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
per_head_dim=16                         3.262592      0.518912      0.356544      0.477120
per_head_dim=32                         3.323552      0.563520      0.358944      0.533344
per_head_dim=64                         3.411744      0.690464      0.360192      0.635296
per_head_dim=128                        3.585920      0.896032      0.488416      2.461696
is_decode=False, use_bwd=True, num_heads=32, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
bs=2                                    10.878624     3.924992      2.123008      4.504256
bs=4                                    21.626017     8.043040      4.071552      9.186080
bs=8                                    43.269279     16.195999     8.124896      18.184799
is_decode=False, use_bwd=True, bs=2, num_kv_heads=None, seq_len=4096, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
num_heads=12                            4.159424      1.519680      0.898816      1.711808
num_heads=16                            5.486912      2.001952      1.142144      2.256960
num_heads=32                            10.886848     3.928896      2.114496      4.502976
num_heads=48                            16.224319     6.085408      3.093696      6.888640
num_heads=72                            24.367489     9.190560      4.642720      10.323552
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, per_head_dim=128, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
seq_len=256                             0.094496      0.060096      0.053184      0.065760
seq_len=512                             0.297440      0.139328      0.112736      0.161664
seq_len=1024                            0.886304      0.361536      0.246848      0.418720
seq_len=2048                            2.857952      1.118368      0.675168      1.294144
seq_len=4096                            10.880512     3.914048      2.119808      4.503936
seq_len=8192                            43.000095     14.913824     7.484128      16.730017
is_decode=False, use_bwd=True, bs=2, num_heads=32, num_kv_heads=None, seq_len=4096, sw_sz=-1
                                        jax           axlearn       jax-cudnn     jax-pallas
per_head_dim=16                         10.150080     1.826656      1.288192      1.718688
per_head_dim=32                         10.277440     2.028608      1.316512      2.048864
per_head_dim=64                         10.463904     2.569408      1.364448      2.540512
per_head_dim=128                        10.875328     3.929568      2.124192      4.502912
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
    # 256 to 8192.
    benchmark_sweep(libraries, common_kwargs, seq_len=[int(2**i) for i in range(8, 14)])
    benchmark_sweep(libraries, common_kwargs, per_head_dim=[16, 32, 64, 128])


benchmark_decode()
bench_flash_attention_fwd_bwd(False)
bench_flash_attention_fwd_bwd(True)

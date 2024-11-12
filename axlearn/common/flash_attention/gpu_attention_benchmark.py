# Copyright © 2023 Apple Inc.

"""FlashAttention kernel benchmarks.

Sample outputs on A100:

bench-head32-seq1024-d64:
   batch_size       Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0         2.0  1.294677    0.308249  0.178258   2.774377        0.198660
1         4.0  2.337943    0.417789  0.268292   5.449353        0.327930
2         8.0  4.341445    0.625852  0.587789  10.795019        0.585716
3        16.0  8.209461    1.210101  0.736935  21.485954        1.101084
batch2-seq2048-d64:
   num_heads       Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0       12.0  1.758803    0.377092  0.292216   4.154821        0.282745
1       16.0  2.274562    0.437537  0.324150   5.495135        0.334550
2       32.0  4.111703    0.646520  0.599221  10.850373        0.558403
3       40.0  5.044082    0.765017  0.676835  13.539443        0.675699
4       56.0  6.987845    1.035317  0.764956  18.896891        0.900977
5       72.0  8.709219    1.234217  0.836006  24.280788        1.129773
batch2-head32-d64:
   seq_len       Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0    128.0  0.141517    0.127410  0.274202   0.102813        0.022059
1    256.0  0.224241    0.128345  0.268729   0.212434        0.042063
2    512.0  0.426494    0.148231  0.268689   0.770047        0.081487
3   1024.0  1.298753    0.316831  0.156720   2.774356        0.198292
4   2048.0  4.031973    0.626030  0.589759  10.851770        0.558741
batch2-head32-seq2048:
   per_head_dim       Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0          16.0  3.858791    0.437569  0.213704  10.560411        0.331421
1          32.0  3.955815    0.514656  0.261292  10.627665        0.437445
2          64.0  4.121394    0.636113  0.584212  10.851702        0.558101
3         128.0  4.439079    0.973939  0.371719  11.237614        0.754280

With backward pass:

grad-head32-seq1024-d64:
   batch_size        Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0         2.0   2.848025    1.942416  1.694485   6.299916        1.133216
1         4.0   5.322315    1.991064  2.380001  12.186511        1.898364
2         8.0  10.041783    2.945663  4.342529  23.935926        2.966447
3        16.0  19.633947    5.056746  7.604803  47.527328        5.060277
grad-batch2-seq2048-d64:
   num_heads        Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0       12.0   3.776024    2.353134  3.311361   9.244321        2.498945
1       16.0   5.015553    2.435218  3.501143  12.162596        2.632030
2       32.0   9.679915    2.942006  4.506633  24.000723        3.347016
3       40.0  11.839152    3.409772  5.224550  29.938248        3.646133
4       56.0  16.324726    5.345623  8.512093  41.617474        5.386331
5       72.0  20.826162    5.967896  9.429584  53.344563        6.108087
grad-batch2-head32-d64:
   seq_len       Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0    128.0  2.084800    1.946231  1.715087   1.179843        0.623846
1    256.0  2.205729    2.110186  1.704993   0.648029        0.340229
2    512.0  2.259852    2.102726  1.554168   1.883081        0.480496
3   1024.0  2.859227    2.069817  1.645913   6.291944        1.120009
4   2048.0  9.606998    2.954207  4.540852  23.863083        2.998823
grad-batch2-head32-seq2048:
   per_head_dim        Jax  Jax Triton    Pallas    PyTorch  PyTorch Triton
0          16.0   9.076053    2.093116  2.735943  23.176823        1.625986
1          32.0   9.133391    2.438204  3.100990  23.311085        2.156206
2          64.0   9.490566    3.001139  4.537563  23.846561        3.010464
3         128.0  10.347649    5.208034  6.846362  24.728912        5.447452

In addition to the dependencies in attention.py, also requires:
torch==2.1.0.dev20230726+cu121
pytorch-triton==2.1.0+9e3e10c5ed
"""
# pylint: skip-file

import jax
import jax.numpy as jnp
import triton  # pytype: disable=import-error
from jax.experimental.pallas.ops.gpu.attention import mha as pallas_mha

from axlearn.common.flash_attention.gpu_attention import (
    cudnn_dot_product_attention,
    flash_attention,
)
from axlearn.common.flash_attention.utils import mha_reference


def _perf_report(prefix: str):
    # 128 is the most common value for per_head_dim.
    batch_size, num_heads, seq_len, per_head_dim = 2, 32, 2048, 128

    # Vary batch size for fixed heads and seq length.
    batch_size_bench = triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[2, 4, 8, 16],
        line_arg="library",
        line_vals=["jax", "jax-triton", "jax-pallas", "jax-cudnn"],
        line_names=["Jax", "Jax Triton", "Pallas", "jax-cudnn"],
        styles=[("blue", "-"), ("purple", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name=f"{prefix}-head{num_heads}-seq1024-d{per_head_dim}",
        args={"num_heads": num_heads, "seq_len": 1024, "per_head_dim": per_head_dim},
    )
    # Vary num heads for fixed batch and seq length.
    num_heads_bench = triton.testing.Benchmark(
        x_names=["num_heads"],
        x_vals=[12, 16, 32, 48, 72],
        line_arg="library",
        line_vals=["jax", "jax-triton", "jax-pallas", "jax-cudnn"],
        line_names=["Jax", "Jax Triton", "Pallas", "jax-cudnn"],
        styles=[("blue", "-"), ("purple", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name=f"{prefix}-batch{batch_size}-seq{seq_len}-d{per_head_dim}",
        args={"batch_size": batch_size, "seq_len": seq_len, "per_head_dim": per_head_dim},
    )
    # Vary seq length for fixed heads and batch size.
    seq_len_bench = triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2**i for i in range(7, 12)],  # 128 to 2048.
        line_arg="library",
        line_vals=["jax", "jax-triton", "jax-pallas", "jax-cudnn"],
        line_names=["Jax", "Jax Triton", "Pallas", "jax-cudnn"],
        styles=[("blue", "-"), ("purple", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name=f"{prefix}-batch{batch_size}-head{num_heads}-d{per_head_dim}",
        args={"batch_size": batch_size, "num_heads": num_heads, "per_head_dim": per_head_dim},
    )
    # Vary per head dim for fixed batch and seq length.
    per_head_dim_bench = triton.testing.Benchmark(
        x_names=["per_head_dim"],
        x_vals=[16, 32, 64, 128],
        line_arg="library",
        line_vals=["jax", "jax-triton", "jax-pallas", "jax-cudnn"],
        line_names=["Jax", "Jax Triton", "Pallas", "jax-cudnn"],
        styles=[("blue", "-"), ("purple", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name=f"{prefix}-batch{batch_size}-head{num_heads}-seq{seq_len}",
        args={"batch_size": batch_size, "num_heads": num_heads, "seq_len": seq_len},
    )
    return triton.testing.perf_report(
        [batch_size_bench, num_heads_bench, seq_len_bench, per_head_dim_bench]
    )


@_perf_report("fwd")
def bench_flash_attention(
    batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, library: str
):
    warmup = 25
    rep = 500

    if library.startswith("jax"):
        q = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        k = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        v = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        # Bias is not supported in pallas, so we don't include it here.
        bias = None

        if "triton" in library:
            fn = lambda: flash_attention(q, k, v, bias, causal=True)
        elif "pallas" in library:
            fn = lambda: pallas_mha(q, k, v, segment_ids=None, causal=True)
        elif "cudnn" in library:
            fn = lambda: cudnn_dot_product_attention(q, k, v, bias=bias, causal=True)
        else:
            fn = lambda: mha_reference(q, k, v, bias, causal=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    else:
        raise ValueError(f"Unsupported: {library}")
    return ms


@_perf_report("grad")
def bench_flash_attention_backward(
    batch_size: int, num_heads: int, seq_len: int, per_head_dim: int, library: str
):
    warmup = 25
    rep = 500

    if library.startswith("jax"):
        q = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        k = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        v = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.float16
        )
        # Bias is not supported in pallas, so we don't include it here.
        bias = None

        if "triton" in library:

            @jax.jit
            def test_fn(q, k, v, bias):
                return flash_attention(q, k, v, bias, causal=True).sum()

            test_bwd = jax.grad(test_fn, argnums=(0, 1, 2))
            fn = lambda: test_bwd(q, k, v, bias)
        elif "pallas" in library:

            @jax.jit
            def pallas_fn(q, k, v):
                return pallas_mha(q, k, v, segment_ids=None, causal=True).sum()

            pallas_bwd = jax.grad(pallas_fn, argnums=(0, 1, 2))
            fn = lambda: pallas_bwd(q, k, v)
        elif "cudnn" in library:

            @jax.jit
            def cudnn_fn(q, k, v):
                return cudnn_dot_product_attention(q, k, v, bias=bias, causal=True).sum()

            cudnn_bwd = jax.grad(cudnn_fn, argnums=(0, 1, 2))
            fn = lambda: cudnn_bwd(q, k, v)
        else:

            @jax.jit
            def ref_fn(q, k, v, bias):
                return mha_reference(q, k, v, bias, causal=True).sum()

            ref_bwd = jax.grad(ref_fn, argnums=(0, 1, 2))
            fn = lambda: ref_bwd(q, k, v, bias)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    else:
        raise ValueError(f"Unsupported: {library}")
    return ms


bench_flash_attention.run(save_path=".", print_data=True)
bench_flash_attention_backward.run(save_path=".", print_data=True)

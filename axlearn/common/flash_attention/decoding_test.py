# Copyright © 2025 Apple Inc.
"""Tests GPU and TPU decoding."""
from contextlib import nullcontext
from typing import Literal

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common.attention_bias import causal_mask, sliding_window_causal_mask
from axlearn.common.flash_attention.common import BaseFlashAttention, ReferenceMHA
from axlearn.common.flash_attention.gpu_attention import CuDNNGPUFlashAttentionWithExplicitBias
from axlearn.common.flash_attention.gpu_decoding import GPUDecoding
from axlearn.common.flash_attention.test_utils import generate_attention_data
from axlearn.common.flash_attention.tpu_decoding import TPUDecoding
from axlearn.common.test_utils import TestCase, Tolerance

if jax.default_backend() == "gpu":
    decoding_fns = [GPUDecoding, CuDNNGPUFlashAttentionWithExplicitBias]
    dtypes = [jnp.float32, jnp.float16]
elif jax.default_backend() == "tpu":
    decoding_fns = [TPUDecoding]
    dtypes = [jnp.float32, jnp.bfloat16]
elif jax.default_backend() == "cpu":
    # CPU emulation of pallas kernels.
    decoding_fns = [GPUDecoding, TPUDecoding]
    dtypes = [jnp.float32]
else:
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


class DecodingTest(TestCase):
    """Tests GPU and TPU decoding."""

    @parameterized.product(
        [
            dict(zip(["batch_size", "seq_len", "num_heads", "per_head_dim"], args))
            for args in [
                (1, 1024, 32, 64),
                (4, 512, 48, 64),
                (2, 1024, 16, 128),
                (1, 4096, 8, 128),
                (2, 734, 48, 64),
            ]
        ],
        attention_bias_type=[None],
        input_dtype=dtypes,
        padding=[0, 111],
        kv_head_factor=[1, 4, 8],
        window_len=[-1, 16, 127],
        decoding_fn=decoding_fns,
    )
    def test_decode_against_ref(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        per_head_dim: int,
        attention_bias_type: Literal["2d", "4d", None],
        input_dtype: jnp.dtype,
        padding: int,
        kv_head_factor: int,
        window_len: int,
        decoding_fn: BaseFlashAttention,
    ):
        if seq_len >= 1024 and jax.default_backend() == "cpu":
            self.skipTest("Too slow on CPU.")
        self.assertEqual(num_heads % kv_head_factor, 0)
        assert num_heads % kv_head_factor == 0
        softmax_scale = per_head_dim**0.5
        mask_fn = causal_mask
        if window_len > 0:
            mask_fn = sliding_window_causal_mask(window_len)
        cfg = dict(
            softmax_scale=softmax_scale,
            interpret=(jax.default_backend() == "cpu"),
        )
        q, k, v, bias = generate_attention_data(
            batch_size,
            1,
            seq_len,
            num_heads,
            per_head_dim,
            num_kv_heads=num_heads // kv_head_factor,
            mask_fn=mask_fn,
            attention_bias_type=attention_bias_type,
            dtype=input_dtype,
            query_offset=seq_len - padding - 1,
        )

        fn = decoding_fn.default_config().set(**cfg).instantiate()
        is_supported = fn.is_supported(query=q, key=k, value=v, bias=bias)
        if seq_len % 512 != 0 and decoding_fn is TPUDecoding:
            self.assertFalse(is_supported)
            return
        if (
            jax.default_backend() == "gpu"
            and decoding_fn is CuDNNGPUFlashAttentionWithExplicitBias
            and input_dtype is jnp.float32
        ):
            self.assertFalse(is_supported)
            return
        self.assertTrue(is_supported)

        o = fn(q, k, v, bias)
        with jax.default_matmul_precision(
            "highest"
        ) if input_dtype is jnp.float32 else nullcontext():
            o_ref = ReferenceMHA.default_config().set(**cfg).instantiate()(q, k, v, bias)
        if input_dtype is jnp.float32:
            self.assertNestedAllClose(o, o_ref, rtol=0.001, atol=0.0005)
        # bfloat16 and float16 have occasional outliers that require relaxed tolerances.
        elif input_dtype is jnp.bfloat16:
            self.assertAllCloseWithOutliers(
                o,
                o_ref,
                tolerance_map={
                    1.0: Tolerance(rtol=0.05, atol=1.25),
                    0.99: Tolerance(rtol=0.05, atol=0.4),
                    0.95: Tolerance(rtol=0.05, atol=0.2),
                    0.9: Tolerance(rtol=0.05, atol=0.1),
                    0.8: Tolerance(rtol=0.05, atol=0.05),
                },
            )
        elif input_dtype is jnp.float16:
            self.assertAllCloseWithOutliers(
                o,
                o_ref,
                tolerance_map={
                    1.0: Tolerance(rtol=0.01, atol=0.25),
                    0.98: Tolerance(rtol=0.01, atol=0.05),
                    0.9: Tolerance(rtol=0.01, atol=0.025),
                },
            )
        else:
            raise ValueError(f"Unsupported dtype {input_dtype}")

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
from axlearn.common.flash_attention.gpu_paged_attention import GPUPagedAttention
from axlearn.common.flash_attention.test_utils import (
    generate_attention_data,
    generate_paged_attention_data,
)
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

    tolerance_map = {
        jnp.float32: {
            1.0: Tolerance(rtol=0.05, atol=0.15),
            0.98: Tolerance(rtol=0.01, atol=0.02),
            0.9: Tolerance(rtol=0.01, atol=0.005),
        },
        jnp.bfloat16: {
            1.0: Tolerance(rtol=0.05, atol=1.25),
            0.98: Tolerance(rtol=0.05, atol=0.5),
            0.95: Tolerance(rtol=0.05, atol=0.25),
            0.9: Tolerance(rtol=0.05, atol=0.1),
            0.8: Tolerance(rtol=0.05, atol=0.05),
        },
        jnp.float16: {
            1.0: Tolerance(rtol=0.01, atol=0.25),
            0.98: Tolerance(rtol=0.01, atol=0.05),
            0.9: Tolerance(rtol=0.01, atol=0.025),
        },
    }

    @parameterized.product(
        [
            dict(zip(["batch_size", "seq_len", "num_heads", "per_head_dim"], args))
            for args in [
                (1, 1024, 32, 64),
                (4, 512, 48, 64),
                (2, 1024, 16, 128),
                (1, 4096, 8, 128),
            ]
        ],
        attention_bias_type=[None, "2d", "4d"],
        input_dtype=dtypes,
        padding=[0, 111],
        kv_head_factor=[1, 8],
        window_len=[-1, 127],
        page_size=[16],
    )
    def test_paged_attention_against_ref(
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
        page_size: int,
    ):
        if batch_size * seq_len * per_head_dim >= 262144 and input_dtype == jnp.float32:
            pytest.skip(reason="Shared Memory Explodes")

        decoding_fn = GPUPagedAttention
        softmax_scale = per_head_dim**-0.5
        mask_fn = causal_mask
        if window_len > 0:
            mask_fn = sliding_window_causal_mask(window_len)
        cfg = dict(
            softmax_scale=softmax_scale,
            interpret=(jax.default_backend() == "cpu"),
            is_decoding=True,
        )
        q, k, v, page_tables, bias = generate_paged_attention_data(
            batch_size=batch_size,
            query_len=1,
            kv_len=seq_len,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
            num_kv_heads=num_heads // kv_head_factor,
            mask_fn=mask_fn,
            attention_bias_type=attention_bias_type,
            dtype=input_dtype,
            query_offset=seq_len - padding - 1,
            page_size=page_size,
        )
        input_batch = dict(
            query=q,
            key=k,
            value=v,
            page_tables=page_tables,
            bias=bias,
        )

        fn = decoding_fn.default_config().set(**cfg).instantiate()
        is_supported = fn.is_supported(input_batch=input_batch)
        self.assertTrue(is_supported)

        o = fn(input_batch=input_batch)

        with (
            jax.default_matmul_precision("highest") if input_dtype is jnp.float32 else nullcontext()
        ):
            o_ref = ReferenceMHA.default_config().set(**cfg).instantiate()(input_batch=input_batch)

        if input_dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
            raise ValueError(f"Unsupported dtype {input_dtype}")

        # bfloat16 and float16 have occasional outliers that require relaxed tolerances.
        self.assertAllCloseWithOutliers(
            o,
            o_ref,
            tolerance_map=self.tolerance_map[input_dtype],
        )

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
            is_decoding=True,
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
        input_batch = dict(
            query=q,
            key=k,
            value=v,
            bias=bias,
        )
        fn = decoding_fn.default_config().set(**cfg).instantiate()
        is_supported = fn.is_supported(input_batch=input_batch)
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

        o = fn(input_batch)
        with jax.default_matmul_precision(
            "highest"
        ) if input_dtype is jnp.float32 else nullcontext():
            o_ref = ReferenceMHA.default_config().set(**cfg).instantiate()(input_batch)

        if input_dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
            raise ValueError(f"Unsupported dtype {input_dtype}")

        if input_dtype is jnp.float32 and jax.default_backend() == "cpu":
            # CPU uses pure FP32 arithmetic, and thus has higher precision.
            self.assertNestedAllClose(o, o_ref, rtol=0.001, atol=0.0005)
        # bfloat16 and float16 have occasional outliers that require relaxed tolerances.
        self.assertAllCloseWithOutliers(
            o,
            o_ref,
            tolerance_map=self.tolerance_map[input_dtype],
        )

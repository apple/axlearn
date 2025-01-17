# Copyright © 2023 Apple Inc.

"""Tests TPU FlashAttention kernels."""
from __future__ import annotations

import unittest

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map
from jax.interpreters.pxla import thread_resources
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from axlearn.common.attention_bias import (
    CausalAttentionBias,
    MaskFnAttentionBias,
    ZeroAttentionBias,
    causal_mask,
    sliding_window_causal_mask,
)
from axlearn.common.flash_attention import tpu_attention
from axlearn.common.flash_attention.utils import mha_reference
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape
from axlearn.common.utils import Tensor

# Comment out to test on CPU manually. Technically, this test runs on the CPU, albeit very slowly.
if jax.default_backend() != "tpu":
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


def setUpModule():
    # If on CPU, emulate 4 devices.
    chex.set_n_cpu_devices(4)


def jax_fn_mask(query_position: Tensor, key_position: Tensor) -> Tensor:
    """A MaskFn that calls jax.

    The mask is the same as `causal_mask`.

    However, this implementation requires specially handling to use with
    SplashAttention since `tpu_flash_attention()` needs to wrap this function
    to return numpy values if the input is numpy. (Otherwise we get tracer errors in jit.)
    """
    return jnp.greater_equal(query_position, key_position)


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer."""

    _TEST_CONFIGS = [
        dict(
            batch_size=2,
            kv_len=256,
            num_heads=4,
        ),
        dict(
            batch_size=8,
            kv_len=2048,
            num_heads=4,
        ),
    ]

    @parameterized.product(seq_len=[8, 16, 32, 128], sliding_window_size=[4, 8, 16])
    def test_sliding_window_mask_equivalence(self, seq_len, sliding_window_size):
        shape = (seq_len, seq_len)
        ref_mask = splash_attention_mask.LocalMask(
            shape=shape, window_size=(sliding_window_size, 0), offset=0
        )

        mask_fn = sliding_window_causal_mask(sliding_window_size=sliding_window_size)
        mask_array = np.asarray(mask_fn(np.arange(seq_len)[:, None], np.arange(seq_len)[None, :]))

        test_mask = splash_attention_mask.NumpyMask(array=mask_array)

        for i in range(seq_len):
            self.assertNestedAllClose(ref_mask[i:, i:], test_mask[i:, i:])

    @parameterized.parameters(
        [ZeroAttentionBias(), splash_attention_mask.FullMask((8, 8))],
        [CausalAttentionBias(shape=(8, 8)), splash_attention_mask.CausalMask(shape=(8, 8))],
        [
            MaskFnAttentionBias(sliding_window_causal_mask(4), shape=(8, 8)),
            splash_attention_mask.LocalMask(shape=(8, 8), window_size=(4, 0), offset=0),
        ],
    )
    def test_to_splash_mask(self, mask, expected):
        # pylint: disable-next=protected-access
        splash_mask = tpu_attention._to_splash_mask(mask, mask_shape=(8, 8))
        self.assertEqual(splash_mask, expected)

    @parameterized.product(
        batch_size=[4],
        seq_len=[1024, 32768],
        mask_fn=["zero", "causal", "sliding"],
        sliding_window_size=[1024],
        num_heads=[4],
        per_head_dim=[256],
        mesh=[(4, 1)],
        mesh_axis_names=[("data", "model")],
    )
    def test_forward(
        self,
        batch_size,
        seq_len,
        num_heads,
        per_head_dim,
        mask_fn,
        sliding_window_size,
        mesh,
        mesh_axis_names,
    ):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        q = jax.random.normal(
            k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16
        )
        k = jax.random.normal(
            k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16
        )
        v = jax.random.normal(
            k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=jnp.bfloat16
        )

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            mesh = thread_resources.env.physical_mesh

            def fn(q, k, v):
                q = jax.lax.with_sharding_constraint(
                    q, NamedSharding(mesh, PartitionSpec("data", None, "model", None))
                )
                k = jax.lax.with_sharding_constraint(
                    k, NamedSharding(mesh, PartitionSpec("data", None, "model", None))
                )
                v = jax.lax.with_sharding_constraint(
                    v, NamedSharding(mesh, PartitionSpec("data", None, "model", None))
                )

                softmax_scale = q.shape[-1] ** -0.5
                if mask_fn == "zero":
                    mask = ZeroAttentionBias()
                elif mask_fn == "causal":
                    mask = CausalAttentionBias(shape=(seq_len, seq_len))
                elif mask_fn.startswith("sliding"):
                    mask = MaskFnAttentionBias(
                        sliding_window_causal_mask(sliding_window_size), shape=(seq_len, seq_len)
                    )

                attn = lambda q, k, v: tpu_attention.tpu_flash_attention(
                    q,
                    k,
                    v,
                    mask=mask,
                    softmax_scale=softmax_scale,
                    interpret=(jax.default_backend() == "cpu"),
                )

                partitioned_mha = shard_map(
                    attn,
                    mesh=mesh,
                    in_specs=(
                        PartitionSpec("data", None, "model", None),
                        PartitionSpec("data", None, "model", None),
                        PartitionSpec("data", None, "model", None),
                    ),
                    out_specs=PartitionSpec("data", None, "model", None),
                    check_rep=False,
                )

                return partitioned_mha(q, k, v)

            fn = jax.jit(fn)

        # Trigger compilation.
        fn(q, k, v)
        jax.grad(lambda q, k, v: fn(q, k, v).mean(), argnums=(0, 1, 2))(q, k, v)

    @parameterized.product(
        _TEST_CONFIGS,
        query_length_multiplier=[0.5, 1, 2],
        mask=[None, causal_mask, jax_fn_mask],
        attention_bias_type=[None, "2d", "4d"],
        with_segment_ids=[False, True],
        per_head_dim=[32, 64, 128, 256],
    )
    def test_forward_and_backward(
        self,
        batch_size,
        kv_len,
        num_heads,
        per_head_dim,
        query_length_multiplier,
        mask,
        attention_bias_type,
        with_segment_ids,
    ):
        if jax.default_backend() == "cpu":
            # TODO(dhwang2): this has been broken for a while on CPU.
            pytest.skip(reason="Backward path is broken on CPU")
        # pylint: disable=protected-access
        causal = mask in [causal_mask, jax_fn_mask]

        fallback_to_legacy = (
            per_head_dim % 128 != 0
            or (attention_bias_type is not None)
            or with_segment_ids
            or (query_length_multiplier != 1 and mask is not None)
        )
        print(
            f"{batch_size=}, {kv_len=}, {num_heads=}, \n"
            f"{per_head_dim=}, {query_length_multiplier=}, {mask=}, \n"
            f"{attention_bias_type=}, {with_segment_ids=} \n"
            f"{causal=}, {fallback_to_legacy=}"
        )

        if fallback_to_legacy and mask is jax_fn_mask:
            pytest.skip("Custom masks are not supported by legacy attention.")
        if with_segment_ids and query_length_multiplier != 1:
            pytest.skip("Segment IDs are not supported for Q and K with different lengths.")

        k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
        query_len = int(kv_len * query_length_multiplier)
        q = jax.random.normal(
            k1, (batch_size, query_len, num_heads, per_head_dim), dtype=jnp.bfloat16
        )
        k = jax.random.normal(k2, (batch_size, kv_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (batch_size, kv_len, num_heads, per_head_dim), dtype=jnp.bfloat16)
        attention_bias = None
        if attention_bias_type == "2d":
            attention_bias = jax.random.normal(k4, (1, 1, query_len, kv_len), dtype=jnp.bfloat16)
        elif attention_bias_type == "4d":
            attention_bias = jax.random.normal(
                k4, (batch_size, num_heads, query_len, kv_len), dtype=jnp.bfloat16
            )
        segment_ids = None
        if with_segment_ids:
            segment_ids = jax.random.bernoulli(k5, shape=(batch_size, kv_len)).astype(jnp.int32)
            segment_ids = jnp.cumsum(segment_ids, axis=1)

        softmax_scale = q.shape[-1] ** -0.5

        def ref_fn(q, k, v, bias, ids):
            return mha_reference(q, k, v, bias, ids, causal=causal, softmax_scale=softmax_scale)

        legacy_flash_wrapper = unittest.mock.Mock(wraps=tpu_attention._legacy_tpu_flash_attention)

        if mask is not None:
            mask = MaskFnAttentionBias(mask, shape=(query_len, kv_len))

        def fn(q, k, v, bias, ids):
            record_legacy_call = unittest.mock.patch.object(
                tpu_attention, "_legacy_tpu_flash_attention", legacy_flash_wrapper
            )
            with record_legacy_call:
                return tpu_attention.tpu_flash_attention(
                    q,
                    k,
                    v,
                    bias,
                    ids,
                    mask=mask,
                    softmax_scale=softmax_scale,
                    interpret=(jax.default_backend() == "cpu"),
                )

        # Compare outputs.
        out = fn(q, k, v, attention_bias, segment_ids)
        ref_out = ref_fn(q, k, v, attention_bias, segment_ids)
        self.assertNestedAllClose(out, ref_out, atol=0.05)

        # Compare grads.
        grad_out = jax.grad(lambda q, k, v, b, s: fn(q, k, v, b, s).mean(), argnums=(0, 1, 2))(
            q, k, v, attention_bias, segment_ids
        )
        ref_grad_out = jax.grad(
            lambda q, k, v, b, s: ref_fn(q, k, v, b, s).mean(), argnums=(0, 1, 2)
        )(q, k, v, attention_bias, segment_ids)
        self.assertNestedAllClose(grad_out, ref_grad_out, atol=0.05)

        # Check splash attention is used when it should be.
        if fallback_to_legacy:
            legacy_flash_wrapper.assert_called()
        else:
            legacy_flash_wrapper.assert_not_called()


if __name__ == "__main__":
    absltest.main()

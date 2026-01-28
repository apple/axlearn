# Copyright © 2023 Apple Inc.

"""Tests TPU FlashAttention kernels."""
from __future__ import annotations

import unittest
from contextlib import nullcontext
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src.mesh import thread_resources
from jax.experimental import mesh_utils
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

from axlearn.common.attention_bias import (
    CausalAttentionBias,
    MaskFnAttentionBias,
    SlidingWindowAttentionBias,
    ZeroAttentionBias,
    and_masks,
    causal_mask,
    sliding_window_causal_mask,
)
from axlearn.common.flash_attention import tpu_attention
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.layer import default_mha_dim_to_partition_spec
from axlearn.common.flash_attention.splash_attention_mask import ComputableMask
from axlearn.common.flash_attention.test_utils import generate_attention_data
from axlearn.common.test_utils import TestCase, Tolerance, is_supported_mesh_shape
from axlearn.common.utils import Tensor

# Skip decorator for GPU backend
skipIfGPU = unittest.skipIf(
    jax.default_backend() == "gpu", "TPU attention tests not compatible with GPU backend"
)


def jax_fn_mask(sliding_window_size: int) -> Tensor:
    """A MaskFn that calls jax.

    The mask is the same as `sliding_window_causal_mask`.

    However, this implementation requires specially handling to use with
    SplashAttention since `tpu_flash_attention()` needs to wrap this function
    to return numpy values if the input is numpy. (Otherwise we get tracer errors in jit.)
    """

    def mask(query_position: Tensor, key_position: Tensor):
        return query_position - key_position <= sliding_window_size

    fun = and_masks(causal_mask, mask)
    return fun


# pylint: disable-next=invalid-name
_singleton_mask_fn = jax_fn_mask(5)


@skipIfGPU
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

    @parameterized.product(seq_len=[8, 128], sliding_window_size=[4, 8, 16])
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
        [
            CausalAttentionBias(
                target_positions=jnp.arange(8)[None], source_positions=jnp.arange(8)[None]
            ),
            splash_attention_mask.CausalMask(shape=(8, 8)),
        ],
        [
            SlidingWindowAttentionBias(
                sliding_window_causal_mask(4),
                sliding_window_size=4,
                target_positions=jnp.arange(8)[None],
                source_positions=jnp.arange(8)[None],
            ),
            splash_attention_mask.LocalMask(shape=(8, 8), window_size=(4, 0), offset=0),
        ],
        [
            MaskFnAttentionBias(
                _singleton_mask_fn,
                target_positions=jnp.arange(8)[None],
                source_positions=jnp.arange(8)[None],
            ),
            ComputableMask(shape=(8, 8), mask_fn=_singleton_mask_fn),
        ],
    )
    def test_to_splash_mask(self, mask, expected):
        # _to_splash_mask must work well during jax tracing as it runs in shard_map.
        @jax.jit
        def inside_tracing(mask):
            # pylint: disable-next=protected-access
            splash_mask = tpu_attention._to_splash_mask(mask, mask_shape=(8, 8))
            self.assertEqual(splash_mask, expected)

        inside_tracing(mask)

    @parameterized.product(sliding_window_size=(None, 16), seq_len=(128, 2048))
    def test_computable_mask(self, sliding_window_size, seq_len):
        """Test that ComputableMask with mask_fn produces same results as equivalent splash mask."""
        batch_size = 2
        num_heads = 4
        per_head_dim = 64
        num_kv_heads = num_heads // 2

        # Generate test data
        q, k, v, _ = generate_attention_data(
            batch_size,
            seq_len,
            seq_len,
            num_heads,
            per_head_dim,
            num_kv_heads,
            mask_fn=None,
            dtype=jnp.bfloat16,
        )

        cfg = dict(
            interpret=jax.default_backend() == "cpu",
            softmax_scale=per_head_dim**-0.5,
            tpu_block_size=128,
        )
        fn = tpu_attention.TPUSplashAttention.default_config().set(**cfg).instantiate()

        # Test with ComputableMask.
        if sliding_window_size is not None:
            mask_fn = sliding_window_causal_mask(sliding_window_size=sliding_window_size)
            bias_with_ref_mask = SlidingWindowAttentionBias(
                mask=mask_fn,
                sliding_window_size=sliding_window_size,
                target_positions=jnp.arange(seq_len)[None],
                source_positions=jnp.arange(seq_len)[None],
            )
        else:
            mask_fn = causal_mask
            bias_with_ref_mask = CausalAttentionBias(
                target_positions=jnp.arange(seq_len)[None],
                source_positions=jnp.arange(seq_len)[None],
            )

        prng_key = jax.random.PRNGKey(42)
        # Outputs by splash mask
        input_batch_ref = dict(
            query=q,
            key=k,
            value=v,
            bias=bias_with_ref_mask,
            prng_key=prng_key,
            logit_sink=None,
        )
        out_ref = fn(input_batch_ref)

        # Outputs by computable mask
        bias_with_computable_mask = MaskFnAttentionBias(
            mask=mask_fn,
            target_positions=jnp.arange(seq_len)[None],
            source_positions=jnp.arange(seq_len)[None],
        )
        input_batch_computable = dict(
            query=q,
            key=k,
            value=v,
            bias=bias_with_computable_mask,
            prng_key=prng_key,
            logit_sink=None,
        )
        self.assertTrue(fn.is_supported(input_batch=input_batch_computable, kv_cache_type=None))
        out_computable = fn(input_batch_computable)

        # Both should produce the same output
        self.assertNestedAllClose(out_computable, out_ref)

    @parameterized.product(
        kv_len=[64, 128],
        mask=["sliding"],
        with_segment_ids=[True],
        per_head_dim=[64, 128, 150],
        q_dtype=[jnp.bfloat16],
        kv_dtype=[jnp.bfloat16],
    )
    def test_forward_len_dim(self, kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype):
        self._test_forward(kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype)

    @parameterized.product(
        kv_len=[128],
        mask=["full", "causal", "sliding", "custom"],
        with_segment_ids=[True, False],
        per_head_dim=[128],
        q_dtype=[jnp.bfloat16],
        kv_dtype=[jnp.bfloat16],
    )
    def test_forward_mask(self, kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype):
        self._test_forward(kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype)

    @parameterized.product(
        kv_len=[128],
        mask=["sliding"],
        with_segment_ids=[True],
        per_head_dim=[128],
        q_dtype=[jnp.float32, jnp.bfloat16],
        kv_dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_forward_dtype(self, kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype):
        if q_dtype == jnp.bfloat16 and kv_dtype == jnp.float32:
            self.skipTest("Q must have higher precision than KV.")
        self._test_forward(kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype)

    # Running everything as product tests causes timeout in CI.
    def _test_forward(self, kv_len, mask, with_segment_ids, per_head_dim, q_dtype, kv_dtype):
        batch_size, num_heads = 2, 4
        num_kv_heads = num_heads // 2
        q, k, v, _ = generate_attention_data(
            batch_size,
            kv_len,
            kv_len,
            num_heads,
            per_head_dim,
            num_kv_heads,
            mask_fn=None,
            with_segment_ids=with_segment_ids,
            dtype=q_dtype,
            kv_dtype=kv_dtype,
        )

        match mask:
            case "full":
                bias = ZeroAttentionBias()
            case "causal":
                bias = CausalAttentionBias(
                    target_positions=jnp.arange(kv_len)[None],
                    source_positions=jnp.arange(kv_len)[None],
                )
            case "sliding":
                bias = SlidingWindowAttentionBias(
                    mask=sliding_window_causal_mask(8),
                    sliding_window_size=8,
                    target_positions=jnp.arange(kv_len)[None],
                    source_positions=jnp.arange(kv_len)[None],
                )
            case "custom":
                bias = MaskFnAttentionBias(
                    mask=jax_fn_mask(8),
                    target_positions=jnp.arange(kv_len)[None],
                    source_positions=jnp.arange(kv_len)[None],
                )
            case _:
                raise ValueError(f"{mask=} is not supported.")

        cfg = dict(
            interpret=jax.default_backend() == "cpu",
            softmax_scale=per_head_dim**-0.5,
            tpu_block_size=128,
        )
        ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
        fn = tpu_attention.TPUSplashAttention.default_config().set(**cfg).instantiate()
        prng_key = jax.random.PRNGKey(66)
        input_batch = dict(
            query=q,
            key=k,
            value=v,
            bias=bias,
            prng_key=prng_key,
            logit_sink=None,
        )

        if not fn.is_supported(input_batch=input_batch, kv_cache_type=None):
            fn = tpu_attention.LegacyTPUFlashAttention.default_config().set(**cfg).instantiate()
            if not fn.is_supported(input_batch=input_batch, kv_cache_type=None):
                self.skipTest("Even legacy fallback cannot handle it.")

        out = fn(input_batch)
        ref_out = ref_fn(input_batch)
        self.assertAllCloseWithOutliers(
            out, ref_out, tolerance_map={1.0: Tolerance(atol=5e-2), 0.98: Tolerance(atol=1e-2)}
        )

    @parameterized.product(
        _TEST_CONFIGS,
        query_length_multiplier=[0.5, 1, 2],
        mask=[None, causal_mask, jax_fn_mask(5)],
        attention_bias_type=[None, "2d", "4d"],
        with_segment_ids=[False, True],
        per_head_dim=[64, 128],
        q_dtype=[jnp.float32, jnp.bfloat16],
        kv_dtype=[jnp.float32, jnp.bfloat16],
        matmul_precision=[None, "highest"],
        dropout_rate=[0, 0.5],
        head_group_size=[2, 1],
    )
    # TODO: Try to reduce positional arguments
    # pylint: disable-next=too-many-positional-arguments
    def test_gradient(
        self,
        batch_size,
        kv_len,
        num_heads,
        per_head_dim,
        query_length_multiplier,
        mask,
        attention_bias_type,
        with_segment_ids,
        q_dtype,
        kv_dtype,
        matmul_precision,
        dropout_rate,
        head_group_size,
    ):
        if jax.default_backend() == "cpu":
            # TODO(dhwang2): this has been broken for a while on CPU.
            self.skipTest("Backward path is broken on CPU")
        if mask not in (None, causal_mask) and query_length_multiplier > 1:
            self.skipTest("Sliding window attention does not make sense when q_len != kv_len.")
        if dropout_rate > 0.0 and attention_bias_type is not None:
            self.skipTest(
                "Dropout is only supported with SplashAttention (which requires no bias.)"
            )
        if q_dtype == jnp.bfloat16 and kv_dtype == jnp.float32:
            self.skipTest("Q must have higher precision than KV.")

        # pylint: disable=protected-access
        fallback_to_legacy = attention_bias_type is not None
        num_kv_heads = num_heads // head_group_size
        q, k, v, bias = generate_attention_data(
            batch_size,
            int(kv_len * query_length_multiplier),
            kv_len,
            num_heads,
            per_head_dim,
            num_kv_heads,
            mask_fn=mask,
            attention_bias_type=attention_bias_type,
            with_segment_ids=with_segment_ids,
            dtype=q_dtype,
            kv_dtype=kv_dtype,
        )
        tpu_block_size = 128
        cfg = dict(
            interpret=jax.default_backend() == "cpu",
            softmax_scale=per_head_dim**-0.5,
            tpu_block_size=tpu_block_size,
            dropout_rate=dropout_rate,
        )
        ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
        fn = tpu_attention.TPUSplashAttention.default_config().set(**cfg).instantiate()
        prng_key = jax.random.PRNGKey(66)
        input_batch = dict(
            query=q,
            key=k,
            value=v,
            bias=bias,
            prng_key=prng_key,
            logit_sink=None,
        )

        with jax.default_matmul_precision(matmul_precision) if matmul_precision else nullcontext():
            if not fn.is_supported(input_batch=input_batch, kv_cache_type=None):
                # Check splash attention is used when it should be.
                self.assertEqual(fallback_to_legacy, True)
                fn = tpu_attention.LegacyTPUFlashAttention.default_config().set(**cfg).instantiate()
                if not fn.is_supported(input_batch=input_batch, kv_cache_type=None):
                    self.skipTest("Even legacy fallback cannot handle it.")

            # Compare outputs.
            out = fn(input_batch)
            if dropout_rate > 0.0:
                # Get the dropout mask from pallas function as the reference.
                dropout_mask = fn.get_dropout_mask(input_batch)
                ref_fn = partial(ref_fn, dropout_mask=dropout_mask)
            ref_out = ref_fn(input_batch)
            self.assertAllCloseWithOutliers(
                out, ref_out, tolerance_map={1.0: Tolerance(atol=5e-2), 0.98: Tolerance(atol=1e-2)}
            )

            # Compare grads.
            def grad_fn(float_inputs, aux_inputs, f):
                full_batch = {**float_inputs, **aux_inputs}
                return f(full_batch).mean()

            float_inputs = dict(query=q, key=k, value=v, logit_sink=None)
            aux_inputs = dict(bias=bias, prng_key=prng_key)
            grad_out = jax.grad(grad_fn, argnums=0)(float_inputs, aux_inputs, fn)
            ref_grad_out = jax.grad(grad_fn, argnums=0)(float_inputs, aux_inputs, ref_fn)
            self.assertNestedAllCloseWithOutliers(
                grad_out,
                ref_grad_out,
                tolerance_map={
                    1.0: Tolerance(atol=5e-2),
                    0.98: Tolerance(atol=1e-2),
                },
            )

    @parameterized.product(
        batch_size=[2],
        seq_len=[128, 256],  # to cover 1 block and 2 blocks.
        num_heads=[1, 4],
        per_head_dim=[128],
        logit_sink_values=[0.0, -1.0, 1.0],
        q_dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_logit_sink(
        self,
        batch_size,
        seq_len,
        num_heads,
        per_head_dim,
        logit_sink_values,
        q_dtype,
    ):
        """Test logit sink functionality."""
        # Generate test data
        q, k, v, bias = generate_attention_data(
            batch_size,
            seq_len,
            seq_len,
            num_heads,
            per_head_dim,
            num_heads,  # num_kv_heads = num_heads for simplicity
            mask_fn=None,
            attention_bias_type=None,
            with_segment_ids=False,
            dtype=q_dtype,
            kv_dtype=q_dtype,
        )

        # Create logit sink tensor
        logit_sink = jnp.full((num_heads,), logit_sink_values, dtype=q_dtype)

        tpu_block_size = 128
        interpret = jax.default_backend() == "cpu"
        cfg = dict(
            interpret=interpret,
            softmax_scale=per_head_dim**-0.5,
            tpu_block_size=tpu_block_size,
            dropout_rate=0.0,
        )

        ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
        fn = tpu_attention.TPUSplashAttention.default_config().set(**cfg).instantiate()

        prng_key = jax.random.PRNGKey(42)
        input_batch = dict(
            query=q,
            key=k,
            value=v,
            bias=bias,
            logit_sink=logit_sink,
            prng_key=prng_key,
        )

        # Check if the kernel supports this configuration
        is_supported = fn.is_supported(input_batch=input_batch, kv_cache_type=None)
        if not is_supported:
            self.skipTest("Configuration not supported by TPUSplashAttention")

        # Compare outputs
        out = fn(input_batch)
        ref_out = ref_fn(input_batch)
        self.assertNestedAllClose(out, ref_out, atol=1e-6 if q_dtype == jnp.float32 else 2e-2)

        # Compare gradients
        def grad_fn(float_inputs, aux_inputs, f):
            full_batch = {**float_inputs, **aux_inputs}
            return f(full_batch).mean()

        float_inputs = dict(query=q, key=k, value=v, logit_sink=logit_sink)
        aux_inputs = dict(bias=bias, prng_key=prng_key)
        grad_out = jax.grad(grad_fn, argnums=0)(float_inputs, aux_inputs, fn)
        ref_grad_out = jax.grad(grad_fn, argnums=0)(float_inputs, aux_inputs, ref_fn)
        self.assertNestedAllClose(grad_out, ref_grad_out, atol=1e-6)

    def test_logit_sink_shape_validation(self):
        """Test that logit sink shape validation works correctly."""
        batch_size, seq_len, num_heads, per_head_dim = 2, 128, 4, 128

        q, k, v, bias = generate_attention_data(
            batch_size,
            seq_len,
            seq_len,
            num_heads,
            per_head_dim,
            num_heads,
            mask_fn=None,
            attention_bias_type=None,
            with_segment_ids=False,
            dtype=jnp.float32,
            kv_dtype=jnp.float32,
        )

        # Create logit sink with wrong shape (should be num_heads, not num_heads + 1)
        wrong_logit_sink = jnp.zeros((num_heads + 1,), dtype=jnp.float32)

        cfg = dict(
            interpret=jax.default_backend() == "cpu",
            softmax_scale=per_head_dim**-0.5,
            tpu_block_size=128,
            dropout_rate=0.0,
        )

        fn = tpu_attention.TPUSplashAttention.default_config().set(**cfg).instantiate()

        input_batch = dict(
            query=q,
            key=k,
            value=v,
            bias=bias,
            logit_sink=wrong_logit_sink,
            prng_key=jax.random.PRNGKey(42),
        )

        # This should raise a ValueError due to shape mismatch
        with self.assertRaises(ValueError):
            fn.is_supported(input_batch=input_batch, kv_cache_type=None)

    @parameterized.product(
        kv_len=[1024, 1024 * 16],
        sliding_window_sz=[-1, 1024],
        use_fused_override=[None, False, True],
    )
    def test_splash_fused_bwd_heuristic(self, kv_len, sliding_window_sz, use_fused_override):
        if jax.default_backend() != "cpu":
            self.skipTest("No need to run on non-CPU platforms.")  # This is to save CI time.

        # Use a dummy jit function to avoid materializing big qkv tensors in memory.
        @jax.jit
        def jit_fn():
            q, k, v, bias = generate_attention_data(
                1, kv_len, kv_len, 1, 128, 1, sliding_window_sz=sliding_window_sz
            )
            cfg = dict(
                interpret=jax.default_backend() == "cpu",
                tpu_block_size=512,
            )
            if use_fused_override is not None:
                cfg["backend_overrides"] = {"splash_use_fused_bwd_kernel": use_fused_override}
            fn = tpu_attention.TPUSplashAttention.default_config().set(**cfg).instantiate()
            prng_key = jax.random.PRNGKey(66)
            input_batch = dict(
                query=q,
                key=k,
                value=v,
                bias=bias,
                prng_key=prng_key,
                logit_sink=None,
            )

            is_supported = fn.is_supported(input_batch=input_batch, kv_cache_type=None)
            self.assertTrue(is_supported)

            if use_fused_override is not None:
                expected_use_fused = use_fused_override
            elif sliding_window_sz == -1:
                expected_use_fused = True
            else:
                expected_use_fused = not kv_len / sliding_window_sz >= 4.0
            # pylint: disable-next=protected-access
            self.assertEqual(fn._use_fused, expected_use_fused)

        jit_fn()

    @parameterized.product(
        per_head_dim=[128, 150], mask=["causal", "sliding"], with_segment_ids=[False, True]
    )
    def test_all_gather_forward_mask(self, per_head_dim, mask, with_segment_ids):
        mesh_shape, mesh_axis_names = (2, 1, 2), ("data", "model", "seq")
        if not is_supported_mesh_shape(mesh_shape):
            self.skipTest(f"Unsupported mesh shape: {mesh_shape}")

        batch_size, seq_len, num_heads = 2, 256, 4
        num_kv_heads = num_heads
        q, k, v, _ = generate_attention_data(
            batch_size,
            seq_len,
            seq_len,
            num_heads,
            per_head_dim,
            num_kv_heads,
            mask_fn=None,
            with_segment_ids=with_segment_ids,
            dtype=jnp.bfloat16,
        )

        match mask:
            case "causal":
                bias = CausalAttentionBias(
                    target_positions=jnp.arange(seq_len)[None],
                    source_positions=jnp.arange(seq_len)[None],
                )
            case "sliding":
                bias = SlidingWindowAttentionBias(
                    mask=sliding_window_causal_mask(8),
                    sliding_window_size=8,
                    target_positions=jnp.arange(seq_len)[None],
                    source_positions=jnp.arange(seq_len)[None],
                )
            case _:
                raise ValueError(f"{mask=} is not supported.")

        with Mesh(mesh_utils.create_device_mesh(mesh_shape), mesh_axis_names):
            tpu_block_size = 128
            cfg = dict(
                interpret=jax.default_backend() == "cpu",
                softmax_scale=per_head_dim**-0.5,
                tpu_block_size=tpu_block_size,
                dropout_rate=0.0,
                backend_overrides={"all_gather_attention": True},
            )

            attention = (
                tpu_attention.TPUSplashAttentionWithAllGather.default_config()
                .set(**cfg)
                .instantiate()
            )
            ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()

            prng_key = jax.random.PRNGKey(42)
            input_batch = dict(
                query=q,
                key=k,
                value=v,
                bias=bias,
                prng_key=prng_key,
                logit_sink=None,
            )

            if not attention.is_supported(input_batch=input_batch, kv_cache_type=None):
                self.skipTest("Configuration not supported by TPUSplashAttentionWithAllGather")

            # Build the sharded attention function
            specs = attention.build(input_batch=input_batch)

            mesh = thread_resources.env.physical_mesh
            batch_axis = tuple(x for x in mesh_axis_names if x in ("data", "fsdp")) or None

            # Create sharded input batch
            input_batch = {
                "query": q,
                "key": k,
                "value": v,
                "bias": bias,
                "prng_key": prng_key,
                **specs.additional_kwargs,
            }

            # Create in_specs for shard_map
            in_specs = {
                "query": PartitionSpec(batch_axis, "seq", "model", None),
                "key": PartitionSpec(batch_axis, None, "model", None),
                "value": PartitionSpec(batch_axis, None, "model", None),
                "bias": bias.partition_spec(default_mha_dim_to_partition_spec(mesh_axis_names)),
                "prng_key": PartitionSpec(),
                **specs.additional_in_specs,
            }

            # Run attention with shard_map
            partitioned_fn = shard_map(
                specs.fn,
                mesh=mesh,
                in_specs=(in_specs,),
                out_specs=PartitionSpec(batch_axis, "seq", "model", None),
                check_rep=False,
            )
            out = partitioned_fn(input_batch)

            # Compare with reference
            ref_out = ref_fn(input_batch)
            self.assertAllCloseWithOutliers(
                out, ref_out, tolerance_map={1.0: Tolerance(atol=5e-2), 0.98: Tolerance(atol=1e-2)}
            )

    @parameterized.product(
        batch_size=[4],
        seq_len=[1024],
        num_heads=[4],
        per_head_dim=[128],
        mesh=[
            ((1, 1, 4), ("data", "model", "seq")),
            ((2, 1, 2), ("data", "model", "seq")),
            ((1, 2, 2), ("data", "model", "seq")),
        ],
        sliding_window_sz=[
            512,
        ],
        with_segment_ids=[
            False,
            True,
        ],
    )
    def test_all_gather_attention_gradient(
        self,
        batch_size,
        seq_len,
        num_heads,
        per_head_dim,
        mesh,
        sliding_window_sz,
        with_segment_ids,
    ):
        """Test TPUSplashAttentionWithAllGather gradient computation with sequence parallelism."""
        mesh_shape, mesh_axis_names = mesh
        if not is_supported_mesh_shape(mesh_shape):
            self.skipTest(f"Unsupported mesh shape: {mesh_shape}")

        num_kv_heads = num_heads
        q, k, v, bias = generate_attention_data(
            batch_size,
            seq_len,
            seq_len,
            num_heads,
            per_head_dim,
            num_kv_heads,
            sliding_window_sz=sliding_window_sz,
            with_segment_ids=with_segment_ids,
            dtype=jnp.bfloat16,
        )

        with Mesh(mesh_utils.create_device_mesh(mesh_shape), mesh_axis_names):
            tpu_block_size = 128
            cfg = dict(
                interpret=jax.default_backend() == "cpu",
                softmax_scale=per_head_dim**-0.5,
                tpu_block_size=tpu_block_size,
                dropout_rate=0.0,
                backend_overrides={"all_gather_attention": True},
            )

            attention = (
                tpu_attention.TPUSplashAttentionWithAllGather.default_config()
                .set(**cfg)
                .instantiate()
            )
            ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()

            prng_key = jax.random.PRNGKey(42)
            input_batch = dict(
                query=q,
                key=k,
                value=v,
                bias=bias,
                prng_key=prng_key,
                logit_sink=None,
            )

            if not attention.is_supported(input_batch=input_batch, kv_cache_type=None):
                self.skipTest("Configuration not supported by TPUSplashAttentionWithAllGather")

            # Build the sharded attention function directly
            specs = attention.build(input_batch=input_batch)

            mesh = thread_resources.env.physical_mesh
            batch_axis = tuple(x for x in mesh_axis_names if x in ("data", "fsdp")) or None

            # Create sharded input batch
            input_batch = {
                "query": q,
                "key": k,
                "value": v,
                "bias": bias,
                "prng_key": prng_key,
                **specs.additional_kwargs,
            }

            # Create in_specs for shard_map
            in_specs = {
                "query": PartitionSpec(batch_axis, "seq", "model", None),
                "key": PartitionSpec(batch_axis, None, "model", None),
                "value": PartitionSpec(batch_axis, None, "model", None),
                "bias": bias.partition_spec(default_mha_dim_to_partition_spec(mesh_axis_names)),
                "prng_key": PartitionSpec(),
                **specs.additional_in_specs,
            }

            # Run attention with shard_map
            partitioned_fn = shard_map(
                specs.fn,
                mesh=mesh,
                in_specs=(in_specs,),
                out_specs=PartitionSpec(batch_axis, "seq", "model", None),
                check_rep=False,
            )
            out = partitioned_fn(input_batch)

            # Compare with reference
            ref_out = ref_fn(input_batch)
            self.assertAllCloseWithOutliers(
                out, ref_out, tolerance_map={1.0: Tolerance(atol=5e-2), 0.98: Tolerance(atol=1e-2)}
            )

            # Compare gradients
            def grad_fn(q, k, v):
                input_batch_grad = {
                    "query": q,
                    "key": k,
                    "value": v,
                    "bias": bias,
                    "prng_key": prng_key,
                    **specs.additional_kwargs,
                }
                return partitioned_fn(input_batch_grad).mean()

            grad_out = jax.grad(grad_fn, argnums=(0, 1, 2))(q, k, v)

            def ref_grad_fn(q, k, v):
                return ref_fn(
                    dict(query=q, key=k, value=v, bias=bias, prng_key=prng_key, logit_sink=None)
                ).mean()

            ref_grad_out = jax.grad(ref_grad_fn, argnums=(0, 1, 2))(q, k, v)

            self.assertNestedAllCloseWithOutliers(
                grad_out,
                ref_grad_out,
                tolerance_map={
                    1.0: Tolerance(atol=5e-2),
                    0.98: Tolerance(atol=1e-2),
                },
            )


if __name__ == "__main__":
    absltest.main()

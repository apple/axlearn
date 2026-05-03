# Copyright © 2023 Apple Inc.

"""Tests FlashAttention layers."""

import math
import os

# pylint: disable=ungrouped-imports
from unittest import mock

from jax.sharding import PartitionSpec

from axlearn.common.utils import Tensor

# Due to reference layer using XLA,
# set the following environment variables to avoid OOM in GPU tests.
# pylint: disable=wrong-import-position
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# pylint: enable=wrong-import-position

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from axlearn.common.attention import Dropout, GroupedQKVLinear, GroupedQueryAttention, QKVLinear
from axlearn.common.attention_bias import (
    CausalAttentionBias,
    CompositeAttentionBias,
    MaskFnAttentionBias,
    SegmentIdAttentionBias,
    SlidingWindowAttentionBias,
    TensorAttentionBias,
    and_masks,
    bool_to_bias,
    causal_mask,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.flash_attention.layer import (
    BackendOverrideModifier,
    FlashAttention,
    default_mha_dim_to_partition_spec,
    default_output_dim_to_partition_spec,
)
from axlearn.common.kv_cache.kv_cache import KVCache
from axlearn.common.kv_cache.paged_kv_cache import PagedKVCache
from axlearn.common.kv_cache.sliding_window_kv_cache import SlidingWindowKVCache
from axlearn.common.layers import set_bias_recursively
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape


def _fake_inputs(
    *,
    batch: int,
    num_heads: int,
    kv_len: int,
    query_len: int,
    hidden_dim: int,
    use_bias: bool,
    use_segment_ids: bool,
    input_dtype: jnp.dtype = jnp.bfloat16,
):
    query = jax.random.normal(
        jax.random.PRNGKey(0),
        [batch, query_len, hidden_dim],
        dtype=input_dtype,
    )
    key = jax.random.normal(
        jax.random.PRNGKey(1),
        [batch, kv_len, hidden_dim],
        dtype=input_dtype,
    )
    value = jax.random.normal(
        jax.random.PRNGKey(2),
        [batch, kv_len, hidden_dim],
        dtype=input_dtype,
    )
    if use_bias:
        bias = jax.random.bernoulli(
            jax.random.PRNGKey(3), p=0.5, shape=[batch, num_heads, query_len, kv_len]
        )
        bias = bool_to_bias(bias)
        bias = TensorAttentionBias(bias)
    else:
        bias = CompositeAttentionBias([])
    if use_segment_ids:
        segment_ids = jnp.ones([batch, kv_len], dtype=jnp.int32)
    else:
        segment_ids = None
    return dict(
        query=query, key=key, value=value, attention_logit_biases=bias, segment_ids=segment_ids
    )


def _prepare_layers(
    *,
    num_heads,
    num_kv_heads,
    per_head_dim,
    mesh_axis_names,
    mask,
    kv_cache=KVCache.default_config(),
    inference=False,
    set_layer_bias_recursively=False,
    tpu_block_size=512,
    dropout_rate=0.0,
):
    hidden_dim = num_heads * per_head_dim
    cache_dtype = jnp.bfloat16 if inference else None
    kv_cache = kv_cache.set(cache_dtype=cache_dtype)
    kwargs = dict(
        query_dim=hidden_dim,
        key_dim=hidden_dim,
        value_dim=hidden_dim,
        num_heads=num_heads,
        dtype=jnp.bfloat16,
        dropout=Dropout.default_config().set(rate=dropout_rate),
        input_linear=(
            GroupedQKVLinear.default_config().set(num_kv_heads=num_kv_heads)
            if num_kv_heads is not None
            else QKVLinear.default_config()
        ),
        kv_cache=kv_cache,
    )
    ref_cfg = GroupedQueryAttention.default_config().set(**kwargs)

    mha_spec = default_mha_dim_to_partition_spec(mesh_axis_names)
    if kv_cache.klass == PagedKVCache:
        model_axis = "model" if "model" in mesh_axis_names else None
        mha_spec["nbph"] = PartitionSpec(model_axis, None, None, None)
        # ref cfh only uses non-paged kv cache for simplicity
        ref_cfg.set(kv_cache=KVCache.default_config().set(cache_dtype=cache_dtype))
    test_cfg = (
        FlashAttention.default_config()
        .set(**kwargs)
        .set(
            mha_dim_to_partition_spec=mha_spec,
            output_dim_to_partition_spec=default_output_dim_to_partition_spec(mesh_axis_names),
            tpu_block_size=tpu_block_size,
        )
    )

    ref_cfg.set(mask=mask)
    test_cfg.set(mask=mask)

    set_bias_recursively(ref_cfg, set_layer_bias_recursively)
    set_bias_recursively(test_cfg, set_layer_bias_recursively)

    ref_layer = ref_cfg.set(name="ref").instantiate(parent=None)
    test_layer = test_cfg.set(name="test").instantiate(parent=None)

    # Use the same params for both. Only attention implementation differs.
    params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
    return test_layer, ref_layer, params, hidden_dim


def jax_fn_mask(sliding_window_size: int) -> Tensor:
    def mask(query_position: Tensor, key_position: Tensor):
        return query_position - key_position <= sliding_window_size

    fun = and_masks(causal_mask, mask)
    return fun


class DummyModel(BaseLayer):
    """A dummy model."""

    @config_class
    class Config(BaseLayer.Config):
        layer: GroupedQueryAttention.Config = GroupedQueryAttention.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("layer", cfg.layer)

    def forward(self, *, query, key, value, attention_logit_biases, segment_ids):
        # [batch, target_length, target_dim].
        x = self.layer(
            query,
            key=key,
            value=value,
            attention_logit_biases=attention_logit_biases,
            segment_ids=segment_ids,
        )
        # TODO(markblee,zhaoyi-zhang): The atol needs to increase significantly if using
        # jnp.sum, as we no longer scale by the size of the data dims.
        return jnp.mean(x.data, dtype=query.dtype)


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer.

    Tests are structured to vary one axis at a time (following the decoder_test.py pattern)
    rather than brute-force product of all combinations. Each focused test method exercises
    a specific code path dimension while keeping other parameters at baseline values.
    """

    # Configs testing different attention parameters (MHA, GQA, head dims).
    # Single-device (mesh=(1,1)). seq_len > tpu_block_size (128).
    _ATTN_CONFIGS = [
        dict(
            batch=2,
            seq_len=256,
            num_heads=4,
            num_kv_heads=None,
            per_head_dim=32,
            mesh=(1, 1),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=2,
            seq_len=256,
            num_heads=4,
            num_kv_heads=1,
            per_head_dim=32,
            mesh=(1, 1),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=2,
            seq_len=256,
            num_heads=4,
            num_kv_heads=None,
            per_head_dim=64,
            mesh=(1, 1),
            mesh_axis_names=("data", "model"),
        ),
    ]

    # Configs testing representative mesh sharding axis-name combinations.
    _MESH_CONFIGS = [
        dict(
            batch=8,
            seq_len=256,
            num_heads=4,
            num_kv_heads=None,
            per_head_dim=64,
            mesh=(4, 2),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=8,
            seq_len=256,
            num_heads=4,
            num_kv_heads=None,
            per_head_dim=64,
            mesh=(4, 2),
            mesh_axis_names=("data", "fsdp"),
        ),
        dict(
            batch=8,
            seq_len=256,
            num_heads=4,
            num_kv_heads=None,
            per_head_dim=64,
            mesh=(1, 2, 2, 2),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=256,
            num_heads=4,
            num_kv_heads=None,
            per_head_dim=64,
            mesh=(1, 2, 1, 2, 2),
            mesh_axis_names=("data", "seq", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=256,
            num_heads=8,
            num_kv_heads=4,
            per_head_dim=128,
            mesh=(2, 4),
            mesh_axis_names=("data", "model"),
        ),
    ]

    def test_dropout_support(self):
        """Tests that FlashAttention errors out when custom dropout is used."""

        class OtherDropout(Dropout):
            pass

        required_kwargs = dict(query_dim=128, key_dim=128, value_dim=128, num_heads=2, name="test")
        FlashAttention.default_config().set(
            dropout=Dropout.default_config(), **required_kwargs
        ).instantiate(parent=None)

        with self.assertRaises(NotImplementedError):
            FlashAttention.default_config().set(
                dropout=OtherDropout.default_config(), **required_kwargs
            ).instantiate(parent=None)

    def test_gqa_kv_heads(self):
        """Tests _maybe_repeat_kv_heads."""
        batch_size = 8
        num_heads = 8
        num_kv_heads = 4
        seq_len = 2048
        per_head_dim = 128

        hidden_dim = num_heads * per_head_dim

        mesh = (1, 8)
        mesh_axis_names = ("data", "model")
        devices = [
            mock.Mock(spec=jax.Device, platform="tpu", coords=(0, 0, i), core_on_chip=0)
            for i in range(math.prod(mesh))
        ]
        with Mesh(mesh_utils.create_device_mesh(mesh, devices), mesh_axis_names):
            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                input_linear=(
                    GroupedQKVLinear.default_config().set(num_kv_heads=num_kv_heads)
                    if num_kv_heads is not None
                    else QKVLinear.default_config()
                ),
                mha_dim_to_partition_spec={
                    "btnh": PartitionSpec("data", None, "model", None),
                    "bsnh": PartitionSpec("data", None, "model", None),
                    "bnts": PartitionSpec("data", None, "model", None),
                },
            )
            cfg = FlashAttention.default_config().set(**kwargs)
            layer = cfg.set(name="test").instantiate(parent=None)

            kv = jnp.zeros((batch_size, seq_len, num_kv_heads, per_head_dim))
            repeated = layer._maybe_repeat_kv_heads(kv)  # pylint: disable=protected-access
            self.assertEqual(repeated.shape[2], 8)

    # ---- Backend and sharding tests (lightweight, no computation) ----

    @parameterized.parameters(_MESH_CONFIGS)
    @pytest.mark.for_8_devices
    def test_backend(
        self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names
    ):
        del batch, seq_len
        devices = [
            mock.Mock(spec=jax.Device, platform="tpu", coords=(0, 0, i), core_on_chip=0)
            for i in range(math.prod(mesh))
        ]

        with Mesh(
            mesh_utils.create_device_mesh(
                mesh,
                devices,
                allow_split_physical_axes=True,
            ),
            mesh_axis_names,
        ):
            test_layer, _, _, _ = _prepare_layers(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                mask=CausalAttentionBias.default_config(),
            )
            backend = test_layer._backend()  # pylint: disable=protected-access
            self.assertEqual(backend, "tpu")

    @parameterized.parameters(_MESH_CONFIGS)
    @pytest.mark.for_8_devices
    def test_shard_biases(
        self, batch, seq_len, num_heads, num_kv_heads, per_head_dim, mesh, mesh_axis_names
    ):
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")

        def as_tensor_bias(bias: Tensor) -> CompositeAttentionBias:
            return CompositeAttentionBias([TensorAttentionBias(bias)])

        def as_partition_spec(pytree: CompositeAttentionBias) -> PartitionSpec:
            self.assertIsInstance(pytree, CompositeAttentionBias)
            pytree = jax.tree.leaves(pytree)
            self.assertLen(pytree, 1)
            return next(iter(pytree))

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            test_layer, _, _, _ = _prepare_layers(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                mask=CausalAttentionBias.default_config(),
            )
            bias = jnp.ones((batch, num_heads, seq_len, seq_len))
            bias = as_tensor_bias(bias)
            spec = test_layer._logit_biases_spec(bias)  # pylint: disable=protected-access
            spec = as_partition_spec(spec)
            self.assertEqual(spec, test_layer.config.mha_dim_to_partition_spec["bnts"])

            bias = jnp.ones((batch, 1, seq_len, seq_len))
            bias = as_tensor_bias(bias)
            spec = test_layer._logit_biases_spec(bias)  # pylint: disable=protected-access
            spec = as_partition_spec(spec)
            self.assertEqual(spec[1], None)

            bias = jnp.ones((1, 1, seq_len, seq_len))
            bias = as_tensor_bias(bias)
            spec = test_layer._logit_biases_spec(bias)  # pylint: disable=protected-access
            spec = as_partition_spec(spec)
            self.assertEqual(spec[0], None)
            self.assertEqual(spec[1], None)

            segment_ids = CompositeAttentionBias(
                [SegmentIdAttentionBias(jnp.ones((batch, seq_len)))]
            )
            spec = test_layer._logit_biases_spec(segment_ids)  # pylint: disable=protected-access
            spec = as_partition_spec(spec)
            self.assertIsInstance(spec, PartitionSpec)
            self.assertEqual(spec, test_layer.config.mha_dim_to_partition_spec["btnh"][:2])

    # ---- Forward tests ----

    def _test_forward(
        self,
        *,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        query_len_multiplier=1,
        attn_type="full",
        use_bias=False,
        use_segment_ids=False,
        input_dtype=jnp.bfloat16,
        dropout_rate=0.0,
    ):
        """Shared forward test: compares FlashAttention output against reference."""
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")
        if dropout_rate > 0.0 and jax.default_backend() == "tpu":
            self.skipTest("Dropout is implemented for GPU only.")

        if attn_type == "full":
            mask = None
        elif attn_type == "causal":
            mask = CausalAttentionBias.default_config()
        elif attn_type == "sliding_window":
            mask = SlidingWindowAttentionBias.default_config(sliding_window_size=4)
        elif attn_type == "custom":
            mask = MaskFnAttentionBias.default_config(mask=jax_fn_mask(5))
        else:
            raise ValueError(f"Not supported attn_type {attn_type}.")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            test_layer, ref_layer, params, hidden_dim = _prepare_layers(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                mask=mask,
                dropout_rate=dropout_rate,
                tpu_block_size=128,
            )

            query_len = int(query_len_multiplier * seq_len)
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                kv_len=seq_len,
                query_len=query_len,
                hidden_dim=hidden_dim,
                use_bias=use_bias,
                use_segment_ids=use_segment_ids,
                input_dtype=input_dtype,
            )

            ref_inputs = dict(inputs)
            ref_out, _ = F(
                ref_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=ref_inputs,
                is_training=True,
            )
            test_out, _ = F(
                test_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=inputs,
                is_training=True,
            )
            # TODO(markblee): Test probs.
            # Note: cannot compare results when dropout_rate > 0 and not using segment ids, because
            # cudnn dropout will be used and it uses different PRNG than ours.
            # Note: Dropout result between reference and Flash will be different on multiple
            # devices due to the use of shard_map.
            if dropout_rate == 0.0 or (int(np.prod(mesh)) == 1 and use_segment_ids):
                self.assertNestedAllClose(ref_out.data, test_out.data, atol=0.05)
        jax.clear_caches()

    @parameterized.product(
        _ATTN_CONFIGS,
        attn_type=["full", "causal", "sliding_window", "custom"],
    )
    def test_forward_attn_types(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        attn_type,
    ):
        """Tests forward with each mask type across attention configs (MHA, GQA, head dims)."""
        self._test_forward(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            per_head_dim=per_head_dim,
            mesh=mesh,
            mesh_axis_names=mesh_axis_names,
            attn_type=attn_type,
        )

    @parameterized.product(
        query_len_multiplier=[0.5, 2],
        attn_type=["full", "causal"],
    )
    def test_forward_cross_attn(self, query_len_multiplier, attn_type):
        """Tests forward with different Q/K sequence lengths (cross-attention shapes)."""
        self._test_forward(
            **self._ATTN_CONFIGS[0],
            query_len_multiplier=query_len_multiplier,
            attn_type=attn_type,
        )

    def test_forward_bias(self):
        """Tests forward with explicit attention bias tensor."""
        self._test_forward(**self._ATTN_CONFIGS[0], use_bias=True)

    @parameterized.parameters("full", "causal")
    def test_forward_segment_ids(self, attn_type):
        """Tests forward with segment ID masking."""
        self._test_forward(
            **self._ATTN_CONFIGS[0],
            attn_type=attn_type,
            use_segment_ids=True,
        )

    def test_forward_float32(self):
        """Tests forward with float32 input dtype."""
        self._test_forward(
            **self._ATTN_CONFIGS[0],
            attn_type="causal",
            input_dtype=jnp.float32,
        )

    def test_forward_dropout(self):
        """Tests forward with dropout (needs segment_ids for output comparison)."""
        self._test_forward(
            **self._ATTN_CONFIGS[0],
            attn_type="causal",
            dropout_rate=0.1,
            use_segment_ids=True,
        )

    @parameterized.parameters(_MESH_CONFIGS)
    @pytest.mark.for_8_devices
    def test_forward_mesh(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
    ):
        """Tests forward with various mesh sharding configurations."""
        self._test_forward(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            per_head_dim=per_head_dim,
            mesh=mesh,
            mesh_axis_names=mesh_axis_names,
            attn_type="causal",
        )

    # ---- Backward tests ----

    # pylint: disable=too-many-positional-arguments
    def _test_backward(
        self,
        *,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        query_len_multiplier=1,
        attn_type="full",
        use_bias=False,
        use_segment_ids=False,
        set_layer_bias_recursively=False,
        dropout_rate=0.0,
    ):
        """Shared backward test: compares gradients of FlashAttention against reference."""
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")
        if dropout_rate > 0.0 and jax.default_backend() == "tpu":
            self.skipTest("Dropout is implemented for GPU only.")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            hidden_dim = num_heads * per_head_dim
            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                dropout=Dropout.default_config().set(rate=dropout_rate),
                input_linear=(
                    GroupedQKVLinear.default_config().set(num_kv_heads=num_kv_heads)
                    if num_kv_heads is not None
                    else QKVLinear.default_config()
                ),
            )
            if attn_type == "causal":
                kwargs["mask"] = CausalAttentionBias.default_config()
            elif attn_type == "sliding_window":
                kwargs["mask"] = SlidingWindowAttentionBias.default_config(sliding_window_size=4)
            elif attn_type == "custom":
                kwargs["mask"] = MaskFnAttentionBias.default_config(mask=jax_fn_mask(5))

            # Function to determine whether we want to use a smaller gpu_block_size.
            # Using a non-default gpu_block_size may be required when any of the following are true:
            # 1. We are using NVIDIA B200, which require gpu_block_size=64.
            # 2. q_seq_len or k_seq_len cannot be evenly divided by gpu_block_size.
            # Falling back to non-CuDNN or non-Pallas kernels fails on Jax > 0.6.2.
            def use_smaller_gpu_block(k_seq_len: int, q_seq_len: int) -> bool:
                if (
                    jax.default_backend() == "gpu"
                    and "NVIDIA B200" in jax.devices("gpu")[0].device_kind
                ):
                    return True
                # Ensure k_seq_len and q_seq_len are divisble by gpu_block_size
                elif k_seq_len % 128 != 0 or q_seq_len % 128 != 0:
                    return True
                # Otherwise return false and use the default value (128)
                return False

            ref_layer_cfg = GroupedQueryAttention.default_config().set(**kwargs)
            test_layer_cfg = FlashAttention.default_config().set(
                tpu_block_size=128,
                gpu_block_size=(
                    64 if use_smaller_gpu_block(seq_len, seq_len * query_len_multiplier) else 128
                ),
                mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
                output_dim_to_partition_spec=default_output_dim_to_partition_spec(mesh_axis_names),
                **kwargs,
            )

            ref_cfg = DummyModel.default_config().set(layer=ref_layer_cfg)
            test_cfg = DummyModel.default_config().set(layer=test_layer_cfg)
            set_bias_recursively(ref_cfg, set_layer_bias_recursively)
            set_bias_recursively(test_cfg, set_layer_bias_recursively)
            ref_layer = ref_cfg.set(name="ref").instantiate(parent=None)
            test_layer = test_cfg.set(name="test").instantiate(parent=None)

            # Use the same params for both. Only attention implementation differs.
            params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            query_len = int(query_len_multiplier * seq_len)
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                kv_len=seq_len,
                query_len=query_len,
                hidden_dim=hidden_dim,
                use_bias=use_bias,
                use_segment_ids=use_segment_ids,
            )
            ref_inputs = dict(inputs)

            def loss(params, inputs, layer):
                loss, _ = F(
                    layer,
                    inputs=inputs,
                    state=params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )
                return loss

            ref_value, ref_grads = jax.value_and_grad(loss)(params, ref_inputs, ref_layer)
            test_value, test_grads = jax.value_and_grad(loss)(params, inputs, test_layer)

            # Have slightly higher diffs with layer bias on GPU. We don't see this on TPU or CPU.
            # pylint: disable-next=protected-access
            if set_layer_bias_recursively and test_layer.layer._backend() == "gpu":
                atol, rtol = 5e-4, 5e-2
            # pylint: disable-next=protected-access
            elif dropout_rate > 0.0 and test_layer.layer._backend() == "gpu":
                atol, rtol = 3.5e-4, 1e-3
            # pylint: disable-next=protected-access
            elif num_kv_heads and test_layer.layer._backend() == "cpu":
                atol = 5e-4 if int(np.prod(mesh)) > 1 else 1e-4
                rtol = 1e-2
            # Need to relax for GPU tests
            # pylint: disable-next=protected-access
            elif test_layer.layer._backend() == "gpu":
                atol, rtol = 1.5e-4, 1.5e-2
            # Can be 1e-5 on x86_64/GPU/TPU, needed to be slightly higher on ARM.
            # Multi-device CPU needs higher tolerance due to allreduce rounding in DummyModel.
            else:
                atol = 5e-4 if int(np.prod(mesh)) > 1 else 1e-4
                rtol = 1e-3

            # Note: cannot compare results when dropout_rate > 0 and not using segment ids, because
            # cudnn dropout will be used and it uses different PRNG than ours.
            # Note: Dropout result between reference and Flash will be different on multiple
            # devices due to the use of shard_map.
            if dropout_rate == 0.0 or (int(np.prod(mesh)) == 1 and use_segment_ids):
                self.assertNestedAllClose(ref_value, test_value, atol=atol, rtol=rtol)
                self.assertNestedAllClose(ref_grads, test_grads, atol=atol, rtol=rtol)
        jax.clear_caches()

    @parameterized.product(
        _ATTN_CONFIGS,
        attn_type=["full", "causal", "sliding_window", "custom"],
    )
    def test_backward_attn_types(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        attn_type,
    ):
        """Tests backward with each mask type across attention configs."""
        self._test_backward(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            per_head_dim=per_head_dim,
            mesh=mesh,
            mesh_axis_names=mesh_axis_names,
            attn_type=attn_type,
        )

    @parameterized.product(
        query_len_multiplier=[0.5, 2],
        attn_type=["full", "causal"],
    )
    def test_backward_cross_attn(self, query_len_multiplier, attn_type):
        """Tests backward with different Q/K sequence lengths."""
        self._test_backward(
            **self._ATTN_CONFIGS[0],
            query_len_multiplier=query_len_multiplier,
            attn_type=attn_type,
        )

    def test_backward_bias(self):
        """Tests backward with explicit attention bias tensor."""
        self._test_backward(**self._ATTN_CONFIGS[0], use_bias=True)

    @parameterized.parameters("full", "causal")
    def test_backward_segment_ids(self, attn_type):
        """Tests backward with segment ID masking."""
        self._test_backward(
            **self._ATTN_CONFIGS[0],
            attn_type=attn_type,
            use_segment_ids=True,
        )

    @parameterized.parameters(True, False)
    def test_backward_layer_bias(self, set_layer_bias_recursively):
        """Tests backward with/without recursive layer bias."""
        self._test_backward(
            **self._ATTN_CONFIGS[0],
            attn_type="causal",
            set_layer_bias_recursively=set_layer_bias_recursively,
        )

    def test_backward_dropout(self):
        """Tests backward with dropout (needs segment_ids for gradient comparison)."""
        self._test_backward(
            **self._ATTN_CONFIGS[0],
            attn_type="causal",
            dropout_rate=0.1,
            use_segment_ids=True,
        )

    @parameterized.parameters(_MESH_CONFIGS)
    @pytest.mark.for_8_devices
    def test_backward_mesh(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
    ):
        """Tests backward with various mesh sharding configurations."""
        self._test_backward(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            per_head_dim=per_head_dim,
            mesh=mesh,
            mesh_axis_names=mesh_axis_names,
            attn_type="causal",
        )

    # ---- Extend step (decoding) tests ----

    # pylint: disable=too-many-positional-arguments
    def _test_extend_step(
        self,
        *,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        attn_type,
        dtype=jnp.bfloat16,
    ):
        """Shared extend_step test: compares autoregressive decoding against prefill."""
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")

        named_sharding = dict(zip(mesh_axis_names, mesh))
        if "seq" in named_sharding and named_sharding["seq"] > 1:
            self.skipTest("Unsupported seq dim sharding for decoding.")
        if (
            math.prod(mesh) > 1
            and attn_type == "paged"
            and math.prod(mesh) != named_sharding.get("model", 1)
        ):
            self.skipTest("Paged attention only supports model sharding.")

        if attn_type == "causal":
            mask = CausalAttentionBias.default_config()
            kv_cache = KVCache.default_config()
        elif attn_type == "sliding_window":
            mask = SlidingWindowAttentionBias.default_config(sliding_window_size=4)
            kv_cache = SlidingWindowKVCache.default_config().set(cached_kv_length=4)
        elif attn_type == "paged":
            mask = CausalAttentionBias.default_config()
            kv_cache = PagedKVCache.default_config()
        else:
            raise ValueError(f"Not supported attn_type {attn_type}.")

        page_size = 16 if attn_type == "paged" else None
        # Limit generation length to 16 to save test time.
        seq_len = 16 if page_size is None else max(16, page_size)

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            test_layer, ref_layer, params, hidden_dim = _prepare_layers(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                mask=mask,
                kv_cache=kv_cache,
                inference=True,
            )

            # Prepare inputs
            query = jax.random.normal(
                jax.random.PRNGKey(0),
                [batch, seq_len, hidden_dim],
                dtype=dtype,
            )
            causal_bias = None
            kv_state = None
            return_aux = None

            inputs = dict(
                query=query,
                kv_state=kv_state,
                return_aux=return_aux,
                attention_logit_biases=causal_bias,
            )
            ref_inputs = dict(
                query=query,
                kv_state=kv_state,
                attention_logit_biases=causal_bias,
                return_aux=return_aux,
            )

            ref_out, _ = F(
                ref_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=ref_inputs,
                is_training=False,
            )
            test_out, _ = F(
                test_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=inputs,
                is_training=False,
            )

            # Prepare initial states.
            initial_state = test_layer.init_states(batch_size=batch, max_len=seq_len, dtype=dtype)
            ref_initial_state = ref_layer.init_states(
                batch_size=batch, max_len=seq_len, dtype=dtype
            )
            if page_size is not None:
                # Populate the kv_pages and page_indices.
                max_pages_each_request = (seq_len + page_size - 1) // page_size
                # First page is the padding page.
                num_global_pages = batch * max_pages_each_request + 1
                page_indices = jnp.arange(1, num_global_pages).reshape(
                    (batch, max_pages_each_request)
                )
                # Float32 inference still uses bfloat16 kv cache.
                page_dtype = jnp.bfloat16 if dtype is jnp.float32 else dtype
                for k in ["key", "value"]:
                    initial_state["kv_cache"][k] = jnp.zeros(
                        shape=[
                            test_layer.i_proj.num_kv_heads,
                            num_global_pages,
                            page_size,
                            per_head_dim,
                        ],
                        dtype=page_dtype,
                    )
                initial_state["kv_cache"]["page_indices"] = page_indices

                if dtype is jnp.float32:
                    # Float32 inference still uses bfloat16 kv cache.
                    for k in ["key", "value"]:
                        self.assertEqual(ref_initial_state["kv_cache"][k].dtype, jnp.bfloat16)
                else:
                    for k in ["key", "value"]:
                        self.assertEqual(ref_initial_state["kv_cache"][k].dtype, dtype)
            else:
                if dtype is jnp.float32:
                    # Float32 inference still uses bfloat16 kv cache.
                    for k in ["key", "value"]:
                        self.assertEqual(ref_initial_state["kv_cache"][k].dtype, jnp.bfloat16)
                        self.assertEqual(initial_state["kv_cache"][k].dtype, jnp.bfloat16)
                else:
                    for k in ["key", "value"]:
                        self.assertEqual(ref_initial_state["kv_cache"][k].dtype, dtype)
                        self.assertEqual(initial_state["kv_cache"][k].dtype, dtype)

            # Prepare decoding inputs.
            inputs = dict(
                cached_states=initial_state,
                kv_state=kv_state,
                return_aux=return_aux,
            )
            ref_inputs = dict(
                cached_states=ref_initial_state,
                kv_state=kv_state,
                return_aux=return_aux,
            )

            decoder_output = jnp.zeros(shape=[seq_len, batch, hidden_dim]).astype(dtype)
            ref_decoder_output = jnp.zeros(shape=[seq_len, batch, hidden_dim]).astype(dtype)

            @partial(jax.jit, static_argnames=["layer"])
            def extend_one_step(params, inputs, layer):
                return F(
                    layer,
                    state=params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(5),
                    inputs=inputs,
                    method="extend_step",
                )

            for t in range(seq_len):
                cur_query = jnp.expand_dims(query[:, t, :], axis=1)
                inputs["query"] = cur_query
                ref_inputs["query"] = cur_query
                ref_extend_step_outputs, _ = extend_one_step(params, ref_inputs, ref_layer)
                ref_inputs["cached_states"] = ref_extend_step_outputs[0]
                ref_decoder_output = ref_decoder_output.at[t].set(
                    jnp.squeeze(ref_extend_step_outputs[1].data, axis=1)
                )

                extend_step_outputs, _ = extend_one_step(params, inputs, test_layer)
                inputs["cached_states"] = extend_step_outputs[0]
                decoder_output = decoder_output.at[t].set(
                    jnp.squeeze(extend_step_outputs[1].data, axis=1)
                )

                self.assertNestedAllClose(
                    decoder_output[t],
                    ref_decoder_output[t],
                    rtol=0.02,
                    atol=2e-2,
                )

            decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
            ref_decoder_out_transposed = jnp.transpose(ref_decoder_output, [1, 0, 2])
            # Golden Reference still need to adjust for bf16 loss.
            self.assertNestedAllClose(
                ref_out.data,
                ref_decoder_out_transposed,
                rtol=0.02,
                atol=2e-2,
            )
            self.assertNestedAllClose(
                decoder_out_transposed,
                ref_decoder_out_transposed,
                rtol=0.02,
                atol=2e-2,
            )
            self.assertNestedAllClose(
                ref_out.data,
                test_out.data,
                rtol=0.02,
                atol=2e-2,
            )
        jax.clear_caches()

    @parameterized.product(
        _ATTN_CONFIGS,
        attn_type=["causal", "sliding_window", "paged"],
    )
    def test_extend_step(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        attn_type,
    ):
        """Tests extend_step with each KV cache type across attention configs."""
        self._test_extend_step(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            per_head_dim=per_head_dim,
            mesh=mesh,
            mesh_axis_names=mesh_axis_names,
            attn_type=attn_type,
        )

    @parameterized.parameters("causal", "paged")
    def test_extend_step_float32(self, attn_type):
        """Tests extend_step with float32 dtype."""
        self._test_extend_step(
            **self._ATTN_CONFIGS[0],
            attn_type=attn_type,
            dtype=jnp.float32,
        )

    @parameterized.parameters(
        _MESH_CONFIGS[0],  # (data, model) 2x2
        _MESH_CONFIGS[-1],  # GQA + model sharding (1, 4)
    )
    @pytest.mark.for_8_devices
    def test_extend_step_mesh(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
    ):
        """Tests extend_step with mesh sharding."""
        self._test_extend_step(
            batch=batch,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            per_head_dim=per_head_dim,
            mesh=mesh,
            mesh_axis_names=mesh_axis_names,
            attn_type="causal",
        )

    @pytest.mark.for_8_devices
    def test_extend_step_paged_mesh(self):
        """Tests paged extend_step with model-only sharding."""
        self._test_extend_step(
            batch=8,
            seq_len=256,
            num_heads=8,
            num_kv_heads=None,
            per_head_dim=128,
            mesh=(1, 8),
            mesh_axis_names=("data", "model"),
            attn_type="paged",
        )

    # ---- Logit sink tests ----

    @parameterized.product(
        _ATTN_CONFIGS,
        logit_sink=[True, False],
        attn_type=["full", "causal"],
    )
    def test_logit_sink(
        self,
        batch,
        seq_len,
        num_heads,
        num_kv_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        logit_sink,
        attn_type,
    ):
        """Tests logit sink functionality in FlashAttention."""
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")

        mask = None
        if attn_type == "causal":
            mask = CausalAttentionBias.default_config()

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            # Create layers with logit sink configuration
            hidden_dim = num_heads * per_head_dim
            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                dropout=Dropout.default_config().set(rate=0.0),
                input_linear=(
                    GroupedQKVLinear.default_config().set(num_kv_heads=num_kv_heads)
                    if num_kv_heads is not None
                    else QKVLinear.default_config()
                ),
                logit_sink=logit_sink,
            )

            ref_cfg = GroupedQueryAttention.default_config().set(**kwargs)
            test_cfg = (
                FlashAttention.default_config()
                .set(**kwargs)
                .set(
                    mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
                    output_dim_to_partition_spec=default_output_dim_to_partition_spec(
                        mesh_axis_names
                    ),
                    tpu_block_size=128,
                )
            )

            if mask is not None:
                ref_cfg.set(mask=mask)
                test_cfg.set(mask=mask)

            ref_layer = ref_cfg.set(name="ref").instantiate(parent=None)
            test_layer = test_cfg.set(name="test").instantiate(parent=None)

            # Initialize parameters
            params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

            if logit_sink:
                self.assertIn("sink", params)
                self.assertEqual(params["sink"].shape, (num_heads,))
            else:
                self.assertNotIn("sink", params)

            # Create test inputs
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                kv_len=seq_len,
                query_len=seq_len,
                hidden_dim=hidden_dim,
                use_bias=False,
                use_segment_ids=False,
            )

            ref_inputs = dict(inputs)
            ref_out, _ = F(
                ref_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=ref_inputs,
                is_training=True,
            )
            test_out, _ = F(
                test_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=inputs,
                is_training=True,
            )

            # Compare outputs - they should be close when using the same parameters
            self.assertNestedAllClose(ref_out.data, test_out.data, atol=0.05)

    def test_logit_sink_parameter_initialization(self):
        """Tests that logit sink parameters are properly initialized."""
        num_heads = 4
        per_head_dim = 32
        hidden_dim = num_heads * per_head_dim

        # Test with logit sink enabled
        cfg = FlashAttention.default_config().set(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            num_heads=num_heads,
            logit_sink=True,
            name="test",
        )
        layer = cfg.instantiate(parent=None)
        params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(42))

        # Check that sink parameter exists and has correct shape
        self.assertIn("sink", params)
        self.assertEqual(params["sink"].shape, (num_heads,))
        self.assertEqual(params["sink"].dtype, layer.dtype())  # Default dtype

        # Test with logit sink disabled
        cfg_no_sink = cfg.set(logit_sink=False)
        layer_no_sink = cfg_no_sink.instantiate(parent=None)
        params_no_sink = layer_no_sink.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(42)
        )

        # Check that sink parameter does not exist
        self.assertNotIn("sink", params_no_sink)

    def test_logit_sink_numerical_stability(self):
        """Tests that logit sink improves numerical stability with extreme logits."""
        batch = 2
        seq_len = 16
        num_heads = 2
        per_head_dim = 8
        hidden_dim = num_heads * per_head_dim
        mesh = (1, 1)
        mesh_axis_names = ("data", "model")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            # Create layers with and without logit sink
            base_cfg = FlashAttention.default_config().set(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
                output_dim_to_partition_spec=default_output_dim_to_partition_spec(mesh_axis_names),
                tpu_block_size=128,
            )

            layer_with_sink = base_cfg.set(logit_sink=True, name="with_sink").instantiate(
                parent=None
            )
            layer_without_sink = base_cfg.set(logit_sink=False, name="without_sink").instantiate(
                parent=None
            )

            # Initialize parameters
            params_with_sink = layer_with_sink.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(123)
            )
            params_without_sink = layer_without_sink.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(123)
            )

            # Create inputs that might cause numerical issues (large values)
            query = (
                jax.random.normal(
                    jax.random.PRNGKey(0), [batch, seq_len, hidden_dim], dtype=jnp.bfloat16
                )
                * 10
            )
            inputs = dict(
                query=query,
                key=query,
                value=query,
                attention_logit_biases=CompositeAttentionBias([]),
                segment_ids=None,
            )

            # Test forward pass with both configurations
            out_with_sink, _ = F(
                layer_with_sink,
                prng_key=jax.random.PRNGKey(5),
                state=params_with_sink,
                inputs=inputs,
                is_training=True,
            )

            out_without_sink, _ = F(
                layer_without_sink,
                prng_key=jax.random.PRNGKey(5),
                state=params_without_sink,
                inputs=inputs,
                is_training=True,
            )

            # Both should produce finite outputs
            self.assertTrue(jnp.all(jnp.isfinite(out_with_sink.data)))
            self.assertTrue(jnp.all(jnp.isfinite(out_without_sink.data)))

            # Outputs should be different due to logit sink effect
            self.assertFalse(jnp.allclose(out_with_sink.data, out_without_sink.data, atol=1e-3))

    @parameterized.parameters([True, False])
    def test_logit_sink_backward_pass(self, logit_sink):
        """Tests that gradients flow correctly through logit sink."""
        batch = 2
        seq_len = 8
        num_heads = 2
        per_head_dim = 4
        hidden_dim = num_heads * per_head_dim

        with Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            cfg = FlashAttention.default_config().set(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                logit_sink=logit_sink,
                mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(("data", "model")),
                output_dim_to_partition_spec=default_output_dim_to_partition_spec(
                    ("data", "model")
                ),
                tpu_block_size=128,
                name="test",
            )

            layer = cfg.instantiate(parent=None)
            params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

            def loss_fn(params):
                inputs = dict(
                    query=jax.random.normal(
                        jax.random.PRNGKey(0), [batch, seq_len, hidden_dim], dtype=jnp.bfloat16
                    ),
                    key=None,
                    value=None,
                    attention_logit_biases=CompositeAttentionBias([]),
                    segment_ids=None,
                )
                out, _ = F(
                    layer,
                    prng_key=jax.random.PRNGKey(5),
                    state=params,
                    inputs=inputs,
                    is_training=True,
                )
                return jnp.mean(out.data)

            # Compute gradients
            loss_value, grads = jax.value_and_grad(loss_fn)(params)

            # Check that loss is finite
            self.assertTrue(jnp.isfinite(loss_value))

            # Check that all gradients are finite
            def check_finite(x):
                if isinstance(x, jnp.ndarray):
                    self.assertTrue(jnp.all(jnp.isfinite(x)), f"Non-finite gradient found: {x}")

            jax.tree.map(check_finite, grads)

            # If logit sink is enabled, check that sink gradients exist and are finite
            if logit_sink:
                self.assertIn("sink", grads)
                self.assertTrue(jnp.all(jnp.isfinite(grads["sink"])))
            else:
                self.assertNotIn("sink", grads)

    @parameterized.parameters(
        # (partition_spec, expected_mesh_axes, test_description)
        (PartitionSpec(None), (None,), "length_1_short_spec"),
        (PartitionSpec("data", None), (None,), "length_2_short_spec"),
        (PartitionSpec("data", None, "model"), ("model",), "length_3_exact_boundary"),
        (PartitionSpec("data", "seq", "fsdp", "expert"), ("fsdp",), "length_4_long_spec"),
    )
    def test_create_layer_parameter_specs_with_logit_sink(
        self, bsnh_partition_spec, expected_mesh_axes, test_description
    ):
        """Tests _create_layer_parameter_specs with different partition spec lengths."""
        del test_description  # Unused, just for test readability

        num_heads = 4
        per_head_dim = 32
        hidden_dim = num_heads * per_head_dim

        cfg = FlashAttention.default_config().set(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            num_heads=num_heads,
            logit_sink=True,
            mha_dim_to_partition_spec={
                "bsnh": bsnh_partition_spec,
                "btnh": bsnh_partition_spec,
                "bnts": PartitionSpec(None),
            },
            name="test",
        )

        layer = cfg.instantiate(parent=None)
        # pylint: disable-next=protected-access
        param_specs = layer._create_layer_parameter_specs()

        # Check that sink parameter exists and has correct mesh_axes
        self.assertIn("sink", param_specs)
        self.assertEqual(param_specs["sink"].mesh_axes, expected_mesh_axes)

    def test_create_layer_parameter_specs_without_logit_sink(self):
        """Tests _create_layer_parameter_specs when logit_sink is disabled."""
        num_heads = 4
        per_head_dim = 32
        hidden_dim = num_heads * per_head_dim

        cfg = FlashAttention.default_config().set(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            num_heads=num_heads,
            logit_sink=False,
            name="test",
        )

        layer = cfg.instantiate(parent=None)
        # pylint: disable-next=protected-access
        param_specs = layer._create_layer_parameter_specs()

        # Check that sink parameter does not exist when logit_sink is disabled
        self.assertNotIn("sink", param_specs)

    # ---- Backend override and softmax scale tests ----

    def test_backend_override_modifier(self):
        """Tests BackendOverrideModifier."""
        cfg: DummyModel.Config = DummyModel.default_config()
        cfg.layer = FlashAttention.default_config()

        # By default we expect backend_overrides = None
        self.assertIsNone(cfg.layer.backend_overrides)

        cfg_modifier = (
            BackendOverrideModifier.default_config()
            .set(
                backend_overrides=dict(
                    splash_block_kv_compute=2048,
                )
            )
            .instantiate()
        )

        cfg = cfg_modifier(cfg)
        self.assertDictEqual(cfg.layer.backend_overrides, dict(splash_block_kv_compute=2048))

    def test_backend_override_modifier_ignores_none(self):
        """Tests that BackendOverrideModifier ignores overrides values of None."""
        cfg: DummyModel.Config = DummyModel.default_config()
        cfg.layer = FlashAttention.default_config()

        # By default we expect backend_overrides = None
        self.assertIsNone(cfg.layer.backend_overrides)

        cfg_modifier = (
            BackendOverrideModifier.default_config()
            .set(
                backend_overrides=dict(
                    splash_block_kv_compute=2048,
                    splash_block_q=None,
                )
            )
            .instantiate()
        )

        cfg = cfg_modifier(cfg)
        # We expect splash_block_q to not appear in backend_overrides since its value = None
        self.assertDictEqual(cfg.layer.backend_overrides, dict(splash_block_kv_compute=2048))

    @parameterized.parameters(
        None,
        0.5,
        2.0,
        0.125,
    )
    def test_softmax_scale(self, softmax_scale):
        """Tests that softmax_scale is correctly applied in flash attention."""
        batch = 2
        seq_len = 128
        num_heads = 4
        per_head_dim = 32
        hidden_dim = num_heads * per_head_dim
        mesh = (1, 1)
        mesh_axis_names = ("data", "model")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            # Create layer with specified softmax_scale
            cfg = FlashAttention.default_config().set(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                softmax_scale=softmax_scale,
                mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
                output_dim_to_partition_spec=default_output_dim_to_partition_spec(mesh_axis_names),
                tpu_block_size=128,
                name="test",
            )

            layer = cfg.instantiate(parent=None)
            params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

            # Verify config is set correctly
            self.assertEqual(layer.config.softmax_scale, softmax_scale)

            # Create test inputs
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                kv_len=seq_len,
                query_len=seq_len,
                hidden_dim=hidden_dim,
                use_bias=False,
                use_segment_ids=False,
            )

            # Forward pass should work without errors
            output, _ = F(
                layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=inputs,
                is_training=True,
            )

            # Output should be finite
            self.assertTrue(jnp.all(jnp.isfinite(output.data)))
            self.assertEqual(output.data.shape, (batch, seq_len, hidden_dim))

            # Test that different scales produce different outputs
            if softmax_scale is not None:
                # Create reference layer with default scale
                ref_cfg = cfg.set(softmax_scale=None, name="ref")
                ref_layer = ref_cfg.instantiate(parent=None)

                ref_output, _ = F(
                    ref_layer,
                    prng_key=jax.random.PRNGKey(5),
                    state=params,
                    inputs=inputs,
                    is_training=True,
                )

                # Outputs should be different when scale differs
                self.assertFalse(jnp.allclose(output.data, ref_output.data, atol=1e-3))


if __name__ == "__main__":
    absltest.main()

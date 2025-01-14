# Copyright © 2023 Apple Inc.

"""Tests FlashAttention layers."""
# pylint: disable=ungrouped-imports
import math
import os
from unittest import mock

from jax.sharding import PartitionSpec

from axlearn.common.utils import Tensor

# Due to reference layer using XLA,
# set the following environment variables to avoid OOM in GPU tests.
# pylint: disable=wrong-import-position
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# pylint: enable=wrong-import-position

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from axlearn.common.attention import Dropout, GroupedQueryAttention
from axlearn.common.attention_bias import (
    CompositeAttentionBias,
    SegmentIdAttentionBias,
    TensorAttentionBias,
    bool_to_bias,
    sliding_window_causal_mask,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class, config_for_function
from axlearn.common.flash_attention.layer import (
    FlashAttention,
    default_mha_dim_to_partition_spec,
    default_output_dim_to_partition_spec,
)
from axlearn.common.layers import set_bias_recursively
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape
from axlearn.common.utils import TensorSpec


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
    per_head_dim,
    mesh_axis_names,
    causal,
    sliding_window_size,
    inference=False,
    set_layer_bias_recursively=False,
    dropout_rate=0.0,
):
    hidden_dim = num_heads * per_head_dim
    kwargs = dict(
        query_dim=hidden_dim,
        key_dim=hidden_dim,
        value_dim=hidden_dim,
        num_heads=num_heads,
        dtype=jnp.bfloat16,
        dropout=Dropout.default_config().set(rate=dropout_rate),
    )
    ref_cfg = GroupedQueryAttention.default_config().set(**kwargs)

    if inference:
        ref_cfg.input_linear.set(dtype=jnp.bfloat16, cache_dtype=None)
    test_cfg = (
        FlashAttention.default_config()
        .set(**kwargs)
        .set(
            mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
            output_dim_to_partition_spec=default_output_dim_to_partition_spec(mesh_axis_names),
        )
    )
    if inference:
        test_cfg.input_linear.set(dtype=jnp.bfloat16, cache_dtype=None)

    if sliding_window_size is not None:
        assert causal
        mask_fn = config_for_function(sliding_window_causal_mask).set(
            sliding_window_size=sliding_window_size
        )
        ref_cfg.set(mask=mask_fn)
        test_cfg.set(mask=mask_fn)
    else:
        ref_cfg.set(causal=causal)
        test_cfg.set(causal=causal)

    set_bias_recursively(ref_cfg, set_layer_bias_recursively)
    set_bias_recursively(test_cfg, set_layer_bias_recursively)

    ref_layer = ref_cfg.set(name="ref").instantiate(parent=None)
    test_layer = test_cfg.set(name="test").instantiate(parent=None)

    # Use the same params for both. Only attention implementation differs.
    params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
    return test_layer, ref_layer, params, hidden_dim


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer."""

    _TEST_CONFIGS = [
        dict(
            batch=2,
            seq_len=384,
            num_heads=4,
            per_head_dim=32,
            mesh=(1, 1),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=2,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 1),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=2,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 1),
            mesh_axis_names=("data", "fsdp"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(2, 2),
            mesh_axis_names=("data", "fsdp"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(8, 1),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(4, 1),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(2, 2),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=128,
            mesh=(2, 2),
            mesh_axis_names=("data", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 1, 8, 1),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 1, 4, 1),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 1, 8),
            mesh_axis_names=("data", "expert", "fsdp"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 1, 4),
            mesh_axis_names=("data", "expert", "fsdp"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 2, 4, 1),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 2, 2, 1),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 1, 2, 2),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 2, 1, 2, 1),
            mesh_axis_names=("data", "seq", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=64,
            mesh=(1, 2, 2, 2),
            mesh_axis_names=("data", "expert", "fsdp", "model"),
        ),
        dict(
            batch=8,
            seq_len=2048,
            num_heads=4,
            per_head_dim=128,
            mesh=(1, 2, 1, 2, 2),
            mesh_axis_names=("data", "seq", "expert", "fsdp", "model"),
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

    @parameterized.parameters(
        [kwargs for kwargs in _TEST_CONFIGS if math.prod(kwargs["mesh"]) == 1]
    )
    def test_backend(self, batch, seq_len, num_heads, per_head_dim, mesh, mesh_axis_names):
        del batch, seq_len
        mock_device = mock.Mock(spec=jax.Device)
        mock_device.platform = "tpu"
        mock_device.coords = (0, 0, 0)
        mock_device.core_on_chip = 0
        devices = [mock_device]

        with Mesh(mesh_utils.create_device_mesh(mesh, devices), mesh_axis_names):
            test_layer, _, _, _ = _prepare_layers(
                num_heads=num_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                causal=True,
                sliding_window_size=None,
            )
            backend = test_layer._backend()  # pylint: disable=protected-access
            self.assertEqual(backend, "tpu")

    @parameterized.parameters(_TEST_CONFIGS)
    def test_shard_biases(self, batch, seq_len, num_heads, per_head_dim, mesh, mesh_axis_names):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")

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
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                causal=True,
                sliding_window_size=None,
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

    @parameterized.product(
        _TEST_CONFIGS,
        query_len_multiplier=[0.5, 1, 2],
        causal=[False, True],
        sliding_window_size=[None, 4],
        use_bias=[False, True],
        use_segment_ids=[False, True],
        input_dtype=[jnp.bfloat16, jnp.float32],
        dropout_rate=[0.0, 0.1],
    )
    def test_forward(
        self,
        batch,
        seq_len,
        num_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        query_len_multiplier,
        causal,
        sliding_window_size,
        use_bias,
        use_segment_ids,
        input_dtype,
        dropout_rate,
    ):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")
        if not causal and sliding_window_size is not None:
            pytest.skip(reason="Sliding window attention must be causal.")
        if causal and use_bias:
            # TODO(c_lan): Investigate the numerical errors when both causal and bias are used.
            pytest.skip(reason="Only one of causal and use_bias can be True.")
        if use_segment_ids and query_len_multiplier != 1:
            pytest.skip("Segment IDs are not supported for Q and K with different lengths.")
        # Data=1 with bias matrix in all fp32 format would OOM the H100 SRAM.
        if use_bias and mesh[mesh_axis_names.index("data")] == 1 and input_dtype == jnp.float32:
            pytest.skip(reason="Unsupported large bias matrix in fp32 format.")
        if dropout_rate > 0.0 and jax.default_backend() == "tpu":
            pytest.skip("Dropout is implemented for GPU only.")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            test_layer, ref_layer, params, hidden_dim = _prepare_layers(
                num_heads=num_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                causal=causal,
                sliding_window_size=sliding_window_size,
                dropout_rate=dropout_rate,
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
            if dropout_rate == 0.0 or use_segment_ids:
                self.assertNestedAllClose(ref_out.data, test_out.data, atol=0.05)
        jax.extend.backend.clear_backends()

    @parameterized.product(
        _TEST_CONFIGS,
        query_len_multiplier=[0.5, 1, 2],
        causal=[False, True],
        sliding_window_size=[None, 4],
        use_bias=[False, True],
        use_segment_ids=[False, True],
        set_layer_bias_recursively=[False, True],
        dropout_rate=[0.0, 0.1],
    )
    def test_backward(
        self,
        batch,
        seq_len,
        num_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        query_len_multiplier,
        causal,
        sliding_window_size,
        use_bias,
        use_segment_ids,
        set_layer_bias_recursively,
        dropout_rate,
    ):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")
        if use_segment_ids and query_len_multiplier != 1:
            pytest.skip("Segment IDs are not supported for Q and K with different lengths.")
        if not causal and sliding_window_size is not None:
            pytest.skip(reason="Sliding window attention must be causal.")
        if sliding_window_size is not None and query_len_multiplier > 1:
            # When sliding window is enabled and q_len > kv_len, there might be be fully masked
            # rows.
            pytest.skip(reason="Sliding window attention does not make sense when q_len > kv_len.")
        if dropout_rate > 0.0 and jax.default_backend() == "tpu":
            pytest.skip("Dropout is implemented for GPU only.")

        if causal and use_bias:
            # TODO(c_lan): Investigate the numerical errors when both causal and bias are used.
            pytest.skip(reason="Only one of causal and use_bias can be True.")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):

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

            hidden_dim = num_heads * per_head_dim

            if sliding_window_size is not None:
                mask_fn = config_for_function(sliding_window_causal_mask).set(
                    sliding_window_size=sliding_window_size
                )
            else:
                mask_fn = None

            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                causal=causal and (mask_fn is None),
                mask=mask_fn,
                dropout=Dropout.default_config().set(rate=dropout_rate),
            )
            ref_cfg = DummyModel.default_config().set(
                layer=GroupedQueryAttention.default_config().set(**kwargs),
            )
            test_cfg = DummyModel.default_config().set(
                layer=FlashAttention.default_config().set(
                    tpu_block_size=128,
                    mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
                    output_dim_to_partition_spec=default_output_dim_to_partition_spec(
                        mesh_axis_names
                    ),
                    **kwargs,
                )
            )
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
                atol, rtol = 2.5e-4, 1e-3
            # Can be 1e-5 on x86_64/GPU/TPU, needed to be slightly higher on ARM.
            else:
                atol, rtol = 1e-4, 1e-3

            # Note: cannot compare results when dropout_rate > 0 and not using segment ids, because
            # cudnn dropout will be used and it uses different PRNG than ours.
            if dropout_rate == 0.0 or use_segment_ids:
                self.assertNestedAllClose(ref_value, test_value, atol=atol, rtol=rtol)
                self.assertNestedAllClose(ref_grads, test_grads, atol=atol, rtol=rtol)
        jax.extend.backend.clear_backends()

    @parameterized.product(
        _TEST_CONFIGS, causal=[True], sliding_window_size=[None, 4], use_bias=[True, False]
    )
    def test_extend_step(
        self,
        batch,
        seq_len,
        num_heads,
        per_head_dim,
        mesh,
        mesh_axis_names,
        causal,
        sliding_window_size,
        use_bias,
    ):
        print(
            f"batch={batch}, seq_len={seq_len} (ignored->16), num_heads={num_heads}, \n"
            f"per_head_dim={per_head_dim}, mesh={mesh}, mesh_axis_names={mesh_axis_names}, \n"
            f"causal={causal}, sliding_window_size={sliding_window_size}"
        )
        # Limit generation length to 16 to save test time.
        seq_len = 16
        dtype = jnp.bfloat16

        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")
        if not causal and sliding_window_size is not None:
            pytest.skip(reason="Sliding window attention must be causal.")

        named_sharding = dict(zip(mesh_axis_names, mesh))
        if "seq" in named_sharding and named_sharding["seq"] > 1:
            pytest.skip(reason="Unsupported seq dim sharding for decoding.")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            test_layer, ref_layer, params, hidden_dim = _prepare_layers(
                num_heads=num_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                causal=causal,
                sliding_window_size=sliding_window_size,
                inference=True,
            )
            tpu_block_size = test_layer.config.tpu_block_size
            # pylint: disable-next=protected-access
            if test_layer._backend() == "tpu" and seq_len % tpu_block_size != 0:
                pytest.skip(
                    f"Sequence length  {seq_len} is not divisible by configured block size for "
                    f"tpu {test_layer.config.tpu_block_size }. "
                    "This was unsupported (and the test failed) even prior to adding "
                    "this skip statement."
                )

            # Prepare inputs
            query = jax.random.normal(
                jax.random.PRNGKey(0),
                [batch, seq_len, hidden_dim],
                dtype=dtype,
            )
            causal_bias = None
            if use_bias:
                causal_bias = jax.random.normal(
                    jax.random.PRNGKey(0),
                    [batch, num_heads, seq_len, seq_len],
                    dtype=dtype,
                )
            kv_state = None
            return_aux = {"probs"}

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
            initial_state, initial_output = test_layer.init_states(
                time_step=None,
                query=TensorSpec([batch, seq_len]),
                kv_state=kv_state,
                attention_logit_biases=None,
            )
            ref_initial_state, ref_inital_output = ref_layer.init_states(
                time_step=None,
                query=TensorSpec([batch, seq_len]),
                kv_state=kv_state,
                attention_logit_biases=None,
            )
            self.assertIsNone(initial_output)
            self.assertIsNone(ref_inital_output)
            for k in ["key", "value"]:
                self.assertEqual(ref_initial_state["i_proj"][k].dtype, dtype)
                self.assertEqual(initial_state["i_proj"][k].dtype, dtype)

            # Prepare decoding inputs.
            inputs = dict(
                cached_states=initial_state,
                kv_state=kv_state,
                return_aux=return_aux,
                attention_logit_biases=None,
            )
            ref_inputs = dict(
                cached_states=ref_initial_state,
                kv_state=kv_state,
                return_aux=return_aux,
                attention_logit_biases=None,
            )

            decoder_output = jnp.zeros(shape=[seq_len, batch, hidden_dim]).astype(dtype)
            ref_decoder_output = jnp.zeros(shape=[seq_len, batch, hidden_dim]).astype(dtype)
            for t in range(seq_len):
                cur_query = jnp.expand_dims(query[:, t, :], axis=1)
                inputs["query"] = cur_query
                if use_bias:
                    inputs["attention_logit_biases"] = jnp.expand_dims(
                        causal_bias[:, :, t, :], axis=2
                    )

                ref_inputs["query"] = cur_query
                if use_bias:
                    ref_inputs["attention_logit_biases"] = jnp.expand_dims(
                        causal_bias[:, :, t, :], axis=2
                    )

                ref_extend_step_outputs, _ = F(
                    ref_layer,
                    state=params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(5),
                    inputs=ref_inputs,
                    method="extend_step",
                )
                ref_inputs["cached_states"] = ref_extend_step_outputs[0]
                ref_decoder_output = ref_decoder_output.at[t].set(
                    jnp.squeeze(ref_extend_step_outputs[1].data, axis=1)
                )

                extend_step_outputs, _ = F(
                    test_layer,
                    state=params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(5),
                    inputs=inputs,
                    method="extend_step",
                )
                inputs["cached_states"] = extend_step_outputs[0]
                decoder_output = decoder_output.at[t].set(
                    jnp.squeeze(extend_step_outputs[1].data, axis=1)
                )

                self.assertNestedAllClose(
                    decoder_output[t],
                    ref_decoder_output[t],
                    atol=2e-2,
                )

            decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
            ref_decoder_out_transposed = jnp.transpose(ref_decoder_output, [1, 0, 2])
            # Golden Reference still need to adjust for bf16 loss.
            self.assertNestedAllClose(
                ref_out.data,
                ref_decoder_out_transposed,
                atol=2e-2,
            )
            self.assertNestedAllClose(
                decoder_out_transposed,
                ref_decoder_out_transposed,
                atol=2e-2,
            )
            self.assertNestedAllClose(
                ref_out.data,
                test_out.data,
                atol=2e-2,
            )
        jax.extend.backend.clear_backends()

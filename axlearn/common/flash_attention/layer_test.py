# Copyright © 2023 Apple Inc.

"""Tests FlashAttention layers."""
import math
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from axlearn.common.attention import (
    GroupedQueryAttention,
    _bool_to_bias,
    apply_attention_logit_biases,
    make_causal_biases,
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


def _fake_inputs(
    *,
    batch: int,
    num_heads: int,
    seq_len: int,
    hidden_dim: int,
    causal: bool,
    use_bias: bool,
    use_segment_ids: bool,
):
    query = jax.random.normal(
        jax.random.PRNGKey(0),
        [batch, seq_len, hidden_dim],
        dtype=jnp.bfloat16,
    )
    if causal:
        key = value = None
    else:
        key = jax.random.normal(
            jax.random.PRNGKey(1),
            [batch, seq_len, hidden_dim],
            dtype=jnp.bfloat16,
        )
        value = jax.random.normal(
            jax.random.PRNGKey(2),
            [batch, seq_len, hidden_dim],
            dtype=jnp.bfloat16,
        )
    if use_bias:
        bias = jax.random.bernoulli(
            jax.random.PRNGKey(3), p=0.5, shape=[batch, num_heads, seq_len, seq_len]
        )
        bias = _bool_to_bias(bias)
    else:
        bias = None
    if use_segment_ids:
        segment_ids = jnp.ones([batch, seq_len], dtype=jnp.int32)
    else:
        segment_ids = None
    return dict(
        query=query, key=key, value=value, attention_logit_biases=bias, segment_ids=segment_ids
    )


def _prepare_layers(
    *, num_heads, per_head_dim, mesh_axis_names, causal, sliding_window_size, inference=False
):
    hidden_dim = num_heads * per_head_dim
    kwargs = dict(
        query_dim=hidden_dim,
        key_dim=hidden_dim,
        value_dim=hidden_dim,
        num_heads=num_heads,
        dtype=jnp.bfloat16,
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

    set_bias_recursively(ref_cfg, False)
    set_bias_recursively(test_cfg, False)

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
            per_head_dim=64,
            mesh=(1, 2, 1, 2, 2),
            mesh_axis_names=("data", "seq", "expert", "fsdp", "model"),
        ),
    ]

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
        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            test_layer, _, _, _ = _prepare_layers(
                num_heads=num_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                causal=True,
                sliding_window_size=None,
            )
            bias = jnp.ones((batch, num_heads, seq_len, seq_len))
            spec = test_layer._logit_biases_spec(bias)  # pylint: disable=protected-access
            self.assertEqual(spec, test_layer.config.mha_dim_to_partition_spec["bnts"])

            bias = jnp.ones((batch, 1, seq_len, seq_len))
            spec = test_layer._logit_biases_spec(bias)  # pylint: disable=protected-access
            self.assertEqual(spec[1], None)

            bias = jnp.ones((1, 1, seq_len, seq_len))
            spec = test_layer._logit_biases_spec(bias)  # pylint: disable=protected-access
            self.assertEqual(spec[0], None)
            self.assertEqual(spec[1], None)

    @parameterized.product(
        _TEST_CONFIGS,
        causal=[False, True],
        sliding_window_size=[None, 4],
        use_bias=[False, True],
        use_segment_ids=[False, True],
    )
    def test_forward(
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
        use_segment_ids,
    ):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")
        if not causal and sliding_window_size is not None:
            pytest.skip(reason="Sliding window attention must be causal.")

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            test_layer, ref_layer, params, hidden_dim = _prepare_layers(
                num_heads=num_heads,
                per_head_dim=per_head_dim,
                mesh_axis_names=mesh_axis_names,
                causal=causal,
                sliding_window_size=sliding_window_size,
            )

            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                causal=causal,
                use_bias=use_bias,
                use_segment_ids=use_segment_ids,
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
            self.assertNestedAllClose(ref_out.data, test_out.data, atol=0.05)

    @parameterized.product(
        _TEST_CONFIGS,
        causal=[False, True],
        sliding_window_size=[None, 4],
        use_bias=[False, True],
        use_segment_ids=[False, True],
    )
    def test_backward(
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
        use_segment_ids,
    ):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")

        if not causal and sliding_window_size is not None:
            pytest.skip(reason="Sliding window attention must be causal.")

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
                causal = False
            else:
                mask_fn = None

            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                causal=causal,
                mask=mask_fn,
            )
            ref_cfg = DummyModel.default_config().set(
                layer=GroupedQueryAttention.default_config().set(**kwargs),
            )
            test_cfg = DummyModel.default_config().set(
                layer=FlashAttention.default_config()
                .set(**kwargs, tpu_block_size=128)
                .set(
                    mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
                    output_dim_to_partition_spec=default_output_dim_to_partition_spec(
                        mesh_axis_names
                    ),
                )
            )
            set_bias_recursively(ref_cfg, False)
            set_bias_recursively(test_cfg, False)
            ref_layer = ref_cfg.set(name="ref").instantiate(parent=None)
            test_layer = test_cfg.set(name="test").instantiate(parent=None)
            # Use the same params for both. Only attention implementation differs.
            params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                causal=causal,
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
            # Can be 1e-5 on x86_64/GPU/TPU, needed to be slightly higher on ARM.
            atol = 2e-5
            self.assertNestedAllClose(ref_value, test_value, atol=atol)
            self.assertNestedAllClose(ref_grads, test_grads, atol=atol)

    @parameterized.product(_TEST_CONFIGS, causal=[True], sliding_window_size=[None, 4])
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
    ):
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
            bias = jax.random.normal(
                jax.random.PRNGKey(0),
                [batch, num_heads, seq_len, seq_len],
                dtype=dtype,
            )
            # Note: We need to use causal bias for flash attention input in case of decoding.
            causal_bias = apply_attention_logit_biases(bias, make_causal_biases(seq_len)).astype(
                dtype
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
            initial_state = test_layer.init_states(
                target_batch_size=batch, target_max_len=seq_len, kv_state=kv_state
            )
            ref_initial_state = test_layer.init_states(
                target_batch_size=batch, target_max_len=seq_len, kv_state=kv_state
            )
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
                cached_states=ref_initial_state, kv_state=kv_state, return_aux=return_aux
            )

            decoder_output = jnp.zeros(shape=[seq_len, batch, hidden_dim]).astype(dtype)
            ref_decoder_output = jnp.zeros(shape=[seq_len, batch, hidden_dim]).astype(dtype)
            for t in range(seq_len):
                cur_query = jnp.expand_dims(query[:, t, :], axis=1)
                inputs["query"] = cur_query
                inputs["attention_logit_biases"] = jnp.expand_dims(causal_bias[:, :, t, :], axis=2)

                ref_inputs["query"] = cur_query
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

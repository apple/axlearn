# Copyright © 2023 Apple Inc.

"""Tests LoRa implementations."""
# pylint: disable=no-self-use

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as torchF
from absl.testing import absltest, parameterized
from einops import rearrange

from axlearn.common import utils
from axlearn.common.attention import (
    FusedQKVLinear,
    KVState,
    MultiheadAttention,
    MultiheadOutputLinear,
    QKVLinear,
    QLinear,
)
from axlearn.common.attention_bias import CausalAttentionBias
from axlearn.common.layers import Linear
from axlearn.common.lora import (
    LoraFusedQKVAdapter,
    LoraFusedQKVLinear,
    LoraLinear,
    LoraLinearAdapter,
    LoraMultiheadOutputLinear,
)
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import Tensor


class LoraLinearTest(TestCase):
    def test_set_param_spec_config(self):
        layer_cfg = LoraLinearAdapter.default_config().set(
            name="test",
            rank=2,
            alpha=1.0,
            input_dim=2,
            output_dim=4,
        )
        layer_cfg.lora_down.param_partition_spec = ["data", None]
        layer_cfg.lora_up.param_partition_spec = [None, "data"]
        layer = layer_cfg.instantiate(parent=None)
        param_specs = layer.create_parameter_specs_recursively()
        self.assertEqual(param_specs["lora_down"]["weight"].mesh_axes, ("data", None))
        self.assertEqual(param_specs["lora_up"]["weight"].mesh_axes, (None, "data"))

    def test_forward(self):
        input_dim = 2
        output_dim = 4
        batch_size = 2
        rank = 2
        alpha = 8

        inputs = jax.random.normal(jax.random.PRNGKey(456), (batch_size, input_dim))

        ref_layer_cfg = Linear.default_config().set(
            name="ref_layer", input_dim=input_dim, output_dim=output_dim
        )
        ref_layer = ref_layer_cfg.instantiate(parent=None)
        ref_state = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        ref_outputs, _ = F(
            ref_layer,
            state=ref_state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )
        layer_cfg = LoraLinear.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
        )
        layer_cfg.adapter.set(rank=rank, alpha=alpha)
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123), prebuilt=None
        )
        state["layer"]["weight"] = ref_state["weight"]
        state["layer"]["bias"] = ref_state["bias"]
        outputs, _ = F(
            layer,
            state=state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )

        assert_allclose(outputs, ref_outputs)

    def test_alpha_is_zero(self):
        input_dim = 2
        output_dim = 4
        batch_size = 2
        rank = 2
        alpha = 0

        inputs = jax.random.normal(jax.random.PRNGKey(456), (batch_size, input_dim))

        ref_layer_cfg = Linear.default_config().set(
            name="ref_layer", input_dim=input_dim, output_dim=output_dim
        )
        ref_layer = ref_layer_cfg.instantiate(parent=None)
        ref_state = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        ref_outputs, _ = F(
            ref_layer,
            state=ref_state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )
        layer_cfg = LoraLinear.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
        )
        layer_cfg.adapter.set(rank=rank, alpha=alpha)
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123), prebuilt=None
        )
        state["layer"]["weight"] = ref_state["weight"]
        state["layer"]["bias"] = ref_state["bias"]
        state["adapter"]["lora_up"]["weight"] = jax.random.normal(
            jax.random.PRNGKey(1), (rank, output_dim)
        )
        outputs, _ = F(
            layer,
            state=state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )

        self.assertNestedEqual(outputs, ref_outputs)


class LoraFusedQKVLinearTest(TestCase):
    @parameterized.parameters(
        (QKVLinear.default_config(),),
        (FusedQKVLinear.default_config(),),
        (QLinear.default_config(),),
    )
    def test_forward(self, ref_layer_cfg):
        test_layer_cfg = LoraFusedQKVLinear.default_config().set(layer=ref_layer_cfg)
        model_dim = 6
        num_heads = 2
        per_head_dim = 3
        seq_len = 4
        batch_size = 2
        rank = 2
        alpha = 4
        enable_lora = dict(query=True, key=False, value=True)
        inputs = jax.random.normal(jax.random.PRNGKey(456), (batch_size, seq_len, model_dim))
        if isinstance(ref_layer_cfg, QLinear.Config):
            external_key = jax.random.normal(
                jax.random.PRNGKey(78), (batch_size, seq_len, num_heads, per_head_dim)
            )
            external_value = jax.random.normal(
                jax.random.PRNGKey(90), (batch_size, seq_len, num_heads, per_head_dim)
            )
            key_positions = jnp.arange(seq_len)[None]
            inputs = dict(
                query=inputs,
                kv_state=KVState(
                    k_proj=external_key, v_proj=external_value, key_positions=key_positions
                ),
            )
        else:
            inputs = (inputs,)

        ref_layer_cfg = ref_layer_cfg.set(
            name="ref_test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
        )
        ref_layer = ref_layer_cfg.instantiate(parent=None)
        ref_state = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        ref_outputs, _ = F(
            ref_layer,
            state=ref_state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=inputs,
        )

        layer_cfg = test_layer_cfg.set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
        )
        layer_cfg.adapter.set(rank=rank, alpha=alpha, enable_lora=enable_lora)
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123), prebuilt=None
        )
        for layer_type in ("qkv_proj", "q_proj", "k_proj", "v_proj"):
            if layer_type in ref_state:
                state["layer"][layer_type]["weight"] = ref_state[layer_type]["weight"]
                state["layer"][layer_type]["bias"] = ref_state[layer_type]["bias"]
        outputs, _ = jax.jit(partial(F, layer, is_training=True))(
            state=state,
            prng_key=jax.random.PRNGKey(456),
            inputs=inputs,
        )

        # Expect the same output due to zero initialization of one of the LoRA weights.
        assert_allclose(outputs, ref_outputs)

    @parameterized.parameters(
        (QKVLinear.default_config(),),
        (FusedQKVLinear.default_config(),),
    )
    def test_extend_step(self, ref_layer_cfg):
        model_dim = 6
        num_heads = 2
        lora_linear = LoraFusedQKVLinear.default_config().set(layer=ref_layer_cfg)
        rank = 2
        alpha = 4
        enable_lora = dict(query=True, key=False, value=True)
        lora_linear.adapter.set(rank=rank, alpha=alpha, enable_lora=enable_lora)
        cfg = MultiheadAttention.default_config().set(
            name="test",
            input_linear=lora_linear,
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            mask=CausalAttentionBias.default_config(),
        )
        layer = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Generate input sequences.
        batch, seq_len = 2, 10
        prng_key, data_key = jax.random.split(prng_key)
        query = jax.random.uniform(data_key, [batch, seq_len, model_dim])

        # Compute layer outputs.
        fwd_outputs, _ = F(
            layer,
            inputs=dict(query=query),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute extend_step.
        (cached_states, _), _ = F(
            layer,
            inputs=dict(time_step=None, query=query),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
            method="init_states",
        )
        step_data = []
        for i in range(seq_len):
            step_inputs = dict(
                cached_states=cached_states,
                query=query[:, i : i + 1],
            )
            (cached_states, step_outs), _ = F(
                layer,
                prng_key=jax.random.PRNGKey(0),
                state=layer_params,
                inputs=step_inputs,
                is_training=False,
                method="extend_step",
            )
            step_data.append(step_outs.data)
        step_data = jnp.concatenate(step_data, axis=1)
        self.assertEqual(step_data.dtype, fwd_outputs.data.dtype)
        assert_allclose(step_data, fwd_outputs.data)


class LoraMultiheadOutputLinearTest(TestCase):
    def test_forward(self):
        model_dim = 8
        num_heads = 2
        per_head_dim = 4
        batch_size = 2
        seq_len = 6
        rank = 2
        alpha = 8

        inputs = jax.random.normal(
            jax.random.PRNGKey(456), (batch_size, seq_len, num_heads, per_head_dim)
        )

        ref_layer_cfg = MultiheadOutputLinear.default_config().set(
            name="ref_layer",
            model_dim=model_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
        )
        ref_layer = ref_layer_cfg.instantiate(parent=None)
        ref_state = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        ref_outputs, _ = F(
            ref_layer,
            state=ref_state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )
        layer_cfg = LoraMultiheadOutputLinear.default_config().set(
            name="test",
            model_dim=model_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
        )
        layer_cfg.adapter.set(rank=rank, alpha=alpha)
        layer = layer_cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123), prebuilt=None
        )
        state["layer"]["weight"] = ref_state["weight"]
        state["layer"]["bias"] = ref_state["bias"]
        outputs, _ = F(
            layer,
            state=state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )

        assert_allclose(outputs, ref_outputs)


class LoraFusedQKVAdapterTest(TestCase):
    def test_initialization(self):
        with utils.numeric_checks(True):
            input_dim = 5
            output_dim = 8
            num_heads = 2
            rank = 2
            alpha = 4
            layer = (
                LoraFusedQKVAdapter.default_config()
                .set(
                    name="test",
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=rank,
                    alpha=alpha,
                    num_heads=num_heads,
                    enable_lora=dict(query=True, key=False, value=True),
                )
                .instantiate(parent=None)
            )
            state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            self.assertEqual(jnp.sum(state["lora_up"]["weight"]), 0.0)

    @parameterized.parameters(
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    )
    def test_forward_torch(self, enable_q, enable_k, enable_v):
        batch_size = 3
        seq_len = 2
        input_dim = 10
        output_dim = 10
        num_heads = 2
        rank = 2
        alpha = 12
        enable_lora = dict(query=enable_q, key=enable_k, value=enable_v)
        torch_enable_lora = [enable_q, enable_k, enable_v]

        inputs = jax.random.normal(jax.random.PRNGKey(456), (batch_size, seq_len, input_dim))

        layer = (
            LoraFusedQKVAdapter.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                output_dim=output_dim,
                rank=rank,
                alpha=alpha,
                num_heads=num_heads,
                enable_lora=enable_lora,
            )
            .instantiate(parent=None)
        )
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        state["lora_up"]["weight"] = jax.random.normal(
            jax.random.PRNGKey(1), state["lora_up"]["weight"].shape
        )
        state["lora_down"]["weight"] = jax.random.normal(
            jax.random.PRNGKey(2), state["lora_down"]["weight"].shape
        )
        proj, _ = jax.jit(partial(F, layer, is_training=True))(
            state=state,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )
        outputs = jnp.zeros((3, *proj.shape[1:]), dtype=proj.dtype)
        outputs = outputs.at[np.array(torch_enable_lora)].add(proj)
        outputs = rearrange(outputs, "p b t n h -> b t (p n h)")

        # [B, T, r * s]
        lora_a = as_torch_tensor(rearrange(state["lora_down"]["weight"], "d p r -> d (p r)"))
        lora_b = as_torch_tensor(
            rearrange(state["lora_up"]["weight"], "p r n h -> r (p n h)")
        ).transpose(-2, -1)
        torch_inputs = as_torch_tensor(inputs)
        after_a = torch_inputs @ lora_a
        after_b = torchF.conv1d(
            after_a.transpose(-2, -1), lora_b.unsqueeze(-1), groups=sum(torch_enable_lora)
        )
        torch_out = after_b.transpose(-2, -1)

        out_features = output_dim * 3
        lora_ind = torch.zeros((out_features,), dtype=torch.bool).view(3, -1)
        lora_ind[torch_enable_lora, :] = True
        lora_ind = lora_ind.view(-1)

        def zero_pad(x: Tensor):
            result = x.new_zeros(*x.shape[:-1], out_features)
            result = result.view(-1, out_features)
            result[:, lora_ind] = x.reshape(-1, out_features // 3 * sum(torch_enable_lora))
            return result.view((*x.shape[:-1], out_features))

        torch_out = zero_pad(torch_out) * alpha / rank
        assert_allclose(outputs, torch_out.detach().numpy())


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

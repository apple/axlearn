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
    MultiheadOutputLinear,
    QKVLinear,
    QLinear,
    RoFormerQKVLinear,
)
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
from axlearn.common.utils import Tensor, TensorSpec


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
            inputs = dict(
                query=inputs, kv_state=KVState(k_proj=external_key, v_proj=external_value)
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

    @parameterized.product(
        layer=(
            FusedQKVLinear.default_config(),
            RoFormerQKVLinear.default_config().set(
                rotary_value=False, input_linear=FusedQKVLinear.default_config()
            ),
        ),
    )
    def test_extend_step(self, layer):
        model_dim = 16
        num_heads = 2
        per_head_dim = 4  # change this to 4 to adapt the need of RoPE.
        seq_len = 4
        batch_size = 2
        rank = 2
        alpha = 4
        enable_lora = dict(query=True, key=False, value=True)
        num_enabled = sum(enable_lora.values())
        inputs = jax.random.normal(jax.random.PRNGKey(456), (batch_size, seq_len, model_dim))

        layer_cfg = LoraFusedQKVLinear.default_config().set(
            layer=layer,
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
        state["adapter"]["lora_up"]["weight"] = jax.random.normal(
            jax.random.PRNGKey(1), (num_enabled, rank, num_heads, per_head_dim)
        )
        outputs, _ = jax.jit(partial(F, layer, is_training=False))(
            state=state,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )
        q_proj, k_proj, v_proj = outputs
        forward_outputs = jnp.stack([q_proj, k_proj, v_proj])

        initial_cache_state, init_output = layer.init_states(
            time_step=None,
            query=TensorSpec([batch_size, seq_len], dtype=q_proj.dtype),
        )
        self.assertIsNone(init_output)

        decoder_inputs = dict(cached_states=initial_cache_state)
        decoder_outputs = jnp.zeros(shape=[seq_len, 3, batch_size, num_heads, per_head_dim])
        for t in range(seq_len):
            decoder_inputs["query"] = jnp.expand_dims(inputs[:, t, :], axis=1)
            (updated_states, outputs), _ = F(
                layer,
                state=state,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=decoder_inputs,
                method="extend_step",
            )
            decoder_inputs["cached_states"] = updated_states
            q_proj, k_proj, v_proj = outputs
            k_proj = jnp.expand_dims(k_proj[:, t, :, :], axis=1)
            v_proj = jnp.expand_dims(v_proj[:, t, :, :], axis=1)

            decoder_outputs = decoder_outputs.at[t].set(
                jnp.squeeze(jnp.stack([q_proj, k_proj, v_proj]), axis=2)
            )
        decoder_out_transposed = jnp.transpose(decoder_outputs, [1, 2, 0, 3, 4])
        assert_allclose(
            decoder_out_transposed,
            forward_outputs,
            atol=1e-6,
        )

    def test_prefill_states(self):
        model_dim = 16
        num_heads = 2
        per_head_dim = 3
        seq_len = 4
        batch_size = 2
        rank = 2
        alpha = 4
        enable_lora = dict(query=True, key=False, value=True)
        num_enabled = sum(enable_lora.values())
        inputs = jax.random.normal(jax.random.PRNGKey(456), (batch_size, seq_len, model_dim))

        layer_cfg = LoraFusedQKVLinear.default_config().set(
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
        state["adapter"]["lora_up"]["weight"] = jax.random.normal(
            jax.random.PRNGKey(1), (num_enabled, rank, num_heads, per_head_dim)
        )
        forward_outputs, _ = jax.jit(partial(F, layer, is_training=False))(
            state=state,
            prng_key=jax.random.PRNGKey(456),
            inputs=(inputs,),
        )

        time_step = jnp.arange(batch_size)
        (initial_cache_states, initial_outputs), _ = F(
            layer,
            state=state,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(time_step=time_step, query=inputs),
            method="init_states",
        )
        time_step_mask = jnp.arange(seq_len) < time_step[:, None]
        # [batch, tgt_len, num_heads, per_head_dim].
        decoder_outputs = initial_outputs.query * time_step_mask[..., None, None]
        decoder_inputs = dict(cached_states=initial_cache_states)
        while jnp.any(time_step < seq_len):
            decoder_inputs["query"] = jnp.take_along_axis(
                inputs, time_step[:, None, None], axis=1, mode="clip"
            )
            (updated_states, outputs), _ = F(
                layer,
                state=state,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=decoder_inputs,
                method="extend_step",
            )
            decoder_inputs["cached_states"] = updated_states
            q_proj, _, _ = outputs

            # [batch, tgt_len, 1, 1].
            oh_indices = jax.nn.one_hot(time_step, seq_len)[:, :, None, None]
            decoder_outputs = decoder_outputs + q_proj * oh_indices
            time_step = time_step + 1

        assert_allclose(
            decoder_outputs,
            forward_outputs.query,
            atol=1e-6,
        )


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

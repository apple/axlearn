# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Test for mixture_of_experts.py"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils

from axlearn.common.attention import (
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerFeedForwardLayer,
    TransformerLayer,
)
from axlearn.common.layers import set_bias_recursively
from axlearn.common.mixture_of_experts import (
    Top2Gating,
    TransformerFeedForwardMoE,
    _convert_feedforward_to_moe_parameters,
    convert_dense_to_moe_parameters,
)
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.common.utils import get_recursively, set_recursively, shapes


# pylint: disable=no-self-use,protected-access
class TransformerFeedForwardMoETest(parameterized.TestCase):
    @parameterized.product(is_training=(True, False), outer_batch=(1, 2))
    def test_moe_layer_forward(self, is_training, outer_batch):
        batch_size = 4
        seq_len = 128
        input_dim = 8

        cfg = TransformerFeedForwardMoE.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.hidden_dim = 32
        cfg.num_experts = 8
        cfg.num_groups = 4
        cfg.outer_batch = outer_batch
        layer: TransformerFeedForwardMoE = cfg.instantiate(parent=None)

        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        outputs, _ = F(
            layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        self.assertEqual((batch_size, seq_len, input_dim), outputs.shape)

    def test_moe_layer_aux_loss(self):
        batch_size = 4
        seq_len = 128
        input_dim = 8

        cfg = TransformerFeedForwardMoE.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.hidden_dim = 32
        cfg.num_experts = 8
        cfg.num_groups = 4

        cfg0 = cfg.clone(load_balance_loss_weight=0.01, router_z_loss_weight=0.001)
        layer0: TransformerFeedForwardMoE = cfg0.instantiate(parent=None)
        state = layer0.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        _, collections0 = F(
            layer0,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
            drop_output_collections=[],
        )
        aux_loss0 = collections0.module_outputs["aux_loss"]

        cfg1 = cfg.clone(load_balance_loss_weight=0.1, router_z_loss_weight=0.01)
        layer1: TransformerFeedForwardMoE = cfg1.instantiate(parent=None)
        _, collections1 = F(
            layer1,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
            drop_output_collections=[],
        )
        aux_loss1 = collections1.module_outputs["aux_loss"]
        self.assertNotEqual(aux_loss0, aux_loss1)
        self.assertNotEqual(aux_loss0, 0.0)
        self.assertNotEqual(aux_loss1, 0.0)

    @parameterized.product(
        copy_weights=(True, False),
        num_experts=(1, 8),
        activation=("nn.relu", ("nn.silu", "linear")),
    )
    def test_moe_and_dense_layer_parity(self, copy_weights, num_experts, activation):
        batch_size = 2
        seq_len = 128
        input_dim = 16
        hidden_dim = 64

        cfg = TransformerFeedForwardMoE.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.hidden_dim = hidden_dim
        cfg.num_experts = num_experts
        cfg.activation = activation
        cfg.num_groups = 4
        # A large capacity factor to prevent dropping tokens.
        cfg.gating.train_capacity_factor = 100.0
        cfg.gating.eval_capacity_factor = 100.0
        layer: TransformerFeedForwardMoE = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        cfg_dense = TransformerFeedForwardLayer.default_config().set(name="test")
        cfg_dense.input_dim = input_dim
        cfg_dense.hidden_dim = hidden_dim
        cfg_dense.activation = activation
        layer_dense: TransformerFeedForwardLayer = cfg_dense.instantiate(parent=None)
        state_dense = layer_dense.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )

        if copy_weights:
            # Copy the dense layer weights to all experts in the sparse model.
            if isinstance(activation, tuple):
                for i in range(len(activation)):
                    state[f"wi_{i}_weight"] = jnp.tile(
                        state_dense[f"linear1_{i}"]["weight"], [num_experts, 1, 1]
                    )
            else:
                state["wi_weight"] = jnp.tile(state_dense["linear1"]["weight"], [num_experts, 1, 1])
            state["wo_weight"] = jnp.tile(state_dense["linear2"]["weight"], [num_experts, 1, 1])

        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        outputs, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        outputs_dense, _ = F(
            layer_dense,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state_dense,
            inputs=dict(inputs=inputs),
        )
        if copy_weights:
            # All experts in MoE have the same weights as the dense layers.
            assert_allclose(outputs, outputs_dense)
        else:
            np.testing.assert_raises(AssertionError, assert_allclose, outputs, outputs_dense)

    @parameterized.product(
        expert_capacity=(0, 200),
        num_experts=(1, 8),
        outer_batch=(1, 2),
    )
    def test_top2_gating(
        self,
        expert_capacity,
        num_experts,
        outer_batch,
    ):
        batch_size = 2
        seq_len = 12

        cfg = Top2Gating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.expert_capacity = expert_capacity
        cfg.eval_capacity_factor = (
            100.0  # set to a larger number to prevent token dropping for test.
        )

        layer: Top2Gating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        shape = (outer_batch, batch_size, seq_len, num_experts)

        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)
        gating, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )

        # The number of selected experts should be 2 for all tokens.
        num_experts_per_token = jnp.sum(gating.dispatch_tensor, axis=(-2, -1))
        num_choosen_experts = 1 if num_experts == 1 else 2
        expected = jnp.ones(shape=shape[:-1]) * num_choosen_experts
        assert_allclose(num_experts_per_token, expected)

        # The total probabilities over all experts should be 1 for all tokens.
        prob_per_token = jnp.sum(gating.combine_tensor, axis=(-2, -1))
        expected = jnp.ones(shape=shape[:-1])
        assert_allclose(prob_per_token, expected)

    @parameterized.product(
        is_training=(True, False),
    )
    def test_top2_gating_capacity_factor(self, is_training):
        num_experts = 4
        group_size = 16

        cfg = Top2Gating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.train_capacity_factor = 1.0
        cfg.eval_capacity_factor = 2.0

        layer: Top2Gating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=(2, 4, group_size, num_experts))
        gating, _ = F(
            layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )
        # expert_capacity = group_size * capacity_factor // num_experts.
        if is_training:
            self.assertEqual(gating.dispatch_tensor.shape[-1], 4)
        else:
            self.assertEqual(gating.dispatch_tensor.shape[-1], 8)

    @parameterized.parameters(
        "prenorm",
        "postnorm",
        "hybridnorm",
        "nonorm",
    )
    def test_layer_structure(self, structure):
        batch, seq_len, dim = 2, 3, 4
        cfg = TransformerFeedForwardMoE.default_config().set(
            name="moe",
            input_dim=dim,
            hidden_dim=dim * 4,
            activation="nn.relu",
            structure=structure,
            num_experts=4,
            num_groups=2,
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=[batch, seq_len, dim])
        ref_x, _ = F(
            layer,
            inputs=dict(inputs=inputs),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )

        dispatch_and_combine_fn = partial(
            F,
            module=layer,
            method="_dispatch_and_combine",
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        if structure == "prenorm":
            x, _ = F(
                layer.norm,
                inputs=dict(x=inputs),
                state=layer_params["norm"],
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            x, _ = dispatch_and_combine_fn(inputs=dict(x=x))
            x += inputs
        elif structure == "postnorm":
            x, _ = dispatch_and_combine_fn(inputs=dict(x=inputs))
            x, _ = F(
                layer.norm,
                inputs=dict(x=(x + inputs)),  # pylint: disable=superfluous-parens
                state=layer_params["norm"],
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
        elif structure == "hybridnorm":
            x, _ = F(
                layer.prenorm,
                inputs=dict(x=inputs),
                state=layer_params["prenorm"],
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            x, _ = dispatch_and_combine_fn(inputs=dict(x=x))
            x, _ = F(
                layer.postnorm,
                inputs=dict(x=x),
                state=layer_params["postnorm"],
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            x += inputs
        elif structure == "nonorm":
            x, _ = dispatch_and_combine_fn(inputs=dict(x=inputs))
        else:
            raise ValueError(f"Structure {structure} not supported.")
        assert_allclose(ref_x, x)

    def test_moe_dense_ffn_fan_axes_parity(self):
        cfg = TransformerFeedForwardMoE.default_config().set(name="test")
        cfg.input_dim = 8
        cfg.hidden_dim = 32
        cfg.num_experts = 8
        cfg.num_groups = 4
        cfg.outer_batch = 1
        layer: TransformerFeedForwardMoE = cfg.instantiate(parent=None)
        fans_i = layer._create_layer_parameter_specs()["wi_weight"].fans()
        fans_o = layer._create_layer_parameter_specs()["wo_weight"].fans()

        cfg2 = TransformerFeedForwardLayer.default_config().set(name="test")
        cfg2.input_dim = 8
        cfg2.hidden_dim = 32
        layer2: TransformerFeedForwardLayer = cfg2.instantiate(parent=None)
        linear1_w = layer2._children["linear1"]._create_layer_parameter_specs()["weight"]
        linear1_w.fan_axes = layer2._compute_fan_axes("weight", linear1_w)
        fans2_i = linear1_w.fans()
        linear2_w = layer2._children["linear2"]._create_layer_parameter_specs()["weight"]
        linear2_w.fan_axes = layer2._compute_fan_axes("weight", linear2_w)
        fans2_o = linear2_w.fans()

        for k, v in fans_i.items():
            assert k in fans2_i
            self.assertEqual(v, fans2_i[k])

        for k, v in fans_o.items():
            assert k in fans2_o
            self.assertEqual(v, fans2_o[k])


class ParamConversionTest(parameterized.TestCase):
    @parameterized.product(
        activation=("relu", ("linear", "nn.silu")),
        num_experts=(1, 2),
        bias=(False, True),
    )
    def test_feed_forward_to_moe_parameters(self, *, activation, num_experts: int, bias: bool):
        input_dim, hidden_dim = 4, 16
        cfg_dense = TransformerFeedForwardLayer.default_config().set(name="test")
        cfg_dense.input_dim = input_dim
        cfg_dense.hidden_dim = hidden_dim
        cfg_dense.activation = activation
        cfg_dense.linear1.bias = cfg_dense.linear2.bias = bias
        layer_dense: TransformerFeedForwardLayer = cfg_dense.instantiate(parent=None)
        state_dense = layer_dense.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )

        cfg_moe = TransformerFeedForwardMoE.default_config().set(name="test")
        cfg_moe.input_dim = input_dim
        cfg_moe.hidden_dim = hidden_dim
        cfg_moe.num_experts = num_experts
        cfg_moe.activation = activation
        cfg_moe.num_groups = 1
        # A large capacity factor to prevent dropping tokens.
        cfg_moe.gating.train_capacity_factor = 100.0
        cfg_moe.gating.eval_capacity_factor = 100.0
        layer_moe: TransformerFeedForwardMoE = cfg_moe.instantiate(parent=None)
        state_moe = layer_moe.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        param_specs_moe = layer_moe.create_parameter_specs_recursively()

        if bias:
            with self.assertRaisesRegex(NotImplementedError, "bias"):
                _convert_feedforward_to_moe_parameters(
                    state_dense, num_experts=num_experts, moe_parameter_specs=param_specs_moe
                )
            return
        state_moe_converted = _convert_feedforward_to_moe_parameters(
            state_dense, num_experts=num_experts, moe_parameter_specs=param_specs_moe
        )
        state_moe_converted = jax.tree_util.tree_map(
            lambda spec, param: spec if param is None else param,
            param_specs_moe,
            state_moe_converted,
        )
        self.assertEqual(shapes(state_moe), shapes(state_moe_converted))
        for expert_i in range(num_experts):
            # The dense weights are replicated to each expert.
            if "linear1" in state_dense:
                np.testing.assert_array_equal(
                    state_dense["linear1"]["weight"], state_moe_converted["wi_weight"][expert_i]
                )
            else:
                np.testing.assert_array_equal(
                    state_dense["linear1_0"]["weight"], state_moe_converted["wi_0_weight"][expert_i]
                )
                np.testing.assert_array_equal(
                    state_dense["linear1_1"]["weight"], state_moe_converted["wi_1_weight"][expert_i]
                )
            np.testing.assert_array_equal(
                state_dense["linear2"]["weight"], state_moe_converted["wo_weight"][expert_i]
            )

    def test_dense_to_moe_parameters(self):
        """Tests _convert_feedforward_to_moe_parameters."""
        mesh_shape = (1, 1, 1)
        devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh = jax.sharding.Mesh(devices, ("expert", "data", "model"))
        with mesh:
            num_layers, input_dim, hidden_dim, num_heads = 6, 4, 16, 2
            cfg_dense = RepeatedTransformerLayer.default_config().set(
                name="test",
                input_dim=input_dim,
                num_layers=num_layers,
                layer=TransformerLayer.default_config(),
            )
            cfg_dense.layer.self_attention.attention.set(num_heads=num_heads)
            cfg_ff_dense = cfg_dense.layer.feed_forward.set(
                hidden_dim=hidden_dim, activation=("linear", "nn.silu")
            )
            set_bias_recursively(cfg_dense, bias=False)
            layer_dense = cfg_dense.instantiate(parent=None)
            state_dense = layer_dense.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(123)
            )

            cfg_ff_moe = TransformerFeedForwardMoE.default_config().set(
                hidden_dim=hidden_dim,
                num_experts=3,
                num_groups=1,
                activation=cfg_ff_dense.activation,
            )
            # A large capacity factor to prevent dropping tokens.
            cfg_ff_moe.gating.train_capacity_factor = 100.0
            cfg_ff_moe.gating.eval_capacity_factor = 100.0
            cfg_moe = RepeatedTransformerLayer.default_config().set(
                name="test",
                input_dim=input_dim,
                num_layers=num_layers // 2,
                layer=StackedTransformerLayer.default_config().set(
                    num_layers=2,
                    layer=[cfg_dense.layer.clone(), cfg_dense.layer.clone(feed_forward=cfg_ff_moe)],
                ),
            )
            set_bias_recursively(cfg_moe, bias=False)

            layer_moe = cfg_moe.instantiate(parent=None)
            state_moe = layer_moe.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(123)
            )
            param_specs_moe = layer_moe.create_parameter_specs_recursively()

            state_moe_converted = convert_dense_to_moe_parameters(
                state_dense, target_parameter_specs=param_specs_moe
            )
            print(shapes(state_moe))
            print(shapes(state_moe_converted))
            self.assertEqual(shapes(state_moe), shapes(state_moe_converted))

            # Initialize `gate_weight` randomly.
            gate_weight_path = ("repeat", "layer", "layer1", "feed_forward", "gate_weight")
            gate_weight_spec = get_recursively(state_moe_converted, path=gate_weight_path)
            gate_weight = jax.random.normal(
                jax.random.PRNGKey(1), shape=gate_weight_spec.shape, dtype=gate_weight_spec.dtype
            )
            set_recursively(state_moe_converted, path=gate_weight_path, value=gate_weight)

            # Feed the same inputs to `layer_dense` and `layer_moe`.
            inputs = jax.random.normal(jax.random.PRNGKey(123), (2, 7, input_dim))
            outputs_dense, _ = F(
                module=layer_dense,
                state=state_dense,
                inputs=(inputs,),
                prng_key=jax.random.PRNGKey(456),
                is_training=True,
            )
            outputs_moe, _ = F(
                module=layer_moe,
                state=state_moe_converted,
                inputs=(inputs,),
                prng_key=jax.random.PRNGKey(456),
                is_training=True,
            )
            # Expect the same outputs since all experts are identical to the dense one.
            assert_allclose(outputs_dense.data, outputs_moe.data)


if __name__ == "__main__":
    absltest.main()

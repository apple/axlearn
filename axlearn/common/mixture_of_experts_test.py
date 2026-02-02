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

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from axlearn.common.attention import (
    NormPosition,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerFeedForwardLayer,
    TransformerLayer,
)
from axlearn.common.config import config_for_function
from axlearn.common.layers import RMSNorm, set_bias_recursively
from axlearn.common.mixture_of_experts import (
    AdaptiveLoadBalanceLoss,
    ApproximateTokenDropFreeMoE,
    GateNoise,
    Top2Gating,
    TopKBiasGating,
    TopKDropFreeGating,
    TopKGating,
    TransformerFeedForwardDropFreeMoE,
    TransformerFeedForwardMoE,
    _all_to_all_combine,
    _all_to_all_dispatch,
    _compute_expert_capacity,
    _convert_feedforward_to_moe_parameters,
    _custom_gather,
    approx_max_k,
    convert_dense_to_moe_parameters,
    get_outer_batch_from_mesh,
    sigmoid,
)
from axlearn.common.module import functional as F
from axlearn.common.quantized_dot_general.activation_clipping import TanhActivationClippingLayer
from axlearn.common.quantized_dot_general.layers import (
    DotGeneralQuantizationType,
    QuantizedDotGeneral,
)
from axlearn.common.test_utils import TestCase, assert_allclose, is_supported_mesh_shape
from axlearn.common.utils import (
    HybridMeshShape,
    MeshShape,
    PartitionSpec,
    get_recursively,
    infer_mesh_shape,
    set_recursively,
    shapes,
    with_sharding_constraint,
)


# pylint: disable=no-self-use,protected-access
class TransformerFeedForwardMoETest(TestCase):

    @parameterized.parameters(
        {"group_size": 50, "capacity_factor": 40, "num_experts": 40, "expected_capacity": 50},
        {"group_size": 50, "capacity_factor": 2.5, "num_experts": 40, "expected_capacity": 3},
        {"group_size": 8192, "capacity_factor": 2.5, "num_experts": 40, "expected_capacity": 512},
        {"group_size": 8192, "capacity_factor": 40, "num_experts": 40, "expected_capacity": 8192},
    )
    def test_expert_capacity(self, group_size, capacity_factor, num_experts, expected_capacity):
        capacity = _compute_expert_capacity(
            group_size=group_size,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            expert_capacity=None,
        )
        self.assertEqual(capacity, expected_capacity)

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

    @parameterized.product(
        expert_capacity=(0, 200),
        outer_batch=(1, 2),
        top_k=(1, 2, 8),
    )
    def test_topk_gating(
        self,
        expert_capacity,
        outer_batch,
        top_k,
    ):
        batch_size = 2
        seq_len = 12
        num_experts = 8

        cfg = TopKGating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.expert_capacity = expert_capacity
        cfg.eval_capacity_factor = (
            100.0  # set to a larger number to prevent token dropping for test.
        )
        cfg.num_experts_per_token = top_k

        layer: TopKGating = cfg.instantiate(parent=None)
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

        # Number of selected experts should be exactly `top_k`.
        num_experts_per_token = jnp.sum(gating.dispatch_tensor, axis=(-2, -1))
        expected = jnp.ones(shape=shape[:-1]) * top_k
        assert_allclose(num_experts_per_token, expected)

        # The total probabilities over all experts should be 1 for all tokens.
        prob_per_token = jnp.sum(gating.combine_tensor, axis=(-2, -1))
        expected = jnp.ones(shape=shape[:-1])
        assert_allclose(prob_per_token, expected)

    @parameterized.product(
        expert_capacity=(0, 200),
        outer_batch=(1, 2),
    )
    def test_top2_and_topk_gating_parity(
        self,
        expert_capacity,
        outer_batch,
    ):
        batch_size = 2
        seq_len = 12
        num_experts = 8

        # Inputs
        shape = (outer_batch, batch_size, seq_len, num_experts)
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)

        # Top-2 config.
        cfg = Top2Gating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.expert_capacity = expert_capacity
        cfg.eval_capacity_factor = (
            100.0  # set to a larger number to prevent token dropping for test.
        )
        layer: Top2Gating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        gating, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )

        # Top-K config with K=2.
        cfg1 = TopKGating.default_config().set(name="test")
        cfg1.num_experts = num_experts
        cfg1.expert_capacity = expert_capacity
        cfg1.eval_capacity_factor = (
            100.0  # set to a larger number to prevent token dropping for test.
        )
        cfg1.num_experts_per_token = 2
        layer1: TopKGating = cfg.instantiate(parent=None)
        state1 = layer1.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        gating1, _ = F(
            layer1,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state1,
            inputs=dict(logits=logits),
        )

        # Everything in top-2 outputs and top-k outputs should be the same.
        assert_allclose(gating.dispatch_tensor, gating1.dispatch_tensor)
        assert_allclose(gating.combine_tensor, gating1.combine_tensor)
        assert_allclose(gating.load_balance_loss, gating1.load_balance_loss)
        assert_allclose(gating.router_z_loss, gating1.router_z_loss)

    def test_adaptive_load_balance_loss(self):
        cfg = AdaptiveLoadBalanceLoss.default_config().set(
            name="test", min_value=0.4, max_value=0.6
        )
        cfg.moving_average.min_weight = 0.2
        layer: AdaptiveLoadBalanceLoss = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        value_under_range = cfg.min_value / 2
        value_in_range = (cfg.min_value + cfg.max_value) / 2
        value_over_range = cfg.max_value * 2

        scale, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(value=value_in_range),
        )
        self.assertAlmostEqual(scale, 1)
        scale, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(value=value_under_range),
        )
        self.assertLess(scale, 1)
        scale, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(value=value_over_range),
        )
        self.assertGreater(scale, 1)

        num_values = 100
        values = jax.random.uniform(jax.random.PRNGKey(1), [num_values], minval=0, maxval=1)
        num_dec = num_inc = 0
        prev_scale = 1.0
        for i in range(num_values):
            scale, output_collection = F(
                layer,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state=state,
                inputs=dict(value=values[i]),
            )
            moving_avg = output_collection.summaries["value_average"]
            if moving_avg < cfg.min_value:
                num_dec += 1
                self.assertLess(scale, prev_scale)
            elif moving_avg > cfg.max_value:
                num_inc += 1
                self.assertGreater(scale, prev_scale)
            else:
                self.assertEqual(scale, prev_scale)
            prev_scale = scale
            state = copy.deepcopy(output_collection.state_updates)
        self.assertGreater(num_dec, 0)
        self.assertGreater(num_inc, 0)

    def test_top2_gating_with_adaptive_load_balance_loss(self):
        num_experts = 4
        group_size = 16

        cfg = Top2Gating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.train_capacity_factor = 1.0
        cfg.eval_capacity_factor = 2.0
        cfg.adaptive_load_balance_loss = AdaptiveLoadBalanceLoss.default_config().set(
            # Set a very low range for the target over-capacity ratio.
            min_value=1e-7,
            max_value=1e-6,
        )
        cfg.adaptive_load_balance_loss.moving_average.min_weight = 0.1

        layer: Top2Gating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=(2, 4, group_size, num_experts))
        gating, output_collection = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )
        self.assertIn("adaptive_load_balance_loss", output_collection.summaries)
        self.assertContainsSubset(
            {"scale", "value_average"}, output_collection.summaries["adaptive_load_balance_loss"]
        )
        # We return the adjusted load balance loss.
        self.assertEqual(
            output_collection.summaries["load_balance_loss"],
            gating.load_balance_loss,
        )
        # The adjusted load balance loss is greater than the original load balance loss.
        self.assertGreater(
            output_collection.summaries["load_balance_loss"],
            output_collection.summaries["load_balance_loss_original"],
        )

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

    @parameterized.product(is_training=(True, False), outer_batch=(1, 2))
    def test_moe_quantized_einsum(self, is_training, outer_batch):
        batch_size = 4
        seq_len = 128
        input_dim = 8

        ref_cfg = TransformerFeedForwardMoE.default_config().set(name="test")
        ref_cfg.input_dim = input_dim
        ref_cfg.hidden_dim = 32
        ref_cfg.num_experts = 8
        ref_cfg.num_groups = 4
        ref_cfg.outer_batch = outer_batch
        ref_layer = ref_cfg.instantiate(parent=None)

        state = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        test_cfg = ref_cfg.clone()
        test_cfg.quantized_dot_general = QuantizedDotGeneral.default_config().set(
            quantization_type=DotGeneralQuantizationType.INT_8,
            activation_clipping=TanhActivationClippingLayer.default_config().set(
                clipping_max_abs=1e6
            ),
        )
        test_layer = test_cfg.instantiate(parent=None)

        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        ref_outputs, _ = F(
            ref_layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        test_outputs, _ = F(
            test_layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        err = np.abs(ref_outputs - test_outputs).sum()
        self.assertGreater(err, 0)
        self.assertNestedAllClose(ref_outputs, test_outputs, atol=5e-2)

    @parameterized.product(
        structure=("prenorm", "postnorm", "hybridnorm"),
        is_training=(True, False),
        outer_batch=(1, 2),
    )
    def test_v2_structure(self, structure, is_training, outer_batch):
        # Test equivalence bewteen (prenorm, postnorm, hybridnorm) and v2 structure.
        # prenorm == in_norm (v2)
        # postnorm == out_norm (v2)
        # hybridnorm == (in_norm, res_norm) (v2)
        batch_size = 4
        seq_len = 128
        input_dim = 8
        hidden_dim = 32
        num_experts = 8
        num_groups = 4

        ref_cfg = TransformerFeedForwardMoE.default_config().set(
            name="ref",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_groups=num_groups,
            outer_batch=outer_batch,
            structure=structure,
            norm=RMSNorm.default_config(),
        )
        ref_layer: TransformerFeedForwardMoE = ref_cfg.instantiate(parent=None)
        ref_state = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        ref_outputs, _ = F(
            ref_layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=ref_state,
            inputs=dict(inputs=inputs),
        )

        norm = {}
        if structure == "prenorm":
            norm = {NormPosition.IN_NORM: RMSNorm.default_config()}
        elif structure == "postnorm":
            norm = {NormPosition.OUT_NORM: RMSNorm.default_config()}
        elif structure == "hybridnorm":
            norm = {
                NormPosition.IN_NORM: RMSNorm.default_config(),
                NormPosition.RES_NORM: RMSNorm.default_config(),
            }
        else:
            raise ValueError(f"No equivalent structure is available in v2 to {structure}.")

        cfg = TransformerFeedForwardMoE.default_config().set(
            name="test",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_groups=num_groups,
            outer_batch=outer_batch,
            structure="v2",
            norm=norm,
        )
        layer: TransformerFeedForwardMoE = cfg.instantiate(parent=None)
        state = ref_state
        if structure == "prenorm":
            state["in_norm"] = state["norm"]
            del state["norm"]
        elif structure == "postnorm":
            state["out_norm"] = state["norm"]
            del state["norm"]
        elif structure == "hybridnorm":
            state["in_norm"] = state["prenorm"]
            state["res_norm"] = state["postnorm"]
            del state["prenorm"]
            del state["postnorm"]
        else:
            raise ValueError(f"No equivalent structure is available in v2 to {structure}.")
        outputs, _ = F(
            layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        self.assertNestedAllClose(outputs, ref_outputs)

    @parameterized.parameters(None, 0.0, -1.0)
    def test_residual_gate(self, residual_gate_init):
        """Test residual gating mechanism in TransformerFeedForwardMoE.

        Tests that:
        1. When residual_gate_init is None, no residual_gate_theta parameter is created.
        2. When residual_gate_init is not None, residual_gate_theta parameter is created
           with the correct shape and initial value.
        3. Forward pass runs successfully in both cases.
        """
        batch_size = 2
        seq_len = 8
        input_dim = 16
        hidden_dim = 32
        num_experts = 4
        num_groups = 2

        # Use v2 structure with in_norm for residual gating to be applied
        cfg = TransformerFeedForwardMoE.default_config().set(
            name="test",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_groups=num_groups,
            structure="v2",
            norm={NormPosition.IN_NORM: RMSNorm.default_config()},
            residual_gate_init=residual_gate_init,
        )
        layer: TransformerFeedForwardMoE = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Check parameter existence
        if residual_gate_init is None:
            self.assertNotIn("residual_gate_theta", state)
        else:
            self.assertIn("residual_gate_theta", state)
            # Check shape is scalar
            self.assertEqual(state["residual_gate_theta"].shape, ())
            # Check initial value
            assert_allclose(state["residual_gate_theta"], residual_gate_init)

        # Test forward pass runs successfully
        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        outputs, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            state=state,
            inputs=dict(inputs=inputs),
        )
        # Check output shape matches input shape
        self.assertEqual(outputs.shape, inputs.shape)

    def test_expert_correlation_loss(self):
        """Test expert correlation loss is computed and added to aux_loss when enabled."""
        batch_size, seq_len, dim = 2, 4, 8
        cfg = TransformerFeedForwardMoE.default_config().set(
            name="moe",
            input_dim=dim,
            hidden_dim=dim * 4,
            activation=("nn.silu", "linear"),
            num_experts=4,
            num_groups=1,
            corr_loss_weight=0.1,  # Enable correlation loss
        )
        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=[batch_size, seq_len, dim])

        # Run forward pass and collect summaries
        _, output_collection = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
            drop_output_collections=[],
        )

        # Verify expert_cov_max summary is present
        self.assertIn("expert_cov_max", output_collection.summaries)
        # Verify expert_cov_loss summary is present
        self.assertIn("expert_cov_loss", output_collection.summaries)

        # Verify aux_loss includes correlation loss
        aux_loss = output_collection.module_outputs["aux_loss"]
        self.assertIsNotNone(aux_loss)

        # Test with corr_loss_weight=0 (disabled)
        cfg_no_corr = cfg.clone(corr_loss_weight=0)
        layer_no_corr = cfg_no_corr.instantiate(parent=None)
        _, output_collection_no_corr = F(
            layer_no_corr,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )

        # Verify expert_cov_loss summary is NOT present when disabled
        self.assertNotIn("expert_cov_loss", output_collection_no_corr.summaries)

    def test_dead_neuron_detection(self):
        """Test dead neuron detection produces summaries for TopKGating."""
        batch_size, seq_len, dim = 2, 8, 16
        cfg = TransformerFeedForwardMoE.default_config().set(
            name="moe",
            input_dim=dim,
            hidden_dim=dim * 4,
            activation=("nn.silu", "linear"),
            num_experts=4,
            num_groups=1,
            gating=TopKGating.default_config().set(num_experts_per_token=2),
        )
        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=[batch_size, seq_len, dim])

        # Run forward pass
        _, output_collection = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            state=state,
            inputs=dict(inputs=inputs),
        )

        # Verify expert_dead_neurons summary is present for TopKGating
        self.assertIn("expert_dead_neurons", output_collection.summaries)
        dead_neurons = output_collection.summaries["expert_dead_neurons"]
        # Should be a scalar count
        self.assertEqual(dead_neurons.shape, ())
        # Should be non-negative integer
        self.assertGreaterEqual(dead_neurons, 0)

    def test_rms_norm_summary(self):
        """Test RMS norm summary for linear2 outputs is produced."""
        batch_size, seq_len, dim = 2, 4, 8
        cfg = TransformerFeedForwardMoE.default_config().set(
            name="moe",
            input_dim=dim,
            hidden_dim=dim * 4,
            activation="nn.relu",
            num_experts=4,
            num_groups=1,
        )
        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=[batch_size, seq_len, dim])

        # Run forward pass
        _, output_collection = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(789),
            state=state,
            inputs=dict(inputs=inputs),
        )

        # Verify rms_norm/linear2_outputs summary is present
        self.assertIn("rms_norm/linear2_outputs", output_collection.summaries)
        rms_norm = output_collection.summaries["rms_norm/linear2_outputs"]
        # Should be a scalar
        self.assertEqual(rms_norm.shape, ())
        # Should be positive
        self.assertGreater(rms_norm, 0)

    def test_dynamic_dispatch_combine_topk_gating(self):
        """Test that dynamic dispatch/combine works correctly with TopKGating."""
        batch_size, seq_len, dim = 2, 4, 8

        # Create MoE layer with TopKGating
        cfg = TransformerFeedForwardMoE.default_config().set(
            name="moe",
            input_dim=dim,
            hidden_dim=dim * 4,
            activation="nn.relu",
            num_experts=4,
            num_groups=1,
            gating=TopKGating.default_config().set(num_experts_per_token=2),
        )
        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=[batch_size, seq_len, dim])

        # Run forward pass - should work without errors
        outputs, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            state=state,
            inputs=dict(inputs=inputs),
        )

        # Verify output shape
        self.assertEqual(outputs.shape, (batch_size, seq_len, dim))

    def test_compatibility_with_top2_gating(self):
        """Test backward compatibility - MoE with Top2Gating should still work."""
        batch_size, seq_len, dim = 2, 4, 8

        # Create MoE layer with default Top2Gating
        cfg = TransformerFeedForwardMoE.default_config().set(
            name="moe",
            input_dim=dim,
            hidden_dim=dim * 4,
            activation="nn.relu",
            num_experts=4,
            num_groups=1,
            # Default gating is Top2Gating
        )
        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = jax.random.normal(jax.random.PRNGKey(1), shape=[batch_size, seq_len, dim])

        # Run forward pass - should work without errors
        outputs, output_collection = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            state=state,
            inputs=dict(inputs=inputs),
            drop_output_collections=[],
        )

        # Verify output shape
        self.assertEqual(outputs.shape, (batch_size, seq_len, dim))
        # Verify aux_loss is present
        self.assertIn("aux_loss", output_collection.module_outputs)


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
        state_moe_converted = jax.tree.map(
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


class GetOuterBatchFromMeshTest(absltest.TestCase):
    """Tests get_outer_batch_from_mesh."""

    def test_mesh_shape_is_none(self):
        result = get_outer_batch_from_mesh(
            mesh_axis_names=["data", "model"],
            outer_batch_axis_names=["data"],
            mesh_shape=None,
        )
        self.assertIsNone(result)

    def test_regular_mesh_shape(self):
        mesh_shape: MeshShape = (2, 4, 8)
        result = get_outer_batch_from_mesh(
            mesh_axis_names=["data", "fsdp", "model"],
            outer_batch_axis_names=["data", "fsdp"],
            mesh_shape=mesh_shape,
        )
        self.assertEqual(result, 2 * 4)

    def test_outer_batch_with_hybrid_mesh_shape(self):
        hybrid = HybridMeshShape(ici_mesh_shape=(2, 8, 1), dcn_mesh_shape=(4, 1, 1))
        result = get_outer_batch_from_mesh(
            mesh_axis_names=["data", "model"],
            outer_batch_axis_names=["data"],
            mesh_shape=hybrid,
        )
        self.assertEqual(result, 2 * 4)

    def test_outer_batch_with_hybrid_mesh_shape_with_raises(self):
        hybrid = HybridMeshShape(ici_mesh_shape=(2, 4, 1), dcn_mesh_shape=(-1, 2, 1))
        with self.assertRaisesRegex(NotImplementedError, "Unable to infer number of granules"):
            get_outer_batch_from_mesh(
                mesh_axis_names=["data", "fsdp", "model"],
                outer_batch_axis_names=["data", "fsdp"],
                mesh_shape=hybrid,
            )

    def test_outer_batch_with_infer_mesh_shape(self):
        mesh_shape: MeshShape = (-1, 2, 4)
        inferred_shape = infer_mesh_shape(mesh_shape, num_devices=8)
        result = get_outer_batch_from_mesh(
            mesh_axis_names=["data", "fsdp", "model"],
            outer_batch_axis_names=["data", "fsdp"],
            mesh_shape=inferred_shape,
        )
        self.assertEqual(result, 1 * 2)


class TopKGatingTest(TestCase):
    """Tests for TopKGating functionality."""

    def test_sigmoid_score_fn(self):
        """Test that sigmoid scoring function works correctly."""
        cfg = TopKGating.default_config().set(
            name="test",
            score_fn=config_for_function(sigmoid),
            num_experts=4,
            num_experts_per_token=1,
            expert_capacity=4,
        )
        layer: TopKGating = cfg.instantiate(parent=None)
        inputs = np.array([-1, 1, 2, 3], dtype=np.float32)
        value = layer._score(inputs)
        assert_allclose(value, 1.0 / (1.0 + np.exp(-inputs)))

    @parameterized.parameters(
        {"num_experts_per_token": 2, "num_experts": 4, "outer_batch": 2},
    )
    def test_topk_noisy_gating(
        self,
        num_experts_per_token,
        num_experts,
        outer_batch,
    ):
        """Test that noisy gating with Gumbel noise produces different results."""
        batch_size = 2
        seq_len = 8
        cfg = TopKGating.default_config().set(name="test", noisy_gating=GateNoise.GUMBEL)
        cfg.num_experts = num_experts
        cfg.num_experts_per_token = num_experts_per_token
        layer: TopKGating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        shape = (outer_batch, batch_size, seq_len, num_experts)
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)

        noisy_gating, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )

        cfg = TopKGating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.num_experts_per_token = num_experts_per_token
        layer: TopKGating = cfg.instantiate(parent=None)
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

        noisy_gating_combine_tensor = jnp.asarray(noisy_gating.combine_tensor).astype(jnp.float32)
        gating_combine_tensor = jnp.asarray(gating.combine_tensor).astype(jnp.float32)
        assert not jnp.array_equal(noisy_gating_combine_tensor, gating_combine_tensor)

    @parameterized.parameters(
        {"num_experts_per_token": 2, "num_experts": 4, "outer_batch": 2},
    )
    def test_approx_topk_noisy_gating(
        self,
        num_experts_per_token,
        num_experts,
        outer_batch,
    ):
        """Test that approximate top-k produces same results as exact top-k for small
        num_experts.
        """
        batch_size = 2
        seq_len = 8
        cfg = TopKGating.default_config().set(
            name="test", topk_fn=config_for_function(approx_max_k).set(recall_target=0.95)
        )
        cfg.num_experts = num_experts
        cfg.num_experts_per_token = num_experts_per_token
        layer: TopKGating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        shape = (outer_batch, batch_size, seq_len, num_experts)
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)

        gate_0, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )

        cfg = TopKGating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.num_experts_per_token = num_experts_per_token
        layer: TopKGating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        shape = (outer_batch, batch_size, seq_len, num_experts)
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)

        gate_1, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )

        gate_0_combine_tensor = jnp.asarray(gate_0.combine_tensor).astype(jnp.float32)
        gate_1_combine_tensor = jnp.asarray(gate_1.combine_tensor).astype(jnp.float32)
        # Given there are 4 experts in total, we expect that the approximate topk with recall = 0.95
        # should return the same results as topk.
        assert jnp.array_equal(gate_0_combine_tensor, gate_1_combine_tensor)

    @parameterized.parameters(
        {"num_experts_per_token": 2, "num_experts": 4, "outer_batch": 2},
    )
    def test_capacity_factor_gating(
        self,
        num_experts_per_token,
        num_experts,
        outer_batch,
    ):
        """Test that default capacity factor equals num_experts_per_token."""
        batch_size = 2
        seq_len = 8
        cfg = TopKGating.default_config().set(name="test")
        cfg.num_experts = num_experts
        cfg.num_experts_per_token = num_experts_per_token
        layer: TopKGating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        shape = (outer_batch, batch_size, seq_len, num_experts)
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)

        gate_0, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )

        cfg = TopKGating.default_config().set(name="test", train_capacity_factor=2.0)
        cfg.num_experts = num_experts
        cfg.num_experts_per_token = num_experts_per_token
        layer: TopKGating = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        shape = (outer_batch, batch_size, seq_len, num_experts)
        logits = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)

        gate_1, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(logits=logits),
        )

        gate_0_combine_tensor = jnp.asarray(gate_0.combine_tensor).astype(jnp.float32)
        gate_1_combine_tensor = jnp.asarray(gate_1.combine_tensor).astype(jnp.float32)
        assert jnp.array_equal(gate_0_combine_tensor, gate_1_combine_tensor)


class TransformerFeedForwardDropFreeMoETest(TestCase):
    """Tests for TransformerFeedForwardDropFreeMoE layer."""

    def _random_dense(self, shape, key, dtype, limit) -> jnp.ndarray:
        if limit is None:
            limit = 1 / np.prod(shape)
        x = jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
        return x.astype(dtype)

    def _loss(self, params, inputs, layer, prng_key):
        x, _ = F(
            layer,
            inputs=dict(inputs=inputs),
            state=params,
            is_training=True,
            prng_key=prng_key,
        )
        return jnp.mean(x)

    @parameterized.product(
        structure=["prenorm", "postnorm", "hybridnorm", "nonorm"],
    )
    def test_layer_structure(self, structure):
        mesh = [1, 1, 1, 1, 1, 1]
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")

        batch_size = 4
        seq_len = 128
        model_dim = 512
        num_experts = 8

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(123456), 3)

        input_batch = self._random_dense((batch_size, seq_len, model_dim), k1, jnp.float32, limit=1)

        mesh_axis_names = ["pipeline", "data", "expert", "fsdp", "seq", "model"]

        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            input_batch = with_sharding_constraint(input_batch, PartitionSpec("data", "fsdp"))

            cfg_ref = TransformerFeedForwardMoE.default_config().set(
                name="moe",
                input_dim=model_dim,
                hidden_dim=model_dim * 4,
                activation=("nn.silu", "linear"),
                structure=structure,
                num_experts=num_experts,
                num_groups=1,
                outer_batch=4,
                gating=TopKGating.default_config().set(
                    train_capacity_factor=num_experts, gating_logit_cap=0
                ),
            )
            ref_layer = cfg_ref.instantiate(parent=None)
            params = ref_layer.initialize_parameters_recursively(prng_key=k2)
            ref_value, ref_grads = jax.value_and_grad(self._loss)(
                params, input_batch, ref_layer, k3
            )

            test_cfg = TransformerFeedForwardDropFreeMoE.default_config().set(
                name="moe",
                input_dim=model_dim,
                hidden_dim=model_dim * 4,
                activation=("nn.silu", "linear"),
                structure=structure,
                num_experts=num_experts,
                num_groups=1,
                gating=TopKDropFreeGating.default_config().set(gating_logit_cap=0),
                tiling=(128, 128, 128),
                interpret=True,  # for cpu testing.
                preferred_element_type=jnp.float32,
            )
            test_layer = test_cfg.instantiate(parent=None)
            test_value, test_grads = jax.value_and_grad(self._loss)(
                params, input_batch, test_layer, k3
            )

            assert_allclose(test_value, ref_value)
            self.assertNestedAllClose(test_grads, ref_grads)

    @parameterized.product(seq_load_balance_loss_weight=[None, 0, 1e-4])
    def test_seq_load_balance_loss(self, seq_load_balance_loss_weight):
        batch_size = 4
        seq_len = 128
        model_dim = 512
        num_experts = 8

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(123456), 3)

        input_batch = self._random_dense((batch_size, seq_len, model_dim), k1, jnp.float32, limit=1)

        mesh_axis_names = ["pipeline", "data", "expert", "fsdp", "seq", "model"]

        with Mesh(mesh_utils.create_device_mesh((1, 1, 1, 1, 1, 1)), mesh_axis_names):
            input_batch = with_sharding_constraint(input_batch, PartitionSpec("data", "fsdp"))
            cfg = TransformerFeedForwardDropFreeMoE.default_config().set(
                name="moe",
                input_dim=model_dim,
                hidden_dim=model_dim * 4,
                activation=("nn.silu", "linear"),
                structure="prenorm",
                num_experts=num_experts,
                num_groups=1,
                gating=TopKDropFreeGating.default_config().set(gating_logit_cap=0),
                tiling=(128, 128, 128),
                interpret=True,  # for cpu testing.
                preferred_element_type=jnp.float32,
                seq_load_balance_loss_weight=seq_load_balance_loss_weight,
            )
            layer = cfg.instantiate(parent=None)
            params = layer.initialize_parameters_recursively(prng_key=k2)
            _, collections = F(
                layer,
                inputs=dict(inputs=input_batch),
                state=params,
                is_training=False,
                prng_key=k3,
            )

        if seq_load_balance_loss_weight:
            assert "seq_load_balance_loss" in collections.summaries["gating"]
        else:
            assert "seq_load_balance_loss" not in collections.summaries["gating"]

    def test_bias_based_routing(self):
        batch, seq_len = 4, 128
        num_experts, num_experts_per_token = 4, 2

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(123456), 3)

        logits = self._random_dense((batch, seq_len, num_experts), k1, jnp.float32, limit=1)
        test_cfg = TopKBiasGating.default_config().set(name="test")
        test_cfg.num_experts = num_experts
        test_cfg.num_experts_per_token = num_experts_per_token
        test_layer = test_cfg.instantiate(parent=None)
        test_layer_params = test_layer.initialize_parameters_recursively(prng_key=k2)
        _, test_output_collection = F(
            test_layer,
            inputs=dict(logits=logits, seq_load_balance_loss_weight=None),
            state=test_layer_params,
            is_training=True,
            prng_key=k3,
        )

        assert jnp.array_equal(
            test_output_collection.state_updates["gate_bias"],
            jnp.array([-0.001, 0.001, -0.001, 0.001]),
        )

    def test_expert_parallel(self):
        mesh = [1, 4, 1, 1, 1, 1]
        if not is_supported_mesh_shape(mesh):
            self.skipTest(f"Unsupported mesh {mesh}.")
        if jax.default_backend() == "cpu":
            # Jax CPU doesn't yet support ragged all-to-all.
            # UNIMPLEMENTED: HLO opcode `ragged-all-to-all` is not supported by XLA:CPU
            # ThunkEmitter.
            self.skipTest("Not supported on CPU.")

        batch_size = 4
        seq_len = 128
        model_dim = 512
        num_experts = 8

        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(123456), 3)

        input_batch = self._random_dense((batch_size, seq_len, model_dim), k1, jnp.float32, limit=1)

        @partial(jax.jit, static_argnames=("layer",))
        def value_and_grad(params, input_batch, layer, k3):
            return jax.value_and_grad(self._loss)(params, input_batch, layer, k3)

        mesh_axis_names = ["pipeline", "data", "expert", "fsdp", "seq", "model"]
        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            input_batch = with_sharding_constraint(
                input_batch, PartitionSpec(("data", "fsdp", "expert"))
            )

            ref_cfg = TransformerFeedForwardDropFreeMoE.default_config().set(
                name="moe",
                input_dim=model_dim,
                hidden_dim=model_dim * 4,
                activation=("nn.silu", "linear"),
                structure="prenorm",
                num_experts=num_experts,
                num_groups=1,
                gating=TopKDropFreeGating.default_config().set(gating_logit_cap=0),
                tiling=(128, 128, 128),
            )
            ref_layer = ref_cfg.instantiate(parent=None)
            params = ref_layer.initialize_parameters_recursively(prng_key=k2)
            ref_value, ref_grads = value_and_grad(params, input_batch, ref_layer, k3)

        mesh = [1, 1, 4, 1, 1, 1]
        with Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names):
            input_batch = with_sharding_constraint(
                input_batch, PartitionSpec(("data", "fsdp", "expert"))
            )

            test_cfg = ApproximateTokenDropFreeMoE.default_config().set(
                name="moe",
                input_dim=model_dim,
                hidden_dim=model_dim * 4,
                activation=("nn.silu", "linear"),
                structure="prenorm",
                num_experts=num_experts,
                num_groups=1,
                gating=TopKDropFreeGating.default_config().set(gating_logit_cap=0),
                tiling=(128, 128, 128),
            )
            test_layer = test_cfg.instantiate(parent=None)
            test_value, test_grads = value_and_grad(params, input_batch, test_layer, k3)

        assert_allclose(test_value, ref_value)
        self.assertNestedAllClose(test_grads, ref_grads)

    def test_all2all(self):
        mesh_shape = (4,)
        if not is_supported_mesh_shape(mesh_shape):
            self.skipTest(f"Unsupported mesh {mesh_shape}.")

        all_tokens_per_expert = jnp.array(
            [
                [1, 4, 3, 2, 2, 2, 1, 1],
                [4, 4, 1, 1, 1, 2, 2, 1],
                [4, 3, 2, 1, 1, 2, 2, 1],
                [5, 2, 2, 1, 2, 1, 1, 2],
            ]
        )
        tokens = jnp.repeat(
            jax.lax.broadcasted_iota(jnp.int32, all_tokens_per_expert.shape, 1) + 1,
            repeats=all_tokens_per_expert,
        )
        tokens = tokens.reshape(4, -1)
        mesh = jax.make_mesh(mesh_shape, ("expert",), devices=jax.devices())
        expert_parallel_capacity = 1.25

        @jax.jit
        @partial(
            jax.shard_map,
            in_specs=(P("expert", None), P("expert")),
            out_specs=(P("expert"), P("expert"), P("expert"), P(None)),
            mesh=mesh,
            check_vma=False,
        )
        def shard_map_fn(tokens_per_expert, sorted_inputs):
            sorted_inputs, tokens_per_expert, (dropped_tokens,), residuals = _all_to_all_dispatch(
                sorted_inputs=sorted_inputs.squeeze(),
                tokens_per_expert=tokens_per_expert.squeeze(),
                expert_parallel_capacity=expert_parallel_capacity,
            )

            local_sorted_inputs = sorted_inputs

            sorted_output = _all_to_all_combine(sorted_output=sorted_inputs, residuals=residuals)
            return (
                sorted_output[None],
                local_sorted_inputs[None],
                tokens_per_expert[None],
                dropped_tokens,
            )

        output, local_output, local_tokens_per_expert, dropped_tokens = jax.jit(shard_map_fn)(
            all_tokens_per_expert, tokens
        )

        # Assert original tokens for readability.
        self.assertEqual(
            tokens.tolist(),
            [
                [1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8],
                [1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8],
                [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8],
                [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8],
            ],
        )

        self.assertEqual(
            local_output.tolist(),
            # Here, extra tokens will be dropped.
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0],
                [5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0],
                [7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )

        # Capacity per rank = 16 tokens * 1.25
        self.assertEqual(
            local_tokens_per_expert.tolist(),
            [[9, 11], [8, 5], [6, 7], [6, 5]],
        )
        self.assertEqual(dropped_tokens.tolist(), [7, 0, 0, 0])

        self.assertEqual(
            output.tolist(),
            [
                [1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8],
                [1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8],
                [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8],
                [0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 5, 5, 6, 7, 8, 8],
            ],
        )


class CustomGatherTest(TestCase):
    @parameterized.parameters(1, 8)
    def test_custom_gather(self, repetitions: int):
        k = jax.random.key(42)
        a = jax.random.normal(k, shape=(8192, 2), dtype=jnp.float32)
        idx = jax.random.randint(k, shape=(8192 * repetitions,), minval=0, maxval=8192)
        # Mimic MoE dispatch.
        idx = jnp.argsort(idx) // repetitions

        @partial(jax.jit, static_argnums=(2,))
        def gather_grad(a, idx, use_custom):
            def gather_test(a: jax.Array, idx):
                if use_custom:
                    a = _custom_gather(a, idx, jnp.argsort(idx))
                else:
                    a = a[idx]
                return (a * a).sum()

            return jax.value_and_grad(gather_test)(a, idx)

        ref = gather_grad(a, idx, False)
        test = gather_grad(a, idx, True)
        self.assertNestedAllClose(ref, test)


if __name__ == "__main__":
    absltest.main()

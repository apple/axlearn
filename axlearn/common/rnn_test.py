# Copyright © 2023 Apple Inc.

"""Tests RNN layers."""
# pylint: disable=no-self-use
import jax.random
import pytest
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common.layers import GroupNorm, LayerNorm
from axlearn.common.module import functional as F
from axlearn.common.param_init import GaussianInitializer
from axlearn.common.rnn import LSTMCell, RepeatedRNNLayer, StackedRNNLayer
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import VDict, shapes


class LSTMTest(TestCase):
    """Tests LSTM layers."""

    @parameterized.parameters(
        (None, None, 6, 0.01284),
        (LayerNorm.default_config(), None, 6, 0.02239),
        (None, 0.1, 6, 0.003402),
        (None, None, None, -0.055765),
    )
    def test_lstm_forward(self, norm_cfg, max_cell_value, hidden_dim, expected_output_mean):
        batch_size, seq_len, input_dim = 2, 5, 3

        layer: LSTMCell = (
            LSTMCell.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                norm=norm_cfg,
                max_cell_value=max_cell_value,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
        logging.info("layer params=%s", layer_params)
        if hidden_dim is None:
            self.assertNotIn("output_proj", layer_params)
        else:
            self.assertIn("output_proj", layer_params)

        # Create the initial states.
        init_states, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),  # not used.
            state=layer_params,
            inputs=dict(batch_size=batch_size),
            method="init_states",
        )
        if hidden_dim:
            self.assertEqual(
                shapes(init_states),
                {"c": (batch_size, hidden_dim), "m": (batch_size, layer.output_dim)},
            )
        else:
            self.assertEqual(
                shapes(init_states),
                {"c": (batch_size, layer.output_dim), "m": (batch_size, layer.output_dim)},
            )

        # Step through timesteps of `inputs` and put outputs in `all_step_outputs`.
        inputs = jax.random.normal(jax.random.PRNGKey(123), [seq_len, batch_size, input_dim])
        all_step_outputs = []
        cached_states = init_states
        for timestep in range(seq_len):
            (cached_states, step_outputs), _ = F(
                layer,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),  # not used.
                state=layer_params,
                inputs=dict(cached_states=cached_states, data=inputs[timestep]),
                method="extend_step",
            )
            self.assertSequenceEqual(step_outputs.shape, [batch_size, layer.output_dim])
            self.assertEqual(shapes(cached_states), shapes(init_states))
            if max_cell_value is not None:
                self.assertGreaterEqual(max_cell_value, jnp.abs(cached_states["c"]).max())

            all_step_outputs.append(step_outputs)
        all_step_outputs = jnp.stack(all_step_outputs)
        assert_allclose(all_step_outputs.mean(), expected_output_mean, atol=1e-6, rtol=1e-6)

        forward_outputs, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),  # not used.
            state=layer_params,
            inputs=dict(time_major_inputs=inputs),
        )
        self.assertSequenceEqual(forward_outputs.shape, (seq_len, batch_size, input_dim))
        # forward outputs match step-by-step outputs.
        assert_allclose(forward_outputs, all_step_outputs, atol=1e-6, rtol=1e-6)


class StackedRNNTest(TestCase):
    @parameterized.parameters(
        (None, 2, 5),
        (GroupNorm.default_config().set(num_groups=1, eps=1e-15), 3, 5),
        (None, None, 4),
    )
    @pytest.mark.fp64
    def test_repeat_forward_vs_layerwise(self, norm_cfg, hidden_dim, num_layers):
        batch_size, seq_len, input_dim = 8, 10, 7
        output_dim = input_dim
        layer: RepeatedRNNLayer = (
            RepeatedRNNLayer.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                num_layers=num_layers,
                layer=LSTMCell.default_config().set(
                    hidden_dim=hidden_dim,
                    norm=norm_cfg,
                ),
                param_init=GaussianInitializer.default_config().set(std=10.0),
            )
            .instantiate(parent=None)
        )

        init_key, data_key, prng_key = jax.random.split(jax.random.PRNGKey(567), num=3)
        layer_params = layer.initialize_parameters_recursively(prng_key=init_key)
        dim = hidden_dim if hidden_dim else output_dim

        expected_param_shapes = dict(
            input_proj=dict(weight=(num_layers, input_dim + output_dim, 4, dim))
        )
        if hidden_dim:
            expected_param_shapes["output_proj"] = dict(weight=(num_layers, hidden_dim, output_dim))
        if norm_cfg:
            expected_param_shapes["norm"] = dict(scale=(num_layers, dim), bias=(num_layers, dim))
        self.assertEqual(dict(repeat=dict(layer=expected_param_shapes)), shapes(layer_params))

        # Create the initial states.
        init_states, _ = F(
            layer,
            is_training=True,
            prng_key=prng_key,  # not used.
            state=layer_params,
            inputs=dict(batch_size=batch_size),
            method="init_states",
        )

        if hidden_dim:
            self.assertEqual(
                shapes(init_states),
                {
                    "c": (num_layers, batch_size, hidden_dim),
                    "m": (num_layers, batch_size, layer.output_dim),
                },
            )
        else:
            self.assertEqual(
                shapes(init_states),
                {
                    "c": (num_layers, batch_size, layer.output_dim),
                    "m": (num_layers, batch_size, layer.output_dim),
                },
            )

        inputs = jax.random.normal(data_key, [seq_len, batch_size, input_dim]) * 100.0
        final_states_list = []
        outputs = inputs
        for ll in range(num_layers):
            layer_params_ll = jax.tree.map(lambda param, i=ll: param[i], layer_params)["repeat"][
                "layer"
            ]
            outputs, output_collections = F(
                layer.repeat.layer,  # LSTMCell.
                is_training=True,
                prng_key=prng_key,  # not used.
                state=layer_params_ll,
                inputs=dict(time_major_inputs=outputs),
                drop_output_collections=[],
            )

            self.assertSequenceEqual(outputs.shape, [seq_len, batch_size, layer.output_dim])
            final_states_list.append(output_collections.module_outputs["final_states"])

        # Stack the tree leaves.
        tree_leaves = [jax.tree_util.tree_flatten(t)[0] for t in final_states_list]
        tree_def = jax.tree_util.tree_structure(final_states_list[0])
        final_states = jax.tree_util.tree_unflatten(
            tree_def, [jnp.stack(leaf) for leaf in zip(*tree_leaves)]
        )
        self.assertEqual(shapes(final_states), shapes(init_states))

        forward_outputs, forward_collections = F(
            layer,
            is_training=True,
            prng_key=prng_key,  # not used.
            state=layer_params,
            inputs=dict(time_major_inputs=inputs),
            drop_output_collections=[],
        )
        self.assertSequenceEqual(forward_outputs.shape, (seq_len, batch_size, input_dim))
        # Outputs match layer by layer outputs.
        logging.info("Outputs max=%s, min=%s.", outputs.max(), outputs.min())
        assert_allclose(forward_outputs, outputs, atol=1e-6, rtol=1e-6)
        # States match layer by layer states.
        self.assertNestedAllClose(
            forward_collections.module_outputs["final_states"],
            final_states,
            rtol=1e-6,
            atol=1e-6,
        )

    @parameterized.parameters(
        ([LSTMCell.default_config().set(hidden_dim=4) for _ in range(3)],),
        ([LSTMCell.default_config().set(norm=LayerNorm.default_config()) for _ in range(5)],),
        (
            [
                LSTMCell.default_config().set(hidden_dim=2),
                LSTMCell.default_config().set(norm=LayerNorm.default_config()),
                LSTMCell.default_config(),
            ],
        ),
    )
    @pytest.mark.fp64
    def test_stack_forward_vs_layerwise(self, layer_cfgs):
        batch_size, seq_len, input_dim = 8, 10, 7
        num_layers = len(layer_cfgs)
        output_dims = [3, 2, 5, 4, 1][:num_layers]
        for i in range(num_layers):
            layer_cfgs[i] = layer_cfgs[i].set(output_dim=output_dims[i])

        # StackedRNNLayer.
        layer: StackedRNNLayer = (
            StackedRNNLayer.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                layers=layer_cfgs,
                param_init=GaussianInitializer.default_config().set(std=10.0),
            )
            .instantiate(parent=None)
        )

        init_key, data_key, prng_key = jax.random.split(jax.random.PRNGKey(567), num=3)
        layer_params = layer.initialize_parameters_recursively(prng_key=init_key)
        expected_param_shapes = {}
        for i, layer_cfg in enumerate(layer_cfgs):
            i_dim = input_dim if i == 0 else output_dims[i - 1]
            o_dim = output_dims[i]
            h_dim = layer_cfg.hidden_dim if layer_cfg.hidden_dim else o_dim
            layer_shapes = dict(input_proj=dict(weight=(i_dim + o_dim, 4, h_dim)))
            if layer_cfg.hidden_dim:
                layer_shapes["output_proj"] = dict(weight=(h_dim, o_dim))
            if layer_cfg.norm:
                layer_shapes["norm"] = dict(scale=(h_dim,), bias=(h_dim,))
            expected_param_shapes[f"layer{i}"] = layer_shapes
        self.assertEqual(expected_param_shapes, shapes(layer_params))

        # Create the initial states.
        init_states, _ = F(
            layer,
            is_training=True,
            prng_key=prng_key,  # not used.
            state=layer_params,
            inputs=dict(batch_size=batch_size),
            method="init_states",
        )

        self.assertEqual(len(init_states), num_layers)
        for i, layer_cfg in enumerate(layer_cfgs):
            if layer_cfg.hidden_dim:
                self.assertEqual(
                    shapes(init_states[i]),
                    {
                        "c": (batch_size, layer_cfg.hidden_dim),
                        "m": (batch_size, output_dims[i]),
                    },
                )
            else:
                self.assertEqual(
                    shapes(init_states[i]),
                    {
                        "c": (batch_size, output_dims[i]),
                        "m": (batch_size, output_dims[i]),
                    },
                )

        inputs = jax.random.normal(data_key, [seq_len, batch_size, input_dim]) * 100.0
        final_states_list = []
        outputs = inputs
        for ll in range(num_layers):
            outputs, output_collections = F(
                # pylint: disable-next=protected-access
                layer._layers[ll],  # LSTMCell.
                is_training=True,
                prng_key=prng_key,  # not used.
                state=layer_params[f"layer{ll}"],
                inputs=dict(time_major_inputs=outputs),
                drop_output_collections=[],
            )

            self.assertSequenceEqual(
                outputs.shape,
                # pylint: disable-next=protected-access
                [seq_len, batch_size, layer._layers[ll].output_dim],
            )
            final_states_list.append(output_collections.module_outputs["final_states"])

        forward_outputs, forward_collections = F(
            layer,
            is_training=True,
            prng_key=prng_key,  # not used.
            state=layer_params,
            inputs=dict(time_major_inputs=inputs),
            drop_output_collections=[],
        )
        # Test output dim.
        self.assertSequenceEqual(forward_outputs.shape, (seq_len, batch_size, layer.output_dim))
        # Outputs match layer by layer outputs.
        logging.info("Outputs max=%s, min=%s.", outputs.max(), outputs.min())
        assert_allclose(forward_outputs, outputs, atol=1e-6, rtol=1e-6)
        # States match layer by layer states.

        for ll in range(num_layers):
            self.assertNestedAllClose(
                forward_collections.module_outputs["final_states"][ll],
                final_states_list[ll],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_stack_layer_dim(self):
        with self.assertRaises(ValueError):
            _: StackedRNNLayer = (
                StackedRNNLayer.default_config()
                .set(
                    name="test",
                    input_dim=2,
                    output_dim=3,
                    layers=[LSTMCell.default_config().set(output_dim=2)],
                )
                .instantiate(parent=None)
            )

    @parameterized.parameters(
        (None, 3, 4),
        (LayerNorm.default_config(), None, 2),
        (None, None, 3),
    )
    def test_stack_vs_repeat_forward(self, norm_cfg, hidden_dim, num_layers):
        batch_size, seq_len, input_dim = 8, 12, 9
        stack_layer: StackedRNNLayer = (
            StackedRNNLayer.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                layers=[
                    LSTMCell.default_config().set(
                        hidden_dim=hidden_dim,
                        norm=norm_cfg,
                    )
                    for _ in range(num_layers)
                ],
            )
            .instantiate(parent=None)
        )
        repeat_layer: RepeatedRNNLayer = (
            RepeatedRNNLayer.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                num_layers=num_layers,
                layer=LSTMCell.default_config().set(
                    hidden_dim=hidden_dim,
                    norm=norm_cfg,
                ),
            )
            .instantiate(parent=None)
        )
        init_key, data_key, prng_key = jax.random.split(jax.random.PRNGKey(567), num=3)
        stack_params = stack_layer.initialize_parameters_recursively(prng_key=init_key)
        # Convert `stack_params` to params for `repeat_layer`.
        repeat_params = {
            "repeat": VDict(
                {
                    "layer": jax.tree.map(
                        lambda *xs: jnp.stack(xs),
                        *stack_params.values(),
                    )
                }
            )
        }
        inputs = jax.random.normal(data_key, [seq_len, batch_size, input_dim]) * 100.0
        stack_outputs, stack_collections = F(
            stack_layer,
            is_training=True,
            prng_key=prng_key,  # not used.
            state=stack_params,
            inputs=dict(time_major_inputs=inputs),
            drop_output_collections=[],
        )
        repeat_outputs, repeat_collections = F(
            repeat_layer,
            is_training=True,
            prng_key=prng_key,  # not used.
            state=repeat_params,
            inputs=dict(time_major_inputs=inputs),
            drop_output_collections=[],
        )
        logging.info("Outputs max=%s, min=%s.", repeat_outputs.max(), repeat_outputs.min())
        assert_allclose(repeat_outputs, stack_outputs, atol=1e-6, rtol=1e-6)
        for i in range(num_layers):
            for k, v in repeat_collections.module_outputs["final_states"].items():
                assert_allclose(
                    v[i],
                    stack_collections.module_outputs["final_states"][i][k],
                    atol=1e-6,
                    rtol=1e-6,
                )


if __name__ == "__main__":
    absltest.main()

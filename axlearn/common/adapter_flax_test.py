# Copyright © 2023 Apple Inc.

"""Tests adapter flax layers."""
from typing import Callable

import jax.random
from absl.testing import absltest
from flax import linen as nn
from jax import numpy as jnp
from jax.experimental.pjit import pjit

from axlearn.common import utils
from axlearn.common.adapter_flax import config_for_flax_module
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import InstantiableConfig, config_class
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape


def dummy_inputs(dim, dtype):
    return (jnp.zeros([0, 0, dim], dtype=dtype),), {}


def dummy_inputs_for_norm(dim, dtype):
    args, kwargs = dummy_inputs(dim, dtype)
    kwargs["use_running_average"] = False
    return args, kwargs


class FeedForward(BaseLayer):
    """A dummy feed forward layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures FeedForward."""

        input_dim: int = 0  # The input feature dim.
        hidden_dim: int = 0  # The hidden feature dim.
        output_dim: int = 0  # The output feature dim.
        linear: InstantiableConfig = config_for_flax_module(nn.Dense, dummy_inputs)
        norm: InstantiableConfig = config_for_flax_module(nn.BatchNorm, dummy_inputs_for_norm)

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "linear1",
            cfg.linear.set(
                create_module_kwargs=dict(features=cfg.hidden_dim),
                create_dummy_input_kwargs=dict(dim=cfg.input_dim, dtype=self.dtype()),
            ),
        )
        self._add_child(
            "linear2",
            cfg.linear.set(
                create_module_kwargs=dict(features=cfg.output_dim),
                create_dummy_input_kwargs=dict(dim=cfg.hidden_dim, dtype=self.dtype()),
            ),
        )
        self._add_child(
            "norm",
            cfg.norm.set(
                create_module_kwargs={},
                create_dummy_input_kwargs=dict(dim=cfg.hidden_dim, dtype=self.dtype()),
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x, use_running_average=not self.is_training)
        x = nn.silu(x)
        x = self.linear2(x)
        return x


class FlaxEmbedAttention(BaseLayer):
    """A dummy Flax Embed layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures FlaxEmbedAttention."""

        num_embeddings: int = 0
        feature_dims: int = 0
        embed: InstantiableConfig = config_for_flax_module(nn.Embed, dummy_inputs)

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "embed",
            cfg.embed.set(
                create_module_kwargs=dict(
                    num_embeddings=cfg.num_embeddings, features=cfg.feature_dims
                ),
                create_dummy_input_kwargs=dict(dim=1, dtype=jnp.int32),
            ),
        )

    def forward(self, x):
        "Calls nn.Embed.attend function instead of the default nn.Embed.__call__."
        return self.embed(x, module_method="attend")


class DotReluDot(nn.Module):
    """A Flax layer with sharding annotations.

    https://flax.readthedocs.io/en/latest/guides/parallel_training/flax_on_pjit.html#define-a-layer
    """

    depth: int
    dense_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(
            self.depth,
            kernel_init=nn.with_partitioning(self.dense_init, (None, "model")),
            use_bias=False,
        )(x)
        y = jax.nn.relu(y)
        y = jax.lax.with_sharding_constraint(y, jax.sharding.PartitionSpec("data", "model"))
        w2 = self.param(
            "W2", nn.with_partitioning(self.dense_init, ("model", None)), (self.depth, x.shape[-1])
        )
        z = jnp.dot(y, w2)
        z = jax.lax.with_sharding_constraint(z, jax.sharding.PartitionSpec("data", None))
        return z, None


class FlaxLayerTest(TestCase):
    """Tests FlaxLayer."""

    def test_feed_forward(self):
        batch_size, seq_len, input_dim, hidden_dim, output_dim = 2, 5, 4, 8, 6
        cfg = FeedForward.default_config().set(
            name="test", input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
        cfg.linear.vlog = 5
        layer: FeedForward = cfg.instantiate(parent=None)
        param_specs = layer.create_parameter_specs_recursively()
        layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(1))

        def check_spec_and_param(spec, param):
            self.assertEqual(spec.dtype, param.dtype)
            self.assertSequenceEqual(spec.shape, param.shape)
            self.assertEqual(spec.mesh_axes, None)

        jax.tree.map(check_spec_and_param, param_specs, layer_params)

        self.assertEqual(
            {
                "linear1": {
                    "params": {
                        "bias": (hidden_dim,),
                        "kernel": (input_dim, hidden_dim),
                    },
                },
                "linear2": {
                    "params": {
                        "bias": (output_dim,),
                        "kernel": (hidden_dim, output_dim),
                    },
                },
                "norm": {
                    "batch_stats": {
                        "mean": (hidden_dim,),
                        "var": (hidden_dim,),
                    },
                    "params": {
                        "bias": (hidden_dim,),
                        "scale": (hidden_dim,),
                    },
                },
            },
            utils.shapes(layer_params),
        )

        inputs = jnp.ones([batch_size, seq_len, input_dim], dtype=jnp.float32)
        outputs, output_collection = F(
            layer,
            inputs=(inputs,),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        self.assertEqual((batch_size, seq_len, output_dim), utils.shapes(outputs))
        # TODO(rpang): figure out why the following check is flaky on Rio.
        # self.assertAlmostEqual(-4.712797453976236e-05, outputs.sum().item())
        self.assertEqual(
            [
                ("state_updates/norm/batch_stats/mean", (hidden_dim,)),
                ("state_updates/norm/batch_stats/var", (hidden_dim,)),
            ],
            [(key, value.shape) for key, value in utils.flatten_items(output_collection)],
        )

    def test_custom_module_call(self):
        feature_dims = 16
        num_embeddings = 4
        batch_size = 2
        cfg = FlaxEmbedAttention.default_config().set(
            name="test", num_embeddings=num_embeddings, feature_dims=feature_dims
        )
        cfg.embed.vlog = 5
        layer: FlaxEmbedAttention = cfg.instantiate(parent=None)
        param_specs = layer.create_parameter_specs_recursively()
        layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(1))

        def check_spec_and_param(spec, param):
            self.assertEqual(spec.dtype, param.dtype)
            self.assertSequenceEqual(spec.shape, param.shape)
            self.assertEqual(spec.mesh_axes, None)

        jax.tree.map(check_spec_and_param, param_specs, layer_params)

        self.assertEqual(
            {
                "embed": {
                    "params": {
                        "embedding": (num_embeddings, feature_dims),
                    },
                }
            },
            utils.shapes(layer_params),
        )

        inputs = jnp.ones([batch_size, feature_dims], dtype=jnp.float32)
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertEqual((batch_size, num_embeddings), utils.shapes(outputs))

    def test_sharding(self):
        mesh_shape = (len(jax.devices()) // 2, 2)
        if not is_supported_mesh_shape(mesh_shape):
            self.skipTest(f"Unsupported mesh shape {mesh_shape}")

        with jax.sharding.Mesh(utils.create_device_mesh(mesh_shape=mesh_shape), ("data", "model")):
            dim = 4
            cfg = config_for_flax_module(DotReluDot, dummy_inputs).set(
                create_module_kwargs=dict(depth=dim),
                create_dummy_input_kwargs=dict(dim=dim, dtype=jnp.int32),
            )
            layer = cfg.set(name="test").instantiate(parent=None)
            param_specs = layer.create_parameter_specs_recursively()
            partition_specs = jax.tree.map(lambda spec: spec.mesh_axes, param_specs)

            jit_init_state = pjit(
                layer.initialize_parameters_recursively,
                in_shardings=(None,),
                out_shardings=partition_specs,
            )
            layer_params = jit_init_state(jax.random.PRNGKey(1))

            jax.tree.map(
                lambda x: self.assertFalse(x.value.is_fully_replicated),
                layer_params,
                is_leaf=lambda x: isinstance(x, nn.Partitioned),
            )
            jax.tree.map(
                lambda x: jax.debug.visualize_array_sharding(x.value),
                layer_params,
                is_leaf=lambda x: isinstance(x, nn.Partitioned),
            )


if __name__ == "__main__":
    absltest.main()

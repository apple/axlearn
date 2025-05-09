# Copyright © 2023 Apple Inc.

"""Tests subsampler layers."""

import contextlib
from collections.abc import Sequence
from typing import Optional, Union

import jax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.audio.subsamplers import ConvSubSampler
from axlearn.common import utils
from axlearn.common.layers import BatchNorm
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import safe_not


class ConvSubSamplerTest(TestCase):
    """Tests ConvSubSampler."""

    @parameterized.parameters(
        dict(activation=("nn.tanh", "nn.relu", "nn.silu"), expected=ValueError("pair of string")),
        dict(activation=("nn.tanh",), expected=ValueError("pair of string")),
        dict(activation="nn.tanh"),  # Single value is broadcasted.
        dict(activation=("nn.tanh", None)),  # Some of the values can be None.
        dict(activation=(None, None)),  # Some of the values can be None.
        dict(activation=None),  # Some of the values can be None.
    )
    def test_instantiate(
        self,
        activation: Optional[Union[str, tuple[str, str]]] = None,
        expected: Optional[Exception] = None,
    ):
        """Tests the checks in __init__."""
        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            cfg = ConvSubSampler.default_config().set(
                input_dim=1, output_dim=1, activation=activation
            )
            cfg.set(name="test").instantiate(parent=None)

    @parameterized.parameters(
        dict(input_shape=(), output_dim=5, expected=ValueError("input_shape")),
        # (2, 32, 80, 1) -> (2, 16, 40, 5) -> (2, 8, 20, 5).
        dict(input_shape=(2, 32, 80, 1), output_dim=5, expected=(2, 8, 20, 5)),
    )
    def test_output_shape(
        self,
        input_shape: tuple[int, int],
        output_dim: Union[int, tuple[int, int]],
        expected: Union[tuple[int, int], Exception],
    ):
        """Tests output_shape against specific inputs."""
        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            cfg = ConvSubSampler.default_config().set(input_dim=1, output_dim=output_dim)
            layer = cfg.set(name="test").instantiate(parent=None)
            output_shape = layer.output_shape(input_shape=input_shape)
            self.assertEqual(expected, tuple(output_shape))

    @parameterized.parameters(
        dict(activation=None, expected_activation=[None, None], output_dim=3),
        dict(
            activation=("nn.tanh", None),
            expected_activation=[jax.nn.tanh, None],
            hidden_dim=2,
            output_dim=3,
        ),
        dict(
            activation="nn.relu",
            expected_activation=[jax.nn.relu, jax.nn.relu],
            hidden_dim=2,
            output_dim=3,
        ),
        dict(
            activation=("nn.silu", "nn.gelu"),
            expected_activation=[jax.nn.silu, jax.nn.gelu],
            output_dim=3,
        ),
    )
    def test_activations(
        self,
        activation: Optional[Union[str, tuple]],
        expected_activation: Sequence,
        output_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """Tests that activations and intermediate output dims are read properly."""
        cfg = ConvSubSampler.default_config().set(
            output_dim=output_dim, hidden_dim=hidden_dim, activation=activation
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        # pylint: disable-next=protected-access
        self.assertEqual(expected_activation, layer._activation)
        self.assertEqual(layer.conv1.config.output_dim, hidden_dim or output_dim)
        self.assertEqual(layer.conv2.config.output_dim, output_dim)

    @parameterized.parameters(
        dict(window=3, stride=2, conv_padding=(1, 1), output_dim=10),
        dict(window=5, stride=2, conv_padding=(1, 1), hidden_dim=12, output_dim=8),
        dict(window=5, stride=2, conv_padding=(2, 2), hidden_dim=5, output_dim=3),
        dict(window=5, stride=3, conv_padding=(2, 2), output_dim=6),
    )
    def test_paddings(
        self,
        window: int,
        stride: int,
        conv_padding: tuple[int, int],
        output_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """Tests that padding inputs do not affect outputs."""
        input_dim, num_filters = 1, 80
        cfg = ConvSubSampler.default_config().set(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
        )
        cfg.conv.window = (window, window)
        cfg.conv.strides = (stride, stride)
        cfg.conv.padding = (conv_padding, conv_padding)
        cfg.norm = BatchNorm.default_config()

        # Initialize layer parameters.
        layer = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, data_key1, data_key2 = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        hidden_dim = hidden_dim or output_dim
        self.assertEqual(
            {
                "conv1": dict(weight=(window, window, input_dim, hidden_dim), bias=(hidden_dim,)),
                "norm1": dict(
                    bias=(hidden_dim,),
                    moving_mean=(hidden_dim,),
                    moving_variance=(hidden_dim,),
                    scale=(hidden_dim,),
                ),
                "conv2": dict(weight=(window, window, hidden_dim, output_dim), bias=(output_dim,)),
                "norm2": dict(
                    bias=(output_dim,),
                    moving_mean=(output_dim,),
                    moving_variance=(output_dim,),
                    scale=(output_dim,),
                ),
            },
            utils.shapes(layer_params),
        )

        batch_size, num_frames = 2, 20
        seq_len = jnp.array([15, 15])
        # [batch_size, num_frames, num_filters, input_dim].
        inputs = jnp.repeat(
            jax.random.normal(data_key1, [1, num_frames, num_filters, input_dim]),
            batch_size,
            axis=0,
        )
        # [batch_size, num_frames].
        paddings = jnp.arange(num_frames)[None, :] >= seq_len[:, None]

        padding_data = jax.random.normal(
            data_key2, [batch_size, num_frames, num_filters, input_dim]
        )
        inputs_with_different_paddings = jnp.where(paddings[:, :, None, None], padding_data, inputs)
        outputs, _ = F(
            layer,
            inputs=dict(inputs=inputs_with_different_paddings, paddings=paddings),
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        # Check that the outputs are the same despite differences in padding positions.
        self.assertNestedAllClose(outputs["outputs"][0], outputs["outputs"][1])
        self.assertNestedAllClose(outputs["paddings"][0], outputs["paddings"][1])
        subsampled_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertEqual(tuple(subsampled_shape), outputs["outputs"].shape)
        self.assertEqual(tuple(subsampled_shape)[:2], outputs["paddings"].shape)

    @parameterized.parameters(jnp.float32, jnp.bfloat16)
    def test_activation_summaries(self, dtype):
        """Tests that activation summaries behave as expected."""
        input_dim, num_filters, hidden_dim, output_dim = 1, 80, 12, 8
        prng_key = jax.random.PRNGKey(567)
        prng_key, init_key, data_key = jax.random.split(prng_key, num=3)

        # Initialize layer parameters.
        cfg = ConvSubSampler.default_config().set(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dtype=dtype
        )
        layer = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(init_key)
        dtypes, _ = jax.tree.flatten(jax.tree.map(jnp.dtype, layer_params))
        self.assertTrue(all(dt == dtype for dt in dtypes))

        # Build inputs.
        batch_size, num_frames = 4, 10
        inputs_shape = [batch_size, num_frames, num_filters, input_dim]
        inputs = jax.random.normal(key=data_key, shape=inputs_shape) * 10.0
        lengths = jnp.array([5, 10, 9, 0])
        paddings = jnp.arange(num_frames)[None, :] >= lengths[:, None]
        inputs = inputs.astype(dtype)
        outputs, output_collections = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        # Compute expected summaries.
        input_weights = jnp.sum(lengths)
        inputs = inputs * safe_not(paddings)[:, :, None, None]
        norms = jnp.sqrt(jnp.sum(inputs**2, axis=(2, 3))) / jnp.sqrt(num_filters * input_dim)
        expected_inputs_mean = jnp.sum(inputs) / input_weights / (num_filters * input_dim)
        expected_inputs_norm = jnp.sum(norms) / input_weights

        output_weights = jnp.sum(1 - outputs["paddings"])
        output_norms = jnp.sqrt(jnp.sum(outputs["outputs"] ** 2, axis=(2, 3))) / jnp.sqrt(
            num_filters // 4 * output_dim
        )
        expected_outputs_mean = (
            jnp.sum(outputs["outputs"]) / output_weights / (num_filters // 4 * output_dim)
        )
        expected_outputs_norm = jnp.sum(output_norms) / output_weights

        self.assertNestedAllClose(
            output_collections.summaries["activations/subsampler_inputs_mean"].mean,
            expected_inputs_mean,
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/subsampler_inputs_norm"].mean,
            expected_inputs_norm,
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/subsampler_outputs_mean"].mean,
            expected_outputs_mean,
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/subsampler_outputs_norm"].mean,
            expected_outputs_norm,
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/subsampler_inputs_mean"].weight,
            input_weights,
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/subsampler_outputs_norm"].weight,
            output_weights,
        )


if __name__ == "__main__":
    absltest.main()

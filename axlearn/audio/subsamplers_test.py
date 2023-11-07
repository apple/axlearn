# Copyright © 2023 Apple Inc.

"""Tests subsampler layers."""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import jax
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.audio.subsamplers import ConvSubSampler
from axlearn.common import utils
from axlearn.common.layers import BatchNorm
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase


class ConvSubSamplerTest(TestCase):
    """Tests ConvSubSampler."""

    @parameterized.parameters(
        dict(output_dim=(5,), expected=ValueError("pair of integers")),
        dict(output_dim=(5, 6, 7), expected=ValueError("pair of integers")),
        dict(output_dim=5),  # Single value is broadcasted.
        dict(activation=("nn.tanh", "nn.relu", "nn.silu"), expected=ValueError("pair of string")),
        dict(activation=("nn.tanh",), expected=ValueError("pair of string")),
        dict(activation="nn.tanh"),  # Single value is broadcasted.
        dict(activation=("nn.tanh", None)),  # Some of the values can be None.
        dict(activation=(None, None)),  # Some of the values can be None.
        dict(activation=None),  # Some of the values can be None.
    )
    def test_instantiate(
        self,
        output_dim: Union[int, Tuple[int, int]] = 5,
        activation: Optional[Union[str, Tuple[str, str]]] = None,
        expected: Optional[Exception] = None,
    ):
        """Tests the checks in __init__."""
        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            cfg = ConvSubSampler.default_config().set(
                input_dim=1, output_dim=output_dim, activation=activation
            )
            cfg.set(name="test").instantiate(parent=None)

    @parameterized.parameters(
        dict(input_shape=(), output_dim=5, expected=ValueError("input_shape")),
        # (2, 32, 80, 1) -> (2, 16, 40, 5) -> (2, 8, 20, 5).
        dict(input_shape=(2, 32, 80, 1), output_dim=5, expected=(2, 8, 20, 5)),
    )
    def test_output_shape(
        self,
        input_shape: Tuple[int, int],
        output_dim: Union[int, Tuple[int, int]],
        expected: Union[Tuple[int, int], Exception],
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
        dict(activation=None, expected_activation=[None, None], conv_dims=3),
        dict(
            activation=("nn.tanh", None), expected_activation=[jax.nn.tanh, None], conv_dims=(2, 3)
        ),
        dict(
            activation="nn.relu", expected_activation=[jax.nn.relu, jax.nn.relu], conv_dims=(2, 3)
        ),
        dict(
            activation=("nn.silu", "nn.gelu"),
            expected_activation=[jax.nn.silu, jax.nn.gelu],
            conv_dims=3,
        ),
    )
    def test_activations(
        self,
        activation: Optional[Union[str, Tuple]],
        expected_activation: Sequence,
        conv_dims: Union[int, Tuple[int, int]],
    ):
        """Tests that activations and intermediate output dims are read properly."""
        cfg = ConvSubSampler.default_config().set(output_dim=conv_dims, activation=activation)
        layer = cfg.set(name="test").instantiate(parent=None)

        if not isinstance(conv_dims, tuple):
            conv_dims = (conv_dims, conv_dims)

        # pylint: disable-next=protected-access
        self.assertEqual(expected_activation, layer._activation)
        self.assertEqual(layer.conv1.config.output_dim, conv_dims[0])
        self.assertEqual(layer.conv2.config.output_dim, conv_dims[1])

    @parameterized.parameters(
        dict(window=3, stride=2, conv_padding=(1, 1), conv_dims=10),
        dict(window=5, stride=2, conv_padding=(1, 1), conv_dims=(12, 8)),
        dict(window=5, stride=2, conv_padding=(2, 2), conv_dims=(5, 3)),
        dict(window=5, stride=3, conv_padding=(2, 2), conv_dims=6),
    )
    def test_paddings(
        self,
        window: int,
        stride: int,
        conv_padding: Tuple[int, int],
        conv_dims: Union[int, Tuple[int, int]],
    ):
        """Tests that padding inputs do not affect outputs."""
        input_dim, num_filters = 1, 80
        cfg = ConvSubSampler.default_config().set(input_dim=input_dim, output_dim=conv_dims)
        cfg.conv.window = (window, window)
        cfg.conv.strides = (stride, stride)
        cfg.conv.padding = (conv_padding, conv_padding)
        cfg.norm = BatchNorm.default_config()

        # Initialize layer parameters.
        layer = cfg.set(name="test").instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, data_key1, data_key2 = jax.random.split(prng_key, num=4)
        layer_params = layer.initialize_parameters_recursively(init_key)

        if not isinstance(conv_dims, tuple):
            conv_dims = (conv_dims, conv_dims)

        self.assertEqual(
            {
                "conv1": dict(weight=(window, window, input_dim, conv_dims[0]), bias=conv_dims[:1]),
                "norm1": dict(
                    bias=conv_dims[:1],
                    moving_mean=conv_dims[:1],
                    moving_variance=conv_dims[:1],
                    scale=conv_dims[:1],
                ),
                "conv2": dict(
                    weight=(window, window, conv_dims[0], conv_dims[1]), bias=conv_dims[1:]
                ),
                "norm2": dict(
                    bias=conv_dims[1:],
                    moving_mean=conv_dims[1:],
                    moving_variance=conv_dims[1:],
                    scale=conv_dims[1:],
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

    def test_activation_summaries(self):
        """Tests that activation summaries behave as expected."""
        input_dim, num_filters, conv_dims = 1, 80, (12, 8)
        prng_key = jax.random.PRNGKey(567)
        prng_key, init_key, data_key = jax.random.split(prng_key, num=3)

        # Initialize layer parameters.
        cfg = ConvSubSampler.default_config().set(input_dim=input_dim, output_dim=conv_dims)
        layer = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Build inputs.
        batch_size, num_frames = 4, 10
        inputs_shape = [batch_size, num_frames, num_filters, input_dim]
        inputs = jax.random.normal(key=data_key, shape=inputs_shape) * 10.0
        lengths = jnp.array([5, 10, 9, 0])
        paddings = jnp.arange(num_frames)[None, :] >= lengths[:, None]
        outputs, output_collections = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
        )

        # Compute expected summaries.
        input_weights = jnp.sum(lengths)
        inputs = inputs * (1 - paddings)[:, :, None, None]
        norms = jnp.sqrt(jnp.sum(inputs**2, axis=(2, 3))) / jnp.sqrt(num_filters * input_dim)
        expected_inputs_mean = jnp.sum(inputs) / input_weights / (num_filters * input_dim)
        expected_inputs_norm = jnp.sum(norms) / input_weights

        output_weights = jnp.sum(1 - outputs["paddings"])
        output_norms = jnp.sqrt(jnp.sum(outputs["outputs"] ** 2, axis=(2, 3))) / jnp.sqrt(
            num_filters // 4 * conv_dims[1]
        )
        expected_outputs_mean = (
            jnp.sum(outputs["outputs"]) / output_weights / (num_filters // 4 * conv_dims[1])
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
            output_collections.summaries["activations/subsampler_inputs_mean"].weight, input_weights
        )
        self.assertNestedAllClose(
            output_collections.summaries["activations/subsampler_outputs_norm"].weight,
            output_weights,
        )

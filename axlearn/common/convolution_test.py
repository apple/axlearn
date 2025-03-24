# Copyright © 2024 Apple Inc.
"""Tests convolution layers."""

# pylint: disable=no-self-use
from typing import Optional, Union

import einops
import jax.random
import numpy as np
import torch
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import convolution, utils
from axlearn.common.convolution import (
    Conv1D,
    Conv1DWithPadding,
    Conv2D,
    Conv2DTranspose,
    Conv2DWith1DPadding,
    Conv3D,
    ConvPaddingType,
    StackOverTime,
    compute_conv_paddings,
)
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import shapes


def _copy(src: jnp.ndarray, dst: torch.nn.Parameter):
    with torch.no_grad():
        src = np.asarray(src).copy()
        src = torch.as_tensor(src)
        dst.copy_(src)


class ConvTest(TestCase):
    @parameterized.parameters((1, 1, 1), (1, 2, 1), (2, 1, 2), (3, 1, 3), (3, 2, 5))
    def test_conv_dilate_window(self, window, dilation, expected):
        effective_window = convolution.conv_dilate_window(window=(window,), dilation=(dilation,))[0]
        self.assertEqual(effective_window, expected)

    @parameterized.parameters(
        (10, 3, 1, "SAME", 1, 10),
        (10, 3, 2, "SAME", 1, 5),
        (10, 3, 1, "SAME", 2, 10),
        (10, 3, 2, "SAME", 2, 5),
        (10, 3, 1, "VALID", 1, 8),
        (10, 3, 2, "VALID", 1, 4),
        (10, 3, 1, "VALID", 2, 6),
        (10, 3, 2, "VALID", 2, 3),
        (10, 3, 1, "CAUSAL", 1, 10),
        (10, 3, 2, "CAUSAL", 1, 5),
        (10, 3, 1, "CAUSAL", 2, 10),
        (10, 3, 2, "CAUSAL", 2, 5),
    )
    def test_conv_output_shape(self, in_shape, window, strides, padding, dilation, expected):
        out_shape = convolution.conv_output_shape(
            in_shape=(in_shape,),
            window=(window,),
            strides=(strides,),
            padding=padding,
            dilation=(dilation,),
        )[0]
        self.assertEqual(out_shape, expected)

    @parameterized.parameters(
        ([0, 0, 0, 1], [0, 0, 0, 1], 1, "SAME"),
        ([0], [], 1, "VALID"),
        ([0, 0], [], 1, "VALID"),
        ([0, 0, 0], [0], 1, "VALID"),
        ([0, 0, 0, 0], [0, 0], 1, "VALID"),
        ([0, 0, 0, 1], [0, 0], 1, "VALID"),
        ([0, 0, 0, 0], [0], 2, "VALID"),
        ([0, 0, 0, 1], [0], 2, "VALID"),
        ([0, 0, 1, 1], [0], 2, "VALID"),
        ([0, 0, 0, 0, 0], [0, 0], 2, "VALID"),
        ([0, 0, 0, 0, 1], [0, 0], 2, "VALID"),
        ([0, 0, 0, 1, 1], [0, 0], 2, "VALID"),
        ([0, 0, 1, 1, 1], [0, 1], 2, "VALID"),
        ([0, 0, 0, 0, 0, 0], [0, 0], 2, "VALID"),
        ([0, 0, 0, 0, 0, 1], [0, 0], 2, "VALID"),
        ([0, 0, 0, 0, 1, 1], [0, 0], 2, "VALID"),
        ([0, 0, 0, 1, 1, 1], [0, 0], 2, "VALID"),
        ([0, 0, 1, 1, 1, 1], [0, 1], 2, "VALID"),
    )
    def test_conv_padding(self, input_paddings, expected_paddings, stride: int, padding_cfg: str):
        """Tests conv_output_shape() with SAME and VALID padding cfg."""
        # This test is from lingvo
        # https://github.com/tensorflow/lingvo/blob/master/lingvo/core/conv_layers_with_time_padding_test.py#L157.
        window = 3
        out_paddings = compute_conv_paddings(
            jnp.array([input_paddings]), window=window, stride=stride, conv_padding=padding_cfg
        )
        assert_allclose(out_paddings[0], expected_paddings)

    @parameterized.parameters(
        (5, 1, "SAME", 1, (2, 2)),
        (5, 2, "SAME", 1, (2, 2)),
        (5, 3, "SAME", 1, (2, 2)),
        (5, 1, "SAME", 2, (4, 4)),
        (5, 2, "SAME", 2, (4, 4)),
        (5, 3, "SAME", 2, (4, 4)),
        (5, 1, "VALID", 1, (0, 0)),
        (5, 2, "VALID", 1, (0, 0)),
        (5, 3, "VALID", 1, (0, 0)),
        (5, 1, "VALID", 2, (0, 0)),
        (5, 2, "VALID", 2, (0, 0)),
        (5, 3, "VALID", 2, (0, 0)),
        (5, 1, "CAUSAL", 1, (4, 0)),
        (5, 2, "CAUSAL", 1, (3, 1)),
        (5, 3, "CAUSAL", 1, (2, 2)),
        (5, 1, "CAUSAL", 2, (8, 0)),
        (5, 2, "CAUSAL", 2, (7, 1)),
        (5, 3, "CAUSAL", 2, (6, 2)),
    )
    def test_conv_explicit_padding(
        self, window: int, stride: int, padding: ConvPaddingType, dilation: int, expected
    ):
        """Tests the cases in conv_explicit_padding() description."""
        explicit_padding = convolution.conv_explicit_padding(
            window=(window,),
            strides=(stride,),
            padding=padding,
            dilation=(dilation,),
        )
        assert_allclose(explicit_padding[0], expected)

    @parameterized.parameters(
        (5, 1, "SAME", [0, 0, 0, 0, 1, 1]),
        (5, 2, "SAME", [0, 0, 1]),
        (5, 1, "VALID", [0, 0]),
        (5, 2, "VALID", [0]),
        (5, 1, "SAME", [0, 0, 0, 0, 1, 1]),
        (5, 2, "SAME", [0, 0, 1]),
    )
    def test_conv_output_1d_padding_simple(
        self, window: int, stride: int, padding: ConvPaddingType, expected
    ):
        """Tests the cases in conv_explicit_padding() description."""
        paddings = jnp.array([[0, 0, 0, 0, 1, 1]], dtype=jnp.bool)
        out_paddings = compute_conv_paddings(
            paddings, window=window, stride=stride, conv_padding=padding
        )
        assert_allclose(out_paddings[0], expected)

    @parameterized.parameters(
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0]),
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0]),
        ([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0]),
        ([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0]),
        ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1]),
        ([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1]),
        ([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1]),
        ([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1]),
    )
    def test_conv_output_1d_padding_causal(self, in_paddings, expected):
        """Test the below cases.

        The formula for CAUSAL padding is `(window - stride, stride - 1)`.
        With window=15 and stride=6, padding is (9, 5).
        Below are examples illustrating how input paddings are transformed into output
        paddings across different scenarios.

            left_pad         |         input paddings              -> outputs paddings
        1) |1 1 1|1 1 1|1 1 1|0 0 0|0 0 0|0 0 0|0 0 0|0 0 0|0 0 0| -> 0 0 0
        2) |1 1 1|1 1 1|1 1 1|1 0 0|0 0 0|0 0 0|0 0 0|0 0 0|0 0 0| -> 1 0 0
        3) |1 1 1|1 1 1|1 1 1|1 1 1|1 1 1|0 0 0|0 0 0|0 0 0|0 0 0| -> 1 0 0
        4) |1 1 1|1 1 1|1 1 1|1 1 1|1 1 1|1 0 0|0 0 0|0 0 0|0 0 0| -> 1 1 0
        5) |1 1 1|1 1 1|1 1 1|1 1 1|1 1 1|1 1 1|1 1 1|1 1 1|1 1 1| -> 1 1 1
        6) |1 1 1|1 1 1|1 1 1|0 1 1|1 1 1|1 1 1|1 1 1|1 1 1|1 1 1| -> 0 1 1
        7) |1 1 1|1 1 1|1 1 1|0 0 0|0 0 0|1 1 1|1 1 1|1 1 1|1 1 1| -> 0 1 1
        8) |1 1 1|1 1 1|1 1 1|0 0 0|0 0 0|0 1 1|1 1 1|1 1 1|1 1 1| -> 0 0 1
            |_________________^_________|
                        |_________________^_________|
                                    |_________________^_________|

        Let's take a closer look at case 7). In case 7), the first window component fully
        covers all 0s, so the first component of the output padding should be the last
        0 component, meaning the second component is 1.

        In case 8), however, the first window component does not cover all 0s, so the
        next component should also be 0. If the second component were 1, information
        from the last partial window of the input would be lost.

        In general, the anchor point should be the next position after the right edge
        of the previous window. Since the anchor is defined by the left pad,
        `left_pad = window - stride`, and `right_pad = (window - 1) - left_pad`,
        simplifying to `right_pad = stride - 1`.
        """
        window = 15
        stride = 6
        padding = "CAUSAL"
        explicit_padding = convolution.conv_explicit_padding(
            window=(window,), strides=(stride,), padding=padding, dilation=(1,)
        )
        assert_allclose(explicit_padding[0], (9, 5))

        in_paddings = jnp.array([in_paddings], dtype=jnp.bool)
        out_paddings = compute_conv_paddings(
            in_paddings, window=window, stride=stride, conv_padding=padding
        )[0]
        assert_allclose(out_paddings, expected)

    @parameterized.parameters(
        (3, 1, ((1, 1),), "SAME"),
        (3, 1, ((0, 0),), "VALID"),
        (3, 1, ((2, 0),), "CAUSAL"),
        (3, 2, ((1, 1),), "SAME"),
        (3, 2, ((0, 0),), "VALID"),
        (3, 2, ((1, 1),), "CAUSAL"),
        (5, 2, ((2, 2),), "SAME"),
        (5, 2, ((0, 0),), "VALID"),
        (5, 2, ((3, 1),), "CAUSAL"),
    )
    def test_conv_output_1d_padding_against_str_padding(
        self, window: int, stride: int, padding: ConvPaddingType, ref_padding: ConvPaddingType
    ):
        """Tests conv_output_shape() with explicit padding cfg."""
        batch_size = 5
        seq_len = 5
        paddings = jnp.triu(jnp.ones((batch_size, seq_len)), k=1)

        explicit_padding = convolution.conv_explicit_padding(
            window=(window,), strides=(stride,), padding=ref_padding, dilation=(1,)
        )
        assert_allclose(explicit_padding, padding[:1])

        out_paddings = compute_conv_paddings(
            paddings, window=window, stride=stride, conv_padding=padding
        )
        ref_paddings = compute_conv_paddings(
            paddings, window=window, stride=stride, conv_padding=ref_padding
        )
        assert_allclose(out_paddings, ref_paddings)

    @parameterized.parameters(
        ("SAME", 1, [0, 0, 0, 0, 1, 1], [0, 0, 1]),
        ("VALID", 1, [0, 0, 0, 0, 1, 1], [0]),
        ("CAUSAL", 1, [0, 0, 0, 0, 1, 1], [0, 0, 1]),
        ("SAME", 2, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1]),
        ("VALID", 2, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0]),
        ("CAUSAL", 2, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1]),
    )
    def test_compute_conv_paddings_with_dilation(
        self, padding: ConvPaddingType, dilation: int, paddings, expected
    ):
        """Tests compute_conv_paddings() as described in conv_explicit_padding()."""
        window, stride = 5, 2
        out_paddings = compute_conv_paddings(
            jnp.array([paddings]),
            window=window,
            stride=stride,
            conv_padding=padding,
            dilation=dilation,
        )[0]
        assert_allclose(out_paddings, expected)

    @parameterized.parameters(
        (5, "SAME", None, [0, 0, 0, 1, 1, 1]),
        (5, "SAME", 1, ValueError),
        (5, "SAME", 2, [0, 0, 0, 1, 1, 1]),
        (5, "SAME", 3, ValueError),
        (5, ((1, 1),), None, [0, 0, 0, 1]),
        (5, ((1, 1),), 0, ValueError),
        (5, ((1, 1),), 1, [0, 0, 0, 1]),
        (5, ((1, 1),), 2, [0, 0, 1, 1]),
        (5, ((1, 1),), 3, [0, 1, 1, 1]),
        (5, ((1, 1),), 4, ValueError),
        (5, "VALID", None, [0, 0]),
        (5, "VALID", 0, [0, 0]),
        (5, "VALID", 1, [0, 0]),
        (5, "VALID", 2, [0, 1]),
        (5, "VALID", 3, [1, 1]),
        (5, "VALID", 4, [1, 1]),
        (5, "CAUSAL", None, [0, 0, 0, 1, 1, 1]),
        (5, "CAUSAL", 3, ValueError),
        (5, "CAUSAL", 4, [0, 0, 0, 1, 1, 1]),
        (5, "CAUSAL", 5, ValueError),
    )
    def test_conv_output_1d_padding_with_anchor(self, window, padding, anchor, expected_paddings):
        input_paddings = [0, 0, 0, 1, 1, 1]
        try:
            out_paddings = compute_conv_paddings(
                jnp.array([input_paddings]),
                window=window,
                stride=1,
                conv_padding=padding,
                anchor=anchor,
            )
            assert_allclose(out_paddings[0], expected_paddings)
        except ValueError as e:
            self.assertTrue(isinstance(e, expected_paddings))

    @parameterized.named_parameters(
        ("w3s1d1_VALID", 3, 1, "VALID", None),
        ("w3s1d2_VALID", 3, 1, "VALID", 2),
        ("w3s1d1_SAME", 3, 1, "SAME", None),
        ("w4s1d1_SAME", 4, 1, "SAME", None),
        ("w4s1d3_SAME", 4, 1, "SAME", 3),
        ("w4s1d1_CAUSAL", 4, 1, ((3, 0),), None),
        ("w4s1d5_CAUSAL", 4, 1, ((3, 0),), 5),
    )
    def test_conv1d(
        self,
        window: int,
        strides: int,
        padding: ConvPaddingType,
        dilation: Optional[int],
    ):
        input_dim, output_dim = 4, 6
        cfg = Conv1D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=padding,
            dilation=dilation,
        )
        layer: Conv1D = cfg.instantiate(parent=None)
        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, input_dim, output_dim), bias=(output_dim,)),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 17, input_dim])
        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, output_shape)

        # Compute ref outputs.
        if isinstance(padding, str):
            ref_padding = padding.lower()
            ref_inputs = inputs
        else:
            # torch.nn.Conv1d does not support asymmetric padding, so pad manually and use "valid".
            ref_padding = "valid"
            ref_inputs = jnp.pad(inputs, ((0, 0), padding[0], (0, 0)))
        ref = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            groups=1,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
            dilation=1 if dilation is None else dilation,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(2, 1, 0), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(ref_inputs.transpose(0, 2, 1)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 1))

    @parameterized.named_parameters(
        ("w3s1_VALID", 3, 1, "VALID"),
        ("w3s1_SAME", 3, 1, "SAME"),
        ("w4s1_SAME", 4, 1, "SAME"),
        ("w4s1_CAUSAL", 4, 1, ((3, 0),)),
    )
    def test_depthwise_conv1d(
        self,
        window: int,
        strides: int,
        padding: ConvPaddingType,
    ):
        input_dim = 4
        cfg = Conv1D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=input_dim,
            num_input_dim_groups=input_dim,
            window=window,
            strides=strides,
            padding=padding,
        )
        layer: Conv1D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, 1, input_dim), bias=(input_dim,)),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 7, input_dim])

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, output_shape)

        # Compute ref outputs.
        if isinstance(padding, str):
            ref_padding = padding.lower()
            ref_inputs = inputs
        else:
            # torch.nn.Conv1d does not support asymmetric padding, so pad manually and use "valid".
            ref_padding = "valid"
            ref_inputs = jnp.pad(inputs, ((0, 0), padding[0], (0, 0)))
        ref = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            groups=input_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(2, 1, 0), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(ref_inputs.transpose(0, 2, 1)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 1))

    # Fails if tolerance is made smaller.
    @parameterized.named_parameters(
        {
            "testcase_name": "1x1",
            "window": (1, 1),
            "strides": (1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2_VALID",
            "window": (2, 2),
            "strides": (1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2_SAME",
            "window": (2, 2),
            "strides": (1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2_S2_VALID",
            "window": (2, 2),
            "strides": (2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_VALID",
            "window": (3, 3),
            "strides": (1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_SAME",
            "window": (3, 3),
            "strides": (1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_S2_VALID",
            "window": (3, 3),
            "strides": (2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_S2_PADDING1",
            "window": (3, 3),
            "strides": (2, 2),
            "padding": (1, 1),
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_GROUPS4",
            "window": (3, 3),
            "strides": (1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 4,
        },
    )
    def test_conv2d(
        self,
        window: tuple[int, int],
        strides: tuple[int, int],
        padding: Union[str, tuple[int, int]],
        num_input_dim_groups: int,
    ):
        input_dim, output_dim = 256, 128
        if isinstance(padding, tuple):
            conv_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            conv_padding = padding
        cfg = Conv2D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=conv_padding,
            num_input_dim_groups=num_input_dim_groups,
        )
        layer: Conv2D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(
                weight=(window[0], window[1], input_dim // num_input_dim_groups, output_dim),
                bias=(output_dim,),
            ),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 10, 7, input_dim])

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        ref_padding = padding.lower() if isinstance(padding, str) else padding
        ref = torch.nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
            groups=num_input_dim_groups,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(3, 2, 0, 1), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(inputs.transpose(0, 3, 1, 2)))
        # We currently don't match PyTorch as closely as we would like.
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 1), atol=4e-6)
        # Tests output_shape.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, output_shape)

    @parameterized.named_parameters(
        ("1x1", (1, 1), (1, 1), "VALID", None),
        ("2x2_VALID", (2, 2), (1, 1), "VALID", None),
        ("2x2_SAME", (2, 2), (1, 1), "SAME", None),
        ("2x2_CAUSAL", (2, 2), (1, 1), "CAUSAL", None),
        ("2x2_S2_VALID", (2, 2), (2, 2), "VALID", None),
        ("2x2_S2_CAUSAL", (2, 2), (2, 2), "CAUSAL", None),
        ("3x3_VALID", (3, 3), (1, 1), "VALID", None),
        ("3x3_VALID_A0", (3, 3), (1, 1), "VALID", 0),
        ("3x3_VALID_A1", (3, 3), (1, 1), "VALID", 1),
        ("3x3_VALID_A2", (3, 3), (1, 1), "VALID", 2),
        ("3x3_SAME", (3, 3), (1, 1), "SAME", None),
        ("3x3_CAUSAL", (3, 3), (1, 1), "CAUSAL", None),
        ("3x3_S2_VALID", (3, 3), (2, 2), "VALID", None),
        ("3x3_S2_CAUSAL", (3, 3), (2, 2), "CAUSAL", None),
        ("3x3_S2_PADDING1", (3, 3), (2, 2), (1, 1), None),
    )
    def test_conv2d_with_1d_padding(
        self,
        window: tuple[int, int],
        strides: tuple[int, int],
        padding: Union[str, tuple[int, int]],
        anchor: Optional[int],
    ):
        """Tests that Conv2DWith1DPadding has consistent outputs under different padding lengths.

        Generates a batch of input sequences. Pads the sequences under different lengths.
        Checks that the outputs are the same.
        """
        input_dim, input_channel, output_dim = 4, 7, 6
        if isinstance(padding, tuple):
            conv_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            conv_padding = padding
        cfg = Conv2DWith1DPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=conv_padding,
            anchor=anchor,
        )
        layer: Conv2DWith1DPadding = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window[0], window[1], input_dim, output_dim), bias=(output_dim,)),
            shapes(layer_params),
        )
        # Generate a batch of 10 input sequences.
        batch_size, max_seq_len = 10, 10

        prng_key, input_key = jax.random.split(prng_key)
        inputs = (
            jax.random.normal(input_key, [batch_size, max_seq_len, input_channel, input_dim]) * 100
        )

        # The 10 sequences have length 1 to 10.
        paddings = jnp.triu(jnp.ones((batch_size, max_seq_len)), k=1).astype(jnp.bool)

        # Compute layer outputs.
        (ref_outputs, ref_paddings), _ = F(
            layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(ref_outputs.shape, output_shape)

        random_keys = jax.random.split(input_key, num=2 * max_seq_len)
        for seq_len in range(1, max_seq_len):
            # We create a new batch. The time axis of the new batch is of length seq_len.
            permute_idx = jax.random.permutation(random_keys[2 * (seq_len - 1)], seq_len)
            inputs_batch = jnp.take_along_axis(inputs, permute_idx[:, None, None, None], axis=0)[
                :, :seq_len
            ]
            paddings_batch = jnp.take_along_axis(paddings, permute_idx[:, None], axis=0)[
                :, :seq_len
            ]

            # Generate random data at padding positions.
            random_data = (
                jax.random.normal(
                    random_keys[2 * seq_len - 1],
                    [len(permute_idx), seq_len, input_channel, input_dim],
                )
                * 1000
            )
            inputs_new_batch = jnp.where(
                paddings_batch[:, :, None, None], random_data, inputs_batch
            )

            (outputs_batch, output_paddings_batch), _ = F(
                layer,
                inputs=dict(x=inputs_new_batch, paddings=paddings_batch),
                is_training=True,
                state=layer_params,
                prng_key=prng_key,
            )
            output_len = output_paddings_batch.shape[1]
            if output_len > 0:
                assert_allclose(
                    outputs_batch,
                    jnp.take_along_axis(ref_outputs, permute_idx[:, None, None, None], axis=0)[
                        :, :output_len
                    ],
                )
                assert_allclose(
                    output_paddings_batch,
                    jnp.take_along_axis(ref_paddings, permute_idx[:, None], axis=0)[:, :output_len],
                )

    @parameterized.named_parameters(
        ("1_S1", 1, 1, "VALID", None),
        ("2_S1_VALID", 2, 1, "VALID", None),
        ("2_S2_SAME", 2, 2, "SAME", None),
        ("2_S_CAUSAL", 2, 1, "CAUSAL", None),
        ("2_S2_VALID", 2, 2, "VALID", None),
        ("2_S2_CAUSAL", 2, 2, "CAUSAL", None),
        ("3_S1_VALID", 3, 1, "VALID", None),
        ("3_S1_VALID_A0", 3, 1, "VALID", 0),
        ("3_S1_VALID_A1", 3, 1, "VALID", 1),
        ("3_S1_VALID_A2", 3, 1, "VALID", 2),
        ("3_S1_SAME", 3, 1, "SAME", None),
        ("3_S1_CAUSAL", 3, 1, "CAUSAL", None),
        ("3_S2_VALID", 3, 2, "VALID", None),
        ("3_S2_CAUSAL", 3, 2, "CAUSAL", None),
    )
    def test_conv1d_against_conv2d_with_1d_padding(
        self,
        window: int,
        strides: int,
        padding: ConvPaddingType,
        anchor: Optional[int],
    ):
        input_dim, output_dim = 4, 6
        ref_cfg = Conv2DWith1DPadding.default_config().set(
            name="ref",
            input_dim=input_dim,
            output_dim=output_dim,
            window=(window, 1),
            strides=(strides, 1),
            padding=padding,
            anchor=anchor,
        )
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = Conv1DWithPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=padding,
            anchor=anchor,
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        state = ref_layer.initialize_parameters_recursively(init_key)
        test_state = dict(
            bias=state["bias"], weight=einops.rearrange(state["weight"], "t 1 i o -> t i o")
        )

        # Generate a batch of 10 input sequences.
        batch_size, max_seq_len = 10, 10

        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, input_dim])
        # The 10 sequences have length 1 to 10.
        paddings = jnp.triu(jnp.ones((batch_size, max_seq_len)), k=1).astype(jnp.bool)

        (test_outputs, test_paddings), _ = F(
            test_layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=test_state,
            prng_key=prng_key,
        )
        output_shape = test_layer.output_shape(input_shape=inputs.shape)
        assert_allclose(test_outputs.shape, output_shape)

        inputs = einops.rearrange(inputs, "b t i -> b t 1 i")
        (ref_outputs, ref_paddings), _ = F(
            ref_layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=state,
            prng_key=prng_key,
        )
        output_shape = ref_layer.output_shape(input_shape=inputs.shape)
        assert_allclose(ref_outputs.shape, output_shape)
        ref_outputs = einops.rearrange(ref_outputs, "b t 1 o -> b t o")

        self.assertEqual(ref_paddings.dtype, jnp.bool)
        self.assertEqual(test_paddings.dtype, jnp.bool)
        assert_allclose(ref_paddings, test_paddings)
        assert_allclose(ref_outputs, test_outputs)

    @parameterized.named_parameters(
        {
            "testcase_name": "1x1x1",
            "window": (1, 1, 1),
            "strides": (1, 1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2x2_VALID",
            "window": (2, 2, 2),
            "strides": (1, 1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2x2_SAME",
            "window": (2, 2, 2),
            "strides": (1, 1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2x2_S2_VALID",
            "window": (2, 2, 2),
            "strides": (2, 2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_VALID",
            "window": (3, 3, 3),
            "strides": (1, 1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_SAME",
            "window": (3, 3, 3),
            "strides": (1, 1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_S2_VALID",
            "window": (3, 3, 3),
            "strides": (2, 2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_S2_PADDING1",
            "window": (3, 3, 3),
            "strides": (2, 2, 2),
            "padding": (1, 1, 1),
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_GROUPS4",
            "window": (3, 3, 3),
            "strides": (1, 1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 4,
        },
    )
    def test_conv3d(
        self,
        window: tuple[int, int],
        strides: tuple[int, int],
        padding: Union[str, tuple[int, int]],
        num_input_dim_groups: int,
    ):
        input_dim, output_dim = 4, 8
        if isinstance(padding, tuple):
            conv_padding = (
                (padding[0], padding[0]),
                (padding[1], padding[1]),
                (padding[2], padding[2]),
            )
        else:
            conv_padding = padding
        cfg = Conv3D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=conv_padding,
            num_input_dim_groups=num_input_dim_groups,
        )
        layer: Conv3D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        expected = dict(
            weight=(window[0], window[1], window[2], input_dim // num_input_dim_groups, output_dim),
            bias=(output_dim,),
        )
        self.assertEqual(
            expected,
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)

        batch_size = 2
        inputs = jax.random.normal(input_key, [batch_size, 10, 7, 4, input_dim])

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        ref_padding = padding.lower() if isinstance(padding, str) else padding
        ref = torch.nn.Conv3d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
            groups=num_input_dim_groups,
        )

        # weight.shape: (H, W, D, I, O)
        # ref.weight.shape: (O, I, H, W, D)
        _copy(layer_params["weight"].transpose(4, 3, 0, 1, 2), ref.weight)
        _copy(layer_params["bias"], ref.bias)

        ref_outputs = ref(as_torch_tensor(inputs.transpose(0, 4, 1, 2, 3)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 4, 1))

        # Tests output_shape.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, output_shape)


class ConvTransposeTest(TestCase):
    CONVT_EXPLICIT_PADDING_PARAMS = [
        (3, 1, "SAME", 1, (1, 1)),
        (3, 2, "SAME", 1, (2, 1)),
        (3, 3, "SAME", 1, (2, 2)),
        (3, 4, "SAME", 1, (2, 3)),
        (3, 1, "SAME", 2, (2, 2)),
        (3, 2, "SAME", 2, (3, 2)),
        (3, 3, "SAME", 2, (3, 3)),
        (3, 1, "VALID", 1, (2, 2)),
        (3, 2, "VALID", 1, (2, 2)),
        (3, 3, "VALID", 1, (2, 2)),
        (3, 4, "VALID", 1, (2, 3)),
        (3, 1, "VALID", 2, (4, 4)),
        (3, 2, "VALID", 2, (4, 4)),
        (3, 3, "VALID", 2, (4, 4)),
        (3, 1, "CAUSAL", 1, (2, 0)),
        (3, 2, "CAUSAL", 1, (2, 1)),
        (3, 3, "CAUSAL", 1, (2, 2)),
        (3, 4, "CAUSAL", 1, (2, 3)),
        (3, 1, "CAUSAL", 2, (4, 0)),
        (3, 2, "CAUSAL", 2, (4, 1)),
        (3, 3, "CAUSAL", 2, (4, 2)),
    ]

    @parameterized.parameters(*CONVT_EXPLICIT_PADDING_PARAMS)
    def test_conv_transpose_explicit_padding(self, window, strides, padding, dilation, expected):
        """Tests the cases in conv_transpose_explicit_padding() description."""
        explicit_padding = convolution.conv_transpose_explicit_padding(
            window=(window,),
            strides=(strides,),
            padding=padding,
            dilation=(dilation,),
        )
        assert_allclose(explicit_padding[0], expected)

    @parameterized.parameters(*CONVT_EXPLICIT_PADDING_PARAMS)
    def test_conv_transpose_explicit_padding_against_jax(
        self, window, strides, padding, dilation, expected
    ):
        """Compare with jax.lax.convolution._conv_transpose_padding()."""
        if padding == "CAUSAL":
            self.skipTest("Causal padding is not supported in JAX.")

        # Copied from jax.lax.convolution._conv_transpose_padding.
        def _conv_transpose_padding(k, s, padding):
            if padding == "SAME":
                pad_len = k + s - 2
                if s > k - 1:
                    pad_a = k - 1
                else:
                    pad_a = int(np.ceil(pad_len / 2))
            elif padding == "VALID":
                pad_len = k + s - 2 + max(k - s, 0)
                pad_a = k - 1
            else:
                raise ValueError("Padding mode must be `SAME` or `VALID`.")
            pad_b = pad_len - pad_a
            return pad_a, pad_b

        dilate_window = convolution.conv_dilate_window(window=(window,), dilation=(dilation,))[0]
        ref_padding = _conv_transpose_padding(dilate_window, strides, padding)

        explicit_padding = convolution.conv_transpose_explicit_padding(
            window=(window,),
            strides=(strides,),
            padding=padding,
            dilation=(dilation,),
        )

        assert_allclose(explicit_padding[0], ref_padding)
        assert_allclose(expected, ref_padding)

    @parameterized.parameters(
        (3, 1, "SAME", 1, 4),
        (3, 2, "SAME", 1, 8),
        (3, 3, "SAME", 1, 12),
        (3, 4, "SAME", 1, 16),
        (3, 1, "SAME", 2, 4),
        (3, 2, "SAME", 2, 8),
        (3, 3, "SAME", 2, 12),
        (3, 1, "VALID", 1, 6),
        (3, 2, "VALID", 1, 9),
        (3, 3, "VALID", 1, 12),
        (3, 4, "VALID", 1, 16),
        (3, 1, "VALID", 2, 8),
        (3, 2, "VALID", 2, 11),
        (3, 3, "VALID", 2, 14),
        (3, 1, "CAUSAL", 1, 4),
        (3, 2, "CAUSAL", 1, 8),
        (3, 3, "CAUSAL", 1, 12),
        (3, 4, "CAUSAL", 1, 16),
        (3, 1, "CAUSAL", 2, 4),
        (3, 2, "CAUSAL", 2, 8),
        (3, 3, "CAUSAL", 2, 12),
    )
    def test_conv_transpose_output_shape(self, window, strides, padding, dilation, expected):
        """Tests the cases in conv_transpose_explicit_padding() description."""
        out_shape = convolution.conv_transpose_output_shape(
            in_shape=(4,),
            window=(window,),
            strides=(strides,),
            padding=padding,
            dilation=(dilation,),
        )
        assert_allclose(out_shape[0], expected)

    @parameterized.parameters(
        (3, 1, "SAME", 1, [0, 0, 1, 1]),
        (3, 2, "SAME", 1, [0, 0, 0, 0, 1, 1, 1, 1]),
        (3, 3, "SAME", 1, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 4, "SAME", 1, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3, 1, "SAME", 2, [0, 0, 1, 1]),
        (3, 2, "SAME", 2, [0, 0, 0, 0, 1, 1, 1, 1]),
        (3, 3, "SAME", 2, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 1, "VALID", 1, [0, 0, 1, 1, 1, 1]),
        (3, 2, "VALID", 1, [0, 0, 0, 0, 1, 1, 1, 1, 1]),
        (3, 3, "VALID", 1, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 4, "VALID", 1, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3, 1, "VALID", 2, [0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 2, "VALID", 2, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
        (3, 3, "VALID", 2, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3, 1, "CAUSAL", 1, [0, 0, 1, 1]),
        (3, 2, "CAUSAL", 1, [0, 0, 0, 0, 1, 1, 1, 1]),
        (3, 3, "CAUSAL", 1, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 4, "CAUSAL", 1, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        (3, 1, "CAUSAL", 2, [0, 0, 1, 1]),
        (3, 2, "CAUSAL", 2, [0, 0, 0, 0, 1, 1, 1, 1]),
        (3, 3, "CAUSAL", 2, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
    )
    def test_compute_conv_transpose_paddings(self, window, strides, padding, dilation, expected):
        """Tests the cases in conv_transpose_explicit_padding() description."""
        in_paddings = jnp.array([0, 0, 1, 1], dtype=jnp.bool)[None, :]
        out_paddings = convolution.compute_conv_transpose_paddings(
            in_paddings, window=window, stride=strides, conv_padding=padding, dilation=dilation
        )
        expected = jnp.array(expected).astype(out_paddings.dtype)
        self.assertNestedEqual(out_paddings[0], expected)

    @parameterized.product(
        window=[1, 3],
        strides=[1, 2, 3],
        padding=["SAME", "VALID", "CAUSAL"],
        dilation=[1, 2],
        value=[0, 1],
    )
    def test_compute_conv_transpose_paddings_all0or1(
        self, window, strides, padding, dilation, value
    ):
        """If in_paddings is all valid or invalid, out_paddings must be all valid or invalid."""
        in_paddings = jnp.full([1, 4], fill_value=value, dtype=jnp.bool)
        out_paddings = convolution.compute_conv_transpose_paddings(
            in_paddings, window=window, stride=strides, conv_padding=padding, dilation=dilation
        )
        expected = jnp.full_like(out_paddings, fill_value=value, dtype=jnp.bool)
        self.assertNestedEqual(out_paddings, expected)

    CONVT_PADDINGS_PARAMS = dict(
        in_paddings=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        ],
        window=[1, 3],
        padding=["SAME", "VALID", "CAUSAL"],
        dilation=[1, 2],
    )

    @parameterized.product(**CONVT_PADDINGS_PARAMS, strides=[1, 2, 3])
    def test_compute_conv_transpose_paddings_with_conv_paddings(
        self, in_paddings, window, strides, padding, dilation
    ):
        """Check if ConvT -> Conv preserves information."""
        in_paddings = jnp.array(in_paddings, dtype=jnp.bool)[None, :]
        out_paddings = convolution.compute_conv_transpose_paddings(
            in_paddings, window=window, stride=strides, conv_padding=padding, dilation=dilation
        )

        recon_paddings = convolution.compute_conv_paddings(
            out_paddings, window=window, stride=strides, conv_padding=padding, dilation=dilation
        )
        self.assertNestedEqual(recon_paddings[0], in_paddings[0])

    @parameterized.product(**CONVT_PADDINGS_PARAMS)
    def test_compute_conv_transpose_paddings_against_conv_paddings(
        self, in_paddings, window, padding, dilation
    ):
        # compute_conv_transpose_paddings and compute_conv_paddings are same when window_stride=1
        # (stride of Conv2D) and lhs_dilation=1 (stride of Conv2DTranspose).
        strides = 1
        if padding == "VALID":
            # TODO(dhwang2,ruoming): Currently, anchor is pad_left but it should be the midpoint
            # between [pad_left, pad_right). Otherwise, the consistency of VALID padding is broken.
            # For reference, the midpoint in SAME and CAUSAL is left_pad.
            dilate_window = convolution.conv_dilate_window(window=(window,), dilation=(dilation,))[
                0
            ]
            conv_padding = convolution.conv_explicit_padding(
                window=(window,), strides=(strides,), padding=padding, dilation=(dilation,)
            )[0]
            pad_left, pad_right = conv_padding
            anchor_range = dilate_window - pad_left - pad_right
            mid_point = anchor_range // 2
            anchor = pad_left + mid_point
        else:
            anchor = None

        in_paddings = jnp.array(in_paddings, dtype=jnp.bool)[None, :]
        ref_paddings = convolution.compute_conv_paddings(
            in_paddings,
            window=window,
            stride=strides,
            conv_padding=padding,
            dilation=dilation,
            anchor=anchor,
        )

        test_paddings = convolution.compute_conv_transpose_paddings(
            in_paddings,
            window=window,
            stride=strides,
            conv_padding=padding,
            dilation=dilation,
            anchor=anchor,
        )

        if ref_paddings.shape != test_paddings.shape:
            self.assertEqual(padding, "VALID")
            dilate_window = convolution.conv_dilate_window(window=(window,), dilation=(dilation,))[
                0
            ]
            pad_left = dilate_window - 1
            test_paddings = test_paddings[:, pad_left:-pad_left]

        assert_allclose(ref_paddings, test_paddings)

    CONVT_PARAMS = [
        (3, 1, "SAME", 1, [0, 1, 2, 2]),
        (3, 2, "SAME", 1, [0, 0, 0, 0, 1, 1, 2, 1]),
        (3, 3, "SAME", 1, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 4, "SAME", 1, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]),
        (3, 1, "SAME", 2, [1, 1, 1, 1]),
        (3, 2, "SAME", 2, [0, 0, 0, 1, 0, 2, 0, 2]),
        (3, 3, "SAME", 2, [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]),
        (3, 1, "VALID", 1, [0, 0, 1, 2, 2, 1]),
        (3, 2, "VALID", 1, [0, 0, 0, 0, 1, 1, 2, 1, 1]),
        (3, 3, "VALID", 1, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 4, "VALID", 1, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]),
        (3, 1, "VALID", 2, [0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 2, "VALID", 2, [0, 0, 0, 0, 1, 0, 2, 0, 2, 0, 1]),
        (3, 3, "VALID", 2, [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]),
        (3, 1, "CAUSAL", 1, [0, 0, 1, 2]),
        (3, 2, "CAUSAL", 1, [0, 0, 0, 0, 1, 1, 2, 1]),
        (3, 3, "CAUSAL", 1, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        (3, 4, "CAUSAL", 1, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]),
        (3, 1, "CAUSAL", 2, [0, 0, 1, 1]),
        (3, 2, "CAUSAL", 2, [0, 0, 0, 0, 1, 0, 2, 0]),
        (3, 3, "CAUSAL", 2, [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]),
    ]

    @parameterized.parameters(*CONVT_PARAMS)
    def test_conv1d_transpose_simple(self, window, strides, padding, dilation, expected):
        """Tests the cases in conv_transpose_explicit_padding() description."""
        input_dim, output_dim = 1, 1
        inputs = jnp.array([0, 0, 1, 1], dtype=jnp.float32)[None, :, None]
        cfg = convolution.Conv1DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        layer = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(dict(weight=(window, input_dim, output_dim)), shapes(layer_params))
        layer_params["weight"] = jnp.ones_like(layer_params["weight"])

        (outputs, paddings), _ = F(
            layer, inputs=dict(x=inputs), is_training=True, state=layer_params, prng_key=prng_key
        )
        out_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, out_shape)
        self.assertIsNone(paddings)
        expected = jnp.array(expected).astype(outputs.dtype)
        self.assertNestedEqual(outputs[0, :, 0], expected)

    @parameterized.parameters(*CONVT_PARAMS)
    def test_conv2d_transpose_simple(self, window, strides, padding, dilation, expected):
        """Tests the cases in conv_transpose_explicit_padding() description."""
        window = (window, 1)
        strides = (strides, 1)
        dilation = (dilation, 1)
        input_dim, output_dim = 1, 1
        inputs = jnp.array([0, 0, 1, 1], dtype=jnp.float32)[None, :, None, None]
        cfg = convolution.Conv2DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        layer = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(dict(weight=(*window, input_dim, output_dim)), shapes(layer_params))
        layer_params["weight"] = jnp.ones_like(layer_params["weight"])

        outputs, _ = F(
            layer, inputs=dict(x=inputs), is_training=True, state=layer_params, prng_key=prng_key
        )
        out_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, out_shape)
        expected = jnp.array(expected).astype(outputs.dtype)
        self.assertNestedEqual(outputs[0, :, 0, 0], expected)

    @parameterized.named_parameters(
        {
            "testcase_name": "2x2",
            "window": (2, 2),
            "strides": (1, 1),
            "padding": "VALID",
        },
        {
            "testcase_name": "2x2_S2",
            "window": (2, 2),
            "strides": (2, 2),
            "padding": "VALID",
        },
        {
            "testcase_name": "3x3_S2",
            "window": (3, 3),
            "strides": (2, 2),
            "padding": "VALID",
        },
    )
    def test_conv2d_transpose_against_pytorch(
        self,
        window: tuple[int, int],
        strides: tuple[int, int],
        padding: Union[str, tuple[int, int]],
    ):
        input_dim, output_dim = 4, 8
        if isinstance(padding, tuple):
            deconv_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            deconv_padding = padding
        cfg = Conv2DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=deconv_padding,
            transpose_kernel=True,
        )
        layer: Conv2DTranspose = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(
                weight=(window[0], window[1], output_dim, input_dim),
                bias=(output_dim,),
            ),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 10, 7, input_dim])
        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        if isinstance(padding, tuple):
            ref_padding = padding[0]
        elif isinstance(padding, str):
            ref_padding = padding.lower()
            if ref_padding == "valid":
                ref_padding = 0
        else:
            ref_padding = 0

        ref = torch.nn.ConvTranspose2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(3, 2, 0, 1), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(inputs.transpose(0, 3, 1, 2)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 1))
        # Tests output_shape.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, output_shape)

    @parameterized.parameters(*CONVT_PARAMS)
    def test_conv3d_transpose_simple(self, window, strides, padding, dilation, expected):
        """Tests the cases in conv_transpose_explicit_padding() description."""
        window = (window, 1, 1)
        strides = (strides, 1, 1)
        dilation = (dilation, 1, 1)
        input_dim, output_dim = 1, 1
        inputs = jnp.array([0, 0, 1, 1], dtype=jnp.float32)[None, :, None, None, None]
        cfg = convolution.Conv3DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        layer = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(dict(weight=(*window, input_dim, output_dim)), shapes(layer_params))
        layer_params["weight"] = jnp.ones_like(layer_params["weight"])

        outputs, _ = F(
            layer, inputs=dict(x=inputs), is_training=True, state=layer_params, prng_key=prng_key
        )
        out_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, out_shape)
        expected = jnp.array(expected).astype(outputs.dtype)
        self.assertNestedEqual(outputs[0, :, 0, 0, 0], expected)

    @parameterized.product(window=(1, 3, 5), padding=("SAME", "VALID", "CAUSAL"), dilation=(1, 2))
    def test_conv1d_transpose_against_conv1d(self, window, padding, dilation):
        # Conv1D and Conv1DTranspose are same when window_stride=1
        # (stride of Conv1D) and lhs_dilation=1 (stride of Conv1DTranspose).
        input_dim, output_dim = 4, 6
        ref_cfg = Conv1D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
        )
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = convolution.Conv1DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        ref_states = ref_layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 17, input_dim])
        # Compute layer outputs.
        ref_outputs, _ = F(
            ref_layer, inputs=dict(x=inputs), is_training=True, state=ref_states, prng_key=prng_key
        )

        (test_outputs, _), _ = F(
            test_layer, inputs=dict(x=inputs), is_training=True, state=ref_states, prng_key=prng_key
        )
        if ref_outputs.shape != test_outputs.shape:
            self.assertEqual(padding, "VALID")
            dilate_window = convolution.conv_dilate_window(window=(window,), dilation=(dilation,))[
                0
            ]
            pad_left = dilate_window - 1
            test_outputs = test_outputs[:, pad_left:-pad_left]
        assert_allclose(ref_outputs, test_outputs)

    @parameterized.product(window=(1, 3, 5), padding=("SAME", "VALID", "CAUSAL"), dilation=(1, 2))
    def test_conv2d_transpose_against_conv2d(self, window, padding, dilation):
        # Conv2D and Conv2DTranspose are same when window_stride=1
        # (stride of Conv2D) and lhs_dilation=1 (stride of Conv2DTranspose).
        window = (window, window)
        dilation = (dilation, dilation)
        input_dim, output_dim = 4, 6
        ref_cfg = Conv2D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
        )
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = convolution.Conv2DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
            transpose_kernel=False,
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        ref_states = ref_layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        width, height = 12, 13
        inputs = jax.random.normal(input_key, [2, width, height, input_dim])
        # Compute layer outputs.
        ref_outputs, _ = F(
            ref_layer,
            inputs=dict(x=inputs),
            is_training=True,
            state=ref_states,
            prng_key=prng_key,
        )

        test_outputs, _ = F(
            test_layer,
            inputs=dict(x=inputs),
            is_training=True,
            state=ref_states,
            prng_key=prng_key,
        )
        if ref_outputs.shape != test_outputs.shape:
            self.assertEqual(padding, "VALID")
            dilate_window = convolution.conv_dilate_window(window=window, dilation=dilation)
            pad_left = tuple(w - 1 for w in dilate_window)
            test_outputs = test_outputs[:, pad_left[0] : -pad_left[0], pad_left[1] : -pad_left[1]]

        assert_allclose(ref_outputs, test_outputs)

    @parameterized.product(window=(1, 3, 5), padding=("SAME", "VALID", "CAUSAL"), dilation=(1, 2))
    def test_conv2d_transpose_against_conv2d_with_paddings(self, window, padding, dilation):
        # Conv2DWith1DPadding and Conv2DTransposeWith1DPadding are same when window_stride=1
        # (stride of Conv2D) and lhs_dilation=1 (stride of Conv2DTranspose).
        window = (window, window)
        dilation = (dilation, dilation)
        input_dim, output_dim = 4, 6
        if padding == "VALID":
            # TODO(dhwang2,ruoming): Currently, anchor is pad_left but it should be the midpoint
            # between [pad_left, pad_right). Otherwise, the consistency of VALID padding is broken.
            # For reference, the midpoint in SAME and CAUSAL is left_pad.
            strides = (1, 1)
            dilate_window = convolution.conv_dilate_window(window=window, dilation=dilation)[0]
            conv_padding = convolution.conv_explicit_padding(
                window=window, strides=strides, padding=padding, dilation=dilation
            )
            pad_left, pad_right = conv_padding[0]
            anchor_range = dilate_window - pad_left - pad_right
            mid_point = anchor_range // 2
            anchor = pad_left + mid_point
        else:
            anchor = None

        ref_cfg = Conv2DWith1DPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
            anchor=anchor,
        )
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = convolution.Conv2DTransposeWith1DPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
            anchor=anchor,
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        ref_states = ref_layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        width, height = 12, 13
        inputs = jax.random.normal(input_key, [2, width, height, input_dim])
        paddings = jnp.zeros([2, width], dtype=inputs.dtype).at[:, -2:].set(1)
        # Compute layer outputs.
        (ref_outputs, ref_paddings), _ = F(
            ref_layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=ref_states,
            prng_key=prng_key,
        )

        (test_outputs, test_paddings), _ = F(
            test_layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=ref_states,
            prng_key=prng_key,
        )
        if ref_outputs.shape != test_outputs.shape:
            self.assertEqual(padding, "VALID")
            dilate_window = convolution.conv_dilate_window(window=window, dilation=dilation)
            pad_left = tuple(w - 1 for w in dilate_window)
            test_outputs = test_outputs[:, pad_left[0] : -pad_left[0], pad_left[1] : -pad_left[1]]
            test_paddings = test_paddings[:, pad_left[0] : -pad_left[0]]

        assert_allclose(ref_outputs, test_outputs)
        assert_allclose(ref_paddings, test_paddings)

    @parameterized.product(window=(1, 3, 5), padding=("SAME", "VALID", "CAUSAL"), dilation=(1, 2))
    def test_conv3d_transpose_against_conv3d(self, window, padding, dilation):
        # Conv3D and Conv3DTranspose are same when window_stride=1
        # (stride of Conv3D) and lhs_dilation=1 (stride of Conv3DTranspose).
        window = (window, window, window)
        dilation = (dilation, dilation, dilation)
        input_dim, output_dim = 4, 6
        ref_cfg = Conv3D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
        )
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = convolution.Conv3DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            padding=padding,
            dilation=dilation,
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        ref_states = ref_layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        width, height, depth = 9, 8, 7
        inputs = jax.random.normal(input_key, [2, width, height, depth, input_dim])
        # Compute layer outputs.
        ref_outputs, _ = F(
            ref_layer,
            inputs=dict(x=inputs),
            is_training=True,
            state=ref_states,
            prng_key=prng_key,
        )

        test_outputs, _ = F(
            test_layer,
            inputs=dict(x=inputs),
            is_training=True,
            state=ref_states,
            prng_key=prng_key,
        )
        if ref_outputs.shape != test_outputs.shape:
            self.assertEqual(padding, "VALID")
            dilate_window = convolution.conv_dilate_window(window=window, dilation=dilation)
            pad_left = tuple(w - 1 for w in dilate_window)
            test_outputs = test_outputs[
                :,
                pad_left[0] : -pad_left[0],
                pad_left[1] : -pad_left[1],
                pad_left[2] : -pad_left[2],
            ]

        assert_allclose(ref_outputs, test_outputs)

    @parameterized.product(
        window=(1, 3, 5),
        strides=(1, 2),
        padding=("SAME", "VALID", "CAUSAL"),
        dilation=(1, 2),
        anchor=(None, 1),
    )
    def test_conv1d_transpose_against_conv2d_transpose_with_1d_padding(
        self,
        window,
        strides,
        padding: ConvPaddingType,
        dilation,
        anchor,
    ):
        if anchor is not None:
            dilate_window = convolution.conv_dilate_window(window=(window,), dilation=(dilation,))[
                0
            ]
            anchor = dilate_window - 1

        input_dim, output_dim = 4, 6
        ref_cfg = convolution.Conv2DTransposeWith1DPadding.default_config().set(
            name="ref",
            input_dim=input_dim,
            output_dim=output_dim,
            window=(window, 1),
            strides=(strides, 1),
            padding=padding,
            dilation=(dilation, 1),
            anchor=anchor,
        )
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = convolution.Conv1DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=padding,
            dilation=dilation,
            anchor=anchor,
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        state = ref_layer.initialize_parameters_recursively(init_key)
        test_state = dict(
            bias=state["bias"], weight=einops.rearrange(state["weight"], "t 1 i o -> t i o")
        )

        # Generate a batch of 10 input sequences.
        batch_size, max_seq_len = 10, 10

        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [batch_size, max_seq_len, input_dim])
        # The 10 sequences have length 1 to 10.
        paddings = jnp.triu(jnp.ones((batch_size, max_seq_len)), k=1)

        (test_outputs, test_paddings), _ = F(
            test_layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=test_state,
            prng_key=prng_key,
        )

        inputs = einops.rearrange(inputs, "b t i -> b t 1 i")
        (ref_outputs, ref_paddings), _ = F(
            ref_layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=state,
            prng_key=prng_key,
        )
        ref_outputs = einops.rearrange(ref_outputs, "b t 1 o -> b t o")

        assert_allclose(ref_paddings, test_paddings)
        assert_allclose(ref_outputs, test_outputs)


class StackOverTimeTest(TestCase):
    @parameterized.parameters(
        (
            2,
            (0, 0),
            [[[1, 1, 2, 2], [3, 3, 4, 4]], [[7, 7, 8, 8], [0, 0, 0, 0]]],
            [[0, 0], [0, 1]],
        ),
        (
            3,
            (0, 0),
            [[[1, 1, 2, 2, 3, 3]], [[7, 7, 8, 8, 0, 0]]],
            [[0], [0]],
        ),
        (
            3,
            (2, 0),
            [[[0, 0, 0, 0, 1, 1], [2, 2, 3, 3, 4, 4]], [[0, 0, 0, 0, 7, 7], [0, 0, 0, 0, 0, 0]]],
            [[0, 0], [0, 1]],
        ),
    )
    def test_stack_over_time(self, stride, pad, expected_outputs, expected_output_paddings):
        # Input shape [2, 5, 2].
        inputs = jnp.array(
            [[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0]]],
            dtype=jnp.float32,
        )
        paddings = jnp.array([[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]])
        layer: StackOverTime = (
            StackOverTime.default_config()
            .set(
                name="test",
                stride=stride,
                padding=pad,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        (outputs, output_paddings), _ = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=False,
            state=layer_params,
            prng_key=jax.random.PRNGKey(5),
        )
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, output_shape)
        assert_allclose(jnp.array(expected_outputs, dtype=jnp.float32), outputs)
        assert_allclose(jnp.array(expected_output_paddings, dtype=jnp.int32), output_paddings)

    def test_stack_over_time_data_change(self):
        """Tests that the stacked outputs is masked with the output paddings."""
        np.random.seed(500)
        inputs = np.random.normal(size=[2, 21, 16])
        paddings = np.ones([2, 21], dtype=np.float32)
        paddings[0, :9] = 0
        paddings[1, :14] = 0
        inputs = inputs * (1 - paddings)[:, :, None]

        layer: StackOverTime = (
            StackOverTime.default_config()
            .set(
                name="test",
                stride=2,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        (outputs, output_paddings), _ = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=False,
            state=layer_params,
            prng_key=jax.random.PRNGKey(5),
        )
        output_shape = layer.output_shape(input_shape=inputs.shape)
        assert_allclose(outputs.shape, output_shape)
        assert_allclose(np.array([5, 7], dtype=np.float32), np.sum(1 - output_paddings, axis=1))
        assert_allclose(np.sum(inputs**2, (1, 2)), np.sum(outputs**2, (1, 2)))

    @parameterized.product(stride=(2, 3, 4), pad=("VALID", "SAME", "CAUSAL"))
    def test_stack_consistent_outputs(self, stride, pad):
        """Tests that StackOverTime has consistent outputs under different padding lengths."""
        batch_size, input_dim = 2, 1
        input_length = 7
        layer: StackOverTime = (
            StackOverTime.default_config()
            .set(
                name="test",
                stride=stride,
                padding=pad,
            )
            .instantiate(parent=None)
        )
        expected_output_length = layer.output_shape(input_shape=[1, input_length, 1])[1]
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        for ll in range(4, 11):
            # Batch with another example of length ll.
            length = max(input_length, ll)
            inputs = jnp.ones([batch_size, length, input_dim])
            paddings = jnp.arange(length)[None, :] >= jnp.array([input_length, ll])[:, None]
            (outputs, output_paddings), _ = F(
                layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                is_training=False,
                state=layer_params,
                prng_key=jax.random.PRNGKey(5),
            )
            output_shape = layer.output_shape(input_shape=inputs.shape)
            assert_allclose(outputs.shape, output_shape)
            if pad != "VALID":  # VALID doesn't preserve length.
                self.assertEqual(expected_output_length, np.sum(1 - output_paddings, axis=1)[0])

    @parameterized.parameters(((0, 1), (0, 0)), ((1, 1), (3, 0)), ((1, 1), (0, 3)))
    def test_stack_vs_conv2d_output_len_match(self, conv_padding, stack_padding):
        # Note that to get the same output length, we need to pad the sequence differently
        # for convolution and stacking layer.
        for audio_seq_len in [16000, 16160, 16320, 16480, 16640, 16800, 16960, 17120]:
            sampling_rate, window_size_ms, window_step_ms = 16000, 25, 10
            window_size = window_size_ms * sampling_rate // 1000
            window_step = window_step_ms * sampling_rate // 1000
            seq_len = max(audio_seq_len - window_size, 0) // window_step + 1
            conv_layer: Conv2DWith1DPadding = (
                Conv2DWith1DPadding.default_config()
                .set(
                    name="test_conv",
                    input_dim=3,
                    output_dim=3,
                    window=(3, 3),
                    strides=(2, 2),
                    padding=(conv_padding, (0, 1)),
                )
                .instantiate(parent=None)
            )
            stack_layer: StackOverTime = (
                StackOverTime.default_config()
                .set(name="test_stack", stride=4, padding=stack_padding)
                .instantiate(parent=None)
            )
            # Computes downsampler output shape.
            down_sample_shape1 = conv_layer.output_shape(input_shape=[None, seq_len, 80, 3])
            down_sample_shape2 = conv_layer.output_shape(input_shape=down_sample_shape1)

            # Computes stack output shape.
            stack_shape = stack_layer.output_shape(input_shape=[None, seq_len, 80])
            # Tests that the sequence length dimension matches.
            self.assertEqual(down_sample_shape2[1], stack_shape[1])


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

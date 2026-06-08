# Copyright © 2026 Apple Inc.

"""Tests for streaming convolution layers."""

import random

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.audio.streaming import convolutions as streaming_conv
from axlearn.audio.streaming.test_utils import check_segment_pad_outputs, segment_inputs
from axlearn.common import convolution, ein_ops
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import safe_not


def _get_inputs(shape):
    prng_key = jax.random.PRNGKey(0)
    inputs = jax.random.uniform(prng_key, shape=shape)
    length = jax.random.randint(prng_key, shape=shape[:1], minval=1, maxval=shape[1])
    paddings = jnp.arange(shape[1])[None, :] > length[:, None]
    return inputs, paddings


class CausalConv2DTest(TestCase):
    @parameterized.named_parameters(
        ("1x1", (1, 1), (1, 1), [1, 2, 3, 4, 5, 0]),
        ("2x2", (2, 2), (1, 1), [1, 3, 5, 7, 9, 0]),
        ("2x2_S2", (2, 2), (2, 2), [3, 7, 5]),
        ("3x3", (3, 3), (1, 1), [1, 3, 6, 9, 12, 0]),
        ("3x3_S2", (3, 3), (2, 2), [3, 9, 9]),
    )
    def test_forward(self, window, strides, expected):
        input_dim, output_dim = 1, 1
        cfg = streaming_conv.CausalConv2DWith1DPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding="CAUSAL",
            bias=False,
        )
        layer: streaming_conv.CausalConv2DWith1DPadding = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window[0], window[1], input_dim, output_dim)),
            jax.tree.map(jnp.shape, layer_params),
        )
        layer_params = jax.tree.map(jnp.ones_like, layer_params)

        # Generate input sequences.
        max_seq_len = 6
        inputs = jnp.arange(max_seq_len).astype(jnp.float32) + 1
        inputs = ein_ops.rearrange(inputs, "t -> 1 t 1 1")
        paddings = jnp.zeros((1, max_seq_len), dtype=jnp.bool)
        paddings = paddings.at[0, -1].set(1)

        # Compute layer outputs.
        (outputs, out_paddings), _ = F(
            layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        outputs = ein_ops.rearrange(outputs, "1 t 1 1 -> t")

        expected = jnp.array(expected, dtype=outputs.dtype)
        assert_allclose(outputs, expected)
        assert_allclose(out_paddings, paddings[:, : expected.shape[0]])

    @parameterized.named_parameters(
        ("1x1", (1, 1), (1, 1)),
        ("2x2", (2, 2), (1, 1)),
        ("2x2_S2", (2, 2), (2, 2)),
        ("3x3", (3, 3), (1, 1)),
        ("3x3_S2", (3, 3), (2, 2)),
    )
    def test_extend_step(self, window, strides):
        input_dim, output_dim = 4, 8
        cfg = streaming_conv.CausalConv2DWith1DPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding="CAUSAL",
            bias=False,
        )
        layer: streaming_conv.CausalConv2DWith1DPadding = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window[0], window[1], input_dim, output_dim)),
            jax.tree.map(jnp.shape, layer_params),
        )

        # Generate input sequences.
        batch, seq_len, freq_dim = 2, 10, 8
        shape = (batch, seq_len, freq_dim, input_dim)
        x, paddings = _get_inputs(shape)

        # Compute layer outputs.
        (fwd_output, fwd_paddings), _ = F(
            layer,
            inputs=dict(x=x, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(batch_size=batch, feature_dim=freq_dim, dtype=x.dtype)
        in_stride = layer.in_stride(layer.config)
        step_sizes = (in_stride, in_stride * 2, in_stride * 3)
        step_inputs = dict(x=x, paddings=paddings)
        step_outputs = []
        step_paddings = []
        i = 0
        while i < seq_len:
            step_size = random.choice(step_sizes)
            # pylint: disable-next=cell-var-from-loop
            step_x = jax.tree.map(lambda x: x[:, i : i + step_size], step_inputs)
            i += step_size
            (cached_states, step_outs), _ = F(
                layer,
                prng_key=jax.random.PRNGKey(0),
                state=layer_params,
                inputs=dict(cached_states=cached_states, input_data=step_x),
                is_training=True,
                method="extend_step",
            )
            step_outputs.append(step_outs["x"])
            step_paddings.append(step_outs["paddings"])
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        step_paddings = jnp.concatenate(step_paddings, axis=1)
        assert_allclose(step_outputs, fwd_output)
        assert_allclose(step_paddings, fwd_paddings)

    @parameterized.parameters((3, 1, 2), (3, 2, 1), (4, 2, 2), (5, 2, 3))
    def test_segment_pad(self, window, strides, expected_segment_pad):
        input_dim, output_dim = 4, 8
        cfg = streaming_conv.CausalConv2DWith1DPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=(window, window),
            strides=(strides, 1),
        )
        layer = cfg.instantiate(parent=None)
        segment_pad = streaming_conv.CausalConv2DWith1DPadding.segment_pad(cfg)
        self.assertEqual(segment_pad, expected_segment_pad)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Generate input sequences.
        prng_key, data_key = jax.random.split(prng_key)
        inputs, segment_ids = segment_inputs(
            data_key, segment_pad=segment_pad, stride=strides, suffix_shape=(input_dim, 4)
        )

        # Compute layer outputs.
        (outputs, out_paddings), _ = F(
            layer,
            inputs=dict(x=inputs, paddings=segment_ids == 0),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        self.assertEqual(outputs.shape[:2], out_paddings.shape)
        check_segment_pad_outputs(outputs, safe_not(out_paddings).astype(jnp.int32))


class CausalConv1DTest(TestCase):
    @parameterized.named_parameters(
        ("w1s1", 1, 1),
        ("w2s1", 2, 1),
        ("w2s2", 2, 2),
        ("w3s1", 3, 1),
        ("w3s2", 3, 2),
        ("w4s1", 4, 1),
        ("w4s2", 4, 2),
    )
    def test_extend_step(self, window, strides):
        input_dim, output_dim = 4, 6
        cfg = streaming_conv.CausalConv1D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            bias=False,
        )
        layer = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, input_dim, output_dim)),
            jax.tree.map(jnp.shape, layer_params),
        )

        # Generate input sequences.
        batch, seq_len = 2, 10
        shape = (batch, seq_len, input_dim)
        x, _ = _get_inputs(shape)

        # Compute layer outputs.
        fwd_output, _ = F(
            layer,
            inputs=(x,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(batch_size=batch)
        in_stride = layer.in_stride(layer.config)
        step_sizes = (in_stride, in_stride * 2, in_stride * 3)
        step_outputs = []
        i = 0
        while i < seq_len:
            step_size = random.choice(step_sizes)
            step_x = x[:, i : i + step_size]
            i += step_size
            (cached_states, step_output), _ = F(
                layer,
                prng_key=jax.random.PRNGKey(0),
                state=layer_params,
                inputs=dict(cached_states=cached_states, input_data=step_x),
                is_training=True,
                method="extend_step",
            )
            step_outputs.append(step_output)
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        assert_allclose(step_outputs, fwd_output)

    @parameterized.named_parameters(
        ("w1s1", 1, 1),
        ("w2s1", 2, 1),
        ("w2s2", 2, 2),
        ("w3s1", 3, 1),
        ("w3s2", 3, 2),
        ("w4s1", 4, 1),
        ("w4s2", 4, 2),
    )
    def test_depthwise_conv_extend_step(self, window, strides):
        input_dim = 4
        cfg = streaming_conv.CausalConv1D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=input_dim,
            num_input_dim_groups=input_dim,
            window=window,
            strides=strides,
            bias=False,
        )
        layer = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, 1, input_dim)),
            jax.tree.map(jnp.shape, layer_params),
        )

        # Generate input sequences.
        batch, seq_len = 2, 10
        shape = (batch, seq_len, input_dim)
        x, _ = _get_inputs(shape)

        # Compute layer outputs.
        fwd_output, _ = F(
            layer,
            inputs=(x,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(batch_size=batch)
        in_stride = layer.in_stride(layer.config)
        step_sizes = (in_stride, in_stride * 2, in_stride * 3)
        step_outputs = []
        i = 0
        while i < seq_len:
            step_size = random.choice(step_sizes)
            step_x = x[:, i : i + step_size]
            i += step_size
            (cached_states, step_output), _ = F(
                layer,
                prng_key=jax.random.PRNGKey(0),
                state=layer_params,
                inputs=dict(cached_states=cached_states, input_data=step_x),
                is_training=True,
                method="extend_step",
            )
            step_outputs.append(step_output)
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        assert_allclose(step_outputs, fwd_output)


class CausalConv1DWithPaddingTest(TestCase):
    @parameterized.named_parameters(
        ("w1s1", 1, 1),
        ("w2s1", 2, 1),
        ("w2s2", 2, 2),
        ("w3s1", 3, 1),
        ("w3s2", 3, 2),
        ("w4s1", 4, 1),
        ("w4s2", 4, 2),
    )
    # pylint: disable-next=no-self-use
    def test_forward(self, window, strides):
        input_dim, output_dim = 4, 4
        layer_kwargs = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding="CAUSAL",
            bias=False,
        )

        ref_cfg = convolution.Conv1DWithPadding.default_config().set(name="test", **layer_kwargs)
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = streaming_conv.CausalConv1DWithPadding.default_config().set(
            name="test", **layer_kwargs
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = test_layer.initialize_parameters_recursively(init_key)

        # Generate input sequences.
        batch, seq_len = 2, 10
        shape = (batch, seq_len, input_dim)
        x, paddings = _get_inputs(shape)

        # Compute layer outputs.
        (ref_output, ref_paddings), _ = F(
            ref_layer,
            inputs=dict(x=x, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        (test_output, test_paddings), _ = F(
            test_layer,
            inputs=dict(x=x, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        assert_allclose(ref_output, test_output)
        assert_allclose(ref_paddings, test_paddings)

    @parameterized.named_parameters(
        ("w1s1", 1, 1),
        ("w2s1", 2, 1),
        ("w2s2", 2, 2),
        ("w3s1", 3, 1),
        ("w3s2", 3, 2),
        ("w4s1", 4, 1),
        ("w4s2", 4, 2),
    )
    # pylint: disable-next=no-self-use
    def test_extend_step(self, window, strides):
        input_dim, output_dim = 4, 4
        layer_kwargs = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding="CAUSAL",
            bias=False,
        )

        cfg = streaming_conv.CausalConv1DWithPadding.default_config().set(
            name="test", **layer_kwargs
        )
        layer = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Generate input sequences.
        batch, seq_len = 2, 10
        shape = (batch, seq_len, input_dim)
        x, paddings = _get_inputs(shape)

        # Compute layer outputs.
        (fwd_output, fwd_paddings), _ = F(
            layer,
            inputs=dict(x=x, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        cached_states = layer.init_states(batch_size=batch, dtype=x.dtype)
        in_stride = layer.in_stride(layer.config)
        step_sizes = (in_stride, in_stride * 2, in_stride * 3)
        step_inputs = dict(x=x, paddings=paddings)
        step_outputs = []
        step_paddings = []
        i = 0
        while i < seq_len:
            step_size = random.choice(step_sizes)
            # pylint: disable-next=cell-var-from-loop
            step_x = jax.tree.map(lambda x: x[:, i : i + step_size], step_inputs)
            i += step_size
            (cached_states, step_outs), _ = F(
                layer,
                prng_key=jax.random.PRNGKey(0),
                state=layer_params,
                inputs=dict(cached_states=cached_states, input_data=step_x),
                is_training=True,
                method="extend_step",
            )
            step_outputs.append(step_outs["x"])
            step_paddings.append(step_outs["paddings"])
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        step_paddings = jnp.concatenate(step_paddings, axis=1)
        assert_allclose(step_outputs, fwd_output)
        assert_allclose(step_paddings, fwd_paddings)

    @parameterized.parameters((3, 1, 2), (3, 2, 1), (4, 2, 2), (5, 2, 3))
    def test_segment_pad(self, window, strides, expected_segment_pad):
        input_dim, output_dim = 4, 8
        cfg = streaming_conv.CausalConv1DWithPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
        )
        layer = cfg.instantiate(parent=None)
        segment_pad = streaming_conv.CausalConv1DWithPadding.segment_pad(cfg)
        self.assertEqual(segment_pad, expected_segment_pad)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Generate input sequences.
        prng_key, data_key = jax.random.split(prng_key)
        inputs, segment_ids = segment_inputs(
            data_key, segment_pad=segment_pad, stride=strides, suffix_shape=(input_dim,)
        )

        # Compute layer outputs.
        (outputs, out_paddings), _ = F(
            layer,
            inputs=dict(x=inputs, paddings=segment_ids == 0),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        self.assertEqual(outputs.shape[:2], out_paddings.shape)
        check_segment_pad_outputs(outputs, safe_not(out_paddings).astype(jnp.int32))


class CausalConv1DTransposeTest(TestCase):
    @parameterized.product(window=[1, 2, 3, 4, 5], strides=[1, 2, 3], dilation=[1, 2, 3])
    def test_extend_step(self, window, strides, dilation):
        input_dim, output_dim = 4, 6
        cfg = streaming_conv.CausalConv1DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            dilation=dilation,
            bias=False,
        )
        layer = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, input_dim, output_dim)),
            jax.tree.map(jnp.shape, layer_params),
        )

        # Generate input sequences.
        batch, seq_len = 2, 10
        shape = (batch, seq_len, input_dim)
        x, paddings = _get_inputs(shape)
        inputs = dict(x=x, paddings=paddings)

        # Compute layer outputs.
        in_stride = layer.in_stride(layer.config)
        out_stride = layer.out_stride(layer.config)
        (fwd_output, fwd_paddings), _ = F(
            layer,
            inputs=inputs,
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        self.assertEqual(fwd_output.shape[1], x.shape[1] * out_stride)
        self.assertEqual(list(fwd_output.shape), layer.output_shape(input_shape=x.shape))

        cached_states = layer.init_states(batch_size=batch)
        step_sizes = (in_stride, in_stride * 2, in_stride * 3)
        step_outputs = []
        step_paddings = []
        i = 0
        while i < seq_len:
            step_size = random.choice(step_sizes)
            step_inputs = jax.tree.map(lambda x: x[:, i : i + step_size], inputs)
            i += step_size
            (cached_states, step_output), _ = F(
                layer,
                prng_key=jax.random.PRNGKey(0),
                state=layer_params,
                inputs=dict(cached_states=cached_states, input_data=step_inputs),
                is_training=True,
                method="extend_step",
            )
            step_outputs.append(step_output["x"])
            step_paddings.append(step_output["paddings"])
        step_paddings = jnp.concatenate(step_paddings, axis=1)
        self.assertNestedEqual(step_paddings, fwd_paddings)
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        assert_allclose(step_outputs, fwd_output)

    @parameterized.product(window=[1, 2, 3], strides=[1, 2], dilation=[1, 2])
    def test_extend_step_wo_paddings(self, window, strides, dilation):
        input_dim, output_dim = 4, 6
        cfg = streaming_conv.CausalConv1DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            dilation=dilation,
            bias=False,
        )
        layer = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, input_dim, output_dim)),
            jax.tree.map(jnp.shape, layer_params),
        )

        # Generate input sequences.
        batch, seq_len = 2, 10
        shape = (batch, seq_len, input_dim)
        x, _ = _get_inputs(shape)
        inputs = dict(x=x)

        # Compute layer outputs.
        in_stride = layer.in_stride(layer.config)
        out_stride = layer.out_stride(layer.config)
        (fwd_output, _), _ = F(
            layer,
            inputs=inputs,
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        self.assertEqual(fwd_output.shape[1], x.shape[1] * out_stride)
        self.assertEqual(list(fwd_output.shape), layer.output_shape(input_shape=x.shape))

        cached_states = layer.init_states(batch_size=batch)
        step_sizes = (in_stride, in_stride * 2, in_stride * 3)
        step_outputs = []
        i = 0
        while i < seq_len:
            step_size = random.choice(step_sizes)
            step_inputs = jax.tree.map(lambda x: x[:, i : i + step_size], inputs)
            i += step_size
            (cached_states, step_output), _ = F(
                layer,
                prng_key=jax.random.PRNGKey(0),
                state=layer_params,
                inputs=dict(cached_states=cached_states, input_data=step_inputs),
                is_training=True,
                method="extend_step",
            )
            step_outputs.append(step_output["x"])
        step_outputs = jnp.concatenate(step_outputs, axis=1)
        assert_allclose(step_outputs, fwd_output)

    @parameterized.parameters((3, 2, 1), (4, 2, 1), (5, 2, 2), (5, 3, 1))
    def test_segment_pad(self, window, strides, expected_segment_pad):
        input_dim, output_dim = 4, 8
        cfg = streaming_conv.CausalConv1DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
        )
        layer = cfg.instantiate(parent=None)
        segment_pad = streaming_conv.CausalConv1DTranspose.segment_pad(cfg)
        self.assertEqual(segment_pad, expected_segment_pad)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Generate input sequences. For transpose, in_stride=1.
        prng_key, data_key = jax.random.split(prng_key)
        inputs, segment_ids = segment_inputs(
            data_key, segment_pad=segment_pad, stride=1, suffix_shape=(input_dim,)
        )

        # Compute layer outputs.
        (outputs, out_paddings), _ = F(
            layer,
            inputs=dict(x=inputs, paddings=segment_ids == 0),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )
        self.assertEqual(outputs.shape[:2], out_paddings.shape)
        check_segment_pad_outputs(outputs, safe_not(out_paddings).astype(jnp.int32))


if __name__ == "__main__":
    absltest.main()

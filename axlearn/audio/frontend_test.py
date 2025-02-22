# Copyright © 2023 Apple Inc.
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests audio frontends."""

import functools
import math

import jax
import jax.numpy as jnp
import pytest
import tensorflow as tf
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from axlearn.audio.frontend import LogMelFrontend, normalize_by_mean_std
from axlearn.audio.frontend_utils import (
    linear_to_log_spectrogram,
    magnitude_spectrogram,
    ms_to_samples,
    sharded_fft,
)
from axlearn.audio.frontend_utils_test import (
    _ref_framer,
    _ref_log_mel_spectrogram,
    _ref_pre_emphasis,
)
from axlearn.audio.test_utils import fake_audio
from axlearn.common.config import config_for_function
from axlearn.common.module import functional as F
from axlearn.common.utils import Tensor


class LogMelFrontendTest(parameterized.TestCase, tf.test.TestCase):
    """Tests LogMelFrontend."""

    @parameterized.parameters(
        dict(frame_size_ms=3.90625, hop_size_ms=1),
        dict(frame_size_ms=1, hop_size_ms=0.15625),
    )
    def test_instantiate(self, frame_size_ms, hop_size_ms):
        num_filters, sample_rate = 80, 16_000
        cfg = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            mel_floor=1.0,
        )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            cfg.set(name="test").instantiate(parent=None)

    def _jit_forward(self, layer, inputs, paddings):
        @jax.jit
        def jit_forward(inputs, paddings):
            return F(
                layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state={},
            )

        test_outputs, _ = jit_forward(inputs, paddings)
        return test_outputs

    @parameterized.product(
        [
            dict(frame_size_ms=25, hop_size_ms=10),
            dict(frame_size_ms=25, hop_size_ms=25),
            dict(frame_size_ms=32, hop_size_ms=10),
            dict(frame_size_ms=31.9375, hop_size_ms=10),
        ],
        pre_emphasis=[True, False],
    )
    @pytest.mark.fp64
    def test_against_ref(self, frame_size_ms, hop_size_ms, pre_emphasis):
        sample_rate, batch_size, max_seconds = 16_000, 4, 13
        num_filters = 80

        # Construct fake inputs.
        inputs, paddings = fake_audio(
            prng_key=jax.random.PRNGKey(123),
            batch_size=batch_size,
            seq_len=max_seconds * sample_rate,
            dtype=jnp.float64,
        )

        # Compute ref outputs.
        ref_outputs, ref_paddings = _ref_frontend(
            inputs=inputs,
            paddings=paddings,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            coeff=0.97,
            num_filters=num_filters,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7600.0,
            mel_floor=1.0,
            pre_emphasis=pre_emphasis,
        )

        # Compute test outputs.
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            mel_floor=1.0,
        )
        if not pre_emphasis:
            cfg.pre_emphasis = None
        layer: LogMelFrontend = cfg.set(name="test").instantiate(parent=None)
        test_outputs = self._jit_forward(layer, inputs, paddings)

        # Only compare the non-padding outputs.
        ref_outputs = ref_outputs * (1 - tf.cast(ref_paddings, ref_outputs.dtype))[..., None]
        self.assertAllClose(ref_outputs[..., None], test_outputs["outputs"])
        self.assertAllClose(ref_paddings, test_outputs["paddings"])

        # Check that output shape is consistent.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertSequenceEqual(test_outputs["outputs"].shape, output_shape)

    @pytest.mark.fp64
    def test_normalization(self):
        sample_rate, batch_size, max_seconds = 16_000, 4, 13
        num_filters, frame_size_ms, hop_size_ms = 80, 25, 10
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            mel_floor=1.0,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        def compute_mean_std(outputs):
            # Compute mean/std of features. [batch, num_frames, num_filters, 1] -> [num_filters].
            non_padding = jnp.sum(1 - outputs["paddings"])
            mean = jnp.sum(outputs["outputs"], axis=(0, 1, -1)) / non_padding
            mean_square = jnp.sum(outputs["outputs"] ** 2, axis=(0, 1, -1)) / non_padding
            return mean, jnp.sqrt(mean_square - mean**2)

        inputs, paddings = fake_audio(
            prng_key=jax.random.PRNGKey(123),
            batch_size=batch_size,
            seq_len=max_seconds * sample_rate,
            dtype=jnp.float64,
        )
        test_outputs = self._jit_forward(layer, inputs, paddings)

        # Compute again with scaling by pre-computed mean and std.
        mean, std = compute_mean_std(test_outputs)
        cfg.output_transformation = config_for_function(
            lambda: functools.partial(normalize_by_mean_std, mean=mean.tolist(), std=std.tolist())
        )
        layer = cfg.set(name="test").instantiate(parent=None)
        test_outputs = self._jit_forward(layer, inputs, paddings)

        # Check outputs.
        mean, std = compute_mean_std(test_outputs)
        self.assertTrue(jnp.allclose(mean, 0, atol=1e-5))
        self.assertTrue(jnp.allclose(std, 1.0, atol=1e-5))

    def test_output_dim(self):
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=80,
            sample_rate=16_000,
            frame_size_ms=25,
            hop_size_ms=10,
            output_dim=2,
            mel_floor=1.0,
        )
        with self.assertRaisesRegex(ValueError, "output_dim"):
            cfg.set(name="test").instantiate(parent=None)

    def test_small_input(self):
        sample_rate, batch_size, max_seconds = 16_000, 4, 13
        num_filters = 80

        # Construct fake inputs.
        inputs, paddings = fake_audio(
            prng_key=jax.random.PRNGKey(123),
            batch_size=batch_size,
            seq_len=max_seconds * sample_rate,
            dtype=jnp.float64,
            scale=1.0,
        )

        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=25,
            hop_size_ms=10,
            mel_floor=1.0,
        )
        layer: LogMelFrontend = cfg.set(name="test").instantiate(parent=None)
        test_outputs = self._jit_forward(layer, inputs, paddings)
        num_zero_elements = jnp.size(jnp.where(test_outputs["outputs"] == 0)[0])
        total_elements = jnp.size(test_outputs["outputs"])
        print(f"percentage of zero elements: {num_zero_elements / total_elements * 100}%")
        # When the input signal is [-0.1, 0.1] uniform distributed noise,
        # about 77.90% of the log-mel are zero;
        # When the input signal is [-1, 1] uniform distributed noise,
        # about 62.10% of the log-mel are zero.
        # When the input signal is [-32768, 32767] uniform distributed noise,
        # about 56.02% of the log-mel are zero.
        output_with_large_mel_floor = test_outputs["outputs"]

        cfg_with_small_floor = cfg.set(mel_floor=1e-6)
        layer: LogMelFrontend = cfg_with_small_floor.set(name="test").instantiate(parent=None)
        test_outputs = self._jit_forward(layer, inputs, paddings)
        num_zero_elements = jnp.size(jnp.where(test_outputs["outputs"] == 0)[0])
        total_elements = jnp.size(test_outputs["outputs"])
        print(f"percentage of zero elements: {num_zero_elements / total_elements * 100}%")
        # Regardless whether input signal is [-1, 1] or [-0.1, 0.1] or [-32768, 32768)
        # about 56.02% of the log-mel are zero. This is due to the Mel matrix.
        output_with_correct_mel_floor = test_outputs["outputs"]

        # When the value of the incorrect output is not zero, then it should be right.
        self.assertAllClose(
            jnp.where(
                output_with_large_mel_floor == 0,
                output_with_correct_mel_floor,
                output_with_large_mel_floor,
            ),
            output_with_correct_mel_floor,
        )

    def test_fft(self):
        sample_rate, batch_size, max_seconds = 16_000, 8, 13
        num_filters, frame_size_ms, hop_size_ms = 80, 25, 10
        # Construct fake inputs.
        inputs, paddings = fake_audio(
            prng_key=jax.random.PRNGKey(123),
            batch_size=batch_size,
            seq_len=max_seconds * sample_rate,
            dtype=jnp.float32,
            scale=1.0,
        )
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            mel_floor=1.0,
        )
        # Note we expect the fft_transformation to explicitly config sharding.
        fft_cfg = config_for_function(sharded_fft).set(
            partition_spec=PartitionSpec("data", None, None)
        )
        ref_layer = cfg.clone(name="ref").instantiate(parent=None)

        with Mesh(
            mesh_utils.create_device_mesh((len(jax.devices()), 1)), ("data", "model")
        ) as mesh:
            # The sharded fn should be within a mesh context.
            layer = cfg.clone(name="test", fft=fft_cfg).instantiate(parent=None)
            inputs = jax.device_put(
                inputs,
                NamedSharding(mesh, PartitionSpec("data", None)),
            )
            paddings = jax.device_put(
                paddings,
                NamedSharding(mesh, PartitionSpec("data", None)),
            )
            ref_outputs = self._jit_forward(ref_layer, inputs, paddings)
            test_outputs = self._jit_forward(layer, inputs, paddings)

        self.assertAllClose(ref_outputs["outputs"], test_outputs["outputs"])
        self.assertAllClose(ref_outputs["paddings"], test_outputs["paddings"])

    @parameterized.product(
        [
            dict(frame_size_ms=25, hop_size_ms=10),
            dict(frame_size_ms=25, hop_size_ms=25),
        ],
    )
    @pytest.mark.fp64
    def test_log_stft(self, frame_size_ms, hop_size_ms):
        sample_rate, batch_size, max_seconds = 16_000, 4, 13
        num_filters = 80

        # Construct fake inputs.
        inputs, paddings = fake_audio(
            prng_key=jax.random.PRNGKey(123),
            batch_size=batch_size,
            seq_len=max_seconds * sample_rate,
            dtype=jnp.float64,
        )

        # Compute ref outputs.
        ref_outputs, ref_paddings = _ref_stft_frontend(
            inputs=inputs,
            paddings=paddings,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            num_filters=num_filters,
        )

        def _log_spectogram(x: Tensor, *, dtype: jnp.dtype) -> Tensor:
            x = magnitude_spectrogram(x, dtype=dtype)
            return linear_to_log_spectrogram(x).astype(dtype)

        # Compute test outputs.
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            fft_size=lambda _: 2 * num_filters - 2,
            mel_floor=1.0,
            pre_emphasis=None,
            spectrogram=config_for_function(lambda: _log_spectogram),
        )
        layer: LogMelFrontend = cfg.set(name="test").instantiate(parent=None)
        test_outputs = self._jit_forward(layer, inputs, paddings)
        test_outputs, test_paddings = test_outputs["outputs"], test_outputs["paddings"]

        # Only compare the non-padding outputs.
        ref_outputs = ref_outputs * (1 - tf.cast(ref_paddings, ref_outputs.dtype))[..., None]
        test_outputs = test_outputs * (1 - test_paddings[..., None, None])
        self.assertAllClose(ref_outputs[..., None], test_outputs)
        self.assertAllClose(ref_paddings, test_paddings)

        # Check that output shape is consistent.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertSequenceEqual(test_outputs.shape, output_shape)

    @parameterized.parameters([(jnp.float32,), (jnp.bfloat16,)])
    def test_dtype(self, dtype):
        # Test that the frontend outputs follow the same dtype as inputs.
        sample_rate, batch_size, max_seconds = 16_000, 4, 13
        num_filters = 80
        frame_size_ms, hop_size_ms = 25, 10
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            mel_floor=1.0,
        )
        layer: LogMelFrontend = cfg.set(name="test").instantiate(parent=None)
        inputs, paddings = fake_audio(
            prng_key=jax.random.PRNGKey(123),
            batch_size=batch_size,
            seq_len=max_seconds * sample_rate,
            dtype=dtype,
        )
        test_outputs = self._jit_forward(layer, inputs, paddings)
        test_outputs, test_paddings = test_outputs["outputs"], test_outputs["paddings"]
        self.assertEqual(test_outputs.dtype, inputs.dtype)
        self.assertEqual(test_paddings.dtype, paddings.dtype)


def _ref_frontend(
    *,
    inputs: tf.Tensor,
    paddings: tf.Tensor,
    sample_rate: int,
    frame_size_ms: int,
    hop_size_ms: int,
    coeff: float,
    num_filters: int,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
    mel_floor: float,
    pre_emphasis: bool,
):
    """Lingvo ASR frontend.

    Reference:
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L330
    """
    frame_size = ms_to_samples(frame_size_ms, sample_rate=sample_rate)
    frame_step = ms_to_samples(hop_size_ms, sample_rate=sample_rate)
    fft_size = int(max(512.0, math.pow(2, math.ceil(math.log(frame_size, 2)))))
    inputs, output_paddings = _ref_framer(
        inputs=inputs,
        paddings=paddings,
        frame_size=frame_size + int(pre_emphasis),
        frame_step=frame_step,
    )
    if pre_emphasis:
        inputs = _ref_pre_emphasis(inputs=inputs, coeff=coeff)
    inputs = tf.signal.hann_window(frame_size, dtype=inputs.dtype) * inputs
    outputs = _ref_log_mel_spectrogram(
        inputs=inputs,
        fft_size=fft_size,
        num_filters=num_filters,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        compute_energy=False,
        mel_floor=mel_floor,
    )
    return outputs, output_paddings


def _ref_stft_frontend(
    *,
    inputs: tf.Tensor,
    paddings: tf.Tensor,
    sample_rate: int,
    frame_size_ms: int,
    hop_size_ms: int,
    num_filters: int,
):
    """Tensorflow STFT."""
    frame_size = ms_to_samples(frame_size_ms, sample_rate=sample_rate)
    frame_step = ms_to_samples(hop_size_ms, sample_rate=sample_rate)

    _, output_paddings = _ref_framer(
        inputs=inputs, paddings=paddings, frame_size=frame_size, frame_step=frame_step
    )
    outputs = tf.signal.stft(
        inputs,
        frame_length=frame_size,
        frame_step=frame_step,
        fft_length=(num_filters - 1) * 2,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    # Note: tf.signal.stft appends more padding than necessary.
    outputs = outputs[:, : output_paddings.shape[1]]
    outputs = tf.math.log(
        tf.maximum(tf.math.abs(outputs), tf.experimental.numpy.finfo(outputs.dtype).tiny)
    )
    return outputs, output_paddings

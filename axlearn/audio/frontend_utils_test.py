# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests audio frontend utilities."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from numpy.typing import ArrayLike

from axlearn.audio.frontend_utils import (
    WindowType,
    linear_to_log_mel_spectrogram,
    linear_to_mel_weight_matrix,
    magnitude_spectrogram,
    next_power_of_2,
    pre_emphasis,
    sliding_window,
    windowing,
)
from axlearn.audio.test_utils import fake_audio
from axlearn.common.utils import as_tensor


class SlidingWindowTest(parameterized.TestCase, tf.test.TestCase):
    """Tests sliding window."""

    @parameterized.parameters(
        dict(frame_size=25, frame_step=10, seq_len=16_000 * 15),
        dict(frame_size=27, frame_step=13, seq_len=16_000 * 15),
        dict(frame_size=25, frame_step=25, seq_len=16_000 * 15),
        dict(frame_size=400, frame_step=160, seq_len=16_000 * 15),
    )
    def test_sliding_window(self, frame_size: int, frame_step: int, seq_len: int):
        batch_size = 5
        inputs, paddings = fake_audio(
            prng_key=jax.random.PRNGKey(123), batch_size=batch_size, seq_len=seq_len
        )
        ref_output, ref_paddings = _ref_framer(
            inputs=inputs,
            paddings=paddings,
            frame_size=frame_size,
            frame_step=frame_step,
        )
        fn = functools.partial(sliding_window, window_size=frame_size, stride=frame_step)
        fn = jax.jit(fn)
        test_paddings = jnp.max(fn(paddings), axis=-1)
        test_output = fn(inputs)
        self.assertAllClose(ref_output, test_output)
        self.assertAllClose(ref_paddings, test_paddings)


class PreEmphasisTest(parameterized.TestCase, tf.test.TestCase):
    """Tests pre-emphasis."""

    @parameterized.product(
        # Inputs are [batch, num_frames, frame_size].
        input_shape=[(5, 1298, 400), (5, 1298, 401)],
        coeff=[0.97, 0.0, 1.0],
    )
    @pytest.mark.fp64  # must annotate within @parameterized.parameters
    def test_pre_emphasis(self, input_shape, coeff: float):
        # fp64 seemingly only necessary under jit.
        inputs = jax.random.uniform(
            jax.random.PRNGKey(123),
            shape=input_shape,
            dtype=jnp.float64,
            minval=-32768.0,
            maxval=32768.0,
        )
        ref_outputs = _ref_pre_emphasis(inputs=inputs, coeff=coeff)
        test_outputs = jax.jit(pre_emphasis, static_argnames="coeff")(inputs, coeff=coeff)
        self.assertAllClose(ref_outputs, test_outputs)


class WindowingTest(parameterized.TestCase, tf.test.TestCase):
    """Tests windowing."""

    @parameterized.product(
        # Inputs are [batch, num_frames, frame_size].
        input_shape=[(5, 1298, 400), (5, 1298, 401)],
        window_type=list(WindowType),
        periodic=[True, False],
    )
    @pytest.mark.fp64  # must annotate within @parameterized.parameters
    def test_window(self, input_shape, window_type: WindowType, periodic: bool):
        inputs = jax.random.uniform(
            jax.random.PRNGKey(123),
            shape=input_shape,
            dtype=jnp.float64,
            minval=-32768.0,
            maxval=32768.0,
        )
        self.assertAllClose(
            _ref_window(
                inputs=inputs, window_type=window_type, dtype=tf.float64, periodic=periodic
            ),
            windowing(inputs, window_type=window_type, periodic=periodic),
        )


class SpectrogramTest(parameterized.TestCase, tf.test.TestCase):
    """Tests spectrograms."""

    @parameterized.product(
        # Inputs are [batch, num_frames, frame_size].
        input_shape=[(5, 1298, 400), (5, 1298, 401)],
        compute_energy=[True, False],
    )
    @pytest.mark.fp64  # must annotate within @parameterized.parameters
    def test_magnitude_spectrogram(self, input_shape, compute_energy: bool):
        inputs = jax.random.uniform(
            jax.random.PRNGKey(123),
            shape=input_shape,
            minval=-32768.0,
            maxval=32768.0,
            dtype=np.float64,
        )
        fft_size = next_power_of_2(inputs.shape[-1])
        inputs = _ref_window(inputs=inputs, window_type=WindowType.HANN, dtype=tf.float64)
        ref_spectrogram = _ref_magnitude_spectrogram(
            inputs=inputs, fft_size=fft_size, compute_energy=compute_energy
        )
        fn = jax.jit(magnitude_spectrogram, static_argnames="fft_size")
        test_spectrogram = fn(as_tensor(inputs), fft_size=fft_size)
        if compute_energy:
            test_spectrogram = jnp.square(test_spectrogram)
        self.assertAllClose(ref_spectrogram, test_spectrogram)

    @parameterized.parameters(
        dict(num_filters=80, fft_size=512),
    )
    @pytest.mark.fp64  # must annotate within @parameterized.parameters
    def test_mel_weight_matrix(self, num_filters: int, fft_size: int):
        sample_rate, lower_edge_hertz, upper_edge_hertz = 16_000, 125.0, 7600.0
        test_matrix = linear_to_mel_weight_matrix(
            num_filters=num_filters,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            dtype=jnp.float64,
        )
        ref_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_filters,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            dtype=tf.float64,
        )
        self.assertAllClose(ref_matrix, test_matrix)

    @parameterized.product(
        # Inputs are [batch, num_frames, frame_size].
        input_shape=[(5, 1298, 400), (5, 1298, 401)],
        compute_energy=[True, False],
        num_filters=[80],
    )
    @pytest.mark.fp64  # must annotate within @parameterized.parameters
    def test_log_mel_spectrogram(self, input_shape, compute_energy, num_filters):
        sample_rate, lower_edge_hertz, upper_edge_hertz, mel_floor = 16_000, 125.0, 7600.0, 1.0
        fft_size = next_power_of_2(input_shape[-1])
        spectrogram_fn = jax.jit(magnitude_spectrogram, static_argnames="fft_size")

        inputs = jax.random.uniform(
            jax.random.PRNGKey(123),
            shape=input_shape,
            minval=-32768.0,
            maxval=32768.0,
            dtype=np.float64,
        )
        fft_size = next_power_of_2(inputs.shape[-1])
        inputs = _ref_window(inputs=inputs, window_type=WindowType.HANN, dtype=tf.float64)

        # Compute spectrogram.
        spectrogram = spectrogram_fn(as_tensor(inputs), fft_size=fft_size)
        if compute_energy:
            spectrogram = jnp.square(spectrogram)
        test_matrix = linear_to_mel_weight_matrix(
            num_filters=num_filters,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            dtype=jnp.float64,
        )
        # Wrap the non-hashable mel matrix with partial, so it doesn't need to be in the jit cache.
        mel_spectrogram_fn = functools.partial(
            linear_to_log_mel_spectrogram, weight_matrix=test_matrix
        )
        test_outputs = jax.jit(mel_spectrogram_fn, static_argnames="mel_floor")(
            spectrogram,
            mel_floor=mel_floor,
        )

        # Compute reference outputs.
        ref_outputs = _ref_log_mel_spectrogram(
            inputs=inputs,
            fft_size=fft_size,
            num_filters=num_filters,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
            compute_energy=compute_energy,
            mel_floor=mel_floor,
        )
        self.assertAllClose(ref_outputs, test_outputs)


def _ref_framer(*, inputs: ArrayLike, paddings: ArrayLike, frame_size: int, frame_step: int):
    """Lingvo framer.

    Reference:
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L420
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L404
    """
    outputs = tf.signal.frame(inputs, frame_size, frame_step, pad_end=False)
    output_paddings = tf.signal.frame(paddings, frame_size, frame_step, pad_end=False)
    output_paddings = tf.reduce_max(output_paddings, axis=2)
    return outputs, output_paddings


def _ref_pre_emphasis(*, inputs: ArrayLike, coeff: float):
    """Lingvo pre-emphasis.

    Reference:
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L398
    """
    return inputs[:, :, 1:] - coeff * inputs[:, :, :-1]


def _ref_window(*, inputs: ArrayLike, window_type: WindowType, **kwargs):
    """Lingvo window.

    Reference:
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L244
    """
    frame_size = inputs.shape[-1]
    if window_type == WindowType.HANN:
        tf_window = tf.signal.hann_window(frame_size, **kwargs)
    elif window_type == WindowType.HAMMING:
        tf_window = tf.signal.hamming_window(frame_size, **kwargs)
    else:
        raise NotImplementedError(f"Unrecognized window type: {window_type}")
    return inputs * tf_window


def _ref_magnitude_spectrogram(*, inputs: ArrayLike, fft_size: int, compute_energy: bool):
    """Lingvo spectrogram.

    Reference:
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L456
    """
    spectrogram = tf.abs(tf.signal.rfft(inputs, [fft_size]))
    if compute_energy:
        spectrogram = tf.square(spectrogram)
    return spectrogram


def _ref_log_mel_spectrogram(
    *,
    inputs: ArrayLike,
    fft_size: int,
    num_filters: int,
    sample_rate: int,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
    compute_energy: bool,
    mel_floor: float,
):
    """Lingvo log-mel spectrogram.

    Reference:
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L456
    """
    spectrogram = _ref_magnitude_spectrogram(
        inputs=inputs, fft_size=fft_size, compute_energy=compute_energy
    )
    # Shape of magnitude spectrogram is [num_frames, fft_size // 2 + 1].
    # Mel_weight is [num_spectrogram_bins, num_mel_bins].
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_filters,
        num_spectrogram_bins=fft_size // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        dtype=inputs.dtype,
    )
    # Weight matrix implemented in the magnitude domain.
    batch_size, num_frames, fft_channels = tf.shape(spectrogram)[:3]
    mel_spectrogram = tf.matmul(
        tf.reshape(spectrogram, [batch_size * num_frames, fft_channels]),
        mel_weight_matrix,
    )
    mel_spectrogram = tf.reshape(mel_spectrogram, [batch_size, num_frames, num_filters])
    return tf.math.log(tf.maximum(float(mel_floor), mel_spectrogram))

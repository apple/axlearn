# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/tensorflow:
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Feature extraction for audio tasks."""

import enum
import math
from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax._src.mesh import thread_resources
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from numpy.typing import ArrayLike

from axlearn.common.utils import Tensor


class WindowType(enum.Enum):
    HANN = 0
    HAMMING = 1


class FrequencyScale(enum.Enum):
    # Mel scale (using natural log).
    MEL_LN = 0


def hertz_to_warped_scale(freq: ArrayLike, *, freq_scale: FrequencyScale) -> ArrayLike:
    """Converts frequencies in Hz to the target frequency scale.

    Args:
        freq: Value(s) in Hz.
        freq_scale: Target frequency scale.

    Returns:
        The frequency in the target scale.
    """
    if freq_scale == FrequencyScale.MEL_LN:
        return 1127.0 * np.log(1.0 + (freq / 700.0))
    else:
        raise NotImplementedError(f"Unsupported target scale {freq_scale}.")


def warped_to_hertz_scale(freq: ArrayLike, *, freq_scale: FrequencyScale) -> ArrayLike:
    """Converts frequencies from the source frequency scale to linear scale.

    Args:
        freq: Value(s) in the source scale.
        freq_scale: Source frequency scale.

    Returns:
        The frequency in Hz.
    """
    if freq_scale == FrequencyScale.MEL_LN:
        return 700.0 * (np.exp(freq / 1127.0) - 1.0)
    else:
        raise NotImplementedError(f"Unsupported source scale {freq_scale}.")


def sliding_window(x: Tensor, *, window_size: int, stride: int) -> Tensor:
    """Computes sliding windows.

    Args:
        x: A Tensor of shape `[..., seq_len]`.
        window_size: Size of sliding window.
        stride: Stride of sliding window.

    Returns:
        Windows of shape `[..., num_windows, window_size]` via sliding window on the last axis.
    """
    # NOTE: using `max` instead of `jnp.maximum` is necessary here to treat as constant for jit.
    output_size = max(x.shape[-1] - window_size, 0) // stride + 1
    idx = stride * jnp.arange(output_size)[:, None] + jnp.arange(window_size)[None, :]
    return x[..., idx]


def pre_emphasis(x: Tensor, *, coeff: Tensor) -> Tensor:
    """Applies a pre-emphasis filter to the input frames.

    Args:
        x: Input frames of shape `[..., frame_size]`.
        coeff: Pre-emphasis coefficient.

    Returns:
        Frames of shape `[..., frame_size-1]`.
    """
    return x[..., 1:] - coeff * x[..., :-1]


def windowing(x: Tensor, *, window_type: WindowType, periodic: bool = True) -> Tensor:
    """Applies windowing to the input frames of shape `[..., num_windows, window_size]`."""
    window_size = x.shape[-1]
    is_even = (1 - window_size % 2) * periodic

    if window_type == WindowType.HANN:
        coeffs = jnp.hanning(window_size + is_even)[:window_size]
    elif window_type == WindowType.HAMMING:
        coeffs = jnp.hamming(window_size + is_even)[:window_size]
    else:
        raise NotImplementedError(f"Unrecognized window_type {window_type}.")

    return (x * coeffs).astype(x.dtype)


def magnitude_spectrogram(ffts: Tensor, *, dtype: jnp.dtype) -> Tensor:
    """Computes magnitude of the spectrogram from the FFT matrix.

    Args:
        ffts: FFT of input audio frames of shape `[..., num_frames, fft_size]`.
        dtype: dtype of output tensor.

    Returns:
        A spectrogram of shape `[..., num_frames, num_spectrogram_bins=fft_size // 2 + 1]`.
    """
    out = jnp.abs(ffts)
    fft_size = ffts.shape[-1]
    out = out[..., : fft_size // 2 + 1]
    return out.astype(dtype)


def linear_to_log_spectrogram(x: Tensor) -> Tensor:
    """Converts linear scale spectrograms to log-mel spectrograms.

    Args:
        x: Magnitude or power spectrogram of shape `[..., num_frames, num_spectrogram_bins]`.

    Returns:
        A log spectrogram of shape `[..., num_frames, num_filters]`.
    """
    # Linear to log spectrogram.
    return jnp.log(jnp.maximum(x, jnp.finfo(x.dtype).tiny))


def linear_to_log_mel_spectrogram(
    x: Tensor, *, weight_matrix: ArrayLike, mel_floor: float = 1.0
) -> Tensor:
    """Converts linear scale spectrograms to log-mel spectrograms.

    Args:
        x: Magnitude or power spectrogram of shape `[..., num_frames, num_spectrogram_bins]`.
        weight_matrix: A weight matrix (or config instantiating thereof) of shape
            `[num_spectrogram_bins, num_filters]`.
        mel_floor: Minimum value of the output spectrogram prior to taking log.

    Returns:
        A spectrogram of shape `[..., num_frames, num_filters]`.
    """
    # Linear to mel spectrogram.
    x = jnp.maximum(mel_floor, x @ weight_matrix).astype(x.dtype)
    # Mel to log-mel spectrogram.
    return linear_to_log_spectrogram(x)


def linear_to_mel_weight_matrix(
    *,
    num_filters: int,
    num_spectrogram_bins: int,
    sample_rate: float,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
    dtype: np.dtype = np.float64,
) -> ArrayLike:
    """Computes the mel matrix, for converting linear scale spectrograms to mel scale.

    This implementation is based on `tf.signal.linear_to_mel_weight_matrix`:
    https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/python/ops/signal/mel_ops.py#L89-L215

    Args:
        num_filters: Number of mel bands in the resulting spectrum.
        num_spectrogram_bins: Number of frequency bins in the source spectrogram.
        sample_rate: Sample rate of the source signal.
        lower_edge_hertz: Lower bound on frequencies to include in the mel spectrum.
        upper_edge_hertz: Upper bound on frequencies to include in the mel spectrum.
        dtype: Dtype of the resulting matrix.

    Returns:
        A matrix of shape `[num_spectrogram_bins, num_filters]`.
    """
    freq_scale = FrequencyScale.MEL_LN

    # Compute mel spectrogram bins up to nyquist frequency. Drop the 0th bin.
    linear_freqs = np.linspace(0, sample_rate // 2, num_spectrogram_bins)[1:].astype(dtype)
    spectrogram_bins_mel = hertz_to_warped_scale(linear_freqs, freq_scale=freq_scale)[:, None]

    # Compute lower and upper bound of the output mel spectrum.
    lower_edge_mel = hertz_to_warped_scale(lower_edge_hertz, freq_scale=freq_scale)
    upper_edge_mel = hertz_to_warped_scale(upper_edge_hertz, freq_scale=freq_scale)

    # Compute num_filters triples of (lower_edge, center, upper_edge).
    idx = np.arange(num_filters)[:, None] + np.arange(3)[None, :]
    band_edges_mel = np.linspace(lower_edge_mel, upper_edge_mel, num_filters + 2)[idx].astype(dtype)
    # Split the triples up and reshape them into [1, num_filters] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = tuple(
        np.reshape(t, [1, num_filters]) for t in np.split(band_edges_mel, 3, axis=1)
    )
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)

    # Add back the 0th bin dropped above.
    return np.pad(np.maximum(np.minimum(upper_slopes, lower_slopes), 0.0), [[1, 0], [0, 0]])


def next_power_of_2(n: float):
    """Computes next power of 2. Returns unchanged if already a power of 2."""
    return 2 ** math.ceil(math.log2(n))


def sharded_fft(n: int, partition_spec: PartitionSpec) -> Callable[[Tensor], Tensor]:
    """A manually sharded FFT implementation.

    To get around a jax issue that FFT on GPU replicates the array instead of shard it.
    https://github.com/google/jax/issues/15680.

    Args:
        n: FFT size. Should be a power of 2.
        partition_spec: PartitionSpec for FFT inputs/outputs.

    Returns:
        A callable that computes FFT.
    """
    return shard_map(
        partial(jnp.fft.fft, n=n),
        mesh=thread_resources.env.physical_mesh,
        in_specs=partition_spec,
        out_specs=partition_spec,
    )

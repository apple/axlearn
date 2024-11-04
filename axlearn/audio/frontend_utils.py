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
from typing import Callable, Union

import einops
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


def _ceil_div(numerator: int | Tensor, denominator: int) -> int | Tensor:
    """Computes ceil(numerator / denominator).

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        The result of ceil(numerator / denominator).
    """
    return (numerator + denominator - 1) // denominator


def ms_to_samples(ms: Union[int, float], *, sample_rate: int) -> int:
    """Converts time in milliseconds to number of samples under the given sample rate.

    Args:
        ms: Time in milliseconds.
        sample_rate: Sample rate.

    Returns:
        Number of samples.

    Raises:
        ValueError: If the input is invalid.
    """
    out_size = (sample_rate * ms) / 1000
    if not out_size.is_integer():
        raise ValueError(f"out_size must be an integer, got {out_size}.")
    return int(out_size)


def num_frames(seq_len: int | Tensor, *, frame_size: int, hop_size: int) -> int | Tensor:
    """Computes the seq len of the frames.

    Note: it includes a partial frame.
    Eq: ceil((seq_len - frame_size + hop_size) / hop_size)

    Args:
        seq_len: Length of the input sequence.
        frame_size: Size of the frames.
        hop_size: hop_size of the frames.

    Returns:
        Output size of the frames.

    Raises:
        ValueError: If the input is invalid.
    """
    if seq_len < 0 or frame_size < 1 or hop_size < 0:
        raise ValueError(
            f"Invalid input: seq_len={seq_len}, frame_size={frame_size}, hop_size={hop_size}"
        )
    elif seq_len == 0:
        return 0
    return max(_ceil_div(seq_len - frame_size + hop_size, hop_size), 1)


def frame(x: Tensor, *, frame_size: int, hop_size: int, pad_value: int = 0) -> Tensor:
    """Frames inputs.

    The uses chunk-based indexing for speed and memory optimization, making it a bit complex.
    For example, with a window of 15 and a hop_size of 10, the chunk size is gcd (5). So, we chunk
    the data in increments of 5, and then perform indexing at the chunk level.

    # pylint: disable=line-too-long
    input: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ensure_chunkwise: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 0, 0, 0, 0]
    chunk_x: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [0, 0, 0, 0, 0]]
    chunk_index:     0                1                   2                     3                  4
    in_frame_indices: [[0, 1 ,2]]
    out_frame_indices: [[0], [2]]
    out_frame_indices + in_frame_indices: [[0, 1, 2], [2, 3, 4]]
    frame_x: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
              [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 0, 0, 0, 0]]
    # pylint: enable=line-too-long

    In real world, chunk=1200 (sample_rate=24kHz, window=25ms, hop_size=10ms), so it uses 1200 times
    less indexing.

    Args:
        x: A Tensor of shape `[..., seq_len]`.
        frame_size: Size of frames.
        hop_size: hop_size of frames.

    Returns:
        Windows of shape `[..., num_windows, frame_size]` via frames on the last axis.
    """
    x, ps = einops.pack([x], "* s")  # Ensure rank 2.

    def flatten_size(frames, window, hop_size):
        return frames * hop_size - hop_size + window

    def ensure_chunkwise(x, output_size):
        full_size = flatten_size(output_size, frame_size, hop_size)
        input_size = x.shape[1]
        frac = max(0, full_size - input_size)
        x = jnp.pad(x, ((0, 0), (0, frac)), constant_values=pad_value)
        return x[:, :full_size]

    output_size = num_frames(x.shape[1], frame_size=frame_size, hop_size=hop_size)
    x = ensure_chunkwise(x, output_size)

    # For optimization, the index will be created at the chunk level, rather than for every sample.
    chunk_size = math.gcd(frame_size, hop_size)
    chunk_x = einops.rearrange(x, "b (t c) -> b t c", c=chunk_size)
    num_chunk = chunk_x.shape[1]

    frame_ratio = frame_size // chunk_size
    hop_ratio = hop_size // chunk_size
    assert flatten_size(output_size, frame_ratio, hop_ratio) == num_chunk

    in_frame_indices = jnp.arange(frame_ratio)[jnp.newaxis, :]
    out_frame_indices = (jnp.arange(output_size) * hop_ratio)[:, jnp.newaxis]
    frame_x = chunk_x[:, out_frame_indices + in_frame_indices, :]
    frame_x = einops.rearrange(
        frame_x, "b ow iw c-> b ow (iw c)", ow=output_size, iw=frame_ratio, c=chunk_size
    )
    y = einops.unpack(frame_x, ps, "* f d")[0]
    return y


def frame_paddings(paddings: Tensor, *, frame_size: int, hop_size: int) -> Tensor:
    """Frames paddings.

    Given paddings,
      1 1 1 1 0 0 0 0 1 1
    we have frame_paddings, in frame_size=5 and hop_size=1 case,
      1 1 1 1 0 0
      1 1 1 0 0 0
      1 1 0 0 0 0
      1 0 0 0 0 1
      0 0 0 0 1 1

    out_paddings is as follows,
      1 1 1 1 1 1

    And frame_paddings, in frame_size=5 and hop_size=2 case,
      1 1 0 0
      1 1 0 0
      1 0 0 1
      1 0 0 1
      0 0 1 P

    out_paddings is as follows,
      1 1 1 1

    Args:
        x: A Tensor of shape `[..., seq_len]`.
        frame_size: Size of frames.
        hop_size: hop_size of frames.

    Returns:
        Windows of shape `[..., num_windows, frame_size]` via frames on the last axis.

    Raises:
        ValueError: If the input is invalid.
    """
    if hop_size > frame_size:
        raise ValueError(f"hop_size {hop_size} must be smaller than frame_size {frame_size}.")
    #  |   input paddings  |
    #  |1 1 1 1 0 0 0 0 1 1|
    #   |_max 1_|
    #       |_max 1_|
    #           |_max 1_|
    #               |_max 1_|
    paddings_frame = frame(paddings, frame_size=frame_size, hop_size=hop_size, pad_value=1)
    out_paddings = jnp.max(paddings_frame, axis=-1)
    return out_paddings


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

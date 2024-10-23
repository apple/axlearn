# Copyright © 2023 Apple Inc.
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Audio frontends for feature extraction."""

import functools
from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Protocol, Union

import jax.numpy as jnp

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
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
    maybe_set_config,
)
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


class StageFn(Protocol):
    """A frontend "stage" is a callable that takes a Tensor and emits a Tensor."""

    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        pass


def normalize_by_mean_std(
    x: Tensor, *, mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None
) -> Tensor:
    """Scales the input by subtracting pre-computed `mean` and/or dividing by pre-computed `std`."""
    if mean is not None:
        x = x - jnp.array(mean, dtype=x.dtype)
    if std is not None:
        x = x / jnp.maximum(jnp.array(std, dtype=x.dtype), jnp.finfo(x.dtype).eps)
    return x


def _ms_to_samples(ms: Union[int, float], *, sample_rate: int) -> float:
    """Converts time in milliseconds to number of samples under the given sample rate."""
    return sample_rate / 1000 * ms


class BaseFrontend(BaseLayer):
    """Defines the interface for speech frontend."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseFrontend."""

        # Number of output channels.
        output_dim: Required[int] = REQUIRED
        # Number of filters/bands in the output spectrogram.
        num_filters: Required[int] = REQUIRED
        # Number of input samples per second, e.g., 24000 for 24KHz inputs.
        sample_rate: Required[int] = REQUIRED
        # Size of each frame in ms.
        frame_size_ms: Required[float] = REQUIRED
        # Hop size in ms.
        hop_size_ms: Required[float] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        frame_size = _ms_to_samples(cfg.frame_size_ms, sample_rate=cfg.sample_rate)
        hop_size = _ms_to_samples(cfg.hop_size_ms, sample_rate=cfg.sample_rate)
        if not frame_size.is_integer():
            raise ValueError(f"frame_size must be an integer, got {frame_size}.")
        if not hop_size.is_integer():
            raise ValueError(f"hop_size must be an integer, got {hop_size}.")

        self._frame_size = int(frame_size)
        self._hop_size = int(hop_size)


def _log_mel_spectrogram(
    *, num_filters: int, sample_rate: int, fft_size: int, mel_floor: float
) -> StageFn:
    """Returns a StageFn that computes Log Mel spectrogram."""

    # Mel filterbank, used to convert magnitude spectrogram to mel spectrogram. Only needs to be
    # constructed once.
    filterbank = linear_to_mel_weight_matrix(
        num_filters=num_filters,
        num_spectrogram_bins=fft_size // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7600.0,
    )

    def fn(fft: Tensor, *, dtype: jnp.dtype) -> Tensor:
        # [batch_size, num_frames, fft_size] -> [batch_size, num_frames, fft_size // 2 + 1].
        spectrogram = magnitude_spectrogram(fft, dtype=dtype)
        # Convert to log-mel. [batch, num_frames, num_filters].
        return linear_to_log_mel_spectrogram(
            spectrogram,
            weight_matrix=filterbank,
            mel_floor=mel_floor,
        )

    return fn


def _pre_emphasis(coeff: float) -> StageFn:
    """Returns a StageFn that applies pre-emphasis."""
    # Native python float is fp64, explicitly cast it to fp32.
    return functools.partial(pre_emphasis, coeff=jnp.array(coeff, dtype=jnp.float32))


class LogMelFrontend(BaseFrontend):
    """Computes Log Mel spectrogram features.

    The frontend implements the following stages:
        `Framer -> PreEmphasis -> Window -> FFT -> FilterBank -> MeanStdDev`.
    """

    @config_class
    class Config(BaseFrontend.Config):
        """Configures LogMelFrontend."""

        # Number of output channels. Should always be 1.
        output_dim: int = 1
        # Optional output transformation. See `normalize_by_mean_std` for an example.
        output_transformation: Optional[InstantiableConfig[StageFn]] = None
        # Floor of melfilter bank energy to prevent log(0).
        # Recommend to set to 1e-6 or smaller to capture
        # low-energy signals.
        # TODO(markblee): Deprecate this in favor of setting `mel_floor` on `spectrogram`.
        mel_floor: Required[float] = REQUIRED
        # Pre-emphasis filter. If None, skips pre-emphasis.
        pre_emphasis: Optional[InstantiableConfig[StageFn]] = config_for_function(
            _pre_emphasis
        ).set(coeff=0.97)
        # Computes fft size from frame size.
        fft_size: Callable[[int], int] = next_power_of_2
        # Optional customized FFT implementation. Use `jnp.fft.fft` if None.
        # This can be used to support a sharded implementation of FFT.
        # See `sharded_fft` for an example.
        fft: Optional[InstantiableConfig[StageFn]] = None
        # Constructs mel spectogram from FFT outputs.
        spectrogram: InstantiableConfig[StageFn] = config_for_function(_log_mel_spectrogram)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.output_dim != 1:
            raise ValueError(
                "output_dim should always be 1. Did you mean to configure num_filters instead?"
            )
        self._output_transformation = None
        if cfg.output_transformation is not None:
            self._output_transformation = maybe_instantiate(cfg.output_transformation)

        fft_size = cfg.fft_size(self._frame_size)
        if cfg.fft is not None:
            self._fft = cfg.fft.set(n=fft_size).instantiate()
        else:
            self._fft = partial(jnp.fft.fft, n=fft_size)

        spectrogram = maybe_set_config(
            cfg.spectrogram,
            fft_size=fft_size,
            num_filters=cfg.num_filters,
            sample_rate=cfg.sample_rate,
            mel_floor=cfg.mel_floor,
        )
        self._spectrogram = spectrogram.instantiate()

        self._pre_emphasis = None
        if cfg.pre_emphasis is not None:
            self._frame_size += 1
            self._pre_emphasis = cfg.pre_emphasis.instantiate()

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> dict[str, Tensor]:
        """Computes log-mel spectrogram features.

        Args:
            inputs: Tensor of dtype float32 and shape [batch, seq_len].
            paddings: A 0/1 Tensor of shape [batch, seq_len]. 1's represent padded positions.

        Returns:
            A dict containing:
            - outputs: A Tensor of shape [batch, num_frames, num_filters, 1].
            - paddings: A 0/1 Tensor of shape [batch, num_frames].
        """
        # TODO(markblee): Make these configurable as needed.
        # Framer.
        frames = sliding_window(inputs, window_size=self._frame_size, stride=self._hop_size)
        if self._pre_emphasis is not None:
            frames = self._pre_emphasis(frames)
        # Windowing. Defaults to a Hann window.
        # [batch_size, num_frames, frame_size].
        frames = windowing(frames, window_type=WindowType.HANN)
        # FFT and construct spectrogram.
        # [batch_size, num_frames, fft_size] -> [batch, num_frames, num_filters].
        outputs = self._spectrogram(self._fft(frames), dtype=frames.dtype)
        if self._output_transformation is not None:
            outputs = self._output_transformation(outputs)
        # To identify padding frames, apply the framer to the input padding.
        # Consider a frame padded if it contains at least one padding sample.
        paddings = sliding_window(paddings, window_size=self._frame_size, stride=self._hop_size)
        # TODO(dhwang2): Logically, a partial frame is a valid frame. Explore it later.
        paddings = jnp.max(paddings, axis=-1, keepdims=True)
        outputs = outputs * (1 - paddings)
        return dict(outputs=outputs[..., None], paddings=jnp.squeeze(paddings, axis=-1))

    def output_shape(self, *, input_shape: Sequence[Optional[int]]):
        """Computes the output shape given input shape.

        Args:
            input_shape: Values for the input dimensions [batch_size, seq_len]. Each value can be an
                integer or None, where None can be used if the dimension is not known.

        Returns:
            The output shape. The dimensions are [batch_size, num_frames, num_filters, 1].

        Raises:
            ValueError: If `input_shape` is invalid.
        """
        cfg: LogMelFrontend.Config = self.config
        if len(input_shape) != 2:
            raise ValueError(f"We expect len(input_shape) = 2, but got {len(input_shape)}.")
        batch_size, seq_len = input_shape
        if seq_len is not None:
            num_frames = max(seq_len - self._frame_size, 0) // self._hop_size + 1
        else:
            num_frames = None
        return [batch_size, num_frames, cfg.num_filters, cfg.output_dim]

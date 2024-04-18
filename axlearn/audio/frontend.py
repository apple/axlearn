# Copyright © 2023 Apple Inc.

"""Audio frontends for feature extraction."""

from typing import Callable, Dict, Optional, Sequence, Union

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
    maybe_instantiate,
)
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


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


class LogMelFrontend(BaseLayer):
    """Computes Log Mel spectrogram features.

    The frontend implements the following stages:
        `Framer -> PreEmphasis -> Window -> FFT -> FilterBank -> MeanStdDev`.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures LogMelFrontend."""

        # Number of output channels. Should always be 1.
        output_dim: int = 1
        # Number of filters/bands in the output spectrogram.
        num_filters: Required[int] = REQUIRED
        # Number of input samples per second, e.g., 24000 for 24KHz inputs.
        sample_rate: Required[int] = REQUIRED
        # Size of each frame in ms.
        frame_size_ms: Required[float] = REQUIRED
        # Hop size in ms.
        hop_size_ms: Required[float] = REQUIRED
        # Optional output transformation. See `normalize_by_mean_std` for an example.
        output_transformation: Optional[InstantiableConfig[Callable[[Tensor], Tensor]]] = None
        # Floor of melfilter bank energy to prevent log(0).
        # Set to 1.0 for backwards-compatibility. Recommend to set to 1e-6 or smaller to capture
        # low-energy signals.
        mel_floor: Required[float] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.output_dim != 1:
            raise ValueError(
                "output_dim should always be 1. Did you mean to configure num_filters instead?"
            )
        frame_size = _ms_to_samples(cfg.frame_size_ms, sample_rate=cfg.sample_rate)
        hop_size = _ms_to_samples(cfg.hop_size_ms, sample_rate=cfg.sample_rate)
        if not frame_size.is_integer():
            raise ValueError(f"frame_size must be an integer, got {frame_size}.")
        if not hop_size.is_integer():
            raise ValueError(f"hop_size must be an integer, got {hop_size}.")

        self._frame_size = int(frame_size)
        self._hop_size = int(hop_size)

        self._output_transformation = None
        if cfg.output_transformation is not None:
            self._output_transformation = maybe_instantiate(cfg.output_transformation)

        # Mel filterbank, used to convert magnitude spectrogram to mel spectrogram. Only needs to be
        # constructed once.
        self._fft_size = next_power_of_2(self._frame_size)
        self._filterbank = linear_to_mel_weight_matrix(
            num_filters=cfg.num_filters,
            num_spectrogram_bins=self._fft_size // 2 + 1,
            sample_rate=cfg.sample_rate,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7600.0,
        )
        self._mel_floor = self.config.mel_floor

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Dict[str, Tensor]:
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
        # Framer. Add 1 to frame size for pre-emphasis.
        frames = sliding_window(inputs, window_size=self._frame_size + 1, stride=self._hop_size)
        # Pre-emphasis filter.
        frames = pre_emphasis(frames, coeff=0.97)
        # Windowing. Defaults to a Hann window.
        frames = windowing(frames, window_type=WindowType.HANN)
        # FFT.
        spectrogram = magnitude_spectrogram(frames, fft_size=self._fft_size)
        # Convert to log-mel. [batch, num_frames, num_filters].
        outputs = linear_to_log_mel_spectrogram(
            spectrogram,
            weight_matrix=self._filterbank,
            mel_floor=self._mel_floor,
        )
        if self._output_transformation is not None:
            outputs = self._output_transformation(outputs)
        # To identify padding frames, apply the framer to the input padding.
        # Consider a frame padded if it contains at least one padding sample.
        paddings = sliding_window(paddings, window_size=self._frame_size + 1, stride=self._hop_size)
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
            num_frames = max(seq_len - (self._frame_size + 1), 0) // self._hop_size + 1
        else:
            num_frames = None
        return [batch_size, num_frames, cfg.num_filters, cfg.output_dim]

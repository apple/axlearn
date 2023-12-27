# Copyright © 2023 Apple Inc.

"""Speech encoder layers."""

from math import prod
from typing import Any, Dict, Optional, Sequence

import jax.numpy as jnp

from axlearn.audio.frontend import LogMelFrontend
from axlearn.audio.spectrum_augmenter import SpectrumAugmenter
from axlearn.audio.subsamplers import ConvSubSampler
from axlearn.common.attention import SinusoidalPositionalEmbedding
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.conformer import RepeatedConformerLayer
from axlearn.common.layers import Dropout, Linear
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


class SpeechFeatureLayer(BaseLayer):
    """Computes speech features from audio waveform."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures SpeechFeatureLayer."""

        # Output dimension.
        output_dim: Required[int] = REQUIRED
        # Converts raw waveforms to features.
        frontend: BaseLayer.Config = LogMelFrontend.default_config()
        # Applies feature augmentation. Should not affect the shape of inputs.
        augmenter: Optional[BaseLayer.Config] = SpectrumAugmenter.default_config()
        # Applies feature subsampling.
        subsampler: BaseLayer.Config = ConvSubSampler.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("frontend", cfg.frontend)
        if cfg.augmenter is not None:
            self._add_child("augmenter", cfg.augmenter)
        self._add_child(
            "subsampler",
            cfg.subsampler.set(
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Dict[str, Any]:
        """Computes speech features.

        The default transformations and corresponding shapes are:
            -> Audio waveform: [batch_size, seq_len].
            -> Log mel filter banks: [batch_size, num_frames, num_freq, 1].
                `num_frames` is (roughly) around
                `(seq_len / cfg.frontend.sample_rate) * (1000 / cfg.frontend.hop_size_ms)`
                and `num_freq` is `cfg.frontend.num_filters`.
                See `cfg.frontend` for more specifics.
            -> Spectrum augmented filter banks: [batch_size, num_frames, num_freq, 1].
            -> Convolution subsampled filter banks:
                [batch_size, subsampled_frames, subsampled_freq, output_dim]. The
                `subsampled_frames`, `subsampled_freq`, and `output_dim` are determined by the
                convolution(s) in the subsampler.

        Args:
            inputs: A float Tensor of shape [batch_size, seq_len]. Values need not be normalized.
            paddings: A 0/1 Tensor of shape [batch_size, seq_len]. 1's represent padded positions.

        Returns:
            A dict containing:
            - outputs: A Tensor of shape
                [batch_size, subsampled_frames, subsampled_freq, output_dim].
            - paddings: A 0/1 Tensor of shape [batch_size, subsampled_frames].
        """
        # Compute frontend features.
        features = self.frontend(inputs=inputs, paddings=paddings)
        x = features["outputs"]

        if "augmenter" in self.children:
            if len(features["outputs"].shape) == 3:
                x = x[..., None]
            # Apply augmentation.
            x = self.augmenter(inputs=x, paddings=features["paddings"])
            if len(features["outputs"].shape) == 3:
                x = jnp.squeeze(x, axis=-1)

        # Apply subsampling.
        # [batch_size, subsampled_frames, subsampled_freq, output_dim].
        subsampled_features = self.subsampler(inputs=x, paddings=features["paddings"])
        x = subsampled_features["outputs"]

        return dict(outputs=x, paddings=subsampled_features["paddings"])

    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        """Computes the speech features output shape.

        Args:
            input_shape: Values for the input dimensions [batch_size, seq_len]. Each value can be an
                integer or None, where None can be used if the shape is not known.

        Returns:
            The output shape. The dimensions are [batch_size, subsampled_frames, feature_dim].

        Raises:
            ValueError: If `input_shape` is invalid.
        """
        if len(input_shape) != 2:
            raise ValueError(f"We expect len(input_shape) = 2, but got {len(input_shape)}.")
        # [batch_size, num_frames, num_filters, cfg.frontend.output_dim].
        output_shape = self.frontend.output_shape(input_shape=input_shape)
        # [batch_size, subsampled_frames, subsampled_filters, cfg.output_dim].
        return self.subsampler.output_shape(input_shape=output_shape)


class SpeechContextNetwork(BaseLayer):
    """Speech context network.

    Reference: Figure 3 in https://arxiv.org/abs/2109.13226.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures SpeechContextNetwork."""

        # Input feature dimension.
        input_dim: Required[int] = REQUIRED
        # Output feature dimension.
        output_dim: Required[int] = REQUIRED
        # Input projection.
        input_linear: Linear.Config = Linear.default_config()
        # Dropout applied after projection.
        dropout: Dropout.Config = Dropout.default_config()
        # Positional embeddings.
        pos_emb: BaseLayer.Config = SinusoidalPositionalEmbedding.default_config()
        # Context layers, e.g. a conformer stack.
        context: BaseLayer.Config = RepeatedConformerLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "input_linear", cfg.input_linear.set(input_dim=cfg.input_dim, output_dim=cfg.output_dim)
        )
        self._add_child("dropout", cfg.dropout)
        self._add_child("pos_emb", cfg.pos_emb.set(dim=cfg.output_dim))
        self._add_child("context", cfg.context.set(input_dim=cfg.output_dim))

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Dict[str, Tensor]:
        """Computes context features.

        Args:
            inputs: A Tensor of shape [batch_size, seq_len, input_dim].
            paddings: A 0/1 Tensor of shape [batch_size, seq_len].

        Returns:
            A dict containing:
            - outputs: A Tensor of shape [batch_size, seq_len, output_dim].
            - output_paddings: A 0/1 Tensor of shape [batch_size, seq_len].
        """
        # [batch, seq_len, input_dim].
        x = self.input_linear(inputs)
        x = self.dropout(x)
        x = x + self.pos_emb(jnp.arange(x.shape[1]))
        x = self.context(inputs=x, paddings=paddings)
        self._add_activation_summary(
            name="speech_context",
            activations=x,
            activation_paddings=paddings,
        )
        return dict(outputs=x * (1 - paddings[..., None]), paddings=paddings)


class ASREncoder(BaseLayer):
    """ASR encoder."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ASREncoder.

        Fields:
            dim: Hidden dimension.
            feature: A layer that takes inputs of shape [batch_size, seq_len] and produces outputs
                of shape [batch_size, num_frames, ...]. The trailing dims after `batch_size` and
                `seq_len` will be flattened prior to passing to the context network.
                This layer is expected to define an `output_shape` method that computes the output
                shape given the input shape [batch_size=None, num_frames=None].
            context: A layer that takes inputs of shape
                [batch_size, num_frames, prod(feature.output_shape([None, None])[2:])] and returns
                outputs of shape [batch_size, num_frames, cfg.dim].
        """

        # Hidden dimension.
        dim: Required[int] = REQUIRED
        # Feature processing.
        feature: Required[BaseLayer.Config] = REQUIRED
        # Context network.
        context: Required[BaseLayer.Config] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("feature", cfg.feature)
        feature_shape = self.feature.output_shape(input_shape=[None, None])
        self._add_child(
            "context", cfg.context.set(input_dim=prod(feature_shape[2:]), output_dim=cfg.dim)
        )

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Dict[str, Tensor]:
        """Computes speech encoder features from waveform.

        Args:
            inputs: A float Tensor of shape [batch_size, seq_len]. Values need not be normalized.
            paddings: A 0/1 Tensor of shape [batch_size, seq_len].

        Returns:
            A dict containing:
            - outputs: A Tensor of shape [batch_size, num_frames, dim].
            - output_paddings: A 0/1 Tensor of shape [batch_size, num_frames].
        """
        speech_features = self.feature(inputs=inputs, paddings=paddings)
        context_features = self.context(
            # Flatten features to [batch_size, num_frames, cfg.context.input_dim].
            inputs=jnp.reshape(
                speech_features["outputs"], speech_features["outputs"].shape[:2] + (-1,)
            ),
            paddings=speech_features["paddings"],
        )
        return context_features

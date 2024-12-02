# Copyright © 2023 Apple Inc.

"""Subsampler layers."""

from collections.abc import Sequence
from typing import Optional, Union

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.convolution import Conv2DWith1DPadding
from axlearn.common.layers import BaseNormalizationLayer, get_activation_fn
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


class ConvSubSampler(BaseLayer):
    """Subsamples the speech by convolution.

    The subsampling uses a pair of convolutions with optional non-linearities in between.
    This is similar but not identical to the behavior in WeNet.

    References:
    - Section 2 in https://arxiv.org/abs/2005.08100.
    - Section 3 in https://arxiv.org/abs/2102.01547.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures ConvSubSampler."""

        # Input channel dim. Typically 1 for spectrogram inputs.
        input_dim: int = 1
        # Output channel dim.
        output_dim: Required[int] = REQUIRED
        # Hidden dim of the conv layers. If None, defaults to output_dim.
        hidden_dim: Optional[int] = None
        # Configures both of the convolutions.
        conv: Conv2DWith1DPadding.Config = Conv2DWith1DPadding.default_config().set(
            window=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )
        # Optional normalization config, to be applied after each convolution.
        norm: Optional[BaseNormalizationLayer.Config] = None
        # Optional activation to be applied after each convolution + norm. This can either be None
        # (in which case no activations are used), a string (used for both convolutions), a pair of
        # strings (used for each convolution respectively), or a pair of string and None (to apply
        # activation to only one convolution).
        activation: Optional[Union[Optional[str], tuple[Optional[str], Optional[str]]]] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        activation = cfg.activation
        if not isinstance(activation, (list, tuple)):
            activation = (activation, activation)
        if len(activation) != 2 or not all(x is None or isinstance(x, str) for x in activation):
            raise ValueError(
                "Expected cfg.activation to be None, a string, or pair of string | None, "
                f"got: {cfg.activation}"
            )
        self._activation = [None if act is None else get_activation_fn(act) for act in activation]

        hidden_dim = cfg.hidden_dim or cfg.output_dim
        self._add_child("conv1", cfg.conv.set(input_dim=cfg.input_dim, output_dim=hidden_dim))
        self._add_child("conv2", cfg.conv.set(input_dim=hidden_dim, output_dim=cfg.output_dim))

        if cfg.norm:
            self._add_child("norm1", cfg.norm.set(input_dim=hidden_dim))
            self._add_child("norm2", cfg.norm.set(input_dim=cfg.output_dim))

    def output_shape(self, *, input_shape: Sequence[Optional[int]]):
        """Computes the output shape after subsampling.

        Args:
            input_shape: Values for the input dimensions
                [batch_size, num_frames, num_freq, num_channels]. Each value can be an integer or
                None, where None can be used if the shape is not known.

        Returns:
            The output shape after subsampling.

        Raises:
            ValueError: If `input_shape` is invalid.
        """
        cfg: ConvSubSampler.Config = self.config
        if len(input_shape) != 4:
            raise ValueError(f"We expect len(input_shape) = 4, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                f"cfg.input_dim = {cfg.input_dim}."
            )
        conv1_shape = self.conv1.output_shape(input_shape=input_shape)
        conv2_shape = self.conv2.output_shape(input_shape=conv1_shape)
        return conv2_shape

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> dict[str, Tensor]:
        """Subsamples the speech.

        Args:
            inputs: A Tensor of shape [batch_size, num_frames, num_freq, input_dim].
            paddings: 0/1 Tensor of shape [batch_size, num_frames].

        Returns:
            A dict containing:
            - outputs: A Tensor of shape
                [batch_size, subsampled_frames, subsampled_freq, output_dim].
            - paddings: 0/1 Tensor of shape [batch, subsampled_frames].
        """
        cfg: ConvSubSampler.Config = self.config
        self._add_activation_summary(
            name="subsampler_inputs", activations=inputs, activation_paddings=paddings
        )
        x, paddings = self.conv1(inputs, paddings=paddings)
        if cfg.norm:
            x = self.norm1(x, paddings=paddings)
        if self._activation[0]:
            x = self._activation[0](x)

        x, paddings = self.conv2(x, paddings=paddings)
        if cfg.norm:
            x = self.norm2(x, paddings=paddings)
        if self._activation[1]:
            x = self._activation[1](x)

        self._add_activation_summary(
            name="subsampler_outputs", activations=x, activation_paddings=paddings
        )
        return dict(outputs=x, paddings=paddings)

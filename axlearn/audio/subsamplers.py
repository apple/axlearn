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
        hidden_dim: int | list[int] | None = None
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

    @classmethod
    def get_hidden_dim_list(cls, cfg: Config) -> list[int]:
        if isinstance(cfg.hidden_dim, int):
            hidden_dim = [cfg.hidden_dim]
        elif cfg.hidden_dim is None:
            hidden_dim = [cfg.output_dim]
        else:
            hidden_dim = list(cfg.hidden_dim)
        return hidden_dim

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        activation = cfg.activation
        hidden_dim = [cfg.input_dim] + self.get_hidden_dim_list(cfg) + [cfg.output_dim]
        self.num_layers = len(hidden_dim) - 1
        if not isinstance(activation, (list, tuple)):
            activation = [activation] * self.num_layers
        if len(activation) != self.num_layers or not all(
            x is None or isinstance(x, str) for x in activation
        ):
            raise ValueError(
                "Expected cfg.activation to be None, a string, or list/tuple of string | None, "
                f"got: {cfg.activation}"
            )
        self._activation = [None if act is None else get_activation_fn(act) for act in activation]

        for i in range(1, len(hidden_dim)):
            self._add_child(
                f"conv{i}", cfg.conv.set(input_dim=hidden_dim[i - 1], output_dim=hidden_dim[i])
            )
        if cfg.norm:
            for i in range(1, len(hidden_dim)):
                self._add_child(f"norm{i}", cfg.norm.set(input_dim=hidden_dim[i]))

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
        for i in range(1, self.num_layers + 1):
            input_shape = self._children[f"conv{i}"].output_shape(input_shape=input_shape)
        return input_shape

    def forward(self, inputs: Tensor, *, segment_ids: Tensor) -> dict[str, Tensor]:
        """Subsamples the speech.

        Args:
            inputs: A Tensor of shape [batch_size, num_frames, num_freq, input_dim].
            segment_ids: An int Tensor of shape [batch_size, num_frames].

        Returns:
            A dict containing:
            - outputs: A Tensor of shape
                [batch_size, subsampled_frames, subsampled_freq, output_dim].
            - segment_ids: An int Tensor of shape [batch_size, subsampled_frames].
        """
        cfg: ConvSubSampler.Config = self.config
        paddings = segment_ids == 0
        self._add_activation_summary(
            name="subsampler_inputs", activations=inputs, activation_paddings=paddings
        )
        x = inputs
        for i in range(1, self.num_layers + 1):
            x, paddings = self._children[f"conv{i}"](x, paddings=paddings)
            segment_ids = self._children[f"conv{i}"].conv_paddings(segment_ids)
            if cfg.norm:
                x = self._children[f"norm{i}"](x, segment_ids=segment_ids)
            if self._activation[i - 1]:
                x = self._activation[i - 1](x)
        self._add_activation_summary(
            name="subsampler_outputs", activations=x, activation_paddings=paddings
        )
        return dict(outputs=x, segment_ids=segment_ids)

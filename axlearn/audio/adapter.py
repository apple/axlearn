# Copyright © 2023 Apple Inc.

"""Audio model adapter for efficient fine-tuning."""

from typing import Optional

import jax

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.layers import BatchNorm, LayerNorm, Linear
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.param_init import DefaultInitializer, WeightInitializer


class AudioModelAdapter(BaseLayer):
    """Adapter layer for efficient fine-tuning of audio models."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures AudioModelAdapter."""

        # Input feature dimension.
        input_dim: Required[int] = REQUIRED
        # Bottleneck dimension (typically much smaller than input_dim).
        bottleneck_dim: Required[int] = REQUIRED
        # Whether to apply layer normalization before the adapter.
        use_layer_norm: bool = True
        # Whether to apply batch normalization in the adapter.
        use_batch_norm: bool = False
        # Scaling factor for the adapter output.
        adapter_scale: float = 1.0
        # Activation function to use.
        activation: str = "relu"
        # Whether to add a residual connection.
        residual: bool = True

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        # Initialize with small weights to make adapter less disruptive initially
        weight_init = WeightInitializer.default_config().set(
            distribution="normal",
            fan="fan_in",
            scale=0.01,
        )

        bias_init = WeightInitializer.default_config().set(
            distribution="normal",
            fan=None,
            scale=0.01,
        )

        param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                ".*weight": weight_init,
                ".*bias": bias_init,
            },
        )

        # Down projection to bottleneck dimension
        self._add_child(
            "down_proj",
            Linear.default_config().set(
                input_dim=cfg.input_dim,
                output_dim=cfg.bottleneck_dim,
                bias=True,
                param_init=param_init,
            ),
        )

        # Optional batch normalization
        if cfg.use_batch_norm:
            self._add_child(
                "batch_norm",
                BatchNorm.default_config().set(
                    input_dim=cfg.bottleneck_dim,
                    decay=0.9,
                ),
            )

        # Up projection back to input dimension
        self._add_child(
            "up_proj",
            Linear.default_config().set(
                input_dim=cfg.bottleneck_dim,
                output_dim=cfg.input_dim,
                bias=True,
                param_init=param_init,
            ),
        )

        # Optional layer normalization
        if cfg.use_layer_norm:
            self._add_child(
                "layer_norm",
                LayerNorm.default_config().set(
                    input_dim=cfg.input_dim,
                ),
            )

    def forward(self, inputs, **_kwargs):
        """Apply the adapter transformation.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_dim].
            **_kwargs: Additional keyword arguments (unused, kept for API compatibility).

        Returns:
            Tensor of the same shape as inputs.
        """
        cfg = self.config
        residual = inputs

        # Apply layer normalization if specified
        x = inputs
        if cfg.use_layer_norm:
            x = self.layer_norm(x)

        # Down projection
        x = self.down_proj(x)

        # Apply batch normalization if specified
        if cfg.use_batch_norm:
            # BatchNorm uses is_training from context automatically
            x = self.batch_norm(x)

        # Activation
        if cfg.activation == "relu":
            x = jax.nn.relu(x)
        elif cfg.activation == "gelu":
            x = jax.nn.gelu(x)

        # Up projection
        x = self.up_proj(x)

        # Scale the output
        if cfg.adapter_scale != 1.0:
            x = x * cfg.adapter_scale

        # Add residual connection if specified
        if cfg.residual:
            x = x + residual

        return x


class ASRModelAdapter(BaseLayer):
    """Adapter for Automatic Speech Recognition (ASR) models."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures ASRModelAdapter."""

        # Feature dimension of the encoder.
        encoder_dim: Required[int] = REQUIRED
        # Bottleneck dimension for encoder adapters.
        encoder_bottleneck_dim: Required[int] = REQUIRED
        # Feature dimension of the decoder.
        decoder_dim: Optional[int] = None
        # Bottleneck dimension for decoder adapters.
        decoder_bottleneck_dim: Optional[int] = None
        # Whether to add adapters to the encoder.
        adapt_encoder: bool = True
        # Whether to add adapters to the decoder.
        adapt_decoder: bool = False
        # Adapter configuration.
        adapter: AudioModelAdapter.Config = AudioModelAdapter.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        if cfg.adapt_encoder:
            self._add_child(
                "encoder_adapter",
                cfg.adapter.clone(
                    input_dim=cfg.encoder_dim,
                    bottleneck_dim=cfg.encoder_bottleneck_dim,
                ),
            )

        if (
            cfg.adapt_decoder
            and cfg.decoder_dim is not None
            and cfg.decoder_bottleneck_dim is not None
        ):
            self._add_child(
                "decoder_adapter",
                cfg.adapter.clone(
                    input_dim=cfg.decoder_dim,
                    bottleneck_dim=cfg.decoder_bottleneck_dim,
                ),
            )

    def adapt_encoder_features(self, features, *, is_training=False, prng_key=None, state=None):
        """Apply adaptation to encoder features.

        Args:
            features: Encoder features to adapt.
            is_training: Whether the model is in training mode.
            prng_key: PRNG key for stochastic operations.
            state: State for the adapter.

        Returns:
            Adapted encoder features.
        """
        cfg = self.config
        if not cfg.adapt_encoder:
            return features

        # Use functional API if state and prng_key are provided
        if state is not None and prng_key is not None:
            outputs, _ = F(
                self.encoder_adapter,
                inputs=features,
                is_training=is_training,
                prng_key=prng_key,
                state=state,
            )
            return outputs

        # Fall back to direct call if no state/prng_key
        return self.encoder_adapter(features)

    def adapt_decoder_features(self, features, *, is_training=False, prng_key=None, state=None):
        """Apply adaptation to decoder features.

        Args:
            features: Decoder features to adapt.
            is_training: Whether the model is in training mode.
            prng_key: PRNG key for stochastic operations.
            state: State for the adapter.

        Returns:
            Adapted decoder features.
        """
        cfg = self.config
        if not cfg.adapt_decoder or not hasattr(self, "decoder_adapter"):
            return features

        # Use functional API if state and prng_key are provided
        if state is not None and prng_key is not None:
            outputs, _ = F(
                self.decoder_adapter,
                inputs=features,
                is_training=is_training,
                prng_key=prng_key,
                state=state,
            )
            return outputs

        # Fall back to direct call if no state/prng_key
        return self.decoder_adapter(features)

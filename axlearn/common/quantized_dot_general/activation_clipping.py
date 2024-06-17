# Copyright © 2024 Apple Inc.

"""Defines BaseActivationClippingLayer and its sub-classes.

Implements activation clipping needed for int8 quantized dot_general.
"""
from typing import Union

from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


class BaseActivationClippingLayer(BaseLayer):
    """Base class for activation clipping.

    This is useful for int8 quantization to control outlandishly large outliers in activation.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Config for Activation Clipping layers"""

        # Toggles clipping summary.
        clipping_summary: bool = False

    def _maybe_add_clipping_summary(
        self, x: Tensor, clipping_max_abs: Union[Tensor, float]
    ) -> None:
        """Adds summary for activation clipping depending on config.

        This summary indicates how many values were clipped in tensor.

        Args:
            x: Tensor before clipping.
            clipping_max_abs: Positive floating point range for clipping.
                This could be directly from config, or inferred by the clipping method.

        """
        cfg = self.config
        if cfg.clipping_summary:
            out_of_range_mask: Tensor = jnp.greater(jnp.abs(x), clipping_max_abs).astype(
                jnp.bfloat16
            )
            out_of_range_mean: Tensor = jnp.mean(out_of_range_mask)
            total_element_count: float = float(jnp.size(out_of_range_mask))
            self.add_summary(
                "clipped_activation", WeightedScalar(out_of_range_mean, total_element_count)
            )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward function for clipping method.

        Input should be clipped to the range of
        [-self.config.clipping_max_abs, self.config.clipping_max_abs],
        or an optimal range as deemed by the clipping method.

        Args:
            x: Input activation to be clipped.

        Returns:
            Clipped activation.
        """
        raise NotImplementedError(type(self))


class HardActivationClippingLayer(BaseActivationClippingLayer):
    """Implements "hard" clipping with jax.numpy.clip.

    This is useful for int8 quantization to control outlandishly large outliers in activation.
    """

    @config_class
    class Config(BaseActivationClippingLayer.Config):
        # Clipping (usually) happens symmetrically, considering both quantization methods
        # implemented are also symmetrical. clipping_max_abs is a positive float number that
        # represents the largest possible absolute value in the activation after clipping.
        clipping_max_abs: Required[float] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.clipping_max_abs is None or cfg.clipping_max_abs <= 0:
            raise ValueError(
                f"clipping_max_abs must be a positive float "
                f"for {self.path()}. Found {cfg.clipping_max_abs}"
            )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Implements "hard" clipping with jax.numpy.clip.

        Args:
            x: Input activation to be clipped.

        Returns:
            Clipped activation.
        """
        cfg = self.config
        self._maybe_add_clipping_summary(x=x, clipping_max_abs=cfg.clipping_max_abs)
        return jnp.clip(x, -cfg.clipping_max_abs, cfg.clipping_max_abs)


class DummyActivationClippingLayer(BaseActivationClippingLayer):
    """Does not clip. Only adds summary."""

    @config_class
    class Config(BaseActivationClippingLayer.Config):
        # Clipping (usually) happens symmetrically, considering both quantization methods
        # implemented are also symmetrical. clipping_max_abs is a positive float number that
        # represents the largest possible absolute value in the activation after clipping.
        clipping_max_abs: Required[float] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.clipping_max_abs is None or cfg.clipping_max_abs <= 0:
            raise ValueError(
                f"clipping_max_abs must be a positive float "
                f"for {self.path()}. Found {cfg.clipping_max_abs}"
            )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Does not clip. Only adds summary.

        Args:
            x: Input activation.

        Returns:
            Input activation.
        """
        cfg = self.config
        self._maybe_add_clipping_summary(x=x, clipping_max_abs=cfg.clipping_max_abs)
        return x


class TanhActivationClippingLayer(BaseActivationClippingLayer):
    """Implements "soft" clipping with jax.numpy.tanh.

    This is useful for int8 quantization to control outlandishly large outliers in activation.
    """

    @config_class
    class Config(BaseActivationClippingLayer.Config):
        # Clipping (usually) happens symmetrically, considering both quantization methods
        # implemented are also symmetrical. clipping_max_abs is a positive float number that
        # represents the largest possible absolute value in the activation after clipping.
        clipping_max_abs: Required[float] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.clipping_max_abs is None or cfg.clipping_max_abs <= 0:
            raise ValueError(
                f"clipping_max_abs must be a positive float "
                f"for {self.path()}. Found {cfg.clipping_max_abs}"
            )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Implements "soft" clipping with jax.numpy.tanh.

        Args:
            x: Input activation to be clipped.

        Returns:
            Clipped activation.
        """
        cfg = self.config
        self._maybe_add_clipping_summary(x=x, clipping_max_abs=cfg.clipping_max_abs)
        return jnp.tanh(x / cfg.clipping_max_abs) * cfg.clipping_max_abs

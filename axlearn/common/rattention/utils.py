"""Utilities for RAttention."""

from typing import Optional

import jax
from jax import numpy as jnp

from axlearn.common.base_layer import ParameterSpec
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.layers import BaseNormalizationLayer, compute_mean_square_with_segment_ids
from axlearn.common.utils import Tensor


class GroupRMSNorm(BaseNormalizationLayer):
    """GroupRMSNorm provides group-wise RMS normalization for inputs.

    This is a specialized version of GroupNorm for RAttention, where both the input and output
    can have the shape [batch_size, seq_length, num_groups, input_dim], where `num_groups`
    corresponds to the number of heads and `input_dim` corresponds to head dimension. The original
    GroupNorm is designed for inputs/outputs shape [batch_size, seq_length, num_groups * input_dim]
    where the extra reshape adds overhead.
    """

    @config_class
    class Config(BaseNormalizationLayer.Config):
        """Configures GroupNorm."""

        # The number of groups.
        num_groups: Required[int] = REQUIRED
        # The epsilon.
        eps: float = 1e-8
        # Cast input to this dtype for the 'forward' call. If None, do not cast.
        forward_dtype: Optional[jnp.dtype] = jnp.float32

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.num_groups, cfg.input_dim], mesh_axes=(None, None)),
        }

    def forward(self, x: Tensor, *, segment_ids: Optional[Tensor] = None) -> Tensor:
        """Applies group normalization.

        Args:
            x: inputs tensor of shape [batch_size, seq_length, num_groups, input_dim]
            segment_ids: An int Tensor of shape [batch_size, seq_len].

        Returns:
            Tensor of the same shape as x.
        """
        cfg = self.config
        orig_dtype = x.dtype
        if cfg.forward_dtype is not None:
            x = x.astype(cfg.forward_dtype)
        reduction_axis = [x.ndim - 1]
        msquare = compute_mean_square_with_segment_ids(
            x=x,
            segment_ids=segment_ids,
            reduction_axis=reduction_axis,
        )
        x = x * jax.lax.rsqrt(msquare + cfg.eps)
        x = x * self.parameters["scale"]
        return x.astype(orig_dtype)

# Copyright © 2026 Apple Inc.

"""Base class and helpers for streaming layers."""

import math
from typing import Sequence

import jax.numpy as jnp

from axlearn.common.module import Module, nowrap
from axlearn.common.utils import Nested, Tensor


class StreamingBase(Module):
    """Abstract base class for streaming layers.

    Subclasses must implement `init_states` and `extend_step`.
    """

    Config = Module.Config

    @classmethod
    def in_stride(cls, cfg: Config) -> int:
        """Returns input stride.

        Examples:
        * Conv with stride=2 returns 2.
        * ConvTranspose with stride=2 returns 1.

        Used to determine how many inputs are required for a single output in `extend_step`.

        Note: not normalized by gcd(in_stride, out_stride); reflects the encoder's compression
        or decoder's dilation factor.
        """
        del cfg
        return 1

    @classmethod
    def out_stride(cls, cfg: Config) -> int:
        """Returns output stride.

        Examples:
        * Conv with stride=2 returns 1.
        * ConvTranspose with stride=2 returns 2.
        """
        del cfg
        return 1

    @classmethod
    def segment_pad(cls, cfg: Config) -> int:
        """Minimum padding length required between different `segment_ids`.

        Convolution layers need `max(conv_padding_left, conv_padding_right)` padding to prevent
        information leaking across segment boundaries. Attention layers need 0, since segment
        boundaries are enforced by `segment_ids`.

        For composite layers with sequential children, accumulate strides to convert each child's
        segment_pad to input coordinates: `max(S1, S2 * I1, S3 * I1 * I2, ...)`, where `Si` is
        layer i's segment_pad and `Ii` is layer i's in_stride.
        """
        del cfg
        return 0

    @nowrap
    def init_states(self, *, batch_size: int, dtype: jnp.dtype) -> Nested[Tensor]:
        """Initializes states for streaming computation.

        Args:
            batch_size: batch size.
            dtype: dtype for the decoding cache.

        Returns:
            Initialized streaming states.
        """
        raise NotImplementedError()

    def extend_step(
        self, *, cached_states: Nested[Tensor], input_data: Nested[Tensor], is_prefill: bool = False
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """Advances one streaming step.

        Args:
            cached_states: states from `init_states` or a previous `extend_step` call.
            input_data: input matching `forward`'s `Args` section.
            is_prefill: True for prefill mode, False for extend-step mode.

        Returns:
            updated_states: updated streaming states.
            output: same structure as `forward`'s output.
        """
        raise NotImplementedError()


def compute_encoder_segment_pad(layer_cfgs: Sequence[StreamingBase.Config]) -> int:
    """Computes segment_pad for sequential encoder (downsampling) layers.

    Formula: `max(S1, S2 * I1, S3 * I1 * I2, ...)`, where `Si` is layer i's segment_pad and `Ii`
    is layer i's in_stride.
    """
    segment_pad, cum_stride = 0, 1
    for cfg in layer_cfgs:
        segment_pad = max(segment_pad, cfg.klass.segment_pad(cfg) * cum_stride)
        cum_stride *= cfg.klass.in_stride(cfg)
    return segment_pad


def compute_decoder_segment_pad(layer_cfgs: Sequence[StreamingBase.Config]) -> int:
    """Computes segment_pad for sequential decoder (upsampling) layers.

    Formula: `max(S1, ceil(S2 / O1), ceil(S3 / (O1 * O2)), ...)`, where `Si` is layer i's
    segment_pad and `Oi` is layer i's out_stride.
    """
    segment_pad, cum_out_stride = 0, 1
    for cfg in layer_cfgs:
        segment_pad = max(segment_pad, math.ceil(cfg.klass.segment_pad(cfg) / cum_out_stride))
        cum_out_stride *= cfg.klass.out_stride(cfg)
    return segment_pad


def next_segment_pos(current_len: int, *, segment_pad: int = 0, stride: int = 1) -> tuple[int, int]:
    """Computes the next valid segment start position.

    Returns the smallest position satisfying:
    1. `next_segment_pos % stride == 0` (stride alignment ensures the next segment produces the
       same output as if it started a new batch).
    2. `next_segment_pos - current_len >= segment_pad`.

    Args:
        current_len: length of the current segment.
        segment_pad: minimum padding required between segments.
        stride: alignment stride; the next position must be divisible by this.

    Returns:
        next_segment_pos: the next valid segment start position.
        effective_segment_pad: the actual padding (`next_segment_pos - current_len`).
    """
    min_pos = current_len + segment_pad
    next_pos = math.ceil(min_pos / stride) * stride
    effective_pad = next_pos - current_len
    return next_pos, effective_pad

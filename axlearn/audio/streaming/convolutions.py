# Copyright © 2026 Apple Inc.

"""Streaming variants of axlearn.common.convolution layers."""

import math
from typing import Optional

import chex
import jax.numpy as jnp

from axlearn.audio.streaming.streaming_base import StreamingBase
from axlearn.common.config import config_class
from axlearn.common.convolution import (
    Conv1D,
    Conv1DTranspose,
    Conv2DWith1DPadding,
    compute_conv_paddings,
    compute_conv_transpose_paddings,
    conv_explicit_padding,
    conv_transpose_explicit_padding,
)
from axlearn.common.ein_ops import rearrange
from axlearn.common.module import Module, nowrap
from axlearn.common.utils import Nested, Tensor, safe_not


class CausalConv2DWith1DPadding(Conv2DWith1DPadding, StreamingBase):
    """Causal `Conv2DWith1DPadding` with `extend_step`."""

    Config = Conv2DWith1DPadding.Config

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.padding = "CAUSAL"
        return cfg

    def __init__(self, cfg: Config, *, parent: Module | None):
        super().__init__(cfg, parent=parent)

        if cfg.padding != "CAUSAL":
            raise ValueError(f"Only CAUSAL padding is supported (got {cfg.padding}).")

    @classmethod
    def _cache_size(cls, cfg) -> int:
        time_padding = cls._time_padding(cfg)
        return time_padding[0]

    @classmethod
    def _time_padding(cls, cfg) -> tuple[int, int]:
        dilation = cfg.dilation or (1,) * len(cfg.window)
        time_padding = conv_explicit_padding(
            window=cfg.window[:1],
            strides=cfg.strides[:1],
            padding=cfg.padding,
            dilation=dilation[:1],
        )
        return time_padding[0]

    @classmethod
    def in_stride(cls, cfg: Config) -> int:
        return cfg.strides[0]

    @classmethod
    def segment_pad(cls, cfg: Config) -> int:
        time_padding = cls._time_padding(cfg)
        return max(time_padding)

    @nowrap
    # pylint: disable-next=arguments-differ
    def init_states(
        self, *, batch_size: int, feature_dim: int, dtype: jnp.dtype = jnp.float32
    ) -> Nested[Tensor]:
        """Initializes states for streaming computation.

        Args:
            batch_size: batch size.
            feature_dim: frequency dimension.
            dtype: dtype for the decoding cache.

        Returns:
            Initialized streaming states.
        """
        cfg = self.config
        cache_len = self._cache_size(cfg)
        shape = (batch_size, cache_len, feature_dim, cfg.input_dim)
        return dict(
            x=jnp.zeros(shape, dtype=dtype),
            paddings=jnp.zeros(shape[:2], dtype=jnp.bool),
        )

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        input_data: Nested[Tensor],
        is_prefill: bool = False,
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """Advances one streaming step.

        Args:
            cached_states: states from `init_states` or a previous `extend_step` call.
            input_data: dict with `x` [BTFD] and `paddings` [BT]. `T` must be a multiple of
                the time stride.
            is_prefill: True for prefill mode, False for extend-step mode.

        Returns:
            updated_states: updated streaming states.
            output_data: dict with `x` [BTFD] and `paddings` [BT].
        """
        del is_prefill
        cfg = self.config
        stride = cfg.strides[0]
        x, paddings = input_data["x"], input_data["paddings"]
        prev_context = cached_states["x"]
        prev_paddings = cached_states["paddings"]
        cache_len = self._cache_size(cfg)
        batch, time_steps = x.shape[:2]

        chex.assert_shape(x, (None, None, None, cfg.input_dim))
        chex.assert_shape(prev_context, (batch, cache_len, x.shape[2], x.shape[3]))
        chex.assert_shape(prev_paddings, (batch, cache_len))

        if time_steps % stride:
            raise ValueError(
                f"The number of time steps {time_steps} must be divisible by the stride {stride}."
            )

        x = x * rearrange(safe_not(paddings), "b s -> b s 1 1")
        context = jnp.concat([prev_context, x], axis=1)
        context_paddings = jnp.concat([prev_paddings, paddings], axis=1)
        context = context * safe_not(rearrange(context_paddings, "b s -> b s 1 1"))

        # Past context is explicit, so the convolution must not use left padding.
        dilation = cfg.dilation or (1,) * len(cfg.window)
        step_padding = ((0, 0),) + conv_explicit_padding(
            window=cfg.window, strides=cfg.strides, padding="CAUSAL", dilation=dilation
        )[1:]
        output = self._conv(context, strides=cfg.strides, padding=step_padding, dilation=dilation)

        output_paddings = self.conv_paddings(paddings)
        output = output * rearrange(safe_not(output_paddings), "b s -> b s 1 1")

        next_context = context[:, -cache_len:] if cache_len > 0 else prev_context
        next_paddings = context_paddings[:, -cache_len:] if cache_len > 0 else prev_paddings
        updated_states = dict(x=next_context, paddings=next_paddings)
        return updated_states, dict(x=output, paddings=output_paddings)


class CausalConv1D(Conv1D, StreamingBase):
    """Causal `Conv1D` with `extend_step`."""

    Config = Conv1D.Config

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.padding = "CAUSAL"
        return cfg

    @classmethod
    def _cache_size(cls, cfg) -> int:
        time_padding = cls._time_padding(cfg)
        return time_padding[0]

    @classmethod
    def _time_padding(cls, cfg) -> tuple[int, int]:
        dilation = cfg.dilation or 1
        time_padding = conv_explicit_padding(
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(dilation,),
        )
        return time_padding[0]

    @classmethod
    def in_stride(cls, cfg: Config) -> int:
        return cfg.strides

    @classmethod
    def segment_pad(cls, cfg: Config) -> int:
        time_padding = cls._time_padding(cfg)
        return max(time_padding)

    @nowrap
    def init_states(self, *, batch_size: int, dtype: jnp.dtype = jnp.float32) -> Nested[Tensor]:
        """Initializes states for streaming computation.

        Args:
            batch_size: batch size.
            dtype: dtype for the decoding cache.

        Returns:
            Initialized streaming states.
        """
        cfg = self.config
        cache_len = self._cache_size(cfg)
        shape = (batch_size, cache_len, cfg.input_dim)
        return dict(x=jnp.zeros(shape, dtype=dtype))

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        input_data: Tensor,
        is_prefill: bool = False,
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """Advances one streaming step.

        Args:
            cached_states: states from `init_states` or a previous `extend_step` call.
            input_data: a Tensor of shape [BTD].
            is_prefill: True for prefill mode, False for extend-step mode.

        Returns:
            updated_states: updated streaming states.
            output: a Tensor of shape [BTD].
        """
        del is_prefill
        cfg = self.config
        stride = cfg.strides
        x = input_data
        prev_context = cached_states["x"]
        cache_len = self._cache_size(cfg)
        batch, time_steps = x.shape[:2]
        chex.assert_shape(x, (None, None, cfg.input_dim))
        chex.assert_shape(prev_context, (batch, cache_len, cfg.input_dim))

        if time_steps % stride:
            raise ValueError(
                f"The number of time steps {time_steps} must be divisible by the stride {stride}."
            )

        context = jnp.concat([prev_context, x], axis=1)

        # Past context is explicit, so the convolution must not use left padding.
        dilation = cfg.dilation or 1
        conv_padding = conv_explicit_padding(
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding="VALID",
            dilation=(dilation,),
        )
        output = self._conv(
            context, strides=(cfg.strides,), padding=conv_padding, dilation=(dilation,)
        )

        next_context = context[:, -cache_len:] if cache_len > 0 else prev_context
        chex.assert_shape(next_context, (batch, cache_len, cfg.input_dim))
        updated_states = dict(x=next_context)
        return updated_states, output


class CausalConv1DWithPadding(CausalConv1D):
    """Causal `Conv1DWithPadding` with `extend_step`.

    Inherits `CausalConv1D` rather than `Conv1DWithPadding` to minimize change.
    """

    @config_class
    class Config(CausalConv1D.Config):
        """Configures CausalConv1DWithPadding."""

        # An optional integer in [left_time_padding, window - right_time_padding) specifying the
        # anchor position within the convolution window used to determine output paddings:
        # the output token is valid iff the input at the anchor position of the corresponding
        # window is valid. Defaults to left time padding. See Conv2DWith1DPadding for details.
        anchor: Optional[int] = None

    # We add a kwargs "paddings" to the forward method.
    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor, *, paddings: Tensor) -> tuple[Tensor, Tensor]:
        """Computes convolution outputs and paddings.

        Args:
            x: a Tensor of shape [BTD].
            paddings: 0/1 Tensor of shape [BT].

        Returns:
            output: a Tensor of shape [BTD].
            paddings: 0/1 Tensor of shape [BT].
        """
        chex.assert_rank(x, paddings.ndim + 1)
        x = x * safe_not(paddings)[..., None]
        output = super().forward(x)
        output_paddings = self.conv_paddings(paddings)
        output = output * rearrange(safe_not(output_paddings), "b s -> b s 1")
        return output, output_paddings

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        input_data: Nested[Tensor],
        is_prefill: bool = False,
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """Advances one streaming step.

        Args:
            cached_states: states from `init_states` or a previous `extend_step` call.
            input_data: dict with `x` [BTD] and `paddings` [BT].
            is_prefill: True for prefill mode, False for extend-step mode.

        Returns:
            updated_states: updated streaming states.
            output_data: dict with `x` [BTD] and `paddings` [BT].
        """
        del is_prefill
        x, paddings = input_data["x"], input_data["paddings"]
        assert x.shape[:2] == paddings.shape
        x = x * rearrange(safe_not(paddings), "b s -> b s 1")
        updated_states, output = super().extend_step(cached_states=cached_states, input_data=x)

        output_paddings = self.conv_paddings(paddings)
        output = output * rearrange(safe_not(output_paddings), "b s -> b s 1")
        return updated_states, dict(x=output, paddings=output_paddings)

    @nowrap
    def conv_paddings(self, paddings: Tensor) -> Tensor:
        """Computes output paddings (or segment_ids) given input paddings (or segment_ids).

        Args:
            paddings: 0/1 paddings or segment_ids int Tensor of shape [BT].

        Returns:
            output_paddings: 0/1 paddings or segment_ids int Tensor of shape [BT].
        """
        cfg = self.config
        return compute_conv_paddings(
            paddings,
            window=cfg.window,
            stride=cfg.strides,
            conv_padding=cfg.padding,
            dilation=cfg.dilation,
            anchor=cfg.anchor,
        )


############################## Transposed Convolution ##############################################


class CausalConv1DTranspose(Conv1DTranspose, StreamingBase):
    """Causal `Conv1DTranspose` with `extend_step`."""

    Config = Conv1DTranspose.Config

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.padding = "CAUSAL"
        return cfg

    @classmethod
    def cache_size(cls, cfg) -> int:
        pad_left, _ = cls._time_padding(cfg)
        # pad_left is in output coordinates; convert to input coordinates by dividing by stride.
        cache_size = math.ceil(pad_left / cls.out_stride(cfg))
        return cache_size

    @classmethod
    def _time_padding(cls, cfg) -> tuple[int, int]:
        return conv_transpose_explicit_padding(
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(cfg.dilation,),
        )[0]

    @classmethod
    def out_stride(cls, cfg: Config) -> int:
        return cfg.strides

    @classmethod
    def segment_pad(cls, cfg: Config) -> int:
        time_padding = cls._time_padding(cfg)
        # time_padding is in output coordinates.
        # Example: w=4, s=2, segment_ids=[1,1,P,2], padding=(3,1), segment_pad=1.
        # P (len(P) = segment_pad) separates two segments to prevent cross-segment leakage.
        #   segment_ids:  * * *|1 * 1 * P * 2|*
        #              pad|<->|
        #                 * * * 1
        #                   * * 1 *
        #                     * 1 * 1
        #                       1 * 1 *
        #                         * 1 * P
        #                           1 * P *
        #                             * P * 2
        #                               P * 2 *
        #
        # The strided zeros (*) between input samples also act as padding.
        # So effective_sed_pad = max(max(time_padding) - (out_stride - 1), 0),
        # and segment_pad = ceil(effective_sed_pad / out_stride). Simplifies to:
        return max(time_padding) // cls.out_stride(cfg)

    @nowrap
    def init_states(self, *, batch_size: int, dtype: jnp.dtype = jnp.float32) -> Nested[Tensor]:
        """Initializes states for streaming computation.

        Args:
            batch_size: batch size.
            dtype: dtype for the decoding cache.

        Returns:
            Initialized streaming states.
        """
        cfg = self.config
        cache_len = self.cache_size(cfg)
        shape = (batch_size, cache_len, cfg.input_dim)
        return dict(x=jnp.zeros(shape, dtype=dtype), paddings=jnp.ones(shape[:2], dtype=jnp.bool))

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        input_data: Nested[Tensor],
        is_prefill: bool = False,
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """Advances one streaming step.

        Args:
            cached_states: states from `init_states` or a previous `extend_step` call.
            input_data: dict with `x` [BTD] and optional `paddings` [BT].
            is_prefill: True for prefill mode, False for extend-step mode.

        Returns:
            updated_states: updated streaming states.
            output: dict with `x` [BTD] and optional `paddings` [BT].
        """
        del is_prefill
        cfg = self.config
        x = input_data["x"]
        prev_context = cached_states["x"]
        cache_len = int(self.cache_size(cfg))
        batch, time_steps = x.shape[:2]
        chex.assert_shape(x, (None, None, cfg.input_dim))
        chex.assert_shape(prev_context, (batch, cache_len, cfg.input_dim))

        context = jnp.concat([prev_context, x], axis=1)

        if "paddings" in input_data:
            paddings = input_data["paddings"]
            prev_paddings = cached_states["paddings"]
            chex.assert_shape(paddings, (batch, time_steps))
            chex.assert_shape(prev_paddings, (batch, cache_len))
            context_paddings = jnp.concat([prev_paddings, paddings], axis=1)
            context = context * rearrange(safe_not(context_paddings), "b s -> b s 1")

        # Past context is explicit, so the convolution must not use left padding.
        pad_left, pad_right = self._time_padding(cfg)
        conv_padding = ((0, pad_right),)
        pad_complete = pad_left % cfg.strides == 0
        output_len = self.output_shape(input_shape=x.shape)[1]
        output = self._conv(
            context, strides=(cfg.strides,), padding=conv_padding, dilation=(cfg.dilation,)
        )
        if not pad_complete:
            output = output[:, -output_len:]
        outputs = dict(x=output)
        updated_states = cached_states.copy()
        next_context = context[:, -cache_len:] if cache_len > 0 else prev_context
        updated_states["x"] = next_context

        if pad_complete:
            anchor = pad_left
        else:
            # For window=4, stride=2:
            # - `pad_left` = 3 (window - 1).
            # - `cache_len` = 2 = ceil(pad_left / stride).
            #
            # Given input x = [0, 0, 1] and a past context = [1, 1], the concat context becomes
            # [1 1 | 0 0 1]. The CORRECT dilated input is [* 1 * | 0 * 0 * 1], but ConvTranspose
            # expands the concat context into the dilated input space:
            # [1 * 1 * | 0 * 0 * 1]
            #        ^ anchor (built-in anchor, incorrect)
            #            ^ desired anchor (correct alignment)
            #  ^ compromising anchor (adjusted to fit within constraints)
            #
            # The built-in anchor points to the wrong position, misaligning the output. The
            # desired shift exceeds the window size, so we compromise to a position inside the
            # window and stride one more window to capture the final component.
            dilated_pad = cache_len * cfg.strides
            redundant_pad = dilated_pad - pad_left
            # desired_anchor = pad_left + redundant_pad, but exceeds window size.
            # pad_left + redundant_pad - window = redundant_pad - 1 because pad_left = w - 1.
            anchor = redundant_pad - 1  # compromising anchor.
            pad_total = pad_left + pad_right
            conv_padding = ((0, pad_total - anchor),)

        if "paddings" in input_data:
            output_paddings = compute_conv_transpose_paddings(
                context_paddings,
                window=cfg.window,
                stride=cfg.strides,
                conv_padding=conv_padding,
                dilation=cfg.dilation,
                anchor=anchor,
            )
            if not pad_complete:
                output_paddings = output_paddings[:, -output_len:]
            output = output * rearrange(safe_not(output_paddings), "b s -> b s 1")
            outputs.update(dict(x=output, paddings=output_paddings))
            updated_states["paddings"] = (
                context_paddings[:, -cache_len:] if cache_len > 0 else prev_paddings
            )

        return updated_states, outputs

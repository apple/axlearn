# Copyright © 2024 Apple Inc.
# pylint: disable=too-many-lines
"""Convolution layers."""

from collections.abc import Sequence
from typing import Literal, Optional, Union

import chex
import jax
from jax import numpy as jnp

from axlearn.common import ein_ops
from axlearn.common.base_layer import BaseLayer, FactorizationSpec, ParameterSpec
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import nowrap
from axlearn.common.param_init import FanAxes
from axlearn.common.utils import Tensor

# The padding type for jax.lax.conv_general_dilated API. Either the strings ‘SAME’, or ‘VALID’, or
# 'CAUSAL' or a sequence of n (low, high) integer pairs that give the padding to apply before and
# after each spatial dimension. The number of tuple is 1 for NHC, 2 for NHWC and 3 for NHWDC.
ConvPaddingType = Union[str, Sequence[tuple[int, int]]]

SUPPORT_CONV_PADDING = ("SAME", "VALID", "CAUSAL")


# TODO(yuanliu939): Make this take `BaseConv.Config` directly.
def _check_conv_cfg(
    *,
    window: Sequence[int],
    strides: Sequence[int],
    padding: ConvPaddingType,
    dilation: Optional[Sequence[int]],
    input_dim: int,
    output_dim: int,
    num_input_dim_groups: int,
):
    if any(w < 1 for w in window):
        raise ValueError(f"window ({window}) must be a positive integer.")

    if any(s < 1 for s in strides):
        raise ValueError(f"strides ({strides}) must be a positive integer.")

    if isinstance(padding, str):
        if padding not in SUPPORT_CONV_PADDING:
            raise ValueError(f"{padding} padding is not supported.")
    else:
        padding_flattened = jax.tree.leaves(padding)
        if any(p < 0 for p in padding_flattened):
            raise ValueError("Negative padding is not supported")

    if dilation is not None and any(d < 1 for d in dilation):
        raise ValueError(f"dilation ({dilation}) must be a positive integer.")

    if input_dim % num_input_dim_groups != 0:
        raise ValueError(
            f"input_dim ({input_dim}) must be divisible by "
            f"num_input_dim_groups({num_input_dim_groups})."
        )

    if output_dim % num_input_dim_groups != 0:
        raise ValueError(
            f"output_dim ({output_dim}) must be divisible by "
            f"num_input_dim_groups({num_input_dim_groups})."
        )


class BaseConv(BaseLayer):
    """Base class for convolution layers."""

    @config_class
    class Config(BaseLayer.Config):
        """Config class for BaseConv."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        # The number of groups in which the input is split along the channel axis.
        # input_dim and output_dim must both be divisible by num_input_dim_groups. For example,
        # - At num_input_dim_groups=1, all inputs are convolved to all outputs (the default).
        # - At num_input_dim_groups=2, the operation is equivalent to concatenating two conv layers
        #   side by side, each seeing half the input and producing half the output channels.
        # - At num_input_dim_groups=input_dim, each input channel is convolved with its own
        #   set of filters (of size output_dim / input_dim); if further output_dim == K * input_dim,
        #   where K is a positive integer, the operation is also known as a "depthwise convolution".
        num_input_dim_groups: int = 1

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if not name.endswith("weight"):
            return None
        if len(parameter_spec.shape) < 2:
            raise NotImplementedError(
                "Default _compute_fan_axes requires weight parameters to have at least 2 axes "
                f"shape({name}) = {parameter_spec.shape}"
            )
        # All other axes represent receptive field.
        return FanAxes(in_axis=-2, out_axis=-1)


# Copied from jax.lax._dilate_shape
# https://github.com/jax-ml/jax/blob/2d78b172266870bd755b039f6faa2056a51930f9/jax/_src/lax/lax.py#L5763
def conv_dilate_window(*, window: Sequence[int], dilation: Optional[Sequence[int]]):
    """Returns dilated effective window size.

    Args:
        window: convolution window.
        dilation: convolution dilation.

    Returns:
        The dilated effective window size.
    """
    if dilation is None or all(d == 1 for d in dilation):
        return window

    return tuple(max(1 + d * (w - 1), 0) for w, d in zip(window, dilation))


# Copied from subroutine in jax.lax.reduce_window.
# Extend lax.padtype_to_pads for CAUSAL.
def conv_explicit_padding(
    *,
    window: Sequence[int],
    strides: Sequence[int],
    padding: ConvPaddingType,
    dilation: Optional[Sequence[int]] = None,
) -> ConvPaddingType:
    """Returns the explicit padding for "SAME", "VALID", and "CAUSAL" modes.

    Each mode follows the formulas below:
    * SAME: (pad_total//2, pad_total - pad_total//2) s.t. pad_total = window-1
    * VALID: (0, 0)
    * CAUSAL: (window - stride, stride - 1)

    Note: In the above equation, `window` will be replaced with `dilate_window` when dilation > 1.
        dilate_window = (window - 1) * dilation + 1. Check conv_dilate_window()

    For example, window=5 and stride=2,
    * SAME: padding = (2, 2)
                pad|           |pad
    paddings:   0 0|0 0 0 0 1 1|1 1
                |___^___|
                    |___^___|
                        |___^___|

    * VALID: padding = (0, 0)
               |           |
    paddings:  |0 0 0 0 1 1|
               |^_______|

    * CAUSAL: padding = (3, 1)
                pad  |           |pad
    paddings:   0 0 0|0 0 0 0 1 1|1
                |_____^_|
                    |_____^_|
                        |_____^_|


    For example, window=5, stride=2 and dilation=2
        -> dilate_window = 9 (== (window-1)*dilation + 1) and pad_total = 8
    * SAME: padding = (4, 4)
                    pad|                   |pad
    paddings:   0 0 0 0|0 0 0 0 0 0 0 0 1 1|1 1 1 1
                |_______^_______|
                    |_______^_______|
                        |_______^_______|
                            |_______^_______|
                                |_______^_______|

    * VALID: padding = (0, 0)
                |                   |pad
    paddings:   |0 0 0 0 0 0 0 0 1 1|
                |^_______________|

    * CAUSAL: padding = (7, 1)
                        pad  |                   |pad
    paddings:   0 0 0 0 0 0 0|0 0 0 0 0 0 0 0 1 1|1
                |_____________^_|
                    |_____________^_|
                        |_____________^_|
                            |_____________^_|
                                |_____________^_|

    For "CAUSAL", the first component is time and treated as "CAUSAL", while the remaining
    components are handled with "SAME" padding.

    Args:
        window: convolution window.
        strides: convolution strides.
        padding: convolution padding.
        dilation: convolution dilation.

    Returns:
        The padding tuple.

    Raises:
        ValueError: If padding is not supported.
    """
    if not isinstance(padding, str):
        return padding
    window = conv_dilate_window(window=window, dilation=dilation)

    def same_padding(window):
        pad_total = tuple(w - 1 for w in window)
        pad_left = tuple(pt // 2 for pt in pad_total)
        pad_right = tuple(pt - pl for pt, pl in zip(pad_total, pad_left))
        return tuple(zip(pad_left, pad_right))

    if padding == "SAME":
        return same_padding(window)
    elif padding == "VALID":
        return ((0, 0),) * len(window)
    elif padding == "CAUSAL":
        causal_padding = ((window[0] - strides[0], strides[0] - 1),)
        if len(window) > 1:
            causal_padding += same_padding(window[1:])
        return causal_padding
    else:
        raise ValueError(f"{padding} padding is not supported.")


def conv_output_shape(
    in_shape: Sequence[Optional[int]],
    *,
    window: Sequence[int],
    strides: Sequence[int],
    padding: ConvPaddingType,
    dilation: Optional[Sequence[int]] = None,
) -> Sequence[int]:
    """Returns output size for convolution.

    Follow https://www.tensorflow.org/api_docs/python/tf/nn/convolution
    * SAME: ceil(in_size / stride)
    * VALID: ceil((in_size - (window - 1) * dilation) / stride)

    Args:
        in_shape: convolution lhs shape.
        window: convolution window.
        strides: convolution strides.
        padding: convolution padding.
        dilation: convolution dilation.

    Returns:
        The output shape.

    Raises:
        ValueError: If the length of in_shape, window, strides, and padding are not equal.
    """
    if len(in_shape) != len(window) or len(in_shape) != len(strides):
        raise ValueError(
            f"len(in_shape) = {len(in_shape)} must be equal to "
            f"len(window) = {len(window)} and len(strides) = {len(strides)}"
        )

    padding = conv_explicit_padding(
        window=window, strides=strides, padding=padding, dilation=dilation
    )
    pad_amount = tuple(sum(p) for p in padding)
    dilate_window = conv_dilate_window(window=window, dilation=dilation)

    def output_shape(in_shape: Optional[int], dilate_window: int, pad_amount: int, stride: int):
        if in_shape is None:
            return None
        numerator = max(in_shape + pad_amount - (dilate_window - 1), 0)
        # ceil(numerator / stride)
        return (numerator + stride - 1) // stride

    return tuple(map(output_shape, in_shape, dilate_window, pad_amount, strides))


def compute_conv_paddings(
    in_paddings: Tensor,
    *,
    window: int,
    stride: int,
    conv_padding: ConvPaddingType,
    dilation: Optional[int] = None,
    anchor: Optional[int] = None,
):
    """Compute output paddings w.r.t. conv_padding.

    The output paddings value is determined by the padding value at the anchor point in the
    window. If anchor is None, the default anchor point is the left time padding from conv
    padding config. See `Conv2DWith1DPadding.Config` in details.

    Args:
        in_paddings: A Tensor of shape [batch_size, seq_len].
        window: convolution window size of the time axis.
        stride: convolution stride size of the time axis.
        conv_padding: "SAME", "VALID", "CAUSAL" or ((left_time_padding, right_time_padding),)
        dilation: convolution dilation size of the time axis.
        anchor: an optional integer in the range of [left_time_padding, window - right_time_padding)
            that specifies the anchor position within the convolution window that is used to
            determine output paddings. Specifically, the output token is valid iff the input token
            at the anchor position of the corresponding window is valid.
            If None, anchor defaults to conv_padding[0] (i.e. left_time_padding).

    Returns:
        out_paddings: A Tensor of shape [batch_size, seq_len].

    Raises:
        ValueError: If anchor is not between left_time_padding and right_time_padding.
    """
    chex.assert_rank(in_paddings, 2)
    dilation = dilation or 1
    conv_padding = conv_explicit_padding(
        window=(window,), strides=(stride,), padding=conv_padding, dilation=(dilation,)
    )
    window = conv_dilate_window(window=(window,), dilation=(dilation,))[0]
    left_pad, right_pad = conv_padding[0]
    pad_total = window - 1

    if anchor is None:
        # valid_window = pad_total - left_pad - right_pad
        # anchor_global = valid_window // 2
        # anchor = anchor_global + left_pad
        anchor = left_pad
    elif not left_pad <= anchor < window - right_pad:
        raise ValueError(f"anchor ({anchor}) must in range [{left_pad}, {window - right_pad}).")

    # This is a method to avoid using jax.pad, by leveraging the property that the valid_window
    # is always within the input sequence.
    # Note: transform anchor from window frame to input sequence frame.
    start_index = anchor - left_pad
    valid_window = pad_total - left_pad - right_pad
    valid_window_right_pad = valid_window - start_index
    seq_len = in_paddings.shape[1]
    limit_index = max(seq_len - valid_window_right_pad, start_index)
    if seq_len < start_index:
        start_index = 0
        limit_index = 0
    out_paddings = jax.lax.slice_in_dim(
        in_paddings, start_index=start_index, limit_index=limit_index, stride=stride, axis=1
    )
    return out_paddings


class Conv1D(BaseConv):
    """The 1D convolution layer.

    Kernel weights have the WIO layout and in the shape of (window, input_dim, output_dim).
    Both inputs and outputs will be in the NWC layout.
    """

    @config_class
    class Config(BaseConv.Config):
        """Configures Conv1D."""

        window: Required[int] = REQUIRED  # The convolution window.
        strides: int = 1  # The convolution strides.
        # Paddings: "SAME", "VALID", "CAUSAL", or (left, right).
        # For causal convolution, set padding to (window - 1, 0).
        padding: ConvPaddingType = ((0, 0),)
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.
        # The convolution dilation, indicating dilation factor applied to the weight. It is also
        # known as atrous convolution or dilated convolution. If None, assume 1.
        dilation: Optional[int] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        dilation = cfg.dilation or 1
        _check_conv_cfg(
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(dilation,),
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            num_input_dim_groups=cfg.num_input_dim_groups,
        )
        if cfg.padding not in SUPPORT_CONV_PADDING:
            left, right = cfg.padding[0]
            if any(p < 0 for p in (left, right)):
                raise NotImplementedError("Negative padding is not supported")
        params = dict(
            weight=ParameterSpec(
                shape=[cfg.window, cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim],
                mesh_axes=cfg.param_partition_spec,
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        dilation = cfg.dilation or 1
        conv_padding = conv_explicit_padding(
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(dilation,),
        )
        return self._conv(x=x, strides=(cfg.strides,), padding=conv_padding, dilation=(dilation,))

    def _conv(
        self,
        x: Tensor,
        *,
        strides: Sequence[int],
        padding: ConvPaddingType,
        dilation: Optional[Sequence[int]],
    ) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=strides,
            padding=padding,
            rhs_dilation=dilation,
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    @nowrap
    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 3:
            raise ValueError(f"We expect len(input_shape) = 3, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                f"cfg.input_dim = {cfg.input_dim}."
            )

        in_shape = input_shape[1:2]
        dilation = cfg.dilation or 1
        out_shape = conv_output_shape(
            in_shape,
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(dilation,),
        )
        return [input_shape[0], *out_shape, cfg.output_dim]


class Conv1DWithPadding(Conv1D):
    """The 1-D convolution with 1-D padding on the time axis."""

    @config_class
    class Config(Conv1D.Config):
        """Configures Conv1DWithPadding."""

        # An optional integer in the range of [left_time_padding, window - right_time_padding)
        # that specifies the anchor position within the convolution window that is used to
        # determine output paddings. Specifically, the output token is valid iff the input token
        # at the anchor position of the corresponding window is valid.
        # If None, defaults to left time padding. See Conv2DWith1DPadding more details.
        anchor: Optional[int] = None

    # We add a kwargs "paddings" to the forward method.
    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor, *, paddings: Tensor) -> tuple[Tensor, Tensor]:
        """Computes convolution outputs and paddings.

        Args:
            x: A Tensor of shape [batch_size, seq_len, frequency, input_dim].
            paddings: 0/1 boolean Tensor of shape [batch_size, seq_len].

        Returns:
            output: A Tensor of shape [batch_size, seq_len, frequency, output_dim].
            paddings: 0/1 boolean Tensor of shape [batch_size, seq_len].
        """
        cfg = self.config
        chex.assert_rank(x, paddings.ndim + 1)
        # Apply padding to the input.
        x = x * (1 - paddings[..., None])

        # Apply Conv1D.
        output = super().forward(x)

        # Compute paddings conv output.
        output_paddings = compute_conv_paddings(
            paddings,
            window=cfg.window,
            stride=cfg.strides,
            conv_padding=cfg.padding,
            dilation=cfg.dilation,
            anchor=cfg.anchor,
        )
        # Apply padding to the outputs.
        output = output * (1 - output_paddings[..., None])
        return output, output_paddings


# The accuracy of the output of this layer currently doesn't match that of PyTorch
# quite as closely as we would like. See layers_test.py:test_conv2d().
class Conv2D(BaseConv):
    """The 2-D convolution layer.

    Kernel weights have the HWIO layout and in the shape of (window[0], window[1], input_dim,
    output_dim). Both inputs and outputs will be in the NHWC layout.
    """

    @config_class
    class Config(BaseConv.Config):
        """Configures Conv2D."""

        window: tuple[int, int] = (1, 1)  # The convolution window.
        strides: tuple[int, int] = (1, 1)  # The convolution strides.
        # Paddings: "SAME", "VALID", "CAUSAL" or ((top, bottom), (left, right)).
        # Note: Sequence models use the first component to represent time.
        padding: ConvPaddingType = ((0, 0), (0, 0))
        # The convolution dilation. If None, assume all 1's.
        dilation: Optional[tuple[int, int]] = None
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            num_input_dim_groups=cfg.num_input_dim_groups,
        )
        params = dict(
            weight=ParameterSpec(
                shape=list(cfg.window)
                + [cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim],
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        conv_padding = conv_explicit_padding(
            window=cfg.window, strides=cfg.strides, padding=cfg.padding, dilation=cfg.dilation
        )
        return self._conv(x=x, strides=cfg.strides, padding=conv_padding, dilation=cfg.dilation)

    def _conv(
        self,
        x: Tensor,
        *,
        strides: Sequence[int],
        padding: ConvPaddingType,
        dilation: Optional[Sequence[int]],
    ) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=strides,
            padding=padding,
            rhs_dilation=dilation,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    @nowrap
    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 4:
            raise ValueError(f"We expect len(input_shape) = 4, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                f"cfg.input_dim = {cfg.input_dim}."
            )

        in_shape = input_shape[1:3]
        out_shape = conv_output_shape(
            in_shape,
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
        )
        return [input_shape[0], *out_shape, cfg.output_dim]


class Conv2DWith1DPadding(Conv2D):
    """The 2-D convolution with 1-D padding on the time axis."""

    @config_class
    class Config(Conv2D.Config):
        """Configures Conv2DWith1DPadding.

        The output paddings value is determined by the padding value at the anchor point in the
        window. If anchor is None, the default anchor point is the left time padding from conv
        padding config.

        For examples with window=5,
        1. "SAME" padding case,
            * padding=(2,2): (0 0 0 0 0)
            * anchor index is 2: (0 0 |0| 0 0)
                        pad  |           | pad
            paddings:     0 0|0 0 0 1 1 1|1 1
                          |___0___|
                            |___0___|
                              |___0___|
                                |___1___|
                                  |___1___|
                                    |___1___|

        2. "VALID" padding case,
            * padding=(0,0): (0 0 0 0 0)
            * anchor index is 0:  (|0| 0 0 0 0)
                    pad |           | pad
            paddings:   |0 0 0 1 1 1|
                        |0_______|
                          |0_______|

        3. The legacy "VALID" padding case,
            * padding=(0,0) and anchor=4: (0 0 0 0 0)
            * anchor index is 4:  (0 0 0 0 |0|)
                    pad |           | pad
            paddings:   |0 0 0 1 1 1|
                        |________1|
                          |________1|

        4. "CAUSAL" padding case,
            * padding=(4,0): (0 0 0 0 0)
            * anchor index is 4:  (0 0 0 0 |0|)
                        pad      |           | pad
            paddings:     0 0 0 0|0 0 0 1 1 1|
                          |_______0|
                            |_______0|
                              |_______0|
                                |_______1|
                                  |_______1|
                                    |_______1|

        5. "CAUSAL" with lookahead=1,
            * padding=(3, 1): (0 0 0 0 0)
            * anchor index is 3:  (0 0 0 |0| 0)
                        pad    |           | pad
            paddings:     0 0 0|0 0 0 1 1 1|1
                          |_____0_|
                            |_____0_|
                              |_____0_|
                                |_____1_|
                                  |_____1_|
                                    |_____1_|

        6. Arbitrary padding case,
            * padding=(2,1): (0 0 0 0 0)
            * anchor index is 2:  (0 0 |0| 0 0)
                        pad  |           | pad
            paddings:     0 0|0 0 0 1 1 1|1
                          |___0___|
                            |___0___|
                              |___0___|
                                |___1___|
                                  |___1___|
        """

        # An optional integer in the range of [left_time_padding, window - right_time_padding)
        # that specifies the anchor position within the convolution window that is used to
        # determine output paddings. Specifically, the output token is valid iff the input token
        # at the anchor position of the corresponding window is valid.
        # If None, defaults to left time padding.
        anchor: Optional[int] = None

    # We add a kwargs "paddings" to the forward method.
    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor, *, paddings: Tensor) -> tuple[Tensor, Tensor]:
        """Computes convolution outputs and paddings.

        Args:
            x: A Tensor of shape [batch_size, seq_len, frequency, input_dim].
            paddings: 0/1 boolean Tensor of shape [batch_size, seq_len].

        Returns:
            output: A Tensor of shape [batch_size, seq_len, frequency, output_dim].
            paddings: 0/1 boolean Tensor of shape [batch_size, seq_len].
        """
        cfg = self.config
        # Apply padding to the input.
        assert len(x.shape) == len(paddings.shape) + 2
        x = x * (1 - paddings[..., None, None])

        # Apply Conv2D.
        output = super().forward(x)
        # Compute paddings conv output.
        dilation = 1 if cfg.dilation is None else cfg.dilation[0]
        output_paddings = compute_conv_paddings(
            paddings,
            window=cfg.window[0],
            stride=cfg.strides[0],
            conv_padding=cfg.padding,
            dilation=dilation,
            anchor=cfg.anchor,
        )
        # Apply padding to the outputs.
        output = output * (1 - output_paddings[..., None, None])
        return output, output_paddings


class Conv3D(BaseConv):
    """The 3-D convolution layer.

    Kernel weights have the HWDIO layout and in the shape of (window[0], window[1],
    window[2], input_dim, output_dim). Both inputs and outputs will be in the NHWDC layout.
    """

    @config_class
    class Config(BaseConv.Config):
        """Configures Conv3D."""

        window: tuple[int, int, int] = (1, 1, 1)  # The convolution window.
        strides: tuple[int, int, int] = (1, 1, 1)  # The convolution strides.

        # Paddings: "SAME" or "VALID, or ((top, bottom), (left, right), (front, back))
        padding: ConvPaddingType = (
            (0, 0),
            (0, 0),
            (0, 0),
        )
        # The convolution dilation. If None, assume all 1's.
        dilation: Optional[tuple[int, int, int]] = None

        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            num_input_dim_groups=cfg.num_input_dim_groups,
        )
        params = dict(
            weight=ParameterSpec(
                shape=list(cfg.window)
                + [cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim],
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        conv_padding = conv_explicit_padding(
            window=cfg.window, strides=cfg.strides, padding=cfg.padding, dilation=cfg.dilation
        )
        return self._conv(x=x, strides=cfg.strides, padding=conv_padding, dilation=cfg.dilation)

    def _conv(
        self,
        x: Tensor,
        *,
        strides: Sequence[int],
        padding: ConvPaddingType,
        dilation: Optional[Sequence[int]],
    ) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=strides,
            padding=padding,
            rhs_dilation=dilation,
            dimension_numbers=("NHWDC", "HWDIO", "NHWDC"),
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    @nowrap
    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 5:
            raise ValueError(f"We expect len(input_shape) = 5, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                f"cfg.input_dim = {cfg.input_dim}."
            )

        in_shape = input_shape[1:4]
        out_shape = conv_output_shape(
            in_shape,
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
        )
        return [input_shape[0], *out_shape, cfg.output_dim]


############################## Transposed Convolution ##############################################


# Based on jax.lax.convolution._conv_transpose_padding, but ours is more intuitive.
def conv_transpose_explicit_padding(
    *,
    window: Sequence[int],
    strides: Sequence[int],
    padding: ConvPaddingType,
    dilation: Sequence[int],
) -> ConvPaddingType:
    """Convert str padding to tuple padding for conv_transpose.

    Each mode follows the formulas below,
    * SAME: (min(window-1, ceil((w+s-2)/2)), max(stride-1, floor((w+s-2)/2)))
        pad_total = window+stride-2
        when stride > window -> (window-1, stride-1)
    * VALID: (window-1, max(stride-1, window-1))
        pad_total = window+stride-2 + max(window-stride, 0)
        when stride > window -> (window-1, stride-1)
    * CAUSAL: (window-1, stride-1)
        pad_total = window+stride-2

    Note: output_size = input_size*stride - (window+stride-2) + pad_total
                      = input_size*stride  <- "SAME", "CAUSAL"
                      = input_size*stride + max(window-stride, 0)  <- "VALID"

    Note: In the above equation, `window` will be replaced with `dilate_window` when dilation > 1.
        dilate_window = (window - 1) * dilation + 1. Check conv_dilate_window()

    The following illustration demonstrates how Conv Transpose operates, assuming all kernel values
    are set to 1 for simplicity in showcasing output values.

    In the window=3 and stride=1 case, this function creates outputs as follows:
    * "SAME" padding=(1, 1)
                    pad|       |pad
        paddings:     0|0 0 1 1|0
                      0 0 0  -> 0
                        0 0 1  -> 1
                          0 1 1  -> 2
                            1 1 0  -> 2

    * "VALID" padding=(2, 2)
                    pad  |       |pad
        paddings:     0 0|0 0 1 1|0 0
                      0 0 0  -> 0
                        0 0 0  -> 0
                          0 0 1  -> 1
                            0 1 1  -> 2
                              1 1 0  -> 2
                                1 0 0  -> 1

    * "CAUSAL" padding=(2, 0)
                    pad  |       |pad
        paddings:     0 0|0 0 1 1|
                      0 0 0  -> 0
                        0 0 0  -> 0
                          0 0 1  -> 1
                            0 1 1  -> 2

    In the window=3 and stride=2 case, this function creates outputs as follows:
    * "SAME" padding=(2, 1)
                    pad  |             |pad
        paddings:     0 0|0 * 0 * 1 * 1|0
                      0 0 0  -> 0
                        0 0 0  -> 0
                          0 0 0  -> 0
                            0 0 0  -> 0
                              0 0 1  -> 1
                                0 1 0  -> 1
                                  1 0 1  -> 2
                                    0 1 0  -> 1

    * "VALID" padding=(2, 2)
                    pad  |             |pad
        paddings:     0 0|0 * 0 * 1 * 1|0 0
                      0 0 0  -> 0
                        0 0 0  -> 0
                          0 0 0  -> 0
                            0 0 0  -> 0
                              0 0 1  -> 1
                                0 1 0  -> 1
                                  1 0 1  -> 2
                                    0 1 0  -> 1
                                      1 0 0  -> 1

    * "CAUSAL" padding=(2, 1)
                    pad  |             |pad
        paddings:     0 0|0 * 0 * 1 * 1|0
                      0 0 0  -> 0
                        0 0 0  -> 0
                          0 0 0  -> 0
                            0 0 0  -> 0
                              0 0 1  -> 1
                                0 1 0  -> 1
                                  1 0 1  -> 2
                                    0 1 0  -> 1

    In the window=3 and stride=3 case, this function creates outputs as follows:
    * "SAME", "VALID" and "CAUSAL" padding=(2, 2)
                    pad  |                   |pad
        paddings:     0 0|0 * * 0 * * 1 * * 1|0 0
                      0 0 0  -> 0
                        0 0 0  -> 0
                          0 0 0  -> 0
                            0 0 0  -> 0
                              0 0 0  -> 0
                                0 0 0  -> 0
                                  0 0 1  -> 1
                                    0 1 0  -> 1
                                      1 0 0  -> 1
                                        0 0 1  -> 1
                                          0 1 0  -> 1
                                            1 0 0  -> 1

    In the window=3 and stride=4 case, this function creates outputs as follows:
    * "SAME", "VALID" and "CAUSAL" padding=(2, 3)
                    pad  |                         |pad
        paddings:     0 0|0 * * * 0 * * * 1 * * * 1|0 0 0
                      0 0 0  -> 0
                        0 0 0  -> 0
                          0 0 0  -> 0
                            0 0 0  -> 0
                              0 0 0  -> 0
                                0 0 0  -> 0
                                  0 0 0  -> 0
                                    0 0 0  -> 0
                                      0 0 1  -> 1
                                        0 1 0  -> 1
                                          1 0 0  -> 1
                                            0 0 0  -> 0
                                              0 0 1  -> 1
                                                0 1 0  -> 1
                                                  1 0 0  -> 1
                                                    0 0 0  -> 0
        Here is how to compute output_size, given the above example,
          1.          |_|  -(window-1)
          2.              |_______________________|  (input_size-1)*stride + 1
          3.          |_|                           |___|  + pad_total

        So, output_size = -(window-1) + (input_size-1)*stride + 1 + pad_total
                        = input_size*stride - (window+stride-2) + pad_total
                        = input_size*stride  <- "SAME", "CAUSAL"
                        = input_size*stride + max(window-stride, 0)  <- "VALID"

    OTHO, when dilation > 1, dilate_window = (window - 1) * dilation + 1.
    For example, when window=3 and dilation=2, dilate_window=5.

    In the stride=2 case, this function creates outputs as follows:
    * "SAME" padding=(3, 2)
                    pad    |             |pad
        paddings:     0 0 0|0 * 0 * 1 * 1|0 0
                      0 * 0 * 0  -> 0
                        0 * 0 * 0  -> 0
                          0 * 0 * 0  -> 0
                            0 * 0 * 1  -> 1
                              0 * 0 * 0  -> 0
                                0 * 1 * 1  -> 2
                                  0 * 0 * 0  -> 0
                                    1 * 1 * 0  -> 2

    * "VALID" padding=(4, 4)
                    pad      |             |pad
        paddings:     0 0 0 0|0 * 0 * 1 * 1|0 0 0 0
                      0 * 0 * 0  -> 0
                        0 * 0 * 0  -> 0
                          0 * 0 * 0  -> 0
                            0 * 0 * 0  -> 0
                              0 * 0 * 1  -> 1
                                0 * 0 * 0  -> 0
                                  0 * 1 * 1  -> 2
                                    0 * 0 * 0  -> 0
                                      1 * 1 * 0  -> 2
                                        0 * 0 * 0  -> 0
                                          1 * 0 * 0  -> 1

    * "CAUSAL" padding=(4, 1)
                    pad      |             |pad
        paddings:     0 0 0 0|0 * 0 * 1 * 1|0
                      0 * 0 * 0  -> 0
                        0 * 0 * 0  -> 0
                          0 * 0 * 0  -> 0
                            0 * 0 * 0  -> 0
                              0 * 0 * 1  -> 1
                                0 * 0 * 0  -> 0
                                  0 * 1 * 1  -> 2
                                    0 * 0 * 0  -> 0

    For "CAUSAL", the first component is time and treated as "CAUSAL", while the remaining
    components are handled with "SAME" padding.

    Args:
        window: convolution window.
        strides: transposed convolution strides. It's lhs_dilation, not window_stride.
        padding: convolution padding.
        dilation: convolution dilation, a.k.a rhs_dilation.

    Returns:
        The padding tuple.

    Raises:
        ValueError: If padding is not supported.
    """
    if not isinstance(padding, str):
        return padding

    window = conv_dilate_window(window=window, dilation=dilation)

    def same_padding(window, strides):
        pad_left = tuple(min(w - 1, (w + s - 1) // 2) for w, s in zip(window, strides))
        pad_right = tuple(max(s - 1, (w + s - 2) // 2) for w, s in zip(window, strides))
        return tuple(zip(pad_left, pad_right))

    if padding == "SAME":
        return same_padding(window, strides)
    elif padding == "VALID":
        pad_left = tuple(w - 1 for w in window)
        pad_right = tuple(max(s - 1, w - 1) for w, s in zip(window, strides))
        return tuple(zip(pad_left, pad_right))
    elif padding == "CAUSAL":
        causal_padding = ((window[0] - 1, strides[0] - 1),)
        if len(window) > 1:
            causal_padding += same_padding(window[1:], strides[1:])
        return causal_padding
    else:
        raise ValueError(f"{padding} padding is not supported.")


def conv_transpose_output_shape(
    in_shape: Sequence[Optional[int]],
    *,
    window: Sequence[int],
    strides: Sequence[int],
    padding: ConvPaddingType,
    dilation: Sequence[int],
) -> Sequence[int]:
    """Returns output size for conv transpose.

    Each mode follows the formulas below,
    * SAME: padding=(min(window-1, ceil((w+s-2)/2)), max(stride-1, floor((w+s-2)/2)))
        pad_total = window+stride-2
        output_size = input_size*stride
    * VALID: padding=(window-1, max(stride-1, window-1))
        pad_total = window+stride-2 + max(window-stride, 0)
        output_size = input_size*stride + max(window-stride, 0)
    * CAUSAL: padding=(window-1, stride-1)
        pad_total = window+stride-2
        output_size = input_size*stride

    Note: In the above equation, `window` will be replaced with `dilate_window` when dilation > 1.
        dilate_window = (window - 1) * dilation + 1. Check conv_dilate_window()

    Refer to
    https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967

    Args:
        in_shape: convolution lhs shape.
        window: convolution window.
        strides: convolution strides.
        padding: convolution padding.
        dilation: convolution dilation.

    Returns:
        The output shape.

    Raises:
        ValueError: If the length of in_shape, window, strides, and padding are not equal.
    """
    if len(in_shape) != len(window) or len(in_shape) != len(strides):
        raise ValueError(
            f"len(in_shape) = {len(in_shape)} must be equal to "
            f"len(window) = {len(window)} and len(strides) = {len(strides)}"
        )

    window = conv_dilate_window(window=window, dilation=dilation)

    def output_shape(in_shape: Optional[int], window: int, stride: int):
        if in_shape is None:
            return None

        if padding == "SAME":
            return in_shape * stride
        elif padding == "VALID":
            return in_shape * stride + max(window - stride, 0)
        elif padding == "CAUSAL":
            return in_shape * stride
        else:
            raise ValueError(f"{padding} padding is not supported.")

    return tuple(map(output_shape, in_shape, window, strides))


def compute_conv_transpose_paddings(
    in_paddings: Tensor,
    *,
    window: int,
    stride: int,
    conv_padding: ConvPaddingType,
    dilation: int = 1,
    anchor: Optional[int] = None,
):
    """Compute output paddings w.r.t. conv_padding for conv transpose.

    The output paddings value is determined by the padding value at the anchor point in the
    window. If anchor is None, the default anchor point is the left time padding from conv
    padding config. See `Conv2DWith1DPadding.Config` in details.

    In the window=3 and stride=1 case, this function creates paddings as follows:

    The following illustration demonstrates how Conv Transpose operates, assuming all kernel values
    are set to 1 for simplicity in showcasing output values.

    In the window=3 and stride=1 case, this function creates outputs as follows:
    * "SAME" padding=(1, 1)
                    pad|       |pad
        paddings:     0|0 0 1 1|1
                        |_____|
                      * 0 *  -> 0
                        * 0 *  -> 0
                          * 1 *  -> 1
                            * 1 0  -> 1

    * "VALID" padding=(2, 2)
                    pad  |       |pad
        paddings:     0 0|0 0 1 1|1 1
                          |_________|
                      * * 0  -> 0
                        * * 0  -> 0
                          * * 1  -> 1
                            * * 1  -> 1
                              * * 1  -> 1
                                * * 1  -> 1

    * "CAUSAL" padding=(2, 0)
                    pad  |       |pad
        paddings:     0 0|0 0 1 1|
                          |_____|
                      * * 0  -> 0
                        * * 0  -> 0
                          * * 1  -> 1
                            * * 1  -> 1

    In the window=3 and stride=2 case, this function creates outputs as follows:
    * "SAME" padding=(2, 1)
                    pad  |             |pad
        paddings:     0 0|0 * 0 * 1 * 1|1
                          |_____________|
                      * * 0  -> 0
                        * * 0  -> 0
                          * * 0  -> 0
                            * * 0  -> 0
                              * * 1  -> 1
                                * * 1  -> 1
                                  * * 1  -> 1
                                    * * 1  -> 1

    * "VALID" padding=(2, 2)
                    pad  |             |pad
        paddings:     0 0|0 * 0 * 1 * 1|1 1
                          |_______________|
                      * * 0  -> 0
                        * * 0  -> 0
                          * * 0  -> 0
                            * * 0  -> 0
                              * * 1  -> 1
                                * * 1  -> 1
                                  * * 1  -> 1
                                    * * 1  -> 1
                                      * * 1  -> 1

    * "CAUSAL" padding=(2, 1)
                    pad  |             |pad
        paddings:     0 0|0 * 0 * 1 * 1|1
                          |_____________|
                      * * 0  -> 0
                        * * 0  -> 0
                          * * 0  -> 0
                            * * 0  -> 0
                              * * 1  -> 1
                                * * 1  -> 1
                                  * * 1  -> 1
                                    * * 1  -> 1

    In the window=3 and stride=3 case, this function creates outputs as follows:
    * "SAME", "VALID" and "CAUSAL" padding=(2, 2)
                    pad  |                   |pad
        paddings:     0 0|0 * * 0 * * 1 * * 1|1 1
                          |_____________________|
                      * * 0  -> 0
                        * * 0  -> 0
                          * * 0  -> 0
                            * * 0  -> 0
                              * * 0  -> 0
                                * * 0  -> 0
                                  * * 1  -> 1
                                    * * 1  -> 1
                                      * * 1  -> 1
                                        * * 1  -> 1
                                          * * 1  -> 1
                                            * * 1  -> 1

    OTHO, when dilation > 1, dilate_window = (window - 1) * dilation + 1.
    For example, when window=3 and dilation=2, dilate_window=5.

    In the stride=2 case, this function creates outputs as follows:
    * "SAME" padding=(3, 2)
                    pad    |             |pad
        paddings:     0 0 0|0 * 0 * 1 * 1|1 1
                            |_____________|
                      * * * 0 *  -> 0
                        * * * 0 *  -> 0
                          * * * 0 *  -> 0
                            * * * 0 *  -> 0
                              * * * 1 *  -> 1
                                * * * 1 *  -> 1
                                  * * * 1 *  -> 1
                                    * * * 1 *  -> 1

    * "VALID" padding=(4, 4)
                    pad      |             |pad
        paddings:     0 0 0 0|0 * 0 * 1 * 1|1 1 1 1
                              |___________________|
                      * * * * 0  -> 0
                        * * * * 0  -> 0
                          * * * * 0  -> 0
                            * * * * 0  -> 0
                              * * * * 1  -> 1
                                * * * * 1  -> 1
                                  * * * * 1  -> 1
                                    * * * * 1  -> 1
                                      * * * * 1  -> 1
                                        * * * * 1  -> 1
                                          * * * * 1  -> 1

    * "CAUSAL" padding=(4, 1)
                    pad      |             |pad
        paddings:     0 0 0 0|0 * 0 * 1 * 1|1
                              |_____________|
                      * * * * 0  -> 0
                        * * * * 0  -> 0
                          * * * * 0  -> 0
                            * * * * 0  -> 0
                              * * * * 1  -> 1
                                * * * * 1  -> 1
                                  * * * * 1  -> 1
                                    * * * * 1  -> 1

    Args:
        in_paddings: A Tensor of shape [batch_size, seq_len].
        window: convolution window size of the time axis.
        stride: convolution stride size of the time axis.
        conv_padding: "SAME", "VALID", "CAUSAL" or ((left_time_padding, right_time_padding),)
        dilation: convolution dilation size of the time axis.
        anchor: an optional integer in the range of [0, window)
            that specifies the anchor position within the convolution window that is used to
            determine output paddings. Specifically, the output token is valid iff the input token
            at the anchor position of the corresponding window is valid.
            If None, anchor defaults to conv_padding[0] (i.e. left_time_padding).

    Returns:
        out_paddings: A Tensor of shape [batch_size, seq_len].

    Raises:
        ValueError: If anchor is not between left_time_padding and window.
    """

    chex.assert_rank(in_paddings, 2)
    conv_padding = conv_transpose_explicit_padding(
        window=(window,), strides=(stride,), padding=conv_padding, dilation=(dilation,)
    )
    window = conv_dilate_window(window=(window,), dilation=(dilation,))[0]
    # Note: in transposed conv, left_pad + right_pad >= window - 1.
    # See conv_transpose_explicit_padding().
    left_pad, right_pad = conv_padding[0]

    if anchor is None:
        anchor = left_pad
    # elif not left_pad <= anchor < window:
    elif not anchor < window:
        raise ValueError(f"anchor ({anchor}) must in range [0, {window}).")

    # Consider the case where window=3, strides=2, dilation=2, and padding="SAME"
    # explicit padding=(3, 2)
    #                 pad    |             |pad
    #     paddings:     0 0 0|0 * 0 * 1 * 1|1 1
    #                         |_____________|
    #                   * * * 0 *  -> 0
    #                     * * * 0 *  -> 0
    #                       * * * 0 *  -> 0
    #                         * * * 0 *  -> 0
    #                           * * * 1 *  -> 1
    #                             * * * 1 *  -> 1
    #                               * * * 1 *  -> 1
    #                                 * * * 1 *  -> 1

    # |0 0 1 1| ->  |0 * 0 * 1 * 1|
    def dilate_paddings(paddings):
        most, last = jnp.split(paddings, [paddings.shape[1] - 1], axis=1)
        dilated = ein_ops.repeat(most, "b t -> b (t s)", s=stride)
        return jnp.concatenate([dilated, last], axis=1)

    in_paddings = dilate_paddings(in_paddings)

    # |0 * 0 * 1 * 1| ->  0 0 0|0 * 0 * 1 * 1|1 1
    #                           |_____________|   which is |0 * 0 * 1 * 1|1
    window_pad_total = window - 1  # Note: we already check `anchor < window`` always.
    window_right_pad = window_pad_total - anchor
    assert window_right_pad >= 0, f"{anchor=} < {window=} always."
    # Note: left_pad + right_pad >= window + stride - 2 >= window - 1 == anchor + window_right_pad
    valid_right_pad = right_pad - window_right_pad
    if valid_right_pad >= 0:
        out_paddings = jnp.pad(in_paddings, ((0, 0), (0, valid_right_pad)), mode="edge")
    else:
        out_paddings = in_paddings[:, :valid_right_pad]

    start_index = anchor - left_pad
    if start_index < 0:
        out_paddings = jnp.pad(out_paddings, ((0, 0), (-start_index, 0)), mode="edge")
    else:
        out_paddings = out_paddings[:, start_index:]
    return out_paddings


class Conv1DTranspose(BaseConv):
    """The 1D transposed convolution layer."""

    @config_class
    class Config(BaseConv.Config):
        """Configures Conv1DTranspose."""

        window: int = 1
        strides: int = 1
        padding: Required[ConvPaddingType] = REQUIRED
        dilation: int = 1  # Dilation for dilated Convolution.
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.
        # An optional integer in the range of [0, window)
        # that specifies the anchor position within the convolution window that is used to
        # determine output paddings. Specifically, the output token is valid iff the input token
        # at the anchor position of the corresponding window is valid.
        # If None, defaults to left time padding. See compute_conv_transpose_paddings more details.
        anchor: Optional[int] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(cfg.dilation,),
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            num_input_dim_groups=cfg.num_input_dim_groups,
        )
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.window, cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim),
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(
        self, x: Tensor, *, paddings: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        cfg = self.config
        conv_padding = conv_transpose_explicit_padding(
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(cfg.dilation,),
        )

        if paddings is not None:
            chex.assert_rank(x, paddings.ndim + 1)
            # Apply padding to the input.
            x = x * (1 - paddings[..., None])

        output = self._conv(
            x=x, strides=(cfg.strides,), padding=conv_padding, dilation=(cfg.dilation,)
        )

        if paddings is None:
            output_paddings = None
        else:
            # Compute paddings conv output.
            output_paddings = compute_conv_transpose_paddings(
                paddings,
                window=cfg.window,
                stride=cfg.strides,
                conv_padding=cfg.padding,
                dilation=cfg.dilation,
                anchor=cfg.anchor,
            )
            output = output * (1 - output_paddings[..., None])
        return output, output_paddings

    def _conv(
        self,
        x: Tensor,
        *,
        strides: Sequence[int],
        padding: ConvPaddingType,
        dilation: Sequence[int],
    ) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=(1,),
            padding=padding,
            lhs_dilation=strides,
            rhs_dilation=dilation,
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 3:
            raise ValueError(f"We expect len(input_shape) = 3, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                "cfg.input_dim = {cfg.input_dim}."
            )

        in_shape = input_shape[1:2]
        out_shape = conv_transpose_output_shape(
            in_shape,
            window=(cfg.window,),
            strides=(cfg.strides,),
            padding=cfg.padding,
            dilation=(cfg.dilation,),
        )
        return [input_shape[0], *out_shape, cfg.output_dim]


# TODO(dhwang2): move to convolution transpose section.
class Conv2DTranspose(BaseConv):
    """The 2-D transposed convolution layer."""

    @config_class
    class Config(BaseConv.Config):
        """Configures Conv2DTranspose."""

        window: tuple[int, int] = (1, 1)
        strides: tuple[int, int] = (1, 1)
        padding: Required[ConvPaddingType] = REQUIRED
        dilation: tuple[int, int] = (1, 1)
        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.
        # If True, kernel weights have the HWOI layout, following the format used by
        # keras.layers.Conv2DTranspose.
        # Otherwise, the standard layout HWIO is used, which is more efficient.
        transpose_kernel: bool = False

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        if cfg.transpose_kernel:
            cfg.param_partition_spec = (None, None, "model", None)
        else:
            cfg.param_partition_spec = (None, None, None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            num_input_dim_groups=cfg.num_input_dim_groups,
        )
        if cfg.transpose_kernel:
            io_shape = (cfg.output_dim, cfg.input_dim // cfg.num_input_dim_groups)
        else:
            io_shape = (cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim)
        params = dict(
            weight=ParameterSpec(
                shape=tuple(cfg.window) + io_shape,
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        conv_padding = conv_transpose_explicit_padding(
            window=cfg.window, strides=cfg.strides, padding=cfg.padding, dilation=cfg.dilation
        )
        return self._conv(x=x, strides=cfg.strides, padding=conv_padding, dilation=cfg.dilation)

    def _conv(
        self,
        x: Tensor,
        *,
        strides: Sequence[int],
        padding: ConvPaddingType,
        dilation: Sequence[int],
    ) -> Tensor:
        cfg = self.config

        rhs = self.parameters["weight"]
        # Since `jax.lax.conv_general_dilated` does not support transpose_kernel yet,
        # we transpose kernel here.
        if cfg.transpose_kernel:
            # Flip spatial dims
            rhs = jnp.flip(rhs, axis=(0, 1))
            # Swap input / output channel axes
            rhs = rhs.swapaxes(2, 3)

        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=rhs,
            window_strides=(1, 1),
            padding=padding,
            lhs_dilation=strides,
            rhs_dilation=dilation,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    @nowrap
    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 4:
            raise ValueError(f"We expect len(input_shape) = 4, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                "cfg.input_dim = {cfg.input_dim}."
            )

        in_shape = input_shape[1:3]
        out_shape = conv_transpose_output_shape(
            in_shape,
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
        )
        return [input_shape[0], *out_shape, cfg.output_dim]


class Conv2DTransposeWith1DPadding(Conv2DTranspose):
    """The 2-D convolution transpose with 1-D padding on the time axis."""

    @config_class
    class Config(Conv2DTranspose.Config):
        """Configures Conv2DTransposeWith1DPadding."""

        transpose_kernel: bool = False
        # An optional integer in the range of [0, window)
        # that specifies the anchor position within the convolution window that is used to
        # determine output paddings. Specifically, the output token is valid iff the input token
        # at the anchor position of the corresponding window is valid.
        # If None, defaults to left time padding. See compute_conv_transpose_paddings more details.
        anchor: Optional[int] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.transpose_kernel = False  # Choose better one unlike parent.
        return cfg

    # We add a kwargs "paddings" to the forward method.
    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor, *, paddings: Tensor) -> tuple[Tensor, Tensor]:
        """Computes convolution outputs and paddings.

        Args:
            x: A Tensor of shape [batch_size, seq_len, frequency, input_dim].
            paddings: 0/1 boolean Tensor of shape [batch_size, seq_len].

        Returns:
            output: A Tensor of shape [batch_size, seq_len, frequency, output_dim].
            paddings: 0/1 boolean Tensor of shape [batch_size, seq_len].
        """
        cfg = self.config
        # Apply padding to the input.
        assert len(x.shape) == len(paddings.shape) + 2
        x = x * (1 - paddings[..., None, None])

        # Apply Conv2D.
        output = super().forward(x)
        # Compute paddings conv output.
        output_paddings = compute_conv_transpose_paddings(
            paddings,
            window=cfg.window[0],
            stride=cfg.strides[0],
            conv_padding=cfg.padding,
            dilation=cfg.dilation[0],
            anchor=cfg.anchor,
        )
        # Apply padding to the outputs.
        output = output * (1 - output_paddings[..., None, None])
        return output, output_paddings


class Conv3DTranspose(BaseConv):
    """The 3-D convolution transpose layer."""

    @config_class
    class Config(BaseConv.Config):
        """Configures Conv3DTranspose."""

        window: tuple[int, int, int] = (1, 1, 1)  # The convolution window.
        strides: tuple[int, int, int] = (1, 1, 1)  # The convolution strides.
        # Paddings: "SAME", "VALID or "CAUSAL", or ((top, bottom), (left, right), (front, back))
        padding: Required[ConvPaddingType] = REQUIRED
        dilation: tuple[int, int, int] = (1, 1, 1)  # The convolution dilation.

        output_dim: Required[int] = REQUIRED  # Output feature dim.
        bias: bool = True  # Whether to add a bias.

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, None, None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        _check_conv_cfg(
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            num_input_dim_groups=cfg.num_input_dim_groups,
        )
        params = dict(
            weight=ParameterSpec(
                shape=cfg.window + (cfg.input_dim // cfg.num_input_dim_groups, cfg.output_dim),
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], mesh_axes=(cfg.param_partition_spec[-1],)
            )
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        conv_padding = conv_transpose_explicit_padding(
            window=cfg.window, strides=cfg.strides, padding=cfg.padding, dilation=cfg.dilation
        )
        return self._conv(x=x, strides=cfg.strides, padding=conv_padding, dilation=cfg.dilation)

    def _conv(
        self,
        x: Tensor,
        *,
        strides: Sequence[int],
        padding: ConvPaddingType,
        dilation: Sequence[int],
    ) -> Tensor:
        cfg = self.config
        output = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=(1, 1, 1),
            padding=padding,
            lhs_dilation=strides,
            rhs_dilation=dilation,
            dimension_numbers=("NHWDC", "HWDIO", "NHWDC"),
            feature_group_count=cfg.num_input_dim_groups,
        )
        if cfg.bias:
            output += self.parameters["bias"]
        return output

    @nowrap
    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        cfg = self.config
        if len(input_shape) != 5:
            raise ValueError(f"We expect len(input_shape) = 5, but got {len(input_shape)}.")
        if input_shape[-1] != cfg.input_dim:
            raise ValueError(
                f"input_shape[-1] = {input_shape[-1]} does not match "
                f"cfg.input_dim = {cfg.input_dim}."
            )

        in_shape = input_shape[1:4]
        out_shape = conv_transpose_output_shape(
            in_shape,
            window=cfg.window,
            strides=cfg.strides,
            padding=cfg.padding,
            dilation=cfg.dilation,
        )
        return [input_shape[0], *out_shape, cfg.output_dim]


############################## Others ##############################################################


class StackOverTime(BaseLayer):
    """Stack inputs along the time axis.

    StackOverTime behaves the same as Conv2DWith1DPadding w.r.t. paddings along the time axis.
    Please refer to the docstring of Conv2DWith1DPadding to understand how the padding work
    including "SAME", "VALID", and "CAUSAL" literals. The padding anchor is set to `left padding`.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures StackOverTime."""

        stride: Required[int] = REQUIRED  # Number of frames to stack.

        # Number of paddings to apply along the time axis. The two integers specify the amount
        # of leading and trailing padding, respectively. Alternatively, this can be a
        # convolution padding literals type such as 'SAME', 'VALID', or 'CAUSAL'.
        # Note: For backward compatibility, the default is set to VALID, but in most cases,
        # CAUSAL is more appropriate as it preserves the sequence length.
        padding: Union[tuple[int, int], Literal["SAME", "VALID", "CAUSAL"]] = "VALID"

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> tuple[Tensor, Tensor]:
        """Stacks stride number of frames into one frame along the time axis.

        Args:
            inputs: Tensor of shape [batch, time, input_dim].
            paddings: 0/1 boolean Tensor of shape [batch, time], paddings of the input sequences.

        Returns:
            stacked_inputs: Tensor of shape [batch, time // stride, input_dim * stride].
            stacked_paddings: 0/1 boolean Tensor of shape [batch, time // stride]. An output frame
                is padding if at least one of the stacked input frames is padding.

        Raises:
            ValueError: If stride is <= 1.
        """
        cfg = self.config
        if cfg.stride <= 1:
            raise ValueError(f"stride should be greater than 1, but got {cfg.stride}.")

        # For the last partial frame.
        inputs = inputs * (1 - paddings)[:, :, None]

        padding = cfg.padding
        if isinstance(padding, str):
            padding = conv_explicit_padding(
                window=(cfg.stride,), strides=(cfg.stride,), padding=padding, dilation=(1,)
            )[0]
        inputs = jnp.pad(inputs, ((0, 0), padding, (0, 0)), constant_values=0)

        batch_size, seq_len, input_dim = inputs.shape
        output_length = seq_len // cfg.stride
        new_shape = [batch_size, output_length, input_dim * cfg.stride]
        # Stack inputs over the time dimension.
        stacked_inputs = jnp.reshape(inputs[:, : output_length * cfg.stride, :], new_shape)
        # An output frame is padding if at least one of the stacked input frames is padding.
        stacked_paddings = compute_conv_paddings(
            paddings, window=cfg.stride, stride=cfg.stride, conv_padding=(padding,)
        )
        stacked_inputs = stacked_inputs * (1 - stacked_paddings)[:, :, None]
        return stacked_inputs, stacked_paddings

    @nowrap
    def output_shape(self, *, input_shape: Sequence[Optional[int]]) -> Sequence[Optional[int]]:
        """Computes stacked output shape.

        Args:
            input_shape: The input dimensions are (batch, time, feature_dim).
                If the value of the dimension is not available, use None.

        Returns:
            The output shape. The dimensions are (batch, time, feature_dim).
        """
        cfg = self.config
        batch_size, seq_len, input_dim = input_shape
        padding = cfg.padding
        if isinstance(padding, tuple):
            padding = (padding,)
        out_shape = conv_output_shape(
            [seq_len], window=(cfg.stride,), strides=(cfg.stride,), padding=padding, dilation=(1,)
        )
        return [batch_size, *out_shape, input_dim * cfg.stride]

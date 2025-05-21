# Copyright © 2025 Apple Inc.
#
# The API and docstring in this file is adapted from:
#
# arogozhnikov/einops:
# Copyright (c) 2018 Alex Rogozhnikov.
# Licensed under the MIT license.

"""einops rearrange and repeat minimal implementation for JAX.

Reimplement einops.rearrange and einops.repeat specifically for JAX, covering only the functionality
we use.

einops (https://openreview.net/pdf?id=oapKSVM2bcj) significantly improves code readability,
we decided to remove the dependency and use internal equivalents due to the following concerns:
* Thread safety issues:
 * Backend initialization is not thread-safe. See https://github.com/arogozhnikov/einops/issues/372.
 * It uses global variables internally, so we are not sure if it's thread-safe.
* External dependency: As a personal project, einops isn't what we want to rely on in production.

The main differences from the original implementation are as follows:
This file was implemented from scratch. The concept that `rearrange` can be implemented as
reshape → transpose → reshape, and `repeat` as reshape → tile → reshape, was understood from
the original einops design. The key API differences are:
* `...` is not supported.
* Numpy and TensorFlow are not supported.

For how to use einops pattern syntax, please refer to the official einops tutorial.
We only support rearrange and repeat.
Tutorial: https://einops.rocks/1-einops-basics/

Pattern token names must follow Python variable naming rules ([_a-z][_a-z0-9]*) or be the number 1.
"""

import functools
import math
import re
from typing import NamedTuple, Union

from jax import numpy as jnp

from axlearn.common.utils import Tensor


def rearrange(x: Tensor, pattern: str, **axes_lengths) -> Tensor:
    """JAX implementation of `einops.rearrange`.

    This function provides a concise and readable way to perform common tensor manipulations
    such as reshape, transpose, flatten, stack, and unsqueeze, using a simple pattern language.

    It supports combining and splitting axes using parentheses `()`, including
    axis length constraints via keyword arguments. Only a single level of parentheses
    is allowed, and all axes must be accounted for in both input and output.

    Examples:
    ```python
    >>> rearrange(jnp.ones((32, 30, 40, 3)), 'b h w c -> b h w c').shape
    (32, 30, 40, 3)

    >>> rearrange(jnp.ones((32, 30, 40, 3)), 'b h w c -> (b h) w c').shape
    (960, 40, 3)

    >>> rearrange(jnp.ones((32, 30, 40, 3)), 'b h w c -> h (b w) c').shape
    (30, 1280, 3)

    >>> rearrange(jnp.ones((32, 30, 40, 3)), 'b h w c -> b c h w').shape
    (32, 3, 30, 40)

    >>> rearrange(jnp.ones((32, 30, 40, 3)), 'b h w c -> b (c h w)').shape
    (32, 3600)

    >>> rearrange(jnp.ones((32, 30, 40, 3)), 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2)
    (32, 15, 20, 12)

    >>> rearrange(jnp.ones((2, 3, 6, 4)), 'b t (k g) h -> b t k g h', k=2, g=3).shape
    (2, 3, 2, 3, 4)

    >>> rearrange(jnp.ones((5, 1, 4, 8)), 't 1 i o -> t i o').shape
    (5, 4, 8)
    ```

    Find more examples in einops tutorial. https://einops.rocks/1-einops-basics/

    Args:
        x: Input tensor to be rearranged.
        pattern: A string pattern of the form `"input_pattern -> output_pattern"`.
            e.g., "h w -> (h2 i) (w2 j)".
        **axes_lengths: Optional axis size hints for composite dimensions, such as those inside
            parentheses. For example, in the pattern "(h2 i)", you can provide i=2 to specify
            that the original 'h' axis should be split accordingly.

    Returns:
        A new Tensor that is a rearranged view of the input `x`. If possible, returns the original
        tensor.

    Raises:
        ValueError: If the input pattern is malformed or incompatible with `x.shape`.
        ValueError: If parentheses are unbalanced or nested.
        ValueError: If the axes don't match between input and output patterns.
    """
    plan = _compute_rearrange_plan(x.shape, pattern, **axes_lengths)

    # 1. Reshape for transpose.
    if plan.input_reshape != list(x.shape):
        x = jnp.reshape(x, plan.input_reshape)

    # 2. Transpose if needed.
    if plan.operand != list(range(len(plan.operand))):
        x = jnp.transpose(x, plan.operand)

    # 3. Reshape for output.
    if plan.output_shape != list(x.shape):
        x = jnp.reshape(x, plan.output_shape)
    return x


def repeat(x: Tensor, pattern: str, **axes_lengths) -> Tensor:
    """JAX implementation of `einops.repeat`.

    einops.repeat allows reordering elements and repeating them in arbitrary combinations.
    This operation includes the functionality of `repeat`, `tile`, and `broadcast`.

    Examples for repeat operation:

    ```python
    # a grayscale image (of shape height x width)
    >>> image = np.random.randn(30, 40)

    # change it to RGB format by repeating in each channel
    >>> repeat(image, 'h w -> h w c', c=3).shape
    (30, 40, 3)

    # repeat image 2 times along height (vertical axis)
    >>> repeat(image, 'h w -> (repeat h) w', repeat=2).shape
    (60, 40)

    # repeat image 2 time along height and 3 times along width
    >>> repeat(image, 'h w -> (h2 h) (w3 w)', h2=2, w3=3).shape
    (60, 120)

    # convert each pixel to a small square 2x2. Upsample image by 2x
    >>> repeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (60, 80)

    # pixelate image first by downsampling by 2x, then upsampling
    >>> downsampled = reduce(image, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
    >>> repeat(downsampled, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (30, 40)
    ```

    Find more examples in einops tutorial. https://einops.rocks/1-einops-basics/

    Args:
        x: Input tensor.
        pattern: Transformation pattern string, e.g., "h w -> (h2 h) (w2 w)".
        **axes_lengths: Optional axis size hints for composite dimensions, such as those inside
            parentheses. For example, in the pattern "(h2 h)", you can provide h2=2 to specify
            that the original 'h' axis should be repeated accordingly.

    Returns:
        A new tensor with elements repeated and reshaped according to the pattern.

    Raises:
        ValueError: If input pattern is not canonical, or axis names are duplicated or mismatched.
    """
    plan = _compute_repeat_plan(x.shape, pattern, **axes_lengths)

    # 1. Reshape for tile.
    if plan.input_reshape != list(x.shape):
        x = jnp.reshape(x, plan.input_reshape)

    # 2. Tile if needed.
    if not all(t == 1 for t in plan.operand):
        x = jnp.tile(x, plan.operand)

    # 3. Reshape for output.
    if plan.output_shape != list(x.shape):
        x = jnp.reshape(x, plan.output_shape)
    return x


class _Plan(NamedTuple):
    """Plan for rearrange or repeat.

    Fields:
        input_reshape: Reshape target before operator (e.g. transpose or tile).
        operand: arguments for operator (e.g. transpose or tile).
        output_shape: Final shape after operator (e.g. transpose or tile).
    """

    input_reshape: list[int]
    operand: list[int]
    output_shape: list[int]


@functools.lru_cache(maxsize=256)
def _compute_rearrange_plan(in_shape: tuple[int, ...], pattern: str, **axes_lengths) -> _Plan:
    """Computes the reshape and transpose plan for `rearrange()`.

    This function parses the einops-style pattern and determines:
      - Input reshape (if any),
      - Transpose permutation order,
      - Final output shape.

    It is cached to avoid recomputing shape logic even if invoked repeatedly during JIT tracing
    and compilation. The result helps reduce compile overhead (w/o any impact on GPU/TPU runtime).
    Note: functools.lru_cache() is thread-safe since python v3.3.

    Args:
        in_shape: Shape of the input tensor.
        pattern: Rearrangement pattern, of form "lhs -> rhs".
        axes_lengths: Optional axis sizes for grouped axes.

    Returns:
        A Plan named tuple:
            input_reshape: Reshape target before transpose.
            perm: Permutation order for transpose.
            output_shape: Final shape after transpose + reshape.

    Raises:
        ValueError: If the pattern is malformed or axes don't match the shape.
    """
    # _parse_axes() can handle any spacing around the "->" in the pattern.
    left, right = pattern.split("->")
    left_axes = _parse_axes(left)
    right_axes = _parse_axes(right)
    dim_size_map = _get_input_reshape(shape=in_shape, axes=left_axes, axes_lengths=axes_lengths)
    perm, output_shape = _get_rearrange_shape(axes=right_axes, dim_size_map=dim_size_map)
    return _Plan(_to_input_reshape(dim_size_map), perm, output_shape)


@functools.lru_cache(maxsize=256)
def _compute_repeat_plan(in_shape: tuple[int, ...], pattern: str, **axes_lengths) -> _Plan:
    """Computes the reshape, tile, and output shape plan for `repeat()`.

    This function parses the einops-style repeat pattern and determines:
      - Input shape before tiling,
      - Number of repeats (`tile`) for each axis,
      - Final output shape after tiling.

    It is cached with `lru_cache` to improve compile-time performance in JAX.

    Args:
        in_shape: Shape of the input tensor.
        pattern: Repeat pattern, of form "lhs -> rhs".
        axes_lengths: Additional axis sizes used in repeat expansion.

    Returns:
        A Plan named tuple:
            input_reshape: Shape to reshape into before tiling.
            tile: Repeat counts per axis.
            output_shape: Final shape to reshape into after tiling.

    Raises:
        ValueError: If the lhs pattern is not canonical or if axes are invalid.
    """
    # _parse_axes() can handle any spacing around the "->" in the pattern.
    left, right = pattern.split("->")
    left_axes = _parse_axes(left)
    right_axes = _parse_axes(right)

    # Reuse _get_input_reshape to know lhs's each size.
    dim_size_map = _get_input_reshape(shape=in_shape, axes=left_axes, axes_lengths=dict())
    if _to_input_reshape(dim_size_map) != list(in_shape):
        raise ValueError(f"The lhs {left_axes} must be canonical form.")

    repeat_plan = _get_repeat_shape(
        axes=right_axes,
        dim_size_map=dim_size_map,
        axes_lengths=axes_lengths,
    )
    return repeat_plan


_Axis = Union[str, tuple["_Axis", ...]]
_Axes = tuple[_Axis, ...]


def _parse_axes(axes: str) -> _Axes:
    """Parses an axis pattern string into a hierarchical tuple structure.

    Converts patterns like "b t (g k) h" into ('b', 't', ('g', 'k'), 'h').

    Rules:
        * Grouping is only allowed with a single level: no nested parentheses.
        * Groups must have at least two tokens. Singleton groups like "(a)" are invalid.
        * Axis tokens must match regex [_a-z][_a-z0-9]* (like 'x', '_abc', 'a9z').

    Raises:
        ValueError: On invalid tokens, unmatched characters, or invalid group structure.
    """
    # Tokens in the pattern must follow Python identifier rules.
    identifier = r"[_a-z][_a-z0-9]*"
    # Inside a group, tokens must be Python identifiers only.
    group_axis_name_re = re.compile(rf"^{identifier}$")
    # In general, tokens must be either Python identifiers or The number 1 (used for expand_dims).
    axis_name_re = re.compile(rf"^1$|^{identifier}$")
    # Splitting is based on whether the token is
    # * A Python identifier (axis names).
    # * Enclosed in (): indicating a group. Nested (()) and partial (() not allowed.
    # * The number 1: all "1", " 1 ", " 1", "1 " are valid.
    composite_dim = r"\([^()]*\)"
    one = r"(?<=\s)1(?=\s)|^1(?=\s|$)|(?<=\s)1$"
    split_re = re.compile(rf"{one}|{identifier}|{composite_dim}")

    # Extract tokens like: ['b', 't', '(g k)', 'h']
    tokens = split_re.findall(axes)
    # Validate that the entire input string is covered
    unmatched = split_re.sub("", axes).strip()
    if unmatched:
        raise ValueError(f"Unexpected characters in pattern: '{unmatched}'")

    visited = set()

    def _validate_duplicated(name: str):
        if name != "1" and name in visited:
            raise ValueError(f"Duplicated axis name: '{name}' in {axes=}.")
        visited.add(name)

    def _validate_token(name: str) -> str:
        if not axis_name_re.fullmatch(name):
            raise ValueError(
                f"Invalid axis name: '{name}'. Must match Python {identifier=} or be '1'."
            )
        _validate_duplicated(name)
        return name

    def _validate_group_token(name: str) -> str:
        if not group_axis_name_re.fullmatch(name):
            raise ValueError(f"Invalid axis name: '{name}'. Must match Python {identifier=}.")
        _validate_duplicated(name)
        return name

    parsed = []
    for tok in tokens:
        if tok.startswith("(") and tok.endswith(")"):
            inner = tok[1:-1].strip().split()
            if len(inner) < 2:
                raise ValueError(f"Group '{tok}' must contain at least two axes.")
            group = tuple(_validate_group_token(name) for name in inner)
            parsed.append(group)
        else:
            parsed.append(_validate_token(tok))

    return tuple(parsed)


def _get_input_reshape(
    *, shape: tuple[int, ...], axes: _Axes, axes_lengths: dict[str, int]
) -> dict[str, int]:
    """Computes the input shape to reshape into, based on left-side axes.

    Also builds mapping of axis names to their position and lengths.

    Args:
        shape: The shape of the input tensor.
        axes: Parsed axis tokens from the left side of the pattern.
        **axes_lengths: Optional size hints for composite dimensions, such as those inside
            parentheses (e.g., `k=2, g=3`). For example, if the pattern includes `(h k)` and you
            specify `k=2`, then `h` can be inferred from the tensor's shape. This information is
            stored in and returned as `dim_size_map`.

    Returns:
        dim_size_map: Mapping from axis name to its size, where the keys are ordered by the reshape
            dimensions, e.g., if the reshape dims are "a (x y) b", the keys will be ordered as
            "a", "x", "y", "b".

    Raises:
        ValueError: If the axes don't match the shape, or unspecified composite axes are used.
    """
    axes_lengths = axes_lengths.copy()  # Copied it as it's modified.
    dim_size_map = {}

    # Infer unknown dim in group.
    def infer_dims(group: list[str], group_dim_size: int) -> list[int]:
        known_dim, unknown_ax = 1, ""
        for axis_name in group:
            if axis_name in axes_lengths:
                known_dim *= axes_lengths[axis_name]
            else:
                if unknown_ax:
                    raise ValueError(
                        f"Multiple unknown axes ({unknown_ax}, {axis_name}) in a group are not "
                        "allowed."
                    )
                unknown_ax = axis_name
        if unknown_ax:
            if group_dim_size % known_dim != 0:
                raise ValueError(f"Cannot infer unknown axis {unknown_ax}, not divisible.")
            axes_lengths[unknown_ax] = group_dim_size // known_dim
        return [axes_lengths[ax] for ax in group]

    def update_dim_size_map(axis_name, dim_size):
        assert axis_name not in dim_size_map, "Parser already filtered it out."
        dim_size_map[axis_name] = dim_size

    for axis_name, input_dim_size in zip(axes, shape):
        if axis_name == "1":
            if input_dim_size != 1:
                raise ValueError(f"Expected singleton dim, got shape={shape} at {axis_name=}.")
        elif isinstance(axis_name, str):
            update_dim_size_map(axis_name, input_dim_size)
        else:
            assert isinstance(axis_name, tuple)
            group = list(axis_name)
            dim_sizes = infer_dims(group, input_dim_size)
            for ax, dim in zip(group, dim_sizes):
                update_dim_size_map(ax, dim)

    input_reshape = _to_input_reshape(dim_size_map)
    if math.prod(shape) != math.prod(input_reshape):
        raise ValueError(f"Incompatible shape reshape: {shape} -> {input_reshape}")
    return dim_size_map


def _to_input_reshape(dim_size_map: dict[str, int]) -> list[int]:
    """Infer input_reshape from dim_size_map.

    dict preserves insertion order, since Python 3.7+, which became part of the language spec.
    This conversion relies on it.

    Args:
        dim_size_map: Mapping from axis token to its size. See `_get_input_reshape()`.

    Returns:
        input_reshape: Shape to reshape the input tensor to.
    """

    return list(dim_size_map.values())


def _to_input_axes_map(dim_size_map: dict[str, int]) -> dict[str, int]:
    """Infer input_axes_map from dim_size_map.

    dict preserves insertion order, since Python 3.7+, which became part of the language spec.
    This conversion relies on it.

    Args:
        dim_size_map: Mapping from axis token to its size. See `_get_input_reshape()`.

    Returns:
        input_axes_map: Mapping from axis name to position in reshaped tensor. That is,
            input_axes_map[axis_name] = k iff input_reshape[k] corresponds to axis_name.
    """
    return {axis_name: list(dim_size_map.keys()).index(axis_name) for axis_name in dim_size_map}


class _RearrangeRHSPlan(NamedTuple):
    """Plan for rearrange from RHS axes.

    Fields:
        perm: Axis permutation to be used as the operand in `jnp.transpose`.
        output_shape: Target output shape after rearrangement.
    """

    perm: list[int]
    output_shape: list[int]


def _get_rearrange_shape(*, axes: _Axes, dim_size_map: dict[str, int]) -> _RearrangeRHSPlan:
    """Computes the final output shape and permutation order.

    Based on right-side axes, resolves reshaped dimensions and transpose permutation.

    Args:
        axes: Parsed axis tokens from the right side of the pattern.
        dim_size_map: Mapping from axis name to its size. See `_get_input_reshape()`.

    Returns:
        A RearrangeRHSPlan named tuple:
            perm: Axis permutation for transpose step.
            output_shape: Target output shape after rearrangement.

    Raises:
        ValueError: If any axes are missing or mismatched.
    """
    input_axes_map = _to_input_axes_map(dim_size_map)
    output_shape = []
    perm = []
    flat_axes = set()

    def update_and_shape(axis_name):
        if axis_name not in dim_size_map:
            raise ValueError(f"Missing axis {axis_name} in input.")
        assert axis_name not in flat_axes, "Parser already filtered it out."
        flat_axes.add(axis_name)
        perm.append(input_axes_map[axis_name])
        return dim_size_map[axis_name]

    for axis_name in axes:
        if axis_name == "1":
            output_shape.append(1)
        elif isinstance(axis_name, str):
            shape = update_and_shape(axis_name)
            output_shape.append(shape)
        else:
            assert isinstance(axis_name, tuple)
            group_shape = 1
            for sub in axis_name:
                shape = update_and_shape(sub)
                group_shape *= shape
            output_shape.append(group_shape)

    if flat_axes != set(input_axes_map.keys()):
        raise ValueError(
            f"Mismatch between LHS axes and RHS axes: {flat_axes} vs {set(input_axes_map.keys())}"
        )
    return _RearrangeRHSPlan(perm, output_shape)


def _get_repeat_shape(
    *,
    axes: _Axes,
    dim_size_map: dict[str, int],
    axes_lengths: dict[str, int],
) -> _Plan:
    """Computes shapes for applying `jnp.tile` and the final output shape in `repeat`.

    Args:
        axes: Parsed right-hand side axes from the repeat pattern.
        dim_size_map: Mapping from axis name to its size. See `_get_input_reshape()`.
        axes_lengths: Mapping of new (repeating) axis names to their intended sizes.

    Returns:
        Plan named tuple:
            input_reshape: Shape to reshape to before `jnp.tile`.
            tile: Repeat counts per axis (to pass to `jnp.tile`).
            output_shape: Final shape to reshape to after tiling.

    Raises:
        ValueError: If axes or shapes are invalid, duplicated, or inconsistent.
    """
    # total_dim_size_map contains everything.
    total_dim_size_map = dim_size_map.copy()
    total_dim_size_map.update(axes_lengths)
    input_axes_map = _to_input_axes_map(dim_size_map)
    input_reshape = []
    tile = []
    output_shape = []
    flat_axes = set()
    input_axes_idx = 0

    def update_existing_axis(axis_name: str):
        dim = total_dim_size_map[axis_name]
        input_reshape.append(dim)
        tile.append(1)
        nonlocal input_axes_idx
        if input_axes_map[axis_name] != input_axes_idx:
            raise ValueError("repeat doesn't allow reordering existing axes.")
        input_axes_idx += 1
        return dim

    def update_new_axis(axis_name: str):
        dim = total_dim_size_map[axis_name]
        input_reshape.append(1)
        tile.append(dim)
        return dim

    def update_and_shape(axis_name):
        assert axis_name not in flat_axes, "Parser already filtered it out."
        flat_axes.add(axis_name)
        # Note: A user may pass lhs components in axes_lengths, so check dim_size_map.
        if axis_name in dim_size_map:
            dim = update_existing_axis(axis_name)
        else:
            dim = update_new_axis(axis_name)
        return dim

    for axis_name in axes:
        if axis_name == "1":
            raise ValueError("repeat does not support singleton axis '1'.")
        elif isinstance(axis_name, str):
            shape = update_and_shape(axis_name)
            output_shape.append(shape)
        else:
            assert isinstance(axis_name, tuple)
            group_shape = 1
            for sub in axis_name:
                shape = update_and_shape(sub)
                group_shape *= shape
            output_shape.append(group_shape)

    if flat_axes != set(total_dim_size_map.keys()):
        raise ValueError(
            f"lhs+rhs axes {set(total_dim_size_map.keys())} must be same to rhs axes {flat_axes}."
        )
    return _Plan(input_reshape, tile, output_shape)

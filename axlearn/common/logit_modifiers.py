# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/t5x:
# Copyright 2022 The T5X Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Functions for logit-to-logit transformations.

The top-k and top-p implementations borrow heavily from T5X in using binary search over
the space of float32 thresholds, enabling them to achieve benign scaling for large `k` and `p`
values. This is unlikely to be the fastest approach for small `k`, however it is fastest for most
values of `p`.

Reference: <https://github.com/google-research/t5x/blob/79998013/t5x/binary_search.py>
"""
from functools import partial
from typing import Callable, Literal, Union

import jax
from jax import numpy as jnp

from axlearn.common.config import maybe_instantiate
from axlearn.common.decoding import NEG_INF
from axlearn.common.module import Tensor

# Function that may modify logit values, returning a tensor with identical shape and dtype.
LogitsToLogitsFn = Callable[[Tensor], Tensor]


def chain(*args) -> LogitsToLogitsFn:
    """Returns a function to apply multiple logits-to-logits functions/configs in series."""

    def fn(logits: Tensor) -> Tensor:
        for modifier in args:
            modifier = maybe_instantiate(modifier)
            logits = modifier(logits)
        return logits

    return fn


def scale_by(temperature: float, *, min_temperature: float = 1e-4) -> LogitsToLogitsFn:
    """Build a function that returns logits scaled for temperature sampling.

    Args:
        temperature: Used to scale the logits.
        min_temperature: A minimum value for temperature, to guard against divide-by-temperature
            leading to 'inf'; if provided temperature is lower than this value we raise an error.

    Returns:
        A logits-to-logits function.

    Raises:
        ValueError: If temperature is less than min_temperature.
    """
    if temperature < min_temperature:
        # We want to avoid large_logit / very_small_temperature leading to inf.
        raise ValueError(f"temperature should be >= {min_temperature} for numerical stability.")

    def fn(logits: Tensor) -> Tensor:
        return logits / jnp.maximum(temperature, min_temperature)

    return fn


def top_p_logits(p: Union[float, Tensor]) -> LogitsToLogitsFn:
    """Build a function that returns logits suitably normalized for top-p sampling.

    The minimum number of logits so that the total probability mass >= p will be
    returned as is, others will be set to a minimum value.

    The returned function does many reductions over the last axis of the input array.
    To avoid excessive latency, ensure that this axis is fully replicated over devices.
    Additionally, the returned function benefits from being able to fit the entire input
    logits Tensor into a fast memory tier, so looping over micro-batches may be beneficial.

    Ref: <https://github.com/google-research/t5x/blob/79998013/t5x/binary_search.py#L227>

    Args:
        p: The total cumulative probability to consider for sampling as a scaler or a 1D tensor
            with a leading batch dimension.

    Returns:
        A logits-to-logits function.
    """
    if isinstance(p, float) and not 0.0 < p <= 1.0:
        raise ValueError("`p` must be in (0, 1].")
    elif not isinstance(p, float) and p.ndim != 1:
        raise ValueError("`p` must be a scalar or a 1D tensor.")

    def fn(logits: Tensor) -> Tensor:
        probs = jax.nn.softmax(logits, axis=-1)
        # Probs in a form suitable for efficient TPU reduction.
        reducible_probs = probs
        reduce_axis = reducible_probs.ndim - 1
        if reducible_probs.ndim > 1:
            # As we will be doing many reductions over reduce_axis, transpose it to be
            # the penultimate dimension, so that these reductions happen within vector lanes.
            # (See ref. in docstring for more details).
            reducible_probs = jnp.swapaxes(reducible_probs, -1, -2)
            reduce_axis = reducible_probs.ndim - 2

        def predicate(float32_query: Tensor, top_p: Union[float, Tensor]) -> Tensor:
            float32_query = jnp.expand_dims(float32_query, reduce_axis)
            # [..., 1, float32_query.shape[-1]]
            probability_mass = jnp.sum(
                jnp.where(reducible_probs >= float32_query, reducible_probs, 0.0),
                axis=reduce_axis,
            )
            if not isinstance(top_p, float):
                top_p = top_p.reshape((top_p.shape[0], *(1,) * (probability_mass.ndim - 1)))
            return probability_mass < top_p

        batched_shape = logits.shape[:-1]  # All but the last axis are batched.
        threshold = _float32_binary_search(batched_shape, predicate=partial(predicate, top_p=p))
        return jnp.where(probs >= jnp.expand_dims(threshold, -1), logits, NEG_INF)

    return fn


def top_k_logits(
    k: int, *, break_ties: Literal["all", "smallest_index"] = "all"
) -> LogitsToLogitsFn:
    """Build a function that returns logits suitably normalized for top-k sampling.

    The returned function does many reductions over the last axis of the input array.
    To avoid excessive latency, ensure that this axis is fully replicated over devices.
    Additionally, the returned function benefits from being able to fit the entire input
    logits Tensor into a fast memory tier, so looping over micro-batches may be beneficial.

    Ref: <https://github.com/google-research/t5x/blob/79998013/t5x/binary_search.py#L164>

    Args:
        k: The maximum rank of logit to consider for sampling.
        break_ties: Configures top-k behavior in the case of ties:
            * "all": Return all logits with the tied value (in total more than k).
            * "smallest_index": Return the k tied values with smallest index.
              Currently this only supports k = 1.

    Returns:
        A logits-to-logits function.

    Raises:
        ValueError: If break_ties is invalid.
        NotImplementedError: If break_ties == "smallest_index" and k != 1.
    """

    def fn(logits: Tensor) -> Tensor:
        # Logits in a form suitable for efficient TPU reduction.
        reducible_logits = logits
        reduce_axis = reducible_logits.ndim - 1
        if reducible_logits.ndim > 1:
            # As we will be doing many reductions over reduce_axis, transpose it to be
            # the penultimate dimension, so that these reductions happen within vector lanes.
            # (See ref. in docstring for more details).
            reducible_logits = jnp.swapaxes(reducible_logits, -1, -2)
            reduce_axis = logits.ndim - 2

        def predicate(float32_query: Tensor) -> Tensor:
            float32_query = -float32_query
            float32_query = jnp.expand_dims(float32_query, reduce_axis)
            # [..., 1, float32_query.shape[-1]]
            count_number_gt = jnp.sum(reducible_logits > float32_query, axis=reduce_axis)
            return count_number_gt >= k

        batched_shape = logits.shape[:-1]  # All but the last axis are batched.
        # We negate both the result and each query value inside the predicate so as to end up
        # with the smallest float32 value for which the predicate is False.
        # This allows us to simply check for greater _and_ equal to threshold in
        # order to capture the top-k logits.
        threshold = -1 * _float32_binary_search(batched_shape, predicate=predicate)
        return jnp.where(logits >= jnp.expand_dims(threshold, -1), logits, NEG_INF)

    def smallest_index_fn(logits: Tensor) -> Tensor:
        # Returns the maximum value with smallest index when there are ties for k = 1.
        # Note this only supports for k = 1. We may consider to extend to k > 1 in
        # the future, but the benefits may be limited as determinism is usually
        # not required in those cases.
        if k != 1:
            raise NotImplementedError(f"Only k=1 supportes for {break_ties=}, but got {k}.")
        # Note different from numpy.argmax, jnp.argmax doesn't mention it returns
        # the first maximum value. We assume it follows np.argmax and have a unit
        # test to check this.
        mask = jax.nn.one_hot(jnp.argmax(logits, axis=-1), logits.shape[-1], axis=-1)
        return jnp.where(mask, logits, NEG_INF)

    if break_ties == "all":
        return fn
    elif break_ties == "smallest_index":
        return smallest_index_fn
    else:
        raise ValueError(f"Unsupported {break_ties=}")


def _monotonic_int32_to_float32_bit_mask(x: Tensor) -> Tensor:
    """Converts an int32 value to an int32 representing a float32 bit mask.

    The transformation is monotonic wrt. total ordering, i.e. ordering of the float32 bit
    mask should be consistent with ordering of the original int32 values.

    Args:
        x: int32 Tensor.

    Returns:
        An int32 Tensor with entries that represent float32 bit patterns.
    """
    # Bit mask that's 1 for all non-sign bits and 0 for the sign bit (max int32 value).
    non_sign_bit_mask = jnp.int32((1 << 31) - 1)
    # In int32, the bit pattern with sign bit set and all other bits unset is the most
    # negative bit pattern (int32::MIN), whereas in float32 it's the least
    # negative bit pattern (-0.0). Flipping all the non-sign bits via XOR-ing
    # with a non-sign bit mask makes the overall transformation monotonic wrt.
    # total ordering.
    return x ^ jnp.where(x < 0, non_sign_bit_mask, jnp.int32(0))


def _int32_binary_search(
    batched_shape: tuple[int], *, predicate: Callable[[Tensor], Tensor]
) -> Tensor:
    """Binary search to find the largest finite int32 value for which the predicate is False.

    Ref: <https://github.com/google-research/t5x/blob/79998013/t5x/binary_search.py#L28>

    Args:
        batched_shape: The shape of the Tensor over which the predicate should be evaluated.
        predicate: A monotonic function which accepts int32 Tensors of batched_shape,
            and returns bool Tensors of the same shape.

    Returns:
        For each batched element, the largest int32 value for which `predicate` returned False.
            If all values returned True, return int32::MIN.
    """
    # Initialize the solution to be 0.
    solution = jnp.zeros(batched_shape, dtype=jnp.int32)
    # Special case the sign bit (the `jnp.where` choice args are flipped vs non-sign bits).
    predicate_satisfied = predicate(solution)
    solution = solution | jnp.where(predicate_satisfied, jnp.int32(-(1 << 31)), jnp.int32(0))
    del predicate_satisfied

    def loop_body(i: int, solution: Tensor) -> Tensor:
        # Loop over the non-sign bits.
        bit = jnp.int32(1 << 30 - i)
        # pylint: disable-next=unsupported-binary-operation
        predicate_satisfied = predicate(solution | bit)
        solution = solution | jnp.where(predicate_satisfied, jnp.int32(0), bit)
        return solution

    return jax.lax.fori_loop(0, 31, loop_body, solution)


def _float32_binary_search(
    batched_shape: tuple[int], *, predicate: Callable[[Tensor], Tensor]
) -> Tensor:
    """Binary search to find the largest finite float32 value for which predicate is False.

    We rely on bit-shifts and bit-pattern operations, which are supported in int32 but not float32.
    For this reason much of the implementation assumes int32 values, and we massage float32
    values to and from int32.

    Ref: <https://github.com/google-research/t5x/blob/79998013/t5x/binary_search.py#L118>

    Args:
        batched_shape: The shape of the Tensor over which the predicate should be evaluated.
        predicate: A monotonic function which accepts float32 Tensors of batched_shape,
            and returns bool Tensors of the same shape.

    Returns:
        For each batched element, the largest float32 value for which `predicate` returned False.
    """
    # Create exponent bit mask.
    exponent_bit_mask = jnp.int32((1 << 31) - (1 << (31 - 8)))

    def int32_predicate(x):
        x = _monotonic_int32_to_float32_bit_mask(x)
        # If only the exponent bits are set then x is not finite.
        is_finite = (x & exponent_bit_mask) != exponent_bit_mask
        # For non-finite numbers at int32::MIN we return False,
        # whilst for non-finite numbers at int32::MAX we return True.
        predicate_on_nonfinite = x >= 0
        fp32_x = jax.lax.bitcast_convert_type(x, jnp.float32)
        return jnp.where(is_finite, predicate(fp32_x), predicate_on_nonfinite)

    solution = _int32_binary_search(batched_shape, predicate=int32_predicate)
    float32_solution = _monotonic_int32_to_float32_bit_mask(solution)
    # Bitcast to float32.
    return jax.lax.bitcast_convert_type(float32_solution, jnp.float32)

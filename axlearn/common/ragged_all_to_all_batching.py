# Copyright © 2025 Apple Inc.

"""Custom batching rule for lax.ragged_all_to_all

This module provides a custom batching implementation for ragged_all_to_all operations
that are missing in JAX. The batching rule is needed when using jax.vmap with
functions that contain ragged_all_to_all operations).
"""

from typing import Any, Optional

import jax.numpy as jnp
from jax import custom_batching, lax

from axlearn.common.utils import Tensor


def _create_ragged_all_to_all_with_axis(axis_name: str):
    """Create a ragged_all_to_all function for custom_vmap."""

    @custom_batching.custom_vmap
    def ragged_all_to_all_fixed_axis(
        inputs, outputs, input_offsets, send_sizes, output_offsets, recv_sizes
    ):
        """Custom vmap-compatible wrapper."""
        return lax.ragged_all_to_all(
            inputs,
            outputs,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=axis_name,
        )

    # pylint: disable=unused-argument
    @ragged_all_to_all_fixed_axis.def_vmap
    def ragged_all_to_all_vmap_rule_fixed(
        axis_size,
        in_batched,
        inputs,
        outputs,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
    ):
        """Custom batching rule for ragged_all_to_all."""
        (
            inputs_batched,
            outputs_batched,
            input_offsets_batched,
            send_sizes_batched,
            output_offsets_batched,
            recv_sizes_batched,
        ) = in_batched

        if not inputs_batched:
            # If inputs are not batched, a simple lax.ragged_all_to_all call suffices.
            result = lax.ragged_all_to_all(
                inputs,
                outputs,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=axis_name,
            )
            return result, False

        # When vmap is called in track parallelism, the inputs are batched by number of tracks
        num_tracks = inputs.shape[0]

        def loop_body(i, current_results):
            single_inputs = inputs[i]
            single_outputs = outputs[i] if outputs_batched else outputs
            single_input_offsets = input_offsets[i] if input_offsets_batched else input_offsets
            single_send_sizes = send_sizes[i] if send_sizes_batched else send_sizes
            single_output_offsets = output_offsets[i] if output_offsets_batched else output_offsets
            single_recv_sizes = recv_sizes[i] if recv_sizes_batched else recv_sizes

            # The ragged_all_to_all operation for a single tack
            track_result = lax.ragged_all_to_all(
                single_inputs,
                single_outputs,
                single_input_offsets,
                single_send_sizes,
                single_output_offsets,
                single_recv_sizes,
                axis_name=axis_name,
            )
            return current_results.at[i].set(track_result)

        output_shape_per_track = outputs.shape[1:] if outputs_batched else outputs.shape
        batched_outputs_shape = (num_tracks,) + output_shape_per_track
        initial_results = jnp.zeros(batched_outputs_shape, dtype=outputs.dtype)

        final_results = lax.fori_loop(0, num_tracks, loop_body, initial_results)

        return final_results, True

    return ragged_all_to_all_fixed_axis


# pylint: disable=unused-argument
def ragged_all_to_all_batched(
    inputs: Tensor,
    outputs: Tensor,
    input_offsets: Tensor,
    send_sizes: Tensor,
    output_offsets: Tensor,
    recv_sizes: Tensor,
    *,
    axis_name: str,
    axis_index_groups: Optional[Any] = None,
) -> jnp.ndarray:
    """Drop-in replacement for lax.ragged_all_to_all with batching support.

    This function has the same signature as lax.ragged_all_to_all but supports
    jax.vmap through custom batching rules.

    Args:
        inputs: Input tensor
        outputs: Output tensor
        input_offsets: Input offsets for ragged communication
        send_sizes: Sizes of data to send
        output_offsets: Output offsets for ragged communication
        recv_sizes: Sizes of data to receive
        axis_name: Name of the communication axis
        axis_index_groups: Optional axis index groups

    Returns:
        Result of ragged_all_to_all operation
    """
    # Create a custom_vmap function with the specific axis_name
    ragged_all_to_all_fn = _create_ragged_all_to_all_with_axis(axis_name)
    return ragged_all_to_all_fn(
        inputs, outputs, input_offsets, send_sizes, output_offsets, recv_sizes
    )

# Copyright © 2026 Apple Inc.

"""Test utilities for streaming layers."""

import chex
import jax
import jax.numpy as jnp
import numpy as np

from axlearn.audio.streaming.streaming_base import next_segment_pos
from axlearn.common import ein_ops
from axlearn.common.test_utils import assert_allclose, assert_not_allclose
from axlearn.common.utils import Tensor


def segment_inputs(
    data_key: Tensor,
    *,
    segment_pad: int,
    stride: int,
    suffix_shape: tuple[int, ...] = (),
    t: int = 7,
) -> tuple[Tensor, Tensor]:
    """Generates test inputs for `segment_pad` verification.

    Creates 4 batches: ground truth (batch 0,1), correct pad (batch 2), wrong pad (batch 3).

    Example with t=3, eff_segment_pad=2:
        segment_ids[0] = [1, 1, 1, 0, 0, 0, 0, 0]  # Ground truth seg1
        segment_ids[1] = [2, 2, 2, 0, 0, 0, 0, 0]  # Ground truth seg2
        segment_ids[2] = [1, 1, 1, 0, 0, 2, 2, 2]  # correct pad
        segment_ids[3] = [1, 1, 1, 0, 2, 2, 2, 0]  # wrong pad

    Args:
        data_key: JAX PRNG key for generating random inputs.
        segment_pad: the segment padding value to test.
        stride: the stride of the layer being tested.
        suffix_shape: additional dimensions after the time dimension (e.g., (dim,) for 1D).
        t: length of each segment in samples.

    Returns:
        inputs: input tensor of shape (4, T) + suffix_shape.
        segment_ids: segment ID tensor of shape [4, T].
    """
    _, eff_segment_pad = next_segment_pos(t, segment_pad=segment_pad, stride=stride)
    T = 2 * t + eff_segment_pad
    shape = (2, t) + suffix_shape
    full_shape = (4, T) + suffix_shape

    two_inputs = np.array(jax.random.normal(data_key, shape))
    two_segment_ids = np.ones((2, t)) * np.array([1, 2])[:, None]

    inputs = np.zeros(full_shape)
    segment_ids = np.zeros([4, T])

    # Ground truth.
    inputs[:2, :t] = two_inputs
    segment_ids[:2, :t] = two_segment_ids

    # With segment_pad.
    inputs[2, :t] = two_inputs[0]
    inputs[2, t + eff_segment_pad : t + eff_segment_pad + t] = two_inputs[1]
    segment_ids[2, :t] = two_segment_ids[0]
    segment_ids[2, t + eff_segment_pad : t + eff_segment_pad + t] = two_segment_ids[1]

    # Insufficient segment_pad.
    wrong_pad = eff_segment_pad - 1
    inputs[3, :t] = two_inputs[0]
    inputs[3, t + wrong_pad : t + wrong_pad + t] = two_inputs[1]
    segment_ids[3, :t] = two_segment_ids[0]
    segment_ids[3, t + wrong_pad : t + wrong_pad + t] = two_segment_ids[1]
    return jnp.array(inputs, jnp.float32), jnp.array(segment_ids, jnp.int32)


def check_segment_pad_outputs(
    x: Tensor, segment_ids: Tensor, *, check_wrong: bool = True, atol: float = 1e-6
):
    """Verifies `segment_pad` correctness by comparing outputs.

    Compares ground-truth outputs (batch 0,1) against packed outputs (batch 2,3).
    - Batch 2 (correct padding) should match ground truth exactly.
    - Batch 3 (wrong padding) should differ from ground truth.

    Args:
        x: output tensor of shape [4, T, ...].
        segment_ids: segment ID tensor of shape [4, T].
        check_wrong: whether to check the wrong-pad case (batch 3).
        atol: absolute tolerance.
    """
    mask = segment_ids != 0
    gt_out = ein_ops.rearrange(x[:2], "b t ... -> (b t) ...")
    gt_mask = ein_ops.rearrange(mask[:2], "b t -> (b t)")
    gt_out = gt_out[gt_mask]

    out_with_seg_pad = x[2][mask[2]]
    out_with_wrong_pad = x[3][mask[3]]

    chex.assert_equal_shape((gt_out, out_with_seg_pad))
    assert_allclose(gt_out, out_with_seg_pad, atol=atol)

    if check_wrong:
        # Misaligned packing can result in different output lengths.
        min_t = min(gt_out.shape[0], out_with_wrong_pad.shape[0])
        chex.assert_equal_shape((gt_out[:min_t], out_with_wrong_pad[:min_t]))
        assert_not_allclose(gt_out[:min_t], out_with_wrong_pad[:min_t], atol=atol)

"""Forced alignment using CTC log-posterior.

In this library, we provider an aligner which takes a CTC posterior-gram (size [T, V]) and a
ground truth label sequence (size [W,]), produce the optimal token alignment path from the input
sequences to the label sequence. It can also produce the word level alignment if the word boundary
is known.

We provide 2 implementations: the first one is non-batched (or vectorized) and cannot run
efficiently on accelerators (GPU/TPU), whose purpose is to provide a more readable reference
implementation that the second one can be compared with; the second one is a batched and jittable,
therefore it can run very efficiently on accelerators.

Details of these algorithms can be found in the companion memo.
See docs/audio/ctc_alignment_alg.pdf for discussions and pseduo algorithms.


"""
from dataclasses import asdict, dataclass
from typing import Literal, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from axlearn.common import ein_ops
from axlearn.common.module import NestedTensor, Tensor


class CtcAligner:
    """A CPU based CTC alignment implementation."""

    def __init__(self, vocab_size: int, blank_id: int = 0):
        # Id of blank symbol.
        self.blank_id = blank_id
        assert self.blank_id == 0, "Only blank_id == 0 is supported at the moment."
        # Vocab size, including blank symbol.
        self.vocab_size = vocab_size

    def _check_input(self, log_posterior: np.ndarray, ground_truth: np.ndarray):
        if log_posterior.ndim != 2:
            raise ValueError(
                f"Expecting log_posterior is a 2-D tensor, but got {log_posterior.shape}."
            )
        _, vocab_size = log_posterior.shape
        if vocab_size != self.vocab_size:
            raise ValueError(f"Expecting vocab_size to be {self.vocab_size}, but got {vocab_size}.")
        if ground_truth.ndim != 1:
            raise ValueError(
                f"Expecting ground_truth is a 1-D tensor, but got {ground_truth.shape}."
            )
        if ground_truth.dtype != np.int32:
            raise ValueError(f"Ground_truth must have a int32 dtype, got {ground_truth.dtype}")
        max_label_index = np.max(ground_truth)
        min_label_index = np.min(ground_truth)
        if max_label_index >= vocab_size or min_label_index < 0:
            raise ValueError(
                f"Ground truth label must in range [0, {ValueError}), "
                f"but we got [{min_label_index}, {max_label_index}]."
            )

    def _get_label_from_state(self, state: int, ground_truth: np.ndarray):
        num_labels = ground_truth.shape[0]
        assert state <= 2 * num_labels + 1
        assert state >= 1
        if state % 2 == 0:
            return ground_truth[state // 2 - 1]
        else:
            return self.blank_id

    def align(self, log_posterior: np.ndarray, ground_truth: np.ndarray) -> Optional[np.ndarray]:
        """Produces the optimal alignment path for a given log_posterior and ground_truth pair.

        Args:
            log_posterior: a size [T, V] ndarray with dtype np.float32; T is the number of
                input frames, V is the vocab size, must match with `self.vocab_size`;
            ground_truth: a size [W, ] np ndarray with dtype np.int32; W is the number of tokens.

        Returns:
            alignment: a size [T, ] ndarray with dtype np.int32; alignment[t] indicates the t-th
                input frame is best aligned with a symbol whose index is `alignement[t]`. If None,
                it indicates there is no valid alignemnt can be found.

        Raises:
            ValueError if input shape is not expected or ground_truth contains out-of-vocab label.
        """
        self._check_input(log_posterior, ground_truth)
        num_frames, _ = log_posterior.shape
        num_labels = ground_truth.shape[0]
        # 1. Initialize.
        alpha = np.zeros((num_frames + 1, 2 * num_labels + 2), dtype=np.float32)
        alpha.fill(-np.inf)
        traceback_table = np.zeros((num_frames + 1, 2 * num_labels + 2), dtype=np.int32)
        traceback_table.fill(-1)

        alpha[0, 0] = 0.0

        # 2. Iterate over time steps.
        for t in range(1, num_frames + 1):
            # Including this frame, we have observed t frames, the maximum u index
            # we can go is t * 2
            max_u = min(t * 2, 2 * num_labels + 1)
            # We have frame with index [t+1, num_frames] left. The maximum we can go up with these
            # (num_frames - t) frames are 2 * (num_frames - t). We must arrive at
            # 2 * num_labels at frame `num_frames` to be able to emit the last symbol,
            # otherwise it is an invalid path. Therefore,
            #  u + 2 * (num_frames - t) >= 2 * num_labels and u>=1
            # i.e., u >= max(1, 2 * num_labels - 2 (num_frames - t))
            min_u = max(2 * num_labels - 2 * (num_frames - t), 1)

            for u in range(min_u, max_u + 1):
                if u % 2 == 0:
                    # This means we are emitting w-th label in the ground_truth to enter this state
                    w = u // 2 - 1
                    prev_states_and_scores = [(u, alpha[t - 1, u]), (u - 1, alpha[t - 1, u - 1])]
                    if (w >= 1 and ground_truth[w - 1] != ground_truth[w]) or w == 0:
                        prev_states_and_scores.append((u - 2, alpha[t - 1, u - 2]))

                    prev_states_and_scores.sort(key=lambda x: x[1], reverse=True)
                    best_score = prev_states_and_scores[0][1]
                    if best_score > -np.inf:
                        traceback_table[t, u] = prev_states_and_scores[0][0]
                        alpha[t, u] = best_score + log_posterior[t - 1, ground_truth[w]]
                else:
                    # This means we are emitting a blank symbol to enter this state
                    prev_states_and_scores = [(u, alpha[t - 1, u]), (u - 1, alpha[t - 1, u - 1])]
                    prev_states_and_scores.sort(key=lambda x: x[1], reverse=True)
                    best_score = prev_states_and_scores[0][1]
                    if best_score > -np.inf:
                        traceback_table[t, u] = prev_states_and_scores[0][0]
                        alpha[t, u] = best_score + log_posterior[t - 1, self.blank_id]

        # 3. Traceback.
        alignment_to_label = np.zeros((num_frames,), dtype=np.int32)
        alignment_to_label.fill(-1)
        alignment_to_state = np.zeros((num_frames + 1,), dtype=np.int32)
        alignment_to_state.fill(-1)
        terminal_states = (2 * num_labels, 2 * num_labels + 1)
        terminal_states_and_scores = list((s, alpha[num_frames, s]) for s in terminal_states)
        terminal_states_and_scores.sort(key=lambda x: x[1], reverse=True)
        best_terminal_state, best_terminal_score = terminal_states_and_scores[0]
        if best_terminal_score == -np.inf:
            return None
        else:
            alignment_to_state[num_frames] = best_terminal_state
            alignment_to_label[num_frames - 1] = self._get_label_from_state(
                best_terminal_state, ground_truth
            )
            for t in range(num_frames - 1, 0, -1):
                alignment_to_state[t] = traceback_table[t + 1, alignment_to_state[t + 1]]
                alignment_to_label[t - 1] = self._get_label_from_state(
                    alignment_to_state[t], ground_truth
                )

        return alignment_to_label

    def alignment_to_boundary(self, alignment: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """Converts alignment to boundary.

        Args:
            alignment: a size [T, ] arrary, where alignment[t] indicate the label aligned with
                the t-th frame
            ground_truth: a size [W, ] array indicating the ground truth label sequence

        Returns:
            boundary: a [W, 2] array, where boundary [w, 0] is the starting frame of w-th label,
                boundary[w, 1] is the ending frame of w-th label (exclusive)
        """
        alignment_ext = np.concatenate(
            [alignment, np.array([self.blank_id], dtype=np.int32)], axis=0
        )
        right_shifted_alignment_ext = np.roll(alignment_ext, 1)

        mask = np.logical_and(
            right_shifted_alignment_ext != alignment_ext, alignment_ext != self.blank_id
        )

        if not np.array_equal(alignment_ext[mask], ground_truth):
            raise ValueError(f"{alignment} is not a valid alignment for {ground_truth}")

        start = np.where(mask)[0].astype(np.int32)
        end = np.roll(start, shift=-1)
        end[-1] = alignment.shape[0]
        boundary = np.stack([start, end], axis=0)
        return boundary


@dataclass
class AlignmentOutput:
    """Alignment output class for vectorized and jittable implementation."""

    # `alignment`: a [B, T]-shaped tensor, alignment[b,t] indicates t-th frame in b-th sequence
    # is aligned to label whose id is `alignment[b,t]`
    alignment: Tensor
    # `alignment_padding`: a [B,]-shaped tensor. It indicates the score of each sequence. If a
    # sequence cannot be aligned, its score is -inf.
    alignment_score: Tensor
    # `segments`: a [B, T, 2]-shaped tensor. segments[b, l] indicates the l-th label's start time
    # (inclusive) and end time (exclusive). -1 means this is a padding
    segments: Tensor

    def asdict(self) -> NestedTensor:
        return asdict(self)


def calculate_prev_extend_labels(labels: Tensor, label_paddings: Tensor) -> Tensor:
    """Calculates prev_extend_labels when doing search on lattices.

    Args:
        labels: a [B, L]-shaped tensor
        label_paddings: a [B, L]-shaped 0/1 tensor.

    Returns:
        prev_labels: a [B, 2*(L+1), 3]-shaped tensor.

    Note:
        When doing search on lattices, the y-axis is consisted of length 2*(L+1) extended labels
            (0, blank, labels[0], blank, labels[1], ..., labels[-1], blank).
        When i is odd,  prev_labels[i] = (i - 1, i, -1), which means only (i-1) and i-th extend
            labels can transite to blank sybmol; When i is even,
                prev_labels[i] = (i - 2, i - 1, i) if labels[i// 2 - 1] != labels[i//2], otherwise
                prev_labels[i] = (i -1, i, -1)
    """
    batch_size, max_num_labels = labels.shape
    label_lengths = jnp.sum(1.0 - label_paddings, axis=-1)

    def _body(carry, step_index):
        def _odd_fn(step_index):
            return ein_ops.repeat(
                jnp.array([step_index - 1, step_index, -1], dtype=jnp.int32),
                "three -> b three",
                b=batch_size,
            )

        def _even_fn(step_index):
            def _zero_fn(step_index):
                del step_index
                return ein_ops.repeat(
                    jnp.array([-1, -1, -1], dtype=jnp.int32), "three -> b three", b=batch_size
                )

            def _nonzero_fn(step_index):
                label_index = step_index // 2 - 1
                cur_normal_label = labels[:, label_index]
                prev_normal_label = labels[:, label_index - 1]
                ret_when_prev_equal_curr = ein_ops.repeat(
                    jnp.array([step_index - 1, step_index, -1], dtype=jnp.int32),
                    "three -> b three",
                    b=batch_size,
                )
                ret_when_prev_neq_curr = ein_ops.repeat(
                    jnp.array([step_index - 2, step_index - 1, step_index], dtype=jnp.int32),
                    "three -> b three",
                    b=batch_size,
                )
                return jnp.where(
                    ein_ops.repeat(
                        cur_normal_label == prev_normal_label, "b -> b new_axis", new_axis=3
                    ),
                    ret_when_prev_equal_curr,
                    ret_when_prev_neq_curr,
                )  # [batch_size, 3]

            return jax.lax.cond(step_index == 0, _zero_fn, _nonzero_fn, step_index)

        return carry, jax.lax.cond(step_index % 2 == 0, _even_fn, _odd_fn, step_index)

    _, prev_extended_labels = jax.lax.scan(
        _body, None, xs=jnp.arange(2 * (max_num_labels + 1), dtype=jnp.int32)
    )  # [2 * (L + 1), B, 3]
    prev_extended_labels = jnp.transpose(prev_extended_labels, (1, 0, 2))  # [B, 2*(L+1), 3]
    # Deal with paddings: prev_extended_labels[b, t] = (-1, -1, -1) if t >= (label_length[b] + 1)*2
    is_padding = jnp.arange(2 * (max_num_labels + 1))[None, :] >= 2 * (label_lengths[:, None] + 1)
    prev_extended_labels = jnp.where(is_padding[:, :, None], -1, prev_extended_labels)
    return prev_extended_labels


def find_max_along_last_axis(value: Tensor, indices: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Finds max value and its index alogn the last axis.

    Args:
        value: a [B, U]-shaped tensor
        indices: a [B, U, 3]-shaped tensor. If -1, then it means this is a padding

    Returns:
        max_value: a [B, U]-shaped tensor, where
            max_value[b, u] = max_k(value[b, k) for k in indices[b, u, :] and k > 0; or -inf if
                indices[b, u, :]  < 0

        max_indices: a [B, U]-shaped tensor, where
            max_indices[b, u] = argmax_k(value[b, k) for k \in indices[b, u, :] and k > 0;
            or -1 if incides[b, u, :] < 0
    """
    valid_mask = indices >= 0  # [B, U]
    safe_indices = jnp.where(valid_mask, indices, 0)  # [B, U, 3]
    gathered_values = jnp.take_along_axis(value[:, None], safe_indices, axis=-1)  # [B, U, 3]
    masked_values = jnp.where(valid_mask, gathered_values, -jnp.inf)  # [B, U, 3]

    max_values = jnp.max(masked_values, axis=-1)  # [B, U]
    max_positions = jnp.argmax(masked_values, axis=-1)  # [B, U]
    # Use max_positions to extract the corresponding indices from `indices`
    max_indices = jnp.take_along_axis(indices, max_positions[..., None], axis=-1).squeeze(-1)

    all_invalid_mask = jnp.all(~valid_mask, axis=-1)
    max_values = jnp.where(all_invalid_mask, -jnp.inf, max_values)
    # If max_values is -inf, then the corresponding max_indices should be -1
    max_indices = jnp.where(all_invalid_mask, -1, max_indices)
    max_indices = jnp.where(jnp.isinf(max_values), -1, max_indices)

    return max_values, max_indices


class AlignmentLoopState(NamedTuple):
    """Alignment loop state, internal use only."""

    step: int
    # `stop_alignment`: a [B]-shaped tensor, indicating whether it has reached padding part, which
    # can be used to determine whether to early stop
    stop_alignment: Tensor
    # `search_lattices`: a [B, T+1, 2*(max_num_labels + 1)]-shaped tensor
    search_lattices: Tensor
    # `backtrace`: a [B, T+1, 2*(max_num_labels + 1)]-shaped int tensor
    backtrace: Tensor

    @classmethod
    def init(cls, batch_size: int, max_num_frames: int, max_num_labels: int):
        search_lattices = jnp.full(
            (batch_size, max_num_frames + 1, 2 * (max_num_labels + 1)), fill_value=-jnp.inf
        )
        search_lattices = search_lattices.at[:, 0, 0].set(0.0)
        init_state = AlignmentLoopState(
            step=0,
            stop_alignment=jnp.array([False] * batch_size, dtype=jnp.bool_),
            search_lattices=search_lattices,
            backtrace=jnp.full(
                (batch_size, max_num_frames + 1, 2 * (max_num_labels + 1)), fill_value=-1
            ),
        )
        return init_state


def search_on_lattices(
    log_pos: Tensor,
    log_pos_paddings: Tensor,
    labels: Tensor,
    label_paddings: Tensor,
    blank_id: int,
    loop: Literal["lax", "python"] = "lax",
) -> AlignmentLoopState:
    batch_size, max_num_frames, _ = log_pos.shape
    _, max_num_labels = labels.shape
    frame_lengths = jnp.sum(1.0 - log_pos_paddings, axis=-1).astype(jnp.int32)

    log_pos = jnp.where(log_pos_paddings[:, :, None], -jnp.inf, log_pos)
    prev_extend_labels = calculate_prev_extend_labels(labels=labels, label_paddings=label_paddings)
    # [B, 2*(max_num_labels + 1), 3]

    init_state = AlignmentLoopState.init(batch_size, max_num_frames, max_num_labels)

    def _loop_cond(state: AlignmentLoopState) -> bool:
        not_at_end = state.step < max_num_frames + 1
        terminate_early = jnp.all(state.stop_alignment)
        return jnp.logical_and(not_at_end, ~terminate_early)

    def _loop_body(state: AlignmentLoopState) -> AlignmentLoopState:
        step = state.step

        def zero_step(state: AlignmentLoopState) -> AlignmentLoopState:
            state = state._replace(step=state.step + 1)
            return state

        def non_zero_step(state: AlignmentLoopState) -> AlignmentLoopState:
            step = state.step
            max_values, max_indices = find_max_along_last_axis(
                value=state.search_lattices[:, step - 1, :], indices=prev_extend_labels
            )
            # Both shape: [B, 2*(max_num_labels+1)]

            # For odd-index extend label, using blank posterior
            search_lattices = state.search_lattices.at[:, step, 1::2].set(
                max_values[:, 1::2] + log_pos[:, step - 1, blank_id : blank_id + 1]
            )
            # For even-index extend label, using the log_pos indexed by ground truth label
            safe_labels = jnp.where(labels < 0, 0, labels)
            # pylint: disable-next=invalid-unary-operand-type
            transit_cost = -jnp.take_along_axis(
                log_pos[:, step - 1, :], safe_labels, axis=-1
            )  # shape: [B, num_labels]
            transit_cost = jnp.where(label_paddings, jnp.inf, transit_cost)
            search_lattices = search_lattices.at[:, step, 2::2].set(
                max_values[:, 2::2] - transit_cost
            )
            stop_alignment = (state.step - 1) >= frame_lengths
            max_indices = jnp.where(state.stop_alignment[:, None], -1, max_indices)
            backtrace = state.backtrace.at[:, step, :].set(max_indices)
            step += 1

            return AlignmentLoopState(
                step=state.step + 1,
                stop_alignment=stop_alignment,
                search_lattices=search_lattices,
                backtrace=backtrace,
            )

        return jax.lax.cond(step == 0, zero_step, non_zero_step, state)

    if loop == "lax":
        state = jax.lax.while_loop(_loop_cond, _loop_body, init_state)
    elif loop == "python":
        state = init_state
        while _loop_cond(state):
            state = _loop_body(state)
    else:
        raise NotImplementedError(loop)

    # Also make sure state.backtrace is masked out when step > frame_length.
    steps = jnp.arange(max_num_frames + 1)[None, :, None]
    backtrace = jnp.where(steps > frame_lengths[:, None, None], -1, state.backtrace)
    state = state._replace(backtrace=backtrace)
    return state


def find_alignment_for_last_frame(
    search_lattices: Tensor, frame_length: Tensor, label_length: Tensor
) -> Tensor:
    """Finds alignment for the last frame.

    Args:
        search_lattices: a [B, T+1, 2*(max_num_frames + 1)]-shaped tensor. See `AlignmentLoopState`
            for details.
        frame_length: a [B, ]-shaped int32 tensor.
        label_length: a [B, ]-shaped int32 tensor.

    Returns:
        a [B, ]-shaped int32 tensor, indicating which label the last frame of b-th sequence aligned
            to.

    Note: this function is equivalent to the following python code:
        for b in range(0, B):
            if search_lattices[b, frame_length[b], 2 * label_length[b]] == -inf and
            search_lattices[b, frame_length[b], 2 * label_length[b] +1 ] == -inf:
                last_frame_align[b] = -1
            else:
                last_frame_align[b] = 2 * label_length
                if search_lattices[b, frame_length[b], 2 * label_length] is bigger
                else 2 * label_length + 1
    """

    def _find_alignment_for_last_frame(search_lattices, frame_length, label_length):
        end_with_label_score = search_lattices[frame_length, 2 * label_length]
        end_with_blank_score = search_lattices[frame_length, 2 * label_length + 1]
        max_score_index = jax.lax.cond(
            end_with_blank_score > end_with_label_score,
            lambda: 2 * label_length + 1,
            lambda: 2 * label_length,
        )
        align_index = jax.lax.cond(
            jnp.isinf(end_with_label_score) & jnp.isinf(end_with_blank_score),
            lambda: -1,
            lambda: max_score_index,
        )
        align_score = jnp.maximum(end_with_blank_score, end_with_label_score)
        return align_index, align_score

    return jax.vmap(_find_alignment_for_last_frame, in_axes=0, out_axes=0)(
        search_lattices, frame_length, label_length
    )


def extend_label_alignment_to_label_alignment(
    extend_label_alignment: Tensor, labels: Tensor, blank_id: int
) -> Tensor:
    """Converts alignment to extend label to groud truth labels.

    Args:
        extend_label_alignment: a [B, T]-shaped tensor;
        labels: a [B, L]-shaped tensor
        blank_id: int

    Returns:
        alignment: a [B, T]-shaped tensor.

    This is equivalent to the following python code:
    For b in range(B):
        For t in range(T-1):
            if align_to_extend_lables[b, t] == -1
                alignment[b, t] = -1
            elif align_to_extend_labels[b, t] is even:
               aligment[b, t] = labels[b, align_to_extend_labels[b, t]//2 - 1]
            else:
               aligment[b, t] = blank_id
    """

    def _per_sequence(extend_label_align: Tensor, labels: Tensor) -> Tensor:
        """Performs per-sequence converstion.

        Args:
            extend_label_align: a [T]-shaped tensor
            labels: a [L]-shaped tensor

        Returns:
            alignment: a [T]-shaped tensor
        """

        def _per_step(extend_label_align, labels):
            non_negative_res = jax.lax.cond(
                jnp.logical_and(extend_label_align % 2 == 0, extend_label_align > 0),
                lambda: labels[extend_label_align // 2 - 1],
                lambda: blank_id,
            )
            return jax.lax.cond(extend_label_align == -1, lambda: -1, lambda: non_negative_res)

        return jax.vmap(_per_step, in_axes=(0, None), out_axes=0)(extend_label_align, labels)

    return jax.vmap(_per_sequence, in_axes=0, out_axes=0)(extend_label_alignment, labels)


def alignment_to_segments(alignment: Tensor, blank_id: int) -> Tensor:
    """Converts alignment to segments.

    Args:
        alignment: a [B, T]-shaped tensor, its (b,t)-th element indicates b-th sequence t-th frame
            is aligned to label whose index is alignment[b, t]. If <0, it means it is a padding
        blank_id: a int.
    Returns:
        segments: a [B, T, 2]-shaped tensor. segments[b, l, 0] is the start time of
            l-th label of b-th sequence (inclusive), segments[b, l, 1] is the end time of
            l-th label of b-th sequence (exclusive)
    """
    batch_size, max_num_frames = alignment.shape
    # example: alignment = [[0 1 1 2 -1]]
    alignment_shifted = jnp.concat(
        [jnp.full((batch_size, 1), fill_value=-1), jnp.roll(alignment, shift=1, axis=-1)[:, 1:]],
        axis=-1,
    )
    # example: alignment_shifted = [[-1, 0, 1, 1, 2]]

    diff_mask = alignment != alignment_shifted
    # example: diff_mask = [[1, 1, 0, 1, 1]]
    # excluding blank label and paddings
    diff_mask = jnp.logical_and(diff_mask, jnp.logical_and(alignment != blank_id, alignment >= 0))
    # example: diff_mask = [[0, 1, 0, 1, 0]]

    segment_start_indices = jnp.where(
        diff_mask,
        ein_ops.repeat(jnp.arange(max_num_frames), "t -> b t", b=batch_size),
        max_num_frames + 1,
    )
    segment_start_indices = jnp.sort(segment_start_indices, axis=-1)
    segment_start_indices = jnp.where(
        segment_start_indices == max_num_frames + 1, -1, segment_start_indices
    )

    # example: segment_start_indices = [[1, 3, -1, -1, -1]]
    segment_end_indices = jnp.roll(segment_start_indices, shift=-1)
    segment_end_indices = segment_end_indices.at[:, -1].set(-1)
    # example: segment_end_indices =   [[3, 5, -1, -1, -1]]
    num_segments = jnp.sum(segment_end_indices >= 0, axis=-1)
    segment_end_indices = segment_end_indices.at[jnp.arange(batch_size), num_segments].set(
        jnp.sum(jnp.where(alignment == -1, 0, 1), axis=-1),
    )
    segments = jnp.stack([segment_start_indices, segment_end_indices], axis=-1)
    # example: segments = [[
    #   [1, 3],
    #   [3, 5],
    #   [-1, -1],
    #   [-1, -1],
    #   [-1, -1],
    # ]]
    return segments


def traceback_to_alignment(
    align_state: AlignmentLoopState,
    frame_paddings: Tensor,
    labels: Tensor,
    label_paddings: Tensor,
    blank_id: int,
    loop: Literal["python", "lax"] = "lax",
) -> Tuple[Tensor, Tensor]:
    """Gets alignment from traceback.

    Args:
        align_state: a dataclass object, see more on `AlignmentLoopState`;
        frame_paddings: a [B, T]-shaped bool tensor;
        labels: a [B, L]-shaped int32 tensor;
        label_paddings: a [B, L]-shaped bool tensor;
        blank_id: a int;
        loop: which loop style to use;

    Returns:
        alignment: a [B, T] int32 tensor; alignment[b, t]=k means the t-th frame of b-th sequence is
            aligned to label[b, k]
        alignment_score: a [B] float32 tensor

    """
    batch_size, max_num_frames = frame_paddings.shape
    align_to_extend_labels = jnp.full((batch_size, max_num_frames + 1), fill_value=-1)
    label_lengths = jnp.sum(1.0 - label_paddings, axis=-1).astype(jnp.int32)
    frame_lengths = jnp.sum(1.0 - frame_paddings, axis=-1).astype(jnp.int32)
    last_frame_align, align_score = find_alignment_for_last_frame(
        align_state.search_lattices, frame_lengths, label_lengths
    )
    align_to_extend_labels = align_to_extend_labels.at[jnp.arange(batch_size), frame_lengths].set(
        last_frame_align
    )

    class LoopState(NamedTuple):
        step: int
        align_to_extend_labels: Tensor  # [B, T+1]-shaped int32 tensor
        # The following tensor are read-only in the loop body
        last_frame_alignment: Tensor  # [B, ]-shaped int32 tensor
        label_length: Tensor  # [B, ]-shaped int32 tensor
        frame_length: Tensor  # [B, ]-shaped int32 tensor
        backtrace: Tensor  # [B, T+1, 2*(L+1)]-shaped int32 tensor

    state = LoopState(
        step=max_num_frames,
        align_to_extend_labels=align_to_extend_labels,
        last_frame_alignment=last_frame_align,
        label_length=label_lengths,
        frame_length=frame_lengths,
        backtrace=align_state.backtrace,
    )

    def _loop_body(loop_state: LoopState) -> LoopState:
        step = loop_state.step

        def _per_seq_fn(align_prev_frame, traceback) -> Tensor:
            return jax.lax.cond(
                align_prev_frame < 0, lambda: -1, lambda: traceback[align_prev_frame]
            )

        alignment = jax.vmap(_per_seq_fn, in_axes=(0, 0), out_axes=0)(
            loop_state.align_to_extend_labels[:, step], loop_state.backtrace[:, step]
        )
        alignment = jnp.where(
            loop_state.frame_length >= step,
            alignment,
            loop_state.align_to_extend_labels[:, step - 1],
        )
        align_to_extend_labels = loop_state.align_to_extend_labels.at[:, step - 1].set(alignment)
        loop_state = loop_state._replace(
            align_to_extend_labels=align_to_extend_labels,
            step=step - 1,
        )
        return loop_state

    def _loop_cond(loop_state: LoopState) -> bool:
        return loop_state.step > 0

    if loop == "lax":
        state = jax.lax.while_loop(_loop_cond, _loop_body, state)
    else:
        while _loop_cond(state):
            state = _loop_body(state)

    alignment = extend_label_alignment_to_label_alignment(
        extend_label_alignment=state.align_to_extend_labels[:, 1:], labels=labels, blank_id=blank_id
    )
    return alignment, align_score


def ctc_forced_alignment(
    log_pos: Tensor,
    log_pos_paddings: Tensor,
    labels: Tensor,
    label_paddings: Tensor,
    blank_id: int,
    loop: Literal["lax", "python"] = "lax",
) -> AlignmentOutput:
    """Batched Forced alignment which can run on XLA.

    Args:
        log_pos: a [B, T, V]-shaped log posterior tensor;
        log_pos_paddings: a [B, T]-shaped 0/1 tensor, indicating whether the corresponding frame
            is padding or not;
        labels: a [B, L]-shaped int32 tensor, indicating the label sequences
        label_paddings: a [B, L]-shaped bool tensor, indicating whether the corresponding label is
            padding or not;
        blank_id: a int, indicating the blank sym index.
        loop: which loop style to use, only use python for debugging purpose.

    Returns:
        An `AlignmentOutput` dataclass, see more details in `AlignmentOutput`.
    """
    chex.assert_rank(log_pos, 3)
    chex.assert_rank(log_pos_paddings, 2)
    chex.assert_shape(log_pos_paddings, log_pos.shape[:2])
    chex.assert_rank(labels, 2)
    chex.assert_shape(label_paddings, labels.shape)
    chex.assert_equal(log_pos.shape[0], labels.shape[0])

    search_state = search_on_lattices(
        log_pos,
        log_pos_paddings=log_pos_paddings,
        labels=labels,
        label_paddings=label_paddings,
        blank_id=blank_id,
        loop=loop,
    )

    alignment, align_score = traceback_to_alignment(
        search_state,
        frame_paddings=log_pos_paddings,
        labels=labels,
        label_paddings=label_paddings,
        blank_id=blank_id,
        loop=loop,
    )
    segments = alignment_to_segments(alignment, blank_id=blank_id)
    # Some edge cases are handled here:
    # 1. if the label length is 0, then this is non-alignable.
    # 2. if the log_pos length is 0, then this is also non-alignable.
    log_pos_lengths = jnp.sum(1.0 - log_pos_paddings, axis=-1).astype(jnp.int32)
    label_lengths = jnp.sum(1.0 - label_paddings, axis=-1).astype(jnp.int32)
    can_be_aligned = jnp.logical_and(log_pos_lengths > 0, label_lengths > 0)
    align_score = jnp.where(can_be_aligned, align_score, -jnp.inf)
    alignment = jnp.where(can_be_aligned[:, None], alignment, -1)
    segments = jnp.where(can_be_aligned[:, None, None], segments, -1)

    return AlignmentOutput(alignment=alignment, alignment_score=align_score, segments=segments)

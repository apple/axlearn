# pylint: disable=no-self-use,too-many-statements
"""Test for ctc_aligner.py"""
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.audio.aligner import ctc_aligner
from axlearn.common import test_utils
from axlearn.common.module import Tensor
from axlearn.common.utils import sequence_mask


def generate_test_data(blank_id: int, num_frames: int, vocab_size: int = 64, num_labels: int = 10):
    np.random.seed(12345)
    if num_frames < 5 * (num_labels - 1) + 1:
        raise ValueError(
            f"We need at least {5 * (num_labels - 1) + 1} frames when num_labels = {num_labels}"
        )
    # Not necessarily a log_posterior, but this is enough for testing purpose.
    log_posterior = np.random.randn(num_frames, vocab_size).astype(np.float32)
    ground_truth = np.random.randint(low=1, high=vocab_size, size=(num_labels,)).astype(np.int32)
    # we put a spike according to ground_truth every 5 frames
    for label_idx in range(num_labels):
        time_idx = label_idx * 5
        log_posterior[time_idx, ground_truth[label_idx]] += 10.0
        log_posterior[time_idx, blank_id] -= 10.0

    # All frames except when ground truth labels are fired will have a blank label posterior spike
    log_posterior[:, blank_id] += 10.0
    return log_posterior, ground_truth, vocab_size


def generate_batched_test_data(
    batch_size: int,
    blank_id: int,
    max_num_frames: int,
    max_num_labels: int = 10,
    vocab_size: int = 64,
):
    """Generates batched test data.

    Batched test data for ctc alignment tests are generated according to the following rules:
        - log_posterior is a [batch_size, max_num_frames, vocab_size]-shaped tensor
        - log_posterior_padding is a [batch_size, max_num_frames]-shaped 0/1 tensor
        - labels is a [batch_size, max_num_labels]-shaped tensor
        - label_paddings is a [batch_size, max_num_labels]-shaped 0/1 tensor
        - we assume that for every 5 frames, it has a spike of the corresponding label, and for the
            rest frames, blank posterior dominates (
            therefore we require max_num_frames > 5 * max_num_labels)

    Args:
        batch_size: an int
        blank_id: an int, index of blank symbol
        max_num_frames: an int
        max_num_labels: an int
        vocab_size: an int

    Returns:
        - log_posterior is a [batch_size, max_num_frames, vocab_size]-shaped tensor
        - log_posterior_padding is a [batch_size, max_num_frames]-shaped 0/1 tensor
        - labels is a [batch_size, max_num_labels]-shaped tensor
        - label_paddings is a [batch_size, max_num_labels]-shaped 0/1 tensor
        - expected_alignment_output: expected output
    """
    np.random.seed(1234)
    if max_num_frames <= 5 * max_num_labels:
        raise ValueError(f"Expecting number of frames {max_num_frames} > 5 * {max_num_labels}")

    log_posterior = np.random.normal(size=(batch_size, max_num_frames, vocab_size)).astype(
        np.float32
    )
    frame_lengths = np.random.randint(
        low=1, high=max_num_frames, size=(batch_size,), dtype=np.int32
    )
    log_pos_paddings = 1.0 - sequence_mask(
        lengths=jnp.array(frame_lengths), max_len=max_num_frames, dtype=jnp.int32
    )
    label_lengths = (frame_lengths + 4) // 5
    label_paddings = 1.0 - np.array(
        sequence_mask(lengths=jnp.array(label_lengths), max_len=max_num_labels, dtype=jnp.int32)
    )
    labels = np.random.randint(
        low=1, high=vocab_size, size=(batch_size, max_num_labels), dtype=np.int32
    )
    labels = np.where(label_paddings, -1, labels)

    expected_alignment = np.full((batch_size, max_num_frames), dtype=np.int32, fill_value=-1)
    expected_segments = np.full((batch_size, max_num_frames, 2), dtype=np.int32, fill_value=-1)
    expected_alignment_score = np.zeros((batch_size,), dtype=np.float32)
    for b in range(batch_size):
        for t in range(max_num_frames):
            if t < frame_lengths[b]:
                if t % 5 == 0:
                    label_index = t // 5
                    assert labels[b, label_index] >= 0
                    log_posterior[b, t, labels[b, label_index]] += 10.0
                    log_posterior[b, t, blank_id] -= 10.0
                    expected_alignment[b, t] = labels[b, label_index]
                    expected_alignment_score[b] += log_posterior[b, t, labels[b, label_index]]
                else:
                    log_posterior[b, t, blank_id] += 10.0
                    expected_alignment[b, t] = blank_id
                    expected_alignment_score[b] += log_posterior[b, t, blank_id]

        for l in range(label_lengths[b]):
            expected_segments[b, l, 0] = l * 5
            expected_segments[b, l, 1] = min((l + 1) * 5, frame_lengths[b])

    expected_alignment_output = ctc_aligner.AlignmentOutput(
        alignment=expected_alignment,
        alignment_score=expected_alignment_score,
        segments=expected_segments,
    )

    return (
        jnp.array(log_posterior),
        jnp.array(log_pos_paddings, jnp.bool),
        jnp.array(labels),
        jnp.array(label_paddings, jnp.bool),
    ), expected_alignment_output


class CtcAlignerTest(test_utils.TestCase):
    def test_align(self):
        log_posterior, ground_truth, vocab_size = generate_test_data(blank_id=0, num_frames=50)
        aligner = ctc_aligner.CtcAligner(vocab_size=vocab_size, blank_id=0)
        alignment = aligner.align(log_posterior=log_posterior, ground_truth=ground_truth)
        expected_alignment = np.zeros((50,), dtype=np.int32)
        for label_idx, label in enumerate(ground_truth):
            expected_alignment[label_idx * 5] = label
        self.assertNestedEqual(alignment, expected_alignment)
        boundary = aligner.alignment_to_boundary(alignment, ground_truth)
        self.assertNestedEqual(
            boundary,
            np.array(
                [
                    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                ],
                dtype=np.int32,
            ),
        )

    def test_align2(self):
        log_posterior, ground_truth, vocab_size = generate_test_data(blank_id=0, num_frames=46)
        aligner = ctc_aligner.CtcAligner(vocab_size=vocab_size, blank_id=0)
        alignment = aligner.align(log_posterior=log_posterior, ground_truth=ground_truth)
        expected_alignment = np.zeros((46,), dtype=np.int32)
        for label_idx, label in enumerate(ground_truth):
            expected_alignment[label_idx * 5] = label
        self.assertNestedEqual(alignment, expected_alignment)
        boundary = aligner.alignment_to_boundary(alignment, ground_truth)
        self.assertNestedEqual(
            boundary,
            np.array(
                [
                    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                    [5, 10, 15, 20, 25, 30, 35, 40, 45, 46],
                ],
                dtype=np.int32,
            ),
        )

    def test_alignment_to_boundary(self):
        alignment = np.array([5, 0, 6, 6, 6, 0, 7, 7, 0, 0, 8], dtype=np.int32)
        ground_truth = np.array([5, 6, 7, 8], dtype=np.int32)
        aligner = ctc_aligner.CtcAligner(vocab_size=64, blank_id=0)
        boundary = aligner.alignment_to_boundary(alignment, ground_truth)
        self.assertNestedEqual(
            boundary,
            np.array(
                [[0, 2, 6, 10], [2, 6, 10, 11]],
                dtype=np.int32,
            ),
        )


class CalculatePrevExtendLabelsTest(test_utils.TestCase):
    def setUp(self):
        super().setUp()
        self.labels = jnp.array(
            [
                [1, 2, 3, -1, -1],
                [1, 1, 3, 3, -1],
            ],
            dtype=jnp.int32,
        )
        self.label_paddings = jnp.array(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
            ],
            jnp.bool,
        )

    @classmethod
    def _ref_prev_extend_labels(cls, labels: Tensor, label_paddings: Tensor) -> Tensor:
        labels_np = np.array(labels)
        label_paddings_np = np.array(label_paddings)
        label_lengths = np.sum(1.0 - label_paddings_np, axis=-1)

        batch_size, num_labels = labels_np.shape
        prev_extend_labels = np.full(
            (batch_size, 2 * (num_labels + 1), 3), fill_value=-1, dtype=np.int32
        )

        for b in range(batch_size):
            for step in range(1, 2 * (num_labels + 1)):
                if step % 2 == 0:
                    t = step // 2 - 1
                    if label_paddings_np[b, t]:
                        prev_extend_labels[b, step] = np.array([-1] * 3, dtype=np.int32)
                    elif labels_np[b, t - 1] == labels_np[b, t]:
                        prev_extend_labels[b, step] = np.array([step - 1, step, -1], dtype=np.int32)
                    else:
                        prev_extend_labels[b, step] = np.array(
                            [step - 2, step - 1, step], dtype=np.int32
                        )
                else:
                    if step >= 2 * (label_lengths[b] + 1):
                        prev_extend_labels[b, step] = np.array([-1] * 3, dtype=np.int32)
                    else:
                        prev_extend_labels[b, step] = np.array([step - 1, step, -1], dtype=np.int32)
        return prev_extend_labels

    def test_base(self):
        prev_extend_labels = ctc_aligner.calculate_prev_extend_labels(
            self.labels, self.label_paddings
        )
        expected_prev_extend_labels = self._ref_prev_extend_labels(self.labels, self.label_paddings)
        self.assertNestedEqual(prev_extend_labels, expected_prev_extend_labels)

    def test_many(self):
        def _generate_test_case(max_label_length: int, batch_size: int):
            labels = np.random.randint(
                low=0, high=16, size=(batch_size, max_label_length), dtype=np.int32
            )
            label_length = np.random.randint(
                low=0, high=max_label_length, size=(batch_size,), dtype=np.int32
            )
            label_paddings = 1 - np.array(
                sequence_mask(
                    lengths=jnp.array(label_length), max_len=max_label_length, dtype=np.int32
                )
            )
            return jnp.array(labels), jnp.array(label_paddings, jnp.bool)

        np.random.seed(12345)
        for _ in range(100):
            labels, label_paddings = _generate_test_case(max_label_length=32, batch_size=4)
            prev_extend_labels = ctc_aligner.calculate_prev_extend_labels(labels, label_paddings)
            ref_prev_extend_labels = self._ref_prev_extend_labels(labels, label_paddings)
            self.assertNestedEqual(prev_extend_labels, ref_prev_extend_labels)


class FindMaxAlongLastAxisTest(test_utils.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(1234)
        self.value = np.random.normal(size=(2, 4)).astype(np.float32)
        self.indices = np.full((2, 4, 3), fill_value=-1)
        self.value[0, 1] = 100.0
        self.value[1, 2] = 50.0
        self.value[1, 3] = 100.0

        self.indices[0, 1, :] = np.array([0, 1, -1], dtype=np.int32)
        self.indices[0, 2, :] = np.array([0, 1, 2], dtype=np.int32)
        self.indices[1, 2, :] = np.array([1, 2, -1], dtype=np.int32)
        self.indices[1, 3, :] = np.array([1, 2, 3], dtype=np.int32)

    def test_base(self):
        max_values, max_indices = ctc_aligner.find_max_along_last_axis(self.value, self.indices)

        expected_max_values = np.full((2, 4), fill_value=-np.inf, dtype=np.float32)
        expected_max_indices = np.full((2, 4), fill_value=-1, dtype=np.int32)
        expected_max_values[0, 1] = 100.0
        expected_max_values[0, 2] = 100.0
        expected_max_values[1, 2] = 50.0
        expected_max_values[1, 3] = 100.0

        expected_max_indices[0, 1] = 1
        expected_max_indices[0, 2] = 1
        expected_max_indices[1, 2] = 2
        expected_max_indices[1, 3] = 3

        self.assertNestedEqual(max_values, expected_max_values)
        self.assertNestedEqual(max_indices, expected_max_indices)


class SearchOnLatticesTest(test_utils.TestCase):
    @classmethod
    def _search_max_and_index(cls, index_to_value: Dict[int, float]):
        sorted_items = sorted(index_to_value.items(), key=lambda item: item[1], reverse=True)
        return sorted_items[0]

    def setUp(self):
        # We consider the following unittest case:
        # B=2, T=4, V=3 (blank/a/b), blank_id = 0
        # The first label sequence is "a"; the second label sequence is "a a b"; and the
        # correspoding log_pos is:
        #   The first sequence:
        #       [
        #           [0.9 0.1 0.0]       #    --> blank
        #           [0.1 0.8 0.1]       #    --> a
        #           [0.4 0.6 0.0]       #    --> a
        #           [-inf -inf -inf]    #    padding
        #       ]
        #   The second sequence:
        #       [
        #           [0.1 0.9 0.0]       #   --> a
        #           [0.9 0.1 0.0]       #   --> blank
        #           [0.1 0.9 0.0]       #   --> a
        #           [0.0 0.1 0.9]       #   --> b
        #       ]
        super().setUp()
        batch_size = 2
        max_num_frames = 4
        vocab_size = 3
        max_num_labels = 3

        self.log_pos = np.full((batch_size, max_num_frames, vocab_size), fill_value=-np.inf)
        self.log_pos[0, :, :] = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.4, 0.6, 0.0],
                [-float("inf"), -float("inf"), -float("inf")],
            ],
            dtype=np.float32,
        )
        self.log_pos[1, :, :] = np.array(
            [
                [0.1, 0.9, 0.0],
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.1, 0.9],
            ],
            dtype=np.float32,
        )
        self.log_pos_paddings = np.array([[0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.bool_)
        self.labels = np.array([[1, -1, -1], [1, 1, 2]], dtype=np.int32)
        self.label_paddings = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.bool_)

        self.expected_search_lattices = np.full(
            (batch_size, max_num_frames + 1, 2 * (max_num_labels + 1)),
            fill_value=-np.inf,
            dtype=np.float32,
        )
        self.expected_search_lattices[:, 0, 0] = 0

        self.expected_backtrace = np.full(
            (batch_size, max_num_frames + 1, 2 * (max_num_labels + 1)),
            fill_value=-1,
            dtype=np.int32,
        )

        # For the first sequence:
        #   - at the 0-th frame (index 1),
        self.expected_search_lattices[0, 1, 1] = self.log_pos[0, 0, 0]
        self.expected_backtrace[0, 1, 1] = 0
        self.expected_search_lattices[0, 1, 2] = self.log_pos[0, 0, 1]
        self.expected_backtrace[0, 1, 2] = 0
        #   - at the 1-st frame (index 2),
        self.expected_search_lattices[0, 2, 1] = (
            self.expected_search_lattices[0, 1, 1] + self.log_pos[0, 1, 0]
        )
        self.expected_backtrace[0, 2, 1] = 1
        # There are 2 possible paths leading to (2, 2): (1, 1) and (1, 2)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[0, 1, idx] for idx in (1, 2)}
        )
        self.expected_search_lattices[0, 2, 2] = max_value + self.log_pos[0, 1, 1]
        self.expected_backtrace[0, 2, 2] = max_index
        self.expected_search_lattices[0, 2, 3] = (
            self.expected_search_lattices[0, 1, 2] + self.log_pos[0, 1, 0]
        )
        self.expected_backtrace[0, 2, 3] = 2
        #   - at the 2-nd frame (index 3),
        self.expected_search_lattices[0, 3, 1] = (
            self.expected_search_lattices[0, 2, 1] + self.log_pos[0, 2, 0]
        )
        self.expected_backtrace[0, 3, 1] = 1
        # There are 2 possible paths leading to (3, 2): (2, 1) and (2, 2)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[0, 2, idx] for idx in (1, 2)}
        )
        self.expected_search_lattices[0, 3, 2] = max_value + self.log_pos[0, 2, 1]
        self.expected_backtrace[0, 3, 2] = max_index
        # There are 2 possible paths  leading to (3, 3): (2, 2) and (2, 3)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[0, 2, idx] for idx in (2, 3)}
        )
        self.expected_search_lattices[0, 3, 3] = max_value + self.log_pos[0, 2, 0]
        self.expected_backtrace[0, 3, 3] = max_index

        # For the second sequence:
        #   - at the 0-th frame (index 1),
        self.expected_search_lattices[1, 1, 1] = self.log_pos[1, 0, 0]
        self.expected_backtrace[1, 1, 1] = 0
        self.expected_search_lattices[1, 1, 2] = self.log_pos[1, 0, 1]
        self.expected_backtrace[1, 1, 2] = 0
        #   - at the 1-th frame (index 2),
        self.expected_search_lattices[1, 2, 1] = (
            self.expected_search_lattices[1, 1, 1] + self.log_pos[1, 1, 0]
        )
        self.expected_backtrace[1, 2, 1] = 1
        # There are 2 possible paths leading to (2, 2): (1, 1) and (1, 2)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[1, 1, idx] for idx in (1, 2)}
        )
        self.expected_search_lattices[1, 2, 2] = max_value + self.log_pos[1, 1, 1]
        self.expected_backtrace[1, 2, 2] = max_index
        self.expected_search_lattices[1, 2, 3] = (
            self.expected_search_lattices[1, 1, 2] + self.log_pos[1, 1, 0]
        )
        self.expected_backtrace[1, 2, 3] = 2

        #   - at the 2-nd frame (index 3),
        self.expected_search_lattices[1, 3, 1] = (
            self.expected_search_lattices[1, 2, 1] + self.log_pos[1, 2, 0]
        )
        self.expected_backtrace[1, 3, 1] = 1
        # There are 2 possible paths leading to (3, 2): (2, 1) and (2, 2)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[1, 2, idx] for idx in (1, 2)}
        )
        self.expected_search_lattices[1, 3, 2] = max_value + self.log_pos[1, 2, 1]
        self.expected_backtrace[1, 3, 2] = max_index
        # There are 2 possible paths  leading to (3, 3): (2, 2) and (2, 3)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[1, 2, idx] for idx in (2, 3)}
        )
        self.expected_search_lattices[1, 3, 3] = max_value + self.log_pos[1, 2, 0]
        self.expected_backtrace[1, 3, 3] = max_index
        # There are 2 possible paths  leading to (3, 4): (2, 3) and (2, 4)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[1, 2, idx] for idx in (3, 4)}
        )
        self.expected_search_lattices[1, 3, 4] = max_value + self.log_pos[1, 2, 1]
        self.expected_backtrace[1, 3, 4] = max_index
        #   - at the 3-rd frame (index 4),
        self.expected_search_lattices[1, 4, 1] = (
            self.expected_search_lattices[1, 3, 1] + self.log_pos[1, 3, 0]
        )
        self.expected_backtrace[1, 4, 1] = 1
        # There are 2 possible paths leading to (4, 2): (3, 1) and (3, 2)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[1, 3, idx] for idx in (1, 2)}
        )
        self.expected_search_lattices[1, 4, 2] = max_value + self.log_pos[1, 3, 1]
        self.expected_backtrace[1, 4, 2] = max_index
        # There are 2 possible paths  leading to (4, 3): (3, 2) and (3, 3)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[1, 3, idx] for idx in (2, 3)}
        )
        self.expected_search_lattices[1, 4, 3] = max_value + self.log_pos[1, 3, 0]
        self.expected_backtrace[1, 4, 3] = max_index
        # There are 2 possible paths leading to (4, 4): (3, 3) and (3, 4)
        (max_index, max_value) = self._search_max_and_index(
            {idx: self.expected_search_lattices[1, 3, idx] for idx in (3, 4)}
        )
        self.expected_search_lattices[1, 4, 4] = max_value + self.log_pos[1, 3, 1]
        self.expected_backtrace[1, 4, 4] = max_index
        self.expected_search_lattices[1, 4, 5] = (
            self.expected_search_lattices[1, 3, 4] + self.log_pos[1, 3, 0]
        )
        self.expected_backtrace[1, 4, 5] = 4
        self.expected_search_lattices[1, 4, 6] = (
            self.expected_search_lattices[1, 3, 4] + self.log_pos[1, 3, 2]
        )
        self.expected_backtrace[1, 4, 6] = 4

    @parameterized.parameters(("python",), ("lax",))
    def test_base(self, loop):
        state = ctc_aligner.search_on_lattices(
            log_pos=jnp.array(self.log_pos),
            log_pos_paddings=jnp.array(self.log_pos_paddings),
            labels=jnp.array(self.labels),
            label_paddings=jnp.array(self.label_paddings),
            blank_id=0,
            loop=loop,
        )
        search_lattice = np.array(state.search_lattices)
        search_lattice = np.where(np.isinf(search_lattice), 0.0, search_lattice)
        expected_search_lattice = np.where(
            np.isinf(self.expected_search_lattices), 0.0, self.expected_search_lattices
        )
        self.assertNestedAllClose(search_lattice, expected_search_lattice)
        self.assertNestedEqual(state.backtrace, self.expected_backtrace)

    @parameterized.parameters(("python",), ("lax",))
    def test_to_alignment(self, loop):
        state = ctc_aligner.search_on_lattices(
            log_pos=jnp.array(self.log_pos),
            log_pos_paddings=jnp.array(self.log_pos_paddings),
            labels=jnp.array(self.labels),
            label_paddings=jnp.array(self.label_paddings),
            blank_id=0,
            loop=loop,
        )
        alignment, align_score = ctc_aligner.traceback_to_alignment(
            state,
            frame_paddings=jnp.array(self.log_pos_paddings),
            labels=jnp.array(self.labels),
            label_paddings=jnp.array(self.label_paddings),
            blank_id=0,
            loop=loop,
        )
        self.assertNestedEqual(
            alignment,
            jnp.array(
                [
                    [0, 1, 1, -1],
                    [1, 0, 1, 2],
                ],
                dtype=jnp.int32,
            ),
        )
        self.assertNestedAllClose(
            align_score,
            jnp.array([2.30, 3.60], dtype=jnp.float32),
            atol=1e-5,
        )

    @parameterized.parameters(("python",), ("lax",))
    def test_no_alignment(self, loop):
        # For the second sequence, its labels are "a a b", if there are only 3 frames,
        # no valid aligment can be found.
        # In this unittest, we change the second sequence to only 3 frames.
        self.log_pos_paddings[1, -1] = 1
        state = ctc_aligner.search_on_lattices(
            log_pos=jnp.array(self.log_pos),
            log_pos_paddings=jnp.array(self.log_pos_paddings),
            labels=jnp.array(self.labels),
            label_paddings=jnp.array(self.label_paddings),
            blank_id=0,
            loop=loop,
        )
        alignment, align_score = ctc_aligner.traceback_to_alignment(
            state,
            frame_paddings=jnp.array(self.log_pos_paddings),
            labels=jnp.array(self.labels),
            label_paddings=jnp.array(self.label_paddings),
            blank_id=0,
            loop=loop,
        )
        self.assertNestedEqual(alignment[1, :], jnp.array([-1] * 4, dtype=jnp.int32))
        self.assertTrue(jnp.isinf(align_score[1]))


class FindAlignmentForLastFrameTest(test_utils.TestCase):
    def setUp(self):
        super().setUp()
        # pylint: disable=invalid-name
        T = 4
        L = 3
        search_lattices = np.full((3, (T + 1), 2 * (L + 1)), fill_value=-np.inf)
        frame_lengths = np.array([3, 4, 3], dtype=np.int32)
        label_lengths = np.array([2, 3, 2], dtype=np.int32)

        # the first sequence, both paths (ending with blank or ending with label) are viable.
        search_lattices[0, frame_lengths[0], 2 * label_lengths[0]] = 10.0
        search_lattices[0, frame_lengths[0], 2 * label_lengths[0] + 1] = 20.0
        # the second sequence, neighter paths (ending with blank or ending with label) are viable.
        search_lattices[1, frame_lengths[1], 2 * label_lengths[1]] = -np.inf
        search_lattices[1, frame_lengths[1], 2 * label_lengths[1] + 1] = -np.inf
        # the third sequence, only path ending with label is viable
        search_lattices[2, frame_lengths[2], 2 * label_lengths[2]] = 10.0
        search_lattices[2, frame_lengths[2], 2 * label_lengths[2] + 1] = -np.inf

        self.search_lattices = jnp.array(search_lattices)
        self.frame_lengths = jnp.array(frame_lengths)
        self.label_lengths = jnp.array(label_lengths)

        self.expected_align_index = jnp.array(
            [2 * label_lengths[0] + 1, -1, 2 * label_lengths[2]], dtype=jnp.int32
        )
        self.expected_align_score = jnp.array([20.0, -jnp.inf, 10.0])
        # pylint: enable=invalid-name

    def test_base(self):
        last_frame_alignment, last_frame_align_score = ctc_aligner.find_alignment_for_last_frame(
            self.search_lattices, self.frame_lengths, self.label_lengths
        )

        self.assertNestedEqual(self.expected_align_index, last_frame_alignment)
        self.assertNestedEqual(self.expected_align_score, last_frame_align_score)


class ExtendLabelAlignmentToAlignmentTest(test_utils.TestCase):
    def setUp(self):
        super().setUp()
        # we use the following alignment as test case:
        # num_frames = 4, num_labels = 3, labels = [a, a, b]
        # backtrace, a (5, 6)-shaped Tensor, is like
        # (for simiplicty, we only show one viable path, B means blank)
        #   B  -1  -1   -1   -1   -1
        #   b  -1  -1   -1   -1    4
        #   B  -1  -1   -1   -1   -1
        #   a  -1  -1   -1    3   -1
        #   B  -1  -1    2   -1   -1
        #   a  -1   0   -1   -1   -1
        #   B  -1  -1   -1   -1   -1
        #   o  -1  -1   -1   -1   -1
        #      o   x0   x1   x2   x3
        #
        # Therefore, the extend_label_alignment is [2 3 4 6]
        # and the alignment should be [1, 0, 1, 2] (or [a, B, a, b])
        self.extend_label_alignment = jnp.array([[2, 3, 4, 6]], dtype=jnp.int32)
        self.labels = jnp.array([[1, 1, 2, -1]], dtype=jnp.int32)
        self.expected_alignment = jnp.array([[1, 0, 1, 2]], dtype=jnp.int32)

    def test_base(self):
        alignment = ctc_aligner.extend_label_alignment_to_label_alignment(
            extend_label_alignment=self.extend_label_alignment, labels=self.labels, blank_id=0
        )
        self.assertNestedEqual(self.expected_alignment, alignment)


class AlignmentToSegmentsTest(test_utils.TestCase):
    def setUp(self):
        super().setUp()
        self.alignment = jnp.array(
            [
                [0, 1, 1, 2, -1],
                [1, 2, 2, 3, 3],
            ],
            dtype=jnp.int32,
        )
        expected_segments = np.zeros((2, 5, 2), dtype=jnp.int32)
        expected_segments[0, ...] = np.array(
            [[1, 3], [3, 4], [-1, -1], [-1, -1], [-1, -1]], dtype=jnp.int32
        )
        expected_segments[1, ...] = np.array(
            [
                [0, 1],
                [1, 3],
                [3, 5],
                [-1, -1],
                [-1, -1],
            ],
            dtype=jnp.int32,
        )
        self.expected_segments = jnp.array(expected_segments)

    def test_base(self):
        segments = ctc_aligner.alignment_to_segments(self.alignment, blank_id=0)
        self.assertNestedEqual(segments, self.expected_segments)


class CtcForcedAlignmentTest(test_utils.TestCase):
    @parameterized.parameters(("lax",), ("python",))
    def test_align1(self, loop):
        (
            (log_pos, log_pos_paddings, labels, label_paddings),
            expected_output,
        ) = generate_batched_test_data(
            batch_size=2, blank_id=0, max_num_frames=32, max_num_labels=5, vocab_size=64
        )

        alignment_output = ctc_aligner.ctc_forced_alignment(
            log_pos=log_pos,
            log_pos_paddings=log_pos_paddings,
            labels=labels,
            label_paddings=label_paddings,
            blank_id=0,
            loop=loop,
        )
        self.assertNestedAllClose(alignment_output.alignment_score, expected_output.alignment_score)
        self.assertNestedEqual(alignment_output.alignment, expected_output.alignment)
        self.assertNestedEqual(alignment_output.segments, expected_output.segments)

    def test_align_with_jit(self):
        (
            (log_pos, log_pos_paddings, labels, label_paddings),
            expected_output,
        ) = generate_batched_test_data(
            batch_size=2, blank_id=0, max_num_frames=32, max_num_labels=5, vocab_size=64
        )
        blank_id = 0

        def _fn(log_pos, log_pos_paddings, labels, label_paddings):
            output = ctc_aligner.ctc_forced_alignment(
                log_pos=log_pos,
                log_pos_paddings=log_pos_paddings,
                labels=labels,
                label_paddings=label_paddings,
                blank_id=blank_id,
            )
            return output.asdict()

        jitted_forced_alignment = jax.jit(_fn)
        alignment_output = jitted_forced_alignment(
            log_pos=log_pos,
            log_pos_paddings=log_pos_paddings,
            labels=labels,
            label_paddings=label_paddings,
        )
        self.assertNestedAllClose(
            alignment_output["alignment_score"], expected_output.alignment_score
        )
        self.assertNestedEqual(alignment_output["alignment"], expected_output.alignment)
        self.assertNestedEqual(alignment_output["segments"], expected_output.segments)

    def test_not_alignable(self):
        # In this unittest, we test whether our algorithm is able to deal with cases that CTC
        # cannot produce an alignment. We set log_pos has 4 frames, but the its labels are
        # "a a b b", because a->a and b->b transition need an additional blank symbol, this label
        # is not able to align with log posterior that only has 4 frames.

        log_pos = jax.random.normal(jax.random.PRNGKey(123), shape=(1, 4, 8), dtype=jnp.float32)
        log_pos_paddings = jnp.zeros((1, 4), dtype=jnp.bool)
        labels = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
        label_paddings = jnp.zeros((1, 4), dtype=jnp.bool)

        alignment_output = ctc_aligner.ctc_forced_alignment(
            log_pos=log_pos,
            log_pos_paddings=log_pos_paddings,
            labels=labels,
            label_paddings=label_paddings,
            blank_id=0,
        )
        # pylint: disable=unsubscriptable-object
        self.assertEqual(alignment_output.alignment_score[0].item(), float("-inf"))
        # pylint: enable=unsubscriptable-object
        self.assertNestedEqual(alignment_output.alignment, jnp.array([[-1] * 4], dtype=jnp.int32))

    def _check_alignment_output_when_not_alignable(self, alignment_output):
        align_score = alignment_output.alignment_score
        segments = alignment_output.segments
        alignment = alignment_output.alignment
        self.assertNestedEqual(
            align_score,
            np.full(shape=align_score.shape, fill_value=-float("inf"), dtype=np.float32),
        )
        self.assertNestedEqual(
            segments,
            np.full(shape=segments.shape, fill_value=-1, dtype=np.int32),
        )
        self.assertNestedEqual(
            alignment,
            np.full(shape=alignment.shape, fill_value=-1, dtype=np.int32),
        )

    def test_zero_label_length(self):
        log_pos = jax.random.normal(jax.random.PRNGKey(123), shape=(1, 4, 8), dtype=jnp.float32)
        log_pos_paddings = jnp.zeros((1, 4), dtype=jnp.bool)
        labels = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
        label_paddings = jnp.ones((1, 4), dtype=jnp.bool)
        alignment_output = ctc_aligner.ctc_forced_alignment(
            log_pos=log_pos,
            log_pos_paddings=log_pos_paddings,
            labels=labels,
            label_paddings=label_paddings,
            blank_id=0,
        )
        self._check_alignment_output_when_not_alignable(alignment_output)

    def test_zero_frame_length(self):
        log_pos = jax.random.normal(jax.random.PRNGKey(123), shape=(1, 4, 8), dtype=jnp.float32)
        log_pos_paddings = jnp.ones((1, 4), dtype=jnp.bool)
        labels = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
        label_paddings = jnp.zeros((1, 4), dtype=jnp.bool)
        alignment_output = ctc_aligner.ctc_forced_alignment(
            log_pos=log_pos,
            log_pos_paddings=log_pos_paddings,
            labels=labels,
            label_paddings=label_paddings,
            blank_id=0,
        )
        self._check_alignment_output_when_not_alignable(alignment_output)


if __name__ == "__main__":
    absltest.main()

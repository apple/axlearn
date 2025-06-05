# Copyright © 2023 Apple Inc.

"""Tests for input ranking."""
from collections.abc import Sequence

import tensorflow as tf
from absl.testing import parameterized

from axlearn.common.input_ranking import rank_by_value
from axlearn.common.test_utils import TestCase


class RankByValueTest(TestCase):
    @parameterized.parameters(
        {
            "input_sequence": [0.2, 0.9, 0.5, 0.8, 0.7],
            "expected_ranks": [1, 5, 2, 4, 3],
            "ascending": True,
            "allow_ties": False,
        },
        {
            "input_sequence": [0.3, 0.4, 0.2, 0.7],
            "expected_ranks": [3, 2, 4, 1],
            "ascending": False,
            "allow_ties": False,
        },
        {
            "input_sequence": [0.5, 0.6, 0.3, 0.1, 0.1],
            "expected_ranks": [4, 5, 3, 1, 2],
            "ascending": True,
            "allow_ties": False,
        },
        {
            "input_sequence": [0.5, 0.6, 0.3, 0.1, 0.1],
            "expected_ranks": [4, 5, 3, 1, 1],
            "ascending": True,
            "allow_ties": True,
        },
        {
            "input_sequence": [3, 1, 2, 3, 5, 4],
            "expected_ranks": [3, 6, 5, 3, 1, 2],
            "ascending": False,
            "allow_ties": True,
        },
        {
            "input_sequence": [0.2, 0.9, 0.5, 0.8, 0.7],
            "expected_ranks": [1, 5, 2, 4, 3],
            "ascending": True,
            "allow_ties": True,
        },
        {
            "input_sequence": [],
            "expected_ranks": [],
            "ascending": True,
            "allow_ties": True,
        },
        {
            "input_sequence": [],
            "expected_ranks": [],
            "ascending": True,
            "allow_ties": False,
        },
    )
    def test_rank_by_value(
        self,
        input_sequence: Sequence[float],
        expected_ranks: Sequence[int],
        ascending: bool,
        allow_ties: bool,
    ):
        def gen():
            yield {"label": input_sequence}

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "label": tf.TensorSpec(shape=(len(input_sequence),), dtype=tf.float32),
            },
        )

        ds = rank_by_value(
            input_key="label", output_key="rank", ascending=ascending, allow_ties=allow_ties
        )(ds)
        for ex in ds.as_numpy_iterator():
            self.assertListEqual(expected_ranks, ex["rank"].tolist())

    def test_rank_by_value_unsupported_shape(self):
        def gen():
            yield {"label": [[1, 2, 3, 4], [5, 6, 6, 9]]}

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "label": tf.TensorSpec(shape=(2, 4), dtype=tf.float32),
            },
        )

        with self.assertRaisesRegex(
            NotImplementedError, "Only implemented for rank-1 tensors. Got rank-2."
        ):
            rank_by_value(input_key="label", output_key="rank", ascending=True, allow_ties=True)(ds)

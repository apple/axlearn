# Copyright © 2024 Apple Inc.

"""Tests grain inputs."""

import copy
from typing import Optional

import grain.python as grain
import jax
import numpy as np
from absl import logging
from absl.testing import parameterized
from grain._src.core.sharding import even_split

from axlearn.common.input_fake import fake_grain_source
from axlearn.common.input_grain import (
    Dataset,
    Input,
    maybe_to_iter_dataset,
    prefetch_dataset,
    rekey,
    sample_from_datasets,
    shard_dataset,
    trim_and_pack_dataset,
    unbatch,
)
from axlearn.common.test_utils import TestCase


def _range_dataset(*, start, stop, step=1, seed=None) -> Dataset:
    source = grain.RangeDataSource(start=start, stop=stop, step=step)
    ds = grain.MapDataset.source(source)
    if seed is not None:
        ds = ds.seed(seed)
    return ds


class _PlusOne(grain.MapTransform):
    def map(self, x: int) -> int:
        return x + 1


class UtilsTest(TestCase):
    """Tests processor utils."""

    @parameterized.parameters(
        dict(
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[1, 1],
            take=10,
            expected=[0, 1, 2, 3, 4, 1, 6, 3, 8, 1],
        ),
        dict(
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[2, 1],
            take=10,
            expected=[0, 2, 1, 4, 6, 3, 8, 0, 2, 1],
        ),
        dict(
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[1, 1e-9],
            take=10,
            expected=[0, 2, 4, 6, 8, 0, 2, 4, 6, 8],
        ),
    )
    def test_sample_from_datasets(
        self,
        sources: list[slice],
        weights: list[int],
        take: Optional[int],
        expected: list[int],
    ):
        ds = sample_from_datasets(
            sources=[
                _range_dataset(start=src.start, stop=src.stop, step=src.step) for src in sources
            ],
            weights=weights,
        )
        ds = ds.slice(slice(0, take))
        self.assertCountEqual(expected, list(ds))

    def test_sample_from_datasets_errors(self):
        ds = _range_dataset(start=0, stop=2)
        ds = ds.repeat()
        # Make sure that already-repeated datasets don't error.
        repeated_ds = sample_from_datasets(sources=[ds], weights=[1]).slice(slice(0, 4))
        self.assertEqual([0, 1, 0, 1], list(repeated_ds))

        # Make sure that non-map dataset raises.
        with self.assertRaisesRegex(ValueError, "MapDataset"):
            ds = ds.to_iter_dataset()
            sample_from_datasets(sources=[ds], weights=[1])

    def test_shuffle_dataset(self):
        # Test without repeat.
        ds = sample_from_datasets(
            sources=[
                _range_dataset(start=0, stop=10, step=2),
                _range_dataset(start=1, stop=5, step=2),
            ],
            weights=[2, 1],
        )
        ds = ds.slice(slice(0, 10))
        ds = ds.shuffle(seed=123)
        original = list(ds)
        self.assertEqual([1, 3, 1, 2, 6, 0, 2, 8, 4, 0], original)

        # Test with repeat.
        ds = ds.repeat(num_epochs=2)
        repeated = list(ds)
        self.assertEqual(2 * len(original), len(repeated))
        self.assertEqual(original, repeated[: len(original)])
        # Check that shuffles are different across epochs.
        self.assertNotEqual(original, repeated[len(original) :])

        # Check that shuffles don't cross epochs.
        self.assertCountEqual(repeated[: len(original)], repeated[len(original) :])

    def test_repeat_dataset(self):
        # Test repeat before shuffle.
        ds = _range_dataset(start=0, stop=10)
        ds = ds.repeat(num_epochs=2)
        ds = ds.shuffle(seed=123)
        ds = ds.slice(slice(0, 10))
        # First epoch might have elements from both epochs.
        self.assertEqual([8, 7, 5, 4, 4, 6, 1, 6, 0, 0], list(ds))

    @parameterized.parameters(
        dict(s=slice(0, 5, 2), expected=[0, 2, 4]),
        dict(s=slice(1, 4, 2), expected=[1, 3]),
        dict(s=slice(20, 24, 2), expected=[]),
    )
    def test_slice_dataset(self, s: slice, expected: list[int]):
        ds = _range_dataset(start=0, stop=10).slice(s)
        self.assertCountEqual(expected, list(ds))

    def test_batch(self):
        # [0, 1, 2, 3, 4].
        ds = _range_dataset(start=0, stop=5, seed=123)
        # [1, 2, 3, 4, 5].
        other_ds = ds.map(_PlusOne())
        # [0, 1, 2, 1, 3, 4, 2, 5, 1, 3, ...].
        mixed_ds = sample_from_datasets(sources=[ds, other_ds], weights=[1, 2])
        batched_ds = mixed_ds.batch(3).slice(slice(0, 3))
        expected = [np.array([0, 1, 2]), np.array([1, 3, 4]), np.array([2, 5, 1])]
        self.assertTrue(all(np.all(a == b) for a, b in zip(expected, batched_ds)))

        # Batch np arrays of different shapes.
        ds = fake_grain_source(
            [
                {"input_ids": np.array([1, 2, 3, 4, 5])},
                {"input_ids": np.array([6, 7, 8])},
                {"input_ids": np.array([], dtype=np.int32)},
                {"input_ids": np.array([9, 10, 11, 12])},
            ]
        )
        # By default, naive batching will raise because it'll produce ragged.
        with self.assertRaisesRegex(ValueError, "same structure"):
            print(list(ds.batch(3)))

        batched_ds = ds.batch(3, batch_fn=list)
        expected = [
            [
                {"input_ids": np.array([1, 2, 3, 4, 5])},
                {"input_ids": np.array([6, 7, 8])},
                {"input_ids": np.array([], dtype=np.int32)},
            ],
            [{"input_ids": np.array([9, 10, 11, 12])}],
        ]
        actual = list(batched_ds)
        self.assertNestedEqual(expected, actual)

    @parameterized.parameters(dict(batch_size=3), dict(batch_size=2))
    def test_unbatch(self, batch_size: int):
        # Test unbatch.
        expected = [
            {
                "input_ids": np.array([1, 2, 3, 4, 5]),
                "input_ids_segment_ids": np.array([1, 1, 1, 1, 2]),
                "input_ids_positions": np.array([0, 1, 2, 3, 0]),
            },
            {
                "input_ids": np.array([11, 12, 13, 14, 7]),
                "input_ids_segment_ids": np.array([1, 1, 1, 1, 2]),
                "input_ids_positions": np.array([0, 1, 2, 3, 0]),
            },
            {
                "input_ids": np.array([8, 0, 0, 0, 0]),
                "input_ids_segment_ids": np.array([1, 0, 0, 0, 0]),
                "input_ids_positions": np.array([0, 0, 0, 0, 0]),
            },
        ]
        ds = fake_grain_source(expected)
        # Batch + unbatch should produce the inputs.
        batched_ds = ds.batch(batch_size, drop_remainder=False)
        self.assertEqual(
            # All Tensors initially stacked: [batch_size, seq_len].
            jax.tree.leaves(list(batched_ds))[0].shape,
            (batch_size, 5),
        )
        unbatch_ds = unbatch(maybe_to_iter_dataset(batched_ds))
        actual = list(unbatch_ds)
        self.assertNestedEqual(expected, actual)

    def test_unbatch_invalid(self):
        # Test inconsistent batch dim.
        with self.assertRaisesRegex(ValueError, "batch"):
            ds = fake_grain_source(
                [{"x": np.array([1, 2]), "y": np.array([1, 2, 3])}],
            )
            ds = unbatch(maybe_to_iter_dataset(ds))
            list(ds)

    def test_unbatch_checkpoint(self):
        def convert_examples(x, rng: np.random.Generator):
            if rng.random() > 0.5:
                logging.log_first_n(logging.WARNING, "Injecting an empty example.", 1)
                return {}
            return {"value": x}

        ds = _range_dataset(start=1, stop=10)
        ds = ds.repeat(None).batch(3)
        ds = ds.map(convert_examples, seed=123)
        ds = unbatch(maybe_to_iter_dataset(ds))
        ds = iter(ds)

        max_steps = 10
        values_without_interruption: list[dict] = []
        checkpoints = []

        for _ in range(max_steps):
            checkpoints.append(ds.get_state())
            values_without_interruption.append(next(ds))

        def check(starting_step, ds):
            for i in range(starting_step, max_steps):
                actual = next(ds)
                expected = values_without_interruption[i]
                for k, v in expected.items():
                    self.assertEqual(v, actual[k], msg=f"expected={expected}, actual={actual}")

        # Try resuming from an existing iterator, to ensure that entire state is reset.
        for starting_step in range(max_steps):
            ds.set_state(checkpoints[starting_step])  # Restore using the same iterator as above.
            check(starting_step, ds)

        # Try resuming from a fresh iterator from any step and validate that outcome is the same.
        for starting_step in range(max_steps):
            ds = iter(ds)
            ds.set_state(checkpoints[starting_step])
            check(starting_step, ds)

    @parameterized.parameters(
        dict(
            feature_lens={"input_ids": 5},
            expected=[
                {
                    "input_ids": np.array([1, 2, 3, 4, 5]),
                    "input_ids_segment_ids": np.array([1, 1, 1, 1, 2]),
                    "input_ids_positions": np.array([0, 1, 2, 3, 0]),
                },
                {
                    "input_ids": np.array([11, 12, 13, 14, 7]),
                    "input_ids_segment_ids": np.array([1, 1, 1, 1, 2]),
                    "input_ids_positions": np.array([0, 1, 2, 3, 0]),
                },
                {
                    "input_ids": np.array([8, 0, 0, 0, 0]),
                    "input_ids_segment_ids": np.array([1, 0, 0, 0, 0]),
                    "input_ids_positions": np.array([0, 0, 0, 0, 0]),
                },
            ],
        ),
        dict(
            feature_lens={"input_ids": 6},
            expected=[
                {
                    "input_ids": np.array([1, 2, 3, 4, 5, 6]),
                    "input_ids_segment_ids": np.array([1, 1, 1, 1, 2, 2]),
                    "input_ids_positions": np.array([0, 1, 2, 3, 0, 1]),
                },
                {
                    "input_ids": np.array([11, 12, 13, 14, 7, 8]),
                    "input_ids_segment_ids": np.array([1, 1, 1, 1, 2, 3]),
                    "input_ids_positions": np.array([0, 1, 2, 3, 0, 0]),
                },
            ],
        ),
    )
    def test_packing(self, feature_lens: dict, expected: list):
        examples = [
            {"input_ids": np.array([1, 2, 3, 4])},
            {"input_ids": np.array([5, 6])},
            {"input_ids": np.array([11, 12, 13, 14])},
            {"input_ids": np.array([7])},
            {"input_ids": np.array([8])},
        ]

        def cast_ints(example):
            return jax.tree.map(lambda x: x.astype(np.int32), example)

        ds = fake_grain_source(examples)
        ds = trim_and_pack_dataset(maybe_to_iter_dataset(ds), feature_lengths=feature_lens)
        self.assertNestedEqual(cast_ints(expected), cast_ints(list(iter(ds))))

    def test_rekey(self):
        # Test rekey with repeat, dropping original inputs.
        examples = [{"text": 123}]
        orig_examples = copy.deepcopy(examples)
        ds = fake_grain_source(examples, repeat=2)
        ds = rekey(ds, key_map={"newtext": "text"})
        self.assertEqual([{"newtext": 123}, {"newtext": 123}], list(ds))
        # Ensure that rekey does not modify original.
        self.assertEqual(orig_examples, examples)

        # Test retaining original and nested rekey.
        examples = [{"nested": {"text": 123}}]
        orig_examples = copy.deepcopy(examples)
        ds = fake_grain_source(examples, repeat=2)
        ds = rekey(
            ds,
            key_map={"nested/newtext": "nested/text"},
            retain_original_inputs=True,
            separator="/",
        )
        self.assertEqual(
            [
                {"nested": {"newtext": 123, "text": 123}},
                {"nested": {"newtext": 123, "text": 123}},  # repeated.
            ],
            list(ds),
        )
        # Ensure that rekey does not modify original.
        self.assertEqual(orig_examples, examples)


class InputTest(parameterized.TestCase):
    """Tests Input module."""

    @parameterized.parameters(
        # A single process case.
        dict(process_index=0, process_count=1),
        # A simulated multi-process case.
        dict(process_index=0, process_count=4),
        dict(process_index=1, process_count=4),
        dict(process_index=2, process_count=4),
        dict(process_index=3, process_count=4),
        # Test when number of examples is less than number of processes. In this case, since
        # repeat=False and drop_remainder=False, the latter processes have no inputs.
        dict(process_index=0, process_count=30),
        dict(process_index=29, process_count=30),
    )
    def test_input(self, process_index: int, process_count: int):
        epoch_len, num_epochs = 10, 2
        ds = _range_dataset(start=0, stop=epoch_len, seed=123)

        def source(ds) -> Dataset:
            ds = shard_dataset(ds, process_index=process_index, process_count=process_count)
            ds = ds.shuffle().repeat(num_epochs=2)
            ds = prefetch_dataset(
                maybe_to_iter_dataset(ds),
                multiprocessing_options=grain.MultiprocessingOptions(num_workers=1),
            )
            return ds

        cfg: Input.Config = Input.default_config().set(source=lambda: source(ds))
        grain_input = cfg.set(name="test").instantiate(parent=None)
        self.assertEqual(epoch_len, len(ds))
        examples = list(grain_input)
        num_examples = num_epochs * epoch_len

        # If loading on a single process.
        if process_count == 1:
            self.assertEqual(num_examples, len(examples))
            # First epoch.
            self.assertEqual([4, 3, 2, 9, 0, 1, 8, 6, 7, 5], examples[:10])
            # Check that second epoch uses a different shuffle.
            self.assertEqual([3, 7, 1, 8, 2, 6, 9, 4, 5, 0], examples[10:])

        else:
            shard_options = grain.ShardOptions(shard_index=process_index, shard_count=process_count)

            # Each process gets a split of the source data (our source is not shuffled).
            start, end = even_split(epoch_len, shard_options)
            self.assertEqual((end - start) * num_epochs, len(examples))

            # Only the source examples assigned to this process should appear.
            start, end = even_split(epoch_len, shard_options)
            self.assertSameElements(
                list(range(epoch_len))[slice(process_index, None, process_count)], examples
            )

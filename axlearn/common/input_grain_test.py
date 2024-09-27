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

from axlearn.common.config import config_for_function
from axlearn.common.input_fake import fake_grain_source
from axlearn.common.input_grain import (
    BuildDatasetFn,
    Input,
    batch,
    chain,
    map_over_dataset,
    maybe_to_iter_dataset,
    prefetch_dataset,
    rekey,
    repeat_dataset,
    sample_from_datasets,
    shard_dataset,
    shuffle_dataset,
    slice_dataset,
    trim_and_pack_dataset,
    unbatch,
    with_processor,
)
from axlearn.common.test_utils import TestCase


def _range_dataset(*, start, stop, step=1, seed=None):
    def fn():
        source = grain.RangeDataSource(start=start, stop=stop, step=step)
        ds = grain.MapDataset.source(source)
        if seed is not None:
            ds = ds.seed(seed)
        return ds

    return fn


class _PlusOne(grain.MapTransform):
    def map(self, x: int) -> int:
        return x + 1


class _PlusRandom(grain.RandomMapTransform):
    def random_map(self, x: int, rng: np.random.Generator) -> int:
        return x + rng.integers(10)


# Must provide a seed explicitly for random callable.
@map_over_dataset(seed=123)
def _plus_random(x: int, rng: np.random.Generator) -> int:
    return x + rng.integers(10)


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
        cfg = config_for_function(sample_from_datasets).set(
            sources=[
                config_for_function(_range_dataset).set(
                    start=src.start, stop=src.stop, step=src.step
                )
                for src in sources
            ],
            weights=weights,
        )
        ds_fn: BuildDatasetFn = cfg.instantiate()
        ds = ds_fn()
        ds = ds.slice(slice(0, take))
        self.assertCountEqual(expected, list(ds))

    def test_shuffle_dataset(self):
        # Test without repeat.
        source = config_for_function(sample_from_datasets).set(
            sources=[
                config_for_function(_range_dataset).set(start=0, stop=10, step=2),
                config_for_function(_range_dataset).set(start=1, stop=5, step=2),
            ],
            weights=[2, 1],
        )
        ds_fn = with_processor(
            source, processor=chain(slice_dataset(slice(0, 10)), shuffle_dataset(seed=123))
        )
        original = list(ds_fn())
        self.assertEqual([1, 3, 1, 2, 6, 0, 2, 8, 4, 0], original)

        # Test with repeat.
        repeat_ds_fn = with_processor(ds_fn, processor=repeat_dataset(num_repeat=2))
        repeated = list(repeat_ds_fn())
        self.assertEqual(2 * len(original), len(repeated))
        self.assertEqual(original, repeated[: len(original)])
        # Check that shuffles are different across epochs.
        self.assertNotEqual(original, repeated[len(original) :])

        # Check that shuffles don't cross epochs.
        self.assertCountEqual(repeated[: len(original)], repeated[len(original) :])

    def test_repeat_dataset(self):
        # Test repeat before shuffle.
        source = config_for_function(_range_dataset).set(start=0, stop=10)
        ds_fn = with_processor(
            source,
            processor=chain(
                repeat_dataset(num_repeat=2), shuffle_dataset(seed=123), slice_dataset(slice(0, 10))
            ),
        )
        # First epoch might have elements from both epochs.
        self.assertEqual([8, 7, 5, 4, 4, 6, 1, 6, 0, 0], list(ds_fn()))

    @parameterized.parameters(
        dict(s=slice(0, 5, 2), expected=[0, 2, 4]),
        dict(s=slice(1, 4, 2), expected=[1, 3]),
        dict(s=slice(20, 24, 2), expected=[]),
    )
    def test_slice_dataset(self, s: slice, expected: list[int]):
        ds_fn = with_processor(_range_dataset(start=0, stop=10), processor=slice_dataset(s))
        ds = ds_fn()
        self.assertCountEqual(expected, list(ds))

    def test_map_over_dataset(self):
        source = _range_dataset(start=0, stop=5, seed=123)

        # Test a MapTransform.
        ds_fn = with_processor(source, processor=map_over_dataset(_PlusOne()))
        ds = ds_fn()
        self.assertCountEqual([1, 2, 3, 4, 5], list(ds))

        # Test a RandomMapTransform.
        ds_fn = with_processor(source, processor=map_over_dataset(_PlusRandom()))
        ds = ds_fn()
        # Note that the source is seeded.
        self.assertCountEqual([9, 9, 11, 7, 7], list(ds))

        # Test a callable. Must provide a seed explicitly.
        ds_fn = with_processor(source, processor=_plus_random)
        ds = ds_fn()
        self.assertCountEqual([2, 2, 6, 11, 12], list(ds))

        # Test raising when decorated on a class.
        # TODO(markblee): Investigate why this fails in CI but not in a dev env.
        # with self.assertRaisesRegex(ValueError, "instances"):
        #     map_over_dataset(_PlusOne)

    def test_chain(self):
        ds_fn = with_processor(
            _range_dataset(start=0, stop=5, seed=123),
            processor=chain(
                slice_dataset(slice(1, 3)),  # [1, 2].
                map_over_dataset(_PlusOne()),  # [2, 3].
            ),
        )
        self.assertEqual([2, 3], list(ds_fn()))

        other_ds_fn = with_processor(
            ds_fn,
            processor=chain(
                map_over_dataset(_PlusOne()),
                map_over_dataset(_PlusOne()),
            ),  # [4, 5].
        )
        self.assertEqual([4, 5], list(other_ds_fn()))

        # A mix of chains.
        mixed_ds_fn = sample_from_datasets(sources=[ds_fn, other_ds_fn], weights=[1, 2])
        # Slice the repeated dataset.
        ds_fn = with_processor(mixed_ds_fn, processor=slice_dataset(slice(4, 10)))
        self.assertEqual([4, 5, 2, 4, 5, 3], list(ds_fn()))

    def test_batch(self):
        # [0, 1, 2, 3, 4].
        ds_fn = _range_dataset(start=0, stop=5, seed=123)
        # [1, 2, 3, 4, 5].
        other_ds_fn = with_processor(ds_fn, processor=map_over_dataset(_PlusOne()))
        # [0, 1, 2, 1, 3, 4, 2, 5, 1, 3, ...].
        mixed_ds_fn = sample_from_datasets(sources=[ds_fn, other_ds_fn], weights=[1, 2])
        batched_ds_fn = with_processor(
            mixed_ds_fn, processor=chain(batch(3), slice_dataset(slice(0, 3)))
        )
        expected = [np.array([0, 1, 2]), np.array([1, 3, 4]), np.array([2, 5, 1])]
        self.assertTrue(all(np.all(a == b) for a, b in zip(expected, batched_ds_fn())))

        # Batch np arrays of different shapes.
        ds_fn = fake_grain_source(
            [
                {"input_ids": np.array([1, 2, 3, 4, 5])},
                {"input_ids": np.array([6, 7, 8])},
                {"input_ids": np.array([], dtype=np.int32)},
                {"input_ids": np.array([9, 10, 11, 12])},
            ]
        )
        # By default, naive batching will raise because it'll produce ragged.
        with self.assertRaisesRegex(ValueError, "same structure"):
            list(with_processor(ds_fn, processor=batch(3))())

        batched_ds_fn = with_processor(ds_fn, processor=batch(3, batch_fn=list))
        expected = [
            [
                {"input_ids": np.array([1, 2, 3, 4, 5])},
                {"input_ids": np.array([6, 7, 8])},
                {"input_ids": np.array([], dtype=np.int32)},
            ],
            [{"input_ids": np.array([9, 10, 11, 12])}],
        ]
        actual = list(batched_ds_fn())
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
        ds_fn = fake_grain_source(expected)
        # Batch + unbatch should produce the inputs.
        batch_ds_fn = with_processor(ds_fn, processor=batch(batch_size, drop_remainder=False))
        self.assertEqual(
            # All Tensors initially stacked: [batch_size, seq_len].
            jax.tree.leaves(list(batch_ds_fn()))[0].shape,
            (batch_size, 5),
        )
        unbatch_ds_fn = with_processor(
            batch_ds_fn, processor=chain(maybe_to_iter_dataset(), unbatch())
        )
        actual = list(unbatch_ds_fn())
        self.assertNestedEqual(expected, actual)

    def test_unbatch_invalid(self):
        # Test inconsistent batch dim.
        with self.assertRaisesRegex(ValueError, "batch"):
            ds_fn = with_processor(
                fake_grain_source(
                    [{"x": np.array([1, 2]), "y": np.array([1, 2, 3])}],
                ),
                processor=chain(maybe_to_iter_dataset(), unbatch()),
            )
            list(ds_fn())

    def test_unbatch_checkpoint(self):
        @map_over_dataset(seed=123)
        def convert_examples(x, rng: np.random.Generator):
            if rng.random() > 0.5:
                logging.log_first_n(logging.WARNING, "Injecting an empty example.", 1)
                return {}
            return {"value": x}

        ds_fn = with_processor(
            _range_dataset(start=1, stop=10),
            processor=chain(
                repeat_dataset(None), batch(3), convert_examples, maybe_to_iter_dataset(), unbatch()
            ),
        )
        ds = iter(ds_fn())

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
            ds = iter(ds_fn())
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

        ds_fn = fake_grain_source(examples)
        packed_ds_fn = with_processor(
            ds_fn, processor=chain(maybe_to_iter_dataset(), trim_and_pack_dataset(feature_lens))
        )
        self.assertNestedEqual(cast_ints(expected), cast_ints(list(iter(packed_ds_fn()))))

    def test_rekey(self):
        # Test rekey with repeat, dropping original inputs.
        examples = [{"text": 123}]
        orig_examples = copy.deepcopy(examples)
        ds_fn = with_processor(
            fake_grain_source(examples, repeat=2),
            processor=rekey({"newtext": "text"}),
        )
        self.assertEqual([{"newtext": 123}, {"newtext": 123}], list(ds_fn()))
        # Ensure that rekey does not modify original.
        self.assertEqual(orig_examples, examples)

        # Test retaining original and nested rekey.
        examples = [{"nested": {"text": 123}}]
        orig_examples = copy.deepcopy(examples)
        ds_fn = with_processor(
            fake_grain_source(examples, repeat=2),
            processor=rekey(
                {"nested/newtext": "nested/text"},
                retain_original_inputs=True,
                separator="/",
            ),
        )
        self.assertEqual(
            [
                {"nested": {"newtext": 123, "text": 123}},
                {"nested": {"newtext": 123, "text": 123}},  # repeated.
            ],
            list(ds_fn()),
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
        source = config_for_function(_range_dataset).set(start=0, stop=epoch_len, seed=123)
        processor = config_for_function(chain).set(
            args=[
                config_for_function(shard_dataset).set(
                    process_index=process_index, process_count=process_count
                ),
                config_for_function(shuffle_dataset),
                config_for_function(repeat_dataset).set(num_repeat=2),
                config_for_function(maybe_to_iter_dataset),
                config_for_function(prefetch_dataset).set(
                    multiprocessing_options=grain.MultiprocessingOptions(num_workers=1)
                ),
            ]
        )

        cfg: Input.Config = Input.default_config().set(
            source=config_for_function(with_processor).set(source=source, processor=processor),
        )
        grain_input = cfg.set(name="test").instantiate(parent=None)
        self.assertEqual(epoch_len, len(source.instantiate()()))
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

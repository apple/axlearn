# Copyright © 2024 Apple Inc.

"""Tests grain inputs."""

import copy
from functools import partial
from typing import Optional, Protocol

import grain.python as grain
import jax
import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from grain._src.core.sharding import even_split
from jax.sharding import PartitionSpec

from axlearn.common.config import config_for_function
from axlearn.common.input_dispatch import InputDispatcher
from axlearn.common.input_fake import fake_grain_source
from axlearn.common.input_grain import (
    BuildDatasetFn,
    Dataset,
    DispatchConfig,
    Input,
    RaggedTensor,
    maybe_to_iter_dataset,
    pad_for_evaluation,
    per_feed_batch,
    prefetch_dataset,
    rekey,
    sample_from_datasets,
    shard_dataset,
    shard_dataset_with_proportion,
    unbatch,
)
from axlearn.common.test_utils import TestCase


def range_dataset(*, start, stop, step=1, seed=None) -> Dataset:
    source = grain.RangeDataSource(start=start, stop=stop, step=step)
    ds = grain.MapDataset.source(source)
    if seed is not None:
        ds = ds.seed(seed)
    return ds


class _PlusOne(grain.MapTransform):
    def map(self, element: int) -> int:
        return element + 1


class UtilsTest(TestCase):
    """Tests processor utils."""

    def _test_checkpointing(self, ds: grain.DatasetIterator):
        """Utility to test ds checkpointing."""

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
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[1, 1],
            is_iter_dataset=[False, False],
            take=10,
            expected=[0, 1, 2, 3, 4, 1, 6, 3, 8, 1],
        ),
        dict(
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[2, 1],
            is_iter_dataset=[False, False],
            take=10,
            expected=[0, 2, 1, 4, 6, 3, 8, 0, 2, 1],
        ),
        dict(
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[1, 1e-9],
            is_iter_dataset=[False, False],
            take=10,
            expected=[0, 2, 4, 6, 8, 0, 2, 4, 6, 8],
        ),
        # IterDataset
        dict(
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[1, 1],
            is_iter_dataset=[True, True],
            take=10,
            expected=[0, 1, 2, 3, 4, 1, 6, 3, 8, 1],
        ),
        # Mixture of IterDataset and MapDataset.
        dict(
            sources=[slice(0, 10, 2), slice(1, 5, 2)],
            weights=[1, 1],
            is_iter_dataset=[True, False],
            take=10,
            expected=[0, 1, 2, 3, 4, 1, 6, 3, 8, 1],
        ),
    )
    def test_sample_from_datasets(
        self,
        sources: list[slice],
        weights: list[int],
        is_iter_dataset: list[bool],
        take: Optional[int],
        expected: list[int],
    ):
        sources = [
            range_dataset(start=src.start, stop=src.stop, step=src.step).repeat() for src in sources
        ]
        sources = [
            source.to_iter_dataset() if should_convert else source
            for source, should_convert in zip(sources, is_iter_dataset)
        ]
        ds = sample_from_datasets(
            sources=sources,
            weights=weights,
        )
        ds_iter = iter(ds)
        result = []
        for _ in range(take):
            result.append(next(ds_iter))
        self.assertCountEqual(expected, list(result))

    def test_sample_from_datasets_errors(self):
        ds = range_dataset(start=0, stop=2)
        ds = ds.repeat()
        # Make sure that already-repeated datasets don't error.
        repeated_ds = sample_from_datasets(sources=[ds], weights=[1]).slice(slice(0, 4))
        self.assertEqual([0, 1, 0, 1], list(repeated_ds))

    def test_shuffle_dataset(self):
        # Test without repeat.
        ds = sample_from_datasets(
            sources=[
                range_dataset(start=0, stop=10, step=2).repeat(),
                range_dataset(start=1, stop=5, step=2).repeat(),
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
        ds = range_dataset(start=0, stop=10)
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
        ds = range_dataset(start=0, stop=10).slice(s)
        self.assertCountEqual(expected, list(ds))

    def test_batch(self):
        # [0, 1, 2, 3, 4].
        ds = range_dataset(start=0, stop=5, seed=123).repeat()
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

        ds = range_dataset(start=1, stop=10)
        ds = ds.repeat(None).batch(3)
        ds = ds.random_map(convert_examples, seed=123)
        ds = unbatch(maybe_to_iter_dataset(ds))
        ds = iter(ds)
        self._test_checkpointing(ds)

    def test_unbatch_empty_batch(self):
        # Test with skip_empty_batch=True.
        ds = fake_grain_source(
            [
                {"x": np.array([]), "y": np.array([])},
                {"x": np.array([]), "y": np.array([])},
                {"x": np.array([1, 2]), "y": np.array([1, 2])},
            ]
        )
        ds = unbatch(maybe_to_iter_dataset(ds), skip_empty_batch=True)
        ds = iter(ds)
        self.assertEqual({"x": 1, "y": 1}, next(ds))
        self.assertEqual({"x": 2, "y": 2}, next(ds))

        # Test with skip_empty_batch=False.
        with self.assertRaisesRegex(AssertionError, "(0, 0)"):
            ds = fake_grain_source([{"x": np.array([]), "y": np.array([])}])
            ds = unbatch(maybe_to_iter_dataset(ds))
            list(ds)

    def test_unbatch_ragged(self):
        # Test unbatching with ragged tensors.
        ds = fake_grain_source(
            [
                {
                    "x": np.array([1, 2, 3]),
                    "y": np.array([1, 2, 3]),
                    "z": RaggedTensor(
                        [
                            np.array([1, 2, 3, 4]),
                            np.array([5, 6, 7, 8]),
                            np.array([9, 10, 11, 12]),
                        ]
                    ),
                },
            ]
        )
        ds = unbatch(maybe_to_iter_dataset(ds))
        ds = iter(ds)
        self.assertNestedEqual({"x": 1, "y": 1, "z": np.array([1, 2, 3, 4])}, next(ds))

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

    @parameterized.parameters(False, True)
    def test_pad_for_evaluation(self, to_iter_dataset: bool):
        # Ensure that number of examples is a multiple of this batch size.
        per_feed_batch_size = 3

        ds = fake_grain_source(examples=list(range(7)))
        if to_iter_dataset:
            ds = maybe_to_iter_dataset(ds)
        ds = pad_for_evaluation(ds, per_feed_batch_size=per_feed_batch_size)
        examples = list(ds)

        # Make sure length is expected.
        self.assertEqual(9, len(examples))
        # Make sure values are expected.
        self.assertEqual(list(range(7)) + [0, 0], examples)

        # Try batching.
        ds = ds.batch(3)
        self.assertNestedEqual(
            [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 0, 0])], list(ds)
        )

    def test_pad_for_evaluation_max(self):
        ds = fake_grain_source(examples=list(range(7)))
        ds = maybe_to_iter_dataset(ds)
        # Test raising if count exceeds max.
        with self.assertRaisesRegex(ValueError, "counting"):
            pad_for_evaluation(ds, per_feed_batch_size=3, max_num_examples=3)

    def test_pad_for_evaluation_checkpoint(self):
        ds = fake_grain_source(examples=[{"x": i} for i in range(7)])
        ds = pad_for_evaluation(ds, per_feed_batch_size=5)
        self._test_checkpointing(iter(ds))

    @parameterized.parameters(
        dict(num_shards=2, shard_index=0),
        dict(num_shards=2, shard_index=1),
    )
    def test_per_feed_batch(self, num_shards: int, shard_index: int):
        input_dispatcher = InputDispatcher.default_config().set(
            global_logical_batch_size=10,
            num_physical_feeds=num_shards,
            physical_feed_index=shard_index,
        )

        # Test against map ds.
        def map_source(dispatch_config):
            map_ds = fake_grain_source(list(range(10)))
            map_ds = map_ds.map(lambda x: x + 1)
            return per_feed_batch(
                map_ds, global_batch_size=10, dispatch_config=dispatch_config, drop_remainder=False
            )

        # Configure dispatch.
        cfg = Input.default_config().set(source=map_source, input_dispatcher=input_dispatcher)
        x = cfg.set(name="test").instantiate(parent=None)
        # Should use the per-feed batch size.
        self.assertNestedEqual([np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])], list(x))

        # Test against iter ds.
        def iter_source(dispatch_config):
            iter_ds = fake_grain_source(list(range(10)))
            iter_ds = maybe_to_iter_dataset(iter_ds)
            iter_ds = iter_ds.map(lambda x: x + 1)
            return per_feed_batch(
                iter_ds, global_batch_size=10, dispatch_config=dispatch_config, drop_remainder=False
            )

        cfg = Input.default_config().set(source=iter_source, input_dispatcher=input_dispatcher)
        x = cfg.set(name="test").instantiate(parent=None)
        # Should use the per-feed batch size.
        self.assertNestedEqual([np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])], list(x))


class _PerProcessFn(Protocol):
    """Processes per-host data."""

    def __call__(self, ds: Dataset, *, dispatch_config: DispatchConfig) -> Dataset:
        ...


class InputTest(parameterized.TestCase):
    """Tests Input module."""

    def _input_config(
        self,
        source_ds: Dataset,
        *,
        per_process: Optional[_PerProcessFn] = None,
    ):
        def ds_fn() -> BuildDatasetFn:
            def source(dispatch_config):
                ds = source_ds
                ds = shard_dataset(ds, dispatch_config=dispatch_config)
                if callable(per_process):
                    ds = per_process(ds, dispatch_config=dispatch_config)
                ds = prefetch_dataset(
                    maybe_to_iter_dataset(ds),
                    multiprocessing_options=grain.MultiprocessingOptions(num_workers=1),
                )
                return ds

            return source

        cfg: Input.Config = Input.default_config().set(source=config_for_function(ds_fn))
        return cfg.set(name="test")

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
        ds = range_dataset(start=0, stop=epoch_len, seed=123).map(lambda x: {"x": x})
        cfg = self._input_config(
            ds,
            per_process=lambda ds, **_: ds.shuffle().repeat(num_epochs).batch(1),
        )
        cfg.input_dispatcher = InputDispatcher.default_config().set(
            global_logical_batch_size=process_count,
            num_physical_feeds=process_count,
            physical_feed_index=process_index,
        )
        grain_input: Input = cfg.instantiate(parent=None)
        examples = list(grain_input)
        num_examples = num_epochs * epoch_len

        expected = list(range(10))

        # If loading on a single process.
        if process_count == 1:
            self.assertEqual(num_examples, len(examples))
            first_epoch = [x["x"] for x in examples[:10]]
            second_epoch = [x["x"] for x in examples[10:]]

            # First epoch. Order will be shuffled.
            self.assertNotEqual(expected, first_epoch)
            self.assertSameElements(expected, first_epoch)
            # Check that second epoch uses a different shuffle.
            self.assertNotEqual(first_epoch, second_epoch)
            self.assertSameElements(expected, second_epoch)

        else:
            shard_options = grain.ShardOptions(shard_index=process_index, shard_count=process_count)

            # Each process gets a split of the source data (our source is not shuffled).
            start, end = even_split(epoch_len, shard_options)
            self.assertEqual((end - start) * num_epochs, len(examples))

            # Only the source examples assigned to this process should appear.
            start, end = even_split(epoch_len, shard_options)
            self.assertSameElements(
                list(range(epoch_len))[slice(process_index, None, process_count)],
                [x["x"] for x in examples],
            )

    @parameterized.parameters(
        # A single shard case.
        dict(range_start=0, range_end=1, period=1),
        # Two-shard cases.
        dict(range_start=0, range_end=1, period=2),
        dict(range_start=1, range_end=2, period=2),
        # Non-even shard cases.
        dict(range_start=0, range_end=3, period=8),
        dict(range_start=3, range_end=8, period=8),
        # 3-shard case.
        dict(range_start=0, range_end=2, period=13),
        dict(range_start=2, range_end=7, period=13),
        dict(range_start=7, range_end=13, period=13),
    )
    def test_shard_dataset_with_proportion(self, range_start, range_end, period):
        epoch_len = 20
        num_epochs = 5
        ds = range_dataset(start=0, stop=epoch_len, seed=123)
        ds = shard_dataset_with_proportion(
            ds, range_start=range_start, range_end=range_end, period=period
        ).repeat(num_epochs=num_epochs)

        executed_results = list(ds)

        expected_result = []
        for index in range(epoch_len * num_epochs):
            index_inside_epoch = index % epoch_len
            if range_start <= index_inside_epoch % period < range_end:
                expected_result.append(index_inside_epoch)

        self.assertEqual(executed_results, expected_result)

    def test_batches(self):
        """Test that we validate per-feed logical batch size."""

        global_batch_size = 4
        process_count, process_index = 2, 0
        dispatcher = InputDispatcher.default_config().set(
            global_logical_batch_size=global_batch_size,
            num_physical_feeds=process_count,
            physical_feed_index=process_index,
        )
        # For global_logical_batch_size=4 and num_physical_feeds=2, each feed should produce logical
        # batch of 2.
        cfg = self._input_config(
            range_dataset(start=0, stop=10, seed=123).shuffle().repeat(num_epochs=1),
            per_process=partial(per_feed_batch, global_batch_size=global_batch_size),
        )
        cfg.input_dispatcher = dispatcher
        grain_input: Input = cfg.instantiate(parent=None)
        batch = next(grain_input.batches(iter(grain_input)))
        self.assertEqual(batch.shape[0], grain_input.input_dispatcher.feed_logical_batch_size)

    @parameterized.parameters(
        # Should produce a per-feed batch of 2, taking every `num_shards` example.
        dict(num_shards=2, shard_index=0, expected=[0, 2]),
        dict(num_shards=2, shard_index=1, expected=[1, 3]),
    )
    def test_dispatch_cpu(self, num_shards: int, shard_index: int, expected: list):
        global_batch_size = num_shards * 2
        dispatcher = InputDispatcher.default_config().set(
            global_logical_batch_size=global_batch_size,
            num_physical_feeds=num_shards,
            physical_feed_index=shard_index,
        )

        # Dispatch requires examples to be dicts.
        ds = range_dataset(start=0, stop=10, seed=123).map(lambda x: {"input_ids": x})
        # Each process produces feed_logical_batch_size.
        cfg = self._input_config(
            ds.repeat(num_epochs=None),
            per_process=partial(per_feed_batch, global_batch_size=global_batch_size),
        )
        cfg.partition_spec = PartitionSpec("x")
        cfg.input_dispatcher = dispatcher

        grain_input: Input = cfg.instantiate(parent=None)
        for batch in grain_input:
            # Each batch produces data corresponding to the current shard.
            self.assertEqual(list(batch.keys()), ["input_ids"])
            self.assertEqual(batch["input_ids"].tolist(), expected)
            break

    def test_element_spec(self):
        ds = range_dataset(start=0, stop=10, seed=123).map(lambda x: {"input_ids": x})
        grain_input: Input = self._input_config(ds).instantiate(parent=None)
        # element_spec() requires Tensor-like leaves.
        with self.assertRaisesRegex(ValueError, "Tensor"):
            grain_input.element_spec()

        ds = range_dataset(start=0, stop=10, seed=123).map(lambda x: {"input_ids": np.array(x)})
        cfg = self._input_config(
            ds.repeat(num_epochs=None),
            per_process=lambda ds, **_: ds.batch(2),
        )
        grain_input: Input = cfg.instantiate(parent=None)
        self.assertEqual(
            # Element spec should reflect the per-process shape.
            {"input_ids": jax.ShapeDtypeStruct(shape=(2,), dtype=np.int64)},
            grain_input.element_spec(),
        )


if __name__ == "__main__":
    absltest.main()

# Copyright © 2023 Apple Inc.

"""Tests tf.data inputs."""

# pylint: disable=no-self-use,too-many-lines
import os
import tempfile
from collections.abc import Iterable, Sequence
from typing import Optional, Union
from unittest import mock

import jax
import numpy as np
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.sharding import Mesh

from axlearn.common import test_utils, utils
from axlearn.common.checkpointer import Checkpointer
from axlearn.common.config import config_for_function
from axlearn.common.input_dispatch import InputDispatcher
from axlearn.common.input_fake import fake_serialized_json_source, fake_source, fake_text_source
from axlearn.common.input_tf_data import (
    BuildDatasetFn,
    DatasetToDatasetFn,
    Input,
    _infer_cardinality,
    _pad_for_evaluation,
    _pad_logical_to_physical,
    add_static_fields,
    batch,
    chain,
    concatenate_datasets,
    default_pad_example_fn,
    disable_shuffle_recursively,
    extract_from_sequence,
    identity,
    pack_to_batch,
    pad_to_batch,
    preserve_element_spec,
    ragged_to_tensor,
    rekey,
    remove_fields,
    sample_from_datasets,
    squeeze_fields,
    tfds_dataset,
    tfds_read_config,
    trim_and_pad_tensor,
    trim_to_batch,
    unpack,
    with_processor,
)
from axlearn.common.utils import as_numpy_array


def build_ds_fn(
    is_training: bool, *, texts: Sequence[str], data_dir: Optional[str] = None
) -> BuildDatasetFn:
    del is_training, data_dir

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            yield from texts

        ds = tf.data.Dataset.from_generator(data_gen, output_types=tf.string)
        # Set the cardinality of the generated dataset.
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(texts)))
        return ds

    return ds_fn


class SamplingTest(parameterized.TestCase):
    @parameterized.parameters(
        {"weights": [1.0, 0.0, 0.0], "expected": ["a", "b", "c", "d", "e"]},
        {"weights": [0.0, 1.0, 0.0], "expected": ["g", "h"]},
        {"weights": [0.0, 0.0, 1.0], "expected": ["w", "x", "y", "z"]},
    )
    def test_sampling_dataset_basic(self, weights, expected):
        sources = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c", "d", "e"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["g", "h"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["w", "x", "y", "z"],
            ),
        ]

        sampling_ds_cfg = config_for_function(sample_from_datasets).set(
            is_training=False,
            sources=sources,
            weights=weights,
        )
        ds_fn = sampling_ds_cfg.instantiate()

        actual = list(ds_fn().take(len(expected)))
        self.assertEqual(expected, actual)

        sources = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c"],
            ),
            config_for_function(build_ds_fn).set(
                texts=[],
            ),
            config_for_function(build_ds_fn).set(
                texts=["y", "z"],
            ),
        ]

        sampling_ds_cfg = config_for_function(sample_from_datasets).set(
            is_training=False,
            sources=sources,
            weights=weights,
        )
        ds_fn = sampling_ds_cfg.instantiate()

        # Dataset with zero cardinality.
        with self.assertRaises(ValueError):
            list(ds_fn().take(1))

    def test_sampling_dataset(self):
        tf.random.set_seed(1)
        sources = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c", "d", "e"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["g", "h"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["w", "x", "y", "z"],
            ),
        ]

        sampling_ds_cfg = config_for_function(sample_from_datasets).set(
            is_training=False,
            sources=sources,
            weights=[1 / 3, 1 / 3, 1 / 3],
            seed=1,
        )
        ds_fn = sampling_ds_cfg.instantiate()

        # Note that dataset ends when a dataset becomes empty.
        expected = ["a", "g", "w", "h", "b", "c", "d"]
        actual = [bytes.decode(x.numpy(), "utf-8") for x in ds_fn().take(len(expected))]
        assert expected == actual

        sources = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c", "d", "e"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["g", "h"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["w", "x", "y", "z"],
            ),
        ]

        sampling_ds_cfg = config_for_function(sample_from_datasets).set(
            is_training=False,
            sources=sources,
            weights=[0.0, 0.5, 0.5],
            seed=1,
        )
        ds_fn = sampling_ds_cfg.instantiate()

        expected = ["g", "w", "x", "h"]
        actual = [bytes.decode(x.numpy(), "utf-8") for x in ds_fn().take(len(expected))]
        assert expected == actual

    def test_autotune_ram_budget(self):
        sources = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c", "d", "e"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["w", "x", "y", "z"],
            ),
        ]

        sampling_ds_cfg = config_for_function(sample_from_datasets).set(
            is_training=False,
            sources=sources,
            weights=[0.5, 0.5],
            seed=1,
            autotune_ram_budget_gb=4,
        )
        ds_fn = sampling_ds_cfg.instantiate()
        dataset = ds_fn()
        # pylint: disable=protected-access
        for component in dataset._data_inputs:
            options = component.options()
            self.assertTrue(options.experimental_warm_start)
            self.assertTrue(options.autotune.enabled)
            self.assertEqual(options.autotune.ram_budget, int(2 * 1024**3))
        # pylint: enable=protected-access

        expected = ["w", "a", "b", "c"]
        actual = [bytes.decode(x.numpy(), "utf-8") for x in ds_fn().take(len(expected))]
        assert expected == actual


class ConcatenateDatasetsTest(parameterized.TestCase):
    def test_raises_when_empty(self):
        with self.assertRaises(ValueError):
            concatenate_datasets(is_training=False, sources=[])

    def test_noop_for_one(self):
        sources = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c", "d", "e"],
            ),
        ]

        ds_fn = concatenate_datasets(is_training=False, sources=sources)

        expected = ["a", "b", "c", "d", "e"]
        actual = [bytes.decode(x.numpy(), "utf-8") for x in ds_fn()]
        assert expected == actual

    def test_concatenates_in_order(self):
        sources = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c", "d", "e"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["g", "h"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["w", "x", "y", "z"],
            ),
        ]

        ds_fn = concatenate_datasets(is_training=False, sources=sources)

        expected = ["a", "b", "c", "d", "e"] + ["g", "h"] + ["w", "x", "y", "z"]
        actual = [bytes.decode(x.numpy(), "utf-8") for x in ds_fn()]
        assert expected == actual


class TfdsTest(parameterized.TestCase):
    @parameterized.parameters(False, True)
    def test_tfds_read_config(self, is_training, read_parallelism=2, decode_parallelism=32):
        read_config = tfds_read_config(
            is_training=is_training,
            read_parallelism=read_parallelism,
            decode_parallelism=decode_parallelism,
        )
        self.assertEqual(read_config.input_context.num_input_pipelines, jax.process_count())
        self.assertEqual(read_config.input_context.input_pipeline_id, jax.process_index())
        if is_training:
            self.assertEqual(read_config.num_parallel_calls_for_decode, decode_parallelism)
            self.assertEqual(read_config.num_parallel_calls_for_interleave_files, read_parallelism)
            self.assertEqual(read_config.interleave_cycle_length, read_parallelism)
        else:
            self.assertEqual(read_config.num_parallel_calls_for_decode, 1)
            self.assertEqual(read_config.num_parallel_calls_for_interleave_files, 1)
            self.assertEqual(read_config.interleave_cycle_length, 1)

    @parameterized.parameters((1, 0), (16, 4))
    def test_tfds_read_config_with_custom_sharding(self, num_shards, shard_index):
        read_config = tfds_read_config(
            is_training=True,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        self.assertEqual(read_config.input_context.num_input_pipelines, num_shards)
        self.assertEqual(read_config.input_context.input_pipeline_id, shard_index)

    @parameterized.parameters(
        ("inputs_pretokenized"),
        ("prefix_ids"),
    )
    def test_tfds_decoders_ci(self, field_name: str):
        def tfds_custom_decoder() -> dict[str, tfds.decode.Decoder]:
            @tfds.decode.make_decoder()
            def custom_fn(field_value, _):
                return field_value

            # pylint: disable=no-value-for-parameter
            return {field_name: custom_fn()}

        decoders = config_for_function(tfds_custom_decoder)
        custom_decoders = decoders.instantiate()
        assert isinstance(
            custom_decoders[field_name], tfds.decode.base.DecoderFn
        ), "The decoder fn is not of type tfds.decode.base.DecoderFn"


def _text_ds(texts: list[str], *, repeat=1) -> tf.data.Dataset:
    dataset = {
        "text": [],
        "index": [],
        "is_valid": [],
    }

    for index, text in enumerate(texts):
        dataset["index"].append(tf.constant(index, dtype=tf.int32))
        dataset["text"].append(tf.constant(text, dtype=tf.string))
        dataset["is_valid"].append(tf.constant(True, dtype=tf.bool))

    return tf.data.Dataset.from_tensor_slices(dataset).repeat(repeat)


def _tokens_ds(tokens: list[list[int]], *, repeat=1) -> tf.data.Dataset:
    dataset = {
        "tokens": [],
        "index": [],
        "is_valid": [],
    }

    for index, tok in enumerate(tokens):
        dataset["index"].append(tf.constant(index, dtype=tf.int32))
        dataset["tokens"].append(tf.constant(tok, dtype=tf.int32))
        dataset["is_valid"].append(tf.constant(True, dtype=tf.bool))

    return tf.data.Dataset.from_tensor_slices(dataset).repeat(repeat)


class PadTest(test_utils.TestCase):
    @parameterized.parameters(None, 1, 5)
    def test_infer_cardinality(self, num_repeats: Optional[int]):
        text_examples = ["a", "b", "c"]
        ds = _text_ds(text_examples).repeat(num_repeats)
        cardinality = _infer_cardinality(ds)
        if num_repeats is None:
            self.assertEqual(cardinality, tf.data.INFINITE_CARDINALITY)
        else:
            self.assertEqual(cardinality, len(text_examples) * num_repeats)

    def test_pad_for_evaluation(self):
        text_examples = ["a", "b", "c"]
        ds = _pad_for_evaluation(
            _text_ds(["a", "b", "c"]),
            per_feed_batch_size=2,
            pad_example_fn=default_pad_example_fn,
        )
        example_ix = 0
        for example in ds:
            text = example["text"].numpy().decode()
            index = example["index"].numpy()
            is_valid = example["is_valid"].numpy()
            if example_ix < len(text_examples):
                self.assertEqual(text, text_examples[example_ix])
                self.assertEqual(index, example_ix)
                self.assertEqual(is_valid, True)
            else:
                self.assertEqual(text, "")
                self.assertEqual(index, 0)
                self.assertEqual(is_valid, False)
            example_ix += 1
        self.assertEqual(example_ix, 4)

    @parameterized.product(
        num_logical_feeds=(1, 2),
        logical_batch_size=(1, 2, 4),
        logical_feed_index=(0, 1),
        physical_batch_size=(2, 4, 8),
    )
    def test_pad_logical_to_physical_for_logical_feed(
        self,
        num_logical_feeds: int,
        logical_feed_index: int,
        logical_batch_size: int,
        physical_batch_size: int,
    ):
        # Skip unsupported combinations.
        if logical_batch_size % num_logical_feeds != 0:
            return
        if logical_batch_size > physical_batch_size:
            return
        if logical_feed_index >= num_logical_feeds:
            return
        text_examples = ["a", "b", "c", "d"]
        per_feed_physcal_batch_size = physical_batch_size // num_logical_feeds
        with mock.patch("jax.process_count", return_value=num_logical_feeds):
            ds = _pad_logical_to_physical(
                _text_ds(text_examples),
                global_batch_size=physical_batch_size,
                global_logical_batch_size=logical_batch_size,
                num_logical_feeds=num_logical_feeds,
                logical_feed_index=logical_feed_index,
                pad_example_fn=default_pad_example_fn,
            ).batch(per_feed_physcal_batch_size)
        per_feed_logical_batch_size = logical_batch_size // num_logical_feeds
        num_batches = 0
        input_iter = iter(ds)
        for input_batch in input_iter:
            text = [el.decode() for el in input_batch["text"].numpy()]
            start_ix = per_feed_logical_batch_size * num_batches
            end_ix = start_ix + per_feed_logical_batch_size
            # Logical part of the batch.
            self.assertSequenceEqual(
                text[:per_feed_logical_batch_size],
                text_examples[start_ix:end_ix],
            )
            index = input_batch["index"].numpy()
            self.assertNestedEqual(
                index[:per_feed_logical_batch_size],
                np.arange(start_ix, end_ix, dtype=np.int32),
            )
            is_valid = input_batch["is_valid"].numpy()
            self.assertNestedEqual(
                is_valid[:per_feed_logical_batch_size],
                np.array([True] * per_feed_logical_batch_size),
            )
            # Padded part of the batch.
            per_feed_padded_batch_size = per_feed_physcal_batch_size - per_feed_logical_batch_size
            if per_feed_padded_batch_size > 0:
                self.assertSequenceEqual(
                    text[per_feed_logical_batch_size:],
                    [""] * per_feed_padded_batch_size,
                )
                self.assertNestedEqual(
                    index[per_feed_logical_batch_size:],
                    np.array([0] * per_feed_padded_batch_size, dtype=np.int32),
                )
                self.assertNestedEqual(
                    is_valid[per_feed_logical_batch_size:],
                    np.array([False] * per_feed_padded_batch_size),
                )
            # Physical to logical dispatch tensor.
            dispatch = input_batch[utils.PHYSICAL_TO_LOGICAL_DISPATCH_KEY].numpy()
            self.assertEqual(dispatch.shape, (per_feed_physcal_batch_size, logical_batch_size))
            expected_dispatch = np.zeros(
                (per_feed_physcal_batch_size, logical_batch_size), dtype=bool
            )
            logical_dispatch_start = logical_feed_index * per_feed_logical_batch_size
            expected_dispatch[
                np.arange(per_feed_logical_batch_size),
                np.arange(
                    logical_dispatch_start, logical_dispatch_start + per_feed_logical_batch_size
                ),
            ] = True
            self.assertNestedEqual(dispatch, expected_dispatch)
            num_batches += 1
            if num_batches == 1:
                self._check_iterator_saveable(input_iter)
        self.assertEqual(num_batches, len(text_examples) // per_feed_logical_batch_size)

    @parameterized.product(
        num_physical_feeds=(2, 4),
        physical_batch_size=(4, 8, 16),
    )
    def test_pad_logical_to_physical_for_physical_feed(
        self,
        num_physical_feeds: int,
        physical_batch_size: int,
    ):
        # Test that non-logical feed returns appropriately padded data.
        text_examples = ["a", "b", "c", "d"]
        per_feed_physcal_batch_size = physical_batch_size // num_physical_feeds
        logical_feed_logical_batch_size = 2
        with mock.patch("jax.process_count", return_value=num_physical_feeds):
            ds = _pad_logical_to_physical(
                _text_ds(text_examples),
                global_batch_size=physical_batch_size,
                global_logical_batch_size=logical_feed_logical_batch_size,
                num_logical_feeds=1,
                logical_feed_index=None,
                pad_example_fn=default_pad_example_fn,
            ).batch(per_feed_physcal_batch_size)
        input_iter = iter(ds)
        num_batches = 0
        for input_batch in input_iter:
            self.assertSequenceEqual(
                [el.decode() for el in input_batch["text"].numpy()],
                [""] * per_feed_physcal_batch_size,
            )
            self.assertNestedEqual(
                input_batch["index"].numpy(),
                np.array([0] * per_feed_physcal_batch_size, dtype=np.int32),
            )
            self.assertNestedEqual(
                input_batch["is_valid"].numpy(),
                np.array([False] * per_feed_physcal_batch_size),
            )
            self.assertNestedEqual(
                input_batch[utils.PHYSICAL_TO_LOGICAL_DISPATCH_KEY],
                np.zeros(
                    (per_feed_physcal_batch_size, logical_feed_logical_batch_size), dtype=bool
                ),
            )
            num_batches += 1
            if num_batches == 1:
                self._check_iterator_saveable(input_iter)
        self.assertEqual(num_batches, len(text_examples) // logical_feed_logical_batch_size)

    def _check_iterator_saveable(self, iterator: Iterable):
        # Check that we can save the data iterator.
        with tempfile.TemporaryDirectory() as td:
            save_dir = os.path.join(td, "ckpt")
            step = 100
            ckptr = (
                Checkpointer.default_config()
                .set(name="ckptr", dir=save_dir)
                .instantiate(parent=None)
            )
            with Mesh(jax.devices(), "data"):
                ckptr.save(step=step, state={"iterator": iterator}, evaler_summaries=None)
                ckptr.wait_until_finished()
            self.assertTrue(os.path.exists(os.path.join(save_dir, f"step_{step:08d}", "index")))

    @parameterized.product(
        num_physical_feeds=(2, 4),
        num_logical_feeds=(1, 2),
        physical_batch_size=(4, 8, 16),
    )
    def test_input_dispatcher(
        self,
        num_physical_feeds: int,
        num_logical_feeds: int,
        physical_batch_size: int,
    ):
        """Checks that Input with input_dispatcher generates the same physical batches and global
        logical batches as a manual feed generated through `_pad_logical_to_physical` and
        `dispatch_input_batch`.
        """
        text_examples = [[1, 2], [3, 4], [5, 6], [7, 8]]
        feed_logical_batch_size = 2
        feed_physical_batch_size = physical_batch_size // num_physical_feeds
        if feed_physical_batch_size < feed_logical_batch_size:
            return  # skip invalid cases
        num_logical_batches_per_feed = len(text_examples) // feed_logical_batch_size
        logical_feed_indices = list(
            range(num_physical_feeds - num_logical_feeds, num_physical_feeds)
        )
        global_logical_batch_size = feed_logical_batch_size * len(logical_feed_indices)

        # Mappings from physical_feed_index to physical feed batches.
        manual_feeds = {}
        input_feeds = {}
        for physical_feed_index in range(num_physical_feeds):
            if physical_feed_index in logical_feed_indices:
                logical_feed_index = logical_feed_indices.index(physical_feed_index)
            else:
                logical_feed_index = None
            physical_feed_ds = _tokens_ds(text_examples)
            if global_logical_batch_size != physical_batch_size:
                with mock.patch("jax.process_count", return_value=num_physical_feeds):
                    physical_feed_ds = _pad_logical_to_physical(
                        physical_feed_ds,
                        global_batch_size=physical_batch_size,
                        global_logical_batch_size=global_logical_batch_size,
                        num_logical_feeds=len(logical_feed_indices),
                        logical_feed_index=logical_feed_index,
                        pad_example_fn=default_pad_example_fn,
                    )
            physical_feed_ds = physical_feed_ds.batch(feed_physical_batch_size)
            manual_feed_batches = list(iter(physical_feed_ds))
            self.assertLen(manual_feed_batches, num_logical_batches_per_feed)

            def source_fn() -> BuildDatasetFn:
                def fn() -> tf.data.Dataset:
                    return _tokens_ds(text_examples)

                return fn

            input_generator = (
                Input.default_config()
                .set(
                    name="input",
                    is_training=True,
                    source=config_for_function(source_fn),
                    processor=config_for_function(identity),
                    batcher=config_for_function(batch).set(
                        global_batch_size=feed_logical_batch_size,
                        pad_example_fn=default_pad_example_fn,
                        repeat=1,
                    ),
                    input_dispatcher=InputDispatcher.default_config().set(
                        global_logical_batch_size=global_logical_batch_size,
                        global_physical_batch_size=physical_batch_size,
                        logical_feed_indices=logical_feed_indices,
                        num_physical_feeds=num_physical_feeds,
                        physical_feed_index=physical_feed_index,
                    ),
                )
                .instantiate(parent=None)
            )

            input_it = iter(input_generator.dataset())
            self.assertIsInstance(input_it, tf.data.Iterator)
            self._check_iterator_saveable(input_it)

            input_batches = list(input_generator.batches(input_it))
            print(f"input_batch={input_batches[0]}")
            self.assertLen(input_batches, num_logical_batches_per_feed)

            for manual_batch, input_batch in zip(manual_feed_batches, input_batches):
                print(f"manual_batch={manual_batch}")
                print(f"input_batch={input_batch}")
                self.assertNestedEqual(manual_batch, input_batch)

            manual_feeds[physical_feed_index] = manual_feed_batches
            input_feeds[physical_feed_index] = input_batches

        for step in range(num_logical_batches_per_feed):
            # For each step, assemble global logical batches from `manual_feeds` and
            # `input_feeds` and check that they are equal.
            manual_global_batch = utils.dispatch_input_batch(
                jax.tree.map(
                    lambda *xs: jnp.concatenate(as_numpy_array(xs), axis=0),
                    *[batches[step] for batches in manual_feeds.values()],
                )
            )
            input_global_batch = input_generator.input_dispatcher.physical_to_logical_batch(
                jax.tree.map(
                    lambda *xs: jnp.concatenate(as_numpy_array(xs), axis=0),
                    *[batches[step] for batches in input_feeds.values()],
                )
            )
            self.assertNestedEqual(manual_global_batch, input_global_batch)


class BatchTest(test_utils.TestCase):
    @parameterized.parameters(False, True)
    def test_eval_pad(self, is_training):
        ds = _text_ds(["a", "b", "c"])
        ds = batch(
            global_batch_size=2, is_training=is_training, pad_example_fn=default_pad_example_fn
        )(ds)
        batch_index = 0
        for input_batch in ds:
            if is_training or batch_index == 0:
                self.assertSequenceEqual(input_batch["text"].numpy().tolist(), [b"a", b"b"])
                self.assertSequenceEqual(input_batch["index"].numpy().tolist(), [0, 1])
                self.assertSequenceEqual(input_batch["is_valid"].numpy().tolist(), [True, True])
            else:
                # The eval dataset will be padded by empty examples.
                self.assertSequenceEqual(input_batch["text"].numpy().tolist(), [b"c", b""])
                self.assertSequenceEqual(input_batch["index"].numpy().tolist(), [2, 0])
                self.assertSequenceEqual(input_batch["is_valid"].numpy().tolist(), [True, False])
            batch_index += 1
            if batch_index >= 10:
                break

    @parameterized.parameters(False, True)
    def test_batch_for_logical_feed_index(self, is_training):
        text_examples = ["a", "b", "c"]
        ds = _text_ds(text_examples)
        ds = batch(
            global_batch_size=2,
            is_training=is_training,
            pad_example_fn=default_pad_example_fn,
            global_logical_batch_size=1,
            logical_feed_indices=[0],
        )(ds)
        batch_index = 0
        for input_batch in ds:
            text = input_batch["text"].numpy().tolist()
            indices = input_batch["index"].numpy().tolist()
            is_valid = input_batch["is_valid"].numpy().tolist()
            dispatch = input_batch[utils.PHYSICAL_TO_LOGICAL_DISPATCH_KEY].numpy().tolist()
            if is_training or batch_index < len(text_examples):
                expected_index = batch_index % len(text_examples)
                self.assertSequenceEqual(
                    text,
                    [text_examples[expected_index].encode(), b""],
                )
                self.assertSequenceEqual(indices, [expected_index, 0])
                self.assertSequenceEqual(is_valid, [True, False])
                self.assertNestedEqual(dispatch, [[1], [0]])
            else:
                self.assertSequenceEqual(
                    text,
                    [b"", b""],
                )
                self.assertSequenceEqual(indices, [0, 0])
                self.assertSequenceEqual(is_valid, [False, False])
                self.assertNestedEqual(dispatch, [[0], [0]])
            batch_index += 1
            if batch_index >= 10:
                break
        if is_training:
            self.assertEqual(batch_index, 10)
        else:
            self.assertEqual(batch_index, len(text_examples))

    @parameterized.parameters(False, True)
    def test_batch_for_physical_feed_index(self, is_training):
        text_examples = ["a", "b", "c"]
        ds = _text_ds(text_examples)
        with mock.patch("jax.process_index", return_value=1):
            ds = batch(
                global_batch_size=2,
                is_training=is_training,
                pad_example_fn=default_pad_example_fn,
                global_logical_batch_size=1,
                # jax.process_index will not be one of the logical feed indices.
                logical_feed_indices=[0],
            )(ds)
        batch_index = 0
        for input_batch in ds:
            self.assertSequenceEqual(
                input_batch["text"].numpy().tolist(),
                [b"", b""],
            )
            self.assertSequenceEqual(input_batch["index"].numpy().tolist(), [0, 0])
            self.assertSequenceEqual(input_batch["is_valid"].numpy().tolist(), [False, False])
            self.assertNestedEqual(
                input_batch[utils.PHYSICAL_TO_LOGICAL_DISPATCH_KEY].numpy().tolist(), [[0], [0]]
            )
            batch_index += 1
            if batch_index >= 10:
                break
        if is_training:
            self.assertEqual(batch_index, 10)
        else:
            self.assertEqual(batch_index, len(text_examples))

    @parameterized.product(
        is_training=(False, True),
        prefetch_buffer_size=(32, None),
    )
    def test_prefetch_buffer_size(self, is_training, prefetch_buffer_size):
        ds = _text_ds(["a", "b", "c"])
        _ = batch(
            global_batch_size=2,
            is_training=is_training,
            pad_example_fn=default_pad_example_fn,
            prefetch_buffer_size=prefetch_buffer_size,
        )(ds)

    @parameterized.product(
        is_training=(False, True),
        post_batch_processor=(None, lambda x: x),
    )
    def test_post_batch_map_fn(self, is_training, post_batch_processor):
        ds = _text_ds(["a", "b", "c"])
        _ = batch(
            global_batch_size=2,
            is_training=is_training,
            pad_example_fn=default_pad_example_fn,
            post_batch_processor=post_batch_processor,
        )(ds)

    @parameterized.product(
        is_training=(False, True),
        repeat=(None, 1, 2),
    )
    def test_repeat(self, *, is_training, repeat):
        ds = _text_ds(["a", "b", "c"])
        ds = batch(
            global_batch_size=2,
            is_training=is_training,
            pad_example_fn=default_pad_example_fn,
            repeat=repeat,
        )(ds)
        batch_index = 0
        for input_batch in ds:
            if is_training or batch_index % 2 == 0:
                self.assertSequenceEqual(input_batch["text"].numpy().tolist(), [b"a", b"b"])
                self.assertSequenceEqual(input_batch["index"].numpy().tolist(), [0, 1])
            else:
                # The eval dataset will be padded by empty examples.
                self.assertSequenceEqual(input_batch["text"].numpy().tolist(), [b"c", b""])
                self.assertSequenceEqual(input_batch["index"].numpy().tolist(), [2, 0])
            batch_index += 1
            if batch_index >= 10:
                break
        if repeat is None:
            # Repeat indefinitely if is_training, otherwise do not repeat
            # (hence 2 batches after padding).
            self.assertEqual(10 if is_training else 2, batch_index)
        else:
            # If is_training, we discard remaining examples, hence one batch per epoch.
            # Otherwise we have two batches per epoch.
            self.assertEqual(repeat if is_training else 2 * repeat, batch_index)


class UnpackTest(test_utils.TestCase):
    # pylint: disable=no-self-use
    def _ds_fn(self) -> tf.data.Dataset:
        def nested_data_gen():
            for value in ["hello", "world"]:
                yield {"key1": "dummy", "key2": {"key3": {"key4": {"key5": value}}}}

        return tf.data.Dataset.from_generator(
            nested_data_gen,
            output_signature={
                "key1": tf.TensorSpec(shape=(), dtype=tf.string),
                "key2": {"key3": {"key4": {"key5": tf.TensorSpec(shape=(), dtype=tf.string)}}},
            },
        )

    def test_unpack_flattens_nested_path(self):
        ds = self._ds_fn()
        ds = unpack({"new_key2": ("key2", "key3", "key4", "key5"), "new_key1": ("key1",)})(ds)
        for el in ds:
            self.assertEqual(el["key1"], el["new_key1"])
            self.assertEqual(el["key2"]["key3"]["key4"]["key5"], el["new_key2"])


class RekeyTest(test_utils.TestCase):
    DEFAULT_VALUES = ["hello", "world"]

    def _ds_fn(self) -> tf.data.Dataset:
        def data_gen():
            for value in self.DEFAULT_VALUES:
                yield {
                    "key1": value,
                    "key2": value,
                    "key3/key4": value,
                    "key5": {"key6": value},
                }

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                "key1": tf.TensorSpec(shape=(), dtype=tf.string),
                "key2": tf.TensorSpec(shape=(), dtype=tf.string),
                "key3/key4": tf.TensorSpec(shape=(), dtype=tf.string),
                "key5": {"key6": tf.TensorSpec(shape=(), dtype=tf.string)},
            },
        )

    def test_rekey_does_nothing_empty_keymap(self):
        ds = self._ds_fn()
        ds = rekey({})(ds)
        for ix, el in enumerate(ds):
            self.assertEqual(el["key1"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["key2"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["key3/key4"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["key5"]["key6"], self.DEFAULT_VALUES[ix])

    @parameterized.parameters(
        dict(
            key_map={"new_key1": "key1", "new_key2": "key2", "new_key3": "key3"},
            default_value="no",
            expected=[
                {"new_key1": "hello", "new_key2": "hello", "new_key3": "no"},
                {"new_key1": "world", "new_key2": "world", "new_key3": "no"},
            ],
        ),
        # Test rekey paths.
        dict(
            key_map={
                # Maps the literal "key3/key4" to "key3": {"key4": ...}.
                "key3.key4": "key3/key4",
                # Maps "key5": {"key6": ...} to the literal "key5/key6".
                "key5/key6": "key5.key6",
                # Injects a new "key7": {"key8": ...}.
                "key7.key8": "unknown",
            },
            separator=".",
            default_value="no",
            expected=[
                {"key3": {"key4": "hello"}, "key5/key6": "hello", "key7": {"key8": "no"}},
                {"key3": {"key4": "world"}, "key5/key6": "world", "key7": {"key8": "no"}},
            ],
        ),
    )
    def test_rekey_maps_new_keys(self, expected: Sequence[dict], **kwargs):
        ds = self._ds_fn()
        ds = rekey(**kwargs)(ds)
        actual = list(ds)
        expected = tf.nest.map_structure(tf.constant, expected)
        self.assertNestedEqual(expected, actual)

    def test_rekey_changes_element_spec(self):
        ds = self._ds_fn()
        ds = rekey(
            {"new_key1": "key1", "new_key2": "key2", "new_key3": "key3"}, default_value="no"
        )(ds)
        expected = dict(
            new_key1=tf.TensorSpec(shape=(), dtype=tf.string),
            new_key2=tf.TensorSpec(shape=(), dtype=tf.string),
            new_key3=tf.TensorSpec(shape=(), dtype=tf.string),
        )
        self.assertNestedEqual(ds.element_spec, expected)

    def test_rekey_maps_falsey_reference_keys_to_default(self):
        ds = self._ds_fn()
        ds = rekey({"new_key1": "key1", "new_key2": None}, default_value="no")(ds)
        for ix, el in enumerate(ds):
            self.assertEqual(set(el.keys()), {"new_key1", "new_key2"})
            self.assertEqual(el["new_key1"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["new_key2"], "no")

    def test_rekey_maps_original_inputs_if_asked(self):
        ds = self._ds_fn()
        ds = rekey(
            {"new_key1": "key1", "new_key2": None}, default_value="no", retain_original_inputs=True
        )(ds)
        for ix, el in enumerate(ds):
            self.assertEqual(
                set(el.keys()), {"key1", "key2", "new_key1", "new_key2", "key3/key4", "key5"}
            )
            self.assertEqual(el["key1"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["key2"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["key3/key4"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["key5"]["key6"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["new_key1"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["new_key2"], "no")

    def test_rekey_does_not_map_missing_reference_keys_with_none_default(self):
        ds = self._ds_fn()
        ds = rekey(
            {"new_key1": "key1", "new_key2": "key2", "new_key3": "key3"}, default_value=None
        )(ds)
        for ix, el in enumerate(ds):
            self.assertEqual(set(el.keys()), {"new_key1", "new_key2"})
            self.assertEqual(el["new_key1"], self.DEFAULT_VALUES[ix])
            self.assertEqual(el["new_key2"], self.DEFAULT_VALUES[ix])

    def test_rekey_does_not_map_falsey_reference_keys_with_none_default(self):
        ds = self._ds_fn()
        ds = rekey({"new_key1": "key1", "new_key2": None}, default_value=None)(ds)
        for ix, el in enumerate(ds):
            self.assertEqual(set(el.keys()), {"new_key1"})
            self.assertEqual(el["new_key1"], self.DEFAULT_VALUES[ix])


class ProcessorsTest(parameterized.TestCase, tf.test.TestCase):
    def test_processor_for_sample_from_dataset(self):
        def process_fn(is_training: bool, *, add_token: str) -> DatasetToDatasetFn:
            del is_training

            @seqio.map_over_dataset
            def process_example_fn(example: str) -> str:
                example += add_token
                return example

            return process_example_fn

        tf.random.set_seed(1)
        source_cfgs = [
            config_for_function(build_ds_fn).set(
                texts=["a", "b", "c", "d", "e"],
            ),
            config_for_function(build_ds_fn).set(
                texts=["f", "g", "h"],
            ),
        ]
        processor_cfgs = [
            config_for_function(process_fn).set(add_token="_ds1"),
            config_for_function(process_fn).set(add_token="_ds2"),
        ]
        sources = [
            config_for_function(with_processor).set(
                source=ds_cfg,
                processor=processor_cfg,
                is_training=False,
            )
            for ds_cfg, processor_cfg in zip(source_cfgs, processor_cfgs)
        ]

        sampling_ds_cfg = config_for_function(sample_from_datasets).set(
            is_training=False,
            sources=sources,
            weights=[0.5, 0.5],
        )
        ds_fn = sampling_ds_cfg.instantiate()
        actual = list(ds_fn().take(2))

        expected = ["a_ds1", "f_ds2"]
        self.assertEqual(len(list(expected)), len(actual))
        for e, a in zip(expected, actual):
            self.assertAllEqual(e, a)

    def test_squeeze_fields(self):
        examples = [
            {
                "a": tf.constant([[1], [0]]),
                "b": tf.constant([[1], [1]]),
                "c": tf.constant([[[1, 2, 3]], [[4, 5, 6]]]),
                "d": tf.constant([[[[3, 2, 1]], [[6, 5, 4]]]]),
            }
        ]

        def gen():
            yield from examples

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "a": tf.TensorSpec(shape=(2, 1), dtype=tf.int32),
                "b": tf.TensorSpec(shape=(2, 1), dtype=tf.int32),
                "c": tf.TensorSpec(shape=(2, 1, 3), dtype=tf.int32),
                "d": tf.TensorSpec(shape=(1, 2, 1, 3), dtype=tf.int32),
            },
        )

        processor = (
            config_for_function(squeeze_fields).set(axis=dict(a=1, c=None, d=[0, 2])).instantiate()
        )
        ds = processor(ds)
        ds = list(ds.as_numpy_iterator())
        self.assertEqual(
            {
                "a": [1, 0],
                "b": [[1], [1]],
                "c": [[1, 2, 3], [4, 5, 6]],
                "d": [[3, 2, 1], [6, 5, 4]],
            },
            ds[0],
        )

    def test_remove_fields(self):
        examples = [
            {
                "a": tf.constant([1]),
                "b": tf.constant([[1], [1]]),
                "c": tf.constant([[2], [2]]),
            }
        ]

        def gen():
            yield from examples

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "a": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "b": tf.TensorSpec(shape=(2, 1), dtype=tf.int32),
                "c": tf.TensorSpec(shape=(2, 1), dtype=tf.int32),
            },
        )

        # Remove key does not exist in data.
        processor = config_for_function(remove_fields).set(fields=["d"]).instantiate()
        ds = processor(ds)
        ds_list = list(ds.as_numpy_iterator())
        self.assertEqual(
            {
                "a": [1],
                "b": [[1], [1]],
                "c": [[2], [2]],
            },
            ds_list[0],
        )
        # Remove a key in data.
        processor = config_for_function(remove_fields).set(fields=["c"]).instantiate()
        ds = processor(ds)
        ds_list = list(ds.as_numpy_iterator())
        self.assertEqual(
            {
                "a": [1],
                "b": [[1], [1]],
            },
            ds_list[0],
        )


class ExtractFromSequenceTest(parameterized.TestCase):
    COLOR_OPTIONS = ["blue", "green", "yellow", "black"]

    def _data_gen(self):
        def fn():
            yield dict(text="Which color would you like?", options=self.COLOR_OPTIONS)

        return tf.data.Dataset.from_generator(
            fn,
            output_signature={
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
                "options": tf.TensorSpec(shape=(len(self.COLOR_OPTIONS),), dtype=tf.string),
            },
        )

    @parameterized.parameters(0, 1, 2, 3)
    def test_extract_single_index(self, idx: int = 0):
        out_key = "selected_option"
        ds = extract_from_sequence(in_key="options", out_key=out_key, idx=idx)(self._data_gen())
        el = next(iter(ds))
        self.assertEqual(el[out_key], self.COLOR_OPTIONS[idx])

    @parameterized.parameters(slice(0, 1), slice(0, 2), slice(1, 2))
    def test_extract_slice(self, slc: slice):
        out_key = "selected_options"
        ds = extract_from_sequence(in_key="options", out_key=out_key, idx=slc)(self._data_gen())
        el = next(iter(ds))
        self.assertSequenceEqual(
            [v.decode("utf8") for v in el[out_key].numpy()], self.COLOR_OPTIONS[slc]
        )


class PreserveElementSpecTest(parameterized.TestCase):
    def test_preserve_element_spec(self):
        @seqio.map_over_dataset
        def mapper(example):
            example["text"] = tf.py_function(func=lambda x: x, inp=example, Tout=tf.string)
            example["label"] = example["text"]
            return example

        orig_ds = _text_ds(["test"])

        # The mapper by default should produce an unknown shape.
        ds = mapper(orig_ds)
        self.assertEqual(ds.element_spec["text"].shape, tf.TensorShape(None))
        self.assertEqual(ds.element_spec["label"].shape, tf.TensorShape(None))

        # Mapping with preserve_element_spec should retain the spec.
        mapper = preserve_element_spec(mapper, key_map={"label": "text"})
        ds = mapper(orig_ds)
        self.assertEqual(ds.element_spec["text"].shape, tf.TensorShape(()))
        self.assertEqual(ds.element_spec["label"].shape, tf.TensorShape(()))


class WithProcessorTest(parameterized.TestCase):
    def test_with_processor(self):
        # Test that we can instantiate properly.
        ds_fn = with_processor(
            config_for_function(build_ds_fn).set(texts=["test"]),
            processor=config_for_function(identity),
            is_training=False,
        )
        next(iter(ds_fn()))

    def test_with_processor_optional_fields(self):
        # Test that we can instantiate properly.
        ds_fn = with_processor(
            # We deliberately use a fake source without is_training/data_dir params.
            config_for_function(fake_serialized_json_source).set(examples=[{"a": 1}, {"b": 2}]),
            processor=config_for_function(identity),
            is_training=False,
        )
        next(iter(ds_fn()))


class AddStaticFieldsTest(parameterized.TestCase):
    def test_add_static_fields(self):
        ds = fake_text_source(is_training=False)()
        processor = add_static_fields(key_map={"custom_key": "custom_value"})
        actual = processor(ds)

        expected = [
            {"text": tf.constant("eval text 0"), "custom_key": tf.constant("custom_value")},
            {"text": tf.constant("eval text 1"), "custom_key": tf.constant("custom_value")},
        ]
        self.assertEqual(
            actual.element_spec,
            {
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
                "custom_key": tf.TensorSpec(shape=(), dtype=tf.string),
            },
        )
        self.assertSequenceEqual(expected, list(actual.as_numpy_iterator()))


class PadToBatchTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        dict(
            examples=[
                {"a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), "b": tf.constant([1, 2])},
                {"a": tf.constant([[1, 2, 0]]), "b": tf.constant([3])},
            ],
            expected=[
                {
                    "a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6], [0, 0, 0], [0, 0, 0]]),
                    "b": tf.constant([1, 2, 0, 0, 0]),
                },
                {
                    "a": tf.constant([[1, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    "b": tf.constant([3, 0, 0, 0, 0]),
                },
            ],
        ),
    )
    def test_pad_to_batch(self, examples: dict[str, tf.Tensor], expected: dict[str, tf.Tensor]):
        processor = pad_to_batch(batch_size=5)
        source = fake_source(
            is_training=False,
            examples=examples,
            spec={
                "a": tf.TensorSpec(shape=[None, 3], dtype=tf.int32),
                "b": tf.TensorSpec(shape=[None], dtype=tf.int32),
            },
        )
        actual = list(processor(source()))
        tf.nest.map_structure(self.assertAllEqual, expected, actual)


class PackTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        dict(
            examples=[
                {"a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), "b": tf.constant([1, 2])},
                {"a": tf.constant([[1, 2, 0]]), "b": tf.constant([3])},
                {"a": tf.constant([[3, 0, 0]]), "b": tf.constant([4])},
                {"a": tf.constant([[1, 2, 3], [4, 0, 0]]), "b": tf.constant([5, 6, 7, 8])},
            ],
            expected=[
                {
                    "a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6], [1, 2, 0], [3, 0, 0]]),
                    "b": tf.constant([1, 2, 3, 4, 0]),
                },
                {
                    "a": tf.constant([[1, 2, 3], [4, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    "b": tf.constant([5, 6, 7, 8, 0]),
                },
            ],
        ),
        dict(
            examples=[
                {"a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), "b": tf.constant([1, 2])},
                {"a": tf.constant([[1, 2, 0]]), "b": tf.constant([3, 4, 5, 6, 7])},
            ],
            expected=[
                {
                    "a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6], [0, 0, 0], [0, 0, 0]]),
                    "b": tf.constant([1, 2, 0, 0, 0]),
                },
                {
                    "a": tf.constant([[1, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    "b": tf.constant([3, 4, 5, 6, 7]),
                },
            ],
        ),
        # Test a case where each element is multi dimensional.
        dict(
            examples=[
                {"a": tf.ones([2, 2, 2], dtype=tf.int32)},
                {"a": tf.ones([3, 2, 2], dtype=tf.int32) * 2},
                {"a": tf.ones([3, 2, 2], dtype=tf.int32) * 3},
            ],
            expected=[
                {
                    "a": tf.concat(
                        [
                            tf.ones([2, 2, 2], dtype=tf.int32),
                            tf.ones([3, 2, 2], dtype=tf.int32) * 2,
                        ],
                        0,
                    ),
                },
                {
                    "a": tf.concat(
                        [
                            tf.ones([3, 2, 2], dtype=tf.int32) * 3,
                            tf.zeros([2, 2, 2], dtype=tf.int32),
                        ],
                        0,
                    ),
                },
            ],
            spec={"a": tf.TensorSpec(shape=[None, 2, 2], dtype=tf.int32)},
        ),
        # Test a case where an input element already exceeds batch_size.
        # We should raise in this case.
        dict(
            examples=[{"a": tf.ones([6, 2], dtype=tf.int32), "b": tf.constant([1])}],
            expected=tf.errors.InvalidArgumentError,
        ),
    )
    def test_pack_to_batch(
        self,
        examples: Sequence[dict[str, tf.Tensor]],
        expected: Union[type[Exception], Sequence[dict[str, tf.Tensor]]],
        spec: Optional[dict] = None,
    ):
        processor = pack_to_batch(batch_size=5)
        source = fake_source(
            is_training=False,
            examples=examples,
            spec=spec
            or {
                "a": tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                "b": tf.TensorSpec(shape=[None], dtype=tf.int32),
            },
        )
        if isinstance(expected, list):
            actual = list(processor(source()))
            tf.nest.map_structure(self.assertAllEqual, expected, actual)
        else:
            with self.assertRaises(expected):
                list(processor(source()))

    @parameterized.parameters(
        dict(
            examples=[
                {"a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6]]), "b": tf.constant([1, 2])},
                {"a": tf.constant([[1, 2, 0]]), "b": tf.constant([3])},
                {"a": tf.constant([[3, 0, 0]]), "b": tf.constant([4])},
                {"a": tf.constant([[1, 2, 3], [4, 0, 0]]), "b": tf.constant([5, 6, 7, 8])},
            ],
            expected=[
                {
                    "a": tf.constant([[1, 0, 0], [2, 3, 0], [4, 5, 6], [1, 2, 0], [3, 0, 0]]),
                    "b": tf.constant([1, 2, 3, 4, 0]),
                },
                {
                    "a": tf.constant([[1, 2, 3], [4, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    "b": tf.constant([5, 6, 7, 8, 0]),
                },
            ],
        ),
        # Test a case where an input element already exceeds batch_size.
        # We should trim the batch in this case.
        dict(
            examples=[{"a": tf.ones([6, 3], dtype=tf.int32), "b": tf.ones([12], dtype=tf.int32)}],
            expected=[{"a": tf.ones([5, 3], dtype=tf.int32), "b": tf.ones([5], dtype=tf.int32)}],
        ),
    )
    def test_trim_and_pack_to_batch(
        self,
        examples: Sequence[dict[str, tf.Tensor]],
        expected: Sequence[dict[str, tf.Tensor]],
        spec: Optional[dict] = None,
    ):
        processor = chain(trim_to_batch(batch_size=5), pack_to_batch(batch_size=5))
        source = fake_source(
            is_training=False,
            examples=examples,
            spec=spec
            or {
                "a": tf.TensorSpec(shape=[None, 3], dtype=tf.int32),
                "b": tf.TensorSpec(shape=[None], dtype=tf.int32),
            },
        )
        actual_ds = processor(source())
        actual = list(actual_ds)
        tf.nest.map_structure(self.assertAllEqual, expected, actual)
        expected_element_spec = {
            "a": tf.TensorSpec(shape=(5, 3), dtype=tf.int32, name=None),
            "b": tf.TensorSpec(shape=(5,), dtype=tf.int32, name=None),
        }
        tf.nest.map_structure(self.assertAllEqual, actual_ds.element_spec, expected_element_spec)


class ConvertRaggedTensorTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        dict(feature_shapes={"a": [None, 5]}), dict(feature_shapes={"a": [None, 5], "b": [5]})
    )
    def test_ragged_to_tensor(self, feature_shapes):
        examples = [
            {"a": tf.ragged.constant([[1, 2, 3]]), "b": tf.constant([5])},
            {"a": tf.ragged.constant([[1]]), "b": tf.constant([5])},
            {"a": tf.ragged.constant([[1, 2]]), "b": tf.constant([5])},
        ]
        ds = fake_source(is_training=False, examples=examples)()
        processor = ragged_to_tensor(feature_shapes=feature_shapes)
        actual = list(processor(ds))

        expected = [
            {"a": tf.constant([[1, 2, 3, 0, 0]]), "b": tf.constant([5])},
            {"a": tf.constant([[1, 0, 0, 0, 0]]), "b": tf.constant([5])},
            {"a": tf.constant([[1, 2, 0, 0, 0]]), "b": tf.constant([5])},
        ]
        tf.nest.map_structure(self.assertAllEqual, expected, actual)


class TrimAndPadTest(parameterized.TestCase):
    @parameterized.product(
        [
            {
                "max_len": 7,
                "expected_tensor": [
                    [3, 1, 0, 0, 0, 0, 0],
                    [3, 5, 4, 1, 0, 0, 0],
                    [3, 1, 6, 0, 0, 0, 0],
                ],
            },
            {
                "max_len": 5,
                "expected_tensor": [
                    [3, 1, 0, 0, 0],
                    [3, 5, 4, 1, 0],
                    [3, 1, 6, 0, 0],
                ],
            },
            {
                "max_len": 3,
                "expected_tensor": [
                    [3, 1, 0],
                    [3, 5, 4],
                    [3, 1, 6],
                ],
            },
        ],
        [
            {
                "input_tensor": tf.ragged.constant(
                    [
                        [3, 1],
                        [3, 5, 4, 1],
                        [3, 1, 6],
                    ]
                )
            },
            {
                "input_tensor": tf.constant(
                    [
                        [3, 1, 0, 0],
                        [3, 5, 4, 1],
                        [3, 1, 6, 0],
                    ]
                )
            },
        ],
    )
    def test_trim_and_pad_tensor(
        self,
        max_len: int,
        input_tensor: Union[tf.Tensor, tf.RaggedTensor],
        expected_tensor: tf.Tensor,
    ):
        t = trim_and_pad_tensor(input_tensor, max_len=max_len)
        tf.debugging.assert_equal(expected_tensor, t)

    @parameterized.parameters(
        {
            "input_tensor": tf.ragged.constant(
                [
                    [
                        [3, 1],
                        [3, 5, 4, 1],
                        [3, 1, 6],
                    ],
                    [
                        [3, 1, 2, 5],
                        [3, 5, 4, 1],
                        [3, 4],
                    ],
                ]
            ),
            "expected_tensor": [
                [
                    [3, 1, 0],
                    [3, 5, 4],
                    [3, 1, 6],
                ],
                [
                    [3, 1, 2],
                    [3, 5, 4],
                    [3, 4, 0],
                ],
            ],
        },
    )
    def test_trim_and_pad_tensor_nd(
        self, input_tensor: Union[tf.Tensor, tf.RaggedTensor], expected_tensor: tf.Tensor
    ):
        max_len = 3
        t = trim_and_pad_tensor(input_tensor, max_len=max_len)
        tf.debugging.assert_equal(expected_tensor, t)

    @parameterized.parameters(
        {
            "pad_id": -1,
            "max_len": 3,
            "input_tensor": tf.ragged.constant(
                [
                    [3, 1],
                    [3, 5, 4, 1],
                    [3, 1, 6],
                ]
            ),
            "expected_tensor": [
                [3, 1, -1],
                [3, 5, 4],
                [3, 1, 6],
            ],
        },
    )
    def test_trim_and_pad_non_zero_pad_id(
        self,
        pad_id: int,
        max_len: int,
        input_tensor: Union[tf.Tensor, tf.RaggedTensor],
        expected_tensor: tf.Tensor,
    ):
        t = trim_and_pad_tensor(input_tensor, max_len=max_len, pad_id=pad_id)
        tf.debugging.assert_equal(expected_tensor, t)


class DisableShuffleRecursivelyTest(parameterized.TestCase):
    """Tests disable_shuffle_recursively."""

    def test_disable_shuffle_recursively(self):
        cfg = Input.default_config().set(
            source=config_for_function(with_processor).set(
                source=config_for_function(tfds_dataset).set(
                    train_shuffle_buffer_size=10, train_shuffle_files=True
                ),
                processor=config_for_function(identity),
            )
        )
        disable_shuffle_recursively(cfg)
        self.assertEqual(cfg.source.source.train_shuffle_buffer_size, 0)
        self.assertEqual(cfg.source.source.train_shuffle_files, False)


class ElementSpecTest(parameterized.TestCase):
    """Tests Input.element_spec()."""

    def test_element_spec(self):
        cfg = Input.default_config().set(
            source=config_for_function(with_processor).set(
                source=config_for_function(fake_text_source),
                processor=config_for_function(identity),
            ),
            processor=config_for_function(identity),
            batcher=config_for_function(batch).set(
                global_batch_size=2,
                pad_example_fn=default_pad_example_fn,
            ),
            is_training=True,
            name="test",
        )
        self.assertEqual(
            {"text": jax.ShapeDtypeStruct(shape=(2,), dtype=object)},
            cfg.instantiate(parent=None).element_spec(),
        )


if __name__ == "__main__":
    absltest.main()

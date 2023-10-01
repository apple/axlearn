# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# Copyright 2022 The SeqIO Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

# pylint: disable=too-many-lines
"""Input generator based on tf.data."""
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import jax
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    # Necessary for S3 access. See, e.g: https://github.com/tensorflow/tensorflow/issues/51583
    # pytype: disable=import-error
    import tensorflow_io as tfio  # pylint: disable=unused-import
except ImportError:
    pass

from absl import logging
from jax import numpy as jnp
from jax.experimental import multihost_utils
from seqio import map_over_dataset
from typing_extensions import Protocol

from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    ConfigOr,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)
from axlearn.common.module import Module
from axlearn.common.utils import NestedTensor, Tensor, as_numpy_array, get_data_dir


class BuildDatasetFn(Protocol):
    """A function to create a tf.data.Dataset instance."""

    def __call__(self) -> tf.data.Dataset:
        ...


class DatasetToDatasetFn(Protocol):
    """A function to create a tf.data.Dataset instance from the given dataset."""

    def __call__(self, ds: Optional[tf.data.Dataset], **kwargs) -> tf.data.Dataset:
        ...


# TODO(markblee): Deprecate in favor of maybe_set_config.
def maybe_set_is_training(cfg: ConfigOr[Any], *, is_training: Optional[bool]) -> ConfigOr[Any]:
    if is_training is not None and hasattr(cfg, "is_training"):
        return cfg.set(is_training=is_training)
    else:
        return cfg


def tfds_read_config(
    *,
    is_training: bool,
    num_shards: Optional[int] = None,
    shard_index: Optional[int] = None,
    read_parallelism: int = 1,
    decode_parallelism: int = 32,
) -> tfds.ReadConfig:
    """Constructs a ReadConfig for tfds dataset.

    Note: different values of read_parallelism may unintentionally affect the determinism:
    https://www.tensorflow.org/datasets/determinism#determinism_caveat_interleave_args.

    Args:
        is_training: Whether the examples are used for training.
            If True, examples may be read in parallel and shuffled.
            Otherwise examples will be read sequentially to ensure a deterministic order.
        num_shards: Partition the input examples in the dataset split to this number of shards.
            Defaults to jax.process_count().
        shard_index: The shard index, in range [0, num_shards). Defaults to jax.process_index().
        read_parallelism: The number of parallel calls for reading data.
            Only used when is_training=True.
        decode_parallelism: The number of parallel calls for decoding examples.
            Only used when is_training=True.

    Returns:
        A tfds.ReadConfig.
    """
    num_shards = jax.process_count() if num_shards is None else num_shards
    shard_index = jax.process_index() if shard_index is None else shard_index
    num_parallel_calls_for_read = read_parallelism if is_training else 1
    num_parallel_calls_for_decode = decode_parallelism if is_training else 1
    return tfds.ReadConfig(
        interleave_cycle_length=num_parallel_calls_for_read,
        num_parallel_calls_for_interleave_files=num_parallel_calls_for_read,
        num_parallel_calls_for_decode=num_parallel_calls_for_decode,
        input_context=tf.distribute.InputContext(
            num_input_pipelines=num_shards, input_pipeline_id=shard_index
        ),
    )


def _infer_num_shards(builder: tfds.core.DatasetBuilder, split: str) -> Optional[int]:
    """Attempts to infer the number of shards associated with the given split.
    For subsplits, a `split in builder.info.splits` check is not supported.

    Args:
        builder: A tfds builder.
        split: The split or subsplit.

    Returns:
        The number of shards or None if it cannot be inferred.
    """
    try:
        num_shards = builder.info.splits[split].num_shards
    except Exception as e:  # pylint: disable=broad-except
        logging.info("Could not infer num shards in split %s: %s", split, e)
        num_shards = None
    return num_shards


def _infer_num_examples(builder: tfds.core.DatasetBuilder, split: str) -> Optional[int]:
    """Attempts to infer the number of examples associated with the given split.
    For subsplits, a `split in builder.info.splits` check is not supported.

    Args:
        builder: A tfds builder.
        split: The split or subsplit.

    Returns:
        The number of examples or None if it cannot be inferred.
    """
    try:
        num_examples = builder.info.splits[split].num_examples
    except Exception as e:  # pylint: disable=broad-except
        logging.info("Could not infer num examples in split %s: %s", split, e)
        num_examples = None
    return num_examples


def _maybe_shard_examples(
    builder: tfds.core.DatasetBuilder,
    read_config: InstantiableConfig,
    split: str,
    required_shards: int,
    is_training: bool,
    dataset_name: str,
) -> Union[str, tfds.core.splits.SplitArg]:
    """Determines how to split the examples into required number of shards.

    If there are more examples than `required_shards`, split the examples evenly. Otherwise,
        raise ValueError if during training and repeat the examples for all shards if during
        inference.

    Args:
        builder: A tfds builder.
        read_config: A Config that instantiates to a tfds.ReadConfig.
        split: The dataset split.
        required_shards: The required number of shards to split the examples into.
        is_training: Whether the examples are used for training.
        dataset_name: The tensorflow dataset name. For logging purpose only.

    Returns:
        The split for each process/host. If the examples are split evenly into the required
            number of shards, the return type is `tfds.core.splits.SplitArg`; otherwise,
            the split is returned unchanged, which would repeat the examples for all
            processes/hosts.

    Raises:
        ValueError: If the number of available examples is less than the `required_shards`
            during training.
    """
    per_process_split = split
    available_examples = _infer_num_examples(builder, split)
    # If available_examples is not enough to be distributed to required_shards,
    # skip the splitting and each host gets the same copy of the data. This avoids the
    # "Instruction [] corresponds to no data" error when calling builder.as_dataset().
    # Raise ValueError if during training.
    if available_examples is None or available_examples >= required_shards:
        if available_examples is None:
            logging.warning(
                "Could not infer number of examples. "
                "Proceed to split examples anyway. May result in error if "
                "number of examples < number of required_shards."
            )
        shard_index = read_config.shard_index or jax.process_index()  # type: ignore
        per_process_split = tfds.even_splits(
            per_process_split, n=required_shards, drop_remainder=False
        )[shard_index]
    else:
        if is_training:
            raise ValueError(
                f"Number of available examples ({available_examples}) < required_shards"
                f" ({required_shards})"
            )
        logging.info(
            "Repeating examples of %s/%s on each host because available examples "
            "(%s) < required_shards (%s)",
            dataset_name,
            split,
            available_examples,
            required_shards,
        )
    return per_process_split


def tfds_dataset(
    dataset_name: str,
    *,
    split: str,
    is_training: bool,
    shuffle_buffer_size: int,
    shuffle_files: Optional[bool] = None,
    data_dir: Optional[str] = None,
    download: bool = False,
    read_config: Optional[InstantiableConfig] = None,
    decoders: Optional[InstantiableConfig] = None,
) -> BuildDatasetFn:
    """Returns a BuildDatasetFn for the given TFDS dataset name and split.

    Args:
        dataset_name: The tensorflow dataset name.
        split: The dataset split.
        is_training: Whether the examples are used for training.
            If True, examples will be read in parallel and shuffled.
            Otherwise examples will be read sequentially to ensure a deterministic order.
        shuffle_buffer_size: The shuffle buffer size.
            If shuffle_buffer_size <= 0, no shuffling is done.
            If is_training=False, shuffle_buffer_size is always expected to be <= 0.
        shuffle_files: Whether to shuffle files. If None, infer from shuffle_buffer_size > 0.
        data_dir: Used for tfds.load. If None, use the value of the environment variable
            DATA_DIR, TFDS_DATA_DIR, or "~/tensorflow_datasets" (in that order).
        download: Whether to download the examples. If false, use the data under data_dir.
        read_config: a Config that instantiates to a tfds.ReadConfig.
            If None, constructs a default one with tfds_read_config().
        decoders: An optional config instantiating a (nested) mapping of feature names to decoders.
                See: https://www.tensorflow.org/datasets/api_docs/python/tfds/decode/Decoder

    Returns:
        A BuildDatasetFn, which returns a tf.data.Dataset for the specified TFDS.

    Raises:
        ValueError: If shuffle_buffer_size > 0 when is_training is False.
    """
    # TODO(ruomingp,markblee): Update the API to take train_shuffle_buffer_size instead.
    if not is_training and shuffle_buffer_size > 0:
        # Note that it's OK for is_training=True and shuffle_buffer_size=0 so that the inputs are
        # deterministic.
        raise ValueError("Shuffling should be disabled if is_training=False")

    if data_dir is None:
        data_dir = get_data_dir()

    if read_config is None:
        read_config = config_for_function(tfds_read_config).set(is_training=is_training)
    else:
        read_config = read_config.set(is_training=is_training)

    if shuffle_files is None:
        shuffle_files = shuffle_buffer_size > 0

    def fn() -> tf.data.Dataset:
        local_read_config = read_config.clone()
        builder = tfds.builder(dataset_name, data_dir=data_dir)
        if download:
            logging.info("Downloading %s", dataset_name)
            builder.download_and_prepare()
            logging.info("Downloading %s done", dataset_name)

        # If we can infer that the split doesn't have enough shards, fallback to using subsplit API.
        # Note: Without the alias, pytype will complain if we modify the nonlocal `split` below.
        per_process_split = split
        available_shards = _infer_num_shards(builder, split)
        if available_shards is not None:
            required_shards = read_config.num_shards or jax.process_count()  # type: ignore
            if available_shards < required_shards:
                per_process_split = _maybe_shard_examples(
                    builder=builder,
                    read_config=read_config,
                    split=split,
                    required_shards=required_shards,
                    is_training=is_training,
                    dataset_name=dataset_name,
                )
                local_read_config.set(num_shards=1, shard_index=0)

        ds: tf.data.Dataset = builder.as_dataset(
            split=per_process_split,
            shuffle_files=shuffle_files,
            read_config=local_read_config.instantiate(),
            decoders=maybe_instantiate(decoders),
        )
        if shuffle_buffer_size > 0:
            # Subsequent processing may merge/split examples (e.g. for T5), so shuffle examples
            # during training before any processing.
            ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        return ds

    return fn


def tfrecord_dataset(
    glob_path: str,
    is_training: bool,
    shuffle_buffer_size: int,
    features: Dict[str, tf.io.FixedLenFeature],
    compression_type: Optional[str] = None,
    read_parallelism: int = 1,
) -> BuildDatasetFn:
    """Builds a BuildDatasetFn for the given a TFRecord dataset name.

    Args:
        glob_path: The GCS path to the directory containing the TFRecord dataset to load.
        is_training: Whether the examples are used for training.
            If True, examples will be read in parallel and shuffled.
            Otherwise examples will be read sequentially to ensure a deterministic order.
        shuffle_buffer_size: The shuffle buffer size.
            If shuffle_buffer_size <= 0, no shuffling is done.
            If is_training=False, shuffle_buffer_size is always expected to be <= 0.
        features: A list of TFRecord style feature dictionaries to load. This is required to unpack
            the structure of the features.
        compression_type: Optional compression type.
        read_parallelism: The number of parallel calls for reading data.
            Only used when is_training=True.

    Returns:
        A BuildDatasetFn, which returns a tf.data.Dataset for the specified TFRecord dataset with
        unnecessary keys from features dict removed.

    Raises:
        ValueError: If shuffling is not enabled iff is_training is True.
    """
    if is_training != (shuffle_buffer_size > 0):
        raise ValueError("Shuffling should be enabled iff is_training is True")

    def _decode_record(record: Dict[str, tf.Tensor]):
        """Decodes a record to a TensorFlow example."""
        return tf.io.parse_single_example(serialized=record, features=features)

    def fn() -> tf.data.Dataset:
        num_parallel_calls_for_read = read_parallelism if is_training else 1
        glob_files = tf.io.gfile.glob(glob_path)
        # Shuffle files to avoid deterministic loading.
        filenames = tf.data.Dataset.from_tensor_slices(glob_files)
        if is_training and shuffle_buffer_size > 0:
            filenames = filenames.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        ds = tf.data.TFRecordDataset(
            filenames,
            compression_type=compression_type,
            num_parallel_reads=num_parallel_calls_for_read,
        )
        ds = ds.map(_decode_record)
        if shuffle_buffer_size > 0:
            # Subsequent processing may merge/split examples (e.g. for T5), so shuffle examples
            # during training before any processing.
            ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        return ds

    return fn


def sample_from_datasets(
    is_training: bool,
    *,
    sources: Sequence[InstantiableConfig],
    weights: Sequence[float],
    seed: Optional[int] = None,
) -> BuildDatasetFn:
    """Returns a data source formed by sampling from multiple data sources without replacement.
    All source datasets are repeated to prevent the sampling from stopping early and to prevent
    sampling from producing an inaccurate distribution. The caller is responsible for calling e.g.
    `take` if a repeating dataset is not desired, such as if is_training=False.

    You should use with_processor to integrate sources your own processors.
    Note: certain processors (such as batch) requires sufficient number of examples per host.
    When one source is too small so that there is no sufficient data emit from per-host slice,
    you should add `ds = ds.repeat()` before the processor.

    Args:
        is_training: Whether the examples are used for training.
            This parameter will be passed through to all input sources.
        sources: list of tf data source configs to sample from. Note that we can
            use `config_for_function(with_processor)` to integrate processors with sources.
        weights: list or tf.Tensor of probabilities of picking each dataset.
        seed: optional random seed.

    Returns:
        A BuildDatasetFn yielding the interleaved data source.

    Raises:
        ValueError: If sources is empty or if length of sources is not equal to the length of
            weights and processors (if specified).
    """
    if len(sources) < 1:
        raise ValueError("Expected at least one source")

    if len(sources) != len(weights):
        raise ValueError(f"Length of sources {sources} is not equal to length of weights {weights}")

    source_fns = [
        maybe_set_is_training(source, is_training=is_training).instantiate() for source in sources
    ]

    def fn() -> tf.data.Dataset:
        # Note: repeat is called even if not training.
        source_ds_list = [source_fn().repeat() for source_fn in source_fns]
        if any(source_ds.cardinality() == 0 for source_ds in source_ds_list):
            raise ValueError("Expected all cardinalities to be non-zero")

        return tf.data.Dataset.sample_from_datasets(
            source_ds_list,
            weights=weights,
            seed=seed,
            stop_on_empty_dataset=True,
        )

    return fn


def concatenate_datasets(
    is_training: bool,
    *,
    sources: Sequence[InstantiableConfig],
) -> BuildDatasetFn:
    """Concatenates the given datasets sequentially (one after another).

    Args:
        is_training: Whether the examples are used for training.
            This parameter will be passed through to all input sources.
        sources: list of tf data source configs to concatenate. Note that we can
            use `config_for_function(with_processor)` to integrate processors with sources.

    Returns:
        A BuildDatasetFn yielding the interleaved data source.

    Raises:
        ValueError: If sources is empty or if length of sources is not equal to the length of
            weights and processors (if specified).
    """
    if len(sources) < 1:
        raise ValueError("Expected at least one source")

    source_fns = [
        maybe_set_is_training(source, is_training=is_training).instantiate() for source in sources
    ]

    def fn() -> tf.data.Dataset:
        result = source_fns[0]()
        for source_fn in source_fns[1:]:
            result = result.concatenate(source_fn())
        return result

    return fn


def take(num_examples: int) -> DatasetToDatasetFn:
    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.take(num_examples)

    return fn


def unbatch() -> DatasetToDatasetFn:
    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.unbatch()

    return fn


def select_fields(fields: Sequence[str]) -> DatasetToDatasetFn:
    """Filter the dataset to only select the fields specified."""

    return rekey({k: k for k in fields}, retain_original_inputs=False)


def remove_fields(fields: Sequence[str]) -> DatasetToDatasetFn:
    """Filter the dataset to remove the fields specified."""

    def process_fn(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        new_example = {}
        for k, v in example.items():
            if k not in fields:
                new_example[k] = v
        return new_example

    return map_over_dataset(process_fn)


def filter_examples(filter_fn: Callable) -> DatasetToDatasetFn:
    """Filter the dataset with the given filter function."""

    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(filter_fn)

    return fn


def squeeze_fields(axis: Mapping[str, Optional[Union[int, Tuple[int, ...]]]]) -> DatasetToDatasetFn:
    """Squeeze fields specified using the corresponding axis.

    Args:
        axis: a mapping from the field name to the axis used to squeeze the field.
            Fields that are not in this mapping are not changed.

    Returns:
        A dataset where fields specified are squeezed.
    """

    def example_fn(example: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for field, ax in axis.items():
            example[field] = tf.squeeze(example[field], axis=ax)
        return example

    return map_over_dataset(example_fn)


def with_processor(
    source: InstantiableConfig,
    *,
    processor: InstantiableConfig,
    is_training: Optional[bool] = None,
) -> BuildDatasetFn:
    """Returns a BuildDatasetFn that combines the given `source` and `processor`.

    Args:
        source: A config that instantiates to a BuildDatasetFn.
        processor: A config that instantiates to a DatasetToDatasetFn.
        is_training: Whether the result dataset will be used for training.

    Returns:
        A BuildDatasetFn that applies `processor` on `source`.
    """
    source = maybe_set_is_training(source, is_training=is_training).instantiate()
    processor = maybe_set_is_training(processor, is_training=is_training).instantiate()

    def fn() -> tf.data.Dataset:
        ds = source()
        return processor(ds)

    return fn


def chain(*args, is_training: Optional[bool] = None) -> DatasetToDatasetFn:
    if is_training is not None:
        args = [maybe_set_is_training(processor, is_training=is_training) for processor in args]

    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        for processor in args:
            processor = maybe_instantiate(processor)
            ds = processor(ds)
        return ds

    return fn


# A function that takes a nested tf.TensorSpec structure as input and returns a nested structure
# with tf.Tensor as leaf nodes.
PadExampleFn = Callable[[Any], Any]


def default_pad_example_fn(element_spec: Any) -> Any:
    """Returns 0 values (or empty strings) for every tensor."""

    def spec_to_tensor(spec: tf.TensorSpec) -> tf.Tensor:
        return tf.zeros(shape=spec.shape, dtype=spec.dtype)

    example = tf.nest.map_structure(spec_to_tensor, element_spec)
    return example


def batch(
    global_batch_size: int,
    *,
    is_training: bool,
    pad_example_fn: PadExampleFn,
    prefetch_buffer_size: Optional[int] = None,
    post_batch_processor: Optional[ConfigOr[DatasetToDatasetFn]] = None,
    repeat: Optional[int] = None,
) -> DatasetToDatasetFn:
    """Returns a function that generates a tf.data.Dataset object.

    Note: batch(is_training=True) requires sufficient number of examples
    per host. When your data is too small, you should add `ds = ds.repeat()`
    before your batch.

    Args:
        global_batch_size: The global batch size across all replicas.
        is_training: Whether the examples are used for training.
            This parameter will be passed through to all input sources.
        pad_example_fn: Pad examples with the given function. Only used if is_training=False.
        prefetch_buffer_size: Size of prefetch buffer. This allows later
            elements to be prepared while the current element is being
            processed. If not set, `tf.data.experimental.AUTOTUNE` is used.
        post_batch_processor: An optional processor (or config instantiating to a processor) that
            applies batch-wise processing functions.
        repeat: The number of times to repeat the batches from the dataset.
            If None, repeat indefinitely if is_training=True and do not repeat otherwise.
            Otherwise must be a positive integer.

    Returns:
        A DatasetToDataset fn.

    Raises:
        ValueError: If
            - global_batch_size is not divisible by num_feeds,
            - Eval dataset has infinite cardinality, or
            - Repeat is not a positive integer.
    """
    num_feeds = jax.process_count()
    feed_index = jax.process_index()
    if global_batch_size % num_feeds != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size} must be divisible by "
            f"num_feeds ({num_feeds})"
        )
    per_feed_batch_size = global_batch_size // num_feeds

    # pylint: disable-next=too-many-branches
    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        if not is_training:
            ds = ds.cache()
            num_examples = ds.cardinality()
            if num_examples == tf.data.INFINITE_CARDINALITY:
                raise ValueError(f"Eval dataset should not have infinite cardinality: {ds}")
            if num_examples == tf.data.UNKNOWN_CARDINALITY:
                num_examples = (
                    ds.map(lambda *x: 1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    .reduce(0, lambda x, _: x + 1)
                    .numpy()
                )
                logging.warning("Manually counted evaluation dataset size: %s", num_examples)
            target_num_examples = num_examples
            if num_examples % per_feed_batch_size != 0:
                target_num_examples += per_feed_batch_size - num_examples % per_feed_batch_size
            if num_feeds > 1:
                # Ensure that we do not run into the "last batch" problem.
                # See: https://jax.readthedocs.io/en/latest/multi_process.html
                target_num_examples = int(
                    jnp.max(
                        multihost_utils.process_allgather(
                            jnp.array([target_num_examples]), tiled=False
                        )
                    )
                )
            if num_examples < target_num_examples:
                logging.info(
                    "Padding evaluation dataset from %s to %s.", num_examples, target_num_examples
                )
                # Save the element_spec before batching.
                element_spec = ds.element_spec

                def ds_fn():
                    yield pad_example_fn(element_spec)

                pad_ds = tf.data.Dataset.from_generator(
                    ds_fn, output_signature=element_spec
                ).repeat(target_num_examples - num_examples)
                ds = ds.concatenate(pad_ds)

        ds = ds.batch(per_feed_batch_size, drop_remainder=True)

        # Post batch processing methods at batch-level.
        if post_batch_processor:
            ds = maybe_instantiate(post_batch_processor)(ds)

        if not is_training and num_feeds > 1:
            num_eval_batches = ds.cardinality()
            if num_eval_batches == tf.data.UNKNOWN_CARDINALITY:
                # Compute the cardinality. Number of eval batches is typically small.
                num_eval_batches = ds.reduce(0, lambda x, _: x + 1).numpy()
            logging.info("Feed %s has %s eval batches.", feed_index, num_eval_batches)
            multihost_utils.assert_equal(
                num_eval_batches,
                f"Number of eval batches are not all equal ({num_eval_batches})",
            )

        if repeat is None:
            if is_training:
                ds = ds.repeat()
        else:
            if not isinstance(repeat, int) or repeat <= 0:
                raise ValueError(f"Invalid repeat (must be a positive integer): {repeat}")
            ds = ds.repeat(repeat)
        # If `prefetch_buffer_size` is not set, use autotune.
        ds = ds.prefetch(prefetch_buffer_size or tf.data.experimental.AUTOTUNE)
        return ds

    return fn


def identity() -> DatasetToDatasetFn:
    """Identity function, useful for example as batcher when data is already batched."""

    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds

    return fn


def skip_on_error() -> DatasetToDatasetFn:
    """Silently skip examples in the dataset that raise an error."""

    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.apply(tf.data.experimental.ignore_errors())

    return fn


def extract_from_sequence(
    in_key: str, out_key: str, idx: Union[int, slice] = 0
) -> DatasetToDatasetFn:
    """Provides function to extract slice or value at index from keyed sequence.

    E.g. if the input is {key1: [0, 1, 2]} and in_key = "key1", out_key = "key1_element", idx = 0
    then the output will be: dict(key1=[0, 1, 2], key1_element=0).

    Args:
        in_key: Key in input example that points to a sequential container.
        out_key: Key in output example where extracted element(s) will be.
        idx: Index or slice to extract from input container "key".

    Returns:
        Function that extracts slice or index from in_key in input example.
    """

    def fn(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        example[out_key] = example[in_key][idx]
        return example

    return map_over_dataset(fn)


def rekey(
    key_map: Dict[str, str],
    default_value: Optional[Any] = "",
    retain_original_inputs: bool = False,
) -> DatasetToDatasetFn:
    """Replace the feature keys according to mapping in `key_map`.

    Like seqio's rekey, except:
        1. We allow for a configurable default value
            (used if the reference key is falsey--e.g. None--or if missing in the input example).
        2. We optionally allow retain keys not explicitly mentioned in the key-map.

    Ref: <https://github.com/google/seqio/blob/9748501b/seqio/preprocessors.py#L30-L52>

    Args:
        key_map: dictionary mapping new keys to original keys.
            If falsey, return input (to match seqio behavior).
        default_value: value to set new key to if old key-value doesn't exist.
            If None, then we do not write the new key-value pair when missing an old key-value
                or when the provided reference key is falsey (to match seqio).
        retain_original_inputs: whether to retain all the keys provided in the original input
            example (if False, only keys specified in the key map will be in the output).

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict.
    """

    def fn(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        if not key_map:
            return example
        output = example if retain_original_inputs else {}
        for new_key, old_key in key_map.items():
            if not old_key or old_key not in example:
                if default_value is not None:
                    output[new_key] = default_value
                continue
            output[new_key] = example[old_key]
        return output

    return seqio.map_over_dataset(fn)


def shuffle(shuffle_buffer_size: int) -> DatasetToDatasetFn:
    """Shuffle dataset if given buffersize is valid (i.e. > 0).

    Args:
        shuffle_buffer_size: A tf.int64 scalar tf.Tensor, representing
            the number of elements from this dataset from which the new dataset will sample.

    Returns:
        A DatasetToDatasetFn that shuffles the dataset as is
            if shuffle_buffer_size < 0 else the shuffled dataset.
    """

    def fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        if shuffle_buffer_size > 0:
            ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

        return ds

    return fn


def unpack(key_map: Dict[str, Tuple[str, ...]]) -> DatasetToDatasetFn:
    """Provides function to return flattened values according to key map.

    E.g. if the input is {key1: {key2: {key3: value}}} and
    key_map = dict(new_key1=("key1", "key2", "key3")), then the output will be
    dict(new_key1=input["key1"]["key2"]["key3"]).

    Args:
        key_map: Keys describe output key names for paths input paths in value.

    Returns:
        Function that unpacks nested values in example according to key map.
    """

    def fn(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        for new_key, old_path in key_map.items():
            value = example[old_path[0]]
            for step in old_path[1:]:
                value = value[step]
            example[new_key] = value
        return example

    return seqio.map_over_dataset(fn)


def ragged_to_tensor(feature_shapes: Dict[str, Any], default_value: int = 0) -> DatasetToDatasetFn:
    """Converts ragged tensors specified in `feature_shapes`
    to a rectangular tensor of the specified shape, padding with `default_value` as necessary.

    Args:
        feature_shapes: A dict in which keys map to padded shapes.
        default_value: An int value to default to when padding ragged tensors.

    Returns:
        A dataset with full tensors padded with default_value.
    """

    def fn(example: Dict[str, tf.Tensor]):
        for k, v in example.items():
            if isinstance(v, tf.RaggedTensor) and k in feature_shapes:
                example[k] = v.to_tensor(default_value=default_value, shape=feature_shapes[k])
        return example

    return seqio.map_over_dataset(fn)


class Input(Module):
    """A Module to generate input batches with tf.data.Dataset.

    This input module contains three components:

    * source: generates the raw examples
    * processor: processes examples (potentially splitting and merging examples)
    * batcher: converts a stream of examples to a stream of batches

    This structure allows the users to replace source but reuse processor/batcher, e.g.,
    for inference.
    """

    @config_class
    class Config(Module.Config):
        """Configures Input."""

        is_training: Required[bool] = REQUIRED

        # TODO(xianzhi): consider unifying `source`, `processor` and `batcher` with a single
        # BuildDatasetFn.

        # A config that instantiates to a BuildDatasetFn. The result dataset will contain
        # a stream of examples representing one epoch of the source dataset.
        source: Required[InstantiableConfig] = REQUIRED

        # A config that instantiates to a DatasetToDatasetFn, which processes examples from
        # the source dataset and generates the example dataset to be batched, potentially
        # splitting and merging examples.
        processor: Required[InstantiableConfig] = REQUIRED

        # A config that instantiates to a DatasetToDatasetFn, which performs batching of examples.
        batcher: InstantiableConfig = config_for_function(batch)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._source = maybe_set_is_training(cfg.source, is_training=cfg.is_training).instantiate()
        self._processor = maybe_set_is_training(
            cfg.processor, is_training=cfg.is_training
        ).instantiate()
        self._batcher = maybe_set_is_training(
            cfg.batcher, is_training=cfg.is_training
        ).instantiate()

    @property
    def source(self) -> BuildDatasetFn:
        return self._source

    @property
    def processor(self) -> DatasetToDatasetFn:
        return self._processor

    def dataset(self) -> tf.data.Dataset:
        return self._batcher(self._processor(self._source()))

    def __iter__(self) -> Iterable[NestedTensor]:
        for input_batch in self.dataset():
            yield as_numpy_array(input_batch)


def disable_shuffle_recursively(cfg: Input.Config):
    """Disables all shuffling on the input config.

    This is useful for disabling shuffle during eval, or for deterministic input pipelines.
    """

    def enter_fn(_, child, default_kv):
        if isinstance(child, ConfigBase):
            for k in child.keys():
                if k.endswith("shuffle_buffer_size"):
                    setattr(child, k, 0)
                elif k == "shuffle_files":
                    setattr(child, k, False)
                elif "shuffle" in k:
                    logging.warning(
                        "Encountered an unrecognized key %s with 'shuffle' in the name.", k
                    )
        return default_kv

    cfg.visit(visit_fn=lambda k, v: None, enter_fn=enter_fn)


def preserve_element_spec(
    fn: DatasetToDatasetFn, key_map: Optional[Dict[str, str]] = None
) -> DatasetToDatasetFn:
    """Wraps a processor by ensuring that it does not change the dataset element_spec.

    Args:
        fn: The original processor.
        key_map: An optional mapping from new keys to original keys, similar to rekey.
            For instance, a processor may add a new key which is not in the original dataset spec;
            This allows setting those keys to have the same spec as other keys.

    Returns:
        A DatasetToDatasetFn which preserves the original element_spec (modulo the changes applied
        via key_map).
    """

    def process_dataset_fn(ds: tf.data.Dataset):
        orig_spec = ds.element_spec
        ds = fn(ds)
        if key_map:
            for k, v in key_map.items():
                orig_spec[k] = orig_spec[v]

        # Restore the original element_spec.
        def _set_shape(x):
            for k, v in x.items():
                v.set_shape(orig_spec[k].shape)
            return x

        return ds.map(_set_shape, num_parallel_calls=tf.data.AUTOTUNE)

    return process_dataset_fn


def add_static_fields(key_map: Dict[str, Any]) -> DatasetToDatasetFn:
    """Adds a predetermined set of key, value pairs to each example.

    Args:
        key_map: Key, value pairs to add.

    Returns:
        A DatasetToDatasetFn that adds the key, value pairs to each input example.
    """

    @seqio.map_over_dataset
    def fn(example: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in key_map.items():
            example[key] = value
        return example

    return fn


def pad_to_batch(batch_size: int, pad_value: int = 0) -> DatasetToDatasetFn:
    """Pads along the first (batch) dimension.

    Example:

        Input: batch_size=5
        {"a": [[1,0,0],[2,3,0],[4,5,6]], "b": [1,2]},
        {"a": [[1,2,0]],                 "b": [3]},

        Output:
        {
            "a": [[1, 0, 0], [2, 3, 0], [4, 5, 6], [0, 0, 0], [0, 0, 0]],
            "b": [1, 2, 0, 0, 0],
        },
        {
            "a": [[1, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'b': [3, 0, 0, 0, 0],
        }

    Args:
        batch_size: Desired batch size.
        pad_value: Value to pad to batch size.

    Returns:
        A DatasetToDatasetFn that pads to the batch size.
    """

    def pad_fn(v: tf.Tensor):
        paddings = [
            [[0, batch_size - tf.shape(v)[0]]],
            tf.zeros([tf.rank(v) - 1, 2], dtype=tf.int32),
        ]
        return tf.pad(v, paddings=tf.concat(paddings, 0), constant_values=pad_value)

    def process_example_fn(example: Dict[str, tf.Tensor]):
        return tf.nest.map_structure(pad_fn, example)

    return seqio.map_over_dataset(process_example_fn)


def pack_to_batch(batch_size: int, pad_value: int = 0) -> DatasetToDatasetFn:
    """Packs along the first (batch) dimension.

    Specifically, given a dataset of elements with leaves tf.Tensor(shape=[None, ...]), we produce
    elements with leaves tf.Tensor(shape=[batch_size, ...]) by stacking consecutive elements along
    the batch dimension.

    Elements must not initially have a batch size larger than `batch_size`, as it's not clear
    whether trim or pad is the intended behavior. If the caller intends to trim larger initial
    batches, use `trim_to_batch` prior to this processor.

    Different fields are allowed to have different initial batch sizes, but must match in the other
    dimensions (i.e., must be stackable).

    Example:

        Input: batch_size=5
        {"a": [[1,0,0],[2,3,0],[4,5,6]], "b": [1,2]},
        {"a": [[1,2,0]],                 "b": [3]},
        {"a": [[3,0,0]],                 "b": [4]},
        {"a": [[1,2,3],[4,0,0]],         "b": [5,6,7,8]},

        Output:
        {
            "a": [[1, 0, 0], [2, 3, 0], [4, 5, 6], [1, 2, 0], [3, 0, 0]],
            "b": [1, 2, 3, 4, 0],
        },
        {
            "a": [[1, 2, 3], [4, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            'b': [5, 6, 7, 8, 0],
        }

    Args:
        batch_size: Desired batch size.
        pad_value: Value to pad to batch size, if there is a remainder.

    Returns:
        A DatasetToDatasetFn that packs to the batch size.
    """

    def _init_state(dtype):
        return dict(
            # A mutable array to hold a single packed batch.
            accum=tf.TensorArray(dtype=dtype, size=0, dynamic_size=True, clear_after_read=True),
            size=0,  # Total accumulated batch size.
            i=0,  # Index of accum to write to.
        )

    def scan_fn(carry: Dict[str, Any], elem: Dict[str, tf.Tensor]):
        out = {}
        # Produce a new batch if any field cannot be packed any further.
        flush = tf.reduce_any(
            [carry[k]["size"] + tf.shape(v)[0] > batch_size for k, v in elem.items()]
        )
        # TODO(markblee): More generally, we can use tf.nest.map_structure.
        for k, x in elem.items():
            tf.debugging.assert_less_equal(
                tf.shape(x)[0],
                batch_size,
                message="Encountered a batch exceeding batch_size.",
            )
            state = carry[k]
            if flush:
                out[k] = state["accum"].concat()  # Emit the batch.
                state = _init_state(x.dtype)  # Reset state.
            else:
                out[k] = tf.constant([], dtype=x.dtype)  # Emit a dummy.

            state["accum"] = state["accum"].write(state["i"], x)
            state["size"] = state["size"] + tf.shape(x)[0]
            state["i"] = state["i"] + 1
            carry[k] = state
        return carry, out

    def define_shape(element_spec: Any, batch_size: int):
        def fn(example: Dict[str, tf.Tensor]):
            for k, t in example.items():
                t.set_shape((batch_size, *element_spec[k].shape.as_list()[1:]))
            return example

        return seqio.map_over_dataset(fn)

    def fn(ds: tf.data.Dataset):
        element_spec = ds.element_spec
        # Concat a dummy "end of dataset" element to force the last batch to be emitted.
        eod = tf.data.Dataset.from_tensors(
            {
                k: tf.reshape(tf.constant([], dtype=v.dtype), shape=[batch_size, 0])
                for k, v in ds.element_spec.items()
            }
        )
        ds = ds.concatenate(eod)
        # Construct initial carry state.
        state = tf.nest.map_structure(lambda spec: _init_state(spec.dtype), ds.element_spec)
        # Accumulate batches.
        ds = ds.scan(initial_state=state, scan_func=scan_fn)
        # Remove dummy outputs.
        ds = ds.filter(lambda x: tf.reduce_all([tf.shape(v)[0] != 0 for v in x.values()]))
        # Pad the rest to fixed batch.
        ds = pad_to_batch(batch_size, pad_value=pad_value)(ds)
        # Apply shapes back due to scan leaving shapes <unknown>.
        ds = define_shape(element_spec, batch_size)(ds)
        return ds

    return fn


def trim_to_batch(batch_size: int) -> DatasetToDatasetFn:
    """Trims the first (batch) dimension."""

    def trim(example: Dict[str, tf.Tensor]):
        for k, v in example.items():
            example[k] = v[:batch_size]
        return example

    return seqio.map_over_dataset(trim)


def trim_and_pad_tensor(
    t: Union[tf.Tensor, tf.RaggedTensor], max_len: int, pad_id: int = 0
) -> tf.Tensor:
    """Convert a tensor to uniform length by trimming and padding the last dimension.

    If the last dim is longer than ``max_len``, it will be trimmed to length ``max_len``;
    if the last dim is shorter than ``max_len`` it will be padded to ``max_len`` using ``pad_id``.

    Args:
        t: The tensor to trim and pad.
        max_len: The length to trim and pad to.
        pad_id: The token id used to pad sequences to the same length on the right.

    Returns:
        A tensor whose last dimension is trimmed and padded.
    """
    if isinstance(t, tf.RaggedTensor):
        shape = t.bounding_shape()
        t = t.to_tensor(default_value=pad_id)
    else:
        shape = tf.shape(t)

    t = t[..., :max_len]
    pad_amt = max_len - shape[-1]
    if pad_amt > 0:
        t = tf.pad(t, [(0, 0)] * (len(t.shape) - 1) + [(0, pad_amt)], constant_values=pad_id)
    t = tf.ensure_shape(t, t.shape[:-1] + [max_len])

    return t

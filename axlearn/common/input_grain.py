# Copyright © 2024 Apple Inc.

"""Input processing based on `grain`.

See https://github.com/google/grain/tree/main/docs for details.

In contrast with tf.data, pygrain transforms are implemented as pure Python functions, e.g. with
numpy. This means that it is compatible with JAX, SentencePiece (without tf_text), tiktoken, etc.
Further, shuffling is handled by index manipulation so that shuffle buffers are not needed, and so
that shuffling can be done deterministically in an online fashion by specifying a seed.

The API in this module intends to follow that of `input_tf_data` closely. Specifically, we mainly
operate on `grain.MapDataset` and `grain.IterDataset` which expose a similar API as
`tf.data.Dataset`, and `grain.Transformations` which are similar to `tfds` mappers.

Typically, one starts by constructing a source dataset (`BuildDatasetFn`), e.g. via
`array_record_dataset`; and then applies one or more transformations to the source. Once a source
dataset is constructed, it can be configured on the `Input` module to be used as a dataset iterator.

On `grain.MapDataset` vs `grain.IterDataset`:
* `grain.IterDataset` does not support efficient indexing and cardinality, and supports a limited
  set of transforms. Converting back from a `grain.IterDataset` to `grain.MapDataset` is also
  potentially expensive. Thus, if conversion is necessary, it is recommended to so near the end of
  the input pipeline.

On `repeat` and `shuffle`:
* `shuffle` after `repeat` will mix elements from different epochs.
* `shuffle` before `repeat` will produce a unique shuffle each epoch (i.e., the epoch id is
    included in the shuffle seed).
* `repeat` with `num_repeat=None` will produce datasets with size `sys.maxsize`.
"""

import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from array_record.python.array_record_data_source import PathLikeOrFileInstruction
from grain._src.python.data_loader import _determine_worker_count
from grain._src.python.dataset import dataset as dataset_base
from jax.experimental import multihost_utils

from axlearn.common import input_base, utils
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    Required,
    config_class,
    config_for_class,
    maybe_instantiate,
)
from axlearn.common.module import Module

Dataset = Union[grain.MapDataset, grain.IterDataset]
_T = TypeVar("_T")
SequenceOr = Union[Sequence[_T], _T]
Tensor = np.ndarray
# Same as `input_tf_data.PadExampleFn`.
PadExampleFn = Callable[[Any], Any]


class RaggedTensor(list):
    pass


@runtime_checkable
class _CallableTransform(Protocol):
    def __call__(self, example: Any) -> Any:
        ...


@runtime_checkable
class _RandomCallableTransform(Protocol):
    def __call__(self, example: Any, rng: np.random.Generator) -> Any:
        ...


# Grain supports a set of predefined transformations (e.g. grain.MapTransform), as well as callables
# taking a single example as input and outputting a single example.
# Not all grain.Transformations are supported by `ds.map` API.
# https://github.com/google/grain/blob/0f55c56b18912ddf467a542c361507aa9e2961e6/grain/_src/python/dataset/dataset.py#L365-L374
ExampleToExampleFn = Union[
    grain.MapTransform,
    grain.RandomMapTransform,
    _CallableTransform,
    _RandomCallableTransform,
]


@dataclass
class DispatchConfig:
    """Specifies the feed read configs.

    Similar to `PartitionSpec`, it specifies host-partitioning along each mesh dimension. Specifying
    a subset of leading mesh dims implies that the remaining trailing dims are all replicated (i.e.
    correspond to a single shard).

    Attributes:
        shard_index: Indices indicating the index of the feed along each mesh dimension.
        num_shards: Number of feeds along each mesh dimension.
    """

    shard_index: Sequence[int]
    num_shards: Sequence[int]

    def __post_init__(self):
        if not isinstance(self.shard_index, Sequence):
            self.shard_index = [self.shard_index]
        if not isinstance(self.num_shards, Sequence):
            self.num_shards = [self.num_shards]
        if len(self.shard_index) != len(self.num_shards):
            raise ValueError(f"{len(self.shard_index)=} should match {len(self.num_shards)=}.")
        if len(self.num_shards) < 1:
            raise ValueError(f"{self.num_shards=} cannot be empty.")
        if not all(0 <= idx < count for idx, count in zip(self.shard_index, self.num_shards)):
            raise ValueError(
                f"Each {self.shard_index=} should be between 0 and {self.num_shards=}, exclusive."
            )


class BuildDatasetFn(Protocol):
    """A function to create a grain data source."""

    def __call__(self, dispatch_config: DispatchConfig) -> Dataset:
        ...


def _copy_tree(x: _T) -> _T:
    """Copies tree structure without copying values."""
    return jax.tree.map(lambda v: v, x)


def _ragged_batch_size(tensor: Union[Tensor, RaggedTensor]) -> int:  # type: ignore
    """Determines the batch_size of the (optionally ragged) tensor.

    Ragged tensor are represented as list of np.ndarray.

    Args:
        tensor: A tensor, which could be an Tensor or RaggedTensor.

    Returns:
        batch_size of the tensor.
    """
    if isinstance(tensor, RaggedTensor):
        return len(tensor)
    elif hasattr(tensor, "shape"):
        return tensor.shape[0]
    else:
        raise NotImplementedError(type(tensor))


def array_record_dataset(
    paths: Union[PathLikeOrFileInstruction, Sequence[PathLikeOrFileInstruction]],
    *,
    seed: Optional[int],
) -> Dataset:
    """Builds an ArrayRecord dataset.

    Reference:
    https://github.com/google/array_record/blob/a9db114d73d800e1f84d7c7a2ff2b5370a7ba600/python/array_record_data_source.py#L214

    Args:
        paths: One or more array record paths, each of which can be a pathlike (e.g. string)
            or tfds `FileInstruction`. When reading subsets or a large number of files prefer to
            pass `FileInstruction`s.
        seed: Seed for any downstream transformations (e.g. `shuffle` or `random_map`).

    Returns:
        An ArrayRecord dataset.
    """
    source = grain.ArrayRecordDataSource(paths)
    ds = grain.MapDataset.source(source)
    if seed is not None:
        ds = ds.seed(seed)
    return ds


def sample_from_datasets(
    *,
    sources: Sequence[Dataset],
    weights: Sequence[float],
) -> Dataset:
    """Mixes one or more repeated data sources.

    Different from `input_tf_data.sample_from_datasets`, the mixing is deterministic:
    https://github.com/google/grain/blob/ddf825c68b6d2c811f9e599d7fb7ae7572affd8c/grain/_src/python/dataset/transformations/mix.py#L222

    Similar to `input_tf_data.sample_from_datasets`, datasets are repeated automatically, since
    otherwise the mixed dataset would terminate as soon as any source dataset is exhausted.
    Use `ds.slice` to limit to a subset of elements.

    Args:
        sources: One or more data sources to mix. Each should be a `grain.MapDataset`.
        weights: Relative weights for each dataset.

    Returns:
        A Dataset for the mixed data source.
    """

    def _ensure_repeated(sources: Sequence[Dataset]):
        # There is no easy way to check if a grain.IterDataset is repeated.
        for source in sources:
            if isinstance(source, grain.MapDataset) and len(source) != sys.maxsize:
                raise ValueError(
                    f"sample_from_datasets requires each dataset to be repeated, {source} is not."
                )
            if isinstance(source, grain.IterDataset):
                logging.info(
                    "Sampling from grain.IterDataset, please make sure your dataset is repeated."
                )

    _ensure_repeated(sources)
    # If any of the datasets are grain.IterDataset, we should use grain.IterDataset.mix().
    if any(isinstance(ds, grain.IterDataset) for ds in sources):
        return grain.IterDataset.mix(datasets=sources, weights=weights)

    return grain.MapDataset.mix(datasets=sources, weights=weights)


def default_pad_example_fn(example: utils.Nested[Any]) -> utils.Nested[Any]:
    """Returns the "zero-value" for every leaf."""

    def empty_like(leaf: Any) -> Tensor:
        if isinstance(leaf, Tensor):
            return np.empty_like(leaf)
        return type(leaf)()

    return jax.tree.map(empty_like, example)


class _UnbatchDatasetIterator(grain.DatasetIterator):
    """An iterator that unbatches np.arrays along dim=0."""

    def __init__(self, parent: grain.DatasetIterator, *, skip_empty_batch: bool = False):
        super().__init__(parent)
        # Index within the unbatched inputs.
        self._index = 0
        self._current_batch = None
        # Don't advance parent state until all indices in current batch have been yielded.
        self._parent_state = self._parent.get_state()
        self._skip_empty_batch = skip_empty_batch

    def __next__(self):
        example = None

        # Use a loop to avoid having to recursively call next(self).
        while example is None:
            # Note that self._index may initially be non-zero, e.g. if restoring from checkpoint
            # using `set_state`.
            if self._current_batch is None:
                # Possibly raises StopIteration.
                example = next(self._parent)
                leaves, structure = jax.tree.flatten(example)
                if not leaves:
                    example = None
                    continue  # Parent produced an empty batch, continue.

                # Make sure all leaves have same batch dim.
                if not all(
                    _ragged_batch_size(leaves[0]) == _ragged_batch_size(x) for x in leaves[1:]
                ):
                    # Convert RaggedTensors back to lists, so that `shapes` can traverse into leaf
                    # np.arrays for a more interpretable message.
                    example = jax.tree.map(
                        lambda x: list(x) if isinstance(x, RaggedTensor) else x, example
                    )
                    raise ValueError(
                        f"Expected all leaves to have same batch dim: {utils.shapes(example)}"
                    )
                self._current_batch = (leaves, structure)

            leaves, structure = self._current_batch
            assert len(leaves) > 0, self._current_batch
            batch_size = _ragged_batch_size(
                leaves[0]
            )  # All leaves have same batch size due to check above.
            if batch_size == 0 and self._skip_empty_batch:
                example = None
            else:
                assert 0 <= self._index < batch_size, (self._index, batch_size)
                example = jax.tree.unflatten(structure, (x[self._index] for x in leaves))
            self._index += 1

            # Move onto the next batch.
            if self._index >= batch_size:
                self._index = 0
                self._current_batch = None
                self._parent_state = self._parent.get_state()

        return example

    def get_state(self) -> dict[str, Any]:
        return {
            "parent": self._parent_state,
            "index": self._index,
        }

    def set_state(self, state: dict[str, Any]):
        self._parent.set_state(state["parent"])
        self._parent_state = state["parent"]
        self._index = state["index"]
        self._current_batch = None


class _UnbatchIterDataset(grain.IterDataset):
    def __init__(self, parents, *, skip_empty_batch: bool = False):
        super().__init__(parents)
        self._skip_empty_batch = skip_empty_batch

    def __str__(self) -> str:
        return "UnbatchIterDataset"

    def __iter__(self) -> _UnbatchDatasetIterator:
        return _UnbatchDatasetIterator(
            self._parent.__iter__(), skip_empty_batch=self._skip_empty_batch
        )


def unbatch(ds: Dataset, *, skip_empty_batch: bool = False) -> Dataset:
    """Similar to `input_tf_data.unbatch`.

    Unlike `batch`, which naively groups top-level elements, unbatch applies to JAX leaves only.
    For example, lists are not considered for batch dim, but rather as part of tree structure.

    Unlike grain's `flat_map`, there is no limit on the fan out.

    Args:
        ds: A Dataset where each example has leaves with the same batch dim.
        skip_empty_batch: Whether to skip batches with leading batch dim=0. If False, an assertion
            error will be raised upon encountering an empty batch.

    Returns:
        A Dataset with unbatched inputs.
    """
    _ensure_iter_dataset(ds)
    return _UnbatchIterDataset(ds, skip_empty_batch=skip_empty_batch)


def rekey(
    ds: Dataset,
    *,
    key_map: dict[str, str],
    default_value: Optional[Any] = "",
    retain_original_inputs: bool = False,
    separator: Optional[str] = None,
) -> Dataset:
    """Replace the feature keys according to mapping in `key_map`.

    Identical to `input_tf_data.rekey`, except that we return shallow copies of examples.

    Args:
        ds: A Dataset where each example is a dict.
        key_map: A dictionary mapping new keys to original keys.
            If falsey, return input (to match seqio behavior).
        default_value: Value to set new key to if old key-value doesn't exist.
            If None, then we do not write the new key-value pair when missing an old key-value
                or when the provided reference key is falsey (to match seqio).
        retain_original_inputs: Whether to retain all the keys provided in the original input
            example (if False, only keys specified in the key map will be in the output).
        separator: An optional separator. If provided, all keys and values of `key_map` will be
            treated as paths and split by the separator.

    Returns:
        A Dataset with rekeyed examples.
    """

    def has_path(x, path: str) -> bool:
        try:
            utils.get_recursively(x, path, separator=separator)
            return True
        except KeyError:
            return False

    def fn(example: dict[str, Tensor]) -> dict[str, Tensor]:
        if not key_map:
            return example
        output = _copy_tree(example) if retain_original_inputs else {}
        for new_key, old_key in key_map.items():
            if not old_key or not has_path(example, old_key):
                if default_value is not None:
                    utils.set_recursively(
                        output, value=default_value, path=new_key, separator=separator
                    )
                continue
            utils.set_recursively(
                output,
                value=utils.get_recursively(example, old_key, separator=separator),
                path=new_key,
                separator=separator,
            )
        return output

    return ds.map(fn)


def maybe_to_iter_dataset(
    ds: Dataset,
    *,
    read_options: ConfigOr[grain.ReadOptions] = config_for_class(grain.ReadOptions),
) -> grain.IterDataset:
    """Converts `grain.MapDataset` to `grain.IterDataset`.

    If the dataset is already a `grain.IterDataset`, is a no-op.
    See also `On grain.MapDataset vs grain.IterDataset` in file docstring.

    Args:
        ds: A Dataset.
        read_options: Read options when converting to `grain.IterDataset`.

    Returns:
        A `grain.IterDataset`.
    """
    read_options = maybe_instantiate(read_options)
    if isinstance(ds, grain.MapDataset):
        ds = ds.to_iter_dataset(read_options)
    return ds


def _ensure_iter_dataset(ds: Dataset):
    """Raises if not `grain.IterDataset`."""

    if not isinstance(ds, grain.IterDataset):
        raise ValueError(
            f"Expected a {grain.IterDataset.__name__}, got {type(ds)}. "
            f"Please use {maybe_to_iter_dataset.__name__} to convert the dataset."
        )


class _ProportionalSliceMapDataset(dataset_base.MapDataset[_T]):
    """Slices a MapDataset given the integer proportions."""

    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self, parent: dataset_base.MapDataset[_T], *, range_start: int, range_end: int, period: int
    ):
        super().__init__(parent)
        if not range_start < range_end <= period:
            raise ValueError(
                f"{range_start=}, {range_end=}, {period=} is not valid args. Please make sure"
                f"{range_start=} < {range_end=} and {range_end=} <= {period=}"
            )
        self._range_start = range_start
        self._range_end = range_end
        self._period = period
        self._stop = len(parent)
        # Let's denote T(range_start, range_end, period, stop) equals to the total count of
        # examples in this shard.
        # T(range_start, range_end, period, stop) = T(0, range_end, period, stop)
        #                                         - T(0, range_start, period, stop)
        # T(0, x, period, stop) = (stop // period) * x + min(stop % period, x)
        self._length = (
            (self._stop // self._period) * (range_end - range_start)
            + min(self._stop % self._period, range_end)
            - min(self._stop % self._period, range_start)
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        with self._stats.record_self_time():
            # Converts the index back to within range.
            index = index % len(self)
            parent_index = (
                self._range_start
                + index // (self._range_end - self._range_start) * self._period
                + index % (self._range_end - self._range_start)
            )

        return self._parent[parent_index]

    def __str__(self) -> str:
        return f"_ProportionalSliceMapDataset[{self._range_start}:{self._range_end}:{self._period}]"


def shard_dataset(ds: Dataset, dispatch_config: DispatchConfig) -> Dataset:
    """A convenience wrapper around `ds.slice`.

    Specifically, each feed reads `ds[shard_index::num_shards]`.
    E.g., if the dataset has 10 elements split among 4 feeds, feed 0,1 get 3 each, and feed 2,3 get
    2 each.

    Args:
        ds: A Dataset.
        dispatch_config: A dispatch config. See `DispatchConfig`.

    Returns:
        A sharded (sliced) dataset.
    """
    if not len(dispatch_config.shard_index) == len(dispatch_config.num_shards) == 1:
        raise NotImplementedError(dispatch_config)
    return ds.slice(slice(dispatch_config.shard_index[0], None, dispatch_config.num_shards[0]))


def shard_dataset_with_proportion(
    ds: Dataset,
    *,
    range_start: int,
    range_end: int,
    period: int,
) -> Dataset:
    """Shards dataset given within host proportions.

    Given a range_start, range_end, this function will shard examples at:
    [range_start + period * n, range_end + period * n) where n is an non-negative integer.

    This function is used when sharding a Dataset to multiple hosts. Period should be the total
    component weights, and range_start and range_end should be the local component weight.

    For example, if we want to shard a dataset with total weight of 100 to 30 and 70, we should do:

    shard_dataset_with_proportion(ds, range_start=0, range_end=30, period=100) and
    shard_dataset_with_proportion(ds, range_start=30, range_end=100, period=100).

    Args:
        ds: A Dataset.
        range_start: An integer indicating start of the range, inclusive.
        range_end: An integer indicating end of the range, exclusive.
        period: An integer indicating the period to be sampled.

    Returns:
        A sharded (sliced) dataset.
    """
    return _ProportionalSliceMapDataset(
        parent=ds, range_start=range_start, range_end=range_end, period=period
    )


def prefetch_dataset(
    ds: Dataset,
    *,
    multiprocessing_options: Optional[ConfigOr[grain.MultiprocessingOptions]] = None,
) -> Dataset:
    """Prefetches the dataset using multiple Python processes.

    This implicitly enables `grain.GrainPool` and `SharedMemoryArray` for numpy arrays.
    The processor requires a `grain.IterDataset`. See `On grain.MapDataset vs grain.IterDataset` in
    the file docstring.

    Args:
        ds: A Dataset.
        multiprocessing_options: Multiprocessing options.
            Used for setting e.g. number of CPU processes for prefetching. If `.num_workers` is left
            as 0, infers from `os.cpu_count()`.

    Returns:
        A prefetching Dataset.
    """
    if multiprocessing_options is None:
        multiprocessing_options = grain.MultiprocessingOptions()
    else:
        multiprocessing_options = maybe_instantiate(multiprocessing_options)

    # Prefetch requires num_workers > 0.
    if multiprocessing_options.num_workers == 0:
        multiprocessing_options = _determine_worker_count(input_worker_count=None)

    _ensure_iter_dataset(ds)
    return ds.prefetch(multiprocessing_options)


class _FixedLengthDatasetIterator(grain.DatasetIterator):
    """Iterate for a fixed length, truncating or producing padding examples as needed."""

    def __init__(
        self,
        parent: grain.DatasetIterator,
        *,
        pad_example: Any,
        length: int,
    ):
        super().__init__(parent)
        self._pad_example = pad_example
        self._length = length
        self._i = 0

    def __len__(self):
        return self._length

    def __next__(self):
        if self._i >= self._length:
            raise StopIteration
        try:
            element = next(self._parent)
        except StopIteration:
            element = self._pad_example
        self._i += 1
        return element

    def get_state(self):
        return {
            "parent": self._parent.get_state(),
            "i": self._i,
        }

    def set_state(self, state: dict[str, Any]):
        self._parent.set_state(state["parent"])
        self._i = state["i"]


class _FixedLengthIterDataset(grain.IterDataset):
    """An iter dataset that has a fixed length, truncating or padding as needed."""

    def __init__(self, parent: grain.IterDataset, *, pad_example: Any, length: int):
        super().__init__(parent)
        self._length = length
        self._pad_example = pad_example

    def __len__(self):
        return self._length

    def __iter__(self):
        parent_iter = self._parent.__iter__()
        return _FixedLengthDatasetIterator(
            parent_iter,
            pad_example=self._pad_example,
            length=self._length,
        )


# TODO(markblee): De-dup with input_tf_data.
def pad_for_evaluation(
    ds: Dataset,
    *,
    per_feed_batch_size: int,
    pad_example_fn: PadExampleFn = default_pad_example_fn,
    max_num_examples: int = 64_000,
) -> grain.IterDataset:
    """Pads the dataset to be a multiple of `per_feed_batch_size`.

    The processor will ensure that all data feeds pad to the same number of batches to avoid
    potential "last batch" problems.

    Args:
        ds: A Dataset. If an IterDataset, the cardinality will be manually counted, which requires
            iterating through the dataset. This is mostly tolerable for evaluation datasets that are
            relatively small.
        per_feed_batch_size: Per-feed batch size.
        pad_example_fn: A callable that takes an example and returns a padding example.
        max_num_examples: An upper bound on the number of examples expected in the dataset. This is
            mainly to avoid blocking indefinitely in the case where the input dataset is infinite.
            If a manual count of the dataset cardinality exceeds this value, we raise to avoid
            silently truncating the dataset.

    Returns:
        A padded dataset.

    Raises:
        ValueError: If evaluation dataset is empty or infinite; or if a manual count of the dataset
            size exceeds `max_num_examples`.
    """
    try:
        num_examples = len(ds)
        logging.info("Dataset %s has known cardinality: %s", ds, num_examples)
    except TypeError:
        logging.warning("Dataset %s has no known cardinality. Will attempt to count.", ds)
        num_examples = 0
        for _ in ds:
            num_examples += 1
            if num_examples >= max_num_examples:
                # pylint: disable-next=raise-missing-from
                raise ValueError(f"Giving up on counting eval dataset after {max_num_examples}.")

    # This case can happen if a map dataset calls `ds.repeat`.
    if num_examples >= sys.maxsize:
        raise ValueError(f"Evaluation dataset cannot have infinite cardinality: {ds}")
    if num_examples <= 0:
        raise ValueError(f"Evaluation dataset cannot be empty: {ds}")

    target_num_examples = num_examples
    if num_examples % per_feed_batch_size != 0:
        target_num_examples += per_feed_batch_size - num_examples % per_feed_batch_size
    if jax.process_count() > 1:
        # Ensure that we do not run into the "last batch" problem.
        # See: https://jax.readthedocs.io/en/latest/multi_process.html
        target_num_examples = int(
            jnp.max(
                multihost_utils.process_allgather(jnp.array([target_num_examples]), tiled=False)
            )
        )

    if num_examples < target_num_examples:
        pad_example = pad_example_fn(next(iter(ds)))
        logging.info("Padding evaluation dataset from %s to %s.", num_examples, target_num_examples)
        ds = _FixedLengthIterDataset(ds, pad_example=pad_example, length=target_num_examples)

    return ds


def per_feed_batch(
    ds: Dataset, *, global_batch_size: int, dispatch_config: DispatchConfig, **kwargs
):
    """Produces per-feed batches along dim=0.

    Args:
        global_batch_size: Global batch along dim=0.
        dispatch_config: Dispatch config.
        kwargs: Forwarded to `ds.batch`.

    Returns:
        A Dataset batched according to the feed batch size along dim=0.
    """
    num_shards = dispatch_config.num_shards[0]
    if global_batch_size % num_shards != 0:
        raise ValueError(f"{global_batch_size=} should be divisible by {num_shards=}")
    return ds.batch(global_batch_size // num_shards, **kwargs)


class Input(input_base.Input):
    """A Module to generate input batches with `grain`."""

    @config_class
    class Config(input_base.Input.Config):
        """Configures Input.

        Attributes:
            source: A BuildDatasetFn (or a config instantiating to one). The result dataset will
                contain a stream of examples representing one epoch of the source dataset.
        """

        source: Required[ConfigOr[BuildDatasetFn]] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: Input.Config = self.config
        self._source = maybe_instantiate(cfg.source)

    @property
    def source(self) -> BuildDatasetFn:
        return self._source

    def dataset(self) -> grain.IterDataset:
        if "input_dispatcher" in self.children:
            # TODO(markblee): Generalize to support ndim>1.
            read_config = self.input_dispatcher.feed_read_config()
        else:
            read_config = dict(shard_index=[jax.process_index()], num_shards=[jax.process_count()])
        return maybe_to_iter_dataset(self._source(DispatchConfig(**read_config)))

    def element_spec(self) -> utils.Nested[jax.ShapeDtypeStruct]:
        """Infers the element spec.

        Grain requires fetching an example from the dataset to extract the spec. To avoid reading
        actual data, replace your source dataset with one from `input_fake.fake_grain_source`.
        """
        ds = self.dataset()
        if isinstance(ds, grain.MapDataset):
            example = ds[0]
        else:
            example = next(ds.__iter__())  # pylint: disable=unnecessary-dunder-call

        def shape_dtype(x):
            if not hasattr(x, "shape") or not hasattr(x, "dtype"):
                raise ValueError(f"element_spec() requires Tensor-like leaves, got: {x}.")
            return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)

        return jax.tree.map(shape_dtype, example)

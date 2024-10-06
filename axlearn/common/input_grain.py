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
from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable

import grain.python as grain
import jax
import numpy as np
from array_record.python.array_record_data_source import PathLikeOrFileInstruction
from grain._src.python.data_loader import _determine_worker_count
from grain._src.python.dataset.transformations import packing

from axlearn.common import utils
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


class BuildDatasetFn(Protocol):
    """A function to create a grain data source."""

    def __call__(self) -> Dataset:
        ...


def _copy_tree(x: _T) -> _T:
    """Copies tree structure without copying values."""
    return jax.tree.map(lambda v: v, x)


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
    """Mixes one or more data sources.

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

    # Without repeat, mixing stops as soon as the first dataset is exhausted.
    def maybe_repeat(ds: Dataset):
        if not isinstance(ds, grain.MapDataset):
            raise ValueError(
                f"{sample_from_datasets.__name__} requires {grain.MapDataset.__name__}"
            )
        # Only repeat if not already infinite.
        if len(ds) != sys.maxsize:
            ds = ds.repeat()
        return ds

    return grain.MapDataset.mix(
        datasets=[maybe_repeat(source) for source in sources],
        weights=weights,
    )


def default_pad_example_fn(example: utils.Nested[Any]) -> utils.Nested[Any]:
    """Returns the "zero-value" for every leaf."""

    def empty_like(leaf: Any) -> Tensor:
        if isinstance(leaf, Tensor):
            return np.empty_like(leaf)
        return type(leaf)()

    return jax.tree.map(empty_like, example)


class _UnbatchDatasetIterator(grain.DatasetIterator):
    """An iterator that unbatches np.arrays along dim=0."""

    def __init__(self, parent: grain.DatasetIterator):
        super().__init__(stats=None)
        self._parent = parent
        # Index within the unbatched inputs.
        self._index = 0
        self._current_batch = None
        # Don't advance parent state until all indices in current batch have been yielded.
        self._parent_state = self._parent.get_state()

    def __next__(self):
        # Note that self._index may initially be non-zero, e.g. if restoring from checkpoint
        # using `set_state`.
        if self._current_batch is None:
            # Possibly raises StopIteration.
            example = next(self._parent)
            leaves, structure = jax.tree.flatten(example)
            if not leaves:
                return next(self)  # Parent produced an empty batch, continue.

            # Make sure all leaves have same batch dim.
            if not all(leaves[0].shape[0] == x.shape[0] for x in leaves[1:]):
                raise ValueError(
                    f"Expected all leaves to have same batch dim: {utils.shapes(example)}"
                )
            self._current_batch = (leaves, structure)

        leaves, structure = self._current_batch
        assert len(leaves) > 0, self._current_batch
        batch_size = leaves[0].shape[0]  # All leaves have same batch size due to check above.
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
    def __str__(self) -> str:
        return "UnbatchIterDataset"

    def __iter__(self) -> _UnbatchDatasetIterator:
        return _UnbatchDatasetIterator(self._parent.__iter__())


def unbatch(ds: Dataset) -> Dataset:
    """Similar to `input_tf_data.unbatch`.

    Unlike `batch`, which naively groups top-level elements, unbatch applies to JAX leaves only.
    For example, lists are not considered for batch dim, but rather as part of tree structure.

    Unlike grain's `flat_map`, there is no limit on the fan out.

    Args:
        ds: A Dataset where each example has leaves with the same batch dim.

    Returns:
        A Dataset with unbatched inputs.
    """
    _ensure_iter_dataset(ds)
    return _UnbatchIterDataset(ds)


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


def trim_and_pack_dataset(ds: Dataset, *, feature_lengths: utils.Nested[int]) -> Dataset:
    """Similar to `seqio.trim_and_pack_dataset`.

    Different from `seqio.trim_and_pack_dataset`, elements may be packed out of order if doing so
    produces less padding. Further, elements may be truncated (with remainder dropped). See
    `SingleBinPackIterDataset` in `grain` or test cases for details.

    Args:
        ds: A Dataset containing keys in `feature_lengths`.
        feature_lengths: A (nested) mapping from of feature key to target length.
            Packing will happen across 0th dimension. Features must be array-like.

    Returns:
        A Dataset with packed features. Similar to `seqio.trim_and_pack_dataset`, packing introduces
        additional fields for each feature:
        - `{feature}_segment_ids`: segment IDs for each packed example, where 0's represent padding;
        - `{feature}_positions`: positions for each segment, where 0's represent padding.
    """
    _ensure_iter_dataset(ds)
    return packing.SingleBinPackIterDataset(parent=ds, length_struct=feature_lengths)


def shard_dataset(
    ds: Dataset,
    *,
    process_count: Optional[int] = None,
    process_index: Optional[int] = None,
) -> Dataset:
    """A convenience wrapper around `ds.slice`.

    Specifically, each process reads `ds[process_index::process_count]`.
    E.g., if the dataset has 10 elements split among 4 processes, process 0,1 get 3 each, and
    process 2,3 get 2 each.

    Args:
        ds: A Dataset.
        process_count: Number of processes. If None, infers from `jax.process_count()`.
        process_index: Process index. If None, infers from `jax.process_index()`.

    Returns:
        A sharded (sliced) dataset.
    """
    if process_count is None:
        process_count = jax.process_count()
    if process_index is None:
        process_index = jax.process_index()
    if not 0 <= process_index < process_count:
        raise ValueError(f"{process_index=} should be between 0 and {process_count=}, exclusive.")
    return ds.slice(slice(process_index, None, process_count))


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


class Input(Module):
    """A Module to generate input batches with `grain`."""

    @config_class
    class Config(Module.Config):
        """Configures Input.

        Attributes:
            source: A `BuildDatasetFn` producing the source dataset. Can be a `grain.MapDataset` or
                `grain.IterDataset`.
        """

        source: Required[ConfigOr[BuildDatasetFn]] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._source = maybe_instantiate(cfg.source)

    @property
    def source(self) -> BuildDatasetFn:
        return self._source

    def dataset(self) -> grain.IterDataset:
        return maybe_to_iter_dataset(self._source())

    def __iter__(self) -> grain.PyGrainDatasetIterator[utils.NestedTensor]:
        return iter(self.dataset())

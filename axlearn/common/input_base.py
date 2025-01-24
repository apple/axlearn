# Copyright © 2024 Apple Inc.

"""Base Input interface."""

import re
from typing import Iterable, Iterator, NamedTuple, Optional, Protocol, Sequence, Union

import jax
from absl import logging
from jax._src.mesh import thread_resources
from jax.sharding import PartitionSpec

from axlearn.common.config import ConfigOr, config_class, maybe_instantiate
from axlearn.common.input_dispatch import InputDispatcher
from axlearn.common.module import Module
from axlearn.common.utils import (
    Nested,
    Tensor,
    as_numpy_array,
    dispatch_input_batch,
    tree_paths,
    with_sharding_constraint,
)


class InputPartitionFn(Protocol):
    """Partitions the input batch."""

    def __call__(self, input_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Applies sharding constraints to `input_batch` and returns the modified batch.

        Implementations should avoid making in-place updates to `input_batch`.
        """


class PathAndRank(NamedTuple):
    """A tuple (path, rank) used for matching against inputs in a batch.

    Attributes:
        path: An optional path or path regex. None means match everything.
        rank: An optional rank (ndim). None means match everything.
    """

    path: Optional[Union[str, re.Pattern]]
    rank: Optional[int]


def partition_by_path_rank(
    path_rank_to_partition: dict[PathAndRank, PartitionSpec],
) -> InputPartitionFn:
    """Partitions the keys in the input batch by Tensor path and rank (ndim).

    If not within a mesh, the partition fn is a no-op.

    Args:
        path_rank_to_partition: A mapping from (path_regex, rank) to partition spec.
            For each input path, the Tensor will be constrained by the first matching
            (path_regex, rank) rule, where paths are full-matched against `path_regex` and ranks are
            matched against `rank`.
            `path_regex` or `rank` are allowed to be None to match everything.
            If replication is desired, specify a partition spec of None explicitly.
            If leaving the input unconstrained is desired, specify a partition spec of
            `PartitionSpec.UNCONSTRAINED` explicitly.

    Returns:
        A function that applies sharding constraints to an input batch and returns a new batch.

    Raises:
        ValueError: If no rules match for a given input, which is likely an oversight. If leaving
            inputs unconstrained is desired, explicitly specify `PartitionSpec.UNCONSTRAINED`.

    Example:
        To constrain all rank-1 Tensors by ("data",) and rank-2 by ("data", "seq"):
        ```
        partition_by_path_ndim({
            (".*", 1): PartitionSpec("data"),
            (".*", 2): PartitionSpec("data", "seq"),
        })
        ```
    """
    compiled = {}
    for (regex, rank), spec in path_rank_to_partition.items():
        if regex is not None:
            regex = re.compile(regex)
        compiled[(regex, rank)] = spec

    def fn(input_batch: Nested[Tensor]) -> Nested[Tensor]:
        mesh = thread_resources.env.physical_mesh  # type: ignore
        if mesh.empty or mesh.size == 1:
            return input_batch

        def maybe_constrain(path: str, value: Tensor):
            for (path_regex, rank), partition_spec in compiled.items():
                if not (rank is None or value.ndim == rank) or not (
                    path_regex is None or re.fullmatch(path_regex, path)
                ):
                    continue
                if partition_spec is not PartitionSpec.UNCONSTRAINED:
                    value = with_sharding_constraint(value, partition_spec)
                    logging.log_first_n(
                        logging.INFO,
                        "Constraining input_batch[%s] with %s.",
                        len(input_batch),
                        path,
                        partition_spec,
                    )
                return value
            # No rules match. We raise as not-constraining is likely an oversight.
            raise ValueError(
                f"No rules matched input_batch['{path}']. "
                "If you intended to leave the input unconstrained, "
                "specify `PartitionSpec.UNCONSTRAINED` explicitly."
            )

        return jax.tree_map(maybe_constrain, tree_paths(input_batch), input_batch)

    return fn


class Input(Module):
    """A Module to generate input batches.

    Subclasses typically only need to implement the `dataset` method. See `input_tf_data.Input` and
    `input_grain.Input` for example implementations.

    The typical usage within a trainer is:
    1. Construct an iterator using `iter(input.dataset())`.
    2. Iterate over per-feed physical batches using `batches(iterator)`.
    3. Use `host_to_global_device_array` to construct a global physical batch.
    4. Use `dispatch_global_batch` within pjit to construct a global logical batch.

    Example:
        ```
        input = Input.default_config().set(...).instantiate(parent=None)
        input_iter = iter(input.dataset())  # Construct an iterator (used e.g. for checkpointing).

        def train_step(global_physical_batch):
            global_logical_batch = input.dispatch_global_batch(global_physical_batch)
            ...

        for per_feed_physical_batch in input.batches(input_iter):
            global_physical_batch = host_to_global_device_array(per_feed_physical_batch)
            ... = pjit(train_step)(global_physical_batch)
        ```
    """

    @config_class
    class Config(Module.Config):
        """Configures Input.

        Attributes:
            input_dispatcher: If not None, creates an InputDispatcher and uses it for dispatching
                per-feed batches to global batches.
            input_partitioner: If not None, applies additional sharding constraints on each input
                batch during `dispatch_global_batch`.
        """

        input_dispatcher: Optional[InputDispatcher.Config] = None
        input_partitioner: Optional[ConfigOr[InputPartitionFn]] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.input_dispatcher is not None:
            self.input_dispatcher: InputDispatcher = (  # pytype: disable=annotation-type-mismatch
                self._add_child("input_dispatcher", cfg.input_dispatcher)
            )
        self._input_partitioner: Optional[InputPartitionFn] = maybe_instantiate(
            cfg.input_partitioner
        )

    def dataset(self) -> Iterable[Nested[Tensor]]:
        """Returns the input dataset, which should produce per-feed logical batches.

        Each batch is a pytree of arrays which reside on host memory (i.e., leaves can be any array
        type which can be converted to numpy via `as_numpy_array`).

        The dataset should be iterable, i.e., it is expected to support conversion to an iterator
        via `iter(...)`. Although not strictly required, it is recommended for the iterator to be
        checkpointable.
        """
        raise NotImplementedError(type(self))

    def __iter__(self) -> Iterator[Nested[Tensor]]:
        """Iterates over the input dataset.

        The iterator should produce per-feed physical batches (by iterating over the iterable
        returned by `dataset` using `batches()`).

        To obtain a checkpointable iterator, use `iter(dataset())` directly.
        """
        yield from self.batches(iter(self.dataset()))

    def batches(self, it: Iterator[Nested[Tensor]]) -> Iterator[Nested[Tensor]]:
        """Yields per-feed physical input batches (using `input_dispatcher` if configured).

        The caller should use `host_to_global_array` to construct a global physical batch from the
        per-feed physical batches returned from this method.

        See also `dispatch_global_batch` for constructing a global logical batch.
        """
        # Validate that input batches have the proper feed batch size if this is a logical feed.
        should_validate = (
            "input_dispatcher" in self.children
            and self.input_dispatcher.logical_feed_index is not None
        )
        for input_batch in it:
            input_batch = as_numpy_array(input_batch)
            # For the first batch, validate that the per_feed_batch_size is configured properly.
            if should_validate:

                def check_per_feed_batch(x: Tensor):
                    expected = self.input_dispatcher.feed_logical_batch_size
                    actual = x.shape[0]
                    if expected != actual:
                        raise ValueError(
                            f"Expected per-feed batch size to be {expected}, got: {actual}"
                        )

                jax.tree.map(check_per_feed_batch, input_batch)
                should_validate = False

            if "input_dispatcher" in self.children:
                input_batch = self.input_dispatcher.logical_to_physical_batch(input_batch)
            yield input_batch

    def dispatch_global_batch(
        self,
        global_physical_batch: Nested[Tensor],
        *,
        batch_axis_names: Union[str, Sequence[str]] = "data",
    ) -> Nested[Tensor]:
        """Converts a global physical batch to a global logical batch.

        The leaves of the output logical batch are partitioned across `batch_axis_names` along the
        0th (batch) dimension. This should be invoked from within `pjit` so that the sharding
        constraints can be applied.

        If `cfg.input_partitioner` is not None, it will be applied to each logical batch after
        constraining `batch_axis_names`.
        """

        def constrain_batch_axis(batch):
            return jax.tree.map(
                lambda x: with_sharding_constraint(x, PartitionSpec(batch_axis_names)),
                batch,
            )

        if "input_dispatcher" in self.children:
            global_logical_batch = self.input_dispatcher.physical_to_logical_batch(
                constrain_batch_axis(global_physical_batch)
            )
        else:
            global_logical_batch = dispatch_input_batch(
                global_physical_batch, batch_axis_names=batch_axis_names
            )

        global_logical_batch = constrain_batch_axis(global_logical_batch)

        # Further constrain based on user-configured partitioning rules.
        if self._input_partitioner is not None:
            global_logical_batch = self._input_partitioner(global_logical_batch)

        return global_logical_batch

    def element_spec(self) -> Nested[jax.ShapeDtypeStruct]:
        """Returns the per-feed logical batch spec.

        This is used e.g. for AOT compilation and is not strictly required for training.
        """
        raise NotImplementedError(type(self))

# Copyright © 2024 Apple Inc.

"""Base Input interface."""

from typing import Iterable, Iterator, Optional, Sequence, Union

import jax
from jax.sharding import PartitionSpec

from axlearn.common.config import config_class
from axlearn.common.input_dispatch import InputDispatcher
from axlearn.common.module import Module
from axlearn.common.utils import (
    Nested,
    Tensor,
    as_numpy_array,
    dispatch_input_batch,
    with_sharding_constraint,
)


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
        """

        input_dispatcher: Optional[InputDispatcher.Config] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.input_dispatcher is not None:
            self.input_dispatcher: InputDispatcher = (  # pytype: disable=annotation-type-mismatch
                self._add_child("input_dispatcher", cfg.input_dispatcher)
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
        return constrain_batch_axis(global_logical_batch)

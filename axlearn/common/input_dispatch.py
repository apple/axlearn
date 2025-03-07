# Copyright © 2024 Apple Inc.

"""Utility to help dispatching input batches from hosts to devices."""

import copy
import math
from collections.abc import Sequence
from functools import cached_property
from typing import Optional

import jax
import numpy as np
from jax import numpy as jnp
from jax._src.mesh import thread_resources
from jax._src.sharding_impls import get_process_index_and_count
from jax.sharding import PartitionSpec

from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module
from axlearn.common.utils import (
    PHYSICAL_TO_LOGICAL_DISPATCH_KEY,
    Nested,
    Tensor,
    input_partition_spec,
)


class BaseInputDispatcher(Module):
    """Base input dispatch interface."""

    @config_class
    class Config(Module.Config):
        """Configuration for BaseInputDispatcher."""

        global_logical_batch_size: Required[int] = REQUIRED

    @property
    def num_logical_feeds(self) -> int:
        raise NotImplementedError(type(self))

    @property
    def logical_feed_index(self) -> int:
        raise NotImplementedError(type(self))

    @property
    def feed_logical_batch_size(self) -> int:
        raise NotImplementedError(type(self))

    @property
    def partition_spec(self) -> PartitionSpec:
        raise NotImplementedError(type(self))

    def feed_read_config(self) -> dict[str, int]:
        raise NotImplementedError(type(self))

    def logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor]) -> Nested[Tensor]:
        raise NotImplementedError(type(self))

    def physical_to_logical_batch(self, global_physical_batch: Nested[Tensor]) -> Nested[Tensor]:
        raise NotImplementedError(type(self))


class InputDispatcher(BaseInputDispatcher):
    """A Module to dispatch per-feed logical input batches to global logical batches on device.

    The dispatch process consists of three steps:
    - Convert each logical feed batch to a physical feed batch (logical_to_physical_batch);
    - Assemble a global physical batch from per-feed batches (utils.host_to_global_device_array);
    - Convert a global physical batch to a global logical batch (physical_to_logical_batch).

    This process is needed because utils.host_to_global_device_array requires that global batch
    size be divisible by number of devices.

    One should set up the local input generator to read the logical shard as specified by
    `feed_read_config` and batch the examples by `feed_logical_batch_size`.
    One should then call `logical_to_physical_batch` on each per-feed batch, followed by
    `utils.host_to_global_device_array` to generate the input array for pjit, then finally
    `physical_to_logical_batch` inside pjit.
    """

    @config_class
    class Config(BaseInputDispatcher.Config):
        """Configuration for InputDispatcher."""

        # Usually left unset. Defaults to
        # max(feed_logical_batch_size * num_physical_feeds, jax.device_count()).
        global_physical_batch_size: Optional[int] = None

        # The total number of physical feeds across all hosts. Defaults to jax.process_count().
        num_physical_feeds: Optional[int] = None

        # The local physical feed index. Must be in [0, num_physical_feeds).
        # Defaults to jax.process_index().
        physical_feed_index: Optional[int] = None

        # Usually left unset.
        # If not None, a list of length num_logical_feeds. logical_feed_indices[i] is an integer in
        # [0, num_physical_feeds), representing the physical feed index for the i'th logical feed.
        logical_feed_indices: Optional[Sequence[int]] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        cfg = cfg.clone()
        cfg.num_physical_feeds = cfg.num_physical_feeds or jax.process_count()
        cfg.physical_feed_index = cfg.physical_feed_index or jax.process_index()
        if cfg.logical_feed_indices is None:
            num_logical_feeds = min(cfg.global_logical_batch_size, cfg.num_physical_feeds)
            cfg.logical_feed_indices = list(range(num_logical_feeds))
        if cfg.global_physical_batch_size is None:
            num_logical_feeds = len(cfg.logical_feed_indices)
            feed_logical_batch_size = cfg.global_logical_batch_size // num_logical_feeds
            cfg.global_physical_batch_size = max(
                feed_logical_batch_size * cfg.num_physical_feeds, jax.device_count()
            )
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.global_logical_batch_size % self.num_logical_feeds != 0:
            raise ValueError(
                f"{cfg.global_logical_batch_size=} must be divisible by {self.num_logical_feeds=}"
            )
        if cfg.global_physical_batch_size % cfg.num_physical_feeds != 0:
            raise ValueError(
                f"{cfg.global_physical_batch_size=} must be divisible by {cfg.num_physical_feeds=}"
            )
        if self.feed_physical_batch_size < self.feed_logical_batch_size:
            raise ValueError(
                f"{self.feed_physical_batch_size=} must be >= {self.feed_logical_batch_size=}"
            )
        if not 0 <= cfg.physical_feed_index < cfg.num_physical_feeds:
            raise ValueError(
                f"{cfg.physical_feed_index=} must be in range [0, {cfg.num_physical_feeds})"
            )
        if not all(0 <= ix < cfg.num_physical_feeds for ix in cfg.logical_feed_indices):
            raise ValueError(
                f"Invalid physical feed index in {cfg.logical_feed_indices}: must be "
                f"in range [0, {cfg.num_physical_feeds})"
            )
        if len(set(cfg.logical_feed_indices)) != len(cfg.logical_feed_indices):
            raise ValueError(f"{cfg.logical_feed_indices=} must be unique")

    @property
    def num_logical_feeds(self) -> int:
        return len(self.config.logical_feed_indices)

    @property
    def logical_feed_index(self) -> Optional[int]:
        cfg = self.config
        if cfg.physical_feed_index in cfg.logical_feed_indices:
            return cfg.logical_feed_indices.index(cfg.physical_feed_index)
        return None

    @property
    def feed_logical_batch_size(self) -> int:
        return self.config.global_logical_batch_size // self.num_logical_feeds

    @property
    def feed_physical_batch_size(self) -> int:
        cfg = self.config
        return cfg.global_physical_batch_size // cfg.num_physical_feeds

    @property
    def partition_spec(self) -> PartitionSpec:
        return input_partition_spec()

    def feed_read_config(self) -> dict[str, int]:
        """Generates the read configuration for the local physical feed.

        Returns:
            A dict containing:
            - "num_shards": The total number of logical shards to split the source by;
            - "shard_index": The logical shard index to read for the local physical feed.
        """
        cfg = self.config
        # Set the read config to draw unique data for num logical feed indices only.
        num_shards = self.num_logical_feeds
        shard_index = self.logical_feed_index
        if shard_index is None:
            # Read an arbitrary shard. It doesn't matter which shard we read, since the results are
            # not included by the result of `physical_to_logical_batch`.
            # Here we try to distribute the logical shards evenly.
            non_logical_feed_indices = [
                ix for ix in range(cfg.num_physical_feeds) if ix not in cfg.logical_feed_indices
            ]
            assert len(non_logical_feed_indices) == cfg.num_physical_feeds - self.num_logical_feeds
            shard_index = non_logical_feed_indices.index(cfg.physical_feed_index) % num_shards
        return dict(num_shards=num_shards, shard_index=shard_index)

    def logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Converts a per-feed logical batch to a per-feed physical batch.

        Specifically, pads the batch to feed_physical_batch_size and adds a dispatch Tensor under
        key PHYSICAL_TO_LOGICAL_DISPATCH_KEY, which will be used by physical_to_logical_batch later.

        Note that the processing in `logical_to_physical_batch` is entirely host-local, i.e.
        operating on pure numpy arrays rather than JAX arrays.

        Args:
            logical_feed_batch: A per-feed logical batch, where every leaf Tensor should be of
                shape [feed_logical_batch_size, ...].

        Returns:
            A per-feed physical batch, where every leaf Tensor should be of shape
            [feed_physical_batch_size, ...].
        """
        cfg: InputDispatcher.Config = self.config
        if (
            cfg.global_logical_batch_size == cfg.global_physical_batch_size
            and cfg.num_physical_feeds == self.num_logical_feeds
        ):
            return copy.deepcopy(logical_feed_batch)
        feed_physical_batch_size = self.feed_physical_batch_size
        feed_logical_batch_size = self.feed_logical_batch_size

        def pad_to_physical_batch_size(x: Tensor):
            if x.ndim < 1 or x.shape[0] != feed_logical_batch_size:
                raise NotImplementedError(
                    "Shape does not match logical batch size: "
                    f"{x.shape} vs. {feed_logical_batch_size}"
                )
            if cfg.physical_feed_index not in cfg.logical_feed_indices:
                x = np.zeros_like(x)
            if feed_logical_batch_size == feed_physical_batch_size:
                return x
            pad_size = feed_physical_batch_size - feed_logical_batch_size
            assert pad_size >= 0, f"{feed_physical_batch_size} < {feed_logical_batch_size}"
            if not jnp.isdtype(x.dtype, ("numeric", "bool")):
                raise NotImplementedError(f"dtype {x.dtype} is not supported")
            padding = np.zeros([pad_size] + list(x.shape[1:]), dtype=x.dtype)
            return np.concatenate([x, padding], axis=0)

        physical_feed_batch = jax.tree.map(pad_to_physical_batch_size, logical_feed_batch)

        if cfg.physical_feed_index not in cfg.logical_feed_indices:
            # Dispatch matrix is all 0's.
            dispatch = np.zeros(
                [feed_physical_batch_size, cfg.global_logical_batch_size], dtype=bool
            )
        else:
            dispatch_start_ix = self.logical_feed_index * feed_logical_batch_size
            # dispatch_start_ix + [0, feed_logical_batch_size).
            logical_example_indices = dispatch_start_ix + np.arange(feed_logical_batch_size)

            # Construct a one-hot dispatch matrix.
            dispatch = np.zeros(
                [feed_logical_batch_size, cfg.global_logical_batch_size], dtype=bool
            )
            dispatch[np.arange(feed_logical_batch_size), logical_example_indices] = True

            if feed_logical_batch_size < feed_physical_batch_size:
                # Pad dispatch matrix with 0's.
                dispatch = np.pad(
                    dispatch,
                    ((0, feed_physical_batch_size - feed_logical_batch_size), (0, 0)),
                    constant_values=False,
                )

        assert dispatch.shape == (
            feed_physical_batch_size,
            cfg.global_logical_batch_size,
        ), f"{dispatch.shape} vs. {(feed_physical_batch_size, cfg.global_logical_batch_size)}"

        if PHYSICAL_TO_LOGICAL_DISPATCH_KEY in physical_feed_batch:
            raise ValueError(f"{PHYSICAL_TO_LOGICAL_DISPATCH_KEY} already exists")
        physical_feed_batch[PHYSICAL_TO_LOGICAL_DISPATCH_KEY] = dispatch
        return physical_feed_batch

    def physical_to_logical_batch(self, global_physical_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Converts a global physical batch to a global logical batch.

        Args:
            global_physical_batch: A global physical batch, where every leaf Tensor should be of
                shape [global_physical_batch_size, ...].

        Returns:
            A global logical batch, where every leaf Tensor should be of shape
            [global_logical_batch_size, ...].
        """
        cfg = self.config

        def traverse_and_dispatch(data: Nested[Tensor]) -> Nested[Tensor]:
            if isinstance(data, dict):
                # Dispatch from physical batch dimensions to logical batch.
                if PHYSICAL_TO_LOGICAL_DISPATCH_KEY in data:
                    dispatch: Tensor = data.pop(PHYSICAL_TO_LOGICAL_DISPATCH_KEY)
                    assert dispatch.shape == (
                        cfg.global_physical_batch_size,
                        cfg.global_logical_batch_size,
                    ), (
                        f"{dispatch.shape} vs. "
                        f"{(cfg.global_physical_batch_size, cfg.global_logical_batch_size)}"
                    )
                    return jax.tree.map(lambda x: jnp.einsum("p...,pl->l...", x, dispatch), data)
                for key, value in data.items():
                    data[key] = traverse_and_dispatch(value)
            return data

        return traverse_and_dispatch(global_physical_batch)


class SpmdInputDispatcher(BaseInputDispatcher):
    """A variant of InputDispatcher which is mesh/topology aware.

    Specifically, given a global shape and input shardings, we infer the layout of the processes
    (input feeds) across the mesh, and assign each process a corresponding feed index and
    process-local shape that the process should produce.

    These process-local inputs can then be directly used to assemble a global array under the target
    sharding without a separate dispatch step.
    """

    @config_class
    class Config(BaseInputDispatcher.Config):
        """Configuration for SpmdInputDispatcher.

        Attributes:
            partition_spec: A PyTree specifying how inputs should be partitioned.
        """

        partition_spec: Required[PartitionSpec] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: SpmdInputDispatcher.Config = self.config

        mesh = thread_resources.env.physical_mesh
        if mesh.empty:
            raise ValueError("Expected to be initialized within the context of a mesh.")

        if not cfg.partition_spec:
            raise ValueError(f"{cfg.partition_spec=} cannot be empty.")

        # TODO(markblee): For simplicity, we currently restrict to batch-only partitioning, since
        # input implementations currently do not support other configurations. Specifically, we can
        # extend `feed_read_config` to return not just an index but a tuple of indices indicating
        # the position of the feed along dims != 0.
        if any(spec is not None for spec in jax.tree.leaves(cfg.partition_spec[1:])):
            raise NotImplementedError(
                "Partitioning along non-batch dims is currently not supported by input dispatch: "
                f"{cfg.partition_spec}"
            )

        self._partition_spec = cfg.partition_spec
        logical_sharding = jax.NamedSharding(mesh, cfg.partition_spec)

        # Validate that batch partitioning is consistent with logical batch size.
        num_partitions = math.prod(
            mesh.shape[axis] for axis in jax.tree.leaves(logical_sharding.spec[0])
        )
        if cfg.global_logical_batch_size % num_partitions != 0:
            raise ValueError(
                f"{cfg.partition_spec=} attempts to divide batch over {num_partitions=}, "
                f"which is incompatible with {cfg.global_logical_batch_size=}."
            )

        # Infer the physical feeds and feed index along dim=0.
        feed_index, feed_count = get_process_index_and_count(
            logical_sharding, dim=0, ndims=len(mesh.shape)
        )
        self._num_logical_feeds = feed_count
        self._logical_feed_index = feed_index

    @property
    def num_logical_feeds(self) -> int:
        return self._num_logical_feeds

    @property
    def logical_feed_index(self) -> int:
        return self._logical_feed_index

    @cached_property
    def feed_logical_batch_size(self) -> int:
        cfg: SpmdInputDispatcher.Config = self.config
        return cfg.global_logical_batch_size // self._num_logical_feeds

    @property
    def partition_spec(self) -> PartitionSpec:
        return self._partition_spec

    def feed_read_config(self) -> dict[str, int]:
        return dict(num_shards=self._num_logical_feeds, shard_index=self._logical_feed_index)

    def logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor]) -> Nested[Tensor]:
        return jax.tree.map(lambda x: x, logical_feed_batch)

    def physical_to_logical_batch(self, global_physical_batch: Nested[Tensor]) -> Nested[Tensor]:
        return jax.tree.map(lambda x: x, global_physical_batch)

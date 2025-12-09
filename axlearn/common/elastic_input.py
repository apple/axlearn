# Copyright © 2025 Apple Inc.

"""
Support Elastic Training

The motivation is to support elastic training when scaling down from `N` slices
to `N-K` without changing the global batch size. In elastic mode, the data from
the original `K` slices are then dispatched to the healthy `N-K` slices. Empty
paddings are added when necessary.

Two main public APIs provided by this module are `ElasticInput` and
`ElasticSpmdInputDispatcher`.

An example usage is:

``` ElasticInput.default_config().set(
    input=Input.default_config().set(
        source=config_for_function(tfds_dataset).set(
            dataset_name=fake_dataset_name, split="train", is_training=True,
            data_dir=tmp_data_dir,
        ), input_dispatcher=ElasticSpmdInputDispatcher.default_config().set(
            num_max_slices=num_processes,
            global_logical_batch_size=global_batch_size,
        ), processor=config_for_function(identity),
        batcher=config_for_function(per_feed_batch).set(
            feed_batch_size=num_samples_per_feed, is_training=True,
            pad_example_fn=default_pad_example_fn,
        ), is_training=True,
    ),
)
```

Note the loss might be changed slightly (within numerical tolerance) because the
order of the global input batch is changed.
"""

import collections
import math
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import jax
import numpy as np
from jax._src import sharding as jsharding
from jax._src.mesh import thread_resources
from jax.sharding import PartitionSpec

from axlearn.common import input_base
from axlearn.common.config import REQUIRED, Required, config_class, maybe_set_config
from axlearn.common.input_dispatch import BaseInputDispatcher, _validate_logical_feed_shapes
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor


class ElasticSpmdInputDispatcher(BaseInputDispatcher):
    """Similar to SpmdInputDispatcher, but to support elastic training"""

    @config_class
    class Config(BaseInputDispatcher.Config):
        """Config for ElasticSpmdInputDispatcher"""

        partition_spec: Required[PartitionSpec] = REQUIRED

        # Currently we only support scaling down. So the value is the maximum
        # number of slices that the job will use during the whole life circle.
        num_max_slices: Required[int] = REQUIRED

        # If `False`, `feed_read_config` will return the feed index of current
        # process, otherwise the corresponding elastic feed index of current
        # process is returned instead.
        is_read_elastic_feed: bool = False

    @property
    def is_in_elastic_mode(self) -> bool:
        cfg = self.config
        if cfg.num_max_slices is None:
            return False
        else:
            if slice_count() < cfg.num_max_slices:
                return True
            elif slice_count() == cfg.num_max_slices:
                return False
            else:
                # TODO (jtian22): consider supporting scaling up in the future.
                raise ValueError(
                    f"The number of slices at runtime[{slice_count()}] is larger"
                    f"than the configured num_max_slices[{cfg.num_max_slices}]!"
                )

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: ElasticSpmdInputDispatcher.Config = self.config

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
        if self.is_in_elastic_mode:
            num_partitions = num_partitions // slice_count() * cfg.num_max_slices

        if cfg.global_logical_batch_size % num_partitions != 0:
            raise ValueError(
                f"{cfg.partition_spec=} attempts to divide batch over {num_partitions=}, "
                f"which is incompatible with {cfg.global_logical_batch_size=}."
            )

        self._device_physical_batch_size = cfg.global_logical_batch_size // num_partitions

        # Infer the physical feeds and feed index along dim=0.
        _, _, pid2fid = get_process_index_and_count_and_mapping(
            logical_sharding, dim=0, ndims=len(mesh.shape)
        )

        def fid2pids(feed_id):
            pids_found = [p for p, f in pid2fid.items() if feed_id == f]

            if pids_found:
                return pids_found

            # In this case, the mapping between process id and feed id in those
            # unhealthy slices are unknown. However, since the device mesh
            # across slices are created the same, we can use the mapping in
            # slice 0 to infer them. Now we take feed_id of 0 as a reference
            # point
            n_feeds_per_slice = len(set(pid2fid.values())) // slice_count()
            n_processes_per_slice = process_count_per_slice()

            fid_shift = feed_id % n_feeds_per_slice
            reference_pids = [pid for pid, fid in pid2fid.items() if fid == fid_shift]
            reference_pid_shifts = [pid % n_processes_per_slice for pid in reference_pids]
            sid = feed_id // n_feeds_per_slice

            inferred_pids = [sid * n_processes_per_slice + shift for shift in reference_pid_shifts]

            return inferred_pids

        self.feed_count = len(set(pid2fid.values())) // slice_count() * cfg.num_max_slices
        self.feed_index = pid2fid[jax.process_index()]

        assert cfg.global_logical_batch_size % self.feed_count == 0
        self._feed_logical_batch_size = cfg.global_logical_batch_size // self.feed_count

        if self.is_in_elastic_mode:
            adjusted_device_physical_batch_size = math.ceil(
                self._device_physical_batch_size * (cfg.num_max_slices / slice_count())
            )
            padding_per_device = (
                adjusted_device_physical_batch_size - self._device_physical_batch_size
            )
            num_mini_batches = math.ceil(self._device_physical_batch_size / padding_per_device)
            self.elastic_feed_mini_batch_index = self.feed_index % num_mini_batches

            # Each batch from the elastic feed will be split into multiple
            # mini-batches Note that we can't divide `_feed_logical_batch_size`
            # by `num_mini_batches`, because we want to ensure this value is
            # dividable by `padding_per_device`
            self.elastic_feed_mini_batch_size = (
                self._feed_logical_batch_size
                * padding_per_device
                // self._device_physical_batch_size
            )

            elastic_feed_start = self.feed_count // cfg.num_max_slices * slice_count()
            elastic_feed_shift = self.feed_index // num_mini_batches

            # Even in elastic mode, some processes may not be assigned the
            # elastic feed. See test case in `elastic_input_test.py` for
            # example.
            self.elastic_feed_index = elastic_feed_start + elastic_feed_shift
            if self.elastic_feed_index >= self.feed_count:
                self.elastic_feed_index = None

            self.elastic_process_ids = (
                None if self.elastic_feed_index is None else fid2pids(self.elastic_feed_index)
            )

            pids_of_current_feed = fid2pids(self.feed_index)
            self.is_primary = (
                jax.process_index() == pids_of_current_feed[0]
                and self.elastic_feed_mini_batch_index == 0
            )

    @property
    def num_logical_feeds(self) -> int:
        return self.feed_count

    @property
    def logical_feed_index(self) -> int:
        return self.feed_index

    @property
    def feed_logical_batch_size(self) -> int:
        return self._feed_logical_batch_size

    @property
    def partition_spec(self) -> PartitionSpec:
        cfg: ElasticSpmdInputDispatcher.Config = self.config
        return cfg.partition_spec

    @property
    def device_physical_batch_size(self) -> int:
        return self._device_physical_batch_size

    def feed_read_config(self) -> dict[str, int]:
        cfg: ElasticSpmdInputDispatcher.Config = self.config
        if cfg.is_read_elastic_feed:
            return {
                "num_shards": self.num_logical_feeds,
                "shard_index": (
                    self.feed_index if self.elastic_feed_index is None else self.elastic_feed_index
                ),
            }
        else:
            return {"num_shards": self.num_logical_feeds, "shard_index": self.feed_index}

    def logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor]) -> Nested[Tensor]:
        return jax.tree.map(lambda x: x, logical_feed_batch)

    def physical_to_logical_batch(self, global_physical_batch: Nested[Tensor]) -> Nested[Tensor]:
        return jax.tree.map(lambda x: x, global_physical_batch)

    def logical_to_physical_shapes(
        self, logical_feed_shapes: Nested[jax.ShapeDtypeStruct]
    ) -> Nested[jax.ShapeDtypeStruct]:
        """Maps per-feed logical shapes to per-feed physical shapes for AOT compilation."""
        _validate_logical_feed_shapes(logical_feed_shapes)
        # Ensure that we always return ShapeDtypeStructs.
        return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), logical_feed_shapes)


@dataclass
class ElasticDatasetIterator:
    """Iterator for ElasticDataset"""

    primary_iterator: Iterator
    elastic_iterator: Optional[Iterator]

    elastic_process_ids: list[int]
    is_primary_for_checkpoint: bool

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.primary_iterator), (
            None if self.elastic_iterator is None else next(self.elastic_iterator)
        )


@dataclass
class ElasticDataset:
    """A composition of dataset from current process and processes of the elastic feed"""

    primary_dataset: Iterable

    # Dataset from the elastic feed
    elastic_dataset: Optional[Iterable]

    # The process ids which share the same elastic feed id
    elastic_process_ids: list[int]

    # Whether current process is responsible for checkpointing input iterator,
    # among those processes who share the same elastic feed index share the same
    # elastic feed index
    is_primary_for_checkpoint: bool

    def __iter__(self):
        return ElasticDatasetIterator(
            primary_iterator=iter(self.primary_dataset),
            elastic_iterator=None if self.elastic_dataset is None else iter(self.elastic_dataset),
            elastic_process_ids=self.elastic_process_ids,
            is_primary_for_checkpoint=self.is_primary_for_checkpoint,
        )


class ElasticInput(input_base.Input):
    """A general wrapper for Input to support elastic training"""

    @config_class
    class Config(input_base.Input.Config):
        """Configures ElasticInput."""

        input: Required[input_base.Input.Config] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        assert (
            isinstance(cfg.input.input_dispatcher, ElasticSpmdInputDispatcher.Config)
            and cfg.input.input_dispatcher.is_read_elastic_feed is False
        )

        self.primary_input = self._add_child("primary_input", cfg.input)

        if self.is_in_elastic_mode:
            self.elastic_input = self._add_child(
                "elastic_input",
                maybe_set_config(
                    cfg.input,
                    input_dispatcher=maybe_set_config(
                        cfg.input.input_dispatcher, is_read_elastic_feed=True
                    ),
                ),
            )

    @property
    def is_in_elastic_mode(self) -> bool:
        return self.primary_input.input_dispatcher.is_in_elastic_mode

    def dataset(self):
        if self.is_in_elastic_mode:
            return ElasticDataset(
                primary_dataset=self.primary_input.dataset(),
                elastic_dataset=(
                    None
                    if self.elastic_input.input_dispatcher.elastic_feed_index is None
                    else self.elastic_input.dataset()
                ),
                elastic_process_ids=self.elastic_input.input_dispatcher.elastic_process_ids,
                is_primary_for_checkpoint=self.elastic_input.input_dispatcher.is_primary,
            )
        else:
            return ElasticDataset(
                primary_dataset=self.primary_input.dataset(),
                elastic_dataset=None,
                elastic_process_ids=[],
                is_primary_for_checkpoint=True,
            )

    def batches(self, it: Iterator[Nested[Tensor]]) -> Iterator[Nested[Tensor]]:
        assert isinstance(it, ElasticDatasetIterator)
        if self.is_in_elastic_mode:
            dispatcher: ElasticSpmdInputDispatcher = self.elastic_input.input_dispatcher

            def _padded_select(path, x, y):
                start = (
                    dispatcher.elastic_feed_mini_batch_index
                    * dispatcher.elastic_feed_mini_batch_size
                )
                stop = start + dispatcher.elastic_feed_mini_batch_size

                n_pad = stop - y.shape[0]
                if n_pad > 0:
                    # Note that the batch size might not be dividable of the
                    # padding size For example, a batch size of 5 might be split
                    # into 3 groups each of size 2, 2, 1. To ensure that each
                    # padding share the same size, we need to expand the batch
                    # from 5 to 6.
                    if jax.tree_util.keystr(path, simple=True) == "target_labels":
                        y = np.pad(
                            y,
                            [(0, n_pad)] + [(0, 0)] * (y.ndim - 1),
                            "constant",
                            constant_values=-1,
                        )
                    else:
                        y = np.pad(
                            y,
                            [(0, n_pad)] + [(0, 0)] * (y.ndim - 1),
                            "edge",
                        )
                return np.concatenate([x, y[start:stop]], axis=0)

            elastic_batch_iter = (
                None
                if it.elastic_iterator is None
                else self.elastic_input.batches(it.elastic_iterator)
            )
            for input_batch in self.primary_input.batches(it.primary_iterator):
                if elastic_batch_iter is None:
                    elastic_batch = jax.tree.map(lambda x: x.copy(), input_batch)
                    elastic_batch["target_labels"][:] = -1
                else:
                    try:
                        elastic_batch = next(elastic_batch_iter)
                    except StopIteration:
                        return
                yield jax.tree.map_with_path(_padded_select, input_batch, elastic_batch)
        else:
            yield from self.primary_input.batches(it.primary_iterator)


def slice_count() -> int:
    """Returns the number of slices."""
    return len(set(d.slice_index for d in jax.devices() if hasattr(d, "slice_index"))) or 1


def process_count_per_slice() -> int:
    """Returns the number of processes per slice."""
    return (
        len(
            set(
                d.process_index
                for d in jax.devices()
                if hasattr(d, "slice_index") and d.slice_index == 0
            )
        )
        or 1
    )


class NonUniformShardingError(ValueError):
    """Raised when sharding is not uniform across processes."""


# This function is adapted from jax
# https://github.com/jax-ml/jax/blob/671483f4d9e50c38a168e07977aa35d0edfbb48b/jax/_src/sharding_impls.py#L730-L843
# Instead of returning `feed_index` of current process id only, we also return
# the process_id to feed_index mapping. See also
# https://github.com/jax-ml/jax/pull/31996
def get_process_index_and_count_and_mapping(
    tensor_sharding: jsharding.Sharding, dim: int, ndims: int
) -> tuple[int, int, dict[int, int]]:
    """Get current process index and number of unique processes for given dimension.

    This function facilitates mapping of process-level data to individual
    devices. Each process can use its index to obtain the data corresponding
    to that index. If process level data is sharded on multiple dimensions
    this function can be used to build the cross product of indices in
    each sharded axis. Processes that need to load the same data will have
    the same index. For shardings whose per-process data is not distributed
    on a grid, the number of distinct shards will be such that it is possible to
    build the target shape while maintaining a "cube" shape of local-process data.

    For example, in case of 4 hosts with sharding distributed like so:

    1234
    2143

    For dim 0 (rows): all processes need to access all rows, so we return (0, 1)
    For dim 1 (cols):
       process 1 and 2 returns index 0 out of 2 (need cols 0 and 1),
       process 3 and 4 returns index 1 out of 2 (need cols 2 and 3).

    On the other hand, for a sharding like:

    1212
    3434

    Dim 0 (rows): process 1 and 2 returns (0, 2), process 3 and 4 returns (1, 2)
    Dim 1 (cols): process 1 and 3 returns (0, 2), process 2 and 4 returns (1, 2)

    Note: This function requires sharding to be process uniform in dimension
    `dim`:
     each process has the same number of addressable indices in that
    dimension and all index sets across processes are either disjoint or the same.

    For sharding to be process uniform the addressable shards doesn't need to
    form contiguous subtensor, or even a sparse grid  and  in case of
    interleaved high-dimensional tensor it is possible for sharding to be
    process uniform only in some dimensions but not others.

    For example:
      1111 and 12 and 1212 and 1212
      2222     21     2121     1212

    are all sharding uniform, in both dimensions. However

      1122
      2121
      1121
      1222

    is uniform in dimension 0 (both hosts access all rows), but
    is not uniform in dimension 1 (host 1 accesses columns: 0, 1, and 3),
    while host 2 accesses (0, 1, 2, 3).

    Returns:
      A tuple of (index, num_distinct_shards, process_id_to_index) for the given dimension.
      It is guaranteed that `index` will cover 0 to `num_distinct_shards - 1`,
      across all processes.

    Raises:
      NonUniformShardingError: if the sharding is not process uniform in dimension
      `dim`.
    """
    if tensor_sharding.is_fully_addressable or tensor_sharding.is_fully_replicated:
        return (0, 1, {d.process_index: 0 for d in tensor_sharding.device_set})
    # Get device to indices map, we don't care about the concrete
    # global shape here, only to get the distribution of shards across the tensor
    # using (num_devices, num_devices, ...)  This is a universal shape that is
    # compatible with any mesh with num_devices.
    device_map = tensor_sharding.devices_indices_map((tensor_sharding.num_devices,) * ndims)

    # Get the slices for 'dim' for all devices.
    global_slice = {k: v[dim] for k, v in device_map.items()}

    # Contains mapping from process_index to a set of slices for that process.
    process_to_slice = collections.defaultdict(set)
    # Contains global set of slices across all processes.
    all_slices = set()

    # Compute the set of slices for each process and the global set of slices.
    for d, v in global_slice.items():
        key = (v.start, v.stop)
        process_to_slice[d.process_index].add(key)
        all_slices.add(key)

    # Get the set of slices for the current process which we will use to compute
    # the index of the current process.
    current_pid = next(iter(tensor_sharding.addressable_devices)).process_index
    addressable_slices = frozenset(process_to_slice[current_pid])

    # Verify that all processes have the same number of slices.
    slices_per_process = len(addressable_slices)
    if any(len(x) != slices_per_process for x in process_to_slice.values()):
        raise NonUniformShardingError(
            f"{tensor_sharding=} is non-uniform on {dim=} as some processes have "
            "different number of slices."
        )
    unique_processes = list({frozenset(x) for x in process_to_slice.values()})

    # After removing duplicate processes all unique slices should
    # cover the dimension exactly once. If they don't it means that
    # the sharding is not uniform.
    if sum(len(h) for h in unique_processes) != len(all_slices):
        raise NonUniformShardingError(f"{tensor_sharding=} is non-uniform on {dim=}")
    feed_index, feed_count = (unique_processes.index(addressable_slices), len(unique_processes))

    # !!! patch begin
    pid2fid = {}
    for pid, _ in process_to_slice.items():
        pid2fid[pid] = unique_processes.index(frozenset(process_to_slice[pid]))
    # !!! patch end
    return feed_index, feed_count, pid2fid

# Copyright © 2024 Apple Inc.

"""Tests grain inputs that require TPU."""

from typing import Optional, Protocol

import grain.python as grain
import jax
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.sharding import PartitionSpec

from axlearn.common.config import config_for_function
from axlearn.common.input_dispatch import InputDispatcher
from axlearn.common.input_grain import (
    BuildDatasetFn,
    Dataset,
    DispatchConfig,
    Input,
    maybe_to_iter_dataset,
    prefetch_dataset,
    shard_dataset,
)
from axlearn.common.utils import host_to_global_device_array, replicate_to_local_data
from axlearn.common.utils_spmd import setup as setup_spmd


def range_dataset(*, start, stop, step=1, seed=None) -> Dataset:
    source = grain.RangeDataSource(start=start, stop=stop, step=step)
    ds = grain.MapDataset.source(source)
    if seed is not None:
        ds = ds.seed(seed)
    return ds


class _PerProcessFn(Protocol):
    """Processes per-host data."""

    def __call__(self, ds: Dataset, *, dispatch_config: DispatchConfig) -> Dataset:
        ...


class InputTPUTest(parameterized.TestCase):
    """Tests Input module that require TPU."""

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

    @pytest.mark.tpu
    def test_dispatch_tpu(self):
        """Test that logical batching works on every other host.

        Can be run on 2 or more hosts (e.g., v5e-16).
        """
        setup_spmd(jax_backend="tpu")
        process_count = jax.process_count()
        process_index = jax.process_index()
        print(f"{process_count=}, {process_index=}")

        assert process_count % 2 == 0
        logical_batch_size = process_count // 2
        batch_sharding = max(1, logical_batch_size // 2)

        dispatcher = InputDispatcher.default_config().set(
            global_logical_batch_size=logical_batch_size
        )
        ds = range_dataset(start=0, stop=10, seed=123).map(lambda x: {"input_ids": x})
        cfg = self._input_config(
            ds.repeat(num_epochs=1),
            per_process=lambda ds, **_: ds.batch(1),
        )
        cfg.partition_spec = PartitionSpec("x")
        cfg.input_dispatcher = dispatcher

        with jax.sharding.Mesh(np.array(jax.devices()).reshape(batch_sharding, -1), ("x", "y")):
            grain_input: Input = cfg.instantiate(parent=None)
            for batch in grain_input:
                physical_batch = host_to_global_device_array(batch)
                batch = grain_input.dispatch_global_batch(physical_batch)

                self.assertEqual(batch["input_ids"].shape[0], logical_batch_size)
                self.assertEqual(batch["input_ids"].sharding.spec, cfg.partition_spec)
                self.assertEqual(
                    list(range(logical_batch_size)),
                    replicate_to_local_data(batch)["input_ids"].tolist(),
                )
                break


if __name__ == "__main__":
    absltest.main()

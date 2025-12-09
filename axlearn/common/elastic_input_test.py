# Copyright © 2025 Apple Inc.

"""Tests ElasticInput."""

import math
import tempfile
from unittest import mock

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, PartitionSpec

from axlearn.common.config import config_for_function
from axlearn.common.elastic_input import ElasticInput, ElasticSpmdInputDispatcher
from axlearn.common.input_tf_data import (
    Input,
    default_pad_example_fn,
    identity,
    per_feed_batch,
    tfds_dataset,
)


class ElasticInputTest(parameterized.TestCase):
    @parameterized.parameters(
        {"target_pid": 0}, {"target_pid": 1}, {"target_pid": 2}, {"target_pid": 3}
    )
    def test_default_input_dispatcher(self, target_pid: int):
        # This test case is carefully designed to catch some corner cases. We
        # first create a fake dataset with 5feeds. The simulated cluster has 5
        # slices, one process for each slice. In elastic mode, we assume 1 slice
        # is down and only the rest 4 slices are working. To test some corner
        # cases, the batch size per device is assumed to be 5. Now the workload
        # from unhealthy process is redistributed to process 0, 1, 2, 3 ( [2, 2, 1, 0]
        # samples per each process).
        with tempfile.TemporaryDirectory() as tmp_data_dir:
            fake_dataset_name = "fake_dataset"
            split = "train"

            seq_length = 8
            num_processes = 5
            num_devices_per_process = 4

            num_feeds = num_processes  # For InputDispatcher only

            num_samples_per_device = 5
            num_samples_per_feed = num_samples_per_device * num_devices_per_process

            global_batch_size = num_samples_per_feed * num_feeds

            k_unhealthy_processes = 1
            n_pad_per_device = num_samples_per_device * (
                k_unhealthy_processes / (num_processes - k_unhealthy_processes)
            )
            n_pad_per_device_rounded = math.ceil(n_pad_per_device)
            n_pad_per_process = n_pad_per_device_rounded * num_devices_per_process

            def data_gen():
                for i in range(global_batch_size):
                    yield {
                        "input_ids": tf.repeat(i, seq_length),
                        "target_labels": tf.repeat(i + 1, seq_length),
                    }

            tfds.dataset_builders.store_as_tfds_dataset(
                name=fake_dataset_name,
                version="1.0.0",
                features=tfds.features.FeaturesDict(
                    {
                        "input_ids": tfds.features.Tensor(shape=(seq_length,), dtype=tf.int32),
                        "target_labels": tfds.features.Tensor(shape=(seq_length,), dtype=tf.int32),
                    },
                ),
                split_datasets={
                    split: tf.data.Dataset.from_generator(
                        data_gen,
                        output_signature={
                            "input_ids": tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
                            "target_labels": tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
                        },
                    )
                },
                data_dir=tmp_data_dir,
                download_config=tfds.download.DownloadConfig(num_shards=num_feeds),
                disable_shuffling=True,
            )

            input_cfg = ElasticInput.default_config().set(
                input=Input.default_config().set(
                    source=config_for_function(tfds_dataset).set(
                        dataset_name=fake_dataset_name,
                        split=split,
                        is_training=True,
                        data_dir=tmp_data_dir,
                        train_shuffle_buffer_size=0,
                        train_shuffle_files=False,
                    ),
                    input_dispatcher=ElasticSpmdInputDispatcher.default_config().set(
                        num_max_slices=num_processes,
                        global_logical_batch_size=global_batch_size,
                    ),
                    processor=config_for_function(identity),
                    batcher=config_for_function(per_feed_batch).set(
                        feed_batch_size=num_samples_per_feed,
                        is_training=True,
                        pad_example_fn=default_pad_example_fn,
                    ),
                    is_training=True,
                    partition_spec=PartitionSpec(("data",)),
                ),
                name="test_elastic_input",
            )

            # simulate the N - 1 case workload from process 3 of 8 samples are evenly
            # redistributed to process 0 and 1, each of size 4.
            with (
                Mesh([jax.local_devices()[0]] * 16, "data"),
                mock.patch(
                    "axlearn.common.elastic_input.slice_count", return_value=num_processes - 1
                ),
                mock.patch(
                    "axlearn.common.elastic_input.get_process_index_and_count_and_mapping",
                    return_value=(None, None, {0: 0, 1: 1, 2: 2, 3: 3}),
                ),
                mock.patch("jax.process_count", return_value=num_processes - 1),
                mock.patch("jax.process_index", return_value=target_pid),
                mock.patch("jax.local_device_count", return_value=num_devices_per_process),
            ):
                inp = input_cfg.instantiate(parent=None)
                input_iter = iter(inp.dataset())

                print(inp.batches(input_iter))

                first_batch = next(inp.batches(input_iter))

                if target_pid == 0:
                    desired_input_ids = np.repeat(
                        np.concatenate(
                            (
                                # original workload
                                np.arange(num_samples_per_feed),
                                # dispatched workload
                                np.arange(
                                    num_samples_per_feed * (num_processes - 1),
                                    num_samples_per_feed * (num_processes - 1) + n_pad_per_process,
                                ),
                            )
                        ),
                        seq_length,
                    ).reshape(-1, seq_length)

                    assert np.all(first_batch["input_ids"] == desired_input_ids)
                    assert np.all(first_batch["target_labels"] == first_batch["input_ids"] + 1)
                elif target_pid == 2:
                    n_zero_padding = num_devices_per_process * (
                        n_pad_per_device_rounded - math.floor(n_pad_per_device)
                    )
                    desired_input_ids = np.repeat(
                        np.concatenate(
                            (
                                # original workload
                                np.arange(num_samples_per_feed * 2, num_samples_per_feed * 3),
                                # dispatched workload
                                np.arange(
                                    num_samples_per_feed * (num_processes - 1)
                                    + n_pad_per_process * 2,
                                    num_samples_per_feed * (num_processes - 1)
                                    + n_pad_per_process * 3
                                    - n_zero_padding,
                                ),
                            )
                        ),
                        seq_length,
                    ).reshape(-1, seq_length)

                    assert np.all(first_batch["input_ids"][:-n_zero_padding] == desired_input_ids)
                    assert np.all(
                        first_batch["target_labels"][:-n_zero_padding]
                        == first_batch["input_ids"][:-n_zero_padding] + 1
                    )
                    assert np.all(first_batch["target_labels"][-n_zero_padding:] == -1)
                elif target_pid in (3,):
                    # ensure the padding part is not involved in final loss calculation
                    assert np.all(first_batch["target_labels"][-4:] == -1)


if __name__ == "__main__":
    absltest.main()

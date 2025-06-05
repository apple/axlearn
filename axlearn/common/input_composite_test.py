# Copyright © 2023 Apple Inc.

"""Tests composite inputs."""
# pylint: disable=no-self-use

import tensorflow as tf
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import input_tf_data
from axlearn.common.config import config_for_function
from axlearn.common.input_composite import ConcatenatedInput, ZipInput
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor


def make_ds_fn(
    is_training: bool,
    numbers: list[int],
    out_signature="number",
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for index, number in enumerate(numbers):
                yield {out_signature: number, "index": index}

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                out_signature: tf.TensorSpec(shape=(), dtype=tf.int32),
                "index": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
        )

    return ds_fn


class BatchTest(TestCase):
    def _input_config(
        self,
        numbers: list[int],
        *,
        batch_size=2,
        repeat=1,
        out_signature="number",
    ) -> input_tf_data.Input.Config:
        return input_tf_data.Input.default_config().set(
            source=config_for_function(make_ds_fn).set(
                numbers=numbers, out_signature=out_signature
            ),
            processor=config_for_function(input_tf_data.identity),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                pad_example_fn=input_tf_data.default_pad_example_fn,
                repeat=repeat,
            ),
        )

    @parameterized.parameters(False, True)
    def test_concatenation(self, is_training):
        cfg = ConcatenatedInput.default_config().set(
            name="input",
            is_training=is_training,
            inputs=[
                self._input_config([1, 2, 3], batch_size=2),
                self._input_config([11, 12, 13, 14, 15, 16], batch_size=3, repeat=None),
            ],
        )
        dataset = cfg.instantiate(parent=None)
        batch_index = 0
        expected_train_batches = [
            {"index": jnp.asarray([0, 1]), "number": jnp.asarray([1, 2])},
            {"index": jnp.asarray([0, 1, 2]), "number": jnp.asarray([11, 12, 13])},
            {"index": jnp.asarray([3, 4, 5]), "number": jnp.asarray([14, 15, 16])},
            {"index": jnp.asarray([0, 1, 2]), "number": jnp.asarray([11, 12, 13])},
            {"index": jnp.asarray([3, 4, 5]), "number": jnp.asarray([14, 15, 16])},
        ]
        expected_eval_batches = [
            {"index": jnp.asarray([0, 1]), "number": jnp.asarray([1, 2])},
            {"index": jnp.asarray([2, 0]), "number": jnp.asarray([3, 0])},
            {"index": jnp.asarray([0, 1, 2]), "number": jnp.asarray([11, 12, 13])},
            {"index": jnp.asarray([3, 4, 5]), "number": jnp.asarray([14, 15, 16])},
        ]
        expected_batches = expected_train_batches if is_training else expected_eval_batches
        for batch in dataset.dataset():
            print(batch)
            if batch_index >= len(expected_batches):
                break
            self.assertNestedAllClose(as_tensor(expected_batches[batch_index]), batch)
            batch_index += 1
        self.assertEqual(batch_index, len(expected_batches))

    @parameterized.parameters(False, True)
    def test_zipinput(self, is_training):
        cfg = ZipInput.default_config().set(
            name="input",
            is_training=is_training,
            inputs={
                "0": self._input_config(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9], batch_size=3, out_signature="number_1", repeat=None
                ),
                "1": self._input_config(
                    [11, 12, 13, 14, 15, 16, 17, 18, 19],
                    batch_size=2,
                    out_signature="number_2",
                    repeat=None,
                ),
            },
        )
        dataset = cfg.instantiate(parent=None)
        expected_train_batches = [
            {
                "0": {"number_1": jnp.asarray([1, 2, 3]), "index": jnp.asarray([0, 1, 2])},
                "1": {"number_2": jnp.asarray([11, 12]), "index": jnp.asarray([0, 1])},
            },
            {
                "0": {"number_1": jnp.asarray([4, 5, 6]), "index": jnp.asarray([3, 4, 5])},
                "1": {"number_2": jnp.asarray([13, 14]), "index": jnp.asarray([2, 3])},
            },
        ]
        expected_eval_batches = [
            {
                "0": {"number_1": jnp.asarray([1, 2, 3]), "index": jnp.asarray([0, 1, 2])},
                "1": {"number_2": jnp.asarray([11, 12]), "index": jnp.asarray([0, 1])},
            },
            {
                "0": {"number_1": jnp.asarray([4, 5, 6]), "index": jnp.asarray([3, 4, 5])},
                "1": {"number_2": jnp.asarray([13, 14]), "index": jnp.asarray([2, 3])},
            },
        ]
        expected_batches = expected_train_batches if is_training else expected_eval_batches
        for batch_index, batch in enumerate(dataset):
            if batch_index >= len(expected_batches):
                break
            self.assertNestedAllClose(as_tensor(expected_batches[batch_index]), batch)
        self.assertEqual(
            batch_index, len(expected_batches)  # pylint: disable=undefined-loop-variable
        )

        dataset = cfg.instantiate(parent=None)
        for batch_index, batch in enumerate(dataset.dataset()):
            if batch_index >= len(expected_batches):
                break
            self.assertNestedAllClose(as_tensor(expected_batches[batch_index]), batch)
        self.assertEqual(
            batch_index, len(expected_batches)  # pylint: disable=undefined-loop-variable
        )


if __name__ == "__main__":
    absltest.main()

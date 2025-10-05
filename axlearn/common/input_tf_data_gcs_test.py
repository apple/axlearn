# Copyright © 2023 Apple Inc.

"""Tests for tf.data inputs that require GCS access."""

from typing import Optional

import jax
import pytest
import tensorflow_datasets as tfds
from absl.testing import absltest, parameterized

from axlearn.common.config import config_for_function
from axlearn.common.input_tf_data import (
    _infer_num_examples,
    _infer_num_shards,
    _maybe_shard_examples,
    tfds_dataset,
    tfds_read_config,
)


class TfdsGcsTest(parameterized.TestCase):
    """Tests for TFDS functionality that require GCS access.

    These tests require Google Cloud Storage authentication to access
    dataset metadata. They are separated from the main test file to
    allow Bazel to filter them at the target level.
    """

    @parameterized.parameters(
        ("train", 1024), ("validation", 8), ("train[:512]", 1), ("invalid", None)
    )
    @pytest.mark.gs_login  # For pytest compatibility
    def test_infer_num_shards(self, split: str, expected: Optional[int]):
        builder = tfds.builder("c4/en", try_gcs=True)
        self.assertEqual(_infer_num_shards(builder, split), expected)

    @parameterized.parameters(
        ("validation", 1043), ("test", 1063), ("test[:12]", 12), ("invalid", None)
    )
    @pytest.mark.gs_login  # For pytest compatibility
    def test_infer_num_examples(self, split: str, expected: Optional[int]):
        builder = tfds.builder("glue/cola:2.0.0", try_gcs=True)
        self.assertEqual(_infer_num_examples(builder, split), expected)

    @parameterized.parameters(
        ("validation", 5, True, "even split"),
        ("validation", 1044, False, "make copy for each host"),
        ("validation", 1044, True, "raise value error"),
        ("invalid", 5, True, "even split"),
    )
    @pytest.mark.gs_login
    def test_maybe_shard_examples(
        self, split: str, required_shards: int, is_training: bool, expected: str
    ):
        dataset_name = "glue/cola:2.0.0"
        builder = tfds.builder(dataset_name, try_gcs=True)
        read_config = config_for_function(tfds_read_config).set(is_training=is_training)
        if expected == "raise value error":
            with self.assertRaises(ValueError):
                _ = _maybe_shard_examples(
                    builder=builder,
                    read_config=read_config,
                    split=split,
                    required_shards=required_shards,
                    is_training=is_training,
                    dataset_name=dataset_name,
                )
        else:
            per_process_split = _maybe_shard_examples(
                builder=builder,
                read_config=read_config,
                split=split,
                required_shards=required_shards,
                is_training=is_training,
                dataset_name=dataset_name,
            )
            if expected == "even split":
                shard_index = read_config.shard_index or jax.process_index()
                expected_split = tfds.even_splits(split, n=required_shards, drop_remainder=False)[
                    shard_index
                ]
                self.assertTrue(expected_split == per_process_split)
            elif expected == "make copy for each host":
                self.assertTrue(per_process_split == split)

    @parameterized.parameters(
        ("validation", True, "sentence", "foobar"),
        ("test", True, "sentence", "barfoo"),
        ("validation", False, "sentence", "bar bar"),
        ("test", False, "sentence", "foo foo"),
    )
    @pytest.mark.gs_login
    def test_tfds_decoders(self, split: str, is_training: bool, field_name: str, expected: str):
        def tfds_custom_decoder() -> dict[str, tfds.decode.Decoder]:
            @tfds.decode.make_decoder()
            def replace_field_value(field_value, _):
                return field_value + expected

            # pylint: disable=no-value-for-parameter
            return {field_name: replace_field_value()}

        decoders = config_for_function(tfds_custom_decoder)

        dataset_name = "glue/cola:2.0.0"
        source = config_for_function(tfds_dataset).set(
            dataset_name=dataset_name,
            split=split,
            is_training=is_training,
            train_shuffle_buffer_size=8,
            decoders=decoders,
        )
        ds = source.instantiate()

        for input_batch in ds().take(5):
            assert expected in input_batch[field_name].numpy().decode(
                "UTF-8"
            ), f"Missing {expected} string in {field_name} field"


if __name__ == "__main__":
    absltest.main()

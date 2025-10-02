# Copyright © 2023 Apple Inc.

"""Tests language modeling inputs."""

# pylint: disable=no-self-use,too-many-lines
import os
import tempfile
from typing import Any, Literal, Optional

import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
from absl.testing import absltest, parameterized

from axlearn.common import input_tf_data, test_utils
from axlearn.common.config import InstantiableConfig, config_for_class, config_for_function
from axlearn.common.input_fake import fake_source
from axlearn.common.input_lm import (
    InputDataType,
    ModelType,
    PackingMethodType,
    _trim_and_pack_with_segments,
    augment_text_from_inputs_targets_pretokenized,
    joint_truncation_for_seq2seq_lm_input,
    lm_from_seq2seq_text_preprocessor,
    lm_text_preprocessor,
    make_autoregressive_inputs,
    text2text_lm_input,
    text_to_lm_eval_input,
    text_to_lm_training_input,
)
from axlearn.common.input_test_utils import make_ds_fn, t5_sentence_piece_vocab_file


class BaseLmInputTest(test_utils.TestCase):
    """Base class containing utilities for LmTrainingInputTest and LmEvalInputTest.

    Inherited by other test files. DO NOT write test cases in this class.
    """

    newlines_replaced_with = "<n>"

    def _lm_eval_processor_config(
        self,
        *,
        vocab_cfg: InstantiableConfig,
        max_len: int,
        stride: int,
        index_key: Optional[str] = None,
    ):
        return config_for_function(text_to_lm_eval_input).set(
            vocab_cfg=vocab_cfg,
            max_len=max_len,
            replace_newlines_with=self.newlines_replaced_with,
            stride=stride,
            index_key=index_key,
        )

    def _test_fake_text_lm_eval_data(
        self,
        *,
        texts=list[str],
        vocab_cfg: InstantiableConfig,
        expected_batches: list[dict[str, Any]],
    ):
        batch_size, max_len = 3, 12

        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=False,
            source=config_for_function(make_ds_fn).set(texts=texts),
            processor=self._lm_eval_processor_config(
                vocab_cfg=vocab_cfg, max_len=max_len, stride=2
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        # Set TensorFlow seed.
        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for ix, batch in enumerate(dataset):
            batch = {k: v.tolist() for k, v in batch.items()}
            self.assertNestedAllClose(expected_batches[ix], batch)
            if ix > 0:
                # Check the first two batches.
                break


class LmTrainingInputTest(BaseLmInputTest):
    @property
    def vocab_cfg(self) -> InstantiableConfig:
        return config_for_class(seqio.SentencePieceVocabulary).set(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )

    @parameterized.parameters("Lorem ipsum dolor sit amet,", " consectetur adipiscing elit\n")
    def test_training_lm_processor_single_example(self, text: str):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        max_len = 32
        processor = (
            config_for_function(text_to_lm_training_input)
            .set(
                vocab_cfg=self.vocab_cfg,
                max_len=max_len,
                window_size=3,
                max_padding_fraction=0.0,
                packing_method=PackingMethodType.EOS_DELIM_NO_MASK,
            )
            .instantiate()
        )
        ds_fn = (
            config_for_function(make_ds_fn)
            .set(is_training=False, texts=[text] * 10, repeat=1)
            .instantiate()
        )
        # Set TensorFlow seed.
        tf.random.set_seed(123)
        example = next(iter(processor(ds_fn())))
        for key in ["input_ids", "target_labels"]:
            # Shape is as expected.
            self.assertEqual((max_len,), example[key].numpy().shape)
        self.assertTrue("target_num_bytes" in example)
        input_ids, target_labels = example["input_ids"].numpy(), example["target_labels"].numpy()
        self.assertTrue(1 in input_ids)  # EOS somewhere in the inputs.
        self.assertTrue(1 in target_labels)  # EOS somewhere in the targets.
        # The inputs should be one-off the labels.
        self.assertNestedAllClose(target_labels[:-1], input_ids[1:])

    @parameterized.parameters(
        dict(
            packing_method=PackingMethodType.EOS_DELIM_MASK,
            expected_batches=[
                {
                    "input_ids": [
                        [1, 21820, 296, 2, 29],
                        [3155, 1, 21820, 8114, 2],
                    ],
                    "target_labels": [
                        [21820, 296, 2, 29, 3155],
                        [1, 21820, 8114, 2, 29],
                    ],
                    "input_segment_ids": [
                        [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 2],
                    ],
                    "input_positions": [
                        [0, 1, 2, 3, 4],
                        [0, 0, 1, 2, 3],
                    ],
                    "target_num_bytes": [18, 17],
                },
                {
                    "input_ids": [
                        [29, 3155, 1, 21820, 296],
                        [2, 29, 3155, 0, 0],
                    ],
                    "target_labels": [
                        [3155, 1, 21820, 296, 2],
                        [29, 3155, 1, 0, 0],
                    ],
                    "input_segment_ids": [
                        [1, 1, 2, 2, 2],
                        [1, 1, 1, 0, 0],
                    ],
                    "input_positions": [
                        [0, 1, 0, 1, 2],
                        [0, 1, 2, 0, 0],
                    ],
                    "target_num_bytes": [19, 3],
                },
            ],
            max_padding_fraction=1.0,  # Always pad
        ),
        dict(
            packing_method=PackingMethodType.EOS_DELIM_NO_MASK,
            expected_batches=[
                {
                    "input_ids": [
                        [1, 21820, 296, 2, 29],
                        [3155, 1, 21820, 8114, 2],
                    ],
                    "target_labels": [
                        [21820, 296, 2, 29, 3155],
                        [1, 21820, 8114, 2, 29],
                    ],
                    "target_num_bytes": [18, 17],
                },
                {
                    "input_ids": [
                        [29, 3155, 1, 21820, 296],
                        [2, 29, 3155, 0, 0],
                    ],
                    "target_labels": [
                        [3155, 1, 21820, 296, 2],
                        [29, 3155, 1, 0, 0],
                    ],
                    "target_num_bytes": [19, 3],
                },
            ],
            max_padding_fraction=1.0,  # Always pad
        ),
        dict(
            packing_method=PackingMethodType.EOS_DELIM_MASK,
            expected_batches=[
                {
                    "input_ids": [
                        [1, 21820, 296, 2, 29],
                        [3155, 1, 21820, 8114, 2],
                    ],
                    "target_labels": [
                        [21820, 296, 2, 29, 3155],
                        [1, 21820, 8114, 2, 29],
                    ],
                    "input_segment_ids": [
                        [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 2],
                    ],
                    "input_positions": [
                        [0, 1, 2, 3, 4],
                        [0, 0, 1, 2, 3],
                    ],
                    "target_num_bytes": [18, 17],
                },
                {
                    "input_ids": [
                        [29, 3155, 1, 21820, 296],
                        [1, 21820, 8114, 2, 29],
                    ],
                    "target_labels": [
                        [3155, 1, 21820, 296, 2],
                        [21820, 8114, 2, 29, 3155],
                    ],
                    "input_segment_ids": [
                        [1, 1, 2, 2, 2],
                        [1, 1, 1, 1, 1],
                    ],
                    "input_positions": [
                        [0, 1, 0, 1, 2],
                        [0, 1, 2, 3, 4],
                    ],
                    "target_num_bytes": [19, 17],
                },
            ],
            max_padding_fraction=0.0,  # Do not pad
        ),
    )
    def test_fake_text_lm_training_data(
        self,
        packing_method: PackingMethodType,
        expected_batches: list[dict[str, Any]],
        max_padding_fraction: float,
    ):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")
        texts = [
            "hello world\n",
            "hello moon\n",
        ]

        # window_size > len(texts) to repeat the sentence. 18 tokens in total.
        # [    1,  21820,   296,     2,    29,  3155,     1, 21820,  8114,
        #      2,     29,  3155,     1, 21820,   296,     2,    29,  3155]
        window_size = 3

        # Pad the concatenated sequence to 20 tokens:
        # [    1,  21820,   296,     2,    29,  3155,     1, 21820,  8114,    2
        #     29,  3155,     1, 21820,   296,     2,    29,  3155,      0,    0]
        #
        # Or, trim the sequence to 15 tokens:
        # [    1,  21820,   296,     2,    29,  3155,     1, 21820,  8114,    2
        #     29,  3155,     1, 21820,   296]
        batch_size, max_len = 2, 5

        # Disable shuffling to make results interpretable.
        shuffle_buffer_size = 0

        # Test text_to_lm_training_input.
        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=True,
            source=config_for_function(make_ds_fn).set(texts=texts),
            processor=config_for_function(text_to_lm_training_input).set(
                vocab_cfg=self.vocab_cfg,
                max_len=max_len,
                replace_newlines_with=self.newlines_replaced_with,
                window_size=window_size,
                max_padding_fraction=max_padding_fraction,
                shuffle_buffer_size=shuffle_buffer_size,
                packing_method=packing_method,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        # Set TensorFlow seed.
        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for ix, batch in enumerate(dataset):
            self.assertIsNotNone(batch)
            if ix < len(expected_batches):
                # Check for equality for provided batches.
                batch = {k: v.tolist() for k, v in batch.items()}
                self.assertNestedAllClose(expected_batches[ix], batch)
            if ix >= 10 * len(expected_batches):
                # Expect to be able to repeat forever.
                break

        # Test lm_text_preprocessor. Expect same results.
        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=True,
            source=config_for_function(make_ds_fn).set(
                texts=texts,
            ),
            processor=config_for_function(lm_text_preprocessor).set(
                vocab_cfg=self.vocab_cfg,
                max_sequence_length=max_len,
                replace_newlines_with=self.newlines_replaced_with,
                window_size=window_size,
                max_padding_fraction=max_padding_fraction,
                shuffle_buffer_size=shuffle_buffer_size,
                packing_method=packing_method,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        # Reset TensorFlow seed.
        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for ix, batch in enumerate(dataset):
            if ix >= len(expected_batches):
                break
            batch = {k: v.tolist() for k, v in batch.items()}
            self.assertNestedAllClose(expected_batches[ix], batch)

    def test_fake_text_lm_training_data_eval(self):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        # N.B. we do not typically expect users to run the training data pipeline in eval mode.
        # Instead we expect them to prefer `text_to_lm_eval_input`.
        texts = [
            "hello world\n",
            "hello moon\n",
            "hello tiger\n",
            "hello dog\n",
            "hello cat\n",
        ]
        expected_first_and_last_batch = [
            {
                "input_ids": [
                    [1, 21820, 296, 2, 29, 3155],
                    [1, 21820, 8114, 2, 29, 3155],
                    [1, 21820, 3, 17, 4424, 2],
                ],
                "target_labels": [
                    [21820, 296, 2, 29, 3155, 1],
                    [21820, 8114, 2, 29, 3155, 1],
                    [21820, 3, 17, 4424, 2, 29],
                ],
                "target_num_bytes": [19, 18, 17],
            },
            # Last batch should have some fully-padded examples.
            {
                "input_ids": [[1712, 2, 29, 3155, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                "target_labels": [[2, 29, 3155, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                "target_num_bytes": [8, 0, 0],
            },
        ]

        # Test lm_text_preprocessor. Expect same results.
        batch_size, max_len = 3, 6
        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=False,
            source=config_for_function(make_ds_fn).set(texts=texts),
            processor=config_for_function(lm_text_preprocessor).set(
                vocab_cfg=self.vocab_cfg,
                max_sequence_length=max_len,
                replace_newlines_with=self.newlines_replaced_with,
                window_size=10,
                max_padding_fraction=0.5,
                shuffle_buffer_size=0,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )
        # Set TensorFlow seed.
        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for ix, batch in enumerate(dataset):
            batch = {k: v.tolist() for k, v in batch.items()}
            # Compare the first and last batch.
            if ix == 0:
                self.assertNestedAllClose(expected_first_and_last_batch[0], batch)
        # pylint: disable=undefined-loop-variable
        self.assertNestedAllClose(expected_first_and_last_batch[1], batch)

    @parameterized.parameters(
        {
            "input_data_type": InputDataType.SEQ2SEQ_MASK,
            "max_len": 11,
            "expected_inputs": dict(
                input_ids=[
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 0],
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 0],
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17],
                ],
                target_labels=[
                    [-1, -1, -1, -1, -1, -1, -1, 21820, 296, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 21820, 8114, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 21820, 3, 17, 4424],
                ],
                prefix=[[1], [1], [1]],
            ),
        },
        {
            "input_data_type": InputDataType.SEQ2SEQ_NO_MASK,
            "max_len": 15,
            "expected_inputs": dict(
                input_ids=[
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 0, 0, 0, 0, 0],
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 0, 0, 0, 0, 0],
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 0, 0, 0],
                ],
                target_labels=[
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 1, -1, -1, -1, -1, -1],
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 1, -1, -1, -1, -1, -1],
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 1, -1, -1, -1],
                ],
                prefix=[[1], [1], [1]],
            ),
        },
        # Test joint truncation.
        {
            "input_data_type": InputDataType.SEQ2SEQ_MASK,
            "max_len": 5,
            "expected_inputs": dict(
                input_ids=[
                    [2, 29, 3155, 21820, 296],
                    [2, 29, 3155, 21820, 8114],
                    [3155, 21820, 3, 17, 4424],
                ],
                target_labels=[
                    [-1, -1, 21820, 296, 1],
                    [-1, -1, 21820, 8114, 1],
                    [21820, 3, 17, 4424, 1],
                ],
                prefix=[[1], [1], [1]],
            ),
            "joint_truncation": True,
        },
        {
            "input_data_type": InputDataType.SEQ2SEQ_MASK,
            "max_len": 11,
            "expected_inputs": dict(
                input_ids=[
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 0],
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 0],
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424],
                ],
                target_labels=[
                    [-1, -1, -1, -1, -1, -1, -1, 21820, 296, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 21820, 8114, 1, -1],
                    [-1, -1, -1, -1, -1, -1, 21820, 3, 17, 4424, 1],
                ],
                prefix=[[1], [1], [1]],
            ),
            "joint_truncation": True,
        },
        {
            "input_data_type": InputDataType.SEQ2SEQ_NO_MASK,
            "max_len": 15,
            "expected_inputs": dict(
                input_ids=[
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 0, 0, 0, 0, 0],
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 0, 0, 0, 0, 0],
                    [1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 0, 0, 0],
                ],
                target_labels=[
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 1, -1, -1, -1, -1, -1],
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 1, -1, -1, -1, -1, -1],
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 1, -1, -1, -1],
                ],
                prefix=[[1], [1], [1]],
            ),
            "joint_truncation": True,
        },
        # Test longer prefix.
        {
            "input_data_type": InputDataType.SEQ2SEQ_MASK,
            "max_len": 11,
            "expected_inputs": dict(
                input_ids=[
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296],
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114],
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3],
                ],
                target_labels=[
                    [-1, -1, -1, -1, -1, -1, -1, -1, 21820, 296, 1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, 21820, 8114, 1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, 21820, 3, 17],
                ],
                prefix=[[1, 1], [1, 1], [1, 1]],
            ),
        },
        {
            "input_data_type": InputDataType.SEQ2SEQ_NO_MASK,
            "max_len": 15,
            "expected_inputs": dict(
                input_ids=[
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 0, 0, 0, 0],
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 0, 0, 0, 0],
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 0, 0],
                ],
                target_labels=[
                    [-1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 1, -1, -1, -1, -1],
                    [-1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 1, -1, -1, -1, -1],
                    [-1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 1, -1, -1],
                ],
                prefix=[[1, 1], [1, 1], [1, 1]],
            ),
        },
        # Test longer prefix + joint truncation.
        {
            "input_data_type": InputDataType.SEQ2SEQ_MASK,
            "max_len": 5,
            "expected_inputs": dict(
                input_ids=[
                    [2, 29, 3155, 21820, 296],
                    [2, 29, 3155, 21820, 8114],
                    [3155, 21820, 3, 17, 4424],
                ],
                target_labels=[
                    [-1, -1, 21820, 296, 1],
                    [-1, -1, 21820, 8114, 1],
                    [21820, 3, 17, 4424, 1],
                ],
                prefix=[[1, 1], [1, 1], [1, 1]],
            ),
            "joint_truncation": True,
        },
        {
            "input_data_type": InputDataType.SEQ2SEQ_MASK,
            "max_len": 11,
            "expected_inputs": dict(
                input_ids=[
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296],
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114],
                    [21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424],
                ],
                target_labels=[
                    [-1, -1, -1, -1, -1, -1, -1, -1, 21820, 296, 1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, 21820, 8114, 1],
                    [-1, -1, -1, -1, -1, -1, 21820, 3, 17, 4424, 1],
                ],
                prefix=[[1, 1], [1, 1], [1, 1]],
            ),
            "joint_truncation": True,
        },
        {
            "input_data_type": InputDataType.SEQ2SEQ_NO_MASK,
            "max_len": 15,
            "expected_inputs": dict(
                input_ids=[
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 0, 0, 0, 0],
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 0, 0, 0, 0],
                    [1, 1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 0, 0],
                ],
                target_labels=[
                    [-1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 296, 1, -1, -1, -1, -1],
                    [-1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 8114, 1, -1, -1, -1, -1],
                    [-1, 21820, 2, 29, 3155, 2, 29, 3155, 21820, 3, 17, 4424, 1, -1, -1],
                ],
                prefix=[[1, 1], [1, 1], [1, 1]],
            ),
            "joint_truncation": True,
        },
    )
    def test_lm_from_seq2seq_text_preprocessor(
        self,
        input_data_type: InputDataType,
        max_len: int,
        expected_inputs: dict[str, list[list[int]]],
        is_training: bool = True,
        source_key: str = "inputs_pretokenized",
        target_key: str = "targets_pretokenized",
        joint_truncation: bool = False,
    ):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        batch_size = 3

        def ds_fn():
            return fake_source(
                is_training=False,
                examples=[
                    {
                        source_key: "hello",
                        target_key: "hello world",
                        "prefix": tf.constant(expected_inputs["prefix"][0]),
                    },
                    {
                        source_key: "hello",
                        target_key: "hello moon",
                        "prefix": tf.constant(expected_inputs["prefix"][1]),
                    },
                    {
                        source_key: "hello",
                        target_key: "hello tiger",
                        "prefix": tf.constant(expected_inputs["prefix"][2]),
                    },
                ],
            )

        if joint_truncation:
            preprocessor_cfg = config_for_function(input_tf_data.chain).set(
                args=[
                    joint_truncation_for_seq2seq_lm_input(max_sequence_length=max_len),
                    make_autoregressive_inputs(
                        model_type=ModelType.DECODER_ONLY, input_data_type=input_data_type
                    ),
                    input_tf_data.remove_fields(["prefix"]),
                ],
            )
        else:
            preprocessor_cfg = None

        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=is_training,
            source=config_for_function(ds_fn),
            processor=config_for_function(lm_from_seq2seq_text_preprocessor).set(
                is_training=True,
                vocab_cfg=self.vocab_cfg,
                max_sequence_length=max_len,
                input_data_type=input_data_type,
                replace_newlines_with=self.newlines_replaced_with,
                source_key=source_key,
                target_key=target_key,
                autoregressive_processor_cfg=preprocessor_cfg,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            for k in ["input_ids", "target_labels"]:
                self.assertEqual(batch[k].tolist(), expected_inputs[k])
            break

    @parameterized.parameters(
        dict(
            replace_newlines_with="<n>",
            source_key="inputs_pretokenized",
            target_key="targets_pretokenized",
            eos_id=1,
            prompt="prompt ",
            join_with="",
            examples=[
                {
                    "inputs_pretokenized": "hello",
                    "targets_pretokenized": "hello\nworld",
                },
                {
                    "inputs_pretokenized": "hello",
                    "targets_pretokenized": "hello moon",
                },
                {
                    "inputs_pretokenized": "hello",
                    "targets_pretokenized": "hello tiger",
                },
            ],
            expected=[
                {
                    "inputs_pretokenized": tf.constant("prompt hello"),
                    "targets_pretokenized": tf.constant("hello<n>world"),
                    "prefix": tf.constant(1),
                },
                {
                    "inputs_pretokenized": tf.constant("prompt hello"),
                    "targets_pretokenized": tf.constant("hello moon"),
                    "prefix": tf.constant(1),
                },
                {
                    "inputs_pretokenized": tf.constant("prompt hello"),
                    "targets_pretokenized": tf.constant("hello tiger"),
                    "prefix": tf.constant(1),
                },
            ],
        )
    )
    def test_augment_text_from_inputs_targets_pretokenized(
        self,
        replace_newlines_with,
        source_key,
        target_key,
        eos_id,
        prompt,
        join_with,
        examples,
        expected,
    ):
        ds = fake_source(
            is_training=False,
            examples=examples,
        )()
        processor = augment_text_from_inputs_targets_pretokenized(
            replace_newlines_with=replace_newlines_with,
            source_key=source_key,
            target_key=target_key,
            eos_id=eos_id,
            prompt=prompt,
            join_with=join_with,
        )
        actual = list(processor(ds))
        self.assertEqual(actual, expected)

    @parameterized.parameters(
        {
            "expected_inputs": {
                "prefix": [0],
                "input_ids": [0, 4, 5, 6, 1, 2],
                "target_labels": [4, 5, 6, 1, 2, 3],
            },
            "passthrough_keys": [],
        },
        {
            "expected_inputs": {
                "prefix": [0],
                "input_ids": [0, 4, 5, 6, 1, 2],
                "target_labels": [4, 5, 6, 1, 2, 3],
            },
            "passthrough_keys": ["id"],
        },
        {
            "expected_inputs": {
                "prefix": [0],
                "input_ids": [0, 4, 5, 6, 1, 2],
                "target_labels": [4, 5, 6, 1, 2, 3],
            },
            "passthrough_keys": ["input_ids"],
        },
        {
            "expected_inputs": {
                "prefix": [0],
                "input_ids": [0, 4, 5, 6, 1, 2],
                "target_labels": [4, 5, 6, 1, 2, 3],
            },
            "passthrough_keys": ["id", "input_ids"],
        },
        {
            "expected_inputs": {
                "prefix": [0],
                "input_ids": [0, 4, 5, 6, 1, 2],
                "target_labels": [4, 5, 6, 1, 2, 3],
            },
            "passthrough_keys": ["missing_key"],
        },
    )
    def test_make_autoregressive_inputs_passthrough_keys(self, expected_inputs, passthrough_keys):
        batch_size = 1
        examples = [
            {
                "target_labels": tf.constant([1, 2, 3]),
                "input_ids": tf.constant([4, 5, 6]),
                "prefix": tf.constant([0]),
                "id": tf.constant([7, 8, 9]),
            },
        ]

        def ds_fn():
            return fake_source(
                is_training=False,
                examples=examples,
            )

        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=False,
            source=config_for_function(ds_fn),
            processor=config_for_function(make_autoregressive_inputs).set(
                model_type=ModelType.DECODER_ONLY,
                input_data_type=InputDataType.SEQ2SEQ_NO_MASK,
                passthrough_keys=passthrough_keys,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            standard_keys = ["prefix", "input_ids", "target_labels"]

            # Ensure standard keys equal expected.
            for k in standard_keys:
                if k not in passthrough_keys:
                    self.assertEqual(batch[k][0].tolist(), expected_inputs[k])

            # Ensure passthrough keys are not modified.
            for k in passthrough_keys:
                if k in examples[0]:
                    self.assertEqual(batch[k][0].tolist(), examples[0][k].numpy().tolist())
            break

    def test_preprocessing_dataset_with_decoder(self):
        """Tests that lm_text_preprocessor works with a dataset that uses a custom decoder."""
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        texts = ["hello world\n", "hello moon\n", "hello tiger\n", "hello dog\n", "hello cat\n"]

        def _test_builder_with_datadir(data_dir: str) -> tfds.core.DatasetBuilder:
            class TestBuilder(tfds.core.GeneratorBasedBuilder):
                """Test dataset builder."""

                name = "temptext"
                VERSION = tfds.core.Version("1.0.0")

                def _info(self) -> tfds.core.DatasetInfo:
                    return tfds.core.DatasetInfo(
                        builder=self,
                        description="test",
                        features=tfds.features.FeaturesDict({"text": tf.string}),
                    )

                def _split_generators(self, _: tfds.download.DownloadManager):
                    return {"train": self._generate_examples("train")}

                def _generate_examples(self, _):
                    for i, text in enumerate(texts):
                        yield i, {"text": text}

            return TestBuilder(data_dir=data_dir)

        def append_foo(tensor: tf.Tensor) -> tf.Tensor:
            return tf.constant(tf.strings.join([tensor, "foo"]), dtype=tf.string)

        def tfds_custom_decoder() -> dict[str, tfds.decode.Decoder]:
            @tfds.decode.make_decoder()
            def replace_field_value(example: tf.Tensor, _: tfds.features.text_feature.Text):
                return tf.py_function(append_foo, [example], tf.string)

            # pylint: disable=no-value-for-parameter
            return {"text": replace_field_value()}

        shuffle_buffer_size, max_sequence_length, window_size = 8, 6, 10
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = _test_builder_with_datadir(tmpdir)
            builder.download_and_prepare()  # Downloads to tmpdir.
            source = config_for_function(input_tf_data.tfds_dataset).set(
                dataset_name=builder.name,
                split="train",
                is_training=True,
                train_shuffle_buffer_size=shuffle_buffer_size,
                data_dir=tmpdir,
                decoders=config_for_function(tfds_custom_decoder),
            )
            preprocessor = config_for_function(lm_text_preprocessor).set(
                vocab_cfg=self.vocab_cfg,
                max_sequence_length=max_sequence_length,
                replace_newlines_with=self.newlines_replaced_with,
                window_size=window_size,
                max_padding_fraction=0.5,
                shuffle_buffer_size=shuffle_buffer_size,
            )
            dataset_fn = input_tf_data.with_processor(source=source, processor=preprocessor)
            # The following will raise a ValueError if shape information is lost.
            element: dict[str, tf.Tensor] = next(iter(dataset_fn()))
            assert "input_ids" in element


class LmEvalInputTest(BaseLmInputTest):
    @property
    def vocab_cfg(self) -> InstantiableConfig:
        return config_for_class(seqio.SentencePieceVocabulary).set(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )

    @parameterized.parameters(
        ("How long is a piece of string?", "index"),
        ("On the 20th of June", "not_index"),
        ("Here we stand united", None),
    )
    def test_eval_lm_processor_single_example(self, text, index_key):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        max_len = 12
        processor = self._lm_eval_processor_config(
            vocab_cfg=self.vocab_cfg, max_len=max_len, stride=None, index_key="index"
        ).instantiate()
        ds_fn = (
            config_for_function(make_ds_fn)
            .set(is_training=False, texts=[text], repeat=1)
            .instantiate()
        )
        example = next(iter(processor(ds_fn())))
        for key in ["input_ids", "target_labels"]:
            # Shape is as expected.
            self.assertEqual((max_len,), example[key].numpy().shape)
        self.assertTrue("target_num_bytes" in example)
        # Index should have been passed through only for set value of `index_key`.
        self.assertEqual(index_key == "index", index_key in example)

        input_ids, target_labels = example["input_ids"].numpy(), example["target_labels"].numpy()
        self.assertEqual(1, input_ids[0])  # Start of example.
        non_padded_length = target_labels.argmin()
        self.assertNotEqual(1, target_labels[0])  # No EOS at start.
        self.assertEqual(1, target_labels[non_padded_length - 1])  # EOS.
        # The inputs should be one-off the labels.
        self.assertNestedAllClose(
            target_labels[: non_padded_length - 1], input_ids[1:non_padded_length]
        )

    def test_fake_text_lm_eval_data(self):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        texts = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit\n",
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        ]
        expected_batches = [
            {
                "input_ids": [
                    [1, 8410, 15, 51, 3, 15432, 440, 103, 322, 2561, 3, 9],
                    [15, 51, 3, 15432, 440, 103, 322, 2561, 3, 9, 3493, 6],
                    [3, 15432, 440, 103, 322, 2561, 3, 9, 3493, 6, 975, 7549],
                ],
                "target_labels": [
                    [8410, 15, 51, 3, 15432, 440, 103, 322, 2561, 3, 9, 3493],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 975],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7549, 17],
                ],
                "target_num_bytes": [26, 5, 4],
            },
            {
                "input_ids": [
                    [440, 103, 322, 2561, 3, 9, 3493, 6, 975, 7549, 17, 15],
                    [322, 2561, 3, 9, 3493, 6, 975, 7549, 17, 15, 2905, 3],
                    [3, 9, 3493, 6, 975, 7549, 17, 15, 2905, 3, 9, 21981],
                ],
                "target_labels": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 2905],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21981, 159],
                ],
                "target_num_bytes": [4, 1, 5],
            },
        ]
        self._test_fake_text_lm_eval_data(
            texts=texts, vocab_cfg=self.vocab_cfg, expected_batches=expected_batches
        )


class Seq2SeqInputTest(parameterized.TestCase, tf.test.TestCase):
    @property
    def vocab_cfg(self) -> InstantiableConfig:
        return config_for_class(seqio.SentencePieceVocabulary).set(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )

    @parameterized.parameters(
        # Encoder-decoder.
        {
            "is_training": False,
            "model_type": ModelType.ENCODER_DECODER,
            "max_len": 6,
            "expected_inputs": dict(
                source=dict(
                    input_ids=tf.constant(
                        [
                            [21820, 1, 0, 0, 0, 0],
                            [21820, 1, 0, 0, 0, 0],
                            [21820, 1, 0, 0, 0, 0],
                        ]
                    ),
                    positions=tf.constant(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ),
                    input_segment_ids=tf.constant(
                        [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
                    ),
                ),
                target=dict(
                    input_ids=tf.constant(
                        [
                            [1, 21820, 296, 0, 0, 0],
                            [1, 21820, 8114, 0, 0, 0],
                            [1, 21820, 3, 17, 4424, 0],
                        ]
                    ),
                    positions=tf.constant(
                        [[0, 1, 2, 0, 0, 0], [0, 1, 2, 0, 0, 0], [0, 1, 2, 3, 4, 0]]
                    ),
                    input_segment_ids=tf.constant(
                        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]
                    ),
                ),
                target_labels=tf.constant(
                    [
                        [21820, 296, 1, -1, -1, -1],
                        [21820, 8114, 1, -1, -1, -1],
                        [21820, 3, 17, 4424, 1, -1],
                    ]
                ),
                prefix=tf.constant([[1], [1], [1]]),
            ),
            "source_key": "source_ids",
            "target_key": "target_labels",
        },
        # Truncation test. Note that truncated targets do not end in EOS.
        {
            "is_training": False,
            "model_type": ModelType.ENCODER_DECODER,
            "max_len": 4,
            "expected_inputs": dict(
                source=dict(
                    input_ids=tf.constant(
                        [
                            [21820, 1, 0, 0],
                            [21820, 1, 0, 0],
                            [21820, 1, 0, 0],
                        ]
                    ),
                    positions=tf.constant([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),
                    input_segment_ids=tf.constant([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]),
                ),
                target=dict(
                    input_ids=tf.constant(
                        [
                            [1, 21820, 296, 0],
                            [1, 21820, 8114, 0],
                            [1, 21820, 3, 17],
                        ]
                    ),
                    positions=tf.constant([[0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 3]]),
                    input_segment_ids=tf.constant([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1]]),
                ),
                target_labels=tf.constant(
                    [
                        [21820, 296, 1, -1],
                        [21820, 8114, 1, -1],
                        [21820, 3, 17, 4424],
                    ]
                ),
                prefix=tf.constant([[1], [1], [1]]),
            ),
        },
        # Source truncation test. Note that truncated sources do not end in EOS.
        {
            "is_training": False,
            "model_type": ModelType.ENCODER_DECODER,
            "max_len": 1,
            "expected_inputs": dict(
                source=dict(
                    input_ids=tf.constant([[21820], [21820], [21820]]),
                    positions=tf.constant([[0], [0], [0]]),
                    input_segment_ids=tf.constant([[1], [1], [1]]),
                ),
                target=dict(
                    input_ids=tf.constant([[1], [1], [1]]),
                    positions=tf.constant([[0], [0], [0]]),
                    input_segment_ids=tf.constant([[1], [1], [1]]),
                ),
                target_labels=tf.constant([[21820], [21820], [21820]]),
                prefix=tf.constant([[1], [1], [1]]),
            ),
        },
        # Prefix test.
        {
            "is_training": False,
            "model_type": ModelType.ENCODER_DECODER,
            "max_len": 6,
            "expected_inputs": dict(
                source=dict(
                    input_ids=tf.constant(
                        [
                            [21820, 1, 0, 0, 0, 0],
                            [21820, 1, 0, 0, 0, 0],
                            [21820, 1, 0, 0, 0, 0],
                        ]
                    ),
                    positions=tf.constant(
                        [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
                    ),
                    input_segment_ids=tf.constant(
                        [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
                    ),
                ),
                target=dict(
                    input_ids=tf.constant(
                        [
                            [1, 1, 1, 21820, 296, 0],
                            [1, 1, 1, 21820, 8114, 0],
                            [1, 1, 1, 21820, 3, 17],
                        ]
                    ),
                    positions=tf.constant(
                        [[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 5]]
                    ),
                    input_segment_ids=tf.constant(
                        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
                    ),
                ),
                target_labels=tf.constant(
                    [
                        [-1, -1, 21820, 296, 1, -1],
                        [-1, -1, 21820, 8114, 1, -1],
                        [-1, -1, 21820, 3, 17, 4424],
                    ]
                ),
                prefix=tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            ),
        },
        # Decoder-only.
        {
            "is_training": False,
            "model_type": ModelType.DECODER_ONLY,
            "max_len": 7,
            "expected_inputs": dict(
                input_ids=tf.constant(
                    [
                        [1, 21820, 21820, 296, 0, 0, 0],
                        [1, 21820, 21820, 8114, 0, 0, 0],
                        [1, 21820, 21820, 3, 17, 4424, 0],
                    ]
                ),
                input_segment_ids=tf.constant(
                    [
                        [1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0],
                    ]
                ),
                input_positions=tf.constant(
                    [
                        [0, 1, 2, 3, 0, 0, 0],
                        [0, 1, 2, 3, 0, 0, 0],
                        [0, 1, 2, 3, 4, 5, 0],
                    ]
                ),
                target_labels=tf.constant(
                    [
                        [-1, 21820, 296, 1, -1, -1, -1],
                        [-1, 21820, 8114, 1, -1, -1, -1],
                        [-1, 21820, 3, 17, 4424, 1, -1],
                    ]
                ),
                prefix=tf.constant([[1], [1], [1]]),
            ),
        },
        # is_training as True
        {
            "is_training": True,
            "model_type": ModelType.DECODER_ONLY,
            "max_len": 6,
            "expected_inputs": dict(
                input_ids=tf.constant(
                    [
                        [1, 21820, 21820, 296, 0, 0],
                        [1, 21820, 21820, 8114, 0, 0],
                        [1, 21820, 21820, 3, 17, 4424],
                    ]
                ),
                input_segment_ids=tf.constant(
                    [
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1],
                    ]
                ),
                input_positions=tf.constant(
                    [
                        [0, 1, 2, 3, 0, 0],
                        [0, 1, 2, 3, 0, 0],
                        [0, 1, 2, 3, 4, 5],
                    ]
                ),
                target_labels=tf.constant(
                    [
                        [-1, 21820, 296, 1, -1, -1],
                        [-1, 21820, 8114, 1, -1, -1],
                        [-1, 21820, 3, 17, 4424, 1],
                    ]
                ),
                prefix=tf.constant([[1], [1], [1]]),
            ),
        },
        # Prefix test.
        {
            "is_training": False,
            "model_type": ModelType.DECODER_ONLY,
            "max_len": 7,
            "expected_inputs": dict(
                input_ids=tf.constant(
                    [
                        [1, 1, 21820, 21820, 296, 0, 0],
                        [1, 1, 21820, 21820, 8114, 0, 0],
                        [1, 1, 21820, 21820, 3, 17, 4424],
                    ]
                ),
                input_segment_ids=tf.constant(
                    [
                        [1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1],
                    ]
                ),
                input_positions=tf.constant(
                    [
                        [0, 1, 2, 3, 4, 0, 0],
                        [0, 1, 2, 3, 4, 0, 0],
                        [0, 1, 2, 3, 4, 5, 6],
                    ]
                ),
                target_labels=tf.constant(
                    [
                        [-1, -1, 21820, 296, 1, -1, -1],
                        [-1, -1, 21820, 8114, 1, -1, -1],
                        [-1, -1, 21820, 3, 17, 4424, 1],
                    ]
                ),
                prefix=tf.constant([[1, 1], [1, 1], [1, 1]]),
            ),
        },
    )
    def test_fake_text2text_lm_input(
        self,
        is_training: bool,
        model_type: str,
        max_len: int,
        expected_inputs: dict[str, list[list[int]]],
        source_key: str = "source",
        target_key: str = "target",
    ):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        batch_size = 3

        def ds_fn():
            return fake_source(
                is_training=is_training,
                examples=[
                    {
                        source_key: "hello",
                        target_key: "hello world",
                        "prefix": tf.constant(expected_inputs["prefix"][0]),
                    },
                    {
                        source_key: "hello",
                        target_key: "hello moon",
                        "prefix": tf.constant(expected_inputs["prefix"][1]),
                    },
                    {
                        source_key: "hello",
                        target_key: "hello tiger",
                        "prefix": tf.constant(expected_inputs["prefix"][2]),
                    },
                ],
            )

        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=is_training,
            source=config_for_function(ds_fn),
            processor=config_for_function(text2text_lm_input).set(
                target_sentence_piece_vocab=self.vocab_cfg,
                max_target_length=max_len,
                model_type=model_type,
                source_key=source_key,
                target_key=target_key,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            tf.nest.map_structure(self.assertAllEqual, expected_inputs, batch)
            break

    def test_original_keys(self):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")

        # Ensure that keys besides source_key, target_key are returned unchanged.
        processor_fn = text2text_lm_input(
            is_training=False,
            target_sentence_piece_vocab=self.vocab_cfg,
            max_target_length=3,
            source_key="source_text",
            target_key="target_labels",
        )
        examples = [
            {"source_text": "source", "target_labels": "target", "prefix": tf.constant([123])}
        ]
        ds = fake_source(is_training=False, examples=examples)()
        ds = processor_fn(ds)
        tf.nest.map_structure(
            self.assertAllEqual,
            next(iter(ds)),
            {
                "prefix": tf.constant([123]),
                "source": {
                    "input_ids": tf.constant([1391, 1, 0]),
                    "input_segment_ids": tf.constant([1, 1, 0]),
                    "positions": tf.constant([0, 1, 0]),
                },
                "target": {
                    # The 123 prefix from above is prepended.
                    "input_ids": tf.constant([123, 2387, 0]),
                    "input_segment_ids": tf.constant([1, 1, 0]),
                    "positions": tf.constant([0, 1, 0]),
                },
                "target_labels": tf.constant([2387, 1, -1]),
            },
        )

    @parameterized.parameters(
        # Note: Inputs will be detokenized into strings before feeding into the processor.
        # We keep them as pieces so that it's easier to visualize token lengths.
        #
        # Encoder-decoder: test when everything fits into one example.
        {
            "model_type": ModelType.ENCODER_DECODER,
            "max_target_len": 6,
            "inputs": [
                {
                    "source": ["▁short", "▁source"],
                    "target": ["▁short", "▁target"],
                    "prefix": tf.constant([1]),
                },
                {
                    "source": ["▁short", "▁source"],
                    "target": ["▁short", "▁target"],
                    "prefix": tf.constant([1]),
                },
            ],
            "expected": [
                {
                    "source": {
                        "input_ids": ["▁short", "▁source", "</s>", "▁short", "▁source", "</s>"],
                        "positions": [0, 1, 2, 0, 1, 2],
                        "input_segment_ids": [1, 1, 1, 2, 2, 2],
                    },
                    "target": {
                        "input_ids": ["</s>", "▁short", "▁target", "</s>", "▁short", "▁target"],
                        "positions": [0, 1, 2, 0, 1, 2],
                        "input_segment_ids": [1, 1, 1, 2, 2, 2],
                    },
                    "target_labels": ["▁short", "▁target", "</s>", "▁short", "▁target", "</s>"],
                }
            ],
        },
        # Encoder-decoder: test with source/target having different lengths.
        {
            "model_type": ModelType.ENCODER_DECODER,
            "max_target_len": 8,
            "inputs": [
                {
                    "source": ["▁short", "▁source"],
                    "target": ["▁some", "▁longer", "▁target"],
                    "prefix": tf.constant([1]),
                },
                {
                    "source": ["▁source"],
                    "target": ["▁short", "▁target"],
                    "prefix": tf.constant([1]),
                },
            ],
            "expected": [
                {
                    "source": {
                        "input_ids": [
                            "▁short",
                            "▁source",
                            "</s>",
                            "▁source",
                            "</s>",
                            "<pad>",
                            "<pad>",
                            "<pad>",
                        ],
                        "positions": [0, 1, 2, 0, 1, 0, 0, 0],
                        "input_segment_ids": [1, 1, 1, 2, 2, 0, 0, 0],
                    },
                    "target": {
                        "input_ids": [
                            "</s>",
                            "▁some",
                            "▁longer",
                            "▁target",
                            "</s>",
                            "▁short",
                            "▁target",
                            "<pad>",
                        ],
                        "positions": [0, 1, 2, 3, 0, 1, 2, 0],
                        "input_segment_ids": [1, 1, 1, 1, 2, 2, 2, 0],
                    },
                    "target_labels": [
                        "▁some",
                        "▁longer",
                        "▁target",
                        "</s>",
                        "▁short",
                        "▁target",
                        "</s>",
                        -1,
                    ],
                }
            ],
        },
        # Encoder-decoder: test that long inputs are truncated, not packed.
        {
            "model_type": ModelType.ENCODER_DECODER,
            "max_target_len": 2,
            "inputs": [
                {
                    "source": ["▁short"],
                    "target": ["▁some", "▁longer", "▁target"],
                    "prefix": tf.constant([1]),
                },
                {
                    "source": ["▁source"],
                    "target": ["▁short", "▁target"],
                    "prefix": tf.constant([1]),
                },
            ],
            "expected": [
                # First example. Note: "target_*" are truncated, and have no EOS.
                {
                    "source": {
                        "input_ids": ["▁short", "</s>"],
                        "positions": [0, 1],
                        "input_segment_ids": [1, 1],
                    },
                    "target": {
                        "input_ids": ["</s>", "▁some"],
                        "positions": [0, 1],
                        "input_segment_ids": [1, 1],
                    },
                    "target_labels": ["▁some", "▁longer"],
                },
                # Second example. Note: "target_*" are truncated, and have no EOS.
                {
                    "source": {
                        "input_ids": ["▁source", "</s>"],
                        "positions": [0, 1],
                        "input_segment_ids": [1, 1],
                    },
                    "target": {
                        # Note: "target_*" are truncated.
                        "input_ids": ["</s>", "▁short"],
                        "positions": [0, 1],
                        "input_segment_ids": [1, 1],
                    },
                    "target_labels": ["▁short", "▁target"],
                },
            ],
        },
        # Encoder-decoder: test a longer prefix.
        {
            "model_type": ModelType.ENCODER_DECODER,
            "max_target_len": 7,
            "inputs": [
                {
                    "source": ["▁short", "▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1, 1]),
                },
                {
                    "source": ["▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1, 1]),
                },
            ],
            "expected": [
                {
                    "source": {
                        "input_ids": [
                            "▁short",
                            "▁source",
                            "</s>",
                            "▁source",
                            "</s>",
                            "<pad>",
                            "<pad>",
                        ],
                        "positions": [0, 1, 2, 0, 1, 0, 0],
                        "input_segment_ids": [1, 1, 1, 2, 2, 0, 0],
                    },
                    "target": {
                        "input_ids": [
                            "</s>",
                            "</s>",
                            "▁target",
                            "</s>",
                            "</s>",
                            "▁target",
                            "<pad>",
                        ],
                        "positions": [0, 1, 2, 0, 1, 2, 0],
                        "input_segment_ids": [1, 1, 1, 2, 2, 2, 0],
                    },
                    "target_labels": [
                        -1,
                        "▁target",
                        "</s>",
                        -1,
                        "▁target",
                        "</s>",
                        -1,
                    ],
                },
            ],
        },
        # Encoder-decoder: test different max_{source,target}_len.
        {
            "model_type": ModelType.ENCODER_DECODER,
            "max_source_len": 6,
            "max_target_len": 4,
            "inputs": [
                {
                    "source": ["▁some", "▁short", "▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1]),
                },
                {
                    "source": ["▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1]),
                },
            ],
            "expected": [
                {
                    "source": {
                        "input_ids": ["▁some", "▁short", "▁source", "</s>", "▁source", "</s>"],
                        "positions": [0, 1, 2, 3, 0, 1],
                        "input_segment_ids": [1, 1, 1, 1, 2, 2],
                    },
                    "target": {
                        "input_ids": ["</s>", "▁target", "</s>", "▁target"],
                        "positions": [0, 1, 0, 1],
                        "input_segment_ids": [1, 1, 2, 2],
                    },
                    "target_labels": ["▁target", "</s>", "▁target", "</s>"],
                },
            ],
        },
        # Encoder-decoder: test padding.
        {
            "model_type": ModelType.ENCODER_DECODER,
            "max_source_len": 6,
            "max_target_len": 4,
            "inputs": [
                {
                    "source": ["▁some", "▁short", "▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1]),
                },
                {
                    "source": ["▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1]),
                },
            ],
            "expected": [
                {
                    "prefix": tf.constant([1]),
                    "source": {
                        "input_ids": ["▁some", "▁short", "▁source", "</s>", "<pad>", "<pad>"],
                        "positions": [0, 1, 2, 3, 0, 0],
                        "input_segment_ids": [1, 1, 1, 1, 0, 0],
                    },
                    "target": {
                        "input_ids": ["</s>", "▁target", "<pad>", "<pad>"],
                        "positions": [0, 1, 0, 0],
                        "input_segment_ids": [1, 1, 0, 0],
                    },
                    "target_labels": ["▁target", "</s>", -1, -1],
                },
                {
                    "prefix": tf.constant([1]),
                    "source": {
                        "input_ids": ["▁source", "</s>", "<pad>", "<pad>", "<pad>", "<pad>"],
                        "positions": [0, 1, 0, 0, 0, 0],
                        "input_segment_ids": [1, 1, 0, 0, 0, 0],
                    },
                    "target": {
                        "input_ids": ["</s>", "▁target", "<pad>", "<pad>"],
                        "positions": [0, 1, 0, 0],
                        "input_segment_ids": [1, 1, 0, 0],
                    },
                    "target_labels": ["▁target", "</s>", -1, -1],
                },
            ],
            "packing_mode": "pad",
        },
        # Decoder: test packing.
        {
            "model_type": ModelType.DECODER_ONLY,
            "max_source_len": None,
            "max_target_len": 8,
            "inputs": [
                {
                    "source": ["▁short", "▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1]),
                },
                {
                    "source": ["▁source"],
                    "target": ["▁target"],
                    "prefix": tf.constant([1]),
                },
                {
                    "source": ["▁other", "▁source"],
                    "target": ["▁some", "▁short", "▁target"],
                    "prefix": tf.constant([1]),
                },
            ],
            "expected": [
                {
                    "input_ids": [
                        "</s>",
                        "▁short",
                        "▁source",
                        "▁target",
                        "</s>",
                        "▁source",
                        "▁target",
                        "<pad>",
                    ],
                    "input_positions": [0, 1, 2, 3, 0, 1, 2, 0],
                    "input_segment_ids": [1, 1, 1, 1, 2, 2, 2, 0],
                    "target_labels": [-1, -1, "▁target", "</s>", -1, "▁target", "</s>", -1],
                },
                {
                    "input_ids": [
                        "</s>",
                        "▁other",
                        "▁source",
                        "▁some",
                        "▁short",
                        "▁target",
                        "<pad>",
                        "<pad>",
                    ],
                    "input_positions": [0, 1, 2, 3, 4, 5, 0, 0],
                    "input_segment_ids": [1, 1, 1, 1, 1, 1, 0, 0],
                    "target_labels": [
                        -1,
                        -1,
                        "▁some",
                        "▁short",
                        "▁target",
                        "</s>",
                        -1,
                        -1,
                    ],
                },
            ],
            "packing_mode": "pack",
        },
    )
    def test_packing(
        self,
        *,
        model_type: ModelType,
        inputs: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        max_target_len: int,
        max_source_len: Optional[int] = None,
        packing_mode: Literal["pack", "pad", "none"] = "pack",
    ):
        if not os.path.exists(t5_sentence_piece_vocab_file):
            self.skipTest("Missing testdata.")
        vocab = self.vocab_cfg.instantiate()
        source = fake_source(
            is_training=False,
            examples=[
                {
                    "source": tf.constant(vocab.tokenizer.decode_pieces(ex["source"])),
                    "target": tf.constant(vocab.tokenizer.decode_pieces(ex["target"])),
                    "prefix": ex["prefix"],
                }
                for ex in inputs
            ],
        )
        processor = text2text_lm_input(
            is_training=False,
            model_type=model_type,
            target_sentence_piece_vocab=self.vocab_cfg,
            max_target_length=max_target_len,
            max_source_length=max_source_len,
            source_key="source",
            target_key="target",
            packing_mode=packing_mode,
        )
        ds = processor(source())
        self.assertEqual(len(expected), len(list(ds)))

        def to_tensor(val: dict):
            for k, v in val.items():
                if isinstance(v, dict):
                    to_tensor(v)
                    continue
                if v and isinstance(v, list) and any(isinstance(x, str) for x in v):
                    # Detokenize.
                    v = [vocab.tokenizer.piece_to_id(x) if isinstance(x, str) else x for x in v]
                val[k] = tf.constant(v)

        for expect, actual in zip(expected, ds):
            # Convert pieces back to tf.Tensor of IDs.
            to_tensor(expect)
            tf.nest.map_structure(self.assertAllEqual, expect, actual)

    @parameterized.parameters(
        # Test packing with intermediate zeros.
        {
            "max_source_len": 8,
            "max_target_len": 6,
            "inputs": [
                {"source_ids": [1, 0, 2, 3], "target_ids": [0, 0, 1]},
                {"source_ids": [0, 0, 1], "target_ids": [2]},
                # Note: all 0's are treated as non-zero.
                {"source_ids": [0, 0, 0], "target_ids": [0, 3]},
                # Note: trailing padding is not discarded.
                {"source_ids": [1, 0], "target_ids": [0, 1]},
                {"source_ids": [2, 3], "target_ids": []},
            ],
            "expected": [
                {
                    "source_ids": [1, 0, 2, 3, 0, 0, 1, 0],
                    "source_ids_positions": [0, 1, 2, 3, 0, 1, 2, 0],
                    "source_ids_segment_ids": [1, 1, 1, 1, 2, 2, 2, 0],
                    "target_ids": [0, 0, 1, 2, 0, 0],
                    "target_ids_positions": [0, 1, 2, 0, 0, 0],
                    "target_ids_segment_ids": [1, 1, 1, 2, 0, 0],
                },
                {
                    "source_ids": [0, 0, 0, 1, 0, 2, 3, 0],
                    "source_ids_positions": [0, 1, 2, 0, 1, 0, 1, 0],
                    "source_ids_segment_ids": [1, 1, 1, 2, 2, 3, 3, 0],
                    "target_ids": [0, 3, 0, 1, 0, 0],
                    "target_ids_positions": [0, 1, 0, 1, 0, 0],
                    "target_ids_segment_ids": [1, 1, 2, 2, 0, 0],
                },
            ],
        },
    )
    def test_trim_and_pack_with_segments(
        self,
        *,
        inputs: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        max_target_len: int,
        max_source_len: Optional[int] = None,
    ):
        source = fake_source(
            is_training=False,
            examples=[
                {
                    "source_ids": tf.constant(ex["source_ids"], dtype=tf.int32),
                    "target_ids": tf.constant(ex["target_ids"], dtype=tf.int32),
                }
                for ex in inputs
            ],
            spec={
                "source_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
                "target_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
            },
        )
        processor = _trim_and_pack_with_segments(
            {"source_ids": max_source_len, "target_ids": max_target_len}
        )
        ds = list(processor(source()))
        self.assertEqual(len(expected), len(ds))
        for expect, actual in zip(expected, ds):
            self.assertEqual(expect.keys(), actual.keys())
            for k, v in expect.items():
                self.assertAllEqual(tf.constant(v), actual[k])


if __name__ == "__main__":
    absltest.main()

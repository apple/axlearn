# Copyright © 2023 Apple Inc.

"""Tests GLUE inputs."""

# pylint: disable=no-self-use
import os
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pytest
import seqio
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common import input_glue, input_tf_data, utils
from axlearn.common.config import InstantiableConfig, config_for_class, config_for_function
from axlearn.common.input_fake import fake_source
from axlearn.common.input_test_utils import t5_sentence_piece_vocab_file


def _source_cfg(
    examples: Sequence[dict[str, tf.Tensor]], output_signature: dict[str, tf.TensorShape]
) -> InstantiableConfig:
    def ds_fn(is_training: bool):
        return fake_source(is_training=is_training, examples=examples, spec=output_signature)

    return config_for_function(ds_fn)


def _vocab_cfg() -> InstantiableConfig:
    return config_for_class(seqio.SentencePieceVocabulary).set(
        sentencepiece_model_file=t5_sentence_piece_vocab_file,
    )


class InputGlueForRobertaTest(parameterized.TestCase, tf.test.TestCase):
    """Test GLUE inputs for RoBERTa."""

    def _glue_processor_config(
        self,
        *,
        max_len: int,
        input_key: Union[str, tuple[str, str]],
        vocab_cfg: InstantiableConfig = _vocab_cfg(),
        normalization: Optional[InstantiableConfig] = None,
        bos_token: Optional[str] = None,
    ):
        def noop_normalizer(
            # pylint: disable-next=unused-argument
            input_key: Union[str, tuple[str, str]],
        ) -> input_tf_data.DatasetToDatasetFn:
            return lambda ds: ds

        num_inputs = 1 if isinstance(input_key, str) else 2
        return config_for_function(input_glue.text_to_glue_input).set(
            input_key=input_key,
            max_len=max_len,
            vocab_cfg=vocab_cfg,
            normalization=normalization or config_for_function(noop_normalizer),
            truncation=config_for_function(input_glue.multi_sequence_truncation).set(
                # Subtract 2 for BOS/EOS, and potentially 2 more for EOS-separators.
                max_len=(max_len - 2 * num_inputs),
            ),
            concatenation=config_for_function(input_glue.add_special_tokens_for_roberta).set(
                bos_token=bos_token
            ),
        )

    def _glue_input_config(
        self,
        is_training: bool,
        input_key: Union[str, tuple[str, str]],
        max_len: int,
        batch_size: int,
        source_cfg: InstantiableConfig,
        vocab_cfg: InstantiableConfig = _vocab_cfg(),
        normalization: Optional[InstantiableConfig] = None,
        bos_token: Optional[str] = None,
    ):
        return input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=is_training,
            source=source_cfg,
            processor=self._glue_processor_config(
                input_key=input_key,
                max_len=max_len,
                normalization=normalization,
                bos_token=bos_token,
                vocab_cfg=vocab_cfg,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

    def test_input_key(self):
        with self.assertRaisesRegex(ValueError, "input_key"):
            cfg = self._glue_processor_config(max_len=1, input_key=[])  # type: ignore
            cfg.set(is_training=True).instantiate()

    @parameterized.parameters(
        {"is_training": False, "bos_token": None},
        {"is_training": True, "bos_token": None},
        {"is_training": False, "bos_token": "▁test"},
        {"is_training": True, "bos_token": "▁test"},
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_fake_sentence(self, is_training, bos_token):
        examples = [
            {
                "sentence": "this is the first test",
                "label": 1,
            },
            {
                "sentence": "this is a very long first test to test truncation",
                "label": 1,
            },
            {"sentence": "", "label": 1},
        ]
        vocab_cfg = _vocab_cfg()
        vocab = vocab_cfg.instantiate()
        batch_size, max_len = len(examples), 10
        cfg = self._glue_input_config(
            vocab_cfg=vocab_cfg,
            is_training=is_training,
            source_cfg=_source_cfg(
                examples=examples,
                output_signature={
                    "sentence": tf.TensorSpec(shape=(), dtype=tf.string),
                    "label": tf.TensorSpec(shape=(), dtype=tf.int32),
                },
            ),
            input_key="sentence",
            max_len=max_len,
            batch_size=batch_size,
            bos_token=bos_token,
        )
        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            self.assertEqual(
                {"input_ids": (batch_size, max_len), "target_labels": (batch_size,)},
                utils.shapes(batch),
            )
            if not is_training:
                bos_id = vocab.tokenizer.piece_to_id(bos_token) if bos_token else 1
                np.testing.assert_array_equal(
                    batch["input_ids"],
                    [
                        [bos_id, 48, 19, 8, 166, 794, 1, 0, 0, 0],
                        [bos_id, 48, 19, 3, 9, 182, 307, 166, 794, 1],
                        [bos_id, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                )
            break

    @parameterized.parameters(False, True)
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_fake_sentence_pair(self, is_training):
        examples = [
            {
                "sentence1": "this is the first test",
                "sentence2": "this is the second test",
                "label": 1,
            },
            {
                "sentence1": "this is a very long sentence to test truncation",
                "sentence2": "another very long sentence",
                "label": 1,
            },
            {"sentence1": "", "sentence2": "this is the second test", "label": 1},
            {"sentence1": "", "sentence2": "", "label": 1},
        ]
        batch_size, max_len = len(examples), 10
        cfg = self._glue_input_config(
            is_training=is_training,
            source_cfg=_source_cfg(
                examples=examples,
                output_signature={
                    "sentence1": tf.TensorSpec(shape=(), dtype=tf.string),
                    "sentence2": tf.TensorSpec(shape=(), dtype=tf.string),
                    "label": tf.TensorSpec(shape=(), dtype=tf.int32),
                },
            ),
            input_key=["sentence1", "sentence2"],
            max_len=max_len,
            batch_size=batch_size,
        )
        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            self.assertEqual(
                {"input_ids": (batch_size, max_len), "target_labels": (batch_size,)},
                utils.shapes(batch),
            )
            np.testing.assert_array_equal(
                batch["input_ids"],
                [
                    [1, 48, 19, 8, 1, 1, 48, 19, 8, 1],
                    [1, 48, 19, 3, 1, 1, 430, 182, 307, 1],
                    [1, 1, 1, 48, 19, 8, 511, 794, 1, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                ],
            )
            break

    @parameterized.parameters(False, True)
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_fake_multiple_choice_preprocess(self, is_training):
        # Examples from COPA.
        examples = [
            {
                "premise": "I picked up my belongings.",
                "question": "cause",
                "choice1": "I was hunting for a new apartment.",
                "choice2": "I was moving out of my apartment.",
                "label": 1,  # Choice 2.
            },
            {
                "premise": "The girl applied the scissors to the paper.",
                "question": "effect",
                "choice1": "The paper sliced apart.",
                "choice2": "The paper crinkled.",
                "label": 0,  # Choice 1.
            },
        ]
        # Note that batch_size is 2x due to preprocessing.
        batch_size, max_len = len(examples) * 2, 32
        cfg = self._glue_input_config(
            is_training=is_training,
            source_cfg=_source_cfg(
                examples=examples,
                output_signature={
                    "premise": tf.TensorSpec(shape=(), dtype=tf.string),
                    "question": tf.TensorSpec(shape=(), dtype=tf.string),
                    "choice1": tf.TensorSpec(shape=(), dtype=tf.string),
                    "choice2": tf.TensorSpec(shape=(), dtype=tf.string),
                    "label": tf.TensorSpec(shape=(), dtype=tf.int32),
                },
            ),
            normalization=config_for_function(input_tf_data.chain).set(
                args=[
                    input_glue.preprocess_copa(),
                    input_tf_data.unbatch(),
                ],
            ),
            input_key=["premise", "choice"],
            max_len=max_len,
            batch_size=batch_size,
        )
        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            self.assertEqual(
                {"input_ids": (batch_size, max_len), "target_labels": (batch_size,)},
                utils.shapes(batch),
            )
            np.testing.assert_array_equal(
                batch["input_ids"],
                # pylint: disable=line-too-long
                # fmt: off
                [
                    [
                        # "I picked up my belongings. What's the CAUSE for this?"
                        1, 27, 4758, 95, 82, 12770, 7, 5, 363, 31, 7, 8, 3087, 11927, 21, 48, 58, 1, 1,
                        # "I was hunting for a new apartment."
                        27, 47, 9601, 21, 3, 9, 126, 4579, 5, 1, 0, 0, 0,
                    ],
                    [
                        1, 27, 4758, 95, 82, 12770, 7, 5, 363, 31, 7, 8, 3087, 11927, 21, 48, 58, 1, 1,
                        # "I was moving out of my apartment."
                        27, 47, 1735, 91, 13, 82, 4579, 5, 1, 0, 0, 0, 0,
                    ],
                    [
                        # "The girl applied the scissors to the paper. What happened as a RESULT?"
                        1, 37, 3202, 2930, 8, 28958, 12, 8, 1040, 5, 363, 2817, 38, 3, 9, 4083, 4138, 9012, 58, 1, 1,
                        # "The paper sliced apart."
                        37, 1040, 3, 23645, 3943, 5, 1, 0, 0, 0, 0,
                    ],
                    [
                        1, 37, 3202, 2930, 8, 28958, 12, 8, 1040, 5, 363, 2817, 38, 3, 9, 4083, 4138, 9012, 58, 1, 1,
                        # "The paper crinkled."
                        37, 1040, 3, 75, 13419, 1361, 5, 1, 0, 0, 0,
                    ],
                ],
                # fmt: on
                # pylint: enable=line-too-long
            )
            break

    def test_fake_multiple_choice_postprocess(self):
        examples = [
            {"input_ids": tf.constant([1, 2, 3]), "target_labels": tf.constant(0)},
            {"input_ids": tf.constant([4, 5, 6]), "target_labels": tf.constant(0)},
            {"input_ids": tf.constant([7, 8, 9]), "target_labels": tf.constant(1)},
            {"input_ids": tf.constant([10, 11, 12]), "target_labels": tf.constant(1)},
            {"input_ids": tf.constant([13, 14, 15]), "target_labels": tf.constant(2)},
            {"input_ids": tf.constant([16, 17, 18]), "target_labels": tf.constant(2)},
        ]
        expected = [
            {"input_ids": tf.constant([[1, 2, 3], [4, 5, 6]]), "target_labels": tf.constant(0)},
            {"input_ids": tf.constant([[7, 8, 9], [10, 11, 12]]), "target_labels": tf.constant(1)},
            {
                "input_ids": tf.constant([[13, 14, 15], [16, 17, 18]]),
                "target_labels": tf.constant(2),
            },
        ]
        source = fake_source(is_training=False, examples=examples)
        processor = input_glue.postprocess_copa()
        ds = processor(source())

        # Check against expected.
        self.assertEqual(len(expected), len(list(ds)))
        for expect, actual in zip(expected, ds):
            tf.nest.map_structure(self.assertAllEqual, expect, actual)

        # Batching should produce "input_ids" of shape (batch_size, 2, seq_len).
        batch_size, seq_len = 3, 3
        ds = ds.batch(batch_size)
        batch = next(iter(ds))
        self.assertEqual(batch["input_ids"].numpy().shape, (batch_size, 2, seq_len))
        self.assertEqual(batch["target_labels"].numpy().shape, (batch_size,))

        # If grouped examples have different labels, we should fail.
        examples = [
            {"input_ids": tf.constant([1, 2, 3]), "target_labels": tf.constant(1)},
            {"input_ids": tf.constant([4, 5, 6]), "target_labels": tf.constant(0)},
        ]
        source = fake_source(is_training=False, examples=examples)
        processor = input_glue.postprocess_copa()
        ds = processor(source())
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "invalid targets"):
            next(iter(ds))


class TestUtils(parameterized.TestCase, tf.test.TestCase):
    """Tests GLUE input utils."""

    @parameterized.parameters(
        {
            "max_len": 4,
            "examples": [
                {"input1": [1, 2, 3], "input2": [4, 5, 6]},
                {"input1": [1, 2, 3, 4, 5, 6], "input2": [1, 2, 3, 4, 5]},
                {"input1": [1, 2], "input2": [3, 4]},
                {"input1": [], "input2": [5, 6, 7]},
                {"input1": [1, 2, 3, 4, 5, 6], "input2": [1, 2, 3, 4]},
                {"input1": [], "input2": []},
            ],
            "expected": [
                {"input1": [1, 2], "input2": [4, 5]},
                {"input1": [1, 2], "input2": [1, 2]},
                {"input1": [1, 2], "input2": [3, 4]},
                {"input1": [], "input2": [5, 6, 7]},
                {"input1": [1, 2], "input2": [1, 2]},
                {"input1": [], "input2": []},
            ],
        },
        {
            "max_len": 3,
            "examples": [
                {"input1": [1, 2], "input2": [3, 4]},
                {"input1": [1, 2], "input2": [3, 4, 5]},
                {"input1": [1, 2, 3], "input2": [3, 4]},
                {"input1": [1, 2, 3], "input2": [3]},
                {"input1": [1, 2], "input2": [3]},
            ],
            "expected": [
                {"input1": [1, 2], "input2": [3]},
                {"input1": [1], "input2": [3, 4]},
                {"input1": [1, 2], "input2": [3]},
                {"input1": [1, 2], "input2": [3]},
                {"input1": [1, 2], "input2": [3]},
            ],
        },
        {
            "max_len": 8,
            "examples": [
                {"input1": [1, 2, 3], "input2": [4, 5, 6], "input3": [7, 8, 9]},
                {
                    "input1": [1, 2, 3, 4, 5, 6],
                    "input2": [1, 2, 3, 4, 5],
                    "input3": [1, 2, 3, 4, 5],
                },
                {"input1": [1, 2], "input2": [3, 4], "input3": [5, 6]},
                {"input1": [], "input2": [5, 6, 7], "input3": []},
                {"input1": [1, 2, 3, 4, 5, 6], "input2": [1, 2, 3, 4], "input3": [5, 6, 7]},
                {"input1": [], "input2": [], "input3": []},
            ],
            "expected": [
                {"input1": [1, 2, 3], "input2": [4, 5, 6], "input3": [7, 8]},
                {"input1": [1, 2, 3], "input2": [1, 2, 3], "input3": [1, 2]},
                {"input1": [1, 2], "input2": [3, 4], "input3": [5, 6]},
                {"input1": [], "input2": [5, 6, 7], "input3": []},
                {"input1": [1, 2, 3], "input2": [1, 2, 3], "input3": [5, 6]},
                {"input1": [], "input2": [], "input3": []},
            ],
        },
        {
            "max_len": 10,
            "examples": [
                {
                    "input1": [1, 2, 3],
                    "input2": [4, 5, 6],
                    "input3": [7, 8, 9],
                    "input4": [7, 8, 9, 10],
                },
                {
                    "input1": [1, 2, 3, 4, 5, 6],
                    "input2": [1, 2, 3, 4, 5],
                    "input3": [1, 2, 3, 4, 5],
                    "input4": [1, 2, 3, 4, 5],
                },
                {"input1": [1, 2], "input2": [3, 4], "input3": [5, 6], "input4": [7, 8]},
                {"input1": [], "input2": [5, 6, 7], "input3": [], "input4": [1, 2, 3]},
                {"input1": [], "input2": [], "input3": [], "input4": []},
            ],
            "expected": [
                {"input1": [1, 2, 3], "input2": [4, 5], "input3": [7, 8], "input4": [7, 8, 9]},
                {"input1": [1, 2, 3], "input2": [1, 2, 3], "input3": [1, 2], "input4": [1, 2]},
                {"input1": [1, 2], "input2": [3, 4], "input3": [5, 6], "input4": [7, 8]},
                {"input1": [], "input2": [5, 6, 7], "input3": [], "input4": [1, 2, 3]},
                {"input1": [], "input2": [], "input3": [], "input4": []},
            ],
        },
    )
    def test_multiple_sequence_truncation(
        self,
        max_len: int,
        examples: list[dict[str, Sequence[int]]],
        expected: list[dict[str, Sequence[int]]],
    ):
        input_keys = tuple(expected[0].keys())
        mapper_fn = input_glue.multi_sequence_truncation(
            max_len=max_len,
            input_key=input_keys,
        )
        output_signature = {}
        for input_key in input_keys:
            output_signature[input_key] = tf.TensorSpec(shape=[None], dtype=tf.int32)
        ds_fn = (
            _source_cfg(
                examples=examples,
                output_signature=output_signature,
            )
            .set(is_training=False)
            .instantiate()
        )
        ds = mapper_fn(ds_fn())

        for expect, actual in zip(expected, ds):
            for key in input_keys:
                self.assertSequenceEqual(expect[key], actual[key].numpy().tolist())

    @parameterized.parameters(
        {
            "examples": [
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "answers": tf.constant(["answer1", "answer3"]),
                    "entities": tf.constant(["answer1", "answer2", "answer3"]),
                }
            ],
            "expected": [
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "label": tf.constant(1),
                    "entity": tf.constant("answer1"),
                },
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "label": tf.constant(0),
                    "entity": tf.constant("answer2"),
                },
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "label": tf.constant(1),
                    "entity": tf.constant("answer3"),
                },
            ],
        },
        {
            "examples": [
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "answers": tf.constant([]),
                    "entities": tf.constant(["answer1", "answer2", "answer3"]),
                }
            ],
            "expected": [
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "label": tf.constant(0),
                    "entity": tf.constant("answer1"),
                },
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "label": tf.constant(0),
                    "entity": tf.constant("answer2"),
                },
                {
                    "passage": tf.constant("I am a passage."),
                    "query": tf.constant("I am a query"),
                    "label": tf.constant(0),
                    "entity": tf.constant("answer3"),
                },
            ],
        },
    )
    def test_preprocess_record(
        self,
        examples: list[dict[str, Sequence[int]]],
        expected: list[dict[str, Sequence[int]]],
    ):
        mapper_fn = input_glue.preprocess_record()
        ds_fn = (
            _source_cfg(
                examples=examples,
                output_signature={
                    "passage": tf.TensorSpec(shape=(), dtype=tf.string),
                    "query": tf.TensorSpec(shape=(), dtype=tf.string),
                    "answers": tf.TensorSpec(shape=(None,), dtype=tf.string),
                    "entities": tf.TensorSpec(shape=(None,), dtype=tf.string),
                },
            )
            .set(is_training=False)
            .instantiate()
        )
        ds = mapper_fn(ds_fn()).unbatch()
        for expect, actual in zip(expected, ds):
            for key in ["passage", "query", "entity", "label"]:
                self.assertAllEqual(expect[key], actual[key])


class InputGlueForT5Test(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        # Test a simple classification example.
        dict(
            input_example={
                "text1": "Hello",
                "text2": "World",
                "label": 0,
            },
            label_names=["zero", "one"],
            task_name="boolq",
            expected={
                "source_text": b"boolq text1: Hello text2: World",
                "target_text": b"zero",
            },
        ),
        # Test a simple regression example.
        dict(
            input_example={
                "text1": "Hello",
                "text2": "World",
                "label": 1.5,
            },
            label_names=["none"],
            task_name="stsb",
            expected={
                "source_text": b"stsb text1: Hello text2: World",
                # Note: STSB label is converted to (roughly) 26-class classification.
                # https://github.com/google-research/text-to-text-transfer-transformer/blob/50a797f3386d3985c2e387cc20626a6ac1483336/t5/data/preprocessors.py#L816-L855
                "target_text": b"1.6",
            },
        ),
        # Test a simple wsc example.
        dict(
            input_example={
                "text": "Joe paid the detective after he received the final report on the case.",
                "span1_index": 2,
                "span1_text": "the detective",
                "span2_index": 5,
                "span2_text": "he",
                "label": 0,
            },
            label_names=["False", "True"],
            task_name="wsc",
            expected={
                "source_text": b"wsc text: Joe paid * the detective * after # he # received the final report on the case.",  # pylint: disable=line-too-long
                "target_text": b"False",
            },
        ),
        # Test a simple ReCoRD example.
        dict(
            input_example={
                "idx": {
                    "passage": 123,
                    "query": 456,
                },
                "passage": "I am a sample passage.",
                "query": "I am a @placeholder query.",
                "answers": tf.constant(["A", "B"]),
                "entities": tf.constant(["A", "B", "C"]),
            },
            task_name="record",
            label_names=["none"],
            expected={
                "source_text": b"record query: I am a @placeholder query. entities: A, B, C passage: I am a sample passage.",  # pylint: disable=line-too-long
                "target_text": b"A",
                "idx": {
                    "passage": 123,
                    "query": 456,
                },
            },
        ),
        # Test unk label.
        dict(
            input_example={
                "text1": "Hello",
                "text2": "World",
                "label": -1,
            },
            label_names=["zero", "one"],
            task_name="cb",
            expected={
                "source_text": b"cb text1: Hello text2: World",
                "target_text": b"<unk>",
            },
        ),
    )
    def test_add_prefix_concat_sequence_pair(
        self,
        *,
        input_example: dict[str, Any],
        expected: dict[str, Any],
        label_names: Sequence[str],
        task_name: str,
    ):
        # "idx" and "prefix" are just passed through.
        if "idx" not in input_example:
            input_example["idx"] = expected["idx"] = 123
        expected["prefix"] = input_example["prefix"] = tf.constant([1])

        source = fake_source(is_training=False, examples=[input_example])
        processor = input_glue.add_prefix_concat_sequence_pair(
            dataset_name=task_name,
            input_key=("text1", "text2"),
            label_names=label_names,
        )
        for actual in processor(source()):
            tf.nest.map_structure(self.assertAllEqual, expected, actual)
            break

    @parameterized.parameters(
        # Each test ensures that:
        # 1. "source_ids" ends with EOS;
        # 2. "target_ids" begins with prefix;
        # 3. "target_labels" pads prefix, and ends with EOS.
        # Test a basic case with no truncation.
        dict(
            max_source_len=10,
            max_target_len=10,
            input_example=dict(
                prefix=tf.constant([100]),
                source_ids=tf.constant([10, 11, 12, 1]),
                target_labels=tf.constant([20, 21, 22, 23, 1]),
            ),
            expected=dict(
                source_ids=tf.constant([10, 11, 12, 1]),
                target_ids=tf.constant([100, 20, 21, 22, 23]),
                target_labels=tf.constant([20, 21, 22, 23, 1]),
            ),
        ),
        # Test a long prefix.
        dict(
            max_source_len=10,
            max_target_len=10,
            input_example=dict(
                prefix=tf.constant([100, 101, 102]),
                source_ids=tf.constant([10, 11, 12, 1]),
                target_labels=tf.constant([20, 21, 22, 23, 1]),
            ),
            expected=dict(
                source_ids=tf.constant([10, 11, 12, 1]),
                target_ids=tf.constant([100, 101, 102, 20, 21, 22, 23]),
                target_labels=tf.constant([-1, -1, 20, 21, 22, 23, 1]),
            ),
        ),
        # Test truncation, ensuring EOS is still present.
        dict(
            max_source_len=2,
            max_target_len=4,
            input_example=dict(
                prefix=tf.constant([100, 101, 102]),
                source_ids=tf.constant([10, 11, 12, 1]),
                target_labels=tf.constant([20, 21, 22, 23, 1]),
            ),
            expected=dict(
                # "source_ids" is truncated but keeps EOS.
                source_ids=tf.constant([10, 1]),
                # "target_ids" is truncated. It has no EOS.
                target_ids=tf.constant([100, 101, 102, 20]),
                # "target_labels" is prefix-padded, and keeps EOS.
                target_labels=tf.constant([-1, -1, 20, 1]),
            ),
        ),
        # Test prefix longer than max len.
        dict(
            max_source_len=2,
            max_target_len=2,
            input_example=dict(
                prefix=tf.constant([100, 101, 102]),
                source_ids=tf.constant([10, 11, 12, 1]),
                target_labels=tf.constant([20, 21, 22, 23, 1]),
            ),
            expected=dict(
                source_ids=tf.constant([10, 1]),
                target_ids=tf.constant([100, 101]),
                target_labels=tf.constant([-1, 1]),
            ),
        ),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_make_glue_autoregressive_inputs(
        self,
        *,
        input_example: dict[str, Any],
        expected: dict[str, Any],
        max_source_len: int,
        max_target_len: int,
    ):
        # "idx" and "prefix" are just passed through.
        input_example["idx"] = expected["idx"] = 123
        expected["prefix"] = input_example["prefix"]

        source = fake_source(is_training=False, examples=[input_example])
        processor = input_glue.make_glue_autoregressive_inputs(
            vocab_cfg=_vocab_cfg(),
            max_source_length=max_source_len,
            max_target_length=max_target_len,
        )
        for actual in processor(source()):
            tf.nest.map_structure(self.assertAllEqual, expected, actual)
            break


if __name__ == "__main__":
    absltest.main()

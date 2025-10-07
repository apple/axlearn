# Copyright © 2023 Apple Inc.

"""Tests text input processing."""

# pylint: disable=no-self-use
import os
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import numpy as np
import pytest
import seqio
import tensorflow as tf
from absl.testing import absltest, parameterized
from seqio import SentencePieceVocabulary
from transformers.models.bert.tokenization_bert import BasicTokenizer

from axlearn.common import input_text, input_tf_data, test_utils
from axlearn.common.config import InstantiableConfig, config_for_class, config_for_function
from axlearn.common.input_test_utils import (
    assert_oneof,
    extract_text,
    extract_text_ragged,
    make_ds_fn,
    make_ragged_ds_fn,
    make_seq2seq_ds_fn,
    opt_vocab_file,
    t5_sentence_piece_vocab_file,
)
from axlearn.common.input_text import TOKEN_TYPE_IDS, add_token_type_ids, strip_accents, tokenize
from axlearn.common.vocabulary_bpe import BPEVocabulary

_T5_VOCAB_FILE = t5_sentence_piece_vocab_file
_OPT_VOCAB_FILE = opt_vocab_file


class StripAccentsTest(test_utils.TestCase):
    @parameterized.product(
        (
            {"original": "El Capitán", "expected": "El Capitan"},
            {
                "original": "dômes granitiques spectaculaires",
                "expected": "domes granitiques spectaculaires",
            },
            {"original": "Yosemite", "expected": "Yosemite"},
            {"original": "优胜美地", "expected": "优胜美地"},
        ),
        normalization_form=("NFD", "NFKD"),
    )
    def test_strip_accents(self, original: str, expected: str, normalization_form: str):
        def gen():
            yield dict(text=original)

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
            },
        )

        no_accent_ds = strip_accents(fields=["text"], normalization_form=normalization_form)
        actual = next(no_accent_ds(ds).as_numpy_iterator())["text"].decode("utf-8")
        self.assertEqual(actual, expected)

    @parameterized.parameters("NFC", "NFKC")
    def test_strip_accents_invalid_normalization(self, normalization_form: str):
        with self.assertRaises(AssertionError):
            strip_accents(fields=["text"], normalization_form=normalization_form)


class AddTokenTypeIDTest(test_utils.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_queries = ["Where is Yosemite?", "Where is Zion?", "Where is Mt. Rainier?"]
        self.sample_answers = [
            "Central Sierra Nevada in California",
            "Southwestern Utah near the town of Springdale",
            "59 miles southeast of Seattle",
        ]
        self.sample_urls = ["en.wikipedia.org", "utah.com", "nps.gov"]

    def create_source_ds(self, num_examples: int) -> tf.data.Dataset:
        assert 1 <= num_examples <= 3

        def gen():
            yield dict(
                query=self.sample_queries[:num_examples],
                answer=self.sample_answers[:num_examples],
                url=self.sample_urls[:num_examples],
            )

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "query": tf.TensorSpec(shape=(num_examples,), dtype=tf.string),
                "answer": tf.TensorSpec(shape=(num_examples,), dtype=tf.string),
                "url": tf.TensorSpec(shape=(num_examples,), dtype=tf.string),
            },
        )
        return ds

    @parameterized.parameters(
        # For ragged tensors, query has 8 tokens, answer has 5 tokens, and url has 6 tokens.
        {
            "input_key": "query",
            "expected": [0] + [0] * 8 + [0],
            "is_ragged": True,
        },
        {
            "input_key": ["query", "answer"],
            "expected": [0] + [0] * 8 + [0] + [1] * 5 + [1],
            "is_ragged": True,
        },
        {
            "input_key": ["query", "answer", "url"],
            "expected": [0] + [0] * 8 + [0] + [1] * 5 + [1] + [2] * 6 + [2],
            "is_ragged": True,
        },
        # For non-ragged tensors, we truncate to 3 tokens per field.
        {
            "input_key": "query",
            "expected": [0] + [0] * 3 + [0],
            "is_ragged": False,
            "feature_lengths": dict(query=3),
        },
        {
            "input_key": ["query", "answer"],
            "expected": [0] + [0] * 3 + [0] + [1] * 3 + [1],
            "is_ragged": False,
            "feature_lengths": dict(query=3, answer=3),
        },
    )
    @pytest.mark.skipif(not os.path.exists(_T5_VOCAB_FILE), reason="Missing testdata.")
    def test_add_token_type_ids_single_example(
        self,
        input_key: Union[str, Sequence[str]],
        expected: Sequence[int],
        is_ragged: bool,
        feature_lengths: Optional[Mapping[str, int]] = None,
    ):
        ds = self.create_source_ds(num_examples=1)
        vocab = SentencePieceVocabulary(_T5_VOCAB_FILE)
        keys = [input_key] if isinstance(input_key, str) else input_key
        ds = tokenize(
            output_features={
                key: seqio.Feature(
                    vocab,
                    add_eos=False,
                    dtype=tf.int32,
                )
                for key in keys
            },
            with_eos=False,
        )(ds)
        if not is_ragged:
            # pytype: disable=attribute-error
            feature_lengths = {k: (1, v) for k, v in feature_lengths.items()}
            # pytype: enable=attribute-error
            ds = seqio.trim_and_pad_dataset(ds, feature_lengths=feature_lengths)
        ds = add_token_type_ids(input_key=input_key)(ds)

        for batch in ds:
            token_type_ids = batch[TOKEN_TYPE_IDS][0].numpy().tolist()
            self.assertEqual(token_type_ids, expected)

    @parameterized.parameters(
        # For ragged tensors, for each of the three example,
        # query has 8 tokens, 5 tokens, 8 tokens respectively;
        # answer has 5 tokens, 9 tokens, and 6 tokens respectively;
        # url has 6 tokens, 7 tokens, 6 tokens respectively.
        {
            "input_key": ["query", "answer", "url"],
            "expected": [
                [0] + [0] * 8 + [0] + [1] * 5 + [1] + [2] * 6 + [2],
                [0] + [0] * 5 + [0] + [1] * 9 + [1] + [2] * 7 + [2],
                [0] + [0] * 8 + [0] + [1] * 6 + [1] + [2] * 6 + [2],
            ],
            "is_ragged": True,
        },
        # For non-ragged tensors, we truncate to 3 tokens per field.
        {
            "input_key": ["query", "answer"],
            "expected": [
                [0] + [0] * 3 + [0] + [1] * 3 + [1],
                [0] + [0] * 3 + [0] + [1] * 3 + [1],
                [0] + [0] * 3 + [0] + [1] * 3 + [1],
            ],
            "is_ragged": False,
            "feature_lengths": dict(query=3, answer=3),
        },
    )
    @pytest.mark.skipif(not os.path.exists(_T5_VOCAB_FILE), reason="Missing testdata.")
    def test_add_token_type_ids_multiple_examples(
        self,
        input_key: Union[str, Sequence[str]],
        expected: Sequence[int],
        is_ragged: bool,
        feature_lengths: Optional[Mapping[str, int]] = None,
    ):
        ds = self.create_source_ds(num_examples=3)
        vocab = SentencePieceVocabulary(_T5_VOCAB_FILE)
        keys = [input_key] if isinstance(input_key, str) else input_key
        ds = tokenize(
            output_features={
                key: seqio.Feature(
                    vocab,
                    add_eos=False,
                    dtype=tf.int32,
                )
                for key in keys
            },
            with_eos=False,
        )(ds)
        if not is_ragged:

            def trim_and_pad_batch():
                def example_fn(
                    example: dict[str, Union[tf.Tensor, tf.RaggedTensor]],
                ) -> dict[str, tf.Tensor]:
                    # pytype: disable=attribute-error
                    for k, v in feature_lengths.items():
                        # pytype: enable=attribute-error
                        example[k] = input_tf_data.trim_and_pad_tensor(example[k], max_len=v)
                    return example

                return seqio.map_over_dataset(example_fn)

            ds = trim_and_pad_batch()(ds)
        ds = add_token_type_ids(input_key=input_key)(ds)

        for batch in ds:
            for i, per_example_token_type_ids in enumerate(batch[TOKEN_TYPE_IDS]):
                per_example_token_type_ids = per_example_token_type_ids.numpy().tolist()
                self.assertEqual(per_example_token_type_ids, expected[i])


class TokenizeExampleTest(test_utils.TestCase):
    def _test_tokenize_example(self, *, vocab_cfg: InstantiableConfig, newlines_replaced_with: str):
        vocab = vocab_cfg.instantiate()
        newlines_replaced_with_id = vocab.encode(newlines_replaced_with).pop()

        # Test tokenize_example replaces newlines.
        tokens = input_text.tokenize_example(
            "Hello\n", sp_vocab=vocab, replace_newlines_with=newlines_replaced_with
        ).numpy()
        self.assertNestedAllClose(
            np.array([*vocab.encode("Hello"), newlines_replaced_with_id]), tokens
        )

    @parameterized.parameters(
        dict(
            vocab_cfg=config_for_class(BPEVocabulary).set(
                sentencepiece_model_file=_OPT_VOCAB_FILE,
                id_map={0: 1, 1: 0},
                decode_as_control=(0, 2),
            ),
            newlines_replaced_with="\n",
        ),
    )
    @pytest.mark.skipif(not os.path.exists(_OPT_VOCAB_FILE), reason="Missing testdata.")
    def test_tokenize_example(self, vocab_cfg: InstantiableConfig, newlines_replaced_with: str):
        self._test_tokenize_example(
            vocab_cfg=vocab_cfg, newlines_replaced_with=newlines_replaced_with
        )


class NumBytesTest(test_utils.TestCase):
    def _test_num_bytes(self, *, vocab_cfg: InstantiableConfig, newlines_replaced_with: str):
        vocab = vocab_cfg.instantiate()

        pad_id = vocab.pad_id
        newline_id = vocab.encode("\n").pop()
        newlines_replaced_with_id = vocab.encode(newlines_replaced_with).pop()

        # Test num_bytes computes expected value.
        ids = tf.constant(
            [vocab.eos_id, newlines_replaced_with_id, newline_id, pad_id, pad_id, pad_id],
            dtype=tf.int32,
        )
        self.assertEqual(
            3,
            input_text.num_bytes(
                ids, sp_vocab=vocab, newlines_replaced_with=newlines_replaced_with
            ),
        )

    @parameterized.parameters(
        dict(
            vocab_cfg=config_for_class(BPEVocabulary).set(
                sentencepiece_model_file=_OPT_VOCAB_FILE,
                id_map={0: 1, 1: 0},
                decode_as_control=(0, 2),
            ),
            newlines_replaced_with="\n",
        ),
    )
    @pytest.mark.skipif(not os.path.exists(_OPT_VOCAB_FILE), reason="Missing testdata.")
    def test_num_bytes(self, vocab_cfg: InstantiableConfig, newlines_replaced_with: str):
        self._test_num_bytes(vocab_cfg=vocab_cfg, newlines_replaced_with=newlines_replaced_with)


class StringsPackUnpackByteArrayTest(test_utils.TestCase):
    def test_pack_unpack_strings_is_lossless(self):
        """Tests packing strings into byte array and then unpacking again."""
        strings = ["", "What time is it?", "This is a red carpet."]
        tf_strings = tf.convert_to_tensor(strings)
        max_packed_byte_array_len = 30
        packed_strings = input_text.pack_strings(tf_strings, max_packed_byte_array_len)
        self.assertEqual(
            tf.TensorShape((len(strings), max_packed_byte_array_len)), packed_strings.shape
        )
        unpacked_strings = input_text.unpack_strings(packed_strings)
        self.assertEqual(len(strings), len(unpacked_strings))
        self.assertEqual(set(strings), set(unpacked_strings))

    def test_pack_too_long_string_raises(self):
        """Tests that trying to pack a string that is too long will surface an error."""
        strings = ["Once apon a time there was a"]
        tf_strings = tf.convert_to_tensor(strings)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, "Condition x <= y did not hold."
        ):
            input_text.pack_strings(tf_strings, max_packed_byte_array_len=5)


class TestSplitSentences(parameterized.TestCase, tf.test.TestCase):
    """Tests split_sentences."""

    def test_split_sentences(self):
        texts = [
            "this is a pretty long sentence\n",
            "this is sentence one.\n\n\nthis is sentence two. three.",
        ]
        ds_fn = make_ds_fn(False, texts, repeat=1)
        split_fn = input_text.split_sentences()
        ds = split_fn(ds_fn())

        expected = [
            "this is a pretty long sentence",
            "",
            "this is sentence one.",
            "this is sentence two.",
            "three.",
            "",
        ]
        actual = [extract_text(x) for x in ds]
        assert expected == actual

    def test_split_sentences_keys(self):
        texts = [
            "this is a pretty long sentence\n",
            "this is sentence one.\n\n\nthis is sentence two. three.",
        ]
        ds_fn = make_ds_fn(False, texts, repeat=1)
        split_fn = input_text.split_sentences(input_key="text", passthrough_keys=["index"])
        ds = split_fn(ds_fn())

        expected = [
            {"text": "this is a pretty long sentence", "index": 0},
            {"text": "", "index": 0},
            {"text": "this is sentence one.", "index": 1},
            {"text": "this is sentence two.", "index": 1},
            {"text": "three.", "index": 1},
            {"text": "", "index": 1},
        ]
        for expect, actual in zip(expected, ds):
            self.assertEqual(expect["text"], extract_text(actual))
            self.assertEqual(expect["index"], actual["index"].numpy())

    @parameterized.parameters(1, 2, 3, 4)
    def test_split_sentences_sampling(self, max_sentences_per_example):
        sentences = [
            [],
            ["one."],
            ["one.", "two."],
            ["one.", "two.", "three."],
            ["one.", "two.", "three.", "four."],
        ]
        texts = [" ".join(s) for s in sentences]
        ds_fn = make_ds_fn(False, texts, repeat=1)
        split_fn = input_text.split_sentences(max_sentences_per_example=max_sentences_per_example)
        ds = split_fn(ds_fn())

        # Group outputs by documents.
        actual = [extract_text(x) for x in ds]
        split_idxs = [i + 1 for i, s in enumerate(actual) if len(s) == 0]
        actual_sentences = np.split(actual, split_idxs)

        # Compare.
        for original, actual in zip(sentences, actual_sentences):
            # Make sure the document is terminated with separator.
            self.assertEqual(actual[-1], "")
            actual = actual[:-1]
            # Make sure number of sentences is as expected.
            self.assertEqual(len(actual), min(len(original), max_sentences_per_example))
            # Ensure that output sentences is a subset of the original.
            self.assertContainsSubset(actual, original)
            # Ensure that each output samples without replacement.
            self.assertCountEqual(actual, set(actual))


class TestRandomChunking(parameterized.TestCase, tf.test.TestCase):
    """Tests random_chunking."""

    def test_random_chunking(self):
        chunk_size = 3
        examples = [
            {input_text.INPUT_IDS: []},
            {input_text.INPUT_IDS: list(range(2))},
            {input_text.INPUT_IDS: list(range(4))},
            {input_text.INPUT_IDS: list(range(6))},
        ]
        # For each input, we may expect one of several valid outputs.
        target_candidates = [
            [tf.constant([], dtype=tf.int32)],
            [[0, 1]],
            [[0, 1, 2], [1, 2, 3]],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
        ]
        for example, expected_candidates in zip(examples, target_candidates):
            # Create a singleton dataset.
            ds = tf.data.Dataset.from_generator(
                lambda ex=example: iter([ex]),
                output_signature={
                    input_text.INPUT_IDS: tf.TensorSpec(shape=(None), dtype=tf.int32),
                },
            )
            # Apply chunking and compare the single output.
            chunk_fn = input_text.random_chunking(max_len=chunk_size)
            actual = next(chunk_fn(ds).as_numpy_iterator())
            assert_oneof(self, actual[input_text.INPUT_IDS], expected_candidates)


class TestTextNormalize(parameterized.TestCase, tf.test.TestCase):
    """Tests normalization helpers."""

    @parameterized.parameters(False, True)
    def test_bert_normalize_against_hf(self, cased: bool):
        # BasicTokenizer acts as a pre-tokenization step:
        # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L172
        # https://github.com/huggingface/transformers/blob/31ec2cb2badfbdd4c1ac9c6c9b8a74e974984206/src/transformers/models/bert/tokenization_bert.py#L223
        hf_tokenizer = BasicTokenizer(
            do_lower_case=not cased,
            tokenize_chinese_chars=True,
            strip_accents=False,
        )
        texts = [
            # Huggingface BasicTokenizer test queries:
            # https://github.com/huggingface/transformers/blob/31ec2cb2badfbdd4c1ac9c6c9b8a74e974984206/tests/bert/test_tokenization_bert.py#L121-L184
            "ah\u535a\u63a8zz",
            " \tHeLLo!how  \n Are yoU?  ",
            # Custom tests.
            "from bert: John Johanson's house",
            # pylint: disable-next=line-too-long
            f"from tf_text: 株式会社 ＫＡＤＯＫＡＷＡ this  {chr(0xFFFD)}\0  is, å tést! SenTence `🤗` \t\t\f,  ",
        ]
        ds_fn = make_ds_fn(is_training=False, texts=texts, repeat=1)
        mapper_fn = input_text.bert_normalize(cased=cased)

        expected = [" ".join(hf_tokenizer.tokenize(text)) for text in texts]
        actual = [extract_text(x) for x in mapper_fn(ds_fn())]

        self.assertEqual(expected, actual)

    @parameterized.parameters(
        dict(
            normalizer=config_for_function(input_text.roberta_normalize).set(cased=False),
            expected=["ah博推zz \thello!how  \n are you?"],
        ),
        dict(
            normalizer=config_for_function(input_text.roberta_normalize).set(cased=True),
            expected=["ah博推zz \tHeLLo!how  \n Are yoU?"],
        ),
        dict(
            normalizer=config_for_function(input_text.bert_normalize).set(cased=False),
            expected=["ah 博 推 zz hello ! how are you ?"],
        ),
        dict(
            normalizer=config_for_function(input_text.bert_normalize).set(cased=True),
            expected=["ah 博 推 zz HeLLo ! how Are yoU ?"],
        ),
    )
    def test_normalize(self, normalizer: InstantiableConfig, expected: list[str]):
        texts = ["ah\u535a\u63a8zz \tHeLLo!how  \n Are yoU?  "]
        ds_fn = make_ds_fn(False, texts, repeat=1)
        process_fn = normalizer.set(input_key="text").instantiate()
        processed_ds = process_fn(ds_fn())
        self.assertEqual(expected, [extract_text(x) for x in processed_ds])
        # Ensure that other fields are not dropped.
        self.assertTrue(all("index" in x for x in processed_ds))
        # Test with multiple input keys.
        ds_fn = make_seq2seq_ds_fn(False, texts, texts, repeat=1)
        process_fn = normalizer.set(input_key=["source", "target"]).instantiate()
        processed_ds = process_fn(ds_fn())
        for key in ["source", "target"]:
            self.assertEqual(expected, [extract_text(x, input_key=key) for x in processed_ds])

    @parameterized.parameters(
        dict(
            normalizer=config_for_function(input_text.roberta_normalize).set(cased=False),
            expected=[[["ah博推zz \thello!how  \n are you?"], ["i am good  <3", "what about you?!"]]],
        ),
        dict(
            normalizer=config_for_function(input_text.roberta_normalize).set(cased=True),
            expected=[[["ah博推zz \tHeLLo!how  \n Are yoU?"], ["I am good  <3", "What about you?!"]]],
        ),
        dict(
            normalizer=config_for_function(input_text.bert_normalize).set(
                cased=False, reduce_axis=-1
            ),
            expected=[
                [["ah 博 推 zz hello ! how are you ?"], ["i am good < 3", "what about you ? !"]]
            ],
        ),
        dict(
            normalizer=config_for_function(input_text.bert_normalize).set(
                cased=True, reduce_axis=-1
            ),
            expected=[
                [["ah 博 推 zz HeLLo ! how Are yoU ?"], ["I am good < 3", "What about you ? !"]]
            ],
        ),
    )
    def test_normalize_ragged_input(self, normalizer: InstantiableConfig, expected: list[str]):
        texts = [
            {
                "text": [
                    ["ah\u535a\u63a8zz \tHeLLo!how  \n Are yoU?  "],
                    ["I am good  <3", "What about you?!"],
                ]
            }
        ]
        ds_fn = make_ragged_ds_fn(False, texts, repeat=1)
        process_fn = normalizer.set(input_key="text").instantiate()
        processed_ds = process_fn(ds_fn())
        self.assertEqual(expected, [extract_text_ragged(x) for x in processed_ds])
        # Ensure that other fields are not dropped.
        self.assertTrue(all("index" in x for x in processed_ds))


class PreprocessTextTest(parameterized.TestCase):
    """Tests preprocess_chunks."""

    def test_preprocess_chunks(self):
        texts = [
            "x" * 10,
            "y" * 15,
        ]
        ds_fn = make_ds_fn(False, texts, repeat=1)
        ds = ds_fn()
        processed_ds = input_text.preprocess_chunks(chunk_size=4)(ds)
        expected = [
            "xxxx",
            "xxxx",
            "xx",
            "yyyy",
            "yyyy",
            "yyyy",
            "yyy",
        ]
        actual = [extract_text(x) for x in processed_ds]
        self.assertEqual(expected, actual)

        processed_ds = input_text.preprocess_chunks(chunk_size=5)(ds)
        expected = [
            "xxxxx",
            "xxxxx",
            "yyyyy",
            "yyyyy",
            "yyyyy",
        ]
        actual = [extract_text(x) for x in processed_ds]
        self.assertEqual(expected, actual)

        processed_ds = input_text.preprocess_chunks(chunk_size=100)(ds)
        expected = texts
        actual = [extract_text(x) for x in processed_ds]
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    absltest.main()

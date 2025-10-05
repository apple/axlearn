# Copyright © 2023 Apple Inc.

"""Tests T5 inputs."""

import logging
import os

import pytest
import seqio
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common.config import config_for_function
from axlearn.common.input_fake import fake_source
from axlearn.common.input_t5 import (
    _noise_span_to_unique_sentinel,
    _nonnoise_span_to_unique_sentinel,
    _random_segmentation,
    apply_t5_mask,
    make_t5_autoregressive_inputs,
    map_prefix_to_value,
    random_spans_helper,
    random_spans_noise_mask,
    reduce_concat_tokens,
    select_random_chunk,
    split_tokens,
)
from axlearn.common.input_test_utils import t5_sentence_piece_vocab_file


def _t5_vocab():
    return seqio.SentencePieceVocabulary(
        sentencepiece_model_file=t5_sentence_piece_vocab_file, extra_ids=100
    )


def _is_sentinel(vocab, token_id):
    return token_id >= (vocab.vocab_size - vocab.extra_ids)


def _count_sentinels(vocab, ids):
    return tf.reduce_sum(tf.cast(ids >= (vocab.vocab_size - vocab.extra_ids), dtype=tf.int32))


def _interleave(vocab, source_ids, target_labels):
    # Test that source_ids and target_labels are inverses of each other, by interleaving
    # source_ids and target_labels.
    interleaved = []
    t = 1  # t=0 is always a sentinel.
    for s in source_ids:
        if _is_sentinel(vocab, s):
            while t < len(target_labels) and not _is_sentinel(vocab, target_labels[t]):
                interleaved.append(target_labels[t])
                t += 1
            t += 1
        else:
            interleaved.append(s)
    logging.info("source_ids=%s", source_ids)
    logging.info("target_labels=%s", target_labels)
    logging.info("interleaved=%s", interleaved)
    return interleaved


class T5InputTest(parameterized.TestCase, tf.test.TestCase):
    def _assert_oneof(self, actual: tf.Tensor, candidates: list[list[int]]):
        for candidate in candidates:
            try:
                self.assertAllEqual(actual, candidate)
                return
            except AssertionError:
                pass
        raise ValueError(f"Expected {actual} to be equal to one of {candidates}")

    @parameterized.parameters(
        # Test a basic case where length == chunk_size.
        dict(
            chunk_size=2,
            examples=[tf.constant([1, 2])],
            expected_one_of=[[tf.constant([1, 2])]],
        ),
        # Test a basic case where length % chunk_size == 0.
        dict(
            chunk_size=2,
            examples=[tf.constant([1, 2, 3, 4])],
            expected_one_of=[[tf.constant([1, 2]), tf.constant([3, 4])]],
        ),
        # Test a basic case where length % chunk_size != 0.
        dict(
            chunk_size=3,
            examples=[tf.constant([1, 2, 3, 4, 5])],
            expected_one_of=[[tf.constant([1, 2, 3]), tf.constant([4, 5])]],
        ),
        # Test a basic case where examples are smaller than chunk_size.
        dict(
            chunk_size=4,
            examples=[tf.constant([1, 2, 3])],
            expected_one_of=[[tf.constant([1, 2, 3])]],
        ),
        # Test a basic case where examples are sometimes empty (i.e., filtered).
        dict(
            chunk_size=2,
            examples=[tf.constant([1, 2, 3]), tf.constant([]), tf.constant([4])],
            expected_one_of=[
                [tf.constant([1, 2]), tf.constant([3])],
                [tf.constant([4])],
            ],
        ),
        # Test a basic case where examples are long.
        dict(
            chunk_size=2,
            examples=[tf.constant([1, 2, 3, 4, 5])],
            expected_one_of=[[tf.constant([1, 2]), tf.constant([3, 4]), tf.constant([5])]],
        ),
    )
    def test_select_random_chunk(
        self, chunk_size: int, examples: list[tf.Tensor], expected_one_of: list[list[tf.Tensor]]
    ):
        def _check(ds):
            ds = list(ds)
            self.assertGreaterEqual(len(expected_one_of), len(ds))
            for expect, actual in zip(expected_one_of, ds):
                self._assert_oneof(actual["source_ids"].numpy().tolist(), expect)

        source = fake_source(
            is_training=False,
            examples=[{"source_ids": x} for x in examples],
            spec={"source_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32)},
        )
        processor = select_random_chunk(chunk_size=chunk_size)
        _check(processor(source()))

    @parameterized.parameters(
        # Test a basic case with num_examples % batch_size == 0.
        dict(
            repeat=4,
            batch_size=2,
            examples=[tf.constant([1, 2, 3])],
            expected=[
                tf.constant([1, 2, 3] * 2),
                tf.constant([1, 2, 3] * 2),
            ],
        ),
        # Test a basic case when num_examples % batch_size != 0.
        dict(
            repeat=8,
            batch_size=3,
            examples=[tf.constant([1, 2, 3])],
            expected=[
                tf.constant([1, 2, 3] * 3),
                tf.constant([1, 2, 3] * 3),
                tf.constant([1, 2, 3] * 2),
            ],
        ),
        # Test a basic case when num_examples < batch_size.
        dict(
            repeat=2,
            batch_size=3,
            examples=[tf.constant([1, 2, 3])],
            expected=[tf.constant([1, 2, 3] * 2)],
        ),
        # Test a case when examples form a ragged.
        dict(
            batch_size=4,
            examples=[
                tf.constant([1, 2]),
                tf.constant([3, 4, 5, 6]),
                tf.constant([]),
                tf.constant([7, 8]),
                tf.constant([9]),
            ],
            expected=[
                tf.constant([1, 2, 3, 4, 5, 6, 7, 8]),
                tf.constant([9]),
            ],
        ),
        # Test a case when examples initially have padding, which gets stripped.
        dict(
            batch_size=3,
            examples=[
                tf.constant([1, 0, 2]),
                tf.constant([3, 4, 0, 0]),
                tf.constant([0]),
            ],
            expected=[tf.constant([1, 2, 3, 4])],
        ),
    )
    def test_reduce_concat_tokens(
        self, batch_size: int, examples: list[tf.Tensor], expected: list[tf.Tensor], repeat: int = 1
    ):
        def _check(ds):
            ds = list(ds)
            self.assertEqual(len(expected), len(ds))
            for expect, actual in zip(expected, ds):
                self.assertAllEqual(expect, actual["source_ids"])

        source = fake_source(
            is_training=False,
            examples=[{"source_ids": x} for x in examples],
            spec={"source_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32)},
            repeat=repeat,
        )
        processor = reduce_concat_tokens(batch_size=batch_size)
        _check(processor(source()))

    @parameterized.parameters(
        dict(
            desired_source_length=10, noise_density=0.0, mean_noise_span_length=3.0, expected=[9, 1]
        ),
        dict(
            desired_source_length=10,
            noise_density=1.0,
            mean_noise_span_length=1.0,
            expected=[9, 19],
        ),
        dict(
            desired_source_length=10,
            noise_density=0.15,
            mean_noise_span_length=3.0,
            expected=[10, 4],
        ),
        dict(
            desired_source_length=10,
            noise_density=0.5,
            mean_noise_span_length=3.0,
            expected=[14, 10],
        ),
    )
    def test_random_spans_helper(
        self,
        expected: tuple[int, int],
        desired_source_length: int,
        noise_density: float,
        mean_noise_span_length: float,
    ):
        # pylint: disable-next=missing-kwoa
        self.assertAllEqual(
            expected,
            random_spans_helper(
                desired_source_length=desired_source_length,
                noise_density=noise_density,
                mean_noise_span_length=mean_noise_span_length,
            ),
        )

    @parameterized.parameters(
        # Test when length % max_tokens_per_segment == 0.
        dict(
            examples=[tf.constant([1, 2, 3, 4])],
            max_tokens_per_segment=2,
            expected=[tf.constant([1, 2]), tf.constant([3, 4])],
        ),
        # Test when length % max_tokens_per_segment != 0.
        dict(
            examples=[tf.constant([1, 2, 3, 4, 5])],
            max_tokens_per_segment=2,
            expected=[tf.constant([1, 2]), tf.constant([3, 4]), tf.constant([5])],
        ),
        # Test when length == max_tokens_per_segment.
        dict(
            examples=[tf.constant([1, 2, 3])],
            max_tokens_per_segment=3,
            expected=[tf.constant([1, 2, 3])],
        ),
        # Test when length < max_tokens_per_segment.
        dict(
            examples=[tf.constant([1, 2, 3])],
            max_tokens_per_segment=5,
            expected=[tf.constant([1, 2, 3])],
        ),
    )
    def test_split_tokens(
        self, examples: list[tf.Tensor], max_tokens_per_segment: int, expected: list[tf.Tensor]
    ):
        def _check(ds):
            ds = list(ds)
            self.assertEqual(len(expected), len(ds))
            for expect, actual in zip(expected, ds):
                self.assertAllEqual(expect, actual["source_ids"])

        source = fake_source(
            is_training=False,
            examples=[{"source_ids": x} for x in examples],
            spec={"source_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32)},
        )
        processor = split_tokens(max_tokens_per_segment=max_tokens_per_segment)
        _check(processor(source()))

    @parameterized.parameters(
        dict(seq_len=10, num_segments=3),
        dict(seq_len=9, num_segments=3),
        dict(seq_len=3, num_segments=9),
    )
    def test_random_segmentation(self, seq_len: int, num_segments: int):
        actual = _random_segmentation(seq_len, num_segments)
        self.assertEqual(sum(actual), seq_len)
        self.assertEqual(len(actual), min(seq_len, num_segments))
        self.assertFalse(tf.reduce_any(tf.equal(actual, 0)))

    @parameterized.parameters(
        # Test some basic cases.
        dict(
            noise_density=0,
            seq_len=4,
            mean_noise_span_length=2.0,
            # Example output: [False, False, False, False],
        ),
        dict(
            noise_density=0.25,
            seq_len=4,
            mean_noise_span_length=2.0,
            # Example output: [False, False, False, True],
        ),
        # Test when number of tokens to be masked is less than `mean_noise_span_length`.
        dict(
            noise_density=0.2,
            seq_len=10,
            mean_noise_span_length=3.0,
            # Example output: [False, False, False, False, False, False, False, False, True, True],
        ),
        # Test when number of tokens to be masked is not divisible by `mean_noise_span_length`.
        dict(
            noise_density=0.3,
            seq_len=10,
            mean_noise_span_length=2.0,
            # Example output: [False, False, False, False, True, True, False, False, False, True],
        ),
        # Test with length < 2.
        dict(noise_density=0.15, seq_len=1, mean_noise_span_length=2.0),
        # TODO(markblee): Revisit this test, which breaks the original.
        # This is due to possible shape mismatch when computing `interleaved_span_lengths`.
        # dict(noise_density=1.0, seq_len=4, mean_noise_span_length=2.0),
    )
    def test_random_spans_noise_mask(
        self, seq_len: int, noise_density: float, mean_noise_span_length: float
    ):
        expected_noise_tokens = min(round(seq_len * noise_density), seq_len - 1)
        expected_noise_spans = round(expected_noise_tokens / mean_noise_span_length)
        if noise_density > 0:
            expected_noise_spans = max(expected_noise_spans, 1)

        # The first token is always a non-noise token.
        if seq_len < 2:
            expected_noise_tokens = expected_noise_spans = 0

        def _count_tokens(x):
            return int(tf.reduce_sum(tf.cast(x, dtype=tf.int32)))

        def _count_spans(x):
            x = tf.concat([[0], tf.cast(x, dtype=tf.int32)], 0)
            return int(tf.reduce_sum(tf.cast(x[1:] > x[:-1], dtype=tf.int32)))

        actual = random_spans_noise_mask(
            noise_density=noise_density,
            seq_len=seq_len,
            mean_noise_span_length=mean_noise_span_length,
        )
        self.assertEqual(_count_tokens(actual), expected_noise_tokens)
        self.assertEqual(_count_spans(actual), expected_noise_spans)

    @parameterized.parameters(
        # Test an empty input.
        dict(
            input_ids=tf.constant([]),
            noise_mask=tf.constant([], dtype=tf.bool),
            expected=tf.constant([]),
        ),
        # Test everything masked.
        dict(
            input_ids=tf.constant([1, 2, 3]),
            noise_mask=tf.constant([True, True, True]),
            expected=tf.constant([32099]),
        ),
        # Test nothing masked.
        dict(
            input_ids=tf.constant([1, 2, 3]),
            noise_mask=tf.constant([False, False, False]),
            expected=tf.constant([1, 2, 3]),
        ),
        # Test some basic cases.
        dict(
            input_ids=tf.constant([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            noise_mask=tf.constant(
                [False, False, False, True, True, False, True, False, False, True]
            ),
            expected=tf.constant([10, 11, 12, 32099, 15, 32098, 17, 18, 32097]),
        ),
        dict(
            input_ids=tf.constant([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            noise_mask=tf.constant(
                [True, True, False, False, False, False, True, True, True, False]
            ),
            expected=tf.constant([32099, 12, 13, 14, 15, 32098, 19]),
        ),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_noise_span_to_unique_sentinel(
        self, input_ids: tf.Tensor, noise_mask: tf.Tensor, expected: tf.Tensor
    ):
        vocab = _t5_vocab()
        actual = _noise_span_to_unique_sentinel(
            input_ids=input_ids, noise_mask=noise_mask, vocab=vocab
        )
        self.assertAllEqual(expected, actual)

    @parameterized.parameters(
        # Test an empty input.
        dict(
            input_ids=tf.constant([]),
            noise_mask=tf.constant([], dtype=tf.bool),
            expected=tf.constant([]),
        ),
        # Test everything masked.
        dict(
            input_ids=tf.constant([1, 2, 3]),
            noise_mask=tf.constant([True, True, True]),
            expected=tf.constant([1, 2, 3]),
        ),
        # Test nothing masked.
        dict(
            input_ids=tf.constant([1, 2, 3]),
            noise_mask=tf.constant([False, False, False]),
            expected=tf.constant([32099]),
        ),
        # Test some basic cases.
        dict(
            input_ids=tf.constant([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            noise_mask=tf.constant(
                [False, False, False, True, True, False, True, False, False, True]
            ),
            expected=tf.constant([32099, 13, 14, 32098, 16, 32097, 19]),
        ),
        dict(
            input_ids=tf.constant([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            noise_mask=tf.constant(
                [True, True, False, False, False, False, True, True, True, False]
            ),
            expected=tf.constant([10, 11, 32099, 16, 17, 18, 32098]),
        ),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_nonnoise_span_to_unique_sentinel(
        self, input_ids: tf.Tensor, noise_mask: tf.Tensor, expected: tf.Tensor
    ):
        vocab = _t5_vocab()
        actual = _nonnoise_span_to_unique_sentinel(
            input_ids=input_ids, noise_mask=noise_mask, vocab=vocab
        )
        self.assertAllEqual(actual, expected)

    @parameterized.parameters(
        # Test an empty input.
        dict(
            source_ids=tf.constant([]),
            noise_density=0.15,
            mean_noise_span_length=3.0,
            # Example output:
            # expected=[dict(source_ids=tf.constant([]), target_labels=tf.constant([]))],
        ),
        # Test an input of length 1.
        dict(
            source_ids=tf.constant([8]),
            noise_density=0.15,
            mean_noise_span_length=3.0,
            # Example output:
            # expected=[
            #     dict(
            #         source_ids=tf.constant([8]),
            #         target_labels=tf.constant([32099]),
            #     )
            # ],
        ),
        # Test without masking.
        dict(
            source_ids=tf.constant([1, 2, 3]),
            noise_density=0.0,
            mean_noise_span_length=3.0,
            # Example output:
            # expected=[
            #     dict(
            #         source_ids=tf.constant([1, 2, 3]),
            #         target_labels=tf.constant([32099]),
            #     )
            # ],
        ),
        # Test a basic case.
        dict(
            source_ids=tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            noise_density=0.4,
            mean_noise_span_length=2.0,
            # Example output:
            # expected=[
            #     dict(
            #         source_ids=tf.constant([1, 2, 3, 32099, 7, 8, 9, 32098]),
            #         target_labels=tf.constant([32099, 4, 5, 6, 32098, 10]),
            #     )
            # ],
        ),
        # Test when number of tokens to be masked is less than `mean_noise_span_length`.
        dict(
            source_ids=tf.constant([1, 2]),
            noise_density=0.4,
            mean_noise_span_length=2.0,
            # Example output:
            # expected=[
            #     dict(
            #         source_ids=tf.constant([1, 32099]),
            #         target_labels=tf.constant([32099, 2]),
            #     )
            # ],
        ),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_apply_t5_mask(
        self,
        source_ids: tf.Tensor,
        noise_density: float,
        mean_noise_span_length: float,
    ):
        tf.random.set_seed(1234)

        def _check(vocab, ds, source_key="source_ids", target_key="target_labels"):
            ds = list(ds)
            for example in ds:
                # Without chunking, interleaving source/target should recover original.
                actual = _interleave(vocab, example[source_key], example[target_key])
                self.assertAllEqual(source_ids, actual)
                return example

        vocab_cfg = config_for_function(_t5_vocab)
        vocab = vocab_cfg.instantiate()
        source = fake_source(
            is_training=False,
            examples=[{"source_ids": source_ids}],
            spec={"source_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32)},
        )
        processor = apply_t5_mask(
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            vocab_cfg=vocab_cfg,
        )
        _check(vocab, processor(source()))

    @parameterized.parameters(
        # Test a basic input.
        dict(
            examples=[
                dict(source_ids=tf.constant([10, 11, 12, 13, 14, 15, 16, 17])),
                dict(source_ids=tf.constant([20, 21, 22])),
                dict(source_ids=tf.constant([30, 31, 32, 33])),
                dict(source_ids=tf.constant([40, 41, 42, 43, 44, 45])),
            ],
            max_source_length=8,
            max_target_length=8,
            max_chunk_size=6,
            noise_density=0.5,
            mean_noise_span_length=3.0,
            num_sequences_to_concat=3,
            # An example output:
            # expected=[
            #     {
            #         "prefix": tf.constant([1]),
            #         "source_ids": tf.constant([10, 32099, 12, 13, 14, 15, 32098, 1]),
            #         "target_ids": tf.constant([1, 32099, 11, 32098, 20, 21, 22, 30]),
            #         "target_labels": tf.constant([32099, 11, 32098, 20, 21, 22, 30, 1]),
            #     },
            #     {
            #         "prefix": tf.constant([1]),
            #         "source_ids": tf.constant([31, 32099, 1]),
            #         "target_ids": tf.constant([1, 32099, 32, 33]),
            #         "target_labels": tf.constant([32099, 32, 33, 1]),
            #     },
            #     {
            #         "prefix": tf.constant([1]),
            #         "source_ids": tf.constant([40, 41, 42, 32099, 1]),
            #         "target_ids": tf.constant([1, 32099, 43, 44, 45]),
            #         "target_labels": tf.constant([32099, 43, 44, 45, 1]),
            #     },
            # ],
        ),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_make_t5_autoregressive_inputs(self, examples: list[dict[str, tf.Tensor]], **kwargs):
        tf.random.set_seed(1234)
        source = fake_source(
            is_training=False,
            examples=examples,
            spec={"source_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32)},
        )
        vocab_cfg = config_for_function(_t5_vocab)
        processor = make_t5_autoregressive_inputs(vocab_cfg=vocab_cfg, **kwargs)
        vocab = vocab_cfg.instantiate()
        ds = list(processor(source()))

        def assert_strictly_increasing(seq):
            self.assertTrue(tf.reduce_all(tf.constant(seq[:-1]) < tf.constant(seq[1:])))

        for actual in ds:
            # Ensure that source_ids ends with EOS.
            self.assertEqual(actual["source_ids"][-1], vocab.eos_id)
            # Ensure that target_ids is a shifted version of target_labels, and starts with EOS.
            self.assertAllEqual(
                tf.concat([[vocab.eos_id], actual["target_labels"][:-1]], 0),
                actual["target_ids"],
            )
            # Ensure that target_labels ends with EOS.
            self.assertEqual(actual["target_labels"][-1], vocab.eos_id)

            source_ids = actual["source_ids"][:-1].numpy()
            target_labels = actual["target_labels"][:-1].numpy()
            # Due to how we construct our inputs, the interleaved IDs should be strictly increasing.
            assert_strictly_increasing(_interleave(vocab, source_ids, target_labels))

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_make_t5_autoregressive_inputs_validation(self):
        with self.assertRaisesRegex(ValueError, "exceeds max target"):
            make_t5_autoregressive_inputs(
                vocab_cfg=config_for_function(_t5_vocab),
                max_source_length=512,
                max_target_length=1,
            )

    @parameterized.parameters(
        # Packed input.
        dict(
            is_training=True,
            example=dict(
                target=dict(
                    input_ids=tf.constant([1, 2, 3, 4, 1, 5, 6]),
                    positions=tf.constant([0, 1, 2, 3, 0, 1, 2]),
                ),
            ),
            expected=dict(
                dict(
                    target=dict(
                        input_ids=tf.constant([0, 2, 3, 4, 0, 5, 6]),
                        positions=tf.constant([0, 1, 2, 3, 0, 1, 2]),
                    ),
                ),
            ),
        ),
        dict(
            is_training=True,
            example=dict(
                dict(
                    target=dict(
                        input_ids=tf.constant([1, 1, 8, 1, 9, 10, 11]),
                        positions=tf.constant([0, 0, 1, 0, 1, 2, 3]),
                    ),
                ),
            ),
            expected=dict(
                dict(
                    target=dict(
                        input_ids=tf.constant([0, 0, 8, 0, 9, 10, 11]),
                        positions=tf.constant([0, 0, 1, 0, 1, 2, 3]),
                    ),
                ),
            ),
        ),
        # Unpacked input.
        dict(
            is_training=True,
            example=dict(
                dict(
                    target=dict(
                        input_ids=tf.constant([1, 2, 3, 4, 0, 0, 0]),
                    ),
                ),
            ),
            expected=dict(
                dict(
                    target=dict(
                        input_ids=tf.constant([0, 2, 3, 4, 0, 0, 0]),
                    ),
                ),
            ),
        ),
        # Eval input.
        dict(
            is_training=False,
            example=dict(prefix=tf.constant([1])),
            expected=dict(prefix=tf.constant([0])),
        ),
    )
    def test_map_prefix_to_value(
        self, is_training: bool, example: dict[str, tf.Tensor], expected: dict[str, tf.Tensor]
    ):
        source = fake_source(is_training=is_training, examples=[example])
        processor = map_prefix_to_value(is_training, value=0)
        ds = processor(source())
        actual = next(iter(ds))
        if is_training:
            tf.nest.map_structure(self.assertAllEqual, expected, actual)
        else:
            self.assertAllEqual(expected["prefix"], actual["prefix"])


if __name__ == "__main__":
    absltest.main()

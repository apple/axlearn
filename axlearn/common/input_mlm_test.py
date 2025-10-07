# Copyright © 2023 Apple Inc.

"""Tests masked language modeling inputs."""

# pylint: disable=no-self-use
import os
from collections.abc import Iterator
from typing import Optional

import numpy as np
import pytest
import seqio
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common import input_tf_data, utils
from axlearn.common.config import InstantiableConfig, config_for_class, config_for_function
from axlearn.common.input_mlm import (
    MLMAction,
    _ids_to_word_starts,
    apply_mlm_mask,
    apply_mlm_mask_combinatorial_ngram,
    iid_mlm_actions,
    roberta_mlm_actions,
    roberta_mlm_actions_combinatorial_ngram,
    text_to_mlm_input,
)
from axlearn.common.input_test_utils import assert_oneof, make_ds_fn, t5_sentence_piece_vocab_file
from axlearn.common.input_text import random_chunking
from axlearn.common.utils import Tensor


class ApplyMLMTest(parameterized.TestCase, tf.test.TestCase):
    @property
    def vocab_cfg(self) -> InstantiableConfig:
        return config_for_class(seqio.SentencePieceVocabulary).set(
            sentencepiece_model_file=t5_sentence_piece_vocab_file, extra_ids=2
        )

    def _make_ds(self, example: dict[str, Tensor]) -> tf.data.Dataset:
        def ds_fn() -> Iterator[dict[str, Tensor]]:
            yield example

        return tf.data.Dataset.from_generator(
            ds_fn,
            output_signature={
                "input_ids": tf.TensorSpec(shape=tf.shape(example["input_ids"]), dtype=tf.int32),
            },
        )

    def _mask_fn(self, example: dict[str, Tensor], **kwargs) -> Optional[dict[str, Tensor]]:
        vocab = self.vocab_cfg.instantiate()
        pad_id = vocab.pad_id
        mask_id = vocab.vocab_size - 1

        input_ds = self._make_ds(example)
        default_kwargs = dict(
            ignore_input_ids=[pad_id],
            ignore_target_id=0,
            mask_id=mask_id,
            vocab_cfg=self.vocab_cfg,
            is_translation_style=False,
        )
        default_kwargs.update(kwargs)
        fn = apply_mlm_mask(**default_kwargs)
        for output_example in fn(input_ds):
            return output_example

    @parameterized.parameters(
        config_for_function(iid_mlm_actions),
        config_for_function(roberta_mlm_actions),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_apply_mlm_mask(self, actions_cfg: InstantiableConfig):
        vocab = self.vocab_cfg.instantiate()
        pad_id = vocab.pad_id
        mask_id = vocab.vocab_size - 1

        input_ids = tf.constant([1, 2, 3, 4, 5, 6, 7, pad_id])

        # Test do nothing: the inputs should be unchanged, and the targets all zeros.
        actual = self._mask_fn(
            {"input_ids": input_ids}, actions_cfg=actions_cfg.set(select_token_prob=0.0)
        )
        self.assertAllEqual(actual["input_ids"], input_ids)
        self.assertAllEqual(actual["target_labels"], tf.zeros_like(input_ids))

        # Test masking all tokens: inputs should be all mask_id, except for the pad tokens;
        # and targets should be the same as inputs.
        actual = self._mask_fn(
            {"input_ids": input_ids},
            actions_cfg=actions_cfg.set(
                select_token_prob=1.0,
                mask_selected_prob=1.0,
                swap_selected_prob=0.0,
            ),
        )
        self.assertAllEqual(
            actual["input_ids"],
            tf.where(input_ids != 0, x=mask_id, y=input_ids),
        )
        self.assertAllEqual(actual["target_labels"], input_ids)

        # Test swapping all tokens. Expect at least one of the inputs to be different from original,
        # and targets to be same as inputs.
        actual = self._mask_fn(
            {
                "input_ids": input_ids,
            },
            actions_cfg=actions_cfg.set(
                select_token_prob=1.0,
                mask_selected_prob=0.0,
                swap_selected_prob=1.0,
            ),
        )
        self.assertGreater(
            tf.reduce_sum(tf.cast(actual["input_ids"] != input_ids, dtype=tf.int32)), 0
        )
        self.assertAllEqual(actual["target_labels"], input_ids)

        # Test keeping all tokens. Expect inputs and targets to be same as original.
        actual = self._mask_fn(
            {
                "input_ids": input_ids,
            },
            actions_cfg=actions_cfg.set(
                select_token_prob=1.0,
                mask_selected_prob=0.0,
                swap_selected_prob=0.0,
            ),
        )
        self.assertAllEqual(actual["input_ids"], input_ids)
        self.assertAllEqual(actual["target_labels"], input_ids)

        # Test equally probable actions. With a sufficiently large input size, we expect to see at
        # least one of each action: a mask_id, a token equal to original, and a swapped token.
        # We also validate that padding tokens are unchanged and have a corresponding target of 0.
        padding_ids = tf.zeros(10, dtype=tf.int32)
        input_ids = tf.concat([tf.range(1, 1000), padding_ids], 0)
        actual = self._mask_fn(
            {
                "input_ids": input_ids,
            },
            actions_cfg=actions_cfg.set(
                select_token_prob=0.75,
                mask_selected_prob=1.0 / 3.0,
                swap_selected_prob=1.0 / 3.0,
            ),
        )
        self.assertGreater(
            tf.reduce_sum(tf.cast(actual["input_ids"] == mask_id, dtype=tf.int32)), 0
        )
        self.assertGreater(
            tf.reduce_sum(tf.cast(actual["input_ids"] == input_ids, dtype=tf.int32)), 0
        )
        self.assertGreater(
            tf.reduce_sum(tf.cast(actual["input_ids"] != input_ids, dtype=tf.int32)), 0
        )
        self.assertAllEqual(actual["input_ids"][tf.negative(tf.size(padding_ids)) :], padding_ids)
        self.assertAllEqual(
            actual["target_labels"][tf.negative(tf.size(padding_ids)) :], padding_ids
        )

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_apply_mlm_mask_validation(self):
        """Tests some sanity checks for apply_mlm_mask."""
        vocab = self.vocab_cfg.instantiate()
        cfg = config_for_function(apply_mlm_mask).set(
            ignore_input_ids=[],
            ignore_target_id=0,
            mask_id=vocab.vocab_size - 1,
            vocab_cfg=self.vocab_cfg,
        )

        # By default, should succeed without any errors.
        cfg.instantiate()

        # Make sure that invalid `mask_id` is caught.
        with self.assertRaisesRegex(ValueError, "should be different from unk"):
            cfg.set(mask_id=vocab.tokenizer.unk_id()).instantiate()

        # Make sure that ignored `mask_id` is caught.
        with self.assertRaisesRegex(ValueError, "should not be ignored"):
            cfg.set(
                ignore_input_ids=[vocab.vocab_size - 1],
                mask_id=vocab.vocab_size - 1,
            ).instantiate()

        # Make sure that `mask_id` is also a start-of-word.
        with self.assertRaisesRegex(ValueError, "should constitute a start-of-word"):
            cfg.set(
                ignore_input_ids=[],
                # This corresponds to the piece "one", which has no leading ▁, [, or <.
                mask_id=782,
            ).instantiate()

    # TODO(markblee): avoid depending on seed in tests.
    # pylint: disable-next=too-many-statements
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_whole_word_mask(self):
        vocab = self.vocab_cfg.instantiate()
        mask_id = vocab.vocab_size - 1

        # Test a case where there the sequence has a start-of-word, but not at index 0.
        # [one, b, ▁away, ▁as, ▁simply]. In this case, start-of-word is at index 2.
        inputs = [782, 26, 550, 38, 914]
        actual = self._mask_fn(
            {"input_ids": tf.constant(inputs)},
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=1.0,
                swap_selected_prob=0.0,
            ),
        )
        expected = {
            "input_ids": tf.constant([mask_id] * len(inputs)),
            "target_labels": tf.constant(inputs),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # Test that _ids_to_word_starts returns a valid rank.
        inputs = [5, 1]
        self.assertAllEqual(_ids_to_word_starts(inputs, vocab), tf.constant([1], dtype=tf.int64))
        actual = self._mask_fn(
            {"input_ids": tf.constant(inputs, dtype=tf.int32)},
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=1.0,
                swap_selected_prob=0.0,
            ),
        )
        expected = {
            "input_ids": tf.constant([mask_id] * len(inputs)),
            "target_labels": tf.constant(inputs),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # The text is "The new parking fines are positively draconian.".
        # Words in "input_ids" below get grouped as
        # [[37, 126, 3078, [1399, 7], 33, 18294, [3, 3515, 509, 15710, 5], [0]].
        # <pad> belongs to the last group and is an independent word since we check for
        # '▁', '<' and '[' to mark start of words. <pad> is not subject to actions.
        # Test do nothing.
        actual = self._mask_fn(
            {
                "input_ids": tf.constant(
                    [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
                ),
            },
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=0.0,
            ),
        )
        expected = {
            "input_ids": tf.constant(
                [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
            ),
            "target_labels": tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # Test masking all tokens.
        inputs = [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
        actual = self._mask_fn(
            {"input_ids": tf.constant(inputs)},
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=1.0,
                swap_selected_prob=0.0,
            ),
        )
        expected = {
            # Everything masked, except for the final ignored token.
            "input_ids": tf.constant([mask_id] * (len(inputs) - 1) + [0]),
            "target_labels": tf.constant(inputs),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # Test swapping all tokens.
        tf.random.set_seed(1234)
        actual = self._mask_fn(
            {
                "input_ids": tf.constant(
                    [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
                ),
            },
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=0.0,
                swap_selected_prob=1.0,
            ),
        )
        expected = {
            "input_ids": tf.constant(
                [31635, 29735, 9493, 29289, 21709, 2273, 14791, 21207, 1851, 27560, 21676, 23805, 0]
            ),
            "target_labels": tf.constant(
                [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
            ),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # Test keeping all tokens.
        actual = self._mask_fn(
            {
                "input_ids": tf.constant(
                    [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
                ),
            },
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=0.0,
                swap_selected_prob=0.0,
            ),
        )
        expected = {
            "input_ids": tf.constant(
                [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
            ),
            "target_labels": tf.constant(
                [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
            ),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # 'fines' and 'draconian' need to have the same treatment for all their tokens.
        tf.random.set_seed(1234)
        actual = self._mask_fn(
            {
                "input_ids": tf.constant(
                    [37, 126, 3078, 1399, 7, 33, 18294, 3, 3515, 509, 15710, 5, 0]
                ),
            },
            whole_word_mask=True,
            # Equally probable actions.
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=0.75,
                mask_selected_prob=1.0 / 3.0,
                swap_selected_prob=1.0 / 3.0,
            ),
        )
        fines_mask = tf.constant([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        draconian_mask = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        fines_inp = tf.boolean_mask(actual["input_ids"], fines_mask)
        draconian_inp = tf.boolean_mask(actual["input_ids"], draconian_mask)
        fines_tgt = tf.boolean_mask(actual["target_labels"], fines_mask)
        draconian_tgt = tf.boolean_mask(actual["target_labels"], draconian_mask)

        # Test do nothing.
        if fines_tgt[0] == 0:
            self.assertAllEqual(fines_inp, tf.constant([1399, 7]))
            self.assertAllEqual(tf.reduce_all(fines_tgt == 0), True)
        if draconian_tgt[0] == 0:
            self.assertAllEqual(draconian_inp, tf.constant([3, 3515, 509, 15710, 5]))
            self.assertAllEqual(tf.reduce_all(draconian_tgt == 0), True)

        # Test masking all tokens when no "start of word" is detected.
        inputs = [7, 7]
        actual = self._mask_fn(
            {"input_ids": tf.constant(inputs, dtype=tf.int32)},
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=1.0,
                swap_selected_prob=0.0,
            ),
        )
        self.assertAllEqual(_ids_to_word_starts(inputs, vocab), tf.constant([], dtype=tf.int64))
        expected = {
            "input_ids": tf.constant([mask_id] * len(inputs), dtype=tf.int32),
            "target_labels": tf.constant([7, 7], dtype=tf.int32),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # Test masking all tokens when input is empty.
        actual = self._mask_fn(
            {"input_ids": tf.constant([], dtype=tf.int32)},
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=1.0,
                swap_selected_prob=0.0,
            ),
        )
        expected = {
            "input_ids": tf.constant([], dtype=tf.int32),
            "target_labels": tf.constant([], dtype=tf.int32),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

        # Test masking all tokens when input is all ignored.
        actual = self._mask_fn(
            {"input_ids": tf.constant([0, 0], dtype=tf.int32)},
            whole_word_mask=True,
            actions_cfg=config_for_function(iid_mlm_actions).set(
                select_token_prob=1.0,
                mask_selected_prob=1.0,
                swap_selected_prob=0.0,
            ),
        )
        expected = {
            "input_ids": tf.constant([0, 0], dtype=tf.int32),
            "target_labels": tf.constant([0, 0], dtype=tf.int32),
        }
        self.assertAllEqual(actual["input_ids"], expected["input_ids"])
        self.assertAllEqual(actual["target_labels"], expected["target_labels"])

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_ids_to_word_starts(self):
        vocab = seqio.SentencePieceVocabulary(t5_sentence_piece_vocab_file)

        texts = [
            "this sentence has the word preprocessing which gets broken",
            "this is sentence 2.",
            "",
        ]
        expected_list = [[0, 1, 2, 3, 4, 5, 8, 9, 10], [0, 1, 2, 3], []]
        for index, text in enumerate(texts):
            inputs = vocab.encode(text)
            actual = _ids_to_word_starts(inputs, vocab)
            expected = tf.constant(expected_list[index], dtype=tf.int64)
            self.assertAllEqual(actual, expected)

    @parameterized.parameters(
        config_for_function(iid_mlm_actions),
        config_for_function(roberta_mlm_actions),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_apply_mlm_mask_translation(self, actions_cfg: InstantiableConfig):
        vocab = self.vocab_cfg.instantiate()
        pad_id = vocab.pad_id

        input_ids = tf.constant([1, 2, 3, 4, 5, 6, 7, pad_id])

        # Test do nothing: the inputs should be unchanged,
        # and the targets should be the same as inputs.
        actual = self._mask_fn(
            {"input_ids": input_ids},
            actions_cfg=actions_cfg.set(select_token_prob=0.0),
            is_translation_style=True,
        )
        self.assertAllEqual(actual["input_ids"], input_ids)
        self.assertAllEqual(actual["target_labels"], input_ids)

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_apply_mlm_mask_element_spec(self):
        orig_ds = self._make_ds({"input_ids": tf.constant([1, 2, 3])})
        mask_fn = apply_mlm_mask(
            ignore_input_ids=[0], ignore_target_id=0, mask_id=123, vocab_cfg=self.vocab_cfg
        )
        ds = mask_fn(orig_ds)
        for key in ["input_ids", "target_labels"]:
            self.assertEqual(orig_ds.element_spec[key], ds.element_spec[key])


class ApplyMLMTestCombinatorialNgram(parameterized.TestCase, tf.test.TestCase):
    """Tests MLM combinatorial ngram."""

    @property
    def vocab_cfg(self) -> InstantiableConfig:
        return config_for_class(seqio.SentencePieceVocabulary).set(
            sentencepiece_model_file=t5_sentence_piece_vocab_file, extra_ids=2
        )

    def _make_ds(self, example: dict[str, Tensor]) -> tf.data.Dataset:
        def ds_fn() -> Iterator[dict[str, Tensor]]:
            yield example

        return tf.data.Dataset.from_generator(
            ds_fn,
            output_signature={
                "input_ids": tf.TensorSpec(shape=tf.shape(example["input_ids"]), dtype=tf.int32),
            },
        )

    def _mask_fn(self, example: dict[str, Tensor], **kwargs) -> Optional[dict[str, Tensor]]:
        vocab = self.vocab_cfg.instantiate()
        pad_id = vocab.pad_id
        mask_id = vocab.vocab_size - 1

        num_of_examples = kwargs["num_of_examples"]
        del kwargs["num_of_examples"]

        input_ds = self._make_ds(example)
        default_kwargs = dict(
            ignore_input_ids=[pad_id],
            ignore_target_id=0,
            mask_id=mask_id,
            vocab_cfg=self.vocab_cfg,
            is_translation_style=False,
            whole_word_mask=False,
            n=1,
        )
        default_kwargs.update(kwargs)
        fn = apply_mlm_mask_combinatorial_ngram(**default_kwargs)
        for output_example in fn(input_ds).batch(num_of_examples):
            return output_example

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_apply_mlm_mask_combinatorial_ngram(self):
        vocab = self.vocab_cfg.instantiate()
        pad_id = vocab.pad_id
        mask_id = vocab.vocab_size - 1

        actions_cfg = config_for_function(roberta_mlm_actions_combinatorial_ngram)

        # ['▁mask', 'ing', '.'].
        input_ids = [8181, 53, 5, pad_id]
        token_len = len(input_ids)
        word_len = 2

        # Sanity check for ngram length.
        n = 5
        num_of_examples = 1
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError, "Condition x >= y did not hold"
        ):
            actual = self._mask_fn(
                {"input_ids": input_ids},
                actions_cfg=actions_cfg.set(mask_prob=1.0),
                n=n,
                ignore_input_ids=[pad_id],
                ignore_target_id=pad_id,
                num_of_examples=num_of_examples,
            )

        n = 1
        num_of_examples = token_len - n + 1
        # Test default.
        actual = self._mask_fn(
            {"input_ids": input_ids},
            actions_cfg=actions_cfg.set(mask_prob=1.0),
            n=n,
            ignore_input_ids=[pad_id],
            ignore_target_id=pad_id,
            num_of_examples=num_of_examples,
        )
        expected = {
            "input_ids": [
                [mask_id, 53, 5, pad_id],
                [8181, mask_id, 5, pad_id],
                [8181, 53, mask_id, pad_id],
                [8181, 53, 5, pad_id],
            ],
            "target_labels": [
                [8181, pad_id, pad_id, pad_id],
                [pad_id, 53, pad_id, pad_id],
                [pad_id, pad_id, 5, pad_id],
                [pad_id, pad_id, pad_id, pad_id],
            ],
        }
        self.assertEqual(
            {key: (num_of_examples, token_len) for key in ("input_ids", "target_labels")},
            utils.shapes(actual),
        )
        self.assertListEqual(np.array(actual["input_ids"]).tolist(), expected["input_ids"])
        self.assertListEqual(np.array(actual["target_labels"]).tolist(), expected["target_labels"])

        # Test transLation style on.
        actual = self._mask_fn(
            {"input_ids": input_ids},
            actions_cfg=actions_cfg.set(mask_prob=1.0),
            n=n,
            ignore_input_ids=[pad_id],
            ignore_target_id=pad_id,
            is_translation_style=True,
            num_of_examples=num_of_examples,
        )
        expected = {
            "input_ids": [
                [mask_id, 53, 5, pad_id],
                [8181, mask_id, 5, pad_id],
                [8181, 53, mask_id, pad_id],
                [8181, 53, 5, pad_id],
            ],
            "target_labels": [
                [8181, 53, 5, pad_id],
                [8181, 53, 5, pad_id],
                [8181, 53, 5, pad_id],
                [8181, 53, 5, pad_id],
            ],
        }
        self.assertEqual(
            {key: (num_of_examples, token_len) for key in ("input_ids", "target_labels")},
            utils.shapes(actual),
        )
        self.assertListEqual(np.array(actual["input_ids"]).tolist(), expected["input_ids"])
        self.assertListEqual(np.array(actual["target_labels"]).tolist(), expected["target_labels"])

        # Test whole word mask on.
        num_of_examples = word_len - n + 1
        actual = self._mask_fn(
            {"input_ids": input_ids},
            actions_cfg=actions_cfg.set(mask_prob=1.0),
            n=n,
            ignore_input_ids=[pad_id],
            ignore_target_id=pad_id,
            whole_word_mask=True,
            num_of_examples=num_of_examples,
        )
        expected = {
            # fmt: off
            "input_ids": [[mask_id, mask_id, mask_id, pad_id], [8181, 53, 5, pad_id]],
            "target_labels": [
                [8181, 53, 5, pad_id],
                [pad_id, pad_id, pad_id, pad_id],
            ],
            # fmt: on
        }
        self.assertEqual(
            {key: (num_of_examples, token_len) for key in ("input_ids", "target_labels")},
            utils.shapes(actual),
        )
        self.assertListEqual(np.array(actual["input_ids"]).tolist(), expected["input_ids"])
        self.assertListEqual(np.array(actual["target_labels"]).tolist(), expected["target_labels"])

        n = 4
        num_of_examples = token_len - n + 1
        # Test default.
        actual = self._mask_fn(
            {"input_ids": input_ids},
            actions_cfg=actions_cfg.set(mask_prob=1.0),
            n=n,
            # Add punctuation to ignore list.
            ignore_input_ids=[5, pad_id],
            ignore_target_id=pad_id,
            num_of_examples=num_of_examples,
        )
        expected = {
            "input_ids": [
                [mask_id, mask_id, 5, pad_id],
            ],
            "target_labels": [
                [8181, 53, pad_id, pad_id],
            ],
        }
        self.assertEqual(
            {key: (num_of_examples, token_len) for key in ("input_ids", "target_labels")},
            utils.shapes(actual),
        )
        self.assertListEqual(np.array(actual["input_ids"]).tolist(), expected["input_ids"])
        self.assertListEqual(np.array(actual["target_labels"]).tolist(), expected["target_labels"])


class MLMActionsTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        config_for_function(iid_mlm_actions),
        config_for_function(roberta_mlm_actions),
    )
    def test_validate_mlm_actions(self, actions_cfg: InstantiableConfig):
        # Each param should be in [0,1].
        for param in ["select_token_prob", "mask_selected_prob", "swap_selected_prob"]:
            for value in [-1, 2]:
                with self.assertRaisesRegex(ValueError, param):
                    actions_cfg.clone(**{param: value}).instantiate()

        # Validate mask_selected_prob + swap_selected_prob <= 1.
        with self.assertRaisesRegex(ValueError, r"mask_selected_prob \+ swap_selected_prob <= 1"):
            actions_cfg.clone(mask_selected_prob=0.6, swap_selected_prob=0.6).instantiate()

    @parameterized.parameters(0, 2, 5, 32, 40)
    def test_roberta_mlm_actions(self, seq_len: int):
        # With the fairseq masking approach, we always act on some minimum number of tokens.
        actions_cfg = config_for_function(roberta_mlm_actions)
        actions_fn = actions_cfg.instantiate()
        expected_actions = int(seq_len * actions_cfg.select_token_prob)

        for _ in range(10):
            actions = actions_fn(seq_len)
            num_actions = tf.reduce_sum(tf.cast(actions != MLMAction.DO_NOTHING.value, tf.int32))
            # Either expected_actions or expected_actions+1, due to probabilistic rounding.
            self.assertGreaterEqual(num_actions, expected_actions)
            self.assertLessEqual(num_actions, expected_actions + 1)

    @parameterized.product(seq_len=(18, 50), n=(1, 3, 18))
    def test_roberta_mlm_actions_combinatorial_ngram(self, seq_len: int, n: int):
        actions_cfg = config_for_function(roberta_mlm_actions_combinatorial_ngram).set(n=n)
        actions_fn = actions_cfg.instantiate()
        expected_num_actions = tf.repeat(n, tf.cast(seq_len - n + 1, dtype=tf.int32))
        actions = actions_fn(seq_len)
        actual_num_actions = tf.reduce_sum(
            tf.cast(actions != MLMAction.DO_NOTHING.value, tf.int32), axis=1
        )
        self.assertAllEqual(actual_num_actions, expected_num_actions)


class MaskedLmInputTest(parameterized.TestCase, tf.test.TestCase):
    def _mlm_processor_config(
        self,
        *,
        actions_cfg: Optional[InstantiableConfig] = None,
        **kwargs,
    ):
        def noop_normalizer() -> input_tf_data.DatasetToDatasetFn:
            return lambda ds: ds

        mlm_mask_cfg = config_for_function(apply_mlm_mask).set(
            whole_word_mask=True,
        )
        if actions_cfg is not None:
            mlm_mask_cfg.set(actions_cfg=actions_cfg)

        defaults = dict(
            sentence_piece_vocab=config_for_class(seqio.SentencePieceVocabulary).set(
                sentencepiece_model_file=t5_sentence_piece_vocab_file,
                extra_ids=1,
            ),
            normalization=config_for_function(noop_normalizer),
            apply_mlm_mask=mlm_mask_cfg,
            mask_token="▁<extra_id_0>",
        )
        defaults.update(kwargs)
        return config_for_function(text_to_mlm_input).set(**defaults)

    @parameterized.product(is_training=[True, False], truncate=[True, False])
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_fake_text_data(self, is_training: bool, truncate: bool):
        batch_size, max_len = 4, 16
        cfg = input_tf_data.Input.default_config().set(
            name="train" if is_training else "eval",
            is_training=is_training,
            source=config_for_function(make_ds_fn).set(
                texts=[
                    "short",
                    "",
                    "not short",
                    "",
                    "this is a pretty long sentence",
                    "",
                    "this is sentence one. this is sentence two.",
                    "",
                    "this sentence is long and will get filtered out without truncation",
                    "a shorter sentence in the same document.",
                    "",
                ]
            ),
            processor=self._mlm_processor_config(
                actions_cfg=config_for_function(iid_mlm_actions).set(select_token_prob=0.0),
                max_len=max_len,
                truncation=config_for_function(random_chunking) if truncate else None,
                shuffle_buffer_size=0,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )
        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            self.assertEqual(
                {key: (batch_size, max_len) for key in ("input_ids", "target_labels")},
                utils.shapes(batch),
            )
            if not truncate:
                # fmt: off
                self.assertAllEqual(
                    [
                        # ["short", "not short", "this is a pretty long sentence"]
                        # Each is a different document separated by two EOS tokens.
                        [1, 710, 1, 1, 59, 710, 1, 1, 48, 19, 3, 9, 1134, 307, 7142, 1],
                        # ["this is sentence one. this is sentence two."]
                        [1, 1, 48, 19, 7142, 80, 5, 48, 19, 7142, 192, 5, 1, 1, 0, 0],
                        # ["a shorter sentence in the same document.", "short"]
                        # Note that the longer sentence is filtered, but the shorter is kept.
                        [1, 3, 9, 10951, 7142, 16, 8, 337, 1708, 5, 1, 1, 710, 1, 1, 0],
                        # ["not short", "this is a pretty long sentence"]
                        [1, 59, 710, 1, 1, 48, 19, 3, 9, 1134, 307, 7142, 1, 0, 0, 0],
                    ],
                    batch["input_ids"],
                )
                # fmt: on
                self.assertAllEqual(batch["target_labels"], tf.zeros_like(batch["target_labels"]))
            else:
                # Note that in this case, we set `truncate` to True in the processor, so the long
                # sentence is not filtered. However, depending on the random chunk selected, we may
                # have multiple valid outputs.

                # fmt: off
                # Check against first two outputs.
                self.assertAllEqual(batch["input_ids"][:2],
                [
                    # ["short", "not short", "this is a pretty long sentence"]
                    [1, 710, 1, 1, 59, 710, 1, 1, 48, 19, 3, 9, 1134, 307, 7142, 1],
                    # ["this is sentence one. this is sentence two."]
                    [1, 1, 48, 19, 7142, 80, 5, 48, 19, 7142, 192, 5, 1, 1, 0, 0],
                ]
                )
                # Check last output.
                self.assertAllEqual(batch["input_ids"][-1],
                    # ["a shorter sentence in the same document.", "short"]
                    [1, 3, 9, 10951, 7142, 16, 8,337, 1708, 5, 1, 1, 710, 1, 1, 0],
                )
                # Third output depends on chunking:
                assert_oneof(self,
                    batch["input_ids"][2],
                    [
                        # ["this sentence is long and will get filtered out without trunc"]
                        # The truncation happens by selecting a random chunk; in this case, we
                        # selected the first chunk.
                        [1, 48, 7142, 19, 307, 11, 56, 129, 3, 23161, 91, 406, 3, 17, 4312, 75],
                        # ["sentence is long and will get filtered out without truncation"]
                        # The truncation happens by selecting a random chunk; in this case, we
                        # selected the second chunk.
                        [1, 7142, 19, 307, 11, 56, 129, 3, 23161, 91, 406, 3, 17, 4312, 75, 257],
                        # ["sentence is long and will get filtered out without truncation"]
                        # The truncation happens by selecting a random chunk; in this case, we
                        # selected the last chunk.
                        [1, 19, 307, 11, 56, 129, 3, 23161, 91, 406, 3, 17, 4312, 75, 257, 1],
                    ],
                )
                # fmt:on
                self.assertAllEqual(batch["target_labels"], tf.zeros_like(batch["target_labels"]))
            break

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_filtering(self):
        # Test that we filter out long sequences.
        texts = [
            "this is a very long document that should be filtered pre-packing.",
            "",
            "this is short.",
            "",
        ]

        def noop(is_training: bool = False):
            del is_training
            return lambda x: x  # Don't need batching.

        cfg = input_tf_data.Input.default_config().set(
            name="test",
            is_training=False,
            source=config_for_function(make_ds_fn).set(texts=texts, repeat=1),
            processor=self._mlm_processor_config(
                actions_cfg=config_for_function(iid_mlm_actions).set(select_token_prob=0.0),
                max_len=10,
                shuffle_buffer_size=0,
            ),
            batcher=config_for_function(noop),
        )
        ds = cfg.instantiate(parent=None)
        actual = list(ds)

        expected = [
            {
                # "this is short.""
                "input_ids": [1, 1, 48, 19, 710, 5, 1, 1, 0, 0],
                "target_labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        ]
        self.assertEqual(len(expected), len(actual))

        for x, y in zip(expected, actual):
            self.assertEqual(x["input_ids"], y["input_ids"].tolist())
            self.assertEqual(x["target_labels"], y["target_labels"].tolist())


if __name__ == "__main__":
    absltest.main()

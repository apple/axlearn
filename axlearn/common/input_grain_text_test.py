# Copyright © 2024 Apple Inc.

"""Tests grain text utilities."""

import os
from typing import Sequence

import numpy as np
import pytest
import seqio
from absl.testing import parameterized

from axlearn.common.config import config_for_function
from axlearn.common.input_fake import fake_grain_source
from axlearn.common.input_grain import Tensor
from axlearn.common.input_grain_text import (
    Vocabulary,
    count_num_bytes,
    num_bytes,
    tokenize,
    with_regex_mapping,
)
from axlearn.common.input_test_utils import t5_sentence_piece_vocab_file
from axlearn.common.test_utils import TestCase


class _DummyVocabulary:
    """A dummy vocab."""

    def __init__(self, eos_id: int = -1):
        # Note that default -1 will never show up in ord.
        self._eos_id = eos_id

    @property
    def eos_id(self):
        return self._eos_id  # pytype: disable=attribute-error

    @property
    def pad_id(self):
        return 0

    def encode(self, s: str) -> Sequence[int]:
        # No eos by default.
        return [ord(c) for c in s]

    def _decode(self, ids: Sequence[int]) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return "".join(chr(x) for x in ids if x != self.eos_id)

    def decode(self, ids: Sequence[int]) -> str:
        ids_list = ids.tolist() if isinstance(ids, Tensor) else ids
        try:
            ids = ids[: ids_list.index(self.eos_id)]
        except ValueError:
            pass  # No eos.
        # pylint: disable-next=protected-access
        return self._decode(ids)


class VocabularyTest(TestCase):
    """Tests vocab utils."""

    @parameterized.parameters(["", "test is a test"])
    def test_basic(self, text: str):
        vocab = _DummyVocabulary()
        self.assertIsInstance(vocab, Vocabulary)
        self.assertEqual(text, vocab.decode(vocab.encode(text)))

    @parameterized.parameters(
        dict(text="", mapping=[("\n", "<n>")]),
        dict(text="this\nis\natest\n", mapping=[("\n", "<n>")]),
        dict(
            text="this\nis\natest\n",
            mapping=[("\n", "x"), ("x", "<n>")],
        ),
    )
    def test_with_regex_mapping(self, text: str, mapping: Sequence):
        vocab = with_regex_mapping(
            _DummyVocabulary,
            encode_mapping=mapping,
            decode_mapping=list(reversed([(v, k) for k, v in mapping])),
        )(eos_id=-1)
        encoded = vocab.encode(text)
        if text:
            # Make sure encoded does not contain the replaced subsequence.
            for k, _ in mapping:
                replaced = vocab.encode(k)
                with self.assertRaises(AssertionError):
                    self.assertContainsExactSubsequence(replaced, encoded)
        # Make sure we recover the original.
        self.assertEqual(text, vocab.decode(encoded))

        # Test that `_decode` goes through EOS.
        # pylint: disable-next=protected-access
        self.assertEqual(text, vocab._decode([-1] + list(encoded)))

    def test_tokenize(self):
        # Test tokenize with repeat.
        ds = fake_grain_source([{"text": ""}, {"text": "this is a test"}], repeat=2)
        ds = tokenize(ds, vocab=_DummyVocabulary())
        expected: list[dict] = [
            {"text": np.array([], dtype=int)},
            {"text": np.array([116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116])},
        ]
        self.assertNestedEqual(expected, list(ds)[:2])
        self.assertNestedEqual(expected, list(ds)[2:])

        # Test tokenize with keys.
        examples = [
            {"text": "", "text_with_eos": ""},
            {"text": "a test", "text_with_eos": "a test"},
        ]
        ds = fake_grain_source(examples)
        ds = tokenize(
            ds,
            vocab={
                "text": _DummyVocabulary(),
                "text_with_eos": _DummyVocabulary(eos_id=5),
            },
            with_eos=True,
        )
        expected = [
            {"text": np.array([-1]), "text_with_eos": np.array([5])},
            {
                "text": np.array([97, 32, 116, 101, 115, 116, -1]),
                "text_with_eos": np.array([97, 32, 116, 101, 115, 116, 5]),
            },
        ]
        for a, b in zip(expected, list(ds)):
            self.assertTrue(np.all(a["text"] == b["text"]))
            self.assertTrue(np.all(a["text_with_eos"] == b["text_with_eos"]))

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_tokenize_sentencepiece(self):
        vocab = seqio.SentencePieceVocabulary(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )
        self.assertIsInstance(vocab, Vocabulary)

        examples = [{"text": ""}, {"text": "this is a test"}]
        ds = fake_grain_source(examples)
        ds = tokenize(ds, vocab=vocab)
        expected = [{"text": np.array([], dtype=int)}, {"text": np.array([48, 19, 3, 9, 794])}]
        self.assertNestedEqual(expected, list(ds))

    def test_config(self):
        # Test that we can configure transformed vocabs.
        vocab_cls = with_regex_mapping(
            seqio.SentencePieceVocabulary,
            # Need extra space for spm to recognize.
            encode_mapping=[("\n", " <extra_id_0>")],
            decode_mapping=[(" <extra_id_0>", "\n")],
        )
        vocab_cfg = config_for_function(vocab_cls).set(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
            extra_ids=1,
        )
        vocab = vocab_cfg.instantiate()
        self.assertIsInstance(vocab, Vocabulary)

        # Tokenization should apply mapping.
        examples = [{"text": "test\n"}]
        ds = fake_grain_source(examples)
        ds = tokenize(ds, vocab=vocab)
        expected_ids = [794, 32000]
        self.assertNestedEqual([{"text": np.array(expected_ids)}], list(ds))
        self.assertEqual("test\n", vocab.decode(expected_ids))


class BytesTest(TestCase):
    """Tests byte utils."""

    @parameterized.parameters(
        dict(
            # 97 has 1 byte, 200 has 2 bytes, 999 has 2 bytes.
            ids=np.array([97, 200, 200, 0, 999, 999]),
            eos_id=0,
            expected=np.array(10),  # 1+2+2+1+2+2.
        ),
        dict(
            # Test a case with multiple EOS.
            ids=np.array([[97, 200, 200, 0, 1, 1], [0, 0, 0, 999, 999, 999]]),
            eos_id=1,
            # 1+2+2+1+1+1, 1+1+1+2+2+2
            expected=np.array([8, 9]),
        ),
        dict(
            # Test a case where we have EOS at the front.
            ids=np.array([[1, 97, 200, 200, 0, 1]]),
            eos_id=1,
            expected=np.array([8]),
        ),
    )
    def test_num_bytes(self, ids: Tensor, eos_id: int, expected: Tensor):
        actual = num_bytes(ids, vocab=_DummyVocabulary(eos_id), eos_id=eos_id)
        self.assertNestedEqual(expected, actual)

    def test_count_num_bytes(self):
        eos_id = 1
        inputs = [
            {"inputs": np.array([97, 200, 400, 1, 999])},
            {"inputs": np.array([[1, 0, 0], [0, 1, 1]])},
        ]
        ds = fake_grain_source(inputs)
        ds = count_num_bytes(ds, input_key="inputs", vocab=_DummyVocabulary(eos_id), eos_id=eos_id)
        expected = [{"inputs_num_bytes": 8}, {"inputs_num_bytes": np.array([3, 3])}]
        # We should retain original keys.
        expected = [{**x, **y} for x, y in zip(expected, inputs)]
        actual = list(ds)
        self.assertNestedEqual(expected, actual)

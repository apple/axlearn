# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# openai/gpt-2:
# Copyright (c) 2019 OpenAI.
# Licensed under a modified MIT license.

"""Tests BPE vocabulary."""

# pylint: disable=no-self-use,protected-access
import os
from typing import Optional, Union

import pytest
import regex as re
import tensorflow as tf
from absl.testing import parameterized
from transformers import GPT2Tokenizer, PreTrainedTokenizer, RobertaTokenizer

from axlearn.common.input_test_utils import tokenizers_dir
from axlearn.common.test_utils import TestCase
from axlearn.common.vocabulary_bpe import BPEVocabulary

# Note: some tests contain invisible characters.
TEXTS = [
    "",
    "œ\x9c",
    "œ",
    "\x9c",
    "цIи — Боливариа́нхой",
    "  — ",
    "   — ",
    " \x0b-",
    " \x0b\x0b- ",
    " \x85- ",
    " \x85\x85- ",
    "hello world",
    "  this\r  \t\t a\ntest's  11'm'll  \tx  ",
    "ah\u535a\u63a8zz",
    " \tHeLLo!how  \n Are yoU?  ",
    "from bert: John Johanson's house",
    # Note: this is slightly different from other occurrences, as it doesn't include the null
    # character. tf_text doesn't seem to regex_split properly with nulls.
    f"from tf_text: 株式会社 ＫＡＤＯＫＡＷＡ this  {chr(0xFFFD)}  is, å tést! SenTence `🤗` \t\t\f,  ",
    f"from tf_text: 株式会社 ＫＡＤＯ<s>ＫＡＷＡ</s> this  {chr(0xFFFD)}  is, å tést! SenTence `🤗` \t\t\f,  ",
    "<s> \thello <s world<s>test !<pad></s><mask>  \x85<s><<|endoftext|>>",
]
_TOKENIZERS = [
    dict(name="roberta-base", hf_cls=RobertaTokenizer, hf_name=None),
    dict(name="gpt2", hf_cls=GPT2Tokenizer, hf_name=None),
    dict(name="opt", hf_cls=GPT2Tokenizer, hf_name="facebook/opt-125m"),
]


def hf_byte_encode(hf_vocab: Union[RobertaTokenizer, GPT2Tokenizer], text: str):
    # Reference:
    # https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L99
    return "".join(hf_vocab.byte_encoder[b] for b in text.encode("utf-8"))


class BaseBPEVocabularyTest(TestCase):
    """Base class for test methods. Do not add tests here."""

    def _get_tokenizers(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        id_map = None
        if name in {"roberta-base", "opt"}:
            # In the original fairseq/hf vocab, we have:
            # |   0   |   1   |   2   |   3   |   4   | ...
            # |  <s>  | <pad> |  </s> | <unk> |   .   | ...
            #
            # Seqio requires that the pad token be at 0, so we remap <s> and <pad>.
            # OPT has the same mapping.
            id_map = {0: 1, 1: 0}
        elif name == "gpt2":
            # In our SPM-compatible GPT2 vocab, we have:
            # |   0   | ... |     50256     | 50257 | 50258 |
            # |   !   | ... | <|endoftext|> | <unk> | <pad> |
            #
            # Seqio requires that the pad token be at 0, so we remap ! and <pad>.
            id_map = {0: 50258, 50258: 0}
        else:
            raise NotImplementedError(name)

        vocab = BPEVocabulary(
            os.path.join(tokenizers_dir, f"bpe/{name}.model"),
            id_map=id_map,
        )
        hf_vocab = hf_cls.from_pretrained(hf_name or name)
        return vocab, hf_vocab


@pytest.mark.gs_login
class BPEVocabularyTest(BaseBPEVocabularyTest):
    @parameterized.parameters(_TOKENIZERS)
    def test_id_map(self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None):
        vocab, hf_vocab = self._get_tokenizers(name, hf_cls, hf_name)

        def _assert_tokens_equal(seqio_id: int, hf_id: int):
            # Make sure ids are the same.
            self.assertEqual(seqio_id, hf_id)
            # Make sure they correspond to the same piece,
            self.assertEqual(
                vocab.tokenizer.id_to_piece(seqio_id), hf_vocab.convert_ids_to_tokens(hf_id)
            )

        # Notes:
        # GPT2 doesn't originally have UNK, PAD or SEP.
        # OPT also has no UNK or SEP.
        if name != "gpt2":
            self.assertEqual(vocab.vocab_size, hf_vocab.vocab_size)
        if name not in {"gpt2", "opt"}:
            _assert_tokens_equal(vocab.unk_id, hf_vocab.unk_token_id)
            _assert_tokens_equal(vocab.eos_id, hf_vocab.sep_token_id)

        # Check special tokens: <s> and <pad> are remapped for compat with seqio.
        self.assertEqual(vocab.pad_id, 0)

        # Check special tokens: everything else should be the same.
        # Note: seqio doesn't expose bos_id as a field.
        _assert_tokens_equal(vocab.eos_id, hf_vocab.eos_token_id)

        # Check other tokens.
        for i in range(min(vocab.vocab_size, hf_vocab.vocab_size)):
            if i not in vocab._id_map:
                _assert_tokens_equal(i, i)

    def test_id_map_validation(self):
        """Some sanity checks for id_map."""
        with self.assertRaisesRegex(ValueError, "seqio expects pad_id"):
            BPEVocabulary(os.path.join(tokenizers_dir, "bpe/roberta-base.model"))

        with self.assertRaisesRegex(ValueError, "must be bijective"):
            BPEVocabulary(
                os.path.join(tokenizers_dir, "bpe/roberta-base.model"), id_map={0: 1, 1: 1}
            )

    def test_extra_ids(self):
        vocab = BPEVocabulary(
            os.path.join(tokenizers_dir, "bpe/roberta-base.model"),
            id_map={0: 1, 1: 0},
            extra_ids=2,
        )
        self.assertEqual(vocab.tokenizer.piece_to_id("▁<extra_id_0>"), vocab.vocab_size - 1)
        self.assertEqual(vocab.tokenizer.piece_to_id("▁<extra_id_1>"), vocab.vocab_size - 2)

    @parameterized.parameters(_TOKENIZERS)
    def test_py_encode_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, _ = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            ids = vocab.encode(text)
            tf_ids = vocab.encode_tf(tf.constant(text, dtype=tf.string))
            self.assertSequenceEqual(ids, tf_ids.numpy().tolist())

    @parameterized.parameters(_TOKENIZERS)
    def test_py_decode_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, _ = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            # Note: seqio decode truncates the string at first EOS, so we compare against _decode.
            self.assertEqual(text, vocab._decode(vocab.encode(text)))

    @parameterized.parameters(_TOKENIZERS)
    def test_regex_split_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, hf_vocab = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            ref = list(re.findall(hf_vocab.pat, text))
            test = vocab._regex_split_tf(tf.constant(text, dtype=tf.string))

            # Convert to list and extract text.
            test = [x.numpy().decode("utf-8") for x in test]
            self.assertSequenceEqual(ref, test)

    @parameterized.parameters(_TOKENIZERS)
    def test_byte_encode_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, hf_vocab = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            self.assertEqual(
                hf_byte_encode(hf_vocab, text),
                vocab._byte_encode_tf(tf.constant([text]))[0].numpy().decode("utf-8"),
            )

    @parameterized.product(_TOKENIZERS, chunk_size=[1, 5])
    def test_chunk_byte_encode_fake_data(
        self,
        name: str,
        hf_cls: PreTrainedTokenizer,
        *,
        chunk_size: int,
        hf_name: Optional[str] = None,
    ):
        vocab, hf_vocab = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            self.assertEqual(
                hf_byte_encode(hf_vocab, text),
                vocab._chunk_byte_encode_tf(tf.constant([text]), chunk_size=chunk_size)[0]
                .numpy()
                .decode("utf-8"),
            )

    @parameterized.product(_TOKENIZERS)
    def test_tokenize_detokenize_roundtrip(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, _ = self._get_tokenizers(name, hf_cls, hf_name)

        # Make sure tokenize + detokenize recovers original.
        for text in TEXTS:
            tokens = vocab._regex_split_with_ud_tokens_tf(text)
            tokens = vocab._byte_encode_tf(tokens)
            recovered = vocab.tf_tokenizer.detokenize(vocab.tf_tokenizer.tokenize(tokens))
            self.assertSequenceEqual(tokens.numpy().tolist(), recovered.numpy().tolist())

    @parameterized.parameters(_TOKENIZERS)
    def test_byte_decode_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, _ = self._get_tokenizers(name, hf_cls, hf_name)

        # Verify hf_byte_encode + byte_decode recovers original.
        for text in TEXTS:
            tokens = vocab._regex_split_with_ud_tokens_tf(text)
            encoded = vocab._byte_encode_tf(tokens)
            recovered = vocab._byte_decode_tf(encoded)
            self.assertEqual(text.encode("utf-8"), tf.strings.reduce_join(recovered).numpy())

    @parameterized.parameters(_TOKENIZERS)
    def test_encode_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, hf_vocab = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            # Tokenize without adding BOS/EOS or padding.
            ref = hf_vocab.convert_tokens_to_ids(hf_vocab.tokenize(text))
            test = vocab.apply_inv_id_map_tf(vocab.encode_tf(text)).numpy().tolist()
            self.assertSequenceEqual(ref, test)

    @parameterized.parameters(_TOKENIZERS)
    def test_decode_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, _ = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            ids = vocab.encode_tf(text)
            recovered = vocab._decode_tf(ids)
            self.assertEqual(text.encode("utf-8"), recovered.numpy())

    @parameterized.parameters(_TOKENIZERS)
    def test_decode_batch_fake_data(
        self, name: str, hf_cls: PreTrainedTokenizer, hf_name: Optional[str] = None
    ):
        vocab, _ = self._get_tokenizers(name, hf_cls, hf_name)

        for text in TEXTS:
            # Note: encode doesn't support batched yet.
            ids = tf.expand_dims(vocab.encode_tf(text), axis=0)
            ids = tf.tile(ids, (2, 1))
            recovered_batch = vocab._decode_tf(ids)
            for recovered in recovered_batch.numpy():
                self.assertEqual(text.encode("utf-8"), recovered)

    def test_decode_as_control_map_fake_data(self):
        vocab = BPEVocabulary(
            os.path.join(tokenizers_dir, "bpe/opt.model"),
            id_map={0: 1, 1: 0},
            decode_as_control=(0, 2),
        )
        text = "<pad></s>"
        ids = vocab.encode_tf(text)
        self.assertSequenceEqual(ids.numpy().tolist(), [0, 2])
        recovered = vocab._decode_tf(ids)

        # All of them should have been dropped.
        self.assertEqual("", recovered.numpy().decode("utf-8"))

        text = "<pad>this </s>is</s> a<pad> test"
        ids = vocab.encode_tf(text)
        recovered = vocab._decode_tf(ids)
        self.assertEqual("this is a test", recovered.numpy().decode("utf-8"))

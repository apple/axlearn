# Copyright © 2024 Apple Inc.

"""Fuji v3 vocabulary."""

import os
import tempfile
from typing import Optional, Protocol, Sequence, Union

import jax
import numpy as np
import tensorflow.compat.v2 as tf

import axlearn.common.file_system as fs
from axlearn.common.utils import get_data_dir


class InnerTokenizer(Protocol):
    """Defines a protocol of InnerTokenizer which is used in Vocabulary.

    This is a subset of sentencepiece_processor.SentencePieceProcessor API that are used in
    Vocabulary.
    """

    def encode_as_pieces(self, pieces: str) -> list[str]:
        """Encode text input to tokens."""
        pass

    def piece_to_id(self, piece: str) -> int:
        """Encode a token to id."""
        pass


class Vocabulary(Protocol):
    """Defines a protocol of Vocabulary.

    This is a subset of seqio.Vocabulary APIs that are used in text_to_lm_training_input and
    test_to_lm_eval_input.
    """

    @property
    def pad_id(self) -> int:
        pass

    @property
    def eos_id(self) -> Optional[int]:
        pass

    def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
        pass

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        """Detokenizes int32 batched Tensor."""
        pass

    def encode(self, s: str) -> list[int]:
        """Tokenizes string to an int sequence, without adding EOS."""
        pass

    def _decode(self, ids: Sequence[int]) -> str:
        """Detokenizes int sequence to a string, through all EOS."""
        pass

    def decode(self, ids: Sequence[int]) -> str:
        """Detokenizes int32 iterable to a string, up through first EOS."""
        pass

    @property
    def tokenizer(self) -> InnerTokenizer:
        pass


class FujiInnerTokenizer:
    """A wrapper for tokenizer.Tokenizer so that it follows InnerTokenizer Protocol."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode_as_pieces(self, pieces: str) -> list[str]:
        """Encode text input to tokens."""
        return self._tokenizer.encode(pieces, add_special_tokens=False).tokens

    def piece_to_id(self, piece: str) -> int:
        """Encode a token to id."""
        return self._tokenizer.token_to_id(piece)


class FujiV3Vocabulary:
    """A wrapper for tokenizers.Tokenizer so that it follows Vocabulary Protocol.

    Although its name has fuji, but it can be extended to work for all tokenizers.Tokenizer.
    """

    def __init__(self, filename: str):
        # Only require tokenizers if instantiating.
        # pylint: disable-next=import-outside-toplevel
        from tokenizers import Tokenizer

        data_dir = get_data_dir()
        data_dir = (
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
            if data_dir is None or data_dir == "FAKE"
            else data_dir
        )
        filename = os.path.join(data_dir, "tokenizers", "hf", filename)
        if filename.startswith("gs:") or filename.startswith("s3:"):
            # Create a different file for each usage.
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "tokenizer.json")
                fs.copy(filename, path)
                self._tokenizer = Tokenizer.from_file(path)
        else:
            self._tokenizer = Tokenizer.from_file(filename)
        self.vocab = self._tokenizer.get_vocab()
        self.tokenizer = FujiInnerTokenizer(self._tokenizer)

    @property
    def pad_id(self) -> int:
        # Some tokenizers do not have a pad_id.
        # https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418
        for token in ("<|pad_id|>", "<|finetune_right_pad_id|>"):
            if token in self.vocab:
                return self.vocab[token]
        raise ValueError("Unable to infer pad token.")

    @property
    def eos_id(self) -> Optional[int]:
        if "<|end_of_text|>" in self.vocab:
            return self.vocab["<|end_of_text|>"]
        raise ValueError("Unable to infer eos token.")

    @property
    def bos_id(self) -> Optional[int]:
        if "<|begin_of_text|>" in self.vocab:
            return self.vocab["<|begin_of_text|>"]
        raise ValueError("Unable to infer eos token.")

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Encodes a string to token IDs.

        Args:
            s: A tf.Tensor of shape () or (n,) and dtype tf.string.

        Returns:
            A tf.Tensor or RaggedTensor of shape (num_tokens,) or (n, None) and dtype tf.int32.
        """
        need_unpack = False
        if s.ndim == 0:
            s = tf.reshape(s, (1,))
            need_unpack = True

        def helper_en(s):
            res = []
            for item in s.numpy():
                item = item.decode("utf-8")
                encoded = self._tokenizer.encode(item, add_special_tokens=True)
                ids = encoded.ids
                # The return does not include EOS, but we need to remove BOS.
                if len(ids) > 0 and ids[0] == self.bos_id:
                    ids = ids[1:]
                res.append(ids)
            return tf.ragged.constant(res, dtype=tf.int32)

        ret = tf.py_function(
            helper_en, inp=[s], Tout=tf.RaggedTensorSpec([None, None], dtype=tf.int32)
        )
        if need_unpack:
            return ret[0]
        else:
            return ret

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        """Detokenizes int32 batched Tensor."""
        need_unpack = False
        if len(ids.shape) == 1:
            ids = tf.reshape(ids, (1, -1))
            need_unpack = True

        def helper(ids):
            ids = [ids[i].numpy() for i in range(ids.shape[0])]
            ids = [
                item[(item != self.bos_id) & (item != self.eos_id) & (item != self.pad_id)]
                for item in ids
            ]
            s = self._tokenizer.decode_batch(ids, skip_special_tokens=False)
            return tf.convert_to_tensor(s, dtype=tf.string)

        ret = tf.py_function(helper, inp=[ids], Tout=tf.string)
        ret.set_shape(tf.TensorShape((None,)))
        if need_unpack:
            return ret[0]
        else:
            return ret

    def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Tokenizes string Scalar to an int32 Tensor, without adding EOS.

        Args:
            s: A tf.Tensor of shape () or (n,) and dtype tf.string.

        Returns:
            A tf.Tensor or RaggedTensor of shape (num_tokens,) or (n, None) and dtype tf.int32.
        """
        return self._encode_tf(s)

    def encode(self, s: str) -> list[int]:
        """Tokenizes string to an int sequence, without adding EOS."""
        ret = self._tokenizer.encode(s, add_special_tokens=True).ids
        # The return does not include EOS, but we need to remove BOS.
        return ret[1:] if ret[0] == self.bos_id else ret

    def _decode(self, ids: Union[list[int], tuple[int]]) -> str:
        """Detokenizes int32 iterable to a string."""
        # remove BOS, EOS and PAD.
        ids = np.array(ids)
        ids = ids[(ids != self.bos_id) & (ids != self.eos_id) & (ids != self.pad_id)]
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def decode(self, ids: Union[list[int], tuple[int], jax.Array, np.ndarray]) -> str:
        """Detokenizes int32 iterable to a string, up through first EOS."""
        if self.eos_id is not None and self.eos_id in ids:
            if isinstance(ids, (jax.Array, np.ndarray)):
                ids = ids.tolist()  # type: ignore
            ids = ids[: ids.index(self.eos_id) + 1]
        return self._decode(ids)

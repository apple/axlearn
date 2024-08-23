# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# openai/gpt-2:
# Copyright (c) 2019 OpenAI.
# Licensed under a modified MIT license.

"""An implementation of GPT2 BPE as a seqio Vocabulary.

Reference:
https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py
"""

from collections.abc import Sequence
from typing import Any, Optional

import regex as re
import seqio
import tensorflow as tf
import tensorflow_text as tft
from sentencepiece import sentencepiece_model_pb2


# pylint: disable-next=too-many-instance-attributes
class BPEVocabulary(seqio.SentencePieceVocabulary):
    """BPE tokenization in the style of GPT2 and RoBERTa, compatible with seqio.

    Known differences:
      - Encoding is truncated at null bytes, seemingly due to a bug in tf_text.
        Related issue: https://github.com/tensorflow/tensorflow/issues/57018
        See also `_regex_split_tf`.
      - UNK tokens in the input are always split, instead of treated atomically like a user-defined
        token. This is because SentencePiece assigns each piece a single type, which for UNK tokens
        is UNKNOWN rather than USER_DEFINED. Note that SentencePiece requires an UNK token.
        See also `_regex_split_with_ud_tokens_tf`.

    Args:
        sentencepiece_model_file: Path to SentencePiece model.
        extra_ids: Number of extra IDs to add.
        decode_errors: Handling for UTF-8 decoding errors.
            See: https://docs.python.org/3/library/stdtypes.html#bytes.decode
        id_map: Optional token ID remapping from old token ID to new token ID.
        decode_as_control: Token IDs to treat as control tokens during decode, i.e. they will be
            dropped, similarly to how SentencePiece treats control tokens. Should refer to IDs
            assuming id_map has been applied.
    """

    def __init__(
        self,
        sentencepiece_model_file: str,
        extra_ids: int = 0,
        decode_errors: str = "replace",
        id_map: Optional[dict[int, int]] = None,
        decode_as_control: Optional[Sequence[int]] = None,
    ):
        """Instantiates the vocab."""
        super().__init__(sentencepiece_model_file, extra_ids=extra_ids)
        # ID used when static hash table lookup fails.
        self._reserved_id = -100
        self._decode_errors = decode_errors

        self._id_map = {}
        self._inv_id_map = {}
        if id_map:
            if set(id_map.keys()) != set(id_map.values()):
                raise ValueError("id_map must be bijective.")

            self._id_map = id_map
            self._inv_id_map = {v: k for k, v in id_map.items()}
            self._id_map_tf = _static_hash_table(self._id_map, self._reserved_id)
            self._inv_id_map_tf = _static_hash_table(self._inv_id_map, self._reserved_id)

        # Load in user-defined tokens from the model.
        self._model_proto = sentencepiece_model_pb2.ModelProto.FromString(self.sp_model)
        ud_pieces = set()
        for piece in self._model_proto.pieces:
            if piece.type == sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED:
                ud_pieces.add(piece.piece)
        self._ud_pieces = ud_pieces

        # Keeps track of tokens tokens to drop during decode.
        self._decode_as_control_map_tf = None
        if decode_as_control:
            if len(decode_as_control) != len(set(decode_as_control)):
                raise ValueError("decode_as_control must be unique.")

            self._decode_as_control_map_tf = _static_hash_table(
                {id: id for id in decode_as_control}, self._reserved_id
            )

        pad_id = self.pad_id
        if pad_id != seqio.PAD_ID:
            raise ValueError(
                f"Tokenizer uses pad_id {pad_id}, but seqio expects pad_id {seqio.PAD_ID}."
            )

        # Map byte to unicode char.
        byte_encoder = {k.to_bytes(1, "big"): v for k, v in _bytes_to_unicode().items()}
        self._byte_encoder = _static_hash_table(byte_encoder, "")

        # Map unicode char back to single bytes.
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        self._byte_decoder = _static_hash_table(byte_decoder, "")

        # Remap \s to \s\p{Z}\x0b\x85 to match Python regex behavior.
        spc = r"\s\p{Z}\x0b\x85"
        self._spc = spc

        # Pattern used by _regex_split_tf.
        self._pat = (
            rf"'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^{spc}\p{{L}}\p{{N}}]+|[{spc}]+"
        )
        # Pattern used for splitting user-defined pieces.
        self._ud_pat = "(" + "|".join([re.escape(piece) for piece in ud_pieces]) + ")"

    @property
    def id_map(self) -> dict[int, int]:
        return self._id_map

    @property
    def eos_id(self) -> Optional[int]:
        eos_id = self.tokenizer.PieceToId(self._model_proto.trainer_spec.eos_piece)
        return self._id_map.get(eos_id, eos_id)

    @property
    def pad_id(self) -> Optional[int]:
        pad_id = self.tokenizer.PieceToId(self._model_proto.trainer_spec.pad_piece)
        return self._id_map.get(pad_id, pad_id)

    @property
    def unk_id(self) -> Optional[int]:
        unk_id = self.tokenizer.PieceToId(self._model_proto.trainer_spec.unk_piece)
        return self._id_map.get(unk_id, unk_id)

    def apply_id_map_tf(self, ids: tf.Tensor) -> tf.Tensor:
        """Maps the input IDs according to id_map.
        This is applied automatically when invoking `encode_tf`.
        """
        if self._id_map_tf is not None:
            remapped_ids = self._id_map_tf.lookup(ids)
            ids = tf.where(remapped_ids == self._reserved_id, ids, remapped_ids)
        return ids

    def apply_inv_id_map_tf(self, ids: tf.Tensor) -> tf.Tensor:
        """Maps the input IDs using the inverse of id_map."""
        if self._inv_id_map_tf is not None:
            remapped_ids = self._inv_id_map_tf.lookup(ids)
            ids = tf.where(remapped_ids == self._reserved_id, ids, remapped_ids)
        return ids

    def _encode(self, s: str) -> Sequence[int]:
        return self._encode_tf(tf.convert_to_tensor(s, dtype=tf.string)).numpy().tolist()

    def _decode(self, ids: Sequence[int]) -> str:
        return (
            self._decode_tf(tf.convert_to_tensor(ids, dtype=tf.int32))
            .numpy()
            .decode("utf-8", errors=self._decode_errors)
        )

    def _regex_split_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Splits by regex pattern with tf.

        This is analogous to the regex splitting used by GPT2, but without the negative lookahead
        (?!...) which is unsupported in tf (which relies on re2).

        Notes:
          - Null characters are not properly handled by tf_text, which drops everything after a
            null. This seems like a rare enough case and a reasonable enough behavior, so we don't
            special-case it, but is different from the original implementation. Related issue:
            https://github.com/tensorflow/tensorflow/issues/57018

        Reference:
        https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L98
        https://github.com/google/re2/wiki/Syntax

        Args:
            s: A tf.Tensor with dtype tf.string.

        Returns:
            A tf.Tensor of shape [num_tokens] and dtype tf.string.
        """
        # Simulate the lookahead by splitting in two stages.
        # `(?s).*` tells us to keep all delimiters ( `.` does not by default include e.g. `\n`).
        s = tft.regex_split(s, f"[{self._spc}][^{self._spc}]+", keep_delim_regex_pattern="(?s).*")
        tokens = tft.regex_split(s, self._pat, keep_delim_regex_pattern="(?s).*")
        return tokens.flat_values

    def _regex_split_with_ud_tokens_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Wraps _regex_split_tf with special handling for user-defined tokens.

        This handles processing similar to HF, e.g:
        https://github.com/huggingface/transformers/blob/6faf283288ce3390281ad8c1d37ccb13f2d03990/src/transformers/tokenization_utils.py#L520

        In particular, we avoid splitting user-defined tokens via the following strategy:
        1. Find byte-offsets where user-defined tokens exist.
        2. _regex_split_tf as usual, including the user-defined tokens.
        3. Inject the original user-defined tokens at the byte-offsets.
        4. Gather all pieces except the remants of the user-defined token splits.

        Note: SentencePiece assigns a single type per piece, i.e., the UNK token cannot
        simultaneously be UNKNOWN and USER_DEFINED. This means that SentencePiece will split UNK
        tokens in the input string no matter what.

        Args:
            s: A tf.Tensor of shape [] with dtype tf.string.

        Returns:
            A tf.Tensor of shape [num_tokens] and dtype tf.string.
        """
        # Split out user-defined tokens first and track where they begin.
        parts, begin, _ = tft.regex_split_with_offsets(
            s, self._ud_pat, keep_delim_regex_pattern="(?s).*"
        )
        matches = tf.strings.regex_full_match(parts, self._ud_pat)
        match_idx = tf.where(matches)
        match_begin = tf.gather_nd(begin, match_idx)
        ud_tokens = tf.gather_nd(parts, match_idx)

        # Split all tokens, including user-defined tokens.
        parts = self._regex_split_tf(parts)

        # Compute byte-offset where user-defined tokens begin.
        token_lens = tf.strings.length(parts)
        token_begin = tf.cumsum(token_lens, 0, exclusive=True)
        ud_idx = tf.where(token_begin == tf.cast(match_begin[:, None], dtype=token_begin.dtype))

        ud_idx = ud_idx[:, 1, None]
        new_parts = tf.tensor_scatter_nd_update(parts, ud_idx, ud_tokens)

        # Compute indices corresponding to the split parts of each user-defined token.
        ud_idx = tf.stack([ud_idx + 1, ud_idx + 2], axis=-1)
        ud_idx = tf.reshape(ud_idx, [-1, 1])

        # Compute mask excluding the split parts of each user-defined token.
        mask = 1 - tf.scatter_nd(
            ud_idx, tf.ones(tf.shape(ud_idx)[0], tf.int32), tf.cast(tf.shape(token_lens), tf.int64)
        )
        return tf.gather_nd(new_parts, tf.where(mask > 0))

    def _chunk_byte_encode_tf(self, tokens: tf.Tensor, *, chunk_size: int) -> tf.Tensor:
        """Applies _byte_encode_tf in chunks of `chunk_size` examples.

        This avoids large allocs and possible shape overflow during tf.strings.reduce_join.

        Args:
            tokens: Refer to _byte_encode_tf.
            chunk_size: Max number of examples (tokens) in each chunk.

        Returns:
            Refer to _byte_encode_tf.
        """
        # Pad batch to multiple of chunk_size.
        shape = tf.shape(tokens)
        num_chunks = tf.cast(tf.math.ceil(shape[0] / chunk_size), tf.int32)
        target = num_chunks * chunk_size
        padded = tf.pad(tokens, [[0, target - shape[0]]], "CONSTANT", constant_values="")
        # Split into evenly-sized chunks. Note that byte encoding treats each byte independently, so
        # it should be OK to split arbitrarily at byte boundaries.
        chunks = tf.reshape(padded, [num_chunks, chunk_size])
        # Encode chunks in parallel.
        output = tf.map_fn(self._byte_encode_tf, chunks)
        return tf.reshape(output, [-1])[: shape[0]]

    def _byte_encode_tf(self, tokens: tf.Tensor) -> tf.Tensor:
        """Applies bytes_to_unicode mapping in tf.

        TODO(markblee): Consider baking this into SentencePiece normalization rules.

        Reference:
        https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L99

        Args:
            tokens: A tf.Tensor of shape [num_tokens] and dtype tf.string.

        Returns:
            A tf.Tensor of same shape and dtype, with bytes mapped accordingly.
        """
        # Split into bytes. tf strings should be UTF-8 by default.
        byte_values = tf.strings.bytes_split(tokens)
        # Replace bytes with original GPT2 mapping.
        mapped_bytes = self._byte_encoder.lookup(byte_values)
        # Join back into string tokens.
        return tf.strings.reduce_join(
            tf.where(mapped_bytes == "", byte_values, mapped_bytes), axis=-1
        )

    def _byte_decode_tf(self, tokens: tf.Tensor) -> tf.Tensor:
        """Applies inverse of bytes_to_unicode mapping in tf.

        Args:
            tokens: A tf.Tensor of shape [num_tokens] with dtype tf.string.

        Returns:
            A tf.Tensor of same shape and dtype, with bytes mapped accordingly.
        """
        # Split into unicode characters.
        chars = tf.strings.unicode_split(tokens, "UTF-8")
        # Replace chars with original GPT2 mapping.
        mapped_chars = self._byte_decoder.lookup(chars)
        return tf.strings.reduce_join(tf.where(mapped_chars == "", chars, mapped_chars), axis=-1)

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Encodes a string to token IDs.

        Args:
            s: A tf.Tensor of shape [] and dtype tf.string.

        Returns:
            A tf.Tensor of shape [num_tokens] and dtype tf.int32.
        """
        # Apply GPT2 style preprocessing.
        # TODO(markblee): Consider supporting batched.
        if self._ud_pieces:
            parts = self._regex_split_with_ud_tokens_tf(s)
        else:
            parts = self._regex_split_tf(s)

        # Encode in chunks if `parts` is large, i.e., the bounding shape of `parts` exceeds INT_MAX.
        # This avoids overflow during tf.strings.reduce_join on `parts` during `_byte_encode_tf`.
        # The chunk size is determined by the maximum number of examples that can fit in INT_MAX
        # assuming each is the length (in bytes) of the longest example in the batch.
        bytes_per_example = tf.maximum(
            tf.reduce_max(tf.cast(tf.strings.length(parts, unit="BYTE"), dtype=tf.int64)), 1
        )
        # Sanity check that bytes_per_example is within [1, INT_MAX].
        tf.debugging.assert_greater_equal(bytes_per_example, tf.constant(1, dtype=tf.int64))
        tf.debugging.assert_less_equal(bytes_per_example, tf.cast(tf.int32.max, dtype=tf.int64))
        bounding_shape = tf.cast(tf.shape(parts)[0], dtype=tf.int64) * bytes_per_example

        parts = tf.cond(
            bounding_shape >= tf.int32.max,
            lambda: self._chunk_byte_encode_tf(
                parts, chunk_size=tf.cast(tf.int32.max // bytes_per_example, dtype=tf.int32)
            ),
            lambda: self._byte_encode_tf(parts),
        )

        # Encode to IDs and apply remapping.
        ids = super()._encode_tf(parts).flat_values
        return self.apply_id_map_tf(ids)

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        """Decodes token IDs back to a string.

        Args:
            ids: A tf.Tensor of shape [num_tokens] or [batch, num_tokens] and dtype tf.int32.

        Returns:
            A tf.Tensor of shape [] or [batch] with dtype tf.string.
        """
        if self._decode_as_control_map_tf:
            control_ids = self._decode_as_control_map_tf.lookup(ids)
            ids = tf.gather_nd(ids, tf.where(ids != control_ids))
        ids = self.apply_inv_id_map_tf(ids)
        s = super()._decode_tf(ids)
        return self._byte_decode_tf(s)


def _static_hash_table(d: dict[Any, Any], default_value: Any) -> tf.lookup.StaticHashTable:
    """Convert a list of keys and values to a static lookup table."""
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(d.keys())),
            tf.constant(list(d.values())),
        ),
        default_value=tf.constant(default_value),
    )


def _bytes_to_unicode() -> dict[int, str]:
    """Returns mapping from utf-8 bytes to unicode strings.

    Copied from original GPT2 implementation with some comments.

    Reference:
    https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L9

    Original comment:
    The reversible BPE codes work on unicode strings. This means you need a large # of unicode
    characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token
    dataset you end up needing around 5K for decent coverage. This is a significant percentage of
    your normal, say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
    unicode strings. And avoids mapping to whitespace/control characters the BPE code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]  # Copy.
    n = 0
    # From https://github.com/openai/gpt-2/issues/80#issuecomment-487202159:
    # "bytes_to_unicode function takes all control and whitespace characters in code points 0-255
    # and shifts them up by 256 to make them printable. So space (code point 32) becomes Ġ (code
    # point 288)."
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# Copyright © 2024 Apple Inc.

"""Grain text utilities."""

import re
from typing import Any, Optional, Protocol, Sequence, TypeVar, Union, cast, runtime_checkable

import jax
import numpy as np

from axlearn.common.config import ConfigOr, maybe_instantiate
from axlearn.common.input_grain import Dataset, Tensor

_T = TypeVar("_T")
_DictOr = Union[dict[str, _T], _T]


@runtime_checkable
class Vocabulary(Protocol):
    """A basic text vocab interface, similar to `seqio.Vocabulary`."""

    @property
    def pad_id(self):
        """PAD token ID."""

    @property
    def eos_id(self):
        """EOS token ID."""

    def encode(self, s: str) -> Sequence[int]:
        """Tokenizes string to an int sequence."""
        raise NotImplementedError(type(self))

    # This API is useful for e.g. `num_bytes`.
    def _decode(self, ids: Sequence[int]) -> str:
        """Detokenizes int sequence to a string, through all EOS."""
        raise NotImplementedError(type(self))

    def decode(self, ids: Sequence[int]) -> str:
        """Detokenizes int sequence to a string, up through first EOS."""
        raise NotImplementedError(type(self))


def with_regex_mapping(
    vocab_cls: _T,
    *,
    encode_mapping: Sequence[tuple[str, str]],
    decode_mapping: Sequence[tuple[str, str]],
) -> _T:
    """Converts vocabs to vocabs that replace certain patterns.

    For example, this can be used to map "\n" to "<n>" before tokenization (and conversely replace
    "<n>" with "\n" after detokenization).

    NOTE: this is mainly intended for "normalization/denormalization" type regex mapping where we
    almost always want to apply when tokenizing or detokenizing. It is not intended to be a general
    purpose regex mapping processor, which may make sense to decouple from the vocab.

    Args:
        vocab_cls: Original vocab class.
        encode_mapping: A sequence of (source, dest) mappings. Specifically, before encoding a
            string, we `re.sub(source, dest)` for each mapping, in the given order.
        decode_mapping: A sequence of (dest, source) mappings. Specifically, after decoding from IDs
            to string, we `re.sub(dest, source)` for each mapping, in the given order.
            Note that `decode_mapping` should often be specified in reverse order of
            `encode_mapping`, s.t. the last `encode_mapping` is reversed by the first
            `decode_mapping`.

    Returns:
        A vocab that applies the encode/decode mapping.
    """

    def regex_compile(mapping: Sequence[tuple[str, str]]) -> dict[re.Pattern, str]:
        return {re.compile(k): v for k, v in mapping}

    enc_mapping = regex_compile(encode_mapping)
    dec_mapping = regex_compile(decode_mapping)

    class VocabWithRegexMapping(cast(type[Vocabulary], vocab_cls)):
        """A vocab that applies a regex mapping."""

        def encode(self, s: str) -> Sequence[int]:
            for k, v in enc_mapping.items():
                s = re.sub(k, v, s)
            return super().encode(s)

        def _decode(self, ids: Sequence[int]) -> str:
            s = super()._decode(ids)
            for k, v in dec_mapping.items():
                s = re.sub(k, v, s)
            return s

    return VocabWithRegexMapping


def tokenize(
    ds: Dataset, *, vocab: _DictOr[ConfigOr[Vocabulary]], with_eos: bool = False
) -> Dataset:
    """Tokenizes features.

    Args:
        ds: A Dataset.
        vocab: A vocab or a mapping from field to vocab.
            If a mapping is provided, fields will be tokenized with corresponding vocabs.
            If a vocab is provided, it will be broadcasted to all fields.
        with_eos: Whether to append EOS to each field.

    Returns:
        A tokenized dataset.
    """
    vocab = jax.tree.map(maybe_instantiate, vocab)

    def encode(vocab: Vocabulary, s: str) -> Tensor:
        ids = vocab.encode(s)
        if with_eos:
            # Note: numpy can default to floating point if the encoded list is empty, so we
            # explicitly specify int dtype.
            if isinstance(ids, list):
                ids.append(vocab.eos_id)  # Faster to append before converting.
            else:
                ids = np.concatenate([np.asarray(ids, dtype=int), [vocab.eos_id]], axis=-1)
        return np.asarray(ids, dtype=int)

    def fn(example: dict[str, Any]) -> dict[str, Any]:
        output_example = {**example}  # Avoid modifying source keys.
        # TODO(markblee): Consider switching to tree utils. The common case is to have a flat dict,
        # so we keep things simple for now.
        if isinstance(vocab, dict):
            for k, v in vocab.items():
                output_example[k] = encode(v, example[k])
        else:
            for k, v in example.items():
                output_example[k] = encode(vocab, v)
        return output_example

    return ds.map(fn)


def num_bytes(ids: Tensor, *, vocab: Vocabulary, eos_id: int) -> Tensor:
    """Compute the number of bytes contained in token IDs, treating EOS as 1 byte.

    In contrast with `input_text.num_bytes`, we do not assume a SentencePiece vocab; further, any
    newline normalization is assumed to happen within the vocab implementation (e.g., the caller can
    wrap vocabs using `with_regex_mapping`).

    Args:
        ids: The token IDs. Can be a 1D or 2D tensor.
        vocab: A vocabulary. Decoded strings are assumed to not include EOS symbol.
        eos_id: EOS token ID. Each EOS ID will be treated as an additional byte, so it is important
            to keep the EOS ID distinct from padding.

    Returns:
        The number of bytes contained in ids. If `ids` is 1D, the return value will be a scalar
        tensor. If `ids` have shape [batch_size, :], the return value will have shape [batch_size].
    """
    ndim, dtype = ids.ndim, ids.dtype
    # Account for eos_id that would not be present in decoded string.
    eos_counts = np.sum(ids == eos_id, axis=-1, keepdims=ndim == 1)
    if ndim == 1:
        ids = (ids,)
    # TODO(markblee): We can avoid a `_decode` API by iteratively splitting along EOS and decoding
    # individual segments.
    texts = (vocab._decode(x) for x in ids)  # pylint: disable=protected-access
    counts = np.asarray(
        [len(text.encode("utf-8")) + eos_count for text, eos_count in zip(texts, eos_counts)],
        dtype=dtype,
    )
    return counts[0] if ndim == 1 else counts


def count_num_bytes(
    ds: Dataset,
    *,
    input_key: str,
    vocab: Vocabulary,
    eos_id: Optional[int] = None,
    output_key: Optional[str] = None,
) -> Dataset:
    """Counts the number of bytes in `input_key`.

    Args:
        ds: A Dataset where each example should contain:
            `input_key`: An int Tensor with shape [None] or [None, None]. The bytes will be computed
            along the last axis.
        input_key: A key that points to a 1D or 2D Tensor of token IDs.
        vocab: A vocabulary for detokenization.
        eos_id: EOS token ID. See `num_bytes` for details. If None, attempts to infer from vocab.
        output_key: Optional output key. If None, uses `{input_key}_num_bytes`.

    Returns:
        A Dataset with `output_key` added to each example.
    """
    output_key = output_key or f"{input_key}_num_bytes"
    eos_id = vocab.eos_id if eos_id is None else eos_id
    if eos_id is None:
        raise ValueError("Either vocab.eos_id or eos_id must be non-None.")
    assert output_key is not None  # Appease pytype.

    def fn(example: dict[str, Tensor]) -> dict[str, Tensor]:
        return {output_key: num_bytes(example[input_key], vocab=vocab, eos_id=eos_id), **example}

    return ds.map(fn)

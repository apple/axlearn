# Copyright © 2024 Apple Inc.

"""Input processing for language modeling using Grain."""

import functools
from typing import Protocol, Sequence

import numpy as np

from axlearn.common import input_grain, input_grain_text
from axlearn.common.config import ConfigOr, maybe_instantiate
from axlearn.common.input_grain import Dataset, Tensor


class _SplitFn(Protocol):
    """Splits flat input IDs into shape [-1, max_len]."""

    def __call__(self, ids: Tensor, *, max_len: int) -> Tensor:
        ...


# TODO(markblee): If we enforce that each input example is initially shorter than packed length, we
# can preserve the `grain.MapDataset` by using a flat map with fanout <= window_size.
def _make_autoregressive_inputs(
    ds: Dataset,
    *,
    max_len: int,
    split_fn: ConfigOr[_SplitFn],
    input_key: str = "target_labels",
    window_size: int = 1,
) -> Dataset:
    """Produces `input_ids` autoregressively from `target_labels`.

    NOTE: this performs a conversion from `grain.MapDataset` to `grain.IterDataset`.
    See `On grain.MapDataset vs grain.IterDataset` in the file docstring of `input_grain`.

    Args:
        ds: A Dataset where each input example contains:
            `input_key`: A flat int Tensor of shape [None], i.e., length can vary across examples.
        max_len: Max sequence length.
        split_fn: A callable taking flat input IDs and producing batched IDs of shape [-1, max_len].
        input_key: Input key containing `target_labels`.
        window_size: Window size. If > 1, also packs.

    Returns:
        A `grain.IterDataset` with potentially different cardinality than the input dataset.
        Each output example contains:
            `input_key`: An int Tensor with shape [max_len].
            "input_ids": An int Tensor with shape [max_len].
    """
    split_fn = maybe_instantiate(split_fn)

    def process_example_fn(example: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        flat_target_labels = np.concatenate([x[input_key] for x in example], axis=-1)
        flat_input_ids = np.roll(flat_target_labels, 1, axis=0)
        return dict(
            input_ids=split_fn(flat_input_ids, max_len=max_len),
            target_labels=split_fn(flat_target_labels, max_len=max_len),
        )

    # Batch as lists to avoid ragged.
    ds = ds.batch(window_size, drop_remainder=False, batch_fn=list)
    ds = ds.map(process_example_fn)
    ds = input_grain.maybe_to_iter_dataset(ds)
    # After processing, we have non-ragged np.arrays, so we can unbatch.
    ds = input_grain.unbatch(ds)
    return ds


def _trim_or_pad_and_batch(
    ids: Tensor, *, max_len: int, max_padding_fraction: float = 1, pad_id: int = -1
) -> Tensor:
    """Pads or truncates ids so as to divide max length, then group into temporary batch.

    This is similar to the `batch` util in `input_lm`.
    """
    if ids.ndim > 1:
        raise ValueError(f"Expected rank 1 inputs, got: {ids.shape=}")
    remainder = ids.shape[0] % max_len
    # If the remainder isn't long enough to satisfy max_padding_fraction for a given example,
    # drop it, else pad to fill to max_len.
    if remainder > int(max_len * (1 - max_padding_fraction)):
        ids = np.pad(ids, pad_width=((0, max_len - remainder)), constant_values=pad_id)
    elif remainder > 0:
        ids = ids[:-remainder]
    return np.reshape(ids, (-1, max_len))


def text_to_lm_training_input(
    ds: Dataset,
    *,
    vocab: ConfigOr[input_grain_text.Vocabulary],
    max_len: int,
    window_size: int = 128,
    max_padding_fraction: float = 1,
) -> Dataset:
    """Returns a function that generates training inputs for language models from raw text.

    The processing follows `input_lm.text_to_lm_training_input`.

    Args:
        ds: A Dataset where each example contains:
            "text": A string to be tokenized.
        vocab: A vocab or a config instantiating to a vocab. Any text normalization (such as
            `replace_newlines_with`) should be applied directly at the vocab.
        max_len: The maximum number of tokens per sequence.
        window_size: The number of examples to pack before chunking into max_len sequences.
        max_padding_fraction: The maximum fraction of a batch example that we are willing to pad.
            E.g. if this is 0.5 then we will pad an example with >= 0.5 * max_len viable tokens,
            else drop it entirely.

    Returns:
        A `grain.IterDataset` with potentially different cardinality than the input dataset.
        Each output example contains:
            "input_ids": An int Tensor with shape [max_len].
            "target_labels": An int Tensor with shape [max_len].
            "target_num_bytes": A scalar int Tensor.
    """
    vocab = maybe_instantiate(vocab)
    split_fn = functools.partial(
        _trim_or_pad_and_batch, max_padding_fraction=max_padding_fraction, pad_id=vocab.pad_id
    )
    ds = ds.repeat(num_epochs=None)
    ds = input_grain_text.tokenize(ds, vocab={"text": vocab}, with_eos=True)
    ds = input_grain.rekey(ds, key_map={"target_labels": "text"})
    # Flatten, roll, split.
    ds = _make_autoregressive_inputs(
        ds, max_len=max_len, window_size=window_size, split_fn=split_fn
    )
    ds = input_grain_text.count_num_bytes(
        ds, input_key="target_labels", vocab=vocab, output_key="target_num_bytes"
    )
    # Note: In input_lm, there is an additional shuffle so that read order is not dominated by
    # document order. grain.IterDataset currently does not support shuffle, although it may be
    # doable with a shuffle-buffer style shuffling. Since shuffle buffers are memory intensive, we
    # skip the shuffle assuming sufficiently long context and source mixing.
    return ds

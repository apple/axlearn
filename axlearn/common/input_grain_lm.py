# Copyright © 2024 Apple Inc.

"""Input processing for language modeling using Grain."""

import functools
import logging
import sys
from collections.abc import Sequence
from typing import Any, Callable, Optional, Protocol

import grain.python as grain
import numpy as np
from grain._src.python.dataset.transformations.prefetch import MultiprocessPrefetchIterDataset

from axlearn.common import input_grain, input_grain_text
from axlearn.common.config import ConfigOr, maybe_instantiate
from axlearn.common.input_grain import Dataset, SequenceOr, Tensor


class _SplitFn(Protocol):
    """Splits flat input IDs into shape [-1, max_len]."""

    def __call__(self, ids: Tensor, *, max_len: int) -> Tensor:
        ...


_PackingFn = Callable[[Dataset, int, Callable, int, str, grain.ReadOptions], Dataset]


class _StreamingPackingDatasetIterator(grain.DatasetIterator):
    """An iterator that yields packed examples in a streaming fashion.

    This implementation does not require maintaining a fixed buffer of `window_size` elements in
    memory. Instead, it yields packed examples and flushes the buffer as soon as an example is
    ready. This significantly improves the first-time read, especially for datasets which have much
    higher tokens per sequence, as well as reduces the peak memory requirements for packing.

    `window_size` is used for parity with windowed_packing. It will also be used if we want to pack
    multimodal data which is not represented in sequence, thus naturally has a limit in how many
    examples we can pack due to memory limit.

    `max_len` and `input_key` are packing sequence length and keys to look up for packing,
    respectively.
    """

    def __init__(
        self,
        parent: grain.DatasetIterator,
        *,
        max_len: int,
        window_size: Optional[int] = None,
        input_key: str = "target_labels",
    ):
        """Initializes StreamingPackingDataset Iterator.

        Args:
            parent: Parent DatasetIterator to inherit from.
            max_len: A int value representing sequence length of the output examples.
            window_size: An optional int value representing the window_size. If it's set, this
                iterator will behave similar to windowed_packing. This is used for parity with
                windowed_packing.
            input_key: The key in the input examples to use for packing.
        """
        super().__init__(parent)
        self._max_len = max_len
        self._window_size = window_size
        self._input_key = input_key

        # Index of the parent.
        self._index = 0
        # Total number of tokens in `self._current_examples_list`.
        self._current_token_count = 0
        # The examples in the current buffer.
        self._current_examples_list = []
        # For checkpointing support, we need to maintain what exactly are the examples in current
        # sequence. self._parent_sequence_start_state and self._parent_sequence_end_state are used
        # to store to starting and ending state of the examples.

        # If not None, the state of `self._parent` before the first example in
        # `self._current_examples_list` was added.
        # Must be None if `self._current_token_count == 0`.
        self._parent_sequence_start_state = None
        # If not None, the state of `self._parent` before the last example in
        # `self._current_examples_list` was added.
        self._parent_sequence_end_state = None

    def _reach_window_limit(self) -> bool:
        """Determines if we have already reached window limit."""
        return self._window_size is not None and self._index % self._window_size == 0

    def _pop_element(self) -> Optional[dict]:
        """Pops element from self._current_example_list, returns None if the list is empty."""
        # If there is no examples in current sequence, return None.
        if not self._current_examples_list:
            return None
        concat_target_labels = np.concatenate(
            [x[self._input_key] for x in self._current_examples_list], axis=-1
        )
        # Total tokens to pop could be up to self._max_len
        total_tokens_to_pop = min(len(concat_target_labels), self._max_len)
        self._current_token_count -= total_tokens_to_pop
        assert self._current_token_count >= 0
        if self._current_token_count > 0:
            self._current_examples_list = [{self._input_key: concat_target_labels[self._max_len :]}]
            self._parent_sequence_start_state = self._parent_sequence_end_state
        else:
            self._current_examples_list = []
            self._parent_sequence_start_state = None

        # If all the concat target labels is empty, early return.
        if total_tokens_to_pop == 0:
            return None

        return {self._input_key: concat_target_labels[: self._max_len]}

    def __next__(self):
        # Iteratively call __next__ until we yield valid examples.
        while True:
            # If there are still leftover tokens when we have already reached the window limit, we
            # should decide whether to keep this sequence.
            if self._current_token_count > 0 and self._reach_window_limit():
                return self._pop_element()

            # Keeps filling up the sequence until reaching the limit.
            # Termination of this while loop means:
            # 1. Reaches the sequence_length limit, and ready to output one batch.
            # 2. Reaches the window limit.
            while self._current_token_count < self._max_len:
                self._parent_sequence_end_state = self._parent.get_state()
                if not self._parent_sequence_start_state:
                    self._parent_sequence_start_state = self._parent_sequence_end_state
                try:
                    example = next(self._parent)
                except StopIteration as e:
                    next_element = self._pop_element()
                    if next_element is not None:
                        return next_element
                    else:
                        raise e

                self._current_examples_list.append(example)

                self._current_token_count += len(example[self._input_key])
                self._index += 1

                if self._reach_window_limit():
                    break

            # If there is enough token, we always return a sequence.
            if self._current_token_count >= self._max_len:
                return self._pop_element()

            next_element = self._pop_element()
            assert self._current_token_count == 0 and not self._current_examples_list
            # If next element is empty, that suggests that the sequence is dropped.
            if next_element is not None:
                return next_element

    def get_state(self) -> dict[str, Any]:
        # TODO(haoshuoh, markblee): All of the parent_state thing could be wrapped in a Packer
        # class.
        return {
            "parent_sequence_start_state": self._parent_sequence_start_state,
            "parent": self._parent.get_state(),
            "index": self._index,
            "current_token_count": self._current_token_count,
        }

    def set_state(self, state: dict[str, Any]):
        self._parent.set_state(state["parent_sequence_start_state"] or state["parent"])
        self._index = state["index"]

        # Retrieves packer states by loading all the examples from that sequence.
        self._current_token_count = state["current_token_count"]
        self._current_examples_list = []
        self._parent_sequence_start_state = None
        self._parent_sequence_end_state = None
        total_tokens_retrieved = 0

        assert (
            self._current_token_count == 0 if self._parent.get_state() == state["parent"] else True
        )

        while self._parent.get_state() != state["parent"]:
            self._parent_sequence_end_state = self._parent.get_state()
            if not self._parent_sequence_start_state:
                self._parent_sequence_start_state = self._parent_sequence_end_state
            example = next(self._parent)
            total_tokens_retrieved += len(example[self._input_key])
            self._current_examples_list.append(example)

        if total_tokens_retrieved > self._current_token_count:
            # The truncation should only happens to the first example (aka rollover example).
            assert total_tokens_retrieved - self._current_token_count <= len(
                self._current_examples_list[0][self._input_key]
            )
            self._current_examples_list[0] = {
                self._input_key: self._current_examples_list[0][self._input_key][
                    total_tokens_retrieved - self._current_token_count :
                ]
            }
        elif total_tokens_retrieved < self._current_token_count:
            raise ValueError("Grain receives invalid states.")


class _StreamingPackingIterDataset(grain.IterDataset):
    """A class that performs streaming packing."""

    def __init__(
        self,
        parents,
        *,
        max_len: int,
        window_size: Optional[int] = None,
        input_key: str = "target_labels",
    ):
        super().__init__(parents)
        self._max_len = max_len
        self._window_size = window_size
        self._input_key = input_key

    def __str__(self) -> str:
        return "StreamingPackingIterDataset"

    def __iter__(self) -> _StreamingPackingDatasetIterator:
        return _StreamingPackingDatasetIterator(
            self._parent.__iter__(),
            max_len=self._max_len,
            window_size=self._window_size,
            input_key=self._input_key,
        )


def streaming_packing(
    ds: Dataset,
    *,
    max_len: int,
    inner: Callable,
    window_size: Optional[int] = None,
    input_key: str = "target_labels",
    read_options: grain.ReadOptions = grain.ReadOptions(num_threads=1, prefetch_buffer_size=16),
) -> Dataset:
    """Streaming packing given max_len and optional window_size.

    Given a sequence of tokens with arbitraty length, streaming packing will pack examples until it
    reaches the max_len. There is an optional window_size option to make it still compatible with
    windowed_packing. If window_size is None, that means there is no upper bound limit on the
    window size.

    Note that the semantics of inner in this function is slightly different from the one used in
    windowed_packing. In windowed_packing, we expect it to take full window of examples. In
    streaming packing, we expect it to take examples that's within this sequence.

    Args:
        ds: datasets to be packed.
        max_len: Max sequence length.
        inner: A processor that operates on packed examples. It should output examples of shape ...
            or None if the example should be skipped.
        window_size: An upper bound on the window size to use for packing. If None, no upper bound
            is enforced.
        input_key: The keys in the input examples to use for packing.
        read_options: grain.ReadOptions which includes num_threads and prefetch_buffer_size. It is
            used to convert the pipeline to grain.IterDataset.

    Returns:
        A packed dataset with dict which only contains values corresponding to `input_key`.
    """

    def _maybe_call(example: Optional[SequenceOr[dict[str, Tensor]]], *, fn: Callable):
        if example is not None:
            processed_example = fn(example)
            # If this example is already dropped by inner function, we skip it by marking it None.
            if processed_example[input_key].size == 0:
                return None
            # fn returns a tensor with shape [1, ..]. We remove the first dimension.
            for v in processed_example.values():
                assert v.shape[0] == 1
            return {k: v[0, :] for k, v in processed_example.items()}
        return example

    # Converts dataset to IterDataset.
    ds = input_grain.maybe_to_iter_dataset(ds, read_options=read_options)
    ds = _StreamingPackingIterDataset(
        ds,
        max_len=max_len,
        window_size=window_size,
        input_key=input_key,
    )
    # Some examples might be dropped after calling inner. Grain IterDataset will automatically
    # handle it as long as we mark those examples as None.
    ds = ds.map(functools.partial(_maybe_call, fn=inner))
    ds = ds.filter(lambda x: x is not None)
    return ds


# TODO(markblee): Clean up the unused signatures.
def windowed_packing(
    ds: Dataset,
    *,
    max_len: Optional[int] = None,
    inner: Optional[Callable] = None,
    window_size: Optional[int] = None,
    input_key: str = "target_labels",
    read_options: grain.ReadOptions = grain.ReadOptions(num_threads=1, prefetch_buffer_size=16),
) -> Dataset:
    """Windowed packing given window_size.

    Given a sequence of tokens with arbitraty length, windowed packing will first batch the example
    given window_size then unbatch given max_len.

    Args:
        max_len: Max sequence length.
        ds: Datasets to be packed.
        inner: A processor that operates on packed examples. It should output examples of shape
            [1, sequence_length] or None if the example should be skipped.
        window_size: An upper bound on the window size to use for packing.
        input_key: The keys in the input examples to use for packing.
        read_options: grain.ReadOptions which includes num_threads and prefetch_buffer_size. It is
            used to convert the pipeline to grain.IterDataset.

    Returns:
        A packed dataset with dict which only contains values corresponding to `input_key`.
    """
    del max_len
    del input_key
    ds = ds.batch(window_size, drop_remainder=False, batch_fn=list)
    if inner is not None:
        ds = ds.map(inner)
    ds = input_grain.maybe_to_iter_dataset(
        ds,
        read_options=read_options,
    )
    # After processing, we have non-ragged np.arrays, so we can unbatch.
    ds = input_grain.unbatch(ds)
    return ds


# TODO(markblee): If we enforce that each input example is initially shorter than packed length, we
# can preserve the `grain.MapDataset` by using a flat map with fanout <= window_size.
def _make_autoregressive_inputs(
    ds: Dataset,
    *,
    max_len: int,
    packing_fn: _PackingFn,
    input_key: str = "target_labels",
    split_fn: Optional[ConfigOr[_SplitFn]] = None,
    read_options: grain.ReadOptions = grain.ReadOptions(num_threads=1, prefetch_buffer_size=16),
    window_size: int = 1,
) -> Dataset:
    """Produces `input_ids` autoregressively from `target_labels`.

    NOTE: this performs a conversion from `grain.MapDataset` to `grain.IterDataset`.
    See `On grain.MapDataset vs grain.IterDataset` in the file docstring of `input_grain`.

    Args:
        ds: A Dataset where each input example contains:
            `input_key`: A flat int Tensor of shape [None], i.e., length can vary across examples.
        max_len: Max sequence length.
        inner: A processor that operates on packed examples. It should output examples of shape ...
            or None if the example should be skipped.
        window_size: An upper bound on the window size to use for packing. If None, no upper bound
            is enforced.
        input_key: The keys in the input examples to use for packing.
        read_options: grain.ReadOptions which includes num_threads and prefetch_buffer_size. It is
            used to convert the pipeline to grain.IterDataset.

    Returns:
        A `grain.IterDataset` with potentially different cardinality than the input dataset.
        Each output example contains:
            `input_key`: An int Tensor with shape [max_len].
            "input_ids": An int Tensor with shape [max_len].
    """
    split_fn = maybe_instantiate(split_fn)
    if split_fn is None:
        split_fn = lambda ids, **_: ids[None]  # Passthrough ids.  # noqa: E731

    def process_example_fn(example: SequenceOr[dict[str, Tensor]]) -> dict[str, Tensor]:
        if not isinstance(example, Sequence):
            example = [example]
        flat_target_labels = np.concatenate([x[input_key] for x in example], axis=-1)
        flat_input_ids = np.roll(flat_target_labels, 1, axis=0)
        return dict(
            input_ids=split_fn(flat_input_ids, max_len=max_len),
            target_labels=split_fn(flat_target_labels, max_len=max_len),
        )

    packing_fn = functools.partial(
        packing_fn,
        max_len=max_len,
        inner=process_example_fn,
        window_size=window_size,
        read_options=read_options,
    )

    return packing_fn(ds)


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
    read_options: grain.ReadOptions = grain.ReadOptions(num_threads=1, prefetch_buffer_size=16),
    packing_fn: Callable = windowed_packing,
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
        read_options: grain.ReadOptions which includes num_threads and prefetch_buffer_size. It is
            used to convert the pipeline to grain.IterDataset.

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
    if isinstance(ds, MultiprocessPrefetchIterDataset):
        # Dataset types like MultiprocessPrefetchIterDataset have no len() or repeat() function
        logging.info("Skipping repeat for ds: %s`", ds)
    elif len(ds) != sys.maxsize:
        # Only repeat if not already infinite.
        ds = ds.repeat(num_epochs=None)
    ds = input_grain_text.tokenize(ds, vocab={"text": vocab}, with_eos=True)
    ds = input_grain.rekey(ds, key_map={"target_labels": "text"})
    # Flatten, roll, split.
    ds = _make_autoregressive_inputs(
        ds,
        max_len=max_len,
        window_size=window_size,
        split_fn=split_fn,
        read_options=read_options,
        packing_fn=packing_fn,
    )
    ds = input_grain_text.count_num_bytes(
        ds, input_key="target_labels", vocab=vocab, output_key="target_num_bytes"
    )
    # Note: In input_lm, there is an additional shuffle so that read order is not dominated by
    # document order. grain.IterDataset currently does not support shuffle, although it may be
    # doable with a shuffle-buffer style shuffling. Since shuffle buffers are memory intensive, we
    # skip the shuffle assuming sufficiently long context and source mixing.
    return ds


def _drop_empty_targets(example: dict[str, Tensor]) -> dict[str, Tensor]:
    # Drop examples that have 0 target bytes.
    mask = example["target_num_bytes"] > 0
    return {k: v[mask] for k, v in example.items()}


def text_to_lm_eval_input(
    ds: Dataset,
    *,
    vocab: ConfigOr[input_grain_text.Vocabulary],
    max_len: int,
    stride: Optional[int] = None,
    packing_fn: Callable = windowed_packing,
) -> Dataset:
    """Returns a function that generates eval inputs for language models from raw text.

    The processing follows `input_lm.text_to_lm_eval_input`.

    Args:
        ds: A Dataset where each example contains:
            "text": A string to be tokenized.
        vocab: A vocab or a config instantiating to a vocab. Any text normalization (such as
            `replace_newlines_with`) should be applied directly at the vocab.
        max_len: The maximum number of tokens per sequence.
        stride: The stride to use when slicing a tokenized document into examples.
            If None, defaults to max length as stride.

    Returns:
        A `grain.IterDataset` with potentially different cardinality than the input dataset.
        Each output example contains:
            "input_ids": An int Tensor with shape [max_len].
            "target_labels": An int Tensor with shape [max_len].
            "target_num_bytes": A scalar int Tensor.
    """
    vocab = maybe_instantiate(vocab)
    stride = max_len if stride is None else stride
    if not 0 < stride <= max_len:
        raise ValueError(f"Expected {stride=} to be in (0,{max_len}].")
    mask: Tensor = np.broadcast_to(vocab.pad_id, [max_len - stride])

    def strided_slice(example: dict[str, Tensor]) -> dict[str, Tensor]:
        output = {}
        for key, ids in example.items():
            assert ids.ndim == 1, ids
            # Pad to a multiple of `max_len`.
            remainder = ids.shape[0] % max_len
            ids = np.pad(ids, pad_width=((0, max_len - remainder)), constant_values=vocab.pad_id)
            # Produce strided slices.
            slices = [ids[:max_len]]
            for i in range(stride, ids.shape[0] - max_len + 1, stride):
                slice_ids = ids[i : i + max_len]
                if key == "target_labels":
                    slice_ids = np.concatenate([mask, slice_ids[mask.shape[-1] :]], axis=-1)
                slices.append(slice_ids)  # Append to list to stack later.
            output[key] = np.stack(slices)
        return output

    # Tokenize.
    ds = input_grain_text.tokenize(ds, vocab={"text": vocab}, with_eos=True)
    ds = input_grain.rekey(ds, key_map={"target_labels": "text"})

    # Make autoregressive and produce strided slices.
    ds = _make_autoregressive_inputs(ds, packing_fn=packing_fn, max_len=max_len, split_fn=None)
    ds = ds.map(strided_slice)

    # Produce batches.
    ds = input_grain_text.count_num_bytes(
        ds, input_key="target_labels", vocab=vocab, output_key="target_num_bytes"
    )
    ds = ds.map(_drop_empty_targets)
    ds = input_grain.maybe_to_iter_dataset(ds)
    ds = input_grain.unbatch(ds)
    return ds

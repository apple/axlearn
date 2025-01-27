# Copyright © 2023 Apple Inc.

"""Fake input modules."""

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import jax
import numpy as np
import tensorflow as tf

from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, as_numpy_array, as_tensor

if TYPE_CHECKING:
    # TODO(markblee): replace with generic "dataset" definition
    from axlearn.common.input_tf_data import BuildDatasetFn


class EmptyInput(Module):
    """Produces empty inputs."""

    @config_class
    class Config(Module.Config):
        """Configures EmptyInput."""

        is_training: Required[bool] = REQUIRED
        global_batch_size: Required[int] = REQUIRED  # The global batch size.
        total_num_batches: Optional[int] = None  # The total number of batches. If None, unlimited.
        source_length: int = 1024  # The length of a sequence (in tokens).
        max_token_id: int = 2048  # The maximum value a token-ID can take.

    def __init__(self, cfg: Config, *, parent=None):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if not cfg.is_training and cfg.total_num_batches is None:
            raise ValueError(f"total_num_batches should not be None if not is_training: {cfg}")
        self._prng_key = jax.random.PRNGKey(1)
        self._num_batches = 0

    def __next__(self):
        raise NotImplementedError(type(self))

    def __iter__(self):
        self._num_batches = 0
        return self

    def dataset(self):
        return self.__iter__()  # pylint: disable=unnecessary-dunder-call

    def batches(self, it: tf.data.Iterator) -> Iterable[Nested[Tensor]]:
        for input_batch in it:
            yield as_numpy_array(input_batch)


class FakeLmInput(EmptyInput):
    """Produces fake language modeling inputs."""

    def __next__(self):
        cfg = self.config
        self._num_batches += 1
        if cfg.total_num_batches is not None and self._num_batches > cfg.total_num_batches:
            raise StopIteration()
        self._prng_key, tokens_key = jax.random.split(self._prng_key, 2)
        if cfg.global_batch_size <= 0 or cfg.global_batch_size % jax.process_count() != 0:
            raise ValueError(
                f"Global batch size ({cfg.global_batch_size}) "
                f"must be positive and divisible by process count ({jax.process_count()})"
            )
        batch_size = cfg.global_batch_size // jax.process_count()
        tokens = jax.random.randint(
            tokens_key,
            shape=[batch_size, cfg.source_length + 1],
            minval=0,
            maxval=cfg.max_token_id,
            dtype=np.int32,
        )
        return as_tensor(
            dict(
                input_ids=tokens[:, :-1],
                target_labels=tokens[:, 1:],
                target_num_bytes=jax.numpy.ones((batch_size,), dtype=jax.numpy.int32),
            )
        )


class FakeSeq2SeqInput(EmptyInput):
    """Produces fake sequence-to-sequence inputs."""

    @config_class
    class Config(EmptyInput.Config):
        target_length: int = 1024  # The length of a target sequence (in tokens).

    def __next__(self):
        cfg = self.config
        self._num_batches += 1
        if cfg.total_num_batches is not None and self._num_batches > cfg.total_num_batches:
            raise StopIteration()
        if cfg.global_batch_size <= 0 or cfg.global_batch_size % jax.process_count() != 0:
            raise ValueError(
                f"Global batch size ({cfg.global_batch_size}) "
                f"must be positive and divisible by process count ({jax.process_count()})"
            )
        batch_size = cfg.global_batch_size // jax.process_count()
        self._prng_key, src_tokens_key, tgt_tokens_key = jax.random.split(self._prng_key, 3)
        src_tokens = jax.random.randint(
            src_tokens_key,
            shape=[batch_size, cfg.source_length + 1],
            minval=0,
            maxval=cfg.max_token_id,
            dtype=np.int32,
        )
        tgt_tokens = jax.random.randint(
            tgt_tokens_key,
            shape=[batch_size, cfg.target_length + 1],
            minval=0,
            maxval=cfg.max_token_id,
            dtype=np.int32,
        )
        return as_tensor(
            dict(
                source_ids=src_tokens[:, :-1],
                target_ids=tgt_tokens[:, :-1],
                prefix=jax.numpy.full((batch_size, 1), -1),
                target_labels=tgt_tokens[:, 1:],
            )
        )


class FakeSequenceClassificationInput(EmptyInput):
    """Produces fake sequence classification inputs."""

    @config_class
    class Config(EmptyInput.Config):
        """Configures FakeSequenceClassificationInput."""

        num_labels: int = 2  # The number of different classes.

    def __next__(self):
        cfg = self.config
        self._num_batches += 1
        if cfg.total_num_batches is not None and self._num_batches > cfg.total_num_batches:
            raise StopIteration()
        self._prng_key, tokens_key = jax.random.split(self._prng_key, 2)
        if cfg.global_batch_size <= 0 or cfg.global_batch_size % jax.process_count() != 0:
            raise ValueError(
                f"Global batch size ({cfg.global_batch_size}) "
                f"must be positive and divisible by process count ({jax.process_count()})"
            )
        batch_size = cfg.global_batch_size // jax.process_count()
        input_ids = jax.random.randint(
            tokens_key,
            shape=[batch_size, cfg.source_length],
            minval=0,
            maxval=cfg.max_token_id,
            dtype=np.int32,
        )
        target_labels = jax.random.randint(
            tokens_key,
            shape=[batch_size],
            minval=0,
            maxval=cfg.num_labels,
            dtype=np.int32,
        )
        return as_tensor(
            dict(
                input_ids=input_ids,
                target_labels=target_labels,
            )
        )


class FakeExtractiveQuestionAnsweringInput(EmptyInput):
    """Produces fake extractive QA inputs."""

    def __next__(self):
        cfg = self.config
        self._num_batches += 1
        if cfg.total_num_batches is not None and self._num_batches > cfg.total_num_batches:
            raise StopIteration()
        self._prng_key, tokens_key = jax.random.split(self._prng_key, 2)
        if cfg.global_batch_size <= 0 or cfg.global_batch_size % jax.process_count() != 0:
            raise ValueError(
                f"Global batch size ({cfg.global_batch_size}) "
                f"must be positive and divisible by process count ({jax.process_count()})"
            )
        batch_size = cfg.global_batch_size // jax.process_count()
        # TODO(@ivan-s-montero): Use fake_text_source so that tokenization will also be tested.
        input_ids = jax.random.randint(
            tokens_key,
            shape=[batch_size, cfg.source_length],
            minval=0,
            maxval=cfg.max_token_id,
            dtype=np.int32,
        )
        start_positions = jax.random.randint(
            tokens_key,
            shape=[batch_size],
            minval=0,
            maxval=cfg.source_length,
            dtype=np.int32,
        )
        end_positions = jax.random.randint(
            tokens_key,
            shape=[batch_size],
            minval=0,
            maxval=cfg.source_length,
            dtype=np.int32,
        )
        token_type_ids = (
            jax.numpy.zeros_like(input_ids).at[:, jax.numpy.min(start_positions) :].set(1)
        )
        return as_tensor(
            dict(
                input_ids=input_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                token_type_ids=token_type_ids,
            )
        )


def fake_source(
    is_training: bool,
    examples: Sequence[dict[str, tf.Tensor]],
    repeat: int = 1,
    spec: Optional[dict[str, tf.TypeSpec]] = None,
    shuffle_buffer_size: Optional[int] = None,
) -> "BuildDatasetFn":
    if len(examples) == 0:
        raise ValueError("examples cannot be empty")

    def data_gen():
        for _ in range(repeat):
            yield from examples

    def fn() -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            data_gen,
            # If None, attempt to infer spec from elements.
            output_signature=spec or tf.nest.map_structure(tf.type_spec_from_value, examples[0]),
        )
        if is_training:
            ds = ds.repeat()
        if shuffle_buffer_size:
            if not is_training:
                raise ValueError("Shuffling should be disabled if is_training=False")
            ds = ds.shuffle(shuffle_buffer_size)
        return ds

    return fn


def fake_text_source(
    *,
    text_field_name: str = "text",
    repeat: int = 1,
    is_training: bool,
    shuffle_buffer_size: Optional[int] = None,
    batch_size: int = 2,
) -> "BuildDatasetFn":
    return fake_source(
        is_training=is_training,
        examples=[
            {
                text_field_name: ("train" if is_training else "eval") + f" text {ix}",
            }
            for ix in range(2 * batch_size if is_training else batch_size)
        ],
        shuffle_buffer_size=shuffle_buffer_size,
        repeat=repeat,
    )


def fake_serialized_json_source(examples: Sequence[dict[str, Any]]) -> "BuildDatasetFn":
    """Returns a BuildDatasetFn that returns a dataset of jsonlines of examples.

    Args:
        examples: a sequence of dictionaries to dump.

    Returns:
        A BuildDatasetFn, which builds a dataset where each line is the dumped json string
        corresponding to the example.
    """

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for ex in examples:
                yield tf.convert_to_tensor(json.dumps(ex))

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature=tf.TensorSpec(shape=[], dtype=tf.string),
        )

    return ds_fn


def fake_text2text_source(
    *,
    source_key: str = "source_text",
    target_key: str = "target_text",
    is_training: bool,
    shuffle_buffer_size: Optional[int] = None,
) -> "BuildDatasetFn":
    return fake_source(
        is_training=is_training,
        examples=[
            {
                source_key: ("train" if is_training else "eval") + f" source {ix}",
                target_key: ("train" if is_training else "eval") + f" target {ix}",
            }
            for ix in range(2 * 2 if is_training else 2)
        ],
        shuffle_buffer_size=shuffle_buffer_size,
    )


def fake_glue_source(
    *,
    input_key: Union[str, tuple[str, str]],
    label_key: str,
    is_training: bool,
    label_value: Union[int, Sequence[int]] = 0,
    num_examples: Optional[int] = None,
    shuffle_buffer_size: Optional[int] = None,
    spec: Optional[dict[str, tf.TypeSpec]] = None,
) -> "BuildDatasetFn":
    if isinstance(input_key, str):
        input_key = [input_key]
    if num_examples is None:
        num_examples = 2 * 2 if is_training else 2
    return fake_source(
        is_training=is_training,
        examples=[
            {
                # Each example has an idx and label_key, as well as all input_keys.
                "idx": ix,
                label_key: label_value,
                **{key: ("train" if is_training else "eval") + f" {key} {ix}" for key in input_key},
            }
            for ix in range(num_examples)
        ],
        shuffle_buffer_size=shuffle_buffer_size,
        spec=spec,
    )


def fake_classification_source(
    *,
    source_key: str = "source_text",
    target_key: str = "target_text",
    is_training: bool,
    classes: Sequence[str],
    shuffle_buffer_size: Optional[int] = None,
) -> "BuildDatasetFn":
    num_classes = len(classes)
    return fake_source(
        is_training=is_training,
        examples=[
            {
                source_key: ("train" if is_training else "eval")
                + f" classification question {ix}, Choose {'. '.join(classes)}\n Answer:",
                target_key: f"{classes[ix % num_classes]}",
            }
            for ix in range(2 * 2 if is_training else 2)
        ],
        shuffle_buffer_size=shuffle_buffer_size,
    )


def fake_classification_source_instruct_lm(
    *,
    text_key: str = "text",
    is_training: bool,
    classes: Sequence[str] = ("yes", "no"),
    shuffle_buffer_size: Optional[int] = None,
    eoa_text: str = "<eoa>",
    eob_text: str = "<eob>",
) -> "BuildDatasetFn":
    """Returns a BuildDatasetFn containing fake classification examples in the InstructLM format.

    Args:
        text_key: Data field name.
        is_training: A boolean indicating whether it is in the training mode.
        classes: A sequence of strings that specifies the class labels.
        shuffle_buffer_size: Shuffle buffer size.
        eoa_text: Text for the EOA token used in the InstructLM tokenizer.
        eob_text: Text for the EOB token used in the InstructLM tokenizer.

    Returns:
        A BuildDatasetFn containing fake classification examples with the
            "<INSTRUCTION><EOA_TOKEN><RESPONSE><EOB_TOKEN>" format.

    Note: This function is useful as evaler source input during unittest where the evaler
        metric calculator may require specific class labels as targets.
        See `evaler_generative_text_classification.py` for example.
    """
    num_classes = len(classes)
    return fake_source(
        is_training=is_training,
        examples=[
            {
                text_key: ("train" if is_training else "eval")
                + f" classification question {ix}, Choose {'. '.join(classes)}\n Answer:"
                + eoa_text
                + f" {classes[ix % num_classes]}"
                + eob_text,
            }
            for ix in range(2 * 2 if is_training else 2)
        ],
        shuffle_buffer_size=shuffle_buffer_size,
    )


def fake_speech_source(
    *,
    is_training: bool,
    num_examples: int = 100,
    speech_key: str = "speech",
    shuffle_buffer_size: Optional[int] = None,
) -> "BuildDatasetFn":
    """Fake speech data source.

    Args:
        is_training: A boolean indicating whether it is in the training mode.
        num_examples: Integer of number of examples in the dataset.
        shuffle_buffer_size: Shuffle buffer size used for training.

    Returns:
        A BuildDatasetFn producing fake examples with `speech_key` with random integer values
        (assuming 16 bit-depth).
    """
    return fake_source(
        is_training=is_training,
        examples=[
            {
                speech_key: jax.random.randint(
                    jax.random.PRNGKey(ix),
                    minval=-(2**15),
                    maxval=2**15,
                    shape=[ix % 100 + 1],
                ),
            }
            for ix in range(num_examples)
        ],
        shuffle_buffer_size=shuffle_buffer_size,
        spec={speech_key: tf.TensorSpec(shape=(None,), dtype=tf.int16)},
    )


def fake_grain_source(
    examples: Sequence[Any],
    *,
    repeat: Optional[int] = 1,
    shuffle_seed: Optional[int] = None,
):
    """Returns a fake grain input source."""

    if len(examples) == 0:
        raise ValueError("Input examples cannot be empty.")

    # Lazy import to avoid introducing a global dependency.
    # pylint: disable-next=import-outside-toplevel
    from grain._src.python.dataset.transformations import source

    ds = source.SourceMapDataset(examples)
    if shuffle_seed is not None:
        ds = ds.seed(shuffle_seed)
        ds = ds.shuffle()  # Uses the configured seed, if provided.
    ds = ds.repeat(num_epochs=repeat)
    return ds

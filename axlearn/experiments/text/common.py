# Copyright © 2023 Apple Inc.

"""Common utilities used for text-based experiments."""
import os
from dataclasses import dataclass
from typing import Optional

import seqio

from axlearn.common import input_fake, input_tf_data, utils
from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.utils import get_data_dir


@dataclass(frozen=True)
class DataMixtureComponent:
    """Defines a dataset mixture component to be used for training a LMs."""

    name: str
    weight: float
    shuffle_buffer_size: int
    split: str = "train"
    # Unstructured info, for comments on the dataset.
    info: str = ""


def tfds_text_source(
    *,
    is_training: bool,
    dataset_name: str,
    split: str,
    read_parallelism: int = 64,
    decode_parallelism: int = 128,
    train_shuffle_buffer_size: int = 64 * 1024,
    fake_source_cfg: Optional[InstantiableConfig] = None,
) -> input_tf_data.BuildDatasetFn:
    """Builds a BuildDatasetFn with the given source.

    To interleave multiple input sources, use mix_inputs.

    Args:
        is_training: Whether the source is used for training.
        dataset_name: TFDS name.
        split: TFDS split.
        read_parallelism: Read parallelism.
        decode_parallelism: Decode parallelism.
        train_shuffle_buffer_size: Training shuffle buffer size.
            Shuffle buffer size will always be 0 during eval.
        fake_source_cfg: Optional fake source config. Defaults to a fake text source.

    Returns:
        A BuildDatasetFn.
    """
    shuffle_buffer_size = train_shuffle_buffer_size if is_training else 0
    if utils.get_data_dir() == "FAKE":
        source = fake_source_cfg or config_for_function(input_fake.fake_text_source)
        source.set(shuffle_buffer_size=min(64, shuffle_buffer_size))
    else:
        source = config_for_function(input_tf_data.tfds_dataset).set(
            dataset_name=dataset_name,
            split=split,
            read_config=config_for_function(input_tf_data.tfds_read_config).set(
                read_parallelism=read_parallelism,
                decode_parallelism=decode_parallelism,
            ),
            train_shuffle_buffer_size=shuffle_buffer_size,
        )
    return source.set(is_training=is_training).instantiate()


def vocab(
    *, sentencepiece_model_name: str, num_extra_ids: Optional[int] = None
) -> seqio.Vocabulary:
    """Construct a seqio Vocabulary given a data directory and sentence-piece model name.

    Args:
        sentencepiece_model_name: The name of the sentencepiece model file.
            We expect to find this relative to data_dir, but if data_dir is "FAKE"
            then we look for a local copy.
        num_extra_ids: Optionally to set number of extra tokens.

    Returns:
        seqio sentencepiece Vocabulary.
    """
    data_dir = get_data_dir()
    sentence_piece_vocab_file = os.path.join(
        (
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
            if data_dir is None or data_dir == "FAKE"
            else data_dir
        ),
        f"tokenizers/sentencepiece/{sentencepiece_model_name}",
    )
    return seqio.SentencePieceVocabulary(
        sentencepiece_model_file=sentence_piece_vocab_file, extra_ids=num_extra_ids
    )

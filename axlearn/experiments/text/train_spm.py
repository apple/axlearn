"""Train a SentencePiece model using TFDS derived data as the source corpus.

Examples:

    # BPE on c4.
    VOCAB=bpe_128k; \
    axlearn gcp vm start --name=$USER-axlearn-train-spm-128k \
        --vm_type=n2-highmem-128 --retain_vm -- \
    python3 -m axlearn.experiments.text.train_spm \
        --input_dataset_name=c4/en:3.0.1 \
        --data_dir=gs://${BUCKET}/tensorflow_datasets \
        --spm_config_file=axlearn/data/tokenizers/sentencepiece/${VOCAB}.json \
        --model_name=c4 \
        --max_train_examples=100000 \
        --output_dir=gs://${BUCKET}/tensorflow_datasets/tokenizers/sentencepiece

Note: SentencePiece training runs on CPU and typically consumes a lot of memory.
This depends on the corpus size and sentence length, but you'll probably want to use a highmem
machine, e.g. 400GB for around 100M sentences (more if training on full documents).

Note: The SentencePieceTrainer internally filters out sentences that exceed the max sentence length
defined in your config (or 4192 by default). You may consider increasing `--max_train_examples` to
ensure that the number of training examples is actually what you expect. Note that the trainer will
output a log indicating how many sentences are filtered, e.g:
https://github.com/google/sentencepiece/blob/8b02b2ca4cf0700cccb4cec85dd76fc3953f57bb/src/trainer_interface.cc#L413

The full set of default SentencePiece configs can be found here:
https://github.com/google/sentencepiece/blob/901368e0752b57a408ac5c84bca0a219d62c648f/doc/options.md

When replicating an existing SentencePiece model, you can also identify the original trainer
configs used by inspecting the proto:

    from sentencepiece import sentencepiece_model_pb2
    with open("/path/to/sentencepiece.model", "rb") as f:
        proto = sentencepiece_model_pb2.ModelProto.FromString(f.read())
    print(proto.trainer_spec)
"""

import json
import multiprocessing
import os
import re
from collections.abc import Iterator
from typing import Any, Optional

import sentencepiece as spm
import seqio
import tensorflow as tf
from absl import app, flags, logging

from axlearn.common import file_system as fs
from axlearn.common import input_tf_data


def _private_flags():
    # Dataset flags.
    flags.DEFINE_string(
        "input_dataset_name",
        None,
        "Input TFDS name. Also used to lookup the preprocessing function to use.",
        required=True,
    )
    flags.DEFINE_string("input_dataset_split", "train", "Input TFDS split name.")
    flags.DEFINE_string(
        "data_dir",
        os.path.expanduser("~/tensorflow_datasets"),
        "The TDFS data directory (for both input and output). Can be a gs:// url.",
    )
    flags.DEFINE_integer(
        "max_train_examples",
        100_000_000,
        "Maximum number of input dataset examples to use for training.",
    )
    flags.DEFINE_integer(
        "shuffle_buffer_size",
        64 * 1024,
        "Shuffle buffer size for shuffling when loading the TFDS.",
    )
    # SentencePiece flags.
    flags.DEFINE_string(
        "spm_config_file",
        None,
        "Path to SentencePiece JSON config file. "
        "Each key, value pair is passed directly to the trainer as kwargs.",
        required=True,
    )
    # Model name
    flags.DEFINE_string(
        "model_name",
        None,
        "Name of SentencePiece model to generate."
        "Defaults to `FLAGS.spm_config_file` if not provided.",
    )
    flags.DEFINE_string(
        "output_dir",
        None,
        "The output directory. If None, uses {FLAGS.data_dir}/tokenizers/sentencepience.",
    )


FLAGS = flags.FLAGS


def _build_tfds_iterator(is_training: bool) -> Iterator:
    """Builds an iterator backed by TFDS.

    Each input example is assumed to be a dict with a "text" field, which is extracted and
    returned as a byte string.

    Args:
        is_training: Whether the iterator is used for training or eval.

    Returns:
        Data iterator.
    """
    logging.info("Loading tfds dataset by name %s", FLAGS.input_dataset_name)
    ds_fn = input_tf_data.tfds_dataset(
        dataset_name=FLAGS.input_dataset_name,
        split=FLAGS.input_dataset_split,
        is_training=is_training,
        train_shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        data_dir=FLAGS.data_dir,
    )
    ds = ds_fn()
    ds = ds.map(lambda x: x["text"])
    # Filter out empty examples.
    ds = ds.filter(lambda x: tf.strings.length(x) > 0)
    ds = ds.take(FLAGS.max_train_examples)
    return ds.as_numpy_iterator()


def _build_spm_config(model_name: str) -> dict[str, Any]:
    """Builds the sentencpiece trainer config."""
    # Set some default config values.
    configs = dict(
        model_prefix=model_name,
        character_coverage=1,
        model_type="unigram",
        num_threads=min(multiprocessing.cpu_count(), 128),
        input_sentence_size=FLAGS.max_train_examples,
        train_extremely_large_corpus=True,
        shuffle_input_sentence=True,
        max_sentence_length=4192,
    )
    with open(FLAGS.spm_config_file, encoding="utf-8") as f:
        overrides = json.load(f)
    configs.update(overrides)

    # Make sure we define at least unk_id, bos_id, eos_id, since the defaults are
    # incompatible with seqio.
    unk_id = configs.get("unk_id")
    bos_id = configs.get("bos_id")
    eos_id = configs.get("eos_id")
    pad_id = configs.get("pad_id", 0)

    if pad_id != 0 or 0 in (unk_id, bos_id, eos_id):
        raise ValueError("For compat with seqio, pad_id (and only pad_id) should be 0.")
    return configs


def _format_name(config_name: str, model_name: Optional[str] = None) -> str:
    """Formats the output SentencePiece model name."""
    config_name = os.path.basename(config_name)
    config_name = os.path.splitext(config_name)[0]
    config_name = f"{config_name}_{model_name}" if model_name else config_name
    return re.sub("[:/.]", "_", f"spm_{config_name}")


def main(_):
    # Train sentencepiece model.
    model_name = _format_name(FLAGS.spm_config_file, FLAGS.model_name)
    spm_config = _build_spm_config(model_name)
    spm.SentencePieceTrainer.train(  # pylint: disable=no-member
        sentence_iterator=_build_tfds_iterator(is_training=True),
        **spm_config,
    )
    model_file = f"{model_name}.model"
    logging.info("Model output to: %s", model_file)
    # Validate trained model.
    vocab = seqio.SentencePieceVocabulary(model_file)
    eval_it = _build_tfds_iterator(is_training=False)
    for i in range(10):
        query = next(eval_it)
        ids = vocab.encode(query)
        logging.info("== Eval case: %d ==", i)
        logging.info("Query: %s", query)
        logging.info("Encoded IDs: %s", ids)
        logging.info("Decoded: %s", vocab.decode(ids))
    # Copy the files to the output dir.
    output_dir = FLAGS.output_dir or os.path.join(FLAGS.data_dir, "tokenizers", "sentencepiece")
    for vocab_file in fs.glob(f"{model_name}*"):
        output_file = os.path.join(output_dir, os.path.basename(vocab_file))
        logging.info("Copying %s to %s", vocab_file, output_file)
        fs.copy(vocab_file, output_file, overwrite=True)


if __name__ == "__main__":
    _private_flags()
    app.run(main)

# Copyright © 2024 Apple Inc.

"""Conformer on LibriSpeech 960h.

[1] https://arxiv.org/abs/2005.08100

To launch on GKE with 2 x v5e-256:
```
GS_ROOT=gs://my-bucket
CONFIG=conformer-l-rnnt
INSTANCE_TYPE=tpu-v5litepod-256;
ZONE=...
EXP=$(echo "${CONFIG}-$(date +%F-%H%M)" | tr '[:upper:]' '[:lower:]')
OUTPUT_DIR=$GS_ROOT/$USER/experiments/$EXP
axlearn gcp launch --zone=$ZONE --instance_type=$INSTANCE_TYPE --num_replicas=2 --gcp_api=gke \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=image=tpu --bundler_spec=target=tpu --bundler_spec=extras=audio \
    --output_dir=$OUTPUT_DIR --name=$USER-$EXP -- \
    python3 -m axlearn.common.launch_trainer_main \
    --module=audio.conformer.librispeech_trainer --config=$CONFIG \
    --trainer_dir=$OUTPUT_DIR \
    --data_dir=$GS_ROOT/tensorflow_datasets \
    --mesh_selector=$INSTANCE_TYPE --jax_backend=tpu
```
"""

import functools
import os
from typing import Dict, Optional, Protocol

import jax.numpy as jnp
import seqio
import tensorflow as tf
from jax.sharding import PartitionSpec

from axlearn.audio.encoder_asr import SpeechFeatureLayer
from axlearn.audio.evaler_asr import WordErrorRateMetricCalculator
from axlearn.audio.frontend import LogMelFrontend
from axlearn.audio.frontend_utils import sharded_fft
from axlearn.audio.input_asr import pad_example_fn
from axlearn.audio.model_asr import ASRModel
from axlearn.audio.spectrum_augmenter import MaskSampler, SpectrumAugmenter
from axlearn.audio.subsamplers import ConvSubSampler
from axlearn.common import learner
from axlearn.common.checkpointer import every_n_steps_policy as save_every_n_steps
from axlearn.common.config import InstantiableConfig, config_for_class, config_for_function
from axlearn.common.decoding import PrefixMerger
from axlearn.common.evaler import SpmdEvaler
from axlearn.common.evaler import every_n_steps_policy as eval_every_n_steps
from axlearn.common.input_fake import fake_speech_source, fake_text_source
from axlearn.common.input_tf_data import BuildDatasetFn, Input, batch, tfds_dataset
from axlearn.common.layers import Conv2DWith1DPadding
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import get_data_dir
from axlearn.experiments.audio.conformer.common import (
    adam_learner_config,
    asr_input,
    ctc_decoder_config,
    encoder_config,
    rnn_transducer_config,
    stack_config,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn


def source_config(is_training: bool, split: str) -> InstantiableConfig[BuildDatasetFn]:
    """Builds a source dataset config for librispeech.

    Args:
        is_training: Whether source is for training.
        split: Dataset split.

    Returns:
        A config that instantiates to a data source.
    """
    if get_data_dir() == "FAKE":

        def fake_asr_source(is_training: bool):
            def fn():
                text_ds = fake_text_source(is_training=is_training)()
                speech_ds = fake_speech_source(is_training=is_training)()
                return tf.data.Dataset.zip((speech_ds, text_ds)).map(lambda s, t: {**s, **t})

            return fn

        source = config_for_function(fake_asr_source).set(is_training=is_training)
    else:
        source = config_for_function(tfds_dataset).set(
            dataset_name="librispeech:2.1.0",
            split=split,
            # In practice shuffle buffer will always be 0 when is_training=False, but we set it
            # explicitly for golden config readability.
            train_shuffle_buffer_size=16 * 1024 if is_training else 0,
        )
    return source


def feature_config(*, dim: int, jax_backend: Optional[str] = None) -> SpeechFeatureLayer.Config:
    """Constructs speech feature layer config.

    See [1] Section 3.1 for details. The feature layer consists of the log-mel frontend, spectrum
    augmenter, and convolution subsampler.

    Args:
        dim: Feature dimension, after subsampling.
        jax_backend: JAX backend. If "gpu", a `shard_map` implementation of FFT will be used.

    Returns:
        A feature layer config.
    """
    frontend_cfg = LogMelFrontend.default_config().set(
        num_filters=80, sample_rate=16000, frame_size_ms=25, hop_size_ms=10, mel_floor=1e-8
    )
    if jax_backend == "gpu":
        # On GPU, use FFT + 'shard_map' by default until a fix is merged for
        # https://github.com/openxla/xla/issues/24.
        frontend_cfg.set(
            fft=config_for_function(sharded_fft).set(
                partition_spec=PartitionSpec(
                    "data",
                )
            )
        )
    augmenter_cfg = SpectrumAugmenter.default_config().set(
        freq_mask_sampler=MaskSampler.default_config().set(max_num_masks=2, max_mask_length=27),
        time_mask_sampler=MaskSampler.default_config().set(
            max_num_masks=10, max_mask_length_ratio=0.05
        ),
    )
    # (B, L, 80, 1) -> (B, L//2, 40, 512) -> (B, L//4, 20, 512) -> (B, L//4, encoder_dim).
    subsampler_cfg = ConvSubSampler.default_config().set(
        conv=Conv2DWith1DPadding.default_config().set(
            window=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), bias=True
        ),
        norm=None,  # Don't apply BatchNorm after conv2d.
        activation="nn.relu",  # Apply relu in sub-sampling conv layers.
    )
    return SpeechFeatureLayer.default_config().set(
        output_dim=dim, frontend=frontend_cfg, augmenter=augmenter_cfg, subsampler=subsampler_cfg
    )


def _vocab_config(
    sentencepiece_model_file: str = "librispeech_bpe_1024.model",
) -> InstantiableConfig[seqio.SentencePieceVocabulary]:
    """Constructs a config that instantiates to a SentencePiece vocabulary.

    By default, this uses a unigram SentencePiece model trained on librispeech.
    """
    data_dir = get_data_dir()
    if data_dir == "FAKE":
        data_dir = os.path.join(os.path.dirname(__file__), "../../../data")
    sentencepiece_dir = os.path.join(data_dir, "tokenizers/sentencepiece")
    return config_for_class(seqio.SentencePieceVocabulary).set(
        sentencepiece_model_file=os.path.join(sentencepiece_dir, sentencepiece_model_file)
    )


class _InputFn(Protocol):
    def __call__(self, *, is_training: bool, split: str) -> Input.Config:
        ...


def evaler_config_dict(
    *,
    input_fn: _InputFn,
    max_decode_len: Optional[int] = None,
    prefix_merger: Optional[PrefixMerger] = None,
    eval_dtype: jnp.dtype = jnp.float32,
) -> dict[str, SpmdEvaler.Config]:
    """Makes evaler configs for librispeech splits.

    Specifically, we construct WER evalers for dev-clean/other and test-clean/other.

    Args:
        input_fn: A callable (is_training, split) -> input config.
        max_decode_len: Max decode length for decoding evals.
        prefix_merger: Optional prefix merger.
        eval_dtype: Eval dtype.

    Returns:
        A mapping from evaler name to evaler config.
    """
    decode_kwargs = dict(num_decodes=8)
    if prefix_merger:
        decode_kwargs["prefix_merger"] = prefix_merger
    if max_decode_len:
        decode_kwargs["max_decode_len"] = max_decode_len
    decode_metric_calculator = WordErrorRateMetricCalculator.default_config().set(
        model_method="beam_search_decode", vocab=_vocab_config(), model_method_kwargs=decode_kwargs
    )
    evalers: dict[str, SpmdEvaler.Config] = {}
    for name, eval_type, split in [
        ("train", "eval", "train_clean100[:512]+train_clean360[:512]+train_other500[:512]"),
        ("dev_clean", "eval", "dev_clean"),
        ("dev_other", "eval", "dev_other"),
        ("dev_clean", "decoder", "dev_clean"),
        ("dev_other", "decoder", "dev_other"),
        ("test_clean", "decoder", "test_clean"),
        ("test_other", "decoder", "test_other"),
        ("train", "decoder", "train_clean100[:512]+train_clean360[:512]+train_other500[:512]"),
    ]:
        evaler_name = f"{eval_type}_{name}"
        evalers[evaler_name] = SpmdEvaler.default_config().set(
            input=input_fn(is_training=False, split=split),
            eval_policy=config_for_function(eval_every_n_steps).set(n=1000, min_step=1),
            eval_dtype=eval_dtype,
        )
        if eval_type == "decoder":
            evalers[evaler_name].metric_calculator = decode_metric_calculator
    return evalers


def _trainer_config_fn(
    *,
    input_cfg: Input.Config,
    model_cfg: ASRModel.Config,
    evaler_cfgs: dict[str, SpmdEvaler.Config],
    learner_cfg: learner.Learner.Config,
    max_step: int,
) -> TrainerConfigFn:
    """Returns a trainer config fn.

    Args:
        input_cfg: Training input processor config.
        model_cfg: ASR model config.
        evaler_cfgs: A mapping from evaler name to evaler config.
        learner_cfg: A learner config.
        max_step: Max training steps.

    Returns:
        A function that returns a trainer config.
    """

    def config_fn() -> SpmdTrainer.Config:
        cfg: SpmdTrainer.Config = SpmdTrainer.default_config().set(
            name="librispeech_trainer",
            model=model_cfg,
            input=input_cfg,
            evalers=evaler_cfgs,
            learner=learner_cfg,
            max_step=max_step,
        )
        cfg.checkpointer.save_policy = config_for_function(save_every_n_steps).set(n=100)
        cfg.checkpointer.keep_every_n_steps = 100
        cfg.summary_writer.write_every_n_steps = 200
        return cfg

    return config_fn


def model_config(
    *,
    model_name: str,
    eos_id: int,
    vocab_size: int = 1024,
    dropout_rate: Optional[float] = 0.1,
) -> ASRModel.Config:
    """Builds an ASR model config.

    Args:
        model_name: A model name of the format "conformer-{size}-{decoder}", where size can be one
            of "test", "s", "m", "l" and decoder can be one of "ctc", "rnnt".
        eos_id: EOS token ID.
        vocab_size: Vocab size.
        dropout_rate: Dropout rate to apply throughout encoder.

    Returns:
        An ASR model config.
    """
    _, size, decoder = model_name.lower().split("-")  # E.g, "conformer-l-ctc".

    # See Table 1 of Conformer paper [1].
    if size == "test":
        feature_dim, encoder_dim, num_layers, num_heads = 4, 4, 1, 2
    elif size == "l":
        feature_dim, encoder_dim, num_layers, num_heads = 512, 512, 17, 8
    elif size == "m":
        feature_dim, encoder_dim, num_layers, num_heads = 512, 256, 16, 4
    elif size == "s":
        feature_dim, encoder_dim, num_layers, num_heads = 512, 144, 16, 4
    else:
        raise ValueError(f"Unsupported {size=}.")

    if "rnnt" in decoder:
        decoder_cfg = rnn_transducer_config(
            vocab_size=vocab_size, lm_dim=640, emb_dim=128, joint_dim=640, eos_id=eos_id
        )
    elif "ctc" in decoder:
        decoder_cfg = ctc_decoder_config(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unsupported {model_name=}.")

    return ASRModel.default_config().set(
        encoder=encoder_config(
            dim=encoder_dim,
            stack_cfg=stack_config(num_layers=num_layers, num_heads=num_heads),
            feature_cfg=feature_config(dim=feature_dim),
            dropout_rate=dropout_rate,
        ),
        decoder=decoder_cfg,
        dtype=jnp.float32,
    )


def _input_fn(is_training: bool, split: str, eos_id: Optional[int] = None) -> Input.Config:
    """Builds an input config for librispeech ASR."""

    if get_data_dir() == "FAKE":
        max_source_len = max_target_len = 20
        global_batch_size = 2
    else:
        # Use a longer sequence length for eval.
        # For librispeech:2.1.0 these are sufficient to avoid dropping any examples.
        max_source_len = 16_000 * 30 if is_training else 16_000 * 36
        max_target_len = 200
        global_batch_size = 2048 if is_training else 512

    return Input.default_config().set(
        source=source_config(is_training=is_training, split=split),
        processor=config_for_function(asr_input).set(
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            vocab_cfg=_vocab_config(),
            eos_id=eos_id,
        ),
        batcher=config_for_function(batch).set(
            global_batch_size=global_batch_size, pad_example_fn=pad_example_fn
        ),
    )


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""

    train_split = "train_clean100+train_clean360+train_other500"
    return {
        # Used for unit test only.
        "conformer-test-ctc": _trainer_config_fn(
            input_cfg=_input_fn(is_training=True, split=train_split, eos_id=2),
            model_cfg=model_config(
                model_name="conformer-test-ctc", eos_id=2, dropout_rate=None, vocab_size=32
            ),
            evaler_cfgs={},
            learner_cfg=adam_learner_config(lr_scale=0.1),
            max_step=3,
        ),
        "conformer-l-rnnt": _trainer_config_fn(
            input_cfg=_input_fn(is_training=True, split=train_split, eos_id=2),
            model_cfg=model_config(model_name="conformer-l-rnnt", eos_id=2, dropout_rate=0.1),
            evaler_cfgs=evaler_config_dict(
                input_fn=functools.partial(_input_fn, eos_id=2),
                max_decode_len=1000,
            ),
            learner_cfg=adam_learner_config(lr_scale=0.22097),
            max_step=500_000,
        ),
    }

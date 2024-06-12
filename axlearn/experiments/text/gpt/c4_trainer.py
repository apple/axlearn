# Copyright © 2023 Apple Inc.

"""GPT models on C4.

    mkdir -p /tmp/gpt_c4_test;
    python3 -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-test-v1 \
        --trainer_dir=/tmp/gpt_c4_test --data_dir=FAKE --jax_backend=cpu \
        --status_port=7337

    GS_ROOT=gs://my-bucket; \
    CONFIG=fuji-7B-v2; \
    INSTANCE_TYPE=tpu-v4-1024; \
    NUM_TPU_SLICES=1; \
    EXP=$(echo "text-gpt-c4-${CONFIG}-$(date +%F-%H%M)" | tr '[:upper:]' '[:lower:]'); \
    OUTPUT_DIR=$GS_ROOT/$USER/experiments/$EXP; \
    axlearn gcp launch --zone=$ZONE --instance_type=$INSTANCE_TYPE --num_slices=${NUM_TPU_SLICES} \
        --output_dir=$OUTPUT_DIR --name=$USER-$EXP -- \
        python3 -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=$CONFIG \
        --trainer_dir=$OUTPUT_DIR \
        --data_dir=$GS_ROOT/tensorflow_datasets \
        --mesh_selector=$INSTANCE_TYPE --jax_backend=tpu

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
bash Miniconda3-latest-Linux-x86_64.sh; \
bash
conda create -n axlearn python=3.10; \
conda activate axlearn; \
git clone https://github.com/apple/axlearn.git; \
cd axlearn; \
pip install -e .
XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/test_trainer; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-7B-v2-single \
  --trainer_dir=/tmp/test_trainer --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
"""
import functools
from typing import Dict

from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.input_lm import lm_text_preprocessor
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import get_data_dir
from axlearn.experiments.text.common import DataMixtureComponent, vocab
from axlearn.experiments.text.gpt import fuji
from axlearn.experiments.text.gpt.common import (
    evaler_config_dict,
    get_trainer_config_fn,
    make_config_name,
    mixture_train_input_source,
    tfds_input,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

# Sentencepiece vocabs generated from c4/en:3.0.1.
# See bpe_{32k,128k}.json for the sentencepiece settings.
_SENTENCEPIECE_MODEL_NAME = {
    32 * 1024: "bpe_32k_c4.model",
    128 * 1024: "bpe_128k_c4.model",  # TODO(ruoming): build the 128k vocab.
}


def _eval_input_sources(
    *, vocab_cfg: InstantiableConfig, max_sequence_length: int
) -> Dict[str, InstantiableConfig]:
    return {
        name: config_for_function(tfds_input).set(
            dataset_name="c4/en:3.0.1",
            split=split,
            is_training=False,
            vocab_cfg=vocab_cfg,
            max_sequence_length=max_sequence_length,
        )
        for name, split in (("train", "train[:8192]"), ("validation", "validation"))
    }


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    arch = "fuji"
    train_data_mixture_components = [
        DataMixtureComponent(
            name="c4/en:3.0.1",
            split="train",
            shuffle_buffer_size=8192,
            weight=1.0,
        ),
    ]

    config_map = {}
    for version in fuji.Version:
        vocab_size = fuji.VOCAB_SIZE[version]
        vocab_cfg = config_for_function(vocab).set(
            sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME[vocab_size]
        )
        train_input_source = config_for_function(mixture_train_input_source).set(
            data_mixture_components=train_data_mixture_components,
            vocab_cfg=vocab_cfg,
            preprocessor=config_for_function(lm_text_preprocessor).set(max_padding_fraction=0.5),
        )
        if get_data_dir() == "FAKE":
            train_input_source.preprocessor.shuffle_buffer_size = 0
        for model_size in fuji.MODEL_SIZES:
            config_name = make_config_name(
                arch=arch, model_size=model_size, version=f"v{version.value}"
            )
            kwargs = fuji.get_trainer_kwargs(model_size, vocab_size=vocab_size, version=version)
            max_sequence_length = kwargs.pop("max_sequence_length")

            # TODO remove before merging
            kwargs["max_step"] = 1000
            # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
            config_map[config_name] = get_trainer_config_fn(
                train_input_source=train_input_source.clone(
                    max_sequence_length=max_sequence_length
                ),
                evalers=evaler_config_dict(
                    _eval_input_sources(
                        vocab_cfg=vocab_cfg, max_sequence_length=max_sequence_length
                    ),
                ),
                **kwargs,
            )
            kwargs_flash = fuji.get_trainer_kwargs(
                model_size,
                vocab_size=vocab_size,
                version=version,
                flash_attention=True,
            )
            max_sequence_length = kwargs_flash.pop("max_sequence_length")
            # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
            config_map[(f"{config_name}-flash")] = get_trainer_config_fn(
                train_input_source=train_input_source.clone(
                    max_sequence_length=max_sequence_length
                ),
                evalers=evaler_config_dict(
                    _eval_input_sources(
                        vocab_cfg=vocab_cfg, max_sequence_length=max_sequence_length
                    ),
                ),
                **kwargs_flash,
            )
            if model_size == "test":

                def wrapper(config_name: str = config_name):
                    trainer_cfg: SpmdTrainer.Config = config_map[config_name]()
                    trainer_cfg.max_step = 5
                    # Make learning rate large to accentuate any differences.
                    trainer_cfg.learner.optimizer.args[1].learning_rate = 0.3
                    trainer_cfg.learner.optimizer.args[1].update_schedule = 1
                    trainer_cfg.vlog = 1
                    return trainer_cfg

                config_map[
                    make_config_name(
                        arch=arch, model_size="golden-run-test", version=f"v{version.value}"
                    )
                ] = wrapper
            if model_size == "7B":

                def make_single_host_config(base_config_name: str) -> SpmdTrainer.Config:
                    """Make a single-host variant of the base config.

                    gpu-p5.48xlarge 8x1 step time:
                    128K tokens per batch: 2.03s for v1.
                    64K tokens per batch:  1.1s for v1, 1.54s for v2.

                    tpu-v5litepod-32 step time:
                    128K tokens per batch: 1.93s for v1.

                    Args:
                        base_config_name: The multi-host config name.

                    Returns:
                        A trainer config that can run on a single host.
                    """

                    # pytype: disable=annotation-type-mismatch
                    cfg: SpmdTrainer.Config = config_map[base_config_name]().clone()
                    # pytype: enable=annotation-type-mismatch

                    # The original config was supposed to run on >= 32 machines.
                    cfg.input.batcher.global_batch_size //= 32
                    for evaler in cfg.evalers.values():
                        evaler.input.batcher.global_batch_size //= 32
                    return cfg

                config_map[f"{config_name}-single-host"] = functools.partial(
                    make_single_host_config, config_name
                )
                config_map[f"{config_name}-flash-single-host"] = functools.partial(
                    make_single_host_config, f"{config_name}-flash"
                )
    return config_map

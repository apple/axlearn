# Copyright © 2023 Apple Inc.

"""GPT models on C4.

    mkdir -p /tmp/gpt_c4_test;
    python3 -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-test \
        --trainer_dir=/tmp/gpt_c4_test --data_dir=FAKE --jax_backend=cpu

    GS_ROOT=gs://my-bucket; \
    CONFIG=fuji-7B; \
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
  --module=text.gpt.c4_trainer --config=fuji-7B-single \
  --trainer_dir=/tmp/test_trainer --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
"""

from typing import Dict

from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.input_lm import lm_text_preprocessor
from axlearn.common.trainer import SpmdTrainer
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

# A sentencepiece vocab generated from c4/en:3.0.1.
# See bpe_32k.json for the sentencepiece settings.
_SENTENCEPIECE_MODEL_NAME = "bpe_32k_c4.model"


def _eval_input_sources() -> Dict[str, InstantiableConfig]:
    return {
        name: config_for_function(tfds_input).set(
            dataset_name="c4/en:3.0.1",
            split=split,
            is_training=False,
            vocab_cfg=config_for_function(vocab).set(
                sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME
            ),
            max_sequence_length=fuji.MAX_SEQUENCE_LENGTH,
        )
        for name, split in (("train", "train[:8192]"), ("validation", "validation"))
    }


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    arch = "fuji"
    vocab_size = 32_768
    train_data_mixture_components = [
        DataMixtureComponent(
            name="c4/en:3.0.1",
            split="train",
            shuffle_buffer_size=8192,
            weight=1.0,
        ),
    ]
    vocab_cfg = config_for_function(vocab).set(sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME)
    train_input_source = config_for_function(mixture_train_input_source).set(
        data_mixture_components=train_data_mixture_components,
        vocab_cfg=vocab_cfg,
        preprocessor=config_for_function(lm_text_preprocessor).set(max_padding_fraction=0.5),
    )

    config_map = {}
    for model_size in fuji.MODEL_SIZES:
        config_name = make_config_name(arch=arch, model_size=model_size)
        kwargs = fuji.get_trainer_kwargs(model_size, vocab_size=vocab_size)
        # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
        config_map[config_name] = get_trainer_config_fn(
            train_input_source=train_input_source.clone(
                max_sequence_length=kwargs.pop("max_sequence_length", fuji.MAX_SEQUENCE_LENGTH),
            ),
            evalers=evaler_config_dict(_eval_input_sources()),
            **kwargs,
        )
    # Make a variant of fuji-7B that can run on a single machine with 8 80G GPUs.
    # pytype: disable=annotation-type-mismatch
    cfg: SpmdTrainer.Config = config_map["fuji-7B"]().clone()
    # pytype: enable=annotation-type-mismatch
    cfg.input.batcher.global_batch_size = 32
    for evaler in cfg.evalers.values():
        evaler.input.batcher.global_batch_size = 32
    config_map["fuji-7B-single"] = lambda: cfg
    return config_map

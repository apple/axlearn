# Copyright © 2024 Apple Inc.

"""GPT models on Pajama.

First setup your local environment and install AXLearn following `docs/01-start.md`.
Test the trainer locally on CPU using:
```shell
mkdir -p /tmp/gpt_pajama_test;
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.pajama_trainer --config=gala-test-4k-sp-rp \
    --trainer_dir=/tmp/gpt_pajama_test --data_dir=FAKE --jax_backend=cpu \
    --status_port=7337
```

You can run this trainer on GPU using:
```shell
XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/test_trainer; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.pajama_trainer --config=gala-7B-4k-sp-rp \
  --trainer_dir=/tmp/test_trainer --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
```

Launch this on TPU using:
```shell
GS_ROOT=gs://my-bucket; \
CONFIG=gala-7B-4k-sp-rp; \
INSTANCE_TYPE=tpu-v4-1024; \
NUM_TPU_SLICES=1; \
EXP=$(echo "text-gpt-pajama-${CONFIG}-$(date +%F-%H%M)" | tr '[:upper:]' '[:lower:]'); \
OUTPUT_DIR=$GS_ROOT/$USER/experiments/$EXP; \
axlearn gcp launch --zone=$ZONE --instance_type=$INSTANCE_TYPE --num_slices=${NUM_TPU_SLICES} \
    --output_dir=$OUTPUT_DIR --name=$USER-$EXP -- \
    python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.pajama_trainer --config=$CONFIG \
    --trainer_dir=$OUTPUT_DIR \
    --data_dir=$GS_ROOT/tensorflow_datasets \
    --mesh_selector=$INSTANCE_TYPE --jax_backend=tpu
```
"""
from typing import Callable, Dict

from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.input_lm import lm_text_preprocessor
from axlearn.common.utils import get_data_dir
from axlearn.experiments.text.common import DataMixtureComponent, vocab
from axlearn.experiments.text.gpt import gala
from axlearn.experiments.text.gpt.common import (
    REPLACE_NEWLINES_WITH,
    mixture_train_input_source,
    tfds_input,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

# Sentencepiece vocabs generated from c4/en:3.0.1.
# See bpe_{32k,128k}.json for the sentencepiece settings.
_SENTENCEPIECE_MODEL_NAME = {
    32 * 1024: "bpe_32k_c4.model",
}


def _eval_input_sources(
    *, vocab_size: int, max_sequence_length: int
) -> Dict[str, InstantiableConfig]:
    return {
        name: config_for_function(tfds_input).set(
            dataset_name=dataset_name,
            split=split,
            is_training=False,
            vocab_cfg=config_for_function(vocab).set(
                sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME[vocab_size],
            ),
            max_sequence_length=max_sequence_length,
            replace_newlines_with=REPLACE_NEWLINES_WITH,
        )
        for dataset_name, name, split in (
            ("pile/openwebtext_2:1.0.0", "openwebtext_2", "test[:5000]"),
        )
    }


DATASETS = {
    "sp-rp": [
        DataMixtureComponent(
            name="rpg/common_crawl:1.0.0",
            split="train",
            shuffle_buffer_size=8192,
            weight=0.726,
        ),
        DataMixtureComponent(
            name="slimpajama/c4:1.0.0",
            split="train",
            shuffle_buffer_size=8192,
            weight=0.081,
        ),
        DataMixtureComponent(
            name="slimpajama/github:1.0.0",
            split="train",
            shuffle_buffer_size=8192,
            weight=0.049,
        ),
        DataMixtureComponent(
            name="slimpajama/book:1.0.0",
            split="train",
            shuffle_buffer_size=8192,
            weight=0.021,
        ),
        DataMixtureComponent(
            name="slimpajama/arxiv:1.0.0",
            split="train",
            shuffle_buffer_size=8192,
            weight=0.023,
        ),
        DataMixtureComponent(
            name="slimpajama/wikipedia:1.0.0",
            split="train",
            shuffle_buffer_size=8192,
            weight=0.05,
        ),
        DataMixtureComponent(
            name="slimpajama/stackexchange:1.0.0",
            split="train",
            shuffle_buffer_size=8192,
            weight=0.05,
        ),
    ]
}


def _train_input_source_fn(
    *, train_data_mixture_components
) -> Callable[[int, int], InstantiableConfig]:
    def fn(vocab_size: int, max_sequence_length: int) -> InstantiableConfig:
        source_cfg = config_for_function(mixture_train_input_source).set(
            data_mixture_components=train_data_mixture_components,
            vocab_cfg=config_for_function(vocab).set(
                sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME[vocab_size]
            ),
            max_sequence_length=max_sequence_length,
            preprocessor=config_for_function(lm_text_preprocessor).set(max_padding_fraction=0.5),
        )
        if get_data_dir() == "FAKE":
            source_cfg.preprocessor.shuffle_buffer_size = 0
        return source_cfg

    return fn


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    config_map = {}
    for dataset_name, train_data_mixture_components in DATASETS.items():
        dataset_config_map = {}
        dataset_train_input_source = _train_input_source_fn(
            train_data_mixture_components=train_data_mixture_components
        )
        dataset_config_map.update(
            gala.trainer_configs(dataset_train_input_source, _eval_input_sources)
        )

        # Include the dataset name in the config name.
        dataset_config_map = {f"{k}-{dataset_name}": v for k, v in dataset_config_map.items()}
        config_map.update(dataset_config_map)
    return config_map

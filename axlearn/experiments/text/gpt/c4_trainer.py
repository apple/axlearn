# Copyright © 2023 Apple Inc.

"""GPT models on C4.

First setup your local environment and install AXLearn following `docs/01-start.md`.
Test the trainer locally on CPU with fake configs/data (commonly used for testing):
```shell
mkdir -p /tmp/gpt_c4_test;
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-test-v1 \
    --trainer_dir=/tmp/gpt_c4_test --data_dir=FAKE --jax_backend=cpu \
    --status_port=7337
```

Example training Fuji-7B with C4 dataset (can run on a single H100):
```shell
XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/gpt_c4_test; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer --config=fuji-7B-v2-single-host \
  --trainer_dir=/tmp/gpt_c4_test --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
```

Launch this on TPU using:
```shell
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
```
"""


from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.input_lm import lm_text_preprocessor
from axlearn.common.utils import get_data_dir
from axlearn.experiments.text.common import DataMixtureComponent, vocab
from axlearn.experiments.text.gpt import fuji, gspmd
from axlearn.experiments.text.gpt.common import mixture_train_input_source, tfds_input
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

# Sentencepiece vocabs generated from c4/en:3.0.1.
# See bpe_{32k,128k}.json for the sentencepiece settings.
_SENTENCEPIECE_MODEL_NAME = {
    32 * 1024: "bpe_32k_c4.model",
    # TikToken is not yet supported, so we are using sentencepiece for now.
    # Our new grain-based inputs can support TikToken in the future.
    128256: "bpe_128k_c4.model",
}
_train_data_mixture_components = [
    DataMixtureComponent(
        name="c4/en:3.0.1",
        split="train",
        shuffle_buffer_size=8192,
        weight=1.0,
    ),
]


def _eval_input_sources(
    *, vocab_size: int, max_sequence_length: int
) -> dict[str, InstantiableConfig]:
    return {
        name: config_for_function(tfds_input).set(
            dataset_name="c4/en:3.0.1",
            split=split,
            is_training=False,
            vocab_cfg=config_for_function(vocab).set(
                sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME[vocab_size]
            ),
            max_sequence_length=max_sequence_length,
        )
        for name, split in (("train", "train[:8192]"), ("validation", "validation"))
    }


def _train_input_source(*, vocab_size: int, max_sequence_length: int) -> InstantiableConfig:
    source_cfg = config_for_function(mixture_train_input_source).set(
        data_mixture_components=_train_data_mixture_components,
        vocab_cfg=config_for_function(vocab).set(
            sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME[vocab_size]
        ),
        max_sequence_length=max_sequence_length,
        preprocessor=config_for_function(lm_text_preprocessor).set(max_padding_fraction=0.5),
    )
    if get_data_dir() == "FAKE":
        source_cfg.preprocessor.shuffle_buffer_size = 0
    return source_cfg


def named_trainer_configs() -> dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    config_map = {}
    config_map.update(fuji.trainer_configs(_train_input_source, _eval_input_sources))
    config_map.update(gspmd.trainer_configs(_train_input_source, _eval_input_sources))
    return config_map

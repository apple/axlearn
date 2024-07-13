# Copyright © 2024 Apple Inc.

"""GPT models on Pajama with Sigmoid attention.

First setup your local environment and install AXLearn following `docs/01-start.md`.
Test the trainer locally on CPU using:
```shell
mkdir -p /tmp/gpt_pajama_sigmoid_test;
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.pajama_sigmoid_trainer --config=gala-sigmoid-test-4k-sp-rp \
    --trainer_dir=/tmp/gpt_pajama_sigmoid_test --data_dir=FAKE --jax_backend=cpu \
    --status_port=7337
```

You can run this trainer on GPU using:
```shell
XLA_FLAGS=--xla_dump_to=/tmp/xla_dump; \
mkdir -p /tmp/test_trainer; \
python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.pajama_sigmoid_trainer --config=gala-sigmoid-1B-2k-sp-rp \
  --trainer_dir=/tmp/test_trainer --data_dir=gs://axlearn-public/tensorflow_datasets \
  --jax_backend=gpu
```

Launch this on TPU using:
```shell
GS_ROOT=gs://my-bucket; \
CONFIG=gala-sigmoid-7B-4k-sp-rp; \
INSTANCE_TYPE=tpu-v4-1024; \
NUM_TPU_SLICES=1; \
EXP=$(echo "text-gpt-pajama-${CONFIG}-$(date +%F-%H%M)" | tr '[:upper:]' '[:lower:]'); \
OUTPUT_DIR=$GS_ROOT/$USER/experiments/$EXP; \
axlearn gcp launch --zone=$ZONE --instance_type=$INSTANCE_TYPE --num_slices=${NUM_TPU_SLICES} \
    --output_dir=$OUTPUT_DIR --name=$USER-$EXP -- \
    python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.pajama_sigmoid_trainer --config=$CONFIG \
    --trainer_dir=$OUTPUT_DIR \
    --data_dir=$GS_ROOT/tensorflow_datasets \
    --mesh_selector=$INSTANCE_TYPE --jax_backend=tpu
```
"""

from typing import Dict

from axlearn.experiments.text.gpt import gala_sigmoid, pajama_trainer
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

# Mapping between shorthand name and sequence length.
MAX_SEQUENCE_LENGTH = {"2k": 2048}

MODEL_SIZES = {"85M", "1B"}


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    arch = "gala-sigmoid"
    sigmoid_cfg_map = {}

    base_cfg_map = pajama_trainer.named_trainer_configs()
    for base_cfg_name, cfg in base_cfg_map.items():
        model_size = base_cfg_name.split("-")[1]
        if model_size not in MODEL_SIZES:
            continue
        if "flash" in base_cfg_name:
            continue

        for seq_len_name, seq_len in MAX_SEQUENCE_LENGTH.items():
            # Update the arch name.
            sigmoid_cfg_name = base_cfg_name.replace("gala", arch)
            # Add the sequence length.
            sigmoid_cfg_name = sigmoid_cfg_name.replace(model_size, f"{model_size}-{seq_len_name}")
            sigmoid_cfg_map[sigmoid_cfg_name] = gala_sigmoid.build_sigmoid_trainer_config(
                cfg, max_sequence_length=seq_len, flash_attention=False
            )
    return sigmoid_cfg_map

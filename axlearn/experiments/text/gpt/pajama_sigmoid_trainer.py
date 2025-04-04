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


from dataclasses import dataclass

from axlearn.experiments.text.gpt import deterministic_trainer, gala, gala_sigmoid, pajama_trainer
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

# Mapping between sequence length and shorthand name.
MAX_SEQUENCE_LENGTH_NAME: dict[int, str] = {2048: "2k", 4096: "4k"}


@dataclass
class _SigmoidConfigArgs:
    seq_len: int = 4096
    norm_structure: gala.SupportedNormStructure = "hybridnorm"
    position_encoding: gala.SupportedPositionEncoding = "alibi"


MODEL_SIZE_CONFIG_ARGS: dict[str, list[_SigmoidConfigArgs]] = {
    "85M": [
        # Used for hyperparameter search.
        _SigmoidConfigArgs(),
    ],
    "1B": [
        _SigmoidConfigArgs(),
    ],
    "7B": [
        _SigmoidConfigArgs(),
    ],
}


def _filter_softmax_trainer_name(*, base_cfg_name: str, model_size: str) -> bool:
    """Passes trainer names that Sigmoid should use.

    We only create a sigmoid version if:
        - It's a Gala config.
        - If the model size is supported by sigmoid.
        - If it's a 4k seq. length cfg.
        - If it's a default, pre-norm + rope config.

    Args:
        base_cfg_name: Config name of the softmax version.
        model_size: The number of transformer parameters (not including vocab embeddings).

    Returns:
        True if it's a config we should create a sigmoid version of.
    """
    if "gala" not in base_cfg_name:
        return False
    if model_size not in MODEL_SIZE_CONFIG_ARGS:
        # We only need a subset for sigmoid.
        return False
    if "2048" in base_cfg_name:
        # We create multiple sequence versions in this trainer,
        # so skip base configs with non-4k seq. length.
        return False
    if "hybridnorm" in base_cfg_name or "alibi" in base_cfg_name:
        # Skip non-default configs. We generate them in this file.
        return False
    return True


def _config_name_suffix(
    *,
    model_size: str,
    suffix: str,
    norm_structure: gala.SupportedNormStructure,
    position_encoding: gala.SupportedPositionEncoding,
    attn_bias: bool,
    norm_scale_param: bool,
) -> str:
    """Creates the config name suffix for Sigmoid."""
    config_name_parts = [model_size, suffix]
    config_name_parts.append(
        gala.config_name_suffix(
            norm_structure=norm_structure, position_encoding=position_encoding, enable_flash=False
        )
    )
    if not attn_bias:
        config_name_parts.append("noattnbias")
    if not norm_scale_param:
        config_name_parts.append("nscale")
    return "-".join(config_name_parts)


def named_trainer_configs() -> dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    arch = "gala-sigmoid"
    sigmoid_cfg_map = {}

    base_cfg_map = pajama_trainer.named_trainer_configs()
    for base_cfg_name, cfg in base_cfg_map.items():
        model_size = base_cfg_name.split("-")[1]
        if not _filter_softmax_trainer_name(base_cfg_name=base_cfg_name, model_size=model_size):
            continue
        if "flash" in base_cfg_name:
            # Don't create additional configs for all of the flash variants.
            # Sigmoid doesn't support flash here, so we ignore those.
            continue

        for config_args in MODEL_SIZE_CONFIG_ARGS[model_size]:
            seq_len = config_args.seq_len
            norm_structure = config_args.norm_structure
            position_encoding = config_args.position_encoding
            seq_len_name = MAX_SEQUENCE_LENGTH_NAME[seq_len]

            # Update the arch name.
            sigmoid_cfg_name = base_cfg_name.replace("gala", arch)
            # Generate the full config name.
            sigmoid_cfg_name = sigmoid_cfg_name.replace(
                model_size,
                _config_name_suffix(
                    model_size=model_size,
                    suffix=seq_len_name,
                    norm_structure=norm_structure,
                    position_encoding=position_encoding,
                    attn_bias=True,
                    norm_scale_param=True,
                ),
            )
            sigmoid_cfg_map[sigmoid_cfg_name] = gala_sigmoid.build_sigmoid_trainer_config(
                cfg,
                max_sequence_length=seq_len,
                flash_attention=False,
                position_encoding=position_encoding,
                norm_structure=norm_structure,
                attn_bias=True,
            )

    # Create deterministic trainer configs for Sigmoid too.
    # This only supports 4k sequence length.
    for base_cfg_name, cfg in deterministic_trainer.named_trainer_configs().items():
        model_size = base_cfg_name.split("-")[1]
        if not _filter_softmax_trainer_name(base_cfg_name=base_cfg_name, model_size=model_size):
            continue
        for config_args in MODEL_SIZE_CONFIG_ARGS[model_size]:
            norm_structure = config_args.norm_structure
            position_encoding = config_args.position_encoding
            seq_len = 4096
            # Update the arch name.
            sigmoid_cfg_name = base_cfg_name.replace("gala", arch)
            # Remove flash (since we don't use flash).
            sigmoid_cfg_name = sigmoid_cfg_name.replace("-flash", "")
            # Generate the full config name.
            sigmoid_cfg_name = sigmoid_cfg_name.replace(
                model_size,
                _config_name_suffix(
                    model_size=model_size,
                    suffix=f"deterministic-{MAX_SEQUENCE_LENGTH_NAME[seq_len]}",
                    norm_structure=norm_structure,
                    position_encoding=position_encoding,
                    attn_bias=True,
                    norm_scale_param=True,
                ),
            )
            sigmoid_cfg_map[sigmoid_cfg_name] = gala_sigmoid.build_sigmoid_trainer_config(
                cfg,
                max_sequence_length=seq_len,
                flash_attention=False,
                position_encoding=position_encoding,
                norm_structure=norm_structure,
                attn_bias=True,
            )
    return sigmoid_cfg_map

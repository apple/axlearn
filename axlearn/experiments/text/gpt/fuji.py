# Copyright © 2023 Apple Inc.

"""Utilities to set up the 'fuji' GPT model trainer configs.

Architecture names follow apple varieties: Fuji, Gala, Honeycrisp, etc.

The fuji models are set up to imitate LLaMA models:
* LLaMA: https://arxiv.org/abs/2302.13971
* LLaMA 2: https://arxiv.org/abs/2307.09288
* LLaMA 3: https://github.com/meta-llama/llama3
"""

import enum
import functools
import itertools
from typing import Any, List, NamedTuple, Optional, Union

from jax.ad_checkpoint import checkpoint_policies as jax_remat_policies

from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    FusedGroupedQKVLinear,
    FusedQKVLinear,
    GroupedQKVLinear,
    GroupedQueryAttention,
    KVCache,
    MultiheadAttention,
    RematRegexSavePatterns,
    RepeatedTransformerLayer,
    RoFormerQKVLinear,
    StackedTransformerLayer,
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.config import TrainerConfigFn, config_for_function
from axlearn.common.decoder import LmHead
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.flash_attention.layer import FlashBlockSizeModifier
from axlearn.common.flash_attention.remat import save_or_offload_flash_attention_policy
from axlearn.common.layers import RMSNorm
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    FP8ConfigModifier,
    GradientAccumulationModifier,
    MeshShapeModifier,
    ModuleConfigModifier,
    PartitionSpecModifier,
    RematSpecModifier,
)
from axlearn.common.utils import (
    combine_remat_policies,
    extended_checkpoint_policies,
    save_and_offload_only_these_names_regex,
)
from axlearn.experiments.text.gpt.common import (
    STEP_DTYPE,
    SourceBuilder,
    adamw_decoupled_learner_config,
    evaler_config_dict,
    flash_attention_config,
    get_trainer_config_fn,
    make_config_name,
    mesh_shape_from_axes,
)
from axlearn.experiments.text.gpt.common import model_config as common_model_config
from axlearn.experiments.text.gpt.common import scaled_hidden_dim
from axlearn.experiments.trainer_config_utils import V6eFlashConfigModifier

MODEL_SIZES = ("test", "1B", "3B", "7B", "8B", "70B", "405B")


class Version(enum.Enum):
    V1 = 1
    V2 = 2
    V3 = 3
    V3_TIKTOKEN = "3-tiktoken"


# Mapping from Fuji versions to vocab sizes.
VOCAB_SIZE = {
    Version.V1: 32 * 1024,
    Version.V2: 32 * 1024,
    Version.V3: 128 * 1024,
    Version.V3_TIKTOKEN: 128256,
}


# Mapping from Fuji versions to maximum sequence lengths.
MAX_SEQUENCE_LENGTH = {
    Version.V1: 2048,
    Version.V2: 4096,
    Version.V3: 8192,
    Version.V3_TIKTOKEN: 8192,
}


ROPE_THETA = {
    Version.V1: 1e4,
    Version.V2: 1e4,
    Version.V3: 5e5,
    Version.V3_TIKTOKEN: 5e5,
}

# Mapping from Fuji versions to total number of tokens used in training.
TOTAL_TOKENS = {
    Version.V1: {
        "test": 1 * (1024**4),  # 1T tokens
        "7B": 1 * (1024**4),  # 1T tokens
        "70B": int(1.4 * (1024**4)),  # 1.4T tokens
    },
    Version.V2: {
        "test": 2 * (1024**4),  # 2T tokens
        "7B": 2 * (1024**4),  # 2T tokens
        "70B": 2 * (1024**4),  # 2T tokens
    },
    Version.V3: {
        "test": 15 * (1024**4),  # 15T tokens
        "1B": 15 * (1024**4),  # 15T tokens
        "3B": 15 * (1024**4),  # 15T tokens
        "7B": 15 * (1024**4),  # 15T tokens
        "70B": 15 * (1024**4),  # 15T tokens
        "405B": 15 * (1024**4),  # 15T tokens
    },
    Version.V3_TIKTOKEN: {
        "test": 15 * (1024**4),  # 15T tokens
        "1B": 15 * (1024**4),  # 15T tokens
        "3B": 15 * (1024**4),  # 15T tokens
        "8B": 15 * (1024**4),  # 15T tokens
        "70B": 15 * (1024**4),  # 15T tokens
        "405B": 15 * (1024**4),  # 15T tokens
    },
}


def offload_dots_saveable_policy(*_, **__):
    """A rematerialization policy function used in RematSpec to offload dot_general_p
    operations from device to pinned host memory.

    Args:
        *_: Ignored positional arguments.
        **__: Ignored keyword arguments.

    Returns:
        A policy function that offloads dot_general_p from device to pinned host
    memory.
    """
    return config_for_function(extended_checkpoint_policies.offload_dots_saveable).set(
        offload_src="device", offload_dst="pinned_host"
    )


def offload_attention_proj_policy(*_, **__):
    """A rematerialization policy function used in RematSpec to offload attention
    projection intermediates during model execution.

    Args:
        *_: Ignored positional arguments.
        **__: Ignored keyword arguments.

    Returns:
        A checkpoint policy function that offloads native attention projection intermediates
        from device to pinned host memory, enabling memory-efficient training with checkpoint
        support.
    """
    return config_for_function(
        extended_checkpoint_policies.save_and_offload_only_these_names_regex
    ).set(
        names_which_can_be_saved=None,
        names_which_can_be_offloaded=RematRegexSavePatterns.NATIVE_ATTENTION.value,
        offload_src="device",
        offload_dst="pinned_host",
    )


# Llama3 uses 16m tokens after 2.87T tokens.
# https://arxiv.org/pdf/2407.21783
TOKENS_PER_BATCH = {
    Version.V1: 4 * (1024**2),
    Version.V2: 4 * (1024**2),
    Version.V3: 16 * (1024**2),
    Version.V3_TIKTOKEN: 16 * (1024**2),
}


class _Trn2CustomConfig(NamedTuple):
    """Config modifications required to run Fuji models on TRN2."""

    # Module config modifications.
    module_modifications: List[ModuleConfigModifier.Config]
    # Partition spec modifications.
    partition_spec_modifications: List[PartitionSpecModifier.Config]


def _generate_trn2_custom_configs(
    model_size: str,
    *,
    version: Version,
) -> _Trn2CustomConfig:
    """Generate custom module config and PartitionSpec modification for TRN2.

    Args:
        model_size: Size of the Fuji model.
        version: Version of the Fuji model.

    Returns:
        A _Trn2CustomConfig object that contains the generated modifications.
    """
    # TRN2 specific model config modifications.
    trn2_module_modifications = [
        # Neuron compiler has a module to detect repeating blocks and reuse them during compilation.
        # So compile time does not grow with the number of layers.
        ModuleConfigModifier.default_config().set(
            target_config="model.decoder.transformer",
            modification=StackedTransformerLayer.default_config(),
        )
    ]
    # Grouped QKV is only used in fuji-v3 except in fuji-v2 if model is 70B.
    if version == Version.V3 or (model_size == "70B" and version != Version.V1):
        trn2_module_modifications.append(
            ModuleConfigModifier.default_config().set(
                target_config="model.decoder.transformer.layer.self_attention.attention."
                "input_linear.input_linear",
                modification=GroupedQKVLinear.default_config(),
            )
        )

    trn2_partition_spec_modifications = [
        PartitionSpecModifier.default_config().set(
            partition_specs={
                # Vocab parallel embeddings sharding from Megatron LM.
                "model.decoder.emb.token_emb": {
                    "param_partition_spec": (
                        "model",
                        ("expert", "fsdp", "seq"),
                    ),
                    "input_partition_spec": (("data", "fsdp"), None),
                    "output_partition_spec": (("data", "fsdp"), None, None),
                    "embedding_partition_spec": ("model", None),
                },
                # Sequence parallel shardings for norms.
                "model.decoder.transformer.layer.self_attention.norm": {
                    "input_partition_spec": (("data", "fsdp"), "model", None),
                    "output_partition_spec": (("data", "fsdp"), None, None),
                },
                "model.decoder.transformer.layer.feed_forward.norm": {
                    "input_partition_spec": (("data", "fsdp"), "model", None),
                    "output_partition_spec": (("data", "fsdp"), None, None),
                },
                "model.decoder.output_norm": {
                    "input_partition_spec": (("data", "fsdp"), "model", None),
                    "output_partition_spec": (("data", "fsdp"), None, None),
                },
                "model.decoder.transformer.layer.feed_forward.linear2": {
                    "output_partition_spec": (("data", "fsdp"), None, None),
                },
            },
        ),
    ]

    trn2_lm_head_partition_spec = [
        PartitionSpecModifier.default_config().set(
            partition_specs={
                # Vocab parallel embeddings sharding from Megatron LM.
                "model.decoder.lm_head": {
                    "param_partition_spec": (
                        "model",
                        ("expert", "fsdp", "seq"),
                    ),
                },
            },
        ),
    ]
    if model_size in ("70B", "8B"):
        trn2_partition_spec_modifications += trn2_lm_head_partition_spec

    return _Trn2CustomConfig(
        module_modifications=trn2_module_modifications,
        partition_spec_modifications=trn2_partition_spec_modifications,
    )


def get_trainer_kwargs(
    model_size: str,
    *,
    vocab_size: int,
    version: Version,
    flash_attention: bool = False,
) -> dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
    tokens_per_batch = TOKENS_PER_BATCH[version]
    max_step = TOTAL_TOKENS[version][model_size] // tokens_per_batch
    max_sequence_length = MAX_SEQUENCE_LENGTH[version]
    train_batch_size = tokens_per_batch // max_sequence_length

    # Whether to use grouped query attention.
    num_kv_heads = None
    if version in (Version.V3, Version.V3_TIKTOKEN):
        num_kv_heads = 8

    rope_theta = ROPE_THETA[version]

    trn2_config = _generate_trn2_custom_configs(model_size, version=version)

    # dict() is more readable here.
    # pylint: disable=use-dict-literal
    if model_size == "test":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=4,
                hidden_dim=8,
                ffn_dim=scaled_hidden_dim(scale=8 / 3, round_up_to_multiples_of=16),
                num_heads=4,
                num_kv_heads=2,
                vocab_size=32,
                rope_theta=rope_theta,
                shared_lm_head=True,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=6e-4, weight_decay=0.01),
            max_sequence_length=64,
            train_batch_size=32,
            eval_batch_size=32,
            max_step=3000,
            eval_every_n_steps=1500,
            save_every_n_steps=500,
            mesh_shape=mesh_shape_from_axes(data=-1),
        )
    elif model_size == "1B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=16,
                hidden_dim=2048,
                num_heads=32,
                num_kv_heads=num_kv_heads,
                ffn_dim=8192,
                rope_theta=rope_theta,
                shared_lm_head=True,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=3e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(data=-1, fsdp=8),
            mesh_rules=(
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=mesh_shape_from_axes(fsdp=-1, model=4)
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "3B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=28,
                hidden_dim=3072,
                num_heads=24,
                num_kv_heads=num_kv_heads,
                ffn_dim=8192,
                rope_theta=rope_theta,
                shared_lm_head=True,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=3e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(data=-1, fsdp=8),
            mesh_rules=(
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=mesh_shape_from_axes(fsdp=-1, model=4)
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                            GradientAccumulationModifier.default_config().set(
                                grad_acc_steps=4,
                            ),
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "7B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=32,
                hidden_dim=128 * 32,
                num_heads=32,
                num_kv_heads=num_kv_heads,
                rope_theta=rope_theta,
                shared_lm_head=True,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=3e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(data=-1, fsdp=8),
            mesh_rules=(
                # Step time:
                # v1 on tpu-v4-1024 (512 chips):            3.03s
                # v1 on tpu-v5litepod-256x4 (1024 chips):   2.44s
                # v1 on tpu-v5p-512 (256 chips):            2.85s
                # v1 on gpu-p5.48xlarge-256 (256 chips):    2.44s
                # v1 on gpu-p5.48xlarge-512 (512 chips):    1.54s
                #
                # tpu-v4-(1024|2048).
                ("tpu-v4-(1024|2048)", mesh_shape_from_axes(data=-1, fsdp=16)),
                # tpu-v5e.
                (
                    "tpu-v5litepod-256",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_dots_saveable_policy,
                                    ),
                                }
                            ),
                            GradientAccumulationModifier.default_config().set(grad_acc_steps=4),
                        ],
                    ),
                ),
                (
                    "tpu-v5litepod-256-2",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_dots_saveable_policy,
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                # v2 on tpu-v5litepod-256x4: 2.21s (46% MFU), HBM usage: 14GB/chip.
                (
                    "tpu-v5litepod-256-4",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False, policy=jax_remat_policies.dots_saveable
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v6e-256-(2|4|8)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_attention_proj_policy,
                                    ),
                                }
                            ),
                            V6eFlashConfigModifier.default_config(),
                        ],
                    ),
                ),
                (
                    "tpu-v5p-.*",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # fsdp=8 is also ok, only 2% slower step time.
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=64)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=config_for_function(combine_remat_policies).set(
                                            policy_1=save_or_offload_flash_attention_policy(),
                                            policy_2=jax_remat_policies.dots_saveable,
                                        ),
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                # H100/A100 80G.
                # Maximum per-node batch size = 64, hence need >= 32 nodes.
                # Without sequence sharding, the maximum number of devices <= batch_size,
                # so at most 512 GPUs (64 nodes) for training 7B-v3.
                # v2 on gpu-p5.48xlarge-256, step time: 1.78s/step, MFU 39%.
                # TODO(kelvin-zou): need to match 1.5s/step perf on TransformerEngine.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge)-(256|512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=8),
                ),
                (
                    "gpu-(a3-highgpu-8g|a3-megagpu-8g|a3-ultragpu-8g)-(256|512|1024)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=8)
                            )
                        ],
                    ),
                ),
                # Ensure the gpu_block_size is updated for Blackwell (B200 / A4)
                (
                    "gpu-(a4-highgpu-8g)-(256|512|1024)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=8)
                            ),
                            # Modify the GPU block-size for B200 platform (Pallas kernels)
                            FlashBlockSizeModifier.default_config().set(gpu_block_size=64),
                        ],
                    ),
                ),
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=mesh_shape_from_axes(fsdp=-1, model=4)
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "8B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=32,
                hidden_dim=128 * 32,
                num_heads=32,
                num_kv_heads=num_kv_heads,
                ffn_dim=scaled_hidden_dim(scale=3.5, round_up_to_multiples_of=256),
                rope_theta=rope_theta,
                shared_lm_head=False,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=3e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(data=-1, fsdp=8),
            mesh_rules=(
                ("tpu-v4-(1024|2048)", mesh_shape_from_axes(data=-1, fsdp=16)),
                (
                    "tpu-v5litepod-256",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_dots_saveable_policy,
                                    ),
                                }
                            ),
                            GradientAccumulationModifier.default_config().set(grad_acc_steps=4),
                        ],
                    ),
                ),
                (
                    "tpu-v5litepod-256-2",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_dots_saveable_policy,
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v5litepod-256-4",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False, policy=jax_remat_policies.dots_saveable
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                ("tpu-v5p-.*", mesh_shape_from_axes(data=-1, fsdp=8)),
                (
                    # pylint: disable=line-too-long
                    "gpu-(p5.48xlarge|p4de.24xlarge|a3-highgpu-8g|a3-megagpu-8g|a3-ultragpu-8g|a4-highgpu-8g)-(256|512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=8),
                ),
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=mesh_shape_from_axes(fsdp=-1, model=4)
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "70B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=80,
                hidden_dim=128 * 64,
                num_heads=64,
                # No GQA support in V1 models, so num_kv_heads is the same as num_heads.
                num_kv_heads=None if version == Version.V1 else 8,
                # TODO(kelvin-zou): Remove the perf numbers for V5e (OOM).
                ffn_dim=scaled_hidden_dim(scale=3.5, round_up_to_multiples_of=256),
                rope_theta=rope_theta,
                shared_lm_head=False,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=1.5e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # TPU V5e maximum per device batch is 1.
                # with all activation offloading, HBM usage: 14.6GB/chip.
                # TODO(kelvin-zou): Fix the env issue for internal use cases.
                # tpu-v5e-256-4. step time: 14.3736s (59.87% MFU).
                (
                    "tpu-v5litepod-256-4",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_dots_saveable_policy,
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v5p-.*",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(fsdp=-1)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=config_for_function(
                                            save_and_offload_only_these_names_regex
                                        ).set(
                                            names_which_can_be_saved="|".join(
                                                [
                                                    RematRegexSavePatterns.FLASH_ATTENTION.value,
                                                    ".*linear1_0",
                                                ]
                                            ),
                                            names_which_can_be_offloaded=None,
                                            offload_src="device",
                                            offload_dst="pinned_host",
                                        ),
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                # V2 on tpu-v6e-256x4, step time: 4.9s.
                (
                    "tpu-v6e-256-(4|8)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_attention_proj_policy,
                                    ),
                                }
                            ),
                            V6eFlashConfigModifier.default_config(),
                        ],
                    ),
                ),
                # V2 on tpu-v6e-256, step time: 19.5s.
                (
                    "tpu-v6e-256",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=256)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=offload_attention_proj_policy,
                                    ),
                                }
                            ),
                            V6eFlashConfigModifier.default_config(),
                            GradientAccumulationModifier.default_config().set(grad_acc_steps=4),
                        ],
                    ),
                ),
                # H100/A100 80G. Maximum per-node batch size = 16, hence need >= 64 nodes.
                # v2 on gpu-p5.48xlarge 8x64, step time: 12.9s.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge)-(512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=128),
                ),
                (
                    "gpu-(a3-highgpu-8g|a3-megagpu-8g|a3-ultragpu-8g)-(256|512|1024)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=64)
                            ),
                        ],
                    ),
                ),
                (
                    "gpu-(a4-highgpu-8g)-(256|512|1024)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=16)
                            ),
                            # Modify the GPU block-size for B200 platform (Pallas kernels)
                            FlashBlockSizeModifier.default_config().set(gpu_block_size=64),
                        ],
                    ),
                ),
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=mesh_shape_from_axes(fsdp=-1, model=4)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=True,
                                        policy=config_for_function(
                                            save_and_offload_only_these_names_regex
                                        ).set(
                                            names_which_can_be_saved="|".join(
                                                [
                                                    RematRegexSavePatterns.QKV_PROJ.value,
                                                    RematRegexSavePatterns.LINEAR1_X.value,
                                                ]
                                            ),
                                            names_which_can_be_offloaded=None,
                                            offload_src=None,
                                            offload_dst=None,
                                        ),
                                    ),
                                }
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "405B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=126,
                hidden_dim=16384,
                ffn_dim=53248,
                num_heads=128,
                # No GQA support in V1 models, so num_kv_heads is the same as num_heads.
                num_kv_heads=None if version == Version.V1 else 8,
                rope_theta=rope_theta,
                shared_lm_head=False,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=1.5e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                (
                    "tpu-v5p-.*",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(fsdp=-1)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=False,
                                        policy=config_for_function(
                                            save_and_offload_only_these_names_regex
                                        ).set(
                                            names_which_can_be_saved=None,
                                            names_which_can_be_offloaded=None,
                                            offload_src="device",
                                            offload_dst="pinned_host",
                                        ),
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
            ),
        )
    else:
        raise NotImplementedError(f"Unknown model size {model_size}.")
    model_kwargs = trainer_kwargs.pop("model_kwargs")
    model_kwargs.setdefault("vocab_size", vocab_size)
    if version == Version.V3_TIKTOKEN:  # tiktoken tokenizer
        model_kwargs["pad_token_id"] = 128004
        model_kwargs["eos_token_id"] = 128001
    trainer_kwargs["model_cfg"] = model_config(**model_kwargs)
    trainer_kwargs["learner_cfg"] = adamw_decoupled_learner_config(
        max_step=trainer_kwargs["max_step"],
        **trainer_kwargs.pop("learner_kwargs"),
    )
    # pylint: enable=use-dict-literal
    return trainer_kwargs


def model_config(
    *,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: Optional[int],
    vocab_size: int,
    rope_theta: float,
    shared_lm_head: bool,
    dropout_rate: float = 0.0,
    ffn_dim: Optional[Union[int, config.FunctionConfigBase]] = None,
    flash_attention: bool = False,
    stack_cfg: Optional[BaseStackedTransformerLayer.Config] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        num_layers: The number of Transformer Layers.
        hidden_dim: The Transformer layer input/output dim.
        num_heads: The number of attention heads.
        num_kv_heads: The optional number of KV heads. If not None, enables grouped query attention.
        vocab_size: The vocabulary size.
        rope_theta: The theta value used for RoPE positional embeddings.
        shared_lm_head: Whether lm_head shares the parameters with emb.
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        ffn_dim: The feed-forward dimension or config function.
            If None, defaults to a setting from https://arxiv.org/abs/2002.05202.
        flash_attention: Whether to enable flash attention.
        stack_cfg: The transformer stack config.
            If None, defaults to a RepeatedTransformerLayer.
        pad_token_id: Int ID of the inputs to be masked for self-attention.
        eos_token_id: Int ID of the end of sequence token id.

    Returns:
        A causal LM config.
    """
    # SwiGLU from https://arxiv.org/abs/2002.05202.
    activation_fn = ("nn.silu", "linear")
    if ffn_dim is None:
        ffn_dim = scaled_hidden_dim(scale=8 / 3, round_up_to_multiples_of=256)
    if num_kv_heads:
        atten_cfg = GroupedQueryAttention.default_config()
        atten_input_linear = FusedGroupedQKVLinear.default_config().set(num_kv_heads=num_kv_heads)
    else:
        atten_cfg = MultiheadAttention.default_config()
        atten_input_linear = FusedQKVLinear.default_config()
    # RoPE embeddings: https://arxiv.org/abs/2104.09864.
    atten_qkv_linear = RoFormerQKVLinear.default_config().set(
        input_linear=atten_input_linear,
        rotary_value=False,
    )
    atten_qkv_linear.rope_pos_emb_layer.theta = rope_theta
    attention_kv_cache = KVCache.default_config().set(cache_dtype=STEP_DTYPE)

    cfg = common_model_config(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        stack_cfg=stack_cfg if stack_cfg is not None else RepeatedTransformerLayer.default_config(),
        activation_fn=activation_fn,
        ffn_dim=ffn_dim,
        normalization=RMSNorm.default_config().set(eps=1e-5, forward_dtype=None),
        dropout_rate=dropout_rate,
        emb_cfg=TransformerTextEmbeddings.default_config().set(pos_emb=None),
        lm_head_cfg=LmHead.default_config() if not shared_lm_head else None,
        attention_cfg=flash_attention_config() if flash_attention else atten_cfg,
        attention_qkv_linear=atten_qkv_linear,
        attention_kv_cache=attention_kv_cache,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    return cfg


def trainer_configs(
    train_input_source: SourceBuilder, eval_input_sources: SourceBuilder
) -> dict[str, TrainerConfigFn]:
    """Returns a mapping from config_name to TrainerConfigFn's.

    Args:
        train_input_source: A callable (vocab_size, max_sequence_length) -> input source config.
        eval_input_soruces: A callable (vocab_size, max_sequence_length) -> eval input sources.
    """
    arch = "fuji"
    config_map = {}
    for version, model_size, flash_attention in itertools.product(
        Version, MODEL_SIZES, [True, False]
    ):
        if model_size not in TOTAL_TOKENS[version]:  # This combination does not exist.
            continue
        vocab_size = VOCAB_SIZE[version]
        config_name = make_config_name(
            arch=arch,
            model_size=model_size,
            version=f"v{version.value}",
            suffix="-flash" if flash_attention else "",
        )
        kwargs = get_trainer_kwargs(
            model_size, vocab_size=vocab_size, version=version, flash_attention=flash_attention
        )
        max_sequence_length = kwargs.pop("max_sequence_length")
        # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
        config_map[config_name] = get_trainer_config_fn(
            train_input_source=train_input_source(
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
            ),
            evalers=evaler_config_dict(
                eval_input_sources(vocab_size=vocab_size, max_sequence_length=max_sequence_length),
            ),
            **kwargs,
        )

        def make_fp8_config(base_config_name: str) -> SpmdTrainer.Config:
            """Make a FP8 variant of the base config.

            Not all accelerators are compatible with FP8. Support is currently
            available for NVIDIA H100, H200, and B200.

            Args:
                base_config_name: The base config name.

            Returns:
                A trainer config that uses FP8.
            """

            # pytype: disable=annotation-type-mismatch
            cfg: SpmdTrainer.Config = config_map[base_config_name]().clone()
            for accelerator, current_config in cfg.mesh_rules:
                # Only create FP8 configs for accelerators that support them
                if any(
                    supported_accelerator in accelerator
                    for supported_accelerator in [
                        "a3-highgpu-8g",
                        "a3-megagpu-8g",
                        "a3-ultragpu-8g",
                        "a4-highgpu-8g",
                        "p5.48xlarge",
                        "p4de.24xlarge",
                    ]
                ):
                    # If we already are using ChainConfigModifier, just append the FP8ConfigModifier
                    if isinstance(current_config, ChainConfigModifier.Config):
                        current_config.config_modifiers.append(
                            FP8ConfigModifier.default_config().set(fp8_amax_history_length=128)
                        )
                    else:
                        # Create a new ChainConfigModifier, preserving the mesh_shape
                        current_config = ChainConfigModifier.default_config().set(
                            config_modifiers=[
                                MeshShapeModifier.default_config().set(mesh_shape=current_config),
                                FP8ConfigModifier.default_config().set(fp8_amax_history_length=128),
                            ]
                        )
            return cfg

        # Make FP8 config, excluding the test model size
        make_fp8_config_func = functools.partial(make_fp8_config, config_name)
        if model_size != "test":
            config_map[f"{config_name}-fp8"] = make_fp8_config_func

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
        if model_size in ("1B", "3B", "7B", "8B"):

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
                # pylint: disable=cell-var-from-loop
                cfg.input.input_dispatcher.global_logical_batch_size //= (
                    128 if version in (Version.V3, Version.V3_TIKTOKEN) else 32
                )
                for evaler in cfg.evalers.values():
                    evaler.input.input_dispatcher.global_logical_batch_size //= (
                        128 if version in (Version.V3, Version.V3_TIKTOKEN) else 32
                    )
                # pylint: enable=cell-var-from-loop
                return cfg

            # Make single-host config
            make_single_host_config_func = functools.partial(make_single_host_config, config_name)
            config_map[f"{config_name}-single-host"] = make_single_host_config_func

            # Make single-host configs for FP8
            if f"{config_name}-fp8" in config_map:
                make_single_host_fp8_config_func = functools.partial(
                    make_single_host_config, f"{config_name}-fp8"
                )
                config_map[f"{config_name}-fp8-single-host"] = make_single_host_fp8_config_func

    return config_map

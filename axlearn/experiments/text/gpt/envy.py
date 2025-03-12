# Copyright © 2024 Apple Inc.

"""Utilities to set up the 'Envy' MoE style model trainer configs.

Add MoE style model configs for the GPT model class.
- SwitchTransformer <https://arxiv.org/pdf/2101.03961>.
- Apple MoE <https://arxiv.org/pdf/2405.15052>
- Mistral 8x7B <https://arxiv.org/pdf/2401.04088>

We follow most of the practice in switch-transformer for MoE, however there are some key
differences:
- We do not use a T5-style model, but a GPT-style decoder-only model.
- We do not follow the optimizer settings in the paper, instead follow practice from Fuji and Gala.
- We use the same tokenizer as Fuji model classes.
- We increase the hidden dimension per head to 128 for better use of tensorcore in both TPU and GPU.
- We increase the sequence length to 8k instead of 512 in most of the T5 models,
    and increase global tokens/batch to 8M instead of 1M.
- We use rotary positional embeddings instead of the relative positional embeddings.
- We set other hyperparameters arbitrarily when unspecified by the paper.

We follow most of the practice in Mistral 8x7B, however there are some subtle differences:
- We don't use dropless routing in Megablocks, instead we use fixed capacity with drop and padding.
- We use the same tokenizer as Fuji model classes.

Architecture names follow apple varieties: Fuji, Gala, etc.
"""

import functools
from typing import Any, Literal, Sequence, Union, NamedTuple, List
import os
import jax
from jax.ad_checkpoint import checkpoint_policies as jax_remat_policies

from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    FusedGroupedQKVLinear,
    GroupedQueryAttention,
    RoFormerQKVLinear,
    GroupedQKVLinear,
    ScaleKey,
    ScaleQuery,
    TransformerLayer,
    StackedTransformerLayer
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
from axlearn.common.mixture_of_experts import TransformerFeedForwardMoE, get_outer_batch_from_mesh, TopKGatingGather
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    GradientAccumulationModifier,
    MeshShapeModifier,
    RematSpecModifier,
    ModuleConfigModifier,
    PartitionSpecModifier
)
from axlearn.common.utils import HybridMeshShape, MeshShape, PartitionSpec, DataPartitionType
from axlearn.experiments.text.gpt.common import (
    MESH_AXIS_NAMES,
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
from axlearn.experiments.text.gpt.common import (
    mup_simple_adam_update_transformation,
    scaled_hidden_dim,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

MODEL_SIZES = ("test", "Switch-Base", "Switch-Large", "Switch-XXL", "Mistral-8x7B", "Mistral-toy", "Mistral-8x20B")

NUM_EXPERTS = {
    "test": 8,
    "Switch-Base": 128,
    "Switch-Large": 128,
    "Switch-XXL": 64,
    "Mistral-8x7B": 8,
    "Mistral-toy": 8,
    "Mistral-8x20B": 8,
}

# T5 uses 32128 vocab size, we make it 32768 for simplicity.
VOCAB_SIZE = 32 * 1024

MAX_SEQUENCE_LENGTH = {
    "test": 8192,
    "Switch-Base": 8192,
    "Switch-Large": 8192,
    "Switch-XXL": 8192,
    "Mistral-toy": 256,
    "Mistral-8x7B": 2048,
    "Mistral-8x20B": 4096,
}

_BASE_MODEL_HIDDEN_DIM = 768

MOE_OUTER_BATCH_AXIS_NAMES = ("data", "fsdp")

MOE_DIM_TO_MESH_AXIS_MAP = {
    "me": PartitionSpec(None, None),
    "emh": PartitionSpec("expert", "fsdp", "model"),
    "ehm": PartitionSpec("expert", "model", "fsdp"),
    "ogsm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, "model"),
    "ogsM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
    "ogse": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
    "ogec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
    # Dispatch and combine tensors.
    "ogsec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, "expert", None),
    "oegcm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, "model"),
    "oegcM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, None),
    "ogecm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, "expert", None, "model"),
    "ogecM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, "expert", None, None),
    "oegch": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, "model"),
}

def get_ffn_layer_types():
    ffn_type = os.getenv("AXLEARN_MIXTRAL_MOE", "1")
    if ffn_type == "0":
        ffn_layer_types = ["dense"]
    elif ffn_type == "1":
        ffn_layer_types = ["sparse"]
    elif ffn_type == "2":
        ffn_layer_types = ["sparse", "dense"]
    return ffn_layer_types

def common_trainer_kwargs() -> dict[str, Any]:
    """Returns kwargs that are common to all configs."""
    return {
        "model_kwargs": {
            "z_loss_scale": 1e-6,
        },
        "learner_kwargs": {
            "peak_lr": 1e-2,
            "alpha": 1 / 200.0,
            "weight_decay": 3.16e-4,
        },
        "save_every_n_steps": 5000,
        "keep_every_n_steps": 5000,
        "eval_every_n_steps": 25_000,
        "mesh_shape": mesh_shape_from_axes(data=-1),
    }

class _Trn2CustomConfig(NamedTuple):
    """Config modifications required to run Fuji models on TRN2."""

    # Module config modifications.
    module_modifications: List[ModuleConfigModifier.Config]
    # Partition spec modifications.
    partition_spec_modifications: List[PartitionSpecModifier.Config]


def _generate_trn2_custom_configs(
    model_size: str,
) -> _Trn2CustomConfig:
    """Generate custom module config and PartitionSpec modification for TRN2.

    Args:
        model_size: Size of the Envy model.

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
                "model.decoder.output_norm": {
                    "input_partition_spec": (("data", "fsdp"), "model", None),
                    "output_partition_spec": (("data", "fsdp"), None, None),
                },

            },
        ),
    ]

    ffn_layer_types = get_ffn_layer_types()
    if len(ffn_layer_types) == 1:
        target_config="model.decoder.transformer.layer.self_attention.attention.input_linear.input_linear"
        mcm = ModuleConfigModifier.default_config().set(
            target_config=target_config,
            modification=GroupedQKVLinear.default_config(),
        )
        trn2_module_modifications.append(mcm)

        trn2_partition_spec_modifications.append(
            PartitionSpecModifier.default_config().set(
                partition_specs={
                    # Sequence parallel shardings for norms.
                    "model.decoder.transformer.layer.self_attention.norm": {
                        "input_partition_spec": (("data", "fsdp"), "model", None),
                        "output_partition_spec": (("data", "fsdp"), None, None),
                    },
                    "model.decoder.transformer.layer.feed_forward.norm": {
                        "input_partition_spec": (("data", "fsdp"), "model", None),
                        "output_partition_spec": (("data", "fsdp"), None, None),
                    },
                },
            )
        )

        if ffn_layer_types[0] == "dense":
            trn2_partition_spec_modifications[-1].partition_specs[f"model.decoder.transformer.layer.feed_forward.linear2"] = {
                "output_partition_spec": (("data", "fsdp"), None, None),
            }
    elif len(ffn_layer_types) == 2:
        for i in range(2):
            target_config=f"model.decoder.transformer.layer.layer[{i}].self_attention.attention.input_linear.input_linear"
            mcm = ModuleConfigModifier.default_config().set(
                target_config=target_config,
                modification=GroupedQKVLinear.default_config(),
            )
            trn2_module_modifications.append(mcm)
            trn2_partition_spec_modifications.append(
                PartitionSpecModifier.default_config().set( 
                    partition_specs={
                        # Sequence parallel shardings for norms.
                        f"model.decoder.transformer.layer.layer[{i}].self_attention.norm": {
                            "input_partition_spec": (("data", "fsdp"), "model", None),
                            "output_partition_spec": (("data", "fsdp"), None, None),
                        },
                        f"model.decoder.transformer.layer.layer[{i}].feed_forward.norm": {
                            "input_partition_spec": (("data", "fsdp"), "model", None),
                            "output_partition_spec": (("data", "fsdp"), None, None),
                        },
                    },
                )
            )

            if ffn_layer_types[i] == "dense":
                trn2_partition_spec_modifications[-1].partition_specs[f"model.decoder.transformer.layer.layer[{i}].feed_forward.linear2"] = {
                    "output_partition_spec": (("data", "fsdp"), None, None),
                }
    

    # trn2_lm_head_partition_spec = [
    #     PartitionSpecModifier.default_config().set(
    #         partition_specs={
    #             # Vocab parallel embeddings sharding from Megatron LM.
    #             "model.decoder.lm_head": {
    #                 "param_partition_spec": (
    #                     "model",
    #                     ("expert", "fsdp", "seq"),
    #                 ),
    #             },
    #         },
    #     ),
    # ]
    # trn2_partition_spec_modifications += trn2_lm_head_partition_spec

    return _Trn2CustomConfig(
        module_modifications=trn2_module_modifications,
        partition_spec_modifications=trn2_partition_spec_modifications,
    )



def get_trainer_kwargs(
    model_size: str,
    *,
    vocab_size: int,
    max_sequence_length: int,
    flash_attention: bool,
) -> dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
    tokens_per_batch = 8 * (1024**2)  # 8M tokens.
    trn2_config = _generate_trn2_custom_configs(model_size)
    # pylint: disable=use-dict-literal
    if model_size == "test":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=4,
                hidden_dim=8,
                ffn_dim=scaled_hidden_dim(scale=8/3, round_up_to_multiples_of=16),
                num_heads=4,
                num_kv_heads=2,
                vocab_size=32,
                num_experts=8,
                train_capacity_factor=2.0,
                num_groups=2,
                ffn_layer_types=[
                    "dense",
                    "sparse",
                ],
            ),
            learner_kwargs=dict(),
            max_sequence_length=64,
            train_batch_size=16,
            max_step=3000,
            mesh_shape=mesh_shape_from_axes(data=-1),
        )
    elif model_size == "Switch-Base":
        # Num of parameters: 30B.
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=12,
                hidden_dim=12 * 128,
                ffn_dim=scaled_hidden_dim(scale=4, round_up_to_multiples_of=128),
                num_heads=12,
                num_kv_heads=12,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=2,
                ffn_structure="hybridnorm",
                # MoE layer every 2 layers.
                ffn_layer_types=[
                    "dense",
                    "sparse",
                ],
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=tokens_per_batch // max_sequence_length,  # 8M tokens.
            max_step=250_000,
            mesh_shape=mesh_shape_from_axes(fsdp=-1, expert=16),
            mesh_rules=(
                (
                    "tpu-v5p-(1024|2048|4096)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=True,
                                        policy=jax_remat_policies.dots_saveable,
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v6e-256",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "Switch-Large":
        # Num of parameters: 104B.
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=24,
                hidden_dim=16 * 128,
                ffn_dim=scaled_hidden_dim(scale=4, round_up_to_multiples_of=128),
                num_heads=16,
                num_kv_heads=16,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=2,
                ffn_structure="hybridnorm",
                # MoE layer every 2 layers.
                ffn_layer_types=[
                    "dense",
                    "sparse",
                ],
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=tokens_per_batch // max_sequence_length,  # 8M tokens.
            max_step=250_000,  # Most of the evals were done at 100k steps in the paper.
            mesh_shape=mesh_shape_from_axes(fsdp=-1, expert=16),
            mesh_rules=(
                (
                    "tpu-v5p-(2048|4096)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v6e-256-4",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                        ],
                    ),
                ),
                (
                    "tpu-v6e-256",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, expert=16, fsdp=16)
                            ),
                            GradientAccumulationModifier.default_config().set(grad_acc_steps=4),
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "Switch-XXL":
        # Num of parameters: 520B.
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=24,
                hidden_dim=64 * 128,
                ffn_dim=scaled_hidden_dim(scale=2.5, round_up_to_multiples_of=128),
                num_heads=64,
                num_kv_heads=8,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=2,
                ffn_structure="hybridnorm",
                # MoE layer every 2 layers.
                ffn_layer_types=[
                    "dense",
                    "sparse",
                ],
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=tokens_per_batch // max_sequence_length,  # 8M tokens.
            max_step=250_000,  # Most of the evals were done at 100k steps in the paper.
            # TODO(kelvin-zou): not verified with real job.
            mesh_shape=mesh_shape_from_axes(fsdp=-1, expert=16, model=8),
        )
    elif model_size in ["Mistral-8x7B", "Mistral-toy"]:
        # Num of parameters: 47B.
        ffn_layer_types = get_ffn_layer_types()
        tp_degree=int(os.getenv("AXLEARN_TP_DEGREE", 4))
        neuron_mesh = mesh_shape_from_axes(fsdp=-1, model=tp_degree)
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=int(os.getenv("AXLEARN_NUM_LAYERS", 4)),
                hidden_dim=32 * 32 if model_size == "Mistral-toy" else 32 * 128,
                ffn_dim=scaled_hidden_dim(scale=3.5, round_up_to_multiples_of=128),
                num_heads=32,
                num_kv_heads=max(8, tp_degree),
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=1,
                ffn_layer_types=ffn_layer_types,
                outer_batch_size=get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, neuron_mesh),
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=int(os.getenv("AXLEARN_MAX_SEQ_LEN", max_sequence_length)),
            train_batch_size=int(os.getenv("AXLEARN_TRAIN_BATCH_SIZE", 128)),
            max_step=250_000,
            mesh_shape=mesh_shape_from_axes(fsdp=-1, model=8),
            mesh_rules=(
                (
                    "tpu-v5p-4096",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(data=-1, fsdp=16, model=8)
                            ),
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=True,
                                        policy=jax_remat_policies.nothing_saveable,
                                    ),
                                }
                            ),
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
                                mesh_shape=neuron_mesh
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                            RematSpecModifier.default_config().set(
                                remat_policies={
                                    "model.decoder.transformer.layer": RematSpec(
                                        prevent_cse=True,
                                        policy=jax_remat_policies.nothing_saveable,
                                    ),
                                } if os.getenv('AXLEARN_REMAT_LAYER', 'true') == 'true' else {}
                            ),
                        ],
                    ),
                ),
            ),
        )
    elif model_size == "Mistral-8x20B":
        # Num of parameters: 150B.
        ffn_layer_types = get_ffn_layer_types()
        neuron_mesh = mesh_shape_from_axes(fsdp=-1, model=16)
        trainer_kwargs = dict(
            model_kwargs=dict(
                # 40 layers gets to about 150B
                num_layers=int(os.getenv("NUM_LAYERS", 4)),
                hidden_dim=32*192,
                ffn_dim=scaled_hidden_dim(scale=3.5, round_up_to_multiples_of=128),
                num_heads=32,
                num_kv_heads=16,
                num_experts=NUM_EXPERTS[model_size],
                train_capacity_factor=2.0,
                num_groups=1,
                ffn_layer_types=ffn_layer_types,
                outer_batch_size=get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, neuron_mesh),
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=128, # 8M tokens.
            max_step=250_000,
            mesh_shape=mesh_shape_from_axes(fsdp=-1, model=8),
            mesh_rules=(
                (
                    "neuron-(trn2|trn2n).48xlarge-64",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                # TP within the chip, FSDP across chips.
                                # Each TRN2 chip has 4 XLA cores.
                                mesh_shape=neuron_mesh
                            ),
                            *trn2_config.module_modifications,
                            *trn2_config.partition_spec_modifications,
                            # RematSpecModifier.default_config().set(
                            #     remat_policies={
                            #         "model.decoder.transformer.layer": RematSpec(
                            #             prevent_cse=True,
                            #             policy=jax_remat_policies.nothing_saveable,
                            #         ),
                            #     }
                            # ),
                        ],
                    ),
                ),
            ),
        )
    # pylint: enable=use-dict-literal
    else:
        raise NotImplementedError(f"Unknown model size {model_size}.")

    merged_trainer_kwargs = common_trainer_kwargs()
    merged_trainer_kwargs.update(
        {k: v for k, v in trainer_kwargs.items() if k not in ("model_kwargs", "learner_kwargs")}
    )

    # Update the model_kwargs
    model_kwargs: dict[str, Any] = merged_trainer_kwargs.pop("model_kwargs")
    model_kwargs.update(trainer_kwargs.get("model_kwargs", {}))
    model_kwargs.setdefault("vocab_size", vocab_size)

    learner_kwargs: dict[str, Any] = merged_trainer_kwargs.pop("learner_kwargs")
    learner_kwargs.update(trainer_kwargs.get("learner_kwargs", {}))

    mesh_shape = merged_trainer_kwargs.get("mesh_shape", mesh_shape_from_axes(data=-1))
    merged_trainer_kwargs["model_cfg"] = model_config(
        flash_attention=flash_attention, mesh_shape=mesh_shape, **model_kwargs
    )
    # If a model is smaller than the base model, do not scale.
    linear_layer_lr_multiplier = min(_BASE_MODEL_HIDDEN_DIM / model_kwargs["hidden_dim"], 1.0)
    merged_trainer_kwargs["learner_cfg"] = adamw_decoupled_learner_config(
        max_step=trainer_kwargs["max_step"],
        # Enable mup-simple.
        adam_update_transformation=mup_simple_adam_update_transformation(
            linear_layer_lr_multiplier,
        ),
        **learner_kwargs,
    )

    return merged_trainer_kwargs


def model_config(
    *,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    num_experts: int,
    vocab_size: int,
    train_capacity_factor: float,
    num_groups: int,
    ffn_layer_types: Sequence[Literal["dense", "sparse"]],
    ffn_dim: Union[int, config.FunctionConfigBase],
    dropout_rate: float = 0.0,
    flash_attention: bool = False,
    outer_batch_size: int = None,
    mesh_shape: Union[MeshShape, HybridMeshShape],
    **kwargs,
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        num_layers: The number of Transformer Layers.
        hidden_dim: The Transformer layer input/output dim.
        num_heads: The number of attention heads.
        num_kv_heads: The number of attention KV heads.
        num_experts: The number of experts in the MoE layer.
        vocab_size: The vocabulary size.
        train_capacity_factor: The train capacity factor for the MoE layer.
        ffn_layer_types: The types of layer in the feed-forward network, Options: [dense, sparse].
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        ffn_dim: The feed-forward dimension or config function.
            If None, defaults to a setting from https://arxiv.org/abs/2002.05202.
        flash_attention: If True, use flash attention implementation.
        mesh_shape: the mesh shape, used to infer the outer batch size.

    Returns:
        A causal LM config.
    """
    # Use RoPE by default.
    # RoPE <https://arxiv.org/abs/2104.09864> for positional encodings.
    # `CausalAttentionLogitBiasLayer` is already applied in the attention impl.
    attention_mask = None
    # RoPE embeddings: https://arxiv.org/abs/2104.09864.
    attention_qkv_linear = RoFormerQKVLinear.default_config().set(
        input_linear=FusedGroupedQKVLinear.default_config().set(
            cache_dtype=STEP_DTYPE,
            num_kv_heads=num_kv_heads,
        ),
        rotary_value=False,
        cache_dtype=STEP_DTYPE,
    )
    attention_qkv_linear.rope_pos_emb_layer.theta = 5e5
    norm_cfg = RMSNorm.default_config().set(eps=1e-5, forward_dtype=None)

    transformer_layer_cfg = TransformerLayer.default_config()
    if flash_attention:
        transformer_layer_cfg.self_attention.attention = flash_attention_config()
    else:
        transformer_layer_cfg.self_attention.attention = GroupedQueryAttention.default_config()
    transformer_layer_cfg.self_attention.attention.set(
        # Use q/k-norm in keeping with:
        # <https://arxiv.org/abs/2309.14322>
        query_scale=ScaleQuery.default_config().set(norm=norm_cfg.clone()),
        key_scale=ScaleKey.default_config().set(norm=norm_cfg.clone()),
    )
    if outer_batch_size is None:
        outer_batch_size = get_outer_batch_from_mesh(
            MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, mesh_shape
        )
    expert_config = TransformerFeedForwardMoE.default_config().set(
        outer_batch=outer_batch_size,
        num_experts=num_experts,
        input_dim=hidden_dim,
        num_groups=num_groups,
        dim_to_mesh_axis_map=MOE_DIM_TO_MESH_AXIS_MAP,
        gating=TopKGatingGather.default_config(),
    )
    expert_config.gating.top_k = 2
    expert_config.gating.train_capacity_factor = train_capacity_factor

    emb_cfg: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config().set(
        pos_emb=None
    )
    # emb_cfg.token_emb.param_partition_spec = (("expert", "fsdp", "seq"), "model")
    cfg = common_model_config(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        # SwiGLU from https://arxiv.org/abs/2002.05202.
        activation_fn=("nn.silu", "linear"),
        ffn_dim=ffn_dim,
        normalization=norm_cfg,
        dropout_rate=dropout_rate,
        emb_cfg=emb_cfg,
        # Since we pass `layer_cfg`, this is already set.
        attention_cfg=None,
        attention_mask=attention_mask,
        attention_qkv_linear=attention_qkv_linear,
        layer_cfg=transformer_layer_cfg,
        ffn_layer_types=ffn_layer_types,
        expert_cfg=expert_config,
        **kwargs,
    )
    # if flash_attention:
    #     cfg.decoder.transformer.layer.remat_spec = RematSpec(
    #         prevent_cse=False, policy=jax_remat_policies.dots_saveable
    #     )
    return cfg


def trainer_configs(
    train_input_source: SourceBuilder,
    eval_input_sources: SourceBuilder,
) -> dict[str, TrainerConfigFn]:
    """Returns a mapping from config_name to TrainerConfigFn's.

    Args:
        train_input_source: A callable (vocab_size, max_sequence_length) -> input source config.
        eval_input_soruces: A callable (vocab_size, max_sequence_length) -> eval input sources.
    """
    arch = "envy"
    config_map = {}
    vocab_size = VOCAB_SIZE
    for model_size in MODEL_SIZES:
        seq_len = MAX_SEQUENCE_LENGTH[model_size]
        config_name = make_config_name(arch=arch, model_size=model_size)
        kwargs = get_trainer_kwargs(
            model_size,
            vocab_size=vocab_size,
            # Use default flash attention.
            flash_attention=True,
            max_sequence_length=seq_len,
        )

        # Test models sometimes override it to a very small length.
        seq_len = kwargs.pop("max_sequence_length", seq_len)

        # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
        config_map[config_name] = get_trainer_config_fn(
            train_input_source=train_input_source(
                vocab_size=vocab_size, max_sequence_length=seq_len
            ),
            evalers=evaler_config_dict(
                eval_input_sources(vocab_size=vocab_size, max_sequence_length=seq_len),
            ),
            **kwargs,
        )
        # Only Switch-Base model size is runnable on a single node mode.
        if model_size == "Switch-Base":

            def make_single_host_config(base_config_name: str) -> SpmdTrainer.Config:
                """Make a single-host variant of the base config."""

                # pytype: disable=annotation-type-mismatch
                cfg: SpmdTrainer.Config = config_map[base_config_name]().clone()
                # pytype: enable=annotation-type-mismatch
                cfg.input.batcher.feed_batch_size = 8
                for evaler in cfg.evalers.values():
                    evaler.input.batcher.feed_batch_size = 8
                remat_modifier = (
                    RematSpecModifier.default_config()
                    .set(
                        remat_policies={
                            "model.decoder.transformer.layer": RematSpec(
                                prevent_cse=True,
                                policy=jax_remat_policies.nothing_saveable,
                            ),
                        }
                    )
                    .instantiate()
                )
                cfg = remat_modifier(cfg)
                return cfg

            # Make single-host config
            make_single_host_config_func = functools.partial(make_single_host_config, config_name)
            config_map[f"{config_name}-single-host"] = make_single_host_config_func

    return config_map

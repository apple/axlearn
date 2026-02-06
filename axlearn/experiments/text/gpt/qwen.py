# Copyright © 2026 Apple Inc.

"""
Utilities to set up the Qwen-3 style model trainer configs: https://arxiv.org/abs/2505.09388.
"""

from typing import Any, Literal, Optional, Sequence, Union

import jax.numpy as jnp

from axlearn.common import causal_lm, config, decoder
from axlearn.common.attention import (
    FusedGroupedQKVLinear,
    RematRegexSavePatterns,
    RoFormerQKVLinear,
    ScaleKey,
    ScaleQuery,
    TransformerLayer,
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.config import TrainerConfigFn, config_for_function
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.flash_attention.layer import FlashAttention
from axlearn.common.flash_attention.remat import save_or_offload_flash_attention_policy
from axlearn.common.layers import RMSNorm
from axlearn.common.mixture_of_experts import GMMBackend, TopKDropFreeGating
from axlearn.common.mixture_of_experts import TransformerFeedForwardDropFreeMoE as MoE
from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    MeshShapeModifier,
    RematSpecModifier,
    ReplaceLayerConfigModifier,
)
from axlearn.common.utils import (
    HybridMeshShape,
    PartitionSpec,
    save_and_offload_only_these_names_regex,
)
from axlearn.experiments.text.gpt.common import (
    SourceBuilder,
    adamw_decoupled_learner_config,
    evaler_config_dict,
    get_trainer_config_fn,
    make_config_name,
    mesh_shape_from_axes,
)
from axlearn.experiments.text.gpt.common import model_config as common_model_config
from axlearn.experiments.text.gpt.common import (
    mup_simple_adam_update_transformation,
    scaled_hidden_dim,
)

CONFIGS = {
    "30B-A3B": [
        (16 * 1024 * 1024, 32 * 1024),  # 16M tokens, 32k seq len
        (8 * 1024 * 1024, 8 * 1024),  # 8M tokens, 8k seq len
    ],
}

_BASE_MODEL_HIDDEN_DIM = 768


QWEN3_VOCAB_SIZE = 151936
QWEN3_BOS_TOKEN_ID = 151643
QWEN3_EOS_TOKEN_ID = 151645
QWEN3_PAD_TOKEN_ID = 151645
QWEN3_RMS_NORM_EPS = 1e-6
QWEN3_ROPE_THETA = 1_000_000.0
QWEN3_HEAD_DIM = 128


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


def get_trainer_kwargs(
    model_size: str,
    *,
    vocab_size: int,
    batch_size: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
    # pylint: disable=use-dict-literal
    if model_size == "30B-A3B":
        # https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=48,
                hidden_dim=16 * 128,
                ffn_dim=scaled_hidden_dim(scale=3, round_up_to_multiples_of=128),
                moe_ffn_dim=768,
                num_heads=32,
                atten_hidden_dim=32 * QWEN3_HEAD_DIM,
                num_kv_heads=4,
                num_experts=128,
                train_capacity_factor=0,
                num_groups=1,
                ffn_structure="prenorm",
                ffn_layer_types=[
                    "sparse",
                ],
                num_experts_per_token=8,
                tie_word_embeddings=False,
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            max_sequence_length=max_sequence_length,
            train_batch_size=batch_size // max_sequence_length,
            max_step=250_000,
            mesh_shape=mesh_shape_from_axes(fsdp=8, data=-1),
            mesh_rules=(
                (
                    "tpu-v5p-(1024|2048)",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=mesh_shape_from_axes(fsdp=8, data=-1)
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
                                                    RematRegexSavePatterns.FLASH_CONTEXT.value,
                                                ]
                                            ),
                                            names_which_can_be_offloaded="|".join(
                                                [
                                                    RematRegexSavePatterns.QKV_PROJ.value,
                                                    RematRegexSavePatterns.LINEAR1_X.value,
                                                    RematRegexSavePatterns.MOE_GATING.value,
                                                ]
                                            ),
                                            offload_src="device",
                                            offload_dst="pinned_host",
                                        ),
                                    ),
                                }
                            ),
                        ],
                    ),
                ),
                # B200/H100/A100 80G
                (
                    "gpu-(p6-b200.48xlarge|p5.48xlarge|p4de.24xlarge|a3-highgpu-8g)-512",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            MeshShapeModifier.default_config().set(
                                mesh_shape=HybridMeshShape(
                                    ici_mesh_shape=mesh_shape_from_axes(fsdp=8),
                                    dcn_mesh_shape=mesh_shape_from_axes(data=-1),
                                ),
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
                                                    RematRegexSavePatterns.FLASH_CONTEXT.value,
                                                    RematRegexSavePatterns.QKV_PROJ.value,
                                                    RematRegexSavePatterns.O_PROJ.value,
                                                    RematRegexSavePatterns.LINEAR1_X.value,
                                                    RematRegexSavePatterns.LINEAR2_X.value,
                                                    RematRegexSavePatterns.MOE_GATING.value,
                                                ]
                                            ),
                                            names_which_can_be_offloaded="|".join([]),
                                            offload_src="device",
                                            offload_dst="pinned_host",
                                        ),
                                    ),
                                }
                            ),
                            ReplaceLayerConfigModifier.default_config().set(
                                target_cls=MoE,
                                source_config=MoE.default_config().set(
                                    gmm_backend=GMMBackend.TOKAMAX,
                                    tokamax_implementation=["triton", "xla"],
                                ),
                                exclude_keys=["gmm_backend", "tokamax_implementation"],
                            ),
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
    model_kwargs: dict[str, Any] = merged_trainer_kwargs.pop(
        "model_kwargs"
    )  # pytype: disable=annotation-type-mismatch
    model_kwargs.update(trainer_kwargs.get("model_kwargs", {}))
    model_kwargs.setdefault("vocab_size", vocab_size)

    learner_kwargs: dict[str, Any] = merged_trainer_kwargs.pop(
        "learner_kwargs"
    )  # pytype: disable=annotation-type-mismatch
    learner_kwargs.update(trainer_kwargs.get("learner_kwargs", {}))

    merged_trainer_kwargs["model_cfg"] = model_config(**model_kwargs)
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
    moe_ffn_dim: Optional[Union[int, config.FunctionConfigBase]] = None,
    atten_hidden_dim: Optional[int] = None,
    num_experts_per_token: int = 2,
    dropout_rate: float = 0.0,
    tie_word_embeddings: bool = False,
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
        moe_ffn_dim: The feed-forward dimension for the MoE layers. If None, uses `ffn_dim`.
        atten_hidden_dim: The attention hidden dimension. If None, defaults to hidden_dim.
        num_experts_per_token: Number of experts to route each token to.
        tie_word_embeddings: If True, tie the input and output word embeddings.
        kwargs: Default kwargs forwarded to `common_model_config`.

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
            num_kv_heads=num_kv_heads,
        ),
        rotary_value=False,
    )
    attention_qkv_linear.rope_pos_emb_layer.theta = QWEN3_ROPE_THETA
    norm_cfg = RMSNorm.default_config().set(eps=1e-6, forward_dtype=jnp.float32)

    transformer_layer_cfg = TransformerLayer.default_config()
    transformer_layer_cfg.self_attention.attention = FlashAttention.default_config().set(
        causal=True,
        mha_dim_to_partition_spec={
            "btnh": PartitionSpec(("data", "expert", "fsdp"), None, ("seq", "model"), None),
            "bsnh": PartitionSpec(("data", "expert", "fsdp"), None, ("seq", "model"), None),
            "bnts": PartitionSpec(("data", "expert", "fsdp"), None, None, None),
        },
        output_dim_to_partition_spec={
            "btnh": PartitionSpec(("data", "expert", "fsdp"), "seq", "model", None),
            "bnts": PartitionSpec(("data", "expert", "fsdp"), "model", "seq", None),
        },
    )
    transformer_layer_cfg.self_attention.attention.set(
        # Use q/k-norm in keeping with:
        # <https://arxiv.org/abs/2309.14322>
        query_scale=ScaleQuery.default_config().set(norm=norm_cfg.clone()),
        key_scale=ScaleKey.default_config().set(norm=norm_cfg.clone()),
        hidden_dim=atten_hidden_dim,
        o_partition_spec=(("data", "expert", "fsdp"), "seq", "model"),
        tpu_block_size=2048,
        backend_overrides=dict(
            splash_use_fused_bwd_kernel=True,
        ),
    )
    expert_config = MoE.default_config().set(
        num_experts=num_experts,
        input_dim=hidden_dim,
        num_groups=num_groups,
        dim_to_mesh_axis_map={
            "me": PartitionSpec(None, None),
            "emh": PartitionSpec("expert", ("fsdp", "seq"), "model"),
            "ehm": PartitionSpec("expert", "model", ("fsdp", "seq")),
        },
        input_dim_to_partition_spec={
            "bsm": PartitionSpec(("data", "expert", "fsdp"), "seq", None),
        },
        output_dim_to_partition_spec={
            "bsm": PartitionSpec(("data", "expert", "fsdp"), "seq", "model"),
            "emh": PartitionSpec("expert", None, "model"),
            "ehm": PartitionSpec("expert", "model", None),
        },
        gating=TopKDropFreeGating.default_config().set(
            num_experts_per_token=num_experts_per_token,
            train_capacity_factor=train_capacity_factor,
        ),
        load_balance_loss_weight=1e-3,
        tiling=(128, 512, 512),
    )

    emb_cfg: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config().set(
        pos_emb=None
    )
    emb_cfg.token_emb.param_partition_spec = ("model", ("expert", "fsdp", "seq"))
    emb_cfg.token_emb.output_partition_spec = (("data", "expert", "fsdp"), "seq", None)
    cfg = common_model_config(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        # SwiGLU from https://arxiv.org/abs/2002.05202.
        activation_fn=("nn.silu", "linear"),
        ffn_dim=ffn_dim,
        moe_ffn_dim=moe_ffn_dim,
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
        lm_head_cfg=(
            None
            if tie_word_embeddings
            else decoder.LmHead.default_config().set(
                # (vocab, model) partitioning.
                param_partition_spec=("model", ("expert", "fsdp", "seq")),
            )
        ),
        **kwargs,
    )
    cfg.decoder.eos_token_id = QWEN3_EOS_TOKEN_ID
    cfg.decoder.pad_token_id = QWEN3_PAD_TOKEN_ID
    cfg.batch_axis_names = ("data", "expert", "fsdp")
    cfg.decoder.transformer.layer.remat_spec = RematSpec(
        prevent_cse=False,
        policy=save_or_offload_flash_attention_policy(),
    )
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
    arch = "qwen3"
    config_map = {}
    vocab_size = QWEN3_VOCAB_SIZE
    for model_size, configs in CONFIGS.items():
        base_name = make_config_name(arch=arch, model_size=model_size)
        for batch_size, seq_len in configs:
            kwargs = get_trainer_kwargs(
                model_size,
                vocab_size=vocab_size,
                batch_size=batch_size,
                max_sequence_length=seq_len,
            )
            seq_len = kwargs.pop("max_sequence_length", seq_len)
            config_name = f"{base_name}-bs{batch_size // 1024 // 1024}m-seq{seq_len // 1024}k"
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

    return config_map

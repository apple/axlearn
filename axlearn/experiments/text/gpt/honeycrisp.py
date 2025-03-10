# Copyright © 2024 Apple Inc.

"""Utilities to set up the 'Honeycrisp' GPT model trainer configs.

- QK Norm (in keeping with <https://arxiv.org/abs/2309.14322>).
- mup-simple (<https://arxiv.org/abs/2309.14322> Section 3.2.4.).

Architecture names follow apple varieties: Fuji, Gala, Honeycrisp, etc.
"""

import itertools
from typing import Any, Optional, Union

from jax.ad_checkpoint import checkpoint_policies as jax_remat_policies

from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    FusedGroupedQKVLinear,
    GroupedQueryAttention,
    KVCache,
    RoFormerQKVLinear,
    ScaleKey,
    ScaleQuery,
    TransformerLayer,
)
from axlearn.common.base_layer import RematSpec
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
from axlearn.experiments.text.gpt.common import (
    STEP_DTYPE,
    SourceBuilder,
    adastar_learner_config,
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

MODEL_SIZES = ("test", "85M", "3B")

VOCAB_SIZE = 48 * 1024

MAX_SEQUENCE_LENGTH = 4096


# 85M is the base model for mup-simple.
_BASE_MODEL_HIDDEN_DIM = 768


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
    max_sequence_length: int,
    flash_attention: bool,
) -> dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
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
            ),
            learner_kwargs=dict(),
            max_sequence_length=64,
            train_batch_size=16,
            max_step=3000,
            mesh_shape=mesh_shape_from_axes(),  # cpu
        )
    elif model_size == "85M":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=12,
                hidden_dim=_BASE_MODEL_HIDDEN_DIM,
                ffn_dim=scaled_hidden_dim(scale=3.0, round_up_to_multiples_of=32),
                num_heads=12,
                num_kv_heads=6,
            ),
            learner_kwargs=dict(lr_warmup_steps=5_000),
            train_batch_size=1 * 1024 * 1024 // max_sequence_length,  # 1M tokens.
            max_step=600_000,  # 600B tokens // 1M tokens/step.
            save_every_n_steps=1000,
            mesh_shape=mesh_shape_from_axes(data=-1),
            mesh_rules=(
                ("tpu-v4-(32|64)", mesh_shape_from_axes(data=-1)),
                # Step time: 0.67s on tpu-v5litepod-32.
                ("tpu-v5litepod-32", mesh_shape_from_axes(data=-1)),
                ("tpu-v5p-(16|32)", mesh_shape_from_axes(data=-1)),
                # Step time: 0.92s on H100 8x2.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge|a3-highgpu-8g)-(16|32|64)",
                    mesh_shape_from_axes(data=-1),
                ),
                ("gpu-p4d.24xlarge-(16|32|64|128)", mesh_shape_from_axes(data=-1, fsdp=8)),
            ),
        )
    elif model_size == "3B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=26,
                hidden_dim=128 * 24,
                ffn_dim=8064,
                num_heads=24,
                num_kv_heads=8,
            ),
            learner_kwargs=dict(lr_warmup_steps=5_000),
            train_batch_size=8 * 1024 * 1024 // max_sequence_length,  # 8M tokens.
            max_step=750_000,  # 6T tokens // 8M tokens/step.
            mesh_shape=mesh_shape_from_axes(data=-1, fsdp=8),
            mesh_rules=(
                ("tpu-v4-256", mesh_shape_from_axes(fsdp=-1)),
                # tpu-v5e. step time: TBD.
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1, fsdp=256)),
                # H100/A100 80G.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge|a3-highgpu-8g)-(64|128|256|512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=8),
                ),
                # A100 40G.
                ("gpu-p4d.24xlarge-256", mesh_shape_from_axes(data=-1, fsdp=8)),
                # tpu-v5p. step time: 1.48s on v5p-1024, 0.8s on v5p-2048, 0.47s on v5p-4096.
                ("tpu-v5p-(1024|2048|4096)", mesh_shape_from_axes(data=-1)),
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

    merged_trainer_kwargs["model_cfg"] = model_config(
        flash_attention=flash_attention, **model_kwargs
    )
    # If a model is smaller than the base model, do not scale.
    linear_layer_lr_multiplier = min(_BASE_MODEL_HIDDEN_DIM / model_kwargs["hidden_dim"], 1.0)
    merged_trainer_kwargs["learner_cfg"] = adastar_learner_config(
        max_step=merged_trainer_kwargs["max_step"],
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
    vocab_size: int,
    dropout_rate: float = 0.0,
    ffn_dim: Optional[Union[int, config.FunctionConfigBase]] = None,
    flash_attention: bool = False,
    **kwargs,
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        num_layers: The number of Transformer Layers.
        hidden_dim: The Transformer layer input/output dim.
        num_heads: The number of attention heads.
        num_kv_heads: The number of attention KV heads.
        vocab_size: The vocabulary size.
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        ffn_dim: The feed-forward dimension or config function.
            If None, defaults to a setting from https://arxiv.org/abs/2002.05202.
        flash_attention: If True, use flash attention implementation.

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
    attention_qkv_linear.rope_pos_emb_layer.theta = 5e5
    attention_kv_cache = KVCache.default_config().set(cache_dtype=STEP_DTYPE)
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
    transformer_layer_cfg.feed_forward.set(
        add_dead_neuron_summary=True,
        add_value_rms_norm_summary=["linear2_outputs"],
    )

    emb_cfg = TransformerTextEmbeddings.default_config().set(pos_emb=None)
    emb_cfg.token_emb.param_partition_spec = (("expert", "fsdp", "seq"), "model")

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
        attention_kv_cache=attention_kv_cache,
        layer_cfg=transformer_layer_cfg,
        **kwargs,
    )
    if flash_attention:
        cfg.decoder.transformer.layer.remat_spec = RematSpec(
            prevent_cse=False, policy=jax_remat_policies.dots_saveable
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
    arch = "honeycrisp"
    config_map = {}

    vocab_size = VOCAB_SIZE

    # For each model size create a flash/noflash version.
    for model_size, enable_flash in itertools.product(MODEL_SIZES, {True, False}):
        seq_len = MAX_SEQUENCE_LENGTH
        suffix = "flash" if enable_flash else ""
        config_name = make_config_name(arch=arch, model_size=model_size, version=suffix)
        kwargs = get_trainer_kwargs(
            model_size,
            vocab_size=vocab_size,
            flash_attention=enable_flash,
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
    return config_map

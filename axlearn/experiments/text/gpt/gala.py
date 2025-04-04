# Copyright © 2024 Apple Inc.

"""Utilities to set up the 'Gala' GPT model trainer configs.

- QK Norm (in keeping with <https://arxiv.org/abs/2309.14322>).
- mup-simple (<https://arxiv.org/abs/2309.14322> Section 3.2.4.).

Architecture names follow apple varieties: Fuji, Gala, Honeycrisp, etc.

The Gala models are set up for baselines for various papers:
- Sigmoid Attention. See `gala_sigmoid.py`.

"""
import itertools
from typing import Any, Literal, Optional, TypeAlias, Union, cast

from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    ALiBiAttentionLogitBiasLayer,
    FusedQKVLinear,
    KVCache,
    RoFormerQKVLinear,
    ScaleKey,
    ScaleQuery,
    TransformerLayer,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
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
from axlearn.experiments.text.gpt.common import (
    mup_simple_adam_update_transformation,
    scaled_hidden_dim,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

MODEL_SIZES = ("test", "85M", "302M", "1B", "7B")

VOCAB_SIZE = 32 * 1024

MAX_SEQUENCE_LENGTH = 4096

SupportedNormStructure: TypeAlias = Literal["prenorm", "hybridnorm", "postnorm"]
SupportedPositionEncoding: TypeAlias = Literal["rope", "alibi"]

MODEL_SIZE_CONFIG_ARGS_BASE = {
    "norm_structure": "prenorm",
    "position_encoding": "rope",
}

# Additional configs that should be generated, on top of the base configs.
# Base configs use rope+prenorm (see `MODEL_SIZE_CONFIG_ARGS_BASE`).
MODEL_SIZE_CONFIG_ARGS_ADDITIONAL = {
    "1B": [
        {
            "norm_structure": "hybridnorm",
            "position_encoding": "alibi",
            "flash_only": True,
        }
    ],
    "7B": [
        {
            "norm_structure": "hybridnorm",
            "position_encoding": "alibi",
        }
    ],
}


# 85M is the base model for mup-simple.
_BASE_MODEL_HIDDEN_DIM = 768


def get_trainer_kwargs(
    model_size: str,
    *,
    vocab_size: int,
    max_sequence_length: int,
    flash_attention: bool = False,
    norm_structure: SupportedNormStructure = "prenorm",
    position_encoding: SupportedPositionEncoding = "rope",
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
                vocab_size=32,
            ),
            learner_kwargs=dict(
                peak_lr=6e-4,
                weight_decay=0.01,
            ),
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
                num_heads=12,
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            train_batch_size=1 * 1024 * 1024 // max_sequence_length,  # 1M tokens.
            max_step=400_000,  # 400B tokens // 1M tokens/step.
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # tpu-v5e. step time: 0.18s (data=-1, fsdp=8).
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1)),
            ),
        )
    elif model_size == "302M":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=24,
                hidden_dim=64 * 16,
                num_heads=16,
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            train_batch_size=1 * 1024 * 1024 // max_sequence_length,  # 1M tokens.
            max_step=400_000,  # 400B tokens // 1M tokens/step.
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # tpu-v5e. step time: TBD.
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1)),
            ),
        )
    elif model_size == "1B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=24,
                hidden_dim=64 * 32,
                num_heads=32,
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            train_batch_size=1 * 1024 * 1024 // max_sequence_length,  # 1M tokens.
            max_step=300_000,  # 300B tokens // 1M tokens/step.
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # tpu-v5e. step time: 0.87s (data=-1, fsdp=8).
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1, fsdp=8)),
            ),
        )
    elif model_size == "7B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=32,
                hidden_dim=128 * 32,
                num_heads=32,
            ),
            learner_kwargs=dict(peak_lr=0.01, weight_decay=1e-4, lr_warmup_steps=5_000),
            train_batch_size=4 * 1024 * 1024 // max_sequence_length,  # 4M tokens.
            max_step=250_000,  # 1T tokens // 4M tokens/step.
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # tpu-v4. step time: 3.03s.
                ("tpu-v4-(1024|2048)", mesh_shape_from_axes(data=-1, fsdp=16)),
                ("tpu-v5p-(1024|2048)", mesh_shape_from_axes(data=-1, fsdp=16)),
                # tpu-v5e. step time: TBD.
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1, fsdp=32)),
                # H100/A100 80G. Maximum per-node batch size = 64, hence need >= 32 nodes.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge)-(256|512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=8),
                ),
            ),
        )
    # pylint: enable=use-dict-literal
    else:
        raise NotImplementedError(f"Unknown model size {model_size}.")

    model_kwargs: dict[str, Any] = trainer_kwargs.pop("model_kwargs")
    model_kwargs.setdefault("vocab_size", vocab_size)

    # If a model is smaller than the base model, do not scale.
    linear_layer_lr_multiplier = min(_BASE_MODEL_HIDDEN_DIM / model_kwargs["hidden_dim"], 1.0)
    trainer_kwargs["model_cfg"] = model_config(
        flash_attention=flash_attention,
        norm_structure=norm_structure,
        position_encoding=position_encoding,
        **model_kwargs,
    )
    trainer_kwargs["learner_cfg"] = adamw_decoupled_learner_config(
        max_step=trainer_kwargs["max_step"],
        # Enable mup-simple.
        adam_update_transformation=mup_simple_adam_update_transformation(
            linear_layer_lr_multiplier,
        ),
        **trainer_kwargs.pop("learner_kwargs"),
    )

    return trainer_kwargs


def model_config(
    *,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    dropout_rate: float = 0.0,
    ffn_dim: Optional[Union[int, config.FunctionConfigBase]] = None,
    flash_attention: bool = False,
    norm_structure: SupportedNormStructure = "prenorm",
    position_encoding: SupportedPositionEncoding = "rope",
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        num_layers: The number of Transformer Layers.
        hidden_dim: The Transformer layer input/output dim.
        num_heads: The number of attention heads.
        vocab_size: The vocabulary size.
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        ffn_dim: The feed-forward dimension or config function.
            If None, defaults to a setting from https://arxiv.org/abs/2002.05202.
        flash_attention: If True, use flash attention implementation.
        norm_structure: Norm structure to use for ffn and attention.
            Options: See `SupportedNormStructure`.
        position_encoding: Position encoding to use.
            Options: See `SupportedPositionEncoding`.

    Returns:
        A causal LM config.
    """

    transformer_layer_cfg = TransformerLayer.default_config()

    if position_encoding == "rope":
        # RoPE <https://arxiv.org/abs/2104.09864> for positional encodings.
        # `CausalAttentionLogitBiasLayer` is already applied in the attention impl.
        attention_mask = None
        # RoPE embeddings: https://arxiv.org/abs/2104.09864.
        attention_qkv_linear = RoFormerQKVLinear.default_config().set(
            input_linear=FusedQKVLinear.default_config(),
            rotary_value=False,
        )
    elif position_encoding == "alibi":
        attention_mask = ALiBiAttentionLogitBiasLayer.default_config().set(
            num_heads=num_heads,
        )
        attention_qkv_linear = FusedQKVLinear.default_config()
    else:
        raise ValueError(f"'{position_encoding}' not supported!")
    attention_kv_cache = KVCache.default_config().set(cache_dtype=STEP_DTYPE)

    if flash_attention:
        transformer_layer_cfg.self_attention.attention = flash_attention_config()

    # SwiGLU from https://arxiv.org/abs/2002.05202.
    activation_fn = ("nn.silu", "linear")

    if ffn_dim is None:
        ffn_dim = scaled_hidden_dim(scale=8 / 3, round_up_to_multiples_of=256)

    norm_cfg = RMSNorm.default_config().set(eps=1e-5, forward_dtype=None)
    emb_cfg = TransformerTextEmbeddings.default_config().set(
        pos_emb=None,
    )

    transformer_layer_cfg.self_attention.attention.set(
        # Use q/k-norm in keeping with:
        # <https://arxiv.org/abs/2309.14322>
        query_scale=ScaleQuery.default_config().set(norm=norm_cfg.clone()),
        key_scale=ScaleKey.default_config().set(norm=norm_cfg.clone()),
    )

    cfg = common_model_config(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        activation_fn=activation_fn,
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
        ffn_structure=norm_structure,
        atten_structure=norm_structure,
    )
    return cfg


def config_name_suffix(
    *,
    norm_structure: SupportedNormStructure,
    position_encoding: SupportedPositionEncoding,
    enable_flash: bool,
) -> str:
    suffix = []
    if norm_structure != "prenorm":
        suffix.append(norm_structure)
    if position_encoding != "rope":
        suffix.append(position_encoding)
    if enable_flash:
        suffix.append("flash")
    return "-".join(suffix)


def trainer_configs(
    train_input_source: SourceBuilder,
    eval_input_sources: SourceBuilder,
) -> dict[str, TrainerConfigFn]:
    """Returns a mapping from config_name to TrainerConfigFn's.

    Args:
        train_input_source: A callable (vocab_size, max_sequence_length) -> input source config.
        eval_input_soruces: A callable (vocab_size, max_sequence_length) -> eval input sources.
    """
    arch = "gala"
    config_map = {}

    vocab_size = VOCAB_SIZE

    # For each model size, norm structure and position encoding create a flash/noflash version.
    for (
        model_size,
        enable_flash,
    ) in itertools.product(
        MODEL_SIZES,
        {True, False},
    ):
        seq_len = MAX_SEQUENCE_LENGTH
        for config_args in [
            MODEL_SIZE_CONFIG_ARGS_BASE,
            *MODEL_SIZE_CONFIG_ARGS_ADDITIONAL.get(model_size, []),
        ]:
            if config_args.get("flash_only") and not enable_flash:
                continue
            norm_structure = cast(SupportedNormStructure, config_args["norm_structure"])
            position_encoding = cast(SupportedPositionEncoding, config_args["position_encoding"])
            config_name = make_config_name(
                arch=arch,
                model_size=model_size,
                version=config_name_suffix(
                    norm_structure=norm_structure,
                    position_encoding=position_encoding,
                    enable_flash=enable_flash,
                ),
            )
            kwargs = get_trainer_kwargs(
                model_size,
                vocab_size=vocab_size,
                flash_attention=enable_flash,
                max_sequence_length=seq_len,
                norm_structure=norm_structure,
                position_encoding=position_encoding,
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

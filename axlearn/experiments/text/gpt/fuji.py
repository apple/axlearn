# Copyright © 2023 Apple Inc.

"""Utilities to set up the 'fuji' GPT model trainer configs.

Architecture names follow Apple varieties: fuji, gala, honeycrisp, etc.

The fuji models are set up to imitate LLaMA-1 (https://arxiv.org/abs/2302.13971).
"""
import enum
from typing import Any, Dict, Optional, Union

from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    CausalAttentionLogitBiasLayer,
    FusedGroupedQKVLinear,
    FusedQKVLinear,
    RepeatedTransformerLayer,
    RoFormerQKVLinear,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
from axlearn.experiments.text.gpt.common import STEP_DTYPE, learner_config, mesh_shape_from_axes
from axlearn.experiments.text.gpt.common import model_config as common_model_config
from axlearn.experiments.text.gpt.common import scaled_hidden_dim

MODEL_SIZES = ("test", "7B")


class Version(enum.Enum):
    V1 = 1
    V2 = 2
    V3 = 3


# Mapping from Fuji versions to vocab sizes.
VOCAB_SIZE = {
    Version.V1: 32 * 1024,
    Version.V2: 32 * 1024,
    Version.V3: 128 * 1024,
}


# Mapping from Fuji versions to maximum sequence lengths.
MAX_SEQUENCE_LENGTH = {
    Version.V1: 2048,
    Version.V2: 4096,
    Version.V3: 8192,
}


ROPE_THETA = {
    Version.V1: 1e4,
    Version.V2: 1e4,
    Version.V3: 5e5,
}


# Mapping from Fuji versions to total number of tokens used in training.
TOTAL_TOKENS = {
    Version.V1: 1 * (1024**4),  # 1T tokens
    Version.V2: 2 * (1024**4),  # 2T tokens
    Version.V3: 15 * (1024**4),  # 15T tokens
}


def get_trainer_kwargs(model_size: str, *, vocab_size: int, version: Version) -> Dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
    tokens_per_batch = 4 * (1024**2)  # 4M tokens.
    max_step = TOTAL_TOKENS[version] // tokens_per_batch
    max_sequence_length = MAX_SEQUENCE_LENGTH[version]
    train_batch_size = tokens_per_batch // max_sequence_length

    # Whether to use grouped query attention.
    num_kv_heads = None
    if version == Version.V3:
        num_kv_heads = 8

    rope_theta = ROPE_THETA[version]

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
    elif model_size == "7B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=32,
                hidden_dim=128 * 32,
                num_heads=32,
                num_kv_heads=num_kv_heads,
                rope_theta=rope_theta,
            ),
            learner_kwargs=dict(peak_lr=3e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # tpu-v4. step time: 3.03s.
                ("tpu-v4-(1024|2048)", mesh_shape_from_axes(data=-1, fsdp=16)),
                # tpu-v5e. step time: TBD.
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1, fsdp=16)),
                # H100/A100 80G. Maximum per-node batch size = 64, hence need >= 32 nodes.
                # p5.48xlarge 8x64. v1 step time: 1.54s.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge)-(256|512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=8),
                ),
            ),
        )
    else:
        raise NotImplementedError(f"Unknown model size {model_size}.")
    model_kwargs = trainer_kwargs.pop("model_kwargs")
    model_kwargs.setdefault("vocab_size", vocab_size)
    trainer_kwargs["model_cfg"] = model_config(**model_kwargs)
    trainer_kwargs["learner_cfg"] = learner_config(
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
    dropout_rate: float = 0.0,
    ffn_dim: Optional[Union[int, config.FunctionConfigBase]] = None,
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        num_layers: The number of Transformer Layers.
        hidden_dim: The Transformer layer input/output dim.
        num_heads: The number of attention heads.
        num_kv_heads: The optional number of KV heads. If not None, enables grouped query attention.
        vocab_size: The vocabulary size.
        rope_theta: The theta value used for RoPE positional embeddings.
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        ffn_dim: The feed-forward dimension or config function.
            If None, defaults to a setting from https://arxiv.org/abs/2002.05202.

    Returns:
        A causal LM config.
    """
    # SwiGLU from https://arxiv.org/abs/2002.05202.
    activation_fn = ("nn.silu", "linear")
    if ffn_dim is None:
        ffn_dim = scaled_hidden_dim(scale=8 / 3, round_up_to_multiples_of=256)
    if num_kv_heads:
        atten_input_linear = FusedGroupedQKVLinear.default_config().set(num_kv_heads=num_kv_heads)
    else:
        atten_input_linear = FusedQKVLinear.default_config()
    atten_input_linear.cache_dtype = STEP_DTYPE
    # RoPE embeddings: https://arxiv.org/abs/2104.09864.
    atten_qkv_linear = RoFormerQKVLinear.default_config().set(
        cache_dtype=STEP_DTYPE,
        input_linear=atten_input_linear,
        rotary_value=False,
    )
    atten_qkv_linear.rope_pos_emb_layer.theta = rope_theta

    cfg = common_model_config(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        stack_cfg=RepeatedTransformerLayer.default_config(),
        activation_fn=activation_fn,
        ffn_dim=ffn_dim,
        normalization=RMSNorm.default_config().set(eps=1e-5, forward_dtype=None),
        dropout_rate=dropout_rate,
        emb_cfg=TransformerTextEmbeddings.default_config().set(pos_emb=None),
        attention_mask=CausalAttentionLogitBiasLayer.default_config(),
        attention_qkv_linear=atten_qkv_linear,
    )
    return cfg

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
from typing import Any, Dict, Optional, Union

from axlearn.common import causal_lm, config
from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    FusedGroupedQKVLinear,
    FusedQKVLinear,
    GroupedQueryAttention,
    MultiheadAttention,
    RepeatedTransformerLayer,
    RoFormerQKVLinear,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments.text.gpt.common import (
    STEP_DTYPE,
    SourceBuilder,
    evaler_config_dict,
    flash_attention_config,
    get_trainer_config_fn,
    learner_config,
    make_config_name,
    mesh_shape_from_axes,
)
from axlearn.experiments.text.gpt.common import model_config as common_model_config
from axlearn.experiments.text.gpt.common import scaled_hidden_dim
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

MODEL_SIZES = ("test", "7B", "70B")


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
    Version.V1: {
        "test": 1 * (1024**4),  # 1T tokens
        "7B": 1 * (1024**4),  # 1T tokens
        "70B": 1.4 * (1024**4),  # 1.4T tokens
    },
    Version.V2: {
        "test": 2 * (1024**4),  # 2T tokens
        "7B": 2 * (1024**4),  # 2T tokens
        "70B": 2 * (1024**4),  # 2T tokens
    },
    Version.V3: {
        "test": 15 * (1024**4),  # 15T tokens
        "7B": 15 * (1024**4),  # 15T tokens
        "70B": 15 * (1024**4),  # 15T tokens
    },
}


def get_trainer_kwargs(
    model_size: str,
    *,
    vocab_size: int,
    version: Version,
    flash_attention: bool = False,
) -> Dict[str, Any]:
    """Construct default trainer kwargs given a model size."""
    tokens_per_batch = 4 * (1024**2)  # 4M tokens.
    max_step = TOTAL_TOKENS[version][model_size] // tokens_per_batch
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
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(
                peak_lr=6e-4,
                weight_decay=0.01,
            ),
            max_sequence_length=64,
            train_batch_size=32,
            eval_batch_size=32,
            max_step=3000,
            eval_every_n_steps=1500,
            save_every_n_steps=500,
            mesh_shape=mesh_shape_from_axes(data=-1),
        )
    elif model_size == "7B":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=32,
                hidden_dim=128 * 32,
                num_heads=32,
                num_kv_heads=num_kv_heads,
                rope_theta=rope_theta,
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
                ("tpu-v4-.*", mesh_shape_from_axes(data=-1, fsdp=16)),
                # tpu-v5e.
                ("tpu-v5litepod-.*", mesh_shape_from_axes(data=-1, fsdp=16)),
                # tpu-v5p.
                ("tpu-v5p-.*", mesh_shape_from_axes(data=-1, fsdp=8)),
                # H100/A100 80G.
                # Maximum per-node batch size = 64, hence need >= 32 nodes.
                # Without sequence sharding, the maximum number of devices <= batch_size,
                # so at most 512 GPUs (64 nodes) for training 7B-v3.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge|a3-highgpu-8g)-(256|512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=8),
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
                rope_theta=rope_theta,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=1.5e-4, weight_decay=0.1),
            max_sequence_length=max_sequence_length,
            train_batch_size=train_batch_size,
            max_step=max_step,
            mesh_shape=mesh_shape_from_axes(fsdp=-1),
            mesh_rules=(
                # tpu-v5e. step time: TBD.
                ("tpu-v5litepod-256", mesh_shape_from_axes(data=-1, fsdp=256)),
                # H100/A100 80G. Maximum per-node batch size = 16, hence need >= 64 nodes.
                # v2 on gpu-p5.48xlarge 8x64, step time: 12.9s.
                (
                    "gpu-(p5.48xlarge|p4de.24xlarge)-(512|1024)",
                    mesh_shape_from_axes(data=-1, fsdp=128),
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
    flash_attention: bool = False,
    stack_cfg: Optional[BaseStackedTransformerLayer.Config] = None,
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
        flash_attention: Whether to enable flash attention.
        stack_cfg: The transformer stack config.
            If None, defaults to a RepeatedTransformerLayer.

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
        stack_cfg=stack_cfg if stack_cfg is not None else RepeatedTransformerLayer.default_config(),
        activation_fn=activation_fn,
        ffn_dim=ffn_dim,
        normalization=RMSNorm.default_config().set(eps=1e-5, forward_dtype=None),
        dropout_rate=dropout_rate,
        emb_cfg=TransformerTextEmbeddings.default_config().set(pos_emb=None),
        attention_cfg=flash_attention_config() if flash_attention else atten_cfg,
        attention_qkv_linear=atten_qkv_linear,
    )
    return cfg


def trainer_configs(
    train_input_source: SourceBuilder, eval_input_sources: SourceBuilder
) -> Dict[str, TrainerConfigFn]:
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
        if model_size == "7B":

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
                cfg.input.batcher.global_batch_size //= 32
                for evaler in cfg.evalers.values():
                    evaler.input.batcher.global_batch_size //= 32
                return cfg

            config_map[f"{config_name}-single-host"] = functools.partial(
                make_single_host_config, config_name
            )
    return config_map

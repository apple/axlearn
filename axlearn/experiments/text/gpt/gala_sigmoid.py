# Copyright © 2024 Apple Inc.

"""Sigmoid attention version of the Gala architecture.

- Sigmoid-based Attention
- ALiBi instead of RoPE.

"""

from typing import Optional

from axlearn.common.attention import ALiBiAttentionLogitBiasLayer, FusedQKVLinear, SigmoidAttention
from axlearn.common.config import ConfigBase, InstantiableConfig
from axlearn.common.flash_attention.layer import FlashAttention
from axlearn.common.layers import set_norm_recursively
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import own_fields
from axlearn.experiments.text.gpt.common import update_model_remat_config
from axlearn.experiments.text.gpt.gala import SupportedNormStructure, SupportedPositionEncoding


def _set_seq_len_recursively(cfg: ConfigBase, max_sequence_length: int) -> ConfigBase:
    """Sets config.max_sequence_length for all relevant descendant configs in `cfg`.

    This is used to update input configs with a new sequence length.

    Args:
        cfg: The root config under which to find configs with max_sequence_length.
        max_sequence_length: The target max_sequence_length.

    Returns:
        A config with new maximum sequence length.
    """

    def visit_fn(_, value):
        if isinstance(value, ConfigBase) and "max_sequence_length" in value:
            value.max_sequence_length = max_sequence_length

    def enter_fn(_, value, default_kv):
        return (
            None if isinstance(value, ConfigBase) and "max_sequence_length" in value else default_kv
        )

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
    return cfg


def build_sigmoid_trainer_config(
    base_cfg: SpmdTrainer.Config,
    *,
    max_sequence_length: int,
    norm_structure: SupportedNormStructure = "prenorm",
    position_encoding: SupportedPositionEncoding = "alibi",
    attn_bias: bool = True,
    norm_cfg: Optional[ConfigBase] = None,
    flash_attention: bool = False,  # pylint: disable=unused-argument
) -> SpmdTrainer.Config:
    """Builds a sigmoid-based trainer config, based on an existing trainer config.

    Used to convert Gala trainer config to a Gala with sigmoid trainer config.

    Args:
        base_cfg: Config to modify to the sigmoid setting.
        max_sequence_length: Maximum seq. length to be used during training and evaluation.
        norm_structure: Norm structure to use for ffn and attention layers.
            Options: See `SupportedNormStructure`.
        position_encoding: Position encoding to use.
            Options: See `SupportedPositionEncoding`.
        attn_bias: If True, use the sequence length bias.
        norm_cfg: If not None, override the norm in the config. Can be used to
            e.g. set it to `RMSNormStateless` for a stateless version.
        flash_attention: If True, use flash attention implementation.

    Returns:
        A TrainerConfigFn that creates an updated config.
    """

    # TODO(floris_weers): when Sigmoid flash attention is checked in, use `flash_attention`.
    def config_fn() -> InstantiableConfig:
        sigmoid_cfg = base_cfg().clone()
        transformer_layer_cfg = sigmoid_cfg.model.decoder.transformer.layer

        if position_encoding == "alibi":
            # ALiBi <https://arxiv.org/abs/2108.12409> for positional encodings.
            # See RefA for ablations.
            attention_mask = ALiBiAttentionLogitBiasLayer.default_config().set(
                num_heads=transformer_layer_cfg.self_attention.attention.num_heads
            )
            input_linear = transformer_layer_cfg.self_attention.attention.input_linear
            if hasattr(input_linear, "input_linear"):
                # In case ALiBi is used in the source config.
                input_linear = input_linear.input_linear
            attention_qkv_linear = FusedQKVLinear.default_config().set(**dict(input_linear.items()))
        # Create sigmoid attention config, but use original attention child configs.
        sigmoid_attention_cfg_kwargs = dict(transformer_layer_cfg.self_attention.attention.items())
        if sigmoid_attention_cfg_kwargs["klass"] is FlashAttention:
            # Remove flash-specific kwargs, which are not supported.
            to_remove = own_fields(FlashAttention.default_config())
            for key in to_remove:
                del sigmoid_attention_cfg_kwargs[key]
        del sigmoid_attention_cfg_kwargs["klass"]
        sigmoid_attention_cfg = (
            SigmoidAttention.default_config()
            .set(**sigmoid_attention_cfg_kwargs)
            .set(seq_len=max_sequence_length, subtract_seq_len_bias=attn_bias)
        )
        if position_encoding == "alibi":
            sigmoid_attention_cfg = sigmoid_attention_cfg.set(input_linear=attention_qkv_linear)
            sigmoid_cfg.model.decoder.set(attention_mask=attention_mask)

        # Set attention layer to Sigmoid.
        transformer_layer_cfg.self_attention.set(attention=sigmoid_attention_cfg)

        # Update norm structure.
        transformer_layer_cfg.self_attention.set(structure=norm_structure)
        transformer_layer_cfg.feed_forward.set(structure=norm_structure)

        if norm_cfg:
            set_norm_recursively(sigmoid_cfg, norm_cfg)

        update_model_remat_config(
            stack_cfg=sigmoid_cfg.model.decoder.transformer,
            layer_cfg=transformer_layer_cfg,
        )
        _set_seq_len_recursively(sigmoid_cfg, max_sequence_length=max_sequence_length)

        return sigmoid_cfg

    return config_fn

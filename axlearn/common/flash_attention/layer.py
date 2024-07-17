# Copyright © 2023 Apple Inc.

"""FlashAttention layers."""
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.maps import thread_resources
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec

from axlearn.common.attention import GroupedQueryAttention
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import ConfigBase, config_class
from axlearn.common.flash_attention.utils import (
    MultiHeadAttentionImpl,
    flash_attention_implementation,
)
from axlearn.common.module import Module
from axlearn.common.utils import Tensor, with_sharding_constraint


def _check_bias_recursively(cfg: ConfigBase):
    """Ensures that `cfg.bias` is set to False for all descendants."""

    def visit_fn(_, value):
        if isinstance(value, BaseLayer.Config) and getattr(value, "bias", False):
            raise NotImplementedError("cfg.bias is not yet supported.")

    def enter_fn(_, value, default_kv):
        return None if isinstance(value, BaseLayer.Config) and "bias" in value else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
    return cfg


class FlashAttention(GroupedQueryAttention):
    """FlashAttention layer.

    Is a drop-in replacement of GroupedQueryAttention
        (which itself supports MultiheadAttention as a special case), with some limitations:
            * Does not support dropout.
            * Does not support gradients wrt. attention logit biases.
            * Supports a subset of config fields and outputs.
    """

    @config_class
    class Config(GroupedQueryAttention.Config):
        """Configures FlashAttention."""

        # If True, applies additional optimizations in the FlashAttention kernels.
        # Causal attention can still be used when False, by passing logit biases.
        causal: bool = False
        # The block size used to tile attention computation (for TPU only).
        # Should be less than the target sequence length and a multiple of 128 on TPU.
        # TODO(tom_gunter): Expose GPU block-size (currently always 128) & unify.
        tpu_block_size: int = 512

        # SPMD partition specs:
        # B - batch dim,
        # T - target sequence length,
        # S - source sequence length,
        # N - number of attention (query) heads,
        # H - per-head dimension.
        # How to partition flash attention computation, keyed by dims.
        mha_dim_to_partition_spec: Dict[str, Optional[PartitionSpec]] = {}
        # How to partition output values, keyed by dims.
        output_dim_to_partition_spec: Dict[str, Optional[PartitionSpec]] = {}

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        _check_bias_recursively(cfg)  # Bias not supported.
        if getattr(cfg, "atten_logit_cap", None) is not None:
            raise NotImplementedError("cfg.atten_logit_cap is not supported.")
        if cfg.dropout.rate:
            raise NotImplementedError("cfg.dropout.rate is not supported.")
        if cfg.tpu_block_size % 128 != 0:
            raise ValueError("cfg.tpu_block_size must divide 128.")

    @classmethod
    def default_config(cls) -> Config:
        cfg: FlashAttention.Config = super().default_config()
        cfg.dropout.rate = None
        cfg.atten_logit_cap = None
        cfg.mha_dim_to_partition_spec = {
            "btnh": PartitionSpec(None),
            "bsnh": PartitionSpec(None),
            "bnts": PartitionSpec(None),
        }
        cfg.output_dim_to_partition_spec = {
            "btnh": PartitionSpec(None),
            "bnts": PartitionSpec(None),
        }
        return cfg

    def _causal_mask(self, seq_len: int) -> Optional[Tensor]:
        return None  # No need for mask because flash attention supports the causal mode natively.

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        cfg = self.config

        # Repeats key/value heads dim if necessary.
        k_proj = self._repeat_kv_heads(k_proj)
        v_proj = self._repeat_kv_heads(v_proj)

        if jax.default_backend() == "tpu":
            assert (
                q_proj.shape[1] % cfg.tpu_block_size == 0
            ), "Target seq len must divide block size."
            assert (
                k_proj.shape[1] % cfg.tpu_block_size == 0
            ), "Source seq len must divide block size."

        if attention_logit_biases is not None:
            if attention_logit_biases.ndim != 4:
                raise ValueError(
                    f"Expected attention_logit_biases.ndim == 4, got {attention_logit_biases.ndim}"
                )
            attention_logit_biases = attention_logit_biases.astype(q_proj.dtype)

        # During GPU decoding, fall back to plain MHA implementation
        # since the seq_len will not be divisible by block size.
        # For prefilling, seq_len can be > 1 and logit biases may not always be provided,
        # so we retain `cfg.causal`.
        # For decoding, seq_len = 1 and logit biases are always provided,
        # so we set "causal" flag to False.
        causal = cfg.causal
        if q_proj.shape[1] % 128 != 0:
            backend = "xla"
            # TODO(senyut): Implement FlashDecoding kernel and support TPU decoding.
            if q_proj.shape[1] == 1:
                causal = False
        else:
            backend = jax.default_backend()

        jit_attn: MultiHeadAttentionImpl = flash_attention_implementation(
            backend=backend,
            causal=causal,
            softmax_scale=1.0,
            block_size=cfg.tpu_block_size,
        )

        # We need to manually partition pallas | jax-triton calls.
        # Note: shard_map doesn't support kwargs.
        partitioned_mha = shard_map(
            jit_attn,
            mesh=thread_resources.env.physical_mesh,
            in_specs=(
                # QKV [batch_size, seq_len, num_heads, per_head_dim].
                cfg.mha_dim_to_partition_spec["btnh"],
                cfg.mha_dim_to_partition_spec["bsnh"],
                cfg.mha_dim_to_partition_spec["bsnh"],
                # Bias [batch_size, num_heads, seq_len, seq_len].
                cfg.mha_dim_to_partition_spec["bnts"],
            ),
            # O [batch_size, seq_len, num_heads, per_head_dim].
            out_specs=cfg.mha_dim_to_partition_spec["btnh"],
            # Disables a checking pass which jax can't apply when there's a triton | pallas
            # call in the body.
            check_rep=False,
        )

        # Scale query and key.
        q_proj = self.scale_query(q_proj)
        k_proj = self.scale_key(k_proj)

        # Constrain input to conform to partitioned MHA expectations.
        q_proj = with_sharding_constraint(q_proj, cfg.mha_dim_to_partition_spec["btnh"])
        k_proj = with_sharding_constraint(k_proj, cfg.mha_dim_to_partition_spec["bsnh"])
        v_proj = with_sharding_constraint(v_proj, cfg.mha_dim_to_partition_spec["bsnh"])
        if attention_logit_biases is not None:
            if attention_logit_biases.shape[0] != q_proj.shape[0]:
                raise ValueError(
                    "attention_logit_biases must have matching batch dim: "
                    f"{attention_logit_biases.shape} vs. {q_proj.shape[0]}"
                )
            attention_logit_biases = with_sharding_constraint(
                attention_logit_biases, cfg.mha_dim_to_partition_spec["bnts"]
            )

        outputs = with_sharding_constraint(
            partitioned_mha(
                q_proj,
                k_proj,
                v_proj,
                attention_logit_biases,
            ),
            cfg.output_dim_to_partition_spec["btnh"],
        )

        # TODO(markblee): Add output probs and benchmark.
        batch, target_len, num_heads, _ = q_proj.shape
        _, source_len, _, _ = k_proj.shape
        output_probs = with_sharding_constraint(
            jnp.empty((batch, num_heads, target_len, source_len)),
            cfg.output_dim_to_partition_spec["bnts"],
        )
        return outputs, output_probs


def default_mha_dim_to_partition_spec(
    mesh_axis_names: Sequence[str],
) -> Dict[str, Optional[PartitionSpec]]:
    """Builds a default FlashAttention mapping from tensor dims to partition specs for the MHA impl.

    Maps attention heads over the default tensor-parallel axis name if present, and
    shards the batch over the remainder of the axes.

    Args:
        mesh_axis_names: Mesh axis names.

    Returns:
        A dictionary keyed by MHA tensor dims with partition spec values.
    """
    batch_axis_names = tuple(el for el in mesh_axis_names if el != "model")
    tp_axis_name = "model" if "model" in mesh_axis_names else None
    return {
        "btnh": PartitionSpec(batch_axis_names, None, tp_axis_name, None),
        "bsnh": PartitionSpec(batch_axis_names, None, tp_axis_name, None),
        "bnts": PartitionSpec(batch_axis_names, tp_axis_name, None, None),
    }


def default_output_dim_to_partition_spec(
    mesh_axis_names: Sequence[str],
) -> Dict[str, Optional[PartitionSpec]]:
    """Builds a default mapping from tensor dims to partition specs for the FlashAttention outputs.

    Maps attention heads over the default tensor-parallel axis name if present,
    shards the target sequence length over the default sequence-parallel axis name if present,
    and shards the batch over the remainder of the axes.

    Args:
        mesh_axis_names: Mesh axis names.

    Returns:
        A dictionary keyed by FlashAttention output tensor dims with partition spec values.
    """
    batch_axis_names = tuple(el for el in mesh_axis_names if el not in ["seq", "model"])
    tp_axis_name = "model" if "model" in mesh_axis_names else None
    sp_axis_name = "seq" if "seq" in mesh_axis_names else None
    return {
        "btnh": PartitionSpec(batch_axis_names, sp_axis_name, tp_axis_name, None),
        "bnts": PartitionSpec(batch_axis_names, tp_axis_name, sp_axis_name, None),
    }

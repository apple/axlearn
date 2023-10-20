# Copyright © 2023 Apple Inc.

"""FlashAttention layers."""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.maps import thread_resources
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec

from axlearn.common.attention import MultiheadAttention
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import ConfigBase, config_class
from axlearn.common.flash_attention.utils import (
    MultiHeadAttentionImpl,
    flash_attention_implementation,
)
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


def _check_bias_recursively(cfg: ConfigBase):
    """Ensures that `cfg.bias` is set to False for all descendants."""

    def visit_fn(_, value):
        if isinstance(value, BaseLayer.Config) and getattr(value, "bias", False):
            raise NotImplementedError("cfg.bias is not yet supported.")

    def enter_fn(_, value, default_kv):
        return None if isinstance(value, BaseLayer.Config) and "bias" in value else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
    return cfg


class FlashAttention(MultiheadAttention):
    """FlashAttention layer.

    Is a drop-in replacement of MultiheadAttention, with some limitations:
        * Does not support dropout.
        * Does not support gradients wrt. attention logit biases.
        * Supports a subset of config fields and outputs.
    """

    @config_class
    class Config(MultiheadAttention.Config):
        # If True, applies additional optimizations in the FlashAttention kernels.
        # Causal attention can still be used when False, by passing logit biases.
        causal: bool = False
        # The block size used to tile attention computation (for TPU only).
        # Should be less than the target sequence length and a multiple of 128 on TPU.
        # TODO(tom_gunter): Expose GPU block-size (currently always 128) & unify.
        tpu_block_size: int = 512

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        _check_bias_recursively(cfg)  # Bias not supported.
        for key in ["per_dim_scale", "atten_logit_cap"]:
            if getattr(cfg, key, None) is not None:
                raise NotImplementedError(f"cfg.{key} is not supported.")
        if cfg.dropout.rate:
            raise NotImplementedError("cfg.dropout.rate is not supported.")
        if cfg.tpu_block_size % 128 != 0:
            raise ValueError("cfg.tpu_block_size must divide 128.")

    @classmethod
    def default_config(cls) -> Config:
        cfg: FlashAttention.Config = super().default_config()
        cfg.dropout.rate = None
        cfg.per_dim_scale = None
        cfg.atten_logit_cap = None
        return cfg

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        attention_logit_biases: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        cfg = self.config

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

        jit_attn: MultiHeadAttentionImpl = flash_attention_implementation(
            backend=jax.default_backend(),
            causal=cfg.causal,
            softmax_scale=self._scale_query(1),
            block_size=cfg.tpu_block_size,
        )

        mesh = thread_resources.env.physical_mesh
        # We need to manually partition jax-triton calls.
        # Note: shard_map doesn't support kwargs.
        # We assume that the batch is partitioned over all but the last mesh axis name.
        batch_axis_names = mesh.axis_names[:-1]
        # We also assume that the last axis is for tensor-parallelism.
        tensor_parallel_axis_name = mesh.axis_names[-1]
        # TODO(tom_gunter,markblee): Better validation of axis names.
        if tensor_parallel_axis_name != "model":
            raise NotImplementedError("Running without tensor-parallel axis is not supported.")
        partitioned_mha = shard_map(
            jit_attn,
            mesh=mesh,
            in_specs=(
                # QKV [batch_size, seq_len, num_heads, per_head_dim].
                PartitionSpec(batch_axis_names, None, tensor_parallel_axis_name, None),
                PartitionSpec(batch_axis_names, None, tensor_parallel_axis_name, None),
                PartitionSpec(batch_axis_names, None, tensor_parallel_axis_name, None),
                # Bias [batch_size, num_heads, seq_len, seq_len].
                PartitionSpec(batch_axis_names, tensor_parallel_axis_name, None, None),
            ),
            # O [batch_size, seq_len, num_heads, per_head_dim].
            out_specs=PartitionSpec(batch_axis_names, None, tensor_parallel_axis_name, None),
            # Disables a checking pass which jax can't apply when there's a triton | pallas
            # call in the body.
            check_rep=False,
        )

        outputs = partitioned_mha(
            q_proj,
            k_proj,
            v_proj,
            attention_logit_biases,
        )
        batch, target_len, num_heads, _ = q_proj.shape
        _, source_len, _, _ = k_proj.shape
        # TODO(markblee): Add output probs and benchmark.
        return outputs, jnp.empty((batch, num_heads, target_len, source_len))

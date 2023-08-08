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
from axlearn.common.flash_attention.attention import mha
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
    * Does not yet support dropout.
    * Does not support gradients wrt attention logit biases.
    * Supports a subset of config fields and outputs.
    """

    @config_class
    class Config(MultiheadAttention.Config):
        # If True, applies additional optimizations in the FlashAttention kernels.
        # Causal attention can still be used when False, by passing logit biases.
        causal: bool = False

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        _check_bias_recursively(cfg)  # Bias not supported.
        for key in ["per_dim_scale", "atten_logit_cap"]:
            if getattr(cfg, key, None) is not None:
                raise NotImplementedError(f"cfg.{key} is not supported.")
        if cfg.dropout.rate:
            raise NotImplementedError("cfg.dropout.rate is not supported.")

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

        if attention_logit_biases is not None:
            if attention_logit_biases.ndim != 4:
                raise ValueError(
                    f"Expected attention_logit_biases.ndim == 4, got {attention_logit_biases.ndim}"
                )
            attention_logit_biases = attention_logit_biases.astype(q_proj.dtype)

        # shard_map-decorated function needs to be jitted.
        @jax.jit
        def jit_mha(query, key, value, bias):
            return mha(
                query, key, value, bias=bias, causal=cfg.causal, softmax_scale=self._scale_query(1)
            )

        mesh = thread_resources.env.physical_mesh
        # We need to manually partition jax-triton calls.
        # Note: shard_map doesn't support kwargs.
        partitioned_mha = shard_map(
            jit_mha,
            mesh=mesh,
            in_specs=(
                # QKV [batch_size, seq_len, num_heads, per_head_dim].
                PartitionSpec("data", None, "model", None),
                PartitionSpec("data", None, "model", None),
                PartitionSpec("data", None, "model", None),
                # Bias [batch_size, num_heads, seq_len, seq_len].
                PartitionSpec("data", "model", None, None),
            ),
            # O [batch_size, seq_len, num_heads, per_head_dim].
            out_specs=PartitionSpec("data", None, "model", None),
            # Disables a checking pass which jax can't apply when there's a triton_call in the body.
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

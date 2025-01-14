# Copyright © 2023 Apple Inc.

"""FlashAttention layers."""

from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.interpreters.pxla import thread_resources
from jax.sharding import PartitionSpec

from axlearn.common.attention import Dropout, GroupedQueryAttention
from axlearn.common.attention_bias import BaseAttentionBias
from axlearn.common.config import config_class
from axlearn.common.flash_attention.utils import (
    MultiHeadAttentionImpl,
    flash_attention_implementation,
)
from axlearn.common.module import Module
from axlearn.common.utils import Tensor, with_sharding_constraint


class FlashAttention(GroupedQueryAttention):
    """FlashAttention layer.

    Is a drop-in replacement of GroupedQueryAttention
        (which itself supports MultiheadAttention as a special case), with some limitations:
            * Does not support dropout.
            * Does not support gradients wrt. attention logit biases.
            * Supports a subset of config fields and outputs.

    On TPU, we only pay to attend to non-masked entries by using Pallas's SplashAttention
    when supported.
    Otherwise, we pay for every entry unless `mask=causal_mask`.
    Note that using an identical implementation of `causal_mask` that does not
    compare reference equal to `causal_activated_attention` will not trigger the optimization
    when not using SplashAttention.
    """

    @config_class
    class Config(GroupedQueryAttention.Config):
        """Configures FlashAttention."""

        # Deprecated. Use `mask=causal_mask` instead.
        # If True, applies additional optimizations in the FlashAttention kernels.
        # Causal attention can still be used when False, by passing logit biases.
        # TODO (apghml) remove this in favor of `mask`.
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
        mha_dim_to_partition_spec: dict[str, Optional[PartitionSpec]] = {}
        # How to partition output values, keyed by dims.
        output_dim_to_partition_spec: dict[str, Optional[PartitionSpec]] = {}

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if getattr(cfg, "atten_logit_cap", None) is not None:
            raise NotImplementedError("cfg.atten_logit_cap is not supported.")
        # We're checking for an exact class match here.
        # pylint: disable-next=unidiomatic-typecheck
        if type(self.dropout) is not Dropout:
            raise NotImplementedError(
                f"Only {Dropout.__module__}.{Dropout.__qualname__} is supported for "
                "FlashAttention. Got "
                f"{type(self.dropout).__module__}.{type(self.dropout).__qualname__}"
            )
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

    def _backend(self):
        # For compatibility with AOT compilation, we obtain the backend type from physical_mesh.
        global_mesh = thread_resources.env.physical_mesh
        if len(global_mesh.devices):
            backend = global_mesh.devices.flat[0].platform
        else:
            # Fall back to jax.default_backend() if no device is found in physical_mesh.
            backend = jax.default_backend()
        return backend

    def _logit_biases_spec(self, attention_logit_biases: BaseAttentionBias) -> BaseAttentionBias:
        cfg = self.config
        return attention_logit_biases.partition_spec(cfg.mha_dim_to_partition_spec)

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        attention_logit_biases: BaseAttentionBias,
    ) -> tuple[Tensor, Tensor]:
        cfg: FlashAttention.Config = self.config
        backend = self._backend()

        batch, target_len, num_heads, _ = q_proj.shape
        _, source_len, _, _ = k_proj.shape

        attention_logit_biases = attention_logit_biases.astype(q_proj.dtype)

        jit_attn: MultiHeadAttentionImpl = flash_attention_implementation(
            backend=backend,
            softmax_scale=1.0,
            block_size=cfg.tpu_block_size,
            dropout_rate=cfg.dropout.rate,
        )

        attention_logit_biases_spec = self._logit_biases_spec(attention_logit_biases)
        attention_logit_biases = with_sharding_constraint(
            attention_logit_biases, attention_logit_biases_spec
        )

        # Scale query and key.
        q_proj = self.scale_query(q_proj)
        k_proj = self.scale_key(k_proj)

        # Constrain input to conform to partitioned MHA expectations.
        q_proj = with_sharding_constraint(q_proj, cfg.mha_dim_to_partition_spec["btnh"])
        k_proj = with_sharding_constraint(k_proj, cfg.mha_dim_to_partition_spec["bsnh"])
        v_proj = with_sharding_constraint(v_proj, cfg.mha_dim_to_partition_spec["bsnh"])

        # We need to manually partition pallas | jax-triton calls.
        # Note: shard_map doesn't support kwargs.
        partitioned_mha = shard_map(
            jit_attn,
            mesh=thread_resources.env.physical_mesh,
            in_specs=(
                # Q [batch_size, seq_len, num_heads, per_head_dim].
                cfg.mha_dim_to_partition_spec["btnh"],
                # KV [batch_size, seq_len, num_kv_heads, per_head_dim].
                # Note: while num_kv_heads can be different from num_heads, their partition spec
                # should be the same.
                cfg.mha_dim_to_partition_spec["bsnh"],
                cfg.mha_dim_to_partition_spec["bsnh"],
                # Bias that can broadcast to [batch_size, num_heads, seq_len, seq_len].
                attention_logit_biases_spec,
                # PRNG Key.
                PartitionSpec(None),
            ),
            # O [batch_size, seq_len, num_heads, per_head_dim].
            out_specs=cfg.mha_dim_to_partition_spec["btnh"],
            # Disables a checking pass which jax can't apply when there's a triton | pallas
            # call in the body.
            check_rep=False,
        )

        outputs = with_sharding_constraint(
            partitioned_mha(
                # Note: we use dropout layer's prng_key so the dropout result is identical to
                # using self.dropout.forward because we will produce identical mask.
                q_proj,
                k_proj,
                v_proj,
                attention_logit_biases,
                self.dropout.get_prng_key(),
            ),
            cfg.output_dim_to_partition_spec["btnh"],
        )

        # TODO(markblee): Add output probs and benchmark.
        output_probs = with_sharding_constraint(
            jnp.empty((batch, num_heads, target_len, source_len)),
            cfg.output_dim_to_partition_spec["bnts"],
        )
        return outputs, output_probs


def default_mha_dim_to_partition_spec(
    mesh_axis_names: Sequence[str],
) -> dict[str, Optional[PartitionSpec]]:
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
) -> dict[str, Optional[PartitionSpec]]:
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

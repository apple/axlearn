# Copyright © 2023 Apple Inc.

"""FlashAttention layers."""
from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.shard_map import shard_map
from jax.interpreters.pxla import thread_resources
from jax.sharding import PartitionSpec

from axlearn.common.attention import (
    ForwardMode,
    GroupedQueryAttention,
    apply_attention_logit_biases,
    causal_mask,
    make_segment_mask,
)
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
        _check_bias_recursively(cfg)  # Bias not supported.
        if getattr(cfg, "atten_logit_cap", None) is not None:
            raise NotImplementedError("cfg.atten_logit_cap is not supported.")
        # TODO(kelvinzou): enable dropout for flash attention.
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

    def _is_mask_fn_used(self):
        backend = self._backend()
        # bias and segment_ids should also be None to use mask_fn (cf. _tpu_splash_attention in
        # tpu_attention.py).

        return (
            backend == "tpu"
            and self.per_head_dim() % splash_attention_kernel.NUM_LANES == 0
            and self._mask_fn is not None
        )

    def _logit_biases_for_mask(
        self,
        *,
        mode: ForwardMode,
        kv_len: int,
        query_len: Optional[int] = None,
        time_step: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if self._mask_fn is None:
            return None
        elif mode == ForwardMode.EXTEND_STEP:
            # Use biases for decoding.
            return super()._logit_biases_for_mask(mode=mode, kv_len=kv_len, time_step=time_step)
        elif self._is_mask_fn_used():
            # Biases are not needed in favor of mask_fn, which is supported in Splash Attention.
            return None
        elif self._mask_fn is causal_mask:
            # Causal mode is supported natively in Flash Attention.
            return None
        else:
            # Fall back to biases. In the subsequent _compute_attention calls, _mask_fn should not
            # be used.
            return super()._logit_biases_for_mask(
                mode=mode, kv_len=kv_len, query_len=query_len, time_step=time_step
            )

    def _backend(self):
        # For compatibility with AOT compilation, we obtain the backend type from physical_mesh.
        global_mesh = thread_resources.env.physical_mesh
        if len(global_mesh.devices):
            backend = global_mesh.devices.flat[0].platform
        else:
            # Fall back to jax.default_backend() if no device is found in physical_mesh.
            backend = jax.default_backend()
        return backend

    def _logit_biases_spec(self, attention_logit_biases: Tensor) -> Tensor:
        spec = self.config.mha_dim_to_partition_spec["bnts"]
        if spec != PartitionSpec(None):
            if attention_logit_biases.shape[0] == 1:
                spec = PartitionSpec(None, *spec[1:])
            if attention_logit_biases.shape[1] == 1:
                spec = PartitionSpec(spec[0], None, *spec[2:])
        return spec

    def _compute_attention(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        attention_logit_biases: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        backend = self._backend()

        # Repeats key/value heads dim if necessary.
        k_proj = self._repeat_kv_heads(k_proj)
        v_proj = self._repeat_kv_heads(v_proj)

        batch, target_len, num_heads, _ = q_proj.shape
        _, source_len, _, _ = k_proj.shape

        # Merge segment ids into attention_logit_biases.
        if segment_ids is not None and attention_logit_biases is not None:
            if q_proj.shape[1] != k_proj.shape[1]:
                raise ValueError(
                    "segment_ids is only supported for query and key with identical lengths."
                )
            attention_logit_biases = apply_attention_logit_biases(
                make_segment_mask(source_segments=segment_ids, target_segments=segment_ids),
                attention_logit_biases,
            )
            segment_ids = None

        if attention_logit_biases is not None:
            if attention_logit_biases.ndim != 4:
                raise ValueError(
                    f"Expected attention_logit_biases.ndim == 4, got {attention_logit_biases.ndim}"
                )
            bias_shape = attention_logit_biases.shape
            if (bias_shape[0] != 1 and bias_shape[0] != batch) or (
                bias_shape[1] != 1 and bias_shape[1] != num_heads
            ):
                raise ValueError(
                    "attention_logit_biases must broadcast to "
                    f"{(batch, num_heads, target_len, source_len)}, "
                    f"got {attention_logit_biases.shape}."
                )
            attention_logit_biases = attention_logit_biases.astype(q_proj.dtype)

        if attention_logit_biases is None or self._mask_fn is causal_mask:
            mask_fn = self._mask_fn
        else:
            mask_fn = None

        # During GPU decoding, fall back to plain MHA implementation
        # since the seq_len will not be divisible by block size.
        # For prefill, seq_len can be > 1 and logit biases may not always be provided,
        # so we retain `mask_fn`.
        # For decoding, seq_len = 1 and logit biases are always provided,
        # so we set `mask_fn` to None.
        if q_proj.shape[1] % 128 != 0:
            backend = "xla"
            # TODO(senyut): Implement FlashDecoding kernel and support TPU decoding.
            if q_proj.shape[1] == 1:
                mask_fn = None
        elif backend == "gpu" and q_proj.shape[1] != k_proj.shape[1]:
            # TODO(xuan-zou): Generalize GPU Flash Attention for q_len != kv_len.
            # Remove pytest.skip corresponding to q_len != kv_len in layer_test.py once fixed.
            raise NotImplementedError(
                f"Query length {q_proj.shape[1]} must be equal to KV length "
                f"{k_proj.shape[1]} for correctly supported GPU flash attention usage."
            )

        if backend == "tpu":
            assert q_proj.shape[1] % cfg.tpu_block_size == 0, (
                f"Target seq len {q_proj.shape[1]} must be "
                f"divisible by block size {cfg.tpu_block_size}."
            )
            assert k_proj.shape[1] % cfg.tpu_block_size == 0, (
                f"Source seq len {k_proj.shape[1]} must be "
                f"divisible by block size {cfg.tpu_block_size}."
            )

        jit_attn: MultiHeadAttentionImpl = flash_attention_implementation(
            backend=backend,
            mask=mask_fn,
            softmax_scale=1.0,
            block_size=cfg.tpu_block_size,
        )

        q_spec = cfg.mha_dim_to_partition_spec["btnh"]
        segment_ids_spec = (
            PartitionSpec(q_spec[0], q_spec[1])
            if q_spec != PartitionSpec(None)
            else PartitionSpec(None)
        )

        attention_logit_biases_spec = cfg.mha_dim_to_partition_spec["bnts"]
        if attention_logit_biases is not None:
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

        if segment_ids is not None:
            if segment_ids.shape[0] != q_proj.shape[0]:
                raise ValueError(
                    "segment_ids must have matching batch dim: "
                    f"{segment_ids.shape} vs. {q_proj.shape[0]}"
                )
            segment_ids = with_sharding_constraint(segment_ids, segment_ids_spec)

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
                # Bias that can broadcast to [batch_size, num_heads, seq_len, seq_len].
                attention_logit_biases_spec,
                # Segment IDs [batch_size, seq_len].
                segment_ids_spec,
            ),
            # O [batch_size, seq_len, num_heads, per_head_dim].
            out_specs=cfg.mha_dim_to_partition_spec["btnh"],
            # Disables a checking pass which jax can't apply when there's a triton | pallas
            # call in the body.
            check_rep=False,
        )

        outputs = with_sharding_constraint(
            partitioned_mha(q_proj, k_proj, v_proj, attention_logit_biases, segment_ids),
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

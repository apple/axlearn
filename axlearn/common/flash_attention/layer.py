# Copyright © 2023 Apple Inc.

"""FlashAttention layers."""

from collections.abc import Sequence
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.experimental.shard_map import shard_map
from jax.interpreters.pxla import thread_resources
from jax.sharding import PartitionSpec

from axlearn.common.attention import Dropout, ForwardMode, GroupedQueryAttention, KVState
from axlearn.common.attention_bias import BaseAttentionBias
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.config import ConfigBase, ConfigModifier, config_class
from axlearn.common.flash_attention.utils import flash_attention_implementation
from axlearn.common.module import Module
from axlearn.common.utils import Tensor, maybe_shard, with_sharding_constraint


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

        # The block size used to tile attention computation (for TPU only).
        # Should be less than the target sequence length and a multiple of 128 on TPU.
        # TODO(tom_gunter): Expose GPU block-size (currently always 128) & unify.
        tpu_block_size: int = 512
        # The default GPU block-size of 128 works on most accelerators
        # NVIDIA Blackwell (B200) requires a smaller block-size
        gpu_block_size: Optional[int] = None

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

        # Backend specific config overrides.
        # TODO(hanzhi-zhou): Unify tpu_block_size and gpu_block_size with backend_overrides.
        backend_overrides: Optional[dict[str, Any]] = None

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
            raise ValueError("cfg.tpu_block_size must be divisible by 128.")

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = super()._create_layer_parameter_specs()

        # Derive the mesh_axes for sink parameters from mha_dim_to_partition_spec.
        if cfg.logit_sink:
            if len(cfg.mha_dim_to_partition_spec["bsnh"]) < 3:
                params["sink"].mesh_axes = (None,)
            else:
                params["sink"].mesh_axes = (cfg.mha_dim_to_partition_spec["bsnh"][2],)

        return params

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

    def _maybe_repeat_kv_heads(self, key_or_value: Tensor) -> Tensor:
        """Repeats key or value heads dim to be shardable."""
        cfg: FlashAttention.Config = self.config
        partition_spec = cfg.mha_dim_to_partition_spec["bsnh"]
        global_mesh = thread_resources.env.physical_mesh
        if (
            partition_spec == PartitionSpec(None)
            or len(partition_spec) != 4
            or partition_spec[-2] is None
        ):
            return key_or_value

        axis = partition_spec[-2]
        if isinstance(axis, tuple):
            axis_size = np.prod([global_mesh.shape[x] for x in axis])
        else:
            axis_size = global_mesh.shape[axis]
        # There will be sharding error if axis_size > num_heads.
        if cfg.num_heads < axis_size:
            raise ValueError(
                f"num_heads ({cfg.num_heads}) must be greater than or equal to "
                f"the number of devices {axis_size} in the mesh axis {axis}."
            )
        num_head_repeats = axis_size // key_or_value.shape[-2]
        # Repeat along the num_heads dim: [batch, source_length, repeated_num_heads, per_head_dim].
        if num_head_repeats > 1:
            logging.info(
                "Repeating %d KV heads %d times to meet the size of %s, which is %d.",
                key_or_value.shape[-2],
                num_head_repeats,
                axis,
                axis_size,
            )
            key_or_value = jnp.repeat(key_or_value, num_head_repeats, axis=-2)
            if cfg.k_partition_spec != cfg.v_partition_spec:
                raise ValueError(
                    "FlashAttention doesn't support "
                    f"{cfg.k_partition_spec=} != {cfg.v_partition_spec}"
                )
            # This maybe_shard is required when using "seq" > num_kv_heads and DeepSpeed Ulysses
            # style sequence parallelism. It tells the compiler to not reshard from partitioning
            # along the sequence axis to head axis before the `jnp.repeat` above, which otherwise
            # would cause an involuntary full materialization.
            key_or_value = maybe_shard(key_or_value, cfg.k_partition_spec or cfg.q_partition_spec)

        if key_or_value.shape[-2] % axis_size != 0:
            raise ValueError(
                f"repeated_num_heads dim size {key_or_value.shape[-2]} must be "
                f"fully divisible by mesh axis {axis} size {axis_size}."
            )

        return key_or_value

    def _compute_attention(
        self,
        *,
        mode: ForwardMode,
        q_proj: Tensor,
        kv_state: KVState,
        attention_logit_biases: BaseAttentionBias,
    ) -> tuple[Tensor, Tensor]:
        """Computes attention context and probs.

        Note: KV cache may cast k_proj/v_proj in lower precision, so flash attention kernel must
        cast them to q_proj.dtype.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted. See `ForwardMode` for
                details.
            q_proj: [batch_size, target_length, num_heads, per_head_dim].
            kv_state: The KV State dataclass containing k_proj, v_proj, key_positions, and optional
                attributes such as page_indices.
            attention_logit_biases: See ``On attention logit biases`` in the file comments.

        Returns:
            The context of shape [batch_size, target_length, num_heads, per_head_dim],
            and probs of shape [batch, num_heads, target_length, source_length].
        """
        cfg: FlashAttention.Config = self.config
        backend = self._backend()

        k_proj, v_proj = kv_state.k_proj, kv_state.v_proj
        page_indices = kv_state.page_indices

        # Repeats key/value heads dim if necessary.
        if page_indices is None:
            k_proj = self._maybe_repeat_kv_heads(k_proj)
            v_proj = self._maybe_repeat_kv_heads(v_proj)
        attention_logit_biases = attention_logit_biases.astype(q_proj.dtype)

        kv_cache_type = self._get_kv_cache_type(mode)

        # Get logit sink parameter if configured.
        logit_sink = self.parameters.get("sink", None)

        jit_attn = flash_attention_implementation(
            backend=backend,
            query=q_proj,
            key=k_proj,
            value=v_proj,
            bias=attention_logit_biases,
            logit_sink=logit_sink,
            softmax_scale=1.0,
            kv_cache_type=kv_cache_type,
            # TODO(hanzhi-zhou): Refactor backend specific config passing.
            tpu_block_size=cfg.tpu_block_size,
            gpu_block_size=cfg.gpu_block_size or 128,
            dropout_rate=cfg.dropout.rate,
            page_tables=page_indices,
            backend_overrides=cfg.backend_overrides,
        )
        if jit_attn is None:
            # Fall back to standard attention if no backend kernels are supported.
            return super()._compute_attention(
                mode=mode,
                q_proj=q_proj,
                attention_logit_biases=attention_logit_biases,
                kv_state=kv_state,  # Use the original kv_state.
            )

        batch, target_len, num_heads, _ = q_proj.shape
        if page_indices is None:
            _, source_len, _, _ = k_proj.shape
        else:
            source_len = k_proj.shape[1] * page_indices.shape[-1]

        attention_logit_biases_spec = self._logit_biases_spec(attention_logit_biases)
        attention_logit_biases = with_sharding_constraint(
            attention_logit_biases, attention_logit_biases_spec
        )

        # When using paged attention, k/v_proj have different format.
        kv_partition = "bsnh" if page_indices is None else "nbph"

        # Constrain input to conform to partitioned MHA expectations.
        q_proj = with_sharding_constraint(q_proj, cfg.mha_dim_to_partition_spec["btnh"])
        k_proj = with_sharding_constraint(k_proj, cfg.mha_dim_to_partition_spec[kv_partition])
        v_proj = with_sharding_constraint(v_proj, cfg.mha_dim_to_partition_spec[kv_partition])

        # We need to manually partition pallas | jax-triton calls.
        # Note: shard_map doesn't support kwargs.
        input_batch_specs = {
            # Q [batch_size, seq_len, num_heads, per_head_dim].
            "query": cfg.mha_dim_to_partition_spec["btnh"],
            # KV [batch_size, seq_len, repeated_num_heads, per_head_dim].
            # repeated_num_heads should be divided evenly by the n axis.
            "key": cfg.mha_dim_to_partition_spec[kv_partition],
            "value": cfg.mha_dim_to_partition_spec[kv_partition],
            # PRNG Key
            "prng_key": PartitionSpec(None),
            # Bias that can broadcast to [batch_size, num_heads, seq_len, seq_len].
            "bias": attention_logit_biases_spec,
            # Logit sink values of shape [num_heads].
            "logit_sink": (
                PartitionSpec(None)
                if logit_sink is None or len(cfg.mha_dim_to_partition_spec["bsnh"]) < 3
                else PartitionSpec(cfg.mha_dim_to_partition_spec["bsnh"][2])
            ),
            # PagedKVCache's page indices.
            "page_tables": cfg.mha_dim_to_partition_spec.get("bs", PartitionSpec(None)),
        }
        partitioned_mha = shard_map(
            jit_attn,
            mesh=thread_resources.env.physical_mesh,
            in_specs=(input_batch_specs,),
            # O [batch_size, seq_len, num_heads, per_head_dim].
            out_specs=cfg.mha_dim_to_partition_spec["btnh"],
            # Disables a checking pass which jax can't apply when there's a triton | pallas
            # call in the body.
            check_rep=False,
        )

        # Note: we use dropout layer's prng_key so the dropout result is identical to
        # using self.dropout.forward because we will produce identical mask.
        input_batch = {
            "query": q_proj,
            "key": k_proj,
            "value": v_proj,
            "prng_key": self.dropout.get_prng_key(),
            "bias": attention_logit_biases,
            "logit_sink": logit_sink,
            "page_tables": page_indices,
        }
        outputs = with_sharding_constraint(
            partitioned_mha(
                input_batch,
            ),
            cfg.output_dim_to_partition_spec["btnh"],
        )

        # TODO(markblee): Add output probs and benchmark.
        output_probs = with_sharding_constraint(
            jnp.empty((batch, num_heads, target_len, source_len)),
            cfg.output_dim_to_partition_spec["bnts"],
        )
        return outputs, output_probs

    def _get_kv_cache_type(self, mode: ForwardMode):
        # Note: prefill (INIT_STATE) is not decoding because query and key have the same shape.
        # Note: this is a heuristic and it is possible (although not currently common) to do
        # an extend_step even if we aren't in decoding. A more robust method could instead directly
        # look at whether we need gradients or not, which could be done by adding a custom_vjp.
        is_decoding = mode == ForwardMode.EXTEND_STEP
        return type(self.kv_cache) if is_decoding else None


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


class FlashBlockSizeModifier(ConfigModifier):
    """Modifies the tpu_block_size or gpu_block_size config of FlashAttention."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configures FlashBlockSizeModifier."""

        tpu_block_size: Optional[int] = 512
        gpu_block_size: Optional[int] = None

    def __call__(self, cfg: ConfigBase) -> ConfigBase:
        tpu_block_size = self.config.tpu_block_size
        gpu_block_size = self.config.gpu_block_size

        def is_flash_config(cfg):
            return isinstance(cfg, FlashAttention.Config)

        def visit_fn(_, value):
            if is_flash_config(value):
                value = cast(FlashAttention.Config, value)
                value.tpu_block_size = tpu_block_size
                value.gpu_block_size = gpu_block_size

        def enter_fn(_, value, default_kv):
            return None if is_flash_config(value) else default_kv

        cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
        return cfg


class BackendOverrideModifier(ConfigModifier):
    """Modifies the backend_overrides config of Flash Attention."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configures BackendOverrideModifier."""

        backend_overrides: Optional[dict[str, Any]] = None

    def __call__(self, cfg: ConfigBase) -> ConfigBase:
        backend_overrides = self.config.backend_overrides

        def is_flash_config(cfg):
            return isinstance(cfg, FlashAttention.Config)

        def visit_fn(_, value):
            if is_flash_config(value):
                value = cast(FlashAttention.Config, value)
                if backend_overrides:
                    # Instantiate a dict if value.backend_overrides hasn't already been set
                    if value.backend_overrides is None:
                        value.backend_overrides = dict()
                    for override_key, override_value in backend_overrides.items():
                        # Ensure we don't insert any values equal to None
                        if override_value:
                            # Use .update() to avoid overwriting existing overrides
                            value.backend_overrides.update({override_key: override_value})

        def enter_fn(_, value, default_kv):
            return None if is_flash_config(value) else default_kv

        cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
        return cfg

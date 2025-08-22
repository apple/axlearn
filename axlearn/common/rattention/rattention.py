"""Implementation of RAttention with residual linear attention."""
from functools import partial
from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp
from jax._src.mesh import thread_resources
from jax.experimental.shard_map import shard_map

from axlearn.common.attention import (
    BaseQKVLinear,
    ForwardMode,
    KVCache,
    KVState,
    RoFormerQKVLinear,
    RoFormerSinusoidalPositionalEmbedding,
)
from axlearn.common.attention import (
    apply_rotary_position_embeddings as orig_apply_rotary_position_embeddings,
)
from axlearn.common.attention_bias import (
    BaseAttentionBias,
    SegmentIdAttentionBias,
    SlidingWindowAttentionBias,
    as_attention_bias,
)
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.ein_ops import rearrange, repeat
from axlearn.common.flash_attention.layer import FlashAttention
from axlearn.common.layers import BaseLayer, BaseNormalizationLayer
from axlearn.common.module import Module, child_context
from axlearn.common.param_init import ConstantInitializer, FanAxes, WeightInitializer
from axlearn.common.rattention.kernels.linear_attention_kernels import (
    residual_linear_attention,
    residual_linear_attention_linear_scan,
    residual_linear_attention_w_timestep,
    right_shift_and_zero_pad,
)
from axlearn.common.rattention.kernels.utils import FeatureMap, get_feature_map
from axlearn.common.rattention.utils import GroupRMSNorm
from axlearn.common.utils import Nested, NestedTensor, PartitionSpec, Tensor, TensorSpec


def apply_rotary_position_embeddings(
    inputs: Tensor,
    sinusoidal_pos: Tensor,
) -> Tensor:
    """Applies rotary position embeddings to the inputs.

    The original function applies RoPE to query/key/value at once. This helper function makes it
    easy to apply RoPE individually.
    """

    # Only applies RoPE to inputs via query, no effect on key/value.
    return orig_apply_rotary_position_embeddings(
        query=inputs,
        key=inputs,
        value=inputs,
        sinusoidal_pos=sinusoidal_pos,
        rotary_key=False,
        rotary_value=False,
    )[0]


class ResidualLinearAttention(BaseLayer):
    """Residual Linear Attention layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Config for ResidualLinearAttention layer.

        Args:
            input_dim: The input dimension.
            hidden_dim: The hidden dimension.
            num_heads: The number of attention heads.
            num_kv_heads: The number of key/value heads.
            sliding_window_size: The size of the sliding window. If -1, no sliding window is used.
            feat_fn: The feature function to use. Currently only softmax is optimized with kernel.
            chunk_size: The chunk size for the linear attention kernel.
            use_learned_init: Whether to use learned initialization for the linear attention kernel.
            use_qk_scale: Whether to use learned scaling for q_proj and k_proj in linear attention.
            dim_to_partition_spec: The partition spec for the input tensors.
            output_partition_spec: The partition spec for the output tensor.
        """

        input_dim: Required[int] = REQUIRED
        hidden_dim: Required[int] = REQUIRED
        num_heads: Required[int] = REQUIRED
        num_kv_heads: Required[int] = REQUIRED

        sliding_window_size: Required[int] = REQUIRED

        feat_fn: FeatureMap = FeatureMap("softmax")
        chunk_size: int = 512
        # If None/False, no learned initialization is used. The initial state will all zeros.
        # If True, creates trainable params `k_tokens` and `v_tokens`, each of shape
        # (num_heads, num_meta_tokens, per_head_dim), with the given initializer.
        use_learned_init: Optional[bool] = False
        # If None/False, directly reusing qk from SWA.
        # If True, create trainable params `q_scale` and `k_scale`, each of shape [per_head_dim,].
        use_qk_scale: Optional[bool] = False

        dim_to_partition_spec: Optional[dict[str, PartitionSpec]] = None
        output_partition_spec: Optional[PartitionSpec] = None

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        cfg = cfg.set(
            dim_to_partition_spec={
                "bnth": PartitionSpec(("data", "expert", "fsdp"), "model", "seq", None),
                "bnt": PartitionSpec(("data", "expert", "fsdp"), "model", "seq"),
                "bnkv": PartitionSpec(("data", "expert", "fsdp"), "model", None, None),
                "b": PartitionSpec(
                    ("data", "expert", "fsdp"),
                ),
            },
            output_partition_spec=PartitionSpec(("data", "expert", "fsdp"), "model", "seq", None),
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        assert cfg.feat_fn in [FeatureMap.SOFTMAX, FeatureMap.RELU], "Unsupported feature map"
        assert cfg.sliding_window_size >= -1, "Sliding window size must be non-negative."

    def per_head_dim(self) -> int:
        """Returns the dimension of each attention head."""
        cfg = self.config
        return cfg.hidden_dim // cfg.num_heads

    def _create_layer_parameter_specs(self):
        cfg = self.config
        param_dict = {}
        if cfg.use_learned_init:
            num_meta_tokens = 128
            meta_params = dict(
                k_tokens=ParameterSpec(
                    shape=(cfg.num_heads, num_meta_tokens, self.per_head_dim()),
                    mesh_axes=("model", None, None),
                    fan_axes=FanAxes(in_axis=1, out_axis=2, batch_axis=0),
                    initializer=WeightInitializer.default_config()
                    .set(fan="fan_out", distribution="normal")
                    .instantiate(),
                    dtype=cfg.dtype,
                    weight_decay_scale=0.0,
                ),
                v_tokens=ParameterSpec(
                    shape=(cfg.num_heads, num_meta_tokens, self.per_head_dim()),
                    mesh_axes=("model", None, None),
                    fan_axes=FanAxes(in_axis=1, out_axis=2, batch_axis=0),
                    initializer=WeightInitializer.default_config()
                    .set(fan="fan_out", distribution="normal")
                    .instantiate(),
                    dtype=cfg.dtype,
                    weight_decay_scale=0.0,
                ),
            )
            param_dict.update(meta_params)

        if cfg.use_qk_scale:
            param_dict.update(
                q_scale=ParameterSpec(
                    shape=(self.per_head_dim(),),
                    mesh_axes=("model",),
                    initializer=ConstantInitializer.default_config().set(value=1.0).instantiate(),
                    dtype=cfg.dtype,
                    weight_decay_scale=0.0,
                ),
                k_scale=ParameterSpec(
                    shape=(self.per_head_dim(),),
                    mesh_axes=("model",),
                    initializer=ConstantInitializer.default_config().set(value=1.0).instantiate(),
                    dtype=cfg.dtype,
                    weight_decay_scale=0.0,
                ),
            )
        return param_dict

    def _get_linear_attention_impl(
        self, decoding_mode=False
    ) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        """Get the kernel for running linear attention.

        In TPU training, the TPU kernel will be used. In other cases (e.g., GPU or TPU decoding),
        the linear scan kernel will be activated.
        """
        cfg = self.config
        global_mesh = thread_resources.env.physical_mesh
        if len(global_mesh.devices):
            backend = global_mesh.devices.flat[0].platform
        else:
            backend = jax.default_backend()

        in_specs = [
            cfg.dim_to_partition_spec["bnth"],
            cfg.dim_to_partition_spec["bnth"],
            cfg.dim_to_partition_spec["bnth"],
            cfg.dim_to_partition_spec["bnkv"],
        ]
        if backend == "tpu" and not decoding_mode:
            rla_impl = residual_linear_attention
        elif decoding_mode:
            rla_impl = residual_linear_attention_w_timestep
            in_specs.append(cfg.dim_to_partition_spec["b"])  # time_step
        else:
            rla_impl = residual_linear_attention_linear_scan
        rla_impl = partial(
            rla_impl,
            window_size=cfg.sliding_window_size,
            feat_map=cfg.feat_fn,
            chunk_size=cfg.chunk_size,
        )

        sharded_la = shard_map(
            rla_impl,
            mesh=thread_resources.env.physical_mesh,
            in_specs=tuple(in_specs),
            out_specs=cfg.output_partition_spec,
            check_rep=False,
        )
        return sharded_la

    def _prepare_linear_attention_inputs(
        self,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        time_step: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply feat_fn functions to prepare q_proj, k_proj, v_proj for linear attention.

        Args:
            q_proj: [batch_size, seq_len, num_heads, per_head_dim]
            k_proj: [batch_size, seq_len, num_kv_heads, per_head_dim]
            v_proj: [batch_size, seq_len, num_kv_heads, per_head_dim]
            time_step: [batch_size] or None, used for decoding.

        Returns:
            q_proj: [batch_size, seq_len, num_heads, new_per_head_dim]
            k_proj: [batch_size, seq_len, num_kv_heads, new_per_head_dim]
            v_proj: [batch_size, seq_len, num_kv_heads, per_head_dim]

        See kernels/utils.py for details of the supported feature functions.
        """
        del time_step  # Time_step is not used in this function.
        cfg = self.config

        feat_fn = get_feature_map(cfg.feat_fn)
        q_proj = feat_fn.fwd(q_proj)
        k_proj = feat_fn.fwd(k_proj)

        return q_proj, k_proj, v_proj

    def _repeat_kv_heads(self, key_or_value: Tensor) -> Tensor:
        cfg = self.config
        assert cfg.num_kv_heads == key_or_value.shape[-2]
        num_head_repeats = cfg.num_heads // key_or_value.shape[-2]
        return jnp.repeat(key_or_value, num_head_repeats, axis=-2)

    def _compute_init_state(self, batch_size: int) -> Tensor:
        """Compute the initial state for linear attention.

        Args:
            batch_size: The batch size of the input tensor.

        Returns:
            The initial state tensor of shape (batch_size, num_heads, 2 * head_dim, head_dim).
        """
        cfg = self.config
        feat_fn = get_feature_map(cfg.feat_fn)
        if cfg.use_learned_init:
            meta_k_tokens = self.parameters["k_tokens"]
            meta_v_tokens = self.parameters["v_tokens"]
            k = feat_fn.fwd(meta_k_tokens)
            h0 = jnp.einsum("nlk,nlv -> nkv", k, meta_v_tokens)
            h0 = repeat(h0, "n k v -> b n k v", b=batch_size)
        else:
            head_dim = self.per_head_dim()
            h0 = jnp.zeros((batch_size, cfg.num_heads, head_dim * 2, head_dim))
        return h0

    def _compute_linear_attention_parallel(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        time_step: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Parallel computation of linear attention, either for training or prefilling.

        Args:
            q_proj: [batch_size, seq_len, num_heads, per_head_dim]
            k_proj: [batch_size, seq_len, num_kv_heads, per_head_dim]
            v_proj: [batch_size, seq_len, num_kv_heads, per_head_dim]
            time_step: [batch_size] or None, used for decoding.

        Returns:
            state: [batch_size, num_heads, 2 * head_dim, head_dim]
            context: [batch_size, seq_len, num_heads, per_head_dim]
        """
        q_proj = rearrange(q_proj, "b t n h -> b n t h")
        k_proj = rearrange(k_proj, "b t n h -> b n t h")
        v_proj = rearrange(v_proj, "b t n h -> b n t h")

        h0 = self._compute_init_state(batch_size=q_proj.shape[0]).astype(v_proj.dtype)
        if time_step is None:
            sharded_la = self._get_linear_attention_impl(decoding_mode=False)
            rla_output = sharded_la(q_proj, k_proj, v_proj, h0)
        else:
            sharded_la = self._get_linear_attention_impl(decoding_mode=True)
            rla_output = sharded_la(q_proj, k_proj, v_proj, h0, time_step)

        if isinstance(rla_output, tuple):
            context, state = rla_output
        else:
            context, state = rla_output, None
        context = rearrange(context, "b n t h -> b t n h")
        return state, context

    def _compute_linear_attention_step(
        self,
        *,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        time_step: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute linear attention for a single step.

        Args:
            q_proj: [batch_size, 1, num_heads, per_head_dim]
            k_proj: [batch_size, seq_len, num_kv_heads, per_head_dim]
            v_proj: [batch_size, seq_len, num_kv_heads, per_head_dim]
            time_step: [batch_size], used for decoding.
            state: [batch_size, num_heads, 2 * head_dim, head_dim]
        Returns:
            state: [batch_size, num_heads, 2 * head_dim, head_dim]
            context: [batch_size, 1, num_heads, per_head_dim]
        """
        cfg = self.config
        orig_dtype = v_proj.dtype

        q_proj, k_proj, v_proj = map(lambda x: x.astype(jnp.float32), (q_proj, k_proj, v_proj))
        q_proj, k_proj, v_proj = self._prepare_linear_attention_inputs(
            q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, time_step=time_step
        )

        shift_size = cfg.sliding_window_size + 1
        if cfg.sliding_window_size > 0:
            k_proj = right_shift_and_zero_pad(k_proj, shift_size)
            v_proj = right_shift_and_zero_pad(v_proj, shift_size)

        k_proj = self._repeat_kv_heads(k_proj)
        v_proj = self._repeat_kv_heads(v_proj)

        q_proj_t = q_proj.squeeze(axis=1)
        k_proj_t = jnp.take_along_axis(
            k_proj, time_step[:, None, None, None], axis=1, mode="clip"
        ).squeeze(axis=1)
        v_proj_t = jnp.take_along_axis(
            v_proj, time_step[:, None, None, None], axis=1, mode="clip"
        ).squeeze(axis=1)

        new_state = state.astype(jnp.float32) + jnp.einsum("bnk,bnv -> bnkv", k_proj_t, v_proj_t)
        output = jnp.einsum("bnk,bnkv -> bnv", q_proj_t, new_state)
        output = output[:, None, :, :]  # add seq_len dimension
        return new_state.astype(orig_dtype), output.astype(orig_dtype)

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        query: Union[Tensor, TensorSpec],
        qkv_proj: Optional[BaseQKVLinear.Output] = None,
        cached_states: Optional[NestedTensor] = None,
        page_pool: Optional[Nested[Tensor]] = None,
    ) -> tuple[Nested[Optional[Tensor]], Tensor]:
        """Forward function for linear attention."""
        assert page_pool is None
        # Initialize states.
        cfg = self.config
        if qkv_proj is None:
            assert isinstance(query, TensorSpec) and mode == ForwardMode.INIT_STATES
            init_state = self._compute_init_state(batch_size=query.shape[0]).astype(query.dtype)
            time_step = jnp.zeros((query.shape[0],), dtype=jnp.int32)
            return dict(time_step=time_step, state=init_state), None

        q_proj, k_proj, v_proj = qkv_proj
        if cfg.use_qk_scale:
            q_proj = q_proj * self.parameters["q_scale"]
            k_proj = k_proj * self.parameters["k_scale"]

        if mode == ForwardMode.FORWARD:
            state, output = self._compute_linear_attention_parallel(
                q_proj=q_proj, k_proj=k_proj, v_proj=v_proj
            )
            return None, output
        else:
            time_step = cached_states["time_step"]

            if mode == ForwardMode.INIT_STATES:
                assert time_step is not None, "time_step must be provided for prefilling."
                state, output = self._compute_linear_attention_parallel(
                    q_proj=q_proj,
                    k_proj=k_proj,
                    v_proj=v_proj,
                    time_step=time_step,
                )

            elif mode == ForwardMode.EXTEND_STEP:
                prev_state = cached_states["state"]
                state, output = self._compute_linear_attention_step(
                    q_proj=q_proj,
                    k_proj=k_proj,
                    v_proj=v_proj,
                    time_step=time_step,
                    state=prev_state,
                )

                time_step = time_step + query.shape[1]
            else:
                raise ValueError(f"Unrecognized mode {mode}.")
            return dict(time_step=time_step, state=state), output

    def forward(
        self,
        query: Tensor,
        qkv_proj: BaseQKVLinear.Output,
    ) -> Tensor:
        return self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            query=query,
            qkv_proj=qkv_proj,
        )[1]

    def init_states(
        self,
        query: Union[Tensor, TensorSpec],
        qkv_proj: Optional[BaseQKVLinear.Output] = None,
        time_step: Optional[Tensor] = None,
    ) -> tuple[Nested[Tensor], Tensor]:
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            query=query,
            qkv_proj=qkv_proj,
            cached_states=dict(time_step=time_step),
        )

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        query: Tensor,
        qkv_proj: BaseQKVLinear.Output,
        **kwargs,
    ) -> tuple[Nested[Tensor], Tensor]:
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            query=query,
            qkv_proj=qkv_proj,
            cached_states=cached_states,
            **kwargs,
        )


class RAttention(FlashAttention):
    """RAttention layer with residual linear attention and sliding window attention.

    RAttention has two separate branches to handle in-window and out-of-window tokens. The residual
    linear attention branch is used to handle out-of-window tokens, while the standard local sliding
    window attention branch is used to handle in-window tokens. The outputs from two branches are
    merged together with simple addition after applying group-wise RMSNorm.

    Compared with FlashAttention, RAttention will explicitly handle RoPE, instead of relying on
    `input_linear`. This is needed because the residual linear attention branch uses the plain
    q_proj/k_proj (i.e., no RoPE applied).
    """

    @config_class
    class Config(FlashAttention.Config):
        """Config for RAttention layer.
        Args:
            sliding_window_size: The size of the sliding window. If -1, no sliding window is used.
            residual_la: The config for the residual linear attention.
            mixing_group_norm: Whether to use group norm to mixing the outputs from residual linear
                attention and sliding window attention. If None, no group norm is used.
        """

        sliding_window_size: Required[int] = 1024
        residual_la: Optional[ResidualLinearAttention.Config] = None
        rope_theta: float = 500000.0
        # If None, default to standard GroupRMSNorm.
        mixing_norm: Optional[BaseNormalizationLayer.Config] = None

    @classmethod
    def default_config(cls) -> FlashAttention.Config:
        cfg = super().default_config()
        cfg = cfg.set(
            causal=True,
            mha_dim_to_partition_spec={
                "btnh": PartitionSpec(("data", "expert", "fsdp"), "seq", "model", None),
                "bsnh": PartitionSpec(("data", "expert", "fsdp"), "seq", "model", None),
                "bnts": PartitionSpec(("data", "expert", "fsdp"), None, None, None),
            },
            output_dim_to_partition_spec={
                "btnh": PartitionSpec(("data", "expert", "fsdp"), "seq", "model", None),
                "bnts": PartitionSpec(("data", "expert", "fsdp"), "model", "seq", None),
            },
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        """Initializes the RAttention layer."""
        # pylint: disable=non-parent-init-called,super-init-not-called
        BaseLayer.__init__(self, cfg, parent=parent)
        cfg = self.config

        # Configs mask for sliding window.
        assert cfg.mask is None and cfg.causal is True
        if cfg.sliding_window_size >= 0:
            self._mask_tpl = SlidingWindowAttentionBias.default_config(
                sliding_window_size=cfg.sliding_window_size
            )
        else:
            assert cfg.sliding_window_size == -1
            self._mask_tpl = None

        # Configure inputs to multi-headed QKV projection.
        i_proj_cfg = cfg.input_linear
        i_proj_cfg.query_dim = cfg.query_dim

        if i_proj_cfg.klass is RoFormerQKVLinear:
            raise ValueError(
                "Input projection for RAttention cannot be RoFormerQKVLinear as we \
                assume that position embeddings are handled explicitly in RAttention."
            )

        # Original qkv_linear requires query_dim = key_dim = value_dim.
        if hasattr(i_proj_cfg, "key_dim"):
            i_proj_cfg.key_dim = cfg.query_dim
        if hasattr(i_proj_cfg, "value_dim"):
            i_proj_cfg.value_dim = cfg.query_dim

        i_proj_cfg.num_heads = cfg.num_heads
        i_proj_cfg.per_head_dim = self.per_head_dim()
        self._add_child("i_proj", i_proj_cfg)

        # Configure output projection.
        o_proj_cfg = cfg.output_linear
        o_proj_cfg.model_dim = self.output_dim()
        o_proj_cfg.num_heads = cfg.num_heads
        o_proj_cfg.per_head_dim = self.per_head_dim()
        self._add_child("o_proj", o_proj_cfg)

        self._add_child("dropout", cfg.dropout)
        self._add_child("scale_query", cfg.query_scale.set(per_head_dim=self.per_head_dim()))
        self._add_child("scale_key", cfg.key_scale.set(per_head_dim=self.per_head_dim()))

        # Configure norm for merging linear attention and self-attention.
        if cfg.residual_la is not None:
            if cfg.mixing_norm is None:
                norm_cfg = GroupRMSNorm.default_config().set(
                    num_groups=cfg.num_heads,
                )
            norm_cfg = norm_cfg.set(input_dim=self.per_head_dim())
            self._add_child("rla_norm", norm_cfg.clone())
            self._add_child("swa_norm", norm_cfg.clone())

        # Configure rope.
        rope_cfg = RoFormerSinusoidalPositionalEmbedding.default_config().set(
            dim=self.per_head_dim(),
            theta=cfg.rope_theta,
        )
        self._add_child("rope", rope_cfg)

        # Configure linear attention.
        if cfg.residual_la is not None:
            self._add_child(
                "residual_la",
                cfg.residual_la.set(
                    input_dim=cfg.query_dim,
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    sliding_window_size=cfg.sliding_window_size,
                ),
            )

        kv_cache = cfg.kv_cache or KVCache.default_config()
        self._add_child("kv_cache", kv_cache)

    # pylint: disable=too-many-statements,too-many-branches
    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        query: Union[Tensor, TensorSpec],
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        kv_state: Optional[KVState] = None,
        unpadded_len: Optional[int] = None,
        attention_logit_biases: Union[None, Tensor, BaseAttentionBias] = None,
        segment_ids: Optional[Tensor] = None,
        query_positions: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
        return_aux: Optional[set[str]] = None,
        page_pool: Optional[Nested[Tensor]] = None,
    ) -> tuple[Nested[Tensor], Optional[FlashAttention.Output]]:
        """Forward function for RAttention.

        Residual linear attention will reuse the q_proj, k_proj, v_proj from the standard attention.
        It's important to note that unscaled and NoPE q_proj and k_proj are reused (not scaled and
        RoPE-applied version). In the case of kv sharing, we also need to pass the unscaled and NoPE
        k_proj/v_proj as kv_state to the output.

        Notes on intermediate variables:
            * unpadded_len vs time_step: time_step denotes the starting point where unpadded_len
              denotes the length of progression.
            * k_proj/v_proj vs full_k_proj/full_v_proj: the former could be single token during
              extend_step whereas the latter always means the kv for the whole sequence. Residual_la
              always takes in full_k_proj/full_v_proj.

        TODO (bailin-wang): full_k_proj/full_v_proj should be replaced with partial_k_proj/
            partial_v_proj which only stores kv for in-window tokens.
        """
        cfg = self.config
        assert key is None and value is None, "Cross-attention is not supported."

        # 0. Handle states along with query/key positions.
        if query_positions is None:
            query_positions = jnp.arange(query.shape[1])[None]
        if mode in (ForwardMode.EXTEND_STEP, ForwardMode.INIT_STATES):
            assert cached_states is not None
            time_step = cached_states["time_step"]
            query_positions = query_positions + time_step[:, None]  # [batch, steps]

        # 1. Compute `i_proj` considering external kv sharing.
        if kv_state is None:
            kv_kwargs = dict()
        else:
            kv_kwargs = dict(kv_state=kv_state)
        i_proj_output = self.i_proj(query, query_positions=query_positions, **kv_kwargs)
        q_proj, k_proj, v_proj = i_proj_output

        if mode == ForwardMode.FORWARD:
            full_k_proj, full_v_proj = k_proj, v_proj
            key_positions = jnp.arange(k_proj.shape[1])[None, :]
            kv_state = KVState(full_k_proj, full_v_proj, key_positions)

            if cfg.residual_la is not None:
                rla_output = self.residual_la(query, i_proj_output)
            else:
                rla_output = None
            new_cached_states = {}
        else:
            if kv_state is None:
                # Update kv cache of SWA.
                with child_context("kv_cache_extend_step", module=self.kv_cache):
                    swa_state, swa_cache_output = self.kv_cache.extend_step(
                        cached_states["swa_state"],
                        k_proj=k_proj,
                        v_proj=v_proj,
                        key_positions=query_positions,
                        unpadded_len=unpadded_len,
                        page_pool=page_pool,
                    )
                    if mode == ForwardMode.EXTEND_STEP:
                        full_k_proj, full_v_proj, key_positions, _ = swa_cache_output
                        kv_state = KVState(*swa_cache_output)
                    else:
                        # During prefill, q/k/v_proj are the same as in the forward pass.
                        # kv_cache.extend_step() was called to update the kv_cache.
                        assert mode == ForwardMode.INIT_STATES
                        full_k_proj, full_v_proj = k_proj, v_proj
                        key_positions = query_positions
                        kv_state = KVState(k_proj, v_proj, key_positions)
            else:
                # KV sharing branch. Pack the `k_proj` and `v_proj` (possibly updated
                # by i_proj), and the same `key_positions`.
                full_k_proj, full_v_proj = k_proj, v_proj
                key_positions = kv_state.key_positions
                kv_state = KVState(k_proj, v_proj, key_positions)

                # No need to cache the swa_state in when using external kv_state.
                swa_state = None

            # Update recurrent state of LA.
            if cfg.residual_la is None:
                rla_state = None
            else:
                if mode == ForwardMode.INIT_STATES:
                    rla_state, rla_output = self.residual_la.init_states(
                        query, (q_proj, full_k_proj, full_v_proj), unpadded_len
                    )
                else:
                    rla_state, rla_output = self.residual_la.extend_step(
                        cached_states["rla_state"], query, (q_proj, full_k_proj, full_v_proj)
                    )

            step_len = unpadded_len if unpadded_len is not None else query.shape[1]
            new_time_step = time_step + step_len
            new_cached_states = dict(
                swa_state=swa_state, rla_state=rla_state, time_step=new_time_step
            )

        # 2. Compute local attention with rope.
        with child_context("key_rope", module=self.rope):
            k_pos_emb = self.rope.forward(positions=key_positions)
        k_pos_emb = k_pos_emb[:, :, None, :]
        full_k_proj_rope = apply_rotary_position_embeddings(full_k_proj, k_pos_emb)

        with child_context("query_rope", module=self.rope):
            q_pos_emb = self.rope.forward(positions=query_positions)
            q_pos_emb = q_pos_emb[:, :, None, :]  # broadcast to all heads
        q_proj_rope = apply_rotary_position_embeddings(q_proj, q_pos_emb)

        # Prepare attention logits.
        attention_logit_biases = as_attention_bias(attention_logit_biases)
        if self._mask_tpl is not None:
            attention_logit_biases += self._mask_tpl.instantiate(
                target_positions=query_positions, source_positions=key_positions, dtype=q_proj.dtype
            )
        if segment_ids is not None:
            assert mode == ForwardMode.FORWARD, "segment_ids must be None in inference."
            attention_logit_biases += SegmentIdAttentionBias(segment_ids)

        # Assemble the final results.
        if cfg.sliding_window_size >= 0:
            scaled_q_proj_rope, scaled_full_k_proj_rope = self._scale_qk(
                q_proj=q_proj_rope,
                k_proj=full_k_proj_rope,
                query_positions=query_positions,
                key_positions=key_positions,
            )
            swa_context, _ = self._compute_attention(
                mode=mode,
                q_proj=scaled_q_proj_rope,
                kv_state=KVState(scaled_full_k_proj_rope, full_v_proj, key_positions),
                attention_logit_biases=attention_logit_biases,
            )
            if cfg.residual_la is not None:
                swa_context = self.swa_norm(swa_context)
                rla_context = self.rla_norm(rla_output)
                context = swa_context + rla_context
            else:
                context = swa_context
        else:
            context = self.rla_norm(rla_output)

        o_proj = self.o_proj(context)
        return_aux = return_aux or set()
        output = self.Output(
            data=o_proj,
            probs=None,
            kv_state=kv_state if "kv_state" in return_aux else None,
        )
        return new_cached_states, output

    def _repeat_kv_heads(self, key_or_value: Tensor) -> Tensor:
        num_head_repeats = self.config.num_heads // key_or_value.shape[-2]
        return jnp.repeat(key_or_value, num_head_repeats, axis=-2)

    def init_states(
        self,
        *,
        time_step: Optional[Tensor],
        query: Union[Tensor, TensorSpec],
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        kv_state: Optional[KVState] = None,
        attention_logit_biases: Optional[Tensor] = None,
        return_aux: Optional[set[str]] = None,
    ) -> tuple[Nested[Tensor], Optional[FlashAttention.Output]]:
        if key is not None and query.shape[1] != key.shape[1]:
            raise ValueError("Cross-attention extend_step is not supported.")
        cfg = self.config

        init_states = dict(time_step=jnp.zeros([query.shape[0]], dtype=jnp.int32))

        if kv_state is None:
            kv_shape = KVCache.Shape(
                batch_size=query.shape[0],
                kv_len=query.shape[1],
                num_kv_heads=self.i_proj.num_kv_heads,
                per_head_dim=self.per_head_dim(),
            )
            init_states.update(
                swa_state=self.kv_cache.init_states(shape=kv_shape, dtype=query.dtype)
            )

        # `rla_state` for prefilling (i.e., query is Tensor) will be set later.
        if isinstance(query, TensorSpec) and cfg.residual_la is not None:
            init_states.update(rla_state=self.residual_la.init_states(query)[0])

        if time_step is None:
            return init_states, None

        cached_states, output = self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            query=query,
            key=key,
            value=value,
            unpadded_len=time_step,
            cached_states=init_states,
            kv_state=kv_state,
            attention_logit_biases=attention_logit_biases,
            return_aux=return_aux,
        )
        return cached_states, output

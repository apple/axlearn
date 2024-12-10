# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# state-spaces/mamba
# Copyright 2023 Tri Dao and Albert Gu. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# johnma2006/mamba-minimal
# Copyright 2023 John Ma. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# huggingface/transformers
# Copyright 2024 The Huggingface Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# ai21labs/Jamba-v0.1
# (https://huggingface.co/ai21labs/Jamba-v0.1)
# Copyright 2024 The AI21 Jamba authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Implementation of Mamba and Jamba State-space Models (SSMs)."""

import functools
import math
from collections.abc import Sequence
from enum import Enum, unique
from typing import NamedTuple, Optional, Union

import jax
import jax.ad_checkpoint
from einops import rearrange, repeat
from jax import numpy as jnp
from jax._src.mesh import thread_resources
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec

from axlearn.common.attention import (
    BaseTransformerLayer,
    ForwardMode,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerFeedForwardLayer,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Conv1D, GroupNorm, Linear, MultiLinear, NormType, RMSNorm
from axlearn.common.module import Module
from axlearn.common.param_init import FanAxes, Initializer, Shape, constant_initializer, uniform
from axlearn.common.ssm_kernels.mamba_kernels import compute_mamba_scan
from axlearn.common.ssm_kernels.ssd_kernels import (
    ssd,
    ssd_linear_scan_w_hidden_states,
    ssd_linear_scan_w_timestep,
)
from axlearn.common.utils import Nested, Tensor, TensorSpec, with_sharding_constraint


class MambaDtProjInitializer(Initializer):
    """Initializes the weight and bias of a Linear layer as described in the Mamba paper."""

    @config_class
    class Config(Initializer.Config):
        """Configures MambaDtProjInitializer.

        All defaults are from the Mamba code:
        https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba_simple.py#L32
        """

        # Rank of dt before it is projected up to inner_dim.
        dt_rank: Required[int] = REQUIRED
        # Initialization stddev is set to `dt_scale` * 1/sqrt{dt_rank} when random.
        dt_scale: float = 1.0
        # Minimum value of the dt projection's bias after applying softplus.
        dt_min: float = 1e-3
        # Maximum value of the dt projection's bias after applying softplus.
        dt_max: float = 1e-2
        # Clamp dt projection's bias to at least this value.
        dt_init_floor: float = 1e-4
        # One of 'random' or 'constant'.
        # If 'constant', the projection matrix is initialized to a constant; otherwise, random. # pylint: disable=C0301
        mode: str = "random"

    def initialize(
        self,
        name: str,
        *,
        prng_key: Tensor,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> Tensor:
        """Initializes the Mamba dt projection matrix and bias following the official
        implementation."""
        cfg = self.config
        assert cfg.dt_rank > 0, "`dt_rank` must be positive."
        assert cfg.dt_min < cfg.dt_max, "`dt_min` must be < `dt_max`."
        # Initialize projection to preserve variance at initialization; see the Mamba paper.
        if "weight" in name:
            dt_init_std = cfg.dt_rank**-0.5 * cfg.dt_scale
            if cfg.mode == "constant":
                return constant_initializer(dt_init_std).initialize(
                    name,
                    shape=shape,
                    dtype=dtype,
                    prng_key=prng_key,
                )
            elif cfg.mode == "random":
                return uniform(scale=dt_init_std, dtype=dtype)(prng_key, shape)
            else:
                raise ValueError(
                    f"{self.__class__.__name__}'s `mode` must either be 'constant' or 'random'."
                )
        # Initialize bias so that softplus(bias) is between dt_min and dt_max; see the Mamba paper.
        elif "bias" in name:
            dt = jnp.exp(
                uniform(scale=1.0, dtype=dtype)(prng_key, shape)
                * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
                + math.log(cfg.dt_min)
            )
            dt = jnp.clip(dt, a_min=cfg.dt_init_floor)
            # Get inverse of softplus.
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt
        raise ValueError(
            f"{self.__class__.__name__} expects to initialize only weights and biases; "
            f"received '{name}'."
        )


class MambaLogAInitializer(Initializer):
    """Initializes Mamba's log A parameter with the S4D real initialization."""

    def initialize(
        self,
        name: str,
        *,
        prng_key: Tensor,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        """Returns a [state_dim, inner_dim] shaped matrix where each column contains
        log([1, 2, ..., state_dim])."""
        return jnp.log(
            jnp.tile(jnp.arange(1, shape[0] + 1, dtype=jnp.float32)[:, None], (1, shape[1]))
        )


@unique
class MambaRecurrenceOutputMode(Enum):
    """Defines what a Mamba recurrence implementation returns."""

    OUTPUTS = 0  # Return step outputs, but not corresponding hidden states.
    OUTPUTS_AND_STATES = 1  # Return step outputs and corresponding hidden states.


class BaseMambaRecurrence(BaseLayer):
    """An abstract class representing a layer that computes the Mamba recurrence
    from 'continuous' parameters after discretizing them."""

    class Output(NamedTuple):
        data: Tensor  # [batch, target_length, inner_dim]
        states: Tensor  # [batch, target_length, state_dim, inner_dim]

    @config_class
    class Config(BaseLayer.Config):
        """Configures a BaseMambaRecurrence."""

        output_mode: MambaRecurrenceOutputMode = MambaRecurrenceOutputMode.OUTPUTS

    def discretize_parameters(
        self,
        x: Tensor,
        *,
        a: Tensor,
        b: Tensor,
        delta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Computes discretization of SSM parameters, as implemented in Mamba.

        Args:
            x: [batch_size, seq_len, inner_dim]
            a: [state_dim, inner_dim]
            b: [batch_size, seq_len, state_dim]
            delta: [batch_size, seq_len, inner_dim]

        Returns:
            A Tensor of shape [batch_size, seq_len, state_dim, inner_dim] representing
                the multiplicative discretized parameters.
            A Tensor of shape [batch_size, seq_len, state_dim, inner_dim] representing
                the additive discretized parameters.
        """
        # Compute ZOH discretization of a: a_bar = exp(delta a); see Eqn (4) in the Mamba paper.
        # a_bar is [batch_size, seq_len, state_dim, inner_dim].
        a_bar = jnp.exp(
            _at_least_float32(jnp.expand_dims(delta, axis=-2)) * jnp.expand_dims(a, axis=(0, 1))
        )
        # Compute the simplified Euler discretization of b.
        # pylint: disable-next=line-too-long
        # See https://github.com/johnma2006/mamba-minimal/blob/03de542a36d873f6e6c4057ad687278cc6ae944d/model.py#L307.
        # The resulting b_bar is [batch_size, seq_len, state_dim, inner_dim].
        # TODO(swiseman): consider ZOH discretization here as well; see Eqn (4) in the Mamba paper.
        b_bar = jnp.expand_dims(delta, axis=-2) * jnp.expand_dims(b, axis=-1)
        b_bar_x = b_bar * jnp.expand_dims(x, axis=-2)
        return a_bar, b_bar_x

    def forward(
        self, x: Tensor, *, a: Tensor, b: Tensor, c: Tensor, delta: Tensor, d: Tensor
    ) -> Output:
        """Computes the Mamba recurrence output given full-sequence inputs and parameters.

        Args:
            x: [batch_size, seq_len, inner_dim]
            a: [batch_size, seq_len, state_dim, inner_dim]
            b: [batch_size, seq_len, state_dim, inner_dim]
            c: [batch_size, seq_len, state_dim]
            delta: [batch_size, seq_len, inner_dim]
            d: [1, inner_dim]

        Returns:
            An instance of BaseMambaRecurrence.Output.
        """
        raise NotImplementedError(type(self))


class LinearScanMambaRecurrence(BaseMambaRecurrence):
    """A layer that computes the Mamba recurrence with a simple linear scan."""

    def _scan(
        self,
        x: Tensor,
        *,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
    ) -> BaseMambaRecurrence.Output:
        """Computes the recurrence: h_t = a_t * h_{t-1} + b_t; y_t = c_T @ h_t + d_t * x_t
        using a simple scan.

        Args:
            x: [batch_size, seq_len, inner_dim]
            a: [batch_size, seq_len, state_dim, inner_dim]
            b: [batch_size, seq_len, state_dim, inner_dim]
            c: [batch_size, seq_len, state_dim]
            d: [1, inner_dim]

        Returns:
            An instance of BaseMambaRecurrence.Output.
        """
        cfg = self.config

        def advance(
            carry: Nested[Tensor], a_b: tuple[Nested[Tensor]]
        ) -> tuple[Nested[Tensor], Nested[Tensor]]:
            """Updates the SSM state given the previous state.

            Args:
                carry: [batch_size, state_dim, inner_dim]
                a_b: A tuple of [batch_size, state_dim, inner_dim] Tensors.

            Returns:
                A [batch_size, state_dim, inner_dim] as the carry.
                A [batch_size, state_dim, inner_dim] as the output.
            """
            a_i, b_i = a_b  # pytype: disable=bad-unpacking
            new_state = a_i * carry + b_i
            return new_state, new_state

        # jax.lax.scan scans over the leading axis, so we swap axes below.
        # TODO(swiseman): benchmark vmap alternative.
        _, h = jax.lax.scan(  # [seq_len, batch_size, state_dim, inner_dim]
            advance,
            jnp.zeros_like(a[:, 0]),
            (jnp.swapaxes(a, 0, 1), jnp.swapaxes(b, 0, 1)),
        )
        h = jnp.swapaxes(h, 0, 1).astype(x.dtype)
        y = jnp.einsum("bts,btsd->btd", c, h) + jnp.expand_dims(d, axis=0) * x
        states = h if cfg.output_mode == MambaRecurrenceOutputMode.OUTPUTS_AND_STATES else None
        return BaseMambaRecurrence.Output(data=y, states=states)

    def forward(
        self, x: Tensor, *, a: Tensor, b: Tensor, c: Tensor, delta: Tensor, d: Tensor
    ) -> BaseMambaRecurrence.Output:
        # Checkpoint with the `dots_with_no_batch_dims_saveable` policy, a reasonable default that
        # performs at least as well as any other policy in benchmarking.
        @functools.partial(
            jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable
        )
        def _materialize_recurrence(inputs: Tensor) -> Tensor:
            """Materializes per-step parameters and corresponding states. We separate this
            function to allow easy remat.

            Args:
                inputs: Tensor of shape [batch_size, seq_len, inner_dim]

            Returns:
                A Tensor of shape [batch_size, seq_len, inner_dim] representing the output
                    of the Mamba recurrence.
            """
            # Create input-dependent SSM parameters.
            a_bar, b_bar_x = self.discretize_parameters(inputs, a=a, b=b, delta=delta)
            recurrence_output = self._scan(
                inputs,
                a=a_bar,
                b=b_bar_x,
                c=c,
                d=d,
            )
            return recurrence_output

        return _materialize_recurrence(x)


class AssociativeScanMambaRecurrence(LinearScanMambaRecurrence):
    """A layer that computes the Mamba recurrence with an associative scan."""

    def _scan(
        self,
        x: Tensor,
        *,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
    ) -> BaseMambaRecurrence.Output:
        """Computes the recurrence: h_t = a_t * h_{t-1} + b_t; y_t = c_t @ h_t + d_t * x_t
        using an associative scan.

        Args:
            x: [batch_size, seq_len, inner_dim]
            a: [batch_size, seq_len, state_dim, inner_dim]
            b: [batch_size, seq_len, state_dim, inner_dim]
            c: [batch_size, seq_len, state_dim]
            d: [1, inner_dim]

        Returns:
            An instance of BaseMambaRecurrence.Output.
        """
        cfg = self.config

        def advance(left: Nested[Tensor], right: Nested[Tensor]) -> Nested[Tensor]:
            """Implements the following associative binary operation on pairs of Tensors, where
            we denote the first Tensor in the pair by a and the second by b:
                f((a_l, b_l), (a_r, b_r)) = (a_r*a_l, a_r*b_l+b_r).
            This operation has the property that the second element of the pair produced by
                f((a_T, b_T), f((a_{T-1}, b_{T-1}), f( ...  f((a_2, b_2), (a_1, b_1)))))
            is equal to the T-th element of the recurrence h_t = a_t * h_{t-1} + b_t, with h_0 = 0.
            Computing recurrences this way in the context of SSMs was proposed in the S5 paper:
            https://arxiv.org/abs/2208.04933. See Appendix A.

            Args:
                left: A dict mapping 'a' and 'b' keys (resp.) to Tensors of size
                    [batch_size, state_dim, inner_dim].
                right: A dict mapping 'a' and 'b' keys (resp.) to Tensors of size
                    [batch_size, state_dim, inner_dim].

            Returns:
                A dict with the same structure as `left` and `right`.
            """
            a_l, b_l = left["a"], left["b"]
            a_r, b_r = right["a"], right["b"]
            return {"a": a_r * a_l, "b": a_r * b_l + b_r}

        # Obtain a dict where keys 'a' and 'b' map (resp.) to Tensors with shape
        # [batch_size, seq_len, state_dim, inner_dim].
        terms = jax.lax.associative_scan(advance, {"a": a, "b": b}, axis=1)
        h = terms["b"].astype(x.dtype)  # [batch_size, seq_len, state_dim, inner_dim]
        y = jnp.einsum("bts,btsd->btd", c, h) + jnp.expand_dims(d, axis=0) * x
        states = h if cfg.output_mode == MambaRecurrenceOutputMode.OUTPUTS_AND_STATES else None
        return BaseMambaRecurrence.Output(data=y, states=states)


class PallasLinearScanMambaRecurrence(BaseMambaRecurrence):
    """A layer that computes the Mamba recurrence with a Pallas-based linear scan."""

    @config_class
    class Config(BaseMambaRecurrence.Config):
        """Configures a PallasLinearScanMambaRecurrence."""

        # Size of tiling along sequence dimension in the Pallas kernel; the default value has
        # been tuned on a tpu-v5p.
        seq_tile_size: int = 128
        # Size of tiling along 'inner' dimension in the Pallas kernel; the default value has
        # been tuned on a tpu-v5p.
        dim_tile_size: int = 512
        # A mapping from the dimensions of a Mamba input to its PartitionSpec.
        mamba_dim_to_partition_spec: dict[str, PartitionSpec] = {
            "btd": PartitionSpec(None),
            "sd": PartitionSpec(None),
            "bts": PartitionSpec(None),
            "1d": PartitionSpec(None),
        }
        # A PartitionSpec for the recurrence output.
        output_partition_spec: PartitionSpec = PartitionSpec(None)

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        assert (
            cfg.output_mode == MambaRecurrenceOutputMode.OUTPUTS
        ), f"{self.__class__} requires `output_mode` to be `MambaRecurrenceOutputMode.OUTPUTS`."

    def forward(
        self, x: Tensor, *, a: Tensor, b: Tensor, c: Tensor, delta: Tensor, d: Tensor
    ) -> BaseMambaRecurrence.Output:
        cfg = self.config

        # We need to jit a function before shard_mapping it.
        @jax.jit
        def jit_mamba_scan(x, a, b, c, delta, d):
            y = compute_mamba_scan(  # [batch_size, seq_len, inner_dim]
                x,
                a,
                b,
                c,
                delta,
                d,
                seq_tile_size=cfg.seq_tile_size,
                dim_tile_size=cfg.dim_tile_size,
            )
            return y

        partition_specs = cfg.mamba_dim_to_partition_spec
        partitioned_mamba_scan = shard_map(
            jit_mamba_scan,
            mesh=thread_resources.env.physical_mesh,
            in_specs=(
                partition_specs["btd"],
                partition_specs["sd"],
                partition_specs["bts"],
                partition_specs["bts"],
                partition_specs["btd"],
                partition_specs["1d"],
            ),
            out_specs=cfg.output_partition_spec,
            check_rep=False,
        )
        # Enforce sharding constraints on input and output.
        x = with_sharding_constraint(x, partition_specs["btd"])
        a = with_sharding_constraint(a, partition_specs["sd"])
        b = with_sharding_constraint(b, partition_specs["bts"])
        c = with_sharding_constraint(c, partition_specs["bts"])
        delta = with_sharding_constraint(delta, partition_specs["btd"])
        d = with_sharding_constraint(d, partition_specs["1d"])
        y = with_sharding_constraint(
            partitioned_mamba_scan(x, a, b, c, delta, d),
            cfg.output_partition_spec,
        )
        # The Pallas kernel does not return states.
        return BaseMambaRecurrence.Output(data=y, states=None)


def default_mamba_dim_to_partition_specs(
    mesh_axis_names: Sequence[str],
) -> dict[str, PartitionSpec]:
    """Builds a default mapping from tensor dims to partition specs for shard_mapping
    the Pallas-based Mamba implementation.

    The inner dimension is sharded over the default tensor-parallel axis name if present,
    and the the batch is sharded over the remainder of the axes.

    Args:
        mesh_axis_names: Mesh axis names.

    Returns:
        A dictionary keyed by Mamba tensor dims with partition spec values.
    """
    batch_axis_names = tuple(el for el in mesh_axis_names if el != "model")
    tp_axis_name = "model" if "model" in mesh_axis_names else None

    # TODO(swiseman): support sequence parallelism.
    x_spec = PartitionSpec(batch_axis_names, None, tp_axis_name)
    a_spec = PartitionSpec(None, tp_axis_name)
    b_spec = PartitionSpec(batch_axis_names, None, None)
    d_spec = PartitionSpec(None, tp_axis_name)
    partition_specs = {"btd": x_spec, "sd": a_spec, "bts": b_spec, "1d": d_spec}
    return partition_specs


def default_output_partition_spec(
    mesh_axis_names: Sequence[str],
) -> dict[str, PartitionSpec]:
    """Builds a default output partition spec for the shard_mapped Pallas-based Mamba
    implementation.

    The inner dimension is sharded over the default tensor-parallel axis name if present,
    and the the batch is sharded over the remainder of the axes.

    Args:
        mesh_axis_names: Mesh axis names.

    Returns:
        A PartitionSpec.
    """
    batch_axis_names = tuple(el for el in mesh_axis_names if el != "model")
    tp_axis_name = "model" if "model" in mesh_axis_names else None
    # TODO(swiseman): support sequence parallelism.
    return PartitionSpec(batch_axis_names, None, tp_axis_name)


def _at_least_float32(x: Tensor) -> Tensor:
    """Casts a Tensor of type bfloat16 or float16 to float32; Tensors of larger types are unchanged.

    Args:
        x: A Tensor of any shape.

    Returns:
        A Tensor of the same shape as x.
    """
    if x.dtype in [jnp.bfloat16, jnp.float16]:
        return x.astype(jnp.float32)
    return x


class MambaMixerLayer(BaseLayer):
    """A layer that computes the Mamba recurrence over its input.

    Can be substituted for a MultiheadAttention layer.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures a MambaMixerLayer."""

        # `d_model` in the Mamba code.
        input_dim: Required[int] = REQUIRED
        # `d_state` in the original Mamba code.
        state_dim: Required[int] = REQUIRED
        # Rank of dt before up-projection.
        dt_rank: Union[int, str] = "auto"
        input_proj: MultiLinear.Config = MultiLinear.default_config().set(
            bias=False,
            param_partition_spec=(None, None, "model"),
        )
        # A causal convolution. The window defaults to 4, as in the Mamba code:
        # https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba_simple.py#L36
        conv: Conv1D.Config = Conv1D.default_config().set(
            window=4,
            bias=True,
            param_partition_spec=(None, None, "model"),
        )
        x_proj: Linear.Config = Linear.default_config().set(
            bias=False, param_partition_spec=("model", None)
        )
        dt_proj: Linear.Config = Linear.default_config().set(
            bias=True, param_partition_spec=(None, "model")
        )
        out_proj: Linear.Config = Linear.default_config().set(
            bias=False,
            param_partition_spec=(None, "model"),  # TODO(swiseman): investigate.
        )
        param_partition_spec: Nested[PartitionSpec] = (None, "model")
        # `input_dim`-dimensional inputs are projected up by `expansion_factor` before the
        # short convolution and recurrence are applied. Defaults to 2 as in the Mamba code:
        # https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba_simple.py#L37
        expansion_factor: float = 2.0
        cache_dtype: Optional[jnp.dtype] = None
        # The recurrence implementation to use for full-sequence inputs.
        mamba_recurrence: BaseMambaRecurrence = LinearScanMambaRecurrence.default_config()
        # The recurrence implementation to use for inference.
        inference_mamba_recurrence: BaseMambaRecurrence = (
            LinearScanMambaRecurrence.default_config().set(
                output_mode=MambaRecurrenceOutputMode.OUTPUTS_AND_STATES
            )
        )

    class MambaOutput(NamedTuple):
        data: Tensor  # [batch, target_length, input_dim]
        conv_input: Tensor  # [batch, target_length, inner_dim]
        states: Tensor  # [batch, target_length, state_dim, inner_dim]

    class SSMParameters(NamedTuple):
        a: Tensor  # [batch_size, state_dim, inner_dim]
        b: Tensor  # [batch_size, seq_len, state_dim]
        c: Tensor  # [batch_size, seq_len, state_dim]
        delta: Tensor  # [batch_size, seq_len, inner_dim]
        d: Tensor  # [1, inner_dim]

    @property
    def inner_dim(self):
        cfg = self.config
        return int(cfg.input_dim * cfg.expansion_factor)

    @property
    def output_dim(self):
        cfg = self.config
        return cfg.input_dim

    @property
    def dt_rank(self):
        cfg = self.config
        return math.ceil(cfg.input_dim / 16) if cfg.dt_rank == "auto" else cfg.dt_rank

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "input_proj",
            cfg.input_proj.set(
                input_dim=cfg.input_dim,
                num_outputs=2,
                output_dim=self.inner_dim,
            ),
        )
        self._add_child(
            "conv",
            cfg.conv.set(
                padding=((cfg.conv.window - 1, 0),),  # A causal convolution.
                input_dim=self.inner_dim,
                output_dim=self.inner_dim,
                num_input_dim_groups=self.inner_dim,
            ),
        )
        self._add_child(
            "x_proj",
            cfg.x_proj.set(
                input_dim=self.inner_dim,
                output_dim=self.dt_rank + cfg.state_dim * 2,
            ),
        )
        self._add_child(
            "dt_proj",
            cfg.dt_proj.set(
                input_dim=self.dt_rank,
                output_dim=self.inner_dim,
                bias=True,
                param_init=MambaDtProjInitializer.default_config().set(dt_rank=self.dt_rank),
            ),
        )
        self._add_child(
            "out_proj",
            cfg.out_proj.set(
                input_dim=self.inner_dim,
                output_dim=cfg.input_dim,
            ),
        )
        self._add_child("recurrence", cfg.mamba_recurrence)
        self._add_child(
            "inference_recurrence",
            cfg.inference_mamba_recurrence.set(
                output_mode=MambaRecurrenceOutputMode.OUTPUTS_AND_STATES
            ),
        )

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        """Creates log_a and d parameter specs.

        Returns:
            A dict mapping `log_a` and `d` to their respective ParameterSpecs.
        """
        cfg = self.config
        params = dict(
            log_a=ParameterSpec(
                shape=(cfg.state_dim, self.inner_dim),
                mesh_axes=cfg.param_partition_spec,
                initializer=MambaLogAInitializer.default_config().instantiate(),
                dtype=cfg.dtype,
                weight_decay_scale=0.0,
            ),
            d=ParameterSpec(
                shape=(1, self.inner_dim),
                mesh_axes=(None, cfg.param_partition_spec[1]),
                initializer=constant_initializer(1.0),
                dtype=cfg.dtype,
                weight_decay_scale=0.0,
            ),
        )
        return params

    def _project_input(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Projects inputs into Tensors with dimension inner_dim.

        Args:
            inputs: [batch_size, seq_len, input_dim]

        Returns:
            A Tensor of shape [batch_size, seq_len, inner_dim] representing a projected input.
            A Tensor of shape [batch_size, seq_len, inner_dim] representing a projected input
               for the residual connection.
        """
        xz = self.input_proj(inputs)  # [batch_size, length, 2, inner_dim]
        x, z = jnp.split(xz, (1,), axis=-2)  # TODO(swiseman): note split not free.
        x, z = jnp.squeeze(x, axis=-2), jnp.squeeze(z, axis=-2)
        return x, z

    def _ssm_parameters(self, inputs: Tensor) -> SSMParameters:
        """Computes input-dependent SSM parameters, as defined in the Mamba paper.

        Args:
            inputs: [batch_size, seq_len, inner_dim]

        Returns:
            An instance of SSMParameters.
        """
        cfg = self.config
        x_dbl = self.x_proj(inputs)  # [batch_size, seq_len, dt_rank + state_dim*2]
        dt, b, c = jnp.split(
            x_dbl,
            (
                self.dt_rank,
                self.dt_rank + cfg.state_dim,
            ),
            axis=-1,
        )
        delta = jax.nn.softplus(self.dt_proj(dt))  # [batch_size, seq_len, inner_dim]
        a = -jnp.exp(_at_least_float32(self.parameters["log_a"])).astype(inputs.dtype)
        return MambaMixerLayer.SSMParameters(a=a, b=b, c=c, delta=delta, d=self.parameters["d"])

    def _output_from_states(self, inputs: Tensor, *, res: Tensor) -> Tensor:
        """Projects recurrence output back to input dimension.

        Args:
            inputs: [batch_size, seq_len, inner_dim]
            res: [batch_size, seq_len, inner_dim]

        Returns:
            A Tensor of shape [batch_size, seq_len, input_dim] representing the output of the
                Mamba layer.
        """
        y = inputs * jax.nn.silu(res)
        return self.out_proj(y).astype(res.dtype)

    def _full_sequence_forward(
        self, inputs: Tensor, *, recurrence: BaseMambaRecurrence
    ) -> MambaOutput:
        """Computes the Mamba layer output from a full sequence of inputs.

        Args:
            inputs: A Tensor of shape [batch_size, seq_len, input_dim].
            recurrence: A BaseMambaRecurrence to use for computing the recurrence.

        Returns:
            A MambaOutput.
        """
        conv_input, res = self._project_input(inputs)
        conv_states = jax.nn.silu(self.conv(conv_input))
        # Get "continuous" ssm parameters.
        a, b, c, delta, d = self._ssm_parameters(conv_states)
        recurrence_output = recurrence(conv_states, a=a, b=b, c=c, delta=delta, d=d)
        output = self._output_from_states(recurrence_output.data, res=res)
        return MambaMixerLayer.MambaOutput(
            data=output, conv_input=conv_input, states=recurrence_output.states
        )

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        query: Tensor,
        cached_states: Optional[Nested[Tensor]] = None,
    ) -> tuple[Optional[Nested[Tensor]], Tensor]:
        """Computes MambaMixerLayer outputs.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted.
            query: A Tensor of shape [batch, target_length, target_dim].
            cached_states: Optional Nested Tensor as produced by `init_states`.

        Returns:
            An optional NestedTensor of cached states, depending on `mode`.
            A MambaOutput instance, where .data is of the same shape as `query`.

        Raises:
            ValueError: If `mode` is unsupported.
        """
        self.vlog(3, "mamba.input=%s", query.sum())
        if mode == ForwardMode.FORWARD:
            mamba_state, mamba_output = (
                None,
                self._full_sequence_forward(query, recurrence=self.recurrence),
            )
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            mamba_state, mamba_output = self.init_states(
                time_step=cached_states["mamba_layer"],
                query=query,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            mamba_state, mamba_output = self.extend_step(cached_states["mamba_layer"], query)
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        self.vlog(3, "mamba.output=%s", mamba_output.data.sum())
        return dict(mamba_layer=mamba_state), mamba_output

    def forward(self, query: Tensor) -> MambaOutput:
        """Computes the Mamba recurrence over the provided query.

        Args:
            query: [batch, target_length, target_dim]

        Returns:
            A MambaOutput instance where .data is the same shape as `query`.
        """
        _, output = self._forward_for_mode(mode=ForwardMode.FORWARD, query=query)
        return output

    def init_states(
        self,
        *,
        time_step: Optional[Tensor],
        query: Union[Tensor, TensorSpec],
    ) -> tuple[Nested[Tensor], Optional[MambaOutput]]:
        """Initializes cache for autoregressive cached decoding.

        The method supports initializing an empty cache as well as prefilling:
        * To initialize an empty cache, specify `time_step=None`.
            In this case, `query` is allowed to be a TensorSpec.
        * To prefill, provide `time_step` and `query` as Tensors.

        Args:
            time_step: An optional Tensor of shape [batch_size]. Each value is an index into the
                length dimension indicating where decoding will start from.
            query: A Tensor or TensorSpec of shape [batch, target_length, target_dim] corresponding
                to query vector up to `time_step` indices. For batch index `i`, only
                `query[i, :time_step[i], ...]` will affect subsequent decoding.

        Returns:
            A tuple (init_states, output):
            * init_states: A Nested Tensor containing the cached convolution input, ssm state,
                and updated time_step.
            * output: In the prefill case, a MambaOutput instance where .data is the same shape as
                query. Otherwise, if initializing cache from scratch, output will be None.
        """
        cfg: MambaMixerLayer.Config = self.config
        dtype = cfg.cache_dtype or cfg.dtype
        batch_size = query.shape[0]

        if time_step is None:
            init_state = dict(
                conv_input=jnp.zeros((batch_size, cfg.conv.window, self.inner_dim), dtype=dtype),
                state=jnp.zeros((batch_size, 1, cfg.state_dim, self.inner_dim), dtype=dtype),
                time_step=jnp.zeros(batch_size, dtype=jnp.int32),
            )
            return init_state, None

        output = self._full_sequence_forward(query, recurrence=self.inference_recurrence)
        conv_input, states = output.conv_input, output.states
        # Pad conv input so we can take the last window timesteps that precede time_step.
        padded_conv_input = jnp.pad(
            conv_input, ((0, 0), (cfg.conv.window, 0), (0, 0))
        )  # [batch_size, target_length+window, input_dim]
        batch_range = jnp.arange(batch_size)
        time_step_range = time_step[:, None] + jnp.arange(cfg.conv.window)
        conv_input_cache = padded_conv_input[batch_range[:, None], time_step_range]
        # Pad states so we can take the step preceding time_step, even if time_step is zero.
        padded_states = jnp.pad(states, ((0, 0), (1, 0), (0, 0), (0, 0)))
        state = padded_states[batch_range, time_step]
        init_state = dict(
            conv_input=conv_input_cache.astype(dtype),
            state=jnp.expand_dims(state, axis=1).astype(dtype),
            time_step=time_step,
        )
        return init_state, output

    def _conv_update(
        self,
        inputs: Tensor,
        *,
        cached_conv_input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Updates cache of convolutional inputs and returns updated state.

        Args:
            input: [batch, inner_dim]
            cached_conv_input: [batch, width, inner_dim]
            weight: [width, 1, inner_dim]
            bias: [inner_dim]

        Returns:
            A Tensor of shape [batch, inner_dim].
            A Tensor of shape [batch, width, inner_dim], representing the new cache.
        """
        # TODO(swiseman): investigate optimizing this further.
        new_cache = jnp.roll(cached_conv_input, shift=-1, axis=1)
        new_cache = new_cache.at[:, -1].set(inputs)
        # Compute the update in float32 to prevent divergence from the forward implementation.
        conv_state = jnp.sum(
            new_cache * jnp.squeeze(_at_least_float32(weight), axis=1), axis=1
        ).astype(
            inputs.dtype
        )  # [batch, inner_dim]
        if bias is not None:
            conv_state = conv_state + bias
        return conv_state, new_cache

    def _single_step_ssm_update(
        self,
        inputs: Tensor,
        *,
        prev_state: Tensor,
        a_bar: Tensor,
        b_bar_x: Tensor,
        c: Tensor,
        d: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Moves the SSM state forward by a single step.

        Args:
            inputs: [batch_size, 1, inner_dim]
            prev_state: [batch_size, 1, state_dim, inner_dim]
            a_bar: [batch_size, 1, state_dim, inner_dim]
            b_bar_x: [batch_size, 1, state_dim, inner_dim]
            c: [batch_size, 1, state_dim]
            d: [1, inner_dim]

        Returns:
            A Tensor of shape [batch_size, 1, inner_dim] representing the new output.
            A Tensor of shape [batch_size, 1, state_dim, inner_dim] representing the updated state.
        """
        new_state = a_bar * prev_state + b_bar_x
        new_state = new_state.astype(inputs.dtype)
        y = jnp.einsum("bts,btsd->btd", c, new_state) + jnp.expand_dims(d, axis=0) * inputs
        return y, new_state

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        query: Tensor,
    ) -> tuple[Nested[Tensor], MambaOutput]:
        """Computes the next state given the query of the current step. This function is used
        in autoregressive decoding.

        Args:
            cached_states: A Nested[Tensor] containing previous state of shape and index.
            query: Tensor of shape [batch, 1, inner_dim]

        Returns:
            A Nested[Tensor] of convolutional input, current state, and updated timestep.
            A MambaOutput instance, where .data is the same shape as query.
        """
        time_step: Tensor = cached_states["time_step"]
        assert time_step.ndim == 1

        conv_input, res = self._project_input(query)
        conv_state, updated_conv_input = self._conv_update(
            jnp.squeeze(conv_input, axis=1),
            cached_conv_input=cached_states["conv_input"],
            weight=self.parameters["conv"]["weight"],
            bias=self.parameters["conv"]["bias"],
        )
        conv_state = jnp.expand_dims(jax.nn.silu(conv_state), axis=1)  # [batch_size, 1, inner_dim]
        # Create input-dependent SSM parameters.
        a, b, c, delta, d = self._ssm_parameters(conv_state)
        a_bar, b_bar_x = self.inference_recurrence.discretize_parameters(
            conv_state,
            a=a,
            b=b,
            delta=delta,
        )
        # Do single step update.
        y, updated_state = self._single_step_ssm_update(
            conv_state,
            prev_state=cached_states["state"],
            a_bar=a_bar,
            b_bar_x=b_bar_x,
            c=c,
            d=d,
        )
        output = self._output_from_states(y, res=res)
        updated_state = dict(
            conv_input=updated_conv_input,
            state=updated_state,
            time_step=time_step + 1,
        )
        return updated_state, MambaMixerLayer.MambaOutput(
            data=output,
            conv_input=conv_input,
            states=updated_state,
        )


class JambaMixerLayer(MambaMixerLayer):
    """A Jamba-style Mamba layer, which norms the input-dependent SSM parameters."""

    @config_class
    class Config(MambaMixerLayer.Config):
        dt_norm: InstantiableConfig = RMSNorm.default_config()
        b_norm: InstantiableConfig = RMSNorm.default_config()
        c_norm: InstantiableConfig = RMSNorm.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("dt_norm", cfg.dt_norm.set(input_dim=self.dt_rank))
        self._add_child("b_norm", cfg.b_norm.set(input_dim=cfg.state_dim))
        self._add_child("c_norm", cfg.c_norm.set(input_dim=cfg.state_dim))

    def _ssm_parameters(self, inputs: Tensor) -> MambaMixerLayer.SSMParameters:
        """Computes layer-normed versions of the input-dependent SSM parameters.

        Args:
            inputs: [batch_size, seq_len, inner_dim]

        Returns:
            An instance of MambaMixerLayer.SSMParameters.
        """
        cfg = self.config
        x_dbl = self.x_proj(inputs)  # [batch_size, seq_len, dt_rank, state_dim*2]
        dt, b, c = jnp.split(
            x_dbl,
            (
                self.dt_rank,
                self.dt_rank + cfg.state_dim,
            ),
            axis=-1,
        )
        dt, b, c = self.dt_norm(dt), self.b_norm(b), self.c_norm(c)
        delta = jax.nn.softplus(self.dt_proj(dt))  # [batch_size, seq_len, inner_dim]
        a = -jnp.exp(_at_least_float32(self.parameters["log_a"])).astype(inputs.dtype)
        return MambaMixerLayer.SSMParameters(a=a, b=b, c=c, delta=delta, d=self.parameters["d"])


class BaseSSMLayer(BaseLayer):
    """An abstract class representing SSM layers.

    Can be substituted for a BaseTransformerLayer.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures a BaseSSMLayer."""

        input_dim: Required[int] = REQUIRED
        state_dim: Required[int] = REQUIRED

    def _to_transformer_output(self, data: Tensor) -> BaseTransformerLayer.Output:
        """Creates a BaseTransformerLayer.Output from an SSM output."""
        return BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=None,
            self_attention_kv_state=None,
            cross_attention_probs=None,
        )

    def forward(
        self,
        data: Tensor,
        **_kwargs,
    ) -> BaseTransformerLayer.Output:
        """Computes outputs given full-sequence inputs.
        See `axlearn.common.attention.BaseTransformerLayer`

        Args:
            data: A Tensor of shape [batch, target_length, input_dim].

        Returns:
            BaseTransformerLayer.Output.
        """
        raise NotImplementedError(type(self))

    def init_states(
        self,
        *,
        time_step: Optional[Tensor],
        data: Union[Tensor, TensorSpec],
        **_kwargs,
    ):
        """Initializes cached states for incremental computation.

        The method supports initializing an empty cache as well as prefilling:
        * To initialize an empty cache, specify `time_step=None`.
            In this case, `data` is allowed to be a TensorSpec.
        * To prefill, provide `time_step` and `data` as Tensors.

        Args:
            time_step: An optional Tensor of shape [batch]. Each value is an index into the length
                dimension indicating where decoding will start from.
            data: A Tensor or TensorSpec of shape [batch, target_length, input_dim]. For batch index
                `i`, only `data[i, :time_step[i], ...]` will affect subsequent decoding.

        Returns:
            A tuple (init_states, output):
            * init_states: A nested tree of Tensors, which can be used as `cached_states` for the
                initial call of `extend_step()`.
            * output: In the prefill case, a BaseTransformerLayer.Output instance, where .data is of
                the same shape as `data`. Otherwise, if initializing cache from scratch, output will
                be None.
        """
        raise NotImplementedError(type(self))

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        data: Tensor,
        **_kwargs,
    ) -> tuple[Nested[Tensor], BaseTransformerLayer.Output]:
        """Computes incremental outputs.

        Args:
            cached_states: A NestedTensor returned by `init_states()` or a previous invocation of
                `extend_step()`.
            data: A Tensor of shape [target_batch_size, target_step_length, input_dim], where
                `target_step_length` is usually 1. For self-attention, `data` will be used as the
                `query` sequence and will be appended to key and value sequences.

        Returns:
            (updated_cached_states, output), where:
            `updated_cached_states` represents the new cached states incorporating `data`;
            `output` represents the output for the given input data. `output.data` will have the
            same shape as the input data.
        """
        raise NotImplementedError(type(self))


@unique
class BlockResidualMode(Enum):
    """Defines how residual additions are computed in MambaBlock layers."""

    # Cast residual to float32 before adding; this is the default Mamba configuration:
    # https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/models/config_mamba.py#L15
    FP32 = 0
    # Do not cast residual to float32 before adding.
    NOCAST = 1


class MambaBlock(BaseSSMLayer):
    """A MambaMixer layer with RMS normalization and a skip connection.

    Can be substituted for a TransformerLayer.
    """

    @config_class
    class Config(BaseSSMLayer.Config):
        """Configures a Mamba block."""

        norm: InstantiableConfig = RMSNorm.default_config()
        mamba_layer: BaseLayer.Config = MambaMixerLayer.default_config()
        residual_mode: BlockResidualMode = BlockResidualMode.FP32

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))
        self._add_child(
            "mamba", cfg.mamba_layer.set(input_dim=cfg.input_dim, state_dim=cfg.state_dim)
        )

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Union[Tensor, TensorSpec],
        cached_states: Optional[Nested[Tensor]] = None,
        **_kwargs,
    ) -> tuple[Optional[Nested[Tensor]], BaseTransformerLayer.Output]:
        """Computes the standard Mamba block including residual connection over
        the input data.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted.
                  See `axlearn.common.attention.ForwardMode` for details.
            data: A Tensor of shape [batch, target_length, target_dim].
            cached_states: Optional Nested Tensor as produced by `init_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as data.

        Raises:
            ValueError: If `mode` is unsupported.
        """
        cfg: MambaBlock.Config = self.config

        def mamba_thunk(target):
            if mode == ForwardMode.FORWARD:
                state, output = None, self.mamba(query=target)
            elif mode == ForwardMode.INIT_STATES:
                assert cached_states is not None
                state, output = self.mamba.init_states(
                    time_step=cached_states["mamba_block"],
                    query=target,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert cached_states is not None
                state, output = self.mamba.extend_step(
                    cached_states["mamba_block"],
                    target,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")
            return state, output

        if mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            if cached_states["mamba_block"] is None:
                state, _ = mamba_thunk(data)
                return dict(mamba_block=state), None

        skip_input = data
        if cfg.residual_mode == BlockResidualMode.FP32:
            skip_input = _at_least_float32(skip_input)
        target = self.norm(data)
        state, output = mamba_thunk(target)
        output = (output.data + skip_input).astype(target.dtype)
        output = self._to_transformer_output(data=output)

        return dict(mamba_block=state), output

    def forward(
        self,
        data,
        **_kwargs,
    ) -> BaseTransformerLayer.Output:
        """Computes the standard Mamba block including residual connection over
        the input data.

        Args:
            data: A Tensor of shape [batch, target_length, target_dim].

        Returns:
            An Output instance, where .data is of the same shape as data.
        """
        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            data=data,
            cached_states=None,
        )
        return output

    def init_states(
        self,
        *,
        time_step: Optional[Tensor],
        data: Union[Tensor, TensorSpec],
        **_kwargs,
    ) -> tuple[Nested[Tensor], BaseTransformerLayer.Output]:
        """Initializes cache for autoregressive cached decoding.

        The method supports initializing an empty cache as well as prefilling:
        * To initialize an empty cache, specify `time_step=None`.
            In this case, `data` is allowed to be a TensorSpec.
        * To prefill, provide `time_step` and `data` as Tensors.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            data: A Tensor or TensorSpec of shape [batch, target_length, target_dim] corresponding
                to query vector at `time_step` indices. For batch index `i`, only
                `target[i, :time_step[i], ...]` will affect subsequent decoding.

        Returns:
            A tuple (init_states, output):
            * init_states: A Nested Tensor state depending on the `attention` layer implementation.
            * output: In the prefill case, an Output instance, where .data is of the same shape as
                data. Otherwise, if initializing cache from scratch, output will be None.
        """
        return self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            data=data,
            cached_states=dict(mamba_block=time_step),
        )

    def extend_step(
        self,
        cached_states: Nested[Tensor],
        data: Tensor,
        **_kwargs,
    ) -> tuple[Nested[Tensor], BaseTransformerLayer.Output]:
        """Computes incremental outputs.

        Args:
            cached_states: A Nested[Tensor] object containing Tensors which are the
                results of previous attentions, and index used for fast decoding. Contains
                "attention" cached states.
            data: Tensor of shape [batch_size, 1, target_dm] corresponding to query vector at index
                time_step.

        Returns:
            A `NestedTensor` state of key and value pair along with index updated at `time_step`.
            An Output instance, where .data is of the same shape as data.
        """
        return self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            data=data,
            cached_states=cached_states,
        )


class JambaMambaBlock(MambaBlock):
    """A Jamba-style block, which follows a Mamba layer with a feed-forward layer."""

    @config_class
    class Config(MambaBlock.Config):
        feed_forward: InstantiableConfig = TransformerFeedForwardLayer.default_config()

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        # Change default mamba_layer to a JambaMixerLayer.
        cfg.mamba_layer = JambaMixerLayer.default_config()
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("feed_forward", cfg.feed_forward.set(input_dim=cfg.input_dim))

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        data: Union[Tensor, TensorSpec],
        cached_states: Optional[Nested[Tensor]] = None,
        **_kwargs,
    ) -> tuple[Optional[Nested[Tensor]], BaseTransformerLayer.Output]:
        cfg = self.config

        def mamba_thunk(target):
            if mode == ForwardMode.FORWARD:
                state, output = None, self.mamba(query=target)
            elif mode == ForwardMode.INIT_STATES:
                assert cached_states is not None
                state, output = self.mamba.init_states(
                    time_step=cached_states["mamba_block"],
                    query=target,
                )
            elif mode == ForwardMode.EXTEND_STEP:
                assert cached_states is not None
                state, output = self.mamba.extend_step(
                    cached_states["mamba_block"],
                    target,
                )
            else:
                raise ValueError(f"Unrecognized mode {mode}.")
            return state, output

        # Handle the case where we initialize cache from scratch.
        # `data` can be effectively treated as a TensorSpec in this case, so norm doesn't apply.
        if mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            if cached_states["mamba_block"] is None:
                state, _ = mamba_thunk(TensorSpec(shape=data.shape, dtype=data.dtype))
                return dict(mamba_block=state), None

        skip_input = data
        if cfg.residual_mode == BlockResidualMode.FP32:
            skip_input = _at_least_float32(skip_input)
        target = self.norm(data)
        state, output = mamba_thunk(target)
        data = (output.data + skip_input).astype(target.dtype)
        output = self.feed_forward(data).astype(target.dtype)  # Feed-forward norms its input.
        return dict(mamba_block=state), self._to_transformer_output(data=output)


def set_double_shard_weights_config_mamba(
    cfg: Union[MambaBlock.Config, Sequence[MambaBlock.Config]],
    *,
    batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"),
    fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
    tp_axis_names: Union[str, Sequence[str]] = "model",
    seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, seq_axis_names, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, seq_axis_names, tp_axis_names)

    # pytype: disable=attribute-error
    def set_mamba_partition_specs(mamba_layer: MambaMixerLayer.Config):
        # Shard weights.
        mamba_layer.input_proj.param_partition_spec = (fsdp_axis_names, None, tp_axis_names)
        mamba_layer.conv.param_partition_spec = (None, None, tp_axis_names)
        mamba_layer.x_proj.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        mamba_layer.dt_proj.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        # TODO(swiseman): benchmark.
        mamba_layer.out_proj.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Shard activations
        mamba_layer.x_proj.output_partition_spec = (batch_axis_names, seq_axis_names, tp_axis_names)
        mamba_layer.dt_proj.output_partition_spec = (
            batch_axis_names,
            seq_axis_names,
            tp_axis_names,
        )
        mamba_layer.out_proj.output_partition_spec = (
            batch_axis_names,
            seq_axis_names,
            tp_axis_names,
        )

    if not isinstance(cfg, Sequence):
        cfg = [cfg]

    for layer_cfg in cfg:
        set_mamba_partition_specs(layer_cfg.mamba_layer)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)
    # pytype: enable=attribute-error


class StackedSSMLayer(StackedTransformerLayer):
    """Overrides StackedTransformerLayer to expect BaseSSMLayer layers."""

    @config_class
    class Config(StackedTransformerLayer.Config):
        """Configures StackedSSMLayer."""

        layer: BaseSSMLayer = MambaBlock.default_config()


class RepeatedSSMLayer(RepeatedTransformerLayer):
    """Overrides RepeatedTransformerLayer to expect BaseSSMLayer layers."""

    @config_class
    class Config(RepeatedTransformerLayer.Config):
        """Configures RepeatedSSMLayer."""

        layer: BaseSSMLayer = MambaBlock.default_config()


class StackedMixedSSMTransformerLayer(StackedTransformerLayer):
    """Convenience class for specifying Jamba-style mixed-layer models."""

    @config_class
    class Config(StackedTransformerLayer.Config):
        """Configures StackedMixedSSMTransformerLayer.

        Configuration based on:
        https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/configuration_jamba.py
        """

        # Configuration allows for specifying a stack of layers that decomposes into contiguous
        # 'periods' of layers. Within each period, exactly one layer is a transformer layer,
        # and the remaining layers are SSM layers.
        #
        # For example, setting `num_layers`=9, `transformer_layer_period`=3, and
        # `transformer_layer_offset`=1 will result in the following sequence of layers:
        #                               S T S S T S S T S,
        # where 'S' represents an SSM layer and 'T' represents a transformer layer.

        # The number of contiguous layers out of which to make exactly one a transformer layer.
        transformer_layer_period: Required[int] = REQUIRED
        # The offset within the `transformer_layer_period` layers to make a transformer layer.
        transformer_layer_offset: Required[int] = REQUIRED
        ssm_layer: BaseSSMLayer = JambaMambaBlock.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        layers = [
            (
                cfg.layer
                if i % cfg.transformer_layer_period == cfg.transformer_layer_offset
                else cfg.ssm_layer
            )
            for i in range(cfg.num_layers)
        ]
        super().__init__(cfg.set(layer=layers), parent=parent)


# Naming convention for Mamba2:
#   * `SSD` is used to denote any kernel-specific parameters/functions (consistent with the kernel),
#   * `Mamba2`` (where SSD is a sub-module) is used to denote layer-level parameters/functions.


class SSDdtBiasInitializer(Initializer):
    """Initializes the bias of the dt projection in the SSD layer of Mamba2.

    The weight matrix of the dt projection is seperately constructed and initialized.
    """

    @config_class
    class Config(Initializer.Config):
        """Configures SSDdtBiasInitializer.

        The initialization is different from Mamba1 in that there is no low-rank parameterization.
        and we only need to initialize the bias term.

        Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py.
        """

        # Initialization stddev is set to `dt_scale` * 1/sqrt{dt_rank} when random.
        dt_scale: float = 1.0
        # Minimum value of the dt projection's bias after applying softplus.
        dt_min: float = 1e-3
        # Maximum value of the dt projection's bias after applying softplus.
        dt_max: float = 1e-1
        # Clamp dt projection's bias to at least this value.
        dt_init_floor: float = 1e-4

    def initialize(
        self,
        name: str,
        *,
        prng_key: Tensor,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> Tensor:
        """Initializes the SSD dt projection bias following the official implementation."""
        if axes is not None:
            raise ValueError("SSDdtBiasInitializer does not support FanAxes.")
        cfg = self.config
        assert 0 < cfg.dt_min < cfg.dt_max, "`dt_min` must be < `dt_max`."
        dt = jnp.exp(
            uniform(scale=1.0, dtype=dtype)(prng_key, shape)
            * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).astype(
            dtype
        )  # math.log may return float64, so we need to cast to dtype
        dt = jnp.clip(dt, a_min=cfg.dt_init_floor)
        # Get inverse of softplus.
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        return inv_dt


class SSDLLogAInitializer(Initializer):
    """Initializes SSD's log-log A parameter, a = exp(-exp(llog_a))."""

    @config_class
    class Config(Initializer.Config):
        """Configures SSDLLogAInitializer.

        Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py.
        """

        # `A` will be initialized within the range of [a_min, a_max], usually not tuned.
        a_min: int = 1
        a_max: int = 16

    def initialize(
        self,
        name: str,
        *,
        prng_key: Tensor,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        """Returns a [num_heads] shaped vector."""
        if axes is not None:
            raise ValueError("SSDLLogAInitializer does not support FanAxes.")

        cfg = self.config
        return jnp.log(
            jax.random.uniform(prng_key, shape, dtype=dtype, minval=cfg.a_min, maxval=cfg.a_max)
        )


class BaseSSDRecurrence(BaseLayer):
    """An abstract class representing a layer that computes the SSD recurrence."""

    class Output(NamedTuple):
        """Defines the output of the SSD recurrence."""

        data: Tensor  # [batch, num_heads, target_length, head_dim]
        states: Tensor  # [batch, num_heads, target_length, state_dim, head_dim]

    @config_class
    class Config(BaseLayer.Config):
        """Configures a BaseSSDRecurrence."""

        output_mode: MambaRecurrenceOutputMode = MambaRecurrenceOutputMode.OUTPUTS

    def forward(
        self, x: Tensor, *, log_a: Tensor, b: Tensor, c: Tensor, delta: Tensor, d: Tensor
    ) -> Output:
        """Computes the Mamba2's SSD recurrence output given full-sequence inputs and parameters.

        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
            log_a: [num_heads]
            b: [batch_size, num_groups, seq_len, state_dim]
            c: [batch_size, num_groups, seq_len, state_dim]
            delta: [batch_size, num_heads, seq_len]
            d: [head_dim]

        Returns:
            An instance of BaseSSDRecurrence.Output.
        """
        raise NotImplementedError(type(self))


class PallasSSDRecurrence(BaseSSDRecurrence):
    """A layer that computes the Mamba2's SSD recurrence with a Pallas-based chunk-wise scan."""

    @config_class
    class Config(BaseSSDRecurrence.Config):
        """Configures a PallasSSDRecurrence."""

        mamba2_dim_to_partition_spec: dict[str, PartitionSpec] = {
            "bhtd": PartitionSpec(None),
            "bht": PartitionSpec(None),
        }

        output_partition_spec: PartitionSpec = PartitionSpec(None)

    def forward(
        self, x: Tensor, *, log_a: Tensor, b: Tensor, c: Tensor, delta: Tensor, d: Tensor
    ) -> BaseSSDRecurrence.Output:
        """Computes Mamba2's SSD recurrence with a Pallas-based chunk-wise scan.

        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
            log_a: [1, num_heads, 1]
            b: [batch_size, num_groups, seq_len, state_dim]
            c: [batch_size, num_groups, seq_len, state_dim]
            delta: [batch_size, num_heads, seq_len]
            d: [1, num_heads, 1, 1]

        Returns:
            An BaseSSDRecurrence.Output instance, where .data is the same shape as x and .states is
            None (no need to return hidden states during training).

        Unlike the Mamba recurrence, discretizations of parameters are not explicitly computed.
        More specifically, \bar a (i.e., discretized a) is computed outside the kernel whereas
        \bar b is computed implicitly via adding the delta term to the input
            x -- \bar x = x * delta.
        See the following line from the official repo for details -
        https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/ssd_minimal.py#L103

        Note that `ssd` functions need to be wrapped, otherwise the following error will be raised:
            ``NotImplementedError: Mosaic kernels cannot be automatically partitioned.``
        The current version of `ssd` function assumes that h0 is None, so there is no need to
        provide its partition spec.
        """
        cfg = self.config

        sharded_ssd = shard_map(
            ssd,
            mesh=thread_resources.env.physical_mesh,
            in_specs=(
                cfg.mamba2_dim_to_partition_spec["bhtd"],
                cfg.mamba2_dim_to_partition_spec["bhtd"],
                cfg.mamba2_dim_to_partition_spec["bhtd"],
                cfg.mamba2_dim_to_partition_spec["bht"],
            ),
            out_specs=cfg.output_partition_spec,
            check_rep=False,
        )
        # The kernel code `ssd_kernels.py` uses q/k/v notations, which corresponds to b/c/x.
        x_bar = x * jnp.expand_dims(delta, axis=-1)
        loga_bar = log_a * delta
        o = sharded_ssd(c, b, x_bar, loga_bar)

        o = o + d * x
        return BaseSSDRecurrence.Output(data=o, states=None)


class LinearScanSSDRecurrence(BaseSSDRecurrence):
    """A layer that computes the Mamba2's SSD recurrence with a Jax-based linear scan."""

    def forward(
        self,
        x: Tensor,
        *,
        log_a: Tensor,
        b: Tensor,
        c: Tensor,
        delta: Tensor,
        d: Tensor,
        time_step: Optional[Tensor] = None,
    ) -> BaseSSDRecurrence.Output:
        """Computes the Mamba2's SSD recurrence with a Jax-based linear scan.

        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
            log_a: [1, num_heads, 1]
            b: [batch_size, num_groups, seq_len, state_dim]
            c: [batch_size, num_groups, seq_len, state_dim]
            delta: [batch_size, num_heads, seq_len]
            d: [1, num_heads, 1, 1]
            time_step: [batch_size] or None

        Returns:
            An BaseSSDRecurrence.Output instance, where .data is the same shape as x and .states is
            the hidden states of shape [batch_size, num_heads, seq_len, state_dim, head_dim] for
            the given time step if `time_step` is not None, otherwise the full hidden states of
            shape [batch_size, num_heads, seq_len, state_dim, head_dim] is returned.
        """
        # Same procedure as the pallas version above.
        x_bar = x * jnp.expand_dims(delta, axis=-1)
        loga_bar = log_a * delta

        if time_step is None:
            # Return the full hidden states.
            o, states = ssd_linear_scan_w_hidden_states(c, b, x_bar, loga_bar)
        else:
            # Return the hidden states at the given time step.
            o, states = ssd_linear_scan_w_timestep(c, b, x_bar, loga_bar, time_step)

        o = o + d * x
        return BaseSSDRecurrence.Output(data=o, states=states)


class Mamba2MixerLayer(BaseLayer):
    """A layer that computes the Mamba2 recurrence over its input."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures a Mamba2MixerLayer."""

        # `d_model` increases as models get larger.
        input_dim: Required[int] = REQUIRED
        # `d_state` typically in {64, 128}
        state_dim: Required[int] = REQUIRED
        # num_heads = input_dim // head_dim, head_dim is typically 128.
        num_heads: Required[int] = REQUIRED

        # `G` in the paper, typically 8
        num_groups: Required[int] = REQUIRED

        # See sec 8.2 for the parameterization. More details (e.g., conv
        # for bc projection) can be found in the following link:
        # https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba2.py # pylint: disable=C0301

        xz_proj: MultiLinear.Config = MultiLinear.default_config().set(
            bias=False,
            param_partition_spec=(None, None, "model"),
        )

        bc_proj: MultiLinear.Config = MultiLinear.default_config().set(
            bias=False,
            param_partition_spec=(None, None, "model"),
        )
        # A causal convolution. The window defaults to 4, the same as mamba1.
        x_conv: Conv1D.Config = Conv1D.default_config().set(
            window=4,
            bias=True,
            param_partition_spec=(None, None, "model"),
        )
        b_conv: Conv1D.Config = Conv1D.default_config().set(
            window=4,
            bias=True,
            param_partition_spec=(None, None, "model"),
        )
        c_conv: Conv1D.Config = Conv1D.default_config().set(
            window=4,
            bias=True,
            param_partition_spec=(None, None, "model"),
        )

        # `dt_bias` is separately created and initialized.
        dt_proj: Linear.Config = Linear.default_config().set(
            bias=False, param_partition_spec=(None, "model")
        )
        pre_out_proj_norm: InstantiableConfig = GroupNorm.default_config().set(
            norm_type=NormType.RMSNORM,
            norm_axes=-1,
        )
        out_proj: Linear.Config = Linear.default_config().set(
            bias=False,
            param_partition_spec=("model", None),
        )

        expansion_factor: float = 2.0
        cache_dtype: Optional[jnp.dtype] = None
        bc_norm: Optional[InstantiableConfig] = RMSNorm.default_config()
        norm_eps: float = 1e-5
        norm_dtype: Optional[jnp.dtype] = None

        # The recurrence implementation to use for full-sequence inputs.
        ssd_recurrence: BaseSSDRecurrence = PallasSSDRecurrence.default_config()
        # The recurrence implementation to use for inference.
        inference_mamba_recurrence: BaseSSDRecurrence = (
            LinearScanSSDRecurrence.default_config().set(
                output_mode=MambaRecurrenceOutputMode.OUTPUTS_AND_STATES
            )
        )

    class Mamba2Output(NamedTuple):
        """Defines the output of the Mamba2MixerLayer."""

        data: Tensor  # [batch, num_heads, target_length, head_dim]
        ssd_state: Tensor  # [batch, num_heads, state_dim, head_dim]

    class SSDParameters(NamedTuple):
        """Defines the parameters of the SSD recurrence."""

        log_a: Tensor  # [1, num_heads, 1]
        b: Tensor  # [batch_size, num_groups, seq_len, state_dim]
        c: Tensor  # [batch_size, num_groups, seq_len, state_dim]
        delta: Tensor  # [batch_size, num_heads, seq_len]
        d: Tensor  # [1, num_heads, 1, 1]

    # Cache used for internal inference, whereas Mamba2Output is external output.
    class Mamba2Cache(NamedTuple):
        """Defines the cache of the Mamba2MixerLayer for inference."""

        # Naming is a bit different from Mamba1: conv_input -> conv_state.
        x_conv_state: Tensor  # [batch_size, seq_len, inner_dim]
        b_conv_state: Tensor  # [batch_size, seq_len, state_dim * 2]
        c_conv_state: Tensor  # [batch_size, seq_len, state_dim * 2]
        ssd_state: Tensor  # [batch_size, num_heads, state_dim, head_dim]
        time_step: Optional[Tensor] = None  # [batch]

    @property
    def inner_dim(self):
        cfg = self.config
        return int(cfg.input_dim * cfg.expansion_factor)

    @property
    def head_dim(self):
        cfg = self.config
        return self.inner_dim // cfg.num_heads

    @property
    def output_dim(self):
        cfg = self.config
        return cfg.input_dim

    @property
    def group_dim(self):
        cfg = self.config
        return self.inner_dim // cfg.num_groups

    @property
    def bc_state_dim(self):
        cfg = self.config
        return cfg.state_dim * cfg.num_groups

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child(
            "xz_proj",
            cfg.xz_proj.set(
                input_dim=cfg.input_dim,
                num_outputs=2,
                output_dim=self.inner_dim,
                bias=False,
            ),
        )
        self._add_child(
            "bc_proj",
            cfg.bc_proj.set(
                input_dim=cfg.input_dim,
                num_outputs=2,
                output_dim=self.bc_state_dim,
                bias=False,
            ),
        )
        self._add_child(
            "x_conv",
            cfg.x_conv.set(
                padding=((cfg.x_conv.window - 1, 0),),  # A causal convolution.
                input_dim=self.inner_dim,
                output_dim=self.inner_dim,
                num_input_dim_groups=self.inner_dim,
            ),
        )
        self._add_child(
            "b_conv",
            cfg.b_conv.set(
                padding=((cfg.b_conv.window - 1, 0),),  # A causal convolution.
                input_dim=self.bc_state_dim,
                output_dim=self.bc_state_dim,
                num_input_dim_groups=self.bc_state_dim,
            ),
        )
        self._add_child(
            "c_conv",
            cfg.c_conv.set(
                padding=((cfg.c_conv.window - 1, 0),),  # A causal convolution.
                input_dim=self.bc_state_dim,
                output_dim=self.bc_state_dim,
                num_input_dim_groups=self.bc_state_dim,
            ),
        )

        # b/c norm is analoguous to q/k norm in standard attention.
        if cfg.bc_norm:
            self._add_child(
                "b_norm",
                cfg.bc_norm.clone().set(
                    input_dim=cfg.state_dim, eps=cfg.norm_eps, forward_dtype=cfg.norm_dtype
                ),
            )
            self._add_child(
                "c_norm",
                cfg.bc_norm.clone().set(
                    input_dim=cfg.state_dim, eps=cfg.norm_eps, forward_dtype=cfg.norm_dtype
                ),
            )

        self._add_child(
            "dt_proj",
            cfg.dt_proj.set(
                input_dim=cfg.input_dim,
                output_dim=cfg.num_heads,
                bias=False,
            ),
        )
        self._add_child(
            "pre_out_proj_norm",
            cfg.pre_out_proj_norm.set(
                input_dim=self.inner_dim, num_groups=cfg.num_groups, eps=cfg.norm_eps
            ),
        )
        self._add_child(
            "out_proj",
            cfg.out_proj.set(
                input_dim=self.inner_dim,
                output_dim=cfg.input_dim,
                bias=False,
            ),
        )

        self._add_child("recurrence", cfg.ssd_recurrence)
        self._add_child(
            "inference_recurrence",
            cfg.inference_mamba_recurrence.set(
                output_mode=MambaRecurrenceOutputMode.OUTPUTS_AND_STATES
            ),
        )

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        """Creates parameter specs.

        Returns:
            A dict mapping `llog_a`, `dt_bias` and `d` to their respective ParameterSpecs.
        """
        cfg = self.config
        params = dict(
            llog_a=ParameterSpec(
                # Initialize with a shape that avoids expansion later.
                shape=(1, cfg.num_heads, 1),
                mesh_axes=(None, "model", None),
                initializer=SSDLLogAInitializer.default_config().instantiate(),
                dtype=cfg.dtype,
                weight_decay_scale=0.0,
            ),
            dt_bias=ParameterSpec(
                shape=(cfg.num_heads,),
                mesh_axes=("model",),
                initializer=SSDdtBiasInitializer.default_config().instantiate(),
                dtype=cfg.dtype,
                weight_decay_scale=0.0,
            ),
            d=ParameterSpec(
                # Initialize with a shape that avoids expansion later.
                shape=(1, cfg.num_heads, 1, 1),
                mesh_axes=(None, "model", None, None),
                initializer=constant_initializer(1.0),
                dtype=cfg.dtype,
                weight_decay_scale=0.0,
            ),
        )
        return params

    def _project_input(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Projects inputs into tensors with dimension inner_dim.

        Args:
            inputs: [batch_size, seq_len, input_dim]

        Returns:
            x, z of the same size [batch_size, seq_len, inner_dim]
        """
        xz = self.xz_proj(inputs)
        x, z = jnp.split(xz, 2, axis=-2)  # [batch_size, seq_len, 1, inner_dim]
        return jnp.squeeze(x, axis=2), jnp.squeeze(z, axis=2)

    def _ssm_parameters(
        self,
        inputs: Tensor,
        b_input: Optional[Tensor] = None,
        c_input: Optional[Tensor] = None,
    ) -> SSDParameters:
        """Computes input-dependent SSD parameters.

        Args:
            inputs: [batch_size, seq_len, inner_dim]
            b_input: [batch_size, seq_len,  bc_state_dim]. If b_input and c_input
                are given, no need to compute bc_proj.
            c_input: [batch_size, seq_len,  bc_state_dim]. If b_input and c_input
                are given, no need to compute bc_proj.

        Exposing the computation of `b` and `c` is useful to keep track the conv1d input for
        `b_conv` and `c_conv`. During training, `b_input` and `c_input` should be None as
        they represent the results after short conv. During inference, they represent the
        input of short conv.

        Returns:
            An instance of SSMParameters.

        Raises:
            ValueError: If only one of b_input and c_input is provided.

        TODO (bailin-wang): merge b_conv and c_conv for better efficiency.
        """
        cfg = self.config
        if (b_input is None) != (c_input is None):
            raise ValueError("Either both or none of b_input and c_input should be provided.")

        if b_input is None or c_input is None:
            bc = self.bc_proj(inputs)  # [batch_size, seq_len, 2, bc_state_dim]
            bc = rearrange(bc, "b s n d -> b s (n d)")
            b, c = jnp.split(bc, 2, axis=-1)
        else:
            b = b_input
            c = c_input

        b = jax.nn.silu(self.b_conv(b))
        c = jax.nn.silu(self.c_conv(c))

        b = rearrange(b, "b s (g d) -> b g s d", d=cfg.state_dim)
        c = rearrange(c, "b s (g d) -> b g s d", d=cfg.state_dim)

        if "b_norm" in self.children and "c_norm" in self.children:
            b = self.b_norm(b)
            c = self.c_norm(c)

        # `dt` is in float32 for better precision of softplus for the delta term which later will
        # be combined with float32 `log_a`. See also the following link:
        # https://github.com/state-spaces/mamba/blob/6b72c122713bb769cc82c6b8e6d019c53d27d6a1/mamba_ssm/ops/triton/ssd_combined.py#L603.
        dt = self.dt_proj(inputs) + jnp.expand_dims(
            _at_least_float32(self.parameters["dt_bias"]), axis=(0, 1)
        )
        delta = jax.nn.softplus(dt)  # [batch_size, seq_len, num_heads]
        delta = rearrange(delta, "b s h -> b h s")  # [batch_size, num_heads, seq_len]

        log_a = -jnp.exp(
            _at_least_float32(self.parameters["llog_a"])
        )  # a = exp(-exp(llog_a)), log_a = -exp(llog_a * delta)

        return Mamba2MixerLayer.SSDParameters(
            log_a=log_a, b=b, c=c, delta=delta, d=self.parameters["d"]
        )

    def _output_from_states(self, inputs: Tensor, *, z: Tensor) -> Tensor:
        """Projects recurrence output back to input dimension.

        Args:
            inputs: [batch_size, num_heads, seq_len, head_dim]
            z: [batch_size, num_heads, seq_len, head_dim]

        Returns:
            A tensor of shape [batch_size, seq_len, input_dim]

        Note that the num_heads/num_groups dim is contracted in the output.
        """
        cfg = self.config
        y = inputs * jax.nn.silu(z)
        y_for_gnorm = rearrange(y, "b nh l d -> b l (nh d)", nh=cfg.num_heads)
        y_for_proj = self.pre_out_proj_norm(y_for_gnorm)
        return self.out_proj(y_for_proj)

    def forward(self, query: Tensor) -> Mamba2Output:
        """Computes the Mamba2 recurrence over the provided inputs.

        Args:
            query: [batch_size, input_length, input_dim]

        Returns:
            A Mamba2Output instance where .data is the same shape as `inputs`.
        """
        _, output = self._forward_for_mode(mode=ForwardMode.FORWARD, query=query)
        return output

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        query: Tensor,
        cache: Optional[Mamba2Cache] = None,
    ) -> tuple[Optional[Nested[Tensor]], Tensor]:
        """Computes MambaMixerLayer outputs.

        Args:
            mode: {FORWARD, INIT_STATES, EXTEND_STEP}
            query: A Tensor of shape [batch_size, seq_len, input_dim]
            cache: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional cache, depending on `mode`.
            A Mamba2Output instance, where .data is of the same shape as `inputs`.

        Raises:
            ValueError: If `mode` is unsupported.
        """
        self.vlog(3, "mamba2.input=%s", query.sum())
        if mode == ForwardMode.FORWARD:
            mamba_cache, mamba_output = self._full_sequence_forward(
                query, recurrence=self.recurrence
            )
        elif mode == ForwardMode.INIT_STATES:
            assert cache is not None
            mamba_cache, mamba_output = self.prefill_states(
                time_step=cache,
                query=query,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cache is not None
            mamba_cache, mamba_output = self.extend_step(cache, query)
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        self.vlog(3, "mamba2.output=%s", mamba_output.data.sum())
        return dict(mamba_layer=mamba_cache), mamba_output

    def _full_sequence_forward(
        self, inputs: Tensor, *, recurrence: BaseSSDRecurrence
    ) -> tuple[Optional[Mamba2Cache], Mamba2Output]:
        """Computes the Mamba2 layer output from a full sequence of inputs.

        Args:
            inputs: A tensor of shape [batch_size, seq_len, input_dim].
            recurrence: A BaseMambaRecurrence to use for computing the recurrence.

        Returns:
            An optional Mamba2Cache instance. Currently, it is always None.
            A Mamba2Output instance.
        """
        cfg = self.config

        x, z = self._project_input(inputs)
        x_conv = jax.nn.silu(self.x_conv(x))
        x_conv_w_head = rearrange(x_conv, "b s (h d) -> b h s d", d=self.head_dim)
        z_w_head = rearrange(z, "b s (h d) -> b h s d", d=self.head_dim)

        log_a, b, c, delta, d = self._ssm_parameters(inputs)
        recurrence_output = recurrence(x_conv_w_head, log_a=log_a, b=b, c=c, delta=delta, d=d)
        output = self._output_from_states(recurrence_output.data, z=z_w_head)

        ssd_state = recurrence_output.states
        if ssd_state is not None:
            ssd_state = ssd_state.astype(cfg.cache_dtype)

        mamba_cache = None
        mamba_output = Mamba2MixerLayer.Mamba2Output(data=output, ssd_state=ssd_state)
        return mamba_cache, mamba_output

    # pylint: disable=unused-argument
    def init_states(self, *, target_batch_size: int, target_max_len: int) -> Mamba2Cache:
        """Initializes cache for autoregressive cached decoding.

        Args:
            batch_size: The batch size of the target to be decoded.
            target_max_len: The maximum length of the target to be decoded.

        Returns:
            A Mamba2Cache instance.
        """
        cfg = self.config
        dtype = cfg.cache_dtype or cfg.dtype
        cache = Mamba2MixerLayer.Mamba2Cache(
            x_conv_state=jnp.zeros(
                (target_batch_size, cfg.x_conv.window, self.inner_dim), dtype=dtype
            ),
            b_conv_state=jnp.zeros(
                (target_batch_size, cfg.b_conv.window, self.bc_state_dim), dtype=dtype
            ),
            c_conv_state=jnp.zeros(
                (target_batch_size, cfg.c_conv.window, self.bc_state_dim), dtype=dtype
            ),
            ssd_state=jnp.zeros(
                (target_batch_size, cfg.num_heads, cfg.state_dim, self.head_dim), dtype=dtype
            ),
            time_step=jnp.zeros(target_batch_size, dtype=jnp.int32),
        )
        return cache

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        query: Tensor,
    ) -> tuple[Mamba2Cache, Mamba2Output]:
        """Initializes cache for autoregressive cached decoding. It refines the mamba state
        returned from `_full_sequence_forward` to the state at `time_step` for the
        incremental decoding later.

        Args:
            time_step: A Tensor of shape [batch_size]. Each value is an index into the length
                dimension indicating where decoding will start from.
            query: Tensor of shape [batch_size, target_length, target_dim] corresponding to input
                vector up to `time_step` indices. For batch index `i`, only
                `inputs[i, :time_step[i], ...]` will affect subsequent decoding.

        Returns:
            A Mamba2Cache instance containing updated convolution state, ssm state and time_step.
            A Mamba2Output instance where .data is the same shape as query.
        """
        cfg = self.config
        cache_dtype = cfg.cache_dtype or cfg.dtype

        x, z = self._project_input(query)
        x_conv = jax.nn.silu(self.x_conv(x))
        x_conv_w_head = rearrange(x_conv, "b s (h d) -> b h s d", d=self.head_dim)
        z_w_head = rearrange(z, "b s (h d) -> b h s d", d=self.head_dim)

        # Run `bc_proj` outside of `_ssm_parameters` so that we can keep track of the conv1d input.
        bc_input = self.bc_proj(query)  # [batch_size, seq_len, 2, bc_state_dim]
        bc_input = rearrange(bc_input, "b s n d -> b s (n d)")
        b_input, c_input = jnp.split(bc_input, 2, axis=-1)
        log_a, b, c, delta, d = self._ssm_parameters(query, b_input=b_input, c_input=c_input)

        recurrence_output = self.inference_recurrence(
            x_conv_w_head, log_a=log_a, b=b, c=c, delta=delta, d=d, time_step=time_step
        )
        output = self._output_from_states(recurrence_output.data, z=z_w_head)
        mamba_output = Mamba2MixerLayer.Mamba2Output(
            data=output, ssd_state=recurrence_output.states.astype(cache_dtype)
        )

        # Collect and refine conv states and ssd states.
        x_conv_state = x
        b_conv_state = b_input
        c_conv_state = c_input

        # For the full sequence, always in float32, will be down-cast based on cache_dtype.
        cont_ssd_state = recurrence_output.states.astype(cache_dtype)

        batch_size = query.shape[0]
        batch_range = jnp.arange(batch_size)

        # Pad conv input so we can take the last window timesteps that precede time_step.
        x_time_step_range = time_step[:, None] + jnp.arange(cfg.x_conv.window)[None, :]
        padded_x_conv_state = jnp.pad(
            x_conv_state, ((0, 0), (cfg.x_conv.window, 0), (0, 0))
        )  # [batch_size, target_length+window, input_dim]
        cont_x_conv_state = padded_x_conv_state[batch_range[:, None], x_time_step_range]

        b_time_step_range = time_step[:, None] + jnp.arange(cfg.b_conv.window)
        padded_b_conv_state = jnp.pad(b_conv_state, ((0, 0), (cfg.b_conv.window, 0), (0, 0)))
        cont_b_conv_state = padded_b_conv_state[batch_range[:, None], b_time_step_range]

        c_time_step_range = time_step[:, None] + jnp.arange(cfg.c_conv.window)
        padded_c_conv_state = jnp.pad(c_conv_state, ((0, 0), (cfg.c_conv.window, 0), (0, 0)))
        cont_c_conv_state = padded_c_conv_state[batch_range[:, None], c_time_step_range]

        init_cache = Mamba2MixerLayer.Mamba2Cache(
            x_conv_state=cont_x_conv_state.astype(cache_dtype),
            b_conv_state=cont_b_conv_state.astype(cache_dtype),
            c_conv_state=cont_c_conv_state.astype(cache_dtype),
            ssd_state=cont_ssd_state.astype(cache_dtype),
            time_step=time_step,
        )
        return init_cache, mamba_output

    def _single_step_conv_update(
        self,
        inputs: Tensor,
        *,
        conv_state: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Updates cache of convolutional inputs and returns updated state.

        Args:
            inputs: [batch_size, inner_dim]
            conv_state: [batch_size, width, inner_dim]
            weight: [width, 1, inner_dim]
            bias: [inner_dim]

        Returns:
            A tensor of shape [batch_size, inner_dim].
            A tensor of shape [batch_size, width, inner_dim], representing the new conv state.
        """
        new_conv_state = jnp.roll(conv_state, shift=-1, axis=1)
        new_conv_state = new_conv_state.at[:, -1].set(inputs)

        conv_output = jnp.sum(
            new_conv_state * jnp.squeeze(_at_least_float32(weight), axis=1), axis=1
        ).astype(
            inputs.dtype
        )  # [batch_size, inner_dim]
        if bias is not None:
            conv_output = conv_output + bias
        return conv_output, new_conv_state

    def _single_step_ssm_update(
        self,
        x: Tensor,
        *,
        ssm_state: Tensor,
        log_a: Tensor,
        b: Tensor,
        c: Tensor,
        d: Tensor,
        delta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Moves the SSM state forward by a single step.

        Args:
            x: [batch_size, num_heads, 1, head_dim]
            ssm_state: [batch_size, num_heads, state_dim, head_dim]
            log_a: [1, num_heads, 1], always float32
            b: [batch_size, num_groups, 1, state_dim]
            c: [batch_size, num_groups, 1, state_dim]
            delta: [batch_size, num_heads, 1], always float32
            d: [1, head_dim, 1, 1]

        Returns:
            A tensor of shape [batch_size, num_heads, 1, head_dim] for the new output.
            A tensor of shape [batch_size, num_heads, state_dim, head_dim] for the updated state.
        """
        cfg = self.config
        num_head_per_group = cfg.num_heads // cfg.num_groups

        orig_dtype = x.dtype
        acc_dtype = cfg.cache_dtype or cfg.dtype

        # x: [batch_size, num_heads, head_dim]
        # b and c: [batch_size, num_groups, state_dim]
        # d: [batch_size, num_heads]
        x, b, c, d = map(lambda x: jnp.squeeze(x, axis=2), (x, b, c, d))

        # [batch_size, num_heads, state_dim]
        b = repeat(b, "b ng d -> b (ng ngh) d", ngh=num_head_per_group)
        c = repeat(c, "b ng d -> b (ng ngh) d", ngh=num_head_per_group)

        # [batch_size, num_heads, head_dim]
        x_bar = x * delta
        # [batch_size, num_heads, 1]
        loga_bar = log_a * delta
        # [batch_size, num_heads, 1]
        a = jnp.exp(loga_bar)
        # [batch_size, num_heads, state_dim, head_dim]
        a = jnp.expand_dims(a, axis=-1)

        new_ssm_state = a * ssm_state + jnp.einsum("...i,...j->...ij", b, x_bar)
        output = jnp.einsum("...ij,...i->...j", new_ssm_state, c) + d * x

        output = jnp.expand_dims(output.astype(orig_dtype), axis=2)
        new_ssm_state = new_ssm_state.astype(acc_dtype)
        return output, new_ssm_state

    def extend_step(
        self,
        cache: Mamba2Cache,
        query: Tensor,
    ) -> tuple[Mamba2Cache, Mamba2Output]:
        """Computes the next state given the query of the current step. This function is used
        in autoregressive decoding.

        Args:
            cached_states: A Nested[Tensor] containing previous state of shape and index.
            query: Tensor of shape [batch_size, 1, inner_dim]

        Returns:
            A Mamba2Cache instance containing the convolution state, ssm state and time_step.
            A Mamba2Output instance, where .data is the same shape as query.
        """
        time_step: Tensor = cache.time_step
        assert time_step.ndim == 1
        cfg = self.config

        x, z = self._project_input(query)
        x_conv, new_x_conv_state = self._single_step_conv_update(
            jnp.squeeze(x, axis=1),
            conv_state=cache.x_conv_state,
            weight=self.parameters["x_conv"]["weight"],
            bias=self.parameters["x_conv"]["bias"],
        )
        x_conv = jnp.expand_dims(jax.nn.silu(x_conv), axis=1)  # [batch_size, 1, inner_dim]
        x_conv_w_head = rearrange(x_conv, "b s (h d) -> b h s d", d=self.head_dim)
        z_w_head = rearrange(z, "b s (h d) -> b h s d", d=self.head_dim)

        # Obtain ssm parameters.
        bc = self.bc_proj(query)  # [batch_size, seq_len, 2, bc_state_dim]
        bc = rearrange(bc, "b s n d -> b s (n d)")
        b, c = jnp.split(bc, 2, axis=-1)

        b_conv, new_b_conv_state = self._single_step_conv_update(
            jnp.squeeze(b, axis=1),
            conv_state=cache.b_conv_state,
            weight=self.parameters["b_conv"]["weight"],
            bias=self.parameters["b_conv"]["bias"],
        )
        b = jnp.expand_dims(jax.nn.silu(b_conv), axis=1)  # [batch_size, 1, bc_inner_dim]

        c_conv, new_c_conv_state = self._single_step_conv_update(
            jnp.squeeze(c, axis=1),
            conv_state=cache.c_conv_state,
            weight=self.parameters["c_conv"]["weight"],
            bias=self.parameters["c_conv"]["bias"],
        )
        c = jnp.expand_dims(jax.nn.silu(c_conv), axis=1)  # [batch_size, 1, bc_inner_dim]

        b = rearrange(b, "b s (g d) -> b g s d", d=cfg.state_dim)
        c = rearrange(c, "b s (g d) -> b g s d", d=cfg.state_dim)

        if cfg.bc_norm:
            b = self.b_norm(b)
            c = self.c_norm(c)

        dt = self.dt_proj(query) + jnp.expand_dims(
            _at_least_float32(self.parameters["dt_bias"]), axis=(0, 1)
        )
        delta = jax.nn.softplus(dt)  # [batch_size, 1, num_heads]
        delta = rearrange(delta, "b s h -> b h s")  # [batch_size, num_heads, 1]

        log_a = -jnp.exp(
            _at_least_float32(self.parameters["llog_a"])
        )  # a = exp(-exp(llog_a)), log_a = -exp(llog_a)
        d = self.parameters["d"]

        y, new_ssd_state = self._single_step_ssm_update(
            x_conv_w_head,
            ssm_state=cache.ssd_state,
            log_a=log_a,
            b=b,
            c=c,
            d=d,
            delta=delta,
        )
        output = self._output_from_states(y, z=z_w_head)

        new_cache = Mamba2MixerLayer.Mamba2Cache(
            x_conv_state=new_x_conv_state,
            b_conv_state=new_b_conv_state,
            c_conv_state=new_c_conv_state,
            ssd_state=new_ssd_state,
            time_step=time_step + 1,
        )
        mamba2output = Mamba2MixerLayer.Mamba2Output(
            data=output,
            ssd_state=new_ssd_state,
        )
        return new_cache, mamba2output


class JambaMamba2Block(JambaMambaBlock):
    """A JambaMamba2Block along with RMN norm and a feed-forward layer."""

    @config_class
    class Config(JambaMambaBlock.Config):
        """Configures a JambaMamba2Block."""

        num_heads: Required[int] = REQUIRED
        num_groups: Required[int] = REQUIRED

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        cfg.mamba_layer = Mamba2MixerLayer.default_config()
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        cfg.mamba_layer = cfg.mamba_layer.set(num_heads=cfg.num_heads, num_groups=cfg.num_groups)
        super().__init__(cfg, parent=parent)


def set_double_shard_weights_config_mamba2(
    cfg: Union[JambaMamba2Block.Config, Sequence[JambaMamba2Block.Config]],
    *,
    batch_axis_names: Union[str, Sequence[str]] = ("data", "expert", "fsdp"),
    fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
    tp_axis_names: Union[str, Sequence[str]] = "model",
    seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, seq_axis_names, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, seq_axis_names, tp_axis_names)

    def set_mamba2_partition_specs(mamba_layer: Mamba2MixerLayer.Config):
        mamba_layer.xz_proj.param_partition_spec = (fsdp_axis_names, None, tp_axis_names)
        mamba_layer.bc_proj.param_partition_spec = (fsdp_axis_names, None, tp_axis_names)
        mamba_layer.b_conv.param_partition_spec = (None, None, tp_axis_names)
        mamba_layer.c_conv.param_partition_spec = (None, None, tp_axis_names)
        mamba_layer.dt_proj.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        mamba_layer.out_proj.param_partition_spec = (tp_axis_names, fsdp_axis_names)

        mamba_layer.dt_proj.output_partition_spec = (
            batch_axis_names,
            seq_axis_names,
            tp_axis_names,
        )
        mamba_layer.out_proj.output_partition_spec = (
            batch_axis_names,
            seq_axis_names,
            tp_axis_names,
        )

    if not isinstance(cfg, Sequence):
        cfg = [cfg]

    for layer_cfg in cfg:
        set_mamba2_partition_specs(layer_cfg.mamba_layer)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)

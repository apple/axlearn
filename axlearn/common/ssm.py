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
from axlearn.common.layers import Conv1D, Linear, MultiLinear, RMSNorm
from axlearn.common.module import Module
from axlearn.common.param_init import FanAxes, Initializer, Shape, constant_initializer, uniform
from axlearn.common.ssm_kernels.mamba_kernels import compute_mamba_scan
from axlearn.common.utils import Nested, Tensor, with_sharding_constraint


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
        inference_mamba_recurrence: (
            BaseMambaRecurrence
        ) = LinearScanMambaRecurrence.default_config().set(
            output_mode=MambaRecurrenceOutputMode.OUTPUTS_AND_STATES
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
                padding=(cfg.conv.window - 1, 0),  # A causal convolution.
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
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cached states, depending on `mode`.
            A MambaOutput instance, where .data is of the same shape as `query`.

        Raises:
            ValueError: If `mode` is unsupported.
        """
        self.vlog(3, "mamba.input=%s", query.sum())
        if mode == ForwardMode.FORWARD:
            mamba_state, mamba_output = None, self._full_sequence_forward(
                query, recurrence=self.recurrence
            )
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            mamba_state, mamba_output = self.prefill_states(
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

    # pylint: disable=unused-argument
    def init_states(self, *, target_batch_size: int, **_kwargs) -> Nested[Tensor]:
        """Initializes cache for autoregressive cached decoding.

        Args:
            target_batch_size: The batch size of the target to be decoded.

        Returns:
            The cache as a Nested[Tensor].
        """
        cfg = self.config
        dtype = cfg.cache_dtype or cfg.dtype
        cache = dict(
            conv_input=jnp.zeros((target_batch_size, cfg.conv.window, self.inner_dim), dtype=dtype),
            state=jnp.zeros((target_batch_size, 1, cfg.state_dim, self.inner_dim), dtype=dtype),
            time_step=jnp.zeros(target_batch_size, dtype=jnp.int32),
        )
        return cache

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        query: Tensor,
    ) -> tuple[Nested[Tensor], MambaOutput]:
        """Initializes cache for autoregressive cached decoding.

        Args:
            time_step: A Tensor of shape [batch_size]. Each value is an index into the length
                dimension indicating where decoding will start from.
            query: Tensor of shape [batch, target_length, target_dim] corresponding to query vector
                up to `time_step` indices. For batch index `i`, only `query[i, :time_step[i], ...]`
                will affect subsequent decoding.

        Returns:
            A Nested[Tensor] containing the cached convolution input, ssm state,
            and updated time_step.
            A MambaOutput instance where .data is the same shape as query.
        """
        cfg = self.config
        dtype = cfg.cache_dtype or cfg.dtype
        output = self._full_sequence_forward(query, recurrence=self.inference_recurrence)
        conv_input, states = output.conv_input, output.states
        # Pad conv input so we can take the last window timesteps that precede time_step.
        padded_conv_input = jnp.pad(
            conv_input, ((0, 0), (cfg.conv.window, 0), (0, 0))
        )  # [batch_size, target_length+window, input_dim]
        batch_range = jnp.arange(conv_input.shape[0])
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

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> Nested[Tensor]:
        """Initializes cached states for incremental computation.

        Args:
            target_batch_size: The batch size for target sequences.
            target_max_len: The maximum number of tokens in a target sequence.

        Returns:
            A nested tree of Tensors, which can be used as `cached_states` for the initial call
            of `extend_step()`.
        """
        raise NotImplementedError(type(self))

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        data: Tensor,
        **_kwargs,
    ) -> tuple[Nested[Tensor], BaseTransformerLayer.Output]:
        """Initializes cached states for incremental computation.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            data: A Tensor of shape [batch, target_length, input_dim]. For batch index `i`, only
                `data[i, :time_step[i], ...]` will affect subsequent decoding.

        Returns:
            A nested tree of Tensors, which can be used as `cached_states` for the initial call
            of `extend_step()`.
            A BaseTransformerLayer.Output instance, where .data is of the same shape as `data`.
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
        mamba_layer: MambaMixerLayer.Config = MambaMixerLayer.default_config()
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
        data: Tensor,
        cached_states: Optional[Nested[Tensor]] = None,
        **_kwargs,
    ) -> tuple[Optional[Nested[Tensor]], BaseTransformerLayer.Output]:
        """Computes the standard Mamba block including residual connection over
        the input data.

        Args:
            mode: Configures whether `cached_states` are consumed or emitted.
                  See `axlearn.common.attention.ForwardMode` for details.
            data: A Tensor of shape [batch, target_length, target_dim].
            cached_states: Optional NestedTensor as produced by `prefill_states`.

        Returns:
            An optional NestedTensor of cache states, depending on `mode`.
            An Output instance, where .data is of the same shape as data.

        Raises:
            ValueError: If `mode` is unsupported.
        """
        cfg = self.config
        skip_input = data
        if cfg.residual_mode == BlockResidualMode.FP32:
            skip_input = _at_least_float32(skip_input)
        target = self.norm(data)

        if mode == ForwardMode.FORWARD:
            state, output = None, self.mamba(query=target)
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            state, output = self.mamba.prefill_states(
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
        output = (output.data + skip_input).astype(target.dtype)
        return dict(mamba_block=state), self._to_transformer_output(data=output)

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

    def init_states(self, *, target_batch_size: int, target_max_len: int) -> Nested[Tensor]:
        """Initializes cache for autoregressive cached decoding.

        Args:
            target_batch_size: The batch size of the target to be decoded.
            target_max_len: The sequence length of the target to be decoded.

        Returns:
            The cache as a `Nested[Tensor]`.
        """
        return dict(
            mamba_block=self.mamba.init_states(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def prefill_states(
        self,
        *,
        time_step: Nested[Tensor],
        data: Tensor,
        **_kwargs,
    ) -> tuple[Nested[Tensor], BaseTransformerLayer.Output]:
        """Initializes cache for autoregressive cached decoding.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from.
            data: Tensor of shape [batch, target_length, target_dim] corresponding to query vector
                at `time_step` indices. For batch index `i`, only `target[i, :time_step[i], ...]`
                will affect subsequent decoding.

        Returns:
            A `NestedTensor` state depending on the `attention` layer implementation.
            An Output instance, where .data is of the same shape as data.
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
        data: Tensor,
        cached_states: Optional[Nested[Tensor]] = None,
        **_kwargs,
    ) -> tuple[Optional[Nested[Tensor]], BaseTransformerLayer.Output]:
        cfg = self.config
        skip_input = data
        if cfg.residual_mode == BlockResidualMode.FP32:
            skip_input = _at_least_float32(skip_input)
        target = self.norm(data)

        if mode == ForwardMode.FORWARD:
            state, output = None, self.mamba(query=target)
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            state, output = self.mamba.prefill_states(
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

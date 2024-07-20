# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""A generic pipeline layer.

Adapted from:
https://github.com/tensorflow/lingvo/blob/2d46faf8/lingvo/jax/layers/pipeline.py
https://github.com/google/praxis/blob/dafadc1f922dc0cf422f44baf4922c9f0dfe4c15/praxis/layers/pipeline.py

A pipeline layer consists a stack of N identical sub layers, where
  * The variables are stacked across layers. Each stacked variable has shape [N, ...].
  * The inputs are divided into M microbatches and have shape [M, ...].
  * The processing depends on the pipeline schedule. Please refer to the corresponding schedule
    docstring for details.
"""

import dataclasses
import functools
from typing import Callable, NamedTuple, Optional, Protocol, Tuple, Union

import jax.ad_checkpoint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common import param_init
from axlearn.common.base_layer import BaseLayer, FactorizationSpec, NestedParameterSpec
from axlearn.common.config import REQUIRED, Configurable, InstantiableConfig, Required, config_class
from axlearn.common.module import Module, NestedTensor, Tensor, child_context, new_output_collection
from axlearn.common.utils import (
    Nested,
    NestedPartitionSpec,
    VDict,
    get_or_none,
    shapes,
    split_prng_key,
    with_sharding_constraint,
)


def transpose_to_pipeline_stage_inputs(x: Tensor, partition_spec: Optional[PartitionSpec] = None):
    """Transposes `x` from the 'layer-major' layout to the 'pipeline-major' layout.

    Args:
        x: A Tensor of shape [N, M, ...], where x[i, j] represents layerwise inputs for pipeline
            layer[i] and microbatch[j].
        partition_spec: The partition spec for x.

    Returns:
        A Tensor of shape [M + N - 1, N, ...], where x'[t, i] represents the layerwise inputs for
        timestep[t] and layer[i]: x'[i + j, i] == x[i, j].
    """
    n, m = x.shape[:2]
    # [N, M + N, ...].
    x = jnp.pad(x, [(0, 0), (0, n)] + [(0, 0)] * (x.ndim - 2))
    # [N * (M + N), ...].
    x = jnp.reshape(x, [-1] + list(x.shape[2:]))
    # [N * (M + N - 1), ...].
    x = x[:-n]
    # [N, M + N - 1, ...].
    x = jnp.reshape(x, [n, m + n - 1] + list(x.shape[1:]))
    # Apply sharding constraints at the first opportunity after reshapes
    # (i.e. when the input is first in the right shape for the constraint again).
    if partition_spec is not None:
        x = with_sharding_constraint(x, partition_spec)
    # [M + N - 1, N, ...].
    x = jnp.transpose(x, [1, 0] + list(range(2, x.ndim)))
    return x


def transpose_from_pipeline_stage_outputs(
    x: Tensor, partition_spec: Optional[PartitionSpec] = None
):
    """Transposes `x` from the 'pipeline-major' layout to the 'layer-major' layout.

    Args:
        x: A Tensor of shape [M + N - 1, N, ...], where x[t, i] represents the layerwise outputs of
            timestep[t] and layer[i].
        partition_spec: The partition spec for x' (layer-major).

    Returns:
        A Tensor of shape [N, M, ...], where x'[i, j] represents layerwise outputs of pipeline
        layer[i] and microbatch[j]: x'[i, j] == x[i + j, i].
    """
    t, n = x.shape[:2]
    m = t - n + 1
    # [N, M+N-1, ...].
    x = jnp.transpose(x, [1, 0] + list(range(2, x.ndim)))
    # [N * (M+N-1), ...].
    x = jnp.reshape(x, [-1] + list(x.shape[2:]))
    # [N * (M+N), ...].
    x = jnp.pad(x, [(0, n)] + [(0, 0)] * (x.ndim - 1))
    # [N, M+N, ...].
    x = jnp.reshape(x, [n, m + n] + list(x.shape[1:]))
    # Apply sharding constraints at the first opportunity after reshapes
    # (i.e. when the input is first in the right shape for the constraint again).
    if partition_spec is not None:
        x = with_sharding_constraint(x, partition_spec)
    # [N, M, ...].
    x = x[:, :m]
    return x


def _select_input_or_previous_outputs(input_t: Tensor, output_t_1: Tensor) -> Tensor:
    """Computes per-stage inputs by merging input and outputs from the previous time step.

    Args:
        input_t: A Tensor of shape [N, ...], where input_t == microbatch[t] if t < M; otherwise
            dummy values.
        output_t_1: A Tensor of shape [N, ...], representing the carry outputs of time step {t-1},
            where output_t_1[i] represents the output of stage i.

    Returns:
        A Tensor of shape [N, ...].
        output[0, ...] = input_t[0, ...] and output[n, ...] = output_t_1[n-1, ...] for n > 0.
    """
    # Shift outputs right.
    ndim = output_t_1.ndim
    padding = [[1, 0]] + [[0, 0]] * (ndim - 1)
    # Use lax.slice to guarantee the gradient is a pad.
    state_t = jax.lax.slice(jnp.pad(output_t_1, padding), [0] * ndim, output_t_1.shape)
    return jnp.where(
        # For operation semantics of iota, see:
        # https://openxla.org/xla/operation_semantics#iota
        jax.lax.broadcasted_iota("int32", state_t.shape, 0) == 0,
        input_t,
        state_t,
    )


def _mask_invalid_gradients(state: Nested[Tensor], *, is_valid: Tensor) -> Nested[Tensor]:
    """Uses stop_gradient to mask invalid (bubble) microbatch iterations."""

    # The jnp.where will be optimized away by XLA, but in the backward pass it will mask with zeros.
    return jax.tree_util.tree_map(
        lambda x: jnp.where(
            jnp.reshape(is_valid, (-1,) + (1,) * (x.ndim - 1)), x, jax.lax.stop_gradient(x)
        ),
        state,
    )


def _shard_pipeline(x: Tensor, *, axis: int = 0) -> Tensor:
    """Shards axis over 'pipeline'."""
    return with_sharding_constraint(
        x,
        PartitionSpec(
            *(PartitionSpec.UNCONSTRAINED for _ in x.shape[:axis]),
            "pipeline",
            *(PartitionSpec.UNCONSTRAINED for _ in x.shape[axis + 1 :]),
        ),
    )


class _PerStageFn(Protocol):
    """Per-stage implementation."""

    def __call__(
        self,
        state_n: Tensor,
        carry_tn: Tensor,
        prng_key_tn: Tensor,
        x_tn: Tensor,
    ):
        """Computes single stage outputs.

        Should be compatible with `jax.vmap`.

        Args:
            state_n: The parameters of the n'th stage.
            carry_tn: The carry input for the v_carry'th timestep and n'th stage.
            prng_key_tn: The PRNG key for the v_carry'th timestep and n'th stage.
            x_tn: The xs input for the v_carry'th timestep and n'th stage.

        Returns:
            dict(
                carry=<carry output>,
                y=<stage-wise output>,
                output_collection=<auxiliary outputs>,
            ).
        """


class BaseSchedule(Configurable):
    """A pipeline schedule.

    A schedule decides how microbatches are assigned to pipeline stages, effectively controlling the
    number of pipeline iterations, the size of pipeline bubbles, and the communication overhead.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures BaseSchedule."""

        num_stages: Required[int] = REQUIRED
        num_microbatches: Required[int] = REQUIRED

    def scan(
        self,
        fn: _PerStageFn,
        *,
        carry: Nested[Tensor],
        state: Nested[Tensor],
        xs: Nested[Tensor],
    ) -> Nested[Tensor]:
        """Implements the pipeline recurrence.

        The input `fn` specifies the single-stage implementation. Each scan iteration typically
        wraps `jax.vmap(fn)` to process stages in parallel.

        Args:
            fn: Per-stage implementation.
            carry: Microbatched inputs with leaves of shape [M, microbatch_size, ...].
            state: Stage state with leaves of shape [N, ...].
            xs: Side inputs with leaves of shape [num_iterations, ...], typically
                to be scanned over.

        Returns:
            Per-microbatch last-stage outputs with leaves of shape [M, N, microbatch_size, ...].
        """
        raise NotImplementedError(type(self))

    @property
    def num_stages(self) -> int:
        """Number of pipeline stages."""
        cfg: BaseSchedule.Config = self.config
        return cfg.num_stages

    @property
    def num_microbatches(self) -> int:
        """Number of microbatches."""
        cfg: BaseSchedule.Config = self.config
        return cfg.num_microbatches

    @property
    def num_iterations(self) -> int:
        """Number of microbatch iterations."""
        cfg: BaseSchedule.Config = self.config
        return cfg.num_stages + cfg.num_microbatches - 1

    def _is_valid_stage(self, t: Tensor) -> Tensor:
        """Returns a mask indicating whether per-stage values correspond to valid microbatches.

        Args:
            t: A scalar representing current timestep. 0 <= t < num_iterations.

        Returns:
            A mask of shape [N]. 1's indicate valid stages, 0's otherwise.
        """
        stage_id = jnp.arange(self.num_stages, dtype=jnp.int32)
        return jnp.logical_and(stage_id <= t, t - stage_id < self.num_microbatches)


class GPipeSchedule(BaseSchedule):
    """A basic schedule as seen in GPipe and GSPMD.

    Reference:
    https://arxiv.org/abs/1811.06965

    The processing happens in a loop consisting of M+N-1 steps.
    In each step 0 <= t < M+N-1, microbatch 0 <= m < M will be processed by stage (t - m)
    if 0 <= t - m < N.
    Or, expressed in stage-parallel terms, stages will process microbatch slice [t:t-N:-1] at step t
    (assuming that we pad the microbatches with N - 1 dummy microbatches at both ends).
    """

    def scan(
        self,
        fn: _PerStageFn,
        *,
        carry: Nested[Tensor],
        state: Nested[Tensor],
        xs: Nested[Tensor],
    ) -> Nested[Tensor]:
        """See `BaseSchedule.scan` for details."""

        @functools.partial(
            jax.ad_checkpoint.checkpoint,
            prevent_cse=False,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        def scan_fn(
            carry_in: Nested[Tensor],
            xs_t: Tuple[Nested[Tensor], Nested[Tensor]],
        ):
            """Processes timestep `t` in the pipeline (in parallel across pipeline stages).

            Args:
                carry_in: Loop state carried across scan iterations.
                xs_t: A tuple of side inputs. Each has leaves of shape [N, ...] or [1, ...].

            Returns:
                (carry_out, ys_t), where:
                - `carry_out` will be used as `carry_in` in the next scan iteration, and thus
                    has the same structure and shape as `carry_in`.
                - `ys_t` is dict(carry=..., y=..., output_collection=...) and will be stacked as
                    `ys` after scan is done.
                    Note that `carry` does not necessarily have the same structure as
                    `carry_out`, and represents the stage-wise carry output from `fn` with
                    leaves of shape [N, ...]. While only last-stage outputs are needed, we
                    retain [N, ...] for consistent sharding.
                    `y` is a `NestedTensor` representing the stage-wise output of `fn` with
                    leaves of shape [N, ...].
                    `output_collection` is an `OutputCollection` representing the auxiliary
                    outputs of `fn` with leaves of shape [N, ...].
            """
            t = carry_in["t"]

            # Compute vmap inputs. Leaves are of shape [N, ...] representing per-stage inputs.
            vmap_in = self._process_carry_in(carry_in)

            # Use stop_gradient for invalid (bubble) microbatch iterations. This jnp.where will
            # be optimized away by XLA, but in the backward pass it will be masking with zeros.
            vmap_state = _mask_invalid_gradients(state, is_valid=self._is_valid_stage(t))

            # Parallel processing along the N axis.
            vmap_out = jax.vmap(fn, spmd_axis_name="pipeline")(vmap_state, vmap_in, *xs_t)
            carry_out = dict(
                t=t + 1,
                carry_output_t_1=vmap_out["carry"],
                per_stage_inputs=carry_in["per_stage_inputs"],
            )

            # TODO(markblee): Consider slicing out just the last-stage outputs of vmap_out.
            # Note that vmap outputs are typically sharded over stages and may incur extra
            # communication per-iteration (e.g. from broadcasting last stage outputs).
            return carry_out, vmap_out

        _, scan_ys = jax.lax.scan(scan_fn, init=self._init_carry_in(carry), xs=xs)

        # Extract the last-stage outputs at each iteration from the stacked carry. Note
        # that the initial N-1 iterations constitute a pipeline bubble where we don't have
        # any meaningful last-stage outputs yet.
        # Use lax.slice to guarantee the gradient is a pad.
        n = self.num_stages
        scan_ys["carry"] = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(
                jax.lax.slice(x, [n - 1, x.shape[1] - 1] + [0] * (x.ndim - 2), x.shape), 1
            ),
            scan_ys["carry"],
        )
        return scan_ys

    def _init_carry_in(self, carry: Nested[Tensor]) -> Nested[Tensor]:
        """Computes initial loop state.

        Args:
            carry: Microbatched inputs with leaves of shape [M, microbatch_size, ...].

        Returns:
            Initial carry state with the keys:
            - t: A scalar time step initialized to 0.
            - carry_output_t_1: Carry output with same tree structure as `carry` and leaves of shape
                [N, microbatch_size, ...] initialized to zeros.
            - per_stage_inputs: Per-stage input with same tree structure as `carry` and leaves of
                shape [M, N, microbatch_size, ...], obtained by padding along N dim.
        """
        n = self.num_stages

        def pad_carry(v_carry: Tensor):
            """Pads input from [M, microbatch_size, ...] to [M, N, microbatch_size, ...].

            We pad explicitly instead of broadcasting along N to avoid gradient accumulation in the
            backward pass (only the first stage produces non-zero gradients.)
            """
            return jnp.pad(
                jnp.expand_dims(v_carry, 1), [(0, 0), (0, n - 1)] + [(0, 0)] * (v_carry.ndim - 1)
            )

        return dict(
            # Current loop iteration.
            t=jnp.array(0, dtype=jnp.int32),
            # [N, microbatch_size, ...].
            carry_output_t_1=jax.tree_util.tree_map(
                lambda x: jnp.zeros((n,) + x.shape[1:], dtype=x.dtype), carry
            ),
            # [M, N, microbatch_size, ...].
            per_stage_inputs=jax.tree_util.tree_map(pad_carry, carry),
        )

    def _process_carry_in(self, carry_in: Nested[Tensor]) -> Nested[Tensor]:
        """Computes the vmap input for timestep `t`.

        Args:
            carry_in: Input loop state for timestep `t`.

        Returns:
            A nested Tensor with leaves of shape [N, ...]:
            - Stage 0 input will be per_stage_inputs[t % M, :1], that is, microbatch[t] if t < M;
            - Stage 1..N-1 inputs will be v_carry_output_t_1[:N-1], that is, the outputs of stages
                0..N-2 from iteration t-1.
        """
        # When t >= m, we feed dummy inputs to the pipeline until the pipeline is flushed.
        # Note that at the end of all iterations we only extract the last-stage outputs from the
        # stacked vmap outputs.
        idx = carry_in["t"] % self.num_microbatches

        def compute_carry_input(v_input_t: Tensor, v_carry_output_t_1: Tensor) -> Tensor:
            return _select_input_or_previous_outputs(v_input_t[idx], v_carry_output_t_1)

        return jax.tree_util.tree_map(
            compute_carry_input,
            carry_in["per_stage_inputs"],
            carry_in["carry_output_t_1"],
        )


class StreamSchedule(BaseSchedule):
    """A schedule utilizing a "streaming" buffer.

    Reference:
    https://github.com/google/praxis/blob/c41477c601fea125ae58f136f139758c34d121b8/praxis/layers/pipeline.py#L140-L149

    Specifically, microbatches of shape [M, microbatch_size, ...] are reshaped to
    [num_streams=N, M // N, microbatch_size, ...], where the leading `num_streams` dim can be
    sharded over the "pipeline" axis of the mesh.

    First, all M input microbatches are reshaped to a buffer of shape
    [num_streams=N, stream_size=M // N, microbatch_size, ...].
    At each time step `t`, we take a slice of the buffer [:, t % stream_size, microbatch_size, ...].
    This produces a slice of shape [N, microbatch_size, ...] to be used as per-stage inputs.
    These per-stage inputs are combined with per-stage outputs from time step `t-1` and fed to
    stages for parallel processing.

    After computing per-stage outputs for `t`, per-stage inputs are shifted left so that future
    iterations `t' % (M // N)` read new microbatches. Because shifting produces an empty buffer at
    the last stage, we also store last-stage outputs for time step `t`. The updated per-stage
    buffers are scattered back to the original [N, M // N, ...] streams for use in the next step.

    After the full `M + N - 1` iterations, the buffer contains the last-stage outputs for all `M`
    microbatches. We rotate the buffer to obtain the final outputs.

    For instance, with M=6, N=3, where sN(x) represents stage N outputs for input x:

    t=0:    Buffer:           Previous Outputs:    Inputs:
            [    0,     1]    [    -]              [    0]
            [    2,     3]    [    -]              [    -]
            [    4,     5]    [    -]              [    -]

    t=1:    Buffer:           Previous Outputs:    Inputs:
            [    2,     1]    [s0(0)]              [    1]
            [    4,     3]    [s1(-)]              [s0(0)]
            [s2(-),     5]    [s2(-)]              [s1(-)]

    t=2:    Buffer:           Previous Outputs:    Inputs:
            [    2,     3]    [s0(1)]              [    2]
            [    4,     5]    [s1(0)]              [s0(1)]
            [s2(-), s2(-)]    [s2(-)]              [s1(0)]

    t=3:    Buffer:           Previous Outputs:    Inputs:
            [    4,     3]    [s0(2)]              [    3]
            [s2(-),     5]    [s1(1)]              [s0(2)]
            [s2(0), s2(-)]    [s2(0)]              [s1(1)]

    t=4:    Buffer:           Previous Outputs:    Inputs:
            [    4,     5]    [s0(3)]              [    4]
            [s2(-), s2(-)]    [s1(2)]              [s0(3)]
            [s2(0), s2(1)]    [s2(1)]              [s1(2)]

    t=5:    Buffer:           Previous Outputs:    Inputs:
            [s2(-),     5]    [s0(4)]              [    5]
            [s2(0), s2(-)]    [s1(3)]              [s0(4)]
            [s2(2), s2(1)]    [s2(2)]              [s1(3)]

    t=6:    Buffer:           Previous Outputs:    Inputs:
            [s2(-), s2(-)]    [s0(5)]              [s2(-)]
            [s2(0), s2(1)]    [s1(4)]              [s0(5)]
            [s2(2), s2(3)]    [s2(3)]              [s1(4)]

    t=7:    Buffer:           Previous Outputs:    Inputs:
            [s2(0), s2(-)]    [s0(-)]              [s2(-)]
            [s2(2), s2(1)]    [s1(5)]              [s0(-)]
            [s2(4), s2(3)]    [s2(4)]              [s1(5)]

    out:    Buffer:           Previous Outputs:
            [s2(0), s2(1)]    [s0(-)]
            [s2(2), s2(3)]    [s1(-)]
            [s2(4), s2(5)]    [s2(5)]

    """

    def __init__(self, cfg):
        super().__init__(cfg)
        cfg: BaseSchedule.Config = self.config
        if cfg.num_microbatches % cfg.num_stages != 0:
            raise ValueError(
                f"Number of microbatches ({cfg.num_microbatches}) "
                f"should be divisible by number of streams ({self.num_stages})."
            )

    def scan(
        self,
        fn: _PerStageFn,
        *,
        carry: Nested[Tensor],
        state: Nested[Tensor],
        xs: Nested[Tensor],
    ) -> Nested[Tensor]:
        """See `BaseSchedule.scan` for details."""

        n = self.num_stages
        m = self.num_microbatches

        @functools.partial(
            jax.ad_checkpoint.checkpoint,
            prevent_cse=False,
            policy=jax.checkpoint_policies.save_only_these_names("iter_input"),
        )
        def scan_fn(
            carry_in: Nested[Tensor],
            xs_t: Tuple[Nested[Tensor], Nested[Tensor]],
        ):
            """Processes timestep `t` in the pipeline (in parallel across pipeline stages).

            Args:
                carry_in: Loop state carried across scan iterations.
                xs_t: A tuple of side inputs. Each has leaves of shape [N, ...] or [1, ...].

            Returns:
                (carry_out, ys_t), where:
                - `carry_out` will be used as `carry_in` in the next scan iteration, and thus
                    has the same structure and shape as `carry_in`.
                - `ys_t` is dict(y=..., output_collection=...) and will be stacked as
                    `ys` after scan is done.
                    `y` is a `NestedTensor` representing the stage-wise output of `fn` with
                    leaves of shape [N, ...].
                    `output_collection` is an `OutputCollection` representing the auxiliary
                    outputs of `fn` with leaves of shape [N, ...].
            """
            t = carry_in["t"]
            buffer = carry_in["buffer"]
            carry_output_t_1 = carry_in["carry_output_t_1"]
            microbatch_idx = t % (m // n)

            # [N, M // N, microbatch_size, ...] --> [N, microbatch_size, ...].
            buf_col_t = jax.tree_util.tree_map(lambda x: x[:, microbatch_idx], buffer)

            def compute_carry_input(v_input_t: Tensor, v_carry_output_t_1: Tensor) -> Tensor:
                v_input_t = _select_input_or_previous_outputs(v_input_t, v_carry_output_t_1)
                return jax.ad_checkpoint.checkpoint_name(_shard_pipeline(v_input_t), "iter_input")

            # Compute vmap inputs.
            # Leaves are of shape [N, microbatch_size, ...] representing per-stage inputs.
            vmap_in = jax.tree_util.tree_map(compute_carry_input, buf_col_t, carry_output_t_1)

            # Use stop_gradient for invalid (bubble) microbatch iterations. This jnp.where will
            # be optimized away by XLA, but in the backward pass it will be masking with zeros.
            vmap_state = _mask_invalid_gradients(state, is_valid=self._is_valid_stage(t))

            # Parallel processing along the N axis.
            vmap_out = jax.vmap(fn, spmd_axis_name="pipeline")(vmap_state, vmap_in, *xs_t)

            def update_buffer(buf: Tensor, buf_col_t: Tensor, v_out_t: Tensor):
                """Updates the column of `buf` at `microbatch_idx `.

                The updated `buf` will be `concat(buf_col_t[1:], v_out_t[-1])`.

                Args:
                    buf: A Tensor of shape [N, K, ...], where K is the stream size.
                    buf_col_t: The current contents of `buf[:, microbatch_idx]`, of shape [N, ...].
                    v_out_t: The outputs of stages at iteration t, of shape [N, ...].

                Returns:
                    The updated `buf`, where:
                    buf[n, microbatch_idx] = buf_col_t[n+1] = buf[n+1, microbatch_idx]
                    for 0 <= n < N - 1;
                    buf[N-1, microbatch_idx] = v_out_t[N-1].
                """
                # Shift the current slice to the left, such that the next time we arrive at the same
                # microbatch_idx, we read a new microbatch.
                # Note that this incurs some communication across the sharded num_streams dim (e.g.
                # via collective permute).
                padding = [[0, 1]] + [[0, 0]] * (buf_col_t.ndim - 1)
                buf_col_t = jax.lax.slice_in_dim(
                    jnp.pad(buf_col_t, padding), 1, buf_col_t.shape[0] + 1, axis=0
                )
                # Reuse the now-empty buffer corresponding to the last stage with the current final
                # stage outputs.
                buf_col_t = jnp.where(
                    jax.lax.broadcasted_iota("int32", buf_col_t.shape, 0) == n - 1,
                    v_out_t,
                    buf_col_t,
                )
                buf_col_t = jnp.expand_dims(buf_col_t, 1)  # [N, 1, microbatch_size, ...].
                # Update one column of buffer of shape [N, K, ...] (buf[:, microbatch_idx]) with
                # v_inputs_t, i.e. buf[n, microbatch_idx] = buf_col_t[n, 0] for 0 <= n < N.
                return jax.lax.dynamic_update_slice_in_dim(buf, buf_col_t, microbatch_idx, axis=1)

            carry_out = dict(
                t=t + 1,
                carry_output_t_1=vmap_out["carry"],
                buffer=jax.tree_util.tree_map(
                    update_buffer,
                    buffer,
                    buf_col_t,
                    vmap_out["carry"],
                ),
            )
            vmap_out.pop("carry")  # Don't need to stack carry outputs.
            return carry_out, vmap_out

        carry_out, scan_ys = jax.lax.scan(scan_fn, init=self._init_carry_in(carry), xs=xs)

        # We iterate for M+N-1 steps, so the last outputs are written to the first (n-1) % (m // n)
        # positions of each stream. We rotate along the streams dim to fix the ordering.
        offset = (n - 1) % (m // n)

        def rotate_out(x: Tensor):
            if offset > 0:
                x = jnp.concatenate([x[:, offset:], x[:, :offset]], axis=1)
            return jnp.reshape(x, (m,) + x.shape[2:])

        # The buffer is also used for storing last-stage outputs.
        scan_ys["carry"] = jax.tree_util.tree_map(rotate_out, carry_out["buffer"])
        return scan_ys

    def _init_carry_in(self, carry: Nested[Tensor]) -> Nested[Tensor]:
        """Computes initial loop state.

        Args:
            carry: Microbatched inputs with leaves of shape [M, microbatch_size, ...].

        Returns:
            Initial carry state with the keys:
            - t: A scalar time step initialized to 0.
            - carry_output_t_1: Carry output with same tree structure as `carry` and leaves of shape
                [N, microbatch_size, ...] initialized to zeros.
            - buffer: Per-stage input with same tree structure as `carry` and leaves of
                shape [N, M // N, microbatch_size, ...]. The N dim is sharded across "pipeline".
        """
        n = self.num_stages
        m = self.num_microbatches

        def reshape_carry(v_carry: Tensor) -> Tensor:
            """Reshapes from [M, microbatch_size, ...] to [N, M // N, microbatch_size, ...]."""
            v_carry = jnp.reshape(v_carry, (n, m // n) + v_carry.shape[1:])
            return _shard_pipeline(v_carry)

        return dict(
            # Current loop iteration.
            t=jnp.array(0, dtype=jnp.int32),
            # [N, microbatch_size, ...]. Equivalent to the "shift" buffer in praxis.
            carry_output_t_1=jax.tree_util.tree_map(
                lambda x: jnp.zeros((n,) + x.shape[1:], dtype=x.dtype), carry
            ),
            # [N, M // N, microbatch_size, ...]. Equivalent to the "stream" buffer in praxis.
            buffer=jax.tree_util.tree_map(reshape_carry, carry),
        )


class Pipeline(BaseLayer):
    """A generic pipeline layer.

    Different pipeline implementations can be configured via `schedule`.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures Pipeline."""

        layer: Required[InstantiableConfig] = REQUIRED  # The config for the sub layer.
        num_layers: Required[int] = REQUIRED  # Repeat layers specified in `layer` this many times.
        num_microbatches: Required[int] = REQUIRED
        schedule: BaseSchedule.Config = GPipeSchedule.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("layer", cfg.layer)
        self._schedule: BaseSchedule = cfg.schedule.set(
            num_stages=cfg.num_layers,
            num_microbatches=cfg.num_microbatches,
        ).instantiate()

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        cfg: Pipeline.Config = self.config
        specs = VDict(**super().create_parameter_specs_recursively())

        def transform_factorization_spec(
            spec: Optional[FactorizationSpec],
        ) -> Optional[FactorizationSpec]:
            if spec is None:
                return None
            return FactorizationSpec(axes=[None] + list(spec.axes))

        return jax.tree_util.tree_map(
            lambda spec: dataclasses.replace(
                spec,
                shape=(cfg.num_layers, *spec.shape),
                mesh_axes=PartitionSpec("pipeline", *spec.mesh_axes),
                factorization=transform_factorization_spec(spec.factorization),
                fan_axes=param_init.maybe_prepend_axis(
                    spec.fan_axes, axis_type=param_init.FanAxes.AxisType.BATCH_AXIS
                ),
            ),
            specs,
        )

    def initialize_parameters_recursively(
        self,
        prng_key: Union[Tensor, VDict],
        *,
        prebuilt: Optional[NestedTensor] = None,
    ) -> NestedTensor:
        def init(prng_key_i, prebuilt_i):
            return VDict(
                layer=self.layer.initialize_parameters_recursively(
                    prng_key_i, prebuilt=get_or_none(prebuilt_i, "layer")
                )
            )

        cfg: Pipeline.Config = self.config
        return jax.vmap(init)(split_prng_key(prng_key, cfg.num_layers).keys, prebuilt)

    class Output(NamedTuple):
        carry: NestedTensor
        ys: NestedTensor

    def _run(
        self,
        fn: Callable[[NestedTensor, NestedTensor], NestedTensor],
        carry: Optional[NestedTensor] = None,
        *,
        xs: Optional[NestedTensor] = None,
        carry_partition_spec: Optional[NestedPartitionSpec] = None,
        xs_partition_spec: Optional[NestedPartitionSpec] = None,
        ys_partition_spec: Optional[NestedPartitionSpec] = None,
    ):
        """Invokes 'fn' for each sub-layer with inputs already with the microbatch axis.

        Args:
            fn: A function with args (carry, x) returning a dict(carry=..., y=...).
            carry: A nested tensor for the iterative input of the 0'th sub-layer.
                It must have shape [M, microbatch_size, ...].
            xs: A nested tensor with separate inputs for each sub-layer, where each leaf value T is
                a tensor of shape [cfg.num_layers, M, microbatch_size, ...] and T[i, j, ...]
                represents layer-wise inputs of microbatch j to the i'th sub-layer.
            carry_partition_spec: Partition spec for the carry tensors.
                If None, tensors will be replicated.
            xs_partition_spec: Partition spec for the input xs tensors. If None, tensors will be
                replicated except for sharding along the "pipeline" mesh axis.
            ys_partition_spec: Partition spec for the output ys tensors. If None, tensors will be
                replicated except for sharding along the "pipeline" mesh axis.

        Returns:
            A dict with the following keys:
            - carry: A nested tensor with the same structure as the input carry representing the
                iterative output of the last sub-layer.
            - ys: A nested tensor where each leaf value T is a tensor of shape
                [cfg.num_layers, M, microbatch_size, ...] and T[i, ...] represents layer-wise output
                from the i'th sub-layer.
        """
        cfg: Pipeline.Config = self.config
        self.vlog(1, "carry=%s xs=%s", shapes(carry), shapes(xs))

        carry_leaves = jax.tree_util.tree_leaves(carry)
        if not carry_leaves:
            raise ValueError("Expected at least one input leaf.")
        if carry_leaves[0].ndim < 2:
            raise ValueError(
                "Expected leaves to have shape `[num_microbatches, microbatch_size, ...]`; "
                f"instead, found {carry_leaves[0].shape}."
            )

        # Number of microbatches.
        m = carry_leaves[0].shape[0]
        # Number of pipeline stages.
        n = cfg.num_layers

        if carry is None:
            carry = {}
            carry_partition_spec = {}
        if carry_partition_spec is None:
            carry_partition_spec = jax.tree_util.tree_map(
                lambda x: PartitionSpec(*[PartitionSpec.UNCONSTRAINED for _ in x.shape]), carry
            )
        if xs is None:
            xs = {}
            xs_partition_spec = {}
        if xs_partition_spec is None:
            xs_partition_spec = jax.tree_util.tree_map(
                lambda x: PartitionSpec(
                    "pipeline", *[PartitionSpec.UNCONSTRAINED for _ in x.shape[1:]]
                ),
                xs,
            )

        # Transpose from "layer-major" [N, M, ...] to "pipeline-major" [N + M - 1, N, ...].
        #
        # Note: for efficient decoding we may want to skip transposes and keep decoding states in
        # the "pipeline-major" form (i.e., in the shape of [N + M - 1, N, ...]).
        #
        # To be investigated in the future.
        padded_xs = jax.tree_util.tree_map(
            transpose_to_pipeline_stage_inputs, xs, xs_partition_spec
        )
        self.vlog(2, "padded_xs=%s", shapes(padded_xs))

        def stack_and_reshape(*keys):
            keys = jnp.stack(keys)
            return jnp.reshape(keys, [m + n - 1, n] + list(keys.shape[1:]))

        prng_keys = jax.random.split(self.prng_key, (m + n - 1) * n)
        prng_keys = jax.tree_util.tree_map(stack_and_reshape, *prng_keys)

        layer_output_collection = new_output_collection()
        with child_context("layer", output_collection=layer_output_collection) as layer_context:

            def vmap_fn(
                state_n: Tensor, carry_tn: Tensor, prng_key_tn: jax.random.PRNGKey, x_tn: Tensor
            ):
                """See `_PerStageFn` for details."""
                output_collection_tn = new_output_collection()
                with child_context(
                    "iter",
                    module=layer_context.module,
                    state=state_n,
                    prng_key=prng_key_tn,
                    output_collection=output_collection_tn,
                ):
                    carry_tn, y_tn = fn(carry_tn, x_tn)
                self.vlog(3, "output_collection_tn=%s", shapes(output_collection_tn))
                return dict(carry=carry_tn, y=y_tn, output_collection=output_collection_tn)

            carry = jax.tree_util.tree_map(with_sharding_constraint, carry, carry_partition_spec)
            scan_ys = self._schedule.scan(
                vmap_fn, carry=carry, state=layer_context.state, xs=(prng_keys, padded_xs)
            )
            final_carry = jax.tree_util.tree_map(
                with_sharding_constraint, scan_ys.pop("carry"), carry_partition_spec
            )

            ys = scan_ys["y"]
            if ys_partition_spec is None:
                ys_partition_spec = jax.tree_util.tree_map(
                    lambda x: PartitionSpec(
                        "pipeline", *[PartitionSpec.UNCONSTRAINED for _ in x.shape[1:]]
                    ),
                    ys,
                )
            # Transpose from pipeline-major [N + M - 1, N, ...] back to layer-major [N, M, ...].
            ys = jax.tree_util.tree_map(
                transpose_from_pipeline_stage_outputs, ys, ys_partition_spec
            )
            self.vlog(3, "scan_ys.output_collection=%s", shapes(scan_ys["output_collection"]))
            layer_output_collection.update(
                jax.tree_util.tree_map(
                    transpose_from_pipeline_stage_outputs, scan_ys["output_collection"]
                )
            )
            self.vlog(3, "layer_output_collection=%s", shapes(layer_output_collection))

        this_output_collection = self.get_invocation_context().output_collection
        layer_output = this_output_collection.add_child("layer")
        layer_output.module_outputs.update(**layer_output_collection.module_outputs)
        layer_output.state_updates.update(**layer_output_collection.state_updates)
        self.vlog(3, "this_output_collection=%s", shapes(this_output_collection))

        # Each summary value in `layer_output_collection` has shape (N, M, ...). For example,
        # if a repeated layer outputs a scalar summary value, it will have shape [N, M].
        # Below we split the stacked values and output them separately under scope
        # "layer{i}/microbatch{j}" so that scalar summaries can be handled correctly.
        for i in range(n):
            layer_i_output = this_output_collection.add_child(f"layer{i}")
            for j in range(m):
                microbatch_j_output = layer_i_output.add_child(f"microbatch{j}")
                microbatch_j_output.summaries.update(
                    **jax.tree_util.tree_map(
                        lambda x, i=i, j=j: x[i, j], layer_output_collection.summaries
                    )
                )
        return self.Output(carry=final_carry, ys=ys)

    def _to_microbatches(self, inputs):
        """Reshapes inputs from [batch_size, ...] to [M, microbatch_size, ...]."""
        cfg: Pipeline.Config = self.config

        def reshape_and_transpose(x: Tensor):
            # Keep batch partitioning along the 'microbatch_size' dim.
            x = jnp.reshape(x, [-1, cfg.num_microbatches] + list(x.shape[1:]))
            return jnp.transpose(x, [1, 0] + list(range(2, x.ndim)))

        return jax.tree_util.tree_map(reshape_and_transpose, inputs)

    # pylint: disable-next=no-self-use
    def _from_microbatches(self, inputs):
        def transpose_and_reshape(x: Tensor):
            x = jnp.transpose(x, [1, 0] + list(range(2, x.ndim)))
            return jnp.reshape(x, [-1] + list(x.shape[2:]))

        return jax.tree_util.tree_map(transpose_and_reshape, inputs)

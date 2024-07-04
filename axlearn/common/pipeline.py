# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A generic pipeline layer.

https://arxiv.org/abs/1811.06965

Adapted from:
https://github.com/tensorflow/lingvo/blob/2d46faf8/lingvo/jax/layers/pipeline.py

A pipeline layer consists a stack of N identical sub layers, where
  * The variables are stacked across layers. Each stacked variable has shape [N, ...].
  * The inputs are divided into M microbatches and have shape [M, ...].
  * The processing happens in a loop consisting of M+N-1 steps.
    In each step 0 <= t < M+N-1, microbatch 0 <= m < M will be processed by layer (t - m)
    if 0 <= t - m < N.
    Or, expressed in layer-parallel terms, layers will process microbatch slice [t:t-N:-1] at step t
    (assuming that we pad the microbatches with N - 1 dummy microbatches at both ends).
"""

import dataclasses
import functools
from typing import Callable, NamedTuple, Optional, Tuple, Union

import jax.ad_checkpoint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common import param_init
from axlearn.common.base_layer import BaseLayer, FactorizationSpec, NestedParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.module import (
    Module,
    NestedTensor,
    Tensor,
    child_context,
    current_context,
    new_output_collection,
)
from axlearn.common.utils import (
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


class Pipeline(BaseLayer):
    """https://arxiv.org/abs/1811.06965."""

    @config_class
    class Config(BaseLayer.Config):
        layer: Required[InstantiableConfig] = REQUIRED  # The config for the sub layer.
        num_layers: Required[int] = REQUIRED  # Repeat layers specified in `layer` this many times.
        num_microbatches: Required[int] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("layer", cfg.layer)

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
            - carry: A nested tensor with the same structure as iterative_input_0
                representing the iterative output of the last sub-layer.
            - ys: A nested tensor where each leaf value T is a tensor of shape
                [cfg.num_layers, M, microbatch_size, ...] and T[i, ...] represents layer-wise output
                from the i'th sub-layer.
        """
        cfg: Pipeline.Config = self.config
        self.vlog(1, "carry=%s xs=%s", shapes(carry), shapes(xs))
        # Number of microbatches.
        m = jax.tree_util.tree_flatten(carry)[0][0].shape[0]
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

        def pad_carry(v_carry: Tensor, partition_spec: PartitionSpec):
            """Pads input [M, microbatch_size, ...] to [M + N - 1, 1, microbatch_size, ...]."""
            # Pad to shape [M + N - 1, ...].
            v_carry = jnp.pad(v_carry, [(0, n - 1)] + [(0, 0)] * (v_carry.ndim - 1))
            v_carry = with_sharding_constraint(
                v_carry, PartitionSpec(PartitionSpec.UNCONSTRAINED, *partition_spec[1:])
            )
            # Expand to shape [M + N - 1, 1, ...].
            v_carry = jnp.expand_dims(v_carry, 1)
            return v_carry

        # [M + N - 1, 1, microbatch_size, ...].
        padded_carry = jax.tree_util.tree_map(pad_carry, carry, carry_partition_spec)

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

        context = current_context()
        assert context is not None
        prng_keys = jax.random.split(context.prng_key, (m + n - 1) * n)

        def stack_and_reshape(*keys):
            keys = jnp.stack(keys)
            return jnp.reshape(keys, [m + n - 1, n] + list(keys.shape[1:]))

        prng_keys = jax.tree_util.tree_map(stack_and_reshape, *prng_keys)
        layer_output_collection = new_output_collection()
        with child_context("layer", output_collection=layer_output_collection) as layer_context:

            def vmap_fn(
                state_n: Tensor, prng_key_tn: jax.random.PRNGKey, carry_tn: Tensor, x_tn: Tensor
            ):
                """Invokes fn for one microbatch and one layer.

                Args:
                    state_n: The parameters of the n'th layer.
                    prng_key_tn: The PRNG key for the v_carry'th timestep and n'th layer.
                    carry_tn: The carry input for the v_carry'th timestep and n'th layer.
                    x_tn: The xs input for the v_carry'th timestep and n'th layer.

                Returns:
                    dict(
                        carry=<carry output>,
                        y=<layerwise output>,
                        output_collection=<auxiliary outputs>,
                    ).
                """
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

            @functools.partial(
                jax.ad_checkpoint.checkpoint,
                prevent_cse=False,
                policy=jax.checkpoint_policies.nothing_saveable,
            )
            def scan_fn(
                carry_output_t_1: NestedTensor,
                scan_t: Tuple[NestedTensor, NestedTensor, NestedTensor],
            ):
                """Processes timestep v_carry in the pipeline (in parallel across pipeline stages).

                Args:
                    carry_output_t_1: A NestedTensor where each Tensor has shape
                        [N=num_layers, ...], representing carry output of timestep {t-1}.
                    scan_t: A tuple of (prng_key_t, input_t, x_t), each is a NestedTensor where each
                        leaf tensor has shape [N, ...] or [1, ...].

                Returns:
                    carry_output_t, dict(carry=..., y=..., output_collection=...), where
                    - `carry_output_t` and `carry` represents the carry output of timestep t and has
                       the same structure and shape as `carry_carry_output_t_1`;
                    - `y` is a NestedTensor representing the layerwise output of fn with leaves of
                       shape [N, ...].
                    - `output_collection` is an OutputCollection representing the auxiliary outputs
                       of fn with leaves of shape [N, ...];
                """

                def compute_carry_input(
                    v_input_t: Tensor, v_carry_output_t_1: Tensor, partition_spec: PartitionSpec
                ):
                    """Computes the carry input for timestep v_carry.

                    Args:
                        v_input_t: A Tensor of shape [1, ...], where
                            v_input_t of timestep t == microbatch[t] if t < M; otherwise padding.
                        v_carry_output_t_1: A Tensor of shape [N, ...], representing carry output of
                            timestep {t-1}.
                        partition_spec: PartitionSpec for carry values.
                    """
                    # Layer 0 input will be v_input_t, that is, microbatch[t] if t < M.
                    # Layer 1..N-1 inputs will be v_carry_output_t_1[0..N-2], that is, the outputs
                    # of layer 0..N-2 from iteration t - 1.
                    v_carry_input_t = jnp.concatenate([v_input_t, v_carry_output_t_1[:-1]], axis=0)
                    return with_sharding_constraint(
                        v_carry_input_t, PartitionSpec(*(["pipeline"] + list(partition_spec[1:])))
                    )

                # Per-timestep inputs.
                # Each leaf tensor in `prng_key_t` and `x_t` has shape [N, ...].
                prng_key_t, input_t, x_t = scan_t
                carry_input_t = jax.tree_util.tree_map(
                    compute_carry_input, input_t, carry_output_t_1, carry_partition_spec
                )

                # Parallel processing along the N axis.
                vmap_out = jax.vmap(vmap_fn)(layer_context.state, prng_key_t, carry_input_t, x_t)
                self.vlog(3, "vmap_out.output_collection=%s", shapes(vmap_out["output_collection"]))
                return vmap_out["carry"], vmap_out

            carry_t0 = jax.tree_util.tree_map(
                lambda x: jnp.tile(jnp.zeros_like(x[:1]), [n] + [1] * (x.ndim - 1)), carry
            )
            self.vlog(
                2,
                "carry_t0=%s prng_keys=%s padded_carry=%s padded_xs=%s",
                shapes(carry_t0),
                shapes(prng_keys),
                shapes(padded_carry),
                shapes(padded_xs),
            )
            _, scan_ys = jax.lax.scan(
                scan_fn,
                init=carry_t0,
                xs=(prng_keys, padded_carry, padded_xs),
            )
            final_carry = jax.tree_util.tree_map(
                lambda x: x[n - 1 :, -1, ...], scan_ys.pop("carry")
            )
            final_carry = jax.tree_util.tree_map(
                with_sharding_constraint, final_carry, carry_partition_spec
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

# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Mixture of experts implementations.

Reference: https://arxiv.org/abs/2405.15052.
"""

import contextlib
import enum
import re
from functools import partial, reduce
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax import lax
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.interpreters.pxla import thread_resources

import axlearn.common.megablock.ops as mblx
from axlearn.common.attention import NormPosition
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    ConfigModifier,
    ConfigOr,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
    maybe_instantiate,
)
from axlearn.common.ein_ops import rearrange
from axlearn.common.layers import (
    BaseNormalizationLayer,
    Dropout,
    LayerNorm,
    MovingAverage,
    StochasticDepth,
    get_activation_fn,
)
from axlearn.common.module import Module, child_context, nowrap
from axlearn.common.param_init import FanAxes, constant_initializer
from axlearn.common.quantized_dot_general.layers import DenseGeneralBaseLayer
from axlearn.common.ragged_all_to_all_batching import ragged_all_to_all_batched
from axlearn.common.utils import (
    HybridMeshShape,
    MeshShape,
    Nested,
    NestedTensor,
    PartitionSpec,
    Tensor,
    VDict,
    flatten_items,
    get_recursively,
    infer_mesh_shape,
    set_recursively,
    tree_paths,
    with_sharding_constraint,
)


class GateNoise(enum.Enum):
    """Types of noise that can be added to gating logits."""

    # Standard Gumbel noise.
    GUMBEL = "GUMBEL"


# Type definitions for configurable functions.
TopKFn = Callable[[Tensor, int], Tuple[Tensor, Tensor]]
ScoreFn = Callable[[Tensor, int], Tensor]


def approx_max_k(
    *,
    reduction_dimension=-1,
    recall_target=0.95,
    reduction_input_size_override=-1,
    aggregate_to_topk=True,
) -> TopKFn:
    """Creates a configured approximate top-k function.

    This can be used as an alternative to jax.lax.top_k for faster but approximate
    top-k selection, which is useful for large numbers of experts.

    Args:
        reduction_dimension: Dimension to reduce over.
        recall_target: Target recall for the approximation (higher = more accurate).
        reduction_input_size_override: Override for input size.
        aggregate_to_topk: Whether to aggregate results to exact top-k.

    Returns:
        A function that performs approximate top-k selection.
    """

    def fn(operand, k):
        return jax.lax.approx_max_k(
            operand,
            k=k,
            reduction_dimension=reduction_dimension,
            recall_target=recall_target,
            reduction_input_size_override=reduction_input_size_override,
            aggregate_to_topk=aggregate_to_topk,
        )

    return fn


def sigmoid() -> ScoreFn:
    """Creates a sigmoid scoring function as an alternative to softmax.

    Returns:
        A function that applies sigmoid to compute scores.
    """

    def fn(x, axis):  # pylint: disable=unused-argument
        return jax.nn.sigmoid(x)

    return fn


def _router_z_loss(logits: Tensor) -> Tensor:
    """Loss that encourages router logits to remain small and improves stability.

    Reference:
    https://github.com/tensorflow/mesh/blob/fbf7b1e547e8b8cb134e81e1cd350c312c0b5a16/mesh_tensorflow/transformer/moe.py#L1956

    Args:
        logits: A tensor with shape (..., num_experts).

    Returns:
        z_loss: A scalar loss.
    """
    log_z = jax.nn.logsumexp(logits, axis=-1)
    z_loss = jnp.mean(jnp.square(log_z))
    return z_loss


def _cum_sum(
    elements: Tensor, *, axis: int = 0, exclusive: bool = False, reverse: bool = False
) -> Tensor:
    """Same as jax.np.cumsum but with the extra options from tf.cumsum.

    Args:
        elements: Array to compute a cumulative sum over.
        axis: The axis to compute the cumsum over.
        exclusive: If True, perform exclusive cumsum.
            With exclusive=False: cumsum([a, b, c]) --> [a, a + b, a + b + c]
            With exclusive=True: cumsum([a, b, c]) --> [0, a, a + b]
        reverse: (default: False), perform the cumulative sum in reverse.

    Returns:
        The cumulative sum.
    """
    if reverse:
        elements = jnp.flip(elements, axis=axis)

    result = jnp.cumsum(elements, axis=axis)
    if exclusive:
        result = result - elements
    if reverse:
        return jnp.flip(result, axis=axis)
    return result


def _create_over_capacity_ratio_summary(
    *, mask: Tensor, position_in_expert: Tensor, capacity: float
) -> Tensor:
    """Computes the capacity ratio of tokens that were not dispatched due to lack of capcity.

    Args:
        mask: A binary mask.
        position_in_expert: Token position in experts.
        capacity: Expert capacity.

    Returns:
        Expert over capacity ratio.
    """
    masked_position_in_expert = mask * position_in_expert
    ge_capacity = jnp.greater_equal(masked_position_in_expert, capacity)
    over_capacity = jnp.sum(ge_capacity).astype(jnp.float32)
    denom = jnp.sum(mask).astype(jnp.float32)
    over_capacity_ratio = over_capacity / jnp.maximum(jnp.array(1.0, dtype=jnp.float32), denom)
    return over_capacity_ratio


# pytype: disable=bad-return-type
def _compute_expert_capacity(
    *,
    group_size: int,
    num_experts: int,
    expert_capacity: Optional[int],
    capacity_factor: Optional[float],
) -> int:
    """Computes the final expert capacity."""
    if not (capacity_factor or expert_capacity):
        raise ValueError("At least one of `capacity_factor` or `expert_capacity` needs to be set.")
    if capacity_factor:
        # Determine expert capacity automatically depending on the input size.
        auto_expert_capacity = int(group_size * capacity_factor / num_experts)
        if expert_capacity is None:
            expert_capacity = 1
        if expert_capacity < auto_expert_capacity:
            expert_capacity = auto_expert_capacity
            logging.info(
                "Setting expert_capacity=%r (capacity_factor=%r group_size=%r num_experts=%r)",
                expert_capacity,
                capacity_factor,
                group_size,
                num_experts,
            )
    return expert_capacity


# pytype: enable=bad-return-type


def _cap_logits(logits: Tensor, gating_logit_cap: float) -> Tensor:
    if gating_logit_cap > 0.0:
        cap = jnp.array(gating_logit_cap, dtype=logits.dtype)
        logits = cap * jnp.tanh(logits / cap)
    return logits


def get_outer_batch_from_mesh(
    *,
    mesh_axis_names: Sequence[str],
    outer_batch_axis_names: Sequence[str],
    mesh_shape: Optional[Union[MeshShape, HybridMeshShape]],
) -> Optional[int]:
    """Infer MoE outer batch size from mesh shape.

    Args:
        mesh_axis_names: The name of each mesh axis.
        outer_batch_axis_names: The names of the mesh axes corresponding to the outer batch size.
        mesh_shape: The size of each mesh axis corresponding to `mesh_axis_names`.
            If None, the returned outer batch size will also be None.

    Returns:
        The MoE outer batch size. Will be None if `mesh_shape` is None.
    """
    if mesh_shape is None:
        return None

    ici_mesh_shape = (
        mesh_shape.ici_mesh_shape if isinstance(mesh_shape, HybridMeshShape) else mesh_shape
    )
    try:
        ici_mesh_shape = infer_mesh_shape(ici_mesh_shape)
    except ValueError as e:
        # It could happen when running in local, the number of devices can be smaller than the
        # required number of devices from the mesh shape.
        logging.info(e)

    if isinstance(mesh_shape, HybridMeshShape):
        if -1 in mesh_shape.dcn_mesh_shape:
            # TODO(markblee): Improve support for this. At the moment it is not a use-case.
            raise NotImplementedError(
                "Unable to infer number of granules. Please specify dcn_mesh_shape without -1."
            )
        mesh_shape = tuple(x * y for x, y in zip(ici_mesh_shape, mesh_shape.dcn_mesh_shape))
    else:
        mesh_shape = ici_mesh_shape

    return reduce(
        lambda x, y: x * y,
        [mesh_shape[mesh_axis_names.index(el)] for el in outer_batch_axis_names],
    )


class AdaptiveLoadBalanceLoss(BaseLayer):
    """A layer to adjust the aux loss weight based on the overcapacity ratio.

    It maintains a moving average of the overcapacity ratios seen during training and adjusts
    the scale if the average overcapacity ratio falls outside the target range.
    """

    @config_class
    class Config(BaseLayer.Config):
        moving_average: MovingAverage.Config = MovingAverage.default_config()
        max_value: Required[float] = REQUIRED
        min_value: float = 1e-3
        # When adjusting the aux loss scale, increase or decrease by a factor of e^log_step.
        log_step: float = 0.01

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("value_average", cfg.moving_average)

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        return {
            "log_scale": ParameterSpec(
                shape=[],
                dtype=jnp.float32,
                mesh_axes=(None,),
                initializer=constant_initializer(0.0),
                weight_decay_scale=0,
            ),
        }

    def forward(self, value: Tensor) -> Tensor:
        """Adjusts and returns the loss scale based on a moving average of `value`.

        The adjustments are set in OutputCollection.state_updates and won't take effect until
        the state updates are applied.

        Args:
            value: The observed value (e.g., representing the MoE over-capacity ratio).

        Returns:
            The new loss scale, a scalar with a positive float value.
        """
        cfg = self.config
        value = self.value_average(value)
        self.add_summary("value_average", value)
        # Increase log_scale if x > max_value;
        # Decrease log_scale if x < min_value;
        # Otherwise keep log_scale unchanged.
        inc = jnp.greater(value, cfg.max_value).astype(jnp.float32)
        dec = jnp.less(value, cfg.min_value).astype(jnp.float32)
        new_log_scale = self.parameters["log_scale"] + (inc - dec) * cfg.log_step
        self.add_state_update("log_scale", new_log_scale)
        scale = jnp.exp(new_log_scale)
        self.add_summary("scale", scale)
        return scale


class BaseGating(BaseLayer):
    """An abstract class to define the common interface of gating layers.

    Dimensions:
        O: outer batch size
        G: number of groups
        S: per group size
        E: number of experts
        C: capacity per expert
        M: model_dim (same as input_dim and output_dim as in FF layer)
        B: original batch dim
        L: original seq len dim
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures BaseGating."""

        num_experts: Required[int] = REQUIRED

    class Output(NamedTuple):
        # A OG`SEC tensor for combining expert outputs.
        combine_tensor: Tensor
        # A OG`SEC tensor, scattering/dispatching inputs to experts.
        dispatch_tensor: Tensor
        # Load balance loss, for equalizing the expert assignment ratios.
        load_balance_loss: Optional[Tensor] = None
        # Router z loss, for encouraging router logits to remain small.
        router_z_loss: Optional[Tensor] = None

    def forward(self, logits: Tensor) -> NestedTensor:
        """Forward pass of gating.

        Args:
            logits: a tensor of shape OG`SE.

        Returns:
            BaseGating.Output.
        """
        raise NotImplementedError(type(self))

    @nowrap
    def dispatch(
        self,
        inputs: Tensor,
        *,
        dispatch_tensor: Tensor,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_spec: PartitionSpec = PartitionSpec(None),
        combine_tensor: Optional[Tensor] = None,
    ) -> Tensor:
        """Dispatch the input tensors according to dispatch_tensor.

        Args:
            inputs: A tensor with shape [..., G, S, M].
            dispatch_tensor: A tensor with shape [..., G, S, E, C].
            dtype: The required dtype for dispatch tensor.
            partition_spec: Partition spec of dispatch tensor.
            combine_tensor: Optional, may be used by some implementations.

        Returns:
            A dispatched tensor with shape [..., E, G, C, M].
        """
        raise NotImplementedError(type(self))

    @nowrap
    def combine(
        self,
        inputs: Tensor,
        *,
        combine_tensor: Tensor,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_spec: PartitionSpec = PartitionSpec(None),
    ) -> Tensor:
        """Take weighted average / combination of dispatched tensors.

        Args:
            inputs: A tensor with shape [O, G, E, C, M].
            combine_tensor: A tensor with shape [O, G, S, E, C].
            dtype: The required dtype for combine tensor.
            partition_spec: Partition spec of combine tensor.

        Returns:
            A tensor with shape [..., G, S, M].
        """
        raise NotImplementedError(type(self))


class Top2Gating(BaseGating):
    """Computes Top-2 gating for Mixture-of-Experts.

    The methods take gating logits, potentially sharded across tpu cores as inputs.
    We rely on sharding propagation to work universally. Dispatch and combine tensors
    should be explicitly annotated with `utils.with_sharding_constraint` by the caller.

    We perform dispatch/combine via einsum.

    Note that for local_dispatch, the original batch BLM is reshaped to OGSM. There are
    O*G groups and each group is being dispatched independently.

    Reference:
    https://github.com/google/praxis/blob/f8467c730ccac1bf2cf10a68fb18f9e6e1f658b4/praxis/gshard_utils.py#L87
    """

    @config_class
    class Config(BaseGating.Config):
        """Configures Top2Gating."""

        # Soft cap, applied for gating logits, this is a stability fix to avoid extreme values
        # during initial steps. Defaults to 0.0.
        gating_logit_cap: float = 0.0
        # Note using bfloat16 for fprop_dtype could be problematic for mask tensors. Reference:
        # https://github.com/google/praxis/blob/2d85369a6cb04161fa5be88c6669454ff1f60574/praxis/gshard_utils.py#L849
        mask_dtype: jnp.dtype = jnp.int32
        # Set expert_capacity to at least (group_size * capacity_factor) / num_experts. Default
        # to 2.0 for top-2 gating.
        train_capacity_factor: float = 2.0
        eval_capacity_factor: float = 2.0
        # Number of examples per minibatch/group per expert. Each example is typically a vector
        # of size input_dim, representing embedded token or an element of Transformer layer output.
        expert_capacity: Optional[int] = None

        # If not None, adjust the aux loss according to recent over-capacity ratios.
        adaptive_load_balance_loss: Optional[AdaptiveLoadBalanceLoss.Config] = None

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.adaptive_load_balance_loss is not None:
            self._add_child("adaptive_load_balance_loss", cfg.adaptive_load_balance_loss)

    # pylint: disable-next=too-many-statements
    def forward(self, logits: Tensor) -> NestedTensor:
        """Please see comments of BaseGating.forward."""
        cfg = self.config
        if logits.dtype != jnp.float32:
            logits = logits.astype(jnp.float32)
        logits = _cap_logits(logits, cfg.gating_logit_cap)
        raw_gates = jax.nn.softmax(logits, axis=-1)  # along E dim

        expert_capacity = _compute_expert_capacity(
            expert_capacity=cfg.expert_capacity,
            capacity_factor=(
                cfg.train_capacity_factor if self.is_training else cfg.eval_capacity_factor
            ),
            group_size=logits.shape[-2],
            num_experts=cfg.num_experts,
        )

        # top-1 index: OGS tensor.
        index_1 = jnp.argmax(raw_gates, axis=-1)
        # OGSE tensor.
        mask_1 = jax.nn.one_hot(index_1, raw_gates.shape[-1], dtype=cfg.mask_dtype)

        gate_1 = jnp.einsum("ogse,ogse->ogs", raw_gates, mask_1.astype(raw_gates.dtype))
        gates_without_top_1 = jnp.where(mask_1, 0.0, raw_gates)

        # Greedily pick the 2nd expert.
        index_2 = jnp.argmax(gates_without_top_1, axis=-1)

        mask_2 = jax.nn.one_hot(index_2, cfg.num_experts, dtype=cfg.mask_dtype)
        gate_2 = jnp.einsum(
            "ogse,ogse->ogs", gates_without_top_1, mask_2.astype(gates_without_top_1.dtype)
        )

        # Renormalize.
        denom = gate_1 + gate_2 + 1e-9
        gate_1 /= denom
        gate_2 /= denom

        # We reshape the mask as [OGS, E], and compute cumulative sums of assignment
        # indicators for each expert index e \in 0..E-1 independently.
        # cumsum over S dim: mask_1 is OGSE tensor.
        position_in_expert_1 = _cum_sum(mask_1, exclusive=True, axis=-2)

        # OGE tensor (reduce S out of OGSE tensor mask_1).
        # density_1[:, e] represents assignment ratio (num assigned / total) to
        # expert e as top_1 expert without taking capacity into account.
        density_denom = jnp.asarray(1.0, dtype=jnp.float32)

        density_1 = jnp.mean(mask_1.astype(jnp.float32), axis=-2) / density_denom
        # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
        # those of examples not assigned to e with top-2 gating.
        density_1_proxy = jnp.mean(raw_gates, axis=-2, dtype=jnp.float32) / density_denom

        # Compute aux_loss.
        aux_loss = jnp.mean(density_1_proxy * density_1, dtype=jnp.float32)
        aux_loss *= cfg.num_experts * cfg.num_experts

        # Add the over capacity ratio for expert 1.
        over_capacity_1 = _create_over_capacity_ratio_summary(
            mask=mask_1,
            position_in_expert=position_in_expert_1,
            capacity=expert_capacity,
        )

        mask_1 *= jnp.less(position_in_expert_1, expert_capacity).astype(mask_1.dtype)
        position_in_expert_1 = jnp.einsum("ogse,ogse->ogs", position_in_expert_1, mask_1)

        # How many examples in this sequence go to this expert?
        mask_1_count = jnp.einsum("ogse->oge", mask_1)
        # [batch, group] - mostly ones, but zeros where something didn't fit.
        mask_1_flat = jnp.sum(mask_1, axis=-1, dtype=cfg.mask_dtype)
        assert mask_1_count.dtype == cfg.mask_dtype
        assert mask_1_flat.dtype == cfg.mask_dtype

        position_in_expert_2 = _cum_sum(mask_2, exclusive=True, axis=-2) + jnp.expand_dims(
            mask_1_count, -2
        )
        # Add the over capacity ratio for expert 2.
        over_capacity_2 = _create_over_capacity_ratio_summary(
            mask=mask_2,
            position_in_expert=position_in_expert_2,
            capacity=expert_capacity,
        )

        mask_2 *= jnp.less(position_in_expert_2, expert_capacity).astype(mask_2.dtype)
        position_in_expert_2 = jnp.einsum("ogse,ogse->ogs", position_in_expert_2, mask_2)
        mask_2_flat = jnp.sum(mask_2, axis=-1, dtype=cfg.mask_dtype)

        gate_1 *= mask_1_flat.astype(gate_1.dtype)
        gate_2 *= mask_2_flat.astype(gate_2.dtype)

        # OGSC tensor.
        b = jax.nn.one_hot(position_in_expert_1, expert_capacity, dtype=jnp.float32)
        # OGSE tensor.
        a = jnp.expand_dims(gate_1 * mask_1_flat.astype(jnp.float32), axis=-1) * jax.nn.one_hot(
            index_1, cfg.num_experts, dtype=jnp.float32
        )
        # OGSEC tensor.
        first_part_of_combine_tensor = jnp.einsum("ogse,ogsc->ogsec", a, b)

        # OGSC tensor.
        b = jax.nn.one_hot(position_in_expert_2, expert_capacity, dtype=jnp.float32)
        # OGSE tensor
        a = jnp.expand_dims(gate_2 * mask_2_flat, axis=-1) * jax.nn.one_hot(
            index_2, cfg.num_experts, dtype=jnp.float32
        )
        second_part_of_combine_tensor = jnp.einsum("ogse,ogsc->ogsec", a, b)
        # OGSEC tensor.
        combine_tensor = first_part_of_combine_tensor + second_part_of_combine_tensor
        # OGSEC tensor.
        dispatch_tensor = combine_tensor.astype(bool)

        # Counts for tokens that are dispatched to 0, 1 and 2 experts.
        dispatch_count_tensor = jnp.sum(dispatch_tensor.astype(jnp.int32), [-2, -1])
        dispatch_0 = jnp.sum(dispatch_count_tensor == 0)
        dispatch_1 = jnp.sum(dispatch_count_tensor == 1)
        dispatch_2 = jnp.sum(dispatch_count_tensor == 2)

        router_z_loss = _router_z_loss(logits)

        # Adding auxiliary losses and gating statistics to job summary.
        self.add_summary("load_balance_loss", aux_loss)
        self.add_summary("router_z_loss", router_z_loss)
        self.add_summary("dispatch_0", dispatch_0)
        self.add_summary("dispatch_1", dispatch_1)
        self.add_summary("dispatch_2", dispatch_2)
        self.add_summary("over_capacity_1", over_capacity_1)
        self.add_summary("over_capacity_2", over_capacity_2)

        if cfg.adaptive_load_balance_loss is None:
            self.add_summary("load_balance_loss", aux_loss)
        else:
            self.add_summary("load_balance_loss_original", aux_loss)
            aux_loss *= self.adaptive_load_balance_loss(
                jnp.maximum(over_capacity_1, over_capacity_2)
            )
            self.add_summary("load_balance_loss", aux_loss)

        return self.Output(
            combine_tensor=combine_tensor,
            dispatch_tensor=dispatch_tensor,
            load_balance_loss=aux_loss,
            router_z_loss=router_z_loss,
        )

    @nowrap
    def dispatch(
        self,
        inputs: Tensor,
        *,
        dispatch_tensor: Tensor,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_spec: PartitionSpec = PartitionSpec(None),
        combine_tensor: Optional[Tensor] = None,
    ) -> Tensor:
        """Dispatch the input tensors according to dispatch_tensor.

        Args:
            inputs: A tensor with shape [..., G, S, M].
            dispatch_tensor: A tensor with shape [..., G, S, E, C].
            dtype: The required dtype for dispatch tensor.
            partition_spec: Partition spec of dispatch tensor.
            combine_tensor: Optional, not used in this implementation but kept for compatibility.

        Returns:
            A dispatched tensor with shape [..., E, G, C, M].
        """
        del combine_tensor  # Unused in einsum-based dispatch
        dispatch_tensor = dispatch_tensor.astype(dtype)
        dispatch_tensor = with_sharding_constraint(dispatch_tensor, partition_spec)
        return jnp.einsum("ogsec,ogsm->oegcm", dispatch_tensor, inputs)

    @nowrap
    def combine(
        self,
        inputs: Tensor,
        *,
        combine_tensor: Tensor,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_spec: PartitionSpec = PartitionSpec(None),
    ) -> Tensor:
        """Take weighted average / combination of dispatched tensors.

        Args:
            inputs: A tensor with shape [O, G, E, C, M].
            combine_tensor: A tensor with shape [O, G, S, E, C].
            dtype: The required dtype for combine tensor.
            partition_spec: Partition spec of combine tensor.

        Returns:
            A tensor with shape [..., G, S, M].
        """
        combine_tensor = combine_tensor.astype(dtype)
        combine_tensor = with_sharding_constraint(combine_tensor, partition_spec)
        return jnp.einsum("ogecm,ogsec->ogsm", inputs, combine_tensor)


class TopKGating(BaseGating):
    """Generalized Top-K gating for Mixture-of-Experts.

    Enhanced with configurable top-k selection, scoring functions, and noisy gating.
    """

    @config_class
    class Config(BaseGating.Config):
        """Configures TopKGating."""

        # Number of selected experts per token.
        num_experts_per_token: int = 2

        # Soft cap, applied for gating logits, this is a stability fix to avoid extreme values
        # during initial steps. Defaults to 0.0 for backward compatibility.
        gating_logit_cap: float = 0.0

        # Set expert_capacity to at least (group_size * capacity_factor) / num_experts.
        # Default to None, which will use num_experts_per_token as the capacity factor.
        train_capacity_factor: Optional[float] = None
        eval_capacity_factor: Optional[float] = None

        # Number of examples per minibatch/group per expert. Each example is typically a vector
        # of size input_dim, representing embedded token or an element of Transformer layer output.
        # When expert_capacity == None, it will be automatically calculated by
        # (group_size * capacity_factor) / num_experts.
        expert_capacity: Optional[int] = None

        # If not None, adjust the aux loss according to recent over-capacity ratios.
        adaptive_load_balance_loss: Optional[AdaptiveLoadBalanceLoss.Config] = None

        # Noisy Gating. If None, no noise will be added to the gating logits.
        noisy_gating: Optional[GateNoise] = None
        # If None, use jax.lax.top_k for the precise top-k selection.
        # Otherwise, it can be set to config_for_function(approx_max_k).set(recall_target=0.95)
        # for a faster but approximate selection.
        topk_fn: Optional[ConfigOr[TopKFn]] = None
        # Scoring function used to transform logits to gates.
        # If None, uses softmax. Can be set to config_for_function(sigmoid) for sigmoid scoring.
        score_fn: Optional[ConfigOr[ScoreFn]] = None

    @classmethod
    def dispatch_tensor_shape(cls):
        """Returns the shape specification for the dispatch tensor."""
        return "ogsec"

    @classmethod
    def combine_tensor_shape(cls):
        """Returns the shape specification for the combine tensor."""
        return "ogsec"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.adaptive_load_balance_loss is not None:
            self._add_child("adaptive_load_balance_loss", cfg.adaptive_load_balance_loss)

    def _add_gumbel_noise(self, logits: Tensor) -> Tensor:
        """Adds Gumbel noise to logits for noisy gating."""
        _, subkey = jax.random.split(self.prng_key)
        # Generates standard Gumbel(0, 1) noise.
        noise = jax.random.gumbel(subkey, logits.shape, dtype=logits.dtype)
        return logits + noise

    def _top_k(self, raw_gates: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """Selects top-k experts, using configured topk_fn if available."""
        cfg = self.config
        if cfg.topk_fn:
            topk_fn = maybe_instantiate(cfg.topk_fn)
            return topk_fn(raw_gates, k=k)
        return jax.lax.top_k(raw_gates, k=k)

    def _score(self, logits: Tensor, axis: int = -1) -> Tensor:
        """Computes scores from logits using configured score_fn or default softmax."""
        cfg = self.config
        if cfg.score_fn:
            score_fn = maybe_instantiate(cfg.score_fn)
            return score_fn(logits, axis=axis)
        return jax.nn.softmax(logits, axis=axis)

    def _get_normalized_gates(self, raw_gates: Tensor) -> Tensor:
        """Gets normalized gates. If using softmax, already normalized; otherwise normalize."""
        cfg = self.config
        if not cfg.score_fn:
            # Default softmax function already makes raw_gates normalized.
            return raw_gates
        else:
            return raw_gates / raw_gates.sum(axis=-1, keepdims=True)

    def _process_logits(self, logits: Tensor) -> Tensor:
        """Converts input logits into float32, caps values, and optionally adds noise."""
        cfg = self.config
        if logits.dtype != jnp.float32:
            logging.info("Upcasting gating logits to float32")
            logits = logits.astype(jnp.float32)

        # Helpful diagnostic.
        rms_logits = jnp.sqrt(jnp.mean(logits**2))
        self.add_summary("router_logit_rms", rms_logits)
        logits = _cap_logits(logits, cfg.gating_logit_cap)

        # Noised Top-K selection.
        if cfg.noisy_gating is not None:
            if cfg.noisy_gating == GateNoise.GUMBEL:
                logits = self._add_gumbel_noise(logits)
            else:
                raise NotImplementedError(cfg.noisy_gating)

        return logits

    def _get_expert_capacity(self, *, group_size: int) -> int:
        """Computes expert capacity based on configuration."""
        cfg = self.config
        capacity_factor = (
            cfg.train_capacity_factor if self.is_training else cfg.eval_capacity_factor
        )
        if capacity_factor is None:
            # Use num_experts_per_token as the default.
            capacity_factor = float(cfg.num_experts_per_token)
        expert_capacity = _compute_expert_capacity(
            expert_capacity=cfg.expert_capacity,
            capacity_factor=capacity_factor,
            group_size=group_size,
            num_experts=cfg.num_experts,
        )
        return expert_capacity

    def _load_balance_loss(
        self,
        *,
        raw_gates: Tensor,
        gate_assignment: Tensor,
        num_experts_per_token: int,
    ) -> Tensor:
        """Calculates the load balance loss.

        Note this may include the padding tokens. Given the batch is packed, the impacts of the
        padded tokens should be much smaller than that of the majority of the valid tokens.

        Arguments:
            raw_gates: A tensor with shape [O, G, S, E].
            gate_assignment: A tensor with shape [O, G, S, K].
            num_experts_per_token: the number of experts selected per token.

        Returns:
            A scalar tensor representing the load balance loss.
        """
        cfg = self.config
        num_experts = cfg.num_experts
        normalized_gates = self._get_normalized_gates(raw_gates)
        # [O, G, S, K, E]
        assign_indicator = jax.nn.one_hot(gate_assignment, num_experts, dtype=jnp.int32)
        # Check if a token has been assigned to any expert.
        # [O, G, S, E]
        assign_indicator = jnp.max(assign_indicator, axis=-2)
        # [O, G, E]
        # Note this density includes the tokens that may be dropped later due to over capacity.
        density = jnp.mean(assign_indicator, axis=-2, dtype=jnp.float32)
        # [O, G, E]
        density_proxy = jnp.mean(normalized_gates, axis=-2, dtype=jnp.float32)

        return jnp.mean(density * density_proxy) * num_experts**2 / num_experts_per_token

    # pylint: disable-next=too-many-statements
    def forward(self, logits: Tensor) -> NestedTensor:
        """Please see comments of BaseGating.forward.

        Args:
            logits: A tensor with shape [O, G, S, E].

        Returns:
            BaseGating.Output with:
                - combine_tensor: [O, G, S, E, C]
                - dispatch_tensor: [O, G, S, E, C]
                - load_balance_loss: scalar
                - router_z_loss: scalar
        """
        cfg = self.config
        num_experts_per_token = cfg.num_experts_per_token

        # Process logits: upcast to float32, cap, and optionally add noise.
        # [O, G, S, E]
        logits = self._process_logits(logits)

        # Get the router z-loss.
        router_z_loss = _router_z_loss(logits)

        # Compute scores from logits (softmax by default, or custom score_fn).
        # [O, G, S, E]
        raw_gates = self._score(logits, axis=-1)  # along E dim

        # Get the expert capacity.
        expert_capacity = self._get_expert_capacity(group_size=logits.shape[-2])

        # Select top-k experts for each token using configurable top-k function.
        # gate_weights: [O, G, S, K], gate_assignment: [O, G, S, K]
        gate_weights, gate_assignment = self._top_k(raw_gates, k=num_experts_per_token)

        # Get the expert load balance loss.
        # This considers the load balance of all the top-k selected experts.
        load_balance_loss = self._load_balance_loss(
            raw_gates=raw_gates,
            gate_assignment=gate_assignment,
            num_experts_per_token=num_experts_per_token,
        )

        # Calculate the normalization factor for the selected experts.
        # [O, G, S, 1]
        denom = jnp.sum(gate_weights, axis=-1, keepdims=True)

        # Reshape gate_assignment from [O, G, S, K] to [O, G, K, S] then flatten to [O, G, K*S]
        gate_assignment = jnp.swapaxes(gate_assignment, 2, 3)
        gate_assignment = gate_assignment.reshape(
            (gate_assignment.shape[0], gate_assignment.shape[1], -1)
        )

        # One-hot encoding of the selected experts.
        # [O, G, K x S, E]
        gate_assignment_one_hot = jax.nn.one_hot(gate_assignment, cfg.num_experts)

        # Compute position of each token in each expert using cumulative sum.
        # [O, G, K x S, E]
        position_in_each_expert = _cum_sum(
            gate_assignment_one_hot, exclusive=False, axis=-2
        ).astype(jnp.int32)

        # Get the actual capacity needed for each expert (before dropping).
        # [O, G, E]
        real_expert_capacity = position_in_each_expert[:, :, -1, :]

        # Calculate the over capacity ratio for the given batch.
        over_capacity_ratio = jnp.sum(
            jnp.maximum(0, real_expert_capacity - expert_capacity)
        ) / jnp.maximum(1, jnp.sum(real_expert_capacity))
        self.add_summary("over_capacity_ratio", over_capacity_ratio)

        # Adjust positions: subtract 1 and zero out non-selected experts.
        # [O, G, K x S, E]
        position_in_each_expert = position_in_each_expert * gate_assignment_one_hot - 1

        # A token can only be routed to its selected expert when its position in that expert
        # is less than the expert_capacity. Otherwise, it will be dropped.
        # The jax.nn.one_hot operation will assign 0 to values beyond expert_capacity.
        # [O, G, K x S, E, C]
        position_in_each_expert_indicator = jax.nn.one_hot(position_in_each_expert, expert_capacity)

        # Reshape to separate K dimension: [O, G, K, S, E, C]
        position_in_each_expert_indicator_shape = position_in_each_expert_indicator.shape
        position_in_each_expert_indicator = position_in_each_expert_indicator.reshape(
            (
                position_in_each_expert_indicator_shape[0],
                position_in_each_expert_indicator_shape[1],
                num_experts_per_token,
                -1,
                position_in_each_expert_indicator_shape[3],
                position_in_each_expert_indicator_shape[4],
            )
        )

        # Sum over K to get dispatch tensor: [O, G, S, E, C]
        dispatch_tensor = jnp.sum(position_in_each_expert_indicator, axis=2)

        # Compute combine tensor by weighting the dispatch tensor.
        # We need to normalize raw_gates by the sum of selected gate weights.
        # [O, G, S, E]
        expert_weights = raw_gates / denom
        # [O, G, S, E, 1]
        expert_weights = jnp.expand_dims(expert_weights, axis=-1)
        # [O, G, S, E, C]
        combine_tensor = dispatch_tensor * expert_weights

        # Counts for tokens that are dispatched to 0, 1, ..., num_experts_per_token experts.
        # [O, G, S]
        dispatch_count_tensor = jnp.sum(dispatch_tensor.astype(jnp.int32), [-2, -1])

        # Add auxiliary losses and gating statistics to job summary.
        self.add_summary("load_balance_loss", load_balance_loss)
        self.add_summary("router_z_loss", router_z_loss)

        # Summary for number of tokens dispatched to 0, 1, ..., k experts.
        for i in range(num_experts_per_token + 1):
            dispatch_i = jnp.sum(dispatch_count_tensor == i)
            self.add_summary(f"dispatch_{i}", dispatch_i)

        # Adaptive load balance loss if configured.
        if cfg.adaptive_load_balance_loss is None:
            self.add_summary("load_balance_loss", load_balance_loss)
        else:
            self.add_summary("load_balance_loss_original", load_balance_loss)
            load_balance_loss *= self.adaptive_load_balance_loss(over_capacity_ratio)
            self.add_summary("load_balance_loss", load_balance_loss)

        return self.Output(
            combine_tensor=combine_tensor,
            dispatch_tensor=dispatch_tensor,
            load_balance_loss=load_balance_loss,
            router_z_loss=router_z_loss,
        )

    @nowrap
    def dispatch(
        self,
        inputs: Tensor,
        *,
        dispatch_tensor: Tensor,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_spec: PartitionSpec = PartitionSpec(None),
        combine_tensor: Optional[Tensor] = None,
    ) -> Tensor:
        """Dispatch the input tensors according to dispatch_tensor.

        Args:
            inputs: A tensor with shape [..., G, S, M].
            dispatch_tensor: A tensor with shape [..., G, S, E, C].
            dtype: The required dtype for dispatch tensor.
            partition_spec: Partition spec of dispatch tensor.
            combine_tensor: Optional, not used in this implementation but kept for compatibility.

        Returns:
            A dispatched tensor with shape [..., E, G, C, M].
        """
        del combine_tensor  # Unused in einsum-based dispatch
        dispatch_tensor = dispatch_tensor.astype(dtype)
        dispatch_tensor = with_sharding_constraint(dispatch_tensor, partition_spec)
        return jnp.einsum("ogsec,ogsm->oegcm", dispatch_tensor, inputs)

    @nowrap
    def combine(
        self,
        inputs: Tensor,
        *,
        combine_tensor: Tensor,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_spec: PartitionSpec = PartitionSpec(None),
    ) -> Tensor:
        """Take weighted average / combination of dispatched tensors.

        Args:
            inputs: A tensor with shape [O, G, E, C, M].
            combine_tensor: A tensor with shape [O, G, S, E, C].
            dtype: The required dtype for combine tensor.
            partition_spec: Partition spec of combine tensor.

        Returns:
            A tensor with shape [..., G, S, M].
        """
        combine_tensor = combine_tensor.astype(dtype)
        combine_tensor = with_sharding_constraint(combine_tensor, partition_spec)
        return jnp.einsum("ogecm,ogsec->ogsm", inputs, combine_tensor)


class TopKDropFreeGating(TopKGating):
    """Computes Token-drop free Top-K gating for Mixture-of-Experts."""

    class Output(NamedTuple):
        """Output of TopKDropFreeGating."""

        # A [B, S, K] tensor for expert assignments.
        gate_assignment: Tensor
        # A [B, S, K] tensor for the weight of each selected expert.
        expert_weights: Tensor
        # Load balance loss, for equalizing the expert assignment ratios.
        load_balance_loss: Optional[Tensor] = None
        # Router z loss, for encouraging router logits to remain small.
        router_z_loss: Optional[Tensor] = None
        # Seq Load balance loss, for equalizing the expert assignment ratios per sequence.
        seq_load_balance_loss: Optional[Tensor] = None

    # pylint: disable-next=no-self-use
    def _load_balance_loss(
        self,
        *,
        raw_gates: Tensor,
        gate_assignment: Tensor,
        num_experts_per_token: int,
    ) -> Tensor:
        """Calculates the load balance loss.

        Note this may include the padding tokens. Given the batch is packed,
        the impacts of the padded tokens should be much smaller than that of the
        majority of the valid tokens.

        Arguments:
            raw_gates: A tensor with shape [B, S, E].
            gate_assignment: A tensor with shape [B, S, K].
            num_experts_per_token: the number of experts selected per token.

        Returns:
            A scalar tensor representing the load balance loss.
        """
        del num_experts_per_token
        normalized_gates = self._get_normalized_gates(raw_gates)
        num_experts = normalized_gates.shape[-1]
        density = jnp.bincount(gate_assignment.reshape((-1)), length=num_experts) / np.prod(
            gate_assignment.shape
        )
        density_proxy = jnp.mean(normalized_gates.reshape([-1, num_experts]), axis=-2)
        return jnp.mean(density * density_proxy) * num_experts**2

    # pylint: disable-next=no-self-use
    def _seq_load_balance_loss(
        self,
        *,
        raw_gates: Tensor,
        gate_assignment: Tensor,
    ) -> Tensor:
        """Calculates the sequence wise load balance loss.

        Note this may include the padding tokens. Given the batch is packed,
        the impacts of the padded tokens should be much smaller than that of the
        majority of the valid tokens.

        Ref: https://arxiv.org/pdf/2412.19437 (eq 17-20)
        Code:
        https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py#L475-L486

        Arguments:
            raw_gates: A float tensor with shape [B, S, E], where raw_gates[b, s, e] represents
              the score of token[b, s] for expert e.
            gate_assignment: An integer tensor with shape [B, S, K] of value [0, top_k), where
              gate_assignment[b, s, k] represents the expert id for the k'th expert for token[b, s].

        Returns:
            A scalar tensor representing the sequence wise load balance loss.
        """
        batch_size, seq_len, num_experts = raw_gates.shape
        topk = gate_assignment.shape[-1]
        # [B, E], each value representing the number of tokens per expert per sequence.
        num_tokens_per_expert = jax.vmap(lambda x: jnp.bincount(x, length=num_experts))(
            gate_assignment.reshape(batch_size, -1)
        )
        density = num_tokens_per_expert * num_experts / (seq_len * topk)  # eq 18
        normalized_gates = self._get_normalized_gates(raw_gates)  # eq 19
        # [B, E]. density_proxy[b, e] represents the total amount of normalized gating weights
        # for expert 'e'.
        density_proxy = jnp.mean(normalized_gates, axis=1)  # eq 20
        return (density * density_proxy).sum(axis=1).mean()  # eq 17 without the alpha factor

    # pylint: disable-next=too-many-statements
    def forward(
        self, logits: Tensor, seq_load_balance_loss_weight: Optional[float] = None
    ) -> Output:
        cfg = self.config
        if logits.dtype != jnp.float32:
            self.vlog(3, "Upcasting gating logits")
            logits = logits.astype(jnp.float32)
        logits = _cap_logits(logits, cfg.gating_logit_cap)
        # Noised Top-K selection.
        if cfg.noisy_gating is not None:
            if cfg.noisy_gating == GateNoise.GUMBEL:
                logits = self._add_gumbel_noise(logits)
            else:
                raise NotImplementedError(cfg.noisy_gating)

        # Get the router z-loss.
        router_z_loss = _router_z_loss(logits)
        self.add_summary("router_z_loss", router_z_loss)

        # [B, S, E]
        raw_gates = self._score(logits, axis=-1)  # along E dim if needed.

        # [B, S, K], [B, S, K]
        gate_weights, gate_assignment = self._top_k(raw_gates, k=cfg.num_experts_per_token)
        # Get the expert load balance loss.
        # This considers the load balance of all the top-k selected experts.
        load_balance_loss = self._load_balance_loss(
            raw_gates=raw_gates,
            gate_assignment=gate_assignment,
            num_experts_per_token=cfg.num_experts_per_token,
        )
        self.add_summary("load_balance_loss", load_balance_loss)
        if seq_load_balance_loss_weight:
            # Get the expert sequence load balance loss.
            seq_load_balance_loss = self._seq_load_balance_loss(
                raw_gates=raw_gates,
                gate_assignment=gate_assignment,
            )
            self.add_summary("seq_load_balance_loss", seq_load_balance_loss)
        else:
            seq_load_balance_loss = 0
        # Caculate the normalization factor.
        denom = jnp.sum(gate_weights, axis=-1, keepdims=True)
        # Renormalize the gates of the selected expert.
        # [B, S, K]
        expert_weights = gate_weights / denom
        # [B, S, K], [B, S, K]
        return self.Output(
            gate_assignment=gate_assignment,
            expert_weights=expert_weights,
            router_z_loss=router_z_loss,
            load_balance_loss=load_balance_loss,
            seq_load_balance_loss=seq_load_balance_loss,
        )


class TopKBiasGating(TopKDropFreeGating):
    """An implementation of gate with Auxiliary-Loss-Free Load Balancing strategy.

    Ref: DeepSeek V3 - https://arxiv.org/abs/2412.19437
    """

    @config_class
    class Config(TopKDropFreeGating.Config):
        """Config for TopKBiasGating."""

        # Gating bias update rate from Table 3 of https://openreview.net/pdf?id=y1iU5czYpE
        gating_update_rate: float = 1e-3
        # https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/config.json
        routed_scaling_factor: float = 2.5
        # The parameters num_group_of_experts and topk_group are used for Node-Limited Routing,
        # which isn't required for TPU operations. We set these values to 1 to disable this
        # functionality while maintaining them for backward compatibility.
        # ref: https://arxiv.org/pdf/2412.19437 section 2.1.2
        num_group_of_experts: int = 1
        topk_group: int = 1

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return {
            # Used for calculate the actual target sparsity during annealing.
            "gate_bias": ParameterSpec(
                shape=[cfg.num_experts],
                dtype=jnp.float32,
                initializer=constant_initializer(0.0),
            ),
        }

    def _adjust_gating_scores(self, *, raw_gates: Tensor) -> Tensor:
        """Adjusts the gating scores after the score_fn called with the loading bias."""
        return raw_gates + self.parameters["gate_bias"]

    def _update_gating_bias(self, *, gate_assignment: Tensor):
        cfg = self.config
        tokens_per_expert = jnp.bincount(gate_assignment.reshape((-1)), length=cfg.num_experts)
        tokens_per_expert = tokens_per_expert.mean() - tokens_per_expert
        updated_gating_bias = self.parameters["gate_bias"] + cfg.gating_update_rate * jnp.sign(
            tokens_per_expert
        )
        # There will be NO gradients from the main loss to update the gating bias.
        self.add_state_update("gate_bias", updated_gating_bias)

    def _top_k(self, raw_gates: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        adjusted_raw_gates = self._adjust_gating_scores(raw_gates=raw_gates)
        if cfg.topk_fn:
            topk_fn = maybe_instantiate(cfg.topk_fn)
            _, indices = topk_fn(adjusted_raw_gates, k=k)
            values = jnp.take_along_axis(raw_gates, indices, axis=-1)
            return values, indices
        _, indices = jax.lax.top_k(adjusted_raw_gates, k=k)
        values = jnp.take_along_axis(raw_gates, indices, axis=-1)
        if self.is_training:
            self._update_gating_bias(gate_assignment=indices)
        return values, indices

    def _top_k_with_two_gates(
        self, raw_gates: Tensor, adjusted_raw_gates: Tensor, k: int
    ) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        if cfg.topk_fn:
            topk_fn = maybe_instantiate(cfg.topk_fn)
            _, indices = topk_fn(adjusted_raw_gates, k=k)
            values = jnp.take_along_axis(raw_gates, indices, axis=-1)
            return values, indices
        _, indices = jax.lax.top_k(adjusted_raw_gates, k=k)
        values = jnp.take_along_axis(raw_gates, indices, axis=-1)
        return values, indices

    def forward(
        self, logits: Tensor, seq_load_balance_loss_weight: Optional[float] = None
    ) -> TopKDropFreeGating.Output:
        cfg = self.config
        if logits.dtype != jnp.float32:
            self.vlog(3, "Upcasting gating logits")
            logits = logits.astype(jnp.float32)
        logits = _cap_logits(logits, cfg.gating_logit_cap)
        B, S, E = logits.shape  # pylint: disable=invalid-name

        if cfg.num_group_of_experts == 1 and cfg.topk_group == 1:
            # Simple routing without group logic.
            router_z_loss = _router_z_loss(logits)
            self.add_summary("router_z_loss", router_z_loss)

            raw_gates = self._score(logits, axis=-1)
            gate_weights, gate_assignment = self._top_k(raw_gates, k=cfg.num_experts_per_token)
            load_balance_loss = self._load_balance_loss(
                raw_gates=raw_gates,
                gate_assignment=gate_assignment,
                num_experts_per_token=cfg.num_experts_per_token,
            )
            self.add_summary("load_balance_loss", load_balance_loss)
            if seq_load_balance_loss_weight:
                seq_load_balance_loss = self._seq_load_balance_loss(
                    raw_gates=raw_gates,
                    gate_assignment=gate_assignment,
                )
                self.add_summary("seq_load_balance_loss", seq_load_balance_loss)
            else:
                seq_load_balance_loss = 0
            denom = jnp.sum(gate_weights, axis=-1, keepdims=True)
            expert_weights = gate_weights / denom
            if cfg.routed_scaling_factor != 1:
                expert_weights *= cfg.routed_scaling_factor
            return self.Output(
                gate_assignment=gate_assignment,
                expert_weights=expert_weights,
                router_z_loss=router_z_loss,
                load_balance_loss=load_balance_loss,
                seq_load_balance_loss=seq_load_balance_loss,
            )
        else:
            # Group-based routing (Node-Limited Routing).
            logits = logits.reshape(B * S, E)
            router_z_loss = _router_z_loss(logits)
            self.add_summary("router_z_loss", router_z_loss)

            raw_gates = self._score(logits, axis=-1)
            adjusted_raw_gates = self._adjust_gating_scores(raw_gates=raw_gates)
            shape0 = raw_gates.shape[0]
            # [B x S, num_group_of_experts]
            group_scores = (
                super()
                ._top_k(
                    adjusted_raw_gates.reshape(shape0, cfg.num_group_of_experts, -1),
                    k=2,
                )[0]
                .sum(axis=-1)
            )
            # [B X S, topk_group]
            group_idx = super()._top_k(group_scores, k=cfg.topk_group)[1]
            # [B x S, num_group_of_experts]
            group_mask = jnp.zeros_like(group_scores)
            # [B x S, num_group_of_experts]
            group_mask = group_mask.at[jnp.arange(shape0)[:, None], group_idx].set(1)
            # [B x S, num_experts]
            score_mask = jnp.reshape(
                jnp.broadcast_to(
                    jnp.expand_dims(group_mask, -1),
                    (
                        shape0,
                        cfg.num_group_of_experts,
                        cfg.num_experts // cfg.num_group_of_experts,
                    ),
                ),
                (shape0, -1),
            ).astype(jnp.bool_)
            adjusted_raw_gates = jnp.where(score_mask, adjusted_raw_gates, -jnp.inf)
            gate_weights, gate_assignment = self._top_k_with_two_gates(
                raw_gates, adjusted_raw_gates, k=cfg.num_experts_per_token
            )
            if self.is_training:
                self._update_gating_bias(gate_assignment=gate_assignment)

            # add load balance loss metric.
            load_balance_loss = self._load_balance_loss(
                raw_gates=raw_gates,
                gate_assignment=gate_assignment,
                num_experts_per_token=cfg.num_experts_per_token,
            )
            self.add_summary("load_balance_loss", load_balance_loss)

            if seq_load_balance_loss_weight:
                # Reshape to [B, S, E] for sequence load balance loss.
                raw_gates = raw_gates.reshape(B, S, -1)
                seq_load_balance_loss = self._seq_load_balance_loss(
                    raw_gates=raw_gates,
                    gate_assignment=gate_assignment.reshape(B, S, -1),
                )
                self.add_summary("seq_load_balance_loss", seq_load_balance_loss)
            else:
                seq_load_balance_loss = 0
            # Caculate the normalization factor.
            denom = jnp.sum(gate_weights, axis=-1, keepdims=True)
            # Renormalize the gates of the selected expert.
            # [B x S, K]
            expert_weights = gate_weights / denom
            if cfg.routed_scaling_factor != 1:
                expert_weights *= cfg.routed_scaling_factor
            return self.Output(
                gate_assignment=gate_assignment.reshape(B, S, -1),
                expert_weights=expert_weights.reshape(B, S, -1),
                router_z_loss=router_z_loss,
                load_balance_loss=load_balance_loss,
                seq_load_balance_loss=seq_load_balance_loss,
            )


class TransformerFeedForwardMoE(DenseGeneralBaseLayer):
    """A Transformer feed-forward layer with mixture of experts.

    This is a drop-in replacement of the `TransformerFeedForwardLayer` class.

    https://github.com/google/praxis/blob/b059aa12a62a1b675d95a66088f8d0593baa48a5/praxis/layers/transformers.py#L510
    https://github.com/tensorflow/lingvo/blob/da1a75b2fea79ee542e7ae735f92032088eda055/lingvo/jax/layers/transformers.py#L415
    """

    @config_class
    class Config(DenseGeneralBaseLayer.Config):
        """Configures TransformerFeedForwardMoE."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        hidden_dim: Required[Union[int, FunctionConfigBase]] = REQUIRED  # Hidden feature dim.
        # If greater than 1, we reshape all tensors from (batch, seq_len, dim) to
        # (outer_batch, inner_batch, seq_len, dim). This is useful for 3D mesh. Reference:
        # https://github.com/tensorflow/mesh/blob/fbf7b1e547e8b8cb134e81e1cd350c312c0b5a16/mesh_tensorflow/transformer/moe.py#L294-L336
        outer_batch: int = 1
        # The normalization layer config.
        norm: BaseNormalizationLayer.Config = LayerNorm.default_config()
        activation: Union[str, tuple[str, str]] = "nn.relu"
        dropout: InstantiableConfig = Dropout.default_config()
        stochastic_depth: InstantiableConfig = StochasticDepth.default_config()
        # The inner structure of the layer: "prenorm", "postnorm", "hybridnorm", "nonorm", "v2".
        # * prenorm: y = x + feedforward(norm(x))
        # * postnorm: y = norm(x + feedforward(x))
        # * hybridnorm: y = x + postnorm(feedforward(prenorm(x)))
        # * nonorm: y = feedforward(x)   # no residual, which is usually applied externally.
        # v2: see comments NormPosition for details.
        #
        # References:
        # prenorm/postnorm: https://arxiv.org/abs/2002.04745.
        # hybridnorm: https://github.com/google/praxis/blob/main/praxis/layers/transformers.py#L273
        structure: str = "prenorm"
        # outputs = inputs + residual_weight * x. Same as
        # https://github.com/apple/axlearn/blob/521b263c10976f6caf9877a7d40dd48a7261124e/axlearn/common/attention.py#L2630
        residual_weight: float = 1.0
        num_experts: Required[int] = REQUIRED
        # Number of groups for dispatching. Typically should be the same as num devices.
        num_groups: Required[int] = REQUIRED
        # Gating function.
        gating: BaseGating.Config = Top2Gating.default_config()
        # Weight for the load balancing loss. Default to 0.01 as in
        # https://arxiv.org/pdf/2112.06905.
        load_balance_loss_weight: float = 0.01
        # Weight for the router z loss. https://arxiv.org/abs/2202.08906.
        router_z_loss_weight: float = 0.0
        # Weight for the expert correlation loss. Encourages the router to learn
        # uncorrelated representations for the experts. Default to 0 (disabled).
        corr_loss_weight: float = 0

        # SPMD partition params used to represent the MoE layer dimensions.
        # O - outer batch dim
        # M - input dim, same as output dim
        # E - experts dim
        # G - groups dim
        # C - experts capacity dim
        # H - hidden dim
        # S - sequence dim
        dim_to_mesh_axis_map: dict[str, Optional[PartitionSpec]] = {}

        # Initial value for residual gate parameter. If None, residual gating is disabled.
        # When enabled, applies learned gating between input and residual branch.
        residual_gate_init: Optional[float] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        # Table 1 of https://arxiv.org/abs/2405.15052.
        cfg.dim_to_mesh_axis_map = {
            "me": PartitionSpec(None, None),
            "emh": PartitionSpec("expert", None, "model"),
            "ehm": PartitionSpec("expert", "model", None),
            "ogsm": PartitionSpec("data", "expert", None, "model"),
            "ogsec": PartitionSpec("data", "expert", None, None, None),
            "oegcm": PartitionSpec("data", "expert", None, None, "model"),
            "ogecm": PartitionSpec("data", "expert", None, None, "model"),
            "oegch": PartitionSpec("data", "expert", None, None, "model"),
        }
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        if isinstance(cfg.hidden_dim, int):
            hidden_dim = cfg.hidden_dim
        else:
            hidden_dim = cfg.hidden_dim.set(input_dim=cfg.input_dim).instantiate()
        params = dict(
            gate_weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.num_experts),
                mesh_axes=cfg.dim_to_mesh_axis_map["me"],
            ),
            wo_weight=ParameterSpec(
                shape=(cfg.num_experts, hidden_dim, cfg.input_dim),
                mesh_axes=cfg.dim_to_mesh_axis_map["ehm"],
                fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=0),
            ),
        )
        if cfg.residual_gate_init is not None:
            params["residual_gate_theta"] = ParameterSpec(
                shape=(),
                initializer=constant_initializer(cfg.residual_gate_init),
                weight_decay_scale=0.0,
                dtype=jnp.float32,
            )
        if isinstance(cfg.activation, tuple):
            assert len(cfg.activation) == 2, cfg.activation
            # Create a wi_weight projection for each activation.
            for i in range(len(cfg.activation)):
                params[f"wi_{i}_weight"] = ParameterSpec(
                    shape=(cfg.num_experts, cfg.input_dim, hidden_dim),
                    mesh_axes=cfg.dim_to_mesh_axis_map["emh"],
                    fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=0),
                )
        else:
            assert isinstance(cfg.activation, str), cfg.activation
            params["wi_weight"] = ParameterSpec(
                shape=(cfg.num_experts, cfg.input_dim, hidden_dim),
                mesh_axes=cfg.dim_to_mesh_axis_map["emh"],
                fan_axes=FanAxes(in_axis=-2, out_axis=-1, batch_axis=0),
            )
        return params

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("gating", cfg.gating.set(num_experts=cfg.num_experts))
        self._add_child("stochastic_depth", cfg.stochastic_depth)
        # Add norm layers for different structures.

        if cfg.structure == "v2":
            if not isinstance(cfg.norm, dict):
                raise ValueError(f"When structure=v2, cfg.norm must be a dict: {cfg.norm}")
            for position, norm in cfg.norm.items():
                self._add_child(position.value, norm.set(input_dim=cfg.input_dim))
        else:
            if not isinstance(cfg.norm, InstantiableConfig):
                raise ValueError("When structure != v2, cfg.norm must be a config.")
            if cfg.structure in ["prenorm", "postnorm"]:
                self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))
            elif cfg.structure == "hybridnorm":
                self._add_child("prenorm", cfg.norm.set(input_dim=cfg.input_dim))
                self._add_child("postnorm", cfg.norm.set(input_dim=cfg.input_dim))
            elif cfg.structure == "nonorm":
                pass
            else:
                raise NotImplementedError(cfg.structure)

        # Add dropout layers for different structures.
        # Always apply two dropouts in v2 structure.
        if cfg.structure in ["prenorm", "hybridnorm", "nonorm", "v2"]:
            self._add_child("dropout1", cfg.dropout)
            self._add_child("dropout2", cfg.dropout)
        elif cfg.structure in ["postnorm"]:
            self._add_child("dropout", cfg.dropout)
        else:
            raise NotImplementedError(cfg.structure)

    def _expert_correlation_loss(self) -> Tensor:
        """Computes the correlation among the experts based on gate_weight.

        This loss encourages the router to learn uncorrelated representations for different
        experts, which can improve expert diversity and model performance.

        Returns:
            A scalar tensor representing the correlation loss.
        """
        e_w = self.parameters["gate_weight"]
        e_w_rms = jnp.sqrt(jnp.square(e_w).sum(axis=0, keepdims=True))
        e_w_normalized = e_w / e_w_rms
        e_w_cos = jnp.einsum("me,mf->ef", e_w_normalized, e_w_normalized)
        e_w_cos_cov = e_w_cos - jnp.eye(e_w_cos.shape[0])
        e_w_cos_cov_max = jnp.max(jnp.abs(e_w_cos_cov))
        cov_loss = 0.5 * jnp.abs(e_w_cos_cov).mean()
        self.add_summary("expert_cov_max", e_w_cos_cov_max)
        return cov_loss

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        if cfg.structure == "prenorm":
            # (batch, seq_len, input_dim)
            x = self.norm(inputs)
            x = self._dispatch_and_combine(x)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x += inputs
        elif cfg.structure == "postnorm":
            x = self._dispatch_and_combine(inputs)
            x = self.dropout(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x = self.norm(x + inputs)
        elif cfg.structure == "hybridnorm":
            x = self.prenorm(inputs)
            x = self._dispatch_and_combine(x)
            x = self.postnorm(x)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            x += inputs
        elif cfg.structure == "nonorm":
            x = self._dispatch_and_combine(inputs)
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            # We still apply `residual_weight`, since there is usually a residual link outside of
            # this layer.
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
        elif cfg.structure == "v2":
            x = self.in_norm(inputs) if NormPosition.IN_NORM in cfg.norm else inputs
            x = self._dispatch_and_combine(x)
            x = self.res_norm(x) if NormPosition.RES_NORM in cfg.norm else x
            x = self.dropout2(x)
            x = self.stochastic_depth(x)
            if cfg.residual_weight != 1:
                x *= cfg.residual_weight
            # Apply residual gating if configured
            if cfg.residual_gate_init is not None:
                theta = self.parameters["residual_gate_theta"]
                self.add_summary("residual_gate/theta", theta)
                theta_fp32 = theta.astype(jnp.float32)
                p_fp32 = jax.nn.sigmoid(theta_fp32)
                p_fp32 = jnp.clip(p_fp32, 1e-6, 1.0 - 1e-6)
                inputs_fp32 = inputs.astype(jnp.float32)
                x_fp32 = x.astype(jnp.float32)
                sqrt_1_minus_p_sq = jnp.sqrt(1.0 - p_fp32 * p_fp32)
                result_fp32 = sqrt_1_minus_p_sq * inputs_fp32 + p_fp32 * x_fp32
                x = result_fp32.astype(inputs.dtype)
            else:
                x += inputs
            x = self.out_norm(x) if NormPosition.OUT_NORM in cfg.norm else x
        else:
            raise NotImplementedError(cfg.structure)
        return x

    # pylint: disable-next=too-many-statements
    def _dispatch_and_combine(self, x: Tensor) -> Tensor:
        """Runs forward pass on the linear layers and dispatching and combining."""
        input_dtype = x.dtype
        cfg = self.config
        outer_batch = cfg.outer_batch
        if x.shape[0] % outer_batch != 0:
            raise ValueError(
                f"batch_size {x.shape[0]} has to be divisible by outer_batch {outer_batch}."
            )
        token_shape = x.shape[:-1]
        # Number of tokens per outer row.
        num_tokens = np.prod(token_shape) // outer_batch
        num_groups = cfg.num_groups
        if num_tokens % num_groups != 0:
            raise ValueError(
                f"Reshaping input sequence from (batch_size, seq_len, input_dim) to "
                f"(outer_batch, num_groups, group_size, input_dim). "
                f"batch_size({x.shape[0]}) * seq_len({x.shape[1]}) / outer_batch({outer_batch})"
                f" = {num_tokens} must be divisible by num_groups({num_groups})."
            )
        group_len = num_tokens // num_groups
        logging.info("Setting the effective group_size=%r", group_len)
        x = x.reshape([outer_batch, num_groups, group_len, cfg.input_dim])
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogsm"])
        logits = jnp.einsum("ogsm,me->ogse", x, self.parameters["gate_weight"])
        # Perform gating based on logits. Casting to float32 precision is usually needed for
        # stable performance.
        gating = self.gating(logits=logits.astype(jnp.float32))
        combine_tensor = gating.combine_tensor
        dispatch_tensor = gating.dispatch_tensor
        # Collect aux_loss.
        aux_loss = (
            gating.load_balance_loss * cfg.load_balance_loss_weight
            + gating.router_z_loss * cfg.router_z_loss_weight
        )
        # Add the expert correlation loss
        if cfg.corr_loss_weight:
            cov_loss = self._expert_correlation_loss()
            self.add_summary("expert_cov_loss", cov_loss)
            aux_loss += cov_loss * cfg.corr_loss_weight

        self.add_module_output("aux_loss", aux_loss)

        # Support dynamic partition spec lookup for different gating implementations.
        # If the gating class has dispatch_tensor_shape() method, use it to get the correct
        # partition spec. Otherwise, fall back to "ogsec" for backward compatibility.
        if hasattr(cfg.gating.klass, "dispatch_tensor_shape"):
            dispatch_partition_spec = cfg.dim_to_mesh_axis_map[
                cfg.gating.klass.dispatch_tensor_shape()
            ]
        else:
            dispatch_partition_spec = cfg.dim_to_mesh_axis_map["ogsec"]

        # For GatherBasedTopKGating, combine_tensor is [2, O, G, S, K] where combine_tensor[1]
        # contains the expert spot indices. Pass it as needed for compatibility.
        combine_tensor_for_dispatch = combine_tensor[1] if combine_tensor.ndim > 4 else None

        x = self.gating.dispatch(
            x,
            dispatch_tensor=dispatch_tensor,
            dtype=input_dtype,
            partition_spec=dispatch_partition_spec,
            combine_tensor=combine_tensor_for_dispatch,
        )
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["oegcm"])
        x = self._wi_activation(x, dispatch_tensor)
        if cfg.structure in ["prenorm", "hybridnorm", "nonorm", "v2"]:
            x = self.dropout1(x)
        with child_context("wo_einsum", module=self):
            x = self.einsum_maybe_quantized(
                "oegch,ehm->oegcm", activation=x, kernel=self.parameters["wo_weight"]
            )
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["oegcm"])

        # Transpose from oegcm to ogecm format for combine operation.
        # TopKGating and Top2Gating both expect inputs in ogecm format.
        if cfg.gating.klass in [TopKGating, Top2Gating]:
            x = jnp.einsum("oegcm->ogecm", x)
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogecm"])

        # Support dynamic partition spec lookup for combine operation.
        if hasattr(cfg.gating.klass, "combine_tensor_shape"):
            combine_partition_spec = cfg.dim_to_mesh_axis_map[
                cfg.gating.klass.combine_tensor_shape()
            ]
        else:
            combine_partition_spec = cfg.dim_to_mesh_axis_map["ogsec"]

        x = self.gating.combine(
            x,
            combine_tensor=combine_tensor,
            dtype=input_dtype,
            partition_spec=combine_partition_spec,
        )
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogsm"])

        # Add RMS norm summary for linear2 outputs.
        self.add_summary("rms_norm/linear2_outputs", (x**2.0).mean().astype(jnp.float32) ** 0.5)
        # (batch, seq_len, input_dim)
        return x.reshape(token_shape + (cfg.input_dim,))

    def _wi_activation(self, x: Tensor, dispatch_tensor: Tensor) -> Tensor:
        """Applies activation functions to the input projection.

        Args:
            x: Input tensor with shape [O, E, G, C, M].
            dispatch_tensor: Dispatch tensor used for dead neuron detection.

        Returns:
            Activated tensor with shape [O, E, G, C, H].
        """
        cfg = self.config
        if isinstance(cfg.activation, tuple):
            activations = []
            for i, activation in enumerate(cfg.activation):
                with child_context(f"wi_{i}_einsum", module=self):
                    x_i = self.einsum_maybe_quantized(
                        "oegcm,emh->oegch", activation=x, kernel=self.parameters[f"wi_{i}_weight"]
                    )
                # Add dead neuron detection for TopKGating.
                # This helps identify experts or neurons that are never activated, which can
                # indicate issues with routing or initialization.
                if cfg.gating.klass is TopKGating:
                    # valid_position_indicator: [E, C] - indicates which expert capacity slots
                    # actually contain valid tokens.
                    valid_position_indicator = jnp.einsum("ogsec->ec", dispatch_tensor)
                    # Broadcast to [O, E, G, C, H] to match x_i shape.
                    valid_position_indicator = valid_position_indicator[
                        jnp.newaxis, :, jnp.newaxis, :, jnp.newaxis
                    ]
                    invalid_position_indicator = 1 - valid_position_indicator
                    # Mask out invalid positions by subtracting a large value, so they don't
                    # contribute to the maximum.
                    x_i_prime = x_i - 10.0 * invalid_position_indicator
                    # Aggregate over the O, G, and C dimensions to get max activation per expert
                    # per hidden unit: [E, H].
                    max_hidden_units = jnp.max(x_i_prime, axis=[0, 2, 3])
                    # Count neurons that never activate (remain below the masking threshold).
                    num_dead_units = jnp.count_nonzero(
                        jnp.less(max_hidden_units, -10.0).astype(jnp.int32)
                    )
                    self.add_summary(
                        "expert_dead_neurons",
                        num_dead_units,
                    )
                x_i = with_sharding_constraint(x_i, cfg.dim_to_mesh_axis_map["oegch"])
                x_i = get_activation_fn(activation)(x_i)
                activations.append(x_i)
            assert len(activations) == 2, cfg.activation
            return activations[0] * activations[1]
        else:
            with child_context("wi_einsum", module=self):
                x = self.einsum_maybe_quantized(
                    "oegcm,emh->oegch", activation=x, kernel=self.parameters["wi_weight"]
                )
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["oegch"])
            return get_activation_fn(cfg.activation)(x)


def _convert_feedforward_to_moe_parameters(
    source_parameters: Nested[Tensor],
    *,
    num_experts: int,
    moe_parameter_specs: Nested[ParameterSpec],
) -> Nested[Tensor]:
    """Converts parameters of a TransformerFeedForwardLayer to those of a TransformerFeedForwardMoE.

    Args:
        source_parameters: The parameters of a TransformerFeedForwardLayer.
        num_experts: The number of experts for the target TransformerFeedForwardMoE.
        moe_parameter_specs: The parameter specs for the target TransformerFeedForwardMoE.

    Returns:
        Parameters of a TransformerFeedForwardMoE.

    Raises:
        NotImplementedError: if the source TransformerFeedForwardLayer includes bias in its linear
            parameters or upon unexpected source parameter names.
    """
    moe_parameters = jax.tree.map(lambda x: None, moe_parameter_specs)
    for path, value in flatten_items(source_parameters):
        m = re.fullmatch("linear([0-9_]+)/(weight|bias)", path)
        if not m:
            set_recursively(moe_parameters, path=path, value=value)
            continue
        if m.group(2) == "bias":
            raise NotImplementedError("TransformerFeedForwardMoE does not support bias")
        linear_name = m.group(1)
        # linear_name can be "linear1", "linear2", "linear1_0", or "linear1_1" and should map to
        # wi, wo, wi_0, wi_1, respectively.
        m = re.fullmatch("([0-9]+)(|_[0-9]+)", linear_name)
        if not m:
            raise NotImplementedError(f"Unexpected {linear_name} in {path}")
        moe_weight_prefix = "wi" if m.group(1) == "1" else "wo"
        moe_weight_suffix = m.group(2)
        # Shard the dispatch tensor by 'expert'.
        dispatch = with_sharding_constraint(
            jnp.ones([num_experts], dtype=value.dtype), PartitionSpec("expert")
        )
        moe_parameters[f"{moe_weight_prefix}{moe_weight_suffix}_weight"] = jnp.einsum(
            "xy,e->exy", value, dispatch
        )
    return moe_parameters


def convert_dense_to_moe_parameters(
    source_parameters: Nested[Tensor],
    *,
    target_parameter_specs: Nested[ParameterSpec],
) -> Nested[Tensor]:
    """Converts parameters of a dense BaseTransformerLayer to parameters of a target layer.

    Current limitations:
    - The source and target must have the same total number of TransformerLayers.
    - `source_parameters` must represent a `RepeatedTransformerLayer(TransformerLayer)` stack;
    - `target_parameter_specs` must represent a `RepeatedTransformerLayer(StackedTransformerLayer)`
      stack, where every layer in the `StackedTransformerLayer` must be TransformerLayer, containing
      either a (dense) TransformerFeedForwardLayer or a TransformerFeedForwardMoE as its
      `feed_forward` child.

    For example, `target_parameter_specs` may have the following structure, representing
    interleaving dense/MoE layers in a RepeatedTransformerLayer stack.
    - repeat
        - layer (StackedTransformerLayer)
            - layer0 (TransformerLayer)
                - self_attention
                - feed_forward (TransformerFeedForwardLayer)
            - layer1 (TransformerLayer)
                - self_attention
                - feed_forward (TransformerFeedForwardMoE)
                    - wi_weight (or wi_0_weight and wi_1_weight if using .*GLU.)
                    - wo_weight
                    - norm
                    - gate_weight

    Args:
        source_parameters: The dense Transformer parameters.
        target_parameter_specs: The target layer parameter specs.

    Returns:
        The target layer parameters, where:
        - The expert feed-forward weights are replicated from the corresponding dense feed-forward
          weights.
        - The `gate_weight` of TransformerFeedForwardMoE layers will be in the form of
          a ParameterSpec instead of a Tensor, since the conversion does not generate
          `gate_weight`.
        - All other parameters will be copied from the corresponding parameters of
          `source_parameters`.
    """

    def convert_fn(source_parameters: Nested[Tensor]) -> Nested[Tensor]:
        try:
            stage_parameter_specs = get_recursively(target_parameter_specs, ("repeat", "layer"))
        except KeyError as e:
            raise NotImplementedError(
                f"Expected RepeatedTransformerLayer, got {target_parameter_specs}"
            ) from e
        # The target layer is a RepeatedTransformerLayer.
        target_parameters = {"repeat": VDict({"layer": {}})}
        num_stages = jax.tree_util.tree_leaves(stage_parameter_specs)[0].shape[0]
        # The target stage is expected to be a StackedTransformerLayer.
        num_layers_per_stage = len(stage_parameter_specs)
        for layer_i in range(num_layers_per_stage):
            layer_name = f"layer{layer_i}"
            try:
                ff_layer_parameter_specs = get_recursively(
                    stage_parameter_specs, (layer_name, "feed_forward")
                )
            except KeyError as e:
                raise NotImplementedError(
                    f"Expected Repeated(Stacked(TransformerLayer)), got {stage_parameter_specs}"
                ) from e

            num_experts = None
            if "gate_weight" in ff_layer_parameter_specs:
                # The target feed_forward layer is a TransformerFeedForwardMoE.
                num_experts = ff_layer_parameter_specs["gate_weight"].shape[-1]
            source_layer_parameters = source_parameters["repeat"]["layer"]

            def convert_layer(
                layer_index: Tensor,
                source_layer_parameters=source_layer_parameters,
                num_experts=num_experts,
                moe_layer_parameter_specs=ff_layer_parameter_specs,
            ) -> Nested[Tensor]:
                """Converts source_layer_parameters[layer_index] to params of a target layer."""
                layer_parameters = jax.tree.map(lambda w: w[layer_index], source_layer_parameters)
                if not num_experts:
                    return layer_parameters

                layer_parameters["feed_forward"] = _convert_feedforward_to_moe_parameters(
                    layer_parameters["feed_forward"],
                    num_experts=num_experts,
                    moe_parameter_specs=moe_layer_parameter_specs,
                )
                return layer_parameters

            source_layer_indices = [s * num_layers_per_stage + layer_i for s in range(num_stages)]
            target_parameters["repeat"]["layer"][layer_name] = jax.vmap(convert_layer)(
                jnp.asarray(source_layer_indices, dtype=jnp.int32)
            )
        return target_parameters

    def compute_out_sharding(path: str, parameter_spec: ParameterSpec) -> Optional[PartitionSpec]:
        if path.endswith("/gate_weight"):
            # `convert_fn` will not generate gate_weight
            return None
        return parameter_spec.mesh_axes

    out_shardings = jax.tree.map(
        compute_out_sharding, tree_paths(target_parameter_specs), target_parameter_specs
    )
    target_parameters = pjit(convert_fn, out_shardings=out_shardings)(source_parameters)
    target_parameters = jax.tree.map(
        lambda spec, param: spec if param is None else param,
        target_parameter_specs,
        target_parameters,
    )
    return target_parameters


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _custom_gather(
    x: Tensor, idx: Tensor, argsort_idx: Tensor, unique_indices: bool = True
) -> Tensor:
    """Equivalent to `x.at[idx].get(unique_indices=unique_indices)`, but with a gather-based
    backward pass.

    The reason to use this function is that generally, the backward pass of a gather operation --
    scatter -- is notoriously slow on TPU. However, with the following assumptions, the backward
    pass can be implemented as a gather, optionally followed by a reduce sum:

    idx satisfies 0 <= idx < x.shape[0] and one of the following
    1. idx is unique and len(idx) == x.shape[0], i.e. idx is a permutation. In this case, the
       backward is a gather
    2. len(unique(idx)) == x.shape[0] and each value in idx has the same number of duplicates.
       In this case, the backward is a gather followed by a reduction on the duplicates.

    If `idx` doesn't follow above cases, the behavior of this function is undefined.

    See `_custom_gather_bwd` for the backward implementation detail.

    Args:
        x: A tensor of shape [S, ...].
        idx: A tensor of shape [S x K], where K is the number of duplicates of each index.
        argsort_idx: A tensor of shape [S x K] that's equal to jnp.argsort(idx).
        unique_indices: True if K == 1, False otherwise.

    Returns:
        A tensor of shape [S x K, ...]
    """
    return _custom_gather_fwd(x, idx, argsort_idx, unique_indices)[0]


def _custom_gather_fwd(
    x: Tensor, idx: Tensor, argsort_idx: Tensor, unique_indices: bool
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    return x.at[idx].get(unique_indices=unique_indices), (argsort_idx, x)


def _custom_gather_bwd(
    unique_indices: bool, res: tuple[Tensor, Tensor], g: Tensor
) -> tuple[Tensor, None, None]:
    del unique_indices
    argsort_idx, x = res
    reduction_dim, rem = divmod(argsort_idx.shape[0], x.shape[0])
    assert rem == 0
    # Indices must be unique here since it's argsorted idx.
    out = g.at[argsort_idx].get(unique_indices=True)
    if reduction_dim == 1:
        return out, None, None
    out = out.reshape(-1, reduction_dim, *out.shape[1:]).sum(1)
    assert out.shape == x.shape
    return out, None, None


_custom_gather.defvjp(_custom_gather_fwd, _custom_gather_bwd)


def _get_all_to_all_params(
    all_sizes: Tensor,
    ep_shard: Tensor,
    input_all_sizes_nodrop: Optional[Tensor] = None,
    output_all_sizes_nodrop: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes sizes and offsets required by `lax.ragged_all_to_all.`

    If `input_all_sizes_nodrop` or `output_all_sizes_nodrop` is not None, use them to
    compute the input/output offsets, respectively.
    """
    input_sizes = all_sizes[ep_shard]
    input_offsets = jnp.cumulative_sum(
        (all_sizes if input_all_sizes_nodrop is None else input_all_sizes_nodrop)[ep_shard],
        include_initial=True,
    )[:-1]
    output_sizes = all_sizes[:, ep_shard]
    output_offsets = jnp.cumulative_sum(
        all_sizes if output_all_sizes_nodrop is None else output_all_sizes_nodrop,
        include_initial=True,
        axis=0,
    )[ep_shard]
    return input_offsets, input_sizes, output_offsets, output_sizes


def _drop_tokens(all_tokens_per_expert: Tensor, max_size: int) -> tuple[Tensor, Tensor]:
    """Reduce the number of tokens in some experts in some ranks so that after all to all, no rank
    will receive number of tokens >= max_size.

    Args:
        all_tokens_per_expert: Tensor of shape [ep, ep, num_local_experts], where
            ep = expert parallelism degree.
        max_size: An integer specifying the maximum tokens allowed for each rank.

    Returns:
        A tuple of
        - A tensor of shape [ep, ep, num_local_experts].
        - A tensor of shape [ep], indicating the number of tokens dropped for expert parallel
            ranks.
    """
    ep_size = all_tokens_per_expert.shape[0]
    # l = number of local experts. ep1 = ep2 = expert parallel size.
    all_tokens_per_expert = rearrange(all_tokens_per_expert, "ep1 ep2 l -> (ep1 l) ep2")
    # axis0 is the axis corresponding to number of tokens received for all experts in each rank.

    # Modifies `all_tokens_per_expert` such that
    # jnp.all(jnp.cumsum(all_tokens_per_expert) <= max_size).
    orig_cumsum = jnp.cumulative_sum(all_tokens_per_expert, include_initial=True, axis=0)
    cumsum = jnp.minimum(orig_cumsum, max_size)
    all_tokens_per_expert = jnp.diff(cumsum, n=1, axis=0)
    dropped_tokens = orig_cumsum[-1] - cumsum[-1]

    all_tokens_per_expert = rearrange(
        all_tokens_per_expert, "(ep1 l) ep2 -> ep1 ep2 l", ep1=ep_size
    )
    return all_tokens_per_expert, dropped_tokens


# Positional args 3 to 6 for lax.ragged_all_to_all, namely
# (input_offsets, send_sizes, output_offsets, recv_sizes)
_RaggedA2AParams = tuple[Tensor, Tensor, Tensor, Tensor]


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _ragged_all_to_all(
    inputs: Tensor,
    outputs: Tensor,
    fwd_params: _RaggedA2AParams,
    bwd_params: _RaggedA2AParams,
    axis_name: str,
) -> Tensor:
    """Equivalent to `lax.ragged_all_to_all(inputs, outputs, *fwd_params, axis_name=axis_name)`

    The reason to use this function is that it accepts a pre-computed `bwd_params` to avoid
    all-to-alls of fwd_params during the backward pass of `lax.ragged_all_to_all`.
    """
    return _ragged_all_to_all_fwd(inputs, outputs, fwd_params, bwd_params, axis_name)[0]


def _ragged_all_to_all_fwd(
    inputs: Tensor,
    outputs: Tensor,
    fwd_params: _RaggedA2AParams,
    bwd_params: _RaggedA2AParams,
    axis_name: str,
) -> tuple[Tensor, tuple[_RaggedA2AParams, Tensor]]:
    return lax.ragged_all_to_all(inputs, outputs, *fwd_params, axis_name=axis_name), (
        bwd_params,
        inputs,
    )


def _ragged_all_to_all_bwd(axis_name: str, res: tuple[_RaggedA2AParams, Tensor], g: Tensor):
    bwd_params, inputs = res
    return (
        lax.ragged_all_to_all(g, jnp.zeros_like(inputs), *bwd_params, axis_name=axis_name),
        None,
        (None,) * len(bwd_params),
        (None,) * len(bwd_params),
    )


_ragged_all_to_all.defvjp(_ragged_all_to_all_fwd, _ragged_all_to_all_bwd)


def _all_to_all_dispatch(
    sorted_inputs: Tensor,
    tokens_per_expert: Tensor,
    expert_parallel_capacity: float,
    has_track_axis: bool = False,
) -> tuple[Tensor, Tensor, Sequence[Tensor], Sequence[Tensor]]:
    """Perform all-to-all dispatch for expert parallelism.

    Args:
        sorted_inputs: Tensor of shape [batch_tokens * num_experts_per_token, hidden_dim]
        tokens_per_expert: Tensor of shape [num_experts], indicating the number of tokens selected
            for each expert.
        expert_parallel_capacity: Capacity factor for each rank after all-to-all.
        has_track_axis: Whether there is a track axis for vmap context.

    Returns:
        A tuple of
        - sorted_inputs: Tensor of shape
            [batch_tokens * num_experts_per_token * expert_parallel_capacity, hidden_dim]. All
            tokens in this tensor correspond to the local experts on this rank.
        - tokens_per_expert: A tensor of shape [num_local_experts] indicating the number of tokens
            for each local expert.
        - stats: A tuple of tensors for summary purposes. Currently, it contains a single tensor
            of shape [ep_size], indicating the number of tokens dropped for expert parallel ranks.
        - residuals: A tuple of values used for all-to-all combine.
    """
    ep_shard = lax.axis_index("expert")
    all_tokens_per_expert = lax.all_gather(tokens_per_expert, axis_name="expert")
    ep_size = all_tokens_per_expert.shape[0]
    all_tokens_per_expert = all_tokens_per_expert.reshape(ep_size, ep_size, -1)

    temp_shape = list(sorted_inputs.shape)
    temp_shape[0] = int(temp_shape[0] * expert_parallel_capacity)
    all_tokens_per_expert_dropped, dropped_tokens = _drop_tokens(
        all_tokens_per_expert, temp_shape[0]
    )
    all_sizes_nodrop = all_tokens_per_expert.sum(-1)
    all_sizes = all_tokens_per_expert_dropped.sum(-1)  # [ep_size, ep_size]

    sorted_inputs_before_a2a = sorted_inputs
    fwd_params = _get_all_to_all_params(
        all_sizes, ep_shard, input_all_sizes_nodrop=all_sizes_nodrop
    )
    bwd_params = _get_all_to_all_params(
        all_sizes.T, ep_shard, output_all_sizes_nodrop=all_sizes_nodrop.T
    )

    if has_track_axis:
        # Use custom batching function for vmap support
        sorted_inputs = ragged_all_to_all_batched(
            sorted_inputs,
            jnp.zeros_like(sorted_inputs, shape=temp_shape),
            *fwd_params,
            axis_name="expert",
        )
    else:
        sorted_inputs = _ragged_all_to_all(
            sorted_inputs,
            jnp.zeros_like(sorted_inputs, shape=temp_shape),
            fwd_params,
            bwd_params,
            axis_name="expert",
        )

    # After all_to_all, the inputs will have following format where tokens from the same rank are
    # contiguous:
    # [local_ep0_rank0, local_ep1_rank0, ..., local_ep0_rank1, local_ep1_rank1, ...]
    # Reorder so that each local expert's tokens are contiguous, e.g.
    # [local_ep0_rank0, local_ep0_rank1, ..., , local_ep1_rank0, local_ep1_rank1, ...]
    output_sizes_per_expert = all_tokens_per_expert_dropped[:, ep_shard]
    output_expert_indices = lax.broadcasted_iota(jnp.int32, output_sizes_per_expert.shape, 1)

    permute_indices = jnp.argsort(
        jnp.repeat(
            output_expert_indices.reshape(-1),
            output_sizes_per_expert.reshape(-1),
            total_repeat_length=sorted_inputs.shape[0],
        )
    )
    argsort_permute_indices = jnp.argsort(permute_indices)
    sorted_inputs = _custom_gather(sorted_inputs, permute_indices, argsort_permute_indices)

    tokens_per_expert = output_sizes_per_expert.sum(0)
    return (
        sorted_inputs,
        tokens_per_expert,
        (dropped_tokens,),
        (
            permute_indices,
            sorted_inputs_before_a2a,
            fwd_params,
            bwd_params,
            argsort_permute_indices,
        ),
    )


def _all_to_all_combine(
    sorted_output: Tensor, residuals: Sequence[Tensor], has_track_axis: bool = False
) -> Tensor:
    """Perform all-to-all combine for expert parallelism.

    Tokens dropped during _all_to_all_dispatch will have zeros after _all_to_all_combine.

    Args:
        sorted_output: Tensor of shape [batch_tokens * num_experts_per_token, hidden_dim].
        residuals: Sequence of tensors from `_all_to_all_dispatch`.
        has_track_axis: Whether there is a track axis for vmap context.

    Returns:
        A tensor of shape [batch_tokens * num_experts_per_token, hidden_dim].
    """
    (
        permute_indices,
        sorted_inputs_before_a2a,
        fwd_params,
        bwd_params,
        argsort_permute_indices,
    ) = residuals
    sorted_output = _custom_gather(sorted_output, argsort_permute_indices, permute_indices)

    if has_track_axis:
        # Use custom batching function for vmap support
        sorted_output = ragged_all_to_all_batched(
            sorted_output,
            jnp.zeros_like(sorted_output, shape=sorted_inputs_before_a2a.shape),
            *bwd_params,
            axis_name="expert",
        )
    else:
        sorted_output = _ragged_all_to_all(
            sorted_output,
            jnp.zeros_like(sorted_output, shape=sorted_inputs_before_a2a.shape),
            bwd_params,
            fwd_params,
            axis_name="expert",
        )
    return sorted_output


class TransformerFeedForwardDropFreeMoE(TransformerFeedForwardMoE):
    """A Transformer feed-forward layer with mixture of experts with NO token drop."""

    @config_class
    class Config(TransformerFeedForwardMoE.Config):
        """Config for TransformerFeedForwardDropFreeMoE."""

        # Adjustable 3-tuple of ints to use the gmm kernel for the best performance.
        # tiling[0] is the block size for the number of tokens dimension.
        # tiling[1] is the block size for the model_dim.
        # tiling[2] is the block size for the hidden_dim.
        # The tiling blocks have to be multiples of 128.
        tiling: Required[Union[InstantiableConfig, tuple[int, int, int]]] = REQUIRED
        # How to partition the input batch with the expected keys below.
        input_dim_to_partition_spec: dict[str, Optional[PartitionSpec]] = {
            "bsm": PartitionSpec(("data", "fsdp"), "seq", None),
        }
        # How to partition the output batch with the expected keys below.
        # bsm - partition the b, s and m dim.
        # emh - partition the e, m, and h dim.
        # ehm - parittion the e, h, and m dim.
        output_dim_to_partition_spec: dict[str, Optional[PartitionSpec]] = {
            "bsm": PartitionSpec(("data", "fsdp"), "seq", "model"),
            "emh": PartitionSpec(None, None, "model"),
            "ehm": PartitionSpec(None, "model", None),
        }
        # Debug mode for the testing purpose.
        interpret: bool = False
        # preferred element type for gmm, mostly for testing purpose.
        preferred_element_type: Optional[jnp.dtype] = None
        # Weight for the sequence load balancing loss.
        # Ref: https://arxiv.org/html/2412.19437v2#S2 Section 2.1.2:
        # Complementary Sequence-Wise Auxiliary Loss.
        # NOTE: This is an auxiliary loss used with bias-based gating
        # (so called Auxiliary-Loss-Free Load Balancing strategy) in DeepSeek V3.
        # It's an experimental feature, so use it with caution.
        seq_load_balance_loss_weight: Optional[float] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dim_to_mesh_axis_map = {
            "me": PartitionSpec(None, None),
            "emh": PartitionSpec(None, ("fsdp", "seq"), "model"),
            # In general, we should not put FSDP in the last dim as it will require a
            # data-formatting (a copy) after the all-gather. This will increase step time by 2%
            # during training. However, not all models have big enough second dim to be sharded by
            # a large FSDP axis. Therefore, for the default config, we still put fsdp/seq in the
            # last dim.
            "ehm": PartitionSpec(None, "model", ("fsdp", "seq")),
        }
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.interpret:
            assert (
                cfg.preferred_element_type is not None
            ), "When interpret == True, please set preferred_element_type explicitly"

    def _padded_gmm(self, lhs, rhs, tokens_per_expert):
        cfg = self.config
        if isinstance(cfg.tiling, tuple):
            tiling = cfg.tiling
            pad_length = cfg.tiling[0]
        else:
            tiling = cfg.tiling.instantiate()
            # no padding when tiling is a function
            pad_length = 1

        # TODO: Revisit once Mosaic supports highest precision.
        matmul_precision = (
            jax.default_matmul_precision("default")
            if lhs.dtype == jnp.bfloat16
            else contextlib.nullcontext()
        )
        with matmul_precision:
            if lhs.shape[0] % pad_length:
                padded_lhs = lhs
                pad_length -= lhs.shape[0] % pad_length
                padded_lhs = jax.lax.pad(
                    lhs, jnp.array(0.0).astype(lhs.dtype), [(0, pad_length, 0), (0, 0, 0)]
                )
                results = mblx.gmm(
                    padded_lhs,
                    rhs,
                    tokens_per_expert,
                    tiling=tiling,
                    preferred_element_type=cfg.preferred_element_type or jnp.bfloat16,
                    interpret=cfg.interpret,
                )
                results = results[: lhs.shape[0]]
            else:
                results = mblx.gmm(
                    lhs,
                    rhs,
                    tokens_per_expert,
                    tiling=tiling,
                    preferred_element_type=cfg.preferred_element_type or jnp.bfloat16,
                    interpret=cfg.interpret,
                )
        return results

    def _dispatch_hook(
        self, *, sorted_inputs: Tensor, tokens_per_expert: Tensor
    ) -> tuple[Tensor, Tensor, Sequence[Tensor], Sequence[Any]]:
        """Hook for subclasses to perform additional processing during dispatch.

        Args:
            sorted_inputs: A tensor of shape [batch_tokens * num_experts_per_token, hidden_dim].
            tokens_per_expert: A tensor of shape [num_experts].

        Returns:
            A tuple of
            - A tensor with the same shape as `sorted_inputs`.
            - A tensor with the same shape as `tokens_per_expert`.
            - Additional outputs for the shard_map function.
            - Residuals for `_combine_hook`.
        """
        return sorted_inputs, tokens_per_expert, (), ()

    def _combine_hook(self, *, sorted_output: Tensor, residuals: Sequence[Any]) -> Tensor:
        """Hook for subclasses to perform additional processing during combine."""
        del residuals
        return sorted_output

    def _additional_shmap_output_sharding(self, mesh: jax.sharding.Mesh) -> Sequence[PartitionSpec]:
        """Specifies the sharding for _dispatch_hook(...)[2]."""
        del mesh
        return ()

    def _additional_shmap_output_hook(self, out: Sequence[Tensor]):
        """Hook for processing additional shmap output."""
        del out

    # pylint: disable-next=too-many-statements
    def _dispatch_and_combine(self, x: Tensor) -> Tensor:
        """Runs forward pass on the linear layers and dispatching and combining."""
        cfg = self.config
        x = with_sharding_constraint(x, cfg.input_dim_to_partition_spec["bsm"])
        logits = jnp.einsum("bsm,me->bse", x, self.parameters["gate_weight"])

        # Perform gating based on logits. Casting to float32 precision is usually needed for
        # stable performance.
        gating = self.gating(logits.astype(jnp.float32), cfg.seq_load_balance_loss_weight)
        # gate_assignment: [B, S, K] where each value is in [0, E-1], representing which
        # expert to use for a token.
        gate_assignment = with_sharding_constraint(
            gating.gate_assignment,
            cfg.input_dim_to_partition_spec["bsm"],
        )
        # expert_weights: [B, S, K]
        expert_weights = with_sharding_constraint(
            gating.expert_weights,
            cfg.input_dim_to_partition_spec["bsm"],
        )
        # Collect aux_loss.
        aux_loss = (
            gating.load_balance_loss * cfg.load_balance_loss_weight
            + gating.router_z_loss * cfg.router_z_loss_weight
            + gating.seq_load_balance_loss * (cfg.seq_load_balance_loss_weight or 0)
        )
        self.add_module_output("aux_loss", aux_loss)
        num_experts_per_token = cfg.gating.num_experts_per_token

        # Sharding along the contracting dim M requires an additional psum
        # in the implementation, which is inefficient.
        assert cfg.input_dim_to_partition_spec["bsm"][-1] is None
        assert cfg.output_dim_to_partition_spec["emh"][-2] is None

        mesh = thread_resources.env.physical_mesh

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                cfg.input_dim_to_partition_spec["bsm"],
                cfg.input_dim_to_partition_spec["bsm"],
                cfg.input_dim_to_partition_spec["bsm"],
                cfg.output_dim_to_partition_spec["emh"],
                cfg.output_dim_to_partition_spec["emh"],
                cfg.output_dim_to_partition_spec["ehm"],
            ),
            out_specs=(
                cfg.output_dim_to_partition_spec["bsm"],
                *self._additional_shmap_output_sharding(mesh),
            ),
            # Disables a checking pass which jax can't apply when there's a triton | pallas
            # call in the body.
            check_rep=False,
        )
        def wrapper(
            x: Tensor,
            gate_assignment: Tensor,
            expert_weights: Tensor,
            wi_0: Tensor,
            wi_1: Tensor,
            wo: Tensor,
        ) -> tuple[Tensor, ...]:
            """Computes unsorted outputs for one sharded block.

            B', S', M' and H' represents potentially sharded batch_dim, sequence length, model dim
            and hidden dim, respectively, where sharding is controlled by the shard_map's
            `in_specs`.

            Args:
                x: the sharded input batch of the shape [G=B', S', M].
                gate_assignment: [G=B', S', K].
                expert_weights: [G=B', S', K].
                wi_0: the input projection of [E, M, H'].
                wi_1: the input projection of [E, M, H'].
                wo: the output projection of [E, H', M'].

            Returns:
                A tuple of
                - A tensor of shape [G=B', S', M'].
                - ... optional series of tensors from `_dispatch_hook()[2]`.
            """
            logging.info("Setting the effective group_size=%r", x.shape[0])
            B, S, M = x.shape  # pylint: disable=invalid-name
            # [B' x S' x K]
            gate_assignment = gate_assignment.reshape((-1))
            # x[sorted_indices[:, i]] for i in range(S * K) represents tokens sorted
            # by which experts they are assigned to.
            # [B' x S' x K]
            sorted_indices = jnp.argsort(gate_assignment)
            token_indices = sorted_indices // num_experts_per_token
            # Dispatch the tokens.
            combine_indices = jnp.argsort(sorted_indices)
            # [B' x S' x K, M]
            sorted_inputs = _custom_gather(
                x.reshape(-1, M), token_indices, combine_indices, unique_indices=False
            )
            tokens_per_expert = jnp.bincount(gate_assignment, length=cfg.num_experts)

            sorted_inputs, tokens_per_expert, additional_outputs, residuals = self._dispatch_hook(
                sorted_inputs=sorted_inputs,
                tokens_per_expert=tokens_per_expert,
            )

            # [B' x S' x K, H']
            activation_0 = self._padded_gmm(sorted_inputs, wi_0, tokens_per_expert)
            activation_0 = get_activation_fn(cfg.activation[0])(activation_0)

            activation_1 = self._padded_gmm(sorted_inputs, wi_1, tokens_per_expert)
            activation_1 = get_activation_fn(cfg.activation[1])(activation_1)

            intermediate = activation_0 * activation_1

            if cfg.structure in ["prenorm", "hybridnorm", "nonorm", "v2"]:
                intermediate = self.dropout1(intermediate)

            # [B' x S x K, M]
            sorted_output = self._padded_gmm(intermediate, wo, tokens_per_expert)
            if thread_resources.env.physical_mesh.shape["model"] > 1:
                # If output is partitioned across "model", we need to reduce-scatter. Otherwise,
                # we do an allreduce.
                spec = cfg.output_dim_to_partition_spec["bsm"][2]
                if spec and "model" in spec:
                    sorted_output = jax.lax.psum_scatter(
                        sorted_output, "model", scatter_dimension=1, tiled=True
                    )
                else:
                    sorted_output = jax.lax.psum(sorted_output, "model")
            # [B' x S' x K, M']
            sorted_output = self._combine_hook(sorted_output=sorted_output, residuals=residuals)
            # Gather the tokens to their original positions.
            unsorted_output = _custom_gather(sorted_output, combine_indices, sorted_indices)
            output = unsorted_output.reshape(B, S, num_experts_per_token, unsorted_output.shape[-1])
            # Apply the expert weights.
            output *= expert_weights.astype(output.dtype)[..., None]
            # [B', S', M']
            output = jnp.sum(output, axis=-2)
            return output, *additional_outputs

        out, *additional_outputs = wrapper(
            x,
            gate_assignment,
            expert_weights,
            self.parameters["wi_0_weight"],
            self.parameters["wi_1_weight"],
            self.parameters["wo_weight"],
        )
        self._additional_shmap_output_hook(additional_outputs)
        return out


class ApproximateTokenDropFreeMoE(TransformerFeedForwardDropFreeMoE):
    """Mostly the same as `TransformerFeedForwardDropFreeMoE`, but allows expert parallel training.

    To avoid maintaining excessive buffer after all-to-all, a config `expert_parallel_capacity` is
    added (see below for more details). If this factor is less than number of expert parallel
    ranks, we may drop tokens after all-to-all. However, with a load balancing loss, generally we
    should not see any token drop (at least for trillion parameter MoE) after training for a few
    hundred steps using the default value of 1.25.
    """

    @config_class
    class Config(TransformerFeedForwardDropFreeMoE.Config):
        # After all-to-all dispatch, each rank's receiving buffer will have size
        # tokens_per_device * num_experts_per_token * expert_parallel_capacity. Excess tokens
        # exceeding this buffer size will be dropped.
        expert_parallel_capacity: float = 1.25

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dim_to_mesh_axis_map = {
            "me": PartitionSpec(None, None),
            "emh": PartitionSpec("expert", ("fsdp", "seq"), "model"),
            "ehm": PartitionSpec("expert", "model", ("fsdp", "seq")),
        }
        cfg.input_dim_to_partition_spec = {
            "bsm": PartitionSpec(("expert", "data", "fsdp"), "seq", None),
        }

        cfg.output_dim_to_partition_spec = {
            "bsm": PartitionSpec(("expert", "data", "fsdp"), "seq", "model"),
            "emh": PartitionSpec("expert", None, "model"),
            "ehm": PartitionSpec("expert", "model", None),
        }
        return cfg

    def _has_track_axis(self) -> bool:
        """Check if we're in a vmap context with track axis (VectorizedTrackTransformerLayer)."""
        mesh = thread_resources.env.physical_mesh
        return "track" in mesh.axis_names if mesh.axis_names else False

    def _dispatch_hook(
        self, *, sorted_inputs: Tensor, tokens_per_expert: Tensor
    ) -> tuple[Tensor, Tensor, Sequence[Tensor], Sequence[Tensor]]:
        return _all_to_all_dispatch(
            sorted_inputs=sorted_inputs,
            tokens_per_expert=tokens_per_expert,
            expert_parallel_capacity=self.config.expert_parallel_capacity,
            has_track_axis=self._has_track_axis(),
        )

    def _combine_hook(self, *, sorted_output: Tensor, residuals: Sequence[Tensor]) -> Tensor:
        return _all_to_all_combine(
            sorted_output, residuals=residuals, has_track_axis=self._has_track_axis()
        )

    def _additional_shmap_output_sharding(self, mesh: jax.sharding.Mesh) -> Sequence[PartitionSpec]:
        # Note: dropped tokens is duplicated along expert and model parallel axes, and sharded
        # as partial sum across data and sequence parallel axes.
        return (
            PartitionSpec(
                tuple(name for name in mesh.axis_names if name not in ("model", "expert", "track"))
            ),
        )

    def _additional_shmap_output_hook(self, out: Sequence[Tensor]):
        if out:
            assert len(out) == 1
            # TODO: This should probably be a bar graph where each bar is the number
            # of tokens dropped for a rank. However, tf_summary doesn't directly support this.
            self.add_summary("ep_dropped_tokens", jnp.sum(out[0]))


def set_interpret_in_moe_config_recursively(
    cfg: ConfigBase, preferred_element_type: jnp.dtype = jnp.bfloat16
) -> ConfigBase:
    """Recursively enables `interpret=True` for all `TransformerFeedForwardDropFreeMoE.Config`.

    Args:
        cfg: A `ConfigBase` tree containing nested module configurations.
        preferred_element_type: preferred_element_type for gmm.

    Returns:
        The same config object with `interpret=True` set for all MoE layers.
    """

    def visit_fn(_, value):
        if isinstance(value, TransformerFeedForwardDropFreeMoE.Config):
            value.interpret = True
            value.preferred_element_type = preferred_element_type

    def enter_fn(_, value, default_kv):
        return None if isinstance(value, TransformerFeedForwardDropFreeMoE.Config) else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
    return cfg


class V6eGMMTilingModifier(ConfigModifier):
    """Modifies the tiling config of TransformerFeedForwardDropFreeMoE for V6e TPU."""

    def __call__(self, cfg):
        def is_nodrop_config(cfg):
            return isinstance(cfg, TransformerFeedForwardDropFreeMoE.Config)

        def visit_fn(_, value):
            if is_nodrop_config(value):
                value.tiling = (1024, 1024, 1024)

        def enter_fn(_, value, default_kv):
            return None if is_nodrop_config(value) else default_kv

        cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
        return cfg

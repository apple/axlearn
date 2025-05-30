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

import re
import os
import math
from enum import Enum
from functools import reduce, partial
from typing import NamedTuple, Optional, Sequence, Union
# import numpy
# import sys
# numpy.set_printoptions(threshold=sys.maxsize)

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax._src.mesh import thread_resources
import numpy as np
from absl import logging
from jax.experimental.pjit import pjit
import jax.lax as lax
from axlearn.common.attention import NormPosition
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import (
    REQUIRED,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
)
from axlearn.common.utils import PartitionSpec
from axlearn.common.layers import (
    BaseNormalizationLayer,
    Dropout,
    LayerNorm,
    MovingAverage,
    StochasticDepth,
    get_activation_fn,
)
from axlearn.common.neuron_blockwise_mlp import can_use_blockwise_matmul_nki, blockwise_mm
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, child_context
from axlearn.common.param_init import FanAxes, constant_initializer
from axlearn.common.quantized_dot_general.layers import DenseGeneralBaseLayer
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

_USING_SHARDMAP_FFN=int(os.getenv('USE_SHARDMAP_FFN', 1))
_USING_INDEX_SHARDING = int(os.getenv('USE_INDEX_SHARDING', 1))


@jax.jit
def down_proj(x, wo_weight):
    return jnp.einsum("oegch,ehm->oegcm", x, wo_weight)

@jax.jit
def combine_outputs(permuted_output, token_permutation_idx, expert_index, expert_affinities_masked, dest_output):
    # Expected shapes:
    # permuted_output: (O, G, E*C, M)
    # token_permutation_idx: (O, G, S, K)
    # expert_index: (O, G, S, K)
    # expert_affinities_masked: (O, G, S, E)

    batch_idx = jnp.arange(permuted_output.shape[0])[:, None, None, None]
    group_idx = jnp.arange(permuted_output.shape[1])[None, :, None, None]
    seq_idx = jnp.arange(token_permutation_idx.shape[2])[None, None, :, None]

    output_k = permuted_output[
        batch_idx,
        group_idx,
        token_permutation_idx,
        :
    ]  # Shape: (O, G, S, K, M)

    expert_affinities_k = expert_affinities_masked[
        batch_idx,
        group_idx,
        seq_idx,
        expert_index
    ]  # Shape: (O, G, S, K)

    expert_affinities_k_expanded = expert_affinities_k[..., None]  # Shape: (O, G, S, K, 1)

    # Multiply outputs by their corresponding affinities
    weighted_output_k = output_k * expert_affinities_k_expanded  # Shape: (O, G, S, K, M)

    # Sum across the top-k dimension
    combined_output = jnp.sum(weighted_output_k, axis=3)  # Shape: (O, G, S, M)
    dest_output = dest_output.at[0].set(combined_output)  # Shape: (1, O, G, S, M)

    return dest_output

import jax.numpy as jnp
from jax import lax

@partial(jax.jit, static_argnums=(6, 7,))
def blockwise_mlp(
    hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weights, token_position_to_id, block_to_expert, 
    block_size, activation_fns):
    O = hidden_states.shape[0]
    G = hidden_states.shape[1]
    # nki doesn't support batching 'E   NotImplementedError: Batching rule for 'nki_call' not implemented'
    use_vmap = False
    if use_vmap:
        hidden_states = hidden_states.reshape((O*G, 1, 1) + hidden_states.shape[2:])
        expert_affinities_masked = expert_affinities_masked.reshape((O*G, 1, 1) + expert_affinities_masked.shape[2:])
        token_position_to_id = token_position_to_id.reshape((O*G, 1, 1) + token_position_to_id.shape[2:])
        block_to_expert = block_to_expert.reshape((O*G, 1, 1) + block_to_expert.shape[2:])

    if can_use_blockwise_matmul_nki(
        hidden_size=gate_up_proj_weight.shape[1],
        intermediate_size_tp=gate_up_proj_weight.shape[-1],
        block_size=block_size,
        glu_mlp=len(activation_fns) == 2,
    ) and int(os.getenv('AXLEARN_USE_BLOCKWISE_MLP_KERNEL', '1')) == 1:
        blockwise_mlp_per_group = blockwise_mm
    else:
        blockwise_mlp_per_group = blockwise_mm_per_group_native

    if use_vmap:
        batched_blockwise_mlp = jax.vmap(blockwise_mlp_per_group, in_axes=(0,0, None, None, None, 0, 0, None))
        output = batched_blockwise_mlp(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weights, token_position_to_id, block_to_expert, block_size)
        return output
    else:
        outputs = []
        for o in range(O):
            g_outputs = []
            for g in range(G):
                hidden_states_og = hidden_states[o:o+1, g:g+1]
                expert_affinities_masked_og = expert_affinities_masked[o:o+1, g:g+1]
                token_position_to_id_og = token_position_to_id[o:o+1, g:g+1]
                block_to_expert_og = block_to_expert[o:o+1, g:g+1]
                output = blockwise_mlp_per_group(hidden_states_og, expert_affinities_masked_og, gate_up_proj_weight, down_proj_weights, token_position_to_id_og, block_to_expert_og, block_size)
                g_outputs.append(output)
            outputs.append(jnp.concatenate(g_outputs, axis=1))
        return jnp.concatenate(outputs, axis=0)

@partial(jax.jit, static_argnums=(2,3,4,))
def calculate_token_position_to_id(block_position_indices, tokens_indices, 
                                   num_blocks, block_size, total_tokens, dest_output):
        """
        Invert block_position_indices to obtain token_position_to_id.
        """
        O, G, num_tokens, E = block_position_indices.shape

        # Create batch and group indices
        # (O, G, S*top_k, E)
        batch_indices = jnp.arange(O)[:, None, None, None]
        batch_indices = jnp.broadcast_to(batch_indices, (O, G, num_tokens, E))

        # (O, G, S*top_k, E)
        group_indices = jnp.arange(G)[None, :, None, None]
        group_indices = jnp.broadcast_to(group_indices, (O, G, num_tokens, E))

        token_position_to_id = jnp.zeros((O, G, num_blocks * block_size + 1), dtype=jnp.int32)
        token_position_to_id = token_position_to_id.at[batch_indices, group_indices, block_position_indices].set(tokens_indices+1)

        token_position_to_id = token_position_to_id[:, :, 1:]
        token_position_to_id = token_position_to_id - 1
        token_position_to_id = jnp.where(token_position_to_id==-1, total_tokens, token_position_to_id)   
        dest_output = dest_output.at[0].set(token_position_to_id)
        return dest_output

def blockwise_mm_per_group_native(hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weights, token_position_to_id, block_to_expert, block_size):
    with jax.named_scope("take_out_OG"):
        hidden_states = jnp.squeeze(hidden_states, axis=(0,1,))
        expert_affinities_masked = jnp.squeeze(expert_affinities_masked, axis=(0,1,))
        token_position_to_id = jnp.squeeze(token_position_to_id, axis=(0,1,))
        block_to_expert = jnp.squeeze(block_to_expert, axis=(0,1,))

    # add +1 for padding
    with jax.named_scope("add padding"):
        padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
        padding_e = jnp.zeros((1,expert_affinities_masked.shape[1]), dtype=expert_affinities_masked.dtype)
        # (S+1, H)
        hidden_states = jnp.concat([hidden_states, padding_h], axis=0)
        expert_affinities = jnp.concat([expert_affinities_masked, padding_e], axis=0)

    Tplus1, H = hidden_states.shape
    _, E = expert_affinities.shape
    T = Tplus1-1
    B = block_size
    dtype = hidden_states.dtype
    token_position_to_id = token_position_to_id.reshape(-1, B)
    N = token_position_to_id.shape[0]
    I_TP = gate_up_proj_weight.shape[-1]
    output_shape = [T+1, H]
    output_jax = jnp.zeros(output_shape, dtype=dtype)
    def body_fun(b, carry):
        output_jax = carry
        local_token_position_to_id = token_position_to_id[b, :]
        hidden_states_padded = hidden_states
        expert_affinities_padded = expert_affinities
        local_hidden_states = hidden_states_padded[local_token_position_to_id].astype(jnp.float32)
        expert_idx = block_to_expert[b]
        local_expert_affinities = expert_affinities_padded[local_token_position_to_id, expert_idx]
        local_expert_affinities = jnp.reshape(local_expert_affinities, (-1, 1))

        x_0 = jnp.einsum("cm,mh->ch", local_hidden_states, gate_up_proj_weight[expert_idx][:, 0, :])
        x_0 = get_activation_fn("nn.silu")(x_0)
        x_1 = jnp.einsum("cm,mh->ch", local_hidden_states, gate_up_proj_weight[expert_idx][:, 1, :])
        x_1 = get_activation_fn("linear")(x_1)
        gate_up_activation = x_0 * x_1
        
        down_activation = jnp.einsum("ch,hm->cm", gate_up_activation, down_proj_weights[expert_idx])
        scale = down_activation * local_expert_affinities
        output_jax = output_jax.at[local_token_position_to_id].add(scale.astype(output_jax.dtype))
        return output_jax

    init_carry = output_jax
    output_jax = lax.fori_loop(0, N, body_fun, init_carry)
    return output_jax[None, None, None, :-1, :]


def _router_z_loss(logits: Tensor) -> Tensor:
    """Loss that encourages router logits to remain small and improves stability.

    Reference:
    https://github.com/tensorflow/mesh/blob/fbf7b1e547e8b8cb134e81e1cd350c312c0b5a16/mesh_tensorflow/transformer/moe.py#L1956

    Args:
        logits: A tensor with shape (batch, num_experts).

    Returns:
        z_loss: A scalar loss.
    """
    with jax.named_scope("z_loss"):
        # pytype: disable=module-attr
        logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        # pytype: enable=module-attr
        log_z = jnp.squeeze(logits_sum, axis=-1)
        z_loss = jax.lax.square(log_z).mean()
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
            # Round up to a multiple of 4 to avoid possible padding.
            while expert_capacity % 4:
                expert_capacity += 1
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
    mesh_axis_names: Sequence[str],
    outer_batch_axis_names: Sequence[str],
    mesh_shape: Optional[Union[MeshShape, HybridMeshShape]],
) -> Optional[int]:
    """Infer MoE outer batch size from mesh shape."""
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
        cfg: AdaptiveLoadBalanceLoss.Config = self.config
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
        cfg: Top2Gating.Config = self.config
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
        # those of examples not assigned to e with top_k.
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
        self.add_summary("load_balance_loss", WeightedScalar(aux_loss, 1))
        self.add_summary("router_z_loss", WeightedScalar(router_z_loss, 1))
        self.add_summary("dispatch_0", WeightedScalar(dispatch_0, 1))
        self.add_summary("dispatch_1", WeightedScalar(dispatch_1, 1))
        self.add_summary("dispatch_2", WeightedScalar(dispatch_2, 1))
        self.add_summary("over_capacity_1", WeightedScalar(over_capacity_1, 1))
        self.add_summary("over_capacity_2", WeightedScalar(over_capacity_2, 1))

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

class TopKGating(BaseGating):
    """Generalized Top-K gating for Mixture-of-Experts."""

    @config_class
    class Config(BaseGating.Config):
        """Configures TopKGating."""

        # Number of selected experts.
        top_k: int = 2

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
        cfg: TopKGating.Config = self.config
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

        density_1 = jnp.mean(mask_1.astype(jnp.float32), axis=-2)
        # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
        # those of examples not assigned to e with top_k.
        density_1_proxy = jnp.mean(raw_gates, axis=-2, dtype=jnp.float32)

        # Compute aux_loss.
        aux_loss = jnp.mean(density_1_proxy * density_1, dtype=jnp.float32)
        aux_loss *= cfg.num_experts * cfg.num_experts

        gate_1 = jnp.einsum("ogse,ogse->ogs", raw_gates, mask_1.astype(raw_gates.dtype))
        gates_list = [gate_1]  # OGS tensor.
        index_list = [index_1]  # OGS tensor.
        masks_list = [mask_1]  # OGSE tensor.
        raw_gates_i = raw_gates  # OGSE tensor.

        denom = gate_1 + 1e-9

        # Greedily pick top-k experts and find `gates`, `masks`, `index` for all experts.
        for i in range(1, cfg.top_k):
            # Gates without the previous experts.
            raw_gates_i *= 1.0 - masks_list[i - 1].astype(raw_gates_i.dtype)
            index_i = jnp.argmax(raw_gates_i, axis=-1)
            mask_i = jax.nn.one_hot(index_i, cfg.num_experts, dtype=cfg.mask_dtype)
            gate_i = jnp.einsum("ogse,ogse->ogs", raw_gates_i, mask_i.astype(raw_gates_i.dtype))
            denom += gate_i
            gates_list.append(gate_i)
            masks_list.append(mask_i)
            index_list.append(index_i)

        # Renormalize.
        gates_list = [x / denom for x in gates_list]

        # We reshape the mask as [OGS, E], and compute cumulative sums of assignment
        # indicators for each expert index e \in 0..E-1 independently.
        # cumsum over S dim: mask_1 is OGSE tensor.
        position_in_expert_1 = _cum_sum(masks_list[0], exclusive=True, axis=-2)
        # Add the over capacity ratio for expert 1.
        over_capacity_list = [
            _create_over_capacity_ratio_summary(
                mask=masks_list[0],
                position_in_expert=position_in_expert_1,
                capacity=expert_capacity,
            )
        ]
        # Filter valid positions for top 1 selection
        masks_list[0] *= jnp.less(position_in_expert_1, expert_capacity).astype(masks_list[0].dtype)
        position_in_expert_1 = jnp.einsum("ogse,ogse->ogs", position_in_expert_1, masks_list[0])
        # How many examples in this sequence go to this expert?
        mask_1_count = jnp.einsum("ogse->oge", masks_list[0])
        # ogs - mostly ones, but zeros where something didn't fit.
        mask_1_flat = jnp.sum(masks_list[0], axis=-1, dtype=cfg.mask_dtype)
        assert mask_1_count.dtype == cfg.mask_dtype
        assert mask_1_flat.dtype == cfg.mask_dtype
        position_in_expert_list = [position_in_expert_1]
        mask_i_flat_list = [mask_1_flat]
        mask_count_all = mask_1_count

        for i in range(1, cfg.top_k):
            position_in_expert_i = _cum_sum(
                masks_list[i], exclusive=True, axis=-2
            ) + jnp.expand_dims(mask_count_all, -2)
            # Add the over capacity ratio for expert i.
            over_capacity_list.append(
                _create_over_capacity_ratio_summary(
                    mask=masks_list[i],
                    position_in_expert=position_in_expert_i,
                    capacity=expert_capacity,
                )
            )
            # Filter invalid positions for top i selection
            masks_list[i] *= jnp.less(position_in_expert_i, expert_capacity).astype(
                masks_list[i].dtype
            )
            # How many examples in this sequence go to this expert?
            mask_count_all += jnp.einsum("ogse->oge", masks_list[i])
            position_in_expert_i = jnp.einsum("ogse,ogse->ogs", position_in_expert_i, masks_list[i])
            position_in_expert_list.append(position_in_expert_i)
            mask_i_flat_list.append(jnp.sum(masks_list[i], axis=-1, dtype=cfg.mask_dtype))

        # OGSEC tensor.
        combine_tensor = jnp.zeros(
            [*logits.shape[:3], cfg.num_experts, expert_capacity],
            dtype=jnp.float32,
        )

        for gate_i, index_i, position_in_expert_i, mask_i_flat in zip(
            gates_list, index_list, position_in_expert_list, mask_i_flat_list
        ):
            # OGS Filter valid gate values.
            gate_i *= mask_i_flat.astype(gate_i.dtype)
            # OGSC tensor.
            b = jax.nn.one_hot(
                position_in_expert_i.astype(np.int32),
                expert_capacity,
                dtype=jnp.float32,
            )
            # OGSE tensor.
            a = jnp.expand_dims(gate_i * mask_i_flat.astype(jnp.float32), axis=-1) * jax.nn.one_hot(
                index_i, cfg.num_experts, dtype=jnp.float32
            )
            combine_tensor += jnp.einsum("ogse,ogsc->ogsec", a, b)

        # OGSEC tensor.
        dispatch_tensor = combine_tensor.astype(bool)

        # Counts for tokens that are dispatched to 0, 1 and 2 experts.
        dispatch_count_tensor = jnp.sum(dispatch_tensor.astype(jnp.int32), [-2, -1])

        router_z_loss = _router_z_loss(logits)

        # Add auxiliary losses and gating statistics to job summary.
        self.add_summary("load_balance_loss", WeightedScalar(aux_loss, 1))
        self.add_summary("router_z_loss", WeightedScalar(router_z_loss, 1))
        # Summary for number of tokens dispatched to 0, 1, ..., k experts.
        for i in range(cfg.top_k + 1):
            dispatch_i = jnp.sum(dispatch_count_tensor == i)
            self.add_summary(f"dispatch_{i}", WeightedScalar(dispatch_i, 1))
        # Over capacity ratios for top-k experts.
        for i, over_capacity_i in enumerate(over_capacity_list):
            self.add_summary(f"over_capacity_{i + 1}", WeightedScalar(over_capacity_i, 1))

        if cfg.adaptive_load_balance_loss is None:
            self.add_summary("load_balance_loss", aux_loss)
        else:
            self.add_summary("load_balance_loss_original", aux_loss)
            over_capacity_max = over_capacity_list[0]
            for over_capacity_i in over_capacity_list[1:]:
                over_capacity_max = jnp.maximum(over_capacity_max, over_capacity_i)
            aux_loss *= self.adaptive_load_balance_loss(over_capacity_max)
            self.add_summary("load_balance_loss", aux_loss)

        return self.Output(
            combine_tensor=combine_tensor,
            dispatch_tensor=dispatch_tensor,
            load_balance_loss=aux_loss,
            router_z_loss=router_z_loss,
        )
   
class TopKGatingGather(TopKGating):
    """Computes Top-K gating for Mixture-of-Experts.

    The methods take gating logits, potentially sharded across tpu cores as inputs.
    We rely on sharding propagation to work universally. Dispatch and combine tensors
    should be explicitly annotated with `utils.with_sharding_constraint` by the caller.

    We perform dispatch/combine via gather and indexing as opposed to einsum in Top2Gating class.

    Note that for local_dispatch, the original batch BLM is reshaped to OGSM. There are
    O*G groups and each group is being dispatched independently.

    Reference:
    https://github.com/google/praxis/blob/f8467c730ccac1bf2cf10a68fb18f9e6e1f658b4/praxis/gshard_utils.py#L87
    """
    @config_class
    class Config(TopKGating.Config):
        pass
    
    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)

    def compute_expert_mask(self, cfg, expert_index, num_experts):
        """Helper function which computes top_k-hot encoded expert_mask from the given expert_index.

        Arguments:
            expert_index: Tensor of shape (O, G, S, top_k), containing the 'chosen' experts for each token.
        Returns:
            expert_mask: Tensor of shape (O, G, S * top_k, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
        """

        # Construct expert_mask from expert_index, using efficient version of one-hot encoding for xla device
        # Perform operation in float32 to prevent precision issues due to auto-downcasting to bf16
        # (Use float dtype to perform computations in the vector engine for efficiency)
        
        with jax.named_scope("expert_mask"):
            # expert_mask: top_k-hot encoded expert assignment per token -> (T, E)        
            # Initialize expert_mask with zeros
            O, G, Sxtop_k = expert_index.shape
            # expert_index: (O, G, S * top_k)
            expert_index = expert_index.reshape(O, G, Sxtop_k)
            expert_mask = jnp.zeros((O, G, Sxtop_k, num_experts), dtype=jnp.float32)

            idx_O, idx_G, idx_Sxtop_k = jnp.meshgrid(
                jnp.arange(O),
                jnp.arange(G),
                jnp.arange(Sxtop_k),
                indexing='ij'
            )

            # Set expert mask to one for each token at correct expert for all top k
            # expert_mask: O, G, S*topk, E
            expert_mask = expert_mask.at[idx_O[..., None], idx_G[..., None],
                                        idx_Sxtop_k[..., None], expert_index[..., None]].set(1.0)
        return expert_mask
    
    @staticmethod
    def compute_expert_affinities_masked(expert_affinities, expert_mask, normalize_top_k_affinities):
        """Helper function which computes the masked expert_affinities by selecting the chosen experts for each token,
        and normalizes the affinities if needed.

        Arguments:
            expert_affinities: Tensor of shape (O, G, S, E), containing the normalized affinities of each token for each expert.
            expert_mask: Tensor of shape (O, G, S, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
            normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
        Returns:
            expert_affinities_masked: Tensor of shape (O, G, S, E) containing the affinities of just the chosen experts for
                                      each token (after normalization if required).
        ∂"""

        # Apply expert_mask obtain the affinities for the chosen experts
        # expert_affinities_masked -> (O, G, S, E)
        expert_affinities_masked = jnp.where(expert_mask == 0, 0, expert_affinities)
        if normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            norm = jnp.sum(jnp.abs(expert_affinities_masked), axis=-1, keepdims=True)
            norm = jnp.clip(norm, a_min=1e-9)
            expert_affinities_masked = expert_affinities_masked / norm
        return expert_affinities_masked
    
    @staticmethod
    def compute_token_assignments(token_permutation_idx, num_experts, expert_capacity):
        with jax.named_scope("token_assignments"):
            O, G, S, top_k = token_permutation_idx.shape
            token_indices = jnp.arange(S)[None, None, :, None]
            token_indices = jnp.broadcast_to(token_indices, (O, G, S, top_k))
            token_indices = token_indices.reshape(O, G, -1)

            # Create batch and group indices
            batch_indices = jnp.arange(O)[:, None, None, None]
            batch_indices = jnp.broadcast_to(batch_indices, (O, G, S, top_k))
            batch_indices = batch_indices.reshape(O, G, -1)

            group_indices = jnp.arange(G)[None, :, None, None]
            group_indices = jnp.broadcast_to(group_indices, (O, G, S, top_k))
            group_indices = group_indices.reshape(O, G, -1)
            
            token_permutation_idx = token_permutation_idx.reshape(O, G, -1)

            # Create scatter indices
            scatter_indices = jnp.stack(
                [batch_indices, group_indices, token_permutation_idx], axis=-1)
            # token_assignments: (C*E+1,)
            # for each position in an expert's capacity we now need the token id
            # this is basically reverse of the token_permutation_idx
            token_assignments = jnp.zeros((O, G, expert_capacity * num_experts+1), dtype=jnp.int32)
            
            # Perform scatter
            token_assignments = jax.lax.scatter(
                token_assignments,
                scatter_indices,
                token_indices + 1,
                jax.lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0, 1, 2),
                    scatter_dims_to_operand_dims=(0, 1, 2)
                )
            )
            token_assignments = token_assignments[:,:,1:]
            token_assignments = jnp.reshape(token_assignments, shape=(O, G, num_experts, expert_capacity))
        return token_assignments

    @staticmethod
    def compute_aux_loss(cfg, mask, raw_gates):
        with jax.named_scope("aux_loss"):
            # OGE tensor (reduce S out of OGSE tensor mask_1).
            # density_1[:, e] represents assignment ratio (num assigned / total) to
            # expert e as top_1 expert without taking capacity into account.
            density_denom = jnp.asarray(1.0, dtype=jnp.float32)

            density_k = jnp.mean(mask.astype(jnp.float32), axis=-2) / density_denom
            # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
            # those of examples not assigned to e with top_k.
            density_k_proxy = jnp.mean(raw_gates, axis=-2, dtype=jnp.float32) / density_denom

            # Compute aux_loss.
            aux_loss = jnp.mean(density_k_proxy * density_k, dtype=jnp.float32)
            aux_loss *= cfg.num_experts * cfg.num_experts
            
            return aux_loss

    def compute_metrics(self, expert_mask_pre_capacity_drop, expert_mask):
        # stats
        num_dropped = jnp.sum(expert_mask_pre_capacity_drop - expert_mask, axis=[0,1,2])
        total_num_dropped = jnp.sum(num_dropped)
        dispatch_per_expert  = jnp.sum(expert_mask, axis=[0, 1, 2])
        return total_num_dropped, dispatch_per_expert
    
    def add_summaries(self, aux_loss, router_z_loss, total_num_dropped, dispatch_per_expert):    
        # Adding auxiliary losses and gating statistics to job summary.
        self.add_summary("load_balance_loss", WeightedScalar(aux_loss, 1))
        self.add_summary("router_z_loss", WeightedScalar(router_z_loss, 1))
        self.add_summary("total_num_dropped", WeightedScalar(total_num_dropped, 1))
        for i in range(self.config.num_experts):
            self.add_summary(f"dispatch_{i}", WeightedScalar(dispatch_per_expert[i], 1))
        
        # TODO(huilgolr): consolidate with below original summaries
        # self.add_summary("dispatch_0", WeightedScalar(dispatch_0, 1))
        # self.add_summary("dispatch_1", WeightedScalar(dispatch_1, 1))
        # self.add_summary("dispatch_2", WeightedScalar(dispatch_2, 1))
        # self.add_summary("over_capacity_1", WeightedScalar(over_capacity_1, 1))
        # self.add_summary("over_capacity_2", WeightedScalar(over_capacity_2, 1))
        # if cfg.adaptive_load_balance_loss is None:
        #     self.add_summary("load_balance_loss", aux_loss)
        # else:
        #     self.add_summary("load_balance_loss_original", aux_loss)
        #     aux_loss *= self.adaptive_load_balance_loss(
        #         jnp.maximum(over_capacity_1, over_capacity_2)
        #     )
        #     self.add_summary("load_balance_loss", aux_loss)

    def router(self, cfg, logits):
        with jax.named_scope("router"):
            # logits: (O, G, S, E)
            if logits.dtype != jnp.float32:
                logits = logits.astype(jnp.float32)
            logits = _cap_logits(logits, cfg.gating_logit_cap)
            # raw_gates (expert affinities): (O, G, S, E)
            raw_gates = jax.nn.softmax(logits, axis=-1)  # along E dim
        return raw_gates
    
    def compute_expert_capacity(self, cfg, logits):
        with jax.named_scope("expert_capacity"):
            # computing expert capacity scalar
            expert_capacity = _compute_expert_capacity(
                expert_capacity=cfg.expert_capacity,
                capacity_factor=(
                    cfg.train_capacity_factor if self.is_training else cfg.eval_capacity_factor
                ),
                group_size=logits.shape[-2],
                num_experts=cfg.num_experts,
            )
        return expert_capacity
        
    def compute_positions_and_drop(self, expert_mask, raw_gates, expert_capacity, cfg):
        O, G, sk, E = expert_mask.shape
        k = cfg.top_k
        S = sk // k

        with jax.named_scope("position_in_expert"):
            # Compute cumulative sums of assignment
            # indicators for each expert, i.e. index e \in 0..E-1 independently.
            # cumsum over S dim
            # position_in_expert: [O, G, S*topk, E]
            position_in_expert = _cum_sum(expert_mask.astype(jnp.int32), axis=-2).astype(jnp.float32)
            expert_mask_pre_capacity_drop = expert_mask
            expert_mask_k_pre_capacity_drop = expert_mask_pre_capacity_drop.reshape(O, G, k, S, E)
            expert_mask_k_pre_capacity_drop = jnp.sum(expert_mask_k_pre_capacity_drop, axis=2)

            # expert_affinities_masked: [O, G, S, E]
            expert_affinities_masked = self.compute_expert_affinities_masked(
                raw_gates, expert_mask_k_pre_capacity_drop, normalize_top_k_affinities=True
            )
            # jax.debug.print('expert mask pre drop {x}', x=expert_mask)
            # Update expert_mask by accounting for capacity factor (i.e. tokens exceeding capacity are dropped)
            expert_mask = jnp.where(position_in_expert > expert_capacity, 0, expert_mask)

            expert_mask_k = expert_mask.reshape(O, G, k, S, E)
            expert_mask_k = jnp.sum(expert_mask_k, axis=2)
            expert_affinities_masked = jnp.where(expert_mask_k == 0, 0, expert_affinities_masked)

            # Add expert offset to the position_in_expert
            # Perform operation in float32 to prevent precision issues due to auto-downcasting to bf16
            # expert_index_offsets: [E,]
            expert_index_offsets = (
                jnp.arange(cfg.num_experts, dtype=jnp.float32) * expert_capacity
            )

            # position_in_expert_with_offset: [O, G, S*topk, E]
            position_in_expert_with_offset = position_in_expert + expert_index_offsets
            
            # Apply expert_mask and sum along S*topk axis to get tokens to index for each S.
            position_in_expert_with_offset = jnp.where(expert_mask == 0, 0, position_in_expert_with_offset)
            position_in_expert_with_offset = position_in_expert_with_offset.reshape(O, G, k , S, E)
            position_in_expert_with_offset = jnp.sum(position_in_expert_with_offset, axis=2)
            
        return position_in_expert_with_offset, expert_mask, expert_affinities_masked
    
    def compute_expert_index(self, cfg, raw_gates):
        O, G, S, E = raw_gates.shape
        with jax.named_scope("expert_index"):
            # expert index will be (O, G, S, top_k)
            # mapping from tokens to the chosen top_k experts
            # based on affinities
            # top_k happens on last axis of operand, so the expert axis
            k = min(cfg.top_k, cfg.num_experts)
            # expert_index: (O, G, S, top_k)
            _, expert_index = jax.lax.top_k(raw_gates, k)
            expert_index = expert_index.astype(jnp.int32)
            # expert_index: (O, G, top_k, S)
            expert_index = jnp.transpose(expert_index, (0, 1, 3, 2))
            # expert_index: (O, G, S * top_k)
            expert_index = expert_index.reshape(O, G, S*k)
        return expert_index

    # pylint: disable-next=too-many-statements
    def forward(self, logits: Tensor) -> NestedTensor:
        """Please see comments of BaseGating.forward."""
        cfg = self.config
        O, G, S, E = logits.shape
        
        raw_gates = self.router(cfg, logits)
        expert_capacity = self.compute_expert_capacity(cfg, logits)
        # expert_index: (O, G, S*top_k)
        expert_index = self.compute_expert_index(cfg, raw_gates)
        
        # expert_mask: (O, G, S*topk, E)
        expert_mask = self.compute_expert_mask(cfg, expert_index, cfg.num_experts)
        
        # Only use top 1 tokens for calculationg aux loss.
        with jax.named_scope("aux_loss"):
            aux_loss = self.compute_aux_loss(self.config, expert_mask[:, :, :S, :], raw_gates)
        
        # position_in_expert_with_offset: (O, G, k , S, E)
        # expert_mask_after_dropping: (O, G, S*topk, E)
        # expert_affinities_masked: (O, G, S, E)
        position_in_expert_with_offset, expert_mask_after_dropping, expert_affinities_masked = self.compute_positions_and_drop(
            expert_mask, raw_gates, expert_capacity, cfg
        )

        # Reshape expert_index back to O, G, S, topk
        expert_index = expert_index.reshape(O, G, cfg.top_k, S)
        expert_index = jnp.transpose(expert_index, (0, 1, 3, 2))

        with jax.named_scope("token_permutation_idx"):
            batch_idx = jnp.arange(position_in_expert_with_offset.shape[0])[:, None, None, None]
            group_idx = jnp.arange(position_in_expert_with_offset.shape[1])[None, :, None, None]
            seq_idx = jnp.arange(position_in_expert_with_offset.shape[2])[None, None, :, None]

            # token_permutation_idx: (O, G, S, top_k)
            # for each token we get the position in assigned experts from this tensor
            token_permutation_idx = position_in_expert_with_offset[
                batch_idx,
                group_idx,
                seq_idx,
                expert_index
            ].astype(jnp.int32)

        
        token_assignments = self.compute_token_assignments(token_permutation_idx, cfg.num_experts, expert_capacity)

        with jax.named_scope("zero_indexing"):
            # Indexing using these will result in the first token (index 0) being loaded in place of dropped tokens
            # However, the output from these will get masked out in the affinity scaling step
            token_permutation_idx = token_permutation_idx - 1
            token_assignments = token_assignments - 1
            zero_tensor = jnp.zeros(1, dtype=token_permutation_idx.dtype)
            token_permutation_idx = jnp.maximum(token_permutation_idx, zero_tensor)
            token_assignments = jnp.maximum(token_assignments, zero_tensor)
        
        router_z_loss = _router_z_loss(logits)

        with jax.named_scope("metrics"):
            total_num_dropped, dispatch_per_expert = self.compute_metrics(expert_mask, expert_mask_after_dropping)
            self.add_summaries(aux_loss, router_z_loss, total_num_dropped, dispatch_per_expert)
        return self.Output(
            combine_tensor=(token_permutation_idx, expert_index, expert_affinities_masked),
            dispatch_tensor=token_assignments,
            load_balance_loss=aux_loss,
            router_z_loss=router_z_loss,
        )

class TopKGatingGatherBlockwise(TopKGatingGather):
    @config_class
    class Config(TopKGating.Config):
        block_size: int = 512
    
    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
    
    def compute_num_blocks(self, expert_capacity):
        #TODO make block size == expert cap
        num_blocks = math.ceil(expert_capacity / self.config.block_size) * self.config.num_experts
        # num_blocks = min(num_blocks, num_tokens * self.config.top_k)
        logging.info("Setting number of blocks as %d", num_blocks)
        return num_blocks

    def get_token_position_to_id(
        self, cfg, block_position_indices, num_blocks,
    ):
        """
        Invert block_position_indices to obtain token_position_to_id.
        """

        O, G, num_tokens, E = block_position_indices.shape
        mesh = thread_resources.env.physical_mesh
        TP = mesh.shape["model"]

        # (O, G, S, E)
        # sharding_constraint = (("fsdp", "data"), "expert", "model", None)
        # tokens_idx_expanded = with_sharding_constraint(tokens_idx_expanded, sharding_constraint)
        # block_position_indices = with_sharding_constraint(block_position_indices, sharding_constraint)
        # token_position_to_id = with_sharding_constraint(token_position_to_id, sharding_constraint)
        # token_position_to_id_shape = token_position_to_id.shape

        tokens_indices = jnp.arange(num_tokens, dtype=jnp.int32)[None, None, :, None]
        tokens_indices = jnp.broadcast_to(tokens_indices, (O, G, num_tokens, E))

        # Create batch and group indices
        # (O, G, S*top_k, E)
        batch_indices = jnp.arange(O)[:, None, None, None]
        batch_indices = jnp.broadcast_to(batch_indices, (O, G, num_tokens, E))

        # (O, G, S*top_k, E)
        group_indices = jnp.arange(G)[None, :, None, None]
        group_indices = jnp.broadcast_to(group_indices, (O, G, num_tokens, E))

        # (O, G, S*top_k, E)
        # block_position_indices
        
        # Create scatter indices
        scatter_indices = jnp.stack([batch_indices, group_indices, block_position_indices], axis=-1, dtype=jnp.int32)
        
        # token_position_to_id: (O, G, N*B)
        # each entry represents position in block to token id
        # -1 will be for padding tokens
        token_position_to_id = jnp.zeros((O, G, num_blocks * cfg.block_size + 1), dtype=jnp.int32)
        
        token_position_to_id = jax.lax.scatter(
            token_position_to_id,
            scatter_indices,
            tokens_indices + 1,
            jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(0, 1, 2),
                scatter_dims_to_operand_dims=(0, 1, 2)
            )
        )
        # jax.debug.print("token_position_to_id before adjusting, {x}", x=token_position_to_id)
        token_position_to_id = token_position_to_id[:, :, 1:]
        # jax.debug.print("token_position_to_id after removing one from last axis, {x}", x=token_position_to_id)

        token_position_to_id = token_position_to_id - 1
        # jax.debug.print("token_position_to_id after sub1 {x}", x=token_position_to_id)
        token_position_to_id = jnp.where(token_position_to_id==-1, num_tokens,token_position_to_id)
        # jax.debug.print("token_position_to_id after replacing -1 {x}", x=token_position_to_id)
        # zero_tensor = jnp.zeros(1, dtype=token_position_to_id.dtype)
        # token_position_to_id = jnp.maximum(token_position_to_id, zero_tensor)
        token_position_to_id = self._remat_name(token_position_to_id, "blockwisegating.token_position_to_id")
        return token_position_to_id

    def forward(self, logits):
        cfg = self.config
        O, G, S, E = logits.shape
        raw_gates = self.router(cfg, logits)
        expert_capacity = self.compute_expert_capacity(cfg, logits)
        # expert_index: (O, G, S*top_k)
        expert_index = self.compute_expert_index(cfg, raw_gates)
        # expert_mask: (O, G, S*topk, E)
        expert_mask = self.compute_expert_mask(cfg, expert_index, cfg.num_experts)
        # Only use top 1 tokens for calculationg aux loss.
        aux_loss = self.compute_aux_loss(self.config, expert_mask[:, :, :S, :], raw_gates)

        # [O, G, S*topk, E], [O, G, S*topk, E], [O, G, S, E]
        _, expert_mask_after_dropping, expert_affinities_masked = self.compute_positions_and_drop(
            expert_mask, raw_gates, expert_capacity, cfg
        )

        expert_mask_after_dropping = jnp.reshape(expert_mask_after_dropping, (O, G, -1, S, E))
        expert_mask_after_dropping = jnp.sum(expert_mask_after_dropping, axis=2)

        num_dropped = jnp.sum(expert_mask, axis=(0,1,2,3)) - jnp.sum(expert_mask_after_dropping, axis=(0,1,2,3))
        # jax.debug.print('num_dropped, {x}', x=num_dropped)
        # jax.debug.print('total_num_dropped, {x}', x=jnp.sum(num_dropped))
        num_blocks = self.compute_num_blocks(expert_capacity)
        
        # blocks_per_expert: (O, G, E)
        blocks_per_expert = jnp.repeat(math.ceil(expert_capacity/cfg.block_size), E, axis=0)
        blocks_per_expert = jnp.expand_dims(blocks_per_expert, (0, 1))
        # print('blocks per expert', blocks_per_expert.shape)
        # jax.debug.print('blocks_per_expert, {x}', x=blocks_per_expert)
        blocks_ids = jnp.arange(num_blocks, dtype=jnp.int32)

        # num_blocks_idx_expanded: (1, 1, num_blocks, 1)
        num_blocks_idx_expanded = jnp.expand_dims(blocks_ids, (0, 1, 3))
        # num_blocks_idx_expanded: (O, G, num_blocks, E)
        num_blocks_idx_expanded = jnp.broadcast_to(num_blocks_idx_expanded, (O, G, num_blocks, 1))
        # print('num blocks', num_blocks_idx_expanded.shape)
        # jax.debug.print('num blocks: {x}', x=num_blocks_idx_expanded)

        # cumulative_blocks_per_expert: (O, G, E)
        cumulative_blocks_per_expert = jnp.cumsum(blocks_per_expert, axis=2, dtype=jnp.int32)
        # print('num blocks', num_blocks_idx_expanded.shape)
        # print('cumulative_blocks_per_expert.shape', cumulative_blocks_per_expert.shape)
        # jax.debug.print("cumulative_blocks_per_expert: {x}", x=cumulative_blocks_per_expert)

        # print('expert mask shape', expert_mask_after_dropping.shape)
        # jax.debug.print("expert_affinities_masked: {x}", x=expert_affinities_masked)
        # jax.debug.print("expert_mask_after_dropping: {x}", x=expert_mask_after_dropping)

        # block_to_expert: (O, G, N)
        #   N is num blocks
        #   for each block, which expert it belongs to
        block_to_expert = jnp.sum(
            num_blocks_idx_expanded >=  jnp.expand_dims(cumulative_blocks_per_expert, 2)[:,:,:,:-1], 
            axis=3
        )

        # (O, G, S, E)
        # after masking this represents for each token, 
        # the position in blocks if all tokens in blocks were laid out in a linear array
        # b0t0, b0t1,..., b1t0, b1t1,...
        expert_mask_after_dropping = with_sharding_constraint(expert_mask_after_dropping, PartitionSpec(("fsdp", "data"), "expert", None, None))
        block_position_indices = _cum_sum(expert_mask_after_dropping.astype(jnp.int32), axis=-2).astype(jnp.int32)  #constrain shapes around cum_sum to avoid collective-permutes for index-sharding
        block_position_indices = with_sharding_constraint(block_position_indices, PartitionSpec(("fsdp", "data"), "expert", None, None))

        # print("block_position_indices shape", block_position_indices.shape)
        # jax.debug.print("block_position_indices after cumsum:{x}", x=block_position_indices)
        # O G 1 E
        expert_block_offsets = jnp.expand_dims(cumulative_blocks_per_expert * cfg.block_size, 2)
        # print('expert_block_offsets shape', expert_block_offsets.shape)
        
        block_position_indices = block_position_indices.at[:,:,:,1:].set(block_position_indices[:,:,:,1:] + expert_block_offsets[:,:,:,:-1])
        # jax.debug.print("expert_block_offsets :{x}", x=expert_block_offsets)
        # print("block_position_indices shape", block_position_indices.shape)
        block_position_indices = jnp.where(expert_mask_after_dropping==0, 0, block_position_indices)
        # print("block_position_indices after dropping and adding expert block offsets: shape", block_position_indices.shape)
        # jax.debug.print("block_position_indices after dropping and adding expert block offsets: {x}", x=block_position_indices)

        # token_position_to_id: (O, G, N*B)
        # for every position in the block, gets the token id in sequence
        # TODO: use same fn for both shard and unsharded path
        if not _USING_INDEX_SHARDING:
            token_position_to_id = self.get_token_position_to_id(cfg, block_position_indices, num_blocks,)
        else: 
            mesh = thread_resources.env.physical_mesh 
            T = mesh.shape["model"] 
            output = jnp.zeros((T, O, G, num_blocks*cfg.block_size), dtype=jnp.int32) 

            #create full tokens_indices and then shard within TP
            tokens_indices = jnp.arange(S, dtype=jnp.int32)[None, None, :, None]
            tokens_indices = jnp.broadcast_to(tokens_indices, (O, G, S, E))
            token_position_to_id_sm = shard_map(
                calculate_token_position_to_id,
                mesh=thread_resources.env.physical_mesh,
                in_specs=(
                    PartitionSpec(("fsdp", "data"), "expert", "model", None), 
                    PartitionSpec(("fsdp", "data"), "expert", "model", None), 
                    None,
                    None,
                    None,
                    PartitionSpec("model", ("fsdp", "data"), "expert",  None),
                ),
                out_specs=PartitionSpec("model", ("fsdp", "data"), "expert",  None),
                check_rep=False
                )
            output = token_position_to_id_sm(block_position_indices, tokens_indices, num_blocks, cfg.block_size, S, output)  # (TP, O, G, N*B)
            token_position_to_id  = jnp.min(output, axis=0)                                                                  # allreduce to get (O, G, N*B)

        router_z_loss = _router_z_loss(logits)
        return self.Output(
            dispatch_tensor=block_to_expert,
            combine_tensor=(token_position_to_id, expert_affinities_masked),
            load_balance_loss=aux_loss,
            router_z_loss=router_z_loss,
        )

class TopKGatingGatherBlockwiseV2(TopKGatingGather):
    def forward(self, logits):
        cfg = self.config
        O, G, S, E = logits.shape
        k = cfg.top_k
        raw_gates = self.router(cfg, logits)
        expert_capacity = self.compute_expert_capacity(cfg, logits)

        # expert_index: (O, G, S*top_k)
        expert_index = self.compute_expert_index(cfg, raw_gates)

        # expert_mask: (O, G, S*topk, E)
        expert_mask = self.compute_expert_mask(cfg, expert_index, cfg.num_experts)

        # Only use top 1 tokens for calculationg aux loss.
        aux_loss = self.compute_aux_loss(self.config, expert_mask[:, :, :S, :], raw_gates)

        # Compute cumulative sums of assignment
        # indicators for each expert, i.e. index e \in 0..E-1 independently.
        # cumsum over S dim
        # position_in_expert: [O, G, S*topk, E]
        expert_mask = with_sharding_constraint(expert_mask, PartitionSpec(("fsdp", "data"), "expert", None, None))
        position_in_expert = _cum_sum(expert_mask.astype(jnp.int32), axis=-2).astype(jnp.float32)
        position_in_expert = with_sharding_constraint(position_in_expert, PartitionSpec(("fsdp", "data"), "expert", None, None))
        expert_mask_pre_capacity_drop = expert_mask
        expert_mask_k_pre_capacity_drop = expert_mask_pre_capacity_drop.reshape(O, G, k, S, E)
        expert_mask_k_pre_capacity_drop = jnp.sum(expert_mask_k_pre_capacity_drop, axis=2)

        # expert_affinities_masked: [O, G, S, E]
        expert_affinities_masked = self.compute_expert_affinities_masked(
            raw_gates, expert_mask_k_pre_capacity_drop, normalize_top_k_affinities=True
        )

        expert_mask = jnp.where(position_in_expert > expert_capacity, 0, expert_mask)
        expert_mask_k = expert_mask.reshape(O, G, k, S, E)
        expert_mask_k = jnp.sum(expert_mask_k, axis=2)
        expert_affinities_masked = jnp.where(expert_mask_k == 0, 0, expert_affinities_masked)
        expert_mask_k = with_sharding_constraint(expert_mask_k, PartitionSpec(("fsdp", "data"), "expert", None, None))
        
        position_in_expert = _cum_sum(expert_mask_k.astype(jnp.int32), axis=-2).astype(jnp.int32)
        position_in_expert = with_sharding_constraint(position_in_expert, PartitionSpec(("fsdp", "data"), "expert", None, None))

        # Add expert offset to the position_in_expert
        # expert_index_offsets: [E,]
        expert_index_offsets = (
            jnp.arange(cfg.num_experts, dtype=jnp.int32) * expert_capacity
        )
        block_to_expert = jnp.arange(E, dtype=jnp.int32)[None, None, :]
        block_to_expert = jnp.broadcast_to(block_to_expert, (O, G, E))

        # position_in_expert_with_offset: [O, G, S*topk, E]
        position_in_expert_with_offset = position_in_expert + expert_index_offsets

        # Apply expert_mask and sum along S*topk axis to get tokens to index for each S.
        position_in_expert_with_offset = jnp.where(expert_mask_k == 0, 0, position_in_expert_with_offset)

        # token_position_to_id: (O, G, N*B)
        # for every position in the block, gets the token id in sequence
        mesh = thread_resources.env.physical_mesh
        T = mesh.shape["model"]
        output = jnp.zeros((T, O, G, E*expert_capacity), dtype=jnp.int32)

        # create full tokens_indices and then shard within TP
        tokens_indices = jnp.arange(S, dtype=jnp.int32)[None, None, :, None]
        tokens_indices = jnp.broadcast_to(tokens_indices, (O, G, S, E))
        token_position_to_id_sm = shard_map(
            calculate_token_position_to_id,
            mesh=thread_resources.env.physical_mesh,
            in_specs=(
                PartitionSpec(("fsdp", "data"), "expert", "model", None),
                PartitionSpec(("fsdp", "data"), "expert", "model", None),
                None, None, None,
                PartitionSpec("model", ("fsdp", "data"), "expert",  None),
            ),
            out_specs=PartitionSpec("model", ("fsdp", "data"), "expert",  None),
            check_rep=False
            )
        # (TP, O, G, N*B)
        output = token_position_to_id_sm(position_in_expert_with_offset, tokens_indices, E, expert_capacity, S, output)
        # allreduce to get (O, G, N*B)
        token_position_to_id  = jnp.min(output, axis=0)
        router_z_loss = _router_z_loss(logits)
        return self.Output(
            dispatch_tensor=block_to_expert,
            combine_tensor=(token_position_to_id, expert_affinities_masked),
            load_balance_loss=aux_loss,
            router_z_loss=router_z_loss,
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
        norm: Union[
            BaseNormalizationLayer.Config, dict[NormPosition, BaseNormalizationLayer.Config]
        ] = LayerNorm.default_config()
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
        # Gating function. Currently only "top2" is supported.
        gating: BaseGating.Config = Top2Gating.default_config()
        # Weight for the load balancing loss. Default to 0.01 as in
        # https://arxiv.org/pdf/2112.06905.
        load_balance_loss_weight: float = 0.01
        # Weight for the router z loss. https://arxiv.org/abs/2202.08906.
        router_z_loss_weight: float = 0.0

        # SPMD partition params used to represent the MoE layer dimensions.
        # O - outer batch dim
        # M - input dim, same as output dim
        # E - experts dim
        # G - groups dim
        # C - experts capacity dim
        # H - hidden dim
        # S - sequence dim
        dim_to_mesh_axis_map: dict[str, Optional[PartitionSpec]] = {}

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        # Table 1 of https://arxiv.org/abs/2405.15052.
        MOE_OUTER_BATCH_AXIS_NAMES = ("data", "fsdp")
        cfg.dim_to_mesh_axis_map = {
            "me": PartitionSpec(None, None),
            "emh": PartitionSpec("expert", None, "model"),
            "ehm": PartitionSpec("expert", "model", None),
            "ogsm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, "model"),
            "ogsM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
            "ogse": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
            "ogec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
            # Dispatch and combine tensors.
            "ogsec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, "expert", None),
            "oegcm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, "model"),
            "oegcM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, None),
            "ogecm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, "expert", None, "model"),
            "ogecM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, "expert", None, None),
            "oegch": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, "model"),
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
        cfg: TransformerFeedForwardMoE.Config = self.config
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
            x += inputs
            x = self.out_norm(x) if NormPosition.OUT_NORM in cfg.norm else x
        else:
            raise NotImplementedError(cfg.structure)
        return x

    def _dispatch_and_combine_with_gather_gating(self, cfg, group_len, gating, x):
        input_dtype = x.dtype
        with jax.named_scope("dispatch"):
            # token_assignments: (O, G, E, C)
            token_assignments= gating.dispatch_tensor
            
            token_assignments = with_sharding_constraint(token_assignments, cfg.dim_to_mesh_axis_map["ogec"])
            O, G, E, C = token_assignments.shape
            _, _, _, M = x.shape

            idx_o = jnp.arange(O)[:, None, None, None]  # (O, 1, 1, 1)
            idx_g = jnp.arange(G)[None, :, None, None]  # (1, G, 1, 1)
            expert_aligned_hidden_states = x[idx_o, idx_g, token_assignments] # (O, G, E, C, M)
            expert_aligned_hidden_states = jnp.einsum("ogecm->oegcm", expert_aligned_hidden_states)

            expert_aligned_hidden_states = with_sharding_constraint(expert_aligned_hidden_states, cfg.dim_to_mesh_axis_map["oegcM"])

        # Perform MLP operations
        with jax.named_scope("expert_compute"):
            x = self._wi_activation(expert_aligned_hidden_states)
            if cfg.structure in ["prenorm", "hybridnorm", "nonorm"]:
                x = self.dropout1(x)
            if not _USING_SHARDMAP_FFN:
                x = jnp.einsum("oegch,ehm->oegcm", x, self.parameters["wo_weight"])
            else:
                down_proj_sm = shard_map(
                    down_proj, 
                    mesh=thread_resources.env.physical_mesh,
                    in_specs=(
                        cfg.dim_to_mesh_axis_map["oegch"],
                        cfg.dim_to_mesh_axis_map["ehM"],
                    ),
                    out_specs=cfg.dim_to_mesh_axis_map["oegcM"],
                    check_rep=False
                )
                x = down_proj_sm(x, self.parameters["wo_weight"])
                x = self._remat_name(x, f"linear2")
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["oegcM"])
            x = jnp.einsum("oegcm->ogecm", x)
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogecM"])
        
        with jax.named_scope("output_combine"):
            # flatten token outputs
            # (O, G, S, top_k), (O, G, S, top_k), (O, G, S, E)
            token_permutation_idx, expert_index, expert_affinities_masked = gating.combine_tensor
            token_permutation_idx, expert_index, expert_affinities_masked = token_permutation_idx.astype(jnp.int32), expert_index.astype(jnp.int32), expert_affinities_masked.astype(input_dtype)
            token_permutation_idx = with_sharding_constraint(token_permutation_idx, cfg.dim_to_mesh_axis_map["ogse"])
            expert_affinities_masked = with_sharding_constraint(expert_affinities_masked, cfg.dim_to_mesh_axis_map["ogse"])
            permuted_output = jnp.reshape(x, (O, G, E*C, M))
            permuted_output = with_sharding_constraint(permuted_output, cfg.dim_to_mesh_axis_map["ogsM"])

            if _USING_SHARDMAP_FFN:
                mesh = thread_resources.env.physical_mesh
                T = mesh.shape["model"]
                output = jnp.zeros((T, O, G, group_len, cfg.input_dim), dtype=permuted_output.dtype)
                min_k = min(self.config.gating.top_k, self.config.num_experts)
                combine_outputs_sm = shard_map(
                    combine_outputs, 
                    mesh=thread_resources.env.physical_mesh,
                    in_specs=(
                        cfg.dim_to_mesh_axis_map["ogsM"],
                        cfg.dim_to_mesh_axis_map["ogse"],
                        cfg.dim_to_mesh_axis_map["ogse"],
                        cfg.dim_to_mesh_axis_map["ogse"],
                        cfg.dim_to_mesh_axis_map["hoesm"],
                    ),
                    out_specs=cfg.dim_to_mesh_axis_map["hoesm"],
                    check_rep=False
                )
                output = combine_outputs_sm(
                    permuted_output, token_permutation_idx, expert_index, expert_affinities_masked, output
                )
                output = jnp.sum(output, axis=0, dtype=permuted_output.dtype)
            else:
                output = jnp.zeros((O, G, group_len, cfg.input_dim), dtype=input_dtype)
                min_k = min(self.config.gating.top_k, self.config.num_experts)
                for k in range(min_k):
                    # indices: (O, G, S)
                    indices = token_permutation_idx[..., k]
                    # indices: (O, G, S, 1)
                    indices = jnp.expand_dims(indices, axis=3)
                    # index into permuted_output
                    # output_k : (O, G, S, M)
                    output_k = jnp.take_along_axis(permuted_output, indices, axis=2)
                    output_k = with_sharding_constraint(output_k, cfg.dim_to_mesh_axis_map["ogsM"])

                    # expert_affinities_masked: (O, G, S, 1) after indexing the expert
                    kth_expert_index = jnp.expand_dims(expert_index[..., k], axis=-1)
                    expert_affinities_k = jnp.take_along_axis(expert_affinities_masked, kth_expert_index, axis=-1) # Result shape: (O, G, S, 1)

                    output += output_k * expert_affinities_k
            return output

    def _dispatch_and_combine_with_gather_blockwise_gating(self, cfg, gating, hidden_states):
        """
        Args
        - cfg: Config
        - gating: Output of the gating function
          - combine_tensor: 
            - token_position_to_id (O, G, N*B)
            - expert_affinities_masked (O, G, S, E)
          - dispatch_tensor: 
            - block_to_expert (O, G, N)
        - hidden_states: (O, G, S, M)
        """
        mesh = thread_resources.env.physical_mesh
        if isinstance(cfg.hidden_dim, int):
            hidden_dim = cfg.hidden_dim
        else:
            hidden_dim = cfg.hidden_dim.set(input_dim=cfg.input_dim).instantiate()
        
        MOE_OUTER_BATCH_AXIS_NAMES = ("data", "fsdp")
        expert_affinities_masked = gating.combine_tensor[1]
        token_position_to_id = gating.combine_tensor[0]
        block_to_expert = gating.dispatch_tensor
        num_blocks = block_to_expert.shape[-1]
        block_size = token_position_to_id.shape[-1] // num_blocks
        gate_up_weight = jnp.stack([self.parameters["wi_0_weight"],self.parameters["wi_1_weight"],], axis=2)
        gate_up_weight = with_sharding_constraint(gate_up_weight, PartitionSpec("expert", "fsdp", None, "model"))

        # TODO: fix checkpointing as it has needs different out_specs
        partitioned_blockwise_mm = shard_map(
            blockwise_mlp,
            mesh=mesh,
            in_specs=(
                cfg.dim_to_mesh_axis_map["ogsM"], # hidden_states
                cfg.dim_to_mesh_axis_map["ogse"], # expert_affinities_masked
                PartitionSpec("expert", None, None, "model"), # gate_up_proj weight
                PartitionSpec("expert", "model", None), # down_proj weight
                PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None), # token_position_to_id
                PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None), # block_to_expert
                None, # block size
                None, # activation_fns
            ),
            out_specs=(
                PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", "model", None, None)
            ),
            check_rep=False
        )
        outputs = partitioned_blockwise_mm(
            hidden_states,
            expert_affinities_masked,
            gate_up_weight,
            self.parameters["wo_weight"], 
            token_position_to_id, 
            block_to_expert, 
            block_size,
            cfg.activation
        )
        outputs = jnp.sum(outputs, axis=2, dtype=outputs.dtype)
        return outputs

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
        x = x.reshape([outer_batch, num_groups, group_len, cfg.input_dim])
        # O may be lower than fsdp axis
        # TODO(huilgolr): what to do here

        x = with_sharding_constraint(x, PartitionSpec(("data", "fsdp"), "expert", None))
        with jax.named_scope("router"):
            logits = jnp.einsum("ogsm,me->ogse", x, self.parameters["gate_weight"])

        # Perform gating based on logits. Casting to float32 precision is usually needed for
        # stable performance.
        with jax.named_scope("gating"):
            gating = self.gating(logits=logits)

        # Collect aux_loss.
        aux_loss = (
            gating.load_balance_loss * cfg.load_balance_loss_weight
            + gating.router_z_loss * cfg.router_z_loss_weight
        )
        self.add_module_output("aux_loss", aux_loss)
        if isinstance(self.gating, (TopKGatingGatherBlockwise, TopKGatingGatherBlockwiseV2)):
            x = self._dispatch_and_combine_with_gather_blockwise_gating(cfg, gating, x)
        elif isinstance(self.gating, TopKGatingGather):
            x = self._dispatch_and_combine_with_gather_gating(cfg, group_len, gating, x)
        else:
            combine_tensor = gating.combine_tensor.astype(input_dtype)
            dispatch_tensor = gating.dispatch_tensor.astype(input_dtype)
            combine_tensor = with_sharding_constraint(combine_tensor, cfg.dim_to_mesh_axis_map["ogsec"])
            dispatch_tensor = with_sharding_constraint(
                dispatch_tensor, cfg.dim_to_mesh_axis_map["ogsec"]
            )
            x = jnp.einsum("ogsec,ogsm->oegcm", dispatch_tensor, x)
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["oegcm"])
            x = self._wi_activation(x)
            if cfg.structure in ["prenorm", "hybridnorm", "nonorm"]:
                x = self.dropout1(x)
            with child_context("wo_einsum", module=self):
                x = self.einsum_maybe_quantized(
                    "oegch,ehm->oegcm", activation=x, kernel=self.parameters["wo_weight"]
                )
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["oegcm"])
            x = jnp.einsum("oegcm->ogecm", x)
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogecm"])
            x = jnp.einsum("ogecm,ogsec->ogsm", x, combine_tensor)
            x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogsm"])
            # (batch, seq_len, input_dim)

        out = x.reshape(token_shape + (cfg.input_dim,))
        return out

    def _wi_activation(self, x: Tensor) -> Tensor:
        cfg = self.config
        if isinstance(cfg.activation, tuple):
            activations = []
            for i, activation in enumerate(cfg.activation):
                with child_context(f"wi_{i}_einsum", module=self):
                    x_i = self.einsum_maybe_quantized(
                        "oegcm,emh->oegch", activation=x, kernel=self.parameters[f"wi_{i}_weight"]
                    )
                x_i = with_sharding_constraint(x_i, cfg.dim_to_mesh_axis_map["oegch"])
                x_i = self._remat_name(x_i, f"linear1_{i}")
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

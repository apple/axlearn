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
from typing import Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.experimental.pjit import pjit

from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import (
    REQUIRED,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
)
from axlearn.common.layers import (
    BaseNormalizationLayer,
    Dropout,
    LayerNorm,
    StochasticDepth,
    get_activation_fn,
)
from axlearn.common.module import Module
from axlearn.common.param_init import FanAxes
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    PartitionSpec,
    Tensor,
    VDict,
    flatten_items,
    get_recursively,
    set_recursively,
    tree_paths,
    with_sharding_constraint,
)


def _router_z_loss(logits: Tensor) -> Tensor:
    """Loss that encourages router logits to remain small and improves stability.

    Reference:
    https://github.com/tensorflow/mesh/blob/fbf7b1e547e8b8cb134e81e1cd350c312c0b5a16/mesh_tensorflow/transformer/moe.py#L1956

    Args:
        logits: A tensor with shape (batch, num_experts).

    Returns:
        z_loss: A scalar loss.
    """
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
            capacity_factor=cfg.train_capacity_factor
            if self.is_training
            else cfg.eval_capacity_factor,
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
        self.add_summary("load_balance_loss", aux_loss)
        self.add_summary("router_z_loss", router_z_loss)
        self.add_summary("dispatch_0", dispatch_0)
        self.add_summary("dispatch_1", dispatch_1)
        self.add_summary("dispatch_2", dispatch_2)
        self.add_summary("over_capacity_1", over_capacity_1)
        self.add_summary("over_capacity_2", over_capacity_2)

        return self.Output(
            combine_tensor=combine_tensor,
            dispatch_tensor=dispatch_tensor,
            load_balance_loss=aux_loss,
            router_z_loss=router_z_loss,
        )


class TransformerFeedForwardMoE(BaseLayer):
    """A Transformer feed-forward layer with mixture of experts.

    This is a drop-in replacement of the `TransformerFeedForwardLayer` class.

    https://github.com/google/praxis/blob/b059aa12a62a1b675d95a66088f8d0593baa48a5/praxis/layers/transformers.py#L510
    https://github.com/tensorflow/lingvo/blob/da1a75b2fea79ee542e7ae735f92032088eda055/lingvo/jax/layers/transformers.py#L415
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures TransformerFeedForwardMoE."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        hidden_dim: Required[Union[int, FunctionConfigBase]] = REQUIRED  # Hidden feature dim.
        # If greater than 1, we reshape all tensors from (batch, seq_len, dim) to
        # (outer_batch, inner_batch, seq_len, dim). This is useful for 3D mesh. Reference:
        # https://github.com/tensorflow/mesh/blob/fbf7b1e547e8b8cb134e81e1cd350c312c0b5a16/mesh_tensorflow/transformer/moe.py#L294-L336
        outer_batch: int = 1
        norm: BaseNormalizationLayer.Config = LayerNorm.default_config()
        activation: Union[str, Tuple[str, str]] = "nn.relu"
        dropout: InstantiableConfig = Dropout.default_config()
        stochastic_depth: InstantiableConfig = StochasticDepth.default_config()
        # The inner structure of the layer: "prenorm", "postnorm", "hybridnorm", "nonorm".
        # * prenorm: y = x + feedforward(norm(x))
        # * postnorm: y = norm(x + feedforward(x))
        # * hybridnorm: y = postnorm(x + feedforward(prenorm(x)))
        # * nonorm: y = feedforward(x)   # no residual, which is usually applied externally.
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
        dim_to_mesh_axis_map: Dict[str, Optional[PartitionSpec]] = {}

    @classmethod
    def default_config(cls) -> Config:
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

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
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
        if cfg.structure in ["prenorm", "hybridnorm", "nonorm"]:
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
        x = x.reshape([outer_batch, num_groups, group_len, cfg.input_dim])
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogsm"])
        logits = jnp.einsum("ogsm,me->ogse", x, self.parameters["gate_weight"])
        # Perform gating based on logits. Casting to float32 precision is usually needed for
        # stable performance.
        gating = self.gating(logits=logits)
        # Collect aux_loss.
        aux_loss = (
            gating.load_balance_loss * cfg.load_balance_loss_weight
            + gating.router_z_loss * cfg.router_z_loss_weight
        )
        self.add_module_output("aux_loss", aux_loss)
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
        x = jnp.einsum("oegch,ehm->oegcm", x, self.parameters["wo_weight"])
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["oegcm"])
        x = jnp.einsum("oegcm->ogecm", x)
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogecm"])
        x = jnp.einsum("ogecm,ogsec->ogsm", x, combine_tensor)
        x = with_sharding_constraint(x, cfg.dim_to_mesh_axis_map["ogsm"])
        # (batch, seq_len, input_dim)
        return x.reshape(token_shape + (cfg.input_dim,))

    def _wi_activation(self, x: Tensor) -> Tensor:
        cfg = self.config
        if isinstance(cfg.activation, tuple):
            activations = []
            for i, activation in enumerate(cfg.activation):
                x_i = jnp.einsum("oegcm,emh->oegch", x, self.parameters[f"wi_{i}_weight"])
                x_i = with_sharding_constraint(x_i, cfg.dim_to_mesh_axis_map["oegch"])
                x_i = get_activation_fn(activation)(x_i)
                activations.append(x_i)
            assert len(activations) == 2, cfg.activation
            return activations[0] * activations[1]
        else:
            x = jnp.einsum("oegcm,emh->oegch", x, self.parameters["wi_weight"])
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
    moe_parameters = jax.tree_util.tree_map(lambda x: None, moe_parameter_specs)
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
        dispatch = with_sharding_constraint(jnp.ones([num_experts], dtype=value.dtype), ("expert",))
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
                layer_parameters = jax.tree_util.tree_map(
                    lambda w: w[layer_index], source_layer_parameters
                )
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

    out_shardings = jax.tree_util.tree_map(
        compute_out_sharding, tree_paths(target_parameter_specs), target_parameter_specs
    )
    target_parameters = pjit(convert_fn, out_shardings=out_shardings)(source_parameters)
    target_parameters = jax.tree_util.tree_map(
        lambda spec, param: spec if param is None else param,
        target_parameter_specs,
        target_parameters,
    )
    return target_parameters

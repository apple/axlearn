# Copyright © 2023 Apple Inc.

"""A Learner is responsible for computing and applying updates to model params, including:

- Computing and applying updates from gradients through optimizer modules;
- Applying updates on non-differentiable params such as batch norm stats;
- Maintaining Polyak averages of model params (if enabled).
"""

import dataclasses
import enum
from typing import Optional, Sequence, Tuple

import jax
import optax
from jax import numpy as jnp

from axlearn.common.base_layer import NestedParameterSpec
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
)
from axlearn.common.module import Module
from axlearn.common.optimizer_base import NestedOptParam, PartitionedGradientTransformation
from axlearn.common.optimizers import param_ema
from axlearn.common.utils import (
    NestedPartitionSpec,
    NestedTensor,
    Tensor,
    flatten_items,
    match_regex_rules,
    register_per_param_settings,
    tree_paths,
)


class UpdateType(enum.Enum):
    """UpdateType specifies which update types are allowed for the parameter.

    If optimizer updates are not allowed on a parameter, no gradient will be computed (assuming
    that trainer respects the return values of `Learner.should_update_with_optimizers`) and
    no optimizer state will be maintained for the parameter. All optimizer-based updates based on
    gradients or weight decay will be disabled.

    If state updates are not allowed on a parameter, OutputCollection.state_updates will *not*
    be applied on the parameter. State updates are typically used to maintain stats such as
    moving stats for BatchNorm. This can be used to disable such updates.
    """

    # Freeze the parameter completely.
    NO_UPDATE = "no_update"

    # Only optimizer-based updates are allowed. This can be specified instead of ALL_UPDATES to
    # avoid accidental state updates on the parameter.
    OPTIMIZERS = "optimizers"

    # Only state updates are allowed. Compared with ALL_UPDATES, this saves optimizer states.
    STATE_UPDATES = "state_updates"

    # All updates are allowed. The default behavior.
    ALL_UPDATES = "all_updates"


def should_update_with_optimizers(update_type: UpdateType) -> bool:
    return update_type in (UpdateType.OPTIMIZERS, UpdateType.ALL_UPDATES)


def should_apply_state_updates(update_type: UpdateType) -> bool:
    return update_type in (UpdateType.STATE_UPDATES, UpdateType.ALL_UPDATES)


class BaseLearner(Module):
    """The base class of a learner."""

    def init(self, model_params: NestedOptParam) -> NestedTensor:
        """Initializes learner state."""
        raise NotImplementedError(type(self))

    def create_state_partition_specs(
        self, model_param_specs: NestedParameterSpec
    ) -> NestedPartitionSpec:
        """Creates learner state partition_specs."""
        raise NotImplementedError(type(self))

    def update(
        self, *, model_params: NestedOptParam, gradients: NestedTensor, state_updates: NestedTensor
    ) -> NestedTensor:
        """Computes `model_params` updates with `gradients` and `state_updates`.

        Args:
            model_params: A nested structure with OptParams as leaf nodes.
            gradients: Gradients on model_params. Must have the same structure as `model_params`
                except that the leaf values will be None for parameters not to be updated by
                optimizers.
            state_updates: The updated values for non-learnable parameters in `model_params`.
                A potentially trimmed tree of `model_params`.

        Returns:
            The updated model parameters. The learner state updates will be placed in the output
            collection's 'state_update' section.
        """
        raise NotImplementedError(type(self))

    def should_update_with_optimizers(self, model_params: NestedOptParam) -> dict:
        """Returns whether each parameter should be updated with the optimizers.

        This is used in trainer to skip gradient computation in the backward pass.

        Args:
            model_params: A nested dict.

        Returns:
            A nested dict with the same structure as `model_params` with boolean leaf values.
        """
        raise NotImplementedError(type(self))


class Learner(BaseLearner):
    """The learner module."""

    @config_class
    class Config(BaseLearner.Config):
        """Configures Learner."""

        optimizer: Required[InstantiableConfig] = REQUIRED  # The optimizer config.

        # A sequence of (param_path_regex, Optional[UpdateType]) pairs to specify which update
        # types are allowed on each parameter.
        #
        # Given a `param_path`, the first `re.fullmatch(param_path_regex, param_path)` determines
        # the update type.
        #
        # If none of the rules matches the param path, assume value ALL_UPDATES, i.e., all updates
        # are allowed.
        update_rules: Sequence[Tuple[str, Optional[UpdateType]]] = []

        # Set ema.decay to enable Polyak averaging of model params.
        #
        # The averages will be stored in self.state["ema"] and will have the same structure as
        # model_params.
        #
        # See optimizers.param_ema for more details.
        ema: InstantiableConfig = config_for_function(param_ema)
        # Whether to add per variable gradient and norm summaries. Enable it will make
        # the training slower since the summary is computed for every step.
        enable_per_variable_summaries: bool = False

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self.optimizer: PartitionedGradientTransformation = cfg.optimizer.instantiate()
        if not isinstance(self.optimizer, PartitionedGradientTransformation):
            raise ValueError(
                f"optimizer must be a PartitionedGradientTransformation: {cfg.optimizer}"
            )
        if cfg.ema.decay is not None:
            self.ema: PartitionedGradientTransformation = cfg.ema.instantiate()

    def create_state_partition_specs(
        self, model_param_specs: NestedParameterSpec
    ) -> NestedPartitionSpec:
        optimizer_model_param_specs = self._get_optimizer_model_params(model_param_specs)
        partition_state = dict(optimizer=self.optimizer.partition(optimizer_model_param_specs))
        if self.config.ema.decay is not None:
            partition_state["ema"] = self.ema.partition(model_param_specs)
        return partition_state

    def _get_optimizer_model_params(self, model_params: NestedOptParam):
        should_update_params = self.should_update_with_optimizers(model_params)
        return jax.tree_util.tree_map(
            lambda should_update, param: param if should_update else None,
            should_update_params,
            model_params,
        )

    def init(self, model_params: NestedOptParam) -> NestedTensor:
        update_types = self._update_types(model_params)
        register_per_param_settings(update_types, description="learner_update_type")
        state = dict(
            optimizer=self.optimizer.init(self._get_optimizer_model_params(model_params)),
        )
        if self.config.ema.decay is not None:
            state["ema"] = self.ema.init(model_params)
        return state

    def _update_types(self, tree: dict) -> dict:
        cfg = self.config
        return jax.tree_util.tree_map(
            lambda path: match_regex_rules(
                path, rules=cfg.update_rules, default_value=UpdateType.ALL_UPDATES
            ),
            tree_paths(tree),
        )

    def should_update_with_optimizers(self, model_params: dict) -> dict:
        """Returns whether each parameter should be updated with the optimizers.

        Args:
            model_params: A nested dict.

        Returns:
            A nested dict with the same structure as `model_params` with boolean leaf values.
        """
        return jax.tree_util.tree_map(
            should_update_with_optimizers, self._update_types(model_params)
        )

    def update(
        self, *, model_params: NestedOptParam, gradients: NestedTensor, state_updates: NestedTensor
    ) -> NestedTensor:
        """Computes `model_params` updates with `gradients` and `state_updates`.

        Args:
            model_params: A nested structure with OptParams as leaf nodes.
            gradients: Gradients on model_params. Must have the same structure as `model_params`
                except that the leaf values will be None for parameters not to be updated by
                optimizers.
            state_updates: The updated values for non-learnable parameters in `model_params`.
                A potentially trimmed tree of `model_params`.

        Returns:
            The updated model parameters. The learner state updates will be placed in the output
            collection's 'state_update' section.
        """
        cfg = self.config
        optimizer_model_params = self._get_optimizer_model_params(model_params)
        optimizer_parameter_updates, optimizer_state = self.optimizer.update(
            gradients,
            state=self.state["optimizer"],
            params=optimizer_model_params,
        )
        self.add_state_update("optimizer", optimizer_state)
        if cfg.enable_per_variable_summaries:
            param_rms = jax.tree_util.tree_map(
                lambda p: optax.safe_root_mean_squares(p.value, min_rms=1e-3), model_params
            )
            for p, p_n in flatten_items(param_rms):
                self.add_summary(f"param_rms/{p}", p_n)
            grad_rms = jax.tree_util.tree_map(
                lambda p: optax.safe_root_mean_squares(p, min_rms=1e-3), gradients
            )
            for p, g_n in flatten_items(grad_rms):
                self.add_summary(f"grad_rms/{p}", g_n)

        # Set `parameter_updates` to 0 if the param is not updated by the optimizer.
        parameter_updates = jax.tree_util.tree_map(
            lambda should_update_with_optimizers, param, update: (
                update if should_update_with_optimizers else jnp.zeros_like(param.value)
            ),
            self.should_update_with_optimizers(model_params),
            model_params,
            optimizer_parameter_updates,
        )

        updated_model_params = optax.apply_updates(
            jax.tree_util.tree_map(lambda op: op.value, model_params), parameter_updates
        )
        apply_state_updates = jax.tree_util.tree_map(
            should_apply_state_updates,
            self._update_types(state_updates),
        )
        for path, should_apply in flatten_items(apply_state_updates):
            if not should_apply:
                self.vlog(1, "Skipping state update on %s", path)
        filtered_state_updates = jax.tree_util.tree_map(
            lambda should_apply, update: update if should_apply else {},
            apply_state_updates,
            state_updates,
        )
        _apply_updates(updated_model_params, filtered_state_updates)
        if cfg.ema.decay is not None:
            _, ema_state = self.ema.update(
                updates={},
                state=self.state["ema"],
                params=jax.tree_util.tree_map(
                    lambda opt_param, value: dataclasses.replace(opt_param, value=value),
                    model_params,
                    updated_model_params,
                ),
            )
            self.add_state_update("ema", ema_state)
        return updated_model_params


def _apply_updates(base: NestedTensor, updates: NestedTensor) -> NestedTensor:
    """Applies updates from `updates` to `base` in-place, keeping `updates` unchanged.
    Note that keys omitted from `updates` will be untouched in `base`.

    Args:
        base: the state to be updated in-place.
        updates: the updates to apply to `base`.

    Returns:
        The updated state.
    """
    if isinstance(updates, Tensor):
        assert isinstance(base, Tensor), base
        return updates
    for k, v in updates.items():
        if k not in base:
            base[k] = v
        else:
            base[k] = _apply_updates(base[k], v)
    return base

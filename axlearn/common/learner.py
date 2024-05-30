# Copyright © 2023 Apple Inc.

"""A Learner is responsible for computing and applying updates to model params, including:

- Computing and applying updates from gradients through optimizer modules;
- Applying updates on non-differentiable params such as batch norm stats;
- Maintaining Polyak averages of model params (if enabled).
"""

import dataclasses
import enum
from typing import Callable, Mapping, Optional, Protocol, Sequence, Tuple

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
from axlearn.common.module import Module, OutputCollection, child_context, new_output_collection
from axlearn.common.optimizer_base import (
    NestedOptParam,
    OptParam,
    PartitionedGradientTransformation,
)
from axlearn.common.optimizers import param_ema
from axlearn.common.utils import (
    Nested,
    NestedPartitionSpec,
    NestedTensor,
    Tensor,
    flatten_items,
    match_regex_rules,
    prune_tree,
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


def _prune_empty(in_tree: NestedTensor) -> NestedTensor:
    """Returns a shallow copy of the input tree with empty subtrees pruned.

    If a tree would be made empty by removal of its subtrees, it will also be pruned.
    This is a shallow copy because leaf nodes (non-dict values) are not deep-copied.

    Args:
        in_tree: the input tree to be pruned.

    Returns:
        The pruned copy of the input tree.
    """
    # Note that falsey values or empty Tensors are not considered empty.
    return prune_tree(in_tree, lambda _, v: isinstance(v, dict) and not v)


@dataclasses.dataclass
class ForwardOutputs:
    loss: Tensor
    aux: NestedTensor
    output_collection: OutputCollection


@dataclasses.dataclass
class BackwardOutputs:
    updated_params: NestedTensor


@dataclasses.dataclass
class ForwardBackwardOutputs:
    forward_outputs: ForwardOutputs
    backward_outputs: BackwardOutputs


class ForwardFn(Protocol):
    """Represents the model forward function."""

    def __call__(self, *, inputs: NestedTensor, model_params: NestedTensor) -> ForwardOutputs:
        """The forward function of a module.

        Args:
            inputs: The inputs for the forward function.
            model_params: The model params.

        Returns:
            A ForwardOutputs value.
        """
        raise NotImplementedError(self)


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

    def forward_and_backward(
        self, *, fn: ForwardFn, inputs: NestedTensor, opt_params: NestedOptParam
    ) -> ForwardBackwardOutputs:
        """Computes updates to the parameters `opt_params` of loss function `fn`.

        Args:
            fn: The loss function to optimize.
            inputs: Inputs to the loss function that are not optimized. E.g., the input batch.
            opt_params: The model parameters that will be optimized.

        Returns:
            Forward and backward outputs of `fn`.
        """
        raise NotImplementedError(type(self))

    def should_update_with_optimizers(self, model_params: NestedOptParam) -> dict:
        """Returns whether each parameter should be updated with the optimizers.

        This can be used to skip gradient computation in the backward pass.

        Summaries, state updates, and other changes to OutputCollection may be ignored.

        Args:
            model_params: A nested structure with OptParams as leaf nodes.

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

    def _get_optimizer_model_params(self, model_params: NestedOptParam) -> NestedOptParam:
        should_update_params = self.should_update_with_optimizers(model_params)
        return jax.tree_util.tree_map(
            lambda should_update, param: param if should_update else None,
            should_update_params,
            model_params,
        )

    def init(self, model_params: NestedOptParam) -> NestedTensor:
        update_types = self._update_types(model_params)
        register_per_param_settings(
            update_types, description="learner_update_type", path=self.path()
        )
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

    def should_update_with_optimizers(self, model_params: NestedOptParam) -> dict:
        """Returns whether each parameter should be updated with the optimizers.

        Args:
            model_params: A nested structure with OptParams as leaf nodes.

        Returns:
            A nested dict with the same structure as `model_params` with boolean leaf values.
        """
        return jax.tree_util.tree_map(
            should_update_with_optimizers, self._update_types(model_params)
        )

    def update(
        self,
        *,
        model_params: Nested[OptParam],
        gradients: Nested[Tensor],
        state_updates: Nested[Tensor],
    ) -> Nested[Tensor]:
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
        optimizer_model_params = self._get_optimizer_model_params(model_params)
        optimizer_parameter_updates, optimizer_state = self.optimizer.update(
            gradients,
            state=self.state["optimizer"],
            params=optimizer_model_params,
        )
        self.add_state_update("optimizer", optimizer_state)
        return self._compute_updated_params(
            model_params,
            gradients=gradients,
            optimizer_parameter_updates=optimizer_parameter_updates,
            state_updates=state_updates,
        )

    def _compute_updated_params(
        self,
        model_params: Nested[OptParam],
        *,
        gradients: Nested[Tensor],
        optimizer_parameter_updates: Nested[Tensor],
        state_updates: Nested[Tensor],
    ) -> NestedTensor:
        cfg = self.config
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
        state_updates = _prune_empty(state_updates)
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

    def forward_and_backward(
        self, *, fn: ForwardFn, inputs: NestedTensor, opt_params: NestedOptParam
    ) -> ForwardBackwardOutputs:
        should_compute_gradients = self.should_update_with_optimizers(opt_params)
        model_params = jax.tree_util.tree_map(lambda opt_param: opt_param.value, opt_params)
        forward_outputs, gradients = _value_and_grad(
            fn,
            model_params=model_params,
            inputs=inputs,
            should_compute_gradients=should_compute_gradients,
        )
        updated_params = self.update(
            model_params=opt_params,
            gradients=gradients,
            state_updates=forward_outputs.output_collection.state_updates,
        )
        return ForwardBackwardOutputs(
            forward_outputs=forward_outputs,
            backward_outputs=BackwardOutputs(updated_params=updated_params),
        )


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


def _mask_tree(tree: dict, *, keep: dict) -> dict:
    """Mask out tree leaves that are not transformed by the optimizer.

    Args:
        tree: A nested structure with ParameterSpec, OptParams or Tensor as leaf nodes.
        keep: A tree of the same structure as tree, with boolean as leaf nodes. If
                the leaf is True, the original value of tree leaf is kept, otherwise replaced
                with optax.MaskNode().

    Returns:
        A masked tree the same structure as tree, the leaf is masked as MaskNode() if
         the corresponding keep leaf is False.

    """
    # For sub-learner optimizer state, only the subset of parameters
    # that belongs to the optimizer is kept and the rest is masked as optax.MaskNode().
    return jax.tree_util.tree_map(
        lambda should_keep, leaf: leaf if should_keep else optax.MaskedNode(),
        keep,
        tree,
    )


class CompositeLearner(BaseLearner):
    """The composite learner supports different sub learners on different subset of parameters.

    Note that the ema is handled by the master learner instead of sublearners.

    Note also that this class delegates to the `update()` methods of the sub-learners and does not
    call their `forward_and_backward()` methods.
    """

    @config_class
    class Config(BaseLearner.Config):
        """Configures CompositeLearner."""

        # Mapping of the sublearner name to the sublearner config. Sublearner name
        # must be a valid module name, that it matches with regex "^[a-z][a-z0-9_]*$".
        # See module.py for more details.
        learners: Mapping[str, Learner.Config] = {}

        # A sequence of (param_path_regex, name) to specify which learner is applied
        # on each parameter. Given a `param_path_regex`, the first
        # `re.fullmatch(param_path_regex, param_path)` determines the learner.
        # It will raise an error if none of the rules matches the param path.
        rules: Required[Sequence[Tuple[str, str]]] = REQUIRED

        # Ema config. All parameters should share the same ema config.
        # See Learner ema for more details.
        ema: InstantiableConfig = config_for_function(param_ema)

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        for name, learner_cfg in cfg.learners.items():
            # Sub learner should not hold ema.
            if learner_cfg.ema.decay is not None:
                raise ValueError(f"Sublearner {name} ema decay is not None.")
            if name == "ema":
                raise ValueError("Sublearner name cannot be ema.")

            sub_learner = learner_cfg.set(name=name)
            self._add_child(name, sub_learner)

        # Check that learners in the rules exist.
        for _, rule_name in cfg.rules:
            if rule_name not in cfg.learners:
                raise ValueError(f"{rule_name} is not found in the known learners.")
        if cfg.ema.decay is not None:
            # Create a global model ema.
            self.ema: PartitionedGradientTransformation = cfg.ema.instantiate()

    def _learner_tree(self, params: NestedOptParam) -> dict:
        """Returns a tree of the same structure as params where each leaf is the name of the
        sublearner to apply."""
        cfg = self.config
        learner_name_tree = jax.tree_util.tree_map(
            lambda path: match_regex_rules(
                path,
                rules=cfg.rules,
                default_value="",
            ),
            tree_paths(params),
        )
        # Check that all params is covered.
        if not jax.tree_util.tree_reduce(
            lambda x, y: x and (y != ""), learner_name_tree, initializer=True
        ):
            raise ValueError("Composite learner rules do not update all model params.")
        return learner_name_tree

    def create_state_partition_specs(
        self, model_param_specs: NestedParameterSpec
    ) -> NestedPartitionSpec:
        cfg = self.config
        learner_tree = self._learner_tree(params=model_param_specs)
        learner_state = {}
        for name in cfg.learners.keys():
            # Whether each parameter should apply the sub learner.
            should_apply = jax.tree_util.tree_map(
                lambda learner_name, n=name: learner_name == n,
                learner_tree,
            )
            # Mask model specs.
            sub_learner_specs = _mask_tree(tree=model_param_specs, keep=should_apply)
            # Call sub learner partition.
            sub_learner_partition = getattr(self, name).create_state_partition_specs(
                model_param_specs=sub_learner_specs
            )
            # Sublearner's partition.
            learner_state[name] = sub_learner_partition
        if self.config.ema.decay is not None:
            assert "ema" not in learner_state
            learner_state["ema"] = self.ema.partition(model_param_specs)
        return learner_state

    def init(self, model_params: NestedOptParam) -> NestedTensor:
        cfg = self.config
        learner_tree = self._learner_tree(params=model_params)
        register_per_param_settings(learner_tree, description="learner_rule", path=self.path())
        learner_state = {}
        for name in cfg.learners.keys():
            # Whether each parameter should apply the sub learner.
            should_apply = jax.tree_util.tree_map(
                lambda learner_name, n=name: learner_name == n,
                learner_tree,
            )
            # Mask model params.
            sub_learner_model_params = _mask_tree(tree=model_params, keep=should_apply)
            # Call sub learner initialization.
            sub_learner_state = getattr(self, name).init(model_params=sub_learner_model_params)
            # Sub-learner's state.
            learner_state[name] = sub_learner_state
        if self.config.ema.decay is not None:
            learner_state["ema"] = self.ema.init(model_params)
        return learner_state

    def update(
        self, *, model_params: NestedOptParam, gradients: NestedTensor, state_updates: NestedTensor
    ) -> NestedTensor:
        """Computes `model_params` updates with `gradients` and `state_updates`.

        Sub learners apply gradients update and state_updates to model_params.
        Ema is handled in the global learner.

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
        learner_tree = self._learner_tree(params=model_params)
        state_updates = _prune_empty(state_updates)
        state_updates_tree = self._learner_tree(params=state_updates)

        updated_model_params = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p.value), model_params
        )

        for name in cfg.learners.keys():
            # Whether each parameter should apply the sub learner.
            should_apply = jax.tree_util.tree_map(
                lambda learner_name, n=name: learner_name == n,
                learner_tree,
            )
            # Mask model params.
            sub_learner_model_params = _mask_tree(tree=model_params, keep=should_apply)
            # Mask gradients.
            sub_learner_gradients = _mask_tree(tree=gradients, keep=should_apply)
            # Split state_updates.
            # Set updates that do not belong to current sublearner as {}.
            sub_learner_state_updates = jax.tree_util.tree_map(
                lambda learner_name, s, n=name: s if learner_name == n else {},
                state_updates_tree,
                state_updates,
            )

            sub_learner_updated_model_params = getattr(self, name).update(
                model_params=sub_learner_model_params,
                gradients=sub_learner_gradients,
                state_updates=sub_learner_state_updates,
            )
            updated_model_params = jax.tree_util.tree_map(
                lambda apply, new_v, old_v: new_v if apply else old_v,
                should_apply,
                sub_learner_updated_model_params,
                updated_model_params,
            )
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

    def forward_and_backward(
        self, *, fn: ForwardFn, inputs: NestedTensor, opt_params: NestedOptParam
    ) -> ForwardBackwardOutputs:
        with child_context(
            "should_compute_gradients", module=self, output_collection=new_output_collection()
        ):
            should_compute_gradients = self.should_update_with_optimizers(opt_params)
        model_params = jax.tree_util.tree_map(lambda opt_param: opt_param.value, opt_params)
        forward_outputs, gradients = _value_and_grad(
            fn,
            model_params=model_params,
            inputs=inputs,
            should_compute_gradients=should_compute_gradients,
        )
        updated_params = self.update(
            model_params=opt_params,
            gradients=gradients,
            state_updates=forward_outputs.output_collection.state_updates,
        )
        return ForwardBackwardOutputs(
            forward_outputs=forward_outputs,
            backward_outputs=BackwardOutputs(updated_params=updated_params),
        )

    def should_update_with_optimizers(self, model_params: NestedOptParam) -> dict:
        """Returns whether each parameter should be updated with the optimizers.

        Args:
            model_params: A nested structure with OptParams as leaf nodes.

        Returns:
            A nested dict with the same structure as `model_params` with boolean leaf values.
        """
        cfg = self.config
        learner_tree = self._learner_tree(params=model_params)
        should_update = jax.tree_util.tree_map(lambda p: False, model_params)
        for name in cfg.learners.keys():
            # Whether each parameter should apply the sub learner.
            should_apply = jax.tree_util.tree_map(
                lambda learner_name, n=name: learner_name == n,
                learner_tree,
            )
            sub_learner_should_update_with_optimizers = getattr(
                self, name
            ).should_update_with_optimizers(model_params=model_params)
            should_update = jax.tree_util.tree_map(
                lambda apply, new_update, old_update: new_update if apply else old_update,
                should_apply,
                sub_learner_should_update_with_optimizers,
                should_update,
            )
        return should_update


def _split_gradients(
    fun: ForwardFn, *, should_compute_gradients: Nested[bool]
) -> Tuple[Callable, Callable]:
    """Return a function that is the same as `fun` but the first argument is split into two
    arguments `(grad,no_grad)` according to whether `(should compute_gradients)` is True.

    Args:
        fun: The function to transform.
        should_compute_gradients: The parameters to compute gradients for. Has the same
                                  tree structure as the first argument of `fun`.

    Returns:
        A two-tuple of:
        * The transformed version of `fun`.
        * A function to split the parameters into `(grad, no_grad)`.
    """

    def filtered_forward(
        model_parameters_grad: Nested[Tensor],
        model_parameters_no_grad: Nested[Tensor],
        /,
        inputs: Nested[Tensor],
    ) -> ForwardOutputs:
        model_params = jax.tree_util.tree_map(
            lambda compute_grad, pg, png: pg if compute_grad else png,
            should_compute_gradients,
            model_parameters_grad,
            model_parameters_no_grad,
        )
        return fun(model_params=model_params, inputs=inputs)

    def split_params_fn(model_params: Nested[Tensor]):
        dummy_value = None
        model_parameters_grad = jax.tree_util.tree_map(
            lambda compute_gradients, v: v if compute_gradients else dummy_value,
            should_compute_gradients,
            model_params,
        )
        model_parameters_no_grad = jax.tree_util.tree_map(
            lambda compute_gradients, v: dummy_value if compute_gradients else v,
            should_compute_gradients,
            model_params,
        )
        return model_parameters_grad, model_parameters_no_grad

    return filtered_forward, split_params_fn


def _value_and_grad(
    fun: ForwardFn,
    *,
    model_params: Nested[Tensor],
    inputs: Nested[Tensor],
    should_compute_gradients: Nested[bool],
) -> Tuple[ForwardOutputs, Nested[Tensor]]:
    """Computes the value and grad of `fun`.

    Args:
        fun: The function to compute gradients for.
        model_params: The model parameters.
        inputs: The inputs to `fun`.
        should_compute_gradients: The model parameters to compute gradients for.
                                  Has the same tree structure as `model_params`..

    Returns:
        The outputs of `fun` and the gradients.
    """

    fun, split_params_fn = _split_gradients(fun, should_compute_gradients=should_compute_gradients)

    def forward(*args):
        outputs = fun(*args)
        return outputs.loss, (outputs.aux, outputs.output_collection)

    split_params = split_params_fn(model_params)
    result, grads = jax.value_and_grad(forward, has_aux=True)(*split_params, inputs)
    loss, (aux, output_collection) = result
    return ForwardOutputs(loss=loss, aux=aux, output_collection=output_collection), grads

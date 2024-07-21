# Copyright © 2023 Apple Inc.

"""A Learner is responsible for computing and applying updates to model params, including:

- Computing and applying updates from gradients through optimizer modules;
- Applying updates on non-differentiable params such as batch norm stats;
- Maintaining Polyak averages of model params (if enabled).
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, cast

import jax
import optax
from jax import numpy as jnp

from axlearn.common.base_layer import ParameterSpec
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
)
from axlearn.common.learner_base import LearnerModule
from axlearn.common.module import Module, child_context, new_output_collection
from axlearn.common.optimizer_base import OptParam, PartitionedGradientTransformation
from axlearn.common.optimizers import param_ema
from axlearn.common.update_transformation import (
    BackwardOutputs,
    ForwardBackwardOutputs,
    ForwardFn,
    ForwardOutputs,
    ForwardPass,
    Updates,
    UpdateTransformation,
    WrappedPartitionedGradientTransformation,
    mask_tree,
)
from axlearn.common.utils import (
    Nested,
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


def _prune_empty(in_tree: Nested[Tensor]) -> Nested[Tensor]:
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


class BaseLearner(LearnerModule):
    """The base class of a learner."""

    def update(self, *, updates: Updates) -> Nested[Tensor]:
        """Computes `model_params` updates from `update`.

        Args:
            updates: The updates to potentially transform and then apply.

        Returns:
            The updated model parameters. The learner state updates will be placed in the output
            collection's 'state_update' section.
        """
        raise NotImplementedError(type(self))

    def forward_and_backward(
        self, *, fn: ForwardFn, inputs: Nested[Tensor], opt_params: Nested[OptParam]
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

    def should_update_with_optimizers(self, model_params: Nested[OptParam]) -> Nested[bool]:
        """Returns whether each parameter should be updated with the optimizers (delta updates).

        This can be used to skip gradient computation in the backward pass.

        Does not affect whether inplace updates happen.

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
        optimizer = cfg.optimizer
        if not isinstance(optimizer, LearnerModule.Config):
            optimizer = WrappedPartitionedGradientTransformation.default_config().set(
                transformation=optimizer
            )
        # Using self.optimizer: UpdateTransformation doesn't seem to work here.
        self.optimizer = cast(UpdateTransformation, self._add_child("optimizer", optimizer))
        if cfg.ema.decay is not None:
            self.ema: PartitionedGradientTransformation = cfg.ema.instantiate()

    def create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec]) -> Any:
        optimizer_model_param_specs = self._get_optimizer_model_params(model_param_specs)
        partition_state = dict(
            optimizer=self.optimizer.create_state_partition_specs(optimizer_model_param_specs)
        )
        if self.config.ema.decay is not None:
            partition_state["ema"] = self.ema.partition(model_param_specs)
        return partition_state

    def _get_optimizer_model_params(self, model_params: Nested[OptParam]) -> Nested[OptParam]:
        should_update_params = self.should_update_with_optimizers(model_params)
        # Use mask_value=None for consistency with old behavior. This may change in the future.
        return mask_tree(model_params, keep=should_update_params, mask_value=None)

    def init(self, model_params: Nested[OptParam]) -> Nested[Tensor]:
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

    def should_update_with_optimizers(self, model_params: Nested[OptParam]) -> dict:
        """Returns whether each parameter should be updated with the optimizers.

        Args:
            model_params: A nested structure with OptParams as leaf nodes.

        Returns:
            A nested dict with the same structure as `model_params` with boolean leaf values.
        """
        return jax.tree_util.tree_map(
            should_update_with_optimizers, self._update_types(model_params)
        )

    def update(self, updates: Updates) -> Nested[Tensor]:
        """Computes `model_params` updates from `update`.

        Args:
            updates: The updates to potentially transform and then apply.

        Returns:
            The updated model parameters. The learner state updates will be placed in the output
            collection's 'state_update' section.
        """
        new_updates = dataclasses.replace(
            updates, opt_params=self._get_optimizer_model_params(updates.opt_params)
        )
        new_updates = self.optimizer(new_updates)
        return self._compute_updated_params(
            opt_params=updates.opt_params,  # Use updates from prior to masking
            gradients=updates.delta_updates,  # We want the original non-transformed gradients here.
            optimizer_parameter_updates=new_updates.delta_updates,
            state_updates=new_updates.inplace_updates,
        )

    def _compute_updated_params(
        self,
        opt_params: Nested[OptParam],
        *,
        gradients: Nested[Tensor],
        optimizer_parameter_updates: Nested[Tensor],
        state_updates: Nested[Tensor],
    ) -> Nested[Tensor]:
        cfg = self.config
        if cfg.enable_per_variable_summaries:
            param_rms = jax.tree_util.tree_map(
                lambda p: optax.safe_root_mean_squares(p.value, min_rms=1e-3), opt_params
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
            self.should_update_with_optimizers(opt_params),
            opt_params,
            optimizer_parameter_updates,
        )

        updated_model_params = optax.apply_updates(
            jax.tree_util.tree_map(lambda op: op.value, opt_params), parameter_updates
        )
        state_updates = _prune_empty(state_updates)
        apply_state_updates = jax.tree_util.tree_map(
            should_apply_state_updates,
            self._update_types(state_updates),
        )
        for path, should_apply in flatten_items(apply_state_updates):
            if not should_apply:
                self.vlog(1, "Skipping state update on %s", path)
        filtered_state_updates = mask_tree(
            state_updates, keep=apply_state_updates, mask_value=optax.MaskedNode()
        )
        _apply_updates(updated_model_params, filtered_state_updates)
        if cfg.ema.decay is not None:
            _, ema_state = self.ema.update(
                updates={},
                state=self.state["ema"],
                params=jax.tree_util.tree_map(
                    lambda opt_param, value: dataclasses.replace(opt_param, value=value),
                    opt_params,
                    updated_model_params,
                ),
            )
            self.add_state_update("ema", ema_state)
        return updated_model_params

    def forward_and_backward(
        self, *, fn: ForwardFn, inputs: Nested[Tensor], opt_params: Nested[OptParam]
    ) -> ForwardBackwardOutputs:
        should_compute_gradients = self.should_update_with_optimizers(opt_params)
        updates = _value_and_grad(
            fn,
            opt_params=opt_params,
            inputs=inputs,
            should_compute_gradients=should_compute_gradients,
        )
        forward_outputs = updates.forward_pass.get("default").outputs  # type: ignore
        updated_params = self.update(updates)
        return ForwardBackwardOutputs(
            forward_outputs=forward_outputs,
            backward_outputs=BackwardOutputs(updated_params=updated_params),
        )


def _apply_updates(base: Nested[Tensor], updates: Nested[Tensor]) -> Nested[Tensor]:
    """Applies updates from `updates` to `base` in-place, keeping `updates` unchanged.
    Note that keys omitted from `updates` will be untouched in `base`.

    If either `base` or `updates is an `optax.MaskedNode()`, the update is not applied.

    Args:
        base: the state to be updated in-place.
        updates: the updates to apply to `base`.

    Returns:
        The updated state.
    """
    if base == optax.MaskedNode() or updates == optax.MaskedNode():
        return base
    if isinstance(updates, Tensor):
        assert isinstance(base, Tensor), base
        return updates
    for k, v in updates.items():
        if k not in base:
            base[k] = v
        else:
            base[k] = _apply_updates(base[k], v)
    return base


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

    def _learner_tree(self, params: Nested[Any]) -> Nested[str]:
        """Returns a tree of the same structure as params where each leaf is the name of the
        sublearner to apply.
        """
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

    def create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec]) -> Any:
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
            sub_learner_specs = mask_tree(
                tree=model_param_specs, keep=should_apply, mask_value=optax.MaskedNode()
            )
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

    def init(self, model_params: Nested[OptParam]) -> Nested[Tensor]:
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
            sub_learner_model_params = mask_tree(
                tree=model_params, keep=should_apply, mask_value=optax.MaskedNode()
            )
            # Call sub learner initialization.
            sub_learner_state = getattr(self, name).init(model_params=sub_learner_model_params)
            # Sub-learner's state.
            learner_state[name] = sub_learner_state
        if self.config.ema.decay is not None:
            learner_state["ema"] = self.ema.init(model_params)
        return learner_state

    def update(self, updates: Updates) -> Nested[Tensor]:
        """Computes `model_params` updates from `update`.

        Args:
            updates: The updates to potentially transform and then apply.

        Returns:
            The updated model parameters. The learner state updates will be placed in the output
            collection's 'state_update' section.
        """
        cfg = self.config

        updated_model_params = jax.tree_util.tree_map(jnp.zeros_like, updates.param_values())

        for name in cfg.learners.keys():
            # Whether each parameter/state should apply the sub learner.
            def should_apply(tree: Nested[Any]) -> Nested[bool]:
                return jax.tree_util.tree_map(
                    # pylint: disable-next=cell-var-from-loop
                    lambda learner_name, n=name: learner_name == n,
                    self._learner_tree(tree),
                )

            # See the docstring of `learner_test.CompositeLearnerTest.test_learner_masking`
            # for a more detailed explanation of the masking behavior we are mimicking here
            # for backwards compatibility.
            sub_learner_updates = updates.mask(should_apply, fields=("inplace_updates",))
            sub_learner_updates = sub_learner_updates.mask(
                # pylint: disable-next=cell-var-from-loop
                lambda _: should_apply(updates.opt_params),
                fields=(
                    "opt_params",
                    "delta_updates",
                ),
            )
            sub_learner_updated_model_params = getattr(self, name).update(sub_learner_updates)
            updated_model_params = jax.tree_util.tree_map(
                lambda apply, new_v, old_v: new_v if apply else old_v,
                should_apply(updates.param_values()),
                sub_learner_updated_model_params,
                updated_model_params,
            )
        if cfg.ema.decay is not None:
            _, ema_state = self.ema.update(
                updates={},
                state=self.state["ema"],
                params=jax.tree_util.tree_map(
                    lambda opt_param, value: dataclasses.replace(opt_param, value=value),
                    updates.opt_params,
                    updated_model_params,
                ),
            )
            self.add_state_update("ema", ema_state)
        return updated_model_params

    def forward_and_backward(
        self, *, fn: ForwardFn, inputs: Nested[Tensor], opt_params: Nested[OptParam]
    ) -> ForwardBackwardOutputs:
        with child_context(
            "should_compute_gradients", module=self, output_collection=new_output_collection()
        ):
            should_compute_gradients = self.should_update_with_optimizers(opt_params)
        updates = _value_and_grad(
            fn,
            opt_params=opt_params,
            inputs=inputs,
            should_compute_gradients=should_compute_gradients,
        )
        forward_outputs = updates.forward_pass.get("default").outputs  # type: ignore
        updated_params = self.update(updates)
        return ForwardBackwardOutputs(
            forward_outputs=forward_outputs,
            backward_outputs=BackwardOutputs(updated_params=updated_params),
        )

    def should_update_with_optimizers(self, model_params: Nested[OptParam]) -> dict:
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
) -> Tuple[ForwardFn, Callable]:
    """Return a function that is the same as `fun` but where the call signature is now

     `fun(model_params=model_params_grad, inputs=(model_paramgs_nograd, inputs))`

     instead of `fun(model_params=model_params, inputs=inputs)`.

     The split of `model_params` into `model_params_grad, model_params_nograd` is
     according to whether `(should compute_gradients)` is True.

     Values that are ommitted are replaced with `None` for compatibility with old behavior.
     This may change in the future

    Args:
        fun: The function to transform.
        should_compute_gradients: The parameters to compute gradients for. Has the same
                                  tree structure as the first argument of `fun`.

    Returns:
        A two-tuple of:
        * The transformed version of `fun`, which is still a `ForwardFn`.
        * A function to split the parameters into `(grad, no_grad)`.
    """

    def filtered_forward(model_params: Nested[Tensor], *, inputs: Any) -> ForwardOutputs:
        model_params_grad = model_params
        model_params_nograd, inputs = inputs
        model_params = jax.tree_util.tree_map(
            lambda compute_grad, pg, png: pg if compute_grad else png,
            should_compute_gradients,
            model_params_grad,
            model_params_nograd,
        )
        return fun(model_params=model_params, inputs=inputs)

    def split_params_fn(model_params: Nested) -> Tuple[Nested, Nested]:
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


def _as_loss_fn(fun: ForwardFn) -> Callable:
    """Convert a `ForwardFn` to a function with the same signature execpt that it outputs
    `loss, forward_pass`.

    This wrapping makes it compatible with `jax.grad()`.

    Args:
        fun: The function to wrap.

    Returns:
        The wrapped function.
    """

    def forward(model_params: Nested[Tensor], *, inputs: Any) -> Tuple[Tensor, ForwardPass]:
        outputs = fun(model_params=model_params, inputs=inputs)  # type: ignore
        return outputs.loss, ForwardPass(
            # We don't use `forward` here since it is not technically a `ForwardFn`.
            forward_fn=fun,
            model_params=model_params,
            inputs=inputs,
            outputs=outputs,
        )

    return forward


def _value_and_grad(
    fun: ForwardFn,
    *,
    opt_params: Nested[OptParam],
    inputs: Nested[Tensor],
    should_compute_gradients: Optional[Nested[bool]] = None,
) -> Updates:
    """Computes the value and grad of `fun`.

    Args:
        fun: The function to compute gradients for.
        opt_params: The model parameters.
        inputs: The inputs to `fun`.
        should_compute_gradients: The model parameters to compute gradients for.
                                  Has the same tree structure as `model_params`.
                                  If None, all parameters have their gradients computed.

    Returns:
        The gradient `Updates` for `fun`. The returned `updates`  include a "default" key for
        `updates.forward_pass`. State updates are taken from the outputs of `fun`.
    """
    split_params_fn = lambda params: [params, {}]
    if should_compute_gradients is not None:
        fun, split_params_fn = _split_gradients(
            fun, should_compute_gradients=should_compute_gradients
        )

    loss_fun = _as_loss_fn(fun)

    split_params = split_params_fn(opt_params)
    model_params_grad, model_params_nograd = jax.tree_util.tree_map(lambda p: p.value, split_params)
    (_, forward_pass), grads = jax.value_and_grad(loss_fun, has_aux=True)(
        model_params_grad, inputs=(model_params_nograd, inputs)
    )
    return Updates(
        opt_params=opt_params,
        delta_updates=grads,
        inplace_updates=forward_pass.outputs.output_collection.state_updates,
        forward_pass=dict(default=forward_pass),
    )

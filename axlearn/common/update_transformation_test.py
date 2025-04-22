# Copyright © 2024 Apple Inc.
"""Tests for update_transformation.py."""
import dataclasses
from collections.abc import Sequence
from typing import Any, NamedTuple

import chex
import jax
import optax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

import axlearn.common
from axlearn.common import optimizers, schedule, test_utils
from axlearn.common.base_layer import FactorizationSpec, ParameterSpec
from axlearn.common.config import config_for_function, maybe_instantiate
from axlearn.common.learner import Learner
from axlearn.common.module import InvocationContext
from axlearn.common.module import functional as F
from axlearn.common.module import new_output_collection, set_current_context
from axlearn.common.optimizer_base import (
    NestedOptParam,
    OptParam,
    PartitionedGradientTransformation,
)
from axlearn.common.update_transformation import (
    ConditionalUpdateTransformation,
    ForwardOutputs,
    OverrideInplaceUpdateTransformation,
    Updates,
    UpdateTransformation,
    WrappedPartitionedGradientTransformation,
)
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    NestedTree,
    PartitionSpec,
    Tensor,
    VDict,
    tree_paths,
)


class UpdateTransformationTest(test_utils.TestCase):
    """Tests related to the `UpdateTransformation` API."""

    @parameterized.parameters(
        dict(
            optimizer=optimizers.sgd_optimizer(
                learning_rate=0.5,
                decouple_weight_decay=True,
            ),
            use_config=True,
        ),
        dict(
            optimizer=optimizers.sgd_optimizer(
                learning_rate=0.5,
                decouple_weight_decay=True,
            ),
            use_config=False,
        ),
        dict(
            optimizer=optimizers.adamw_optimizer(
                learning_rate=0.9,
                b1=0.9,
                b2=0.99,
                eps=1e-8,
                weight_decay=0.4,
            )
        ),
        dict(
            optimizer=optimizers.adamw_decoupled_optimizer(
                learning_rate=0.85,
                b1=0.95,
                b2=0.99,
                eps=1e-8,
                update_schedule=1.0,
                weight_decay=0.3,
            )
        ),
        dict(
            optimizer=optimizers.adastar_optimizer(
                learning_rate=0.1,
                gradient_ema_decay=0.8,
                gradient_ema_debias=False,
                gradient_square_ema_decay=0.75,
                gradient_square_ema_debias=True,
                eps=1e-9,
                eps_square=0,
                update_ema_decay=None,
                raw_update_clipping_threshold=0.01,
                update_ema_debias=False,
                update_schedule=1.0,
            )
        ),
        dict(
            optimizer=optimizers.chain(
                optimizers.sgd_optimizer(
                    learning_rate=0.5,
                    decouple_weight_decay=True,
                ),
                optimizers.sgd_optimizer(
                    learning_rate=0.5,
                    decouple_weight_decay=True,
                ),
            )
        ),
    )
    def test_wrapped_partitioned_gradient_transformation(
        self, *, use_config: bool = False, optimizer: PartitionedGradientTransformation
    ):
        if use_config:
            optimizer = config_for_function(lambda t=optimizer: t)
        cfg = WrappedPartitionedGradientTransformation.default_config().set(
            name="tmp", transformation=optimizer
        )
        update_transformation: UpdateTransformation = cfg.instantiate(parent=None)
        optimizer: PartitionedGradientTransformation = maybe_instantiate(optimizer)

        # Setup state and call args.
        updates = mock_updates()
        opt_params = updates.opt_params
        grads = updates.delta_updates
        param_specs = updates.param_specs()
        inplace_updates = updates.inplace_updates

        # For backwards compatibility, we make sure the state tree structure is the same.
        actual_init = update_transformation.init(opt_params)
        expected_init = optimizer.init(opt_params)
        self.assertNestedAllClose(actual_init, expected_init)

        actual_specs = update_transformation.create_state_partition_specs(param_specs)
        expected_specs = optimizer.partition(param_specs)
        self.assertNestedAllClose(actual_specs, expected_specs)

        # Check that the final update and optimizer state updates are the same.
        actual_update, actual_output_collection = F(
            module=update_transformation,
            prng_key=None,
            state=actual_init,
            inputs=[updates],
            method="__call__",
            is_training=True,
        )
        ctx = InvocationContext(
            name="root",
            parent=None,
            module=None,
            state={},
            is_training=True,
            prng_key=None,
            output_collection=new_output_collection(),
        )
        with set_current_context(ctx):
            (
                expected_param_update,
                expected_state_update,
            ) = optimizer.update(updates=grads, state=expected_init, params=opt_params)
        expected_update = Updates(
            opt_params=opt_params,
            delta_updates=expected_param_update,
            inplace_updates=inplace_updates,
        )
        self.assertNestedAllClose(actual_update, expected_update)
        self.assertNestedAllClose(actual_output_collection.state_updates, expected_state_update)

        # Check that the summaries are the same.
        self.assertNestedAllClose(actual_output_collection.summaries, ctx.get_summaries())


def mock_params() -> Nested[Tensor]:
    """Returns mock model params."""
    return dict(
        weight=jnp.ones((3, 5)),  # An ordinary param.
        vdict=VDict(bias=jnp.ones((7, 5))),  # Param in a custom pytreee node.
        state=jnp.ones(2, dtype=jnp.int32),  # Simulates a nograd param.
        more_state=jnp.ones(3, dtype=jnp.int32),  # A param with both grad and state update.
        do_not_update=optax.MaskedNode(),  # A param masked out by CompositeLearner.
    )


def mock_updates(state_param_none: bool = True) -> Updates:
    """Create an updates object with various semi-reasonable values."""
    model_params = mock_params()
    if state_param_none:
        model_params["state"] = None
    opt_params = jax.tree.map(
        lambda p: OptParam(
            value=p,
            factorization_spec=FactorizationSpec([None] * p.ndim),
            weight_decay_scale=0.1,
        ),
        model_params,
    )
    delta_updates = jax.tree.map(lambda p: p * -0.1, model_params)
    delta_updates["state"] = None
    inplace_updates = dict(
        state=jnp.arange(2, dtype=jnp.int32),
        more_state=jnp.arange(3, dtype=jnp.int32),
        do_not_update=optax.MaskedNode(),
    )
    updates = Updates(
        opt_params=opt_params, delta_updates=delta_updates, inplace_updates=inplace_updates
    )
    return updates


class UpdatesTest(test_utils.TestCase):
    """Tests related to the `Updates` interface."""

    def test_param_values(self):
        updates = mock_updates()
        actual = updates.param_values()
        expected = mock_params()
        expected["state"] = None
        chex.assert_trees_all_equal_structs(actual, expected)
        self.assertNestedAllClose(actual, expected)

    def test_param_specs(self):
        updates = mock_updates()
        actual = updates.param_specs()
        expected = dict(
            weight=ParameterSpec(
                shape=(3, 5),
                dtype=jnp.float32,
                factorization=FactorizationSpec([None, None]),
                weight_decay_scale=0.1,
            ),
            vdict=VDict(
                bias=ParameterSpec(
                    shape=(7, 5),
                    dtype=jnp.float32,
                    factorization=FactorizationSpec([None, None]),
                    weight_decay_scale=0.1,
                )
            ),
            state=None,
            more_state=ParameterSpec(
                shape=(3,),
                dtype=jnp.int32,
                factorization=FactorizationSpec([None]),
                weight_decay_scale=0.1,
            ),
            do_not_update=optax.MaskedNode(),
        )
        chex.assert_trees_all_equal_structs(actual, expected)
        self.assertEqual(actual, expected)

    @parameterized.product(
        fields=[[], ["opt_params"], ["delta_updates"], ["inplace_updates"], None],
        masked_names=[["weight", "more_state"], ["weight"]],
    )
    def test_mask(self, fields: Sequence[str], masked_names):
        updates = mock_updates()
        kwargs = dict(keep=lambda tree: {k: k in masked_names for k in tree})
        if fields is not None:
            kwargs.update(fields=fields)
        else:
            fields = ("opt_params", "delta_updates", "inplace_updates")
        actual = updates.mask(**kwargs)

        def expected_result(tree: Nested) -> Nested:
            return tree | {k: optax.MaskedNode() for k in tree if k not in masked_names}

        opt_params = expected_result(updates.opt_params)
        delta_updates = expected_result(updates.delta_updates)
        inplace_updates = expected_result(updates.inplace_updates)

        # Expected value of `masked` with default fields.
        expected_default = axlearn.common.update_transformation.Updates(
            opt_params=opt_params, delta_updates=delta_updates, inplace_updates=inplace_updates
        )
        expected = dataclasses.replace(
            updates, **{field: getattr(expected_default, field) for field in fields}
        )

        chex.assert_trees_all_equal_structs(actual, expected)
        self.assertNestedAllClose(actual, expected)


class OverrideInplaceUpdateTransformationTest(test_utils.TestCase):
    """Tests for `OverrideInplaceUpdateTransformation`."""

    def test_override_inplace_update_transformation(self):
        learning_rate = config_for_function(schedule.stepwise).set(
            sub=[0.1, 0.01, 0.001],
            start_step=[100, 200],
        )
        transformation = config_for_function(optimizers.adamw_optimizer).set(
            learning_rate=learning_rate, b1=0.9, b2=0.95, eps=1e-7
        )
        cfg = OverrideInplaceUpdateTransformation.default_config().set(
            name="tmp", transformation=transformation, rules=[".*weight"]
        )
        update_transformation: UpdateTransformation = cfg.instantiate(parent=None)

        updates = mock_updates()
        param_specs = updates.param_specs()
        actual_init = update_transformation.init(updates.opt_params)
        actual_specs = update_transformation.create_state_partition_specs(param_specs)

        # `weight` should be filtered from both init states and specs.
        jax.tree.map(lambda path: self.assertNotIn("weight", path), tree_paths(actual_init))
        jax.tree.map(lambda path: self.assertNotIn("weight", path), tree_paths(actual_specs))

        actual_update, _ = F(
            module=update_transformation,
            prng_key=None,
            state=actual_init,
            inputs=[updates],
            method="__call__",
            is_training=True,
        )

        # `weight` should be in both `inplace_updates` and `delta_updates`.
        out = jax.tree.map(lambda path: "weight" in path, tree_paths(actual_update.delta_updates))
        self.assertTrue(jax.tree.reduce(lambda a, b: a or b, out))
        out = jax.tree.map(lambda path: "weight" in path, tree_paths(actual_update.inplace_updates))
        self.assertTrue(jax.tree.reduce(lambda a, b: a or b, out))


class LearnerStep(NamedTuple):
    state: Any
    model_params: Any


class ConditionalUpdateTransformationTest(test_utils.TestCase):
    """Tests for `OverrideInplaceUpdateTransformation`."""

    @parameterized.parameters("adamw", "chained")
    def test_conditional_update_transformation(self, optimizer_type):
        def get_learner_from_su(should_update_schedule_fn=None):
            if optimizer_type in ("adamw", "chained"):
                optimizer_cfg = config_for_function(optimizers.adamw_optimizer).set(
                    learning_rate=0.1,
                    b1=0.9,
                    b2=0.99,
                    eps=1e-5,
                    weight_decay=0,
                )
            if optimizer_type == "chained":
                optimizer_cfg = config_for_function(optimizers.chain).set(
                    args=[
                        config_for_function(optimizers.clip_by_global_norm).set(max_norm=10),
                        optimizer_cfg,
                    ]
                )
            optimizer_cfg = ConditionalUpdateTransformation.default_config().set(
                inner=optimizer_cfg,
                update_schedule=should_update_schedule_fn,
            )
            cfg = Learner.default_config().set(
                name="test",
                optimizer=optimizer_cfg,
            )
            cfg.ema.decay = None  # ema is not supported if we use conditional update
            learner: Learner = cfg.instantiate(parent=None)
            return learner

        # learner updates at step 0 and step 2 (step is 0-based)
        learner = get_learner_from_su(should_update_schedule_fn=lambda step: step % 2 == 0)
        # learner2 is a regular learner
        learner2 = get_learner_from_su()
        v_spec = ParameterSpec(
            dtype=jnp.float32,
            shape=(4,),
            mesh_axes=PartitionSpec("model"),
            factorization=None,
            weight_decay_scale=1.0,
        )
        param_specs = dict(v_all=v_spec)
        model_params = dict(v_all=jnp.asarray([1, -2, 3, -4], dtype=jnp.float32))

        def opt_params_from_model_params(
            model_params: NestedTensor, param_specs: NestedTree
        ) -> NestedOptParam:
            """Returns a tree of OptParam for Learner.{init,update}."""
            return jax.tree.map(
                lambda param, spec: OptParam(
                    value=param,
                    factorization_spec=spec.factorization if spec is not None else None,
                    weight_decay_scale=spec.weight_decay_scale if spec is not None else 1.0,
                ),
                model_params,
                param_specs,
            )

        params = opt_params_from_model_params(model_params, param_specs=param_specs)
        params2 = opt_params_from_model_params(model_params, param_specs=param_specs)

        state = learner.init(model_params=params)
        state2 = learner2.init(model_params=params)

        def loss_fn(model_params, inputs):
            del inputs
            m = model_params["v_all"]
            loss = -1 * (jnp.arange(1, 5) * m).sum()
            output_collection = new_output_collection()
            return ForwardOutputs(
                loss=loss,
                aux={},
                output_collection=output_collection,
            )

        learner_steps, learner2_steps = [], []
        for step in range(3):
            assert state["optimizer"]["should_update"].count.item() == step
            fwd_bwd_outputs, output_collection = F(
                learner,
                method="forward_and_backward",
                state=state,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                inputs=dict(
                    fn=loss_fn,
                    opt_params=params,
                    inputs=dict(
                        input_batch={},
                    ),
                ),
            )
            fwd_bwd_outputs2, output_collection2 = F(
                learner2,
                method="forward_and_backward",
                state=state2,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                inputs=dict(
                    fn=loss_fn,
                    opt_params=params2,
                    inputs=dict(
                        input_batch={},
                    ),
                ),
            )
            state = output_collection.state_updates
            model_params = fwd_bwd_outputs.backward_outputs.updated_params
            params = opt_params_from_model_params(model_params, param_specs)
            learner_steps.append(LearnerStep(state=state, model_params=model_params))

            state2 = output_collection2.state_updates
            model_params2 = fwd_bwd_outputs2.backward_outputs.updated_params
            params2 = opt_params_from_model_params(model_params2, param_specs)
            learner2_steps.append(LearnerStep(state=state2, model_params=model_params2))

        def check_state_and_model_params_equal(step1, step2):
            self.assertNestedAllClose(
                step1.state["optimizer"]["inner"], step2.state["optimizer"]["inner"]
            )
            self.assertNestedAllClose(step1.model_params["v_all"], step2.model_params["v_all"])

        # Check no updates during "off" step
        check_state_and_model_params_equal(learner_steps[0], learner_steps[1])
        # Check optimizer state correctly accumulates across steps
        check_state_and_model_params_equal(learner_steps[2], learner2_steps[1])


if __name__ == "__main__":
    absltest.main()

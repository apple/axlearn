# Copyright © 2023 Apple Inc.
"""Tests learner."""
import copy
import re
from numbers import Number
from typing import Any, Optional, Union, cast

import chex
import jax.nn
import numpy as np
import optax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

import axlearn.common.update_transformation_test
from axlearn.common import schedule
from axlearn.common.base_layer import FactorizationSpec, ParameterSpec
from axlearn.common.config import REQUIRED, Required, config_class, config_for_function
from axlearn.common.gradient_accumulation import with_minibatch_steps
from axlearn.common.learner import (
    CompositeLearner,
    Learner,
    UpdateType,
    _apply_updates,
    _prune_empty,
    _split_gradients,
    _value_and_grad,
    should_update_with_optimizers,
)
from axlearn.common.metrics import MetricAccumulator, WeightedScalar
from axlearn.common.module import OutputCollection
from axlearn.common.module import functional as F
from axlearn.common.module import new_output_collection
from axlearn.common.optimizer_base import OptParam, OptStateSpec
from axlearn.common.optimizers import (
    AddDecayedWeightsState,
    ParamEmaState,
    adafactor_optimizer,
    adam_optimizer,
    chain,
    clip_by_global_norm,
    sgd_optimizer,
)
from axlearn.common.test_utils import TestCase
from axlearn.common.update_transformation import (
    ForwardOutputs,
    ForwardPass,
    Updates,
    UpdateTransformation,
)
from axlearn.common.utils import (
    Nested,
    PartitionSpec,
    Tensor,
    VDict,
    flatten_items,
    match_regex_rules,
    tree_paths,
)


class LearnerTest(TestCase):
    def test_prune_empty_state(self):
        state = {
            "state": {
                "tensor": jnp.array(0),
                "nested": {
                    "empty": {},
                    "not_empty": jnp.array([]),
                },
            },
            "removed": {
                "nested": {
                    "deep_nested": {},
                },
                "sibling": {
                    "deep_nested": {},
                },
            },
        }
        expected = {
            "state": {
                "tensor": jnp.array(0),
                "nested": {
                    "not_empty": jnp.array([]),
                },
            },
        }
        actual = _prune_empty(state)
        self.assertNestedAllClose(expected, actual)

    @parameterized.product(ema_decay=(None, 0.9), method=("update", "forward_and_backward"))
    def test_learner(self, ema_decay: Optional[float], method: str):
        learning_rate = config_for_function(schedule.stepwise).set(
            sub=[0.1, 0.01, 0.001],
            start_step=[100, 200],
        )
        learning_rate_fn = schedule.as_schedule_fn(learning_rate)
        weight_decay = 1e-4
        step = 0
        sgd_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=learning_rate,
            decouple_weight_decay=True,
            weight_decay=weight_decay,
        )
        optimizer_cfg = config_for_function(chain).set(
            args=(config_for_function(clip_by_global_norm), sgd_cfg),
        )
        cfg = Learner.default_config().set(name="test", optimizer=optimizer_cfg)
        cfg.ema.decay = ema_decay
        learner: Learner = cfg.instantiate(parent=None)

        param_specs = dict(
            v=ParameterSpec(
                dtype=jnp.float32,
                shape=(4,),
                mesh_axes=PartitionSpec("model"),
                factorization=None,
            ),
            c=ParameterSpec(
                dtype=jnp.float32,
                shape=[],
                mesh_axes=PartitionSpec(),
                factorization=None,
            ),
        )
        expected_state_spec = dict(
            optimizer=(
                # clip_by_global_norm.
                optax.EmptyState(),
                # sgd.
                (
                    optax.TraceState(
                        trace=dict(
                            v=OptStateSpec(
                                dtype=jnp.float32, shape=(4,), mesh_axes=PartitionSpec("model")
                            ),
                            c=OptStateSpec(dtype=jnp.float32, shape=[], mesh_axes=PartitionSpec()),
                        ),
                    ),
                    AddDecayedWeightsState(count=None),
                    optax.ScaleByScheduleState(
                        count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec())
                    ),
                ),
            ),
        )
        if ema_decay:
            expected_state_spec["ema"] = ParamEmaState(
                count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
                ema=dict(
                    v=OptStateSpec(dtype=jnp.float32, shape=(4,), mesh_axes=PartitionSpec("model")),
                    c=OptStateSpec(dtype=jnp.float32, shape=[], mesh_axes=PartitionSpec()),
                ),
            )
        self.assertEqual(
            expected_state_spec,
            learner.create_state_partition_specs(param_specs),
        )

        params = dict(
            v=OptParam(
                value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1.0,
            ),
            c=OptParam(
                value=jnp.zeros([], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=0,
            ),
        )
        state = learner.init(model_params=params)

        def loss_fn(model_params, inputs):
            del inputs
            output_collection = new_output_collection()
            output_collection.state_updates["c"] = model_params["c"] + 1
            return ForwardOutputs(
                loss=-jax.nn.log_softmax(model_params["v"])[1],
                aux={},
                output_collection=output_collection,
            )

        loss, grads = jax.value_and_grad(lambda x: loss_fn(x, None).loss)(
            jax.tree.map(lambda p: p.value, params)
        )
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        self.assertNestedAllClose(
            dict(v=jnp.asarray([0.089629, -0.756364, 0.662272, 0.004462]), c=0.0), grads, atol=1e-6
        )

        if method == "update":
            updates = Updates(
                delta_updates=grads,
                opt_params=params,
                inplace_updates=dict(c=params["c"].value + 1),
            )
            inputs = dict(updates=updates)
        elif method == "forward_and_backward":
            inputs = dict(fn=loss_fn, inputs={}, opt_params=params)
        else:
            raise NotImplementedError

        updated_params, output_collection = F(
            learner,
            method=method,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=inputs,
        )

        if method == "forward_and_backward":
            updated_params = updated_params.backward_outputs.updated_params

        v_value = params["v"].value
        expected_new_v = v_value - learning_rate_fn(step) * (grads["v"] + weight_decay * v_value)
        self.assertNestedAllClose(
            updated_params,
            dict(v=expected_new_v, c=1.0),
            atol=1e-6,
        )
        # Optimizer summaries are now logged under `/learner/optimizer/` instead of `/learner/`.
        summaries = output_collection.summaries["optimizer"]
        self.assertNestedAllClose(
            {
                "learning_rate": learning_rate_fn(step),
                "lr_schedule_step": 0,
                "gradient_norm": 1.0093285,
                "schedule_step": 0,
                "schedule_scale": -1.0 * learning_rate_fn(step),
            },
            summaries,
        )
        state_updates = output_collection.state_updates
        expected_state_update = {
            "optimizer": (
                # clip_by_global_norm.
                optax.EmptyState(),
                # sgd.
                (
                    optax.TraceState(trace=grads),
                    optax.EmptyState(),
                    optax.ScaleByScheduleState(count=jnp.ones([], dtype=jnp.int32)),
                ),
            ),
        }
        if ema_decay:
            expected_state_update["ema"] = ParamEmaState(
                count=1,
                ema=jax.tree.map(lambda v: v * (1 - ema_decay), updated_params),
            )
        self.assertNestedAllClose(
            expected_state_update,
            state_updates,
        )

    @parameterized.named_parameters(
        ("default", []),
        ("no_update_v", [(".*v", UpdateType.NO_UPDATE)]),
        ("optimizers_v", [(".*v", UpdateType.OPTIMIZERS)]),
        ("state_updates_v", [(".*v", UpdateType.STATE_UPDATES)]),
        ("all_updates_v", [(".*v", UpdateType.ALL_UPDATES)]),
        ("no_update_c", [(".*c", UpdateType.NO_UPDATE)]),
        ("optimizers_c", [(".*c", UpdateType.OPTIMIZERS)]),
        ("state_updates_c", [(".*c", UpdateType.STATE_UPDATES)]),
        ("all_updates_c", [(".*c", UpdateType.ALL_UPDATES)]),
    )
    def test_update_rules(self, update_rules):
        learning_rate = config_for_function(schedule.stepwise).set(
            sub=[0.1, 0.01, 0.001],
            start_step=[100, 200],
        )
        learning_rate_fn = schedule.as_schedule_fn(learning_rate)
        weight_decay = 1e-4
        step = 0
        sgd_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=learning_rate,
            decouple_weight_decay=True,
            weight_decay=weight_decay,
        )
        optimizer_cfg = config_for_function(chain).set(
            args=(config_for_function(clip_by_global_norm), sgd_cfg),
        )
        cfg = Learner.default_config().set(
            name="test", optimizer=optimizer_cfg, update_rules=update_rules
        )
        learner: Learner = cfg.instantiate(parent=None)

        param_specs = dict(
            v=ParameterSpec(
                dtype=jnp.float32,
                shape=(4,),
                mesh_axes=PartitionSpec("model"),
                factorization=None,
            ),
            c=ParameterSpec(
                dtype=jnp.float32,
                shape=[],
                mesh_axes=PartitionSpec(),
                factorization=None,
            ),
        )
        update_types = {
            k: match_regex_rules(k, rules=update_rules, default_value=UpdateType.ALL_UPDATES)
            for k in param_specs
        }
        sgd_trace = {}
        for k, spec in param_specs.items():
            if should_update_with_optimizers(update_types[k]):
                sgd_trace[k] = OptStateSpec(
                    dtype=spec.dtype,
                    shape=spec.shape,
                    mesh_axes=spec.mesh_axes,
                )
            else:
                sgd_trace[k] = None
        self.assertEqual(
            dict(
                optimizer=(
                    # clip_by_global_norm.
                    optax.EmptyState(),
                    # sgd.
                    (
                        optax.TraceState(trace=sgd_trace),
                        AddDecayedWeightsState(count=None),
                        optax.ScaleByScheduleState(
                            count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec())
                        ),
                    ),
                ),
            ),
            learner.create_state_partition_specs(param_specs),
        )

        params = dict(
            v=OptParam(
                value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1.0,
            ),
            c=OptParam(
                value=jnp.zeros([], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=0,
            ),
        )
        state = learner.init(model_params=params)

        def loss_fn(x):
            return -jax.nn.log_softmax(x["v"])[1]

        loss, grads = jax.value_and_grad(loss_fn)(jax.tree.map(lambda p: p.value, params))
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        self.assertNestedAllClose(
            dict(v=jnp.asarray([0.089629, -0.756364, 0.662272, 0.004462]), c=0.0), grads, atol=1e-6
        )
        for k in grads:
            if not should_update_with_optimizers(update_types[k]):
                grads[k] = None

        updated_params, output_collection = F(
            learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=[
                Updates(
                    delta_updates=grads,
                    opt_params=params,
                    inplace_updates=dict(c=params["c"].value + 1),
                )
            ],
        )
        v_value = params["v"].value
        if should_update_with_optimizers(update_types["v"]):
            expected_new_v = v_value - learning_rate_fn(step) * (
                grads["v"] + weight_decay * v_value
            )
            expected_grad_norm = 1.0093285
        else:
            expected_new_v = v_value
            expected_grad_norm = 0.0

        if update_types["c"] in (UpdateType.ALL_UPDATES, UpdateType.STATE_UPDATES):
            expected_new_c = 1.0
        else:
            expected_new_c = params["c"].value

        self.assertNestedAllClose(
            updated_params,
            dict(v=expected_new_v, c=expected_new_c),
            atol=1e-6,
        )
        summaries = output_collection.summaries
        self.assertNestedAllClose(
            {
                "learning_rate": learning_rate_fn(step),
                "lr_schedule_step": 0,
                "gradient_norm": expected_grad_norm,
                "schedule_step": 0,
                "schedule_scale": -1.0 * learning_rate_fn(step),
            },
            summaries["optimizer"],
        )
        state_updates = output_collection.state_updates
        self.assertNestedAllClose(
            {
                "optimizer": (
                    # clip_by_global_norm.
                    optax.EmptyState(),
                    # sgd.
                    (
                        optax.TraceState(trace=grads),
                        optax.EmptyState(),
                        optax.ScaleByScheduleState(count=jnp.ones([], dtype=jnp.int32)),
                    ),
                ),
            },
            state_updates,
        )

    @parameterized.named_parameters(
        ("default", []),
        ("no_update", [(".*", UpdateType.NO_UPDATE)]),
    )
    def test_update_rules_on_vdict(self, update_rules):
        weight_decay = 1e-4
        learning_rate = 0.1
        optimizer_cfg = config_for_function(adafactor_optimizer).set(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.96,
            clipping_threshold=1,
            multiply_by_parameter_scale=True,
            weight_decay=weight_decay,
            weight_decay_scale_by_learning_rate_exponent=1,
        )
        cfg = Learner.default_config().set(
            name="test", optimizer=optimizer_cfg, update_rules=update_rules
        )
        learner: Learner = cfg.instantiate(parent=None)

        param_specs = VDict(
            v=ParameterSpec(
                dtype=jnp.float32,
                shape=(2, 3),
                mesh_axes=PartitionSpec("model"),
                factorization=None,
            ),
        )
        update_types = {
            k: match_regex_rules(k, rules=update_rules, default_value=UpdateType.ALL_UPDATES)
            for k in param_specs
        }

        params = VDict(
            v=OptParam(
                value=jnp.asarray([[0, 1, 2], [2, -3, 0]], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=None,
            ),
        )
        state = learner.init(model_params=params)

        def loss_fn(params):
            return jnp.mean(jax.vmap(lambda x: -jax.nn.log_softmax(x)[1])(params["v"]))

        loss, grads = jax.value_and_grad(loss_fn)(jax.tree.map(lambda p: p.value, params))
        np.testing.assert_allclose(3.270226, loss, atol=1e-6)
        self.assertNestedAllClose(
            VDict(
                v=jnp.asarray(
                    [[0.045015, -0.377636, 0.33262], [0.4378, -0.49705, 0.05925]], dtype=jnp.float32
                )
            ),
            grads,
            atol=1e-6,
        )
        for k in grads:
            if not should_update_with_optimizers(update_types[k]):
                grads[k] = None

        updated_params, _ = F(
            learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=[
                Updates(
                    delta_updates=grads,
                    opt_params=params,
                    inplace_updates={},
                )
            ],
        )
        v_value = params["v"].value
        if should_update_with_optimizers(update_types["v"]):
            expected_new_v = jnp.asarray(
                [[-0.01291, 1.0129, 1.98707], [1.979163, -2.979153, -0.020817]]
            )
        else:
            expected_new_v = v_value

        self.assertNestedAllClose(dict(v=expected_new_v), updated_params, atol=1e-6)

    def test_per_variable_summaries(self):
        sgd_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=1.0,
            decouple_weight_decay=True,
            weight_decay=1e-4,
        )
        optimizer_cfg = config_for_function(chain).set(
            args=(config_for_function(clip_by_global_norm), sgd_cfg),
        )
        cfg = Learner.default_config().set(
            name="test", optimizer=optimizer_cfg, enable_per_variable_summaries=True
        )
        learner: Learner = cfg.instantiate(parent=None)
        params = dict(
            weight=OptParam(
                value=jnp.asarray([0, 2, 2, -3], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1.0,
            ),
            moving_mean=OptParam(
                value=jnp.array([0, -1, 0, 0], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=0.0,
            ),
        )
        state = learner.init(model_params=params)

        def loss_fn(x):
            return -jax.nn.log_softmax(x["weight"] + x["moving_mean"])[1]

        loss, grads = jax.value_and_grad(loss_fn)(jax.tree.map(lambda p: p.value, params))
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6, rtol=1e-6)
        expected_grad = jnp.asarray([0.089629, -0.756364, 0.662272, 0.004462])
        self.assertNestedAllClose(
            dict(weight=expected_grad, moving_mean=expected_grad), grads, atol=1e-6, rtol=1e-6
        )
        _, output_collection = F(
            learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=[
                Updates(
                    delta_updates=grads,
                    opt_params=params,
                    inplace_updates=dict(moving_mean=params["moving_mean"].value + 1),
                )
            ],
        )
        self.assertNestedAllClose(
            {
                "optimizer/learning_rate": 1.0,
                "optimizer/lr_schedule_step": 0,
                "optimizer/gradient_norm": jnp.sqrt(jnp.sum(2 * expected_grad**2)),
                "param_rms/weight": jnp.sqrt((0 + 4 + 4 + 9) / 4),
                "param_rms/moving_mean": 0.5,
                "grad_rms/weight": jnp.sqrt(jnp.mean(expected_grad**2)),
                "grad_rms/moving_mean": jnp.sqrt(jnp.mean(expected_grad**2)),
                "optimizer/schedule_step": 0,
                "optimizer/schedule_scale": -1.0,
            },
            output_collection.summaries,
        )

    def test_inplace_updates_supersede_delta_updates(self):
        """Tests that inplace updates take precedence over delta updates."""

        class IdentityTransformation(UpdateTransformation):
            def transform_update(self, updates: Updates) -> Updates:
                return updates

        cfg = Learner.default_config().set(
            name="tmp", optimizer=IdentityTransformation.default_config()
        )
        learner = cfg.instantiate(parent=None)
        param_updates = learner.update(
            Updates(
                opt_params=dict(
                    a=OptParam(value=jnp.array(5), factorization_spec=None, weight_decay_scale=None)
                ),
                delta_updates=dict(a=jnp.array(7)),
                inplace_updates=dict(a=jnp.array(3)),
            )
        )
        self.assertEqual(param_updates, dict(a=jnp.array(3)))

    @parameterized.named_parameters(
        ("one_step", 1),  # no accumulation
        ("two_steps", 2),
        ("four_steps", 4),
    )
    def test_gradient_accumulation_init(self, accumulation_steps):
        sgd_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=1.0,
            decouple_weight_decay=True,
            weight_decay=1e-4,
        )
        optimizer_cfg = config_for_function(chain).set(
            args=(config_for_function(clip_by_global_norm), sgd_cfg),
        )
        forward_fn_transformation_cfg = config_for_function(with_minibatch_steps).set(
            steps=accumulation_steps,
            metric_accumulator=MetricAccumulator.default_config(),
        )
        cfg = Learner.default_config().set(
            name="test",
            optimizer=optimizer_cfg,
            forward_fn_transformation=forward_fn_transformation_cfg,
        )
        learner: Learner = cfg.instantiate(parent=None)
        self.assertEqual(accumulation_steps, learner.config.forward_fn_transformation.steps)

    def test_grad_accumulation_numeric(self):
        """Test that the gradient accumulation works as expected."""
        sgd_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=1.0,
            decouple_weight_decay=True,
            weight_decay=1e-4,
        )
        optimizer_cfg = config_for_function(chain).set(
            args=(config_for_function(clip_by_global_norm), sgd_cfg),
        )
        forward_fn_transformation_cfg = config_for_function(with_minibatch_steps).set(
            steps=4,
            metric_accumulator=MetricAccumulator.default_config(),
        )
        cfg = Learner.default_config().set(
            name="test",
            optimizer=optimizer_cfg,
            forward_fn_transformation=forward_fn_transformation_cfg,
        )
        learner: Learner = cfg.instantiate(parent=None)
        params = dict(
            weight=OptParam(
                value=jnp.asarray([0, 2, 2, -3], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1.0,
            ),
            moving_mean=OptParam(
                value=jnp.array([0, -1, 0, 0], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=0.0,
            ),
        )
        state = learner.init(model_params=params)

        def loss_fn(*, model_params, inputs) -> ForwardOutputs:
            del inputs
            loss = -jax.nn.log_softmax(model_params["weight"] + model_params["moving_mean"])[1]
            output_collection = new_output_collection()
            output_collection.state_updates["weight"] = model_params["weight"] + 1
            output_collection.summaries["loss"] = WeightedScalar(loss, 1)
            return ForwardOutputs(loss=loss, aux={}, output_collection=output_collection)

        loss, grads = jax.value_and_grad(lambda x: loss_fn(model_params=x, inputs=None).loss)(
            jax.tree.map(lambda p: p.value, params)
        )
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6, rtol=1e-6)
        expected_grads = jnp.asarray([0.089629, -0.756364, 0.662272, 0.004462])
        # Grad test
        self.assertNestedAllClose(
            dict(
                weight=expected_grads,
                moving_mean=expected_grads,
            ),
            grads,
            atol=1e-6,
            rtol=1e-6,
        )
        # Updated params test
        batch_key, forward_key, param_noise_key = jax.random.split(jax.random.PRNGKey(0), 3)
        updated_params, _ = F(
            learner,
            method="forward_and_backward",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(
                fn=loss_fn,
                opt_params=params,
                inputs=dict(
                    input_batch=jax.random.randint(batch_key, (32, 4096), 1, 100),
                    forward_key=forward_key,
                    param_noise_key=param_noise_key,
                ),
            ),
        )
        updated_params = updated_params.backward_outputs.updated_params
        self.assertNestedAllClose(
            dict(
                weight=jnp.array([1.0, 3.0, 3.0, -2.0]),
                moving_mean=jnp.array([-0.08962882, -0.24363643, -0.6622724, -0.00446236]),
            ),
            updated_params,
            atol=1e-6,
            rtol=1e-6,
        )


class HelperTest(TestCase):
    """Test helper functions."""

    def test__apply_updates(self):
        result = _apply_updates(
            dict(
                a=jnp.array(1),
                c=dict(g=jnp.array(6), h=jnp.array(7)),
                d=optax.MaskedNode(),
                e=jnp.array(4),
            ),
            updates=dict(
                b=tuple(jnp.arange(5)),
                c=dict(h=jnp.array(8)),
                d=jnp.array(3),
                e=optax.MaskedNode(),
                f=jnp.array(5),
            ),
        )
        self.assertEqual(
            result,
            dict(
                a=jnp.array(1),
                b=tuple(jnp.arange(5)),
                c=dict(g=jnp.array(6), h=jnp.array(8)),
                d=optax.MaskedNode(),
                e=jnp.array(4),
                f=jnp.array(5),
            ),
        )

    @staticmethod
    def _forward(
        *,
        model_params: Nested[Tensor],
        inputs: Nested[Tensor],
    ) -> ForwardOutputs:
        loss = inputs * model_params["a"] + model_params["b"]["c"] * model_params["b"]["d"]
        output_collection = new_output_collection()
        output_collection.state_updates["test"] = model_params
        return ForwardOutputs(loss=loss, aux=11, output_collection=output_collection)

    def test__split_gradients(self):
        new_forward, split_args_fn = _split_gradients(
            self._forward, should_compute_gradients=dict(a=True, b=dict(c=True, d=False))
        )
        params = dict(a=3, b=dict(c=5, d=7))
        split_args = split_args_fn(params)
        self.assertEqual(
            split_args, (dict(a=3, b=dict(c=5, d=None)), dict(a=None, b=dict(c=None, d=7)))
        )

        forward_outputs = new_forward(model_params=split_args[0], inputs=(split_args[1], 53))
        self.assertEqual(forward_outputs.loss, 53 * 3 + 5 * 7)
        self.assertEqual(forward_outputs.aux, 11)
        self.assertEqual(forward_outputs.output_collection.state_updates["test"], params)

    def test__value_and_grad(self):
        params = dict(a=3.0, b=dict(c=5.0, d=7.0))
        opt_params = jax.tree.map(
            lambda p: OptParam(value=p, factorization_spec=None, weight_decay_scale=None), params
        )
        updates = _value_and_grad(
            self._forward,
            opt_params=opt_params,
            inputs=53.0,
            should_compute_gradients=dict(a=True, b=dict(c=True, d=False)),
        )

        new_forward, split_args_fn = _split_gradients(
            self._forward, should_compute_gradients=dict(a=True, b=dict(c=True, d=False))
        )
        params = dict(a=3.0, b=dict(c=5.0, d=7.0))
        delta_update = dict(a=53.0, b=dict(c=7.0, d=None))
        model_params_grad, model_params_nograd = split_args_fn(params)
        forward_outputs = self._forward(model_params=params, inputs=53)
        expected = Updates(
            opt_params=opt_params,
            delta_updates=delta_update,
            inplace_updates=dict(test=params),
            forward_pass=dict(
                default=ForwardPass(
                    forward_fn=new_forward,
                    model_params=model_params_grad,
                    inputs=(model_params_nograd, 53.0),
                    outputs=forward_outputs,
                )
            ),
        )
        updates, expected = jax.tree.map(
            lambda x: jnp.asarray(x) if isinstance(x, Number) else x, (updates, expected)
        )
        self.assertEqual(
            updates.forward_pass["default"].forward_fn(
                model_params=model_params_grad, inputs=(model_params_nograd, 53.0)
            ),
            expected.forward_pass["default"].forward_fn(
                model_params=model_params_grad, inputs=(model_params_nograd, 53.0)
            ),
        )
        object.__setattr__(
            expected.forward_pass["default"],
            "forward_fn",
            updates.forward_pass["default"].forward_fn,
        )
        self.assertEqual(updates, expected)


class CompositeLearnerTest(TestCase):
    @parameterized.product(ema_decay=(None, 0.9), method=("update", "forward_and_backward"))
    # pylint: disable-next=too-many-statements
    def test_learner(self, ema_decay: Optional[float], method: str):
        """Sets up two sub learners for encoder/decoder respectively."""
        encoder_lr = 0.1
        opt1_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=encoder_lr, decouple_weight_decay=True, weight_decay=1.0
        )
        opt2_cfg = config_for_function(adam_optimizer).set(
            learning_rate=0.0, b1=0.9, b2=0.99, eps=1e-5, l2_regularizer_weight=1.0
        )
        learner_rules = [(".*encoder.*", "encoder"), (".*decoder.*", "decoder")]

        cfg = CompositeLearner.default_config().set(
            name="test",
            rules=learner_rules,
            learners={
                "encoder": Learner.default_config().set(
                    optimizer=opt1_cfg, enable_per_variable_summaries=True
                ),
                "decoder": Learner.default_config().set(
                    optimizer=opt2_cfg, enable_per_variable_summaries=False
                ),
            },
        )
        cfg.ema.decay = ema_decay
        learner: CompositeLearner = cfg.instantiate(parent=None)

        param_specs = dict(
            encoder=dict(
                weight=ParameterSpec(
                    dtype=jnp.float32,
                    shape=[3, 4],
                    factorization=FactorizationSpec(axes=("row", "col")),
                    mesh_axes=PartitionSpec("data", "model"),
                ),
                bias=ParameterSpec(
                    dtype=jnp.float32,
                    shape=[4],
                    factorization=None,
                    mesh_axes=PartitionSpec("model"),
                ),
                mean=ParameterSpec(
                    dtype=jnp.float32,
                    shape=[2],
                    factorization=None,
                    mesh_axes=PartitionSpec("model"),
                ),
            ),
            decoder=dict(
                head=ParameterSpec(
                    dtype=jnp.float32,
                    shape=[2, 3],
                    factorization=FactorizationSpec(axes=("row", "col")),
                    mesh_axes=PartitionSpec("model", None),
                ),
                scalar=ParameterSpec(
                    dtype=jnp.float32,
                    shape=[],
                    factorization=None,
                    mesh_axes=PartitionSpec(),
                ),
            ),
        )

        # Test partition.
        partition_state = learner.create_state_partition_specs(param_specs)

        # Expected specs from sub optimizer.
        opt1_specs = opt1_cfg.instantiate().partition(param_specs)
        opt2_specs = opt2_cfg.instantiate().partition(param_specs)

        def _check_mask_state(spec):
            self.assertEqual(optax.MaskedNode, spec)

        expected_keys = {"encoder", "decoder"}
        if ema_decay is not None:
            expected_keys.add("ema")
        self.assertEqual(set(partition_state.keys()), expected_keys)
        # sgd states.
        # pytype: disable=attribute-error
        self.assertEqual(
            opt1_specs[0].trace["encoder"],
            partition_state["encoder"]["optimizer"][0].trace["encoder"],
        )
        jax.tree.map(_check_mask_state, partition_state["encoder"]["optimizer"][0].trace["decoder"])
        self.assertSequenceEqual(opt1_specs[1:], partition_state["encoder"]["optimizer"][1:])
        # adam states.
        self.assertEqual(
            opt2_specs[1].mu["decoder"],
            partition_state["decoder"]["optimizer"][1].mu["decoder"],
        )
        jax.tree.map(_check_mask_state, partition_state["decoder"]["optimizer"][1].mu["encoder"])
        self.assertEqual(
            opt2_specs[1].nu["decoder"],
            partition_state["decoder"]["optimizer"][1].nu["decoder"],
        )
        jax.tree.map(_check_mask_state, partition_state["decoder"]["optimizer"][1].nu["encoder"])
        self.assertEqual(opt2_specs[1].count, partition_state["decoder"]["optimizer"][1].count)
        # optax.EmptyState().
        self.assertEqual(opt2_specs[0], partition_state["decoder"]["optimizer"][0])
        self.assertEqual(opt2_specs[-1], partition_state["decoder"]["optimizer"][-1])
        if ema_decay is not None:
            expected_ema_state_spec = ParamEmaState(
                count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
                ema=dict(
                    encoder=dict(
                        weight=OptStateSpec(
                            dtype=jnp.float32,
                            shape=[3, 4],
                            mesh_axes=PartitionSpec("data", "model"),
                        ),
                        bias=OptStateSpec(
                            dtype=jnp.float32,
                            shape=[4],
                            mesh_axes=PartitionSpec("model"),
                        ),
                        mean=OptStateSpec(
                            dtype=jnp.float32,
                            shape=[2],
                            mesh_axes=PartitionSpec("model"),
                        ),
                    ),
                    decoder=dict(
                        head=OptStateSpec(
                            dtype=jnp.float32,
                            shape=[2, 3],
                            mesh_axes=PartitionSpec("model", None),
                        ),
                        scalar=OptStateSpec(
                            dtype=jnp.float32,
                            shape=[],
                            mesh_axes=PartitionSpec(),
                        ),
                    ),
                ),
            )
            self.assertSequenceEqual(partition_state["ema"], expected_ema_state_spec)

        # Test update.
        params = jax.tree.map(
            lambda spec: OptParam(
                value=5.1 * jnp.ones(spec.shape, dtype=spec.dtype),
                factorization_spec=spec.factorization,
                weight_decay_scale=1,
            ),
            param_specs,
        )
        state = learner.init(model_params=params)
        self.assertEqual(set(state.keys()), expected_keys)

        def loss_fn(model_params, inputs):
            del inputs
            output_collection = OutputCollection(
                state_updates=dict(
                    encoder=dict(mean=jnp.array([1.0, 2.0])),
                    decoder=dict(scalar=model_params["decoder"]["scalar"] + 2.7),
                ),
                summaries={},
                module_outputs={},
            )
            result = jax.tree_util.tree_reduce(lambda x, y: x.sum() + y.sum(), model_params)
            return ForwardOutputs(loss=result, aux={}, output_collection=output_collection)

        grads = jax.tree_map(lambda p: jnp.ones_like(p.value), params)

        if method == "update":
            inputs = [
                Updates(
                    delta_updates=grads,
                    opt_params=params,
                    inplace_updates=dict(
                        encoder=dict(mean=jnp.array([1.0, 2.0])),
                        decoder=dict(scalar=params["decoder"]["scalar"].value + 2.7),
                    ),
                )
            ]
            encoder_inputs = copy.deepcopy(inputs)
            del encoder_inputs[0].inplace_updates["decoder"]
            decoder_inputs = copy.deepcopy(inputs)
            del decoder_inputs[0].inplace_updates["encoder"]
        elif method == "forward_and_backward":
            inputs = dict(fn=loss_fn, inputs={}, opt_params=params)
            encoder_inputs = inputs
            decoder_inputs = inputs
        else:
            raise NotImplementedError

        updated_params, output_collection = F(
            learner,
            method=method,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=inputs,
        )

        # Expected updates from sub learner.
        encoder_learner_cfg = Learner.default_config().set(
            name="encoder_learner", optimizer=opt1_cfg, enable_per_variable_summaries=True
        )
        encoder_learner_cfg.ema.decay = ema_decay
        encoder_learner = encoder_learner_cfg.instantiate(parent=None)
        updated_encoder, encoder_collection = F(
            encoder_learner,
            method=method,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=encoder_learner.init(model_params=params),
            inputs=encoder_inputs,
        )
        decoder_learner_cfg = Learner.default_config().set(
            name="decoder_learner", optimizer=opt2_cfg, enable_per_variable_summaries=False
        )
        decoder_learner_cfg.ema.decay = ema_decay
        decoder_learner = decoder_learner_cfg.instantiate(parent=None)
        updated_decoder, decoder_collection = F(
            decoder_learner,
            method=method,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=decoder_learner.init(model_params=params),
            inputs=decoder_inputs,
        )
        if method == "forward_and_backward":
            updated_params = updated_params.backward_outputs.updated_params
            updated_encoder = updated_encoder.backward_outputs.updated_params
            updated_decoder = updated_decoder.backward_outputs.updated_params
        # Test updated params match with sub learners.
        self.assertNestedAllClose(updated_params["decoder"], updated_decoder["decoder"])
        self.assertNestedAllClose(updated_params["encoder"], updated_encoder["encoder"])
        # Test state updates match with sub learners.
        expected_encoder_updates = jax.tree.map(
            # Mask decoder states.
            lambda v, path: optax.MaskedNode() if re.fullmatch(".*decoder.*|.*ema.*", path) else v,
            encoder_collection.state_updates,
            tree_paths(encoder_collection.state_updates),
        )

        expected_decoder_updates = jax.tree.map(
            # Mask encoder states.
            lambda v, path: optax.MaskedNode() if re.fullmatch(".*encoder.*|.*ema.*", path) else v,
            decoder_collection.state_updates,
            tree_paths(decoder_collection.state_updates),
        )
        expected_state_updates = {
            "encoder": expected_encoder_updates,
            "decoder": expected_decoder_updates,
        }
        if ema_decay is not None:
            expected_ema = ParamEmaState(
                count=encoder_collection.state_updates["ema"].count,
                ema=dict(
                    encoder=encoder_collection.state_updates["ema"].ema["encoder"],
                    decoder=decoder_collection.state_updates["ema"].ema["decoder"],
                ),
            )
            expected_state_updates["ema"] = expected_ema

        self.assertNestedAllClose(
            expected_state_updates,
            output_collection.state_updates,
        )
        # Test state_updates match with init state tree structure.
        self.assertEqual(
            {key for key, _ in flatten_items(output_collection.state_updates)},
            {key for key, _ in flatten_items(state)},
        )
        # Test summary.
        expected_summaries = dict(
            encoder={k: v for k, v in encoder_collection.summaries.items() if "decoder" not in k},
            decoder={
                k: v
                for k, v in decoder_collection.summaries.items()
                if ("encoder" not in k or "decoder" not in k)  # No per variable summaries.
            },
        )
        self.assertNestedAllClose(
            expected_summaries,
            output_collection.summaries,
        )
        # pytype: enable=attribute-error

    def test_learner_config(self):
        opt1_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=0.1, decouple_weight_decay=True, weight_decay=1.0
        )
        opt2_cfg = config_for_function(adam_optimizer).set(
            learning_rate=0.0, b1=0.9, b2=0.99, eps=1e-5, l2_regularizer_weight=1.0
        )
        learner_rules = [(".*encoder.*", "encoder"), (".*decoder.*", "decoder")]

        cfg = CompositeLearner.default_config().set(
            name="test",
            rules=learner_rules,
            learners={
                "encoder": Learner.default_config().set(
                    optimizer=opt1_cfg, enable_per_variable_summaries=True
                ),
            },
        )
        with self.assertRaisesRegex(ValueError, ".* is not found in the known learners"):
            # decoder rule does not point to any existing learner.
            cfg.instantiate(parent=None)

        cfg = CompositeLearner.default_config().set(
            name="test",
            rules=[(".*encoder.*", "encoder")],
            learners={
                "encoder": Learner.default_config().set(
                    optimizer=opt1_cfg, enable_per_variable_summaries=True
                ),
                "ema": Learner.default_config().set(
                    optimizer=opt1_cfg, enable_per_variable_summaries=True
                ),
            },
        )
        with self.assertRaisesRegex(ValueError, "Sublearner name cannot be ema"):
            # sublearner name cannot be ema.
            cfg.instantiate(parent=None)

        cfg = CompositeLearner.default_config().set(
            name="test",
            rules=learner_rules,
            learners={
                "encoder": Learner.default_config().set(
                    optimizer=opt1_cfg, enable_per_variable_summaries=True
                ),
                "decoder": Learner.default_config().set(
                    optimizer=opt2_cfg, enable_per_variable_summaries=False
                ),
            },
        )
        learner: CompositeLearner = cfg.instantiate(parent=None)

        param_specs = dict(
            encoder=dict(
                weight=ParameterSpec(
                    dtype=jnp.float32,
                    shape=[3, 4],
                    factorization=FactorizationSpec(axes=("row", "col")),
                    mesh_axes=PartitionSpec("data", "model"),
                ),
            ),
            head=ParameterSpec(
                dtype=jnp.float32,
                shape=[2, 3],
                factorization=FactorizationSpec(axes=("row", "col")),
                mesh_axes=PartitionSpec("model", None),
            ),
        )
        params = jax.tree.map(
            lambda spec: OptParam(
                value=jnp.ones(spec.shape, dtype=spec.dtype),
                factorization_spec=spec.factorization,
                weight_decay_scale=1,
            ),
            param_specs,
        )
        with self.assertRaisesRegex(
            ValueError, "Composite learner rules do not update all model params"
        ):
            # `head` is not covered by any rules.
            learner.init(model_params=params)

    def test_sublearner_ema(self):
        opt1_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=0.1, decouple_weight_decay=True, weight_decay=1.0
        )
        learner_cfg = Learner.default_config().set(optimizer=opt1_cfg)
        learner_cfg.ema.decay = 0.9
        learner_rules = [(".*encoder.*", "encoder")]
        cfg = CompositeLearner.default_config().set(
            name="test",
            rules=learner_rules,
            learners={
                "encoder": learner_cfg,
            },
        )
        with self.assertRaises(ValueError):
            # Sublearner ema is not None.
            cfg.instantiate(parent=None)

    # pylint: disable=no-self-argument
    def test_learner_masking(test_self):
        """In-depth test of the masking of `Learner` and `CompositeLearner`.

         The behavior must exactly match the following.
         1. Mask the leaves of `updates.delta_updates`  with `None` where
            `should_update_with_optimizers()` is `False`.
         2. Mask the leaves of `updates.opt_params` and `updates.inplace_updates`, and also
            the leaves of the arguments to `init()` and `create_state_partition_specs`
            with `optax.MaskedNode()` for leaves that are not assigned to the current sub-learner.
            But only do this if the leaf is not already `None`.
        3. For each leaf that is `None` or `optax.MaskedNode()` in `updates.opt_params`, mask out
           the corresponding leaf in `updates.delta_updates` with the same value.
        4. The optimizer must not change the tree structure.
        5. No implicit masking by key deletion.

        The reason this is required is backwards compatibility with existing state trees and the
        pre-existing `CompositeLearner` implementation.

        """
        updates = axlearn.common.update_transformation_test.mock_updates()

        param_keys = updates.opt_params.keys()
        state_keys = updates.inplace_updates.keys()

        class ExtraCheckingLearner(Learner):
            """A sub-learner that does extra explicit checking that masking was done according
            to the above requirements.
            """

            @config_class
            class Config(Learner.Config):
                # Rules to use for checking that the masking is correct.
                # Maps name of rule -> dictionary of how nodes should be masked under that rule.
                # E.g., dict(rule_1=dict(layer_1=dict(weight_1=optax.MaskedNode))).
                rules: Required[dict[str, dict[str, Union[optax.MaskedNode, None]]]] = REQUIRED

            def init(self, model_params: Nested[OptParam]) -> Nested[Tensor]:
                model_params = cast(dict, model_params)
                test_self.assertEqual(model_params.keys(), param_keys)
                self._check_masking(model_params, rule="2")
                return super().init(model_params)

            def create_state_partition_specs(
                self, model_param_specs: Nested[ParameterSpec]
            ) -> Nested[PartitionSpec]:
                model_param_specs = cast(dict, model_param_specs)
                test_self.assertEqual(model_param_specs.keys(), param_keys)
                self._check_masking(model_param_specs, rule="2")
                return super().create_state_partition_specs(model_param_specs)

            def update(self, updates: Updates) -> Nested[Tensor]:
                test_self.assertSequenceEqual(updates.opt_params.keys(), param_keys)
                test_self.assertSequenceEqual(updates.delta_updates.keys(), param_keys)
                test_self.assertSequenceEqual(updates.inplace_updates.keys(), state_keys)

                self._check_masking(updates.opt_params, rule="2")
                self._check_masking(updates.delta_updates, rule="updates.delta_updates")
                self._check_masking(updates.inplace_updates, rule="2")

                result = super().update(updates)

                chex.assert_trees_all_equal_structs(result, updates.opt_params)
                self._check_masking(result, rule="2")
                return result

            def _check_masking(self, tree: Nested[Any], rule: str):
                """Check that `tree` is masked correctly.

                Args:
                    tree: The tree to check.
                    rule: The rule from `cfg.masking_rules` to check agains.
                """
                cfg = self.config
                tree: dict

                rule_dict: dict[str, Union[optax.MaskedNode, None]] = cfg.rules[rule]

                expected = {k: rule_dict[k] for k in tree if k in rule_dict}
                # Compute actual masked entries.
                masked = [optax.MaskedNode(), None]
                masked = masked + [VDict(bias=m) for m in masked]
                actual = {k: v for k, v in tree.items() if v in masked}

                test_self.assertEqual(actual, expected)

        sublearner_cfg = ExtraCheckingLearner.default_config().set(
            optimizer=config_for_function(sgd_optimizer).set(
                learning_rate=0.01,
                decouple_weight_decay=True,
            ),
            update_rules=[("state", UpdateType.STATE_UPDATES)],
        )
        learner1_params = ["weight", "state"]
        learner2_params = ["vdict/bias", "more_state", "do_not_update"]
        rules = [(name, "l1") for name in learner1_params]
        rules += [(name, "l2") for name in learner2_params]
        learner2_params.remove("vdict/bias")

        # Rule 2 corresponds to rule (2) in the method docstring.
        learner1_rule2 = {k: optax.MaskedNode() for k in learner2_params}
        # We only track the topmost key in the checks in this test.
        learner1_rule2["vdict"] = VDict(bias=optax.MaskedNode())

        # do_not_update was already masked with optax.MaskedNode() from the very start and will
        # remain masked that way.
        learner2_rule2 = {k: optax.MaskedNode() for k in learner1_params + ["do_not_update"]}

        # Rule "updates.delta_updates" corresponds ot rule (1) and (3).
        composite_cfg = CompositeLearner.default_config().set(
            name="tmp",
            learners=dict(
                l1=sublearner_cfg.clone(
                    rules={
                        "2": learner1_rule2,
                        "updates.delta_updates": dict(state=None) | learner1_rule2,
                    }
                ),
                l2=sublearner_cfg.clone(
                    rules={
                        "2": learner2_rule2,
                        "updates.delta_updates": dict(state=None) | learner2_rule2,
                    }
                ),
            ),
            rules=rules,
        )

        learner: CompositeLearner = composite_cfg.instantiate(parent=None)
        learner.create_state_partition_specs(updates.param_specs())
        state = learner.init(updates.opt_params)

        F(learner, method="update", prng_key=None, state=state, inputs=[updates], is_training=True)


if __name__ == "__main__":
    absltest.main()

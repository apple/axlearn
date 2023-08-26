# Copyright © 2023 Apple Inc.

"""Tests learner."""
import re

import jax.nn
import numpy as np
import optax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import schedule
from axlearn.common.base_layer import FactorizationSpec, ParameterSpec
from axlearn.common.config import config_for_function
from axlearn.common.learner import (
    CompositeLearner,
    Learner,
    UpdateType,
    should_update_with_optimizers,
)
from axlearn.common.module import functional as F
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
from axlearn.common.utils import PartitionSpec, VDict, flatten_items, match_regex_rules, tree_paths


class LearnerTest(TestCase):
    @parameterized.parameters(None, 0.9)
    def test_learner(self, ema_decay):
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

        def loss_fn(x):
            return -jax.nn.log_softmax(x["v"])[1]

        loss, grads = jax.value_and_grad(loss_fn)(jax.tree_util.tree_map(lambda p: p.value, params))
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        self.assertNestedAllClose(
            dict(v=jnp.asarray([0.089629, -0.756364, 0.662272, 0.004462]), c=0.0), grads, atol=1e-6
        )

        updated_params, output_collection = F(
            learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(
                gradients=grads,
                model_params=params,
                state_updates=dict(c=params["c"].value + 1),
            ),
        )
        v_value = params["v"].value
        expected_new_v = v_value - learning_rate_fn(step) * (grads["v"] + weight_decay * v_value)
        self.assertNestedAllClose(
            updated_params,
            dict(v=expected_new_v, c=1.0),
            atol=1e-6,
        )
        summaries = output_collection.summaries
        self.assertNestedAllClose(
            {
                "learning_rate": learning_rate_fn(step),
                "lr_schedule_step": 0,
                "gradient_norm": 1.0093285,
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
                ema=jax.tree_util.tree_map(lambda v: v * (1 - ema_decay), updated_params),
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

        loss, grads = jax.value_and_grad(loss_fn)(jax.tree_util.tree_map(lambda p: p.value, params))
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
            inputs=dict(
                gradients=grads,
                model_params=params,
                state_updates=dict(c=params["c"].value + 1),
            ),
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
            },
            summaries,
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

        loss, grads = jax.value_and_grad(loss_fn)(jax.tree_util.tree_map(lambda p: p.value, params))
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
            inputs=dict(
                gradients=grads,
                model_params=params,
                state_updates={},
            ),
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

        loss, grads = jax.value_and_grad(loss_fn)(jax.tree_util.tree_map(lambda p: p.value, params))
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
            inputs=dict(
                gradients=grads,
                model_params=params,
                state_updates=dict(moving_mean=params["moving_mean"].value + 1),
            ),
        )
        self.assertNestedAllClose(
            {
                "learning_rate": 1.0,
                "lr_schedule_step": 0,
                "gradient_norm": jnp.sqrt(jnp.sum(2 * expected_grad**2)),
                "param_rms/weight": jnp.sqrt((0 + 4 + 4 + 9) / 4),
                "param_rms/moving_mean": 0.5,
                "grad_rms/weight": jnp.sqrt(jnp.mean(expected_grad**2)),
                "grad_rms/moving_mean": jnp.sqrt(jnp.mean(expected_grad**2)),
            },
            output_collection.summaries,
        )


class CompositeLearnerTest(TestCase):
    @parameterized.parameters(None, 0.9)
    # pylint: disable-next=too-many-statements
    def test_learner(self, ema_decay):
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
        jax.tree_util.tree_map(
            _check_mask_state, partition_state["encoder"]["optimizer"][0].trace["decoder"]
        )
        self.assertSequenceEqual(opt1_specs[1:], partition_state["encoder"]["optimizer"][1:])
        # adam states.
        self.assertEqual(
            opt2_specs[1].mu["decoder"],
            partition_state["decoder"]["optimizer"][1].mu["decoder"],
        )
        jax.tree_util.tree_map(
            _check_mask_state, partition_state["decoder"]["optimizer"][1].mu["encoder"]
        )
        self.assertEqual(
            opt2_specs[1].nu["decoder"],
            partition_state["decoder"]["optimizer"][1].nu["decoder"],
        )
        jax.tree_util.tree_map(
            _check_mask_state, partition_state["decoder"]["optimizer"][1].nu["encoder"]
        )
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
        params = jax.tree_util.tree_map(
            lambda spec: OptParam(
                value=5.1 * jnp.ones(spec.shape, dtype=spec.dtype),
                factorization_spec=spec.factorization,
                weight_decay_scale=1,
            ),
            param_specs,
        )
        state = learner.init(model_params=params)
        self.assertEqual(set(state.keys()), expected_keys)
        grads = jax.tree_map(lambda p: jnp.ones_like(p.value), params)
        updated_params, output_collection = F(
            learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(
                gradients=grads,
                model_params=params,
                state_updates=dict(
                    encoder=dict(mean=jnp.array([1.0, 2.0])),
                    decoder=dict(scalar=params["decoder"]["scalar"].value + 2.7),
                ),
            ),
        )
        # Expected updates from sub learner.
        encoder_learner_cfg = Learner.default_config().set(
            name="encoder_learner", optimizer=opt1_cfg, enable_per_variable_summaries=True
        )
        encoder_learner_cfg.ema.decay = ema_decay
        encoder_learner = encoder_learner_cfg.instantiate(parent=None)
        updated_encoder, encoder_collection = F(
            encoder_learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=encoder_learner.init(model_params=params),
            inputs=dict(
                gradients=grads,
                model_params=params,
                state_updates=dict(
                    encoder=dict(mean=jnp.array([1.0, 2.0])),
                ),
            ),
        )
        decoder_learner_cfg = Learner.default_config().set(
            name="decoder_learner", optimizer=opt2_cfg, enable_per_variable_summaries=False
        )
        decoder_learner_cfg.ema.decay = ema_decay
        decoder_learner = decoder_learner_cfg.instantiate(parent=None)
        updated_decoder, decoder_collection = F(
            decoder_learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=decoder_learner.init(model_params=params),
            inputs=dict(
                gradients=grads,
                model_params=params,
                state_updates=dict(decoder=dict(scalar=params["decoder"]["scalar"].value + 2.7)),
            ),
        )
        # Test updated params match with sub learners.
        self.assertNestedAllClose(updated_params["decoder"], updated_decoder["decoder"])
        self.assertNestedAllClose(updated_params["encoder"], updated_encoder["encoder"])
        # Test state updates match with sub learners.
        expected_encoder_updates = jax.tree_util.tree_map(
            # Mask decoder states.
            lambda v, path: optax.MaskedNode() if re.fullmatch(".*decoder.*|.*ema.*", path) else v,
            encoder_collection.state_updates,
            tree_paths(encoder_collection.state_updates),
        )

        expected_decoder_updates = jax.tree_util.tree_map(
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
            set(key for key, _ in flatten_items(output_collection.state_updates)),
            set(key for key, _ in flatten_items(state)),
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
        params = jax.tree_util.tree_map(
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


if __name__ == "__main__":
    absltest.main()

# Copyright © 2023 Apple Inc.

"""Tests optimization modules."""
# pylint: disable=no-self-use,too-many-lines
import itertools
from typing import Optional

import jax
import numpy as np
import optax
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import schedule
from axlearn.common.base_layer import FactorizationSpec, NestedParameterSpec, ParameterSpec
from axlearn.common.config import config_for_function
from axlearn.common.optimizer_base import OptParam, OptStateSpec, PartitionedGradientTransformation
from axlearn.common.optimizers import (
    ParamEmaState,
    adafactor_optimizer,
    adam_optimizer,
    adamw_optimizer,
    add_decayed_weights,
    chain,
    clip_by_block_rms,
    clip_by_global_norm,
    copy_partition,
    ema,
    l2_regularizer,
    lion_optimizer,
    opt_param_values,
    param_ema,
    per_param_scale_by_path,
    scale_by_param_block_rms,
    scale_by_trust_ratio,
    scale_update_per_param,
    sgd_optimizer,
    with_partition_fn,
)
from axlearn.common.schedule import Schedule, adafactor_decay_rate, decay_bias_correction
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import (
    NestedPartitionSpec,
    PartitionSpec,
    Tensor,
    VDict,
    flatten_items,
    shapes,
)


def rms_norm(x):
    return jnp.sqrt(jnp.mean(x**2))


def optax_ema_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    def partition_fn(
        param_specs: NestedParameterSpec,
    ) -> NestedPartitionSpec:
        return optax.EmaState(count=None, ema=copy_partition(param_specs))

    return with_partition_fn(base, partition_fn)


class OptimizerTest(TestCase):
    """Tests optimization modules."""

    @parameterized.parameters(
        (
            config_for_function(scale_update_per_param).set(
                per_param_scale=config_for_function(per_param_scale_by_path).set(
                    description="test_partition", scale_by_path=[]
                )
            ),
        ),
        (config_for_function(clip_by_global_norm).set(max_norm=10),),
        (config_for_function(clip_by_block_rms).set(threshold=1),),
        (config_for_function(add_decayed_weights).set(weight_decay=0.1),),
        (
            config_for_function(add_decayed_weights).set(
                weight_decay=0.1, learning_rate_exponent=1, learning_rate=0.1
            ),
        ),
        (
            config_for_function(sgd_optimizer).set(
                learning_rate=0.1, weight_decay=0.01, decouple_weight_decay=False
            ),
        ),
        (
            config_for_function(sgd_optimizer).set(
                learning_rate=0.1, weight_decay=0.01, decouple_weight_decay=True
            ),
        ),
        (
            config_for_function(adamw_optimizer).set(
                learning_rate=0.1, weight_decay=0.01, b1=0.9, b2=0.96, eps=1e-5
            ),
        ),
        (config_for_function(ema).set(decay=0.1, debias=False),),
        (config_for_function(ema).set(decay=0.1, debias=True),),
        (config_for_function(ema).set(decay=0.1, debias=True, accumulator_dtype=jnp.bfloat16),),
        (config_for_function(ema).set(decay=0.1, debias=True, accumulator_dtype=jnp.int8),),
        (
            config_for_function(adafactor_optimizer).set(
                learning_rate=config_for_function(schedule.adafactor),
                b1=0.9,
                b2=0.96,
                multiply_by_parameter_scale=False,
                clipping_threshold=None,
                dtype_momentum=jnp.float32,
                weight_decay=0.01,
                weight_decay_scale_by_learning_rate_exponent=0,
                weight_decay_per_param_scale=None,
                eps=1e-30,
                factored=False,
            ),
        ),
        (
            config_for_function(adafactor_optimizer).set(
                learning_rate=config_for_function(schedule.adafactor),
                b1=0.9,
                b2=0.96,
                multiply_by_parameter_scale=True,
                clipping_threshold=1,
                dtype_momentum=jnp.int8,
                weight_decay=0.01,
                weight_decay_scale_by_learning_rate_exponent=1,
                weight_decay_per_param_scale=None,
                eps=1e-30,
                factored=True,
                apply_scale_by_trust_ratio=True,
            ),
        ),
        (
            config_for_function(lion_optimizer).set(
                learning_rate=0.1,
                weight_decay=0.01,
                b1=0.9,
                b2=0.96,
            ),
        ),
    )
    def test_partition_fn(self, optimizer_cfg):
        """Tests that opt.{init,mesh_axes} are consistent with each other."""
        opt: PartitionedGradientTransformation = optimizer_cfg.instantiate()
        param_specs = dict(
            matrix=ParameterSpec(
                dtype=jnp.float32,
                shape=[3, 4],
                factorization=FactorizationSpec(axes=("row", "col")),
                mesh_axes=("data", "model"),
            ),
            vector=ParameterSpec(
                dtype=jnp.float32,
                shape=[4],
                factorization=None,
                mesh_axes=PartitionSpec(
                    "model",
                ),
            ),
            scalar=ParameterSpec(
                dtype=jnp.float32,
                shape=[],
                factorization=None,
                mesh_axes=PartitionSpec(),
            ),
        )
        opt_specs = opt.partition(param_specs)
        print(opt_specs)
        params = jax.tree_util.tree_map(
            lambda spec: OptParam(
                value=jnp.ones(spec.shape, dtype=spec.dtype),
                factorization_spec=spec.factorization,
                weight_decay_scale=1,
            ),
            param_specs,
        )
        states = opt.init(params)
        print(states)

        def _check_spec(spec: OptStateSpec, state: Tensor):
            self.assertEqual(spec.dtype, state.dtype)
            self.assertSequenceEqual(spec.shape, state.shape)

        jax.tree_util.tree_map(_check_spec, opt_specs, states)

    @parameterized.parameters((0.1, 0, True), (0.1, 0.01, True), (0.1, 0.01, False))
    def test_sgd_optimizer(self, learning_rate, weight_decay, decouple_weight_decay):
        sgd = sgd_optimizer(
            learning_rate=learning_rate,
            decouple_weight_decay=decouple_weight_decay,
            weight_decay=weight_decay,
        )
        params = OptParam(
            value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32),
            factorization_spec=None,
            weight_decay_scale=1.0,
        )
        state = sgd.init(params)

        def loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(loss)(params.value)
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        np.testing.assert_allclose(grads, [0.089629, -0.756364, 0.662272, 0.004462], atol=1e-6)

        updates, _ = sgd.update(grads, state=state, params=params)
        np.testing.assert_allclose(
            updates, -learning_rate * (grads + weight_decay * params.value), atol=1e-6
        )

        updated_params = optax.apply_updates(params.value, updates)
        np.testing.assert_allclose(updated_params, params.value + updates, atol=1e-6)

    @parameterized.parameters((0.1, 0, False), (0.1, 0.01, True), (0.1, 0.0, True))
    def test_adamw_optimizer(self, learning_rate, weight_decay, multiply_by_parameter_scale):
        self._test_optimizer(
            adamw_optimizer(
                learning_rate=learning_rate,
                b1=0.9,
                b2=0.99,
                eps=1e-5,
                weight_decay=weight_decay,
                multiply_by_parameter_scale=multiply_by_parameter_scale,
            )
        )

    @parameterized.parameters((0.1, 0.0), (0.1, 0.01))
    def test_adam_optimizer(self, learning_rate, l2_regularizer_weight):
        # Note l2_regularizer_weight is sufficiently small so that the loss decreases.
        self._test_optimizer(
            adam_optimizer(
                learning_rate=learning_rate,
                b1=0.9,
                b2=0.99,
                eps=1e-5,
                l2_regularizer_weight=l2_regularizer_weight,
            )
        )

    @parameterized.product(
        learning_rate=(0.1, 0.01),
        multiply_by_parameter_scale=(False, True),
        clipping_threshold=(None, 1.0),
        apply_scale_by_trust_ratio=(False, True),
    )
    def test_adafactor_optimizer(
        self,
        learning_rate,
        multiply_by_parameter_scale,
        clipping_threshold,
        apply_scale_by_trust_ratio,
    ):
        self._test_optimizer(
            adafactor_optimizer(
                learning_rate=learning_rate,
                b1=0.9,
                b2=0.98,
                multiply_by_parameter_scale=multiply_by_parameter_scale,
                clipping_threshold=clipping_threshold,
                apply_scale_by_trust_ratio=apply_scale_by_trust_ratio,
            )
        )

    @parameterized.parameters((0.1, 0, False), (0.1, 0.01, False), (0.1, 0.0, True))
    def test_lion_optimizer(self, learning_rate, weight_decay, multiply_by_parameter_scale):
        self._test_optimizer(
            lion_optimizer(
                learning_rate=learning_rate,
                b1=0.9,
                b2=0.99,
                weight_decay=weight_decay,
                multiply_by_parameter_scale=multiply_by_parameter_scale,
            )
        )

    @parameterized.product(
        mu_dtype=(None, jnp.bfloat16, jnp.float32), params_dtype=(jnp.bfloat16, jnp.float32)
    )
    def test_lion_optimizer_dtype(self, mu_dtype, params_dtype):
        """Tests that dtypes are consistent between init, update and partition."""

        # Construct params.
        params = OptParam(
            value=jnp.array(0, dtype=params_dtype),
            factorization_spec=None,
            weight_decay_scale=None,
        )
        param_specs = ParameterSpec(shape=params.shape, dtype=params.dtype)
        grads = jnp.array(0, dtype=params_dtype)

        # Construct states.
        base = lion_optimizer(learning_rate=1.0, b1=0.9, b2=0.99, mu_dtype=mu_dtype)
        init_state = base.init(params)
        partition_state = base.partition(param_specs)
        _, update_state = base.update(grads, init_state, params)

        logging.info("init_state=%s", init_state)
        logging.info("partition_state=%s", partition_state)
        logging.info("update_state=%s", update_state)

        def _check_dtypes(x, y, z):
            self.assertTrue(
                getattr(x, "dtype", None) == getattr(y, "dtype", None) == getattr(z, "dtype", None)
            )

        jax.tree_util.tree_map(_check_dtypes, init_state, partition_state, update_state)

    def _test_optimizer(self, optimizer):
        params = OptParam(
            value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32),
            factorization_spec=None,
            weight_decay_scale=1.0,
        )
        state = optimizer.init(params)

        param_spec = ParameterSpec(shape=[4], mesh_axes=PartitionSpec("model"), factorization=None)
        state_partition_spec = optimizer.partition(param_spec)
        logging.info("state_partition_spec=%s state=%s", state_partition_spec, shapes(state))

        def check_partition_spec(spec: OptStateSpec, tree):
            if spec.mesh_axes is None:
                return
            self.assertIsInstance(tree, Tensor)
            self.assertEqual(list(spec.shape), list(tree.shape))
            self.assertEqual(len(spec.mesh_axes), tree.ndim)

        jax.tree_util.tree_map(check_partition_spec, state_partition_spec, state)

        def compute_loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(compute_loss)(params.value)
        updates, _ = optimizer.update(grads, state=state, params=params)
        updated_params = optax.apply_updates(params.value, updates)
        new_loss = compute_loss(updated_params)
        self.assertLess(new_loss, loss)

    @parameterized.product(
        learning_rate=(0.0, 0.1),
        weight_decay=(None, 0.0, 0.01),
        weight_decay_scale_by_learning_rate_exponent=(None, 0.0, 1.0),
    )
    def test_adafactor_weight_decay(
        self,
        learning_rate: float,
        weight_decay: Optional[float],
        weight_decay_scale_by_learning_rate_exponent: Optional[float],
    ):
        optimizer_kwargs = dict(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.98,
            multiply_by_parameter_scale=False,
            clipping_threshold=None,
            weight_decay=weight_decay,
            weight_decay_scale_by_learning_rate_exponent=(
                weight_decay_scale_by_learning_rate_exponent
            ),
        )
        if weight_decay is not None and weight_decay_scale_by_learning_rate_exponent is None:
            with self.assertRaisesRegex(ValueError, "weight_decay_scale_by_learning_rate_exponent"):
                adafactor_optimizer(**optimizer_kwargs)
            return
        optimizer = adafactor_optimizer(**optimizer_kwargs)
        params = OptParam(
            value=jnp.asarray([4, 1, 2, -3, 100], dtype=jnp.float32),
            factorization_spec=None,
            weight_decay_scale=1.0,
        )
        state = optimizer.init(params)
        print(params)

        def compute_loss(x):
            # x[-1] does not affect loss.
            return -jax.nn.log_softmax(x[:-1])[1]

        grads = jax.grad(compute_loss)(params.value)
        updates, _ = optimizer.update(grads, state=state, params=params)
        updated_value = optax.apply_updates(params.value, updates)
        print(updated_value)
        if not weight_decay or (not learning_rate and weight_decay_scale_by_learning_rate_exponent):
            # No weight decay on value[-1].
            self.assertEqual(updated_value[-1], params.value[-1])
        elif weight_decay_scale_by_learning_rate_exponent:
            # Weight decay on value[-1] is scaled by the learning rate.
            self.assertAlmostEqual(
                updated_value[-1], params.value[-1] * (1 - weight_decay * learning_rate)
            )
        else:
            # Weight decay on value[-1] is not scaled by the learning rate.
            self.assertAlmostEqual(updated_value[-1], params.value[-1] * (1 - weight_decay))
        # If weight_decay_scale_by_learning_rate_exponent is none zero, all updates are controlled
        # by the learning_rate.
        if learning_rate == 0 and (
            not weight_decay or weight_decay_scale_by_learning_rate_exponent
        ):
            self.assertNestedAllClose(updated_value, params.value)
        else:
            self.assertFalse(
                any(u == v for u, v in zip(updated_value[:-1].tolist(), params.value[:-1].tolist()))
            )

    @parameterized.parameters(
        (
            0.9,
            0.999,
            [-1.0123808, 6.0123806, -3.0123806, -8.012381, 100.0],
        ),
        (
            0.9,
            config_for_function(decay_bias_correction).set(decay=0.999),
            [-0.482145, 5.482144, -2.482145, -7.482145, 100.0],
        ),
        (
            0.9,
            config_for_function(adafactor_decay_rate),
            [-0.48214498, 5.4821444, -2.482145, -7.482145, 100.0],
        ),
        (
            config_for_function(decay_bias_correction).set(decay=0.9),
            config_for_function(decay_bias_correction).set(decay=0.999),
            [-23.960178, 14.513834, -23.645355, -23.724747, 100.0],
        ),
    )
    def test_adafactor_beta_schedules(self, b1: Schedule, b2: Schedule, expected_value):
        optimizer = adafactor_optimizer(
            learning_rate=0.1,
            b1=b1,
            b2=b2,
            multiply_by_parameter_scale=True,
            clipping_threshold=1.0,
        )
        params = OptParam(
            value=jnp.asarray([4, 1, 2, -3, 100], dtype=jnp.float32),
            factorization_spec=None,
            weight_decay_scale=1.0,
        )
        state = optimizer.init(params)
        print(params)

        def compute_loss(x):
            # x[-1] does not affect loss.
            return -jax.nn.log_softmax(x[:-1])[1]

        for _ in range(10):
            grads = jax.grad(compute_loss)(params.value)
            updates, _ = optimizer.update(grads, state=state, params=params)
            params.value = optax.apply_updates(params.value, updates)
            print(params.value)
        assert_allclose(params.value, expected_value)

    @parameterized.parameters(([0, 0, 0, 0],), ([1, 2, 3, 4],))
    def test_adamw_multiply_by_parameter_scale(self, params):
        # We set a min scale for param rms.
        params_rms = max(1e-3, jnp.sqrt(jnp.mean(jnp.asarray(params, dtype=jnp.float32) ** 2)))
        optimizer_pps = adamw_optimizer(
            learning_rate=0.01,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0,
            multiply_by_parameter_scale=True,
        )
        optimizer_no_pps = adamw_optimizer(
            learning_rate=0.01,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0,
            multiply_by_parameter_scale=False,
        )
        params_pps = OptParam(
            value=jnp.asarray(params, dtype=jnp.float32),
            factorization_spec=None,
            weight_decay_scale=0.0,
        )
        params_no_pps = OptParam(
            value=jnp.asarray(params, dtype=jnp.float32),
            factorization_spec=None,
            weight_decay_scale=0.0,
        )
        state_pps = optimizer_pps.init(params_pps)
        state_no_pps = optimizer_no_pps.init(params_no_pps)

        def compute_loss(x):
            return -jax.nn.log_softmax(x)[1]

        _, grads_pps = jax.value_and_grad(compute_loss)(params_pps.value)
        updates_pps, _ = optimizer_pps.update(grads_pps, state=state_pps, params=params_pps)
        _, grads_no_pps = jax.value_and_grad(compute_loss)(params_no_pps.value)
        updates_no_pps, _ = optimizer_no_pps.update(
            grads_no_pps, state=state_no_pps, params=params_no_pps
        )

        def check_pps(update_pps, update_no_pps, param_rms):
            assert_allclose(update_pps, update_no_pps * param_rms)

        jax.tree_util.tree_map(check_pps, updates_pps, updates_no_pps, params_rms)

    @parameterized.product(
        optimizer_cfg=[config_for_function(sgd_optimizer).set(decouple_weight_decay=True)],
        param_scale=[0.0, 2.0],
    )
    def test_weight_scaling(self, optimizer_cfg, param_scale):
        # Set config to scale weights.
        scale_update_cfg = config_for_function(scale_update_per_param).set(
            per_param_scale=config_for_function(per_param_scale_by_path).set(
                scale_by_path=[
                    ("(.*/)?base", param_scale),
                ],
                description="weight_update_scale",
            ),
        )

        # Set default options.
        optimizer_cfg.set(
            learning_rate=-1,
            weight_decay=0,
            weight_decay_per_param_scale=None,
        )

        # Chain the optimizer config and scale updates config afterward.
        optimizer = chain(
            optimizer_cfg,
            scale_update_cfg,
        )

        params = dict(
            base=OptParam(
                value=jnp.ones([1, 1], dtype=jnp.float32) * 2.1,
                factorization_spec=None,
                # `scale_update_per_param` shouldn't be affected by `weight_decay_scale`.
                weight_decay_scale=100.0,
            ),
            head=OptParam(
                value=jnp.ones([1, 1], dtype=jnp.float32) * 3.4,
                factorization_spec=None,
                weight_decay_scale=None,
            ),
        )
        state = optimizer.init(params)

        grads = jax.tree_map(jnp.ones_like, opt_param_values(params))

        updates, _ = optimizer.update(grads, state=state, params=params)
        updated_value = optax.apply_updates(opt_param_values(params), updates)

        self.assertNestedAllClose(
            updated_value["base"],
            params["base"].value + 1.0 * param_scale,
        )

        self.assertNestedAllClose(
            updated_value["head"],
            params["head"].value + 1.0,
        )

    @parameterized.named_parameters(
        ("no_per_param_scale", 1.0, None),
        ("bias_0.5", 0.5, None),
        ("sgd_bias_0.2", 0.2, config_for_function(sgd_optimizer).set(decouple_weight_decay=True)),
        (
            "adamw_bias_0.2",
            0.2,
            config_for_function(adamw_optimizer).set(b1=0.9, b2=0.95, eps=1e-6),
        ),
        (
            "adafactor_bias_0.2",
            0.2,
            config_for_function(adafactor_optimizer).set(
                b1=0.9,
                b2=0.999,
                multiply_by_parameter_scale=False,
                clipping_threshold=None,
                weight_decay_scale_by_learning_rate_exponent=1,
            ),
        ),
    )
    def test_weight_decay_per_param_scales(self, bias_scale=1.0, optimizer_cfg=None):
        weight_decay = 0.1
        if bias_scale != 1:
            per_param_scale = config_for_function(per_param_scale_by_path).set(
                scale_by_path=[
                    ("(.*/)?bias", bias_scale),
                ],
                description="weight_decay_scale",
            )
        else:
            per_param_scale = None
        if optimizer_cfg is None:
            optimizer_cfg = config_for_function(add_decayed_weights).set(
                learning_rate=-1,
                learning_rate_exponent=1,
                per_param_scale=per_param_scale,
            )
        else:
            optimizer_cfg.set(
                learning_rate=1,
                weight_decay_per_param_scale=per_param_scale,
            )
        optimizer = optimizer_cfg.set(weight_decay=weight_decay).instantiate()
        params = dict(
            weight=OptParam(
                value=jnp.ones([1, 1], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=None,
            ),
            bias=OptParam(
                value=jnp.ones([1, 1], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=None,
            ),
            moving_mean=OptParam(
                value=jnp.ones([1, 1], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=0.0,
            ),
        )
        state = optimizer.init(params)
        # Start with zero gradients. Only apply weight decay.
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, opt_param_values(params))
        updates, _ = optimizer.update(zero_grads, state=state, params=params)
        updated_value = optax.apply_updates(opt_param_values(params), updates)
        # Weight is decayed with scale 1.
        self.assertNestedAllClose(
            updated_value["weight"], params["weight"].value * (1 - weight_decay)
        )
        # Bias is decayed at scale bias_scale.
        self.assertNestedAllClose(
            updated_value["bias"], params["bias"].value * (1 - weight_decay * bias_scale)
        )
        # moving_mean is not decayed.
        self.assertNestedAllClose(updated_value["moving_mean"], params["moving_mean"].value)

    @parameterized.product(max_norm=(None, 100.0, 0.1), drop_norm=(None, 5.0, 0.5))
    def test_gradient_clipping(self, max_norm, drop_norm):
        clip = clip_by_global_norm(max_norm=max_norm, drop_norm=drop_norm)
        params = jnp.asarray([0, 1, 2, -3], dtype=jnp.float32)
        state = clip.init(params)

        # This test has similarities with learner_test.test_learner.
        # pylint: disable=duplicate-code
        def loss_fn(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(loss_fn)(params)
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        np.testing.assert_allclose(grads, [0.089629, -0.756364, 0.662272, 0.004462], atol=1e-6)
        # pylint: enable=duplicate-code

        g_norm = optax.global_norm(grads)
        updates, _ = clip.update(grads, state=state, params=params)
        if drop_norm is None or g_norm < drop_norm:
            if max_norm is None or g_norm < max_norm:
                np.testing.assert_allclose(updates, grads, atol=1e-6)
            else:
                np.testing.assert_allclose(max_norm, optax.global_norm(updates))
        else:
            np.testing.assert_allclose(updates, jnp.zeros_like(grads))

    @parameterized.product(
        regularizer_weight=(0.0, 1.0),
        per_param_scale=(
            None,
            config_for_function(per_param_scale_by_path).set(
                # Excluding the bias parameters from l2 regularizer.
                description="l2_regularizer_scale",
                scale_by_path=[(".*bias.*", 0)],
            ),
        ),
    )
    def test_l2_regularizer(self, regularizer_weight, per_param_scale):
        apply_l2 = l2_regularizer(
            regularizer_weight=regularizer_weight,
            per_param_scale=per_param_scale,
        )

        params = VDict(
            weight=OptParam(
                value=jnp.array([2, 3], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=None,
            ),
            bias=OptParam(
                value=jnp.array([1], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=None,
            ),
            moving_mean=OptParam(
                value=jnp.array([1], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=0.0,
            ),
        )
        state = apply_l2.init(params)
        self.assertEqual(optax.EmptyState, type(state))
        # Start with zero gradients.
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, opt_param_values(params))
        updates, _ = apply_l2.update(zero_grads, state=state, params=params)
        updated_value = optax.apply_updates(opt_param_values(params), updates)
        if not per_param_scale:
            per_param_l2_weight = VDict(
                weight=regularizer_weight,
                bias=regularizer_weight,
                moving_mean=0.0,
            )
        else:
            per_param_l2_weight = VDict(
                weight=regularizer_weight,
                bias=0.0,
                moving_mean=0.0,
            )
        self.assertNestedAllClose(
            updated_value,
            jax.tree_util.tree_map(lambda p, s: p.value + p.value * s, params, per_param_l2_weight),
        )

    def test_scale_by_trust_ratio(self):
        trust_ratio = scale_by_trust_ratio()
        params = VDict(
            x=OptParam(
                value=jnp.asarray([[0, 0, 0, 0], [0, 1, 2, -3]], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1,
            )
        )
        state = trust_ratio.init(params)
        self.assertEqual(optax.ScaleByTrustRatioState, type(state))

        grads = VDict(x=jnp.asarray([[1e-5] * 4, [1] * 4], dtype=jnp.float32))
        updates, _ = trust_ratio.update(grads, state=state, params=params)

        np.testing.assert_allclose(
            updates["x"], [[1e-5] * 4, [1.8708287, 1.8708287, 1.8708287, 1.8708287]], atol=1e-6
        )

    @parameterized.parameters(100.0, 1e-3)
    def test_clip_by_block_rms(self, max_norm):
        clip = clip_by_block_rms(threshold=max_norm)
        params = VDict(x=jnp.asarray([[0, 0, 0, 0], [0, 1, 2, -3]], dtype=jnp.float32))
        state = clip.init(params)
        self.assertEqual(optax.EmptyState, type(state))

        def loss(params):
            return -jax.nn.log_softmax(params["x"])[:, 1].mean()

        loss, grads = jax.value_and_grad(loss)(params)
        assert_allclose(loss, 1.399186)
        assert_allclose(
            [[0.125, -0.375, 0.125, 0.125], [0.044814, -0.378182, 0.331136, 0.002231]], grads["x"]
        )

        g_norm = jax.vmap(rms_norm)(grads["x"])
        assert_allclose([0.216506, 0.252332], g_norm)

        updates, _ = clip.update(grads, state=state, params=params)
        if max_norm > 1:
            np.testing.assert_allclose(updates["x"], grads["x"], atol=1e-6)
        else:
            np.testing.assert_allclose(
                jax.vmap(rms_norm)(updates["x"]), [max_norm] * 2, atol=1e-6, rtol=1e-6
            )

    @parameterized.parameters(100.0, 1e-3)
    def test_scale_by_param_block_rms(self, threshold):
        scale = scale_by_param_block_rms(threshold)
        params = VDict(
            x=OptParam(
                value=jnp.asarray([[0, 0, 0, 0], [0, 1, 2, -3]], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1,
            )
        )
        p_norm = jax.vmap(rms_norm)(params["x"].value)
        state = scale.init(params)
        self.assertEqual(optax.EmptyState, type(state))

        grads = VDict(x=jnp.asarray([[1e-5] * 4, [1] * 4], dtype=jnp.float32))

        g_norm = jax.vmap(rms_norm)(grads["x"])
        assert_allclose([1e-5, 1.0], g_norm)

        updates, _ = scale.update(grads, state=state, params=params)
        np.testing.assert_allclose(
            jax.vmap(rms_norm)(updates["x"]), jnp.maximum(p_norm, threshold) * g_norm
        )

    # pylint: disable=too-many-branches
    @parameterized.parameters(
        itertools.product(
            (jnp.float32, jnp.bfloat16, jnp.int16, jnp.int8), (True, False), (0.9, 0.1)
        )
    )
    def test_ema_parity(self, accumulator_dtype, debias, momentum):
        float_dtypes = [jnp.float32, jnp.bfloat16]
        # Compare against a fp32 reference if using int for experimental.
        ref_dtype = accumulator_dtype if accumulator_dtype in float_dtypes else jnp.float32
        ref = optax_ema_partition(optax.ema(momentum, debias=debias, accumulator_dtype=ref_dtype))
        exp = ema(momentum, debias=debias, accumulator_dtype=accumulator_dtype)

        # Partition on the 'expert' and 'model' axes.
        num_experts, model_dim, hidden_dim = 8, 150, 512
        parameter_partition_specs = dict(
            w=ParameterSpec(
                dtype=jnp.float32,
                shape=[num_experts, model_dim, hidden_dim],
                mesh_axes=PartitionSpec("expert", None, "model"),
            ),
            x=ParameterSpec(
                dtype=jnp.float32,
                shape=[num_experts, hidden_dim],
                mesh_axes=PartitionSpec("expert", "model"),
            ),
            b=ParameterSpec(
                dtype=jnp.float32,
                shape=[hidden_dim],
                mesh_axes=PartitionSpec("model"),
            ),
        )

        # Check state.ema partition specs are the same.
        ref_partition: PartitionSpec = ref.partition(parameter_partition_specs)
        exp_partition: PartitionSpec = exp.partition(parameter_partition_specs)
        for k, v in ref_partition.ema.items():
            exp_v = exp_partition.ema[k]
            self.assertEqual(accumulator_dtype, exp_v.dtype)
            self.assertEqual(v.shape, exp_v.shape)
            self.assertEqual(v.mesh_axes, exp_v.mesh_axes)
        # Check state.scale partition specs are right for each dtype.
        for k, v in ref_partition.ema.items():
            exp_scale = exp_partition.scale
            if accumulator_dtype in float_dtypes or not v.shape:
                # No quantization.
                self.assertEqual(exp_scale[k].shape, (1,))
                self.assertEqual(exp_scale[k].mesh_axes, PartitionSpec(None))
            else:
                # Int accumulators that require quantization.
                self.assertEqual(exp_scale[k].shape, v.shape[1:])
                self.assertEqual(exp_scale[k].mesh_axes, PartitionSpec(*v.mesh_axes[1:]))

        # Check init behaves the same except for additional scale values.
        opt_params = dict(
            w=OptParam(
                value=jax.random.normal(
                    jax.random.PRNGKey(1),
                    [num_experts, model_dim, hidden_dim],
                    dtype=jnp.bfloat16,
                ),
                factorization_spec=None,
                weight_decay_scale=1,
            ),
            x=OptParam(
                value=jnp.zeros([num_experts, hidden_dim], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1,
            ),
            b=OptParam(
                value=jnp.zeros([hidden_dim], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1,
            ),
        )
        ref_opt_state = ref.init(opt_params)
        exp_opt_state = exp.init(opt_params)

        # Check that exp_opt_state matches exp_partition.
        partition_spec_map = dict(flatten_items(exp_partition))
        for path, value in flatten_items(exp_opt_state):
            partition_spec = partition_spec_map.get(path)
            logging.info(
                "State: %s=%s(%s) state_spec=%s", path, value.dtype, value.shape, partition_spec
            )
            if partition_spec is None:
                continue
            self.assertSequenceEqual(
                value.shape, partition_spec.shape, msg=f"{path}: {partition_spec} vs {value.shape}"
            )
            self.assertLen(
                partition_spec.mesh_axes,
                len(value.shape),
                msg=f"{path}: {partition_spec} vs {value.shape}",
            )

        self.assertEqual(exp_opt_state.count, ref_opt_state.count)
        if accumulator_dtype in float_dtypes:
            self.assertNestedAllClose(exp_opt_state.ema, ref_opt_state.ema)
        else:
            # Check de-quantized equivalence.
            for k, v in ref_opt_state.ema.items():
                assert_allclose(v, exp_opt_state.ema[k].astype(v.dtype) * exp_opt_state.scale[k])

        self.assertEqual(exp_opt_state.scale.keys(), exp_opt_state.ema.keys())
        for k, v in exp_opt_state.scale.items():
            # The scale factors should be fp32.
            self.assertEqual(v.dtype, jnp.float32)

        # Check update behaves the same, excepting for any quantization error.
        for step in range(10):
            updates = jax.tree_util.tree_map(
                lambda x, key=jax.random.PRNGKey(100 + step): jax.random.normal(key, x.shape),
                opt_params,
            )
            ref_scaled_updates, ref_opt_state = ref.update(updates, ref_opt_state, opt_params)
            exp_scaled_updates, exp_opt_state = exp.update(updates, exp_opt_state, opt_params)
            self.assertEqual(exp_opt_state.count, ref_opt_state.count)
            if accumulator_dtype in float_dtypes:
                self.assertNestedAllClose(exp_scaled_updates, ref_scaled_updates)
                self.assertNestedAllClose(exp_opt_state.ema, ref_opt_state.ema)
            else:
                # Quantization error should be upper bounded by O(max_value / qstep_size).
                for k, v in ref_scaled_updates.items():
                    atol = 2 * jnp.max(jnp.abs(v)) / jnp.iinfo(accumulator_dtype).max
                    assert_allclose(exp_scaled_updates[k], v, atol=atol)

    @parameterized.product(
        decay=(None, 0.9, decay_bias_correction(0.9)),
        dtype=(jnp.float32, jnp.bfloat16),
    )
    def test_param_ema(self, decay, dtype):
        opt = param_ema(decay=decay)
        param_specs = dict(
            v=ParameterSpec(
                dtype=dtype,
                shape=[4],
                mesh_axes=PartitionSpec("model"),
            ),
        )
        opt_specs = opt.partition(param_specs)
        if decay is None:
            self.assertEqual(optax.EmptyState(), opt_specs)
        else:
            self.assertEqual(
                ParamEmaState(
                    count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
                    ema=dict(
                        v=OptStateSpec(dtype=dtype, shape=[4], mesh_axes=PartitionSpec("model"))
                    ),
                ),
                opt_specs,
            )

        params = dict(
            v=OptParam(
                value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32),
                factorization_spec=None,
                weight_decay_scale=1.0,
            ),
        )
        state: ParamEmaState = opt.init(params)
        if decay is None:
            self.assertEqual(optax.EmptyState(), state)
        else:
            self.assertNestedAllClose(
                ParamEmaState(
                    count=0, ema=jax.tree_util.tree_map(lambda p: jnp.zeros_like(p.value), params)
                ),
                state,
            )

        _, new_state = opt.update({}, state=state, params=params)
        if decay is None:
            self.assertEqual(optax.EmptyState(), new_state)
        else:
            self.assertEqual(new_state.count, 1)
            if isinstance(decay, float):
                self.assertNestedAllClose(
                    jax.tree_util.tree_map(lambda p: (1 - decay) * p.value, params),
                    new_state.ema,
                )
            else:
                self.assertNestedAllClose(
                    jax.tree_util.tree_map(lambda p: p.value, params),
                    new_state.ema,
                )


if __name__ == "__main__":
    absltest.main()

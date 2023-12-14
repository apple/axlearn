# Copyright © 2023 Apple Inc.

"""Tests param initializers."""
# pylint: disable=duplicate-code
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.param_init import (
    PARAM_REGEXP_BIAS,
    PARAM_REGEXP_SCALE,
    PARAM_REGEXP_WEIGHT,
    ConstantInitializer,
    DefaultInitializer,
    FanAxes,
    PerGroupInitializer,
    WeightInitializer,
    maybe_prepend_axis,
)
from axlearn.common.test_utils import TestCase, assert_allclose


class DefaultInitializerTest(TestCase):
    @parameterized.parameters(0.0, 0.1, -0.1)
    def test_bias(self, bias_init_value):
        cfg = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_BIAS: ConstantInitializer.default_config().set(value=bias_init_value)
            }
        )
        init: DefaultInitializer = cfg.instantiate()

        bias = init.initialize("bias", prng_key=jax.random.PRNGKey(1), shape=[4], dtype=jnp.float16)
        self.assertEqual(bias.dtype, jnp.float16)
        if bias_init_value:
            np.testing.assert_array_equal(bias, jnp.ones_like(bias) * bias_init_value)
        else:
            np.testing.assert_array_equal(bias, jnp.zeros_like(bias))

    def test_scale(self):
        init: DefaultInitializer = DefaultInitializer.default_config().instantiate()
        scale = init.initialize(
            "scale", prng_key=jax.random.PRNGKey(1), shape=[4], dtype=jnp.bfloat16
        )
        self.assertEqual(scale.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(scale, jnp.ones_like(scale))

    def test_init_by_param_name_add(self):
        cfg = DefaultInitializer.default_config()
        init: DefaultInitializer = cfg.instantiate()
        weight_shape = [1000, 10]
        with self.assertRaisesRegex(NotImplementedError, "Unsupported parameter name"):
            weight = init.initialize(
                "codebook",
                prng_key=jax.random.PRNGKey(1),
                shape=weight_shape,
                dtype=jnp.float32,
                axes=FanAxes(in_axis=0, out_axis=1),
            )
        scale = 2.0
        cfg.init_by_param_name = {
            ".*codebook$": WeightInitializer.default_config().set(
                fan="fan_in",
                scale=scale,
                distribution="uniform",
            )
        }
        init: DefaultInitializer = cfg.instantiate()
        weight = init.initialize(
            "codebook",
            prng_key=jax.random.PRNGKey(1),
            shape=weight_shape,
            dtype=jnp.float32,
            axes=FanAxes(in_axis=0, out_axis=1),
        )
        variance = scale / weight_shape[0]
        data_range = jnp.sqrt(3 * variance)
        assert_allclose(jnp.min(weight), -data_range, rtol=1e-4)
        assert_allclose(jnp.max(weight), data_range, rtol=1e-4)
        # Tests that default keys are set.
        bias_shape = [1000]
        bias = init.initialize(
            "bias",
            prng_key=jax.random.PRNGKey(1),
            shape=bias_shape,
            dtype=jnp.float32,
        )
        assert_allclose(bias, jnp.zeros(bias_shape))

    def test_init_by_param_name_update(self):
        init = DefaultInitializer.default_config().instantiate()
        # Test default weight initializer config.
        self.assertEqual(init.config.init_by_param_name[PARAM_REGEXP_WEIGHT].fan, "fan_avg")
        self.assertEqual(init.config.init_by_param_name[PARAM_REGEXP_WEIGHT].scale, 1.0)
        self.assertEqual(
            init.config.init_by_param_name[PARAM_REGEXP_WEIGHT].distribution, "uniform"
        )

        cfg = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan="fan_out", scale=2.0, distribution="uniform"
                )
            }
        )
        # Tests that config is updated.
        weight_init_cfg = cfg.init_by_param_name[PARAM_REGEXP_WEIGHT]
        self.assertEqual(weight_init_cfg.fan, "fan_out")
        self.assertEqual(weight_init_cfg.scale, 2.0)
        self.assertEqual(weight_init_cfg.distribution, "uniform")
        weight_init_cfg = cfg.instantiate().config.init_by_param_name[PARAM_REGEXP_WEIGHT]
        self.assertEqual(weight_init_cfg.fan, "fan_out")
        self.assertEqual(weight_init_cfg.scale, 2.0)
        self.assertEqual(weight_init_cfg.distribution, "uniform")

    def test_init_by_param_name_order(self):
        init = DefaultInitializer.default_config().instantiate()
        # Tests the order.
        self.assertEqual(
            list(init.config.init_by_param_name.keys()),
            [PARAM_REGEXP_BIAS, PARAM_REGEXP_SCALE, PARAM_REGEXP_WEIGHT],
        )
        cfg = DefaultInitializer.default_config().set(
            init_by_param_name={
                ".*bbias": ConstantInitializer.default_config().set(value=-10),
                # Note this overwrites the default entry.
                PARAM_REGEXP_BIAS: ConstantInitializer.default_config().set(value=2.718),
                "bias": ConstantInitializer.default_config().set(value=6),
            }
        )
        init: DefaultInitializer = cfg.instantiate()
        self.assertEqual(
            list(init.config.init_by_param_name.keys()),
            [".*bbias", PARAM_REGEXP_BIAS, "bias", PARAM_REGEXP_SCALE, PARAM_REGEXP_WEIGHT],
        )
        bias_shape = [10]
        bias = init.initialize(
            "bias",
            prng_key=jax.random.PRNGKey(1),
            shape=bias_shape,
            dtype=jnp.float32,
        )
        # "bias" matches with ".*bias$" and "bias". The first match is ".*bias$".
        assert_allclose(bias, 2.718 * jnp.ones(bias_shape))

        bias2 = init.initialize(
            "bbias",
            prng_key=jax.random.PRNGKey(1),
            shape=bias_shape,
            dtype=jnp.float32,
        )
        # "bbias" matches with ".*bbias" and ".*bias$". The first match is ".*bbias".
        assert_allclose(bias2, -10 * jnp.ones(bias_shape))


class WeightInitializerTest(TestCase):
    # pylint: disable-next=no-self-use
    def test_none_fan(self):
        scale = 1.0
        init: WeightInitializer = (
            WeightInitializer.default_config()
            .set(fan=None, scale=scale, distribution="uniform")
            .instantiate()
        )
        weight_shape = [100, 100]
        weight = init.initialize(
            "weight", prng_key=jax.random.PRNGKey(1), shape=weight_shape, dtype=jnp.float32
        )
        std_err = 1 / (12 * np.sqrt(np.prod(weight_shape))) * scale * 2
        self.assertBetween(np.mean(weight), 0.0 - 6 * std_err, 0.0 + 6 * std_err)
        assert_allclose(jnp.min(weight), -scale, rtol=1e-4)
        assert_allclose(jnp.max(weight), scale, rtol=1e-4)


class PerGroupInitializerTest(TestCase):
    # pylint: disable-next=no-self-use
    def test_single_group(
        self,
    ):
        init_cfg = WeightInitializer.default_config()
        init = init_cfg.instantiate()
        per_group_init = (
            PerGroupInitializer.default_config()
            .set(
                initializer=init_cfg,
                num_groups=1,
            )
            .instantiate()
        )

        shape = [3, 3, 4, 6]

        prng_key = jax.random.PRNGKey(123)
        init_args = dict(
            prng_key=prng_key,
            shape=shape,
            dtype=jnp.float32,
            axes=FanAxes(in_axis=-2, out_axis=-1),
        )
        weight_standard_init = init.initialize(
            "weight",
            **init_args,
        )
        weight_per_group_init = per_group_init.initialize(
            "weight",
            **init_args,
        )

        # no difference when using num_groups=1
        assert_allclose(weight_standard_init, weight_per_group_init)

    # pylint: disable-next=no-self-use
    def test_constant_init(
        self,
    ):
        init_cfg = ConstantInitializer.default_config().set(value=0.5)
        init = init_cfg.instantiate()
        per_group_init = (
            PerGroupInitializer.default_config()
            .set(
                initializer=init_cfg,
                num_groups=6,
            )
            .instantiate()
        )

        shape = [3, 3, 4, 6]

        prng_key = jax.random.PRNGKey(123)
        init_args = dict(
            prng_key=prng_key,
            shape=shape,
            dtype=jnp.float32,
            axes=FanAxes(in_axis=-2, out_axis=-1),
        )

        weight_standard_init = init.initialize(
            "weight",
            **init_args,
        )
        weight_per_group_init = per_group_init.initialize(
            "weight",
            **init_args,
        )

        # no difference when using a deterministic initializer with num_groups != 1
        assert_allclose(weight_standard_init, weight_per_group_init)

    @parameterized.parameters(
        (32, 1),
        (1024, 32),
    )
    def test_per_group_initialization(
        self,
        dim: int,
        num_input_dim_groups: int,
    ):
        init_cfg = WeightInitializer.default_config().set(
            fan="fan_out",
            distribution="normal",
        )
        init = init_cfg.instantiate()
        per_group_init = (
            PerGroupInitializer.default_config()
            .set(
                initializer=init_cfg,
                num_groups=num_input_dim_groups,
            )
            .instantiate()
        )

        shape = (1, 1, dim // num_input_dim_groups, dim)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        init_args = dict(
            prng_key=prng_key,
            shape=shape,
            dtype=jnp.float32,
            axes=FanAxes(in_axis=-2, out_axis=-1),
        )

        weight_standard_init = init.initialize(
            "weight",
            **init_args,
        )
        weight_per_group_init = per_group_init.initialize(
            "weight",
            **init_args,
        )

        # test the shapes
        self.assertEqual(shape, weight_standard_init.shape)
        self.assertEqual(shape, weight_per_group_init.shape)

        expected_std_per_group = np.sqrt(1.0 / (dim // num_input_dim_groups))
        expected_std = np.sqrt(1.0 / dim)
        # Choosing high variance as we comparing the std of a random sample
        # to the std of the generating distribution
        assert_allclose(expected_std_per_group, weight_per_group_init.std(), atol=1e-2, rtol=1e-1)
        assert_allclose(expected_std, weight_standard_init.std(), atol=1e-2, rtol=1e-1)


class FanAxesTest(TestCase):
    def test_normalize(self):
        fan_axes = FanAxes(in_axis=0, out_axis=(1,), batch_axis=(2, 1))
        fan_axes = fan_axes.canonicalize()
        self.assertEqual(fan_axes.in_axis, (0,))
        self.assertEqual(fan_axes.out_axis, (1,))
        self.assertEqual(fan_axes.batch_axis, (1, 2))

    def test_eq(self):
        self.assertEqual(
            FanAxes(in_axis=(0,), out_axis=1),
            FanAxes(in_axis=0, out_axis=(1,)),
        )
        self.assertNotEqual(
            FanAxes(in_axis=(0,), out_axis=1),
            42,
        )
        self.assertNotEqual(
            FanAxes(in_axis=(0,), out_axis=1),
            FanAxes(in_axis=0, out_axis=(2,)),
        )

    def test_prepend_axis(self):
        initial = FanAxes(in_axis=0, out_axis=1, batch_axis=-1)
        self.assertEqual(
            initial.prepend_axis(axis_type=FanAxes.AxisType.OUT_AXIS),
            FanAxes(in_axis=1, out_axis=(0, 2), batch_axis=-1),
        )
        self.assertEqual(
            initial.prepend_axis(axis_type=FanAxes.AxisType.NONE),
            FanAxes(in_axis=1, out_axis=2, batch_axis=-1),
        )

    def test_append_axis(self):
        initial = FanAxes(in_axis=-1, out_axis=-2, batch_axis=0)
        self.assertEqual(
            initial.append_axis(axis_type=FanAxes.AxisType.OUT_AXIS),
            FanAxes(in_axis=-2, out_axis=(-1, -3), batch_axis=0),
        )
        self.assertEqual(
            initial.append_axis(axis_type=FanAxes.AxisType.NONE),
            FanAxes(in_axis=-2, out_axis=-3, batch_axis=0),
        )

    def test_maybe_prepend_axis(self):
        initial = FanAxes(in_axis=0, out_axis=1, batch_axis=-1)
        self.assertEqual(
            maybe_prepend_axis(initial, axis_type=FanAxes.AxisType.OUT_AXIS),
            FanAxes(in_axis=1, out_axis=(0, 2), batch_axis=-1),
        )


if __name__ == "__main__":
    absltest.main()

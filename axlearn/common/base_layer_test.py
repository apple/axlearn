# Copyright © 2023 Apple Inc.

"""Tests BaseLayer."""
import dataclasses
import math
from functools import partial
from typing import Dict, List, Optional

import jax.ad_checkpoint
import jax.core
import jax.interpreters.ad
import jax.random
import numpy as np
from absl.testing import absltest, parameterized
from jax import checkpoint_policies as jax_remat_policies
from jax import numpy as jnp

from axlearn.common import param_init, utils
from axlearn.common.base_layer import (
    BaseLayer,
    NestedTensor,
    ParameterNoise,
    ParameterSpec,
    RematSpec,
    no_remat,
)
from axlearn.common.config import config_class
from axlearn.common.module import Module, OutputCollection
from axlearn.common.module import functional as F
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    DefaultInitializer,
    FanAxes,
    WeightInitializer,
)
from axlearn.common.test_utils import TestCase, assert_allclose


class TestLayer(BaseLayer):
    """A dummy layer."""

    def _create_layer_parameter_specs(self):
        return {
            "moving_mean": ParameterSpec(
                shape=[],
                mesh_axes=None,
                dtype=jnp.float32,
                initializer=param_init.constant_initializer(1.0),
            )
        }

    def forward(self, x):
        self.add_summary("x", x)
        self.add_state_update("moving_mean", 0.1 * x + 0.9 * self.state["moving_mean"])
        y = x - self.state["moving_mean"]
        self.add_module_output("forward", y)
        return y


class TestParentLayer(BaseLayer):
    """A parent layer."""

    @config_class
    class Config(BaseLayer.Config):
        child: TestLayer.Config = TestLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("child", cfg.child)

    def forward(self, x):
        return self.child(x)


class TestIntermediateOutputLayer(BaseLayer):
    """A parent layer."""

    @config_class
    class Config(BaseLayer.Config):
        child1: TestLayer.Config = TestLayer.default_config()
        child2: TestLayer.Config = TestLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("child1", cfg.child1)
        self._add_child("child2", cfg.child2)

    def forward(self, x):
        expected_child1_output = self.child1(x)
        out = self.child2(expected_child1_output)

        child1_output = utils.get_recursively(self.get_module_outputs(), "child1/forward")
        return {"child1_output": child1_output, "out": out}


class ParameterScaler(ParameterNoise):
    """A dummy param noise layer."""

    @config_class
    class Config(ParameterNoise.Config):
        scale: float = 1.0

    def apply(self, prng_key: jax.random.KeyArray, params: NestedTensor) -> NestedTensor:
        cfg = self.config
        return jax.tree_util.tree_map(lambda x: x * cfg.scale, params)


def _callback_primitive(forward, backward):
    def forward_impl(x):
        forward()
        return x

    def backward_impl(x):
        backward()
        return (x,)

    prim = jax.core.Primitive("passthrough_with_callback")
    prim.def_impl(forward_impl)
    prim.def_abstract_eval(forward_impl)
    jax.interpreters.ad.deflinear(prim, backward_impl)
    return prim.bind


class TestRematLayer(BaseLayer):
    """A dummy remat layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures TestRematLayer."""

        output_name: str = "op"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        self.op = jnp.square
        self._forward_calls = []
        self._backward_calls = []
        self._callback = _callback_primitive(
            lambda: self._forward_calls.append(None), lambda: self._backward_calls.append(None)
        )

    def forward(self, x):
        h = self._callback(self.op(x))
        y = self._remat_name(h, self.config.output_name)
        return y

    @property
    def num_forward(self):
        return len(self._forward_calls)

    @property
    def num_backward(self):
        return len(self._backward_calls)


class TestRematParentLayer(BaseLayer):
    """A dummy parent layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures TestRematParentLayer."""

        child_names: List[str] = []

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        for name in cfg.child_names:
            self._add_child(
                name,
                TestRematLayer.default_config().set(output_name=name, remat_spec=cfg.remat_spec),
            )

    def forward(self, x):
        return self.layer2(self.layer1(x))


class BaseLayerTest(TestCase):
    """Tests BaseLayer."""

    def test_forward(self):
        test_module: TestLayer = (
            TestLayer.default_config().set(name="test").instantiate(parent=None)
        )
        state = test_module.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(456))
        self.assertEqual({"moving_mean": 1.0}, state)
        y, output_collection = jax.jit(partial(F, test_module, is_training=True))(
            prng_key=jax.random.PRNGKey(123), state=state, inputs=(jnp.asarray(5.0),)
        )
        self.assertEqual(4, y)
        self.assertEqual(
            OutputCollection(
                summaries={"x": 5}, state_updates={"moving_mean": 1.4}, module_outputs={}
            ),
            output_collection,
        )

    def test_intermediate_output(self):
        test_module: TestIntermediateOutputLayer = (
            TestIntermediateOutputLayer.default_config().set(name="test").instantiate(parent=None)
        )
        state = test_module.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(456))
        y, output_collection = jax.jit(partial(F, test_module, is_training=True))(
            prng_key=jax.random.PRNGKey(123), state=state, inputs=(jnp.asarray(5.0),)
        )
        self.assertEqual(4, y["child1_output"])
        self.assertEqual(3, y["out"])
        self.assertEqual(output_collection.module_outputs, {})

    def test_parent_forward(self):
        test_module: TestParentLayer = (
            TestParentLayer.default_config().set(name="test").instantiate(parent=None)
        )
        state = test_module.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(456))
        self.assertEqual({"child": {"moving_mean": 1.0}}, state)
        y, output_collection = jax.jit(partial(F, test_module, is_training=True))(
            prng_key=jax.random.PRNGKey(123), state=state, inputs=(jnp.asarray(5.0),)
        )
        self.assertEqual(4, y)
        self.assertEqual(
            OutputCollection(
                summaries={"child": {"x": 5}},
                state_updates={"child": {"moving_mean": 1.4}},
                module_outputs={},
            ),
            output_collection,
        )

    def test_remat_name(self):
        var_tag = "save_var"
        test_module: TestRematLayer = (
            TestRematLayer.default_config()
            .set(name="test", output_name=var_tag)
            .instantiate(parent=None)
        )
        state = test_module.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(456))
        # Check that the output variable was tagged with <var_tag>.
        jaxpr = jax.make_jaxpr(partial(F, test_module, is_training=True))(
            prng_key=jax.random.PRNGKey(123), state=state, inputs=(jnp.ones([]),)
        )
        tagged_params = [el for el in jaxpr.eqns if "name" in el.params]
        self.assertEqual(len(tagged_params), 1)
        tagged_param = tagged_params.pop()
        self.assertIsInstance(tagged_param.primitive, jax.core.Primitive)
        self.assertEqual(tagged_param.primitive.name, "name")
        self.assertEqual(f"{type(test_module).__name__}.{var_tag}", tagged_param.params.get("name"))

    def test_remat_causes_additional_forwards(
        self, remat_spec=RematSpec(policy=jax_remat_policies.nothing_saveable)
    ):
        test_module: TestRematParentLayer = (
            TestRematParentLayer.default_config()
            .set(name="test", child_names=["layer1", "layer2"])
            .instantiate(parent=None)
        )
        test_module_remat = test_module.config.set(remat_spec=remat_spec).instantiate(parent=None)
        state = test_module.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(456))
        state_remat = test_module_remat.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(456)
        )

        def loss(inputs, state, module):
            return F(
                module,
                is_training=True,
                state=state,
                prng_key=jax.random.PRNGKey(123),
                inputs=(inputs,),
            )[0]

        v, g = jax.value_and_grad(partial(loss, state=state, module=test_module))(jnp.ones([]))
        v_remat, g_remat = jax.value_and_grad(
            partial(loss, state=state_remat, module=test_module_remat)
        )(jnp.ones([]))
        # Check values and grads line up.
        self.assertEqual(v, v_remat)
        self.assertEqual(g, g_remat)
        # Check both have same number of __call__.
        self.assertEqual(state, state_remat)
        # Check that both have the same number of "backward" calls.
        for k, v in test_module.children.items():
            self.assertEqual(v.num_backward, test_module_remat.children[k].num_backward)
        # Check that the first layer has more forward calls in the remat case.
        self.assertGreater(test_module_remat.layer1.num_forward, test_module.layer1.num_forward)

    @parameterized.parameters(
        None, ParameterScaler.default_config(), ParameterScaler.default_config().set(scale=0)
    )
    def test_apply_parameter_noise_recursively(self, param_noise_cfg):
        test_module: TestLayer = (
            TestLayer.default_config()
            .set(name="test", param_noise=param_noise_cfg)
            .instantiate(parent=None)
        )
        state = test_module.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(456))
        noisy_state = test_module.apply_parameter_noise_recursively(
            prng_key=jax.random.PRNGKey(789), params=state
        )
        # apply_parameter_noise_recursively creates a copy of "state".
        self.assertIsNot(noisy_state, state)
        if param_noise_cfg is None or param_noise_cfg.scale == 1:
            self.assertNestedAllClose(state, noisy_state)
        else:
            self.assertEqual(param_noise_cfg.scale, 0)
            for (orig_path, orig_value), (noisy_path, noisy_value) in zip(
                utils.flatten_items(state), utils.flatten_items(noisy_state)
            ):
                self.assertEqual(orig_path, noisy_path)
                self.assertNestedAllClose(jnp.zeros_like(orig_value), noisy_value)

    @parameterized.parameters(None, (jnp.array([0, 10, 5, 7]),), (jnp.array([0, 0, 0, 0]),))
    def test_activation_summary(self, lengths):
        test_layer: TestLayer = TestLayer.default_config().set(name="test").instantiate(parent=None)
        data_key, init_key, prng_key = jax.random.split(jax.random.PRNGKey(567), num=3)
        batch_size = 4

        if lengths is None:
            inputs = jax.random.normal(key=data_key, shape=[batch_size, 3, 2, 4]) * 10.0
            paddings = None
        else:
            max_len = 10
            inputs = jax.random.normal(key=data_key, shape=[batch_size, max_len, 2, 4]) * 10.0
            paddings = jnp.arange(max_len)[None, :] >= lengths[:, None]
        layer_params = test_layer.initialize_parameters_recursively(prng_key=init_key)
        _, output_collections = F(
            test_layer,
            inputs=dict(activations=inputs, name="inputs", activation_paddings=paddings),
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
            method="_add_activation_summary",
        )
        if lengths is None:
            expected_mean = jnp.mean(inputs)
            expected_norm = jnp.mean(jnp.sqrt(jnp.mean(inputs**2, axis=(1, 2, 3))))
            self.assertEqual(
                output_collections.summaries["activations/inputs_mean"].weight, batch_size
            )
            self.assertEqual(
                output_collections.summaries["activations/inputs_norm"].weight, batch_size
            )
        else:
            num_frames = jnp.sum(1 - paddings)
            self.assertEqual(
                output_collections.summaries["activations/inputs_mean"].weight, num_frames
            )
            self.assertEqual(
                output_collections.summaries["activations/inputs_norm"].weight, num_frames
            )
            inputs_with_padding = inputs * (1 - paddings)[:, :, None, None]
            expected_mean = jnp.sum(inputs_with_padding) / jnp.maximum(1, num_frames) / (2 * 4)
            inputs_norm = jnp.sqrt(jnp.sum(inputs_with_padding**2, axis=(2, 3)) / (2 * 4))
            expected_norm = jnp.sum(inputs_norm) / jnp.maximum(1, num_frames)

        assert_allclose(
            output_collections.summaries["activations/inputs_mean"].mean,
            expected_mean,
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            output_collections.summaries["activations/inputs_norm"].mean,
            expected_norm,
            atol=1e-6,
            rtol=1e-6,
        )

    @parameterized.parameters(True, False)
    # pylint: disable-next=no-self-use
    def test_activation_summary_toy_example(self, with_paddings):
        test_layer: TestLayer = TestLayer.default_config().set(name="test").instantiate(parent=None)
        init_key, prng_key = jax.random.split(jax.random.PRNGKey(112))

        # shape [2, 2, 3]
        inputs = jnp.array([[[-1, 1, -2], [-1, 1, 0]], [[0, 0, 1], [0, 2, -1]]]).astype(jnp.float32)

        if with_paddings:
            paddings = jnp.array([[0, 1], [1, 1]])
            expected_mean = -2 / 3.0
            expected_norm = np.sqrt((1 + 1 + 4) / 3)
        else:
            paddings = None
            expected_mean = 0 / 12.0
            expected_norm = 0.5 * (np.sqrt((1 + 1 + 4 + 1 + 1) / 6) + np.sqrt((1 + 1 + 4) / 6))

        layer_params = test_layer.initialize_parameters_recursively(prng_key=init_key)
        _, output_collections = F(
            test_layer,
            inputs=dict(activations=inputs, name="inputs", activation_paddings=paddings),
            is_training=True,
            prng_key=prng_key,
            state=layer_params,
            method="_add_activation_summary",
        )

        assert_allclose(
            output_collections.summaries["activations/inputs_mean"].mean,
            expected_mean,
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            output_collections.summaries["activations/inputs_norm"].mean,
            expected_norm,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_no_remat_inheritance(self):
        # Check that @no_remat is preserved by inheritance unless the method
        # is explicitly overriden by one without @no_remat.
        class AnotherTestLayer(BaseLayer):
            @no_remat
            def fn(self, st: str):
                pass

        class Subclass1(AnotherTestLayer):
            pass

        class Subclass2(AnotherTestLayer):
            def fn(self, st: str):
                pass

        self.assertTrue(hasattr(AnotherTestLayer.fn, "_no_remat"))
        self.assertTrue(hasattr(Subclass1.fn, "_no_remat"))
        self.assertFalse(hasattr(Subclass2.fn, "_no_remat"))

    def test_no_remat(self):
        # pylint: disable=missing-class-docstring
        # Checks that using @no_remat allows calling a function with a non-JAX type.
        class AnotherTestLayer(BaseLayer):
            @config_class
            class Config(BaseLayer.Config):
                remat_spec: Optional[RematSpec] = RematSpec(
                    policy=jax_remat_policies.nothing_saveable
                )

            def forward(self, x):
                b = self.fn("three")
                x = b * x
                return x.sum()

            @no_remat
            def fn(self, st: str):
                if st == "three":
                    return 3

        # Pytype doesn't like us directly accessing the _no_remat attribute, so we use getattr.
        self.assertTrue(getattr(AnotherTestLayer.fn, "_no_remat", False))

        layer = AnotherTestLayer.default_config().set(name="tmp").instantiate(parent=None)
        params = {}
        rng = jax.random.PRNGKey(0)
        jit_value_and_grad = jax.jit(
            lambda *args, inputs, **kwargs: jax.value_and_grad(
                lambda inputs: F(layer, *args, inputs=inputs, is_training=True, **kwargs)[0]
            )(inputs)
        )
        _ = jit_value_and_grad(prng_key=rng, state=params, inputs=[jax.numpy.ones(5)])


class ComputeFanAxesTest(TestCase):
    """Tests compute_fan_axes."""

    class DefaultFanLayer(BaseLayer):
        """A layer using default _compute_fan_axes."""

        def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
            return {
                "weight": ParameterSpec(
                    shape=(6, 4, 4, 8, 12),  # B, H, W, I, O
                    dtype=jnp.float32,
                ),
            }

    class CustomFanLayer(DefaultFanLayer):
        """A layer with FanAxes (no batched axis)."""

        def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> FanAxes:
            return FanAxes(in_axis=(1, 2), out_axis=3)

    class BatchedCustomFanLayer(DefaultFanLayer):
        """A layer with FanAxes (with batched axis)."""

        def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> FanAxes:
            return FanAxes(in_axis=(1, 2), out_axis=3, batch_axis=0)

    @parameterized.parameters(
        (
            DefaultFanLayer,
            FanAxes(in_axis=-2, out_axis=-1),
            {
                "fan_in": 6 * 4 * 4 * 8,
                "fan_out": 6 * 4 * 4 * 12,
                "fan_avg": 6 * 4 * 4 * 10,
            },
        ),
        (
            CustomFanLayer,
            FanAxes(in_axis=(1, 2), out_axis=3),
            {
                "fan_in": 6 * 12 * 4 * 4,
                "fan_out": 6 * 12 * 8,
                "fan_avg": 6 * 12 * 12,
            },
        ),
        (
            BatchedCustomFanLayer,
            FanAxes(in_axis=(1, 2), out_axis=3, batch_axis=0),
            {
                "fan_in": 12 * 4 * 4,
                "fan_out": 12 * 8,
                "fan_avg": 12 * 12,
            },
        ),
    )
    def test_fan(self, cls, fan_axes, fans):
        for dist in ("uniform", "normal", "truncated_normal"):
            for scale in (1.0, 2.0):
                for fan_type in ("fan_in", "fan_out", "fan_avg"):
                    cfg = cls.default_config().set(name="test")
                    cfg.param_init = DefaultInitializer.default_config().set(
                        init_by_param_name={
                            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                                fan=fan_type, scale=scale, distribution=dist
                            )
                        }
                    )
                    layer: BaseLayer = cfg.instantiate(parent=None)
                    # pylint: disable-next=protected-access
                    param_spec_map = layer._create_layer_parameter_specs()
                    self.assertEqual(
                        # pylint: disable-next=protected-access
                        layer._compute_fan_axes("weight", param_spec_map["weight"]),
                        fan_axes,
                    )
                    spec = dataclasses.replace(param_spec_map["weight"], fan_axes=fan_axes)
                    self.assertEqual(spec.fans(), fans)
                    layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(1))
                    weight = layer_params["weight"]
                    fan = fans[fan_type]
                    self.assertEqual(weight.dtype, jnp.float32)
                    expected_std = scale / math.sqrt(fan)
                    actual_std = np.std(weight)
                    self.assertBetween(actual_std, expected_std / 1.5, expected_std * 1.5)

    def test_fan_axes_in_create_parameter_specs_recursively(self):
        layer_cfg = self.BatchedCustomFanLayer.default_config().set(name="test")
        layer = layer_cfg.instantiate(parent=None)
        specs = layer.create_parameter_specs_recursively()
        self.assertEqual(
            specs["weight"].fan_axes, FanAxes(in_axis=(1, 2), out_axis=3, batch_axis=0)
        )


if __name__ == "__main__":
    absltest.main()

# Copyright © 2023 Apple Inc.

"""Pipeline layer tests."""

# pylint: disable=no-self-use,duplicate-code,protected-access
from typing import Optional, cast

import jax.random
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.base_layer import BaseLayer, ParameterSpec, RematSpec
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module, OutputCollection
from axlearn.common.module import functional as F
from axlearn.common.pipeline import (
    BaseSchedule,
    GPipeSchedule,
    Pipeline,
    StreamSchedule,
    _mask_invalid_gradients,
    _select_input_or_previous_outputs,
    transpose_from_pipeline_stage_outputs,
    transpose_to_pipeline_stage_inputs,
)
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import Nested, PartitionSpec, Tensor, VDict, cast_floats, shapes


class TransposeTest(absltest.TestCase):
    def test_transpose_functions(self):
        num_layers, num_microbatches = 3, 5
        layer_indices = jnp.tile(jnp.arange(num_layers)[:, None], (1, num_microbatches))
        microbatch_indices = jnp.tile(jnp.arange(num_microbatches)[None, :], (num_layers, 1))
        # [num_layers, num_microbatches, 2].
        inputs = jnp.stack([layer_indices, microbatch_indices], axis=-1)

        # Transpose to pipeline inputs.
        transposed = transpose_to_pipeline_stage_inputs(inputs)
        logging.info("transposed=%s", transposed)
        for i in range(num_layers):
            for j in range(num_microbatches):
                t = i + j
                assert_allclose(jnp.asarray([i, j]), transposed[t, i])

        # Transpose from pipeline outputs.
        outputs = transpose_from_pipeline_stage_outputs(transposed)
        assert_allclose(outputs, inputs)


class TestLayer(BaseLayer):
    """A dummy testing layer."""

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        return dict(
            inc=ParameterSpec(
                shape=[], mesh_axes=[], initializer=param_init.constant_initializer(1)
            )
        )

    def init_forward_state(self, batch_size):
        return jnp.zeros([batch_size])

    def forward(self, carry, forward_state):
        logging.info("TestLayer: carry=%s forward_state=%s", shapes(carry), shapes(forward_state))
        self.add_summary("carry_mean", jnp.mean(carry))
        self.add_state_update("inc", 1)
        self.add_module_output("out", 2)
        return carry + self.parameters["inc"], forward_state + carry


class TestComplicatedLayer(BaseLayer):
    """A dummy testing layer."""

    @config_class
    class Config(BaseLayer.Config):
        layer1: TestLayer.Config = TestLayer.default_config()
        layer2: TestLayer.Config = TestLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("layer1", cfg.layer1)
        self._add_child("layer2", cfg.layer2)

    def init_forward_state(self, batch_size):
        return jnp.zeros([batch_size], dtype=self.dtype())

    def forward(self, carry, forward_state):
        carry, forward_state = self.layer1(carry, forward_state)
        carry, forward_state = self.layer2(carry, forward_state)
        return carry, forward_state


# TODO(markblee): Rename dummy layers to avoid confusion with the actual test cases.
class TestPipeline(Pipeline):
    """A dummy pipeline layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.layer = TestLayer.default_config()
        return cfg

    def init_forward_state(self, batch_size):
        cfg = self.config  # type: Pipeline.Config
        microbatch_size = batch_size // cfg.num_microbatches
        layer_state = self.layer.init_forward_state(microbatch_size)
        return dict(
            layer=jax.tree.map(
                lambda x: jnp.tile(x[None, None, :], [cfg.num_layers, cfg.num_microbatches, 1]),
                layer_state,
            )
        )

    def forward(self, carry, forward_state):
        carry = self._to_microbatches(carry)

        def fn(carry, forward_state_tn):
            return self.layer(carry, forward_state_tn["layer"])

        carry, forward_state = self._run(fn, carry, xs=forward_state)
        carry = self._from_microbatches(carry)
        return carry, dict(layer=forward_state)


class DummyMLP(BaseLayer):
    """A dummy MLP layer."""

    @config_class
    class Config(BaseLayer.Config):
        input_dim: Required[int] = REQUIRED
        hidden_dim: Required[int] = REQUIRED

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg: DummyMLP.Config = self.config
        return dict(
            linear1=ParameterSpec(
                shape=[cfg.input_dim, cfg.hidden_dim],
                initializer=param_init.gaussian_initializer(1),
            ),
            linear2=ParameterSpec(
                shape=[cfg.hidden_dim, cfg.input_dim],
                initializer=param_init.gaussian_initializer(1),
            ),
        )

    def init_forward_state(self, batch_size):
        return jnp.zeros([batch_size])

    def forward(self, x: Tensor, forward_state):
        x = jnp.einsum("bd,dh->bh", x, self.parameters["linear1"])
        x = jax.nn.relu(x)
        x = jnp.einsum("bh,hd->bd", x, self.parameters["linear2"])
        self.add_summary("carry_mean", jnp.mean(x))
        return x, forward_state


class PipelineTest(TestCase):
    @parameterized.product(
        remat_spec=[None, RematSpec(prevent_cse=False)],
        dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_pipeline(self, remat_spec: Optional[RematSpec], dtype: jnp.dtype):
        batch_size, microbatch_size, num_layers = 14, 2, 4
        num_microbatches = batch_size // microbatch_size
        layer: TestPipeline = (
            TestPipeline.default_config()
            .set(
                name="test",
                num_layers=num_layers,
                num_microbatches=num_microbatches,
                remat_spec=remat_spec,
                vlog=3,
            )
            .instantiate(parent=None)
        )
        self.assertEqual(
            PartitionSpec("pipeline"),
            layer.create_parameter_specs_recursively()["layer"]["inc"].mesh_axes,
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
        logging.info("layer params=%s", layer_params)

        input_forward_state = layer.init_forward_state(batch_size)
        inputs = (jnp.arange(batch_size, dtype=jnp.float32), input_forward_state)
        (carry, output_forward_state), output_collection = F(
            layer,
            prng_key=jax.random.PRNGKey(2),
            state=cast_floats(layer_params, to_dtype=dtype),
            inputs=cast_floats(inputs, to_dtype=dtype),
            is_training=True,
            drop_output_collections=(),
        )
        logging.info("forward_state=%s", output_forward_state)
        logging.info("state_outputs=%s", shapes(output_collection.state_updates))
        logging.info("module_outputs=%s", shapes(output_collection.module_outputs))

        # Ensure carry dtype matches input.
        jax.tree.map(lambda x: self.assertEqual(x.dtype, dtype), carry)

        assert_allclose(carry, jnp.arange(num_layers, num_layers + batch_size))
        self.assertEqual(shapes(input_forward_state), shapes(output_forward_state))
        assert_allclose(
            output_forward_state["layer"],
            jnp.reshape(
                jnp.arange(batch_size)[None, :] + jnp.arange(num_layers)[:, None],
                (num_layers, microbatch_size, num_microbatches),
            ).transpose([0, 2, 1]),
        )
        self.assertEqual(
            OutputCollection(
                summaries={
                    "layer": {},
                    **{
                        f"layer{i}": {
                            f"microbatch{j}": {"carry_mean": tuple()}
                            for j in range(num_microbatches)
                        }
                        for i in range(num_layers)
                    },
                },
                state_updates={
                    "layer": {"inc": (num_layers, num_microbatches)},
                    **{
                        f"layer{i}": {f"microbatch{j}": {} for j in range(num_microbatches)}
                        for i in range(num_layers)
                    },
                },
                module_outputs={
                    "layer": {"out": (num_layers, num_microbatches)},
                    **{
                        f"layer{i}": {f"microbatch{j}": {} for j in range(num_microbatches)}
                        for i in range(num_layers)
                    },
                },
            ),
            shapes(output_collection),
        )
        assert_allclose(
            [[3.5 + i + j for j in range(num_microbatches)] for i in range(num_layers)],
            [
                [
                    output_collection.summaries[f"layer{i}"][f"microbatch{j}"]["carry_mean"]
                    for j in range(num_microbatches)
                ]
                for i in range(num_layers)
            ],
        )

    @parameterized.parameters(None, RematSpec(prevent_cse=False))
    def test_pipeline_prebuilt(self, remat_spec):
        for multiple_values in range(3):
            batch_size, microbatch_size, num_layers = 14, 2, 4
            num_microbatches = batch_size // microbatch_size
            layer: TestPipeline = (
                TestPipeline.default_config()
                .set(
                    name="test",
                    layer=TestComplicatedLayer.default_config(),
                    num_layers=num_layers,
                    num_microbatches=num_microbatches,
                    remat_spec=remat_spec,
                )
                .instantiate(parent=None)
            )
            self.assertEqual(
                PartitionSpec("pipeline"),
                layer.create_parameter_specs_recursively()["layer"]["layer1"]["inc"].mesh_axes,
            )
            self.assertEqual(
                PartitionSpec("pipeline"),
                layer.create_parameter_specs_recursively()["layer"]["layer2"]["inc"].mesh_axes,
            )
            prebuilt = VDict(
                {
                    "layer": {
                        "layer1": {"inc": jnp.zeros(4, dtype=jnp.float32)},
                        "layer2": {"inc": jnp.ones(4, dtype=jnp.float32) * multiple_values},
                    }
                }
            )
            layer_params = layer.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(1), prebuilt=prebuilt
            )
            layer_params["layer"] = prebuilt["layer"]
            logging.info("layer params=%s", layer_params)

            input_forward_state = layer.init_forward_state(batch_size)
            (carry, output_forward_state), output_collection = F(
                layer,
                prng_key=jax.random.PRNGKey(2),
                state=layer_params,
                inputs=(jnp.arange(batch_size, dtype=jnp.float32), input_forward_state),
                is_training=True,
            )
            logging.info("forward_state=%s", output_forward_state)
            logging.info("output_collection=%s", output_collection)
            assert_allclose(
                carry,
                jnp.arange(num_layers * multiple_values, num_layers * multiple_values + batch_size),
            )
            self.assertEqual(shapes(input_forward_state), shapes(output_forward_state))
            assert_allclose(
                output_forward_state["layer"],
                jnp.reshape(
                    (
                        jnp.arange(batch_size)[None, :]
                        + jnp.arange(num_layers)[:, None] * multiple_values
                    )
                    * 2,
                    (num_layers, microbatch_size, num_microbatches),
                ).transpose([0, 2, 1]),
            )
            self.assertEqual(
                {
                    "layer": {},
                    **{
                        f"layer{i}": {
                            f"microbatch{j}": {
                                "layer1": {"carry_mean": tuple()},
                                "layer2": {"carry_mean": tuple()},
                            }
                            for j in range(num_microbatches)
                        }
                        for i in range(num_layers)
                    },
                },
                shapes(output_collection.summaries),
            )
            assert_allclose(
                [
                    [3.5 + i * multiple_values + j for j in range(num_microbatches)]
                    for i in range(num_layers)
                ],
                [
                    [
                        output_collection.summaries[f"layer{i}"][f"microbatch{j}"]["layer1"][
                            "carry_mean"
                        ]
                        for j in range(num_microbatches)
                    ]
                    for i in range(num_layers)
                ],
            )

    @parameterized.product(
        schedule=[
            dict(cls=GPipeSchedule, batch_size=14),
            # StreamSchedule requires num stages to divide num microbatches.
            dict(cls=StreamSchedule, batch_size=16),
        ],
        remat_spec=[None, RematSpec(prevent_cse=False)],
    )
    def test_pipeline_gradients(self, schedule: dict, remat_spec: Optional[RematSpec]):
        """Test gradients against a ref implementation."""
        schedule_cls = schedule["cls"]
        batch_size, microbatch_size, num_stages, input_dim = schedule["batch_size"], 2, 4, 8
        num_microbatches = batch_size // microbatch_size

        class DummyScheduleWithNaNs(schedule_cls):
            """Wraps carry input by filling bubbles with NaNs."""

            def _compute_carry_input(
                self,
                per_stage_inputs: Nested[Tensor],
                carry_output_t_1: Nested[Tensor],
                *,
                t: Tensor,
            ) -> Tensor:
                x = super()._compute_carry_input(per_stage_inputs, carry_output_t_1, t=t)

                def inject_nans(x):
                    return jnp.where(self._is_valid_stage(x, t=t), x, jnp.nan)

                return jax.tree.map(inject_nans, x)

        layer: TestPipeline = (
            TestPipeline.default_config()
            .set(
                name="test",
                num_layers=num_stages,
                num_microbatches=num_microbatches,
                remat_spec=remat_spec,
                layer=DummyMLP.default_config().set(input_dim=input_dim, hidden_dim=input_dim * 2),
                schedule=DummyScheduleWithNaNs.default_config(),
                vlog=3,
            )
            .instantiate(parent=None)
        )

        def test_fn(layer_params, data, prng_key):
            layer_outputs, output_collection = F(
                layer,
                inputs=data,
                state=layer_params,
                is_training=True,
                prng_key=prng_key,
            )
            # Ignore forward state.
            data, _ = layer_outputs
            return jnp.sum(data**2), output_collection

        def ref_fn(layer_params, data):
            linear1 = layer_params["layer"]["linear1"]
            linear2 = layer_params["layer"]["linear2"]
            for i in range(num_stages):
                data = jnp.einsum("bd,dh->bh", data, linear1[i])
                data = jax.nn.relu(data)
                data = jnp.einsum("bh,hd->bd", data, linear2[i])
            return jnp.sum(data**2)

        layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(1))
        dummy_forward_state = layer.init_forward_state(batch_size)
        inputs = jax.random.uniform(
            jax.random.PRNGKey(2), [batch_size, input_dim], dtype=jnp.float32
        )
        (test_out, output_collection), test_grads = jax.value_and_grad(test_fn, has_aux=True)(
            layer_params, (inputs, dummy_forward_state), jax.random.PRNGKey(3)
        )
        ref_out, ref_grads = jax.value_and_grad(ref_fn)(layer_params, inputs)

        self.assertNestedAllClose(test_out, ref_out)
        self.assertNestedAllClose(test_grads, ref_grads)
        jax.tree.map(lambda x: self.assertFalse(jnp.isnan(x).any().item()), output_collection)

    @parameterized.product(
        schedule=[
            # StreamSchedule requires num stages to divide num microbatches.
            dict(cls=StreamSchedule, batch_size=16),
        ],
        remat_spec=[None, RematSpec(prevent_cse=False)],
    )
    def test_schedule_equivalence(self, schedule: dict, remat_spec: Optional[RematSpec]):
        """Tests equivalence with GPipeSchedule."""
        schedule_cls = schedule["cls"]
        batch_size, microbatch_size, num_layers, input_dim = schedule["batch_size"], 2, 4, 8
        num_microbatches = batch_size // microbatch_size
        ref_cfg: TestPipeline.Config = TestPipeline.default_config().set(
            name="test",
            num_layers=num_layers,
            num_microbatches=num_microbatches,
            remat_spec=remat_spec,
            layer=DummyMLP.default_config().set(input_dim=input_dim, hidden_dim=input_dim * 2),
        )
        test_cfg = ref_cfg.clone(
            schedule=cast(type[BaseSchedule], schedule_cls).default_config(),
        )

        def compute_outputs(layer: TestPipeline, inputs):
            params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
            forward_state = layer.init_forward_state(batch_size)
            return F(
                layer,
                inputs=(inputs, forward_state),
                state=params,
                is_training=True,
                prng_key=jax.random.PRNGKey(2),
            )

        ref_layer: TestPipeline = ref_cfg.instantiate(parent=None)
        test_layer: TestPipeline = test_cfg.instantiate(parent=None)

        inputs = jax.random.uniform(
            jax.random.PRNGKey(2), [batch_size, input_dim], dtype=jnp.float32
        )
        ref_outputs = compute_outputs(ref_layer, inputs)
        test_outputs = compute_outputs(test_layer, inputs)

        self.assertNestedAllClose(ref_outputs, test_outputs)


class GPipeScheduleTest(TestCase):
    """Tests GPipeSchedule."""

    def test_is_valid_stage(self):
        schedule = (
            GPipeSchedule.default_config()
            .set(
                num_stages=3,
                num_microbatches=5,
            )
            .instantiate()
        )
        self.assertEqual(7, schedule.num_iterations)
        expected = jnp.array(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, True, True],
                [False, False, True],
            ]
        )
        self.assertTrue(
            jnp.all(
                expected == jax.vmap(schedule._is_valid_stage)(jnp.arange(schedule.num_iterations))
            )
        )


class StreamScheduleTest(TestCase):
    """Tests StreamSchedule."""

    def test_is_valid_stage(self):
        # Note that num_stages must divide num_microbatches.
        schedule = (
            StreamSchedule.default_config()
            .set(
                num_stages=3,
                num_microbatches=6,
            )
            .instantiate()
        )
        self.assertEqual(8, schedule.num_iterations)
        expected = jnp.array(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, True, True],
                [False, False, True],
            ]
        )
        self.assertTrue(
            jnp.all(
                expected == jax.vmap(schedule._is_valid_stage)(jnp.arange(schedule.num_iterations))
            )
        )


class UtilsTest(TestCase):
    """Tests utilities."""

    def test_select_input_or_previous_outputs(self):
        shape = [3, 4]
        x = jax.random.uniform(jax.random.PRNGKey(1), shape=shape)
        y = jax.random.uniform(jax.random.PRNGKey(2), shape=shape)
        z = _select_input_or_previous_outputs(x, y)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(y.shape, z.shape)
        self.assertNestedAllClose(z[0], x[0])
        self.assertNestedAllClose(z[1:], y[:-1])

    def test_mask_invalid_gradients(self):
        state = {"x": jnp.ones([2, 2]), "y": jnp.ones([2, 3])}
        is_valid = jnp.array([True, False])

        def fn(state):
            state = _mask_invalid_gradients(state, is_valid=is_valid)
            return jnp.sum(state["x"]) * 2 + jnp.sum(state["y"]) * 3

        expected = {
            "x": jnp.array([[2.0, 2.0], [0.0, 0.0]]),
            "y": jnp.array([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]]),
        }
        self.assertNestedAllClose(expected, jax.grad(fn)(state))


if __name__ == "__main__":
    absltest.main()

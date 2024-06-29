# Copyright © 2023 Apple Inc.

"""Pipeline layer tests."""

# pylint: disable=no-self-use,duplicate-code
from typing import Dict, Optional

import jax.random
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.base_layer import BaseLayer, ParameterSpec, RematSpec
from axlearn.common.config import config_class
from axlearn.common.module import Module, OutputCollection
from axlearn.common.module import functional as F
from axlearn.common.pipeline import (
    Pipeline,
    transpose_from_pipeline_stage_outputs,
    transpose_to_pipeline_stage_inputs,
)
from axlearn.common.test_utils import assert_allclose
from axlearn.common.utils import PartitionSpec, VDict, shapes


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

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
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
            layer=jax.tree_util.tree_map(
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


class PipelineTest(parameterized.TestCase):
    @parameterized.parameters(None, RematSpec(prevent_cse=False))
    def test_pipeline(self, remat_spec):
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
        (carry, output_forward_state), output_collection = F(
            layer,
            prng_key=jax.random.PRNGKey(2),
            state=layer_params,
            inputs=(jnp.arange(batch_size, dtype=jnp.float32), input_forward_state),
            is_training=True,
            drop_output_collections=(),
        )
        logging.info("forward_state=%s", output_forward_state)
        logging.info("state_outputs=%s", shapes(output_collection.state_updates))
        logging.info("module_outputs=%s", shapes(output_collection.module_outputs))
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


if __name__ == "__main__":
    absltest.main()

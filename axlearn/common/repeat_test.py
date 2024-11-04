# Copyright © 2023 Apple Inc.

"""Tests repeat layer."""

import itertools
from typing import Optional

import jax.random
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import param_init
from axlearn.common.base_layer import BaseLayer, ParameterSpec, RematSpec
from axlearn.common.config import REQUIRED, Required, config_class, config_for_function
from axlearn.common.layers import RedirectToSharedModule
from axlearn.common.layers_test import ParentLayer
from axlearn.common.module import Module, OutputCollection, child_context
from axlearn.common.module import functional as F
from axlearn.common.repeat import Repeat, _drop_by_regex
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    PartitionSpec,
    Tensor,
    VDict,
    get_recursively,
    shapes,
)


class TestLayer(BaseLayer):
    """A dummy layer."""

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        return dict(
            inc=ParameterSpec(
                shape=[], mesh_axes=[], initializer=param_init.constant_initializer(1)
            )
        )

    def init_forward_state(self, batch_size):
        return jnp.zeros([batch_size], dtype=self.dtype())

    def forward(self, *, carry, forward_state):
        logging.info("TestLayer: carry=%s forward_state=%s", shapes(carry), shapes(forward_state))
        forward_state = forward_state + carry
        self.add_summary("carry_mean", jnp.mean(carry))
        self.add_module_output("state", forward_state)
        self.add_state_update("inc", 2)
        return carry + self.parameters["inc"], forward_state


class TestComplicatedLayer(BaseLayer):
    """A dummy layer with children."""

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

    def forward(self, *, carry, forward_state):
        carry, forward_state = self.layer1(carry=carry, forward_state=forward_state)
        carry, forward_state = self.layer2(carry=carry, forward_state=forward_state)
        return carry, forward_state


class TestRepeat(Repeat):
    """A dummy repeat layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.layer = TestLayer.default_config()
        return cfg

    def init_forward_state(self, batch_size):
        cfg = self.config
        layer_state = self.layer.init_forward_state(batch_size)
        return dict(
            layer=jax.tree.map(
                lambda x: jnp.tile(x[None, :], [cfg.num_layers, 1]),
                layer_state,
            )
        )

    def initialize_parameters_recursively(
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
    ) -> NestedTensor:
        params = super().initialize_parameters_recursively(prng_key=prng_key, prebuilt=prebuilt)
        params["dummy"] = jnp.ones(1)
        return params

    def forward(self, *, carry, forward_state):
        def fn(carry, forward_state_tn):
            return self.layer(carry=carry, forward_state=forward_state_tn["layer"])

        carry, forward_state = self._run(fn, carry, xs=forward_state)
        return carry, dict(layer=forward_state)


class TestEnsemble(BaseLayer):
    """A dummy ensemble layer."""

    @config_class
    class Config(BaseLayer.Config):
        num_layers: Required[int] = REQUIRED
        dummy_layer: TestLayer.Config = TestLayer.default_config()
        repeat_layer: TestRepeat.Config = TestRepeat.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("dummy_layer", cfg.dummy_layer)
        self._share_with_descendants(
            self.dummy_layer,
            shared_module_name="shared_layer",
        )
        self._add_child("repeat_layer", cfg.repeat_layer.set(num_layers=cfg.num_layers))

    def init_forward_state(self, batch_size):
        layer_state = self.repeat_layer.init_forward_state(batch_size)
        return dict(repeat_layer=layer_state)

    def forward(self, *, carry, forward_state):
        carry, forward_state = self.repeat_layer(
            carry=carry, forward_state=forward_state["repeat_layer"]
        )
        return carry, dict(repeat_layer=forward_state)

    def forward_first_n(self, *, n, carry, forward_state):
        substate = _get_first_n(n, self.state["repeat_layer"])
        with child_context("repeat_layer", state=substate):
            carry, forward_state = self.repeat_layer(
                carry=carry, forward_state=forward_state["repeat_layer"]
            )
            return carry, dict(repeat_layer=forward_state)


class RepeatTest(TestCase):
    """Tests repeat layer."""

    @parameterized.product(
        dtype=(jnp.float32, jnp.bfloat16),
        remat_spec=(None, RematSpec(prevent_cse=False)),
        drop_output=(None, config_for_function(_drop_by_regex).set(rules=["module_outputs.*"])),
        num_layers_total=(4, 6),
    )
    def test_repeat(self, dtype, remat_spec, drop_output, num_layers_total):
        batch_size, num_layers = 14, 4
        cfg = TestEnsemble.default_config().set(
            name="test", num_layers=num_layers_total, dtype=dtype
        )
        cfg.repeat_layer.set(remat_spec=remat_spec, drop_output=drop_output)
        layer: TestEnsemble = cfg.instantiate(parent=None)
        self.assertEqual(
            PartitionSpec(None),
            layer.create_parameter_specs_recursively()["repeat_layer"]["layer"]["inc"].mesh_axes,
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
        logging.info("layer params=%s", layer_params)

        input_forward_state = layer.init_forward_state(batch_size)
        if num_layers_total == num_layers:
            method = "forward"
            inputs = dict(
                carry=jnp.arange(batch_size, dtype=dtype),
                forward_state=input_forward_state,
            )
        else:
            method = "forward_first_n"
            input_forward_state = _get_first_n(num_layers, input_forward_state)
            inputs = dict(
                carry=jnp.arange(batch_size, dtype=dtype),
                forward_state=input_forward_state,
                n=num_layers,
            )

        (carry, output_forward_state), output_collection = F(
            layer,
            prng_key=jax.random.PRNGKey(2),
            state=layer_params,
            inputs=inputs,
            method=method,
            is_training=True,
            drop_output_collections=(),
        )
        logging.info("forward_state=%s", output_forward_state)
        logging.info("output_collection=%s", output_collection)
        assert_allclose(carry, jnp.arange(num_layers, num_layers + batch_size, dtype=dtype))
        self.assertEqual(shapes(input_forward_state), shapes(output_forward_state))
        assert_allclose(
            output_forward_state["repeat_layer"]["layer"],
            jnp.reshape(
                jnp.arange(batch_size)[None, :] + jnp.arange(num_layers, dtype=dtype)[:, None],
                (num_layers, batch_size),
            ),
        )
        # Check output collection.
        self.assertEqual(
            OutputCollection(
                state_updates={
                    "repeat_layer": {
                        # State update values are stacked across layers.
                        "layer": {"inc": (num_layers,)},
                        **{f"layer{i}": {} for i in range(num_layers)},
                    },
                },
                module_outputs={
                    "repeat_layer": {
                        # Module output values are stacked across layers.
                        "layer": (
                            {"state": (num_layers, batch_size)} if drop_output is None else {}
                        ),
                        **{f"layer{i}": {} for i in range(num_layers)},
                    },
                },
                summaries={
                    "repeat_layer": {
                        "layer": {},
                        # Summary values are unstacked and placed in separate "layer{i}" scopes.
                        **{f"layer{i}": {"carry_mean": tuple()} for i in range(num_layers)},
                    }
                },
            ),
            shapes(output_collection),
        )
        if drop_output is not None:
            self.assertEqual(
                get_recursively(output_collection.module_outputs, "repeat_layer/layer"),
                {},
            )
        else:
            assert_allclose(
                get_recursively(output_forward_state, "repeat_layer/layer"),
                get_recursively(output_collection.module_outputs, "repeat_layer/layer/state"),
            )
        assert_allclose(
            [2] * num_layers,
            get_recursively(output_collection.state_updates, "repeat_layer/layer/inc"),
        )
        # Check summaries.
        self.assertEqual(
            {
                "repeat_layer": {
                    "layer": {},
                    **{f"layer{i}": {"carry_mean": tuple()} for i in range(num_layers)},
                }
            },
            shapes(output_collection.summaries),
        )
        assert_allclose(
            0.5 * (batch_size - 1) + jnp.arange(num_layers, dtype=dtype),
            [
                output_collection.summaries["repeat_layer"][f"layer{i}"]["carry_mean"]
                for i in range(num_layers)
            ],
        )
        if remat_spec is None:
            # pylint: disable-next=protected-access
            self.assertEmpty(layer.repeat_layer._remat_methods)
        else:
            # pylint: disable-next=protected-access
            self.assertSequenceEqual(layer.repeat_layer._remat_methods, ["forward"])

    @parameterized.parameters(
        itertools.product((jnp.float32, jnp.bfloat16), (None, RematSpec(prevent_cse=False)))
    )
    def test_repeat_prebuilt_forward(self, dtype, remat_spec):
        # Testing using modified parameters for feed-forwarding.
        for multiple_values in range(2):
            batch_size, num_layers = 14, 4
            cfg = TestEnsemble.default_config().set(name="test", num_layers=num_layers, dtype=dtype)
            cfg.repeat_layer.remat_spec = remat_spec
            cfg.repeat_layer.layer = TestComplicatedLayer.default_config()
            layer: TestEnsemble = cfg.instantiate(parent=None)
            repeat_layer_prebuilt = VDict(
                {
                    "layer": {
                        "layer1": {"inc": jnp.zeros(4, dtype=dtype)},
                        "layer2": {"inc": jnp.ones(4, dtype=dtype) * multiple_values},
                    }
                }
            )
            prebuilt = {"dummy_layer": {"inc": None}, "repeat_layer": repeat_layer_prebuilt}
            layer_params_repeated_prebuilt = layer.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(1), prebuilt=prebuilt
            )
            layer_params_repeated_prebuilt["repeat_layer"] = repeat_layer_prebuilt
            input_forward_state = layer.init_forward_state(batch_size)
            (carry, output_forward_state), output_collection = F(
                layer,
                prng_key=jax.random.PRNGKey(2),
                state=layer_params_repeated_prebuilt,
                inputs=dict(
                    carry=jnp.arange(batch_size, dtype=dtype),
                    forward_state=input_forward_state,
                ),
                is_training=True,
            )
            assert_allclose(
                carry,
                jnp.arange(
                    num_layers * multiple_values,
                    num_layers * multiple_values + batch_size,
                    dtype=dtype,
                ),
            )
            assert_allclose(
                output_forward_state["repeat_layer"]["layer"],
                jnp.reshape(
                    (
                        jnp.arange(batch_size)[None, :]
                        + jnp.arange(num_layers, dtype=dtype)[:, None] * multiple_values
                    )
                    * 2,
                    (num_layers, batch_size),
                ),
            )
            assert_allclose(
                0.5 * (batch_size - 1) + jnp.arange(num_layers, dtype=dtype) * multiple_values,
                [
                    output_collection.summaries["repeat_layer"][f"layer{i}"]["layer1"]["carry_mean"]
                    for i in range(num_layers)
                ],
            )
            if remat_spec is None:
                # pylint: disable-next=protected-access
                self.assertEmpty(layer.repeat_layer._remat_methods)
            else:
                # pylint: disable-next=protected-access
                self.assertSequenceEqual(layer.repeat_layer._remat_methods, ["forward"])

    @parameterized.product(
        dtype=[jnp.float32, jnp.bfloat16],
        remat_spec=[None, RematSpec(prevent_cse=False)],
    )
    def test_shared_module(self, dtype, remat_spec):
        """Test repeat with shared modules."""
        batch_size, num_layers = 14, 4

        cfg = ParentLayer.default_config().set(
            shared_modules=["shared_layer"],
            children=dict(
                shared_layer=TestLayer.default_config().set(remat_spec=remat_spec, dtype=dtype),
                # Repeat to a shared module.
                repeat=TestRepeat.default_config().set(
                    layer=RedirectToSharedModule.default_config().set(
                        shared_module="shared_layer",
                        remat_spec=remat_spec,
                    ),
                    num_layers=num_layers,
                ),
                # Test nested repeat to a shared module.
                nested=ParentLayer.default_config().set(
                    children=dict(
                        repeat=TestRepeat.default_config().set(
                            layer=RedirectToSharedModule.default_config().set(
                                shared_module="shared_layer",
                                remat_spec=remat_spec,
                            ),
                            num_layers=num_layers,
                        ),
                    ),
                ),
            ),
        )
        layer = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
        logging.info("layer params=%s", layer_params)

        input_forward_state = layer.shared_layer.init_forward_state(batch_size)
        input_forward_state = dict(
            layer=jax.tree.map(
                lambda x: jnp.tile(x[None, :], [num_layers, 1]),
                input_forward_state,
            )
        )

        for forward_path, output_path, remat_methods in [
            ("repeat", "repeat/layer{i}/shared_layer/carry_mean", ["forward"]),
            (
                "nested/repeat",
                "nested/repeat/layer{i}/shared_layer/carry_mean",
                ["forward", "forward"],
            ),
        ]:
            (carry, output_forward_state), output_collection = F(
                layer,
                prng_key=jax.random.PRNGKey(2),
                state=layer_params,
                inputs=dict(
                    carry=jnp.arange(batch_size, dtype=dtype),
                    forward_state=input_forward_state,
                    path=tuple(forward_path.split("/")),
                ),
                is_training=True,
            )
            assert_allclose(carry, jnp.arange(num_layers, num_layers + batch_size, dtype=dtype))
            self.assertEqual(shapes(input_forward_state), shapes(output_forward_state))
            assert_allclose(
                output_forward_state["layer"],
                jnp.reshape(
                    jnp.arange(batch_size)[None, :] + jnp.arange(num_layers, dtype=dtype)[:, None],
                    (num_layers, batch_size),
                ),
            )

            assert_allclose(
                0.5 * (batch_size - 1) + jnp.arange(num_layers, dtype=dtype),
                [
                    get_recursively(output_collection.summaries, output_path.format(i=i))
                    for i in range(num_layers)
                ],
            )

            # Test remat spec.
            if remat_spec is None:
                # pylint: disable-next=protected-access
                self.assertEmpty(layer.shared_layer._remat_methods)
            else:
                # pylint: disable-next=protected-access
                self.assertSequenceEqual(layer.shared_layer._remat_methods, remat_methods)


def _get_first_n(n, tree):
    return jax.tree.map(lambda x: x[:n], tree)


if __name__ == "__main__":
    absltest.main()

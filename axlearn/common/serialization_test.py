# Copyright © 2024 Apple Inc.
#
# The code in this file is adapted from:
#
# google/flax:
# Copyright 2024 The Flax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for serialization utils."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest, parameterized
from jax.tree_util import Partial

from axlearn.common import flax_struct, serialization


@flax_struct.dataclass
class _Point:
    x: float
    y: float
    meta: Any = flax_struct.field(pytree_node=False)


@flax_struct.dataclass
class _Box:
    value: int


def _to_state_dict(box: _Box):
    return {"value": box.value}


def _from_state_dict(box: _Box, state: Any):
    return box.replace(value=state["value"])


serialization.register_serialization_state(_Box, _to_state_dict, _from_state_dict, override=True)


class SerializationTest(parameterized.TestCase):
    def test_dataclass_serialization(self):
        p = _Point(x=1, y=2, meta={"dummy": True})
        state_dict = serialization.to_state_dict(p)
        self.assertEqual(state_dict, {"x": 1, "y": 2})
        restored_p = serialization.from_state_dict(p, {"x": 3, "y": 4})
        expected_p = _Point(x=3, y=4, meta={"dummy": True})
        self.assertEqual(restored_p, expected_p)

        with self.assertRaises(ValueError):  # Invalid field.
            serialization.from_state_dict(p, {"z": 3})
        with self.assertRaises(ValueError):  # Missing field.
            serialization.from_state_dict(p, {"x": 3})

    def test_pass_through_serialization(self):
        p = _Box(value=123)
        state_dict = serialization.to_state_dict(p)
        self.assertEqual(state_dict, {"value": 123})
        restored_box = serialization.from_state_dict(p, state_dict)
        expected_box = _Box(value=123)
        self.assertEqual(restored_box, expected_box)

    def test_model_serialization(self):
        initial_params = {
            "params": {
                "kernel": jnp.array([[1.0]], dtype=jnp.float32),
                "bias": jnp.array([0.0], dtype=jnp.float32),
            }
        }
        state = serialization.to_state_dict(initial_params)
        self.assertEqual(state, {"params": {"kernel": np.ones((1, 1)), "bias": np.zeros((1,))}})
        state = {"params": {"kernel": np.zeros((1, 1)), "bias": np.zeros((1,))}}
        restored_model = serialization.from_state_dict(initial_params, state)
        self.assertEqual(restored_model, state)

    def test_partial_serialization(self):
        add_one = Partial(jnp.add, 1)
        state = serialization.to_state_dict(add_one)
        self.assertEqual(state, {"args": {"0": 1}, "keywords": {}})
        restored_add_one = serialization.from_state_dict(add_one, state)
        self.assertEqual(add_one.args, restored_add_one.args)

    def test_optimizer_serialization(self):
        initial_params = {
            "params": {
                "kernel": jnp.array([[1.0]], dtype=jnp.float32),
                "bias": jnp.array([0.0], dtype=jnp.float32),
            }
        }
        tx = optax.sgd(0.1, momentum=0.1)
        tx_state = tx.init(initial_params)
        state = serialization.to_state_dict(tx_state)
        expected_state = {
            "0": {
                "trace": {
                    "params": {
                        "bias": np.array([0.0], dtype=jnp.float32),
                        "kernel": np.array([[0.0]], dtype=jnp.float32),
                    }
                }
            },
            "1": {},
        }
        self.assertEqual(state, expected_state)
        state = jax.tree.map(lambda x: x + 1, expected_state)
        restored_tx_state = serialization.from_state_dict(tx_state, state)
        tx_state_plus1 = jax.tree.map(lambda x: x + 1, tx_state)
        self.assertEqual(restored_tx_state, tx_state_plus1)

    def test_collection_serialization(self):
        @flax_struct.dataclass
        class DummyDataClass:
            x: float

            @classmethod
            def initializer(cls, shape):
                del shape
                return cls(x=0.0)  # pytype: disable=wrong-keyword-args

        variables = {"state": {"dummy": DummyDataClass(x=2.0)}}
        serialized_state_dict = serialization.to_state_dict(variables)
        self.assertEqual(serialized_state_dict, {"state": {"dummy": {"x": 2.0}}})
        deserialized_state = serialization.from_state_dict(variables, serialized_state_dict)
        self.assertEqual(variables, deserialized_state)


if __name__ == "__main__":
    absltest.main()

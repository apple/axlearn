# Copyright © 2024 Apple Inc.
#
# The code in this file is adapted from:
#
# google/flax:
# Copyright 2024 The Flax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for struct utils."""

import dataclasses
import functools
import sys
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest
from jax._src.tree_util import prefix_errors

from axlearn.common import flax_struct


@flax_struct.dataclass
class _Point:
    x: float
    y: float
    meta: Any = flax_struct.field(pytree_node=False)


class StructTest(absltest.TestCase):
    def test_no_extra_fields(self):
        p = _Point(x=1, y=2, meta={})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.new_field = 1

    def test_mutation(self):
        p = _Point(x=1, y=2, meta={})
        new_p = p.replace(x=3)
        self.assertEqual(new_p, _Point(x=3, y=2, meta={}))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.y = 3

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="Slots require Python >= 3.10")
    def test_slots(self):
        @functools.partial(flax_struct.dataclass, frozen=False, slots=True)
        class SlotsPoint:
            x: float
            y: float

        p = SlotsPoint(x=1.0, y=2.0)
        p.x = 3.0  # Can assign to existing fields.
        self.assertEqual(p, SlotsPoint(x=3.0, y=2.0))
        with self.assertRaises(AttributeError):
            p.z = 0.0  # Can't create new fields by accident.

    def test_pytree_nodes(self):
        p = _Point(x=1, y=2, meta={"abc": True})
        leaves = jax.tree_util.tree_leaves(p)
        self.assertEqual(leaves, [1, 2])
        new_p = jax.tree.map(lambda x: x + x, p)
        self.assertEqual(new_p, _Point(x=2, y=4, meta={"abc": True}))

    def test_keypath_error(self):
        (e,) = prefix_errors(_Point(1.0, [2.0], meta={}), _Point(1.0, 2.0, meta={}))
        with self.assertRaisesRegex(ValueError, r"in_axes\.y"):
            raise e("in_axes")

    def test_double_wrap_no_op(self):
        class Dummy:
            a: int

        self.assertFalse(hasattr(Dummy, "_axlearn_dataclass"))

        # pylint: disable-next=invalid-name
        Dummy = flax_struct.dataclass(Dummy)
        self.assertTrue(hasattr(Dummy, "_axlearn_dataclass"))

        # pylint: disable-next=invalid-name
        Dummy = flax_struct.dataclass(Dummy)  # no-op
        self.assertTrue(hasattr(Dummy, "_axlearn_dataclass"))

    def test_wrap_pytree_node_no_error(self):
        @flax_struct.dataclass
        class Dummy(flax_struct.PyTreeNode):
            a: int

        del Dummy

    def test_chex_tree_leaves_compatibility(self):
        """Tests that the treedef of a chex.dataclass is the same as a flax_struct.PyTreeNode.

        This is needed to ensure backwards compatibility for serialization / deserialization.
        """
        flattened = []
        for cls in Chex, Struct:
            instance = cls(
                field_d=jnp.array(0),
                field_b=jnp.array(1),
                field_a=jnp.array(2),
                field_c=jnp.array(3),
            )
            # tree_flatten_with_path is not preserved because Chex does not support this so the
            # fallback jax implementation with numbered keys gets used.
            flattened.append(jax.tree_util.tree_leaves(instance))
        chex.assert_trees_all_equal(*flattened)

    def test_constructor_order(self):
        """Tests that the constructor called using positional arguments uses the same order
        the fields were declared in.
        """
        expected = Struct(
            field_b=jnp.array(5),
            field_a=jnp.array(6),
            field_d=jnp.array(7),
            field_c=jnp.array(8),
        )
        actual = Struct(
            jnp.array(5),
            jnp.array(6),
            jnp.array(7),
            jnp.array(8),
        )
        self.assertEqual(actual, expected)

    def test_flatten_order(self):
        """Tests the `flatten_order` argument of `flax_struct.dataclass`."""

        @functools.partial(flax_struct.dataclass, flatten_order=None)
        class C:
            field_b: int
            field_a: int

        result = jax.tree_util.tree_leaves(C(field_b=1, field_a=2))
        expected = (1, 2)
        self.assertSequenceEqual(result, expected)


@chex.dataclass
class Chex:
    field_b: jax.Array
    field_a: jax.Array
    field_d: jax.Array
    field_c: jax.Array


class Struct(flax_struct.PyTreeNode):
    field_b: jax.Array
    field_a: jax.Array
    field_d: jax.Array
    field_c: jax.Array


if __name__ == "__main__":
    absltest.main()

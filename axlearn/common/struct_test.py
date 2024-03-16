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

import jax
import pytest
from absl.testing import absltest
from jax._src.tree_util import prefix_errors

from axlearn.common import struct


@struct.dataclass
class _Point:
    x: float
    y: float
    meta: Any = struct.field(pytree_node=False)


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
        @functools.partial(struct.dataclass, frozen=False, slots=True)
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
        new_p = jax.tree_util.tree_map(lambda x: x + x, p)
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
        Dummy = struct.dataclass(Dummy)
        self.assertTrue(hasattr(Dummy, "_axlearn_dataclass"))

        # pylint: disable-next=invalid-name
        Dummy = struct.dataclass(Dummy)  # no-op
        self.assertTrue(hasattr(Dummy, "_axlearn_dataclass"))

    def test_wrap_pytree_node_no_error(self):
        @struct.dataclass
        class Dummy(struct.PyTreeNode):
            a: int

        del Dummy

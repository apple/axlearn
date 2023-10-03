# Copyright © 2023 Apple Inc.

"""Tests common utils."""
import dataclasses
import sys
from collections import OrderedDict
from typing import Any, Iterable, NamedTuple, Optional, Sequence, Type

# pylint: disable=no-self-use
import chex
import jax
import jaxlib
import numpy as np
import pytest
import tensorflow as tf
import torch
from absl.testing import absltest, parameterized
from flax import serialization
from jax import numpy as jnp
from jax.experimental import checkify, mesh_utils
from jax.sharding import PartitionSpec

from axlearn.common import learner, optimizers
from axlearn.common.base_layer import BaseLayer, FactorizationSpec, ParameterSpec
from axlearn.common.config import config_class, config_for_function, similar_names
from axlearn.common.layers import BatchNorm, LayerNorm, Linear
from axlearn.common.module import Module
from axlearn.common.repeat import Repeat
from axlearn.common.test_utils import (
    ParamInitSpec,
    TestCase,
    TestWithTemporaryCWD,
    ThirdPartyInitializer,
    is_supported_mesh_shape,
    prng_impl,
    read_param_init_specs_recursively,
    read_per_param_settings,
)
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import (
    NestedTensor,
    StackedKeyArray,
    VDict,
    as_numpy_array,
    as_tensor,
    cast_floats,
    check_jax_type,
    check_param_shape_alignment,
    complete_partition_spec_tree,
    copy_recursively,
    count_model_params,
    create_device_mesh,
    flatten_items,
    get_data_dir,
    get_recursively,
    input_partition_spec,
    match_regex_rules,
    prune_tree,
    runtime_checks,
    set_data_dir,
    set_recursively,
    shard_input_batch,
    split_prng_key,
    tree_paths,
    validate_float_dtype,
    vectorized_tree_map,
)


class Combo(NamedTuple):
    head: Any
    tail: Any


class TreeUtilsTest(TestCase):
    def test_tree_paths(self):
        tree = {"a": 1, "b": [2, {"c": 3}]}
        self.assertEqual({"a": "a", "b": ["b/0", {"c": "b/1/c"}]}, tree_paths(tree))

        # Tuple.
        self.assertEqual(("0", ("1/0", "1/1"), "2"), tree_paths(("a", ("b", "c"), "d")))

        # NamedTuple.
        self.assertEqual(
            Combo(head="head", tail=Combo(head="tail/head", tail="tail/tail")),
            tree_paths(Combo(head=1, tail=Combo(head=2, tail=3))),
        )

        @chex.dataclass
        class DataclassCombo:
            scalar: int
            dataclass_combo: Any
            none: Type[None]
            nested_tensor: NestedTensor

        # Dataclass.
        self.assertEqual(
            DataclassCombo(
                scalar="scalar",
                dataclass_combo=DataclassCombo(
                    scalar="dataclass_combo/scalar",
                    dataclass_combo=Combo(
                        head="dataclass_combo/dataclass_combo/head",
                        tail="dataclass_combo/dataclass_combo/tail",
                    ),
                    none=None,
                    nested_tensor={},
                ),
                none=None,
                nested_tensor={
                    "a": ["nested_tensor/a/0", "nested_tensor/a/1"],
                    "c": None,
                },
            ),
            tree_paths(
                DataclassCombo(
                    scalar=1,
                    dataclass_combo=DataclassCombo(
                        scalar="hello",
                        dataclass_combo=Combo(head="head", tail="tail"),
                        none=None,
                        nested_tensor={},
                    ),
                    none=None,
                    nested_tensor={"a": [1, 2], "c": None},
                )
            ),
        )

        # None is preserved, similar to an empty list.
        self.assertEqual({"a": "a", "b": None, "c": []}, tree_paths({"a": 1, "b": None, "c": []}))

    def test_flatten_items(self):
        tree = {"a": 1, "b": [2, {"c": 3, "d": 4}], "e": None}
        # Note that we don't have ("e", None), since None is not considered a tree leaf.
        self.assertEqual([("a", 1), ("b/0", 2), ("b/1/c", 3), ("b/1/d", 4)], flatten_items(tree))
        self.assertEqual(
            [("a", 1), ("b.0", 2), ("b.1.c", 3), ("b.1.d", 4)],
            flatten_items(tree, separator="."),
        )
        kv = [("a", 1), ("b", 2)]
        d1 = OrderedDict(kv)
        d2 = OrderedDict(reversed(kv))
        self.assertEqual([("a", 1), ("b", 2)], sorted(flatten_items(d1)))
        self.assertEqual([("a", 1), ("b", 2)], sorted(flatten_items(d2)))

    def assertTensorEqual(self, a, b):
        self.assertIsInstance(a, jnp.ndarray)
        self.assertIsInstance(b, jnp.ndarray)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)

    def test_as_tensor(self):
        # From a number.
        self.assertTensorEqual(jnp.ones([], dtype=jnp.int32), as_tensor(1))
        # From a numpy array.
        self.assertTensorEqual(
            jnp.ones([2], dtype=jnp.float32), as_tensor(np.ones([2], dtype=np.float32))
        )
        # From a TF tensor.
        self.assertTensorEqual(
            jnp.ones([3], dtype=jnp.bfloat16),
            as_tensor(tf.ones([3], dtype=tf.bfloat16)),
        )
        # From a Torch tensor.
        self.assertTensorEqual(
            jnp.ones([4, 1], dtype=jnp.float16),
            as_tensor(torch.ones([4, 1], dtype=torch.float16)),
        )
        # From a nested structure.
        jax.tree_util.tree_map(
            self.assertTensorEqual,
            {
                "a": jnp.ones([1], dtype=jnp.float32),
                "b": [jnp.asarray([2]), {"c": jnp.asarray([[4]])}],
            },
            as_tensor(
                {
                    "a": np.ones([1], dtype=np.float32),
                    "b": [torch.as_tensor([2]), {"c": tf.convert_to_tensor([[4]])}],
                }
            ),
        )

    def assertNumpyArrayEqual(self, a, b):
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)

    def test_as_numpy_array(self):
        # From a number.
        self.assertNumpyArrayEqual(np.ones([], dtype=np.int64), as_numpy_array(1))
        # From a numpy array.
        self.assertNumpyArrayEqual(
            np.ones([2], dtype=np.float32), as_numpy_array(np.ones([2], dtype=np.float32))
        )
        # From a TF tensor.
        self.assertNumpyArrayEqual(
            np.ones([3], dtype=np.float16),
            as_numpy_array(tf.ones([3], dtype=tf.float16)),
        )
        # From a Torch tensor.
        self.assertNumpyArrayEqual(
            np.ones([4, 1], dtype=np.float32),
            as_numpy_array(torch.ones([4, 1], dtype=torch.float)),
        )
        # From a nested structure.
        jax.tree_util.tree_map(
            self.assertNumpyArrayEqual,
            {
                "a": np.ones([1], dtype=np.float32),
                "b": [np.array([2], dtype=np.int64), {"c": np.array([[4]], dtype=np.int32)}],
            },
            as_numpy_array(
                {
                    "a": jnp.ones([1], dtype=jnp.float32),
                    "b": [torch.as_tensor([2]), {"c": tf.convert_to_tensor([[4]])}],
                }
            ),
        )

    def test_vdict_tree_def(self):
        tree = VDict(a=jnp.arange(10), b=jnp.arange(7) - 3, c=None)
        # Note that 'None' is considered part of the tree structure, not tree leaves.
        self.assertEqual(
            "PyTreeDef(CustomNode(VDict[['a', 'b', 'c']], [*, *, None]))",
            str(jax.tree_util.tree_structure(tree)),
        )
        self.assertLen(jax.tree_util.tree_leaves(tree), 2)

    def test_vectorized_tree_map(self):
        tree = VDict(a=jnp.arange(10), b=jnp.arange(7) - 3)
        self.assertEqual(VDict(a="a", b="b"), tree_paths(tree))
        self.assertNestedAllClose([("a", tree["a"]), ("b", tree["b"])], flatten_items(tree))

        # Stack 3 trees together.
        stacked_tree = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), tree, tree, tree)
        self.assertEqual(type(stacked_tree), VDict)
        self.assertEqual(
            VDict(a=(3, 10), b=(3, 7)), jax.tree_util.tree_map(lambda t: t.shape, stacked_tree)
        )

        # jax.tree_util.tree_map() treats VDict similarly to dict.
        self.assertEqual(
            VDict(a=45 * 3, b=0), jax.tree_util.tree_map(lambda t: t.sum(), stacked_tree)
        )
        # vectorized_tree_map() vectorizes 'fn' on VDict and processes the 3 trees separately.
        self.assertNestedAllClose(
            VDict(a=jnp.asarray([45, 45, 45]), b=jnp.asarray([0, 0, 0])),
            vectorized_tree_map(lambda t: t.sum(), stacked_tree),
        )

        # Nested VDict.
        tree2 = VDict(c=stacked_tree)
        stacked_tree2 = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), tree2, tree2)
        self.assertEqual(
            VDict(c=VDict(a=(2, 3, 10), b=(2, 3, 7))),
            jax.tree_util.tree_map(lambda t: t.shape, stacked_tree2),
        )
        self.assertNestedAllClose(
            VDict(c=VDict(a=jnp.full([2, 3], 45), b=jnp.full([2, 3], 0))),
            vectorized_tree_map(lambda t: t.sum(), stacked_tree2),
        )

    def test_vectorized_tree_map_with_empty_vdict(self):
        self.assertNestedAllClose(
            VDict(x=None),
            vectorized_tree_map(
                lambda x, y: x + y,
                VDict(x=None),
                VDict(x=None),
            ),
        )
        self.assertNestedAllClose(
            VDict(a=VDict(x=None), b=jnp.asarray([4, 6])),
            vectorized_tree_map(
                lambda x, y: x + y,
                VDict(a=VDict(x=None), b=jnp.asarray([1, 2])),
                VDict(a=VDict(x=None), b=jnp.asarray([3, 4])),
            ),
        )

    def test_vdict_serialization(self):
        state_dict = dict(a=jnp.arange(10), b=jnp.arange(7) - 3)
        tree = VDict(**state_dict)
        v_state_dict = serialization.to_state_dict(tree)
        self.assertEqual(v_state_dict, state_dict)
        new_tree = serialization.from_state_dict(VDict, state=v_state_dict)
        self.assertEqual(new_tree, tree)
        # Check if `to_bytes` works as expected.
        self.assertEqual(serialization.to_bytes(state_dict), serialization.to_bytes(tree))

    def test_vdict_ref_count(self):
        x = jnp.arange(10)
        self.assertEqual(2, sys.getrefcount(x))
        v_dict = VDict(x=x)
        self.assertEqual(3, sys.getrefcount(x))
        self.assertEqual(2, sys.getrefcount(v_dict))
        values, keys = v_dict.tree_flatten()
        # tree_flatten should not increase ref count on `v_dict`.
        self.assertEqual(2, sys.getrefcount(v_dict))
        # `keys` should not increase ref count on `x`. Only `values` should.
        self.assertEqual(4, sys.getrefcount(x))
        self.assertSequenceEqual(["x"], keys)
        self.assertLen(values, 1)

    def test_get_and_set_recursively(self):
        tree = {"a": {"b": 2, "c": {"d": 3, "e": 4}}}
        self.assertEqual({"a": {"b": 2, "c": {"d": 3, "e": 4}}}, get_recursively(tree, ""))
        self.assertEqual({"a": {"b": 2, "c": {"d": 3, "e": 4}}}, get_recursively(tree, []))
        self.assertEqual({"b": 2, "c": {"d": 3, "e": 4}}, get_recursively(tree, "a"))
        self.assertEqual(2, get_recursively(tree, "a/b"))
        self.assertEqual(2, get_recursively(tree, ["a", "b"]))
        self.assertEqual({"d": 3, "e": 4}, get_recursively(tree, "a/c"))
        self.assertEqual(3, get_recursively(tree, "a.c.d", separator="."))

        with self.assertRaises(KeyError):
            get_recursively(tree, "a/foo")
        with self.assertRaises(KeyError):
            get_recursively(tree, ["a", "foo"])

        set_recursively(tree, value="bar", path="a/foo/b")
        self.assertEqual("bar", get_recursively(tree, "a/foo/b"))
        set_recursively(tree, value="boo", path="a.foo.b", separator=".")
        self.assertEqual("boo", get_recursively(tree, "a/foo/b"))
        set_recursively(tree, value="bar", path=["a", "foo", "b"])
        self.assertEqual("bar", get_recursively(tree, "a/foo/b"))
        with self.assertRaises(ValueError):
            set_recursively(tree, value="bar", path="")

    def test_copy_recursively(self):
        source = {"a": {"b": 2, "c": {"d": 3, "e": 4}}}
        self.assertEqual(
            {"a": {"b": 2}},
            copy_recursively(source=source, target=None, path=("a", "b")),
        )
        self.assertEqual(
            {"a": {"b": 2}},
            copy_recursively(source=source, target=None, path="a/b"),
        )
        self.assertEqual(
            {"a": {"b": 2}},
            copy_recursively(source=source, target=None, path="a.b", separator="."),
        )
        target = {"a": 1, "f": 3}
        self.assertEqual(
            {"a": {"b": 2}, "f": 3},
            copy_recursively(source=source, target=target, path="a/b"),
        )
        self.assertEqual(
            {"a": {"b": 2, "c": {"d": 3, "e": 4}}, "f": 3},
            copy_recursively(source=source, target=target, path="a/c"),
        )
        # Mutating `target` does not mutate source.
        # pylint: disable-next=unsubscriptable-object
        target["a"]["c"]["d"] = 10  # pytype: disable=unsupported-operands
        self.assertEqual(3, source["a"]["c"]["d"])

        # When path="", copy the entire source.
        target = copy_recursively(source=source, target=None, path="")
        self.assertEqual(source, target)
        # Mutating `target` does not mutate source.
        # pylint: disable-next=unsubscriptable-object
        target["a"]["b"] = 10
        self.assertEqual(2, source["a"]["b"])

    def test_split_prng_key(self):
        original_key = jax.random.PRNGKey(1234)

        def fn(key: jax.random.KeyArray):
            return jax.random.normal(key, [3, 2])

        base_results = []
        key = original_key
        for _ in range(10):
            key, child_key = jax.random.split(key)
            base_results.append(fn(child_key))
        base_results = jnp.stack(base_results)

        def batch(fn):
            return lambda split_keys: jax.vmap(fn)(split_keys.keys)

        split_keys = split_prng_key(original_key, 10)
        self.assertIsInstance(split_keys, StackedKeyArray)
        self.assertNestedAllClose(batch(fn)(split_keys), base_results)

        # Splitting the keys again is a no-op.
        resplit_keys = split_prng_key(split_keys, 10)
        self.assertNestedAllClose(resplit_keys, split_keys)
        self.assertNestedAllClose(batch(fn)(resplit_keys), base_results)

        # Splitting the keys again with the wrong number of keys.
        with self.assertRaisesRegex(AssertionError, "9"):
            split_prng_key(split_keys, 9)

        # Split keys by multiple dims.
        split_keys = split_prng_key(original_key, (2, 5))
        batch_results = batch(batch(fn))(split_keys)
        self.assertSequenceEqual(batch_results.shape, [2, 5, 3, 2])
        self.assertNestedAllClose(batch_results.reshape(base_results.shape), base_results)

        # Splitting the keys again is a no-op.
        resplit_keys = split_prng_key(split_keys, (2, 5))
        self.assertNestedAllClose(resplit_keys, split_keys)

    @parameterized.parameters(
        ((1, 1), ("data", "model")),
        ((1, 1, 1), ("pipeline", "data", "model")),
    )
    def test_input_partition_spec(self, mesh_shape, mesh_axis_names):
        if not is_supported_mesh_shape(mesh_shape):
            pytest.skip(reason=f"Unsupported mesh {mesh_shape}.")
        devices = mesh_utils.create_device_mesh(mesh_shape)
        with jax.sharding.Mesh(devices, mesh_axis_names):
            self.assertSequenceEqual(
                input_partition_spec(),
                PartitionSpec(
                    mesh_axis_names,
                ),
            )

    @parameterized.parameters(
        ((1, 4), ("data", "model"), "data"),
        ((1, 2, 2, 2), ("replica", "data", "fsdp", "model"), ("replica", "data", "fsdp")),
    )
    def test_shard_input_batch(
        self,
        mesh_shape: Sequence[int],
        mesh_axis_names: Sequence[str],
        batch_axis_names: Sequence[str],
    ):
        if not is_supported_mesh_shape(mesh_shape):
            pytest.skip(reason=f"Unsupported mesh {mesh_shape}.")
        devices = mesh_utils.create_device_mesh(mesh_shape)
        with jax.sharding.Mesh(devices, mesh_axis_names):
            sharded_batch = shard_input_batch(
                jnp.ones(jnp.prod(jnp.asarray(mesh_shape))),
                batch_axis_names=batch_axis_names,
            )
            # Check that the batch has been sharded.
            self.assertEqual(sharded_batch.sharding.spec, PartitionSpec(batch_axis_names))

    def test_complete_partition_spec_tree(self):
        data = dict(
            replicated=dict(a=1, b=2),
            sharded=VDict(c=3, d=4),
        )
        partition_by_x = PartitionSpec("x")
        partial_partition_spec = dict(replicated=None, sharded=partition_by_x)
        self.assertEqual(
            complete_partition_spec_tree(
                jax.tree_util.tree_structure(data), partial_partition_spec
            ),
            dict(
                replicated=dict(a=None, b=None), sharded=VDict(c=partition_by_x, d=partition_by_x)
            ),
        )
        param_spec = ParameterSpec(
            shape=[1, 2, 3],
            mesh_axes=["x", "y", "z"],
            factorization=FactorizationSpec(axes=[None, "row", "col"]),
        )
        self.assertEqual(
            complete_partition_spec_tree(
                jax.tree_util.tree_structure(data), dict(replicated=None, sharded=param_spec)
            ),
            dict(replicated=dict(a=None, b=None), sharded=VDict(c=param_spec, d=param_spec)),
        )

    @parameterized.parameters((jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16))
    def test_cast_floats(self, from_dtype, to_dtype):
        in_tree = {
            "w1": jnp.ones(2, dtype=from_dtype),
            "w2": jnp.zeros(3, dtype=jnp.int32),
        }
        out_tree = cast_floats(in_tree, to_dtype=to_dtype)

        def check_type(x):
            if x.dtype in [jnp.float32, jnp.bfloat16]:
                self.assertEqual(x.dtype, to_dtype)

        jax.tree_util.tree_map(check_type, out_tree)

        self.assertEqual(out_tree["w2"].dtype, in_tree["w2"].dtype)

    def test_count_model_params(self):
        tree = {
            "a": jnp.asarray([1]),
            "b": [jnp.asarray([2]), {"c": jnp.asarray([3]), "d": jnp.asarray([4])}],
            "e": None,
        }
        self.assertEqual(4, count_model_params(tree))

    def test_check_param_shape_alignment(self):
        target_tree = {
            "linear1": {
                "weight": jnp.zeros((32, 64)),
                "bias": jnp.zeros((64, 1)),
                "linear2": {
                    "weight": jnp.zeros((16, 32)),
                    "bias": jnp.zeros((32, 16)),
                },
            }
        }

        align_target_tree = {
            "linear1": {
                "weight": jnp.zeros((32, 64)),
                "bias": jnp.zeros((64, 1)),
                "linear2": {
                    "weight": jnp.zeros((16, 32)),
                    "bias": jnp.zeros((32, 16)),
                },
            }
        }

        misalign_target_tree = {
            "linear1": {
                "weight": jnp.zeros((15, 64)),
                "bias": jnp.zeros((64, 1)),
                "linear2": {
                    "weight": jnp.zeros((16, 32)),
                    "bias": jnp.zeros((32, 16)),
                },
            }
        }

        self.assertEqual(None, check_param_shape_alignment(target_tree, align_target_tree))
        error_msg = "(linear1/weight/0) shape is different: source: (32), target: (15)."
        self.assertEqual(error_msg, check_param_shape_alignment(target_tree, misalign_target_tree))

    def test_check_jax_type(self):
        check_jax_type(args=(1, 1.0, jax.numpy.ones(1), None, [{"key": 1}]))
        with self.assertRaises(ValueError):
            check_jax_type(args=([{"key": "1"}],))


class SimilarNamesTest(TestCase):
    @parameterized.parameters(
        ("test", ["test0", "other1"], ["test0"]),
        ("other", ["test1", "test0"], []),
        ("", ["test1", "test0"], []),
        ("aaa", ["aaa"], ["aaa"]),
        ("aaaab", ["aaa", "aaab"], ["aaab", "aaa"]),  # Test sorting by score.
        ("test", ["test1", "test0"], ["test0", "test1"]),  # Test sorting by alphabetical.
    )
    def test_similar_names(self, name: str, candidates: Iterable[str], expected: Iterable[str]):
        self.assertEqual(similar_names(name, candidates), expected)


class ContextManagerTest(TestWithTemporaryCWD):
    def test_runtime_checks(self):
        def f(x):
            checkify.check(x != 0, "cannot be zero!")
            return 1 / x

        # Jittable checks will fail by default, because we didn't checkify.
        with self.assertRaisesRegex(ValueError, "not functionalized"):
            jax.jit(f)(0)

        # With runtime_checks enabled, we should be able to crash with jittable checks without
        # needing to checkify.
        with runtime_checks():
            with self.assertRaisesRegex(jaxlib.xla_extension.XlaRuntimeError, "cannot be zero!"):
                jax.jit(f)(0)

    def test_prng_impl(self):
        self.assertEqual(jax.config.jax_default_prng_impl, "rbg")
        with prng_impl("threefry2x32"):
            self.assertEqual(jax.config.jax_default_prng_impl, "threefry2x32")
        self.assertEqual(jax.config.jax_default_prng_impl, "rbg")


class _TestParentLayer(BaseLayer):
    """A dummy parent layer."""

    @config_class
    class Config(BaseLayer.Config):
        child1: Linear.Config = Linear.default_config()
        child2: Linear.Config = Linear.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("child2", cfg.child2.set(input_dim=4, output_dim=5))
        self._add_child("child1", cfg.child1.set(input_dim=2, output_dim=3))


class _TestRepeatLayer(Repeat):
    """A dummy repeat layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.num_layers = 2
        return cfg


class ReadParamInitSpecsRecursivelyTest(TestCase):
    """Tests read_param_init_specs_recursively."""

    def test_ordering(self):
        layer = (
            Linear.default_config()
            .set(name="test", input_dim=2, output_dim=3)
            .instantiate(parent=None)
        )
        param_init_specs = read_param_init_specs_recursively(layer)
        self.assertSequenceEqual(param_init_specs["weight"].shape, [2, 3])
        self.assertSequenceEqual(param_init_specs["bias"].shape, [3])

    def test_nested(self):
        layer = _TestParentLayer.default_config().set(name="test").instantiate(parent=None)
        param_init_specs = read_param_init_specs_recursively(layer)
        self.assertSequenceEqual(param_init_specs["child1"]["weight"].shape, [2, 3])
        self.assertSequenceEqual(param_init_specs["child1"]["bias"].shape, [3])
        self.assertSequenceEqual(param_init_specs["child2"]["weight"].shape, [4, 5])
        self.assertSequenceEqual(param_init_specs["child2"]["bias"].shape, [5])

        # Check fan_axes.
        for child in ["child1", "child2"]:
            self.assertEqual(param_init_specs[child]["weight"].fan_axes.in_axis, -2)
            self.assertEqual(param_init_specs[child]["weight"].fan_axes.out_axis, -1)
            self.assertEqual(param_init_specs[child]["bias"].fan_axes, None)

    def test_flatten(self):
        layer = (
            Linear.default_config()
            .set(name="test", input_dim=2, output_dim=3)
            .instantiate(parent=None)
        )
        param_init_specs = read_param_init_specs_recursively(layer)
        self.assertEqual(
            [(name, init_spec.shape) for name, init_spec in flatten_items(param_init_specs)],
            [("weight", (2, 3)), ("bias", [3])],
        )

    def test_repeat_layer(self):
        layer = (
            _TestRepeatLayer.default_config()
            .set(name="test", layer=Linear.default_config().set(input_dim=2, output_dim=3))
            .instantiate(parent=None)
        )
        param_init_specs = read_param_init_specs_recursively(layer)
        self.assertSequenceEqual(param_init_specs["layer"]["weight"].shape, [2, 3])
        self.assertSequenceEqual(param_init_specs["layer"]["bias"].shape, [3])

    def test_delegates(self):
        class TestLayer(Linear):
            def initialize_parameters_recursively(
                self, prng_key: jax.random.KeyArray, *, prebuilt: Optional[NestedTensor] = None
            ) -> NestedTensor:
                params = super().initialize_parameters_recursively(prng_key, prebuilt=prebuilt)
                params["dummy"] = {"test": 1}
                return params

        layer = (
            TestLayer.default_config()
            .set(name="test", input_dim=2, output_dim=3)
            .instantiate(parent=None)
        )
        delegates = {
            "dummy": ParamInitSpec(
                shape=None,
                initializer=ThirdPartyInitializer.default_config()
                .set(library="dummy_delegate")
                .instantiate(),
                fan_axes=None,
            ),
        }
        param_init_specs = read_param_init_specs_recursively(layer, delegates=delegates)
        self.assertSequenceEqual(param_init_specs["weight"].shape, [2, 3])
        self.assertSequenceEqual(param_init_specs["bias"].shape, [3])
        self.assertIs(param_init_specs["dummy"].initializer, delegates["dummy"].initializer)


class ReadPerParamSettingsTest(TestCase):
    @parameterized.parameters(
        config_for_function(optimizers.adamw_optimizer).set(
            b1=0.9, b2=0.96, eps=1e-5, learning_rate=100.0
        ),
        config_for_function(optimizers.sgd_optimizer).set(
            learning_rate=100.0, decouple_weight_decay=True
        ),
        config_for_function(optimizers.adafactor_optimizer).set(
            learning_rate=100.0,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            multiply_by_parameter_scale=True,
            clipping_threshold=1.0,
            weight_decay_scale_by_learning_rate_exponent=1.0,
        ),
        config_for_function(optimizers.adafactor_optimizer).set(
            learning_rate=100.0,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            multiply_by_parameter_scale=False,
            clipping_threshold=1.0,
            dtype_momentum=jnp.int8,
            weight_decay_scale_by_learning_rate_exponent=1.0,
        ),
    )
    def test_add_decayed_weights(self, opt_cfg):
        def config_fn():
            trainer_cfg = SpmdTrainer.default_config()
            trainer_cfg.model = _TestParentLayer.default_config().set(name="test")
            per_param_scale = config_for_function(optimizers.per_param_scale_by_path).set(
                description="weight_decay_scale",
                scale_by_path=[
                    ("(.*/)?bias", 0.0),
                ],
            )
            optimizer_cfg = opt_cfg.set(
                weight_decay=5.0,
                weight_decay_per_param_scale=per_param_scale,
            )
            trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
            return trainer_cfg

        # pylint: disable-next=attribute-defined-outside-init
        self.named_trainer_configs = lambda: {"test": config_fn}
        weight_decays = read_per_param_settings(module=self, config_name="test")
        self.assertIn("weight_decay_scale", weight_decays)
        self.assertEqual(weight_decays["weight_decay_scale"]["child1"]["weight"], 1.0)
        self.assertEqual(weight_decays["weight_decay_scale"]["child1"]["bias"], 0.0)
        self.assertEqual(weight_decays["weight_decay_scale"]["child2"]["weight"], 1.0)
        self.assertEqual(weight_decays["weight_decay_scale"]["child2"]["bias"], 0.0)

    @parameterized.parameters(0.0, 3.5)
    def test_l2_regularizer(self, l2_regularizer_weight):
        def config_fn():
            trainer_cfg = SpmdTrainer.default_config()
            trainer_cfg.model = BatchNorm.default_config().set(name="test_model", input_dim=3)
            per_param_scale = config_for_function(optimizers.per_param_scale_by_path).set(
                description="l2_regularizer_scale",
                scale_by_path=[
                    ("(.*/)?bias", 0.0),
                ],
            )
            optimizer_cfg = config_for_function(optimizers.adam_optimizer).set(
                learning_rate=100.0,
                b1=0.9,
                b2=0.98,
                eps=1e-9,
                l2_regularizer_weight=l2_regularizer_weight,
                l2_regularizer_per_param_scale=per_param_scale,
            )
            trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
            return trainer_cfg

        # pylint: disable-next=attribute-defined-outside-init
        self.named_trainer_configs = lambda: {"test": config_fn}
        settings = read_per_param_settings(module=self, config_name="test")
        if l2_regularizer_weight:
            self.assertIn("l2_regularizer_scale", settings)
            l2_regs = settings["l2_regularizer_scale"]
            self.assertEqual(l2_regs["bias"], 0.0)
            self.assertEqual(l2_regs["scale"], 1.0)
            self.assertEqual(l2_regs["moving_mean"], 0.0)
            self.assertEqual(l2_regs["moving_variance"], 0.0)
        else:
            self.assertNotIn("l2_regularizer_scale", settings)

    def test_repeat_layer(self):
        def config_fn():
            trainer_cfg = SpmdTrainer.default_config()
            trainer_cfg.model = _TestRepeatLayer.default_config().set(
                name="test_model", layer=LayerNorm.default_config().set(input_dim=3)
            )
            per_param_scale = config_for_function(optimizers.per_param_scale_by_path).set(
                description="l2_regularizer_scale",
                scale_by_path=[
                    ("(.*/)?scale", 0.0),
                ],
            )
            optimizer_cfg = config_for_function(optimizers.adam_optimizer).set(
                learning_rate=100.0,
                b1=0.9,
                b2=0.98,
                eps=1e-9,
                l2_regularizer_weight=3.5,
                l2_regularizer_per_param_scale=per_param_scale,
            )
            trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
            return trainer_cfg

        # pylint: disable-next=attribute-defined-outside-init
        self.named_trainer_configs = lambda: {"test": config_fn}
        settings = read_per_param_settings(module=self, config_name="test")
        self.assertIn("l2_regularizer_scale", settings)
        l2_regs = settings["l2_regularizer_scale"]
        self.assertEqual(l2_regs["layer"]["bias"], 1.0)
        self.assertEqual(l2_regs["layer"]["scale"], 0.0)

    def test_two_per_param_scales(self):
        def config_fn():
            trainer_cfg = SpmdTrainer.default_config()
            trainer_cfg.model = Linear.default_config().set(
                name="test_model", input_dim=3, output_dim=2
            )
            l2_per_param_scale = config_for_function(optimizers.per_param_scale_by_path).set(
                description="l2_regularizer_scale",
                scale_by_path=[
                    (".*bias.*", 0),
                ],
            )
            freeze_per_param_scale = config_for_function(optimizers.per_param_scale_by_path).set(
                description="update_scale",
                scale_by_path=[
                    (".*weight.*", 0),
                ],
            )

            optimizer_cfg = config_for_function(optimizers.chain).set(
                args=[
                    config_for_function(optimizers.adam_optimizer).set(
                        learning_rate=100.0,
                        b1=0.9,
                        b2=0.98,
                        eps=1e-9,
                        l2_regularizer_weight=2.0,
                        l2_regularizer_per_param_scale=l2_per_param_scale,
                    ),
                    config_for_function(optimizers.scale_update_per_param).set(
                        per_param_scale=freeze_per_param_scale
                    ),
                ]
            )
            trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
            return trainer_cfg

        # pylint: disable-next=attribute-defined-outside-init
        self.named_trainer_configs = lambda: {"test": config_fn}
        settings = read_per_param_settings(module=self, config_name="test")
        # l2_per_param_scale.
        self.assertIn("l2_regularizer_scale", settings)
        l2_regs = settings["l2_regularizer_scale"]
        self.assertDictEqual(l2_regs, {"bias": 0.0, "weight": 1.0})
        # freeze_per_param_scale.
        self.assertIn("update_scale", settings)
        update_scales = settings["update_scale"]
        self.assertDictEqual(update_scales, {"bias": 1.0, "weight": 0.0})

    def test_learner_update_types(self):
        def config_fn():
            trainer_cfg = SpmdTrainer.default_config()
            trainer_cfg.model = Linear.default_config().set(
                name="test_model", input_dim=3, output_dim=2
            )
            trainer_cfg.learner.update_rules = [
                # Freeze weight.
                (".*weight.*", learner.UpdateType.NO_UPDATE),
            ]

            optimizer_cfg = config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                decouple_weight_decay=0.01,
            )
            trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
            return trainer_cfg

        # pylint: disable-next=attribute-defined-outside-init
        self.named_trainer_configs = lambda: {"test": config_fn}
        all_per_param_settings = read_per_param_settings(module=self, config_name="test")
        self.assertCountEqual(
            ["learner_update_type", "weight_decay_scale"], all_per_param_settings.keys()
        )
        # learner_update_type.
        self.assertDictEqual(
            all_per_param_settings["learner_update_type"],
            {"bias": learner.UpdateType.ALL_UPDATES, "weight": learner.UpdateType.ALL_UPDATES},
        )


class ValidateFloatDtypeTest(TestCase):
    """Tests validate_float_dtype."""

    @parameterized.parameters(jnp.float16, jnp.int32, jnp.int16)
    def test_validate_float_dtype_raises_for_invalid_dtypes(self, dtype: jnp.dtype):
        with self.assertRaisesRegex(ValueError, "float dtype"):
            validate_float_dtype(dtype)

    @parameterized.parameters(jnp.float32, jnp.bfloat16)
    def test_validate_float_dtype__for_valid_dtypes(self, dtype: jnp.dtype):
        validate_float_dtype(dtype)


class DataDirTest(TestCase):
    """Tests data_dir."""

    @parameterized.parameters(
        ("$DATA_DIR",),
        ("$DATA_DIR", "FAKE"),
        ("FAKE",),
        ("FAKE", "$DATA_DIR"),
        ("dir1", "dir2", "set_data_dir conflict"),
    )
    def test_get_and_set(
        self,
        data_dir1: Optional[str],
        data_dir2: Optional[str] = None,
        exception_regexp: Optional[str] = None,
    ):
        with set_data_dir(data_dir=data_dir1):
            self.assertEqual(get_data_dir(), data_dir1)
            if data_dir2 is not None:
                if exception_regexp:
                    with self.assertRaisesRegex(ValueError, exception_regexp):
                        with set_data_dir(data_dir=data_dir2):
                            pass
                else:
                    with set_data_dir(data_dir=data_dir2):
                        self.assertEqual(get_data_dir(), data_dir2)
                self.assertEqual(get_data_dir(), data_dir1)

    def test_exception_handling(self):
        try:
            with set_data_dir("dir1"):
                self.assertEqual("dir1", get_data_dir())
                raise RuntimeError()
        except RuntimeError:
            pass
        # Check that "dir2" is popped even with an exception.
        self.assertEqual("FAKE", get_data_dir())


class MatchRegexRulesTest(TestCase):
    """Tests match_regex_rules."""

    def test(self):
        rules = [
            (".*/bias", "b"),
            ("special/weight", "sw"),
            (".*/weight", "w"),
            ("ignored/weight", "iw"),
        ]
        self.assertEqual("b", match_regex_rules("layer/bias", rules=rules))
        self.assertEqual("w", match_regex_rules("layer/weight", rules=rules))
        # "special/weight" matches the "sw" rule first.
        self.assertEqual("sw", match_regex_rules("special/weight", rules=rules))
        # "ignored/weight" matches the "w" rule first.
        self.assertEqual("w", match_regex_rules("ignored/weight", rules=rules))
        # Full match only.
        self.assertIsNone(match_regex_rules("layer/weight_", rules=rules))
        self.assertEqual("w", match_regex_rules("not_special/weight", rules=rules))
        # Custom default value.
        self.assertEqual("d", match_regex_rules("layer/scale", rules=rules, default_value="d"))


class PruneTreeTest(TestCase):
    """Tests prune_tree."""

    def test(self):
        in_tree = {
            "a": {
                "b": {"d": "test"},
                "c": {
                    "b": None,
                    "e": 123,
                },
            },
            "f": 345,
        }
        # Prune by path.
        self.assertEqual(
            {"a": {"c": {"e": 123}}, "f": 345}, prune_tree(in_tree, lambda k, _: "b" in k)
        )
        # Prune by path with prefix/separator.
        self.assertEqual(
            {"a": {"c": {"b": None, "e": 123}}, "f": 345},
            prune_tree(in_tree, lambda k, _: k == "prefix:a:b", prefix="prefix", separator=":"),
        )
        # Prune by value.
        self.assertEqual(
            {"a": {"b": {"d": "test"}, "c": {"b": None}}},
            prune_tree(in_tree, lambda _, v: isinstance(v, int)),
        )


@dataclasses.dataclass(frozen=True)
class DummyDevice:
    """Mock device for testing."""

    platform: str
    device_kind: str
    process_index: int


@dataclasses.dataclass(frozen=True)
class DummyTpuDevice(DummyDevice):
    """Mock TPU device for testing."""

    coords: Sequence[int]
    core_on_chip: int = 0


@dataclasses.dataclass(frozen=True)
class DummyMultiSliceTpuDevice(DummyTpuDevice):
    """Mock multi-slice TPU device for testing."""

    slice_index: int = 0


class DeviceMeshTest(TestCase):
    @parameterized.parameters(
        {"logical_mesh": (2, 8)},
        {"logical_mesh": (4, 4)},
        {"logical_mesh": (1, 2, 8)},
    )
    def test_create_device_mesh_tpuv4(self, logical_mesh: Sequence[int]):
        physical_mesh = (4, 4, 1)
        coords = [
            (x, y, z)
            for x in range(physical_mesh[0])
            for y in range(physical_mesh[1])
            for z in range(physical_mesh[2])
        ]
        devices = [
            DummyTpuDevice(
                platform="tpu",
                device_kind="TPU v4",
                process_index=ix // 4,
                coords=coord,
            )
            for ix, coord in enumerate(coords)
        ]
        # Check that the constructed mesh has the expected shape.
        self.assertEqual(
            create_device_mesh(mesh_shape=logical_mesh, devices=devices).shape, logical_mesh
        )

    @parameterized.parameters(
        {"logical_mesh": (2, 16)},
        {"logical_mesh": (2, 4, 4)},
    )
    def test_create_device_mesh_multi_slice_tpuv4(self, logical_mesh: Sequence[int]):
        slice_physical_mesh = (4, 4, 1)
        num_slices = 2
        coords = [
            (x, y, z)
            for x in range(slice_physical_mesh[0])
            for y in range(slice_physical_mesh[1])
            for z in range(slice_physical_mesh[2])
        ]
        devices = [
            DummyMultiSliceTpuDevice(
                platform="tpu",
                device_kind="TPU v4",
                process_index=(len(coords) * slice_index + ix) // 4,
                coords=coord,
                slice_index=slice_index,
            )
            for ix, coord in enumerate(coords)
            for slice_index in range(num_slices)
        ]
        # Check that the constructed mesh has the expected shape.
        device_mesh = create_device_mesh(mesh_shape=logical_mesh, devices=devices)
        self.assertEqual(device_mesh.shape, logical_mesh)
        # Check that the sub_mesh along the first axis only contains devices from one of the slices.
        for ix, sub_mesh in enumerate(device_mesh):
            self.assertTrue(all(el.slice_index == ix for el in sub_mesh.flatten()))

    @parameterized.parameters(
        {"logical_mesh": (8, 2, 4)},
        {"logical_mesh": (16, 4)},
        {"logical_mesh": (2, 32)},
    )
    def test_create_device_mesh_gpu(self, logical_mesh: Sequence[int] = (8, 2, 4)):
        num_gpus_per_process = 8
        num_granules = 8
        devices = [
            DummyDevice(
                platform="gpu",
                device_kind="gpu",
                process_index=(num_gpus_per_process * granule_index + ix) // num_gpus_per_process,
            )
            for ix in range(num_gpus_per_process)
            for granule_index in range(num_granules)
        ]
        # Check that the constructed mesh has the expected shape.
        device_mesh = create_device_mesh(mesh_shape=logical_mesh, devices=devices)
        self.assertEqual(device_mesh.shape, logical_mesh)


if __name__ == "__main__":
    absltest.main()

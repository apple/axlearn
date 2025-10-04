# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/jax:
# Copyright 2018 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# AI-Hypercomputer/maxtext:
# Copyright 2024 The MaxText Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Common utilities."""

import collections
import contextlib
import copy
import dataclasses
import functools
import math
import numbers
import os
import re
import sys
import threading
import traceback
import types
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import cache
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import attr
import jax
import numpy as np
from absl import logging
from jax import numpy as jnp
from jax._src.ad_checkpoint import name_p
from jax._src.array import local_to_global_shape
from jax._src.lax import lax as lax_internal
from jax._src.mesh import thread_resources
from jax._src.tree_util import KeyEntry, KeyPath
from jax.ad_checkpoint import Offloadable, Recompute, Saveable
from jax.experimental import mesh_utils, multihost_utils
from jax.extend.core import Primitive
from jax.sharding import PartitionSpec

from axlearn.common import serialization
from axlearn.common.config import (
    ConfigBase,
    ConfigOr,
    FunctionConfigBase,
    config_for_function,
    is_named_tuple,
    maybe_instantiate,
    register_validator,
)

# New code should use Nested[XX] instead of NestedXX.
# Old definitions are provided for backwards compatibility.
_NestedT = TypeVar("_NestedT")
Nested = Union[_NestedT, dict[str, "Nested[_NestedT]"]]

Tensor = jax.Array
NestedTree = Union[Any, dict[str, Any]]
NestedTensor = Union[Tensor, dict[str, Any]]  # DEPRECATED, use Nested[Tensor].
NestedPartitionSpec = Optional[Union[PartitionSpec, dict[str, Any]]]

T = TypeVar("T")

# The device mesh shape in the form of a tuple of ints.
# We avoid subscripting Sequence[int] so it can be used for isinstance checks.
MeshShape = Sequence

_enable_numeric_checks = False
_enable_xla_runtime_errors = False

# The set of supported floating point dtypes.
_supported_float_dtypes = [jnp.bfloat16, jnp.float32]


@dataclasses.dataclass
class HybridMeshShape:
    """A mesh shape for hybrid (i.e., ICI and DCN) parallelism.

    For example, with mesh axes (data, model):
    - Pure fsdp on a v4-8:
        HybridMeshShape(ici_mesh_shape=(1, 4), dcn_mesh_shape=(1, 1))
    - Two-way data parallelism over 2 H100 nodes, and fsdp within-node:
        HybridMeshShape(ici_mesh_shape=(1, 8), dcn_mesh_shape=(2, 1))
    """

    ici_mesh_shape: MeshShape
    dcn_mesh_shape: MeshShape

    def __post_init__(self) -> None:
        if len(self.ici_mesh_shape) != len(self.dcn_mesh_shape):
            raise ValueError(
                f"{self.ici_mesh_shape=} should have the same length as {self.dcn_mesh_shape}."
            )

    def __len__(self):
        assert len(self.ici_mesh_shape) == len(self.dcn_mesh_shape)
        return len(self.ici_mesh_shape)


# "device" = Accelerator memory, e.g. HBM.
# "pinned_host" = Page locked memory on CPU, which can be address directly by accelerators by
# direct memory access (DMA). For TPU, "pinned_host" memory layout follows TPU device tile
# layout and usually cannot be zero-copy converted to a CPU-tensor.
MemoryKind = Literal["device", "pinned_host"]


@dataclasses.dataclass
class TensorSpec:
    """Specification of a Tensor.

    Used to describe model parameters and optimizer states.
    """

    shape: Sequence[int]
    dtype: Optional[jnp.dtype] = None
    mesh_axes: Optional[PartitionSpec] = None
    memory_kind: Optional[MemoryKind] = None

    @property
    def sharding(self) -> jax.sharding.Sharding:
        mesh = thread_resources.env.physical_mesh
        return jax.sharding.NamedSharding(
            mesh,
            PartitionSpec() if self.mesh_axes is None else self.mesh_axes,
            memory_kind=self.memory_kind,
        )


NestedTensorSpec = Optional[Union[TensorSpec, dict[str, Any]]]
RematType = Union[type(Saveable), Offloadable, type(Recompute)]
SavePattern = Union[str, re.Pattern, None]

# Register a config validator for RematType.
register_validator(
    match_fn=lambda x: isinstance(x, (type(Saveable), Offloadable, type(Recompute))),
    validate_fn=lambda x: None,
)


class RematPolicy(Protocol):
    def __call__(self, prim: Primitive, *args: Any, **params: Any) -> Union[RematType, bool]:
        ...


def save_and_offload_only_these_names_regex(
    *,
    names_which_can_be_saved: SavePattern,
    names_which_can_be_offloaded: SavePattern,
    offload_src: str,
    offload_dst: str,
) -> RematPolicy:
    """Adapted from jax source code to support regex.

    Reference:
    https://github.com/jax-ml/jax/blob/0d36b0b433a93c707f86dac89b0c05d40302775a/jax/_src/ad_checkpoint.py#L120

    Args:
        names_which_can_be_saved: A regex pattern for names which can be saved.
        names_which_can_be_offloaded: A regex pattern for names which can be offloaded.
        offload_src: The source device for offloading.
        offload_dst: The target device for offloading.

    Returns:
        A policy function that offloads and saves only the tensors that match the given
        regex patterns.
    """

    def policy(prim, *_, **params):
        if str(prim) == str(name_p):
            if names_which_can_be_saved and re.fullmatch(names_which_can_be_saved, params["name"]):
                return Saveable
            if names_which_can_be_offloaded and re.fullmatch(
                names_which_can_be_offloaded, params["name"]
            ):
                return Offloadable(src=offload_src, dst=offload_dst)
        return Recompute  # not saveable unless it's in the allow-list

    return policy


def offload_dots_saveable(offload_src: str, offload_dst: str) -> RematPolicy:
    """Extract from offload_dot_with_no_batch_dims and remove no-batch-dims limit.

    https://github.com/google/jax/blob/f4158ace933482844c145a6b919bf5dc86e084ba/jax/_src/ad_checkpoint.py#L81C1-L90C1
    This would remove the need to match the names for activation tensors.

    Args:
        offload_src: The source device for offloading.
        offload_dst: The target device for offloading.

    Returns:
        A policy fun that offloads dot_general_p to the target device and recomputes all other.
    """

    # pylint: disable-next=unused-argument
    def policy(prim, *_, **params):
        if str(prim) == str(lax_internal.dot_general_p):
            return Offloadable(src=offload_src, dst=offload_dst)
        return Recompute

    return policy


class RematCombineFn(Protocol):
    def __call__(
        self,
        p1: RematType,
        p2: RematType,
        *,
        prim: Primitive,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> RematType:
        """Protocol for remat policy combine function.

        Args:
            p1: Remat type returned by policy 1 for `prim`.
            p2: Remat type returned by policy 2 for `prim`.
            prim: The jax primitive for which the remat type will be applied.
            args: Positional arguments passed to RematPolicy
            kwargs: Keyword arguments passed to RematPolicy.

        Returns:
            RematType: The final remat type for `prim`.
        """


def default_remat_combine_fn(preferred_remat_type: Optional[RematType] = None) -> RematCombineFn:
    """The default remat policy combine function.

    If the two policies return conflicting remat types and neither is `Recompute`:
    - If `preferred_remat_type` is None, raises `RuntimeError`.
    - If `preferred_remat_type` is not None, `preferred_remat_type` will be the resulting
      remat type.

    Args:
        preferred_remat_type: Indicates how to resolve remat type conflicts.

    Returns:
        A `RematCombineFn` for use in `combine_remat_policies`.
    """

    def combine_fn(
        p1: RematType,
        p2: RematType,
        *,
        prim: Primitive,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ):
        del args, kwargs
        if p1 is not Recompute and p2 is not Recompute:
            if p1 is not p2:
                if preferred_remat_type is None:
                    raise RuntimeError(
                        f"Conflict in remat policies for primitive {prim}. "
                        f"Got policy 1 = {p1}, policy 2 = {p2}. "
                        "Please specify preferred_remat_type to resolve conflicts."
                    )
                else:
                    return preferred_remat_type
            return p1
        else:
            if p1 is not Recompute:
                return p1
            return p2

    return combine_fn


def combine_remat_policies(
    policy_1: ConfigOr[RematPolicy],
    policy_2: ConfigOr[RematPolicy],
    *,
    combine_fn: ConfigOr[RematCombineFn] = default_remat_combine_fn(),
):
    """Returns a remat policy that combines the two policies with `combine_fn`.

    Args:
        policy_1: Remat policy 1.
        policy_2: Remat policy 2.
        combine_fn: A function that combines and potentially resolves conflicts of the remat types
            from the two policies. The default `combine_fn` chooses the policy that does not return
            `Recompute` and raises `RuntimeError` if both are not `Recompute` and are different.

    Returns:
        A `RematPolicy`.
    """
    policy_1 = maybe_instantiate(policy_1)
    policy_2 = maybe_instantiate(policy_2)
    combine_fn = maybe_instantiate(combine_fn)

    def convert_to_enum(p: Union[RematType, bool]) -> RematType:
        if isinstance(p, bool):
            p = Saveable if p else Recompute
        return p

    def policy(prim, *args, **kwargs):
        p1 = convert_to_enum(policy_1(prim, *args, **kwargs))
        p2 = convert_to_enum(policy_2(prim, *args, **kwargs))
        return combine_fn(p1, p2, prim=prim, args=args, kwargs=kwargs)

    return policy


extended_checkpoint_policies = types.SimpleNamespace(
    offload_dots_saveable=offload_dots_saveable,
    save_and_offload_only_these_names_regex=save_and_offload_only_these_names_regex,
    combine_remat_policies=combine_remat_policies,
)


@contextlib.contextmanager
def runtime_checks(enabled: bool = True):
    old_state = _enable_xla_runtime_errors

    def switch(value):
        global _enable_xla_runtime_errors  # pylint: disable=global-statement
        _enable_xla_runtime_errors = value
        jax.config.update("jax_experimental_unsafe_xla_runtime_errors", value)

    switch(enabled)
    yield
    switch(old_state)


@contextlib.contextmanager
def numeric_checks(enabled: bool = True):
    old_state = _enable_numeric_checks

    def switch(value):
        global _enable_numeric_checks  # pylint: disable=global-statement
        _enable_numeric_checks = value
        jax.config.update("jax_debug_nans", value)

    switch(enabled)
    yield
    switch(old_state)


def check_numerics(x: Tensor, msg_fmt: str = "", **msg_kwargs):
    """Checks that all elements in `x` are finite."""
    global _enable_numeric_checks  # pylint: disable=global-statement,global-variable-not-assigned
    if _enable_numeric_checks:
        assert bool(jnp.isfinite(x).all()), f"Check numerics {msg_fmt.format(**msg_kwargs)}: {x}"
    return x


def shapes(nested_tensor: NestedTensor) -> NestedTree:
    """Returns a tree of the same structure as `nested_tensor` but with corresponding shapes instead
    of tensors."""
    return jax.tree.map(lambda x: getattr(x, "shape", x), nested_tensor)


def _concat(*, prefix: str, suffix: str, separator: str):
    return f"{prefix}{separator}{suffix}" if prefix else f"{suffix}"


def _key_entry_to_str(key_entry: KeyEntry) -> str:
    # Although (e.g.) DictKey does have its own __str__ implementation, calling
    # str(DictKey('a')) produces "['a']" instead of just "a".
    if isinstance(key_entry, jax.tree_util.DictKey):
        key = key_entry.key
    elif isinstance(key_entry, jax.tree_util.GetAttrKey):
        key = key_entry.name
    elif isinstance(key_entry, jax.tree_util.SequenceKey):
        key = key_entry.idx
    elif isinstance(key_entry, jax.tree_util.FlattenedIndexKey):
        key = key_entry.key
    else:
        raise RuntimeError(f"Unknown key entry type {type(key_entry)}: {key_entry}.")

    # Use f-string instead of calling str() because it matches the behavior of the previous
    # implementation and differs from str() for (e.g.) enums.
    return f"{key}"


def tree_paths(
    tree: NestedTree, separator: str = "/", is_leaf: Optional[Callable[[Any], bool]] = None
) -> NestedTree:
    """Returns a tree of the same structure as `nested_tensor` but with corresponding paths instead
    of values.

    E.g.,
        tree_paths({'a': 1, 'b': [2, {'c': 3}]}) = {'a': 'a', 'b': ['b/0', {'c': 'b/1/c'}]}

    Args:
        tree: A nested structure.
        separator: The separator between parts of a path.
        is_leaf: A Callable to evaluate whether the given node should be considered a leaf when
                 it otherwise would not, similarly to the is_leaf in jax.tree.map.

    Returns:
        A nested structure with the same structure as `tree`, but each leaf will be a string path.
        Note that None is not considered a leaf by jax.tree_util, hence also preserved by
        tree_paths.
    """
    return jax.tree.map_with_path(
        lambda kp, _: separator.join(_key_entry_to_str(k) for k in kp), tree, is_leaf=is_leaf
    )


def flatten_items(
    tree: Nested[Tensor], separator: str = "/", is_leaf: Optional[Callable[[Any], bool]] = None
) -> Sequence[tuple[str, Tensor]]:
    """Flattens `tree` and returns a list of (path, value) pairs."""
    flat_paths_and_values, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)
    return list(
        (separator.join(_key_entry_to_str(k) for k in path), value)
        for path, value in flat_paths_and_values
    )


@jax.tree_util.register_pytree_with_keys_class
class VDict(dict):
    """A dict with Tensor leaf nodes whose values should be vectorized."""

    def __repr__(self):
        return f"VDict({super().__repr__()})"

    def tree_flatten_with_keys(self):
        # Convert dict_values and dict_keys to lists to avoid holding reference to the VDict.
        # We sort the keys so that tree_map works with VDicts that have different key orderings,
        # matching jax's behavior for dicts.
        items = sorted(self.items(), key=lambda x: x[0])
        if not items:
            return ((), ())
        keys, values = zip(*items)
        aux = keys
        keys = [jax.tree_util.DictKey(k) for k in keys]
        key_values = list(zip(keys, values))
        return key_values, aux

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


# Register VDict as a dict for serialization.
serialization.register_serialization_state(
    VDict,
    # pylint: disable-next=protected-access
    ty_to_state_dict=serialization._dict_state_dict,
    # pylint: disable-next=protected-access
    ty_from_state_dict=serialization._restore_dict,
)


def vectorized_tree_map(fn, tree, *rest):
    """Similar to jax.tree.map(), but vectorizes `fn` on VDict's."""

    def vectorized_fn(*nodes):
        if isinstance(nodes[0], VDict):
            if not jax.tree_util.tree_leaves(nodes[0]):
                # This can happen when all VDict values are None and cause issues with jax.vmap.
                return nodes[0]
            nodes = [dict(**node) for node in nodes]
            result = jax.vmap(functools.partial(vectorized_tree_map, fn))(*nodes)
            return VDict(**result)
        return fn(*nodes)

    return jax.tree.map(vectorized_fn, tree, *rest, is_leaf=lambda t: isinstance(t, VDict))


def expand_vdicts(tree: NestedTensor) -> NestedTensor:
    """Expands each VDict in `tree` to a list.

    Args:
        tree: A nested tree of Tensors. All leaf nodes under a VDict must be tensors with the same
            dim 0 size.

    Returns:
        Returns a tree where every VDict is replaced by a list of dicts, where the length of the
        list equals to the dim 0 size of tensors in the VDict and list element i corresponds to
        slice i of the VDict tensors. The only exception is empty VDicts, which are not expanded.
    """

    def fn(value: Union[Tensor, VDict]) -> NestedTensor:
        if not isinstance(value, VDict):
            return value

        leaves = jax.tree_util.tree_leaves(value)
        if not leaves:
            # An empty VDict.
            return value

        non_tensor_leaves = [leaf for leaf in leaves if not isinstance(leaf, Tensor)]
        if non_tensor_leaves:
            raise ValueError(
                f"Expected a tree of Tensors, got {type(non_tensor_leaves[0])} in {tree}"
            )

        scalar_tensors = [leaf for leaf in leaves if not leaf.shape]
        if scalar_tensors:
            raise ValueError(
                f"Expected a tree of vectorized Tensors, got scalar {scalar_tensors} in {tree}"
            )

        vdict_size = leaves[0].shape[0]
        different_vdict_size_tensors = [leaf for leaf in leaves if leaf.shape[0] != vdict_size]
        if different_vdict_size_tensors:
            raise ValueError(
                "Expected a tree of vectorized Tensors of same dim 0, "
                f"got {different_vdict_size_tensors[0].shape[0]} vs. {vdict_size} in {tree}"
            )

        expanded: list[VDict] = []
        for ind in range(vdict_size):
            value_i: VDict = jax.tree.map(lambda x, i=ind: x[i], value)
            expanded_i = {k: expand_vdicts(v) for k, v in value_i.items()}
            expanded.append(expanded_i)
        return expanded

    return jax.tree.map(fn, tree, is_leaf=lambda x: isinstance(x, VDict))


class StackedKeyArray(NamedTuple):
    keys: Union[Tensor, "StackedKeyArray"]


def split_prng_key(
    prng_key: Union[StackedKeyArray, Tensor], num_keys: Union[int, Sequence[int]]
) -> StackedKeyArray:
    """Splits prng_key to keys iteratively and return the stacked keys.

    Args:
        prng_key: The input key. This can be a single key or a StackedKeyArray that was previously
            split.
        num_keys: The number of keys to split to. This can be a single integer or a sequence of
            integers. The total number of keys to generate will be the product of the integers.

    Returns:
        A nested StackedKeyArray, whose nesting level is the same as `len(num_keys)` and whose value
        is a key array with shape [num_keys...] and can be used by jax.lax.scan().
    """
    if isinstance(num_keys, int):
        num_keys = (num_keys,)

    if isinstance(prng_key, StackedKeyArray):

        def verify_key_shape(x):
            assert x.shape[: len(num_keys)] == num_keys, f"{x.shape} vs. {num_keys}"
            return x

        return jax.tree.map(verify_key_shape, prng_key)

    total_num_keys = np.prod(num_keys)
    child_prng_keys = []
    for _ in range(total_num_keys):
        # Generate the child keys iteratively to be consistent with how a parent module
        # generates child module keys.
        prng_key, child_key = jax.random.split(prng_key)
        child_prng_keys.append(child_key)

    def stack_and_reshape(*keys):
        # Reshape keys from [num_layers, ...] to [num_stages, num_layers_per_stage, ...].
        keys = jnp.stack(keys, axis=0)
        keys = jax.tree.map(lambda x: x.reshape(list(num_keys) + list(x.shape[1:])), keys)
        return keys

    # pylint: disable-next=no-value-for-parameter
    keys = jax.tree.map(stack_and_reshape, *child_prng_keys)

    for _ in num_keys:
        keys = StackedKeyArray(keys=keys)
    return keys


def as_tensor(x: Any):
    """Converts `x` to Tensor recursively.

    Args:
        x: a jnp array, numpy array, TF/PyTorch Tensor, or a nested structure of arrays or Tensors.

    Returns:
        A nested structure with the same structure as `x` but with values converted to Tensors.

    Raises:
        NotImplementedError: If conversion for the input type is unsupported.
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (numbers.Number, np.ndarray)):
        return jnp.asarray(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return jnp.asarray(x.numpy())
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree.map(as_tensor, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def as_numpy_array(x: Any):
    """Converts `x` to numpy ndarray recursively.

    Args:
        x: a jnp array, numpy array, TF/PyTorch Tensor, or a nested structure of arrays or Tensors.

    Returns:
        A nested structure with the same structure as `x` but with values converted to numpy array.

    Raises:
        NotImplementedError: If conversion for the input type is unsupported.
    """
    if isinstance(x, (numbers.Number, Tensor)):
        return np.array(x)
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return x.numpy()
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree.map(as_numpy_array, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def with_sharding_constraint(x: Tensor, shardings):
    mesh = thread_resources.env.physical_mesh
    if mesh.empty or mesh.size == 1:
        return x
    return jax.lax.with_sharding_constraint(x, shardings)


def maybe_shard(x: NestedTensor, partition_spec: Optional[PartitionSpec]) -> NestedTensor:
    if partition_spec is None:
        return x
    return with_sharding_constraint(x, PartitionSpec(*partition_spec))


def replicate_to_local_data(x: NestedTensor) -> NestedTensor:
    """Replicates and converts Tensors in `x` to local DeviceArrays.

    Args:
        x: The tensor to replicate.

    Returns:
        Replicated tensor.
    """
    return multihost_utils.process_allgather(x, tiled=True)


def complete_partition_spec_tree(
    treedef: jax.tree_util.PyTreeDef, partition_spec_tree: NestedTree
) -> NestedTree:
    """Adapted from flatten_axes(), but with a simplified API and more error logging and messages.

    Original:
    https://github.com/google/jax/blob/cdf4177f9219c9c0f70e243a097740e46138fc35/jax/_src/api_util.py#L277-L315

    Args:
        treedef: The tree structure of data to be partitioned according to `partition_spec_tree`.
        partition_spec_tree: A nested structure with PartitionSpecs or ParamPartitionSpecs and Nones
            at the leaves. Must be a tree prefix of `treedef`.

    Returns:
        A complete tree of PartitionSpecs or ParamPartitionSpecs and Nones that have the exact same
        structure as `treedef`.

    Raises:
        ValueError: If an unsupported type is encountered, or if partition_spec_tree is not a tree
            prefix of treedef.
    """
    proxy = object()
    dummy = jax.tree_util.tree_unflatten(treedef, [object()] * treedef.num_leaves)
    axes = []

    def replace_none_with_proxy(tree):
        if tree is None:
            return proxy
        if isinstance(tree, PartitionSpec) or dataclasses.is_dataclass(tree):
            return tree
        if is_named_tuple(tree):
            return type(tree)(*[replace_none_with_proxy(x) for x in tree])
        if isinstance(tree, (tuple, list)):
            return type(tree)([replace_none_with_proxy(x) for x in tree])
        if isinstance(tree, dict):
            return type(tree)([(k, replace_none_with_proxy(v)) for k, v in tree.items()])
        raise ValueError(f"{type(tree)}: {tree}")

    partition_spec_tree_with_proxy = replace_none_with_proxy(partition_spec_tree)

    def add_leaves(i, x):
        axes.extend([i] * len(jax.tree_util.tree_flatten(x)[0]))

    try:
        jax.tree.map(add_leaves, partition_spec_tree_with_proxy, dummy)
    except ValueError as err:
        logging.info("[complete_partition_spec_tree] ValueError: %s", err)
        logging.info(
            "[complete_partition_spec_tree] partition_spec_tree_with_proxy=%s",
            jax.tree_util.tree_structure(partition_spec_tree_with_proxy),
        )
        logging.info("[complete_partition_spec_tree] dummy=%s", jax.tree_util.tree_structure(dummy))
        for path, value in flatten_items(partition_spec_tree_with_proxy):
            logging.info(
                "[complete_partition_spec_tree] partition_spec_tree_with_proxy leaf: %s=%s",
                path,
                value,
            )
        for path, value in flatten_items(dummy):
            logging.info("[complete_partition_spec_tree] dummy leaf: %s=%s", path, value)

        raise ValueError(
            f"specification must be a tree prefix of the "
            f"corresponding value, got specification {partition_spec_tree} "
            f"for value tree {treedef}. Original ValueError: {err}"
        ) from None
    axes = [None if a is proxy else a for a in axes]
    assert (
        len(axes) == treedef.num_leaves
    ), f"({len(axes)} vs. {treedef.num_leaves}) {axes} {treedef}"
    return jax.tree_util.tree_unflatten(treedef, axes)


def input_partition_spec() -> PartitionSpec:
    """Returns partition spec for the input batch.

    We partition the inputs along all axes. For example, if the mesh has shape (64, 4) and axis
    names of ("data", "model"), the partition spec will be (("data", "model"), None...) so that the
    batch axis of every global tensor will be partitioned 256 (= 64 * 4) ways.

    Must be called within the context of a Mesh.
    """
    mesh = thread_resources.env.physical_mesh
    return PartitionSpec(
        mesh.axis_names,
    )


# Key associated with per-example dataset dispatch index tensor, indicating which logical
# batch index the example maps to.
PHYSICAL_TO_LOGICAL_DISPATCH_KEY = "__physical_to_logical_batch_dispatch"


def dispatch_input_batch(
    input_batch: NestedTensor, *, batch_axis_names: Union[str, Sequence[str]] = "data"
) -> NestedTensor:
    """Constrains all leaf values in the input batch, then (optionally) dispatches examples
    to a subset along the batch axis.

    The dispatchings are applied to all nested dicts which contain a special dispatching key in
    their root.

    This is deprecated in favor of `axlearn.common.input_dispatch`.

    Args:
        input_batch: The input batch, where the first dimension of each leaf is the batch dim.
        batch_axis_names: The name(s) of the batch axes.

    Returns:
        A nested tensor like the input batch, where each leaf contains
            a subset of the input batch, and has been wrapped with sharding annotations.
            N.B. some internal key-value pairs (like PHYSICAL_TO_LOGICAL_DISPATCH_KEY)
            may be dropped after use if present.
    """
    logging.log_first_n(
        logging.WARNING,
        "dispatch_input_batch is deprecated. Please use `axlearn.common.input_dispatch` instead.",
        n=1,
    )

    # Constrain the input batch.
    input_batch = jax.tree.map(
        lambda x: with_sharding_constraint(x, PartitionSpec(batch_axis_names)), input_batch
    )

    def traverse_and_dispatch(data: NestedTensor) -> NestedTensor:
        if isinstance(data, dict):
            # Dispatch from physical batch dimensions to logical batch.
            if PHYSICAL_TO_LOGICAL_DISPATCH_KEY in data:
                dispatch = data.pop(PHYSICAL_TO_LOGICAL_DISPATCH_KEY)
                return jax.tree.map(lambda x: jnp.einsum("b...,bl->l...", x, dispatch), data)
            for key, value in data.items():
                data[key] = traverse_and_dispatch(value)
        return data

    return traverse_and_dispatch(input_batch)


class DataPartitionType(Enum):
    # Data are fully partitioned across all devices.
    FULL = "full"
    # Data are fully replicated across all devices.
    REPLICATED = "replicated"


def data_partition_type_to_spec(
    partition: Union[DataPartitionType, Nested[PartitionSpec]],
) -> Nested[PartitionSpec]:
    """Returns a PartitionSpec for the given partition type."""
    if partition == DataPartitionType.FULL:
        return input_partition_spec()
    elif partition == DataPartitionType.REPLICATED:
        return PartitionSpec(None)
    elif isinstance(partition, PartitionSpec):
        return partition
    elif isinstance(partition, dict):
        return {k: data_partition_type_to_spec(v) for k, v in partition.items()}
    else:
        raise NotImplementedError(f"Unsupported partition: {partition}")


def host_to_global_array(
    host_arrays: Nested[Union[np.ndarray, Tensor]],
    *,
    partition: Union[Nested[PartitionSpec], DataPartitionType] = DataPartitionType.FULL,
) -> Nested[Tensor]:
    """Converts the given host device arrays to global device arrays.

    Must be called within the context of a Mesh.

    Args:
        host_arrays: A nested tree of device arrays in host memory. Usually these present the
            per-host portion of the global input batch. We currently assume that per-host portions
            form a uniform sharding across the batch.
        partition: How the global array should be partitioned.

    Returns:
        A nested tree with the same structure as `host_arrays`, but global device arrays at the
        leaves. Each global device array is partitioned according to `partition`.

    Raises:
        NotImplementedError: If the given `partition` type is not supported.
    """
    if isinstance(partition, DataPartitionType):
        logging.log_first_n(
            logging.WARNING,
            "Passing DataPartitionType is deprecated. Please specify a PartitionSpec directly.",
            n=1,
        )

    mesh = thread_resources.env.physical_mesh
    partition_specs = complete_partition_spec_tree(
        jax.tree_util.tree_structure(host_arrays),
        data_partition_type_to_spec(partition),
    )
    process_count = jax.process_count()

    def make_array(x: np.ndarray, partition_spec: PartitionSpec):
        if partition == DataPartitionType.FULL:
            global_shape = (x.shape[0] * process_count, *x.shape[1:])
        elif partition == DataPartitionType.REPLICATED:
            global_shape = (x.shape[0], *x.shape[1:])
        elif isinstance(partition, (PartitionSpec, dict)):
            global_shape = None  # Allow jax to infer.
        else:
            raise NotImplementedError(f"Unsupported partition: {partition}")
        return jax.make_array_from_process_local_data(
            sharding=jax.sharding.NamedSharding(mesh, partition_spec),
            local_data=x,
            global_shape=global_shape,
        )

    return jax.tree.map(make_array, host_arrays, partition_specs)


def host_to_global_specs(
    host_arrays: Nested[jax.ShapeDtypeStruct], *, partition: PartitionSpec
) -> Nested[jax.ShapeDtypeStruct]:
    """Converts the given host-local specs to global array specs.

    The API has the same semantics as `host_to_global_array`, which takes Tensors instead of specs.
    Please refer to `host_to_global_array` docstring for details.
    """
    mesh = thread_resources.env.physical_mesh
    partition_specs = complete_partition_spec_tree(
        jax.tree_util.tree_structure(host_arrays),
        data_partition_type_to_spec(partition),
    )

    def make_array_spec(x: jax.ShapeDtypeStruct, partition_spec: PartitionSpec):
        # `local_to_global_shape` is also used by `jax.make_array_from_process_local_data`.
        # It uses the process indices from devices in the mesh, which allows it to be compatible
        # with the fake devices used in AOT.
        sharding = jax.sharding.NamedSharding(mesh, partition_spec)
        global_shape = local_to_global_shape(sharding, local_shape=x.shape)
        # We use the sharding from `partition`, as host-local arrays do not have sharding.
        return jax.ShapeDtypeStruct(shape=global_shape, dtype=x.dtype, sharding=sharding)

    return jax.tree.map(make_array_spec, host_arrays, partition_specs)


def host_to_global_device_array(*args, **kwargs) -> Nested[Tensor]:
    """A deprecated alias for `host_to_global_array`.

    Please use `host_to_global_array` instead.
    """
    logging.log_first_n(
        logging.WARN, "host_to_global_device_array is renamed to host_to_global_array.", 1
    )
    return host_to_global_array(*args, **kwargs)


# TODO(markblee): Remove partition arg.
def global_to_host_array(
    global_arrays: Nested[Tensor],
    *,
    partition: Optional[DataPartitionType] = DataPartitionType.FULL,
) -> Nested[Tensor]:
    """Extracts host addressable data from each Tensor in `global_arrays`.

    Args:
        global_arrays: A nested Tensor.
            Each leaf Tensor must be uniformly partitioned across each dim.
        partition: Deprecated.

    Returns:
        A nested Tensor with the same structure as `global_array`. Each leaf Tensor will have shape
        `process_shape` where `process_shape` will be equal to `global_shape` if the global Tensors
        are replicated. If the global Tensors are partitioned across hosts, the `process_shape` will
        represent the host-local portion.
    """
    if partition is not None:
        logging.log_first_n(logging.WARNING, "Specifying partition is deprecated.", n=1)

    def index_to_shard(
        shards: list[jax.Shard], global_shape: Sequence[int]
    ) -> dict[tuple, jax.Shard]:
        """Returns a mapping from (sorted) indices to shards.

        Each key is a tuple of length `len(global_shape)`.
        Each element of the tuple is a `(start, limit)` tuple, specifying the start and limit
        indices of the shard along the global shape dim.
        """
        index_to_shard = []
        for shard in shards:
            index = tuple(
                (s.start or 0, s.stop or global_shape[dim]) for dim, s in enumerate(shard.index)
            )
            index_to_shard.append((index, shard))
        index_to_shard.sort(key=lambda x: x[0])
        return dict(index_to_shard)

    def get_local_array(value: Tensor) -> np.ndarray:
        # Note that we ensure consistent ordering of addressable shards by sorting by index below.
        local_shards: list[jax.Shard] = value.addressable_shards
        if not local_shards:
            raise ValueError(f"No local shards for {value}")

        # If value is replicated, return any shard.
        if value.is_fully_replicated:
            return np.asarray(local_shards[0].data)

        # A mapping from (unique) global index to local shards.
        global_index_to_local_shard = index_to_shard(local_shards, value.shape)
        # A mapping from dim -> local slices.
        dim_to_local_slices: list[list[slice]] = [[] for _ in range(value.ndim)]

        # For each dim, we bucket global slices into the corresponding local slice index.
        for dim, slices in enumerate(dim_to_local_slices):
            global_to_local = {}
            for global_index in global_index_to_local_shard:
                global_slice = global_index[dim]
                size = global_slice[1] - global_slice[0]
                if global_slice not in global_to_local:
                    # This exploits the fact that global slices are already sorted.
                    bucket_idx = len(global_to_local)
                    global_to_local[global_slice] = bucket_idx
                else:
                    # Along a given dim, global slices can appear multiples times if the array is
                    # replicated along a different dim, in which case the local slice is the same.
                    bucket_idx = global_to_local[global_slice]

                # We require uniform sharding along each dim.
                assert not slices or (slices[-1].stop - slices[-1].start) == size
                start = bucket_idx * size
                slices.append(slice(start, start + size))

        # The local shape can be inferred from the last offset along each dim.
        local_shape = tuple(local_slices[-1].stop for local_slices in dim_to_local_slices)

        # Build the final output array.
        output = np.empty(local_shape, dtype=value.dtype)
        for local_index, local_shard in zip(
            zip(*dim_to_local_slices), global_index_to_local_shard.values()
        ):
            output[tuple(local_index)] = local_shard.data
        return output

    return jax.tree.map(get_local_array, global_arrays)


def get_recursively(
    x: NestedTensor, path: Union[str, Sequence[str]], separator: Optional[str] = "/"
) -> NestedTensor:
    """Recursively indexes through the nested tensor.

    Args:
        x: The tensor to index.
        path: The sequence of keys used to recursively
            index the nested tensor. If `isinstance(path, str)`, it will be split
            into sequence of strings based on the `separator`.
        separator: If not None, the delimiter to split ``path`` by if `isinstance(path, str)`.

    Returns:
        NestedTensor

    Raises:
        KeyError: If the input path is invalid.
    """
    if not path:
        return x
    is_str = isinstance(path, str)
    if is_str:
        path = path.split(separator) if separator else [path]

    for idx, key in enumerate(path):
        if key not in x:
            prefix = separator.join(path[: idx + 1]) if separator and is_str else path[: idx + 1]
            raise KeyError(f"No entries found at path '{prefix}'")
        x = x[key]

    return x


def set_recursively(
    x: NestedTensor,
    *,
    value: Tensor,
    path: Union[str, Sequence[str]],
    separator: Optional[str] = "/",
):
    """Sets x[path...] = value, where path can be a multi-part index.

    If any part of the path does not exist in `x`, new sub dicts will be created, e.g.,

    x = {}
    set_recursive(x, value=1, path="a/b/c")
    # x = {"a": {"b": {"c": 1}}}

    Args:
        x: The tensor to index.
        value: The value to set at x[path].
        path: The sequence of keys used to recursively
            index the nested tensor. If `isinstance(path, str)`, it will be split
            into sequence of strings based on the `separator`.
        separator: If not None, the delimiter to split ``path`` by if `isinstance(path, str)`.

    Raises:
        ValueError: If the input path is empty.
    """
    if not path:
        raise ValueError("path must not be empty")
    if isinstance(path, str):
        path = path.split(separator) if separator else [path]

    for key in path[:-1]:
        if key not in x:
            x[key] = {}
        x = x[key]
    x[path[-1]] = value


def copy_recursively(
    *,
    source: NestedTensor,
    target: NestedTensor,
    path: Union[str, Sequence[str]],
    separator: str = "/",
) -> NestedTensor:
    """Sets target[path] = source[path].

    Args:
        source: The source tree.
        target: The target tree.
        path: The sequence of keys used to recursively
            index the nested tensor. If `isinstance(path, str)`, it will be split
            into sequence of strings based on the `delimiter`
        separator: The delimiter to split ``path`` by if `isinstance(path, str)`.

    Returns:
        The updated `target` or `source` if `path` is empty.
    """
    if not path:
        return copy.deepcopy(source)

    if isinstance(path, str):
        path = path.split(separator)

    x = source
    if target is None:
        target = type(source)()
    y = target
    for key in path[:-1]:
        x = x[key]
        if key not in y or not isinstance(y[key], type(x)):
            y[key] = type(x)()
        y = y[key]
    y[path[-1]] = copy.deepcopy(x[path[-1]])
    return target


def cast_floats(
    in_tree: Union[NestedTensor, NestedTensorSpec], to_dtype: Optional[jnp.dtype]
) -> Union[NestedTensor, NestedTensorSpec]:
    """Maps valid float arrays found in the inputs to the requested dtype in {float32, bfloat16}.

    Args:
        in_tree: The input values.
        to_dtype: Float type to cast values to; value is constrained to be in {float32, bfloat16}.
            If None, do not cast.

    Returns:
        The inputs after casting.

    Raises:
        ValueError: If to_dtype is unsupported.
    """
    if to_dtype is None:
        # Still make a copy of the tree.
        return jax.tree.map(lambda x: x, in_tree)

    if to_dtype not in _supported_float_dtypes:
        raise ValueError(f"to_dtype must be one of {_supported_float_dtypes}")

    from_dtype = jnp.float32 if to_dtype == jnp.bfloat16 else jnp.bfloat16

    def cast(x: Union[Tensor, TensorSpec]) -> Union[Tensor, TensorSpec]:
        if x.dtype == from_dtype:
            if isinstance(x, TensorSpec):
                return dataclasses.replace(x, dtype=to_dtype)
            else:
                return x.astype(to_dtype)
        return x

    return jax.tree.map(cast, in_tree)


@runtime_checkable
class PerParamFn(Protocol[T]):
    """A callable that operates on each parameter."""

    def __call__(self, params: Union[Nested[Tensor], Nested[TensorSpec]]) -> Nested[T]:
        """This protocol requires a callable that accepts either a nested Tensor or
        a nested TensorSpec as input and returns a processed value for each parameter.

        Args:
            params: A value of type NestedTensor or NestedTensorSpec.

        Returns:
            A value of type Nested[T], which is the processed value for each parameter.
        """


def per_param_dtype_by_path(
    default_dtype: Optional[jnp.dtype] = None,
    *,
    update_rules: Optional[Sequence[tuple[str, Optional[jnp.dtype]]]] = None,
) -> PerParamFn[jnp.dtype]:
    """Returns a function that assigns a dtype to each parameter based on the provided update
    rules. Each rule consists of a regex pattern that matches a parameter path, and a dtype to
    assign the parameter to. If no rule matches, the parameter is assigned to the provided
    `default_dtype`. If `default_dtype` is None, keep the original dtype as it is.

    Args:
        default_dtype: The dtype to use if none of the regex patterns match
            the parameter path.
        update_rules: A list of (regex, dtype) pairs. The first regex pattern fully matching the
            parameter path determines the dtype for the parameter.

    Returns:
        A function assigns each parameter to the appropriate dtype based on the update rules
        or the default dtype.

    Example:
        tree = {
            'conv1_weights': jnp.ones((3, 3), dtype=jnp.float32),
            'conv2_weights': jnp.ones((3, 3), dtype=jnp.float32),
            'fc1_weights': jnp.ones((10, 10), dtype=jnp.float32),
            'fc2_weights': jnp.ones((10, 10), dtype=jnp.float32),
        }
        default_dtype = jnp.float32
        update_rules = [
            ("^fc.*", jnp.bfloat16),
        ]
        cast_fn = per_param_dtype_by_path(default_dtype, update_rules)
        per_param_dtype = cast_fn(tree)
        Result:
        per_param_dtype = {
            'conv1_weights': jnp.float32,
            'conv2_weights': jnp.float32,
            'fc1_weights': jnp.bfloat16,
            'fc2_weights': jnp.bfloat16,
        }
    """

    def fn(
        tree: Union[Nested[Tensor], Nested[TensorSpec]],
    ) -> Union[Nested[Tensor], Nested[TensorSpec]]:
        if update_rules is None:
            return jax.tree.map(lambda x: default_dtype, tree_paths(tree))

        return jax.tree.map(
            lambda path: match_regex_rules(path, rules=update_rules, default_value=default_dtype),
            tree_paths(tree),
        )

    return fn


def cast_floats_per_param(
    in_tree: Union[NestedTensor, NestedTensorSpec],
    per_param_dtype: Nested[jnp.dtype],
) -> Union[NestedTensor, NestedTensorSpec]:
    """Cast each parameter in a tree to a specified dtype.

    Args:
        in_tree: The input values, which is a NestedTensor or NestedTensorSpec.
        per_param_dtype: Target dtype for each parameter in the `tree`.
            If None, no casting and will keep the original dtype.

    Returns:
        Union[NestedTensor, NestedTensorSpec]: A tree with the same shape as `in_tree`,
            but with all tensors or tensor specs cast to the specified data type.

    Raises:
        ValueError: If an unsupported dtype is provided in `per_param_dtype`.
    """

    def cast_per_param(
        x: Union[Tensor, TensorSpec], to_dtype: jnp.dtype
    ) -> Union[Tensor, TensorSpec]:
        if to_dtype is None:
            return x

        if to_dtype not in _supported_float_dtypes:
            raise ValueError(f"to_dtype must be one of {_supported_float_dtypes}")

        from_dtype = jnp.float32 if to_dtype == jnp.bfloat16 else jnp.bfloat16

        if x.dtype == from_dtype:
            if isinstance(x, TensorSpec):
                return dataclasses.replace(x, dtype=to_dtype)
            else:
                return x.astype(to_dtype)

        return x

    return jax.tree.map(cast_per_param, in_tree, per_param_dtype)


def canonicalize_per_param_dtype(
    param_dtype: Union[jnp.dtype, ConfigOr[PerParamFn[jnp.dtype]]],
) -> ConfigOr[PerParamFn[jnp.dtype]]:
    """Canonicalize the input `param_dtype` to a consistent format of
    `ConfigOr[PerParamFn[jnp.dtype]]`, which handles three possible cases:

    1. If `param_dtype` is `None`, it returns a configuration of default
       per_param_dtype_by_path function.
    2. If `param_dtype` is a `jnp.dtype`, it returns a configuration of
       per_param_dtype_by_path with `param_dtype` as `default_dtype`.
    3. If `param_dtype` is already an instance of `ConfigOr[PerParamFn[jnp.dtype]]`,
       it returns the `param_dtype` as it is.

    Args:
        param_dtype: A `jnp.dtype` or a `ConfigOr[PerParamFn[jnp.dtype]]`.

    Returns:
        ConfigOr[PerParamFn[jnp.dtype]]: A ConfigOr[PerParamFn[jnp.dtype]] that wraps the
        `param_dtype` as `default_dtype` or return `param_dtype` directly if it is already
        an instance of `ConfigOr[PerParamFn[jnp.dtype]]`.

    Raises:
        ValueError: If `param_dtype` does not match any of the required types.
    """

    if param_dtype is None:
        return config_for_function(per_param_dtype_by_path)
    # Check if param_dtype is an instance of jnp.dtype
    elif hasattr(param_dtype, "dtype") and isinstance(param_dtype.dtype, jnp.dtype):
        return config_for_function(per_param_dtype_by_path).set(
            default_dtype=param_dtype,
        )
    # Check if param_dtype is an instance of ConfigOr[PerParamFn[jnp.dtype]]
    elif isinstance(param_dtype, PerParamFn) or (
        isinstance(param_dtype, FunctionConfigBase) and isinstance(param_dtype.fn, PerParamFn)
    ):
        return param_dtype
    raise ValueError(
        f"{param_dtype} does not match any required types, should be "
        "jnp.dtype or ConfigOr[PerParamFn[jnp.dtype]]."
    )


def count_model_params(tree: NestedTensor) -> int:
    """Count the number of parameters in a model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))


def check_param_shape_alignment(
    source_tree: NestedTensor, target_tree: NestedTensor
) -> Union[None, str]:
    """Check param shape alignment between two parameter trees.

    This function assumes both trees have the same structures.

    Args:
        source_tree: The source parameter tree,
          which can be obtained via trainer.trainer_state.model or else.
        target_tree: The target parameter tree,
          which can be obtained via trainer.trainer_state.model or else.

    Returns:
        None if shape matches.
        A message indicating which parameter shapes are mismatched.
        e.g. "(linear1/weight/0) shape is different: source: (32), target: (15)."
    """
    param_shape_source = jax.tree.map(lambda x: x.shape, source_tree)
    param_shape_target = jax.tree.map(lambda x: x.shape, target_tree)
    output_str = []
    flatten_param_shape_source = dict(flatten_items(param_shape_source))
    flatten_param_shape_target = dict(flatten_items(param_shape_target))
    for k in flatten_param_shape_source:
        if flatten_param_shape_source[k] != flatten_param_shape_target[k]:
            output_str.append(
                f"({k}) shape is different: source: ({flatten_param_shape_source[k]}), "
                f"target: ({flatten_param_shape_target[k]})."
            )
    if len(output_str) == 0:
        return None
    else:
        return "\n".join(output_str)


def check_jax_type(
    *,
    args: Optional[Sequence] = None,
    kwargs: Optional[dict] = None,
    pretty_named_args: Optional[dict] = None,
    msg: Optional[str] = None,
):
    """Checks that the supplied arguments are valid JAX types and raise ValueError if not.

    Args:
        args: Positional arguments of a function call to check.
        kwargs: Keyword arguments of a function call to check.
        pretty_named_args: Arguments that already have a human readable name to check.
        msg: A prefix to print with a line break before the error message produced by this function.

    Raises:
        ValueError: If the supplied arguments are not valid jax types.
    """
    if pretty_named_args is None:
        pretty_named_args = {}
    if args is not None:
        pretty_named_args.update({f"args[{i}]": args[i] for i in range(len(args))})
    if kwargs is not None:
        pretty_named_args.update({f"kwargs[{key}]": kwargs[key] for key in kwargs})

    for name, arg in pretty_named_args.items():
        values, _ = jax.tree_util.tree_flatten(arg)
        for value in values:
            if not isinstance(value, (type(None), jax.Array, int, float)):
                if msg is None:
                    msg = ""
                else:
                    msg += "\n"
                msg += f"Argument {name} has leaf with non-JAX type {type(value)}"
                raise ValueError(msg)


def validate_float_dtype(dtype: jnp.dtype):
    """Validates if the provided dtype is both a float and amongst the set supported.

    Args:
        dtype: dtype to validate.

    Raises:
        ValueError: If the provided dtype does not fulfil the validation check.
    """
    if dtype not in _supported_float_dtypes:
        raise ValueError(f"float dtype ({dtype}) must be one of {_supported_float_dtypes}.")


def partial_with_fn_metadata(fn, *args, **kwargs):
    """Wraps a function with functools.partial, retaining original function's metadata."""
    partial_fn = functools.partial(fn, *args, **kwargs)
    return functools.update_wrapper(partial_fn, fn)


def prune_tree(
    in_tree: NestedTensor,
    should_prune: Callable[[str, NestedTensor], bool],
    *,
    prefix: str = "",
    separator: str = "/",
):
    """Returns a shallow copy of the input tree with subtrees pruned based on `should_prune`.

    This is a shallow copy because leaf nodes (non-dict values) are not deep-copied.

    Args:
        in_tree: The input tree to be pruned.
        should_prune: A callable which takes (path, subtree) as input and returns a boolean. The
            subtree provided will have already been pruned. If the callable returns True, the
            subtree itself will be dropped.
        prefix: Path prefix.
        separator: Separator used to join path parts.

    Returns:
        The pruned copy of the input tree.
    """
    if isinstance(in_tree, dict):
        # Use type() so that if in_tree is a VDict, out_tree is also a VDict.
        out_tree = type(in_tree)()
        for k, v in in_tree.items():
            path = _concat(prefix=prefix, suffix=k, separator=separator)
            v = prune_tree(v, should_prune, prefix=path, separator=separator)
            if not should_prune(path, v):
                out_tree[k] = v
        in_tree = out_tree
    return in_tree


def non_empty_leaf_merge_fn(primary: Any, secondary: Any):
    """This function chooses the non-empty leaf. If both leaves are non-empty, an error
    will be raised.
    """
    is_primary_empty = False
    is_secondary_empty = False
    try:
        is_primary_empty = len(primary) == 0
        is_secondary_empty = len(secondary) == 0
    except TypeError:
        # A TypeError will be raised if primary/secondary don't have length,
        # e.g. if they are scalars.
        pass
    if primary is None or is_primary_empty:
        return secondary
    if secondary is None or is_secondary_empty:
        return primary
    raise ValueError(
        f"Encountered incompatible subtree leaves: {primary=}, {secondary=}. Specify "
        "a custom override function to resolve incompatible subtree merges."
    )


def tree_merge(
    primary: Nested[Any],
    *,
    secondary: Nested[Any],
    leaf_merge_fn: Callable[[Any, Any], Any],
) -> Nested[Any]:
    """Merge `secondary` into `primary`. The result contains deep copies of subtrees from both.

    Two trees are mergable if there does not exists a path in `secondary` that is a subpath of any
    path in `primary`. If there are identical path with different leaves, `leaf_merge_fn` is used to
    determine which leaf is kept in the resulting tree.
    """
    if isinstance(primary, dict) ^ isinstance(secondary, dict):
        raise ValueError(f"Trying to merge incompatible subtrees: {primary=}, {secondary=}")
    # Use the override function if primary or secondary is a leaf.
    if not (isinstance(primary, dict) or isinstance(secondary, dict)):
        return copy.deepcopy(leaf_merge_fn(primary, secondary))
    # Use type() so that if primary is a VDict, out_tree is also a VDict.
    out_tree = type(primary)(primary)
    for k in secondary:
        if k in primary:
            out_tree[k] = tree_merge(
                primary[k], secondary=secondary[k], leaf_merge_fn=leaf_merge_fn
            )
        else:
            out_tree[k] = copy.deepcopy(secondary[k])
    return out_tree


@dataclasses.dataclass
class DataDirStack(threading.local):
    """See `install_context_stack` on how to ensure thread-safety of the global stack."""

    stack: list[Optional[str]]


_global_data_dir_stack = DataDirStack(stack=[])


def push_data_dir(data_dir: Optional[str]):
    _global_data_dir_stack.stack.append(data_dir)


def pop_data_dir() -> Optional[str]:
    return _global_data_dir_stack.stack.pop(-1)


@contextlib.contextmanager
def set_data_dir(data_dir: Optional[str]):
    """Sets the environment variable DATA_DIR to the given `data_dir`.

    Args:
        data_dir: The data_dir.

    Raises:
        ValueError: If the environment variable DATA_DIR is already set to a different value.
    """
    cur_data_dir = get_data_dir()
    if (
        data_dir is not None
        and cur_data_dir is not None
        and data_dir != "FAKE"
        and cur_data_dir != "FAKE"
        and data_dir != cur_data_dir
    ):
        raise ValueError(f"set_data_dir conflict: data_dir={data_dir} DATA_DIR={cur_data_dir}")

    push_data_dir(data_dir)
    try:
        yield
    finally:
        popped = pop_data_dir()
        assert popped == data_dir, f"{popped} vs. {data_dir}"


def get_data_dir() -> Optional[str]:
    if _global_data_dir_stack.stack:
        return _global_data_dir_stack.stack[-1]
    return os.environ.get("DATA_DIR")


def get_or_none(x: Optional[dict], key: Any) -> Optional[Any]:
    return None if x is None else x.get(key)


T = TypeVar("T")


def match_regex_rules(
    x: str, *, rules: Sequence[tuple[str, T]], default_value: Optional[T] = None
) -> Optional[T]:
    """Matches the given string against a sequence of regex-based rules.

    Args:
        x: The str to match against the rules.
        rules: A sequence of (regex, value) pairs.
        default_value: The value to return if none of the rules matches `x`.

    Returns:
        The value from the first matching rule or `default_value`.
    """
    for regex, value in rules:
        if re.fullmatch(regex, x):
            return value
    return default_value


def _register_per_param_settings(
    settings: NestedTree, *, description: str, path: Optional[str] = None
):
    del settings, description, path


def register_per_param_settings(
    settings: NestedTree, *, description: str, path: Optional[str] = None
) -> NestedTree:
    """Registers per-parameter setting.

    This function can be patched in testing to inspect per-param settings.

    Args:
        settings: A nested tree of per parameter settings, e.g. a per parameter learner update rule.
        description: A string description of the per-param settings.
        path: An optional string of where the per param settings is registered.

    Returns:
        A nested tree of per parameter settings.
    """
    _register_per_param_settings(settings, description=description, path=path)
    if logging.vlog_is_on(1):
        for param_path, param_setting in flatten_items(settings):
            logging.info(
                "Per-param setting %s registered in %s: %s=%s",
                description,
                path,
                param_path,
                param_setting,
            )
    return settings


def _reshape_mesh_to_rings(a: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    """Reshapes device mesh to rings for 64x4 or 32x8 mesh shape.

    Adapted from maxtext and made some code simplifications. Reference:
    https://github.com/AI-Hypercomputer/maxtext/blob/7f0dcef34f4857476d19b4ca9ceada654246c0b0/MaxText/max_utils.py#L474.

    64x4 and 32x8 are non-native mesh sizes on v6e and v5e and require careful arrangement of
    devices to achieve good performance.
    """
    b = []
    if shape == (64, 4):
        for i in range(8):
            b.append([])
            for j in range(8):
                a_i = i * 2
                a_j = j * 2
                # Forms a ring of size 4.
                b[i].append([a[a_i, a_j], a[a_i, a_j + 1], a[a_i + 1, a_j + 1], a[a_i + 1, a_j]])
    elif shape == (32, 8):
        for i in range(8):
            b.append([])
            for j in range(4):
                a_i = i * 2
                a_j = j * 4
                # Forms a ring of size 8.
                b[i].append(
                    [
                        a[a_i, a_j],
                        a[a_i, a_j + 1],
                        a[a_i, a_j + 2],
                        a[a_i, a_j + 3],
                        a[a_i + 1, a_j + 3],
                        a[a_i + 1, a_j + 2],
                        a[a_i + 1, a_j + 1],
                        a[a_i + 1, a_j],
                    ]
                )
    else:
        raise ValueError(f"The target mesh shape {shape} is not implemented.")
    return np.reshape(np.array(b), shape)


def _maybe_get_special_mesh(
    mesh_shape: MeshShape, *, devices: np.ndarray
) -> Optional[tuple[int, int]]:
    """Checks if any of the special mesh shapes are applicable."""
    if int(np.prod(mesh_shape)) != 256:
        return None
    if getattr(devices[0], "device_kind", None) not in [
        "TPU v5e",
        "TPU v6e",
        "TPU v6 lite",
        "TPU v5 lite",
    ]:
        return None

    filtered_mesh = tuple(filter(lambda x: x != 1, mesh_shape))
    target_shapes = [(64, 4), (32, 8)]
    return None if filtered_mesh not in target_shapes else filtered_mesh


def build_standard_mesh(mesh_shape: MeshShape, *, devices: np.ndarray) -> np.ndarray:
    logging.info("Building device mesh.")
    mesh_shape = infer_mesh_shape(mesh_shape, num_devices=devices.size)
    try:
        if (shape := _maybe_get_special_mesh(mesh_shape, devices=devices)) is not None:
            # If any of the special mesh shapes is applicable, use them.
            mesh = mesh_utils.create_device_mesh([16, 16], devices=devices)
            mesh = _reshape_mesh_to_rings(mesh, shape=shape)
            mesh = mesh.reshape(mesh_shape)
            logging.log_first_n(logging.INFO, "Using custom mesh: %s", 1, str(mesh))
            return mesh
        return mesh_utils.create_device_mesh(mesh_shape, devices=devices)
    except NotImplementedError as e:
        logging.warning(
            "mesh_utils.create_device_mesh cannot handle shape %s: %s. "
            "Falling back to the naive mesh. Performance may be reduced.",
            mesh_shape,
            e,
        )
        return devices.reshape(mesh_shape)


def create_hybrid_device_mesh(
    mesh_shape: HybridMeshShape,
    *,
    devices: Sequence[Any],
    process_is_granule: bool = False,
) -> np.ndarray:
    """Extends the method to have an option to fall back to naive mesh.

    Reference:
    https://github.com/google/jax/blob/1189d61bc086fcfb548e73235a601ec46c3623c5/jax/experimental/mesh_utils.py#L324

    Args:
        mesh_shape: Shape of the logical mesh for both ICI and DCN.
            The ICI mesh corresponds to the faster/inner network, ordered by increasing network
            intensity, e.g. [data, fsdp, model] where model has the most network communication
            requirements.
            The DCN mesh corresponds to the slower/outer network in the same order as the ICI mesh.
            We expect the shapes to be fully specified, i.e., they should not contain -1 dims.
        devices: The devices to construct a mesh for.
        process_is_granule: If True, this function will treat processes as the units of the
            slower/outer network by looking for "process_index" attributes on devices. Otherwise it
            will treat slices as the units and look for "slice_index" attributes on devices.

    Raises:
        ValueError: If the number of granules to which the `devices` belong doesn't equal the
            product of `dcn_mesh_shape`, or if the number of devices belonging to any single granule
            does not equal the product of `mesh_shape`.

    Returns:
        A np.ndarray of JAX devices with `ici_mesh_shape * dcn_mesh_shape` as its shape that can be
        fed into jax.sharding.Mesh for hybrid parallelism.
    """
    device_attr = "process_index" if process_is_granule else "slice_index"
    assert hasattr(devices[0], device_attr)
    granule_dict = collections.defaultdict(list)
    for dev in devices:
        granule_dict[getattr(dev, device_attr)].append(dev)
    granules = list(granule_dict[key] for key in sorted(granule_dict.keys()))
    if np.prod(mesh_shape.dcn_mesh_shape) != len(granules):
        raise ValueError(
            f"Number of slices/granules {len(granules)} must equal the product of "
            f"dcn_mesh_shape {mesh_shape.dcn_mesh_shape}"
        )
    per_granule_meshes = [
        build_standard_mesh(mesh_shape.ici_mesh_shape, devices=np.asarray(granule))
        for granule in granules
    ]
    granule_mesh = np.arange(len(granules)).reshape(mesh_shape.dcn_mesh_shape)
    blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(granule_mesh)
    device_mesh = np.block(blocks.tolist())
    return device_mesh


def create_device_mesh(
    mesh_shape: Union[MeshShape, HybridMeshShape],
    *,
    devices: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Constructs a device mesh.

    If `mesh_shape` is specified as a `HybridMeshShape`, we use the `ici_mesh_shape` and
    `dcn_mesh_shape` directly to construct the mesh.

    If `mesh_shape` is specified as a `MeshShape`, we first determine whether we are running in a
    TPU or GPU environment.
        - If running in a TPU environment:
            - If multi-slice/granule, we split the first non-singleton axis of the configured mesh
                shape across the slices.
        - If running in a GPU environment:
            - If multi-node, and the first non-singleton axis divides the number of processes
                (GPU-nodes/granules), we split the first axis across the processes.

    In all other cases we construct a standard mesh according to the configured mesh_shape.

    Args:
        mesh_shape: The desired logical mesh shape.
        devices: The devices that will be used to construct the mesh.
            If None, defaults to jax.devices().

    Returns:
        A numpy array containing the JAX devices with shape determined by the config mesh_shape.

    Raises:
        NotImplementedError: If not all devices have the same platform.
    """
    if devices is None:
        devices = jax.devices()
    devices = np.asarray(devices)

    # Check if the devices are part of a multi-granule configuration.
    # <https://github.com/google/jax/blob/b81b79c1b0d2ec/jax/experimental/mesh_utils.py#L313>
    device_platform = devices[0].platform
    device_attr = "process_index" if device_platform != "tpu" else "slice_index"
    is_multi_granule_env = hasattr(devices[0], device_attr)
    if not all(el.platform == device_platform for el in devices):
        raise NotImplementedError(f"Not all devices had platform: {device_platform}.")

    num_granules = (
        max(getattr(el, device_attr) for el in devices.flatten()) + 1 if is_multi_granule_env else 1
    )
    num_devices = len(devices)
    assert (
        num_devices % num_granules == 0
    ), "Number of devices must be divisible by number of granules."
    num_devices_per_granule = num_devices // num_granules

    # Fallback to a standard mesh if on GPU with incompatible multi-granule mesh.
    if (
        device_platform == "gpu"
        and isinstance(mesh_shape, MeshShape)
        and mesh_shape[0] % num_granules != 0
    ):
        logging.warning("Falling back to ICI-only mesh on GPU, performance may be reduced.")
        return build_standard_mesh(mesh_shape, devices=devices)

    # Canonicalize to HybridMeshShape. If DCN mesh is not specified, break the first non-singleton
    # device axis (the least communication intensive) over the number of slices/granules. If all
    # axes are singletons, this is effectively a no-op, since this implies a single-granule
    # environment.
    if isinstance(mesh_shape, MeshShape):
        mesh_shape = infer_mesh_shape(mesh_shape, num_devices=num_devices)
        for axis, dim in enumerate(mesh_shape):
            if dim % num_granules == 0:
                break
            elif dim != 1:
                raise ValueError(
                    f"First non-singleton mesh axis {axis} with value {dim} must be divisible by "
                    f"the number of slices/granules {num_granules}."
                )
        else:
            raise ValueError(
                f"At least one axis of {mesh_shape=} must be divisible by {num_granules=}."
            )

        if num_granules > 1:
            logging.info("Building multi-slice/granule device mesh over axis %s.", axis)
        # Truncate intra-slice/granule mesh.
        mesh_shape = (*mesh_shape[:axis], dim // num_granules, *mesh_shape[axis + 1 :])
        logging.info("Inferred intra-slice/granule mesh shape: %s", mesh_shape)
        # Configure data center (inter-slice/granule) mesh.
        dcn_mesh_shape = (1,) * axis + (num_granules,) + (1,) * len(mesh_shape[axis + 1 :])
        logging.info("Inferred inter-slice/granule mesh shape: %s", dcn_mesh_shape)

        mesh_shape = HybridMeshShape(ici_mesh_shape=mesh_shape, dcn_mesh_shape=dcn_mesh_shape)
    else:
        # Infer -1 values in the mesh.
        mesh_shape = HybridMeshShape(
            ici_mesh_shape=infer_mesh_shape(
                mesh_shape.ici_mesh_shape, num_devices=num_devices_per_granule
            ),
            dcn_mesh_shape=infer_mesh_shape(mesh_shape.dcn_mesh_shape, num_devices=num_granules),
        )
    logging.info("Using hybrid mesh shape: %s.", mesh_shape)

    # Check that we have the right number of devices.
    assert num_granules * num_devices_per_granule == len(devices)
    if np.prod(mesh_shape.dcn_mesh_shape) != num_granules:
        raise ValueError(
            f"Product of DCN mesh {mesh_shape.dcn_mesh_shape} does not match {num_granules=}."
        )
    if np.prod(mesh_shape.ici_mesh_shape) != num_devices_per_granule:
        raise ValueError(
            f"Product of ICI mesh {mesh_shape.ici_mesh_shape} does not match "
            f"{num_devices_per_granule=}."
        )

    # Return a standard mesh if not a multi-granule env.
    if num_granules == 1:
        return build_standard_mesh(mesh_shape.ici_mesh_shape, devices=devices)

    return create_hybrid_device_mesh(
        mesh_shape,
        devices=devices,
        process_is_granule=device_attr == "process_index",
    )


def infer_mesh_shape(mesh_shape: MeshShape, *, num_devices: Optional[int] = None) -> MeshShape:
    """Infer the value for -1 from len(jax.devices()) and other dims if there is -1 in mesh shape.

    Args:
        mesh_shape: The original MeshShape, which might have -1 in one axis.
        num_devices: The devices that will be used to construct the mesh.
            If None, defaults to len(jax.devices()).

    Returns
        A new MeshShape with inferred value for -1.
    """
    if -1 not in mesh_shape:
        return mesh_shape

    if mesh_shape.count(-1) > 1:
        raise ValueError(f"Only one axis can be -1 in {mesh_shape=}.")

    # Handle the case with one -1.
    prod = math.prod(mesh_shape, start=-1)
    if num_devices is None:
        num_devices = len(jax.devices())
    if num_devices % prod != 0:
        raise ValueError(
            f"Unable to infer -1 in mesh shape {mesh_shape} as num_devices {num_devices} "
            f"is not a multiple of the product {prod} of mesh axes."
        )

    return tuple(x if x != -1 else num_devices // prod for x in mesh_shape)


def thread_stack_traces() -> Sequence[Sequence[str]]:
    """Retrieves the current python stack traces."""
    grouped_lines = []
    for thread in threading.enumerate():
        lines = []
        thread_id = thread.ident
        lines.append(f"Thread: {thread.name}({thread_id})")
        # pylint: disable-next=protected-access
        for line in traceback.format_stack(sys._current_frames()[thread_id]):
            lines.append(f">>> {line.rstrip()}")
        grouped_lines.append(lines)
    return grouped_lines


def pytree_children(node: Any) -> Sequence[tuple[KeyEntry, Any]]:
    """Generate the (key, value) pairs for the immediate children of a pytree `node`.

    Reference: jax._src.tree_util.generate_key_paths()

    Example:
        ```
        assert pytree_children(dict(a=[1,2])) == [(DictKey('a'), [1,2])]
        ```
    """
    flat = jax.tree_util.default_registry.flatten_one_level(node)
    if flat is None:
        return []

    if isinstance(node, tuple) and hasattr(node, "_fields") and flat[1] == type(node):
        # Handle namedtuple as a special case, based on heuristic.
        return [(jax.tree_util.GetAttrKey(s), getattr(node, s)) for s in node._fields]

    key_children, _ = jax.tree_util.default_registry.flatten_one_level_with_keys(node)
    if key_children:
        return key_children

    return [(jax.tree_util.FlattenedIndexKey(i), c) for i, c in enumerate(flat[0])]


def find_cycles(tree: Nested) -> dict[str, KeyPath]:
    """Find a cycle in pytree `tree` if one exists.

    This function finds a descendant which has reference equality with one of its own
    ancestors, if one exists.

    Args:
        tree: The tree to find cycles in.

    Returns:
        If no cycle is found, an empty dict.
        If a cycle is found a dict with keys:
        * descendant: The KeyPath to the descendant.
        * ancestor: The KeyPath to the ancestor.
    """

    def _find_cycles(tree: Nested, *, key_path: KeyPath, seen: list[int]) -> dict[str, KeyPath]:
        # DFS and check if path to root contains repeats.
        # This is quadratic time in the depth of the tree but could be made linear
        # time with a small amount of additional implementation complexity.
        uid = id(tree)
        if uid in seen:
            result = dict(descendant=key_path[:], ancestor=key_path[: seen.index(uid)])
            return result
        seen.append(uid)
        items = pytree_children(tree)
        for key, child in items:
            key_path.append(key)
            result = _find_cycles(child, key_path=key_path, seen=seen)
            key_path.pop()
            if result:
                return result
        seen.pop()
        return {}

    return _find_cycles(tree, key_path=[], seen=[])


def raise_for_cycles(tree: Any):
    """Raise an informative error message if `tree` contains cycles."""

    cycles = find_cycles(tree)
    if cycles:
        raise ValueError(
            "Circular reference in args, kwargs, or context.\n"
            "Descendant refers to ancestor.\n"
            f"Descendant KeyPath: {cycles['descendant']}.\n"
            f"Ancestor KeyPath: {cycles['ancestor']}."
        )


@dataclasses.dataclass
class DeviceUsage:
    """Usage measurements for a device."""

    device_id: int
    device_duty_cycle_percent: Optional[float] = None
    device_utilization: Optional[float] = None
    hbm_memory_usage_bytes: Optional[int] = None
    hbm_memory_total_bytes: Optional[int] = None
    hbm_memory_bandwidth_utilization: Optional[float] = None


def sequence_mask(*, lengths: Tensor, max_len: int, dtype: jnp.dtype = jnp.bool) -> Tensor:
    """Computes a mask over sequence positions for each given length.

    Args:
        lengths: [...]. int32
        max_len: T, int
        dtype: outputs dtype.

    Returns:
        Tensor [..., T]. 1 is valid and 0 is padding.
    """
    prefix_axis = tuple(range(lengths.ndim))
    # [..., T]
    sequence = jnp.expand_dims(jnp.arange(max_len), axis=prefix_axis)
    # [..., 1]
    lengths = lengths[..., jnp.newaxis]
    return (sequence < lengths).astype(dtype)


def safe_not(mask: Tensor) -> Tensor:
    """Inverts a boolean mask.

    Commonly used to switch between paddings and mask.

    Args:
        mask: A boolean tensor.

    Returns:
        A boolean tensor of the same shape.
    """
    return ~(mask.astype(jnp.bool))


def validate_contains_paths(x: Nested[Tensor], paths: Sequence[str]):
    """Raises ValueError if any of the given `paths` are not present in `x`."""
    for path in paths:
        try:
            get_recursively(x, path)
        except KeyError as e:
            raise ValueError(
                f"Input is expected to contain '{path}'; "
                f"instead, it contains: '{jax.tree_util.tree_structure(x)}'."
            ) from e


def prune_empty(in_tree: Nested[Tensor]) -> Nested[Tensor]:
    """Returns a shallow copy of the input tree with empty subtrees pruned.

    If a tree would be made empty by removal of its subtrees, it will also be pruned.
    This is a shallow copy because leaf nodes (non-dict values) are not deep-copied.

    Args:
        in_tree: the input tree to be pruned.

    Returns:
        The pruned copy of the input tree.
    """
    # Note that falsey values or empty Tensors are not considered empty.
    return prune_tree(in_tree, lambda _, v: isinstance(v, dict) and not v)


def own_fields(cfg: ConfigBase) -> Sequence[str]:
    """Returns fields that are defined by `cfg`, rather than any of its ancestors."""

    bases = cfg.__class__.__bases__
    if len(bases) > 1:
        raise ValueError(f"Configs should not use multiple inheritance: {bases}")

    @cache
    def get_base_keys(base: type):
        return attr.fields_dict(base)

    base_keys = get_base_keys(bases[0])
    return [k for k in cfg.keys() if k not in base_keys]

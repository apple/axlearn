# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/jax:
# Copyright 2018 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Common utilities."""
import contextlib
import copy
import dataclasses
import functools
import numbers
import os
import re
import threading
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jax
import numpy as np
from absl import logging
from flax import serialization
from jax import numpy as jnp
from jax.experimental import maps, mesh_utils, multihost_utils, pjit
from jax.sharding import PartitionSpec
from jax.tree_util import register_pytree_node_class

from axlearn.common.config import is_named_tuple

Tensor = jax.Array
# Recursive type annotations not supported by pytype yet.
NestedTree = Union[Any, Dict[str, Any]]  # Union[Any, Dict[str, "NestedTree"]]
# Union[..., Dict[str, "NestedTensor"]]
NestedTensor = Union[Tensor, Dict[str, Any]]
# NestedPartitionSpec = Optional[Union[PartitionSpec, Dict[str, "NestedPartitionSpec"]]]
NestedPartitionSpec = Optional[Union[PartitionSpec, Dict[str, Any]]]

_enable_numeric_checks = False
_enable_xla_runtime_errors = False

# The set of supported floating point dtypes.
_supported_float_dtypes = [jnp.bfloat16, jnp.float32]


@dataclasses.dataclass
class TensorSpec:
    """Specification of a Tensor.

    Used to describe model parameters and optimizer states.
    """

    shape: Sequence[int]
    dtype: Optional[jnp.dtype] = None
    mesh_axes: Optional[PartitionSpec] = None

    @property
    def sharding(self) -> jax.sharding.Sharding:
        mesh = maps.thread_resources.env.physical_mesh
        return jax.sharding.NamedSharding(mesh, self.mesh_axes)


# NestedTensorSpec = Optional[Union[TensorSpec, Dict[str, "NestedTensorSpec"]]]
NestedTensorSpec = Optional[Union[TensorSpec, Dict[str, Any]]]


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
    return jax.tree_util.tree_map(lambda x: getattr(x, "shape", x), nested_tensor)


def _concat(*, prefix: str, suffix: str, separator: str):
    return f"{prefix}{separator}{suffix}" if prefix else f"{suffix}"


def tree_paths(tree: NestedTree, separator: str = "/") -> NestedTree:
    """Returns a tree of the same structure as `nested_tensor` but with corresponding paths instead
    of values.

    E.g.,
        tree_paths({'a': 1, 'b': [2, {'c': 3}]}) = {'a': 'a', 'b': ['b/0', {'c': 'b/1/c'}]}

    Args:
        tree: A nested structure.
        separator: The separator between parts of a path.

    Returns:
        A nested structure with the same structure as `tree`, but each leaf will be a string path.
        Note that None is not considered a leaf by jax.tree_util, hence also preserved by
        tree_paths.
    """

    def visit(tree, prefix):
        if tree is None:
            # None is considered part of the tree structure, not a tree leaf.
            return tree
        elif hasattr(tree, "items"):
            return type(tree)(
                (k, visit(v, _concat(prefix=prefix, suffix=k, separator=separator)))
                for k, v in tree.items()
            )
        elif is_named_tuple(tree):
            return type(tree)(**visit(tree._asdict(), prefix))
        elif isinstance(tree, (list, tuple)):
            return type(tree)(
                [
                    visit(v, _concat(prefix=prefix, suffix=k, separator=separator))
                    for k, v in enumerate(tree)
                ]
            )
        else:
            return prefix

    return visit(tree, "")


@dataclasses.dataclass
class PathAndValue:
    path: str
    value: Any


def flatten_items(tree: NestedTensor, separator="/") -> Sequence[Tuple[str, Tensor]]:
    """Flattens `tree` and returns a list of (path, value) pairs."""
    paths = tree_paths(tree, separator=separator)
    paths_and_values = jax.tree_util.tree_map(
        # pylint: disable-next=unnecessary-lambda
        lambda path, value: PathAndValue(path, value),
        paths,
        tree,
    )
    flat_paths_and_values, _ = jax.tree_util.tree_flatten(paths_and_values)
    return list((pv.path, pv.value) for pv in flat_paths_and_values)


@register_pytree_node_class
class VDict(dict):
    """A dict with Tensor leaf nodes whose values should be vectorized."""

    def __repr__(self):
        return f"VDict({super().__repr__()})"

    def tree_flatten(self):
        # Convert dict_values and dict_keys to lists to avoid holding reference to the VDict.
        return (list(self.values()), list(self.keys()))

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


# Register VDict as a dict for Flax serialization.
serialization.register_serialization_state(
    VDict,
    # pylint: disable-next=protected-access
    ty_to_state_dict=serialization._dict_state_dict,
    # pylint: disable-next=protected-access
    ty_from_state_dict=serialization._restore_dict,
)


def vectorized_tree_map(fn, tree, *rest):
    """Similar to jax.tree_util.tree_map(), but vectorizes `fn` on VDict's."""

    def vectorized_fn(*nodes):
        if isinstance(nodes[0], VDict):
            if not jax.tree_util.tree_leaves(nodes[0]):
                # This can happen when all VDict values are None and cause issues with jax.vmap.
                return nodes[0]
            nodes = [dict(**node) for node in nodes]
            result = jax.vmap(functools.partial(vectorized_tree_map, fn))(*nodes)
            return VDict(**result)
        return fn(*nodes)

    return jax.tree_util.tree_map(
        vectorized_fn, tree, *rest, is_leaf=lambda t: isinstance(t, VDict)
    )


class StackedKeyArray(NamedTuple):
    keys: Union[jax.random.KeyArray, "StackedKeyArray"]


def split_prng_key(
    prng_key: Union[StackedKeyArray, jax.random.KeyArray], num_keys: Union[int, Sequence[int]]
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

        return jax.tree_util.tree_map(verify_key_shape, prng_key)

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
        keys = jax.tree_util.tree_map(lambda x: x.reshape(list(num_keys) + list(x.shape[1:])), keys)
        return keys

    # pylint: disable-next=no-value-for-parameter
    keys = jax.tree_util.tree_map(stack_and_reshape, *child_prng_keys)

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
        return jax.tree_util.tree_map(as_tensor, x)
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
        return jax.tree_util.tree_map(as_numpy_array, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def with_sharding_constraint(x, shardings):
    mesh = jax.experimental.maps.thread_resources.env.physical_mesh  # type: ignore
    if mesh.empty or mesh.size == 1:
        return x
    return pjit.with_sharding_constraint(x, shardings)


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
        jax.tree_util.tree_map(add_leaves, partition_spec_tree_with_proxy, dummy)
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
    mesh = maps.thread_resources.env.physical_mesh
    return PartitionSpec(
        mesh.axis_names,
    )


def shard_input_batch(
    input_batch: NestedTensor, *, batch_axis_names: Union[str, Sequence[str]] = "data"
) -> NestedTensor:
    """Constrains all leaf values in the input batch to be partitioned over one axis.

    Args:
        input_batch: The inputs to be constrained.
        batch_axis_names: The name(s) of the batch axes.

    Returns:
        Inputs wrapped with sharding constraints.
    """
    return jax.tree_util.tree_map(
        lambda x: with_sharding_constraint(x, PartitionSpec(batch_axis_names)), input_batch
    )


class DataPartitionType(Enum):
    # Data are fully partitioned across all devices.
    FULL = "full"
    # Data are fully replicated across all devices.
    REPLICATED = "replicated"


def data_partition_type_to_spec(partition: DataPartitionType) -> PartitionSpec:
    """Returns a PartitionSpec for the given partition type."""
    if partition == DataPartitionType.FULL:
        return input_partition_spec()
    elif partition == DataPartitionType.REPLICATED:
        return None
    else:
        raise NotImplementedError(f"Unsupported partition: {partition}")


def host_to_global_device_array(
    host_arrays: NestedTensor, *, partition: DataPartitionType = DataPartitionType.FULL
) -> NestedTensor:
    """Converts the given host device arrays to global device arrays.

    Must be called within the context of a Mesh.

    We cannot use `multihost_utils.host_local_array_to_global_array` since the local mesh may not
    be contiguous. According to yashkatariya@google.com,
    "using `jax.make_array_from_single_device_arrays` is the right solution."

    Args:
        host_arrays: a nested tree of device arrays in host memory. Usually these present the
            per-host portion of the global input batch.
        partition: how the global array should be partitioned.

    Returns:
        A nested tree with the same structure as `host_arrays`, but global device arrays at the
        leaves. Each global device array is partitioned according to `partition`.

    Raises:
        NotImplementedError: if the given `partition` type is not supported.
    """
    mesh = maps.thread_resources.env.physical_mesh
    partition_spec = data_partition_type_to_spec(partition)

    local_devices = mesh.local_devices

    def put_to_devices_fully_partitioned(x: Tensor) -> List[Tensor]:
        len_local_devices = len(local_devices)
        if x.shape[0] % len_local_devices != 0:
            raise ValueError(f"({x.shape}) cannot be sharded across {len_local_devices} devices.")
        # np.reshape is faster than np.split, jnp.reshape, and jnp.split.
        xs = np.reshape(x, (len_local_devices, x.shape[0] // len_local_devices, *x.shape[1:]))
        return [jax.device_put(x_i, device) for x_i, device in zip(xs, local_devices)]

    def put_to_devices_replicated(x: Tensor) -> List[Tensor]:
        # Replicate `x` to every local device.
        return [jax.device_put(x, device) for device in local_devices]

    if partition == DataPartitionType.FULL:
        put_to_devices = put_to_devices_fully_partitioned
    elif partition == DataPartitionType.REPLICATED:
        put_to_devices = put_to_devices_replicated
    else:
        raise NotImplementedError(f"Unsupported partition: {partition}")

    device_arrays = jax.tree_util.tree_map(put_to_devices, host_arrays)
    partition_specs = complete_partition_spec_tree(
        jax.tree_util.tree_structure(host_arrays),
        partition_spec,
    )

    def make_gda(x, device_buffers, partition_spec):
        if partition == DataPartitionType.FULL:
            global_batch_size = x.shape[0] * jax.process_count()
        elif partition == DataPartitionType.REPLICATED:
            global_batch_size = x.shape[0]
        else:
            raise NotImplementedError(f"Unsupported partition: {partition}")
        global_shape = tuple([global_batch_size] + list(x.shape[1:]))
        return jax.make_array_from_single_device_arrays(
            shape=global_shape,
            sharding=jax.sharding.NamedSharding(mesh, partition_spec),
            arrays=device_buffers,
        )

    return jax.tree_util.tree_map(make_gda, host_arrays, device_arrays, partition_specs)


def global_to_host_array(
    global_arrays: NestedTensor, *, partition: DataPartitionType = DataPartitionType.FULL
) -> NestedTensor:
    """Extracts host addressable rows from each Tensor in `global_arrays`.

    Args:
        global_arrays: A NestedTensor.
            Each leaf Tensor must have shape [global_batch_size, ...] with identical
            global_batch_size across tensors.
            The tensors must be partitioned in the same way and can be partitioned only along the
            batch axis.
        partition: How the global array should be partitioned.

    Returns:
        A NestedTensor with the same structure as `global_array`. Each leaf Tensor will have shape
        [host_batch_size, ...] where `host_batch_size` will be equal to `global_batch_size` if the
        global Tensors are replicated or `global_batch_size // process_count` if the global Tensors
        are partitioned across hosts.
    """

    def sort_global_shards(global_shards: List[jax.Shard]) -> List[jax.Shard]:
        # We should sort jax.Array.global_shards by using this function to guarantee
        # round-trip equality of host_to_global_device_array and global_to_host_array.
        # Shards are sorted in-place.
        global_shards.sort(key=lambda shard: shard.index)
        return global_shards

    global_array_items = flatten_items(global_arrays)
    if not global_array_items:
        return global_arrays  # no leaf Tensor.
    first_path, first_value = global_array_items[0]
    sorted_first_value_shards = sort_global_shards(first_value.global_shards)
    first_value_shard_is_local = [shard.data is not None for shard in sorted_first_value_shards]
    batch_size = first_value.shape[0]

    def get_local_array(path: str, value: Tensor) -> Tensor:
        if value.shape[0] != batch_size:
            raise ValueError(
                f"Value batch size mismatch: {batch_size} @ {first_path} vs. "
                f"{value.shape[0]} @ {path} of {shapes(global_arrays)}"
            )
        sorted_value_shards = sort_global_shards(value.global_shards)
        value_shard_is_local = [shard.data is not None for shard in sorted_value_shards]
        if value_shard_is_local != first_value_shard_is_local:
            raise ValueError(
                f"Value shard mismatch: {first_value_shard_is_local} @ {first_path} vs. "
                f"{value_shard_is_local} @ {path}"
            )
        local_data = [shard.data for shard in sorted_value_shards if shard.data is not None]
        if not local_data:
            raise ValueError(f"No local shard found: {sorted_value_shards}.")
        if partition == DataPartitionType.FULL:
            return np.concatenate(local_data, axis=0)
        elif partition == DataPartitionType.REPLICATED:
            return local_data[0]
        else:
            raise NotImplementedError(f"Unsupported partition: {partition}")

    return jax.tree_util.tree_map(get_local_array, tree_paths(global_arrays), global_arrays)


def get_recursively(
    x: NestedTensor, path: Union[str, Sequence[str]], separator: str = "/"
) -> NestedTensor:
    """Recursively indexes through the nested tensor.

    Args:
        x: The tensor to index.
        path: The sequence of keys used to recursively
            index the nested tensor. If `isinstance(path, str)`, it will be split
            into sequence of strings based on the `separator`.
        separator: The delimiter to split ``path`` by if `isinstance(path, str)`.

    Returns:
        NestedTensor

    Raises:
        KeyError: If the input path is invalid.
    """
    if not path:
        return x
    is_str = isinstance(path, str)
    if is_str:
        path = path.split(separator)

    for idx, key in enumerate(path):
        if key not in x:
            prefix = separator.join(path[: idx + 1]) if is_str else path[: idx + 1]
            raise KeyError(f"No entries found at path '{prefix}'")
        x = x[key]

    return x


def set_recursively(
    x: NestedTensor, *, value: Tensor, path: Union[str, Sequence[str]], separator: str = "/"
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
        separator: The delimiter to split ``path`` by if `isinstance(path, str)`.

    Raises:
        ValueError: If the input path is empty.
    """
    if not path:
        raise ValueError("path must not be empty")
    if isinstance(path, str):
        path = path.split(separator)

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


def cast_floats(in_tree: NestedTensor, to_dtype: Optional[jnp.dtype]) -> NestedTensor:
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
        return jax.tree_util.tree_map(lambda x: x, in_tree)

    if to_dtype not in _supported_float_dtypes:
        raise ValueError(f"to_dtype must be one of {_supported_float_dtypes}")

    from_dtype = jnp.float32 if to_dtype == jnp.bfloat16 else jnp.bfloat16

    def cast(x: Tensor) -> Tensor:
        if x.dtype == from_dtype:
            return x.astype(to_dtype)
        return x

    return jax.tree_util.tree_map(cast, in_tree)


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
    param_shape_source = jax.tree_util.tree_map(lambda x: x.shape, source_tree)
    param_shape_target = jax.tree_util.tree_map(lambda x: x.shape, target_tree)
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
                msg += f"Argument f{name} has leaf with non-JAX type {type(value)}"
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
        out_tree = {}
        for k, v in in_tree.items():
            path = _concat(prefix=prefix, suffix=k, separator=separator)
            v = prune_tree(v, should_prune, prefix=path, separator=separator)
            if not should_prune(path, v):
                out_tree[k] = v
        in_tree = out_tree
    return in_tree


@dataclasses.dataclass
class DataDirStack(threading.local):
    """See `install_context_stack` on how to ensure thread-safety of the global stack."""

    stack: List[Optional[str]]


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


def get_or_none(x: Optional[Dict], key: Any) -> Optional[Any]:
    return None if x is None else x.get(key)


T = TypeVar("T")


def match_regex_rules(
    x: str, *, rules: Sequence[Tuple[str, T]], default_value: Optional[T] = None
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


def _register_per_param_settings(settings: NestedTree, *, description: str):
    del settings, description


def register_per_param_settings(settings: NestedTree, *, description: str):
    """Registers per-parameter setting.

    This function can be patched in testing to inspect per-param settings.
    """
    _register_per_param_settings(settings, description=description)
    if logging.vlog_is_on(1):
        for path, setting in flatten_items(settings):
            logging.info("Per-param setting %s: %s=%s", description, path, setting)
    return settings


def create_device_mesh(
    mesh_shape: Sequence[int], *, devices: Optional[Sequence[Any]] = None
) -> np.ndarray:
    """Constructs a device mesh.

    We first determine whether we are running in a TPU or GPU environment.
        - If running in a TPU environment:
            - If multi-slice/granule, we split the first axis of the configured
                mesh shape across the slices.
        - If running in a GPU environment:
            - If the first axis divides the number of processes (GPU-nodes/granules), we
                split the first axis across the processes.

    In all other cases we construct a standard mesh according to the configured mesh_shape.

    TODO(tom_gunter): Allow for more inter/intra granule mesh config flexibility.

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

    def build_standard_mesh():
        logging.info("Building device mesh.")
        try:
            return mesh_utils.create_device_mesh(mesh_shape, devices=devices)
        except NotImplementedError as e:
            logging.warning(
                "mesh_utils.create_device_mesh cannot handle shape %s: %s. "
                "Falling back to the naive mesh. Performance may be reduced.",
                mesh_shape,
                e,
            )
            return devices.reshape(mesh_shape)

    # Check if the devices are part of a multi-granule configuration.
    # <https://github.com/google/jax/blob/b81b79c1b0d2ec/jax/experimental/mesh_utils.py#L313>
    device_platform = devices[0].platform
    attr = "process_index" if device_platform != "tpu" else "slice_index"
    is_multi_granule_env = hasattr(devices[0], attr)
    if not all(el.platform == device_platform for el in devices):
        raise NotImplementedError(f"Not all devices had platform: {device_platform}.")

    # Return standard mesh if not a multi-slice/granule env.
    if not is_multi_granule_env:
        return build_standard_mesh()

    ici_mesh_shape = mesh_shape
    num_granules = max([getattr(el, attr) for el in devices.flatten()]) + 1

    # Return standard mesh if on GPU with incompatible multi-slice/granule mesh.
    if device_platform == "gpu" and ici_mesh_shape[0] % num_granules != 0:
        logging.warning("Falling back to ICI-only mesh on GPU, performance may be reduced.")
        return build_standard_mesh()

    # We only break the first device axis (the least communication intensive) across granules.
    assert (
        ici_mesh_shape[0] % num_granules == 0
    ), "First mesh shape axis must divide num slices/granules."
    logging.info("Building multi-slice/granule device mesh.")
    # Truncate intra-slice/granule mesh.
    ici_mesh_shape = (ici_mesh_shape[0] // num_granules, *ici_mesh_shape[1:])
    logging.info("Inferred intra-slice/granule mesh shape: %s", ici_mesh_shape)
    # Configure data center (inter-slice/granule) mesh.
    dcn_mesh_shape = (num_granules,) + (1,) * len(ici_mesh_shape[1:])
    logging.info("Inferred inter-slice/granule mesh shape: %s", dcn_mesh_shape)
    # Check we have the right number of devices.
    total_parallelism = np.product(dcn_mesh_shape) * np.product(ici_mesh_shape)
    assert total_parallelism == len(devices), (
        f"Num devices {len(devices)} does not match the product of "
        f"inter and intra slice/granule parallelism {total_parallelism}."
    )
    return mesh_utils.create_hybrid_device_mesh(
        ici_mesh_shape,
        dcn_mesh_shape=dcn_mesh_shape,
        devices=devices,
        process_is_granule=attr == "process_index",
    )

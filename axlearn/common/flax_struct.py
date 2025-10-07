# Copyright © 2024 Apple Inc.
#
# The code in this file is adapted from:
#
# google/flax:
# Copyright 2024 The Flax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Adapted from flax.struct with minor changes."""

import dataclasses
from typing import Any, Literal, TypeVar

import jax
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet

from axlearn.common import serialization, utils

_T = TypeVar("_T")


def _sort_fields(klass: _T) -> _T:
    """Sorts all dataclass fields of `klass` so that the order returned by
    `dataclasses.fields(klass)` is in ascended sorted order by field name.

    Pseudo-fields like `ClassVar` and `InitVar` are also sorted.

    This function mutates `klass`.
    """
    klass.__dataclass_fields__ = dict(
        sorted(klass.__dataclass_fields__.items(), key=lambda x: x[0])
    )


def field(pytree_node: bool = True, **kwargs):
    return dataclasses.field(metadata={"pytree_node": pytree_node}, **kwargs)


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def dataclass(klass: _T, flatten_order: Literal[None, "asc"] = "asc", **kwargs) -> _T:
    """Same as `flax.struct.dataclass`. See also `PyTreeNode`.

    Equivalent to `flax.struct.dataclass`.

    Args:
        klass: The class that will be transformed by the decorator.
        flatten_order: The sort order used when flattening this as a pytree. If None, no sorting.
        kwargs: Forwarded to `dataclasses.dataclass`.

    Returns:
        The new class.
    """
    # Check if already a `_dataclass`.
    if "_axlearn_dataclass" in klass.__dict__:
        return klass

    kwargs.setdefault("frozen", True)
    dataklass = dataclasses.dataclass(**kwargs)(klass)  # type: ignore
    if flatten_order == "asc":
        _sort_fields(dataklass)  # type: ignore
    dataklass.replace = dataclasses.replace

    # Data fields are fields that are transformable by jax. All other fields are meta fields, which
    # are not touched by jax transforms.
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(dataklass):
        if field_info.metadata.get("pytree_node", True):
            data_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    def flatten_func(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    def flatten_with_keys(x) -> tuple[tuple, tuple]:
        data = tuple((jax.tree_util.GetAttrKey(name), getattr(x, name)) for name in data_fields)
        meta = tuple(getattr(x, name) for name in meta_fields)
        return data, meta

    # Note that meta, data are tuples as produced by `flatten_with_keys`.
    def unflatten_func(meta: tuple, data: tuple):
        # Support unflattening from chex.dataclass which requires handling lists.
        data = tuple(data)
        return dataklass(**dict(zip(meta_fields + data_fields, meta + data)))

    jax.tree_util.register_pytree_with_keys(
        dataklass, flatten_with_keys, unflatten_func, flatten_func
    )

    def to_state_dict(x) -> utils.Nested[Any]:
        return {name: serialization.to_state_dict(getattr(x, name)) for name in data_fields}

    def from_state_dict(x, state: utils.Nested[Any]):
        # Shallow copy the state so we can pop the restored fields.
        state = state.copy()
        updates = {}
        for name in data_fields:
            if name not in state:
                raise ValueError(
                    f"Missing field {name} in state dict while restoring an instance of "
                    f"{klass.__name__}, at path {serialization.current_path()}"
                )
            value = getattr(x, name)
            value_state = state.pop(name)
            updates[name] = serialization.from_state_dict(value, value_state, name=name)
        if state:
            names = ",".join(state.keys())
            raise ValueError(
                f'Unknown field(s) "{names}" in state dict while restoring an instance of '
                f"{klass.__name__} at path {serialization.current_path()}"
            )
        return x.replace(**updates)

    serialization.register_serialization_state(dataklass, to_state_dict, from_state_dict)

    # Add a _axlearn_dataclass flag to distinguish from regular dataclasses.
    setattr(dataklass, "_axlearn_dataclass", True)

    return dataklass


_N = TypeVar("_N", bound="PyTreeNode")


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
class PyTreeNode:
    """Base class for dataclasses that should act like a JAX pytree node.

    Equivalent to `flax.struct.PyTreeNode`.
    """

    def __init_subclass__(cls):
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(type(self))

    def replace(self: _N, **overrides) -> _N:
        del overrides
        raise NotImplementedError(type(self))

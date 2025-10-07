# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Foundation module system for AXLearn.

Why do we need Module?
======================

1. JAX is functional - it has no built-in concept of stateful objects like layers or models.  Module
   provides the base class for all objects. The functional() method converts a Module instance into
   a pure JAX function.

2. Deep learning needs hierarchies - networks are composed of layers within layers.  Module provides
   _add_child() to form parent-child trees. For example:
   - A trainer contains a model, optimizer, and data loader
   - A model contains layers, which contain sub-layers

3. Hierarchical calls need shared context - when a model calls its layers, they all need access to
   the same PRNG keys, training mode, etc.  Module automatically wraps public methods to propagate
   InvocationContext, which carries:
   - PRNG keys (automatically split for each child)
   - Training/evaluation mode
   - State (parameters for neural layers)
   - Output collection (summaries, metrics)
   This avoids manually threading these through every method call.

What is a Module?
=================

A Module is a configurable, composable unit of computation that:
- Has a nested Config class holding its hyperparameters
- Maintains a frozen instance of this Config, preventing accidental changes
- Has a name and parent (parent=None for root modules)
- Can have child modules, forming a tree structure
- Automatically wrap public methods to take and propagate the parameter InvocationContext.
- Collects and propagates outputs (summaries, metrics, state updates)


An Example Module
=================

```
class MyModule(Module):
    @config_class
    class Config(Module.Config):
        dropout_rate: float = 0.1
        child: Module.Config = ...

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        # Calls cfg.child.instantiate(parent=self) to instantiate the child module.
        self._add_child("child", cfg.child)

    def _compute_scale(self, x: Tensor) -> float:
        # Private methods are NOT wrapped - no automatic context access.
        # Called directly during computation without context overhead.
        return jnp.sqrt(x.shape[-1])

    @nowrap
    def dropout_rate(self) -> float:
        # Public methods marked with @nowrap are NOT wrapped, so it
        # cannot access context (self.prng_key, self.is_training will fail).
        return self.config.dropout_rate

    def forward(self, x: Tensor) -> Tensor:
        # Public methods WITHOUT @nowrap ARE automatically wrapped.
        # This method receives InvocationContext and can access:

        # 1. Call private method (no context needed)
        scale = self._compute_scale(x)

        # 2. Call @nowrap-ed public method (no context)
        dropout_rate = self.dropout_rate()

        # 3. Access context properties (from InvocationContext)
        key = self.prng_key  # Automatically split PRNG key for this module
        is_training = self.is_training  # Training mode flag

        # 4. Call child module's wrapped method - context automatically propagates!
        # Child receives its own context with split PRNG key, same training mode, etc.
        x = self.child(x)

        # Apply dropout using context
        if is_training:
            keep_mask = jax.random.bernoulli(key, 1 - dropout_rate, x.shape)
            x = x * keep_mask / (1 - dropout_rate)

        return x * scale
```

Module automatically wraps `forward()` via `_wrap_method_with_auto_child_context` during
`__init__`. This allows parent modules to call `self.my_child.forward(x)` without manually
creating or passing InvocationContext - it propagates automatically through the module tree.

"""

import contextlib
import copy
import dataclasses
import functools
import hashlib
import inspect
import os.path
import re
import threading
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union

import jax
import numpy as np
from absl import logging
from typing_extensions import Protocol

from axlearn.common import flax_struct, traceback_util
from axlearn.common.config import REQUIRED, Configurable, Required, RequiredFieldValue, config_class
from axlearn.common.summary import Summary
from axlearn.common.traceback_util import annotate_stack, no_stack_summary
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    Tensor,
    partial_with_fn_metadata,
    prune_tree,
    raise_for_cycles,
)

_CallableT = TypeVar("_CallableT", bound=Callable)


def nowrap(fun: _CallableT) -> _CallableT:
    """Marks the specified module method as one that doesn't need to be wrapped.

    Methods decorated with `@nowrap` are helper methods that don't require wrapping, and
    `_methods_to_wrap_for_auto_child_context()` will not return them.

    This is especially useful in cases where a public method (i.e., one that is not explicitly
    prefixed with `_`) does not need an invocation context, such as methods that do not attempt
    to access state or PRNG keys.

    For instance::

        >>> from axlearn.common import module
        >>> class Foo(module.Module):
        ...   @module.nowrap
        ...   def init_states(self, batch_size: int):
        ...     return dict(time_step=jnp.zeros(batch_size, dtype=jnp.int32))

    Args:
        fun: The Module method to mark as nowrap.

    Returns:
        The given function ``fun`` marked as nowrap.
    """
    # pylint: disable-next=protected-access
    fun._nowrap = True
    return fun


def _generate_seed_from_name(name: str) -> np.int64:
    """Generates a random seed from a name string.

    Copied from
    https://github.com/tensorflow/lingvo/blob/3d16483b749a1181330ae9ce318688e7518d63c9/lingvo/jax/base_layer.py#L249-L260

    Args:
        name: A string.

    Returns:
        An integer seed in the range [0, 2**31 - 1).
    """
    md5 = hashlib.md5()
    md5.update(name.encode("utf-8"))
    return np.int64(int(md5.hexdigest(), 16) % (2**31 - 1))


_HOW_TO_CALL_MODULE_MULTIPLE_TIMES = (
    "To call the same module multiple times, "
    "create child contexts with distinct names with child_context(). "
    "See invoke_self_multiple_times and invoke_child_multiple_times in module_test.py for "
    "examples."
)


class OutputConflictError(ValueError):
    def __init__(self, error_message: str):
        super().__init__(f"{error_message}. {_HOW_TO_CALL_MODULE_MULTIPLE_TIMES}")


class ChildNameConflictError(ValueError):
    pass


class InvalidDescendantError(ValueError):
    pass


class OutputCollection(NamedTuple):
    """Implicit outputs from module invocations.

    Usually users do not interact with `OutputCollection` directly, but via
    `InvocationContext.add_{summary, state_update, module_output}()`.
    """

    summaries: NestedTensor
    state_updates: NestedTensor
    module_outputs: NestedTensor

    def __contains__(self, name: str) -> bool:
        return name in self.summaries

    def add_child(self, name: str) -> "OutputCollection":
        if not re.fullmatch("^[a-z][a-z0-9_]*$", name):
            raise ValueError(f'Invalid child name "{name}"')
        if name in self:
            raise OutputConflictError(f"{name} already present")
        child = new_output_collection()
        self.summaries[name] = child.summaries
        self.state_updates[name] = child.state_updates
        self.module_outputs[name] = child.module_outputs
        return child

    def update(self, collection: "OutputCollection"):
        self.summaries.update(**collection.summaries)
        self.state_updates.update(**collection.state_updates)
        self.module_outputs.update(**collection.module_outputs)


def new_output_collection():
    return OutputCollection(summaries={}, state_updates={}, module_outputs={})


def propagate_repeated_output_collections(
    repeated_output_collection: OutputCollection,
    *,
    child_name_prefix: str,
    target_output_collection: OutputCollection,
):
    """Propagates contents from `repeated_output_collection` to `target_target_output_collection`.

    Specifically:
    * module_outputs and state_updates from `repeated_output_collection` will be added to
      target_output_collection[child_name_prefix].
    * For each summary value `v` of `repeated_output_collection`, `v[i]` will be added to
      target_output_collection[f"{child_name_prefix}{i}"] for 0 <= i < N = num_children.

    Args:
        repeated_output_collection: An OutputCollection produced by a Jax loop (e.g., jax.vmap
            or jax.scan). Each leaf tensor has shape [N, ...].
        child_name_prefix: The child name prefix used for children to be added to
            `target_output_collection`.
        target_output_collection: The target OutputCollection.
    """
    # Fill `target_output_collection[child_name_prefix]` with `repeated_output_collection`.
    child_output = target_output_collection.add_child(child_name_prefix)
    child_output.module_outputs.update(**repeated_output_collection.module_outputs)
    child_output.state_updates.update(**repeated_output_collection.state_updates)

    # Each summary value in `repeated_output_collection` has shape (N, ...). For example,
    # if a repeated layer outputs a scalar summary value, it will have shape [N].
    # Below we split the stacked values and output them separately under scope
    # "{child_name_prefix}{i}" so that scalar summaries can be handled correctly.
    summary_values = jax.tree_util.tree_leaves(repeated_output_collection.summaries)
    if summary_values:
        first_summary_value = summary_values[0]
        assert first_summary_value.shape, "Stacked summaries should have a leading stack dimension."
        num_children = first_summary_value.shape[0]
        for i in range(num_children):
            child_i_output = target_output_collection.add_child(f"{child_name_prefix}{i}")
            child_i_output.summaries.update(
                jax.tree.map(lambda x, i=i: x[i], repeated_output_collection.summaries)
            )


T = TypeVar("T")


@typing.runtime_checkable  # Needed for isinstance checks to work.
class Summable(Protocol):
    # Objects of the same type which adhere to this protocol may be added.
    def __add__(self, other: T) -> T:
        ...


# TODO(markblee): Link to docs on invocation contexts.
@functools.partial(flax_struct.dataclass, frozen=False)
class InvocationContext:  # pylint: disable=too-many-instance-attributes
    """The invocation context for `Module.__call__()`.

    Attributes:
        name: The context name. Must be unique among sibling contexts.
        parent: The parent context, or None if `self` is the root context.
        module: The Module associated with the context.
        state: The state of the module.
        is_training: Whether the invocation should run in the training mode.
        prng_key: The pseudo-random number generator key (can be None if the computation does not
            require random numbers).
        output_collection: See `OutputCollection`.
    """

    name: str = flax_struct.field(pytree_node=False)
    parent: Optional["InvocationContext"] = flax_struct.field(pytree_node=True)
    module: Optional["Module"] = flax_struct.field(pytree_node=False)
    state: NestedTensor = flax_struct.field(pytree_node=True)
    is_training: bool = flax_struct.field(pytree_node=False)
    prng_key: Optional[Tensor] = flax_struct.field(pytree_node=True)
    output_collection: OutputCollection = flax_struct.field(pytree_node=True)

    def path(self):
        if self.parent is None:
            return self.name
        return self.parent.path() + "." + self.name

    # pylint: disable-next=too-many-branches
    def add_child(self, name: str, **override_kwargs) -> "InvocationContext":
        """Creates a child context with the given `name`.

        Args:
            name: The child context name. Must be unique among the siblings.
            override_kwargs: Overrides of the child context fields.

        Returns:
            The child context. Default values for fields not specified in `override_kwargs`:
            - `module` defaults to self.module.children[name];
            - `state` defaults to the state corresponding to child_context.module, that is:
              self.state[child_module.name] if child_context.module is a a child of self.module, or
              self.state if child_context.module is self.module;
            - `is_training` defaults to self.is_training;
            - `prng_key` defaults to fold_in(self.prng_key, hash(name));
            - `output_collection` defaults to self.output_collection.add_child(name).

        Raises:
            ValueError: if "parent" is specified in `override_kwargs`.
            NotImplementedError: if a field doesn't have a default value.
        """
        if "parent" in override_kwargs:
            raise ValueError("Overriding parent is not allowed")
        kwargs = {}  # type: dict[str, Any]
        for field in dataclasses.fields(self):
            k = field.name
            if k in override_kwargs:
                kwargs[k] = override_kwargs[k]
            elif k == "name":
                kwargs[k] = name
            elif k == "parent":
                kwargs[k] = self
            elif k == "module":
                # Defaults to the child module with the given `name`.
                kwargs[k] = self.module.children[name]
            elif k == "state":
                # Defaults to the state corresponding to child_context.module.
                module: Module = kwargs["module"]
                if module is self.module:
                    kwargs[k] = self.state
                elif module.parent is self.module:
                    kwargs[k] = self._get_child_state(kwargs["module"])
                else:
                    raise ValueError(
                        f"state must be specified explicit for module {module.path()} when "
                        f"creating a child context of {self.path()} (module={self.module.path()})"
                    )
            elif k == "is_training":
                kwargs[k] = self.is_training
            elif k == "prng_key":
                if self.prng_key is None:
                    kwargs[k] = None
                else:
                    kwargs[k] = jax.random.fold_in(self.prng_key, _generate_seed_from_name(name))
            elif k == "output_collection":
                kwargs[k] = self.output_collection.add_child(name)
            else:
                raise NotImplementedError(f"Missing default value for {k}")
        return InvocationContext(**kwargs)

    def _get_child_state(self, child: "Module") -> NestedTensor:
        # Module state can be None.
        if self.state is None:
            return None
        return self.state.get(child.name)

    def add_summary(
        self,
        name: str,
        value: Nested[Union[Summary, Tensor]],
    ):
        """Adds the named value to the `OutputCollection.summaries`.

        Args:
            name: The name of the item to add.
            value: The value to add.
        """

        def validate(leaf):
            if isinstance(leaf, Summary):
                leaf.validate()

        jax.tree.map(validate, value, is_leaf=lambda x: isinstance(x, Summary))

        self.output_collection.summaries[name] = value

    def add_state_update(self, name: str, value: Tensor):
        """Adds a state update to the output collection."""
        self.output_collection.state_updates[name] = value

    def add_module_output(self, name: str, value: Tensor):
        """Add module output to the output collection.

        Args:
            name: The output name.
            value: The output tensor.

        Raises:
            OutputConflictError: If outputs of method already exist.
        """
        if name in self.output_collection.module_outputs:
            raise OutputConflictError(f"Outputs of name '{name}' already exist")

        self.output_collection.module_outputs[name] = value

    def set_state_update(self, value: Any):
        """Sets the state update field of the output collection.

        Useful when writing a "transparent module" that needs its state to have a specific
        structure (e.g., a tuple) for backwards compatibility.
        """
        # E.g., optimizer state from a PartitionedGradientTransformation may be a tuple, so we
        # have to assign it via the parent.
        parent = self.parent
        if parent is not None:
            parent.output_collection.state_updates[self.name] = value
        self.output_collection = self.output_collection._replace(state_updates=value)

    def get_summaries(self):
        return self.output_collection.summaries

    def get_state_updates(self):
        return self.output_collection.state_updates

    def get_module_outputs(self):
        return self.output_collection.module_outputs

    def functional(self, method_fn: Callable) -> "_Functional":
        """Transforms `method_fn` (with this context) into a pure functional Callable.

        The returned Callable will have the same behavior as `method_fn`, except that it runs
        inside this context instead of the current context and returns
        an OutputCollection in addition to the method output instead of mutating the context it
        runs it.

        This context and the arguments to `method_fn` are not modified by the call.

        Args:
            method_fn: The function to call.

        Returns:
            The callable described above.
        """
        return _Functional(method_fn=method_fn, context=self, require_parent=False)


@dataclass
class ContextStack(threading.local):
    """See `install_context_stack` on how to ensure thread-safety of the global stack."""

    stack: list[InvocationContext]
    thread_id: int


_global_context_stack = ContextStack(stack=[], thread_id=threading.get_ident())


def clone_context_stack() -> list[InvocationContext]:
    """Returns a copy of the current InvocationContext stack.

    This is often used together with `install_context_stack` to ensure that different threads
    do not interfere with each other. See `install_context_stack`.
    """
    return list(_global_context_stack.stack)


def install_context_stack(stack: list[InvocationContext]):
    """Installs the given context stack.

    `install_context_stack` should be called in every child thread to ensure that each thread
    uses its own context stack:

    ```
    def my_thread(context_stack):
        install_context_stack(context_stack)
        ...

    t = Thread(target=my_thread, kwargs=dict(context_stack=clone_context_stack()))
    ```

    Args:
        stack: The context stack to install. This is usually returned by `clone_context_stack()`.

    Raises:
        ValueError: if the current thread id does not match the global context stack thread id.
    """
    current_thread_id = threading.get_ident()
    if _global_context_stack.thread_id == current_thread_id:
        raise ValueError("install_context_stack should only be called on a different thread")
    _global_context_stack.thread_id = current_thread_id
    _global_context_stack.stack = stack


def current_context() -> Optional[InvocationContext]:
    if not _global_context_stack.stack:
        return None
    return _global_context_stack.stack[-1]


@contextlib.contextmanager
def set_current_context(context: InvocationContext, *, require_parent: bool = True):
    if _global_context_stack.stack:
        cur_context = _global_context_stack.stack[-1]
        if context.parent is not cur_context and require_parent:
            raise ValueError(
                f"context ({context.path()})'s parent "
                f"must match the current context ({cur_context.path()}). "
                "Did you create the context via child_context()?"
            )
    else:
        if context.parent is not None:
            raise ValueError(
                "The first context in the stack must be a root context with parent=None. "
                "Usually a root context is created by module.functional()."
            )
    try:
        _global_context_stack.stack.append(context)
        if context.name != "remat":  # "remat" is already automatically added to JAX scope by JAX.
            with jax.named_scope(f"{context.name}[{context.module.__class__.__qualname__}]"):
                yield context
        else:
            yield context
    finally:
        _global_context_stack.stack.pop(-1)


@contextlib.contextmanager
def child_context(name: str, **kwargs):
    if _global_context_stack.thread_id != threading.get_ident():
        raise RuntimeError(
            "Each thread should call install_context_stack() to install its own stack."
        )
    context = current_context().add_child(name, **kwargs)
    with set_current_context(context) as c:
        yield c


@traceback_util.wrap
@no_stack_summary
def _call_method_in_context(
    module: "Module", *args, method_fn: Callable, method_name: str, **kwargs
):
    """Call the given method within the invocation context corresponding to `module` and passing
    it `args` and `kwargs`.

    Args:
        module: The module whose context the mnethod should be called in.
        *args: Positional arguments to `method_fn`.
        method_fn: The method to call.
        method_name: The name of the method to call.
        **kwargs: Keyword arguments to `method_fn`.

    Returns:
        The output of `method_fn(*args, **kwawrgs)` when called from within the invocation context
        of `module`.
    """
    if len(args) > 1:
        logging.log_first_n(
            logging.WARNING,
            "Multiple positional arguments for %s.%s. Consider using keyword arguments instead.",
            3,
            type(module),
            method_name,
        )

    # Use ExitStack since we need to repeatedly enter a context in a loop.
    # This cannot be done with a parenthesized context manager since, confusingly,
    # even though you can do something like `with (mgr1, mgr2)`,
    # it is not allowed to do `z = (mgr1, mgr2)` and then `with z`.
    # We prefer the ExitStack() approach over recursion since it does not add unnecessary frames to
    # the stack, which can make it harder to use a debugger with AXLearn.
    with contextlib.ExitStack() as stack:
        context = current_context()
        if context is not None:
            try:
                # Enter context for descendant module if not already in it.
                reversed_path_to_descendant = list(
                    reversed(context.module.path_to_descendant_module(module))
                )
                while reversed_path_to_descendant:
                    stack.enter_context(child_context(reversed_path_to_descendant.pop()))
            except InvalidDescendantError as e:
                # If an ancestor shared this module, use the shared module context since the module
                # may not be a descendant of the current module.
                try:
                    shared_module = context.module.get_shared_module(module)
                    stack.enter_context(child_context(**shared_module._asdict()))
                except InvalidDescendantError:
                    raise ValueError(
                        f"Module {module.path()} is not a descendant of {context.module.path()}, "
                        "nor does any ancestor share the module."
                    ) from e

        # pylint: disable-next=protected-access
        # Save call information on the stack so we can get this information from the traceback
        # object.
        method_fn = annotate_stack(
            module_call=True,
            module_type=type(module),
            method_name=method_name,
            arg_types=[type(a) for a in args],
            kwarg_types={k: type(v) for k, v in kwargs.items()},
        )(method_fn)
        return method_fn(module, *args, **kwargs)


class _PostInitMeta(type):
    """A metaclass that invokes `__post_init__`."""

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        instance = super().__call__(*args, **kwds)
        maybe_post_init = getattr(instance, "__post_init__", None)
        if callable(maybe_post_init):
            maybe_post_init()
        return instance


def _wrap_method_with_auto_child_context(*, method_fn: Callable, method_name: str) -> Callable:
    """Wraps a method by proxying through `_call_method_in_context`.

    Note that this does not bind any instance to the `self` parameter of the method.
    We keep this function separate from a `Module` method to avoid confounding the `self` argument
    of this function with the `self` argument in `wrap_method_fn`.

    Callers of this function should either bind the returned function to an instance, e.g. using
    `partial(method_fn, instance)`, or supply an instance explicitly as the first arg.
    """
    method_fn_in_context = functools.partial(
        _call_method_in_context, method_fn=method_fn, method_name=method_name
    )
    if not traceback_util.is_stack_summary_enabled():
        method_fn = functools.wraps(method_fn)(method_fn_in_context)
        return method_fn

    @no_stack_summary
    @functools.wraps(method_fn)
    def wrap_method_fn(self, *args, **kwargs):
        # Wrap method so it is called in a child context and add special handling of
        # TypeErrors to make it easier to see issues where a wrapped method is called
        # by the user with the wrong signature.
        try:
            return method_fn_in_context(self, *args, **kwargs)
        except TypeError as e:
            # Make it easier to see what call triggered the error in CI.
            # When running in an environment like TPUs where stack summaries are available,
            # this is unecessary and we would have slightly cleaner summaries without it.
            if getattr(e, "_handled", False):
                raise
            args_types = [type(arg) for arg in args]
            kwargs_types = {k: type(v) for k, v in kwargs.items()}
            new_exc = TypeError(
                f"Type error when calling {self}.{method_fn} "
                f"with args={args_types} and kwargs={kwargs_types}"
            )
            setattr(new_exc, "_handled", True)
            raise new_exc from e

    return wrap_method_fn


class Module(Configurable, metaclass=_PostInitMeta):
    """A node in a tree of Modules."""

    @config_class
    class Config(Configurable.Config):
        """Module config.

        Attributes:
            name: Name of this module.
            vlog: The maximum vlog level. If None, vlog is disabled.
        """

        name: Required[str] = REQUIRED
        vlog: Optional[int] = None

    def __init__(self, cfg: Config, *, parent: Optional["Module"]):
        super().__init__(cfg)
        cfg = self.config
        self._name = cfg.name
        self._parent = parent  # Avoid adding parent to self._modules.
        self._children: dict[str, "Module"] = {}
        # Mapping from descendant module name to relative path from current module.
        self._paths_to_shared_modules: dict[str, list[str]] = {}
        # Mapping from modules being shared by the current module, to the shared module name.
        self._shared_module_names: dict["Module", str] = {}
        self._vlog_level = cfg.vlog

    def __post_init__(self):
        # Wrap methods after `__init__`, allowing access to child modules.
        for method_name, method_fn in self._wrapped_methods_for_auto_child_context().items():
            setattr(self, method_name, method_fn)

    def _wrapped_methods_for_auto_child_context(self) -> dict[str, Callable]:
        """Returns methods that have been wrapped and bound to `self`.

        This ensures that module methods are bound to the instance that defined the method, rather
        than the instance that the method is assigned to in `__post_init__`.

        For example, `self.child._wrapped_methods_for_auto_child_context()` returns methods bound to
        `self.child` rather than `self`, which affects what `self.config` points to within the
        wrapped method.

        On the other hand, `self.child._methods_to_wrap_for_auto_child_context()` returns un-bound
        methods of `self.child`. Subclasses will typically override this method to control which
        methods of the subclass to wrap.
        """
        methods = self._methods_to_wrap_for_auto_child_context()
        return self._wrap_methods_with_auto_child_context(methods)

    def _wrap_methods_with_auto_child_context(
        self, methods: dict[str, Callable]
    ) -> dict[str, Callable]:
        """Wrap each method in `methods` with an auto child context.

        See `_wrapped_methods_for_auto_child_context()` for more details.

        Args:
            methods: The methods to wrap. Each key is the method name and each value is the
                method itself.

        Returns:
            The wrapped methods.
        """
        wrapped = {}

        for method_name, method_fn in methods.items():
            # method_fn is not bound to any instance.
            self.vlog(1, "Wrapping method %s of %s with auto child context", method_name, self)
            # Wrap method with automatic child context.
            method_fn = _wrap_method_with_auto_child_context(
                method_fn=method_fn, method_name=method_name
            )
            # Bind method_fn to self and override self.<method>.
            wrapped[method_name] = partial_with_fn_metadata(method_fn, self)

        return wrapped

    def _methods_to_wrap_for_auto_child_context(self) -> dict[str, Callable]:
        """Returns methods to be wrapped in `_wrapped_methods_for_auto_child_context`.

        These methods should not be bound to any instance (i.e., `self` should be left as a required
        first argument to the returned methods). Such a binding instead happens in
        `_wrapped_methods_for_auto_child_context`, which is invoked automatically in
        `__post_init__`.
        """

        def _should_wrap_method(method: str) -> bool:
            # Only public methods defined in subclasses of Module need to be wrapped.
            if hasattr(Module, method) or method.startswith("_"):
                return False
            fn = getattr(type(self), method, None)
            if not inspect.isfunction(fn):
                return False
            fn_sig = inspect.signature(fn)
            if "self" not in fn_sig.parameters:
                return False
            if hasattr(fn, "_nowrap"):
                return False
            return True

        return {
            method: getattr(type(self), method)
            for method in dir(self)
            if _should_wrap_method(method)
        }

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            try:
                return self.__dict__[name]
            except KeyError as e:
                raise AttributeError(name) from e
        else:
            try:
                return self._children[name]
            except KeyError as e:
                raise AttributeError(f"{name} is not an attribute or child of {self}") from e

    @property
    def parent(self):
        return self._parent

    @property
    def name(self):
        return self._name

    def path(self):
        if self.parent is None:
            return self.name
        return f"{self.parent.path()}.{self.name}"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self)}@{self.path()}"

    def vlog_is_on(self, level: int) -> bool:
        return self._vlog_level is not None and level <= self._vlog_level

    def vlog(self, level: int, msg: str, *args, **kwargs):
        if self.vlog_is_on(level):
            logging.info(f"@{self.path()} {msg}", *args, **kwargs)

    def vprint(self, level: int, msg: str, *args, **kwargs):
        """Prints debug info with if level <= config.vlog.

        Prints with jax.debug.print(prefix + msg, *args, **kwargs) so that it can handle tensors in
        args and kwargs, where prefix includes information about the time and caller.

        Args:
            level: The verbosity level of the print message.
            msg: The msg for jax.debug.print().
            *args: The args for jax.debug.print, may contain Tensors.
            **kwargs: The kwargs for jax.debug.print, may contain Tensors.
        """
        if self.vlog_is_on(level):
            caller_frame = inspect.stack()[1]  # Get the frame of the caller (index 1).
            filename = os.path.basename(caller_frame.filename)
            line_number = caller_frame.lineno
            prefix = f"{filename}:{line_number}] @{self.path()} "
            jax.debug.print(prefix + msg, *args, **kwargs)

    def _add_child(
        self,
        name: str,
        child_config: "Module.Config",
        **kwargs,
    ) -> "Module":
        """Adds a child module.

        Args:
            name: The child module name.
            child_config: The config of the child module.
            **kwargs: Additional kwargs for child module's __init__ method.

        Returns:
            The created child module.

        Raises:
            ValueError: upon invalid `name`.
            TypeError: if child_config is a Module.Config.
            ChildNameConflictError: if the child already exists.
        """
        if not re.fullmatch("^[a-z][a-z0-9_]*$", name):
            raise ValueError(f'Invalid child name "{name}"')
        child_config = copy.deepcopy(child_config)
        if not isinstance(child_config, Module.Config):
            raise TypeError(f"add_child expects a Module config, got {child_config}")
        if not isinstance(child_config.name, RequiredFieldValue) and child_config.name != name:
            raise ValueError(
                f"child_config already has a different name: {child_config.name} vs. {name}"
            )
        child_config.name = name
        module = child_config.instantiate(parent=self, **kwargs)
        if name in self._children:
            raise ChildNameConflictError(f"Child {name} already exists")
        self._children[name] = module
        return module

    def path_to_descendant_module(self, module: "Module") -> list[str]:
        """Returns the relative path from `self` to `module`.

        Args:
            module: The descendant module.

        Returns:
            A sequence of module names along the path from `self` to `module`.

        Raises:
            ValueError: if `module` is not a descendant of `self`.
        """
        relative_path = []
        while module is not None and module is not self:
            relative_path.append(module.name)
            module = module.parent
        path = list(reversed(relative_path))
        if module is None:
            raise InvalidDescendantError(
                f"Module at {'.'.join(path)} is not a descendant of {self.path()}"
            )
        return path

    def _share_with_descendants(self, module: "Module", *, shared_module_name: str):
        """Share `module` with self's descendant modules.

        Args:
            module: The module to share. Must be a descendant module of `self`.
            shared_module_name: The name under which to share. For each descendant module,
                the shared module can be retrieved via `get_shared_module(shared_module_name)`.
                It is an error to share multiple modules from the same Module under the same
                `shared_module_name`.
                If there are multiple ancestor Modules sharing modules under the same name, the
                nearest ancestor "wins".

        Raises:
            ValueError: if `module` is not a descendant of `self` or `shared_shared_module_name`
                is already taken.
        """
        # This checks that `module` is a descendant of `self`.
        relative_path = self.path_to_descendant_module(module)
        if shared_module_name in self._paths_to_shared_modules:
            raise ValueError(
                f"Shared module already exists under name {shared_module_name}: "
                f"{self._paths_to_shared_modules[shared_module_name]} vs. {relative_path}"
            )
        self._paths_to_shared_modules[shared_module_name] = relative_path
        self._shared_module_names[module] = shared_module_name

    class SharedModuleInfo(NamedTuple):
        name: str
        module: "Module"
        state: NestedTensor

    def get_shared_module(self, shared_module_or_name: Union["Module", str]) -> SharedModuleInfo:
        """Gets the shared module and state with the given name from a nearest ancestor.

        Shared modules should be registered via `_share_with_descendants`.

        Args:
            shared_module_name: The name of the shared module.

        Returns:
            The shared module and corresponding state.

        Raises:
            ValueError: if `shared_module_name` has not been shared by any ancestor.
        """
        # pylint: disable=protected-access
        context = self.get_invocation_context()

        def context_shares_module(ctx: InvocationContext) -> bool:
            if isinstance(shared_module_or_name, str):
                return shared_module_or_name in ctx.module._paths_to_shared_modules
            elif isinstance(shared_module_or_name, Module):
                return shared_module_or_name in ctx.module._shared_module_names
            raise ValueError(f"{shared_module_or_name=} must be a string or Module.")

        while context is not None and not context_shares_module(context):
            context = context.parent
        if context is None:
            raise InvalidDescendantError(
                f"Module '{self.path()}' does not have an ancestor that shares "
                f"{shared_module_or_name=}."
            )

        if isinstance(shared_module_or_name, Module):
            shared_module_or_name = context.module._shared_module_names[shared_module_or_name]
        assert isinstance(shared_module_or_name, str)

        target_module, target_state = context.module, context.state
        # pylint: disable-next=protected-access
        path_from_ancestor = context.module._paths_to_shared_modules[shared_module_or_name]
        for part in path_from_ancestor:
            if part not in target_module.children:
                raise InvalidDescendantError(
                    f"Module '{target_module.path()}' does not contain '{part}' from path "
                    f"'{'.'.join(path_from_ancestor)}'"
                )
            if part not in target_state:
                raise InvalidDescendantError(
                    f"Module '{target_module.path()}' state does not contain '{part}' from path "
                    f"'{'.'.join(path_from_ancestor)}'. The state contains: {target_state.keys()}"
                )
            target_module, target_state = target_module.children[part], target_state[part]

        return Module.SharedModuleInfo(
            module=target_module, state=target_state, name=shared_module_or_name
        )

    def get_invocation_context(self) -> InvocationContext:
        context = current_context()
        if not context:
            raise RuntimeError(
                "Module invocation context not found. "
                "Did you invoke the module inside functional(...)?"
            )
        if context.module is not self:
            raise RuntimeError(
                f"Module mismatch: {context.module} vs. {self}. "
                "Are the module methods wrapped for auto-child-context?"
            )
        return context

    @property
    def children(self) -> dict[str, "Module"]:
        return self._children

    @property
    def is_training(self) -> bool:
        return self.get_invocation_context().is_training

    @property
    def prng_key(self) -> Tensor:
        return self.get_invocation_context().prng_key

    @property
    def state(self):
        return self.get_invocation_context().state

    def add_summary(self, name: str, value: Union[Summable, Tensor, Summary]):
        """Adds the named value to `OutputCollection.summaries`.

        Args:
            name: The name of the item to add.
            value: The value to add.
        """
        return self.get_invocation_context().add_summary(name, value)

    def add_state_update(self, name: str, value: Tensor):
        """Adds a state update to the output collection of the current context."""
        return self.get_invocation_context().add_state_update(name, value)

    def add_module_output(self, name: str, value: Tensor):
        return self.get_invocation_context().add_module_output(name, value)

    def get_module_outputs(self):
        return self.get_invocation_context().get_module_outputs()

    @no_stack_summary
    def __call__(self, *args, **kwargs) -> Any:
        """A shortcut for self.forward(*args, **kwargs).

        Args:
            *args: positional args.
            **kwargs: keyword args.

        Returns:
            The method output.

        Raises:
            ValueError: If invoking from outside of an InvocationContext, or a context with invalid
                module path.
        """
        return self.forward(*args, **kwargs)


@functools.partial(flax_struct.dataclass, frozen=False)
class _Functional:
    """A pure functional call to `method_fn`."""

    # The function to call.
    method_fn: Callable = flax_struct.field(pytree_node=False)
    # The context to call method_fn in.
    # This will be copied to prevent method_fn from mutating the original.
    context: InvocationContext = flax_struct.field(pytree_node=True)
    # Whether to require that context.parent is current_context().
    require_parent: bool = flax_struct.field(pytree_node=False)
    # Whether to copy the argument pytrees to prevent method_fn from mutating the original.
    copy_args_tree: bool = flax_struct.field(pytree_node=False, default=True)

    def __call__(self, *args, **kwargs) -> tuple[Any, OutputCollection]:
        """Invokes method_fn in a pure functional fashion.

        The invocation will not depend on external inputs or have any side effects. The results only
        depend on the given inputs. All outputs are reflected in the return value.

        Args:
            *args: Positional arguments to method_fn.
            **kwargs: Keyword arguments to method_fn.

        Returns:
            (method_outputs, output_collection), where
            - method_outputs are the return value of the method.
            - output_collection is an OutputCollection containing summaries and state updates.

        Raises:
            ValueError: If there are circular references in args, kwargs, or context.
        """
        call = getattr(self.method_fn, "__qualname__", None) or getattr(self.method_fn, "__name__")
        logging.vlog(1, "functional: %s.%s (*%s, **%s)", call, self.method_fn, args, kwargs)

        # Some badly behaved tests call F() with an InvocationContext.state that contains
        # circular references.
        # This results in a cryptic error that doesn't make the root cause obvious.
        # So we raise a clearer error explicitly.
        raise_for_cycles(dict(context=self.context, args=args, kwargs=kwargs))
        context = self.context
        if self.copy_args_tree:
            context, args, kwargs = jax.tree.map(lambda x: x, (self.context, args, kwargs))

        with set_current_context(context, require_parent=self.require_parent):
            # pylint: disable-next=not-an-iterable,not-a-mapping,not-callable
            method_outputs = self.method_fn(*args, **kwargs)
        return method_outputs, context.output_collection


def functional(
    module: Module,
    prng_key: Optional[Tensor],
    state: NestedTensor,
    inputs: Union[Sequence[Any], dict[str, Any]],
    *,
    method: str = "forward",
    is_training: bool,
    drop_output_collections: Sequence[str] = ("module_outputs",),
    copy_args_tree: bool = True,
) -> tuple[Any, OutputCollection]:
    """Invokes <module>.<method> in a pure functional fashion.

    The invocation will not depend on external inputs or have any side effects. The results only
    depend on the given inputs. All outputs are reflected in the return value.

    Args:
        module: The Module to invoke.
        prng_key: The pseudo-random number generator key (can be None if the computation does not
            require random numbers).
        state: The input state of the module, including model parameters if the module contains a
            model.
        inputs: The inputs for the method. If it's a sequence, it represents the positional args.
            If it's a dict, it represents keyword args.
        method: The Module method to invoke.
        is_training: Whether the invocation should run in the training mode.
        drop_output_collections: The output collection types to drop.
            Defaults to dropping all module outputs.
        copy_args_tree: Whether to copy the `inputs` pytree to prevent method_fn from mutating the
            original. Defaults to True.

    Returns:
        (method_outputs, output_collection), where
        - method_outputs are the return value of the method.
        - output_collection is an OutputCollection containing summaries and state updates.

    Raises:
        ValueError: If there are circular references in args, kwargs, or context.
    """
    context = InvocationContext(
        name="root",
        parent=None,
        module=module,
        state=state,
        output_collection=new_output_collection(),
        is_training=is_training,
        prng_key=prng_key,
    )

    args = []
    kwargs = {}
    if isinstance(inputs, dict):
        kwargs = inputs
    else:
        args = inputs
    method_fn = getattr(module, method)

    fn = _Functional(
        context=context, method_fn=method_fn, require_parent=True, copy_args_tree=copy_args_tree
    )
    method_outputs, output_collection = fn(*args, **kwargs)

    for output_collection_type in drop_output_collections:
        getattr(output_collection, output_collection_type).clear()
    return method_outputs, output_collection


def scan_in_context(
    fn,
    *,
    carry: NestedTensor,
    xs: NestedTensor,
    drop_output: Optional[Callable[[str], bool]] = None,
    child_name_prefix: str = "iter",
    unroll: Union[int, bool] = 1,
    remat_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[NestedTensor, NestedTensor]:
    """A thin wrapper around `jax.lax.scan` which is compatible with `OutputCollection`.

    In particular, summaries and outputs added by `add_summary` and `add_module_output` respectively
    are accumulated in `current_context().output_collection`, subject to any output filtering.
    Specifically, summaries from iteration `i` will be placed in
    `summaries[f"{child_name_prefix}{i}"]`; module outputs will be stacked and placed in
    `module_outputs[child_name_prefix]`.

    Args:
        fn: A function with args (carry, x) returning a dict(carry=..., y=...).
        carry: The initial value of the loop carry, to be accumulated across scan.
        xs: A dict with at least "x" as a key, where each leaf is a tensor of shape
            [num_scan_iters, ...]. At scan iteration i:
            - xs["x"][i, ...] represents the inputs to `fn`.
            - xs[key][i, ...] is provided as a kwarg to the ith invocation context.
        drop_output: A callable that takes a path and outputs a decision of whether to drop the
            output at the given path, where True means we drop. By default, the callable is None,
            meaning nothing is dropped.
        child_name_prefix: The child name prefix used for children to be added to
            `target_output_collection`.
        unroll: If a positive integer is provided, it determines how many unrolled loop iterations
            to run within a single rolled iteration of the loop. If a boolean is provided, it will
            determine if the loop is competely unrolled (i.e. unroll=True) or left completely rolled
            (i.e. unroll=False).
        remat_kwargs: Optional dict passed to `jax.checkpoint` to enable rematerialization.
            Common options include:
              - `prevent_cse`: (bool) Whether to disable common subexpression elimination.
                    If left unspecified, defaults to False following recommendations from the JAX
                    documentation.
                    Raises a ValueError if `prevent_cse` is set to True.
              - `policy`: A checkpoint policy from `jax.checkpoint_policies`.
            If provided, the scan body will be wrapped as:
                `scan_fn = jax.checkpoint(scan_fn, **remat_kwargs)`
            Otherwise, `jax.checkpoint` is not used.
            See https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html.

    Returns:
        The scan outputs (carry, ys):
            - carry: A NestedTensor with the same structure as the input `carry`, representing its
                value at the final iteration.
            - ys: A NestedTensor with tensor leaves T of shape [num_scan_iters, ...], with T[i, ...]
                representing the `fn` outputs and output collection of the ith scan iteration,
                respesctively.

    Raises:
        ValueError: If `current_context()` is None, or if invalid remat_kwargs are passed.
    """

    ctx = current_context()
    if ctx is None:
        raise ValueError("Expected current_context() to not be None.")

    def scan_fn(carry_i: NestedTensor, scan_i: NestedTensor):
        output_collection_i = new_output_collection()
        x_i = scan_i.pop("xs")
        with child_context(
            "iter",
            module=ctx.module,
            output_collection=output_collection_i,
            **scan_i,
        ):
            carry_i, y_i = fn(carry_i, x_i)

        # Filter output collection.
        if drop_output is not None:
            pruned_collection_i = new_output_collection()._asdict()
            pruned_collection_i.update(
                prune_tree(
                    output_collection_i._asdict(),
                    lambda path, _: drop_output(path),
                )
            )
            output_collection_i = OutputCollection(**pruned_collection_i)

        return carry_i, dict(y_i=y_i, output_collection=output_collection_i)

    if remat_kwargs is not None:
        if "prevent_cse" in remat_kwargs:
            if remat_kwargs["prevent_cse"]:
                raise ValueError(
                    "`prevent_cse=True` is not recommended inside `jax.checkpoint` over `scan`."
                    "Set `prevent_cse=False` or omit the flag entirely."
                )
        else:
            remat_kwargs["prevent_cse"] = False
        scan_fn = jax.checkpoint(scan_fn, **remat_kwargs)
    carry, scan_ys = jax.lax.scan(scan_fn, init=carry, xs=xs, unroll=unroll)
    propagate_repeated_output_collections(
        scan_ys.pop("output_collection"),
        child_name_prefix=child_name_prefix,
        target_output_collection=ctx.output_collection,
    )

    return carry, scan_ys["y_i"]

# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Base class of modules.

Design choices:
* All module hyper-parameters are encapsulated by the module's config.
* Every module has a name, default dtype, and an optional param_init config.
* Every module has a parent except the root module.
* A module's config is frozen upon __init__. This prevents the config from being modified by
  accident.
* Module.config returns a copy of the module's config. This allows the caller to make changes
  without affecting the original config.

Module.__init__() wraps public methods of subclasses of Module to propagate child context
automatically. Specifically, suppose we have class MyModule with a method `do_foo`:

```
class MyModule(Module):
    def do_foo(self, ...):
        ...
```

`MyModule.__init__` will identify `MyModule.do_foo` as one of the methods to wrap through
`Module._methods_to_wrap_for_auto_child_context` (which can be overridden by subclasses, e.g.,
in RedirectToSharedModule). It will then wrap the method via
`Module._wrap_method_with_auto_child_context` and install the wrapped function as `self.do_foo`.

This allows MyModule's parents to invoke `do_foo` as `self.my_child.do_foo(...)` without having
to create the child context explicitly.
"""
import contextlib
import copy
import dataclasses
import hashlib
import inspect
import os.path
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union

import jax
import numpy as np
from absl import logging
from typing_extensions import Protocol

from axlearn.common import traceback_util
from axlearn.common.config import REQUIRED, Configurable, Required, RequiredFieldValue, config_class
from axlearn.common.summary import Summary
from axlearn.common.traceback_util import annotate_stack, no_stack_summary
from axlearn.common.utils import Nested, NestedTensor, Tensor, partial_with_fn_metadata, prune_tree


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
                jax.tree_util.tree_map(lambda x, i=i: x[i], repeated_output_collection.summaries)
            )


T = TypeVar("T")


class Summable(Protocol):
    # Objects of the same type which adhere to this protocol may be added.
    def __add__(self, other: T) -> T:
        ...


# TODO(markblee): Link to docs on invocation contexts.
@dataclass
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

    name: str
    parent: Optional["InvocationContext"]
    module: Optional["Module"]
    state: NestedTensor
    is_training: bool
    prng_key: Optional[Tensor]
    output_collection: OutputCollection

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
        kwargs = {}  # type: Dict[str, Any]
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

        jax.tree_util.tree_map(validate, value, is_leaf=lambda x: isinstance(x, Summary))

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


@dataclass
class ContextStack(threading.local):
    """See `install_context_stack` on how to ensure thread-safety of the global stack."""

    stack: List[InvocationContext]
    thread_id: int


_global_context_stack = ContextStack(stack=[], thread_id=threading.get_ident())


def clone_context_stack() -> List[InvocationContext]:
    """Returns a copy of the current InvocationContext stack.

    This is often used together with `install_context_stack` to ensure that different threads
    do not interfere with each other. See `install_context_stack`.
    """
    return list(_global_context_stack.stack)


def install_context_stack(stack: List[InvocationContext]):
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
def set_current_context(context: InvocationContext):
    if _global_context_stack.stack:
        cur_context = _global_context_stack.stack[-1]
        if context.parent is not cur_context:
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
    @no_stack_summary
    # Save call information on the stack so we can get this information from the traceback object.
    @annotate_stack(
        module_call=True,
        module_type=type(module),
        method_name=method_name,
        arg_types=[type(a) for a in args],
        kwarg_types={k: type(v) for k, v in kwargs.items()},
    )
    def thunk():
        # pylint: disable-next=protected-access
        return module._call_thunk(*args, method_fn=method_fn, **kwargs)()

    if len(args) > 1:
        logging.log_first_n(
            logging.WARNING,
            "Multiple positional arguments for %s.%s. Consider using keyword arguments instead.",
            3,
            type(module),
            method_name,
        )

    context = current_context()
    if context is None:
        return thunk()

    @no_stack_summary
    def call_thunk_in_context(reversed_path):
        if not reversed_path:
            return thunk()
        with child_context(reversed_path.pop()):
            return call_thunk_in_context(reversed_path)

    return call_thunk_in_context(list(reversed(context.module.path_to_descendant_module(module))))


class Module(Configurable):
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
        self._children: Dict[str, "Module"] = {}
        # Mapping from descendant module name to relative path from current module.
        self._paths_to_shared_modules: Dict[str, List[str]] = {}
        self._vlog_level = cfg.vlog
        # TODO(markblee): Consider using a metaclass.
        for method_name, method_fn in self._methods_to_wrap_for_auto_child_context().items():
            # method_fn is not bound to any instance.
            self.vlog(1, "Wrapping method %s of %s with auto child context", method_name, self)
            # Wrap method with automatic child context.
            method_fn = self._wrap_method_with_auto_child_context(
                method_fn=method_fn, method_name=method_name
            )
            # Bind method_fn to self and override self.<method>.
            method_fn = partial_with_fn_metadata(method_fn, self)
            setattr(self, method_name, method_fn)

    def _methods_to_wrap_for_auto_child_context(self) -> Dict[str, Callable]:
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
            return True

        return {
            method: getattr(type(self), method)
            for method in dir(self)
            if _should_wrap_method(method)
        }

    def _wrap_method_with_auto_child_context(self, *, method_fn, method_name):
        @no_stack_summary
        def wrap_method_fn(self, *args, method_fn=method_fn, **kwargs):
            return _call_method_in_context(
                self, *args, method_fn=method_fn, method_name=method_name, **kwargs
            )

        return wrap_method_fn

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

    def path_to_descendant_module(self, module: "Module") -> Optional[List[str]]:
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
        if module is None:
            raise ValueError(f"Module {module.path()} is not a descendant of {self.path()}")
        return list(reversed(relative_path))

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

    class SharedModuleInfo(NamedTuple):
        module: "Module"
        state: NestedTensor

    def get_shared_module(self, shared_module_name: str) -> SharedModuleInfo:
        """Gets the shared module and state with the given name from a nearest ancestor.

        Shared modules should be registered via `_share_with_descendants`.

        Args:
            shared_module_name: The name of the shared module.

        Returns:
            The shared module and corresponding state.

        Raises:
            ValueError: if `shared_module_name` has not been shared by any ancestor.
        """
        context = self.get_invocation_context()
        while (
            context is not None
            # pylint: disable-next=protected-access
            and shared_module_name not in context.module._paths_to_shared_modules
        ):
            context = context.parent
        if context is None:
            raise ValueError(
                f"Module '{self.path()}' does not have an ancestor that shares "
                f"'{shared_module_name}'."
            )
        target_module, target_state = context.module, context.state
        # pylint: disable-next=protected-access
        path_from_ancestor = context.module._paths_to_shared_modules[shared_module_name]
        for part in path_from_ancestor:
            if part not in target_module.children:
                raise ValueError(
                    f"Module '{target_module.path()}' does not contain '{part}' from path "
                    f"'{'.'.join(path_from_ancestor)}'"
                )
            if part not in target_state:
                raise ValueError(
                    f"Module '{target_module.path()}' state does not contain '{part}' from path "
                    f"'{'.'.join(path_from_ancestor)}'. The state contains: {target_state.keys()}"
                )
            target_module, target_state = target_module.children[part], target_state[part]
        return Module.SharedModuleInfo(module=target_module, state=target_state)

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
    def children(self) -> Dict[str, "Module"]:
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

    def _call_thunk(self, *args, method_fn, **kwargs) -> Callable[[], Any]:
        # Build nullary that that evaluates <method_fn(self, *args, **kwargs)> when called.
        @no_stack_summary
        def nullary():
            try:
                return method_fn(self, *args, **kwargs)
            except TypeError as e:
                args_types = [type(arg) for arg in args]
                kwargs_types = {k: type(v) for k, v in kwargs.items()}
                raise TypeError(
                    f"Type error when calling {self}.{method_fn} "
                    f"with args={args_types} and kwargs={kwargs_types}"
                ) from e

        return nullary


def functional(
    module: Module,
    prng_key: Optional[Tensor],
    state: NestedTensor,
    inputs: Union[Sequence[Any], Dict[str, Any]],
    *,
    method: str = "forward",
    is_training: bool,
    drop_output_collections: Sequence[str] = ("module_outputs",),
) -> Tuple[Any, OutputCollection]:
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

    Returns:
        (method_outputs, output_collection), where
        - method_outputs are the return value of the method.
        - output_collection is an OutputCollection containing summaries and state updates.
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

    method_fn = getattr(module, method)
    logging.vlog(1, "functional: %s.%s %s(%s)", module, method, method_fn, inputs)
    with set_current_context(context):
        if isinstance(inputs, dict):
            input_args, input_kwargs = [], inputs
        else:
            input_args, input_kwargs = inputs, {}
        method_outputs = method_fn(*input_args, **input_kwargs)

    for output_collection_type in drop_output_collections:
        getattr(context.output_collection, output_collection_type).clear()

    return method_outputs, context.output_collection


def scan_in_context(
    fn,
    *,
    carry: NestedTensor,
    xs: NestedTensor,
    drop_output: Optional[Callable[[str], bool]] = None,
    child_name_prefix: str = "iter",
) -> Tuple[NestedTensor, NestedTensor]:
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

    Returns:
        The scan outputs (carry, ys):
            - carry: A NestedTensor with the same structure as the input `carry`, representing its
                value at the final iteration.
            - ys: A NestedTensor with tensor leaves T of shape [num_scan_iters, ...], with T[i, ...]
                representing the `fn` outputs and output collection of the ith scan iteration,
                respesctively.
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

    carry, scan_ys = jax.lax.scan(scan_fn, init=carry, xs=xs)
    propagate_repeated_output_collections(
        scan_ys.pop("output_collection"),
        child_name_prefix=child_name_prefix,
        target_output_collection=ctx.output_collection,
    )

    return carry, scan_ys["y_i"]

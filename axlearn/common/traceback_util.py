# Copyright © 2023 Apple Inc.

"""This module provides a `wrap` decorator that can be used to print a human-readable
stack summary when an exception occurs in a wrapped Module method. This stack summary
omits things like `nullary()`, `thunk()`, and things filtered by JAX. It also provides additional
information about the type signature that Module methods were called with.

To enable this functionality, we need to be able to annotate the stack with information when
functions are called about whether they should be included in the stack summary and what
the type signature is for Module methods. This functionality is provided by a flexible
stack annotation system. We provide the decorator `annotate_stack()` that allows converting
a function into one whose stack frame is annotated with additional information. These annotations
can be retrieved from a traceback by using `walk_annotated_tb()`.

In comparison to JAX's stack filtering mechanism, JAX only supports removing lines from
the stack trace and requires the same treatment for all functions in the same file.

Here is a self-contained example demonstrating the usage of this module.
```
from axlearn.common import traceback_util
from axlearn.common.traceback_util import no_stack_summary, annotate_stack


# Any Exception encountered during calls to this wrapped method will generate a stack summary.
@traceback_util.wrap
def f():
    g()

# Omit the call to `g` from the stack summary. Calls that `g` makes to other functions will still be
# included unless they themselves are annotated with @no_stack_summary.
@no_stack_summary
def g():
    # This shows how to call a module method such that the type signature of the call is
    # included in the stack summary. To do so, define a thunk that is annotated with the
    # metadata for the call that you want to be included in the stack summary.
    @annotate_stack(
        module_call=True,
        module_type=object,
        method_name="h",
        arg_types=[int],
        kwarg_types={},
    )
    def thunk():
        return h(3)

    thunk()


def h(x: int):
    raise Exception(str(x))

# Since `h` raises an exception, calling `f` will cause a stack summary for this Exception to be
# displayed.
f()
```
"""
import builtins
import functools
import linecache
import os
import sys
import traceback
import types
from collections.abc import Iterator
from typing import Any, Callable, Optional

import jax._src.traceback_util
from absl import logging


def is_stack_summary_enabled() -> bool:
    """Check AXLEARN_ENABLE_STACK_SUMMARY env variable enabled.

    AXLEARN_ENABLE_STACK_SUMMARY env variable controls the wrapping mechanism. By default,
    it's enabled, which wraps functions such as traceback, remat, and exception handling.

    Returns:
        Boolean indicating whether stack summary is enabled or not.

    Raises:
        ValueError: if the env variable is not "true" or "false"
    """
    enable = os.getenv("AXLEARN_ENABLE_STACK_SUMMARY", "true")
    if enable not in ["true", "false"]:
        raise ValueError("AXLEARN_ENABLE_STACK_SUMMARY must be 'true', 'false' or unset.")
    return enable == "true"


def wrap(fn: Callable) -> Callable:
    """Wraps the given function so that if an Exception `e` is encountered, it has an Exception
    constructed with `e` as the cause. The value of `e._stack_summary` is
    set to the generated exception, and `e` is reraised.

    We set `sys.excepthook` such that when an Exception with a `_stack_summary` attribute is
    encountered, the `_stack_summary` gets printed. This means that code that is unaware
    of stack summaries can still catch the original Exception normally.

    If a wrapped function calls other wrapped functions, the wrapper on the inner functions
    will be ignored. This behavior is similar to that of JAX's traceback filtering wrappers.

    Args:
        fn: The function to wrap.

    Returns:
        The wrapped function.
    """
    if not is_stack_summary_enabled():
        return fn

    return _InContextException.wrap(fn)


class _InContextException(Exception):
    """This is used to print a stack summary about an Exception encountered during a call to a
    wrapped Module method.

    The error message provided by this exception includes a stack summary with information
    about what arguments Module methods were called with. Functions decorated with
    `@no_stack_summary` are elided. Functions hidden from the stack trace by JAX are also
    elided.

    The original raw stack trace is still viewable above the stack summary.

     Example of what the stack summary looks like:
     ```
     Stack Summary (most recent call last):
       [...]
       File "test.py", line 35, in <module>
         jax.jit(F, static_argnames=['is_training'], static_argnums=0)(layer,
       File "test.py", line 761, in functional
         method_outputs = method_fn(*input_args, **input_kwargs)
       Wrapped call __main__.OuterLayer.forward()
       File "test.py", line 21, in forward
         self.sublayer()
       Wrapped call __main__.SubLayer.forward()
       File "test.py", line 26, in forward
         self.do_something(
       Wrapped call __main__.SubLayer.do_something(str)
       File "test.py", line 31, in do_something
         raise TypeError("oops")
     TypeError: oops
     ```
    """

    @classmethod
    def wrap(cls, fn: Callable) -> Callable:
        """Wraps the given function so that if an Exception `e` is encountered, it has an instance
        of _InContextException constructed with `e` as the cause. The value of `e._stack_summary` is
        set to the generated exception, and `e` is reraised.

        If a wrapped function calls other wrapped functions, the wrapper on the inner functions
        will be ignored. This behavior is similar to that of JAX's traceback filering wrappers.

        Args:
            fn: The function to wrap.

        Returns:
            The wrapped function.
        """

        # We create a wrapper that calls the original function and handles any exceptions
        # such that a stack summary will be generated.
        @no_stack_summary
        @functools.wraps(fn)
        def in_context_exception_wrapper(*args, **kwargs):
            # pylint: disable=used-before-assignment
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                try:
                    raise Exception()  # pylint: disable=raise-missing-from,broad-exception-raised
                except Exception as local:  # pylint: disable=broad-exception-caught
                    tb = local.__traceback__
                # We now walk through the stack above the current call to check if we are the
                # outermost in_context_exception_wrapper. If so, we generate the stack summary.
                # If not, we do not since the outermost one already will.
                own_code = tb.tb_frame.f_code
                parent_frame = tb.tb_frame.f_back
                if all(
                    own_code is not frame.f_code for frame, _ in traceback.walk_stack(parent_frame)
                ):
                    in_context_exception = cls()
                    try:
                        # Raise it so that it gets its __traceback__ set.
                        raise in_context_exception from e
                    except cls as in_context_exception_copy:
                        # The __traceback__ attribute is only set on _in_context_exception_copy.
                        in_context_exception = in_context_exception_copy
                    e._stack_summary = in_context_exception  # pylint: disable=protected-access
                # We use raise with no arguments to exclude the outermost
                # in_context_exception_wrapper from the traceback of e which also excludes it
                # from the stack summary. Note that @no_stack_summary does not affect the
                # outermost call because the traceback stops before the annotation wrapper
                # for the outermost call.
                raise

        return in_context_exception_wrapper

    def __str__(self):
        return (
            "\n\nAn error was encountered in a wrapped Module method.\n"
            "Below is an AXLearn stack summary, which may be easier to read.\n"
            "Immediately above is the stack frame in which the stack summary was "
            "initialized.\n"
            "You can probably ignore that frame.\n"
            "Further above that is the original Python Exception with the raw stack "
            "trace, which caused this Exception.\n\n" + self._format_summary()
        )

    def _format_summary(self) -> str:
        """Returns a formatted stack summary."""
        tb = self.__cause__.__traceback__
        tb = tb.tb_next  # Don't include the frame where InContextException was raised.
        tb = jax._src.traceback_util.filter_traceback(tb)  # pylint: disable=protected-access

        lines = ["Stack Summary (most recent call last):"]

        # Returns fully qualified name of cls.
        def fqname(cls):
            output = cls.__qualname__
            if cls.__module__ != "builtins":
                output = cls.__module__ + "." + output
            return output

        for frame, aux in _walk_annotated_tb(tb):
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            line = linecache.getline(filename, lineno).strip()
            if aux.get("module_call", False):
                module = fqname(aux["module_type"])
                args = ", ".join(fqname(arg) for arg in aux["arg_types"])
                kwargs = ", ".join(
                    f"{key}: {fqname(value)}" for key, value in aux["kwarg_types"].items()
                )
                full_args = ", ".join(arg for arg in [args, kwargs] if arg != "")
                lines.append(f'  Wrapped call {module}.{aux["method_name"]}' f"({full_args})")
            # Check value to see if function has _stack_summary set.
            if aux.get("include_stack_summary", True):
                lines.append(f'  File "{filename}", line {lineno}, ' f"in {frame.f_code.co_name}")
                lines.append(f"    {line}")
        lines.extend(traceback.format_exception_only(type(self.__cause__), self.__cause__))
        return "\n".join(lines)


def no_stack_summary(fn: Callable) -> Callable:
    """A decorator that wraps `fn` so that calls to it are excluded from the stack summary generated
    by `wrap`.

    The function `fn` can still be viewed in the raw stack trace that is printed when `wrap` is
    used.

    Args:
        fn: The function to mark.

    Returns:
        A wrapped version of `fn` that will be excluded from the stack summary.
    """

    return annotate_stack(include_stack_summary=False)(fn)


# We need to resort to this because Python provides no reliable method of getting
# a function object from a traceback. Although Python provides various fixed metadata about the
# function, there is no way of getting the function object itself that correctly handles anonymous
# functions. Otherwise, we could just set an attribute on the function.
def annotate_stack(**aux) -> Callable:
    """A decorator that creates a wrapper around `fn` that annotates the call with annotations given
    by the supplied keyword arguments.

    These annotations can be retrieved from a traceback using `walk_annotated_tb()`.

    Example:
        ```
        fn = lambda x: x
        annotated_fn = annotate_stack(my_annotation="Hello, world!")(fn)
        ```

    Args:
        aux: The auxiliary data with which to annotate calls to the wrapped function.

    Returns:
        A wrapped function that calls `fn` after annotating the call with `aux`.
    """

    if not is_stack_summary_enabled():
        return lambda fn: fn

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def stack_annotation_wrapper(*args, **kwargs):
            # pylint: disable=unused-variable
            aux_ = aux
            return fn(*args, **kwargs)

        return stack_annotation_wrapper

    return decorator


def _is_annotation_frame(frame: types.FrameType) -> bool:
    """Returns whether the given frame is an annotation frame.

    Args:
        frame: The frame to check.

    Returns:
        Whether `frame` corresponds to a `stack_annotation_wrapper` generated by `annotate_stack`.
    """
    code = annotate_stack()(lambda: None).__code__
    return frame.f_code is code


def _walk_annotated_tb(tb: types.TracebackType) -> Iterator[tuple[types.FrameType, dict[str, Any]]]:
    """Similar to `traceback.walk_tb`, except that annotation frames are elided
    and the iterator returns a tuple of (frame, aux) instead of (frame, lineno).

    The `aux` value is obtained from the immediately preceding contiguous sequence of annotation
    stack frames. If the immediately preceding stack frame is not an annotation stack frame,
    `aux` is an empty dict.

    For example,
    ```
    @annotate_stack(a=3)
    @annotate_stack(b=4)
    def f()
        pass
    ```
    results in the call to `f` having `aux=dict(a=3,b=4)`.

    Args:
        tb: The traceback to walk.

    Returns:
        A generator yielding `(frame, aux)` pairs.
    """
    aux = {}
    for frame, _ in traceback.walk_tb(tb):
        if _is_annotation_frame(frame):
            aux.update(frame.f_locals["aux_"])
            continue
        else:
            yield frame, aux
            aux = {}


# The excepthook patch below is needed because we don't want to replace the actual exception with
# the InContextException that wraps it until right before the exception is printed. The reason
# for this is that if we replace it earlier, then try/except statements that expect the original
# exception type would break.
#
# We have the `type` argument shadow the builtin name because that is what Python's excepthook does.
def _excepthook(
    # pylint: disable=redefined-builtin,redefined-outer-name
    type: type[BaseException],
    value: BaseException,
    traceback: types.TracebackType,
    old_excepthook: Optional[Callable] = sys.excepthook,
) -> Any:
    """Calls the original `sys.excepthook` with `value`. If `value` has a `_stack_summary`
    attribute, the original `sys.excepthook` is called with the value, type, and traceback of
    that attribute instead.

    Args:
        type: The type of the exception to handle.
        value: The exception to handle.
        traceback: The traceback of the exception to handle.
        old_excepthook: The old excepthook to call. Defaults to the value of `sys.excepthoook`
                        at the time this module was first initialized.

    Returns:
        The output from the call to the original `sys.excepthook`.
    """
    if hasattr(value, "_stack_summary"):
        stack_summary = value._stack_summary  # pylint: disable=protected-access
        return old_excepthook(
            builtins.type(stack_summary), stack_summary, stack_summary.__traceback__
        )
    return old_excepthook(type, value, traceback)


logging.debug("traceback_util: patching excepthook")
sys.excepthook = _excepthook

# IPython doesn't use excepthook.
try:
    from IPython import get_ipython  # type: ignore

    def _ipython_exception_handler(self, etype, evalue, tb, tb_offset=None):
        _excepthook(
            etype,
            evalue,
            tb,
            old_excepthook=lambda *args: self.showtraceback(args, tb_offset=tb_offset),
        )

    _ipython = get_ipython()
    if _ipython is not None:
        _ipython.set_custom_exc((Exception,), _ipython_exception_handler)
except ModuleNotFoundError:
    pass

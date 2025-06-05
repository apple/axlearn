# Copyright © 2024 Apple Inc.

"""A file system interface supporting multiple cloud environments.

Several libraries exist today to abstract file system management, including general purpose
libraries like `tf.io` and `fsspec`, or specific libraries like `google-cloud-storage` SDK.

This module provides a common interface between these options.
"""
# TODO(markblee): Consider supporting a pathlib-like interface.

import contextlib
import functools
import os
from typing import IO, Sequence, TypeVar, Union
from urllib.parse import urlparse

import tensorflow as tf
from absl import logging
from tensorflow import errors as tf_errors


@contextlib.contextmanager
def _wrap_exception(
    source_exc: Union[type[Exception], tuple[type[Exception], ...]], target_exc: type[Exception]
):
    """Surfaces any `source_exc` as `target_exc` instead.

    Users can use multiple contexts to wrapping different `target_exc`, e.g:
    ```
    with (
        _wrap_exception(tf_errors.OpError, OpError),
        _wrap_exception(tf_errors.NotFound, NotFoundError),
    ):
        ...
    ```

    Note that contexts are entered in order and exited in reverse order, so more general exceptions
    should be wrapped first.

    Args:
        source_exc: One or more exceptions to catch.
        target_exc: Exception to raise instead.
    """
    try:
        yield
    except source_exc as e:
        raise target_exc(str(e)) from e


class OpError(Exception):
    """FileSystem-level error."""


class NotFoundError(OpError):
    """Resource not found error."""


_F = TypeVar("_F")


# Wrapping errors allows us to change implementations without requiring callsites to change
# exception handling behavior.
def _wrap_tf_errors(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with (
            _wrap_exception(tf_errors.OpError, OpError),
            _wrap_exception(tf_errors.NotFoundError, NotFoundError),
        ):
            return fn(*args, **kwargs)

    return wrapped


@functools.cache
def _gs_client():
    """Creates or retrieves the GCS client."""
    # pylint: disable=import-outside-toplevel
    # Avoid introducing a global dependency on `gcp`.
    from google.cloud import storage  # pytype: disable=import-error

    from axlearn.cloud.gcp.config import gcp_settings
    from axlearn.cloud.gcp.utils import get_credentials

    return storage.Client(project=gcp_settings("project"), credentials=get_credentials())


@_wrap_tf_errors
def isdir(path: str) -> bool:
    """Analogous to tf.io.gfile.isdir."""
    return tf.io.gfile.isdir(path)


@_wrap_tf_errors
def listdir(path: str) -> list[str]:
    """Analogous to tf.io.gfile.listdir."""
    return tf.io.gfile.listdir(path)


@_wrap_tf_errors
def glob(pattern: Union[str, Sequence[str]]) -> list[str]:
    """Analogous to tf.io.gfile.glob."""
    if isinstance(pattern, (list, tuple)):
        results = set()
        for p in pattern:
            results.update(glob(p))
        return list(results)

    parsed = urlparse(pattern)

    # tf.io.gfile.glob is prohibitively slow for gs paths containing many prefixes.
    if parsed.scheme == "gs":
        # https://googleapis.github.io/google-api-python-client/docs/dyn/storage_v1.objects.html#list
        try:
            # pylint: disable=import-outside-toplevel
            from google.api_core.exceptions import GoogleAPIError
            from google.api_core.exceptions import NotFound as GoogleAPINotFound

            client = _gs_client()
        except (ModuleNotFoundError, ImportError, RuntimeError, SystemExit) as e:
            logging.warning("Failed to initialize gs client: %s. Falling back to slow glob.", e)
        else:
            bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
            # TODO(markblee): gsutil/gcloud storage seem to have custom logic for handling wildcards
            # which is even more performant, but introduces extra complexity.
            with (
                _wrap_exception(GoogleAPIError, OpError),
                _wrap_exception(GoogleAPINotFound, NotFoundError),
            ):
                return [
                    os.path.join(f"gs://{bucket}", blob.name)
                    for blob in client.list_blobs(bucket_or_name=bucket, match_glob=prefix)
                ]

    return tf.io.gfile.glob(pattern)


@_wrap_tf_errors
def exists(path: str) -> bool:
    """Analogous to tf.io.gfile.exists."""
    return tf.io.gfile.exists(path)


@_wrap_tf_errors
def remove(path: str):
    """Analogous to tf.io.gfile.remove."""
    return tf.io.gfile.remove(path)


@_wrap_tf_errors
def copy(src: str, dst: str, overwrite: bool = False):
    """Analogous to tf.io.gfile.copy."""
    return tf.io.gfile.copy(src, dst, overwrite=overwrite)


@_wrap_tf_errors
# pylint: disable-next=redefined-builtin
def open(path: str, mode: str = "r") -> IO:
    """Analogous to tf.io.gfile.GFile."""
    return tf.io.gfile.GFile(path, mode)


@_wrap_tf_errors
def readfile(path: str) -> str:
    with open(path, mode="r") as f:
        return str(f.read())


@_wrap_tf_errors
def makedirs(path: str):
    """Analogous to tf.io.gfile.makedirs."""
    return tf.io.gfile.makedirs(path)


@_wrap_tf_errors
def rmtree(path: str):
    """Analogous to tf.io.gfile.rmtree."""
    return tf.io.gfile.rmtree(path)

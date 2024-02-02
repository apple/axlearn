# Copyright © 2023 Apple Inc.

"""Utilities for testing GCP tooling."""

import contextlib
from typing import Any, Dict, Optional, Sequence, Union
from unittest import mock


@contextlib.contextmanager
def mock_gcp_settings(module_name: Union[str, Sequence[str]], settings: Dict[str, str]):
    def gcp_settings(key: str, default: Optional[Any] = None, required: bool = True):
        value = settings.get(key, default)
        if required and value is None:
            raise ValueError(f"{key} is required")
        return value

    if isinstance(module_name, str):
        module_name = [module_name]

    mocks = [mock.patch(f"{m}.gcp_settings", side_effect=gcp_settings) for m in module_name]
    with contextlib.ExitStack() as stack:
        # Boilerplate to register multiple mocks at once.
        for m in mocks:
            stack.enter_context(m)
        yield

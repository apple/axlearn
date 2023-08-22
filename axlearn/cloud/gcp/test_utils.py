# Copyright © 2023 Apple Inc.

"""Utilities for testing GCP tooling."""

import contextlib
from typing import Any, Dict, Optional
from unittest import mock


@contextlib.contextmanager
def mock_gcp_settings(module_name: str, settings: Dict[str, str]):
    def gcp_settings(key: str, default: Optional[Any] = None, required: bool = True):
        value = settings.get(key, default)
        if required and value is None:
            raise ValueError(f"{key} is required")
        return value

    with mock.patch(f"{module_name}.gcp_settings", side_effect=gcp_settings):
        yield

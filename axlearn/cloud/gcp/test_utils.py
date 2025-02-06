# Copyright © 2023 Apple Inc.

"""Utilities for testing GCP tooling."""

import contextlib
from collections.abc import Sequence
from typing import Any, Optional, Union
from unittest import mock

from absl import flags

from axlearn.cloud.gcp import config


@contextlib.contextmanager
def mock_gcp_settings(module_name: Union[str, Sequence[str]], settings: dict[str, str]):
    def gcp_settings(
        key: str,
        *,
        fv: Optional[flags.FlagValues] = None,
        default: Optional[Any] = None,
        required: bool = True,
    ):
        del fv
        value = settings.get(key, default)
        if required and value is None:
            raise ValueError(f"{key} is required")
        return value

    def gcp_settings_from_active_config(project_or_zone: str):
        return settings.get(project_or_zone, None)

    if isinstance(module_name, str):
        module_name = [module_name]

    mocks = [mock.patch(f"{m}.gcp_settings", side_effect=gcp_settings) for m in module_name]
    if "project" in settings or "zone" in settings:
        mocks.append(
            mock.patch(
                f"{config.__name__}._gcp_settings_from_active_config",
                side_effect=gcp_settings_from_active_config,
            ),
        )

    with contextlib.ExitStack() as stack:
        # Boilerplate to register multiple mocks at once.
        for m in mocks:
            stack.enter_context(m)
        yield

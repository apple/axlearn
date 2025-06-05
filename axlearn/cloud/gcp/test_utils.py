# Copyright © 2023 Apple Inc.

"""Utilities for testing GCP tooling."""

import contextlib
from collections.abc import Sequence
from typing import Any, Optional, Union
from unittest import mock

from absl import flags

from axlearn.cloud.gcp import config


def default_mock_settings() -> dict[str, str]:
    """Default settings to use in tests."""

    return {
        "project": "settings-project",
        "env_id": "settings-env-id",
        "zone": "settings-zone",
        "permanent_bucket": "settings-permanent-bucket",
        "private_bucket": "settings-private-bucket",
        "ttl_bucket": "settings-ttl-bucket",
        "gke_cluster": "settings-cluster",
        "gke_reservation": "settings-reservation",
        "service_account_email": "settings-service-account-email",
        "k8s_service_account": "settings-account",
        "docker_repo": "settings-repo",
        "default_dockerfile": "settings-dockerfile",
        "location_hint": "settings-location-hint",
        "subnetwork": "projects/test_project/regions/test_region/subnetworks/test_subnetwork",
    }


@contextlib.contextmanager
def mock_gcp_settings(
    module_name: Union[str, Sequence[str]], settings: Optional[dict[str, str]] = None
):
    if settings is None:
        settings = default_mock_settings()

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


@contextlib.contextmanager
def mock_job(job, *, bundler_kwargs: Optional[dict] = None):
    if bundler_kwargs is None:
        bundler_kwargs = {}
    mock_instance = mock.MagicMock()
    mock_bundler = mock.MagicMock(**bundler_kwargs)
    mock_cfg = mock.MagicMock(**{"instantiate.return_value": mock_instance})
    if hasattr(job, "from_flags"):
        mock_construct = mock.patch.object(job, "from_flags", return_value=mock_cfg)
    elif hasattr(job, "default_config"):
        mock_construct = mock.patch.object(job, "default_config", return_value=mock_cfg)
    with mock_construct:
        yield mock_cfg, mock_bundler, mock_instance

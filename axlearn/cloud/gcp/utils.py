# Copyright © 2023 Apple Inc.

"""GCP general-purpose utilities."""

import re
import subprocess
import sys
from typing import Optional, Sequence

import google.auth
from absl import flags, logging
from google.auth import exceptions as gauthexceptions
from google.auth import impersonated_credentials
from google.auth.credentials import Credentials

from axlearn.cloud.common.utils import infer_cli_name
from axlearn.cloud.gcp.scopes import DEFAULT_APPLICATION


def common_flags():
    """Defines common GCP flags."""
    flags.DEFINE_string("project", None, "The GCP project name.")
    flags.DEFINE_string("zone", None, "The GCP zone name.")


def get_credentials(
    *,
    impersonate_account: Optional[str] = None,
    impersonate_scopes: Optional[Sequence[str]] = None,
) -> Credentials:
    """Get gcloud credentials, or exits if unauthenticated.

    Args:
        impersonate_account: Service account to impersonate, if not None.
        impersonate_scopes: Scopes of the impersonation token,
            following https://developers.google.com/identity/protocols/oauth2/scopes.

    Returns:
        An authorized set of credentials.
    """

    try:
        credentials, project_id = google.auth.default()
        logging.info("Using credential for project id = %s", project_id)
    except (gauthexceptions.RefreshError, gauthexceptions.DefaultCredentialsError):
        logging.error("Please run '%s gcp auth' before this script.", infer_cli_name())
        logging.error("Please also verify if default project id is correct.")
        sys.exit(1)

    if impersonate_account:
        credentials = impersonated_credentials.Credentials(
            source_credentials=credentials,
            target_principal=impersonate_account,
            # If no scope provided, use the same scopes provided by
            # `gcloud auth application-default login`.
            target_scopes=impersonate_scopes or DEFAULT_APPLICATION,
        )

    return credentials


def running_from_vm() -> bool:
    """Check if we're running from GCP VM.

    Reference:
    https://cloud.google.com/compute/docs/instances/detect-compute-engine#use_the_metadata_server_to_detect_if_a_vm_is_running_in
    """
    out = subprocess.run(
        ["curl", "-s", "metadata.google.internal", "-i"],  # Curl silently.
        check=False,
        capture_output=True,
        text=True,
    )
    return (out.returncode == 0) and "Metadata-Flavor: Google" in out.stdout


def is_valid_resource_name(name: str) -> bool:
    """Validates names (e.g. TPUs, VMs, jobs) to ensure compat with GCP.

    Reference:
    https://cloud.google.com/compute/docs/naming-resources#resource-name-format
    """
    return re.fullmatch(r"^[a-z]([-a-z0-9]*[a-z0-9])?", name) is not None

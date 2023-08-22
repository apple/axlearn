# Copyright © 2023 Apple Inc.

"""GCP general-purpose utilities."""

import re
import subprocess
import sys

from absl import flags, logging
from google.auth import exceptions as gauthexceptions
from oauth2client.client import (
    ApplicationDefaultCredentialsError,
    GoogleCredentials,
    HttpAccessTokenRefreshError,
)

from axlearn.cloud.common.utils import infer_cli_name


def common_flags():
    """Defines common GCP flags."""
    flags.DEFINE_string("project", None, "The GCP project name.")
    flags.DEFINE_string("zone", None, "The GCP zone name.")


def get_credentials() -> GoogleCredentials:
    """Get gcloud credentials, or exits if unauthenticated.

    Returns:
        An authorized set of credentials.
    """
    try:
        credentials = GoogleCredentials.get_application_default()
        credentials.get_access_token()
    except (
        ApplicationDefaultCredentialsError,
        gauthexceptions.RefreshError,
        HttpAccessTokenRefreshError,
    ):
        logging.error("Please run '%s gcp auth' before this script.", infer_cli_name())
        sys.exit(1)
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

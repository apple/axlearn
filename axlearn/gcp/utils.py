# Copyright © 2023 Apple Inc.

"""GCP general-purpose utilities."""

import json
import logging as pylogging
import os
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Sequence

import pkg_resources
from absl import flags, logging
from google.auth import exceptions as gauthexceptions
from oauth2client.client import (
    ApplicationDefaultCredentialsError,
    GoogleCredentials,
    HttpAccessTokenRefreshError,
)

from axlearn.gcp import ROOT_MODULE_NAME


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


class FilterDiscoveryLogging(pylogging.Filter):
    """Filters out noisy 'discovery' logging."""

    def filter(self, record):
        if record.levelno == 20:
            return record.module != "discovery"
        return True


def configure_logging(level: int):
    """Configures the logging level and adds FilterDiscoveryLogging.

    Args:
        level: Logging verbosity.
    """
    # Utility to configure logging.
    logging.set_verbosity(level)
    logging.get_absl_handler().addFilter(FilterDiscoveryLogging())


def handle_popen(proc: subprocess.Popen):
    """Waits for subprocess to exit, if return != 0 raises with stdout/stderr.

    Args:
        proc: Subprocess to handle.

    Raises:
        RuntimeError: If subprocess returncode != 0.
    """
    proc.wait()
    if proc.returncode != 0:
        stderr, stdout = proc.communicate()
        raise RuntimeError(f"Popen command {proc.args} failed to return 0: \n{stderr}\n{stdout}")


def concat_cmd_list(cmd_list: Sequence[str], delimiter: str = " ", quote: str = "") -> str:
    """Convert a shell command list into a single command with appropriate quotes and delimiter.

    Args:
        cmd_list: List of commands to be executed in a shell.
        delimiter: To delineate between each command in cmd_list.
        quote: String used to indicate single command not to be broken up.

    Returns:
        Single string to be executed in shell
    """
    concat = ""
    for cmd in cmd_list:
        if re.match(f"^{quote}.*{quote}$", cmd):
            token = cmd
        else:
            token = quote + cmd + quote
        if concat:
            concat += delimiter
        concat += token
    return concat


def generate_taskname() -> str:
    """Generate a unique task name."""
    return f"{os.environ['USER'].replace('_', '')}-{uuid.uuid4().hex.lower()[:6]}"


# TODO(markblee): Consider using git python.
def get_git_revision(revision: str) -> str:
    """Gets the commit hash for the revision."""
    return subprocess.check_output(["git", "rev-parse", revision]).decode("ascii").strip()


def get_git_branch() -> str:
    """Returns current git branch."""
    return subprocess.check_output(["git", "branch", "--show-current"]).decode("ascii").strip()


def get_git_status() -> str:
    """Returns current git status."""
    return subprocess.check_output(["git", "status", "--short"]).decode("ascii").strip()


def list_tags(image: str, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """List tags for an image hosted in Artifact or Container Registry.

    Args:
        image: The image, e.g. us-docker.pkg.dev/project/repo.
        tag: Optional tag to filter by.

    Returns:
        A list of dict, where each has a "tags" field containing all matching tags.
    """
    args = [
        "gcloud",
        "container",
        "images",
        "list-tags",
        "--format=json",
    ]
    if tag:
        args.append(f"--filter='tags:{tag}'")
    args.append(image)
    out = subprocess.check_output(concat_cmd_list(args), shell=True)
    return json.loads(out)


def get_package_root(root_module_name: str = ROOT_MODULE_NAME) -> str:
    """Returns the absolute path of the package root, as defined by the directory with name
    `root_module_name`.

    Note that the installed package may not include pyproject.toml or a git directory (as it is
    rooted at `root_module_name`, not the root of the project).

    Args:
        root_module_name: Name of the root module.

    Returns:
        The absolute path of the project root.

    Raises:
        ValueError: If run from outside the package.
    """
    init = curr = os.path.dirname(os.path.realpath(__file__))
    while curr and curr != "/":
        if os.path.basename(curr) == root_module_name:
            return curr
        curr = os.path.dirname(curr)
    raise ValueError(f"Not running within {root_module_name} (searching up from '{init}').")


def get_repo_root() -> str:
    """Returns the absolute path of the repo root, as defined by a directory with `.git` containing
    `ROOT_MODULE_NAME` as a subdirectory.

    Returns:
        The absolute path of the project root.

    Raises:
        ValueError: If run from outside a repo containing `ROOT_MODULE_NAME`.
    """
    repo_root = os.path.dirname(get_package_root())
    if not os.path.exists(os.path.join(repo_root, ".git")):
        raise ValueError(f"Not running within a repo (no .git directory under '{repo_root})")
    return repo_root


def running_from_source() -> bool:
    """Returns whether this function is called from source (instead of an installed package).

    Returns:
        True iff running from source.
    """
    try:
        get_repo_root()
        return True
    except ValueError as e:
        logging.info(str(e))
    return False


def get_pyproject_version() -> str:
    """Returns the project version, e.g. X.Y.Z."""
    return pkg_resources.get_distribution(ROOT_MODULE_NAME).version


def running_from_vm() -> bool:
    """Check if we're running from GCP VM.

    Reference:
    https://cloud.google.com/compute/docs/instances/detect-compute-engine#use_the_metadata_server_to_detect_if_a_vm_is_running_in
    """
    out = subprocess.run(
        "curl metadata.google.internal -i",
        check=False,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return (out.returncode == 0) and "Metadata-Flavor: Google" in out.stdout


def parse_kv_flags(kv_flags: Sequence[str]) -> Dict[str, str]:
    """Parses sequence of k:v into a dict.

    Args:
        kv_flags: A sequence of strings in the format "k:v". If a key appears twice, the last
            occurrence "wins".

    Returns:
        A dict where keys and values are parsed from "k:v".

    Raises:
        ValueError: If a member of `kv_flags` isn't in the format "k:v".
    """
    metadata = {}
    for kv in kv_flags:
        parts = kv.split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected key:value, got {kv}")
        metadata[parts[0]] = parts[1]
    return metadata


def is_valid_resource_name(name: str) -> bool:
    """Validates names (e.g. TPUs, VMs, jobs) to ensure compat with GCP.

    Reference:
    https://cloud.google.com/compute/docs/naming-resources#resource-name-format
    """
    return re.fullmatch(r"^[a-z]([-a-z0-9]*[a-z0-9])?", name) is not None


def format_table(*, headings: List[str], rows: List[List[str]]) -> str:
    """Formats headings and rows as a table.

    Args:
        headings: Sequence of headings, one for each column.
        rows: Sequence of rows. Each row is itself a sequence of strings, containing values for each
            column. It is assumed that each row has the same number of columns as `headings`.

    Returns:
        A string formatted as a table, consisting of the provided headings and rows.
    """
    rows = [[h.upper() for h in headings]] + rows
    max_lens = [max([len(str(row[i])) for row in rows]) for i in range(len(headings))]
    fmt = "".join([f"{{:<{max_len + 6}}}" for max_len in max_lens])
    return "\n" + "\n".join([fmt.format(*[str(v) for v in row]) for row in rows]) + "\n"


def infer_cli_name() -> str:
    """Attempts to infer the CLI name."""
    return os.path.basename(os.environ.get("AXLEARN_CLI_NAME", ROOT_MODULE_NAME))

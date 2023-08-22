# Copyright © 2023 Apple Inc.

"""General-purpose utilities."""

import logging as pylogging
import os
import shlex
import subprocess
import uuid
from typing import Dict, List, Sequence, Union

import pkg_resources
from absl import logging

from axlearn.cloud import ROOT_MODULE_NAME


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
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Popen command {proc.args} returned non-zero exit code {proc.returncode}:\n"
            f"stdout={stdout}\n"
            f"stderr={stderr}"
        )


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


def subprocess_run(argv: Union[str, Sequence[str]], *args, **kwargs) -> subprocess.CompletedProcess:
    """Runs a command via subprocess.run.

    Main differences are:
    - Automatically splits or joins argv depending on whether shell=True.
    - Logs errors if capture_output=True and subprocess raises.

    Args:
        argv: The command. Can be a string or sequence of strings.
        *args: Forwarded to `subprocess.run`.
        **overrides: Forwarded to `subprocess.run`.

    Returns:
        A completed process.

    Raises:
        CalledProcessError: If the command fails.
    """
    try:
        if not kwargs.get("shell") and isinstance(argv, str):
            argv = shlex.split(argv)
        elif kwargs.get("shell") and isinstance(argv, list):
            argv = shlex.join(argv)
        # pylint: disable-next=subprocess-run-check
        return subprocess.run(argv, *args, **kwargs)
    except subprocess.CalledProcessError as e:
        # Emit the captured stdout/stderr.
        if kwargs.get("capture_output"):
            logging.error(
                "Command %s failed: code=%s, stdout=%s, stderr=%s",
                e.cmd,
                e.returncode,
                e.stdout,
                e.stderr,
            )
        raise  # Re-raise.


def canonicalize_to_list(v: Union[str, Sequence[str]], *, delimiter: str = ",") -> List[str]:
    """Converts delimited strings to lists."""
    if not v:
        return []  # Note: "".split(",") returns [""].
    if isinstance(v, str):
        v = [elem.strip() for elem in v.split(delimiter)]
    return list(v)

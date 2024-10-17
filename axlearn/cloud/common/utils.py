# Copyright © 2023 Apple Inc.

"""General-purpose utilities."""

import dataclasses
import logging as pylogging
import os
import shlex
import signal
import subprocess
import uuid
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import pkg_resources
import psutil
from absl import app, logging

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


def generate_job_name() -> str:
    """Generate a unique job name."""
    return f"{os.environ['USER'].replace('_', '')}-{uuid.uuid4().hex.lower()[:6]}"


def generate_job_id() -> str:
    """Generate a unique job uuid."""
    return str(uuid.uuid4())


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
    # TODO(markblee): Fix for nightly
    return pkg_resources.get_distribution(ROOT_MODULE_NAME).version


def parse_kv_flags(kv_flags: Sequence[str], *, delimiter: str = ":") -> dict[str, str]:
    """Parses sequence of k:v into a dict.

    Args:
        kv_flags: A sequence of strings in the format "k:v". If a key appears twice, the last
            occurrence "wins".
        delimiter: The separator between the key and value.

    Returns:
        A dict where keys and values are parsed from "k:v".

    Raises:
        ValueError: If a member of `kv_flags` isn't in the format "k:v".
    """
    metadata = {}
    for kv in kv_flags:
        parts = kv.split(delimiter, maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Expected key{delimiter}value, got {kv}")
        metadata[parts[0]] = parts[1]
    return metadata


def format_table(*, headings: list[str], rows: list[list[str]]) -> str:
    """Formats headings and rows as a table.

    Args:
        headings: Sequence of headings, one for each column.
        rows: Sequence of rows. Each row is itself a sequence of strings, containing values for each
            column. It is assumed that each row has the same number of columns as `headings`.

    Returns:
        A string formatted as a table, consisting of the provided headings and rows.
    """
    rows = [[h.upper() for h in headings]] + rows
    max_lens = [max(len(str(row[i])) for row in rows) for i in range(len(headings))]
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
        error_msg = (
            f"Command {e.cmd} failed: code={e.returncode}, stdout={e.stdout}, stderr={e.stderr}"
        )
        if kwargs.get("capture_output"):
            logging.error(error_msg)
        raise ValueError(error_msg) from e  # Re-raise.


def canonicalize_to_list(v: Union[str, Sequence[str]], *, delimiter: str = ",") -> list[str]:
    """Converts delimited strings to lists."""
    if not v:
        return []  # Note: "".split(",") returns [""].
    if isinstance(v, str):
        v = [elem.strip() for elem in v.split(delimiter)]
    return list(v)


def canonicalize_to_string(v: Union[str, Sequence[str]], *, delimiter: str = ",") -> str:
    """Converts lists to delimited strings."""
    if not v:
        return ""
    if not isinstance(v, str) and isinstance(v, Sequence):
        v = delimiter.join([elem.strip() for elem in v])
    return str(v)


def parse_action(
    argv: Sequence[str], *, options: Sequence[str], default: Optional[str] = None
) -> str:
    """Parses action from argv, or exits with usage info.

    The action is inferred from the first positional arg in argv[1:] (where argv[0] is interpreted
    as the CLI name).

    Args:
        argv: CLI arguments, possibly including --flags.
        options: Possible actions.
        default: Optional default action if unable to infer action from argv.

    Returns:
        The chosen action.

    Raises:
        absl.app.UsageError: if an invalid action (or no action and default is None) is provided.
    """
    assert default is None or default in options
    action = None
    for arg in argv[1:]:
        arg = arg.strip()
        if not arg.startswith("--"):
            action = arg
            break
    if action not in options:
        action = default
    if action is None or action not in options:  # No action nor default provided.
        raise app.UsageError(f"Invalid action: {action}. Expected one of [{','.join(options)}].")
    return action


def send_signal(popen: subprocess.Popen, sig: int = signal.SIGKILL):
    """Sends a signal (default SIGKILL) to the process (and child processes)."""
    # Note: kill() might leave orphan processes if proc spawned child processes.
    # We use psutil to recursively kill() all children.
    # If changing this fn, please run the `test_send_signal` test manually.
    try:
        parent = psutil.Process(popen.pid)
    except psutil.NoSuchProcess:
        return  # Nothing to do.
    for child in parent.children(recursive=True):
        try:
            child.send_signal(sig)
        except psutil.NoSuchProcess:
            pass  # Ignore NoSuchProcess exception and continue with the next child.
    popen.send_signal(sig)


def copy_blobs(from_prefix: str, *, to_prefix: str):
    """Replicates blobs with the from_prefix to the to_prefix."""

    # tf.io, which `fs` uses for some APIs, increases import time significantly, which hurts CLI
    # experience.
    # pylint: disable-next=import-outside-toplevel
    from axlearn.common import file_system as fs

    # As tf_io.gfile.copy requires a path to a file when reading from cloud storage,
    # we traverse the `from_prefix` to find and copy all suffixes.
    if not fs.isdir(from_prefix):
        # Copy the file.
        logging.debug("Copying file %s", from_prefix)
        fs.copy(from_prefix, to_prefix, overwrite=True)
        return
    for blob in fs.glob(os.path.join(from_prefix, "*")):
        if fs.isdir(blob):
            sub_directory = os.path.basename(blob)
            logging.info("Copying sub-directory %s", sub_directory)
            to_prefix = os.path.join(to_prefix, sub_directory)
            fs.makedirs(to_prefix)
        copy_blobs(blob, to_prefix=to_prefix)


def merge(base: dict, overrides: dict):
    """Recursively merge overrides into base."""
    if not isinstance(base, dict):
        return overrides
    for k, v in overrides.items():
        base[k] = merge(base.get(k), v)
    return base


_Row = list[Any]


@dataclasses.dataclass(repr=False)
class Table:
    """A table which can be pretty-printed."""

    headings: _Row
    rows: list[_Row]

    def __post_init__(self):
        if not isinstance(self.headings, Sequence):
            raise ValueError(f"Expected headings to be a sequence: {self.headings}")
        if not isinstance(self.rows, Sequence):
            raise ValueError(f"Expected rows to be a sequence: {self.rows}")
        for row in self.rows:
            self._check_row(row)

    def _check_row(self, row: _Row):
        if not isinstance(row, Sequence):
            raise ValueError(f"Expected row to be a sequence: {row}")
        if len(self.headings) != len(row):
            raise ValueError(f"Expected row to have {len(self.headings)} columns.")

    def add_row(self, row: _Row):
        """Adds a row to the table."""
        self._check_row(row)
        self.rows.append(row)

    def add_col(self, key: str, col: list[Any]):
        """Adds a named column to the table. The name will be added as a heading."""
        col = list(col)
        if not self.rows:
            self.headings.append(key)
            self.rows = col
        elif len(self.rows) != len(col):
            raise ValueError(f"Expected column to have {len(self.rows)} rows.")
        else:
            self.headings.append(key)
            for i, row in enumerate(self.rows):
                row.append(col[i])

    def get_col(self, *keys: str) -> list[_Row]:
        """Gets one or more named columns from the table."""
        idx = [self.headings.index(k) for k in keys]
        return [[row[i] for i in idx] for row in self.rows]

    def sort(self, key: Callable[[_Row], Any], reverse: bool = False):
        """Sorts the table. Heading remains unchanged."""
        self.rows.sort(key=key, reverse=reverse)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Table) and other.headings == self.headings and other.rows == self.rows
        )

    def __repr__(self) -> str:
        """Formats the table for printing."""
        return format_table(headings=self.headings, rows=self.rows)

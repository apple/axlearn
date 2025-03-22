# Copyright © 2025 Apple Inc.
"""Defines the GitSummary class to hold a summary of a git repository checkout."""

import dataclasses
import enum
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Union

from absl import logging


@dataclasses.dataclass(frozen=True, kw_only=True)
class GitSummaryMember:
    """Generic class to define various git summary attributes and how the values
    for this attributes are collected and exposed as a label and/or a file.

    Attributes:
        label: Value is exposed as label with this name.
        file: Value is exposed as file with this filename.
        cmd: Command to execute in `execute_command`. Should be specified as a sequence of
            command line arguments, e.g. `("git", "status")`.
        indicator_label: If True, value is exposed as an indicator label
            instead of the raw value. Defaults to False.
    """

    label: str
    file: str
    cmd: Sequence[str]
    indicator_label: bool = False

    def execute_command(self, cwd: str) -> str:
        """Execute GitSummaryMember's command.

        Args:
            cwd: Path where the command should get executed.

        Raises:
            subprocess.CalledProcessError: If executed command fails.

        Returns:
            str: Stdout of the executed command.
        """
        return subprocess.run(
            self.cmd, cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip("\n")

    def to_label(self, val: str) -> str:
        """Convert collected value to a label.

        Args:
            val: The collected value.

        Returns:
            str: The label.
        """
        if self.indicator_label:
            return str(int(bool(val)))
        else:
            return val

    def to_file(self, *, folder: str, val: str) -> str:
        """Store collected value to a file.

        Args:
            folder: The folder to store the file in.
            val: The value to store.

        Returns:
            str: Path to the saved file.
        """
        filepath = Path(folder) / self.file
        try:
            with open(filepath, "wt", encoding="utf-8") as fb:
                fb.write(val)
        except IOError as e:
            raise IOError(f"Failed to write to file {filepath}.") from e
        return str(filepath)


class GitSummaryMembers(enum.Enum):
    # pylint: disable=invalid-name
    """Define GitSummary Components.

    Attributes:
        commit: HEAD's git-sha.
        branch: Active branch.
        origin: URL of a Git remote repository named “origin”.
        diff: Git diff ("" if clean).
        porcelain: Git porcelain ("" if clean).
    """
    commit = GitSummaryMember(
        label="git-commit", file=".git.commit", cmd=("git", "rev-parse", "HEAD")
    )
    branch = GitSummaryMember(
        label="git-branch",
        file=".git.branch",
        cmd=("git", "branch", "--show-current"),
        indicator_label=False,
    )
    origin = GitSummaryMember(
        label="git-origin", file=".git.origin", cmd=("git", "remote", "get-url", "origin")
    )
    diff = GitSummaryMember(
        label="git-diff-dirty",
        file=".git.diff",
        cmd=("git", "--no-pager", "diff", "--no-color"),
        indicator_label=True,
    )
    porcelain = GitSummaryMember(
        label="git-porcelain-dirty",
        file=".git.porcelain",
        cmd=("git", "status", "--porcelain"),
        indicator_label=True,
    )


@dataclasses.dataclass(kw_only=True)
class GitSummary:
    """Retrieve different summaries for a git repo.

    Git summaries include things like the current commit, branch, origin,
    and whether the workspace is dirty.

    Attributes:
        path: Path where the git summary was collected.
        required: If True, raises a CalledProcessError.
            if summary was not requested in a valid git repo.
        root: Relative path to the git repository root
            or None if invalid git summary.
        summary: Summary members as name:value (see GitSummaryMembers).

    Raises:
        FileNotFoundError: If required=True and git is not found in the system path.
        subprocess.CalledProcessError: If required=True and path does not point to valid git repo.
    """

    path: str
    required: bool = False
    root: Optional[str] = dataclasses.field(init=False, default=None)
    summary: dict[GitSummaryMembers, str] = dataclasses.field(init=False, default_factory=dict)

    def __post_init__(self):
        """Post initialize GitSummary object"""
        if shutil.which("git", path=os.environ.get("PATH", "")) is None:
            if self.required:
                raise FileNotFoundError("Could not find `git` in PATH with required=True.")
            else:
                logging.error(
                    "Git executable not available, "
                    "collecting git summary from a git checkout will not work."
                )
        get_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=self.path,
            check=self.required,
            text=True,
            capture_output=True,
        )
        if get_root.returncode == 0:
            self.root = os.path.relpath(
                os.path.realpath(os.path.normpath(get_root.stdout.strip("\n"))),
                os.path.realpath(os.path.normpath(self.path)),
            )
            self.summary = {t: t.value.execute_command(cwd=self.path) for t in GitSummaryMembers}
        else:
            logging.warning(
                "Folder %s is not a valid git checkout, will return an invalid GitSummary.",
                self.path,
            )

    def to_labels(self) -> dict[str, str]:
        """Return git summaries as a dict mapping labels to values."""
        return {key.value.label: key.value.to_label(val) for key, val in self.summary.items()}

    def to_disk(self, out_dir: Union[str, Path]) -> list[str]:
        """Store git summaries as files at a specific destination and returns the file paths.

        Args:
            destination: Path to folder where summary should be saved.

        Returns:
            list[str]: List of paths to the serialized summary components.
        """

        summary_files = []
        for key, val in self.summary.items():
            output_path = key.value.to_file(val=val, folder=out_dir)
            summary_files.append(output_path)
        return summary_files

    def is_valid(self) -> bool:
        """Return True for a valid git summary,
        i.e. collected from valid git repository checkout, and False otherwise."""
        return self.root is not None

    def is_dirty(self) -> bool:
        """Return True if git diff or git porcelain are not empty.

        Raises:
            NotImplementedError if called on invalid git summary.
        """

        if not self.root:
            raise NotImplementedError(
                "Cannot determine if the repository is dirty because the git summary is invalid."
            )
        return bool(self.summary[GitSummaryMembers.porcelain])

    def __getitem__(self, key: GitSummaryMembers) -> Optional[str]:
        """Expose git summary attribute values that were collected at instance __init__.

        Equivalent to obj.summary.get(key, None) as invalid summary is empty.

        Args:
            key: The key to retrieve the value for.

        Returns:
            Optional[str]: The value associated with the key,
                or None if the summary is not valid.
        """
        return self.summary[key] if self.root else None

    def __bool__(self) -> bool:
        """Helper returning True if the summary is valid and clean,
        and False if the summary is invalid or dirty.

        Returns:
            bool: True if the summary is valid and clean, False otherwise.
        """
        return self.is_valid() and not self.is_dirty()

# Copyright © 2025 Apple Inc.
"""Tests for GitSummary class"""

import contextlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from absl.testing import absltest, parameterized

from axlearn.cloud.common.git_summary import GitSummary, GitSummaryMembers


@contextlib.contextmanager
def temporary_env(changes: Dict[str, Optional[str]]):
    """Temporarily change environment variables inside the context.

    Args:
        changes (Dict[str, Optional[str]]): dict with custom env variables.
            If a value is None, the environment variable will be temporarily deleted
    """

    def update_env(update):
        for k, v in update.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v

    backup = {k: os.environ.get(k) for k in changes.keys()}
    update_env(changes)
    try:
        yield
    finally:
        update_env(backup)


class GitSummaryTest(parameterized.TestCase):
    def test_missing_git(self):
        with self.assertRaises(FileNotFoundError):
            with temporary_env({"PATH": None}):
                _ = GitSummary(path=".", required=True)
            assert not GitSummary(path=".", required=False).is_valid()

    def test_invalid_summary(self):
        with tempfile.TemporaryDirectory() as td:
            summary = GitSummary(path=td)
            assert not summary.is_valid()
            assert summary.root is None
            assert len(summary.summary) == 0
            with self.assertRaises(NotImplementedError):
                _ = summary.is_dirty()
            with self.assertRaises(subprocess.CalledProcessError):
                _ = GitSummary(path=td, required=True)

    def test_valid_summary(self):
        git_sha_pattern = r"^[a-f0-9]{7,40}$"
        with tempfile.TemporaryDirectory() as repo:
            # create a valid repo and test .is_valid
            origin = "git@github.com:YYY/XXX.git"
            checkedin_file = "checkedin_file"
            subprocess.run(
                "git init "
                f"&& git remote add origin {origin}"
                f"&& touch {checkedin_file} "
                f"&& git add {checkedin_file} "
                "&& git -c user.name=XXX -c user.email=xxx@yyy.com commit -m init",
                cwd=repo,
                check=True,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            summary = GitSummary(path=repo)
            assert summary.is_valid()
            assert not summary.is_dirty()
            assert summary.root == "."
            assert re.match(git_sha_pattern, summary[GitSummaryMembers.commit]) is not None
            assert summary[GitSummaryMembers.branch] in {"main", "master"}
            assert summary[GitSummaryMembers.origin] == origin
            assert not summary[GitSummaryMembers.diff]
            assert not summary[GitSummaryMembers.porcelain]
            self._test_labels(summary)
            with tempfile.TemporaryDirectory() as summary_dir:
                self._test_labels_disk_consistency(summary, destination=summary_dir)

            # now make porcelain dirty and test
            subprocess.run(["touch", "new_file"], cwd=repo, check=True)
            summary = GitSummary(path=repo)
            assert summary.is_dirty()
            assert summary[GitSummaryMembers.porcelain] == "?? new_file"
            assert not summary[GitSummaryMembers.diff]
            with tempfile.TemporaryDirectory() as summary_dir:
                self._test_labels_disk_consistency(summary, destination=summary_dir)

            # now make diff dirty and test
            subprocess.run(["echo change >> checkedin_file"], cwd=repo, check=True, shell=True)
            summary = GitSummary(path=repo)
            assert summary.is_dirty()
            assert summary[GitSummaryMembers.porcelain] == " M checkedin_file\n?? new_file"
            diff = summary[GitSummaryMembers.diff]
            assert all(
                s in (diff or "")
                for s in [
                    "diff --git a/checkedin_file b/checkedin_file",
                    "--- a/checkedin_file",
                    "+++ b/checkedin_file",
                    "@@ -0,0 +1 @@",
                    "+change",
                    "-",
                ]
            )
            with tempfile.TemporaryDirectory() as summary_dir:
                self._test_labels_disk_consistency(summary, destination=summary_dir)

    def _test_labels(self, summary):
        asdict_keys = set(summary.to_labels().keys())
        expected_keys = {
            "git-commit",
            "git-branch",
            "git-origin",
            "git-diff-dirty",
            "git-porcelain-dirty",
        }
        assert asdict_keys == expected_keys

    def _test_labels_disk_consistency(self, summary: GitSummary, destination: str):
        summary.to_disk(out_dir=destination)
        labels = summary.to_labels()
        expected_files = {
            ".git.commit",
            ".git.branch",
            ".git.origin",
            ".git.diff",
            ".git.porcelain",
        }
        for m in GitSummaryMembers:
            filepath = Path(destination) / m.value.file
            assert m.value.file in expected_files
            expected_files.remove(m.value.file)
            assert filepath.exists()
            with open(filepath, "rt", encoding="utf-8") as fd:
                if m.value.indicator_label:
                    assert labels[m.value.label] == str(int(bool(fd.read())))
                else:
                    assert labels[m.value.label] == fd.read()
        assert not expected_files


if __name__ == "__main__":
    absltest.main()

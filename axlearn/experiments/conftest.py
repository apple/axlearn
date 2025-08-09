# Copyright © 2023 Apple Inc.

"""Configures pytest."""


def pytest_addoption(parser):
    # pylint: disable-next=import-outside-toplevel
    from axlearn.common.test_utils import pytest_addoption_atomic

    pytest_addoption_atomic(
        parser,
        "--update",
        action="store_true",
        default=False,
        help="If true, update all golden files.",
    )
    # Since these option will only affect the golden file tests, we limit the scope to `golden-file`
    pytest_addoption_atomic(
        parser,
        "--golden-file-no-verify",
        action="store_true",
        default=False,
        help="If set, the test will skip the comparison of golden files.",
    )

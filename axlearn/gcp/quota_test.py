# Copyright © 2023 Apple Inc.

"""Tests quota utilities."""

import os
from typing import Sequence

from absl.testing import parameterized

from axlearn.gcp.quota import get_project_resources, get_user_projects


class QuotaUtilsTest(parameterized.TestCase):
    """Tests quota utils."""

    def test_get_project_resources(self):
        expected = dict(
            team1=dict(test=8),
            team2=dict(test=8),
        )
        actual = get_project_resources(
            os.path.join(os.path.dirname(__file__), "testdata/project-quotas.config"),
        )
        self.assertEqual(expected, actual)

    @parameterized.parameters(
        dict(user="user1", expected=["team1", "team2"]),
        dict(user="user2", expected=["team2"]),
        dict(user="user3", expected=[]),
    )
    def test_get_user_projects(self, user: str, expected: Sequence[str]):
        self.assertEqual(
            expected,
            get_user_projects(
                os.path.join(os.path.dirname(__file__), "testdata/project-quotas.config"),
                user,
            ),
        )

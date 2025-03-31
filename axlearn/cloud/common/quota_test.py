# Copyright © 2023 Apple Inc.

"""Tests quota utilities."""

import copy
import tempfile
from collections.abc import Sequence

import toml
from absl.testing import parameterized

from axlearn.cloud.common.quota import QuotaInfo, get_resource_limits, get_user_projects

_MOCK_TOTAL_RESOURCES = [{"resource_type1": 16, "resource_type2": 8}]
_MOCK_PROJECT_RESOURCES = {
    "team1": {"resource_type1": 0.3},
    "team2": {"resource_type1": 0.6, "resource_type2": 1.0},
}
_MOCK_PROJECT_MEMBERSHIP = {"team1": ["user1"], "team2": ["user[12]"], "team3": [".*"]}
_MOCK_CONFIG = {
    "toml-schema": {"version": "1"},
    "total_resources": _MOCK_TOTAL_RESOURCES,
    "project_resources": _MOCK_PROJECT_RESOURCES,
    "project_membership": _MOCK_PROJECT_MEMBERSHIP,
}


class QuotaUtilsTest(parameterized.TestCase):
    """Tests quota utils."""

    def test_resource_limits(self):
        # Make sure fractions are converted properly.
        with tempfile.NamedTemporaryFile("r+") as f:
            toml.dump(_MOCK_CONFIG, f)
            f.seek(0)
            self.assertEqual(
                QuotaInfo(
                    total_resources=_MOCK_TOTAL_RESOURCES,
                    project_resources=_MOCK_PROJECT_RESOURCES,
                    project_membership=_MOCK_PROJECT_MEMBERSHIP,
                ),
                get_resource_limits(f.name),
            )

        # Test a case where the totals don't add up.
        with tempfile.NamedTemporaryFile("r+") as f:
            broken_config = copy.deepcopy(_MOCK_CONFIG)
            broken_config["project_resources"]["team1"]["resource_type1"] = 0.8
            toml.dump(broken_config, f)
            f.seek(0)
            self.assertEqual(
                QuotaInfo(
                    total_resources=_MOCK_TOTAL_RESOURCES,
                    project_resources={
                        "team1": {"resource_type1": 0.8},
                        "team2": {"resource_type1": 0.6, "resource_type2": 1.0},
                    },
                    project_membership=_MOCK_PROJECT_MEMBERSHIP,
                ),
                get_resource_limits(f.name),
            )

        # Test a case where the set of resource types changed.
        with tempfile.NamedTemporaryFile("r+") as f:
            config = copy.deepcopy(_MOCK_CONFIG)
            # Remove "resource_type2" from "total_resources".
            for resources in config["total_resources"]:
                resources.pop("resource_type2", None)
            toml.dump(config, f)
            f.seek(0)
            self.assertEqual(
                QuotaInfo(
                    total_resources=[{"resource_type1": 16}],
                    project_resources=_MOCK_PROJECT_RESOURCES,
                    project_membership=_MOCK_PROJECT_MEMBERSHIP,
                ),
                get_resource_limits(f.name),
            )

        # Implicitly we have 1 tier.
        with tempfile.NamedTemporaryFile("r+") as f:
            config = copy.deepcopy(_MOCK_CONFIG)
            toml.dump(config, f)
            f.seek(0)
            self.assertEqual(
                _MOCK_TOTAL_RESOURCES,
                get_resource_limits(f.name).total_resources,
            )

        # Test specifying tiers explicitly.
        with tempfile.NamedTemporaryFile("r+") as f:
            config = copy.deepcopy(_MOCK_CONFIG)
            config["total_resources"] = [{"resource_type1": 10}, {"resource_type1": 10}]
            toml.dump(config, f)
            f.seek(0)
            self.assertEqual(
                [{"resource_type1": 10}, {"resource_type1": 10}],
                get_resource_limits(f.name).total_resources,
            )

        # Test that it returns user project membership
        with tempfile.NamedTemporaryFile("r+") as f:
            toml.dump(_MOCK_CONFIG, f)
            f.seek(0)
            self.assertEqual(
                _MOCK_PROJECT_MEMBERSHIP,
                get_resource_limits(f.name).project_membership,
            )

    @parameterized.parameters(
        dict(user="user1", expected=["team1", "team2", "team3"]),
        dict(user="user2", expected=["team2", "team3"]),
        dict(user="user12", expected=["team3"]),
    )
    def test_get_user_projects(self, user: str, expected: Sequence[str]):
        """Tests `get_user_projects()` and `QuotaInfo.user_projects()`."""
        with tempfile.NamedTemporaryFile("r+") as f:
            toml.dump(_MOCK_CONFIG, f)
            f.seek(0)
            self.assertEqual(
                expected,
                get_user_projects(f.name, user),
            )

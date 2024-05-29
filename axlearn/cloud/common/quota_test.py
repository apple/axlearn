# Copyright © 2023 Apple Inc.

"""Tests quota utilities."""

import copy
import tempfile
from typing import Sequence

import toml
from absl.testing import parameterized

from axlearn.cloud.common.quota import QuotaInfo, get_resource_limits, get_user_projects

_mock_config = {
    "toml-schema": {"version": "1"},
    "total_resources": [
        {
            "resource_type1": 16,
            "resource_type2": 8,
        }
    ],
    "project_resources": {
        "team1": {"resource_type1": 0.3},
        "team2": {"resource_type1": 0.6, "resource_type2": 1.0},
    },
    "project_membership": {
        "team1": ["user1"],
        "team2": ["user[12]"],
        "team3": [".*"],
    },
}


class QuotaUtilsTest(parameterized.TestCase):
    """Tests quota utils."""

    def test_resource_limits(self):
        # Make sure fractions are converted properly.
        with tempfile.NamedTemporaryFile("r+") as f:
            toml.dump(_mock_config, f)
            f.seek(0)
            self.assertEqual(
                QuotaInfo(
                    total_resources=[{"resource_type1": 16, "resource_type2": 8}],
                    project_resources={
                        "team1": {"resource_type1": 0.3},
                        "team2": {"resource_type1": 0.6, "resource_type2": 1.0},
                    },
                ),
                get_resource_limits(f.name).quota_info(),
            )

        # Test a case where the totals don't add up.
        with tempfile.NamedTemporaryFile("r+") as f:
            broken_config = copy.deepcopy(_mock_config)
            broken_config["project_resources"]["team1"]["resource_type1"] = 0.8
            toml.dump(broken_config, f)
            f.seek(0)
            self.assertEqual(
                QuotaInfo(
                    total_resources=[{"resource_type1": 16, "resource_type2": 8}],
                    project_resources={
                        "team1": {"resource_type1": 0.8},
                        "team2": {"resource_type1": 0.6, "resource_type2": 1.0},
                    },
                ),
                get_resource_limits(f.name).quota_info(),
            )

        # Test a case where the set of resource types changed.
        with tempfile.NamedTemporaryFile("r+") as f:
            config = copy.deepcopy(_mock_config)
            # Remove "resource_type2" from "total_resources".
            for resources in config["total_resources"]:
                resources.pop("resource_type2", None)
            toml.dump(config, f)
            f.seek(0)
            self.assertEqual(
                QuotaInfo(
                    total_resources=[{"resource_type1": 16}],
                    project_resources={
                        "team1": {"resource_type1": 0.3},
                        "team2": {"resource_type1": 0.6, "resource_type2": 1.0},
                    },
                ),
                get_resource_limits(f.name).quota_info(),
            )

        # Implicitly we have 1 tier.
        with tempfile.NamedTemporaryFile("r+") as f:
            config = copy.deepcopy(_mock_config)
            toml.dump(config, f)
            f.seek(0)
            self.assertEqual(
                [{"resource_type1": 16, "resource_type2": 8}],
                get_resource_limits(f.name).total_resources,
            )

        # Test specifying tiers explicitly.
        with tempfile.NamedTemporaryFile("r+") as f:
            config = copy.deepcopy(_mock_config)
            config["total_resources"] = [{"resource_type1": 10}, {"resource_type1": 10}]
            toml.dump(config, f)
            f.seek(0)
            self.assertEqual(
                [{"resource_type1": 10}, {"resource_type1": 10}],
                get_resource_limits(f.name).total_resources,
            )

        # Test that it returns user project membership
        with tempfile.NamedTemporaryFile("r+") as f:
            toml.dump(_mock_config, f)
            f.seek(0)
            self.assertEqual(
                {
                    "team1": ["user1"],
                    "team2": ["user[12]"],
                    "team3": [".*"],
                },
                get_resource_limits(f.name).project_membership,
            )

    @parameterized.parameters(
        dict(user="user1", expected=["team1", "team2", "team3"]),
        dict(user="user2", expected=["team2", "team3"]),
        dict(user="user12", expected=["team3"]),
    )
    def test_get_user_projects(self, user: str, expected: Sequence[str]):
        """Tests `get_user_projects()` and `UserQuotaInfo.user_projects()`."""
        with tempfile.NamedTemporaryFile("r+") as f:
            toml.dump(_mock_config, f)
            f.seek(0)
            self.assertEqual(
                expected,
                get_user_projects(f.name, user),
            )

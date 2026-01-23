# Copyright © 2024 Apple Inc.

"""Tests system characteristics."""

from absl.testing import parameterized

from axlearn.cloud.gcp import system_characteristics


class SystemCharacteristicsTest(parameterized.TestCase):
    """Tests system_characteristics."""

    @parameterized.parameters(
        # Valid 7x topology
        dict(
            topology="4x4x4",
            tpu_version="7x",
            expected=((2, 2, 1), (2, 2, 4)),
        ),
        # Another valid 7x topology
        dict(
            topology="8x8x8",
            tpu_version="7x",
            expected=((2, 2, 1), (4, 4, 8)),
        ),
        # Topology with different dimensions
        dict(
            topology="2x2x1",
            tpu_version="7x",
            expected=((2, 2, 1), (1, 1, 1)),
        ),
        # Invalid: unknown TPU version
        dict(
            topology="4x4x4",
            tpu_version="v5p",
            expected=None,
        ),
        # Invalid: dimension mismatch (2D topology for 3D bounds)
        dict(
            topology="4x4",
            tpu_version="7x",
            expected=None,
        ),
        # Invalid: dimension mismatch (4D topology for 3D bounds)
        dict(
            topology="4x4x4x4",
            tpu_version="7x",
            expected=None,
        ),
    )
    def test_get_host_bounds(self, topology: str, tpu_version: str, expected):
        result = system_characteristics.get_host_bounds(topology, tpu_version)
        self.assertEqual(result, expected)

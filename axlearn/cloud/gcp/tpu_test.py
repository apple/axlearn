# Copyright © 2023 Apple Inc.

"""Tests TPU utilities."""

from typing import Union

from absl.testing import parameterized

from axlearn.cloud.gcp.tpu import (
    infer_tpu_cores,
    infer_tpu_resources,
    infer_tpu_type,
    infer_tpu_version,
    infer_tpu_workers,
)
from axlearn.common.compiler_options import NotTpuError


class TpuUtilsTest(parameterized.TestCase):
    """Tests TPU utils."""

    @parameterized.parameters(
        dict(tpu_type="v4-128", version="v4", cores=128, workers=16),
        dict(tpu_type="v5litepod-16", version="v5litepod", cores=16, workers=4),
        dict(tpu_type="v3-64", version="v3", cores=64, workers=8),
    )
    def test_infer_utils(self, tpu_type: str, version: str, cores: int, workers: int):
        self.assertEqual(version, infer_tpu_version(tpu_type))
        self.assertEqual(cores, infer_tpu_cores(tpu_type))
        self.assertEqual(workers, infer_tpu_workers(tpu_type))

    @parameterized.parameters(
        dict(instance_type="v4-8", expected="v4-8"),
        dict(instance_type="tpu-v4-8", expected="v4-8"),
        dict(instance_type="gpu", expected=NotTpuError("Invalid")),
        dict(instance_type=None, expected=NotTpuError("Invalid")),
    )
    def test_infer_tpu_type(self, instance_type, expected: Union[str, Exception]):
        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                self.assertEqual(expected, infer_tpu_type(instance_type))
        else:
            self.assertEqual(expected, infer_tpu_type(instance_type))

    @parameterized.parameters(
        dict(instance_type="v4-128", num_replicas=1, expected={"v4": 128}),
        dict(instance_type="tpu-v4-128", num_replicas=1, expected={"v4": 128}),
        dict(instance_type="v4-128", num_replicas=2, expected={"v4": 256}),
        dict(instance_type="tpu-v4-128", num_replicas=2, expected={"v4": 256}),
    )
    def test_infer_resources(self, instance_type, num_replicas, expected):
        self.assertEqual(expected, infer_tpu_resources(instance_type, num_replicas))

    def test_unknown_tpu_version(self):
        with self.assertRaisesRegex(ValueError, "Unknown TPU version"):
            infer_tpu_version("v5lite-16")

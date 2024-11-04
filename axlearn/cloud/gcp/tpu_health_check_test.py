# Copyright © 2024 Apple Inc.

"""Tests TPU health check."""

import os
import tempfile
from unittest import mock

from absl.testing import parameterized

from axlearn.cloud.gcp import tpu_health_check_main
from axlearn.cloud.gcp.tpu_health_check import (
    _parse_spec_and_check_if_should_skip,
    global_health_check,
    pairwise_slice_health_check,
    single_slice_health_check,
)


class TpuUtilsTest(parameterized.TestCase):
    def test_parsing(self):
        with mock.patch.dict(
            os.environ, {"HOSTNAME": "h", "NODE_NAME": "n", "MEGASCALE_NUM_SLICES": "1"}
        ):
            self.assertEqual(
                _parse_spec_and_check_if_should_skip("single=1.1", check_type="single"),
                1.1,
            )
            self.assertEqual(
                _parse_spec_and_check_if_should_skip(
                    "pairwise=2.2,global=1.1", check_type="global"
                ),
                1.1,
            )
            self.assertEqual(
                _parse_spec_and_check_if_should_skip(
                    "pairwise=2.2,global=1.1", check_type="single"
                ),
                None,
            )
        self.assertEqual(
            _parse_spec_and_check_if_should_skip("pairwise=2.2,global=1.1", check_type="single"),
            None,
        )
        with self.assertRaises(RuntimeError):
            _parse_spec_and_check_if_should_skip("pairwise=2.2,global=1.1", check_type="pairwise")

    def test_global_health_check(self):
        # On CPU CI, this should pass.
        with mock.patch("os.kill") as mock_exit, mock.patch.dict(
            os.environ, {"HOSTNAME": "h", "NODE_NAME": "n", "MEGASCALE_NUM_SLICES": "1"}
        ):
            global_health_check("global=180", output_dir="")
            mock_exit.assert_not_called()

    def _check_failure_file(self, folder: str, keyword: str):
        for f in os.listdir(folder):
            if not f.startswith("."):
                with open(os.path.join(folder, f), encoding="utf-8") as file:
                    self.assertIn(keyword, file.read())
                break
        else:
            self.fail("should not reach here")

    def test_global_health_check_timeout(self):
        with mock.patch(
            "os.kill"
        ) as mock_exit, tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"HOSTNAME": "h", "NODE_NAME": "n", "MEGASCALE_NUM_SLICES": "1"}
        ):
            global_health_check("global=0.000001", output_dir=d)
            mock_exit.assert_called_once()
            self._check_failure_file(d, "timeout")

    def test_raises_with_no_megascale_env(self):
        with self.assertRaises(RuntimeError):
            single_slice_health_check("single=1", output_dir="")
        with self.assertRaises(RuntimeError):
            pairwise_slice_health_check("pairwise=1", output_dir="")

    def test_global_health_check_failure(self):
        with mock.patch("os.kill") as mock_exit, mock.patch(
            f"{tpu_health_check_main.__name__}.main", lambda: False
        ), tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"HOSTNAME": "h", "NODE_NAME": "n", "MEGASCALE_NUM_SLICES": "1"}
        ):
            global_health_check("global=180", output_dir=d)
            mock_exit.assert_called_once()
            self._check_failure_file(d, "program error")

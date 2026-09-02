# Copyright © 2026 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ManagedMLDiagnostics wrapper."""

import os
import sys
from unittest import mock
from absl import flags
from absl.testing import absltest, parameterized

from axlearn.common.managed_mldiagnostics import ManagedMLDiagnostics, MLDiagnosticsConfig


class ManagedMLDiagnosticsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        ManagedMLDiagnostics._instance = None

    def test_initialize_run_success(self):
        mock_machinelearning_run = mock.MagicMock()
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.machinelearning_run": mock_machinelearning_run,
        }

        with mock.patch.dict(sys.modules, modules_mock), mock.patch.dict(os.environ, {"AXLEARN_JOB_NAME": "test_run"}):
            sys.modules["google_cloud_mldiagnostics"].machinelearning_run = mock_machinelearning_run

            cfg = MLDiagnosticsConfig(
                gcs_path="gs://test", region="us-central1", enable_xprof=True
            )
            diagnostics = ManagedMLDiagnostics(cfg)
            mock_machinelearning_run.assert_called_once_with(
                name="test_run",
                region="us-central1",
                gcs_path="gs://test",
                on_demand_xprof=True,
            )
            self.assertTrue(diagnostics._is_enabled)

    def test_initialize_run_only_once(self):
        mock_machinelearning_run = mock.MagicMock()
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.machinelearning_run": mock_machinelearning_run,
        }

        with mock.patch.dict(sys.modules, modules_mock), mock.patch.dict(os.environ, {"AXLEARN_JOB_NAME": "test_run"}):
            sys.modules["google_cloud_mldiagnostics"].machinelearning_run = mock_machinelearning_run

            cfg = MLDiagnosticsConfig(gcs_path="gs://test", region="us-central1", enable_xprof=True)
            cfg2 = MLDiagnosticsConfig(gcs_path="gs://test2", region="us-central2", enable_xprof=True)
            diagnostics = ManagedMLDiagnostics(cfg)
            diagnostics2 = ManagedMLDiagnostics(cfg2)

            mock_machinelearning_run.assert_called_once_with(
                name="test_run",
                region="us-central1",
                gcs_path="gs://test",
                on_demand_xprof=True,
            )

    def test_initialize_run_import_error(self):
        with mock.patch.dict(sys.modules, {"google_cloud_mldiagnostics": None}), mock.patch.dict(os.environ, {"AXLEARN_JOB_NAME": "test_run"}):
            cfg = MLDiagnosticsConfig(gcs_path="gs://test", region="us-central1", enable_xprof=True)
            diagnostics = ManagedMLDiagnostics(cfg)
            self.assertFalse(diagnostics._is_enabled)

    def test_initialize_run_exception(self):
        mock_machinelearning_run = mock.MagicMock(side_effect=RuntimeError("Some error"))
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.machinelearning_run": mock_machinelearning_run,
        }

        with mock.patch.dict(sys.modules, modules_mock), mock.patch.dict(os.environ, {"AXLEARN_JOB_NAME": "test_run"}):
            sys.modules["google_cloud_mldiagnostics"].machinelearning_run = mock_machinelearning_run

            cfg = MLDiagnosticsConfig(gcs_path="gs://test", region="us-central1", enable_xprof=True)
            diagnostics = ManagedMLDiagnostics(cfg)
            self.assertFalse(diagnostics._is_enabled)

    def test_initialize_run_missing_name(self):
        mock_machinelearning_run = mock.MagicMock()
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.machinelearning_run": mock_machinelearning_run,
        }

        with mock.patch.dict(sys.modules, modules_mock), mock.patch.dict(os.environ, {}, clear=True):
            sys.modules["google_cloud_mldiagnostics"].machinelearning_run = mock_machinelearning_run

            cfg = MLDiagnosticsConfig(gcs_path="gs://test", region="us-central1", enable_xprof=True)
            diagnostics = ManagedMLDiagnostics(cfg)
            mock_machinelearning_run.assert_not_called()
            self.assertFalse(diagnostics._is_enabled)

    def test_start_stop_xprof(self):
        mock_xprof_cls = mock.MagicMock()
        mock_xprof_inst = mock_xprof_cls.return_value
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.xprof": mock_xprof_cls,
        }

        with mock.patch.dict(sys.modules, modules_mock):
            sys.modules["google_cloud_mldiagnostics"].xprof = mock_xprof_cls
            diagnostics = ManagedMLDiagnostics()
            diagnostics._is_enabled = True

            diagnostics.start_xprof()
            mock_xprof_inst._ensure_initialized.assert_called_once()
            mock_xprof_inst.start.assert_called_once()

            diagnostics.stop_xprof()
            mock_xprof_inst.stop.assert_called_once()


if __name__ == "__main__":
    absltest.main()

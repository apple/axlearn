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

import sys
from unittest import mock
from absl import flags
from absl.testing import absltest, parameterized

from axlearn.common.managed_mldiagnostics import ManagedMLDiagnostics, MLDiagnosticsConfig


class ManagedMLDiagnosticsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        ManagedMLDiagnostics._instance = None

    @parameterized.parameters(
        ("loss", 0.5, "LOSS"),
        ("model/loss", 0.5, "LOSS"),
        ("learner/loss", 0.5, "LOSS"),
        ("eval/loss", 0.5, "LOSS"),
        ("learner/optimizer/learning_rate", 0.1, "LEARNING_RATE"),
        ("learner/optimizer/gradient_norm", 1.2, "GRADIENT_NORM"),
        ("num_model_params", 1e9, "TOTAL_WEIGHTS"),
        ("average_step_time", 0.5, "STEP_TIME"),
        ("unmapped_metric", 123.4, None),
    )
    def test_record_metric(self, path, raw_value, expected_metric_type_name):
        mock_metric_types = mock.MagicMock()
        # Set up mock enum values from the actual mapping
        from axlearn.common.managed_mldiagnostics import _METRIC_TO_METRIC_TYPE_NAME
        for name in set(_METRIC_TO_METRIC_TYPE_NAME.values()):
            setattr(mock_metric_types.MetricType, name, name)

        mock_metrics = mock.MagicMock()

        # Mock the import of google_cloud_mldiagnostics
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.metric_types": mock_metric_types,
            "google_cloud_mldiagnostics.metrics": mock_metrics,
        }

        with mock.patch.dict(sys.modules, modules_mock):
            # Resolve dependencies in sys.modules
            sys.modules["google_cloud_mldiagnostics"].metric_types = mock_metric_types
            sys.modules["google_cloud_mldiagnostics"].metrics = mock_metrics

            diagnostics = ManagedMLDiagnostics()
            diagnostics._is_enabled = True
            diagnostics.record_metric(path, raw_value, step=10)

            if expected_metric_type_name is not None:
                mock_metrics.record.assert_called_once_with(
                    expected_metric_type_name, float(raw_value), step=10
                )
            else:
                mock_metrics.record.assert_not_called()

    def test_initialize_run_success(self):
        mock_machinelearning_run = mock.MagicMock()
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.machinelearning_run": mock_machinelearning_run,
        }

        with mock.patch.dict(sys.modules, modules_mock):
            sys.modules["google_cloud_mldiagnostics"].machinelearning_run = mock_machinelearning_run

            cfg = MLDiagnosticsConfig(run_name="test_run", gcs_path="gs://test", region="us-central1")
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

        with mock.patch.dict(sys.modules, modules_mock):
            sys.modules["google_cloud_mldiagnostics"].machinelearning_run = mock_machinelearning_run

            cfg = MLDiagnosticsConfig(run_name="test_run", gcs_path="gs://test", region="us-central1")
            cfg2 = MLDiagnosticsConfig(run_name="test_run2", gcs_path="gs://test2", region="us-central2")
            diagnostics = ManagedMLDiagnostics(cfg)
            diagnostics2 = ManagedMLDiagnostics(cfg2)

            mock_machinelearning_run.assert_called_once_with(
                name="test_run",
                region="us-central1",
                gcs_path="gs://test",
                on_demand_xprof=True,
            )

    def test_initialize_run_import_error(self):
        with mock.patch.dict(sys.modules, {"google_cloud_mldiagnostics": None}):
            cfg = MLDiagnosticsConfig(run_name="test_run", gcs_path="gs://test", region="us-central1")
            diagnostics = ManagedMLDiagnostics(cfg)
            self.assertFalse(diagnostics._is_enabled)

    def test_initialize_run_exception(self):
        mock_machinelearning_run = mock.MagicMock(side_effect=RuntimeError("Some error"))
        modules_mock = {
            "google_cloud_mldiagnostics": mock.MagicMock(),
            "google_cloud_mldiagnostics.machinelearning_run": mock_machinelearning_run,
        }

        with mock.patch.dict(sys.modules, modules_mock):
            sys.modules["google_cloud_mldiagnostics"].machinelearning_run = mock_machinelearning_run

            cfg = MLDiagnosticsConfig(run_name="test_run", gcs_path="gs://test", region="us-central1")
            diagnostics = ManagedMLDiagnostics(cfg)
            self.assertFalse(diagnostics._is_enabled)


if __name__ == "__main__":
    absltest.main()

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

"""Managed ML Diagnostics wrapper"""

import logging
import threading
from typing import Any, Optional

from axlearn.common.config import ConfigBase, config_class


# Mapping of concrete metric names to Google Cloud ML Diagnostics MetricType names.
_METRIC_TO_METRIC_TYPE_NAME = {
    # 1. Model Quality
    "loss": "LOSS",
    "model/loss": "LOSS",
    "learner/loss": "LOSS",
    "eval/loss": "LOSS",
    "learner/optimizer/learning_rate": "LEARNING_RATE",
    "learner/optimizer/gradient_norm": "GRADIENT_NORM",
    "num_model_params": "TOTAL_WEIGHTS",

    # 2. Model Performance
    "average_step_time": "STEP_TIME",
}


class ManagedMLDiagnostics:
    """Singleton wrapper for Google Cloud ML Diagnostics."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ManagedMLDiagnostics, cls).__new__(cls)
            return cls._instance

    def __init__(self, cfg: Optional["MLDiagnosticsConfig"] = None):
        with self._lock:
            if hasattr(self, "_initialized"):
                return
            self._initialized = True
            self._is_enabled = False
            self._xprof = None

            if cfg is None or not (cfg.run_name and cfg.gcs_path):
                return
            try:
                from google_cloud_mldiagnostics import machinelearning_run
                # Initialize ML Diagnostics run (deferred)
                kwargs = {
                    "name": cfg.run_name,
                    "gcs_path": cfg.gcs_path,
                    "on_demand_xprof": True,
                }
                if cfg.region:
                    kwargs["region"] = cfg.region
                machinelearning_run(**kwargs)
                self._is_enabled = True
                logging.info(f"Successfully created Google Cloud ML Diagnostics run (deferred): {cfg.run_name}")

            except Exception as e:
                logging.error(f"Failed to start ML Diagnostics run (deferred): {e}", exc_info=True)


    def record_metric(self, path: str, raw_value: Any, step: int):
        """Record a training metric to ML Diagnostics."""
        if not self._is_enabled:
            return

        try:
            from google_cloud_mldiagnostics import metric_types, metrics as mldiag_metrics
            path_lower = path.lower()
            metric_val = float(raw_value)

            metric_type_name = _METRIC_TO_METRIC_TYPE_NAME.get(path_lower)
            if metric_type_name:
                metric_type = getattr(metric_types.MetricType, metric_type_name)
                mldiag_metrics.record(metric_type, metric_val, step=step)
        except Exception as e:
            logging.error(
                f"Failed to record metric {path} to ML Diagnostics: {e}", exc_info=True)

    def start_xprof(self):
        """Starts ML diagnostics xprof tracing if available."""
        if not self._is_enabled:
            return
        with self._lock:
            if self._xprof is None:
                try:
                    from google_cloud_mldiagnostics import xprof as mldiag_xprof
                    self._xprof = mldiag_xprof() if mldiag_xprof is not None else None
                except ImportError:
                    logging.warning(
                        "google-cloud-mldiagnostics is enabled but xprof import failed."
                    )
                    return
            if self._xprof is not None:
                self._xprof._ensure_initialized()
                self._xprof.start()

    def stop_xprof(self):
        """Stops ML diagnostics xprof tracing."""
        if self._xprof is not None:
            with self._lock:
                self._xprof.stop()


@config_class
class MLDiagnosticsConfig(ConfigBase):
    """Configuration for ML Diagnostics."""
    enable: bool = False
    run_name: Optional[str] = None
    region: Optional[str] = None
    gcs_path: Optional[str] = None


def is_ml_diagnostics_enabled(cfg: Optional[MLDiagnosticsConfig]) -> bool:
    """Returns True if ML Diagnostics is configured and enabled."""
    return cfg is not None and cfg.enable


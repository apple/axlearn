# Copyright © 2025 Apple Inc.

"""Tests for lws_utils."""

import copy
from unittest import mock

import pytest

from axlearn.cloud.common.utils import AcceleratorConfig
from axlearn.cloud.gcp.jobset_utils import TPUJobBuilder
from axlearn.cloud.gcp.lws_utils import TPULeaderWorkerTemplate


class TestTPULeaderWorkerTemplate:
    """Tests TPULeaderWorkerTemplate."""

    @pytest.mark.parametrize(
        "enable_tpu_slice_auto_provisioning,expected_label_present",
        [
            # Test when auto provisioning is enabled
            (True, True),
            # Test when auto provisioning is disabled
            (False, False),
            # Test when auto provisioning is None (not set)
            (None, False),
        ],
    )
    def test_build_pod_inject_slice_selector(
        self, enable_tpu_slice_auto_provisioning, expected_label_present
    ):
        """Test that inject-slice-selector label is added correctly based on config."""
        # Create a basic config
        cfg = TPULeaderWorkerTemplate.default_config()
        cfg.name = "test-lws"
        cfg.job_name = "test-job"
        cfg.command = "echo test"
        cfg.project = "test-project"
        cfg.output_dir = "gs://test-bucket/output"
        cfg.accelerator = AcceleratorConfig(
            instance_type="tpu-v5litepod-8",
            num_replicas=1,
        )
        cfg.enable_tpu_slice_auto_provisioning = enable_tpu_slice_auto_provisioning

        # Create a mock bundler
        mock_bundler = mock.Mock()

        # Mock the parent class's _build_pod to return a simple pod structure
        base_pod = {
            "metadata": {
                "labels": {
                    "existing-label": "existing-value",
                }
            },
            "spec": {"containers": []},
        }

        # Mock TPUJobBuilder._build_pod (the parent class method)
        with mock.patch.object(
            TPUJobBuilder,
            "_build_pod",
            return_value=copy.deepcopy(base_pod),
        ) as mock_base_build_pod:
            # Create the template instance
            template = cfg.instantiate(bundler=mock_bundler)

            # Call _build_pod which should call super()._build_pod()
            result = template._build_pod()  # pylint: disable=protected-access

            # Verify the parent's _build_pod was called
            mock_base_build_pod.assert_called_once()

            # Check if the inject-slice-selector label is present
            inject_label = "tpu-provisioner.cloud.google.com/inject-slice-selector"
            if expected_label_present:
                assert inject_label in result["metadata"]["labels"]
                assert result["metadata"]["labels"][inject_label] == "true"
            else:
                assert inject_label not in result["metadata"]["labels"]

            # Verify existing labels are still present
            assert "existing-label" in result["metadata"]["labels"]
            assert result["metadata"]["labels"]["existing-label"] == "existing-value"

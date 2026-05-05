# Copyright © 2025 Apple Inc.

"""Tests for lws_utils."""

import copy
from unittest import mock

from absl.testing import absltest, parameterized

from axlearn.cloud.common.utils import AcceleratorConfig
from axlearn.cloud.gcp.jobset_utils import TPUJobBuilder
from axlearn.cloud.gcp.lws_utils import BaseLeaderWorkerTemplate, TPULeaderWorkerTemplate


class TestTPULeaderWorkerTemplate(parameterized.TestCase):
    """Tests TPULeaderWorkerTemplate."""

    @parameterized.parameters(
        # Test when auto provisioning is enabled
        dict(enable_tpu_slice_auto_provisioning=True, expected_label_present=True),
        # Test when auto provisioning is disabled
        dict(enable_tpu_slice_auto_provisioning=False, expected_label_present=False),
        # Test when auto provisioning is None (not set)
        dict(enable_tpu_slice_auto_provisioning=None, expected_label_present=False),
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
                self.assertIn(inject_label, result["metadata"]["labels"])
                self.assertEqual(result["metadata"]["labels"][inject_label], "true")
            else:
                self.assertNotIn(inject_label, result["metadata"]["labels"])

            # Verify existing labels are still present
            self.assertIn("existing-label", result["metadata"]["labels"])
            self.assertEqual(result["metadata"]["labels"]["existing-label"], "existing-value")


class TestBaseLeaderWorkerTemplatePodMutators(absltest.TestCase):
    """Tests that pod_mutators field exists on BaseLeaderWorkerTemplate."""

    def test_pod_mutators_field_exists(self):
        """BaseLeaderWorkerTemplate.Config should have pod_mutators."""
        cfg = BaseLeaderWorkerTemplate.default_config()
        self.assertTrue(hasattr(cfg, "pod_mutators"))
        self.assertFalse(cfg.pod_mutators)

    def test_pod_mutators_field_on_tpu_template(self):
        """TPULeaderWorkerTemplate should inherit pod_mutators via TPUJobBuilder."""
        cfg = TPULeaderWorkerTemplate.default_config()
        self.assertTrue(hasattr(cfg, "pod_mutators"))
        self.assertFalse(cfg.pod_mutators)


if __name__ == "__main__":
    absltest.main()

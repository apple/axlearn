# Copyright © 2025 Apple Inc.

"""Tests k8s HealthCheckPolicy module."""

from unittest import mock

from absl import flags

from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import k8s_health_check_policy
from axlearn.cloud.gcp.pathways_utils import NOTARY_PROXY_HTTP_PORT
from axlearn.common.test_utils import TestCase

FLAGS = flags.FLAGS


class LWSHealthCheckPolicyTest(TestCase):
    """Tests LWSHealthCheckPolicy."""

    def _health_check_policy_config(
        self,
        **kwargs,
    ) -> k8s_health_check_policy.LWSHealthCheckPolicy.Config:
        fv = flags.FlagValues()
        cfg = k8s_health_check_policy.LWSHealthCheckPolicy.default_config().set()

        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                setattr(fv, key, value)
        fv.name = "fake-name"
        fv.project = "fake-project"
        fv.mark_as_parsed()
        cfg = from_flags(cfg, fv)
        self.assertIsNotNone(fv["name"])
        self.assertIsNotNone(fv["project"])
        return cfg

    def test_instantiate(self):
        """Tests basic instantiation."""
        cfg = self._health_check_policy_config(
            project="fake-project",
        )
        self.assertIsInstance(cfg, k8s_health_check_policy.LWSHealthCheckPolicy.Config)
        self.assertEqual(cfg.project, "fake-project")

        health_check = cfg.set().instantiate()
        self.assertEqual(health_check.name, "fake-name")
        self.assertEqual(health_check.service_name, "fake-name-service")

    def test_default_values(self):
        """Tests default configuration values."""
        cfg = self._health_check_policy_config(
            project="fake-project",
        )
        health_check = cfg.instantiate()

        # Check default health check settings
        self.assertEqual(health_check.config.check_interval_sec, 10)
        self.assertEqual(health_check.config.timeout_sec, 5)
        self.assertEqual(health_check.config.healthy_threshold, 1)
        self.assertEqual(health_check.config.unhealthy_threshold, 3)
        # Default port should be NOTARY_PROXY_HTTP_PORT
        self.assertEqual(health_check.health_check_port, NOTARY_PROXY_HTTP_PORT)

    def test_custom_health_check_config(self):
        """Tests custom health check configuration."""
        cfg = self._health_check_policy_config(
            project="fake-project",
        )
        cfg.check_interval_sec = 20
        cfg.timeout_sec = 10
        cfg.healthy_threshold = 2
        cfg.unhealthy_threshold = 5
        cfg.health_check_port = 9999

        health_check = cfg.instantiate()

        self.assertEqual(health_check.config.check_interval_sec, 20)
        self.assertEqual(health_check.config.timeout_sec, 10)
        self.assertEqual(health_check.config.healthy_threshold, 2)
        self.assertEqual(health_check.config.unhealthy_threshold, 5)
        self.assertEqual(health_check.health_check_port, 9999)

    def test_build_health_check_policy_structure(self):
        """Tests the structure of the built HealthCheckPolicy."""
        cfg = self._health_check_policy_config(
            project="fake-project",
        )
        cfg.namespace = "test-namespace"
        health_check = cfg.instantiate()

        # Mock the Kubernetes API call to get LeaderWorkerSet
        mock_lws = {
            "metadata": {
                "uid": "fake-lws-uid-12345",
            }
        }
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = mock_lws

            # pylint: disable-next=protected-access
            policy_dict = health_check._build_health_check_policy()

        # Verify basic structure
        self.assertEqual(policy_dict["apiVersion"], "networking.gke.io/v1")
        self.assertEqual(policy_dict["kind"], "HealthCheckPolicy")
        self.assertEqual(policy_dict["metadata"]["name"], "fake-name-health-check")
        self.assertEqual(policy_dict["metadata"]["namespace"], "test-namespace")

    def test_build_health_check_policy_spec(self):
        """Tests the spec of the HealthCheckPolicy."""
        cfg = self._health_check_policy_config(
            project="fake-project",
        )
        cfg.namespace = "test-namespace"
        health_check = cfg.instantiate()

        # Mock the Kubernetes API call to get LeaderWorkerSet
        mock_lws = {
            "metadata": {
                "uid": "fake-lws-uid-12345",
            }
        }
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = mock_lws

            # pylint: disable-next=protected-access
            policy_dict = health_check._build_health_check_policy()

        spec = policy_dict["spec"]

        # Verify default settings
        default_cfg = spec["default"]
        self.assertEqual(default_cfg["checkIntervalSec"], 10)
        self.assertEqual(default_cfg["timeoutSec"], 5)
        self.assertEqual(default_cfg["healthyThreshold"], 1)
        self.assertEqual(default_cfg["unhealthyThreshold"], 3)

        # Verify TCP health check config
        config = default_cfg["config"]
        self.assertEqual(config["type"], "TCP")
        self.assertEqual(config["tcpHealthCheck"]["port"], NOTARY_PROXY_HTTP_PORT)

        # Verify targetRef
        target_ref = spec["targetRef"]
        self.assertEqual(target_ref["group"], "")
        self.assertEqual(target_ref["kind"], "Service")
        self.assertEqual(target_ref["name"], "fake-name-service")

    def test_execute_calls_api(self):
        """Tests that execute() calls the K8s API correctly."""
        cfg = self._health_check_policy_config(
            project="fake-project",
        )
        cfg.namespace = "test-namespace"
        health_check = cfg.instantiate()

        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.create_namespaced_custom_object.return_value = {"status": "created"}

            result = health_check.execute()

            # Verify the API was called correctly
            mock_instance.create_namespaced_custom_object.assert_called_once()
            call_kwargs = mock_instance.create_namespaced_custom_object.call_args.kwargs
            self.assertEqual(call_kwargs["group"], "networking.gke.io")
            self.assertEqual(call_kwargs["version"], "v1")
            self.assertEqual(call_kwargs["namespace"], "test-namespace")
            self.assertEqual(call_kwargs["plural"], "healthcheckpolicies")

            # Verify body structure
            body = call_kwargs["body"]
            self.assertEqual(body["kind"], "HealthCheckPolicy")
            self.assertEqual(body["metadata"]["name"], "fake-name-health-check")

            self.assertEqual(result, {"status": "created"})


if __name__ == "__main__":
    import unittest

    unittest.main()

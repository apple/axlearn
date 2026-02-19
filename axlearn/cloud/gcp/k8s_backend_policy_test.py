# Copyright © 2025 Apple Inc.

"""Tests k8s GCPBackendPolicy module."""

from unittest import mock

from axlearn.cloud.gcp import k8s_backend_policy
from axlearn.common.test_utils import TestCase


class LWSGCPBackendPolicyTest(TestCase):
    """Tests LWSGCPBackendPolicy."""

    def _backend_policy_config(
        self,
        **kwargs,
    ) -> k8s_backend_policy.LWSGCPBackendPolicy.Config:
        cfg = k8s_backend_policy.LWSGCPBackendPolicy.default_config().set(
            name="fake-name",
            project="fake-project",
            **kwargs,
        )
        return cfg

    def test_instantiate(self):
        """Tests basic instantiation."""
        cfg = self._backend_policy_config()
        self.assertIsInstance(cfg, k8s_backend_policy.LWSGCPBackendPolicy.Config)
        self.assertEqual(cfg.project, "fake-project")

        backend_policy = cfg.set().instantiate()
        self.assertEqual(backend_policy.name, "fake-name")
        self.assertEqual(backend_policy.service_name, "fake-name-service")

    def test_default_timeout(self):
        """Tests default timeout value."""
        cfg = self._backend_policy_config()
        backend_policy = cfg.instantiate()
        self.assertEqual(backend_policy.timeout_sec, 3600)

    def test_build_backend_policy_structure(self):
        """Tests the structure of the built GCPBackendPolicy."""
        cfg = self._backend_policy_config(namespace="test-namespace")
        backend_policy = cfg.instantiate()

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
            policy_dict = backend_policy._build_backend_policy()

        # Verify basic structure
        self.assertEqual(policy_dict["apiVersion"], "networking.gke.io/v1")
        self.assertEqual(policy_dict["kind"], "GCPBackendPolicy")
        self.assertEqual(policy_dict["metadata"]["name"], "fake-name-backend-policy")
        self.assertEqual(policy_dict["metadata"]["namespace"], "test-namespace")

    def test_build_backend_policy_owner_reference(self):
        """Tests the owner reference of the GCPBackendPolicy."""
        cfg = self._backend_policy_config(namespace="test-namespace")
        backend_policy = cfg.instantiate()

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
            policy_dict = backend_policy._build_backend_policy()

        # Verify owner reference
        owner_refs = policy_dict["metadata"]["ownerReferences"]
        self.assertEqual(len(owner_refs), 1)
        owner_ref = owner_refs[0]
        self.assertEqual(owner_ref["apiVersion"], "leaderworkerset.x-k8s.io/v1")
        self.assertEqual(owner_ref["kind"], "LeaderWorkerSet")
        self.assertEqual(owner_ref["name"], "fake-name")
        self.assertEqual(owner_ref["uid"], "fake-lws-uid-12345")
        self.assertTrue(owner_ref["controller"])
        self.assertTrue(owner_ref["blockOwnerDeletion"])

    def test_build_backend_policy_spec(self):
        """Tests the spec of the GCPBackendPolicy."""
        cfg = self._backend_policy_config(namespace="test-namespace")
        backend_policy = cfg.instantiate()

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
            policy_dict = backend_policy._build_backend_policy()

        spec = policy_dict["spec"]

        # Verify default settings
        default_cfg = spec["default"]
        self.assertEqual(default_cfg["timeoutSec"], 3600)

        # Verify targetRef
        target_ref = spec["targetRef"]
        self.assertEqual(target_ref["group"], "")
        self.assertEqual(target_ref["kind"], "Service")
        self.assertEqual(target_ref["name"], "fake-name-service")

    def test_execute_calls_api(self):
        """Tests that execute() calls the K8s API correctly."""
        cfg = self._backend_policy_config(namespace="test-namespace")
        backend_policy = cfg.instantiate()

        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.create_namespaced_custom_object.return_value = {"status": "created"}

            result = backend_policy.execute()

            # Verify the API was called correctly
            mock_instance.create_namespaced_custom_object.assert_called_once()
            call_kwargs = mock_instance.create_namespaced_custom_object.call_args.kwargs
            self.assertEqual(call_kwargs["group"], "networking.gke.io")
            self.assertEqual(call_kwargs["version"], "v1")
            self.assertEqual(call_kwargs["namespace"], "test-namespace")
            self.assertEqual(call_kwargs["plural"], "gcpbackendpolicies")

            # Verify body structure
            body = call_kwargs["body"]
            self.assertEqual(body["kind"], "GCPBackendPolicy")
            self.assertEqual(body["metadata"]["name"], "fake-name-backend-policy")
            self.assertEqual(body["spec"]["default"]["timeoutSec"], 3600)

            self.assertEqual(result, {"status": "created"})


if __name__ == "__main__":
    import unittest

    unittest.main()

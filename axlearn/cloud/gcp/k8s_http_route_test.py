# Copyright © 2025 Apple Inc.

"""Tests k8s HTTPRoute module."""

from unittest import mock

from absl import flags

from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import k8s_http_route
from axlearn.common.test_utils import TestCase

FLAGS = flags.FLAGS


class LWSHTTPRouteTest(TestCase):
    """Tests LWSHTTPRoute."""

    def _http_route_config(
        self,
        **kwargs,
    ) -> k8s_http_route.LWSHTTPRoute.Config:
        fv = flags.FlagValues()
        cfg = k8s_http_route.LWSHTTPRoute.default_config().set()

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
        cfg = self._http_route_config(
            project="fake-project",
        )
        self.assertIsInstance(cfg, k8s_http_route.LWSHTTPRoute.Config)
        self.assertEqual(cfg.project, "fake-project")

        http_route = cfg.set().instantiate()
        self.assertEqual(http_route.name, "fake-name")
        self.assertEqual(http_route.service_name, "fake-name-service")

    def test_default_values(self):
        """Tests default configuration values."""
        cfg = self._http_route_config(
            project="fake-project",
        )
        http_route = cfg.instantiate()

        # Check default gateway settings
        self.assertEqual(http_route.config.gateway_namespace, "long-running-inference")
        self.assertEqual(http_route.config.gateway_name, "bastion-inference-gateway")
        self.assertEqual(http_route.config.https_gateway_name, "https-gateway")
        self.assertEqual(http_route.config.https_gateway_namespace, "gateway-system")
        self.assertEqual(http_route.config.http_port, 8080)
        self.assertEqual(http_route.config.grpc_port, 9000)

    def test_custom_gateway_config(self):
        """Tests custom gateway configuration."""
        cfg = self._http_route_config(
            project="fake-project",
            gateway_namespace="custom-namespace",
            gateway_name="custom-gateway",
            http_port=8888,
            grpc_port=9999,
        )
        http_route = cfg.instantiate()

        self.assertEqual(http_route.config.gateway_namespace, "custom-namespace")
        self.assertEqual(http_route.config.gateway_name, "custom-gateway")
        self.assertEqual(http_route.config.http_port, 8888)
        self.assertEqual(http_route.config.grpc_port, 9999)

    def test_build_http_route_structure(self):
        """Tests the structure of the built HTTPRoute."""
        cfg = self._http_route_config(
            project="fake-project",
            namespace="test-namespace",
        )
        cfg.namespace = "test-namespace"
        http_route = cfg.instantiate()

        # Mock the K8s API call to get LWS
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = {
                "metadata": {"uid": "fake-uid"}
            }

            # pylint: disable-next=protected-access
            route_dict = http_route._build_http_route()

            # Verify basic structure
            self.assertEqual(route_dict["apiVersion"], "gateway.networking.k8s.io/v1")
            self.assertEqual(route_dict["kind"], "HTTPRoute")
            self.assertEqual(route_dict["metadata"]["name"], "fake-name-route")
            self.assertEqual(route_dict["metadata"]["namespace"], "test-namespace")

            # Verify parentRefs
            parent_refs = route_dict["spec"]["parentRefs"]
            self.assertEqual(len(parent_refs), 1)

            # HTTPS gateway parent ref
            self.assertEqual(parent_refs[0]["name"], "https-gateway")
            self.assertEqual(parent_refs[0]["namespace"], "gateway-system")

    def test_build_http_route_rules(self):
        """Tests the routing rules in the HTTPRoute."""
        cfg = self._http_route_config(
            project="fake-project",
            namespace="test-namespace",
        )
        cfg.namespace = "test-namespace"
        http_route = cfg.instantiate()

        # Mock the K8s API call to get LWS
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = {
                "metadata": {"uid": "fake-uid"}
            }

            # pylint: disable-next=protected-access
            route_dict = http_route._build_http_route()

            rules = route_dict["spec"]["rules"]
            self.assertEqual(len(rules), 2)

            # Rule 1: gRPC routing (header match)
            grpc_rule = rules[0]
            self.assertIn("headers", grpc_rule["matches"][0])
            header_match = grpc_rule["matches"][0]["headers"][0]
            self.assertEqual(header_match["type"], "Exact")
            self.assertEqual(header_match["name"], "serve-name")
            # Name without "-long-running" suffix stays the same
            self.assertEqual(header_match["value"], "fake-name")

            # gRPC backend
            grpc_backend = grpc_rule["backendRefs"][0]
            self.assertEqual(grpc_backend["name"], "fake-name-service")
            self.assertEqual(grpc_backend["namespace"], "test-namespace")
            self.assertEqual(grpc_backend["port"], 9000)

            # Rule 2: REST routing (path match)
            rest_rule = rules[1]
            path_match = rest_rule["matches"][0]["path"]
            self.assertEqual(path_match["type"], "PathPrefix")
            # Name without "-long-running" suffix stays the same
            self.assertEqual(path_match["value"], "/serve/fake-name")

            # URL rewrite filter
            filters = rest_rule["filters"]
            self.assertEqual(len(filters), 1)
            self.assertEqual(filters[0]["type"], "URLRewrite")
            self.assertEqual(filters[0]["urlRewrite"]["path"]["replacePrefixMatch"], "/")

            # REST backend
            rest_backend = rest_rule["backendRefs"][0]
            self.assertEqual(rest_backend["name"], "fake-name-service")
            self.assertEqual(rest_backend["namespace"], "test-namespace")
            self.assertEqual(rest_backend["port"], 8080)

    def test_execute_calls_api(self):
        """Tests that execute() calls the K8s API correctly."""
        cfg = self._http_route_config(
            project="fake-project",
        )
        http_route = cfg.instantiate()

        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = {
                "metadata": {"uid": "fake-uid"}
            }
            mock_instance.create_namespaced_custom_object.return_value = {"status": "created"}

            result = http_route.execute()

            # Verify get_namespaced_custom_object was called to fetch LWS
            mock_instance.get_namespaced_custom_object.assert_called_once()

            # Verify the create API was called correctly
            mock_instance.create_namespaced_custom_object.assert_called_once()
            call_kwargs = mock_instance.create_namespaced_custom_object.call_args.kwargs
            self.assertEqual(call_kwargs["group"], "gateway.networking.k8s.io")
            self.assertEqual(call_kwargs["version"], "v1")
            # Default namespace is "default"
            self.assertEqual(call_kwargs["namespace"], "default")
            self.assertEqual(call_kwargs["plural"], "httproutes")

            # Verify body structure
            body = call_kwargs["body"]
            self.assertEqual(body["kind"], "HTTPRoute")
            self.assertEqual(body["metadata"]["name"], "fake-name-route")

            self.assertEqual(result, {"status": "created"})

    def test_suffix_removal_in_path(self):
        """Tests that -long-running suffix is removed from path match and gRPC header."""
        # Test with name that has -long-running suffix
        cfg = self._http_route_config(
            project="fake-project",
            namespace="test-namespace",
        )
        # Override the name to include -long-running suffix
        cfg.name = "rfc-364-oliver-long-running"
        cfg.namespace = "test-namespace"
        http_route = cfg.instantiate()

        # Mock the K8s API call to get LWS
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = {
                "metadata": {"uid": "fake-uid"}
            }

            # pylint: disable-next=protected-access
            route_dict = http_route._build_http_route()

            rules = route_dict["spec"]["rules"]

            # Rule 1: gRPC routing - header value should have suffix removed
            grpc_rule = rules[0]
            header_match = grpc_rule["matches"][0]["headers"][0]
            # Suffix should be stripped: rfc-364-oliver-long-running -> rfc-364-oliver
            self.assertEqual(header_match["value"], "rfc-364-oliver")

            # Rule 2: REST routing - should have suffix removed
            rest_rule = rules[1]
            path_match = rest_rule["matches"][0]["path"]
            # Suffix should be stripped: rfc-364-oliver-long-running -> rfc-364-oliver
            self.assertEqual(path_match["value"], "/serve/rfc-364-oliver")

            # Service name should keep the full name
            rest_backend = rest_rule["backendRefs"][0]
            self.assertEqual(rest_backend["name"], "rfc-364-oliver-long-running-service")

    def test_double_suffix_removal(self):
        """Tests suffix removal when name has -long-running-long-running."""
        cfg = self._http_route_config(
            project="fake-project",
            namespace="test-namespace",
        )
        # Override the name to include double -long-running suffix
        cfg.name = "rfc-333-oliver-serv-long-running-long-running"
        cfg.namespace = "test-namespace"
        http_route = cfg.instantiate()

        # Mock the K8s API call to get LWS
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_api:
            mock_instance = mock_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = {
                "metadata": {"uid": "fake-uid"}
            }

            # pylint: disable-next=protected-access
            route_dict = http_route._build_http_route()

            rules = route_dict["spec"]["rules"]

            # Rule 1: gRPC routing - header value should remove only one -long-running
            grpc_rule = rules[0]
            header_match = grpc_rule["matches"][0]["headers"][0]
            # Only one suffix stripped: ...long-running-long-running -> ...long-running
            self.assertEqual(header_match["value"], "rfc-333-oliver-serv-long-running")

            # Rule 2: REST routing - should remove only one -long-running
            rest_rule = rules[1]
            path_match = rest_rule["matches"][0]["path"]
            # Only one suffix stripped: ...long-running-long-running -> ...long-running
            self.assertEqual(path_match["value"], "/serve/rfc-333-oliver-serv-long-running")

            # Service name should keep the full name
            rest_backend = rest_rule["backendRefs"][0]
            self.assertEqual(
                rest_backend["name"], "rfc-333-oliver-serv-long-running-long-running-service"
            )


if __name__ == "__main__":
    import unittest

    unittest.main()

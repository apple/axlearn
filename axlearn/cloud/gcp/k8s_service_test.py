"""Tests k8s service module."""

from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import k8s_service
from axlearn.cloud.gcp.pathways_utils import NOTARY_PROXY_GRPC_PORT, NOTARY_PROXY_HTTP_PORT
from axlearn.common.test_utils import TestCase

FLAGS = flags.FLAGS


class GKELWSService(TestCase):
    """Tests GKEService with LWS(TPU)."""

    def _service_config(
        self,
        *,
        command: str,
        **kwargs,
    ) -> k8s_service.LWSService.Config:
        fv = flags.FlagValues()
        cfg = k8s_service.LWSService.default_config().set()

        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                # Use setattr rather than set_default to set flags.
                setattr(fv, key, value)
        fv.name = "fake-name"
        fv.project = "fake-project"
        fv.mark_as_parsed()
        cfg = from_flags(cfg, fv, command=command)
        # Test that retries are configured on fv by default.
        self.assertIsNotNone(fv["name"])
        self.assertIsNotNone(fv["project"])
        return cfg

    def test_instantiate(
        self,
    ):
        cfg = self._service_config(
            command="test-command",
            project="fake-project",
            ports=[9000],
        )
        self.assertIsInstance(cfg, k8s_service.LWSService.Config)
        self.assertEqual(cfg.project, "fake-project")
        gke_lws_service = cfg.set().instantiate()
        self.assertEqual(cfg.name + "-service", gke_lws_service.name)
        self.assertEqual(cfg.ports, gke_lws_service.ports)

    def test_non_default_namespace(
        self,
    ):
        cfg = self._service_config(
            command="test-command",
            project="fake-project",
            namespace="non-default",
            ports=[9000],
        )
        self.assertIsInstance(cfg, k8s_service.LWSService.Config)
        self.assertEqual(cfg.namespace, "non-default")
        gke_lws_service = cfg.set().instantiate()
        self.assertEqual(cfg.name + "-service", gke_lws_service.name)
        self.assertEqual(cfg.namespace, gke_lws_service.config.namespace)

        # Mock kubernetes API to test _build_service uses the non-default namespace
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_custom_api:
            mock_instance = mock_custom_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = {
                "metadata": {"uid": "fake-uid"}
            }

            # Call _build_service
            # pylint: disable-next=protected-access
            service_dict = gke_lws_service._build_service()

            # Verify that get_namespaced_custom_object was called with non-default namespace
            mock_instance.get_namespaced_custom_object.assert_called_once()
            call_args = mock_instance.get_namespaced_custom_object.call_args
            self.assertEqual(call_args.kwargs["namespace"], "non-default")

            # Verify the service is properly configured
            self.assertIsNotNone(service_dict)

    @parameterized.parameters([True, False])
    def test_gke_gateway_route_target_ports(self, gke_gateway_route):
        """Tests that target_ports are auto-set when gke_gateway_route=True."""
        cfg = self._service_config(
            command="test-command",
            project="fake-project",
            ports=["8080", "9000"],
            protocol_list=["TCP", "TCP"],
            target_ports=["8080", "9000"],  # These should be overridden when gke_gateway_route=True
            port_names=["http", "grpc"],
        )
        cfg.gke_gateway_route = gke_gateway_route
        gke_lws_service = cfg.set().instantiate()

        if gke_gateway_route:
            # When gke_gateway_route=True, target_ports should be auto-set to notary ports
            self.assertEqual(
                gke_lws_service.target_ports,
                [str(NOTARY_PROXY_HTTP_PORT), str(NOTARY_PROXY_GRPC_PORT)],
            )
            self.assertEqual(gke_lws_service.port_names, ["http", "grpc"])
        else:
            # When gke_gateway_route=False, target_ports should use config values
            self.assertEqual(gke_lws_service.target_ports, ["8080", "9000"])
            self.assertEqual(gke_lws_service.port_names, ["http", "grpc"])

    def test_gke_gateway_route_grpc_app_protocol(self):
        """Tests that appProtocol is set for GRPC port when gke_gateway_route=True."""
        cfg = self._service_config(
            command="test-command",
            project="fake-project",
            ports=["8080", "9000"],
            protocol_list=["TCP", "TCP"],
            target_ports=["8080", "9000"],
            port_names=["http", "grpc"],
        )
        cfg.gke_gateway_route = True
        gke_lws_service = cfg.set().instantiate()

        # Mock kubernetes API to test _build_service
        with mock.patch("kubernetes.client.CustomObjectsApi") as mock_custom_api:
            mock_instance = mock_custom_api.return_value
            mock_instance.get_namespaced_custom_object.return_value = {
                "metadata": {"uid": "fake-uid"}
            }

            # Call _build_service
            # pylint: disable-next=protected-access
            service_dict = gke_lws_service._build_service()

            # Verify the service is properly configured
            self.assertIsNotNone(service_dict)
            spec = service_dict.get("spec")
            self.assertIsNotNone(spec)

            # Check ports - should have appProtocol for grpc port
            ports = spec.ports
            self.assertEqual(len(ports), 2)

            # Find the grpc port
            grpc_port = next((p for p in ports if p.name == "grpc"), None)
            self.assertIsNotNone(grpc_port)
            assert grpc_port is not None  # For pytype
            self.assertEqual(grpc_port.app_protocol, "kubernetes.io/h2c")

            # HTTP port should not have app_protocol set
            http_port = next((p for p in ports if p.name == "http"), None)
            self.assertIsNotNone(http_port)
            self.assertIsNone(getattr(http_port, "app_protocol", None))

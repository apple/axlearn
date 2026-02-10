"""k8s HTTPRoute module for gke_gateway_route feature."""

import copy
import logging
from typing import Any

import kubernetes as k8s
from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable, generate_job_name
from axlearn.cloud.gcp.config import default_project
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested


class LWSHTTPRoute(FlagConfigurable):
    """LWS HTTPRoute for gke_gateway_route feature.

    Creates an HTTPRoute K8s object that routes traffic from GKE Gateway
    directly to the LWS service.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures LWSHTTPRoute.

        Attributes:
            name: The name of the LWS resource.
            project: The GCP project.
            namespace: The namespace of the service.
            gateway_name: The name of the Gateway for HTTP routing.
            gateway_namespace: The namespace where HTTPRoute is created.
            https_gateway_name: The name of the HTTPS Gateway.
            https_gateway_namespace: The namespace of the HTTPS Gateway.
            http_port: The HTTP port on the service.
            grpc_port: The gRPC port on the service.
        """

        name: Required[str] = REQUIRED
        project: Required[str] = REQUIRED
        namespace: str = "default"
        gateway_name: str = "bastion-inference-gateway"
        gateway_namespace: str = "long-running-inference"
        https_gateway_name: str = "https-gateway"
        https_gateway_namespace: str = "gateway-system"
        http_port: int = 8080
        grpc_port: int = 9000

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the HTTPRoute.", **common_kwargs)
        flags.DEFINE_string("project", None, "The GCP project name.", **common_kwargs)
        flags.DEFINE_string(
            "namespace",
            "default",
            "Namespace of the service.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "gateway_namespace",
            "long-running-inference",
            "Namespace where HTTPRoute is created.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "gateway_name",
            "bastion-inference-gateway",
            "Name of the Gateway for HTTP routing.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "https_gateway_name",
            "https-gateway",
            "Name of the HTTPS Gateway.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "https_gateway_namespace",
            "gateway-system",
            "Namespace of the HTTPS Gateway.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "http_port",
            8080,
            "HTTP port on the service.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "grpc_port",
            9000,
            "gRPC port on the service.",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default("project", default_project())
        fv.set_default("namespace", fv.namespace or "default")
        fv.set_default("gateway_namespace", fv.gateway_namespace or "long-running-inference")
        fv.set_default("gateway_name", fv.gateway_name or "bastion-inference-gateway")
        fv.set_default("https_gateway_name", fv.https_gateway_name or "https-gateway")
        fv.set_default("https_gateway_namespace", fv.https_gateway_namespace or "gateway-system")
        fv.set_default("http_port", fv.http_port or 8080)
        fv.set_default("grpc_port", fv.grpc_port or 9000)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: LWSHTTPRoute.Config = super().from_flags(fv, **kwargs)
        cfg.name = fv.name
        cfg.namespace = fv.namespace
        cfg.gateway_namespace = fv.gateway_namespace
        cfg.gateway_name = fv.gateway_name
        cfg.https_gateway_name = fv.https_gateway_name
        cfg.https_gateway_namespace = fv.https_gateway_namespace
        cfg.http_port = fv.http_port
        cfg.grpc_port = fv.grpc_port
        return cfg

    @classmethod
    def default_config(cls):
        return super().default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        logging.info("LWSHTTPRoute class init")
        self._config = copy.deepcopy(cfg)
        self.name = cfg.name
        self.service_name = f"{cfg.name}-service"
        self.service_namespace = cfg.namespace

    def _build_http_route(self) -> Nested[Any]:
        """Builds a config for an HTTPRoute.

        Returns:
            A nested dict corresponding to a K8s HTTPRoute config.
        """
        cfg = self.config
        logging.info("LWSHTTPRoute class build")
        logging.info(str(self.config))

        # Import utils here to avoid circular dependency
        from axlearn.cloud.gcp.utils import (  # pylint: disable=import-outside-toplevel
            custom_leaderworkerset_kwargs,
        )

        # Fetch the LeaderWorkerSet to get its UID for owner reference
        api_kwargs = custom_leaderworkerset_kwargs()
        custom_api = k8s.client.CustomObjectsApi()
        lws = custom_api.get_namespaced_custom_object(
            group=api_kwargs["group"],
            version=api_kwargs["version"],
            namespace=self.service_namespace,
            plural=api_kwargs["plural"],
            name=self.name,
        )

        # Build the HTTPRoute spec
        # NOTE: HTTPRoute MUST be in the same namespace as LWS for owner references to work.
        # Cross-namespace owner references are not supported by Kubernetes.
        http_route = {
            "apiVersion": "gateway.networking.k8s.io/v1",
            "kind": "HTTPRoute",
            "metadata": {
                "name": f"{self.name}-route",
                "namespace": cfg.namespace,  # Use LWS namespace for owner reference to work
                "ownerReferences": [
                    {
                        "apiVersion": f'{api_kwargs["group"]}/{api_kwargs["version"]}',
                        "kind": "LeaderWorkerSet",
                        "name": self.name,
                        "uid": lws["metadata"]["uid"],
                        "controller": True,
                        "blockOwnerDeletion": True,
                    }
                ],
            },
            "spec": {
                "parentRefs": [
                    {
                        "group": "gateway.networking.k8s.io",
                        "kind": "Gateway",
                        "name": cfg.https_gateway_name,
                        "namespace": cfg.https_gateway_namespace,
                    },
                ],
                "rules": [
                    # Rule 1: gRPC routing (header match)
                    # Because Bolt-Serve always add -long-running as suffix to model name
                    # Strip it
                    {
                        "matches": [
                            {
                                "headers": [
                                    {
                                        "type": "Exact",
                                        "name": "serve-name",
                                        "value": self.name.removesuffix("-long-running"),
                                    }
                                ],
                                "path": {"type": "PathPrefix", "value": "/"},
                            }
                        ],
                        "backendRefs": [
                            {
                                "name": self.service_name,
                                "namespace": self.service_namespace,
                                "port": cfg.grpc_port,
                            }
                        ],
                    },
                    # Rule 2: REST routing (path match with URL rewrite)
                    # Because Bolt-Serve always add -long-running as suffix to model name
                    # Strip it
                    {
                        "matches": [
                            {
                                "path": {
                                    "type": "PathPrefix",
                                    "value": f'/serve/{self.name.removesuffix("-long-running")}',
                                }
                            }
                        ],
                        "filters": [
                            {
                                "type": "URLRewrite",
                                "urlRewrite": {
                                    "path": {
                                        "type": "ReplacePrefixMatch",
                                        "replacePrefixMatch": "/",
                                    }
                                },
                            }
                        ],
                        "backendRefs": [
                            {
                                "name": self.service_name,
                                "namespace": self.service_namespace,
                                "port": cfg.http_port,
                            }
                        ],
                    },
                ],
            },
        }

        return http_route

    def execute(self):
        """Creates the HTTPRoute in the cluster."""
        logging.info("LWSHTTPRoute class execute")
        http_route = self._build_http_route()
        logging.info("Submitting LWSHTTPRoute body=%s", http_route)

        return k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            group="gateway.networking.k8s.io",
            version="v1",
            namespace=self.config.namespace,  # Use LWS namespace to match metadata
            plural="httproutes",
            body=http_route,
        )

# Copyright © 2025 Apple Inc.

"""k8s GCPBackendPolicy module for gke_gateway_route feature."""

import copy
import logging
from typing import Any

import kubernetes as k8s

from axlearn.cloud.common.utils import FlagConfigurable
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested

# Default timeout for GCPBackendPolicy in seconds
DEFAULT_BACKEND_TIMEOUT_SEC = 3600


class LWSGCPBackendPolicy(FlagConfigurable):
    """LWS GCPBackendPolicy for gke_gateway_route feature.

    Creates a GCPBackendPolicy K8s object that configures backend timeout
    for the LWS service when using GKE Gateway routing.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures LWSGCPBackendPolicy.

        Attributes:
            name: The name of the LWS resource.
            project: The GCP project.
            namespace: The namespace of the service.
        """

        name: Required[str] = REQUIRED
        project: Required[str] = REQUIRED
        namespace: str = "default"

    @classmethod
    def default_config(cls):
        return super().default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        logging.info("LWSGCPBackendPolicy class init")
        self._config = copy.deepcopy(cfg)
        self.name = cfg.name
        self.service_name = f"{cfg.name}-service"
        self.timeout_sec = DEFAULT_BACKEND_TIMEOUT_SEC

    def _build_backend_policy(self) -> Nested[Any]:
        """Builds a config for a GCPBackendPolicy.

        Returns:
            A nested dict corresponding to a K8s GCPBackendPolicy config.
        """
        cfg = self.config
        logging.info("LWSGCPBackendPolicy class build")
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
            namespace=cfg.namespace,
            plural=api_kwargs["plural"],
            name=self.name,
        )

        # Build the GCPBackendPolicy spec
        backend_policy = {
            "apiVersion": "networking.gke.io/v1",
            "kind": "GCPBackendPolicy",
            "metadata": {
                "name": f"{self.name}-backend-policy",
                "namespace": cfg.namespace,
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
                "default": {
                    "timeoutSec": self.timeout_sec,
                },
                "targetRef": {
                    "group": "",
                    "kind": "Service",
                    "name": self.service_name,
                },
            },
        }

        return backend_policy

    def execute(self):
        """Creates the GCPBackendPolicy in the cluster."""
        logging.info("LWSGCPBackendPolicy class execute")
        backend_policy = self._build_backend_policy()
        logging.info("Submitting LWSGCPBackendPolicy body=%s", backend_policy)

        return k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            group="networking.gke.io",
            version="v1",
            namespace=self.config.namespace,
            plural="gcpbackendpolicies",
            body=backend_policy,
        )

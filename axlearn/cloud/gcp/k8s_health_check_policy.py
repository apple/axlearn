# Copyright © 2025 Apple Inc.

"""k8s HealthCheckPolicy module for gke_gateway_route feature."""

import copy
import logging
from typing import Any, Optional

import kubernetes as k8s
from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable, generate_job_name
from axlearn.cloud.gcp.config import default_project
from axlearn.cloud.gcp.pathways_utils import NOTARY_PROXY_HTTP_PORT
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested


class LWSHealthCheckPolicy(FlagConfigurable):
    """LWS HealthCheckPolicy for gke_gateway_route feature.

    Creates a HealthCheckPolicy K8s object that configures health checks
    for the LWS service when using GKE Gateway routing.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures LWSHealthCheckPolicy.

        Attributes:
            name: The name of the LWS resource.
            project: The GCP project.
            namespace: The namespace of the service.
            check_interval_sec: Interval between health checks in seconds.
            timeout_sec: Timeout for each health check in seconds.
            healthy_threshold: Number of consecutive successes to mark healthy.
            unhealthy_threshold: Number of consecutive failures to mark unhealthy.
            health_check_port: The port to use for TCP health check.
        """

        name: Required[str] = REQUIRED
        project: Required[str] = REQUIRED
        namespace: str = "default"
        check_interval_sec: int = 10
        timeout_sec: int = 5
        healthy_threshold: int = 1
        unhealthy_threshold: int = 3
        health_check_port: Optional[int] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the HealthCheckPolicy.", **common_kwargs)
        flags.DEFINE_string("project", None, "The GCP project name.", **common_kwargs)
        flags.DEFINE_integer(
            "health_check_interval_sec",
            10,
            "Interval between health checks in seconds.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "health_check_timeout_sec",
            5,
            "Timeout for each health check in seconds.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "health_check_healthy_threshold",
            1,
            "Number of consecutive successes to mark healthy.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "health_check_unhealthy_threshold",
            3,
            "Number of consecutive failures to mark unhealthy.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "health_check_port",
            None,
            "Port to use for TCP health check. Defaults to NOTARY_PROXY_HTTP_PORT.",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default("project", default_project())
        fv.set_default("health_check_interval_sec", fv.health_check_interval_sec or 10)
        fv.set_default("health_check_timeout_sec", fv.health_check_timeout_sec or 5)
        fv.set_default("health_check_healthy_threshold", fv.health_check_healthy_threshold or 1)
        fv.set_default("health_check_unhealthy_threshold", fv.health_check_unhealthy_threshold or 3)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: LWSHealthCheckPolicy.Config = super().from_flags(fv, **kwargs)
        cfg.name = fv.name
        cfg.check_interval_sec = fv.health_check_interval_sec
        cfg.timeout_sec = fv.health_check_timeout_sec
        cfg.healthy_threshold = fv.health_check_healthy_threshold
        cfg.unhealthy_threshold = fv.health_check_unhealthy_threshold
        if hasattr(fv, "health_check_port") and fv.health_check_port:
            cfg.health_check_port = fv.health_check_port
        return cfg

    @classmethod
    def default_config(cls):
        return super().default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        logging.info("LWSHealthCheckPolicy class init")
        self._config = copy.deepcopy(cfg)
        self.name = cfg.name
        self.service_name = f"{cfg.name}-service"
        # Default to NOTARY_PROXY_HTTP_PORT if not specified
        self.health_check_port = cfg.health_check_port or NOTARY_PROXY_HTTP_PORT

    def _build_health_check_policy(self) -> Nested[Any]:
        """Builds a config for a HealthCheckPolicy.

        Returns:
            A nested dict corresponding to a K8s HealthCheckPolicy config.
        """
        cfg = self.config
        logging.info("LWSHealthCheckPolicy class build")
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

        # Build the HealthCheckPolicy spec
        health_check_policy = {
            "apiVersion": "networking.gke.io/v1",
            "kind": "HealthCheckPolicy",
            "metadata": {
                "name": f"{self.name}-health-check",
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
                    "checkIntervalSec": cfg.check_interval_sec,
                    "timeoutSec": cfg.timeout_sec,
                    "healthyThreshold": cfg.healthy_threshold,
                    "unhealthyThreshold": cfg.unhealthy_threshold,
                    "config": {
                        "type": "TCP",
                        "tcpHealthCheck": {
                            "port": self.health_check_port,
                        },
                    },
                },
                "targetRef": {
                    "group": "",
                    "kind": "Service",
                    "name": self.service_name,
                },
            },
        }

        return health_check_policy

    def execute(self):
        """Creates the HealthCheckPolicy in the cluster."""
        logging.info("LWSHealthCheckPolicy class execute")
        health_check_policy = self._build_health_check_policy()
        logging.info("Submitting LWSHealthCheckPolicy body=%s", health_check_policy)

        return k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            group="networking.gke.io",
            version="v1",
            namespace=self.config.namespace,
            plural="healthcheckpolicies",
            body=health_check_policy,
        )

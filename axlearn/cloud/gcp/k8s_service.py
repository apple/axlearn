""" k8s service module."""
import copy
import logging
from typing import Any, Optional

import kubernetes as k8s
from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable, generate_job_name
from axlearn.cloud.gcp.config import default_project
from axlearn.cloud.gcp.utils import custom_leaderworkerset_kwargs
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested


class Service(FlagConfigurable):
    """Service interface"""

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures Service
        Attributes:
            name: The name of LWS  resource.
            project: The poject to use within the k8s cluster.
        """

        name: Required[str] = REQUIRED
        project: Required[str] = REQUIRED

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the service.", **common_kwargs)
        flags.DEFINE_string("project", None, "The GCP project name.", **common_kwargs)

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default("project", default_project())

    def _delete(self):
        """Cleans up the service. Called on termination when all retries are exhausted.

        Note that `_delete` is not called when `_execute` finishes successfully. It is up
        to the implementation of `_execute` to clean up properly.
        """

    def _build_service(self) -> Any:
        """Performs some computation. The return value can be implementation dependent."""
        raise NotImplementedError(type(self))


class LWSService(Service):
    """LWS Service"""

    @config_class
    class Config(Service.Config):
        """Configures Service
        Attributes:
            namespace: The namespace to use within the k8s cluster.
            protocol: protocol for service , ex: TCP, HTTP
            port: the exposed port of service
            targetport: the application port of leader pod
            service_type: Type of Service , ex: ClusterIP
        """

        namespace: str = None
        protocol: Optional[str] = None
        port: Optional[int] = None
        targetport: Optional[int] = None
        service_type: Optional[str] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the service.", **common_kwargs)
        flags.DEFINE_string("namespace", None, "Namespace of the service.", **common_kwargs)
        flags.DEFINE_string("protocol", None, "Protocol of the service.", **common_kwargs)
        flags.DEFINE_string("service_type", None, "Type of the service.", **common_kwargs)
        flags.DEFINE_integer("port", None, "Port of the service.", **common_kwargs)
        flags.DEFINE_integer("targetport", None, "TargetPort of the service.", **common_kwargs)

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        fv.set_default("namespace", fv.namespace or "default")
        fv.set_default("protocol", fv.protocol or "TCP")
        fv.set_default("service_type", fv.service_type or "ClusterIP")
        fv.set_default("port", fv.port or 9000)
        fv.set_default("targetport", fv.targetport or 9000)

    @classmethod
    def default_config(cls):
        return super().default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        logging.info("LWSService class init")
        self._config = copy.deepcopy(cfg)
        self.name = cfg.name + "-service"
        self.protocol = cfg.protocol
        self.port = cfg.port
        self.targetport = cfg.targetport
        self.service_type = cfg.service_type
        self.label_name = cfg.name

    def _build_service(self) -> Nested[Any]:
        """
        Builds a config for a Service
        Returns:
            A nested dict corresponding to a k8s Service config
        """
        logging.info("LWSservice class build")
        logging.info(str(self.config))
        api_kwargs = custom_leaderworkerset_kwargs()

        namespace = "default"
        group = "leaderworkerset.x-k8s.io"  # Replace with actual API group for LWS
        version = "v1"  # Replace with CRD version
        plural = "leaderworkersets"  # Replace with CRD plural name
        lws_name = self.name.split("-service")[0]
        custom_api = k8s.client.CustomObjectsApi()

        # Fetch the CR object
        lws = custom_api.get_namespaced_custom_object(
            group=group, version=version, namespace=namespace, plural=plural, name=lws_name
        )

        return dict(
            metadata=k8s.client.V1ObjectMeta(
                name=self.name,
                owner_references=[
                    k8s.client.V1OwnerReference(
                        api_version=f"{api_kwargs['group']}/{api_kwargs['version']}",
                        kind="LeaderWorkerSet",
                        name=lws_name,  ### self.name is a name+"-service"
                        uid=lws["metadata"]["uid"],
                        controller=True,
                        block_owner_deletion=True,
                    )
                ],
            ),
            spec=k8s.client.V1ServiceSpec(
                selector={"app": self.label_name},
                ports=[
                    k8s.client.V1ServicePort(
                        protocol=self.protocol,
                        port=self.port,
                        target_port=self.targetport,
                    )
                ],
                type=self.service_type,
            ),
        )

    def execute(self):
        logging.info("LWSservice class execute")
        service = self._build_service()
        logging.info("Submitting LWSservice body=%s ", service)
        v1 = k8s.client.CoreV1Api()
        return v1.create_namespaced_service(namespace=self.config.namespace, body=service)

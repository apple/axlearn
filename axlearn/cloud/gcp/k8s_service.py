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
            protocol_list: protocol for service , ex: TCP, HTTP
            ports: the exposed port of service
            target_ports: the application port of leader pod
            service_type: Type of Service , ex: ClusterIP
            port_names: Names of ports in a service
        """

        namespace: str = None
        protocol_list: Optional[list[str]] = None
        ports: Optional[list[str]] = None
        target_ports: Optional[list[str]] = None
        service_type: Optional[str] = None
        port_names: Optional[list[str]] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the service.", **common_kwargs)
        flags.DEFINE_string("namespace", None, "Namespace of the service.", **common_kwargs)
        flags.DEFINE_list("protocol_list", [], "list of Protocols of the service.", **common_kwargs)
        flags.DEFINE_string("service_type", None, "Type of the service.", **common_kwargs)
        flags.DEFINE_list("ports", [], "Ports of the service.", **common_kwargs)
        flags.DEFINE_list("target_ports", [], "Target Ports of the service.", **common_kwargs)
        flags.DEFINE_list(
            "port_names",
            [],
            "Port Names to identify target and port of the service.",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        fv.set_default("namespace", fv.namespace or "default")
        fv.set_default("protocol_list", fv.protocol_list or ["TCP"])
        fv.set_default("service_type", fv.service_type or "ClusterIP")
        fv.set_default("ports", fv.ports or ["9000"])
        fv.set_default("target_ports", fv.target_ports or ["9000"])
        fv.set_default("port_names", fv.port_names or ["tcp_port"])

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        max_ports = max(
            len(fv.protocol_list), len(fv.ports), len(fv.target_ports), len(fv.port_names)
        )
        if (
            (len(fv.protocol_list) != max_ports)
            or (len(fv.ports) != max_ports)
            or (len(fv.target_ports) != max_ports)
            or (len(fv.port_names) != max_ports)
        ):
            raise ValueError(
                "The count of values of protocol_list, ports, target_ports, port_names should match"
            )

        cfg: Service.Config = super().from_flags(fv, **kwargs)
        cfg.ports = fv.ports
        cfg.namespace = fv.namespace
        cfg.name = fv.name
        cfg.protocol_list = fv.protocol_list
        cfg.service_type = fv.service_type
        cfg.ports = fv.ports
        cfg.target_ports = fv.target_ports
        cfg.port_names = fv.port_names
        return cfg

    @classmethod
    def default_config(cls):
        return super().default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        logging.info("LWSService class init")
        self._config = copy.deepcopy(cfg)
        self.name = cfg.name + "-service"
        self.protocol_list = cfg.protocol_list
        self.ports = cfg.ports
        self.target_ports = cfg.target_ports
        self.service_type = cfg.service_type
        self.port_names = cfg.port_names
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
        group = api_kwargs["group"]
        version = api_kwargs["version"]
        plural = api_kwargs["plural"]
        lws_name = self.name.split("-service")[0]
        custom_api = k8s.client.CustomObjectsApi()

        # Fetch the CR object
        lws = custom_api.get_namespaced_custom_object(
            group=group, version=version, namespace=namespace, plural=plural, name=lws_name
        )

        ports_map_list = []
        for i in range(len(self.ports)):
            print(self.protocol_list)
            ports_map_list.append(
                k8s.client.V1ServicePort(
                    protocol=self.protocol_list[i],
                    port=int(self.ports[i]),
                    target_port=int(self.target_ports[i]),
                    name=self.port_names[i],
                )
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
                ports=ports_map_list,
                type=self.service_type,
            ),
        )

    def execute(self):
        logging.info("LWSservice class execute")
        service = self._build_service()
        logging.info("Submitting LWSservice body=%s ", service)
        v1 = k8s.client.CoreV1Api()
        return v1.create_namespaced_service(namespace=self.config.namespace, body=service)

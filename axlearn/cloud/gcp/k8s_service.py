import logging
import shlex
import subprocess
from collections.abc import Sequence
from typing import Any, Optional

import kubernetes as k8s
from absl import flags

from axlearn.cloud.common.bundler import BaseDockerBundler
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.utils import generate_job_name, subprocess_run
from axlearn.cloud.gcp.config import default_env_id, default_project, default_zone
from axlearn.cloud.gcp.jobset_utils import BaseReplicatedJob
from axlearn.cloud.gcp.lws_utils import BaseLeaderWorkerTemplate
from axlearn.cloud.gcp.utils import (
    custom_jobset_kwargs,
    custom_leaderworkerset_kwargs,
    delete_k8s_jobset,
    delete_k8s_leaderworkerset,
    delete_k8s_service,
)
from axlearn.common.config import REQUIRED, ConfigOr, Required, config_class, maybe_instantiate, ConfigBase
from axlearn.common.utils import Nested

class Service:
    """Service interface"""

    @config_class
    class Config(ConfigBase):
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
        super().set_defaults(fv)
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


class LWSService:
    """SLWS ervice"""

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

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        fv.set_default("namespace", fv.namespace or "default")

    def __init__(self, cfg: Config):

        logging.info("LWSService class init")
        self.name = cfg.name + "-service"
        self.config = cfg

    def _delete(self):
        delete_k8s_service(self.name, namespace=self.config.namespace)

    def _build_service(self) -> Nested[Any]:
        """
        Builds a config for a Service
        Returns:
            A nested dict corresponding to a k8s Service config
        """
        logging.info("LWSservice class build")
        logging.info(str(self.config))

        return dict(
            metadata=k8s.client.V1ObjectMeta(name=self.name),
            spec=k8s.client.V1ServiceSpec(
                selector={"app": self.config.name}, #### may be make labels more generic  #####
                ports=[
                    k8s.client.V1ServicePort(
                        protocol=self.config.protocol,
                        port=self.config.port,
                        target_port=self.config.targetport,
                    )
                ],
                type=self.config.service_type,  
            ),
        )

    def execute(self):
        
        logging.info("LWSservice class execute")
        service = self._build_service()
        logging.info("Submitting LWSservice body=%s ", service)
        v1 = k8s.client.CoreV1Api()
        return v1.create_namespaced_service(namespace=self.config.namespace, body=service)
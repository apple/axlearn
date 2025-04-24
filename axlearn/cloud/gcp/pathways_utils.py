# Copyright © 2025 Apple Inc.

"""Utilities for building Pathways Jobset specs."""

import copy
import logging
import os
from typing import Any, Sequence

from axlearn.cloud.common.bastion import BASTION_JOB_VERSION_ENV_VAR
from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.gcp.jobset_utils import (
    _ANNOTATION_NODE_SERVICE_ACCOUNT,
    _METADATA_GOOGLE_INTERNAL_IP,
    BASTION_JOB_VERSION_LABEL,
    BaseReplicatedJob,
    TPUReplicatedJob,
    _LoadBalancer,
)
from axlearn.cloud.gcp.system_characteristics import USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS
from axlearn.common.compiler_options import infer_tpu_type
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested

# The port used by pathways proxy server.
# The specific value is not important, as long as clients and servers use the same port.
_PATHWAYS_PROXY_PORT = 29000
# The port used by pathways resource manager server.
# The specific value is not important, as long as clients and servers use the same port.
_PATHWAYS_RESOURCE_MANAGER_PORT = 29001
# The port used by pathways worker server.
# The specific value is not important, as long as clients and servers use the same port.
_PATHWAYS_WORKER_PORT = 29001
# Pin to specific pathways image version for stable release.
# Oldest available is jax-0.5.1, although axlearn is using jax-0.4.38.
# Verified the backwards compatibility works.
_PATHWAYS_IMAGE_TAG = "jax-0.5.1"
# The docker image used by pathways proxy container.
_PATHWAYS_PROXY_IMAGE = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:{_PATHWAYS_IMAGE_TAG}"
)
# The docker image used by pathways resource manager container and worker container.
_PATHWAYS_SERVER_IMAGE = (
    f"us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:{_PATHWAYS_IMAGE_TAG}"
)
# The container name of pathways resourcemanager.
_PATHWAYS_RESOURCE_MANAGER_CONTAINER_NAME = "pathways-rm"
# The container name of pathways proxy.
_PATHWAYS_PROXY_CONTAINER_NAME = "pathways-proxy"
# The node pool name for pathways-head pod.
_PATHWAYS_HEAD_NODE_POOL_NAME = "pathways-head"
# The k8s replicatedJob name for pathways-head pods.
_PATHWAYS_HEAD_REPLICATED_JOB_NAME = "pathways-head"
# The k8s replicatedJob name for pathways-worker pods.
_PATHWAYS_WORKER_REPLICATED_JOB_NAME = "pathways-worker"


def get_pathways_head_address(job_name: str) -> str:
    """Returns the address of the pathways head pod."""
    # There will be only one pathways-head pod, so it is already 0-0.
    # First 0 means the first replicatedJob of pathways-head,
    # the second 0 means the first pod in the replicatedJob.
    return f"{job_name}-{_PATHWAYS_HEAD_REPLICATED_JOB_NAME}-0-0.{job_name}"


def get_pathways_tpu_version(gke_machine_type: str) -> str:
    """Map from GKE machine type to TPU version.

    References:
    https://github.com/google/pathways-job/blob/4417de7aa23d3c2316e400a3a327512834374475/internal/controller/pathwaysjob_controller.go#L70-L82
    """
    pathways_tpu_devices = {
        # v6e
        "ct6e-standard-4t": "tpuv6e",
        # v5p
        "ct5p-hightpu-4t": "tpuv5",
        # v5e
        "ct5lp-hightpu-4t": "tpuv5e",
        # v4
        "ct4p-hightpu-4t": "tpuv4",
    }
    return pathways_tpu_devices[gke_machine_type.lower()]


class PathwaysReplicatedJob(BaseReplicatedJob):
    """Builds a replicated jobspec for Pathways on TPU, to be used with JobSet API."""

    @config_class
    class Config(BaseReplicatedJob.Config):
        inner: Required[TPUReplicatedJob.Config] = REQUIRED

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        return cfg.set(inner=TPUReplicatedJob.default_config())

    def __init__(self, cfg: BaseReplicatedJob.Config, *, bundler: Bundler):
        super().__init__(cfg, bundler=bundler)
        self._bundler = bundler
        self._inner: TPUReplicatedJob = cfg.inner.instantiate(bundler=self._bundler)
        self._tpu_type = infer_tpu_type(cfg.inner.accelerator.instance_type)
        if self._tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {self._tpu_type}")
        if cfg.inner.enable_pre_provisioner:
            raise NotImplementedError("Pre-provisioner is currently not supported")

    def _update_env_list(self, env_list: list[dict], name: str, value: str):
        for env in env_list:
            if env.get("name") == name:
                env["value"] = value
                return
        env_list.append({"name": name, "value": value})

    def _build_pathways_head_container(self) -> dict:
        """Build the container for the 'pathways-head' role."""
        # pylint: disable-next=protected-access
        container = self._inner._build_container()

        head_container = copy.deepcopy(container)

        env_list = head_container.get("env", [])
        self._update_env_list(
            env_list,
            "JAX_BACKEND_TARGET",
            f"grpc://localhost:{_PATHWAYS_PROXY_PORT}",
        )
        self._update_env_list(env_list, "XCLOUD_ENVIRONMENT", "GCP")
        self._update_env_list(env_list, "JAX_PLATFORMS", "proxy")
        self._update_env_list(env_list, "ENABLE_PATHWAYS_PERSISTENCE", "1")
        self._update_env_list(env_list, "TPU_SKIP_MDS_QUERY", "true")

        env_list.append(
            {
                "name": "HOST_ADDRESS",
                "valueFrom": {
                    "fieldRef": {"fieldPath": "metadata.labels['jobset.sigs.k8s.io/coordinator']"}
                },
            }
        )

        head_container["env"] = env_list

        # TODO(ethanli): Define the resources for pathways head.
        resources = {
            "requests": {"cpu": "1000m", "memory": "32Gi"},
            "limits": {"cpu": "32000m", "memory": "128Gi"},
        }
        head_container["resources"] = resources

        return head_container

    def _build_pathways_head_sidecar_containers(self) -> list[Nested[Any]]:
        """Builds a config for the pathways containers which orchestrate resource management
        and pathways proxy communications.

        Returns:
            A list of nested dict corresponding to a pathways resource
            manager config and a pathways proxy config.
        """

        cfg: TPUReplicatedJob.Config = self._inner.config

        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        staging_location = f"{cfg.output_dir}/pathways-staging"
        pathways_tpu_version = get_pathways_tpu_version(system.gce_machine_type)

        return [
            dict(
                name=_PATHWAYS_PROXY_CONTAINER_NAME,
                image=_PATHWAYS_PROXY_IMAGE,
                # https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/#pod-sidecar-containers
                # SideCar container is an init container with restartPolicy as "Always".
                restartPolicy="Always",
                args=[
                    f"--resource_manager_address=localhost:{_PATHWAYS_RESOURCE_MANAGER_PORT}",
                    f"--server_port={_PATHWAYS_PROXY_PORT}",
                    f"--gcs_scratch_location={staging_location}",
                ],
                ports=[dict(containerPort=_PATHWAYS_PROXY_PORT)],
            ),
            dict(
                name=_PATHWAYS_RESOURCE_MANAGER_CONTAINER_NAME,
                image=_PATHWAYS_SERVER_IMAGE,
                # https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/#pod-sidecar-containers
                # SideCar container is an init container with restartPolicy as "Always".
                restartPolicy="Always",
                env=[
                    {
                        "name": "TPU_SKIP_MDS_QUERY",
                        "value": "true",
                    },
                ],
                args=[
                    f"--server_port={_PATHWAYS_RESOURCE_MANAGER_PORT}",
                    "--node_type=resource_manager",
                    f"--instance_count={cfg.accelerator.num_replicas}",
                    f"--instance_type={pathways_tpu_version}:{system.topology}",
                    f"--gcs_scratch_location={staging_location}",
                ],
            ),
        ]

    def _build_pathways_head_pod(self) -> Nested[Any]:
        """Builds a pathways head pod. The pod includes a head container,
        a proxy container and a resource manager container.
        """

        cfg: TPUReplicatedJob.Config = self._inner.config

        annotations, labels, volumes, tolerations = {}, {}, [], []

        if os.environ.get(BASTION_JOB_VERSION_ENV_VAR):
            labels.update({BASTION_JOB_VERSION_LABEL: os.environ.get(BASTION_JOB_VERSION_ENV_VAR)})

        volumes.append(dict(name="shared-output", emptyDir={}))

        if cfg.gcsfuse_mount:
            annotations.update(
                {
                    "gke-gcsfuse/volumes": "true",
                    "gke-gcsfuse/cpu-limit": cfg.gcsfuse_mount.cpu,
                    "gke-gcsfuse/memory-limit": cfg.gcsfuse_mount.memory,
                    "gke-gcsfuse/ephemeral-storage-limit": cfg.gcsfuse_mount.ephemeral_gb,
                }
            )

        node_selector = {
            "cloud.google.com/gke-nodepool": _PATHWAYS_HEAD_NODE_POOL_NAME,
        }

        head_container = self._build_pathways_head_container()
        init_containers = self._build_pathways_head_sidecar_containers()

        # Hardcode metadata.google.internal ip address to avoid transient DNS resolution issue.
        metadata_host_alias = dict(
            ip=_METADATA_GOOGLE_INTERNAL_IP,
            hostnames=["metadata", "metadata.google.internal"],
        )
        head_pod_spec = {
            "terminationGracePeriodSeconds": 60,
            # Fail if any pod fails, and allow retries to happen at JobSet level.
            "restartPolicy": "Never",
            "hostAliases": [metadata_host_alias],
            "nodeSelector": node_selector,
            "tolerations": tolerations,
            "containers": [head_container],
            "initContainers": init_containers,
            "volumes": volumes,
            "serviceAccountName": cfg.service_account,
            "hostNetwork": True,
            "dnsPolicy": "ClusterFirstWithHostNet",
        }

        if cfg.priority_class:
            head_pod_spec["priorityClassName"] = cfg.priority_class

        return {
            "metadata": {
                "annotations": annotations,
                "labels": labels,
            },
            "spec": head_pod_spec,
        }

    def _build_pathways_head_job(self):
        logging.debug("Building a head job.")
        cfg: TPUReplicatedJob.Config = self._inner.config

        annotations = _LoadBalancer(
            jobset_name=cfg.name, replicated_job_name=_PATHWAYS_HEAD_REPLICATED_JOB_NAME
        ).metadata
        spec = dict(
            parallelism=1,
            completions=1,
            backoffLimit=0,
            template=self._build_pathways_head_pod(),
        )
        head_job = dict(
            metadata=dict(annotations=annotations),
            spec=spec,
        )

        return head_job

    def _build_pathways_worker_container(self) -> dict:
        """Build the container for the 'pathways-worker' role."""
        cfg: TPUReplicatedJob.Config = self._inner.config
        # pylint: disable-next=protected-access
        container = self._inner._build_container()

        worker_container = copy.deepcopy(container)
        env_list = worker_container.get("env", [])

        self._update_env_list(
            env_list, "MEGASCALE_COORDINATOR_ADDRESS", f"{get_pathways_head_address(cfg.name)}"
        )
        env_list.extend(
            [
                {
                    "name": "MEGASCALE_NUM_SLICES",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": (
                                "metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']"
                            )
                        }
                    },
                },
                {
                    "name": "MEGASCALE_SLICE_ID",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": "metadata.labels['jobset.sigs.k8s.io/job-index']",
                        }
                    },
                },
            ]
        )
        worker_container["env"] = env_list

        worker_container["args"] = [
            f"--server_port={_PATHWAYS_WORKER_PORT}",
            f"--resource_manager_address={get_pathways_head_address(cfg.name)}:"
            + f"{_PATHWAYS_RESOURCE_MANAGER_PORT}",
            f"--gcs_scratch_location={cfg.output_dir}/pathways-staging",
        ]

        worker_container["image"] = _PATHWAYS_SERVER_IMAGE

        ports = worker_container.get("ports", [])
        ports.append({"containerPort": _PATHWAYS_WORKER_PORT})
        worker_container["ports"] = ports

        # Command will be executed by the head node, and it will compile the model and
        # distribute works to workers.
        # So workers doesn't need to execute the command by themselves.
        worker_container.pop("command")

        return worker_container

    def _build_pathways_worker_pod(self) -> Nested[Any]:
        """Conoverts a worker pod to a new pod for the 'pathways-workers' role."""
        cfg: TPUReplicatedJob.Config = self._inner.config
        # pylint: disable-next=protected-access
        pod = self._inner._build_pod()
        worker_pod = copy.deepcopy(pod)

        pod_spec = worker_pod.get("spec", {})
        # Use default value - OnFailure.
        pod_spec.pop("restartPolicy")
        # Need to enable host network to improve head <> worker communucation.
        # It should not be required but current Pathways only support host network.
        pod_spec["hostNetwork"] = True
        # Only set dnsPolicy if it's not already set
        pod_spec["dnsPolicy"] = "ClusterFirstWithHostNet"
        pod_spec["containers"] = [self._build_pathways_worker_container()]
        worker_pod["spec"] = pod_spec

        # Service account for nodes.
        if cfg.service_account:
            metadata = worker_pod.get("metadata", {})
            annotations = metadata.get("annotations", {})
            node_service_account = f"{cfg.service_account}@{cfg.project}.iam.gserviceaccount.com"
            annotations.update(
                {
                    _ANNOTATION_NODE_SERVICE_ACCOUNT: node_service_account,
                }
            )
            worker_pod["metadata"]["annotations"] = annotations

        return worker_pod

    def _build_pathways_worker_job(self):
        """See `BaseReplicatedJob` docstring for details."""

        logging.debug("Building a worker job.")

        cfg: TPUReplicatedJob.Config = self._inner.config

        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]

        annotations = _LoadBalancer(
            jobset_name=cfg.name, replicated_job_name=_PATHWAYS_WORKER_REPLICATED_JOB_NAME
        ).metadata

        annotations.update(
            {"alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"}
        )

        spec = dict(
            parallelism=system.vms_per_slice,
            completions=system.vms_per_slice,
            # Default value for suspend and resume.
            # References:
            # https://github.com/google/pathways-job/blob/4417de7aa23d3c2316e400a3a327512834374475/internal/controller/pathwaysjob_controller.go#L651
            backoffLimit=system.vms_per_slice * 4,
            template=self._build_pathways_worker_pod(),
        )
        worker_job = dict(
            metadata=dict(annotations=annotations),
            spec=spec,
        )
        return worker_job

    def __call__(self) -> Sequence[Nested[Any]]:
        cfg: TPUReplicatedJob.Config = self._inner.config

        replicated_jobs = [
            dict(
                name=_PATHWAYS_HEAD_REPLICATED_JOB_NAME,
                replicas=1,
                template=self._build_pathways_head_job(),
            ),
            dict(
                name=_PATHWAYS_WORKER_REPLICATED_JOB_NAME,
                replicas=cfg.accelerator.num_replicas,
                template=self._build_pathways_worker_job(),
            ),
        ]

        return replicated_jobs

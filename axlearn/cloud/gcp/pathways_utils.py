# Copyright © 2025 Apple Inc.

"""Utilities for building Pathways Jobset specs."""

import copy
import logging
import os
from typing import Any, Optional, Sequence, Union

from absl import flags

from axlearn.cloud.common.bastion import BASTION_JOB_VERSION_ENV_VAR
from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import parse_kv_flags
from axlearn.cloud.gcp.jobset_utils import (
    _ANNOTATION_NODE_SERVICE_ACCOUNT,
    _METADATA_GOOGLE_INTERNAL_IP,
    BASTION_JOB_VERSION_LABEL,
    BaseReplicatedJob,
    TPUReplicatedJob,
    _LoadBalancer,
)
from axlearn.cloud.gcp.lws_utils import BaseLeaderWorkerTemplate, TPULeaderWorkerTemplate
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
    support_twisted_topology,
)
from axlearn.cloud.gcp.tpu import infer_tpu_workers
from axlearn.cloud.gcp.utils import validate_jobset_name
from axlearn.common.compiler_options import (
    default_xla_options,
    infer_tpu_type,
    xla_flags_from_options,
)
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
# There is no guarantee that this image will work with newer Jax releases.
# This image version extends GRPC timeout for long context models, based on jax-0.5.3-patch060625
# This image extends GRPC timeout for long context models.
_PATHWAYS_IMAGE_TAG = "shm_proxy_settings"
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
# The k8s replicatedJob name for pathways-head pods.
_PATHWAYS_HEAD_REPLICATED_JOB_NAME = "pathways-head"
# The k8s replicatedJob name for pathways-worker pods.
_PATHWAYS_WORKER_REPLICATED_JOB_NAME = "pathways-worker"

# Add node-selector for cpu workload to avoid sharing nodes with system services.
_PATHWAYS_HEAD_NODE_POOL_SELECTOR_KEY = "axlearn/nodepool_type"
_PATHWAYS_HEAD_NODE_POOL_SELECTOR_VALUE = "workload"
# The back off limit of pathways pods.
# Note that the head pod will back of exact this many times.
# While workers will share #workers * _PATHWAYS_BACK_OFF_LIMIT total times.
_PATHWAYS_BACK_OFF_LIMIT = 32


def parse_xla_flag_value(value: str) -> Union[int, bool, str]:
    """Attempts to convert an XLA flag string value to int.

    If conversion fails, returns the original string (stripped).
    """
    stripped_value_str = value.strip()
    try:
        return int(stripped_value_str)
    except ValueError:
        return stripped_value_str


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


def get_megascale_options(
    xla_options: dict[str, Union[str, bool, int]]
) -> dict[str, Union[str, bool, int]]:
    """Filters XLA options for those pertaining to Megascale.

    Args:
        xla_options: A dictionary of XLA options.

    Returns:
        A dictionary containing only Megascale-related XLA options
        (those starting with 'megascale').
    """
    return {k: v for k, v in xla_options.items() if k.startswith("megascale")}


def get_xla_options(
    xla_options: dict[str, Union[str, bool, int]]
) -> dict[str, Union[str, bool, int]]:
    """Filters XLA options for those starting with 'xla_'.

    Args:
        xla_options: A dictionary of XLA options.

    Returns:
        A dictionary containing only XLA-specific options (those starting with 'xla').
    """
    return {k: v for k, v in xla_options.items() if k.startswith("xla_")}


def round_up_to_power_of_2(n):
    """
    Rounds an integer up to the nearest power of 2.

    Args:
        n (int): The number to round up. Must be a positive integer.

    Returns:
        int: The smallest power of 2 that is greater than or equal to n.

    Examples:
        round_up_to_power_of_2(7)   -> 8
        round_up_to_power_of_2(8)   -> 8
        round_up_to_power_of_2(9)   -> 16
        round_up_to_power_of_2(32)  -> 32
    """
    assert isinstance(n, int) and n > 0
    return 1 << (n - 1).bit_length()


class PathwaysReplicatedJob(BaseReplicatedJob):
    """Builds a replicated jobspec for Pathways on TPU, to be used with JobSet API."""

    @config_class
    class Config(BaseReplicatedJob.Config):
        """Configures PathwaysReplicatedJob.

        Attributes:
            inner: The wrapped TPUReplicatedJob configuration.
            pathways_head_cpu: CPU request for pathways-head container.
            pathways_head_mem: Memory request for pathways-head container.
        """

        inner: Required[TPUReplicatedJob.Config] = REQUIRED
        pathways_xla_flags: list[str] = []
        pathways_head_cpu: Optional[str] = None
        pathways_head_mem: Optional[str] = None

    @classmethod
    def define_flags(cls, fv):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # XLA flags and megascale flags have to be passed at jobset creation time.
        # The XLA flags automatically get passed to the pathways proxy and the
        # Megascale flags get passed to pathways workers. A single flag is used since
        # the implementation details could change later.
        flags.DEFINE_list(
            "pathways_xla_flags",
            [],
            "Set XLA and Megascale flags. Defaults are set by compiler_options.py. "
            "Example: 'xla_tpu_x=24,megascale_y=true'",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "pathways_head_cpu",
            None,
            "CPU request for pathways-head container in cores. Default is 1 core.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "pathways_head_mem",
            None,
            "Memory request for pathways-head container in GiB. Default is 16GiB",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv):
        super().set_defaults(fv)
        fv.set_default("pathways_head_cpu", fv.pathways_head_cpu or "1")
        fv.set_default("pathways_head_mem", fv.pathways_head_mem or "16")

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        return cfg.set(inner=TPUReplicatedJob.default_config())

    def __init__(self, cfg: BaseReplicatedJob.Config, *, bundler: Bundler):
        super().__init__(cfg, bundler=bundler)
        self._bundler = bundler
        self._inner: TPUReplicatedJob = cfg.inner.instantiate(bundler=self._bundler)
        pathways_cfg: PathwaysReplicatedJob.Config = self.config
        self._tpu_type = infer_tpu_type(cfg.inner.accelerator.instance_type)
        if self._tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {self._tpu_type}")
        if cfg.inner.enable_pre_provisioner:
            raise NotImplementedError("Pre-provisioner is currently not supported")
        self._is_single_head = True
        xla_and_mxla_options = default_xla_options(
            instance_type=self._tpu_type,
            num_slices=cfg.inner.accelerator.num_replicas,
            backend="tpu",
        )
        pathways_xla_flags = parse_kv_flags(pathways_cfg.pathways_xla_flags, delimiter="=")
        for k, v in pathways_xla_flags.items():
            k = k.lstrip("--")
            v = parse_xla_flag_value(v)
            xla_and_mxla_options[k] = v

        # Needs to be passed to pathways-proxy.
        self._xla_options = get_xla_options(xla_and_mxla_options)
        # Needs to be passed as command arguments to each pathways-worker.
        self._mxla_options = get_megascale_options(xla_and_mxla_options)

        # Validate pathways-head name length.
        validate_jobset_name(
            name=cfg.inner.name,
            num_workers=1,
            num_replicas=1,
            job_name=_PATHWAYS_HEAD_REPLICATED_JOB_NAME,
        )
        # Validate pathways-worker name length.
        validate_jobset_name(
            name=cfg.inner.name,
            num_workers=infer_tpu_workers(self._tpu_type),
            num_replicas=cfg.inner.accelerator.num_replicas,
            job_name=_PATHWAYS_WORKER_REPLICATED_JOB_NAME,
        )

    def _update_env_list(self, env_list: list[dict], name: str, value: str):
        for env in env_list:
            if env.get("name") == name:
                env["value"] = value
                return
        env_list.append({"name": name, "value": value})

    def _get_pathways_head_address(
        self, pathways_worker_replicated_job_index: Optional[int] = None
    ) -> str:
        """Returns the address of the pathways-head pod.
        There will be only one pathways-head pod, so it is always 0-0.
        First 0 means the first replicatedJob of pathways-head,
        the second 0 means the first pod in the replicatedJob.
        """
        assert pathways_worker_replicated_job_index is None
        cfg: PathwaysReplicatedJob.Config = self.config
        return f"{cfg.name}-{_PATHWAYS_HEAD_REPLICATED_JOB_NAME}-0-0.{cfg.name}"

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
        # This is required to be able to run a Jax client when using
        # IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS=true.
        # In Jax 0.6.2 and beyond this flag can be renamed to
        # IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS as well.
        self._update_env_list(env_list, "TEST_UNDECLARED_OUTPUTS_DIR", "true")
        # Threshold for using shared memory between Jax client and Pathways proxy.
        # Setting it to 1 byte so effectively all Jax device_put use shared memory.
        self._update_env_list(env_list, "IFRT_PROXY_LARGE_TRANSFER_THRESHOLD", "1")
        self._update_env_list(
            env_list, "IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY", "/tmp/ifrt_proxy"
        )
        env_list.append(
            {
                "name": "HOST_ADDRESS",
                "valueFrom": {
                    "fieldRef": {"fieldPath": "metadata.labels['jobset.sigs.k8s.io/coordinator']"}
                },
            }
        )

        # pylint: disable=line-too-long
        env_list.append(
            {
                "name": "NUM_REPLICAS",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.annotations['jobset.sigs.k8s.io/replicatedjob-replicas']"
                    }
                },
            }
        )
        # pylint: enable=line-too-long

        env_list.append(
            {
                "name": "REPLICA_ID",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "metadata.annotations['jobset.sigs.k8s.io/job-index']"
                    }
                },
            }
        )

        head_container["env"] = env_list

        cpu_req = f"{float(self.config.pathways_head_cpu) * 1000}m"
        mem_req = f"{self.config.pathways_head_mem}Gi"
        resources = {
            "requests": {"cpu": cpu_req, "memory": mem_req},
        }
        head_container["resources"] = resources

        volume_mounts = head_container.get("volumeMounts", [])
        volume_mounts.append(dict(name="shared-memory", mountPath="/tmp/ifrt_proxy"))
        head_container["volumeMounts"] = volume_mounts

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

        # If multi-head, every pathways-head will only
        # be connected to one pathways instance (a pathways-worker replicated job).
        pathways_instance_count = cfg.accelerator.num_replicas if self._is_single_head else 1

        cmd_args = [
            f"--resource_manager_address=localhost:{_PATHWAYS_RESOURCE_MANAGER_PORT}",
            f"--server_port={_PATHWAYS_PROXY_PORT}",
            f"--gcs_scratch_location={staging_location}",
        ]
        cmd_args.extend(xla_flags_from_options(self._xla_options).split())

        instance_type = f"{pathways_tpu_version}:{system.topology}"
        if support_twisted_topology(self._tpu_type):
            instance_type = f"{instance_type}_untwisted"
        return [
            dict(
                name=_PATHWAYS_PROXY_CONTAINER_NAME,
                image=_PATHWAYS_PROXY_IMAGE,
                # https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/#pod-sidecar-containers
                # SideCar container is an init container with restartPolicy as "Always".
                restartPolicy="Always",
                args=cmd_args,
                env=[
                    # This is required for GKE Workload Identity and Mac Jax Client support.
                    # TODO(samos123): Remove this once this becomes the default.
                    {"name": "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS", "value": "true"},
                    {"name": "XLA_FLAGS", "value": f"--xla_dump_to=/output/{cfg.name}/xla"},
                    {
                        "name": "IFRT_PROXY_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY",
                        "value": "/tmp/ifrt_proxy",
                    },
                ],
                ports=[dict(containerPort=_PATHWAYS_PROXY_PORT)],
                volumeMounts=[
                    dict(name="shared-output", mountPath="/output"),
                    dict(name="shared-memory", mountPath="/tmp/ifrt_proxy"),
                ],
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
                    f"--instance_count={pathways_instance_count}",
                    f"--instance_type={instance_type}",
                    f"--gcs_scratch_location={staging_location}",
                ],
                volumeMounts=[dict(name="shared-output", mountPath="/output")],
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
        volumes.append(dict(name="shared-memory", emptyDir=dict(medium="Memory")))

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
            _PATHWAYS_HEAD_NODE_POOL_SELECTOR_KEY: _PATHWAYS_HEAD_NODE_POOL_SELECTOR_VALUE,
        }

        head_container = self._build_pathways_head_container()
        init_containers = [
            *self._build_pathways_head_sidecar_containers(),
            # pylint: disable-next=protected-access
            self._inner._build_uploader_container(),
        ]

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
            backoffLimit=_PATHWAYS_BACK_OFF_LIMIT,
            template=self._build_pathways_head_pod(),
        )
        head_job = dict(
            metadata=dict(annotations=annotations),
            spec=spec,
        )

        return head_job

    def _build_pathways_worker_container(
        self, pathways_worker_replicated_job_index: Optional[int] = None
    ) -> dict:
        """Build the container for the 'pathways-worker' role."""
        cfg: TPUReplicatedJob.Config = self._inner.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        host_memory = GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS[system.gce_machine_type]
        # pylint: disable-next=protected-access
        container = self._inner._build_container()

        worker_container = copy.deepcopy(container)
        env_list = worker_container.get("env", [])

        pathways_head_address = self._get_pathways_head_address(
            pathways_worker_replicated_job_index=pathways_worker_replicated_job_index
        )

        self._update_env_list(env_list, "MEGASCALE_COORDINATOR_ADDRESS", pathways_head_address)

        if self._is_single_head:
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
        else:
            env_list.extend(
                [
                    {
                        "name": "MEGASCALE_NUM_SLICES",
                        "value": "1",
                    },
                    {
                        "name": "MEGASCALE_SLICE_ID",
                        "value": "0",
                    },
                ]
            )

        worker_container["env"] = env_list

        worker_container["args"] = [
            f"--server_port={_PATHWAYS_WORKER_PORT}",
            f"--resource_manager_address={pathways_head_address}:"
            + f"{_PATHWAYS_RESOURCE_MANAGER_PORT}",
            f"--gcs_scratch_location={cfg.output_dir}/pathways-staging",
            # Recycling host memory gives a slight increase in performance.
            "--tpu_pinned_host_allocation_recycle=true",
            # The flag below is needed for better H2D performance.
            # We use 1/4 of the host memory, rounding up to power of 2 as premapped buffer.
            # Note that pathways worker requires this flag to be a power of 2.
            f"--tpu_premapped_buffer_size={round_up_to_power_of_2(host_memory//4)*(1<<30)}",
        ]
        mega_scale_args = xla_flags_from_options(self._mxla_options).split()
        worker_container["args"].extend(mega_scale_args)

        worker_container["image"] = _PATHWAYS_SERVER_IMAGE

        ports = worker_container.get("ports", [])
        ports.append({"containerPort": _PATHWAYS_WORKER_PORT})
        worker_container["ports"] = ports

        # Command will be executed by the head node, and it will compile the model and
        # distribute works to workers.
        # So workers doesn't need to execute the command by themselves.
        worker_container.pop("command")

        return worker_container

    def _build_pathways_worker_pod(
        self, pathways_worker_replicated_job_index: Optional[int] = None
    ) -> Nested[Any]:
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
        pod_spec["containers"] = [
            self._build_pathways_worker_container(pathways_worker_replicated_job_index)
        ]
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

    def _build_pathways_worker_job(
        self,
        pathways_worker_replicated_job_index: Optional[int] = None,
    ):
        """See `BaseReplicatedJob` docstring for details."""

        logging.debug("Building a worker job.")

        cfg: TPUReplicatedJob.Config = self._inner.config

        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]

        replicated_job_name = (
            _PATHWAYS_WORKER_REPLICATED_JOB_NAME
            if self._is_single_head
            else f"{_PATHWAYS_WORKER_REPLICATED_JOB_NAME}-{pathways_worker_replicated_job_index}"
        )

        annotations = _LoadBalancer(
            jobset_name=cfg.name, replicated_job_name=replicated_job_name
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
            backoffLimit=system.vms_per_slice * _PATHWAYS_BACK_OFF_LIMIT,
            template=self._build_pathways_worker_pod(pathways_worker_replicated_job_index),
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


# TODO (ethanli): Consider refactoring with the modifiers pattern.
class PathwaysMultiheadReplicatedJob(PathwaysReplicatedJob):
    """Builds a replicated jobspec for Pathways on TPU for multi-host inference use case,
    to be used with JobSet API. There will be one pathways-head job for each pathways-worker
    replicated job. For a job with num_replicas=N, there will be N pathways-head job
    and N pathways-worker replicated jobs.
    """

    def __init__(self, cfg: PathwaysReplicatedJob.Config, *, bundler: Bundler):
        super().__init__(cfg, bundler=bundler)
        self._is_single_head = False
        cfg: PathwaysMultiheadReplicatedJob.Config = self.config
        # Validate pathways-head name length.
        validate_jobset_name(
            name=cfg.inner.name,
            num_workers=1,
            num_replicas=cfg.inner.accelerator.num_replicas,
            job_name=_PATHWAYS_HEAD_REPLICATED_JOB_NAME,
        )
        # Validate pathways-worker name length.
        # pytype: disable=wrong-arg-types
        validate_jobset_name(
            name=cfg.inner.name,
            num_workers=infer_tpu_workers(self._tpu_type),
            # In the multi-head pathways setup, there is only one
            # replica of replicated job per worker group. And we have
            # num_replicas of such replicated job. So the k8s format of
            # num_replicas is always {replica_index}-0.
            num_replicas=f"{cfg.inner.accelerator.num_replicas}-0",
            job_name=_PATHWAYS_WORKER_REPLICATED_JOB_NAME,
        )
        # pytype: enable=wrong-arg-types

    def _get_pathways_head_address(
        self, pathways_worker_replicated_job_index: Optional[int] = None
    ) -> str:
        """Returns the address of the pathways head pod.
        In the multi-head Pathways setup, there will be one pathways-head pod
        per corresponding pathways-worker k8s job. The pathways-workers from
        the replicated_job of specified index is configured to connect to
        their corresponding pathways-head pod.

        Args:
            pathways_worker_replicated_job_index: the index of the pathways-workers replicated job.

        Returns:
            The network address of the pathways-head pod.
        """
        cfg: PathwaysMultiheadReplicatedJob.Config = self.config

        return (
            f"{cfg.name}-{_PATHWAYS_HEAD_REPLICATED_JOB_NAME}"
            f"-{pathways_worker_replicated_job_index}-0.{cfg.name}"
        )

    def __call__(self) -> Sequence[Nested[Any]]:
        cfg: TPUReplicatedJob.Config = self._inner.config

        replicated_jobs = [
            dict(
                name=_PATHWAYS_HEAD_REPLICATED_JOB_NAME,
                replicas=cfg.accelerator.num_replicas,
                template=self._build_pathways_head_job(),
            ),
        ]

        for i in range(0, cfg.accelerator.num_replicas):
            replicated_jobs.append(
                dict(
                    name=f"{_PATHWAYS_WORKER_REPLICATED_JOB_NAME}-{i}",
                    replicas=1,
                    template=self._build_pathways_worker_job(i),
                ),
            )

        return replicated_jobs


class PathwaysLeaderWorkerTemplate(BaseLeaderWorkerTemplate):
    """Builds a LeaderWorkerTemplate spec for TPUs"""

    @config_class
    class Config(BaseLeaderWorkerTemplate.Config):
        """Configures PathwaysLeaderWorkerTemplate
        Attributes:
            inner: The wrapped TPUReplicatedJob configuration.
            pathways_head_cpu: CPU request for pathways-head container.
            pathways_head_mem: Memory request for pathways-head container.
        """

        inner: Required[TPULeaderWorkerTemplate.Config] = REQUIRED
        pathways_xla_flags: list[str] = []
        pathways_head_cpu: Optional[str] = None
        pathways_head_mem: Optional[str] = None

        target_port: Optional[int] = None
        enable_service: bool = None

    @classmethod
    def define_flags(cls, fv):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # XLA flags and megascale flags have to be passed at jobset creation time.
        # The XLA flags automatically get passed to the pathways proxy and the
        # Megascale flags get passed to pathways workers. A single flag is used since
        # the implementation details could change later.
        flags.DEFINE_list(
            "pathways_xla_flags",
            [],
            "Set XLA and Megascale flags. Defaults are set by compiler_options.py. "
            "Example: 'xla_tpu_x=24,megascale_y=true'",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "pathways_head_cpu",
            None,
            "CPU request for pathways-head container in cores. Default is 1 core.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "pathways_head_mem",
            None,
            "Memory request for pathways-head container in GiB. Default is 16GiB",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "enable_service",
            False,
            "Whether to enable creation of service for LWS",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "target_port",
            None,
            "port where a service can access application, set at head container",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv):
        super().set_defaults(fv)
        fv.set_default("pathways_head_cpu", fv.pathways_head_cpu or "1")
        fv.set_default("pathways_head_mem", fv.pathways_head_mem or "16")
        fv.set_default("target_port", fv.target_port or 9000)
        fv.set_default("enable_service", fv.enable_service or False)

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        return cfg.set(inner=TPULeaderWorkerTemplate.default_config())

    def __init__(self, cfg, *, bundler):
        super().__init__(cfg, bundler=bundler)
        self._bundler = bundler
        self._inner: TPULeaderWorkerTemplate = cfg.inner.instantiate(bundler=self._bundler)
        self._tpu_type = infer_tpu_type(cfg.accelerator.instance_type)
        if self._tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {self._tpu_type}")

    def _build_pathways_worker_container(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self.config
        # pylint: disable-next=protected-access
        container = self._inner._build_container()

        worker_container = copy.deepcopy(container)
        args = [
            f"--server_port={_PATHWAYS_WORKER_PORT}",
            "--resource_manager_address=$(LWS_LEADER_ADDRESS):"
            + f"{_PATHWAYS_RESOURCE_MANAGER_PORT}",
            f"--gcs_scratch_location={cfg.output_dir}/pathways-staging",
        ]
        worker_container["args"] = args
        ports = worker_container.get("ports", [])
        ports.append({"containerPort": _PATHWAYS_WORKER_PORT})
        worker_container["ports"] = ports
        worker_container["image"] = _PATHWAYS_SERVER_IMAGE

        worker_container.pop("command")
        return worker_container

    def build_worker_pod(self) -> dict:
        # pylint: disable-next=protected-access
        cfg: TPULeaderWorkerTemplate.Config = self._inner.config
        # pylint: disable-next=protected-access
        pod = self._inner._build_pod()
        worker_pod = copy.deepcopy(pod)

        pod_spec = worker_pod.get("spec", {})
        pod_spec.pop("restartPolicy")
        pod_spec["HostNetwork"] = True
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

    def _build_pathways_proxy_container(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self._inner.config
        staging_location = f"{cfg.output_dir}/pathways-staging"

        return dict(
            name=_PATHWAYS_PROXY_CONTAINER_NAME,
            image=_PATHWAYS_PROXY_IMAGE,
            args=[
                f"--resource_manager_address=localhost:{_PATHWAYS_RESOURCE_MANAGER_PORT}",
                f"--server_port={_PATHWAYS_PROXY_PORT}",
                f"--gcs_scratch_location={staging_location}",
            ],
            env=[{"name": "IFRT_PROXY_USE_INSECURE_GRPC_CREDENTIALS", "value": "true"}],
            ports=[dict(containerPort=_PATHWAYS_PROXY_PORT)],
        )

    def _build_pathways_rm_container(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self._inner.config
        staging_location = f"{cfg.output_dir}/pathways-staging"

        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        pathways_tpu_version = get_pathways_tpu_version(system.gce_machine_type)

        return dict(
            name=_PATHWAYS_RESOURCE_MANAGER_CONTAINER_NAME,
            image=_PATHWAYS_SERVER_IMAGE,
            env=[
                {
                    "name": "TPU_SKIP_MDS_QUERY",
                    "value": "true",
                },
                {
                    "name": "HOST_ADDRESS",
                    "value": "$(LWS_LEADER_ADDRESS)",
                },
            ],
            args=[
                f"--server_port={_PATHWAYS_RESOURCE_MANAGER_PORT}",
                "--node_type=resource_manager",
                "--instance_count=1",
                f"--instance_type={pathways_tpu_version}:{system.topology}",
                f"--gcs_scratch_location={staging_location}",
            ],
            ports=[dict(containerPort=_PATHWAYS_RESOURCE_MANAGER_PORT)],
        )

    def _build_head_container(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self._inner.config
        cpu_req = f"{float(self.config.pathways_head_cpu) * 1000}m"
        mem_req = f"{self.config.pathways_head_mem}Gi"
        resources = {
            "requests": {"cpu": cpu_req, "memory": mem_req},
            "limits": {"cpu": cpu_req, "memory": mem_req},
        }
        return dict(
            name=cfg.name,
            image=cfg.image_id or self._bundler.id(cfg.name),
            command=["bash", "-c", cfg.command],
            env=[
                {
                    "name": "XCLOUD_ENVIRONMENT",
                    "value": "GCP",
                },
                {
                    "name": "JAX_PLATFORMS",
                    "value": "proxy",
                },
                {
                    "name": "JAX_BACKEND_TARGET",
                    "value": f"grpc://$(LWS_LEADER_ADDRESS):{_PATHWAYS_PROXY_PORT}",
                },
                {
                    "name": "TEST_UNDECLARED_OUTPUTS_DIR",
                    "value": "true",
                },
            ],
            imagePullPolicy="Always",
            resources=resources,
            ports=[dict(containerPort=self.config.target_port)]
            if self.config.enable_service
            else [],
        )

    def build_leader_pod(self) -> Nested[Any]:
        # pylint: disable-next=protected-access
        cfg: TPUReplicatedJob.Config = self._inner.config

        annotations, labels, volumes, tolerations = {}, {}, [], []

        if os.environ.get(BASTION_JOB_VERSION_ENV_VAR):
            labels.update({BASTION_JOB_VERSION_LABEL: os.environ.get(BASTION_JOB_VERSION_ENV_VAR)})

        volumes.append(dict(name="shared-output", emptyDir={}))
        labels = {"app": cfg.name}

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
            _PATHWAYS_HEAD_NODE_POOL_SELECTOR_KEY: _PATHWAYS_HEAD_NODE_POOL_SELECTOR_VALUE,
        }

        containers = [
            self._build_head_container(),
            self._build_pathways_proxy_container(),
            self._build_pathways_rm_container(),
        ]

        metadata_host_alias = dict(
            ip=_METADATA_GOOGLE_INTERNAL_IP,
            hostnames=["metadata", "metadata.google.internal"],
        )

        leader_pod_spec = {
            "terminationGracePeriodSeconds": 60,
            "hostAliases": [metadata_host_alias],
            "nodeSelector": node_selector,
            "tolerations": tolerations,
            "containers": containers,
            "volumes": volumes,
            "serviceAccountName": cfg.service_account,
            "hostNetwork": True,
            "dnsPolicy": "ClusterFirstWithHostNet",
        }

        if cfg.priority_class:
            leader_pod_spec["priorityClassName"] = cfg.priority_class

        return {
            "metadata": {
                "annotations": annotations,
                "labels": labels,
            },
            "spec": leader_pod_spec,
        }

    def __call__(self) -> Nested[Any]:
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        return dict(
            subGroupPolicy=dict(
                subGroupSize=system.vms_per_slice,
                subGroupPolicyType="LeaderExcluded",
            ),
            size=system.vms_per_slice + 1,
            leaderTemplate=self.build_leader_pod(),
            workerTemplate=self.build_worker_pod(),
        )

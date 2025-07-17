# Copyright © 2025 Apple Inc.

"""Utilities for building Jobset specs."""

import io
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

from absl import flags

from axlearn.cloud.common.bastion import (
    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR,
    BASTION_JOB_VERSION_ENV_VAR,
    deserialize_jobspec,
)
from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import (
    AcceleratorConfig,
    FlagConfigurable,
    accelerator_flags,
    namespaced,
    parse_kv_flags,
)
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
)
from axlearn.cloud.gcp.tpu import get_default_env, infer_tpu_workers
from axlearn.cloud.gcp.utils import validate_jobset_name
from axlearn.common.compiler_options import infer_tpu_type
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested

# Set 80% of the max value as the requested memory.
_MEMORY_REQUEST_PERCENTAGE = 0.8


# A label added to the jobset to indicate job version.
BASTION_JOB_VERSION_LABEL = "bastion-job-version"

# The metadata.google.internal IP.
# https://cloud.google.com/compute/docs/troubleshooting/troubleshoot-metadata-server#failed-request
_METADATA_GOOGLE_INTERNAL_IP = "169.254.169.254"

# Kubernetes pod annotation keys. Used by TPUReplicatedJob to support multi NIC.
# Refer to GKE TPU provisioner for more context:
# https://github.com/GoogleCloudPlatform/ai-on-gke/blob/5f256eed7075a5cb8e73cd72328aea46237b8ce6/tpu-provisioner/internal/cloud/common.go#L29-L31
_ANNOTATION_ADDITIONAL_NODE_NETWORKS = "tpu-provisioner.cloud.google.com/additional-node-networks"
_ANNOTATION_NODE_SERVICE_ACCOUNT = "tpu-provisioner.cloud.google.com/node-service-account"


# Use kw_only=True so that subclasses can have a mix of default and non-default attributes.
@dataclass(kw_only=True)
class VolumeMount:
    """Generic volume mount."""

    name: str
    mount_path: str
    read_only: bool = False


@dataclass(kw_only=True)
class GCSFuseMount(VolumeMount):
    """Configures the GCS FUSE mount.

    https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver#sidecar-container
    https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver#consume-ephemeral-volume-pod

    Attributes:
        gcs_path: GCS path, including gs:// prefix.
        mount_path: Path within local fs to mount to.
        cpu: Defaults to 250m. Increase if higher throughput needed.
        memory: Defaults to 256Mi. Set proportionally to number of files processed (not filesize).
        ephemeral_gb: Defaults to 5Gi. Used for staging temp files before uploading to GCS.
        shared_memory: Default to 1Gi. Used for e.g. Grain-related jobs which store prefetch
            elements in shared_memory. Setting it to 0 means unlimited shared_memory.
        http_client_timeout: Defaults to 0s. Specifies how long the Cloud Storage FUSE HTTP client
            can wait to get a response from the server before timing out. Setting it to 0s means no
            limit.
        read_only: Whether the mount should be read-only.
    """

    gcs_path: str
    name: str = "gcs-fuse-csi-ephemeral"
    mount_path: str = "/output"
    cpu: str = "250m"
    memory: str = "256Mi"
    ephemeral_gb: str = "5Gi"
    shared_memory: str = "1Gi"
    http_client_timeout: str = "0s"


@dataclass(kw_only=True)
class HostMount(VolumeMount):
    """Configures the hostPath mount.

    https://kubernetes.io/docs/concepts/storage/volumes/#hostpath

    Attributes:
        host_path: Host path to mount into the container.
        type: Type of the host path.
    """

    host_path: str
    type: str = "Directory"


@dataclass
class _LoadBalancer:
    """Configures the load balancer which exposes a K8s replicated job.
        The jobset-controller will take care of creating the load balancer
        based on the metadata.
        TODO(liang-he): move the load balancer creation to the Bastion.

    Attributes:
        service_name: The service name of the load balancer.
            Default name is <jobset_name>-<replicated_job_name>-service.
        target_port: The port number of the container which the load
            balancer targets.
        port: The port number of the load balancer.
    """

    def __init__(
        self, *, jobset_name: str, replicated_job_name: str, target_port: int = 9000, port: int = 80
    ):
        """
        Initializes a LoadBalancer instance.

        Args:
            jobset_name: Name of the jobset.
            replicated_job_name: Name of the replicated job.
            target_port: The container port the load balancer targets. Defaults to 9000.
            port: The load balancer's exposed port. Defaults to 80.
        """
        self.service_name = f"{jobset_name}-{replicated_job_name}-service"
        self.target_port = target_port
        self.port = port
        self.metadata = {
            "axlearn/replicatedjob-load-balancer-service-name": self.service_name,
            "axlearn/replicatedjob-load-balancer-target-port": str(self.target_port),
            "axlearn/replicatedjob-load-balancer-port": str(self.port),
        }


class BaseReplicatedJob(FlagConfigurable):
    """Common base class between single and composite replicated jobs."""

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures BaseReplicatedJob.

        Attributes:
            name: Name of the jobset. Also used for inferring docker image.
            output_dir: An optional GCS path to upload job outputs to.
                Each host's output will be placed in `"{output_dir}/output/$HOSTNAME/"`.
                This directory is used by the sidecar container to sync outputs to GCS using gsutil.
                Ensure that `output_dir` is a valid GCS path (e.g., `gs://your-bucket/path`).
        """

        name: Required[str] = REQUIRED
        output_dir: Optional[str] = None

    @classmethod
    def define_flags(cls, fv):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # NOTE: the parent typically sets these flags, so we leave them as None.
        flags.DEFINE_string("name", None, "Name of the job.", **common_kwargs)
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )

    def __init__(self, cfg: Config, *, bundler: Bundler):
        super().__init__(cfg)
        self._bundler = bundler

    def __call__(self) -> Sequence[Nested[Any]]:
        """Builds replicated job specs for k8s JobSet API.

        https://kubernetes.io/docs/concepts/workloads/controllers/job/

        Returns:
            A sequence of dicts, each containing top-level keys "name", "replicas", and "template".
            The "template" should correspond to a k8s Job config.
        """
        raise NotImplementedError(type(self))


@namespaced(mapping="inner")
class CompositeReplicatedJob(BaseReplicatedJob):
    """Builds a composite replicated job spec."""

    @config_class
    class Config(BaseReplicatedJob.Config):
        """Configures SingleReplicatedJob.

        Attributes:
            inner: A mapping from job_name to child replicated job.
        """

        inner: Required[dict[str, BaseReplicatedJob.Config]] = REQUIRED

    def __init__(self, cfg: Config, **kwargs):
        super().__init__(cfg, **kwargs)
        self._inner = {
            namespace: child.instantiate(**kwargs) for namespace, child in cfg.inner.items()
        }

    def __call__(self) -> Sequence[Nested[Any]]:
        composite = []
        for child in self._inner.values():
            composite.extend(child())
        return composite


class SingleReplicatedJob(BaseReplicatedJob):
    """Builds a single replicated job spec."""

    @config_class
    class Config(BaseReplicatedJob.Config):
        """Configures SingleReplicatedJob.

        Attributes:
            job_name: Name of the replicated job.
            command: Command to be executed.
            accelerator: Accelerator configuration.
            project: GCP Project.
            env_vars: Optional env vars to set.
            gcsfuse_mount: Optional configs for the GCS FUSE sidecar and volume mount.
                See `GCSFuseMount` for details.
            host_mounts: List of volumes from host to mount into the container.
                See `HostMount` for details.
            service_account: Optional service account to execute the job as.
            enable_pre_provisioner: Whether to enable pre-provisioner.
        """

        # We disambiguate from "name" which often refers to the jobset.
        job_name: Required[str] = REQUIRED
        command: Required[str] = REQUIRED
        project: Required[str] = REQUIRED
        accelerator: AcceleratorConfig = AcceleratorConfig()
        env_vars: dict[str, str] = {}
        gcsfuse_mount: Optional[GCSFuseMount] = None
        host_mounts: Optional[Sequence[HostMount]] = None
        service_account: Optional[str] = None
        # This config is made Optional for backwards compatibility. Default is False.
        enable_pre_provisioner: Optional[bool] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        accelerator_flags(**common_kwargs)
        flags.DEFINE_string("job_name", "job", "Replicated job name.", **common_kwargs)
        flags.DEFINE_string("command", None, "Command to execute.", **common_kwargs)
        flags.DEFINE_multi_string("env", [], "Env var in the format key:value.", **common_kwargs)
        flags.DEFINE_multi_string(
            "gcsfuse_mount_spec",
            None,
            "GCS FUSE mount spec in the format key=value.",
            **common_kwargs,
        )
        flags.DEFINE_multi_string(
            "host_mount_spec",
            None,
            "Host mount spec in the format key=value, separated by comma. You can specify multiple "
            "host mounts by using this flag repeatedly. Example: "
            "--host_mount_spec=name=tmp,host_path=/tmp,mount_path=/host-tmp "
            "--host_mount_spec=name=home,host_path=/home,mount_path=/host-home",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "service_account",
            None,
            "If specified, will run job as the service account.",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "enable_pre_provisioner", None, "Whether to enable pre-provisioner.", **common_kwargs
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: SingleReplicatedJob.Config = super().from_flags(fv, **kwargs)
        cfg.service_account = cfg.service_account or gcp_settings(
            "k8s_service_account", default="default", fv=fv
        )
        cfg.accelerator.set(instance_type=fv.instance_type, num_replicas=fv.num_replicas)
        # pylint: disable=missing-kwoa
        # pytype: disable=missing-parameter
        if fv.gcsfuse_mount_spec:
            cfg.gcsfuse_mount = GCSFuseMount(**parse_kv_flags(fv.gcsfuse_mount_spec, delimiter="="))
        if fv.host_mount_spec:
            cfg.host_mounts = [
                HostMount(**parse_kv_flags(item.split(","), delimiter="="))
                for item in fv.host_mount_spec
            ]
        # pytype: enable=missing-parameter
        return cfg


class TPUReplicatedJob(SingleReplicatedJob):
    """Builds a replicated jobspec for TPU, to be used with JobSet API."""

    @config_class
    class Config(SingleReplicatedJob.Config):
        """Configures TPUReplicatedJob.

        Attributes:
            reservation: If specified, the TPU reservation name. This is not necessarily specific to
                GKE and can be the same as e.g. the QRM reservation.
                https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/reservations/list
            reservation_project: GCP project to which the TPU reservation belongs. This is needed
                for shared reservations. If specified, the TPU provisioner will instead use the
                full reservation name for reservation affinity in the format:
                "projects/<reservation_project>/reservations/<reservation>"
                https://github.com/GoogleCloudPlatform/ai-on-gke/blob/889ec98f9b9a7aec05eb0f9890ada1f4c59b6159/tpu-provisioner/internal/cloud/gke.go#L328-L334
            enable_tpu_ici_resiliency: Whether to enable TPU ICI resiliency.
                If True, the job will persist through some types of network failure, but with
                degraded performance.
                If None, we leave it to GCP to determine whether it's appropriate for the requested
                TPU topology.
            location_hint: If set, the job will be scheduled to run on this TPU location.
                If None, we leave it to GCP to determine where the TPUs are located.
            enable_tpu_smart_repair: Whether to enable TPU smart repair.
                GKE 1.29.3-gke.1154000 or above is required.
            priority_class: Optional; The GKE PriorityClass for the job.
                https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption
                Note: 1. Values need to be pre-defined in each cluster.
                      2. Job level priority is enforced by pod level priority of the leader pod.
                         This is managed by jobset controller.
                      3. For TPU slice, this requires alpha.jobset.sigs.k8s.io/exclusive-topology
                      4. [2024-11-11] Does not work on multi-slice TPU training yet.
            additional_node_networks: Optional; comma-separated list of <network-name>:<subnet-name>
                to attach to the node pool. This is needed to support multiple NIC.
                Refer to GKE TPU provisioner for more context:
                https://github.com/GoogleCloudPlatform/ai-on-gke/blob/5f256eed7075a5cb8e73cd72328aea46237b8ce6/tpu-provisioner/internal/cloud/common.go#L29-L31
        """

        reservation: Optional[str] = None
        reservation_project: Optional[str] = None
        enable_tpu_ici_resiliency: Optional[bool] = None
        location_hint: Optional[str] = None
        enable_tpu_smart_repair: bool = False
        priority_class: Optional[str] = None
        additional_node_networks: Optional[str] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("reservation", None, "TPU reservation.", **common_kwargs)
        flags.DEFINE_string(
            "reservation_project", None, "TPU reservation project.", **common_kwargs
        )
        flags.DEFINE_boolean(
            "enable_tpu_ici_resiliency",
            None,
            "Whether to enable TPU ICI resiliency. If None, the decision is left to GCP, as "
            "not all TPU types support this flag.",
            **common_kwargs,
        )
        # Only supported in clusters with PriorityClass setup.
        # TODO(ethanli): infer it from the JobMetadata.priority.
        flags.DEFINE_string(
            "priority_class",
            None,
            "The GKE PriorityClass for the job.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: TPUReplicatedJob.Config = super().from_flags(fv, **kwargs)
        default_env = get_default_env(
            tpu_type=infer_tpu_type(fv.instance_type),
            num_tpu_slices=fv.num_replicas,
            job_name=cfg.name,
        )
        # NOTE: we allow fv.env flags to override the defaults.
        cfg.env_vars = {**default_env, **cfg.env_vars, **parse_kv_flags(fv.env)}
        cfg.reservation = cfg.reservation or gcp_settings("gke_reservation", required=False, fv=fv)
        cfg.reservation_project = cfg.reservation_project or gcp_settings(
            "gke_reservation_project", required=False, fv=fv
        )
        # Only read from the config file since users shouldn't need to configure this.
        cfg.location_hint = gcp_settings("location_hint", required=False, fv=fv)
        cfg.enable_tpu_smart_repair = bool(
            gcp_settings("enable_tpu_smart_repair", required=False, fv=fv)
        )
        cfg.additional_node_networks = gcp_settings(
            "additional_node_networks", required=False, fv=fv
        )
        return cfg

    def __init__(self, cfg: Config, *, bundler: Bundler):
        super().__init__(cfg, bundler=bundler)
        cfg: TPUReplicatedJob.Config = self.config
        if cfg.output_dir is None:
            raise ValueError("cfg.output_dir is required.")
        self._tpu_type = infer_tpu_type(cfg.accelerator.instance_type)
        if self._tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {self._tpu_type}")
        validate_jobset_name(
            cfg.name,
            num_workers=infer_tpu_workers(self._tpu_type),
            num_replicas=cfg.accelerator.num_replicas,
            job_name=cfg.job_name,
        )
        self._output_volume_mount = dict(name="shared-output", mountPath="/output")
        if cfg.additional_node_networks and not cfg.service_account:
            raise ValueError("service_account must be set if additional_node_networks is set.")
        self._load_balancer = _LoadBalancer(jobset_name=cfg.name, replicated_job_name=cfg.job_name)

    def _maybe_add_volume_mount(self, volume_mounts: list[dict], *, spec: Optional[VolumeMount]):
        if spec:
            volume_mounts.append(
                dict(name=spec.name, mountPath=spec.mount_path, readOnly=spec.read_only)
            )

    def _build_container(self) -> Nested[Any]:
        """Builds a config for a single container.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: TPUReplicatedJob.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        volume_mounts = [self._output_volume_mount]

        if cfg.gcsfuse_mount:
            self._maybe_add_volume_mount(volume_mounts, spec=cfg.gcsfuse_mount)
            self._maybe_add_volume_mount(
                volume_mounts, spec=VolumeMount(name="shared-memory", mount_path="/dev/shm")
            )

        if cfg.host_mounts:
            for mount in cfg.host_mounts:
                self._maybe_add_volume_mount(volume_mounts, spec=mount)

        env_vars = {**cfg.env_vars}
        if cfg.enable_tpu_ici_resiliency is not None:
            env_vars["ENABLE_ICI_RESILIENCY"] = str(cfg.enable_tpu_ici_resiliency).lower()

        resources = {"limits": {"google.com/tpu": system.chips_per_vm}}
        # # Set request memory by host machine type.
        # machine_memory_gi = GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS.get(
        #     system.gce_machine_type, None
        # )
        # if machine_memory_gi is not None:
        #     request_memory_gi = machine_memory_gi * _MEMORY_REQUEST_PERCENTAGE
        #     resources["limits"]["memory"] = f"{machine_memory_gi}Gi"
        #     resources["requests"] = {"memory": f"{math.floor(request_memory_gi)}Gi"}

        k8s_env_vars = [dict(name=k, value=str(v)) for k, v in env_vars.items()]
        k8s_env_vars.append(
            {"name": "NODE_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}}
        )
        k8s_env_vars.append(
            {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}
        )

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            # https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#tpu-chips-node-pool
            # https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-multislice#run_workload
            ports=[
                dict(containerPort=8471),  # Port using which TPU VMs communicate.
                dict(containerPort=8080),  # Port for MXLA coordinator.
                dict(containerPort=8431),  # Port to export TPU runtime metrics.
                dict(containerPort=self._load_balancer.target_port),  # Port for load balancer.
            ],
            securityContext=dict(privileged=True),
            # TODO(markblee): Improve SIGTERM behavior for command.
            command=["bash", "-c", cfg.command],
            resources=resources,
            # Env var values should always be strings.
            env=k8s_env_vars,
            volumeMounts=volume_mounts,
            imagePullPolicy="Always",
        )

    def _build_uploader_container(
        self, src: str = "/output", output_volume_mount: Optional[dict] = None
    ) -> Nested[Any]:
        """Builds a config for the uploader container which sync logs to the output dir.

        The sidecar container runs an loop to periodically sync outputs to GCS until the Pod is
        terminated.
        When the main container exits, Kubernetes will then send a termination signal (SIGTERM)
        to the uploader container, allowing it to exit gracefully.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: TPUReplicatedJob.Config = self.config
        output_volume_mount = output_volume_mount or self._output_volume_mount
        dst = f"{cfg.output_dir}/output/$HOSTNAME/"
        interval_s = 60
        sync_command = f"while true; do gsutil -m rsync -r {src} {dst}; sleep {interval_s}; done"
        resources = {
            # "requests": {"cpu": "100m", "memory": "128Mi"},
            # "limits": {"cpu": "500m", "memory": "256Mi"},
        }
        return dict(
            name="output-uploader",
            image="google/cloud-sdk:alpine",
            # https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/#pod-sidecar-containers
            # SideCar container is an init container with restartPolicy as "Always".
            restartPolicy="Always",
            command=["/bin/sh", "-c"],
            args=[sync_command],
            resources=resources,
            volumeMounts=[output_volume_mount],
        )

    def _build_shared_memory_volumes(self, shared_memory: str) -> Nested[Any]:
        return {
            "name": "shared-memory",
            "emptyDir": {"medium": "Memory", "sizeLimit": shared_memory},
        }

    def _build_pod(self) -> Nested[Any]:
        """Builds a config for a single Pod, which is a set of containers.

        https://kubernetes.io/docs/concepts/workloads/pods

        Returns:
            A nested dict corresponding to a k8s Pod template, including the pod metadata and spec.
        """
        cfg: TPUReplicatedJob.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        annotations, labels, selector, volumes, tolerations = {}, {}, {}, [], []

        volumes.append(dict(name="shared-output", emptyDir={}))
        if cfg.gcsfuse_mount:
            # Increases the shared memory volumes when enabled gcsfuse. This is useful when grain
            # prefetch is enabled.
            volumes.append(self._build_shared_memory_volumes(cfg.gcsfuse_mount.shared_memory))
            # Mount a GCS bucket as a volume.
            annotations.update(
                {
                    "gke-gcsfuse/volumes": "true",
                    "gke-gcsfuse/cpu-request": cfg.gcsfuse_mount.cpu,
                    "gke-gcsfuse/memory-request": cfg.gcsfuse_mount.memory,
                    "gke-gcsfuse/ephemeral-storage-request": cfg.gcsfuse_mount.ephemeral_gb,
                    # GCSFuse will set limits=request if we only set requests:
                    # https://github.com/GoogleCloudPlatform/gcs-fuse-csi-driver/blob/main/pkg/webhook/config.go#L110
                    "gke-gcsfuse/cpu-limit": "0",
                    "gke-gcsfuse/memory-limit": "0",
                    "gke-gcsfuse/ephemeral-storage-limit": "0",
                }
            )
            # Parse GCSFuseMount path into bucket, prefix.
            parsed = urlparse(cfg.gcsfuse_mount.gcs_path)
            # https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver#consume-ephemeral-volume-pod
            # Caveat: --implicit-dirs might have negative impacts on i/o performance. See
            # https://github.com/googlecloudplatform/gcsfuse/blob/master/docs/semantics.md .
            # See https://cloud.google.com/storage/docs/cloud-storage-fuse/config-file for more
            # details about mountOptions.
            # The mountOptions are following https://github.com/AI-Hypercomputer/maxtext/pull/1070.
            volumes.append(
                dict(
                    name=cfg.gcsfuse_mount.name,
                    csi=dict(
                        driver="gcsfuse.csi.storage.gke.io",
                        readOnly=cfg.gcsfuse_mount.read_only,
                        volumeAttributes=dict(
                            bucketName=parsed.netloc,
                            # pylint: disable=line-too-long
                            mountOptions=f"only-dir={parsed.path.lstrip('/')},implicit-dirs,metadata-cache:ttl-secs:-1,metadata-cache:stat-cache-max-size-mb:-1,metadata-cache:type-cache-max-size-mb:-1,kernel-list-cache-ttl-secs=-1,gcs-connection:http-client-timeout:{cfg.gcsfuse_mount.http_client_timeout}",
                            gcsfuseMetadataPrefetchOnMount="false",  # Improves first-time read.
                            disableMetrics="false",  # Enables GCSFuse metrics by default.
                        ),
                    ),
                )
            )
        if cfg.host_mounts:
            for mount in cfg.host_mounts:
                volumes.append(
                    dict(
                        name=mount.name,
                        hostPath=dict(path=mount.host_path, type=mount.type),
                    )
                )

        # If running from bastion, a scheduling tier will be specified in env.
        # Tier "0" corresponds to reserved; otherwise we use preemptible.
        tier = os.environ.get("BASTION_TIER", None)

        if tier == "0" and cfg.reservation is not None:
            logging.info("Found tier=%s in env. Using reservation=%s", tier, cfg.reservation)
            selector.update({"cloud.google.com/reservation-name": cfg.reservation})
            if cfg.reservation_project is not None:
                selector.update({"cloud.google.com/reservation-project": cfg.reservation_project})
            labels.update({"bastion-tier": "reserved"})
        elif tier != "disabled":
            logging.info("Found tier=%s in env. Using spot quota", tier)
            selector.update({"cloud.google.com/gke-spot": "true"})
            tolerations.append(
                {
                    "key": "cloud.google.com/gke-spot",
                    "operator": "Equal",
                    "value": "true",
                    "effect": "NoSchedule",
                }
            )
            labels.update({"bastion-tier": "spot"})

        if cfg.enable_tpu_ici_resiliency is not None:
            selector.update(
                {
                    "cloud.google.com/gke-tpu-ici-resiliency": str(
                        cfg.enable_tpu_ici_resiliency
                    ).lower()
                }
            )

        if cfg.location_hint is not None:
            selector.update({"cloud.google.com/gke-location-hint": str(cfg.location_hint).lower()})

        if cfg.enable_pre_provisioner:
            # Used by pre-provisioner.
            selector.update({PRE_PROVISIONER_LABEL: cfg.name})
        elif tier != "disabled":
            # Used by GCP auto-provisioner.
            selector.update(
                {
                    # NOTE: This is an arbitrary key, with a value that must be unique to the
                    # jobset. This forces the jobset to be associated with its own node pool;
                    # without this, the TPU provisioner may create a node pool and the scheduler may
                    # schedule a different jobset onto the node pool, which can cause conflicts if
                    # the original jobset attempts to restart (node pool conflict). This is more
                    # reliable at the moment but doesn't take advantage of node pool sharing. GCP is
                    # working on a fix.
                    "provisioner-nodepool-id": cfg.name,
                }
            )

        if os.environ.get(BASTION_JOB_VERSION_ENV_VAR):
            labels.update({BASTION_JOB_VERSION_LABEL: os.environ.get(BASTION_JOB_VERSION_ENV_VAR)})

        if os.environ.get(_BASTION_SERIALIZED_JOBSPEC_ENV_VAR):
            spec = deserialize_jobspec(
                io.StringIO(os.environ.get(_BASTION_SERIALIZED_JOBSPEC_ENV_VAR))
            )

            labels.update({"job-priority": str(spec.metadata.priority)})
            labels.update({"user-id": spec.metadata.user_id})

            # For job-priority to be populated to node labels when tpu-provisioner is used.
            selector.update({"job-priority": str(spec.metadata.priority)})

        annotations.update(
            {
                # Disable gcp auto-provisioner or not.
                # https://github.com/GoogleCloudPlatform/ai-on-gke/blob/b199de1d5326f257fa6fc21d99e45b5b4621bb20/tpu-provisioner/internal/controller/creation_controller.go#L40
                "tpu-provisioner.cloud.google.com/disable-autoprovisioning": (
                    "true" if cfg.enable_pre_provisioner else "false"
                ),
            }
        )

        if cfg.enable_tpu_smart_repair:
            labels.update({"cloud.google.com/gke-tpu-auto-restart": "true"})
            annotations.update(
                {
                    # The list of labels to be copied to node pools by tpu-provisioner.
                    # https://github.com/GoogleCloudPlatform/ai-on-gke/blob/main/tpu-provisioner/internal/cloud/common.go#L27-L28
                    # pylint: disable=line-too-long
                    "tpu-provisioner.cloud.google.com/copy-labels": "cloud.google.com/gke-tpu-auto-restart"
                }
            )

        # Hardcode metadata.google.internal ip address to avoid transient DNS resolution issue.
        metadata_host_alias = dict(
            ip=_METADATA_GOOGLE_INTERNAL_IP,
            hostnames=["metadata", "metadata.google.internal"],
        )

        spec = dict(
            # NOTE: Don't set hostNetwork or dnsPolicy for compat with Workload Identity.
            terminationGracePeriodSeconds=60,
            # Fail if any pod fails, and allow retries to happen at JobSet level.
            restartPolicy="Never",
            # https://kubernetes.io/docs/tasks/network/customize-hosts-file-for-pods/#adding-additional-entries-with-hostaliases
            hostAliases=[metadata_host_alias],
            nodeSelector={
                "cloud.google.com/gke-tpu-accelerator": system.gke_accelerator,
                "cloud.google.com/gke-tpu-topology": system.topology,
                **selector,
            },
            tolerations=tolerations,
            containers=[self._build_container()],
            initContainers=[self._build_uploader_container()],
            serviceAccountName=cfg.service_account,
            volumes=volumes,
        )

        if cfg.priority_class:
            spec["priorityClassName"] = cfg.priority_class

        # Handles additional network.
        if cfg.additional_node_networks:
            node_service_account = f"{cfg.service_account}@{cfg.project}.iam.gserviceaccount.com"
            annotations.update(
                {
                    _ANNOTATION_ADDITIONAL_NODE_NETWORKS: cfg.additional_node_networks,
                    _ANNOTATION_NODE_SERVICE_ACCOUNT: node_service_account,
                }
            )
            spec["hostNetwork"] = True
            spec["dnsPolicy"] = "ClusterFirstWithHostNet"

        return dict(
            metadata=dict(annotations=annotations, labels=labels),
            spec=spec,
        )

    def __call__(self) -> Sequence[Nested[Any]]:
        """See `BaseReplicatedJob` docstring for details."""
        cfg: TPUReplicatedJob.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        job_spec = dict(
            metadata=dict(annotations=self._load_balancer.metadata),
            spec=dict(
                parallelism=system.vms_per_slice,
                completions=system.vms_per_slice,
                backoffLimit=0,  # Fail the job if any node fails. Retries happen at JobSet level.
                template=self._build_pod(),
            ),
        )
        # NOTE: the suffix here impacts how long job names can be.
        return [
            dict(
                name=cfg.job_name,
                replicas=cfg.accelerator.num_replicas,
                template=job_spec,
            )
        ]


class GPUReplicatedJob(SingleReplicatedJob):
    """Builds a replicated job spec for a generic GPU job (A3, A3 Mega, A3 Ultra, A4),
    to be used with JobSet API.
    """

    Config = SingleReplicatedJob.Config

    def _build_init_containers(self) -> list[Nested[Any]]:
        return []

    def _build_main_container(self) -> Nested[Any]:
        """Builds the base container with common elements across all GPU jobs

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: GPUReplicatedJob.Config = self.config

        volume_mounts = [
            {"name": "shared-memory", "mountPath": "/dev/shm"},
            {"name": "nvidia-install-dir-host", "mountPath": "/usr/local/nvidia/lib64"},
        ]

        # These are common across all GPUReplicatedJobs, used for connecting between replicas
        env_vars: dict[str, Nested[str]] = {}
        env_vars["DISTRIBUTED_COORDINATOR"] = f"{cfg.name}-{cfg.job_name}-0-0.{cfg.name}:8080"
        env_vars["NUM_PROCESSES"] = f"{cfg.accelerator.num_replicas}"

        # List of XLA flags across all A3 and A4 instances
        global_gpu_xla_flags = [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_enable_triton_gemm=false",
            "--xla_gpu_enable_pipelined_all_gather=true",
            "--xla_gpu_enable_pipelined_reduce_scatter=true",
            "--xla_gpu_enable_pipelined_all_reduce=true",
            "--xla_gpu_enable_while_loop_double_buffering=true",
            "--xla_gpu_enable_all_gather_combine_by_dim=false",
            "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
            "--xla_disable_hlo_passes=rematerialization",
        ]
        env_vars["XLA_FLAGS"] = " ".join(global_gpu_xla_flags)
        # Leave trailing space for A3 / A4-specific XLA flags to be added later
        env_vars["XLA_FLAGS"] += " "

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            ports=[
                dict(containerPort=8080),  # Port for MXLA coordinator.
            ],
            securityContext=dict(privileged=True),
            # TODO(markblee): Improve SIGTERM behavior for command.
            resources=dict(limits={"nvidia.com/gpu": "8"}),
            env=env_vars,
            volumeMounts=volume_mounts,
        )

    def _build_volumes(self) -> Nested[Any]:
        """Builds a config for volumes."""
        volumes = [
            {
                "name": "shared-memory",
                "emptyDir": {"medium": "Memory"},
            },
            {
                "name": "nvidia-install-dir-host",
                "hostPath": {"path": "/home/kubernetes/bin/nvidia/lib64"},
            },
        ]

        return volumes

    def _build_pod(self) -> Nested[Any]:
        """Builds a config for a single Pod, which is a set of containers.

        https://kubernetes.io/docs/concepts/workloads/pods

        Returns:
            A nested dict corresponding to a k8s Pod template, including the pod metadata and spec.
        """
        cfg: GPUReplicatedJob.Config = self.config
        volumes = self._build_volumes()
        annotations = {
            "kubectl.kubernetes.io/default-container": cfg.name,
        }

        containers = [self._build_main_container()]
        init_containers = self._build_init_containers()

        return dict(
            metadata=dict(annotations=annotations),
            spec=dict(
                terminationGracePeriodSeconds=60,
                # Fail if any pod fails, and allow retries to happen at JobSet level.
                restartPolicy="Never",
                initContainers=init_containers,
                hostNetwork=True,
                dnsPolicy="ClusterFirstWithHostNet",
                containers=containers,
                serviceAccountName=cfg.service_account,
                volumes=volumes,
            ),
        )

    def _build_job(self) -> Nested[Any]:
        """Builds a config for a single Job, which is a set of Pods.

        https://kubernetes.io/docs/concepts/workloads/controllers/job/

        Returns:
            A nested dict corresponding to a k8s Job config, including the job metadata and spec.
        """
        cfg: GPUReplicatedJob.Config = self.config

        return dict(
            spec=dict(
                parallelism=cfg.accelerator.num_replicas,
                completions=cfg.accelerator.num_replicas,
                backoffLimit=0,  # Fail the job if any node fails. Retries happen at JobSet level.
                template=self._build_pod(),
            ),
        )

    def __call__(self) -> Sequence[Nested[Any]]:
        """See `BaseReplicatedJob` docstring for details."""
        cfg: GPUReplicatedJob.Config = self.config
        job_spec = dict(
            spec=dict(
                parallelism=cfg.accelerator.num_replicas,
                completions=cfg.accelerator.num_replicas,
                backoffLimit=0,  # Fail the job if any node fails. Retries happen at JobSet level.
                template=self._build_pod(),
            ),
        )
        # NOTE: the suffix here impacts how long job names can be.
        return [dict(name="job", replicas=1, template=job_spec)]


class A3HighReplicatedJob(GPUReplicatedJob):
    """Builds a replicated job spec for an a3-high GPU job, to be used with JobSet API."""

    Config = GPUReplicatedJob.Config

    def _build_volumes(self) -> Nested[Any]:
        """Builds a config for volumes."""

        return super()._build_volumes() + [
            {
                "name": "tcpx-socket",
                "emptyDir": {},
            },
            {
                "name": "tcpx-nccl-plugin-volume",
                "emptyDir": {},
            },
        ]

    def _build_main_container(self) -> Nested[Any]:
        """Builds the config for the container running the job.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: A3HighReplicatedJob.Config = self.config

        base_main_container: Nested[Any] = super()._build_main_container()
        volume_mounts = base_main_container["volumeMounts"] + [
            {"name": "tcpx-socket", "mountPath": "/run/tcpx"},
            {"name": "tcpx-nccl-plugin-volume", "mountPath": "/usr/local/tcpx"},
        ]

        env_vars = base_main_container["env"]

        # XLA flags for a3-high (H100 with TCPX)
        platform_xla_flags = [
            # Allows combining multiple all reduce into single all reduce.
            "--xla_gpu_all_reduce_contiguous",
            # Increase combine threshold to 1GB for improved performance.
            # A3 and TCPX performs bad with smaller message sizes.
            "--xla_gpu_all_reduce_combine_threshold_bytes=1073741824",
            "--xla_gpu_all_gather_combine_threshold_bytes=1073741824",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824",
        ]
        # Add platform-specific XLA flags to the common flags
        # (see global_gpu_xla_flags in GPUReplicatedJob)
        env_vars["XLA_FLAGS"] += " ".join(platform_xla_flags)

        env_vars.update(
            {
                "LD_LIBRARY_PATH": "/usr/local/tcpx/lib64:/usr/local/nvidia/lib64",
                # Set to 0 to encourage rail alignment.
                "NCCL_CROSS_NIC": "0",
                # TCPX only supports Ring algorithm.
                "NCCL_ALGO": "Ring",
                # TCPX only supports Simple protocol.
                "NCCL_PROTO": "Simple",
                "NCCL_DEBUG": "WARN",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV,TUNING,NET,VERSION",
                # Enable GPU Direct RDMA when GPU and NIC are same PCI switch.
                "NCCL_NET_GDR_LEVEL": "PIX",
                # TCPX requires disabling PXN.
                "NCCL_P2P_PXN_LEVEL": "0",
                # The NCCL_GPU_DIRECTTCPX variables can not be tweaked.
                "NCCL_GPUDIRECTTCPX_FORCE_ACK": "0",
                "NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP": "1000",
                "NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS": "1000000",
                "NCCL_GPUDIRECTTCPX_TX_BINDINGS": (
                    "eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
                ),
                "NCCL_GPUDIRECTTCPX_RX_BINDINGS": (
                    "eth1:22-35,124-139;eth2:22-35,124-139;eth3:74-87,178-191;eth4:74-87,178-191"
                ),
                "NCCL_GPUDIRECTTCPX_SOCKET_IFNAME": "eth1,eth2,eth3,eth4",
                "NCCL_GPUDIRECTTCPX_CTRL_DEV": "eth0",
                "NCCL_GPUDIRECTTCPX_UNIX_CLIENT_PREFIX": "/run/tcpx",
                # Improves performance but can be tweaked.
                "NCCL_DYNAMIC_CHUNK_SIZE": "524288",
                "NCCL_P2P_NET_CHUNKSIZE": "524288",
                "NCCL_P2P_PCI_CHUNKSIZE": "524288",
                "NCCL_P2P_NVL_CHUNKSIZE": "1048576",
                # The number of sockets per thread improves performance.
                "NCCL_NSOCKS_PERTHREAD": "4",
                "NCCL_SOCKET_NTHREADS": "1",
                # Use the system NIC for NCCL control plane comms.
                "NCCL_SOCKET_IFNAME": "eth0",
                # TCPX is not compatible with NVLS.
                "NCCL_NVLS_ENABLE": "0",
            }
        )

        # Override env vars with user provided env vars.
        env_vars.update(cfg.env_vars)
        # K8s expects each env variable to be a dict.
        k8s_env_vars = [{"name": name, "value": value} for name, value in env_vars.items()]
        k8s_env_vars.append(
            {
                "name": "PROCESS_ID",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": (
                            "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                        ),
                    }
                },
            },
        )

        user_cmd = cfg.command
        if user_cmd is None:
            raise ValueError("Command should not be None.")
        user_cmd += "; touch /run/tcpx/terminated"
        command = ["bash", "-c", user_cmd]

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            ports=[
                dict(containerPort=8080),  # Port for MXLA coordinator.
            ],
            securityContext=dict(privileged=True),
            # TODO(markblee): Improve SIGTERM behavior for command.
            command=command,
            resources=dict(limits={"nvidia.com/gpu": "8"}),
            env=k8s_env_vars,
            volumeMounts=volume_mounts,
        )

    def _build_a3_high_tcpx_init_container(self) -> Nested[Any]:
        """Builds the init container for TCPX use with a3-high"""

        volume_mounts = [
            {
                "name": "tcpx-nccl-plugin-volume",
                "mountPath": "/var/lib/tcpx",
            },
        ]
        command = ["bash", "-c", "/scripts/container_entry.sh install"]
        return dict(
            name="tcpx-nccl-plugin-installer",
            image=(
                "us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/nccl-plugin-gpudirecttcpx-dev:v3.1.7"
            ),
            command=command,
            env=[{"name": "LD_LIBRARY_PATH", "value": "/usr/local/nvidia/lib64"}],
            volumeMounts=volume_mounts,
        )

    def _build_a3_high_sidecar_container(self) -> Nested[Any]:
        """Builds a sidecar container which is required by A3
        for GPU to GPU RDMA like networking.

        Returns:
            A nested dict of the sidecar container.
        """
        volume_mounts = [
            {
                "name": "nvidia-install-dir-host",
                "mountPath": "/usr/local/nvidia/lib64",
            },
            {
                "name": "tcpx-socket",
                "mountPath": "/run/tcpx",
            },
        ]
        # See the reference for TCPX on a3-high linked here:
        # https://cloud.google.com/compute/docs/gpus/gpudirect#provide-access
        command = [
            "bash",
            "-c",
            'set -x; /tcpgpudmarxd/build/app/tcpgpudmarxd --gpu_nic_preset a3vm  \
                --gpu_shmem_type fd --uds_path /run/tcpx \
                --setup_param "--verbose 128 2 0" & \n\
            while [ ! -f /run/tcpx/terminated ]; do sleep 10; done;',
        ]

        return dict(
            name="tcpx-daemon",
            image="us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/tcpgpudmarxd-dev:latest",
            securityContext={"privileged": True},
            command=command,
            env=[{"name": "LD_LIBRARY_PATH", "value": "/usr/local/nvidia/lib64"}],
            volumeMounts=volume_mounts,
        )

    def _build_init_containers(self) -> list[Nested[Any]]:
        return [self._build_a3_high_tcpx_init_container(), self._build_a3_high_sidecar_container()]


class A3MegaReplicatedJob(GPUReplicatedJob):
    """Builds a replicated job spec for an a3-mega GPU job, to be used with JobSet API."""

    Config = GPUReplicatedJob.Config

    def _build_a3_mega_tcpx_init_container(self) -> Nested[Any]:
        """Builds a config for a single container."""
        volume_mounts = [
            {
                "name": "tcpx-nccl-plugin-volume",
                "mountPath": "/var/lib/tcpx",
            },
        ]
        # a3-mega uses TCPXO, slightly different from a3-high TCPX. See reference:
        # https://cloud.google.com/cluster-toolkit/docs/machine-learning/a3-mega-enable-gpudirect-tcpxo
        command = [
            "bash",
            "-c",
            'set -ex; chmod 755 /scripts/container_entry.sh; \n\
             /scripts/container_entry.sh install; \n\
             mkdir -p /usr/lib/tcpx/lib64; \n\
             cp -r /var/lib/tcpxo/lib64/. /usr/lib/tcpx/lib64; \n\
             echo "installation finishes";',
        ]
        return dict(
            name="tcpx-nccl-plugin-installer",
            image=(
                # pylint: disable=line-too-long
                "us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/nccl-plugin-gpudirecttcpx-dev:latest"
            ),
            command=command,
            env=[{"name": "LD_LIBRARY_PATH", "value": "/usr/local/nvidia/lib64"}],
            volumeMounts=volume_mounts,
        )

    def _build_a3_mega_tcpx_sidecar_container(self) -> Nested[Any]:
        """Builds a sidecar container which is required by A3
        for GPU to GPU RDMA like networking.

        Returns:
            A nested dict of the sidecar container.
        """
        volume_mounts = [
            {
                "name": "nvidia-install-dir-host",
                "mountPath": "/usr/local/nvidia/lib64",
            },
            {
                "name": "tcpx-socket",
                "mountPath": "/run/tcpx",
            },
        ]
        # a3-mega uses TCPXO, slightly different from a3-high TCPX. See reference:
        # https://cloud.google.com/cluster-toolkit/docs/machine-learning/a3-mega-enable-gpudirect-tcpxo
        command = [
            "bash",
            "-c",
            "set -ex; chmod 755 /fts/entrypoint_rxdm_container.sh; \n\
            /fts/entrypoint_rxdm_container.sh --num_hops=2 --num_nics=8 \
                --uid= --alsologtostderr &\n\
            while [ ! -f /run/tcpx/terminated ]; do sleep 10; done;",
        ]

        return dict(
            name="tcpx-daemon",
            image="us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpxo/tcpgpudmarxd-dev:latest",
            securityContext={"privileged": True},
            command=command,
            env=[{"name": "LD_LIBRARY_PATH", "value": "/usr/local/nvidia/lib64"}],
            volumeMounts=volume_mounts,
            restartPolicy="Always",
        )

    def _build_init_containers(self) -> list[Nested[Any]]:
        return [
            self._build_a3_mega_tcpx_init_container(),
            self._build_a3_mega_tcpx_sidecar_container(),
        ]

    def _build_main_container(self) -> Nested[Any]:
        """Builds the config for the container running the job.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: A3MegaReplicatedJob.Config = self.config

        base_main_container: Nested[Any] = super()._build_main_container()
        volume_mounts = base_main_container["volumeMounts"] + [
            {"name": "tcpx-socket", "mountPath": "/run/tcpx"},
            {"name": "tcpx-nccl-plugin-volume", "mountPath": "/usr/local/tcpx"},
            {"name": "aperture-devices", "mountPath": "/dev/aperture_devices"},
        ]

        env_vars = base_main_container["env"]

        # A list of XLA flags and their functions is linked here:
        # https://docs.jax.dev/en/latest/xla_flags.html#gpu-xla-flags
        # These flags have been tuned by GCP for a3-mega (H100 with TCPXO)
        platform_xla_flags = [
            "--xla_gpu_enable_highest_priority_async_stream=true",
            "--xla_gpu_all_reduce_combine_threshold_bytes=134217728",
            "--xla_gpu_all_gather_combine_threshold_bytes=1073741824",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=33554432",
        ]
        # Add platform-specific XLA flags to the common flags
        # (see global_gpu_xla_flags in GPUReplicatedJob)
        env_vars["XLA_FLAGS"] += " ".join(platform_xla_flags)

        env_vars.update(
            {
                "LD_LIBRARY_PATH": "/usr/local/tcpx/lib64:/usr/local/nvidia/lib64",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                # The NCCL_FASTRAK config cannot be changed
                # pylint: disable=line-too-long
                # This config is based on: https://github.com/AI-Hypercomputer/gpu-recipes/blob/dc6ef1afc1492f05e5741356f00cf645a9f1b795/src/helm-charts/a3mega/nemo-training/templates/nemo-launcher-job.yaml
                "NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY": "/dev/aperture_devices",
                "NCCL_FASTRAK_CTRL_DEV": "eth0",
                "NCCL_FASTRAK_IFNAME": "eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8",
                "NCCL_FASTRAK_USE_LLCM": "1",
                "NCCL_FASTRAK_NUM_FLOWS": "2",
                "NCCL_FASTRAK_USE_SNAP": "1",
                "NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS": "600000",
                "NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL": "0",
                "NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING": "0",
                "NCCL_MIN_NCHANNELS": "4",
                "NCCL_TUNER_PLUGIN": "libnccl-tuner.so",
                "NCCL_TUNER_CONFIG_PATH": "/root/axlearn/cloud/gcp/nccl/a3_mega/tuner_config.txtpb",
                "NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE": (
                    "/root/axlearn/cloud/gcp/nccl/a3_mega/guest_config.txtpb"
                ),
                # Set to 0 to encourage rail alignment.
                "NCCL_CROSS_NIC": "0",
                "NCCL_ALGO": "Ring,Tree",
                # TCPX only supports Simple protocol.
                "NCCL_PROTO": "Simple",
                "NCCL_DEBUG": "WARN",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV,TUNING,NET,VERSION",
                "NCCL_NET_GDR_LEVEL": "PIX",
                "NCCL_DYNAMIC_CHUNK_SIZE": "524288",
                "NCCL_P2P_NET_CHUNKSIZE": "524288",
                "NCCL_P2P_PCI_CHUNKSIZE": "524288",
                "NCCL_P2P_NVL_CHUNKSIZE": "1048576",
                # Use the system NIC for NCCL control plane comms.
                "NCCL_SOCKET_IFNAME": "eth0",
                # TCPX is not compatible with NVLS.
                "NCCL_NVLS_ENABLE": "0",
            }
        )

        # Override env vars with user provided env vars.
        env_vars.update(cfg.env_vars)
        # K8s expects each env variable to be a dict.
        k8s_env_vars = [{"name": name, "value": value} for name, value in env_vars.items()]
        k8s_env_vars.append(
            {
                "name": "PROCESS_ID",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": (
                            "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                        ),
                    }
                },
            },
        )

        user_cmd = cfg.command
        if user_cmd is None:
            raise ValueError("Command should not be None.")
        user_cmd += "; touch /run/tcpx/terminated"
        command = ["bash", "-c", user_cmd]

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            ports=[
                dict(containerPort=8080),  # Port for MXLA coordinator.
            ],
            securityContext=dict(privileged=True),
            # TODO(markblee): Improve SIGTERM behavior for command.
            command=command,
            resources=dict(limits={"nvidia.com/gpu": "8"}),
            env=k8s_env_vars,
            volumeMounts=volume_mounts,
        )

    def _build_volumes(self) -> Nested[Any]:
        """Builds a config for volumes."""

        return super()._build_volumes() + [
            {
                "name": "tcpx-socket",
                "emptyDir": {},
            },
            {
                "name": "tcpx-nccl-plugin-volume",
                "emptyDir": {},
            },
            {
                "name": "aperture-devices",
                "hostPath": {"path": "/dev/aperture_devices"},
            },
        ]


class A3UltraReplicatedJob(GPUReplicatedJob):
    """Builds a replicated job spec for an a3-ultra GPU job, to be used with JobSet API."""

    Config = GPUReplicatedJob.Config

    def _build_main_container(self) -> Nested[Any]:
        """Builds the config for the container running the job.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: A3UltraReplicatedJob.Config = self.config

        base_main_container: Nested[Any] = super()._build_main_container()
        volume_mounts = base_main_container["volumeMounts"] + [
            {"name": "gib", "mountPath": "/usr/local/gib"},
        ]

        env_vars = base_main_container["env"]

        # These flags have been tuned by GCP for a3-ultra (H200 with InfiniBand),
        # see the following reference:
        # https://github.com/AI-Hypercomputer/gpu-recipes/blob/dc6ef1afc1492f05e5741356f00cf645a9f1b795/src/helm-charts/a3ultra/maxtext-training/templates/maxtext-configmap.yaml#L26-L38
        platform_xla_flags = [
            "--xla_gpu_graph_level=0",
            "--xla_gpu_all_reduce_combine_threshold_bytes=2147483648",
            "--xla_gpu_all_gather_combine_threshold_bytes=2147483648",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=16777216",
        ]
        # Add platform-specific XLA flags to the common flags
        # (see global_gpu_xla_flags in GPUReplicatedJob)
        env_vars["XLA_FLAGS"] += " ".join(platform_xla_flags)

        env_vars.update(
            {
                "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "NCCL_DEBUG": "WARN",
                "NCCL_CROSS_NIC": "0",
                "NCCL_NET_GDR_LEVEL": "PIX",
                "NCCL_P2P_NET_CHUNKSIZE": "131072",
                "NCCL_P2P_PCI_CHUNKSIZE": "131072",
                "NCCL_P2P_NVL_CHUNKSIZE": "524288",
                "NCCL_NVLS_CHUNKSIZE": "524288",
                "NCCL_IB_GID_INDEX": "3",
                "NCCL_IB_ADAPTIVE_ROUTING": "1",
                "NCCL_IB_QPS_PER_CONNECTION": "4",
                "NCCL_IB_TC": "52",
                "NCCL_IB_FIFO_TC": "84",
                "NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE": (
                    "/root/axlearn/cloud/gcp/nccl/a3_ultra/guest_config.txtpb"
                ),
                "NCCL_TUNER_CONFIG_PATH": (
                    "/root/axlearn/cloud/gcp/nccl/a3_ultra/tuner_config.txtpb"
                ),
            }
        )

        # Override env vars with user provided env vars.
        env_vars.update(cfg.env_vars)
        # K8s expects each env variable to be a dict.
        k8s_env_vars = [{"name": name, "value": value} for name, value in env_vars.items()]
        k8s_env_vars.append(
            {
                "name": "PROCESS_ID",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": (
                            "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                        ),
                    }
                },
            },
        )

        command = ["bash", "-c", cfg.command]

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            ports=[
                dict(containerPort=8080),  # Port for MXLA coordinator.
            ],
            securityContext=dict(privileged=True),
            # TODO(markblee): Improve SIGTERM behavior for command.
            command=command,
            resources=dict(limits={"nvidia.com/gpu": "8"}),
            env=k8s_env_vars,
            volumeMounts=volume_mounts,
        )

    def _build_volumes(self) -> Nested[Any]:
        """Builds a config for volumes."""

        return super()._build_volumes() + [
            {
                "name": "gib",
                "hostPath": {"path": "/home/kubernetes/bin/gib"},
            },
        ]


class A4HighReplicatedJob(GPUReplicatedJob):
    """Builds a replicated job spec for an a4-high GPU job, to be used with JobSet API."""

    Config = GPUReplicatedJob.Config

    def _build_main_container(self) -> Nested[Any]:
        """Builds the config for the container running the job.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: A4HighReplicatedJob.Config = self.config

        base_main_container: Nested[Any] = super()._build_main_container()
        volume_mounts = base_main_container["volumeMounts"] + [
            {"name": "gib", "mountPath": "/usr/local/gib"},
        ]

        env_vars = base_main_container["env"]

        # These flags have been tuned by GCP for a4-high (B200 with InfiniBand)
        # See Maxtext reference for XLA flags:
        # https://github.com/AI-Hypercomputer/gpu-recipes/blob/main/training/a4/llama3-1-70b/maxtext-pretraining-gke/values.yaml
        platform_xla_flags = [
            "--xla_gpu_all_reduce_combine_threshold_bytes=2147483648",
            "--xla_gpu_all_gather_combine_threshold_bytes=2147483648",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=2147483648",
            "--xla_gpu_cudnn_gemm_fusion_level=3",
            "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL",
        ]
        # Add platform-specific XLA flags to the common flags
        # (see global_gpu_xla_flags in GPUReplicatedJob)
        env_vars["XLA_FLAGS"] += " ".join(platform_xla_flags)

        env_vars.update(
            {
                "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "NCCL_DEBUG": "WARN",
                "NCCL_CROSS_NIC": "0",
                "NCCL_NET_GDR_LEVEL": "PIX",
                "NCCL_P2P_NET_CHUNKSIZE": "131072",
                "NCCL_P2P_PCI_CHUNKSIZE": "131072",
                "NCCL_P2P_NVL_CHUNKSIZE": "524288",
                "NCCL_NVLS_CHUNKSIZE": "524288",
                "NCCL_IB_GID_INDEX": "3",
                "NCCL_IB_ADAPTIVE_ROUTING": "1",
                "NCCL_IB_QPS_PER_CONNECTION": "4",
                "NCCL_IB_TC": "52",
                "NCCL_IB_FIFO_TC": "84",
                "NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE": (
                    "/root/axlearn/cloud/gcp/nccl/a4_high/guest_config.txtpb"
                ),
                "NCCL_TUNER_CONFIG_PATH": "/root/axlearn/cloud/gcp/nccl/a4_high/tuner_config.txtpb",
            }
        )

        # Override env vars with user provided env vars.
        env_vars.update(cfg.env_vars)
        # K8s expects each env variable to be a dict.
        k8s_env_vars = [{"name": name, "value": value} for name, value in env_vars.items()]
        k8s_env_vars.append(
            {
                "name": "PROCESS_ID",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": (
                            "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                        ),
                    }
                },
            },
        )

        command = ["bash", "-c", cfg.command]

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            ports=[
                dict(containerPort=8080),  # Port for MXLA coordinator.
            ],
            securityContext=dict(privileged=True),
            # TODO(markblee): Improve SIGTERM behavior for command.
            command=command,
            resources=dict(limits={"nvidia.com/gpu": "8"}),
            env=k8s_env_vars,
            volumeMounts=volume_mounts,
        )

    def _build_volumes(self) -> Nested[Any]:
        """Builds a config for volumes."""

        return super()._build_volumes() + [
            {
                "name": "gib",
                "hostPath": {"path": "/home/kubernetes/bin/gib"},
            },
        ]

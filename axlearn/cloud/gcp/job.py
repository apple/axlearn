# Copyright © 2023 Apple Inc.

"""Utilities for executing commands on GCP.

Note that these utilities do not handle resource management.
"""
import atexit
import io
import logging
import math
import os
import pathlib
import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union
from urllib.parse import urlparse

import kubernetes as k8s
from absl import flags
from google.auth.credentials import Credentials

from axlearn.cloud.common.bastion import (
    _BASTION_SERIALIZED_JOBSPEC_ENV_VAR,
    BASTION_JOB_VERSION_ENV_VAR,
    deserialize_jobspec,
)
from axlearn.cloud.common.bundler import BaseDockerBundler
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.utils import parse_kv_flags, subprocess_run
from axlearn.cloud.gcp.config import default_project, default_zone, gcp_settings
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.scopes import DEFAULT_TPU_SCOPES
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
)
from axlearn.cloud.gcp.tpu import (
    get_queued_tpu_node,
    get_tpu_node,
    infer_tpu_type,
    qrm_resource,
    tpu_resource,
)
from axlearn.cloud.gcp.utils import (
    custom_jobset_kwargs,
    delete_k8s_jobset,
    get_credentials,
    running_from_vm,
)
from axlearn.common.config import REQUIRED, ConfigBase, Required, config_class
from axlearn.common.utils import Nested

# Set 80% of the max value as the requested memory.
_MEMORY_REQUEST_PERCENTAGE = 0.8

# A label added to the jobset to indicate job version.
BASTION_JOB_VERSION_LABEL = "bastion-job-version"

# The metadata.google.internal IP.
# https://cloud.google.com/compute/docs/troubleshooting/troubleshoot-metadata-server#failed-request
_METADATA_GOOGLE_INTERNAL_IP = "169.254.169.254"


class GCPJob(Job):
    """Base GCP Job definition."""

    @config_class
    class Config(Job.Config):
        """Configures GCPJob."""

        # GCP project.
        project: Required[str] = REQUIRED
        # GCP zone.
        zone: Required[str] = REQUIRED
        # If not none, the current job will be executed as the service account.
        service_account: Optional[str] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("project", default_project(), "The GCP project name.", **common_kwargs)
        flags.DEFINE_string("zone", default_zone(), "The GCP zone name.", **common_kwargs)
        flags.DEFINE_string(
            "service_account",
            None,
            "If specified, will run job as the service account. "
            "Otherwise will fallback to application-default credentials.",
            **common_kwargs,
        )

    def _get_job_credentials(
        self,
        impersonate_scopes: Optional[Sequence[str]] = None,
    ) -> Credentials:
        """Returns the credentials the job runs as.

        Note that credentials are temporary and should be created on demand.

        Args:
            impersonate_scopes: Scopes of the impersonation token,
                following https://developers.google.com/identity/protocols/oauth2/scopes

        Returns:
            The temporary credentials, possibly impersonating `cfg.service_account`.
        """
        return get_credentials(
            impersonate_account=self.config.service_account,
            impersonate_scopes=impersonate_scopes,
        )


@config_class
class AcceleratorConfig(ConfigBase):
    """Configures job resources, e.g. TPU or GPU.

    Attributes:
        instance_type: Instance type, e.g. tpu-v4-8.
        num_replicas: Number of replicas, e.g. TPU slices.
    """

    instance_type: Required[str] = REQUIRED
    num_replicas: int = 1


def accelerator_flags(flag_values: flags.FlagValues, **kwargs):
    """Defines resource flags, e.g. --instance_type and --num_replicas."""
    flags.DEFINE_string(
        "instance_type",
        # --instance_type is often defined at the launcher, so use any existing value by default.
        getattr(flag_values, "instance_type", None),
        "Instance type.",
        flag_values=flag_values,
        **kwargs,
    )
    flags.DEFINE_integer(
        "num_replicas", 1, "Number of replicas.", flag_values=flag_values, **kwargs
    )


class TPUQRMJob(GCPJob):
    """Executes arbitrary commands on TPU-VMs."""

    @config_class
    class Config(GCPJob.Config):
        """Configures TPUQRMJob.

        Attributes:
            accelerator: TPU configuration.
        """

        accelerator: AcceleratorConfig = AcceleratorConfig()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._local_home = pathlib.Path.home()
        self._use_iap = None  # Infer from public IP.

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        accelerator_flags(flag_values=fv, allow_override=True)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: TPUQRMJob.Config = super().from_flags(fv, **kwargs)
        cfg.accelerator.set(instance_type=fv.instance_type, num_replicas=fv.num_replicas)
        return cfg

    def _ensure_ssh_keys(self):
        """Ensures SSH keys exist, or raises ValueError. Only necessary on remote VM."""
        # Seem to need to nuke this every time to avoid MITM warnings.
        hosts_file = self._local_home / ".ssh/google_compute_known_hosts"
        if hosts_file.exists():
            hosts_file.unlink()

        ssh_key = self._local_home / ".ssh/google_compute_engine"
        proc = subprocess_run(f"ssh-add {ssh_key}", check=False, capture_output=True)
        if proc.returncode:
            logging.warning("SSH key %s does not exist yet.", ssh_key)

    def _infer_iap(self):
        """Infers whether instance has public IP. If not, we tunnel through IAP."""
        if self._use_iap is None:
            cfg: TPUQRMJob.Config = self.config
            if cfg.accelerator.num_replicas > 1:
                node = get_queued_tpu_node(
                    cfg.name,
                    qrm_resource(self._get_job_credentials(DEFAULT_TPU_SCOPES)),
                )
            else:
                node = get_tpu_node(
                    cfg.name,
                    tpu_resource(self._get_job_credentials(DEFAULT_TPU_SCOPES)),
                )
            if node is None:
                raise ValueError(f"Expected TPU {cfg.name} to exist")
            for endpoint in node.get("networkEndpoints", []):
                for access_config in endpoint.get("accessConfig", []):
                    if access_config.get("natIP", None):
                        logging.info("Detected a public IP, not using IAP.")
                        self._use_iap = False
                        return False
            logging.info("Didn't find a public IP, using IAP.")
            self._use_iap = True
        return self._use_iap

    def _execute_remote_cmd(
        self,
        cmd: str,
        *,
        worker: Union[int, str] = "all",
        detached_session: Optional[str] = None,
        batch_size: Union[int, str] = 100,
        extra_ssh_flags: str = "",
        **kwargs,
    ) -> Sequence[subprocess.CompletedProcess]:
        """Executes a command on existing TPU-VM(s).

        Args:
            cmd: Command to run.
            worker: Worker ID. Defaults to "all".
            wait: Whether to wait for process to complete. If True, waits for command to complete,
                and returns a completed process. Caller can inspect outputs or exit codes. If False,
                spawns and returns a process. Caller can listen to logs in realtime.
            detached_session: If not None, run commands behind `screen` in detached mode. This is
                useful for persisting commands even if SSH is terminated. If not None, should be a
                string containing the session name.
            batch_size: Number of concurrent command executions. If 'all', run all commands
                simultaneously.
            extra_ssh_flags: Extra gcloud ssh flags.
            **kwargs: Forwarded to subprocess.

        Returns:
            A list of completed subprocesses. Each corresponds to execution of the command on a
            single slice.

        Raises:
            ValueError: If the name of the detached screen session is too long.
        """
        cfg: TPUQRMJob.Config = self.config
        from_vm = running_from_vm()
        cmd = _prepare_cmd_for_gcloud_ssh(f"pushd /root && {cmd}")
        if from_vm:
            self._ensure_ssh_keys()
            extra_ssh_flags = f"--internal-ip {extra_ssh_flags}"
        elif self._infer_iap():
            # Infer IAP flag if not running from VM.
            extra_ssh_flags = f"--tunnel-through-iap {extra_ssh_flags}"
        cmd = f"sudo bash -c {cmd}"
        if detached_session:
            # Even though the official limit is 100 chars, screen seems to silently exit even before
            # that.
            if len(detached_session) > 80:
                raise ValueError(f"Screen name {detached_session} is too long.")
            cmd = f"sudo screen -dmS {detached_session} {cmd}"
        logging.debug("Executing remote command on worker [%s]: '%s'", worker, cmd)
        if cfg.accelerator.num_replicas > 1:
            slices = [f"{cfg.name}-{i}" for i in range(cfg.accelerator.num_replicas)]
        else:
            slices = [cfg.name]
        procs = []
        for s in slices:
            cmd_for_slice = (
                f"gcloud alpha compute -q tpus tpu-vm ssh {s} "
                f"--project={cfg.project} "
                f"--zone={cfg.zone} "
                f"--worker={worker} "
                f"--batch-size={batch_size} "
                f'{extra_ssh_flags} --command="{cmd}"'
            )
            proc = subprocess_run(cmd_for_slice, **_prepare_subprocess_kwargs(kwargs))
            procs.append(proc)
        return procs

    def _execute(self) -> Any:
        """Performs some computation on remote TPU-VMs."""
        cfg: TPUQRMJob.Config = self.config
        self._execute_remote_cmd(cfg.command)

    def execute(self) -> Any:
        """Wraps _execute with ssh-agent and retries. All args and kwargs are forwarded."""
        if running_from_vm():
            _start_ssh_agent()
        return super().execute()


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
        read_only: Whether the mount should be read-only.
    """

    gcs_path: str
    name: str = "gcs-fuse-csi-ephemeral"
    mount_path: str = "/output"
    cpu: str = "250m"
    memory: str = "256Mi"
    ephemeral_gb: str = "5Gi"


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


class GKEJob(GCPJob):
    """Base GKE Job interface."""

    @config_class
    class Config(GCPJob.Config):
        """Configures GKEJob.

        Attributes:
            env_vars: Optional env vars to set.
            namespace: The namespace to use within the k8s cluster.
                https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
            gcsfuse_mount: Optional configs for the GCS FUSE sidecar and volume mount.
                See `GCSFuseMount` for details.
            enable_pre_provisioner: Whether to enable pre-provisioner.
            queue: The Kueue LocalQueue to use. If not set, no queue is used.
            output_dir: Optional; The output directory of the GKE job outputs.
                Each host's output will be placed in `"{output_dir}/output/$HOSTNAME/"`.
                This directory is used by the sidecar container to sync outputs to GCS using gsutil.
                Ensure that `output_dir` is a valid GCS path (e.g., `gs://your-bucket/path`).
            priority_class: Optional; The GKE PriorityClass for the job.
                https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption
                Note: 1. Values need to be pre-defined in each cluster.
                      2. Job level priority is enforced by pod level priority of the leader pod.
                         This is managed by jobset controller.
                      3. For TPU slice, this requires alpha.jobset.sigs.k8s.io/exclusive-topology
                      4. [2024-11-11] Does not work on multi-slice TPU training yet.
            host_mounts: List of volumes from host to mount into the container.
                See `HostMount` for details.
        """

        env_vars: dict[str, str] = {}
        namespace: str = "default"
        gcsfuse_mount: Optional[GCSFuseMount] = None
        # This config is made Optional for backwards compatibility. Default is False.
        enable_pre_provisioner: Optional[bool] = None
        queue: Optional[str] = None
        output_dir: Optional[str] = None
        priority_class: Optional[str] = None
        host_mounts: Optional[list[HostMount]] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        flags.DEFINE_string(
            "namespace", "default", "K8s namespace.", flag_values=fv, allow_override=True
        )
        flags.DEFINE_multi_string(
            "gcsfuse_mount_spec",
            None,
            "GCS FUSE mount spec in the format key=value.",
            flag_values=fv,
        )
        flags.DEFINE_string(
            "queue",
            None,
            "The name of the Kueue LocalQueue to use. If not set, no queue is used.",
            flag_values=fv,
        )
        flags.DEFINE_multi_string(
            "host_mount_spec",
            None,
            "Host mount spec in the format key=value, separated by comma. You can specify multiple "
            "host mounts by using this flag repeatedly. Example: "
            "--host_mount_spec=name=tmp,host_path=/tmp,mount_path=/host-tmp "
            "--host_mount_spec=name=home,host_path=/home,mount_path=/host-home",
            flag_values=fv,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: GKEJob.Config = super().from_flags(fv, **kwargs)
        cfg.service_account = cfg.service_account or gcp_settings(
            "k8s_service_account", default="default", fv=fv
        )
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


class TPUGKEJob(GKEJob):
    """A TPU job represented as a k8s JobSet.

    See also `gke_runner` as an example.
    """

    @config_class
    class Config(GKEJob.Config):
        """Configures TPUGKEJob.

        Attributes:
            accelerator: TPU configuration.
            reservation: If specified, the TPU reservation name. This is not necessarily specific to
                GKE and can be the same as e.g. the QRM reservation.
                https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/reservations/list
            enable_tpu_ici_resiliency: Whether to enable TPU ICI resiliency.
                If True, the job will persist through some types of network failure, but with
                degraded performance.
                If None, we leave it to GCP to determine whether it's appropriate for the requested
                TPU topology.
            location_hint: If set, the job will be scheduled to run on this TPU location.
                If None, we leave it to GCP to determine where the TPUs are located.
            enable_tpu_smart_repair: Whether to enable TPU smart repair.
                GKE 1.29.3-gke.1154000 or above is required.
        """

        accelerator: AcceleratorConfig = AcceleratorConfig()
        reservation: Optional[str] = None
        enable_tpu_ici_resiliency: Optional[bool] = None
        location_hint: Optional[str] = None
        enable_tpu_smart_repair: bool = False

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        accelerator_flags(**common_kwargs)
        flags.DEFINE_string("reservation", None, "TPU reservation.", **common_kwargs)
        flags.DEFINE_boolean(
            "enable_tpu_ici_resiliency",
            None,
            "Whether to enable TPU ICI resiliency. If None, the decision is left to GCP, as "
            "not all TPU types support this flag.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: TPUGKEJob.Config = super().from_flags(fv, **kwargs)
        cfg.accelerator.set(instance_type=fv.instance_type, num_replicas=fv.num_replicas)
        cfg.reservation = cfg.reservation or gcp_settings("gke_reservation", required=False, fv=fv)
        # Only read from the config file since users shouldn't need to configure this.
        cfg.location_hint = gcp_settings("location_hint", required=False, fv=fv)
        cfg.enable_tpu_smart_repair = bool(
            gcp_settings("enable_tpu_smart_repair", required=False, fv=fv)
        )
        return cfg

    def __init__(self, cfg: Config):
        bundler_cfg = cfg.bundler
        bundler_cfg = getattr(bundler_cfg, "inner", bundler_cfg)
        if bundler_cfg is None or not issubclass(bundler_cfg.klass, BaseDockerBundler):
            raise NotImplementedError(f"Only docker bundler supported, got: {bundler_cfg}")
        self._tpu_type = infer_tpu_type(cfg.accelerator.instance_type)
        if self._tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {self._tpu_type}")
        super().__init__(cfg)
        self._output_volume_mount = dict(name="shared-output", mountPath="/output")

    def _maybe_add_volume_mount(self, volume_mounts: list[dict], *, spec: Optional[VolumeMount]):
        if spec:
            volume_mounts.append(
                dict(
                    name=spec.name,
                    mountPath=spec.mount_path,
                    readOnly=spec.read_only,
                ),
            )

    def _build_container(self) -> Nested[Any]:
        """Builds a config for a single container.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: TPUGKEJob.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        volume_mounts = [self._output_volume_mount]

        self._maybe_add_volume_mount(volume_mounts, spec=cfg.gcsfuse_mount)
        if cfg.host_mounts:
            for mount in cfg.host_mounts:
                self._maybe_add_volume_mount(volume_mounts, spec=mount)

        env_vars = {**cfg.env_vars}
        if cfg.enable_tpu_ici_resiliency is not None:
            env_vars["ENABLE_ICI_RESILIENCY"] = str(cfg.enable_tpu_ici_resiliency).lower()

        resources = {"limits": {"google.com/tpu": system.chips_per_vm}}
        # Set request memory by host machine type.
        machine_memory_gi = GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS.get(
            system.gce_machine_type, None
        )
        if machine_memory_gi is not None:
            request_memory_gi = machine_memory_gi * _MEMORY_REQUEST_PERCENTAGE
            resources["limits"]["memory"] = f"{machine_memory_gi}Gi"
            resources["requests"] = {"memory": f"{math.floor(request_memory_gi)}Gi"}

        k8s_env_vars = [dict(name=k, value=str(v)) for k, v in env_vars.items()]
        k8s_env_vars.append(
            {
                "name": "NODE_IP",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "status.hostIP",
                    }
                },
            },
        )
        k8s_env_vars.append(
            {
                "name": "NODE_NAME",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "spec.nodeName",
                    }
                },
            },
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

    def _build_uploader_container(self) -> Nested[Any]:
        """Builds a config for the uploader container which sync logs to the output dir.

        The sidecar container runs an loop to periodically sync outputs to GCS until the Pod is
        terminated.
        When the main container exits, Kubernetes will then send a termination signal (SIGTERM)
        to the uploader container, allowing it to exit gracefully.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: TPUGKEJob.Config = self.config

        dst = f"{cfg.output_dir}/output/$HOSTNAME/"
        interval_s = 60

        sync_command = f"while true; do gsutil -m rsync -r /output {dst}; sleep {interval_s}; done"

        volume_mounts = [self._output_volume_mount]

        resources = {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "256Mi"},
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
            volumeMounts=volume_mounts,
        )

    def _build_pod(self) -> Nested[Any]:
        """Builds a config for a single Pod, which is a set of containers.

        https://kubernetes.io/docs/concepts/workloads/pods

        Returns:
            A nested dict corresponding to a k8s Pod template, including the pod metadata and spec.
        """
        cfg: TPUGKEJob.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        annotations, labels, selector, volumes, tolerations = {}, {}, {}, [], []

        volumes.append(dict(name="shared-output", emptyDir={}))
        if cfg.gcsfuse_mount:
            # Mount a GCS bucket as a volume.
            annotations.update(
                {
                    "gke-gcsfuse/volumes": "true",
                    "gke-gcsfuse/cpu-limit": cfg.gcsfuse_mount.cpu,
                    "gke-gcsfuse/memory-limit": cfg.gcsfuse_mount.memory,
                    "gke-gcsfuse/ephemeral-storage-limit": cfg.gcsfuse_mount.ephemeral_gb,
                }
            )
            # Parse GCSFuseMount path into bucket, prefix.
            parsed = urlparse(cfg.gcsfuse_mount.gcs_path)
            # https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver#consume-ephemeral-volume-pod
            # Caveat: --implicit-dirs might have negative impacts on i/o performance. See
            # https://github.com/googlecloudplatform/gcsfuse/blob/master/docs/semantics.md .
            volumes.append(
                dict(
                    name=cfg.gcsfuse_mount.name,
                    csi=dict(
                        driver="gcsfuse.csi.storage.gke.io",
                        readOnly=cfg.gcsfuse_mount.read_only,
                        volumeAttributes=dict(
                            bucketName=parsed.netloc,
                            mountOptions=f"only-dir={parsed.path.lstrip('/')},implicit-dirs",
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
            labels.update({"bastion-tier": "reserved"})
        else:
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
            selector.update(
                {
                    PRE_PROVISIONER_LABEL: cfg.name,
                }
            )
        else:
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

        return dict(
            metadata=dict(annotations=annotations, labels=labels),
            spec=spec,
        )

    def _build_job(self) -> Nested[Any]:
        """Builds a config for a single Job, which is a set of Pods.

        https://kubernetes.io/docs/concepts/workloads/controllers/job/

        Returns:
            A nested dict corresponding to a k8s Job config, including the job metadata and spec.
        """
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        return dict(
            spec=dict(
                parallelism=system.vms_per_slice,
                completions=system.vms_per_slice,
                backoffLimit=0,  # Fail the job if any node fails. Retries happen at JobSet level.
                template=self._build_pod(),
            ),
        )

    def _build_jobset(self) -> Nested[Any]:
        """Builds a config for a JobSet, which is a set of Jobs.

        https://github.com/kubernetes-sigs/jobset/blob/d49514bee57da8ac9aec2fcea06c3a13c21afeae/docs/concepts/README.md

        Returns:
            A nested dict corresponding to a k8s JobSet config.
        """
        cfg: TPUGKEJob.Config = self.config

        annotations = {
            # The exclusive topology annotation will ensure that all Pods will have affinity
            # rules added that will ensure that they are fully scheduled on the same
            # pod-slice node-pools.
            "alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool",
        }
        if cfg.queue:
            annotations["kueue.x-k8s.io/queue-name"] = cfg.queue

        return dict(
            metadata=dict(
                name=cfg.name,
                annotations=annotations,
            ),
            spec=dict(
                failurePolicy=dict(maxRestarts=cfg.max_tries - 1),
                replicatedJobs=[
                    # NOTE: the suffix here impacts how long job names can be.
                    dict(
                        name="job",
                        replicas=cfg.accelerator.num_replicas,
                        template=self._build_job(),
                    ),
                ],
            ),
        )

    def _delete(self):
        cfg: TPUGKEJob.Config = self.config
        # Issues a delete request for the JobSet and proactively delete its descendants. This is not
        # fully blocking; after the call returns there can be a delay before everything is deleted.
        delete_k8s_jobset(cfg.name, namespace=cfg.namespace)

    def _execute(self) -> Any:
        """Submits a JobSet to the cluster."""
        cfg: TPUGKEJob.Config = self.config
        api_kwargs = custom_jobset_kwargs()
        custom_object = dict(
            apiVersion=f"{api_kwargs['group']}/{api_kwargs['version']}",
            kind="JobSet",
            **self._build_jobset(),
        )
        logging.info("Submitting JobSet body=%s api_kwargs=%s", custom_object, api_kwargs)
        return k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            namespace=cfg.namespace,
            body=custom_object,
            **api_kwargs,
        )


class GPUGKEJob(GKEJob):
    """A GPU job represented as a k8s JobSet.

    See also `gke_runner` as an example.
    """

    @config_class
    class Config(GKEJob.Config):
        """Configures GPUGKEJob.

        Attributes:
            accelerator: GPU configuration.
        """

        accelerator: AcceleratorConfig = AcceleratorConfig()

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        accelerator_flags(**common_kwargs)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: GPUGKEJob.Config = super().from_flags(fv, **kwargs)
        cfg.accelerator.set(instance_type=fv.instance_type, num_replicas=fv.num_replicas)
        return cfg

    def __init__(self, cfg: Config):
        bundler_cfg = cfg.bundler
        bundler_cfg = getattr(bundler_cfg, "inner", bundler_cfg)
        if bundler_cfg is None or not issubclass(bundler_cfg.klass, BaseDockerBundler):
            raise NotImplementedError(f"Only docker bundler supported, got: {bundler_cfg}")
        super().__init__(cfg)
        if cfg.gcsfuse_mount:
            raise NotImplementedError("GCSFuse is not supported on GKE with GPU.")
        if cfg.enable_pre_provisioner:
            raise NotImplementedError("Pre-provisioner is not supported on GKE with GPU.")
        instance_type = cfg.accelerator.instance_type
        if not instance_type.startswith("gpu-a3-highgpu"):
            raise NotImplementedError(
                f"The instance type {instance_type} is not supported on GKE with GPU. "
                "Only gpu-a3-highgpu-8g is supported."
            )

    def _build_a3_sidecar_container(self) -> Nested[Any]:
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
            image="us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/tcpgpudmarxd-dev:v2.0.11",
            securityContext={"privileged": True},
            command=command,
            env=[{"name": "LD_LIBRARY_PATH", "value": "/usr/local/nvidia/lib64"}],
            volumeMounts=volume_mounts,
        )

    def _build_main_container(self) -> Nested[Any]:
        """Builds the config for the container running the job.

        Returns:
            A nested dict corresponding to a k8s Container config.
        """
        cfg: GPUGKEJob.Config = self.config

        volume_mounts = [
            {"name": "shared-memory", "mountPath": "/dev/shm"},
            {"name": "tcpx-socket", "mountPath": "/run/tcpx"},
            {"name": "nvidia-install-dir-host", "mountPath": "/usr/local/nvidia/lib64"},
            {"name": "tcpx-nccl-plugin-volume", "mountPath": "/usr/local/tcpx"},
        ]

        env_vars: dict[str, str] = {}
        env_vars["DISTRIBUTED_COORDINATOR"] = f"{cfg.name}-job-0-0.{cfg.name}:8080"
        env_vars["NUM_PROCESSES"] = f"{cfg.accelerator.num_replicas}"

        default_xla_flags = [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            # Allows combining multiple all reduce into single all reduce.
            "--xla_gpu_all_reduce_contiguous",
            # Increase combine threshold to 1GB for improved performance.
            # A3 and TCPX performs bad with smaller message sizes.
            "--xla_gpu_all_reduce_combine_threshold_bytes=1073741824",
            "--xla_gpu_all_gather_combine_threshold_bytes=1073741824",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824",
        ]
        env_vars["XLA_FLAGS"] = " ".join(default_xla_flags)

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
                    "eth1:8-21,112-125;eth2:8-21,112-125;" "eth3:60-73,164-177;eth4:60-73,164-177"
                ),
                "NCCL_GPUDIRECTTCPX_RX_BINDINGS": (
                    "eth1:22-35,124-139;eth2:22-35,124-139;" "eth3:74-87,178-191;eth4:74-87,178-191"
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

    def _build_a3_init_container(self) -> Nested[Any]:
        """Builds a config for a single container."""
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
                "us-docker.pkg.dev/gce-ai-infra/gpudirect-tcpx/"
                "nccl-plugin-gpudirecttcpx-dev:v3.1.7"
            ),
            command=command,
            env=[{"name": "LD_LIBRARY_PATH", "value": "/usr/local/nvidia/lib64"}],
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
            {
                "name": "tcpx-socket",
                "emptyDir": {},
            },
            {
                "name": "tcpx-nccl-plugin-volume",
                "emptyDir": {},
            },
        ]

        return volumes

    def _build_pod(self) -> Nested[Any]:
        """Builds a config for a single Pod, which is a set of containers.

        https://kubernetes.io/docs/concepts/workloads/pods

        Returns:
            A nested dict corresponding to a k8s Pod template, including the pod metadata and spec.
        """
        cfg: GPUGKEJob.Config = self.config
        volumes = self._build_volumes()
        annotations = {
            "kubectl.kubernetes.io/default-container": cfg.name,
        }

        containers = [self._build_main_container(), self._build_a3_sidecar_container()]
        init_containers = [self._build_a3_init_container()]

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
        cfg: GPUGKEJob.Config = self.config

        return dict(
            spec=dict(
                parallelism=cfg.accelerator.num_replicas,
                completions=cfg.accelerator.num_replicas,
                backoffLimit=0,  # Fail the job if any node fails. Retries happen at JobSet level.
                template=self._build_pod(),
            ),
        )

    def _build_jobset(self) -> Nested[Any]:
        """Builds a config for a JobSet, which is a set of Jobs.

        https://github.com/kubernetes-sigs/jobset/blob/d49514bee57da8ac9aec2fcea06c3a13c21afeae/docs/concepts/README.md

        Returns:
            A nested dict corresponding to a k8s JobSet config.
        """
        cfg: GPUGKEJob.Config = self.config
        annotations = {}
        if cfg.queue:
            annotations["kueue.x-k8s.io/queue-name"] = cfg.queue

        return dict(
            metadata=dict(
                name=cfg.name,
                annotations=annotations,
            ),
            spec=dict(
                failurePolicy=dict(maxRestarts=cfg.max_tries - 1),
                replicatedJobs=[
                    # NOTE: the suffix here impacts how long job names can be.
                    dict(
                        name="job",
                        replicas=1,
                        template=self._build_job(),
                    ),
                ],
            ),
        )

    def _delete(self):
        cfg: GPUGKEJob.Config = self.config
        # Issues a delete request for the JobSet and proactively delete its descendants. This is not
        # fully blocking; after the call returns there can be a delay before everything is deleted.
        delete_k8s_jobset(cfg.name, namespace=cfg.namespace)

    def _execute(self) -> Any:
        """Submits a JobSet to the cluster."""
        cfg: GPUGKEJob.Config = self.config
        api_kwargs = custom_jobset_kwargs()
        custom_object = dict(
            apiVersion=f"{api_kwargs['group']}/{api_kwargs['version']}",
            kind="JobSet",
            **self._build_jobset(),
        )
        logging.info("Submitting JobSet body=%s api_kwargs=%s", custom_object, api_kwargs)
        return k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            namespace=cfg.namespace,
            body=custom_object,
            **api_kwargs,
        )


class CPUJob(GCPJob):
    """Executes arbitrary commands on CPU VMs."""

    def _execute_remote_cmd(
        self, cmd: str, *, detached_session: Optional[str] = None, **kwargs
    ) -> subprocess.CompletedProcess:
        """Executes a command on an existing VM.

        Args:
            cmd: Command to run.
            detached_session: If not None, run commands behind `screen` in detached mode. This is
                useful for persisting commands even if SSH is terminated. If not None, should be a
                string containing the session name.
            **kwargs: Forwarded to subprocess.

        Returns:
            A subprocess, either live or completed.
        """
        cfg: CPUJob.Config = self.config
        logging.debug("Executing remote command: '%s'", cmd)
        cmd = _prepare_cmd_for_gcloud_ssh(f"pushd /root && {cmd}")
        # Use login shell. Note `-i` is not interactive.
        cmd = f"sudo -i bash -c {cmd}"
        if detached_session:
            cmd = f"sudo screen -dmS {detached_session} {cmd}"
        # Run via screen to persist command after SSH.
        cmd = (
            f"gcloud compute -q ssh {cfg.name} "
            f"--project={cfg.project} "
            f"--zone={cfg.zone} "
            f'--command="{cmd}"'
        )
        proc = subprocess_run(cmd, **_prepare_subprocess_kwargs(kwargs))
        logging.debug("Finished launching: '%s'.", cmd)
        return proc

    def _execute(self) -> Any:
        """Performs some computation on remote VMs."""
        cfg: CPUJob.Config = self.config
        self._execute_remote_cmd(cfg.command)


def _prepare_subprocess_kwargs(kwargs: dict) -> dict:
    """Enable check=True and capture all outputs by default."""
    kwargs.setdefault("text", True)
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", kwargs.keys().isdisjoint(["stdout", "stderr"]))
    return kwargs


def _kill_ssh_agent():
    """Terminates ssh-agent, e.g. as started by `_start_ssh_agent`."""
    subprocess_run("ssh-agent -k", check=False, capture_output=True)
    os.environ.pop("SSH_AUTH_SOCK", None)
    os.environ.pop("SSH_AGENT_PID", None)


def _start_ssh_agent():
    """Starts ssh-agent for SSH key handling.

    The ssh-agent is automatically terminated when the program exits.
    """
    # pylint: disable=line-too-long
    if not os.getenv("SSH_AGENT_PID"):
        logging.info("ssh-agent is not running, starting it now...")
        process = subprocess_run("ssh-agent -s", stdout=subprocess.PIPE, check=True, text=True)
        # Example format:
        # Linux:
        # SSH_AUTH_SOCK=/tmp/ssh-g4aYlFVLLugX/agent.52090; export SSH_AUTH_SOCK;\nSSH_AGENT_PID=52091; export SSH_AGENT_PID;\necho Agent pid 52091;\n
        # Mac:
        # SSH_AUTH_SOCK=/var/folders/j0/blx8mk5j1hlc0k110xsbrxw00000gn/T//ssh-ZAf5XlQX7tWM/agent.7841; export SSH_AUTH_SOCK;\nSSH_AGENT_PID=7842; export SSH_AGENT_PID;\necho Agent pid 7842;\n
        match = re.search(
            r"SSH_AUTH_SOCK=([^;]+);.*SSH_AGENT_PID=([^;]+);",
            process.stdout,
            re.MULTILINE | re.DOTALL,
        )
        auth_sock, agent_pid = match.groups()  # pytype: disable=attribute-error
        os.environ["SSH_AUTH_SOCK"] = auth_sock
        os.environ["SSH_AGENT_PID"] = agent_pid
        atexit.register(_kill_ssh_agent)
    logging.info("ssh-agent is running.")


def _prepare_cmd_for_gcloud_ssh(cmd: str) -> str:
    """Handles bash escapes to ensure `cmd` is compatible with gcloud `--command`."""
    cmd = shlex.quote(cmd)
    cmd = cmd.replace('"', '\\"')  # Escape double quotes for --command.
    cmd = cmd.replace("$", r"\$")  # Escape $ for --command.
    return cmd


def docker_command(
    cmd: str,
    *,
    image: str,
    detached_session: Optional[str] = None,
    env: Optional[Sequence[str]] = None,
    volumes: Optional[dict[str, str]] = None,
    extra_docker_flags: Optional[Sequence[str]] = None,
) -> str:
    """Wraps a command with docker run.

    Args:
        cmd: Command to run.
        image: Docker image name.
        detached_session: If not None, runs in detached mode with the given name.
        env: Optional env vars to expose to container.
        volumes: Optional mapping of source/target volumes to mount.
        extra_docker_flags: Optional extra flags for docker run.

    Returns:
        The docker command.
    """
    cmd = _prepare_cmd_for_gcloud_ssh(f"pushd /root && {cmd}")
    cmd = f"/bin/bash -c {cmd}"
    env = " ".join([f"-e {e}" for e in (env or [])])
    volumes = " ".join([f"-v {src}:{dst}" for src, dst in (volumes or {}).items()])
    extra_docker_flags = " ".join(extra_docker_flags or [])
    detached = f"-d --name={detached_session}" if detached_session else ""
    cmd = (
        f"docker run --rm --privileged -u root --network=host {detached} {env} {volumes} "
        f"{extra_docker_flags} {image} {cmd}"
    )
    logging.debug("Docker run command: %s", cmd)
    return cmd

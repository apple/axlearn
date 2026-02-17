# Copyright © 2023 Apple Inc.

"""Utilities for executing commands on GCP.

See also ``On configuration`` in `axlearn/cloud/gcp/job.py`.
"""

import enum
import json
import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from typing import Any, Optional, cast

import kubernetes as k8s
from absl import flags

from axlearn.cloud.common.bastion import BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR
from axlearn.cloud.common.bundler import BaseDockerBundler
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.utils import generate_job_name, subprocess_run
from axlearn.cloud.gcp.config import default_env_id, default_project, default_zone
from axlearn.cloud.gcp.jobset_utils import BaseReplicatedJob
from axlearn.cloud.gcp.k8s_health_check_policy import LWSHealthCheckPolicy
from axlearn.cloud.gcp.k8s_http_route import LWSHTTPRoute
from axlearn.cloud.gcp.k8s_service import LWSService
from axlearn.cloud.gcp.lws_utils import BaseLeaderWorkerTemplate
from axlearn.cloud.gcp.system_characteristics import (
    _SystemCharacteristics,
    get_subblock_characteristics,
    get_system_characteristics,
)
from axlearn.cloud.gcp.utils import (
    custom_jobset_kwargs,
    custom_leaderworkerset_kwargs,
    delete_k8s_jobset,
    delete_k8s_leaderworkerset,
)
from axlearn.common.compiler_options import infer_tpu_version
from axlearn.common.config import REQUIRED, ConfigOr, Required, config_class, maybe_instantiate
from axlearn.common.utils import Nested


class _ServiceProtocol(enum.Enum):
    """https://kubernetes.io/docs/reference/networking/service-protocols/"""

    TCP = "TCP"
    UDP = "UDP"
    SCTP = "SCTP"


class _ServiceType(enum.Enum):
    """https://cloud.google.com/kubernetes-engine/docs/concepts/service#types-of-services sss"""

    CLUSTER_IP = "ClusterIP"
    NODE_PORT = "NodePort"
    LOAD_BALANCER = "LoadBalancer"
    EXTERNAL_NAME = "ExternalName"


def get_topology_assignment() -> Optional[list[list[str]]]:
    """Retrieves TPU topology assignments from the environment variable.

    When TPU slice auto-provisioning is enabled, Bastion passes topology assignments
    through an environment variable. These assignments specify which TPU slices should be
    used for the job, enabling precise control over TPU resource allocation.

    Example topology assignment:
        [["sub-block-id", "sub-block-id"]]

    This is the assignment for a job asking for tpu-7x-256, that needs 128 chips, using
    2 sub-blocks (64 chips per sub-block). This job will run on a TPU slice formed by
    2 sub-blocks. Each inner array represents the TPU slice info for a job's replica.

    Returns:
        A list of lists of strings representing topology assignments, where each inner list
        contains slice identifiers for a particular job replica. Returns None if the
        environment variable is not set or if parsing fails.
    """
    topology_assignments_env = os.environ.get(BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR)
    if not topology_assignments_env:
        logging.info("No %s environment variable set.", BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR)
        return None

    try:
        return json.loads(topology_assignments_env)
    except json.JSONDecodeError as e:
        logging.warning(
            "Failed to parse topology assignments from env var %s, value: %s, error: %s",
            BASTION_JOB_TOPOLOGY_ASSIGNMENT_ENV_VAR,
            topology_assignments_env,
            e,
        )
        return None


class GCPJob(Job):
    """Base GCP Job definition."""

    @config_class
    class Config(Job.Config):
        """Configures GCPJob."""

        # Name of the job.
        name: Required[str] = REQUIRED
        # GCP project.
        project: Required[str] = REQUIRED
        # GCP zone.
        zone: Required[str] = REQUIRED
        # GCP env_id.
        env_id: Optional[str] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the job.", **common_kwargs)
        flags.DEFINE_string("project", None, "The GCP project name.", **common_kwargs)
        flags.DEFINE_string("zone", None, "The GCP zone name.", **common_kwargs)
        flags.DEFINE_string(
            "env_id",
            None,
            "The env_id, used along with project to identify `gcp_settings`.",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default("project", default_project())
        fv.set_default("zone", default_zone())
        fv.set_default("env_id", default_env_id())


# TODO(markblee): Rename to GKEJobSet.
class GKEJob(GCPJob):
    """Base GKE JobSet interface."""

    @config_class
    class Config(GCPJob.Config):
        """Configures GKEJob.

        Attributes:
            builder: A builder that returns one or more replicated job specs.
            namespace: The namespace to use within the k8s cluster.
                https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
            queue: The Kueue LocalQueue to use. If not set, no queue is used.
            annotations: JobSet annotations (or config instantiating to annotations).
            labels: JobSet labels (or config instantiating to labels).
        """

        builder: Required[BaseReplicatedJob.Config] = REQUIRED
        namespace: str = "default"
        # TODO(markblee): queue can be expressed with `annotations`.
        queue: Optional[str] = None
        annotations: Optional[ConfigOr[dict]] = None
        labels: Optional[ConfigOr[dict]] = None
        enable_tpu_slice_auto_provisioning: Optional[bool] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the job.", **common_kwargs)
        flags.DEFINE_string(
            "queue",
            None,
            "The name of the Kueue LocalQueue to use. If not set, no queue is used.",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "enable_tpu_slice_auto_provisioning",
            None,
            "Auto provision TPU slices based on the topology assignment.",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv):
        super().set_defaults(fv)
        fv.set_default("max_tries", fv.max_tries or 10)
        fv.set_default("retry_interval", fv.retry_interval or 60)

    def __init__(self, cfg: Config, *, bundler: BaseDockerBundler):
        super().__init__(cfg)
        cfg: GKEJob.Config = self.config
        self._bundler = bundler
        # This instantiatees a builder for constructing replicated job specs, which will be managed
        # together under the jobset represented by this class.
        # Note the distinction from bundlers, which are responsible for bundling any code assets
        # required to run the job.

        # Pass enable_tpu_slice_auto_provisioning from GKEJob to the builder
        builder_cfg = cfg.builder
        if (
            hasattr(builder_cfg, "enable_tpu_slice_auto_provisioning")
            and cfg.enable_tpu_slice_auto_provisioning is not None
        ):
            builder_cfg.enable_tpu_slice_auto_provisioning = cfg.enable_tpu_slice_auto_provisioning

        self._builder: BaseReplicatedJob = builder_cfg.instantiate(bundler=bundler)

    def _delete(self):
        cfg: GKEJob.Config = self.config
        # Issues a delete request for the JobSet and proactively delete its descendants. This is not
        # fully blocking; after the call returns there can be a delay before everything is deleted.
        delete_k8s_jobset(cfg.name, namespace=cfg.namespace)

    def _lookup_system_by_node_selectors(
        self, node_selector: dict[str, str]
    ) -> Optional[tuple[str, _SystemCharacteristics]]:
        """Looks up system characteristics from node selectors.

        Args:
            node_selector: Kubernetes node selector dict

        Returns:
            Tuple of tpu type and _SystemCharacteristics object if TPU selectors found
            None otherwise.
        """
        gke_accelerator = node_selector.get("cloud.google.com/gke-tpu-accelerator")
        topology = node_selector.get("cloud.google.com/gke-tpu-topology")

        if not (gke_accelerator and topology):
            return None

        return get_system_characteristics(gke_accelerator, topology)

    def _get_tpu_replicated_job_topology_selection(
        self,
        replicated_jobs: Sequence[Nested[Any]],
        topology_assignments: list[list[str]],
    ) -> dict[str, list[list[str]]]:
        """Builds topology selection mapping from replicated jobs to subblock assignments.

        This method analyzes TPU replicated jobs to determine their subblock requirements
        and distributes topology assignments accordingly. It's used for TPU slice
        auto-provisioning with super slicing support.

        Args:
            replicated_jobs: List of replicated job specs from the JobSet.
            topology_assignments: List of subblock ID lists, where each inner list
                represents subblocks for one replica. Format: [["sb-1", "sb-2"], ["sb-3"]].

        Returns:
            A dict mapping replicated job name to its topology assignment. The topology
            assignment is a list of lists, where each inner list contains subblock IDs
            for a specific replica of that job.

            Example return value:
            {
                "tpu-worker": [["sb-1", "sb-2"], ["sb-3", "sb-4"]]
            }

        Raises:
            ValueError: If TPU version doesn't support subblock super slicing.
            ValueError: If requested resources don't match topology assignments.
            ValueError: If insufficient topology assignments are provided.
            ValueError: If system lookup fails for job's node selectors.
        """
        result = {}
        used_indices = set()  # Track which assignments have been used

        for job in replicated_jobs:
            # Extract node selector
            node_selector: dict[str, str] = cast(
                dict[str, str],
                job.get("template", {})
                .get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("nodeSelector", {}),
            )

            # Look up system characteristics
            gke_accelerator = node_selector.get("cloud.google.com/gke-tpu-accelerator")
            topology = node_selector.get("cloud.google.com/gke-tpu-topology")

            if not (gke_accelerator and topology):
                # Not a TPU job, skip
                continue

            maybe_system_chars = self._lookup_system_by_node_selectors(node_selector)

            if maybe_system_chars is None:
                job_name = str(job.get("name"))
                raise ValueError(
                    f"Could not find system characteristics for job '{job_name}' with "
                    f"accelerator='{gke_accelerator}' and topology='{topology}'. "
                    f"This combination is not defined in "
                    "USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS."
                )

            tpu_type, system_chars = maybe_system_chars
            job_name = str(job.get("name"))
            num_replicas = int(job.get("replicas", 1))

            # Get TPU version and check for subblock support
            tpu_version = infer_tpu_version(tpu_type)
            subblock_chars = get_subblock_characteristics(tpu_version)

            if subblock_chars is None:
                raise ValueError(
                    f"TPU version '{tpu_version}' (from device type '{system_chars.device_type}') "
                    f"does not support subblock super slicing. Only TPU versions with configured "
                    f"subblock mappings are supported for slice auto-provisioning."
                )

            # Calculate Subblocks
            job_vms = system_chars.vms_per_slice
            subblock_vms = subblock_chars.vms_per_slice

            if job_vms % subblock_vms != 0:
                raise ValueError(
                    f"Job '{job_name}' requires {job_vms} VMs, which is not "
                    f"evenly divisible by subblock size {subblock_vms} VMs. "
                    f"Job topology: {topology}, Subblock topology: {subblock_chars.topology}"
                )

            # Calculate subblocks needed per replica
            subblocks_per_replica = job_vms // subblock_vms

            # Find matching assignments for each replica
            job_assignments = []
            for replica_idx in range(num_replicas):
                # Find first unused assignment with correct number of subblocks
                found_assignment = None
                for idx, assignment in enumerate(topology_assignments):
                    if idx not in used_indices and len(assignment) == subblocks_per_replica:
                        found_assignment = assignment
                        used_indices.add(idx)
                        break

                if found_assignment is None:
                    raise ValueError(
                        f"Could not find unused topology assignment with {subblocks_per_replica} "
                        f"subblock(s) for job '{job_name}' replica {replica_idx}. "
                        f"Total assignments: {len(topology_assignments)}, "
                        f"already used: {len(used_indices)}."
                    )

                job_assignments.append(found_assignment)

            result[job_name] = job_assignments
            logging.info(
                "Job '%s' (%s, topology=%s): %d replica(s) × %d subblock(s)",
                job_name,
                system_chars.device_type,
                topology,
                num_replicas,
                subblocks_per_replica,
            )

        # Check if there are unused topology assignments
        if len(used_indices) < len(topology_assignments):
            logging.warning(
                "Not all topology assignments were consumed. Used %d out of %d assignment(s).",
                len(used_indices),
                len(topology_assignments),
            )

        return result

    def _build_jobset(self) -> Nested[Any]:
        """Builds a config for a JobSet, which is a set of Jobs.

        https://github.com/kubernetes-sigs/jobset/blob/d49514bee57da8ac9aec2fcea06c3a13c21afeae/docs/concepts/README.md

        Returns:
            A nested dict corresponding to a k8s JobSet config.
        """
        cfg: GKEJob.Config = self.config
        annotations = maybe_instantiate(cfg.annotations or {})
        labels = maybe_instantiate(cfg.labels or {})
        if cfg.queue:
            annotations["kueue.x-k8s.io/queue-name"] = cfg.queue

        replicated_jobs = self._builder()

        # Bastion passes the job metadata to the runner through env vars
        # If the job has topology assigned, its also in the env var
        # Try to parse the env var and get the topology assignments.
        topology_assignment = get_topology_assignment()
        if cfg.enable_tpu_slice_auto_provisioning and topology_assignment:
            slice_selection_dict = self._get_tpu_replicated_job_topology_selection(
                replicated_jobs, topology_assignment
            )
            slice_selection = json.dumps(slice_selection_dict)
            logging.info("Adding slice selection: %s to job set", slice_selection)
            labels.update({"tpu-provisioner.cloud.google.com/slice-autoprovisioning": "sync"})
            annotations.update(
                {"tpu-provisioner.cloud.google.com/slice-selection": slice_selection}
            )

            # Finally, make sure to remove the exclusive topology annotation, that is not required
            # When using slice auto provisioning
            annotations.pop("alpha.jobset.sigs.k8s.io/exclusive-topology", None)

        return dict(
            metadata=dict(name=cfg.name, annotations=annotations, labels=labels),
            spec=dict(
                failurePolicy=dict(maxRestarts=cfg.max_tries - 1),
                replicatedJobs=replicated_jobs,
            ),
        )

    def _execute(self) -> Any:
        """Submits a JobSet to the cluster."""
        cfg: GKEJob.Config = self.config
        api_kwargs = custom_jobset_kwargs()
        custom_object = dict(
            apiVersion=f"{api_kwargs["group"]}/{api_kwargs["version"]}",
            kind="JobSet",
            **self._build_jobset(),
        )
        logging.info("Submitting JobSet body=%s api_kwargs=%s", custom_object, api_kwargs)
        return k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            namespace=cfg.namespace,
            body=custom_object,
            **api_kwargs,
        )


def exclusive_topology_annotations() -> dict:
    """Used for TPU GKEJob.

    The exclusive topology annotation will ensure that all Pods will have affinity rules added that
    will ensure that they are fully scheduled on the same pod-slice node-pools.
    """
    return {"alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool"}


class CPUJob(GCPJob):
    """Executes arbitrary commands on CPU VMs."""

    @config_class
    class Config(GCPJob.Config):
        """Configures CPUJob.

        Attributes:
            command: Command to execute.
        """

        command: Required[str] = REQUIRED

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


class GKELeaderWorkerSet(GCPJob):
    """Base GKE LeaderWorkerSet interface"""

    @config_class
    class Config(GCPJob.Config):
        """Configures GKELeaderWorkerSet.
        Attributes:
            builder: A builder that returns one or more statefulset specs.
            namespace: The namespace to use within the k8s cluster.
            annotations: LeaderWorkerSet annotations.
            labels: LeaderWorkerSet labels.
            num_replicas: number of LWS replicas.
        """

        builder: Required[BaseLeaderWorkerTemplate.Config] = REQUIRED
        namespace: str = "default"
        annotations: Optional[ConfigOr[dict]] = None
        labels: Optional[ConfigOr[dict]] = None
        num_replicas: int = 1
        enable_service: bool = False
        ports: list[str] = None
        targetports: list[str] = None
        service_type: str = None
        protocol_list: list[str] = None
        port_names: list[str] = None
        service: Optional[LWSService.Config] = None
        gke_gateway_route: bool = False
        http_route: Optional[LWSHTTPRoute.Config] = None
        health_check_policy: Optional[LWSHealthCheckPolicy.Config] = None
        enable_tpu_slice_auto_provisioning: Optional[bool] = None

    @classmethod
    def set_defaults(cls, fv):
        super().set_defaults(fv)
        fv.set_default("max_tries", fv.max_tries or 10)
        fv.set_default("retry_interval", fv.retry_interval or 60)

        fv.set_default("gke_gateway_route", fv.gke_gateway_route or False)
        # When gke_gateway_route is set, enable_service is implicitly True
        if fv.gke_gateway_route:
            fv.set_default("enable_service", True)
        else:
            fv.set_default("enable_service", fv.enable_service or False)
        fv.set_default("targetports", fv.targetports or ["9090"])
        fv.set_default("ports", fv.ports or ["9090"])
        fv.set_default("protocol_list", fv.protocol_list or [_ServiceProtocol.TCP.value])
        fv.set_default("port_names", fv.port_names or ["tcp-port"])
        fv.set_default("service_type", fv.service_type or _ServiceType.CLUSTER_IP.value)

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the LeaderWorkerSet.", **common_kwargs)
        flags.DEFINE_boolean(
            "enable_service",
            False,
            "Whether to enable creation of service for LWS",
            **common_kwargs,
        )
        ##### https://cloud.google.com/kubernetes-engine/docs/how-to/exposing-apps ####
        ## Available types: ClusterIP(default), NodePort, LoadBalancer, ExternalName, Headless ##
        flags.DEFINE_enum(
            "service_type",
            None,
            [v.value for v in _ServiceType],
            help="Service type for LWS",
            flag_values=fv,
        )
        flags.DEFINE_list(
            "ports",
            [],
            "External ports where application is exposed through service",
            **common_kwargs,
        )

        flags.DEFINE_list(
            "targetports",
            [],
            " Application port which the service redirects to",
            **common_kwargs,
        )
        flags.DEFINE_list(
            "port_names",
            [],
            " Port Names which map the port and targetport in service",
            **common_kwargs,
        )
        #### https://kubernetes.io/docs/reference/networking/service-protocols/ #####
        #### Available types: TCP, UDP, SCTP #####
        flags.DEFINE_list(
            "protocol_list",
            [],
            "Protocol list needed for different port and targetport combinations",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "gke_gateway_route",
            False,
            "Enable gke_gateway_route with notary-proxy sidecars for direct gateway routing",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "enable_tpu_slice_auto_provisioning",
            None,
            "Auto provision TPU slices based on the topology assignment.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: GKELeaderWorkerSet.Config = super().from_flags(fv, **kwargs)
        cfg.num_replicas = fv.num_replicas
        cfg.enable_service = fv.enable_service
        cfg.ports = fv.ports
        cfg.targetports = fv.targetports
        cfg.protocol_list = fv.protocol_list
        cfg.port_names = fv.port_names
        cfg.service_type = fv.service_type
        cfg.gke_gateway_route = fv.gke_gateway_route
        return cfg

    def __init__(self, cfg: Config, *, bundler: BaseDockerBundler):
        super().__init__(cfg)
        cfg: GKELeaderWorkerSet.Config = self.config
        self._bundler = bundler

        # Pass enable_tpu_slice_auto_provisioning from GKEJob to the builder
        builder_cfg = cfg.builder
        if (
            hasattr(builder_cfg, "enable_tpu_slice_auto_provisioning")
            and cfg.enable_tpu_slice_auto_provisioning is not None
        ):
            builder_cfg.enable_tpu_slice_auto_provisioning = cfg.enable_tpu_slice_auto_provisioning

        # This instantiatees a builder for constructing replicated job specs, which will be managed
        # together under the leaderworkerset represented by this class.
        # Note the distinction from bundlers, which are responsible for bundling any code assets
        # required to run the job.
        self._builder: BaseLeaderWorkerTemplate = cfg.builder.instantiate(bundler=bundler)

        # Wire gke_gateway_route flag to service and http_route configs
        if cfg.service is not None:
            cfg.service.set(name=cfg.name, gke_gateway_route=cfg.gke_gateway_route)
        if cfg.http_route is not None:
            cfg.http_route.set(name=cfg.name, namespace=cfg.namespace)
        if cfg.health_check_policy is not None:
            cfg.health_check_policy.set(name=cfg.name, namespace=cfg.namespace)

    def _delete(self):
        cfg: GKELeaderWorkerSet.Config = self.config
        # Issues a delete request for the LeaderWorkerSet and proactively delete its descendants.
        # This is not fully blocking; after the call returns there can be a delay before
        # everything is deleted.
        delete_k8s_leaderworkerset(cfg.name, namespace=cfg.namespace)

    def _build_leaderworkerset(self) -> Nested[Any]:
        """
        Builds a config for a LeaderWorkerSet, which is a set for multi-host inference

        Returns:
            A nested dict corresponding to a k8s LWS config
        """
        cfg: GKELeaderWorkerSet.Config = self.config
        annotations = maybe_instantiate(cfg.annotations or {})
        labels = maybe_instantiate(cfg.labels or {})

        # If the topology is set and slice auto provisioning is configured
        # set the necessary annotations
        topology_assignment = get_topology_assignment()
        if cfg.enable_tpu_slice_auto_provisioning and topology_assignment:
            # Add TPU slice selection
            logging.info("Adding slice selection: %s to leader worker set", topology_assignment)

            # Note, we use async here rather than the jobset sync. Async will immediatly create
            # the pods before the slice has been created. Once sync is supported for leader worker
            # set we should consider switching.
            labels["tpu-provisioner.cloud.google.com/slice-autoprovisioning"] = "async"

            # For Leader worker sets, we only support topology assignments to workers.
            # The format of the topology assignments (list of subblock groups) is what
            # is expected by the TPU provisioner.
            annotations.update(
                {
                    "tpu-provisioner.cloud.google.com/slice-selection": json.dumps(
                        {
                            "workers": topology_assignment,
                        }
                    )
                }
            )

            # Remove exclusive topology annotation, the tpu provisioner will ensure replica
            # affinity by injecting slice based node selectors, so we don't need to use
            # the exclusive topology annotations
            exclusive_topology_annotation = exclusive_topology_annotations_leaderworkerset()
            for key in exclusive_topology_annotation:
                annotations.pop(key, None)

        return dict(
            metadata=dict(name=cfg.name, annotations=annotations, labels=labels),
            spec=dict(
                replicas=cfg.num_replicas,
                leaderWorkerTemplate=self._builder(),
            ),
        )

    def _execute(self):
        cfg: GKELeaderWorkerSet.Config = self.config

        api_kwargs = custom_leaderworkerset_kwargs()
        custom_object = dict(
            apiVersion=f"{api_kwargs["group"]}/{api_kwargs["version"]}",
            kind="LeaderWorkerSet",
            **self._build_leaderworkerset(),
        )
        logging.info("submitting LeaderWorkerSet: %s", custom_object)

        lws_resp = k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            namespace=cfg.namespace,
            body=custom_object,
            **api_kwargs,
        )

        #### Creating a  Service #######
        if cfg.enable_service:
            service_resp = cfg.service.instantiate().execute()
            logging.info("Service created %s", str(service_resp))
        else:
            cfg.service = None

        #### Creating HTTPRoute for gke_gateway_route #######
        if cfg.gke_gateway_route and cfg.http_route:
            http_route_resp = cfg.http_route.instantiate().execute()
            logging.info("HTTPRoute created %s", str(http_route_resp))

        #### Creating HealthCheckPolicy for gke_gateway_route #######
        if cfg.gke_gateway_route and cfg.health_check_policy:
            health_check_resp = cfg.health_check_policy.instantiate().execute()
            logging.info("HealthCheckPolicy created %s", str(health_check_resp))

        return lws_resp


def exclusive_topology_annotations_leaderworkerset() -> dict:
    """Used for TPU GKELeaderWorkerSet.

    The exclusive topology annotation will ensure that all Pods will have affinity
    rules added that will ensure that they are fully scheduled on the same pod-slice
    node-pools.
    """
    return {
        "leaderworkerset.sigs.k8s.io/subgroup-exclusive-topology": "cloud.google.com/gke-nodepool"
    }

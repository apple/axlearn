# Copyright © 2024 Apple Inc.

"""Utilities to provision TPU node pools."""

import hashlib
import io
import os
import time
from typing import NamedTuple, Optional

from absl import flags, logging

from axlearn.cloud.common.bastion import _BASTION_SERIALIZED_JOBSPEC_ENV_VAR, deserialize_jobspec
from axlearn.cloud.common.utils import AcceleratorConfig, FlagConfigurable
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import GKEJob
from axlearn.cloud.gcp.job_flink import FlinkTPUGKEJob
from axlearn.cloud.gcp.jobset_utils import TPUReplicatedJob
from axlearn.cloud.gcp.node_pool import (
    construct_node_pool_name,
    create_node_pools,
    delete_node_pools,
)
from axlearn.cloud.gcp.pathways_utils import PathwaysLeaderWorkerTemplate
from axlearn.cloud.gcp.system_characteristics import (
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
    _SystemCharacteristics,
)
from axlearn.cloud.gcp.tpu import infer_tpu_type
from axlearn.common.config import REQUIRED, Required, config_class

FLAGS = flags.FLAGS

# TODO(muyang_yu): avoid listing job types one by one.
_INFERENCE_JOBS = (FlinkTPUGKEJob,)
_SUPPORTED_BUILDER_TYPES = (TPUReplicatedJob.Config, PathwaysLeaderWorkerTemplate.Config)


class NodePoolProvisioner(FlagConfigurable):
    """Node pool provisioner."""

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures node pool provisioning.

        Attributes:
            project: GCP Project name.
            zone: GCP zone name.
            cluster: K8s cluster name.
            name: The name of the provisioner.
            service_account_email: Service account email for node pools.
            retry_interval: Number of seconds to retry node pool creation or deletion.
            wait_timeout: Number of seconds to wait for node pool creation or deletion.
            labels: Optional extra labels to apply to all node pools.
            disk_type: Optional disk type for node pool nodes (e.g. "pd-balanced").
            boot_disk_kms_key: Optional KMS key for boot disk encryption.
            enable_confidential_storage: Whether to enable confidential storage on nodes.
        """

        project: Required[str] = REQUIRED
        zone: Required[str] = REQUIRED
        cluster: Required[str] = REQUIRED
        name: Required[str] = REQUIRED
        # If not none, node pools will be created with the service account email.
        service_account_email: Optional[str] = None
        retry_interval: int = 30
        wait_timeout: int = 30 * 60
        labels: Optional[dict[str, str]] = None
        disk_type: Optional[str] = None
        boot_disk_kms_key: Optional[str] = None
        enable_confidential_storage: Optional[bool] = None

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: NodePoolProvisioner.Config = super().from_flags(fv, **kwargs)
        cfg.project = gcp_settings("project", fv=fv)
        cfg.zone = gcp_settings("zone", fv=fv)
        cfg.cluster = gcp_settings("gke_cluster", fv=fv)
        cfg.service_account_email = gcp_settings("service_account_email", required=False, fv=fv)
        cfg.disk_type = gcp_settings("gke_disk_type", required=False, fv=fv)
        cfg.boot_disk_kms_key = gcp_settings("gke_boot_disk_kms_key", required=False, fv=fv)
        enable_cs = gcp_settings("gke_enable_confidential_storage", required=False, fv=fv)
        cfg.enable_confidential_storage = enable_cs.strip().lower() == "true" if enable_cs else None
        return cfg

    def create_for(self, job: GKEJob):
        """Creates node pools for the job."""
        raise NotImplementedError(type(self))

    def delete_for(self, job: GKEJob):
        """Deletes node pools for the job."""
        raise NotImplementedError(type(self))


class _JobProvisioningConfig(NamedTuple):
    """Extracted provisioning parameters for a TPU job.

    Attributes:
        acc_cfg: Accelerator config (provides num_replicas, instance_type, etc.).
        job_sys_property: System characteristics for the TPU type.
        topology: GKE topology string, or None if not applicable.
        use_spot_vm: Whether to use spot VMs.
        reservation: Reservation name (fully-qualified if cross-project), or None.
        location_hint: Location hint for node pool placement, or None.
        enable_tpu_ici_resiliency: Whether TPU ICI resiliency is enabled.
        enable_tpu_smart_repair: Whether TPU smart repair is enabled.
        job_priority: Job priority from the bastion jobspec, or None.
    """

    acc_cfg: AcceleratorConfig
    job_sys_property: _SystemCharacteristics
    topology: Optional[str]
    use_spot_vm: bool
    reservation: Optional[str]
    location_hint: Optional[str]
    enable_tpu_ici_resiliency: bool
    enable_tpu_smart_repair: bool
    job_priority: Optional[int]


class _NodePoolBatch(NamedTuple):
    """A batch of node pools to create or delete.

    Attributes:
        names: Node pool names (e.g. from construct_node_pool_name).
        additional_labels_list: Per-pool label dicts, parallel to names.
            Empty list for delete batches (labels are not used on delete).
    """

    names: list[str]
    additional_labels_list: list[dict[str, str]]


class TPUNodePoolProvisioner(NodePoolProvisioner):
    """TPU node pool provisioner."""

    def _get_job_provisioning_config(self, job: GKEJob) -> _JobProvisioningConfig:
        """Extracts and resolves all provisioning parameters from the job config.

        Args:
            job: The GKEJob whose builder config carries TPU accelerator settings.

        Returns:
            A _JobProvisioningConfig with all parameters needed by create_node_pools
            or delete_node_pools.

        Raises:
            TypeError: If the job's builder config is not a supported type.
        """
        job_cfg: GKEJob.Config = job.config
        builder_cfg: TPUReplicatedJob.Config | PathwaysLeaderWorkerTemplate.Config = job_cfg.builder

        # TODO(markblee,ethanli,muyang_yu): Refactor so we do not need to make assumptions about
        # TPUGKEJob implementation and internals.
        if not isinstance(builder_cfg, _SUPPORTED_BUILDER_TYPES):
            raise TypeError(
                "Expected"
                + f"{TPUReplicatedJob.Config}"
                + f"{PathwaysLeaderWorkerTemplate.Config},"
                + f"got {type(builder_cfg)}."
            )

        if isinstance(builder_cfg, PathwaysLeaderWorkerTemplate.Config):
            # pylint: disable-next=protected-access
            builder_cfg = builder_cfg.inner
        acc_cfg = builder_cfg.accelerator
        reservation = builder_cfg.reservation
        reservation_project = builder_cfg.reservation_project
        location_hint = builder_cfg.location_hint
        enable_tpu_ici_resiliency = builder_cfg.enable_tpu_ici_resiliency
        enable_tpu_smart_repair = builder_cfg.enable_tpu_smart_repair
        tpu_type = infer_tpu_type(acc_cfg.instance_type)
        job_sys_property = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[tpu_type]

        tier = os.environ.get("BASTION_TIER", 0)
        if tier == "0" and reservation is not None:
            # By default, a bare reservation name is assumed to belong to the same project
            # as the node pool. For shared reservations owned by a different project, GKE
            # requires the fully-qualified format:
            # projects/{project}/reservations/{reservation}.
            if reservation_project is not None and not reservation.startswith("projects/"):
                reservation = f"projects/{reservation_project}/reservations/{reservation}"
            logging.info("Found tier=%s in env. Using reservation=%s", tier, reservation)
            use_spot_vm = False
        else:
            logging.info("Found tier=%s in env. Using spot quota", tier)
            use_spot_vm = True
            reservation = None

        job_priority = None
        if os.environ.get(_BASTION_SERIALIZED_JOBSPEC_ENV_VAR):
            spec = deserialize_jobspec(
                io.StringIO(os.environ.get(_BASTION_SERIALIZED_JOBSPEC_ENV_VAR))
            )
            job_priority = spec.metadata.priority

        topology = job_sys_property.topology
        if job_sys_property.gce_machine_type == "ct6e-standard-8t":
            # If we customize chips_per_vm to use ct6e-standard-8t for v6e,
            # it is required not to set topology.
            topology = None
        elif isinstance(job, _INFERENCE_JOBS):
            # Inference jobs like Flink/Beam jobs use node pool as single
            # host nodes, we don't set topology for them.
            topology = None

        return _JobProvisioningConfig(
            acc_cfg=acc_cfg,
            job_sys_property=job_sys_property,
            topology=topology,
            use_spot_vm=use_spot_vm,
            reservation=reservation,
            location_hint=location_hint,
            enable_tpu_ici_resiliency=enable_tpu_ici_resiliency,
            enable_tpu_smart_repair=enable_tpu_smart_repair,
            job_priority=job_priority,
        )

    def _build_node_pool_names_and_labels(
        self,
        job: GKEJob,
        prov_cfg: _JobProvisioningConfig,
        *,
        start_index: int,
        end_index: int,
    ) -> _NodePoolBatch:
        """Builds node pool names and per-pool label dicts for a range of replica indices.

        Args:
            job: The GKEJob providing namespace and name.
            prov_cfg: Extracted provisioning config.
            start_index: First replica index (inclusive).
            end_index: Last replica index (exclusive).

        Returns:
            A _NodePoolBatch with parallel `names` and `additional_labels_list`.
        """
        cfg: TPUNodePoolProvisioner.Config = self.config
        job_cfg: GKEJob.Config = job.config

        node_pool_names = []
        additional_labels_list = []
        for i in range(start_index, end_index):
            node_pool_names.append(
                construct_node_pool_name(
                    jobset_namespace=job_cfg.namespace, jobset_name=job_cfg.name, index=i
                )
            )
            # This is required because the jobset-controller-manager will
            # inject this node-selector to pods.
            # https://github.com/nstogner/jobset/commit/59c93ca5b0df408b7b6f19edcbd255079c8e0b2a
            # TODO(ethanli): remove this hack once jobset-controller-manager
            #  supports disabling node-selector injections
            job_key = hashlib.sha1(
                f"{job_cfg.namespace}/{job_cfg.name}-job-{i}".encode()
            ).hexdigest()
            # Copy cfg.labels so per-pool additions (`job-key`,
            # `job-priority`, smart-repair) don't leak back into the
            # shared cfg.labels dict across pools or invocations.
            additional_labels = dict(cfg.labels) if cfg.labels else {}
            additional_labels.update({"job-key": job_key})

            # Populate job-priority label to nodes.
            if prov_cfg.job_priority is not None:
                additional_labels.update({"job-priority": str(prov_cfg.job_priority)})

            if prov_cfg.enable_tpu_smart_repair:
                additional_labels.update({"cloud.google.com/gke-tpu-auto-restart": "true"})

            additional_labels_list.append(additional_labels)

        return _NodePoolBatch(names=node_pool_names, additional_labels_list=additional_labels_list)

    def create_for(self, job: GKEJob):
        """Creates named node pools for the job."""

        cfg: TPUNodePoolProvisioner.Config = self.config
        prov_cfg = self._get_job_provisioning_config(job)
        num_node_pools = prov_cfg.acc_cfg.num_replicas

        node_pool_names, additional_labels_list = self._build_node_pool_names_and_labels(
            job, prov_cfg, start_index=0, end_index=num_node_pools
        )

        start_time = time.perf_counter()
        create_node_pools(
            node_pool_names,
            project=cfg.project,
            zone=cfg.zone,
            cluster=cfg.cluster,
            pre_provisioner_id=cfg.name,
            num_nodes_per_pool=prov_cfg.job_sys_property.vms_per_slice,
            machine_type=prov_cfg.job_sys_property.gce_machine_type,
            topology=prov_cfg.topology,
            use_spot_vm=prov_cfg.use_spot_vm,
            reservation=prov_cfg.reservation,
            location_hint=prov_cfg.location_hint,
            enable_tpu_ici_resiliency=prov_cfg.enable_tpu_ici_resiliency,
            service_account_email=cfg.service_account_email,
            additional_labels_list=additional_labels_list,
            disk_type=cfg.disk_type,
            boot_disk_kms_key=cfg.boot_disk_kms_key,
            enable_confidential_storage=cfg.enable_confidential_storage,
            retry_interval=cfg.retry_interval,
            wait_timeout=cfg.wait_timeout,
        )

        elapsed_time = time.perf_counter() - start_time
        logging.info(
            "%s node pools for %s creation took %s seconds", num_node_pools, cfg.name, elapsed_time
        )

    def delete_for(self, job: GKEJob):
        """Deletes node pools of the job."""

        cfg: TPUNodePoolProvisioner.Config = self.config
        job_cfg: GKEJob.Config = job.config
        builder_cfg: TPUReplicatedJob.Config = job_cfg.builder

        # TODO(markblee,ethanli,muyang_yu): Refactor so we do not need to make assumptions about
        # TPUGKEJob implementation and internals.
        if not isinstance(builder_cfg, _SUPPORTED_BUILDER_TYPES):
            raise TypeError(f"Expected {_SUPPORTED_BUILDER_TYPES}" + f"got {type(builder_cfg)}.")

        num_node_pools = builder_cfg.accelerator.num_replicas
        node_pool_names = []

        for i in range(num_node_pools):
            node_pool_names.append(
                construct_node_pool_name(
                    jobset_namespace=job_cfg.namespace, jobset_name=job_cfg.name, index=i
                )
            )

        start_time = time.perf_counter()
        delete_node_pools(
            node_pool_names,
            project=cfg.project,
            zone=cfg.zone,
            cluster=cfg.cluster,
            retry_interval=cfg.retry_interval,
            wait_timeout=cfg.wait_timeout,
        )

        elapsed_time = time.perf_counter() - start_time
        logging.info("Node pool deletion took %s seconds", elapsed_time)

    def create_pools_to(self, job: GKEJob, target_replicas: int, *, wait: bool = True) -> int:
        """Idempotently ensures node pools exist for indices [0..target_replicas).

        Existing pools are skipped (create_node_pools checks pool status and
        no-ops if already RUNNING). Use this on scale-up before patching the
        workload's replica count so newly-admitted replicas have nodes to
        schedule onto.

        Args:
            job: The GKEJob whose node pools should be created.
            target_replicas: Number of pools to ensure exist (indices 0..N-1).
            wait: When True (default), block until all pools are RUNNING
                (or `cfg.wait_timeout` elapses). When False, fire creates for
                missing pools and return immediately with the count of pools
                already RUNNING.

        Returns:
            Number of pools in [0..target_replicas) whose status is RUNNING
            at the end of the call. With wait=True this is always
            target_replicas; with wait=False it can be less.
        """
        if target_replicas <= 0:
            return 0

        cfg: TPUNodePoolProvisioner.Config = self.config
        prov_cfg = self._get_job_provisioning_config(job)

        node_pool_names, additional_labels_list = self._build_node_pool_names_and_labels(
            job, prov_cfg, start_index=0, end_index=target_replicas
        )
        logging.info(
            "create_pools_to: ensuring %s node pools exist for %s (wait=%s)",
            target_replicas,
            cfg.name,
            wait,
        )
        start_time = time.perf_counter()
        ready = create_node_pools(
            node_pool_names,
            project=cfg.project,
            zone=cfg.zone,
            cluster=cfg.cluster,
            pre_provisioner_id=cfg.name,
            num_nodes_per_pool=prov_cfg.job_sys_property.vms_per_slice,
            machine_type=prov_cfg.job_sys_property.gce_machine_type,
            topology=prov_cfg.topology,
            use_spot_vm=prov_cfg.use_spot_vm,
            reservation=prov_cfg.reservation,
            location_hint=prov_cfg.location_hint,
            enable_tpu_ici_resiliency=prov_cfg.enable_tpu_ici_resiliency,
            service_account_email=cfg.service_account_email,
            additional_labels_list=additional_labels_list,
            disk_type=cfg.disk_type,
            boot_disk_kms_key=cfg.boot_disk_kms_key,
            enable_confidential_storage=cfg.enable_confidential_storage,
            retry_interval=cfg.retry_interval,
            wait_timeout=cfg.wait_timeout if wait else 0,
        )
        elapsed_time = time.perf_counter() - start_time
        msg = (
            "create_pools_to: %s/%s pools ready for %s after %s seconds"
            if wait
            else "create_pools_to: dispatched create requests; "
            "%s/%s pools currently RUNNING for %s after %s seconds"
        )
        logging.info(msg, ready, target_replicas, cfg.name, elapsed_time)
        return ready

    def delete_pools_above(
        self,
        job: GKEJob,
        target_replicas: int,
        max_replicas: int,
        *,
        wait: bool = True,
    ) -> int:
        """Deletes node pools for indices [target_replicas..max_replicas).

        Pools that do not exist are skipped gracefully (delete_node_pools
        treats NOT_EXIST as already-deleted). Use this on scale-down AFTER
        patching the workload's replica count so the soon-to-be-orphaned
        replicas have terminated and pods can drain before pool deletion.

        Args:
            job: The GKEJob whose node pools should be cleaned up.
            target_replicas: Number of pools to keep (indices 0..target-1 stay).
            max_replicas: Upper bound of indices to consider for deletion.
                Pass the max from the scaling spec so all previously-created
                pools beyond target_replicas are cleaned up.
            wait: When True (default), block until all pools are deleted
                (or `cfg.wait_timeout` elapses). When False, fire deletes for
                still-existing pools and return immediately with the count of
                pools that still exist.

        Returns:
            Number of pools in [target_replicas..max_replicas) that still
            exist at the end of the call (status not NOT_EXIST). With
            wait=True this is always 0; with wait=False it can be more.
        """
        if target_replicas >= max_replicas:
            return 0

        cfg: TPUNodePoolProvisioner.Config = self.config
        prov_cfg = self._get_job_provisioning_config(job)

        node_pool_names, _ = self._build_node_pool_names_and_labels(
            job, prov_cfg, start_index=target_replicas, end_index=max_replicas
        )
        logging.info(
            "delete_pools_above: deleting %s node pools (indices %s..%s) for %s (wait=%s)",
            max_replicas - target_replicas,
            target_replicas,
            max_replicas,
            cfg.name,
            wait,
        )
        start_time = time.perf_counter()
        remaining = delete_node_pools(
            node_pool_names,
            project=cfg.project,
            zone=cfg.zone,
            cluster=cfg.cluster,
            retry_interval=cfg.retry_interval,
            wait_timeout=cfg.wait_timeout if wait else 0,
        )
        elapsed_time = time.perf_counter() - start_time
        msg = (
            "delete_pools_above: %s pools remaining for %s after %s seconds"
            if wait
            else "delete_pools_above: dispatched delete requests; "
            "%s pools still existing for %s after %s seconds"
        )
        logging.info(msg, remaining, cfg.name, elapsed_time)
        return remaining

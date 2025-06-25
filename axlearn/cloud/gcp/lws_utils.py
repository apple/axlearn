# Copyright © 2025 Apple Inc.

"""Utilities for building LeaderWorkerSet specs"""

import logging
import os
from typing import Any, Optional, Sequence

from absl import flags

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import (
    AcceleratorConfig,
    FlagConfigurable,
    accelerator_flags,
    parse_kv_flags,
)
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS
from axlearn.cloud.gcp.tpu import get_default_env
from axlearn.common.compiler_options import infer_tpu_type
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested

# Kubernetes pod annotation keys. Used by TPUReplicatedJob to support multi NIC.
# Refer to GKE TPU provisioner for more context:
# https://github.com/GoogleCloudPlatform/ai-on-gke/blob/5f256eed7075a5cb8e73cd72328aea46237b8ce6/tpu-provisioner/internal/cloud/common.go#L29-L31
_ANNOTATION_ADDITIONAL_NODE_NETWORKS = "tpu-provisioner.cloud.google.com/additional-node-networks"
_ANNOTATION_NODE_SERVICE_ACCOUNT = "tpu-provisioner.cloud.google.com/node-service-account"


class BaseLeaderWorkerTemplate(FlagConfigurable):
    """
    Common base class for LeaderWorker Templates
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """
        Configures BaseLeaderWorker.
        Attributes:
            name: Name of the LeaderWorkerSet
            command: Command to be executed.
            accelerator: Accelerator configuration.
            env_vars: Optional env vars to set.
            service_account: Optional service account to execute the job as.
            output_dir: An optional GCS path to upload LWS outputs to.
        """

        name: Required[str] = REQUIRED
        # TODO: Change this to be a list of str[], to support different commands
        # between leader and workers
        command: Required[str] = REQUIRED
        accelerator: AcceleratorConfig = AcceleratorConfig()
        env_vars: dict[str, str] = {}
        service_account: Optional[str] = None
        output_dir: Optional[str] = None

    @classmethod
    def define_flags(cls, fv):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        accelerator_flags(**common_kwargs)
        # NOTE: the parent typically sets these flags, so we leave them as None.
        flags.DEFINE_string("name", None, "Name of the LWS.", **common_kwargs)
        flags.DEFINE_string("command", None, "Command to execute.", **common_kwargs)
        flags.DEFINE_multi_string("env", [], "Env var in the format key:value.", **common_kwargs)
        flags.DEFINE_string(
            "service_account",
            None,
            "If specified, will run job as the service account.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "enable_pre_provisioner", None, "Whether to enable pre-provisioner.", **common_kwargs
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: BaseLeaderWorkerTemplate.Config = super().from_flags(fv, **kwargs)
        cfg.service_account = cfg.service_account or gcp_settings(
            "k8s_service_account", default="default", fv=fv
        )
        cfg.accelerator.set(instance_type=fv.instance_type, num_replicas=fv.num_replicas)
        return cfg

    def __init__(self, cfg: Config, *, bundler: Bundler):
        super().__init__(cfg)
        self._bundler = bundler

    def __call__(self) -> Sequence[Nested[Any]]:
        """Builds LeaderWorkerTemplate for the LWS API.

        Returns:
        A nested dict corresponding to a LeaderWorkerTemplate config.
        """
        raise NotImplementedError(type(self))


class TPULeaderWorkerTemplate(BaseLeaderWorkerTemplate):
    """Builds a LeaderWorkerTemplate spec for TPUs"""

    @config_class
    class Config(BaseLeaderWorkerTemplate.Config):
        """Configures TPULeaderWorkerTemplate
        Attributes:
            reservation: If specified, the TPU reservation name. This is not necessarily specific to
                GKE and can be the same as e.g. the QRM reservation.
                https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/reservations/list
            reservation_project: GCP project to which the TPU reservation belongs. This is needed
                for shared reservations. If specified, the TPU provisioner will instead use the
                full reservation name for reservation affinity in the format:
                "projects/<reservation_project>/reservations/<reservation>"
                https://github.com/GoogleCloudPlatform/ai-on-gke/blob/889ec98f9b9a7aec05eb0f9890ada1f4c59b6159/tpu-provisioner/internal/cloud/gke.go#L328-L334
        """

        reservation: Optional[str] = None
        reservation_project: Optional[str] = None
        enable_tpu_ici_resiliency: Optional[bool] = None
        location_hint: Optional[str] = None
        enable_tpu_smart_repair: bool = False
        priority_class: Optional[str] = None
        additional_node_networks: Optional[str] = None
        # This config is made Optional for backwards compatibility
        enable_pre_provisioner: Optional[bool] = None

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
        flags.DEFINE_string(
            "priority_class",
            None,
            "The GKE PriorityClass for the job.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: TPULeaderWorkerTemplate.Config = super().from_flags(fv, **kwargs)
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
        cfg.location_hint = gcp_settings("location_hint", required=False, fv=fv)
        cfg.enable_tpu_smart_repair = bool(
            gcp_settings("enable_tpu_smart_repair", required=False, fv=fv)
        )
        cfg.additional_node_networks = gcp_settings(
            "additional_node_networks", required=False, fv=fv
        )
        return cfg

    def __init__(self, cfg, *, bundler):
        super().__init__(cfg, bundler=bundler)
        cfg: TPULeaderWorkerTemplate.Config = self.config
        self._tpu_type = infer_tpu_type(cfg.accelerator.instance_type)
        if self._tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {self._tpu_type}")
        if cfg.additional_node_networks and not cfg.service_account:
            raise ValueError("service_account must be set if additional_node_networks is set.")

    def _build_container(self) -> dict:
        """Build the container to be used in the leader template

        Returns:
            A nested dict correspoding to a k8s Container config.
        """
        cfg: TPULeaderWorkerTemplate.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        env_vars = {**cfg.env_vars}
        k8s_env_vars = [dict(name=k, value=str(v)) for k, v in env_vars.items()]
        resources = {"limits": {"google.com/tpu": system.chips_per_vm}}
        if cfg.enable_tpu_ici_resiliency is not None:
            env_vars["ENABLE_ICI_RESILIENCY"] = str(cfg.enable_tpu_ici_resiliency).lower()

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            command=["bash", "-c", cfg.command],
            resources=resources,
            env=k8s_env_vars,
            imagePullPolicy="Always",
        )

    def _build_pod(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        annotations, labels, selector, tolerations = {}, {}, {}, []

        tier = os.environ.get("BASTION_TIER", None)
        if tier == "0" and cfg.reservation is not None:
            logging.info("Found tier=%s in env. Using reservation=%s", tier, cfg.reservation)
            selector.update({"cloud.google.com/reservation-name": cfg.reservation})
            if cfg.reservation_project:
                selector.update({"cloud.google.com/reservation-project": cfg.reservation_project})
            labels.update({"bastion-tier": "reserved"})
        if cfg.location_hint is not None:
            selector.update({"cloud.google.com/gke-location-hint": str(cfg.location_hint).lower()})
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

        if cfg.enable_pre_provisioner:
            # Used by pre-provisioner.
            selector.update({PRE_PROVISIONER_LABEL: cfg.name})

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

        if cfg.enable_tpu_ici_resiliency is not None:
            selector.update(
                {
                    "cloud.google.com/gke-tpu-ici-resiliency": str(
                        cfg.enable_tpu_ici_resiliency
                    ).lower()
                }
            )

        spec = dict(
            nodeSelector={
                "cloud.google.com/gke-tpu-accelerator": system.gke_accelerator,
                "cloud.google.com/gke-tpu-topology": system.topology,
                **selector,
            },
            tolerations=tolerations,
            containers=[self._build_container()],
            serviceAccountName=cfg.service_account,
        )

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

        if cfg.priority_class:
            spec["priorityClassName"] = cfg.priority_class

        return dict(metadata=dict(annotations=annotations, labels=labels), spec=spec)

    def __call__(self) -> Nested[Any]:
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        return dict(
            size=system.vms_per_slice,
            workerTemplate=self._build_pod(),
        )

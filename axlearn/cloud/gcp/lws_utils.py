# Copyright © 2025 Apple Inc.

"""Utilities for building LeaderWorkerSet specs"""

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
    parse_kv_flags,
)
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
)
from axlearn.cloud.gcp.tpu import get_default_env, infer_tpu_workers
from axlearn.common.compiler_options import infer_tpu_type
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested



class BaseLeaderWorkerTemplate(FlagConfigurable):
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
        """
        name: Required[str] = REQUIRED
        # TODO: Change this to be a list of str[], to support different commands
        # between leader and workers
        command: Required[str] = REQUIRED
        accelerator: AcceleratorConfig = AcceleratorConfig()
        env_vars: dict[str, str] = {}
        service_account: Optional[str] = None

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
    
    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("reservation", None, "TPU reservation.", **common_kwargs)
        flags.DEFINE_string(
            "reservation_project", None, "TPU reservation project.", **common_kwargs
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
        return cfg
    
    def __init__(self, cfg, *, bundler):
        super().__init__(cfg, bundler=bundler)
        cfg: TPULeaderWorkerTemplate.Config = self.config
        self._tpu_type = infer_tpu_type(cfg.accelerator.instance_type)
        if self._tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {self._tpu_type}")


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

        return dict(
            name=cfg.name,
            image=self._bundler.id(cfg.name),
            command=["bash", "-c", cfg.command],
            resources=resources,
            env=k8s_env_vars,
            imagePullPolicy="Always",
        )
    
    def _build_worker_pod(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        annotations, labels, selector, tolerations = {}, {}, {}, {}
        if cfg.reservation is not None:
            logging.info("Using reservation=%s", cfg.reservation)
            selector.update({"cloud.google.com/reservation-name": cfg.reservation})
        if cfg.reservation_project is not None:
            selector.update({"cloud.google.com/reservation-project": cfg.reservation_project})
            labels.update({"bastion-tier": "reserved"})
        
        spec = dict(
            nodeSelector={
                "cloud.google.com/gke-tpu-accelerator": system.gke_accelerator,
                "cloud.google.com/gke-tpu-topology": system.topology,
                **selector,
            },
            containers=[self._build_container()], 
            serviceAccountName=cfg.service_account,
        )

        return dict(
            metadata=dict(annotations=annotations, labels=labels),
            spec=spec
        )

    def _build_pathways_jetstream_container(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        staging_location = f"{cfg.output_dir}/pathways-staging"
        checkpoint_location = f"gs://lkolluru-axlearn-pathways/maxtext/llama-3-405b/final/bf16/scanned/0/items"  #### Need to change ###

        return [
            dict(
                name="pathways-proxy",
                image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
                #restartPolicy="Always",
                imagePullPolicy="Always",
                args=[
                    f"--resource_manager_address=localhost:38677",
                    f"--server_port=38681",
                    f"--gcs_scratch_location={staging_location}",
                ],
                ports=[dict(containerPort="38681")],
            ),
            dict(
                name="pathways-rm",
                image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
                # https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/#pod-sidecar-containers
                # SideCar container is an init container with restartPolicy as "Always".
                # restartPolicy="Always",
                imagePullPolicy="Always",
                env=[
                    {
                        "name": "TPU_SKIP_MDS_QUERY",
                        "value": "true",
                    },
                    #### Not sure on this one #####
                    {
                        "name": "HOST_ADDRESS",
                        "value": "localhost",
                    },
                ],
                args=[
                    f"--server_port=38677",
                    "--node_type=resource_manager",
                    f"--instance_count={cfg.accelerator.num_replicas}",
                    f"--instance_type={pathways_tpu_version}:{system.topology}",
                    f"--gcs_scratch_location={staging_location}",
                ],
            ),
            dict(
                name="jax-tpu",
                image="us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-pathways:v0.2.0",
                # restartPolicy="Always",
                imagePullPolicy="Always",
                env=[
                    {
                        "name": "LOG_LEVEL",
                        "value": "INFO",
                    },
                ],
                args=[
                    f"MaxText/configs/v5e/inference/llama3_405b_v5e-64.yml",
                    f"model_name=llama3.1-405b",
                    f"load_parameters_path=",
                    f"max_prefill_predict_length=1024",
                    f"max_target_length=2048",
                    f"async_checkpointing=false",
                    f"steps=1",
                    f"ici_fsdp_parallelism=1",
                    f"ici_autoregressive_parallelism=2",
                    f"ici_tensor_parallelism=8",
                    f"scan_layers=false",
                    f"weight_dtype=bfloat16",
                    f"per_device_batch_size=10",
                    f"enable_single_controller=true",
                    f"quantization=int8",
                    f"quantize_kvcache=true",
                    f"checkpoint_is_quantized=true",
                    f"enable_model_warmup=true",
                ],
                ports=[dict(containerPort="9000")],
                startupProbe=dict(
                    httpGet=dict(
                        path= "/healthcheck",
                        port= "8000",
                        scheme = "HTTP",
                    ),
                    periodSeconds="1",
                    initialDelaySeconds="600",
                    failureThreshold="10000",
                ),
                livenessProbe=dict(
                    httpGet=dict(
                        path= "/healthcheck",
                        port= "8000",
                        scheme = "HTTP",
                    ),
                    periodSeconds="60",
                    failureThreshold="10",
                ),
                readinessProbe=dict(
                    httpGet=dict(
                        path= "/healthcheck",
                        port= "8000",
                        scheme = "HTTP",
                    ),
                    periodSeconds="60",
                    failureThreshold="10",
                ),

            ),
            dict(
                name="jetstream-http",
                image="us-docker.pkg.dev/cloud-tpu-images/inference/jetstream-http:v0.2.3",
                #restartPolicy="Always",
                imagePullPolicy="Always",
                ports=[dict(containerPort="8000")],
            ),
        ]

        

    def _build_leader_pod(self) -> dict:
        cfg: TPULeaderWorkerTemplate.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        annotations, labels, selector, tolerations = {}, {}, {}, {}

        if cfg.reservation is not None:
            logging.info("Using reservation=%s", cfg.reservation)
            selector.update({"cloud.google.com/reservation-name": cfg.reservation})
        if cfg.reservation_project is not None:
            selector.update({"cloud.google.com/reservation-project": cfg.reservation_project})
            labels.update({"bastion-tier": "reserved"})
        
        ### Labels Update #####
        labels.update({"app": "jetstream-pathways"})
        
        spec = dict(
            nodeSelector={
                "cloud.google.com/gke-tpu-accelerator": system.gke_accelerator,
                "cloud.google.com/gke-tpu-topology": system.topology,
                **selector,
            },
            containers=[self._build_pathways_jetstream_container()], 
            serviceAccountName=cfg.service_account,
        )

        return dict(
            metadata=dict(annotations=annotations, labels=labels),
            spec=spec
        )

    
    def __call__(self) -> Nested[Any]:
        cfg: TPULeaderWorkerTemplate.Config = self.config
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        return dict(
            size=system.vms_per_slice+1,
            leaderTemplate =self._build_leader_pod(),
            workerTemplate =self._build_worker_pod(),
        )





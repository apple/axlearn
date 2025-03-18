# Copyright © 2025 Apple Inc.

"""A helper module to launch and manage Apache Flink + Beam bundles on GKE."""
import logging
import math
import time
from typing import Any, Dict

import kubernetes as k8s

from axlearn.cloud.gcp import job
from axlearn.cloud.gcp.job import GKEJob
from axlearn.cloud.gcp.jobs.tpu_utils import get_default_env
from axlearn.cloud.gcp.jobset_utils import TPUReplicatedJob
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_CPU_CHARACTERISTICS,
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
    _SystemCharacteristics,
)
from axlearn.cloud.gcp.utils import BEAM_SUBMITTER_LABEL, delete_flink_deployment, delete_k8s_job
from axlearn.common.compiler_options import infer_tpu_type

_FINK_MAIN_CONTAINER_CPU_PERCENTAGE = 0.4

_FINK_MAIN_CONTAINER_MEMORY_PERCENTAGE = 0.4
_PYTHON_HARNESS_MEMORY_PERCENTAGE = 0.4

_TIMEOUT_SECS = 600


def _custom_flinkdeployment_kwargs() -> dict[str, str]:
    """Common kwargs needed for CustomObjectsApi Flink deployment."""
    return dict(group="flink.apache.org", version="v1beta1", plural="flinkdeployments")


class FlinkTPUGKEJob(job.GKEJob):
    """A Job that submits a Flink + Beam bundle and monitors its status."""

    builder = TPUReplicatedJob
    Config = GKEJob.Config

    def _delete(self):
        """This is a non-blocking method to delete the flink deployment and submitter job.
        It is called when GKERunner gives up retrying this job.
        """
        cfg: GKEJob.Config = self.config
        # Delete all deployments submitted by this job.
        try:
            delete_k8s_job(cfg.name, namespace=cfg.namespace)
        except k8s.client.ApiException:
            logging.info("%s does not exist, no need to delete.", cfg.name)

        flink_deployment_name = self._get_flink_cluster_name()
        try:
            delete_flink_deployment(flink_deployment_name, namespace=cfg.namespace)
        except k8s.client.ApiException:
            logging.info("%s does not exist, no need to delete.", flink_deployment_name)

    def _cleanup(self):
        """This is a blocking method to delete the flink deployment and submitter job.
        It is called at the beginning of execution for every retry.
        """
        self._delete()
        cfg: job.TPUGKEJob.Config = self.config
        while True:
            try:
                k8s.client.CustomObjectsApi().get_namespaced_custom_object_status(
                    namespace=cfg.namespace,
                    name=self._get_flink_cluster_name(),
                    **_custom_flinkdeployment_kwargs(),
                )
            except k8s.client.ApiException as e:
                if e.status == 404:
                    logging.info("Flink deployment does not exist or is deleted.")
                    break
                raise
            logging.info("Waiting for Flink cluster to be deleted, waiting 5 seconds...")
            time.sleep(5)

        while True:
            try:
                _ = k8s.client.BatchV1Api().read_namespaced_job_status(
                    namespace=cfg.namespace, name=cfg.name
                )
            except k8s.client.ApiException as e:
                if e.status == 404:
                    logging.info("Submitter job does not exist or is deleted.")
                    break
                raise
            logging.info("Waiting for submitter job to be deleted, waiting 5 seconds...")
            time.sleep(5)

    def _get_flink_cluster_name(self) -> str:
        return f"{self.config.name}-flink-cluster"

    def _execute(self) -> Any:
        """Submits a Flink Cluster and a Beam job submitter to the cluster."""
        cfg: job.TPUGKEJob.Config = self.config

        # When to retry, cleaning up the previous deployments.
        # And this is a noop for the initial execution.
        self._cleanup()

        system = self._get_system_info()

        # 1) Create a Flink cluster and wait for it to be ready.
        flink_cluster_object = self._build_flink_deployment(system)
        logging.info("Submitting Flink deployment body=%s", flink_cluster_object)
        k8s.client.CustomObjectsApi().create_namespaced_custom_object(
            namespace=cfg.namespace,
            body=flink_cluster_object,
            **_custom_flinkdeployment_kwargs(),
        )

        start_time = time.perf_counter()
        while True:
            flink_deployment = k8s.client.CustomObjectsApi().get_namespaced_custom_object_status(
                namespace=cfg.namespace,
                name=self._get_flink_cluster_name(),
                **_custom_flinkdeployment_kwargs(),
            )
            # pylint: disable=line-too-long
            # Please refer to https://nightlies.apache.org/flink/flink-kubernetes-operator-docs-release-1.8/docs/custom-resource/reference/#jobmanagerdeploymentstatus
            # for all possible status. Also note that the FlinkDeployment is in "standalone" mode.
            # The Job Manager is ready when "replica" number of task managers are initialized and
            # registered to job manager.
            if flink_deployment.get("status", {}).get("jobManagerDeploymentStatus", "") == "READY":
                break
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > _TIMEOUT_SECS:
                raise RuntimeError(f"Flink deployment is not ready after {_TIMEOUT_SECS} seconds.")
            logging.info("Flink cluster is not ready yet, waiting 5 seconds...")
            time.sleep(5)

        # 2) When the Flink cluster is ready, get its IP address. This is where jobs should be
        # submitted to
        core_api = k8s.client.CoreV1Api()
        jobmanager_pods = core_api.list_namespaced_pod(
            namespace=cfg.namespace,
            label_selector=f"app={flink_deployment['metadata']['name']},component=jobmanager",
        )
        # TODO(muyang_yu): consider using pod name instead of id.
        jobmanager_ip = jobmanager_pods.items[0].status.pod_ip

        # 3) Create a job to submit user's pipeline to the Flink cluster
        job_submission = self._build_job_submission_deployment(jobmanager_ip)
        logging.info("Submitting Job job_submission=%s", job_submission)
        return k8s.client.BatchV1Api().create_namespaced_job(
            namespace=cfg.namespace,
            body=job_submission,
        )

    def _get_system_info(self) -> _SystemCharacteristics:
        tpu_type = infer_tpu_type(self.config.accelerator.instance_type)
        if tpu_type not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise NotImplementedError(f"Missing system characteristics for {tpu_type}")
        return USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[tpu_type]

    def _build_resource(
        self, system: _SystemCharacteristics, cpu_percentage: float, memory_percentage: float
    ) -> Dict[str, Any]:
        resource = {}
        machine_cpu = GCE_MACHINE_TYPE_TO_CPU_CHARACTERISTICS.get(system.gce_machine_type, None)
        if machine_cpu:
            resource["cpu"] = math.floor(machine_cpu * cpu_percentage)

        machine_memory_gi = GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS.get(
            system.gce_machine_type, None
        )
        if machine_memory_gi is not None:
            resource["memory"] = f"{math.floor(machine_memory_gi*memory_percentage)}Gi"
        return resource

    def _build_resources(
        self, system: _SystemCharacteristics, memory_percentage: float
    ) -> Dict[str, Any]:
        machine_memory_gi = GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS.get(
            system.gce_machine_type, None
        )
        resources = {"limits": {"google.com/tpu": system.chips_per_vm}}
        if machine_memory_gi is not None:
            request_memory_gi = machine_memory_gi * memory_percentage
            resources["limits"]["memory"] = f"{machine_memory_gi}Gi"
            resources["requests"] = {"memory": f"{math.floor(request_memory_gi)}Gi"}
        return resources

    def _build_checkpoint_path(self):
        return f"{self.config.output_dir}/flink_checkpoints"

    def _build_flink_deployment(self, system: _SystemCharacteristics) -> Dict[str, Any]:
        cfg: job.GKEJob.Config = self.config
        return dict(
            apiVersion="flink.apache.org/v1beta1",
            kind="FlinkDeployment",
            metadata=dict(namespace=cfg.namespace, name=self._get_flink_cluster_name()),
            spec=dict(
                image="flink:1.18",
                flinkVersion="v1_18",
                serviceAccount=cfg.service_account,
                # Standalone mode supports initing Task Manager before beam
                # pipeline is submitted. This can avoid Task Manager initialization
                # taking too long and the job submission timeouts.
                mode="standalone",
                podTemplate=dict(
                    spec=dict(
                        initContainers=[
                            # pylint: disable=protected-access
                            # pytype: disable=attribute-error
                            self._builder._build_uploader_container(
                                src="/opt/flink/log",
                                output_volume_mount=dict(
                                    mountPath="/opt/flink/log", name="flink-logs"
                                ),
                            )
                        ],
                        containers=[
                            dict(
                                name="flink-main-container",
                                volumeMounts=[dict(mountPath="/opt/flink/log", name="flink-logs")],
                            )
                        ],
                        volumes=[dict(name="flink-logs", emptyDir={})],
                    )
                ),
                flinkConfiguration={
                    # We reply on JAX mesh config to do data parallelism, every host will
                    # offer only one task slot.
                    "taskmanager.numberOfTaskSlots": "1",
                    "taskmanager.memory.task.off-heap.size": "16g",
                    "taskmanager.network.bind-host": "0.0.0.0",
                    "rest.address": "0.0.0.0",
                },
                # job manager's responsibility is lightweight, it is only responsible to
                # accept one request from one job submitter in this setup. So a minimum
                # resource is good enough.
                jobManager=dict(resource=dict(memory="2g", cpu=1)),
                taskManager=dict(
                    replicas=cfg.accelerator.num_replicas,
                    resource=self._build_resource(
                        system=system,
                        cpu_percentage=_FINK_MAIN_CONTAINER_CPU_PERCENTAGE,
                        memory_percentage=_FINK_MAIN_CONTAINER_MEMORY_PERCENTAGE,
                    ),
                    podTemplate=dict(
                        spec=dict(
                            nodeSelector={
                                PRE_PROVISIONER_LABEL: cfg.name,
                                "cloud.google.com/gke-accelerator-count": str(system.chips_per_vm),
                                "cloud.google.com/gke-location-hint": str(
                                    self._builder.config.location_hint
                                ),
                                "cloud.google.com/gke-tpu-accelerator": system.gke_accelerator,
                                "cloud.google.com/gke-tpu-topology": system.topology,
                            },
                            tolerations=[
                                dict(
                                    key="google.com/tpu",
                                    operator="Equal",
                                    value="present",
                                    effect="NoSchedule",
                                )
                            ],
                            initContainers=[
                                # pylint: disable=protected-access
                                # pytype: disable=attribute-error
                                self._builder._build_uploader_container(
                                    src="/opt/flink/log",
                                    output_volume_mount=dict(
                                        mountPath="/opt/flink/log", name="flink-logs"
                                    ),
                                )
                            ],
                            containers=[
                                dict(
                                    name="python-harness",
                                    volumeMounts=[
                                        dict(mountPath="/opt/flink/log", name="flink-logs")
                                    ],
                                    image=self._bundler.id(cfg.name),
                                    args=["-worker_pool"],
                                    env=[
                                        dict(name="BEAM_EXTERNAL_HOST", value="0.0.0.0"),
                                        dict(name="JAX_PLATFORMS", value="tpu"),
                                        dict(
                                            name="NODE_IP",
                                            valueFrom=dict(
                                                fieldRef=dict(
                                                    apiVersion="v1", fieldPath="status.hostIP"
                                                )
                                            ),
                                        ),
                                        dict(
                                            name="NODE_NAME",
                                            valueFrom=dict(
                                                fieldRef=dict(
                                                    apiVersion="v1", fieldPath="spec.nodeName"
                                                )
                                            ),
                                        ),
                                        # In beam batch inference, every worker is independent
                                        # and don't talk to each other. So every worker only
                                        # need to see itself.
                                        dict(name="TPU_WORKER_HOSTNAMES", value="localhost"),
                                        dict(
                                            name="TPU_WORKER_ID",
                                            value="0",
                                        ),
                                    ]
                                    + [
                                        dict(name=k, value=str(v))
                                        for k, v in get_default_env(
                                            tpu_type=infer_tpu_type(cfg.accelerator.instance_type),
                                            num_tpu_slices=cfg.accelerator.num_replicas,
                                            job_name=cfg.name,
                                        ).items()
                                    ],
                                    resources=self._build_resources(
                                        system=system,
                                        memory_percentage=_PYTHON_HARNESS_MEMORY_PERCENTAGE,
                                    ),
                                    ports=[
                                        dict(
                                            containerPort=50000, name="harness-port", protocol="TCP"
                                        )
                                    ],
                                ),
                                dict(
                                    name="flink-main-container",
                                    volumeMounts=[
                                        dict(mountPath="/opt/flink/log", name="flink-logs")
                                    ],
                                ),
                            ],
                            volumes=[dict(name="flink-logs", emptyDir={})],
                        )
                    ),
                ),
            ),
        )

    def _build_job_submission_deployment(self, job_manager_ip: str) -> Dict[str, Any]:
        cfg: job.GKEJob.Config = self.config
        user_command = cfg.command
        user_command += (
            f" --flink_master_address={job_manager_ip}"
            f" --flink_parallelism={cfg.accelerator.num_replicas}"
            # Replicate output to /output/beam_pipline_log
            f" 2>&1 | tee /output/beam_pipline_log"
        )
        return dict(
            apiVersion="batch/v1",
            kind="Job",
            metadata=dict(name=cfg.name),
            spec=dict(
                backoffLimit=0,
                template=dict(
                    metadata=dict(
                        labels=dict(
                            app=cfg.name,
                            app_type=BEAM_SUBMITTER_LABEL,
                        )
                    ),
                    spec=dict(
                        serviceAccountName=cfg.service_account,
                        volumes=[dict(name="shared-output", emptyDir={})],
                        # Makes sure all logs are uploaded before terminating the pod.
                        terminationGracePeriodSeconds=100,
                        # pylint: disable=protected-access
                        # pytype: disable=attribute-error
                        initContainers=[self._builder._build_uploader_container()],
                        containers=[
                            dict(
                                name=cfg.name,
                                env=[dict(name="PYTHONUNBUFFERED", value="1")],
                                image=self._bundler.id(cfg.name),
                                volumeMounts=[dict(name="shared-output", mountPath="/output")],
                                command=["/bin/sh", "-c"],
                                args=[
                                    user_command,
                                ],
                            )
                        ],
                        restartPolicy="Never",
                    ),
                ),
            ),
        )

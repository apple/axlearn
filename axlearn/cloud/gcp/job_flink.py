# Copyright © 2025 Apple Inc.

"""A helper module to launch and manage Apache Flink + Beam bundles on GKE."""

import enum
import logging
import math
import os
import time
from typing import Any, Dict, Optional

import kubernetes as k8s
import requests
from absl import flags

from axlearn.cloud.gcp import job
from axlearn.cloud.gcp.jobset_utils import TPUReplicatedJob
from axlearn.cloud.gcp.node_pool import PRE_PROVISIONER_LABEL
from axlearn.cloud.gcp.system_characteristics import (
    GCE_MACHINE_TYPE_TO_CPU_CHARACTERISTICS,
    GCE_MACHINE_TYPE_TO_MEMORY_CHARACTERISTICS,
    USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS,
    _SystemCharacteristics,
)
from axlearn.cloud.gcp.tpu import get_default_env, infer_tpu_cores, infer_tpu_workers
from axlearn.cloud.gcp.utils import BEAM_SUBMITTER_LABEL, delete_flink_deployment, delete_k8s_job
from axlearn.common.compiler_options import infer_tpu_type, infer_tpu_version
from axlearn.common.config import config_class

_FINK_MAIN_CONTAINER_CPU_PERCENTAGE = 0.4

_FINK_MAIN_CONTAINER_MEMORY_PERCENTAGE = 0.4
_PYTHON_HARNESS_MEMORY_PERCENTAGE = 0.4

_TIMEOUT_SECS = 1800

_FLINK_VERSION = "1.18"


class FlinkJobStatus(enum.Enum):
    """Flink Job Status.

    See also:
    https://nightlies.apache.org/flink/flink-docs-master/api/java/org/apache/flink/api/common/JobStatus.html

    Attributes:
        CANCELED: Job has been cancelled.
        CANCELLING: Job is being cancelled.
        CREATED: Job is newly created, no task has started to run.
        FAILED: The job has failed and is currently waiting for the cleanup to complete.
        FAILING: JobSet has failed.
        FINISHED: All of the job's tasks have successfully finished.
        INITIALIZING: The job has been received by the Dispatcher,
                      and is waiting for the job manager to receive leadership and to be created.
        RECONCILING: The job is currently reconciling and waits for task execution
                     report to recover state.
        RESTARTING: The job is currently undergoing a reset and total restart.
        RUNNING: Some tasks are scheduled or running, some may be pending, some may be finished.
        SUSPENDED: The job has been suspended which means that it has been stopped
                   but not been removed from a potential HA job store.
    """

    CANCELED = "CANCELED"
    CANCELLING = "CANCELLING"
    CREATED = "CREATED"
    FAILED = "FAILED"
    FAILING = "FAILING"
    FINISHED = "FINISHED"
    INITIALIZING = "INITIALIZING"
    RECONCILING = "RECONCILING"
    RESTARTING = "RESTARTING"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"


def _custom_flinkdeployment_kwargs() -> dict[str, str]:
    """Common kwargs needed for CustomObjectsApi Flink deployment."""
    return dict(group="flink.apache.org", version="v1beta1", plural="flinkdeployments")


# TODO(muyang_yu,markblee): Refactor to move logic into builder, which allows decoupling the
# management of namespaced_custom_objects from the construction of the flink cluster specs and the
# construction of the flink job specs.
class FlinkTPUGKEJob(job.GKEJob):
    """A Job that submits a Flink + Beam bundle and monitors its status."""

    @config_class
    class Config(job.GKEJob.Config):
        """Configures FlinkTPUGKEJob.

        Attributes:
            flink_threads_per_worker: Threads per worker.
        """

        flink_threads_per_worker: int = 1

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_integer(
            "flink_threads_per_worker",
            1,
            "For advanced users only: the number of threads per worker. "
            "A large number of flink_threads_per_worker can better utilize I/O of the node, but"
            "it will also use more TPU memory, since every inference thread will load the model "
            "once if the model loading is implemented in a singleton way. \n"
            "If this is not set, job_flink will set it to be chips_per_vm based on the TPU type.",
            **common_kwargs,
        )

    @classmethod
    def default_config(cls):
        return super().default_config().set(builder=TPUReplicatedJob.default_config())

    def __init__(self, cfg: Config, *, bundler):
        super().__init__(cfg, bundler=bundler)
        if not isinstance(cfg.builder, TPUReplicatedJob.Config):
            raise NotImplementedError(type(cfg.builder))
        # Job_manager_ip is assigned after the flink cluster is ready
        # Before that job_manager_ip is None
        self.job_manager_ip: Optional[str] = None

    def _delete(self):
        """This is a non-blocking method to delete the flink deployment and submitter job.
        It is called when GKERunner gives up retrying this job.
        """
        cfg: FlinkTPUGKEJob.Config = self.config
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
        cfg: FlinkTPUGKEJob.Config = self.config
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

    def _get_single_node_topology(self) -> str:
        """This method returns the single node topology for the configured TPU type."""
        cfg: FlinkTPUGKEJob.Config = self.config
        tpu_type = infer_tpu_type(cfg.builder.accelerator.instance_type)
        cores, hosts = infer_tpu_cores(tpu_type), infer_tpu_workers(tpu_type)
        if cores % hosts != 0:
            raise ValueError(
                f"Number of cores:{cores} is not divisible by hosts:{hosts} for TPU type:{tpu_type}"
            )
        single_host_cores = cores // hosts
        single_host_tpu_name = f"{infer_tpu_version(tpu_type)}-{single_host_cores}"
        if single_host_tpu_name not in USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS:
            raise RuntimeError(f"Can't find specs for {single_host_tpu_name}.")
        return USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[single_host_tpu_name].topology

    def _get_num_of_tpu_nodes(self, system: _SystemCharacteristics) -> int:
        cfg: FlinkTPUGKEJob.Config = self.config
        return cfg.builder.accelerator.num_replicas * system.vms_per_slice

    def _wait_for_tpu_workers_all_ready(self, job_manager_ip: str, system: _SystemCharacteristics):
        expected_tpu_workers = self._get_num_of_tpu_nodes(system)
        ready_tpu_workers = 0

        start_time = time.perf_counter()
        while True:
            # Please refer to:
            # https://nightlies.apache.org/flink/flink-docs-master/docs/ops/metrics/#rest-api-integration
            # for the endpoint and
            # https://nightlies.apache.org/flink/flink-docs-master/docs/ops/metrics/#cluster
            # for details about this metrics
            response = requests.get(
                f"http://{job_manager_ip}:8081/jobmanager/metrics?get=numRegisteredTaskManagers",
                timeout=30,
            )
            # When the flink cluster is just brought up, this metrics may not be available yet.
            if response and response.json():
                ready_tpu_workers = int(response.json()[0]["value"])
            if ready_tpu_workers == expected_tpu_workers:
                logging.info(
                    "All %s/%s TPU workers are ready.", ready_tpu_workers, expected_tpu_workers
                )
                break
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > _TIMEOUT_SECS:
                raise RuntimeError(
                    f"Not all of {expected_tpu_workers} workers are "
                    f"ready after {_TIMEOUT_SECS} seconds."
                )
            logging.info(
                "%s/%s TPU workers are ready, waiting 30 seconds...",
                ready_tpu_workers,
                expected_tpu_workers,
            )
            time.sleep(30)

    def _execute(self) -> Any:
        """Submits a Flink Cluster and a Beam job submitter to the cluster."""
        cfg: FlinkTPUGKEJob.Config = self.config

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
        self.job_manager_ip = jobmanager_pods.items[0].status.pod_ip

        # 3) Make sure all TPU pods are ready and registered to the flink cluster before submitting
        # the beam pipeline, otherwise the submitter may time out before TPU workers are all ready.
        self._wait_for_tpu_workers_all_ready(self.job_manager_ip, system)

        # 4) Create a job to submit user's pipeline to the Flink cluster
        job_submission = self._build_job_submission_deployment(self.job_manager_ip, system)
        logging.info("Submitting Job job_submission=%s", job_submission)
        return k8s.client.BatchV1Api().create_namespaced_job(
            namespace=cfg.namespace,
            body=job_submission,
        )

    def _get_system_info(self) -> _SystemCharacteristics:
        cfg: FlinkTPUGKEJob.Config = self.config
        tpu_type = infer_tpu_type(cfg.builder.accelerator.instance_type)
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
            resource["memory"] = f"{math.floor(machine_memory_gi * memory_percentage)}Gi"
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

    def _build_tpu_node_selector(self, system: _SystemCharacteristics):
        cfg: FlinkTPUGKEJob.Config = self.config
        node_selector = {
            PRE_PROVISIONER_LABEL: cfg.name,
            "cloud.google.com/gke-accelerator-count": str(system.chips_per_vm),
            "cloud.google.com/gke-tpu-accelerator": system.gke_accelerator,
            # In batch inference, we use every node independently, because they don't need to
            # communicate with each other. When we provision the nodepool, we removed the
            # placement_policy from it so that nodes' preemption doesn't impact the whole
            # nodepool. And here at pod level, we use single node's topology instead of the
            # whole slice's topology. So that jax.device_count() gets the number of chips in a
            # single node.
            "cloud.google.com/gke-tpu-topology": self._get_single_node_topology(),
        }

        location_hint = cfg.builder.location_hint
        if location_hint:
            node_selector["cloud.google.com/gke-location-hint"] = str(location_hint).lower()
        return node_selector

    def _build_flink_deployment(self, system: _SystemCharacteristics) -> Dict[str, Any]:
        cfg: FlinkTPUGKEJob.Config = self.config
        return dict(
            apiVersion="flink.apache.org/v1beta1",
            kind="FlinkDeployment",
            metadata=dict(namespace=cfg.namespace, name=self._get_flink_cluster_name()),
            spec=dict(
                image=f"flink:{_FLINK_VERSION}",
                flinkVersion=f"v{_FLINK_VERSION.replace('.', '_')}",
                serviceAccount=cfg.builder.service_account,
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
                            # pytype: enable=attribute-error
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
                    # taskmanager.numberOfTaskSlots controls the number of concurrent
                    # threads per worker.
                    "taskmanager.numberOfTaskSlots": f"{cfg.flink_threads_per_worker}",
                    "taskmanager.memory.task.off-heap.size": "16g",
                    "taskmanager.network.bind-host": "0.0.0.0",
                    "rest.address": "0.0.0.0",
                    # Store checkpointing for retry.
                    "execution.checkpointing.interval": "10m",
                    "execution.checkpointing.mode": "EXACTLY_ONCE",
                    # Fixed-delay restart strategy.
                    "restart-strategy.type": "fixed-delay",
                    "restart-strategy.fixed-delay.attempts": "3",  # Max 30 restarts
                    # Delay 10m so that TPU node have enough time to recover.
                    "restart-strategy.fixed-delay.delay": "10m",
                },
                # job manager's responsibility is lightweight, it is only responsible to
                # accept one request from one job submitter in this setup. So a minimum
                # resource is good enough.
                jobManager=dict(resource=dict(memory="2g", cpu=1)),
                taskManager=dict(
                    # We use large slices as multiple independent single nodes in inference
                    replicas=self._get_num_of_tpu_nodes(system),
                    resource=self._build_resource(
                        system=system,
                        cpu_percentage=_FINK_MAIN_CONTAINER_CPU_PERCENTAGE,
                        memory_percentage=_FINK_MAIN_CONTAINER_MEMORY_PERCENTAGE,
                    ),
                    podTemplate=dict(
                        spec=dict(
                            nodeSelector=self._build_tpu_node_selector(system=system),
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
                                # pylint: enable=protected-access
                                # pytype: enable=attribute-error
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
                                            tpu_type=infer_tpu_type(
                                                cfg.builder.accelerator.instance_type
                                            ),
                                            # Every pod is independent to each other, so they
                                            # believe they run in single slice.
                                            num_tpu_slices=1,
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

    def _build_job_submission_deployment(
        self, job_manager_ip: str, system: _SystemCharacteristics
    ) -> Dict[str, Any]:
        cfg: FlinkTPUGKEJob.Config = self.config
        user_command = cfg.builder.command
        # --flink_parallelism controls the number of replicas of all stages in the Beam pipeline
        # it executes.
        # A reasonable large number of --flink_parallelism can enable better I/O performance.
        # But if it is too large, it takes large amount of memory and time to initialize them.
        # And since this is the only job running on the flink cluster, we are using all task
        # slots from all taskmasters for this job, which is a reasonable number.
        flink_parallelism = self._get_num_of_tpu_nodes(system) * cfg.flink_threads_per_worker
        user_command += (
            f" --flink_master={job_manager_ip}:8081"
            f" --parallelism={flink_parallelism}"
            # This directory is used by FlinkRunner to store artifacts like
            # JARs, Python wheels, custom files from the main session and
            # intermediate outputs. All workers will share it.
            f" --artifacts_dir={os.path.join(cfg.builder.output_dir, 'artifacts_dir')}"
            f" --flink_version={_FLINK_VERSION}"
            f" --runner=FlinkRunner"
            # The following two flags makes sure the Flink Job manager routes the
            # execution to "EXTERNAL" flink task managers. And the flink cluster
            # has configured in a way that the routing can be done via "localhost:50000"
            f" --environment_type=EXTERNAL"
            f" --environment_config=localhost:50000"
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
                        serviceAccountName=cfg.builder.service_account,
                        volumes=[dict(name="shared-output", emptyDir={})],
                        # Makes sure all logs are uploaded before terminating the pod.
                        terminationGracePeriodSeconds=100,
                        # pylint: disable=protected-access
                        # pytype: disable=attribute-error
                        initContainers=[self._builder._build_uploader_container()],
                        # pylint: enable=protected-access
                        # pytype: enable=attribute-error
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

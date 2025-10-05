# Copyright © 2024 Apple Inc.

"""Runs and monitors Jobsets on GKE.

This also supports mounting GCS paths as GCS FUSE volumes.
The common usage is via a bastion using `axlearn gcp launch`.

However, it's also possible to run locally for debugging (examples below).

Example:

    # A dummy v5e-16 job.
    # If `gke_cluster` is configured in the config file, you can omit --cluster.
    # This command will block until completion.
    axlearn gcp launch run --instance_type=tpu-v5litepod-16 \
        --cluster=my-tpu-cluster \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile \
        -- "sleep 10; echo hello >> /output/tmp.txt"

"""

import enum
import os
import time
from typing import Optional, cast

import kubernetes as k8s
import requests
from absl import flags, logging

from axlearn.cloud.common.bastion import (
    BASTION_JOB_VERSION_ENV_VAR,
    JobLifecycleEvent,
    JobLifecycleState,
)
from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.event_queue import BaseQueueClient
from axlearn.cloud.common.utils import generate_job_name
from axlearn.cloud.gcp.cloud_build import parse_tag_from_image_id, wait_for_cloud_build
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.event_queue import event_queue_from_config
from axlearn.cloud.gcp.job import GKEJob, GKELeaderWorkerSet
from axlearn.cloud.gcp.job_flink import FlinkJobStatus, FlinkTPUGKEJob
from axlearn.cloud.gcp.jobset_utils import BASTION_JOB_VERSION_LABEL, TPUReplicatedJob
from axlearn.cloud.gcp.node_pool import (
    PRE_PROVISIONER_LABEL,
    delete_node_pools,
    list_node_pools_by_label_key,
)
from axlearn.cloud.gcp.node_pool_provisioner import NodePoolProvisioner
from axlearn.cloud.gcp.runners import utils as runner_utils
from axlearn.cloud.gcp.runners.base import BaseRunnerJob
from axlearn.cloud.gcp.utils import custom_jobset_kwargs
from axlearn.cloud.gcp.vertexai_tensorboard import (
    VertexAITensorboardUploader,
    is_vertexai_tensorboard_configured,
)
from axlearn.common.config import REQUIRED, Required, config_class, maybe_instantiate

FLAGS = flags.FLAGS


# TODO(markblee): Move this to the builder.
def _infer_reservation(jobset_spec: dict) -> Optional[str]:
    """Infers reservation given a jobset spec."""
    try:
        for job in jobset_spec["replicatedJobs"]:
            node_selector = job["template"]["spec"]["template"]["spec"]["nodeSelector"]
            # If any job has a reservation selector, return it.
            reservation = node_selector.get("cloud.google.com/reservation-name", None)
            if reservation is not None:
                return reservation
    except (TypeError, KeyError):
        logging.warning("Failed to infer reservation.")
    return None


def _infer_processor_type(jobset_spec: dict) -> Optional[str]:
    """Infers processor_type(e.g cpu, tpu) given a jobset spec."""
    try:
        has_tpu = False
        has_cpu = False
        for job in jobset_spec["replicatedJobs"]:
            node_selector = job["template"]["spec"]["template"]["spec"]["nodeSelector"]
            # Check node selector to decide whether it is a TPU or CPU job.
            # Note that the replicated job builder is expected to set these node selector.
            # So that this function is able to infer processor type for jobs.

            # This node pool selector is set by a GKE webhook for Jobsets automatically.
            tpu_type = node_selector.get("cloud.google.com/gke-tpu-accelerator", None)
            has_tpu = has_tpu or tpu_type is not None

            # This selector is set by CPU replicated job builder.
            node_pool_type = node_selector.get("axlearn/nodepool_type", None)
            has_cpu = has_cpu or node_pool_type == "workload"

        if has_tpu:
            # In a hybrid job, we considered it as "TPU"
            return "tpu"
        elif has_cpu:
            return "cpu"
    except (TypeError, KeyError):
        logging.warning("Failed to infer processor type.")
    return None


# TODO(markblee): Move this to the builder.
def _infer_job_version(jobset_spec: dict) -> Optional[int]:
    """Infers job version given a jobset spec."""
    try:
        for job in jobset_spec["replicatedJobs"]:
            labels = job["template"]["spec"]["template"]["metadata"]["labels"]
            # If any job has a job version label, return it.
            job_version = labels.get(BASTION_JOB_VERSION_LABEL, None)

            if job_version is not None:
                return int(job_version)
    except (TypeError, KeyError) as e:
        logging.warning("Failed to infer job version: %s.", e)
    return None


# TODO(ethanli): Move this to the builder.
def _infer_job_count(jobset_spec: dict) -> Optional[int]:
    """Infers job count given a jobset spec."""
    try:
        total_job_count = 0
        for job in jobset_spec["replicatedJobs"]:
            total_job_count += int(job["replicas"])

        return total_job_count
    except (TypeError, KeyError) as e:
        logging.warning("Failed to infer job count: %s.", e)
    return None


class GKERunnerJob(BaseRunnerJob):
    """Launches and monitors a GKE job via k8s JobSet API."""

    @config_class
    class Config(BaseRunnerJob.Config):
        """Configures GKERunnerJob.

        Attributes:
            name: The name of the jobset.
            inner: GKE job configuration.
            output_dir: Output directory for artifacts (e.g. XLA dumps).
            namespace: K8s namespace propagated to inner.
            cluster: GKE cluster.
            status_interval_seconds: Interval to poll status.
            vertexai_tb_uploader: Optional VertexAI Tensorboard Uploader.
            enable_pre_provisioner: Whether to enable pre-provisioner.
            pre_provisioner: Optional pre-provisioner configuration.
            event_publisher: Optional event publisher configuration.
            bundler: Bundler config.
        """

        name: Required[str] = REQUIRED
        inner: Required[GKEJob.Config] = REQUIRED
        output_dir: Required[str] = REQUIRED
        namespace: str = "default"
        cluster: Required[str] = REQUIRED
        status_interval_seconds: float = 30
        vertexai_tb_uploader: Optional[VertexAITensorboardUploader.Config] = None
        # This config is made Optional for backwards compatibility. Default is False.
        enable_pre_provisioner: Optional[bool] = None
        pre_provisioner: Optional[NodePoolProvisioner.Config] = None
        # The event publisher sends events into queue.
        event_publisher: Optional[BaseQueueClient.Config] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues = FLAGS):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the job.", **common_kwargs)
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )
        flags.DEFINE_string("namespace", None, "K8s namespace.", **common_kwargs)
        flags.DEFINE_string("cluster", None, "GKE cluster name.", **common_kwargs)
        flags.DEFINE_boolean(
            "enable_pre_provisioner", None, "Whether to enable pre-provisioner.", **common_kwargs
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        # Don't override `name` if already specified, since the default is non-deterministic.
        # NOTE: Accessing fv.name directly reads any values or default values set on either --name
        # or its aliases. On the other hand, accessing fv["name"].default ignores any values or
        # default values set by aliases.
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default("namespace", "default")
        fv.set_default(
            "output_dir", f"gs://{gcp_settings('ttl_bucket', fv=fv)}/axlearn/jobs/{fv.name}"
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: GKERunnerJob.Config = super().from_flags(fv, **kwargs)
        cfg.cluster = cfg.cluster or gcp_settings("gke_cluster", required=False, fv=fv)
        # The pre_provisioner will be configured by default in the construction of the runner
        # config, which allows its flags to be defined up-front. Here, we disable it if user decides
        # to opt-out.
        if not fv.enable_pre_provisioner:
            cfg.pre_provisioner = None
        cfg.event_publisher = event_queue_from_config(flag_values=fv)
        if is_vertexai_tensorboard_configured(fv):
            cfg.vertexai_tb_uploader = VertexAITensorboardUploader.from_flags(fv)
        return cfg

    def __init__(self, cfg: Config, *, bundler: Bundler):
        cfg = cfg.clone(max_tries=cfg.inner.max_tries, retry_interval=cfg.inner.retry_interval)
        super().__init__(cfg, bundler=bundler)
        cfg = self.config
        self._inner: GKEJob = cfg.inner.instantiate(bundler=self._bundler)

        # Log sync process.
        self._tb_uploader = None
        if cfg.vertexai_tb_uploader:
            self._tb_uploader: VertexAITensorboardUploader = cfg.vertexai_tb_uploader.set(
                summary_dir=cfg.output_dir
            ).instantiate()

        self._pre_provisioner = None
        if cfg.pre_provisioner is not None:
            self._pre_provisioner: NodePoolProvisioner = cfg.pre_provisioner.set(
                name=cfg.name,
            ).instantiate()

        self._event_publisher: BaseQueueClient = maybe_instantiate(cfg.event_publisher)

    class Status(enum.Enum):
        """GKE JobSet status.

        See also:
        https://github.com/kubernetes-sigs/jobset/blob/89bb4d61480275ea3ace72e1ac5bef5914110a66/api/jobset/v1alpha2/jobset_types.go#L54
        https://github.com/kubernetes-sigs/jobset/blob/5eb9a2a3c2d6b8b2c94acb3c34e8b4c3f34ea390/pkg/controllers/jobset_controller.go#L260

        Attributes:
            UNKNOWN: Unknown status.
            NOT_STARTED: JobSet does not exist yet.
            PENDING: JobSet exists but not all jobs are ready.
            COMPLETED: JobSet is complete.
            FAILED: JobSet has failed.
            SUSPENDED: JobSet is suspended.
            STARTUPPOLICYCOMPLETED: JobSet completed StartupPolicy.
            READY: JobSet is ready (all Jobs are ready).
            SUCCEEDED: JobSet succeeded (all Jobs succeeded). Typically also manifests as COMPLETED.
            UPDATING: Job will be relaunched with new specs.
            RESCHEDULED: Job was rescheduled onto a different tier.
        """

        UNKNOWN = "UNKNOWN"
        NOT_STARTED = "NOT_STARTED"
        PENDING = "PENDING"
        COMPLETED = "COMPLETED"
        FAILED = "FAILED"
        SUSPENDED = "SUSPENDED"
        STARTUPPOLICYCOMPLETED = "STARTUPPOLICYCOMPLETED"
        READY = "READY"
        SUCCEEDED = "SUCCEEDED"
        UPDATING = "UPDATING"
        RESCHEDULED = "RESCHEDULED"

    # TODO(markblee): Consider moving some of the logic here into the inner impl.
    def _get_status(self) -> Status:
        cfg: GKERunnerJob.Config = self.config

        try:
            resp = k8s.client.CustomObjectsApi().get_namespaced_custom_object_status(
                name=cfg.name,
                namespace=cfg.inner.namespace,
                **custom_jobset_kwargs(),
            )

            tier = os.environ.get("BASTION_TIER", 0)
            reservation = _infer_reservation(resp["spec"])
            processor_type = _infer_processor_type(resp["spec"])
            if runner_utils.should_recreate_job(tier, reservation, processor_type=processor_type):
                return GKERunnerJob.Status.RESCHEDULED

            expected_job_version = os.environ.get(BASTION_JOB_VERSION_ENV_VAR, None)
            current_job_version = _infer_job_version(resp["spec"])

            # If the job is expected to run with a newer version, relaunch it.
            if expected_job_version is not None and (
                current_job_version is None or int(expected_job_version) > current_job_version
            ):
                logging.info(
                    "Current job version is %s; expected job version is %s",
                    current_job_version,
                    expected_job_version,
                )
                return GKERunnerJob.Status.UPDATING

            # According to stogner@google.com, it's possible for "conditions" to be missing until
            # the overall jobset has completed. However, if the jobset does complete, "conditions"
            # should be a reliable indicator of overall completion status.
            conditions = resp["status"].get("conditions", [])

            # Attempt to infer overall success/failure from jobset conditions.
            # https://github.com/kubernetes-sigs/jobset/blob/89bb4d61480275ea3ace72e1ac5bef5914110a66/api/jobset/v1alpha2/jobset_types.go#L54
            for condition in conditions:
                if condition.get("status", "").lower() == "true":
                    status = condition["type"]
                    logging.info(
                        "Job reached status %s with reason %s", status, condition.get("reason")
                    )
                    return GKERunnerJob.Status[status.upper()]

            # As a fallback, accumulate statuses across replicated jobs.
            # Note that "active" status is not disjoint from "ready":
            # https://github.com/kubernetes-sigs/jobset/blob/5eb9a2a3c2d6b8b2c94acb3c34e8b4c3f34ea390/pkg/controllers/jobset_controller.go#L260
            # Note also that one or more jobs can enter "failed" but subsequently get retried at the
            # jobset level.
            # Considering the above, the main signal we get from these statuses is if all child jobs
            # are "ready", in which case we consider the job as actually running.
            statuses = {k: 0 for k in ["failed", "ready", "succeeded"]}
            for job in resp["status"]["replicatedJobsStatus"]:
                for status in statuses:
                    statuses[status] += job.get(status, 0)
            logging.info("Statuses: %s", statuses)

            # The job can enter PENDING state in a few different ways:
            # 1. If any slice fails, and we're waiting on jobset to retry, we consider the job
            #     PENDING. Note that if jobset fails overall, it'll show up in the "conditions"
            #     above.
            # 2. If all replicated job statuses above report 0, none of the jobs have started.
            if (retryable_failure := statuses.get("failed", 0)) or all(
                v == 0 for v in statuses.values()
            ):
                if retryable_failure:
                    logging.info("One or more child jobs failed, waiting for jobset to retry.")
                # Take this opportunity to reschedule if needed.
                if runner_utils.should_recreate_job(
                    tier,
                    reservation,
                    processor_type=processor_type,
                    is_pending=True,
                ):
                    return GKERunnerJob.Status.RESCHEDULED
                return GKERunnerJob.Status.PENDING

            # Return status if all replicas agree.
            total_job_count = _infer_job_count(resp["spec"])
            for status, count in statuses.items():
                # TODO(markblee): Fix this by refactoring _get_status to inner.
                # By doing so, we can also get rid of the other GKERunnerJob subclasses.
                if count == total_job_count:
                    return GKERunnerJob.Status[status.upper()]
        except k8s.client.exceptions.ApiException as e:
            if e.status == 404:
                return GKERunnerJob.Status.NOT_STARTED
            raise
        except KeyError as e:
            # Can happen if job was just submitted.
            logging.warning("Got KeyError: %s, attempting to ignore.", e)
        return GKERunnerJob.Status.UNKNOWN

    def _delete(self):
        # TODO(markblee): Make delete a public method.
        self._inner._delete()  # pylint: disable=protected-access
        if self._pre_provisioner is not None:
            self._pre_provisioner.delete_for(self._inner)

    def _reschedule(self):
        """Reschedules the jobset onto the appropriate tier.

        If we can identify that the node pool has incorrect selectors for the current scheduling
        tier, delete it to force provisioner to recreate with the right specs.
        We identify the node pool by looking for the `provisioner-nodepool-id` label set during
        creation (by auto-provisioner),
        or by node_pool.PRE_PROVISIONER_LABEL label set during provisioning (by pre-provisioner)

        TODO(markblee): Refactor this logic when jobset recreation is fixed in:
        https://github.com/GoogleCloudPlatform/ai-on-gke/tree/main/tpu-provisioner
        """
        cfg: GKERunnerJob.Config = self.config
        # Delete the jobset first, so that provisioner does not attempt to recreate the existing
        # node-pools.
        logging.info("Deleting jobset %s", cfg.name)
        self._inner._delete()  # pylint: disable=protected-access

        node_pool_label_key = "provisioner-nodepool-id"
        if self._pre_provisioner is not None:
            # TODO(ethanli): move this logic to pre-provisioner.
            node_pool_label_key = PRE_PROVISIONER_LABEL

        node_pools_dict = list_node_pools_by_label_key(
            project=cfg.project, zone=cfg.zone, cluster=cfg.cluster, label_key=node_pool_label_key
        )
        node_pools = node_pools_dict.get(cfg.name, [])
        if len(node_pools) == 0:
            logging.info("Could not infer node pool, skipping delete.")
            return
        node_pools_to_delete = []
        for node_pool in node_pools:
            node_pool_config = node_pool.get("config", {})
            reservation_affinity = node_pool_config.get("reservationAffinity", {})
            taints = node_pool_config.get("taints", [])
            tier = os.environ.get("BASTION_TIER", 0)
            has_reservation = (
                reservation_affinity.get("key") == "compute.googleapis.com/reservation-name"
                and len(reservation_affinity.get("values", [])) > 0
            )
            has_spot = any(
                taint.get("key") == "cloud.google.com/gke-spot"
                and taint.get("value") == "true"
                and taint.get("effect") == "NO_SCHEDULE"
                for taint in taints
            )
            logging.info(
                "Found existing node pool %s with tier %s.\n"
                "The reservation affinity is: %s\n"
                "The taints are: %s",
                node_pool["name"],
                tier,
                reservation_affinity,
                taints,
            )
            if (str(tier) == "0" and not has_reservation) or (str(tier) != "0" and not has_spot):
                logging.info(
                    "Since there is a mismatch, we will attempt to delete %s.",
                    node_pool["name"],
                )
                node_pools_to_delete.append(node_pool["name"])
            else:
                logging.info("Node pool appears to have the right specs.")
        if len(node_pools_to_delete) > 0:
            start_time = time.perf_counter()
            delete_node_pools(
                node_pools_to_delete,
                project=cfg.project,
                zone=cfg.zone,
                cluster=cfg.cluster,
                retry_interval=cfg.retry_interval,
                wait_timeout=30 * 60 * len(node_pools_to_delete),
            )
            elapsed_time = time.perf_counter() - start_time
            logging.info(
                "Node pools %s deletion took %s seconds", node_pools_to_delete, elapsed_time
            )

    def _execute(self):
        cfg: GKERunnerJob.Config = self.config

        # Keep track of last status to prevent duplicate events.
        last_job_status = None

        while True:
            status = self._get_status()

            # Don't retry if FAILED, since we ask GKE to handle retries.
            # Note that job remains ACTIVE until all retries are exhausted.
            if status == GKERunnerJob.Status.FAILED:
                self._maybe_publish(
                    cfg.name, msg="Job failed with error", state=JobLifecycleState.FAILED
                )
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return
            elif status in {
                GKERunnerJob.Status.SUCCEEDED,
                GKERunnerJob.Status.COMPLETED,
            }:
                self._maybe_publish(cfg.name, msg="Job succeeds", state=JobLifecycleState.SUCCEEDED)
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return
            elif status == GKERunnerJob.Status.RESCHEDULED:
                logging.info("Jobset does not match scheduling tier. Rescheduling the jobset...")
                self._reschedule()
            elif status == GKERunnerJob.Status.UPDATING:
                logging.info("Newer job version is available. Relaunching the jobset...")
                self._inner._delete()  # pylint: disable=protected-access
            elif status == GKERunnerJob.Status.NOT_STARTED:
                logging.info("Task has not started. Starting it now...")
                # pylint: disable-next=protected-access
                image_id = self._inner._builder.config.image_id
                try:
                    # Note: while the wait is blocking, the bastion will kill the runner process
                    # when it needs to reschedule.
                    if not image_id:
                        self._bundler.wait_until_finished(cfg.name)
                    else:
                        tag = parse_tag_from_image_id(image_id)
                        wait_for_cloud_build(project_id=cfg.project, image_id=image_id, tags=[tag])
                except RuntimeError as e:
                    logging.error("Bundling failed: %s. Aborting the job.", e)
                    return

                # Provision node pools for the job to run.
                if self._pre_provisioner is not None:
                    self._pre_provisioner.create_for(self._inner)

                self._inner.execute()
            else:
                # Ensure VertexAI Tensorboard Uploader is running.
                if self._tb_uploader:
                    self._tb_uploader.upload()
                logging.info("Task %s has status: %s", cfg.name, status)
                # Only emit events when status changes.
                if status == GKERunnerJob.Status.READY and status != last_job_status:
                    self._maybe_publish(
                        cfg.name, msg="Job is running", state=JobLifecycleState.RUNNING
                    )
                    last_job_status = status
            time.sleep(cfg.status_interval_seconds)

    def _maybe_publish(self, job_name: str, *, msg: str, state: JobLifecycleState):
        # Publish events to event queue.
        if not self._event_publisher:
            return
        self._event_publisher.publish(JobLifecycleEvent(job_name, state, msg))


class FlinkGKERunnerJob(GKERunnerJob):
    """A GKERunnerJob that uses FlinkGKEJob."""

    @classmethod
    def default_config(cls):
        # The Flink runner currently is special cased for this setting.
        # TODO(muyang_yu,markblee): Generalize and remove the restriction.
        cfg = super().default_config()
        return cfg.set(
            inner=FlinkTPUGKEJob.default_config().set(builder=TPUReplicatedJob.default_config())
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg = super().from_flags(fv, **kwargs)
        cfg.vertexai_tb_uploader = None
        return cfg

    def _get_flink_job_status(self, timeout: int = 5) -> GKERunnerJob.Status:
        """Fetches the Flink job status from the JobManager REST API.

        Args:
            timeout: Timeout in seconds for the HTTP request.

        Returns:
            The translated Flink job status as a `GKERunnerJob.Status`.

        Raises:
            RuntimeError: If the request fails or an error occurs.
        """
        flink_jobmanager_ip = cast(FlinkTPUGKEJob, self._inner).job_manager_ip
        # TODO(jinglu1): Move more of the logic to FlinkTPUGKEJob by exposing a get_job_status
        api_url = f"http://{flink_jobmanager_ip}:8081/jobs/overview"
        try:
            response = requests.get(api_url, timeout=timeout)
            response.raise_for_status()  # Raise error for bad responses (4xx or 5xx)
            # Response structure:
            # https://nightlies.apache.org/flink/flink-docs-master/docs/ops/rest_api/#jobs-overview
            data = response.json()
            jobs = data.get("jobs", [])
            if len(jobs) == 0:
                # This method is only called after job submission, so no jobs likely means failure.
                return GKERunnerJob.Status.FAILED
            state = jobs[-1].get("state")
            if not state:
                return GKERunnerJob.Status.FAILED
            if state == FlinkJobStatus.FINISHED.value:
                return GKERunnerJob.Status.SUCCEEDED
            if state in (FlinkJobStatus.FAILED.value, FlinkJobStatus.FAILING.value):
                return GKERunnerJob.Status.FAILED
            return GKERunnerJob.Status.READY
        except requests.Timeout as e:
            raise RuntimeError("Request to Flink JobManager timed out.") from e
        except requests.ConnectionError as e:
            raise RuntimeError("Failed to connect to Flink JobManager.") from e
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Flink JobManager returned error status: {e.response.status_code}"
            ) from e
        except requests.RequestException as e:
            raise RuntimeError(f"Unexpected error while getting Flink job status: {e}") from e

    def _get_status(self) -> GKERunnerJob.Status:
        """Retrieves the current status of the job.

        Returns:
            GKERunnerJob.Status:
                SUCCEEDED: When the job succeeded.
                PENDING: When the job hasn't started yet.
                READY: When the job is running.
                UNKNOWN: All other cases.

        Raises:
            RuntimeError: When the job fails, and GKE runner will retry it.
        """
        cfg: GKERunnerJob.Config = self.config
        try:
            resp = k8s.client.CustomObjectsApi().get_namespaced_custom_object_status(
                name=cfg.name,
                namespace=cfg.inner.namespace,
                group="batch",
                version="v1",
                plural="jobs",
            )

            status = resp.get("status", {})
            conditions = status.get("conditions", [])
            condition = conditions[-1] if conditions else {}

            # If a job complete or failed, it is shown in the last condition of its status.
            if condition.get("type") == "Complete" and condition.get("status") == "True":
                return self._get_flink_job_status()
            elif condition.get("type") == "Failed" and condition.get("status") == "True":
                return GKERunnerJob.Status.FAILED

            # Otherwise, we rely on the active/succeeded/failed to derive its status.
            # Note that we currently set restartPolicy="Never" for this job and rely on GKERunner
            # to retry the whole job submitter and flink cluster bundle as a whole. So when the
            # code passed the finish condition check and comes to here, there are only two more
            # valid cases remaining:
            # active == 0 and succeeded == 0 and failed == 0 means PENDING
            # active == 1 means READY
            active = status.get("active", 0)
            succeeded = status.get("succeeded", 0)
            failed = status.get("failed", 0)

            # The job has not started running yet.
            if active == 0 and succeeded == 0 and failed == 0:
                return GKERunnerJob.Status.PENDING

            # Check if the job is still active.
            if active > 0:
                return GKERunnerJob.Status.READY

            # If we can't determine the status, return UNKNOWN
            return GKERunnerJob.Status.UNKNOWN

        except k8s.client.exceptions.ApiException as e:
            if e.status == 404:
                return GKERunnerJob.Status.NOT_STARTED
            raise
        except KeyError as e:
            # Can happen if job was just submitted.
            logging.warning("Got KeyError: %s, attempting to ignore.", e)
        return GKERunnerJob.Status.UNKNOWN


class LWSRunnerJob(BaseRunnerJob):
    """Launches and monitors a GKE job via k8s LWS API."""

    @config_class
    class Config(BaseRunnerJob.Config):
        """Configures LWSRunnerJob.

        Attributes:
            name: The name of the LeaderWorkerSet.
            inner: LWS job configuration.
            output_dir: Output directory for artifacts (e.g. XLA dumps).
            namespace: K8s namespace propagated to inner.
            cluster: GKE cluster.
            status_interval_seconds: Interval to poll status.
            vertexai_tb_uploader: Optional VertexAI Tensorboard Uploader.
            enable_pre_provisioner: Whether to enable pre-provisioner.
            pre_provisioner: Optional pre-provisioner configuration.
            event_publisher: Optional event publisher configuration.
            bundler: Bundler config.
        """

        name: Required[str] = REQUIRED
        inner: Required[GKELeaderWorkerSet.Config] = REQUIRED
        output_dir: Required[str] = REQUIRED
        namespace: str = "default"
        cluster: Required[str] = REQUIRED
        status_interval_seconds: float = 30
        vertexai_tb_uploader: Optional[VertexAITensorboardUploader.Config] = None
        # This config is made Optional for backwards compatibility. Default is False.
        enable_pre_provisioner: Optional[bool] = None
        pre_provisioner: Optional[NodePoolProvisioner.Config] = None
        # The event publisher sends events into queue.
        event_publisher: Optional[BaseQueueClient.Config] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues = FLAGS):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("name", None, "Name of the LWS object.", **common_kwargs)
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )
        flags.DEFINE_string("namespace", None, "K8s namespace.", **common_kwargs)
        flags.DEFINE_string("cluster", None, "GKE cluster name.", **common_kwargs)
        flags.DEFINE_boolean(
            "enable_pre_provisioner", None, "Whether to enable pre-provisioner.", **common_kwargs
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        # Don't override `name` if already specified, since the default is non-deterministic.
        # NOTE: Accessing fv.name directly reads any values or default values set on either --name
        # or its aliases. On the other hand, accessing fv["name"].default ignores any values or
        # default values set by aliases.
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default("namespace", "default")
        fv.set_default(
            "output_dir", f"gs://{gcp_settings('ttl_bucket', fv=fv)}/axlearn/jobs/{fv.name}"
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: LWSRunnerJob.Config = super().from_flags(fv, **kwargs)
        cfg.cluster = cfg.cluster or gcp_settings("gke_cluster", required=False, fv=fv)
        # The pre_provisioner will be configured by default in the construction of the runner
        # config, which allows its flags to be defined up-front. Here, we disable it if user decides
        # to opt-out.
        if not fv.enable_pre_provisioner:
            cfg.pre_provisioner = None
        cfg.event_publisher = event_queue_from_config(flag_values=fv)
        if is_vertexai_tensorboard_configured(fv):
            cfg.vertexai_tb_uploader = VertexAITensorboardUploader.from_flags(fv)
        return cfg

    def __init__(self, cfg: Config, *, bundler: Bundler):
        cfg = cfg.clone(max_tries=cfg.inner.max_tries, retry_interval=cfg.inner.retry_interval)
        super().__init__(cfg, bundler=bundler)
        cfg = self.config
        self._inner: GKELeaderWorkerSet = cfg.inner.instantiate(bundler=self._bundler)

        # Log sync process.
        self._tb_uploader = None
        if cfg.vertexai_tb_uploader:
            self._tb_uploader: VertexAITensorboardUploader = cfg.vertexai_tb_uploader.set(
                summary_dir=cfg.output_dir
            ).instantiate()

        self._pre_provisioner = None
        if cfg.pre_provisioner is not None:
            self._pre_provisioner: NodePoolProvisioner = cfg.pre_provisioner.set(
                name=cfg.name,
            ).instantiate()

        self._event_publisher: BaseQueueClient = maybe_instantiate(cfg.event_publisher)

    class Status(enum.Enum):
        """GKE LeaderWorkerSet status.

        See also:
        https://github.com/kubernetes-sigs/lws/blob/main/api/leaderworkerset/v1/leaderworkerset_types.go#L342


        Attributes:
            UNKNOWN: Unknown status.
            FAILED: lws has failed.
            UPDATING: lws is updating
            PROGRESSING: lws replicas are being created
            RUNNING: lws is running with all workers and healthy leader and worker sets
        """

        UNKNOWN = "UNKNOWN"
        FAILED = "FAILED"
        UPDATING = "UPDATING"
        PROGRESSING = "PROGRESSING"
        RUNNING = "RUNNING"
        NOT_STARTED = "NOT_STARTED"

    def _get_status(self) -> Status:
        """Retrieves the current status of the job.

        Returns:
            LWSRunnerJob.Status:
                UPDATING: When the job succeeded.
                FAILED: When the job hasn't started yet.
                RUNNING: When the job is running.
                UNKNOWN: All other cases.

        Raises:
            RuntimeError: When the job fails, and LWS runner will retry it.
        """
        cfg: LWSRunnerJob.Config = self.config
        try:
            resp = k8s.client.CustomObjectsApi().get_namespaced_custom_object_status(
                name=cfg.name,
                namespace=cfg.inner.namespace,
                group="leaderworkerset.x-k8s.io",
                version="v1",
                plural="leaderworkersets",
            )

            status = resp.get("status", {})
            conditions = status.get("conditions", [])

            condition_available = None
            condition_progressive = None
            condition_update_in_progress = None

            for condition in conditions:
                if condition.get("type") == "Progressing":
                    condition_progressive = condition.get("status")
                if condition.get("type") == "Available":
                    condition_available = condition.get("status")
                if condition.get("type") == "UpdateInProgress":
                    condition_update_in_progress = condition.get("status")

            if condition_update_in_progress:
                return LWSRunnerJob.Status.UPDATING

            # If LeaderWorkerSet is running fine , condition is Available=True
            if condition_available:
                return LWSRunnerJob.Status.RUNNING

            # If LeaderWorkerSet is deploying/updating, condition is Progressing=True
            if condition_progressive:
                return LWSRunnerJob.Status.PROGRESSING

            # If LeaderWorkerSet is failed , condition is Progressing=False and Available=False
            if not condition_available and not condition_progressive:
                return LWSRunnerJob.Status.FAILED

            # If we can't determine the status, return UNKNOWN
            return LWSRunnerJob.Status.UNKNOWN

        except k8s.client.exceptions.ApiException as e:
            if e.status == 404:
                return LWSRunnerJob.Status.NOT_STARTED
            raise
        except KeyError as e:
            # Can happen if job was just submitted.
            logging.warning("Got KeyError: %s, attempting to ignore.", e)
        return LWSRunnerJob.Status.UNKNOWN

    def _delete(self):
        # TODO(markblee): Make delete a public method.
        self._inner._delete()  # pylint: disable=protected-access
        if self._pre_provisioner is not None:
            self._pre_provisioner.delete_for(self._inner)

    def _execute(self):
        cfg: LWSRunnerJob.Config = self.config

        # Keep track of last status to prevent duplicate events.
        last_job_status = None
        while True:
            status = self._get_status()

            # Don't retry if FAILED, since we ask GKE to handle retries.
            # Note that LeaderWorkerSet remains ACTIVE until all retries are exhausted.
            if status == LWSRunnerJob.Status.FAILED:
                self._maybe_publish(
                    cfg.name,
                    msg="LeaderWorkerSet failed with error",
                    state=JobLifecycleState.FAILED,
                )
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return

            elif status == LWSRunnerJob.Status.NOT_STARTED:
                logging.info("Task has not started. Starting it now...")
                # pylint: disable-next=protected-access
                image_id = self._inner._builder.config.image_id
                try:
                    # Note: while the wait is blocking, the bastion will kill the runner process
                    # when it needs to reschedule.
                    if not image_id:
                        self._bundler.wait_until_finished(cfg.name)
                    else:
                        tag = parse_tag_from_image_id(image_id)
                        wait_for_cloud_build(project_id=cfg.project, image_id=image_id, tags=[tag])
                except RuntimeError as e:
                    logging.error("Bundling failed: %s. Aborting the job.", e)
                    return

                # Provision node pools for the job to run.
                if self._pre_provisioner is not None:
                    self._pre_provisioner.create_for(self._inner)

                self._inner.execute()
            else:
                # Ensure VertexAI Tensorboard Uploader is running.
                if self._tb_uploader:
                    self._tb_uploader.upload()
                logging.info("Task %s has status: %s", cfg.name, status)
                # Only emit events when status changes.
                if status == LWSRunnerJob.Status.RUNNING and status != last_job_status:
                    self._maybe_publish(
                        cfg.name, msg="LeaderWorkerSet is running", state=JobLifecycleState.RUNNING
                    )
                    last_job_status = status
            time.sleep(cfg.status_interval_seconds)

    def _maybe_publish(self, job_name: str, *, msg: str, state: JobLifecycleState):
        # Publish events to event queue.
        if not self._event_publisher:
            return
        self._event_publisher.publish(JobLifecycleEvent(job_name, state, msg))

# Copyright © 2024 Apple Inc.

"""Runs and monitors Jobsets on GKE.

This also supports mounting GCS paths as GCS FUSE volumes.

Example:

    # A dummy v5e-16 job.
    # If `gke_cluster` is configured in the config file, you can omit --cluster.
    axlearn gcp gke start --instance_type=tpu-v5litepod-16 \
        --cluster=my-tpu-cluster \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile \
        -- "sleep 10; echo hello >> /output/tmp.txt"

    # List running jobs.
    axlearn gcp gke list

    # To stop a job.
    axlearn gcp gke stop --name=...

"""

import enum
import os
import sys
import time
from typing import Optional, Sequence, Type, cast

import kubernetes as k8s
from absl import app, flags, logging

from axlearn.cloud.common.bundler import get_bundler_config
from axlearn.cloud.common.utils import (
    configure_logging,
    generate_job_name,
    parse_action,
    parse_kv_flags,
)
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import GCPJob, GKEJob, TPUGKEJob, custom_jobset_kwargs
from axlearn.cloud.gcp.jobs.tpu_runner import with_tpu_training_defaults
from axlearn.cloud.gcp.utils import (
    catch_auth,
    delete_k8s_jobset,
    k8s_jobset_table,
    list_k8s_jobsets,
    load_kube_config,
    running_from_vm,
)
from axlearn.cloud.gcp.vertexai_tensorboard import VertexAITensorboardUploader
from axlearn.common.config import REQUIRED, Required, config_class

FLAGS = flags.FLAGS


def _infer_reservation(jobset_spec: dict) -> Optional[str]:
    """Infers reservation given a jobset spec."""
    try:
        for job in jobset_spec["replicatedJobs"]:
            node_selector = job["template"]["spec"]["template"]["spec"]["nodeSelector"]
            # If any job has a reservation selector, return it.
            if reservation := node_selector.get("cloud.google.com/reservation-name", None):
                return reservation
    except (TypeError, KeyError):
        logging.warning("Failed to infer reservation.")
    return None


class GKERunnerJob(GCPJob):
    """Launches and monitors a GKE job via k8s JobSet API."""

    inner: Type[GKEJob]

    @config_class
    class Config(GCPJob.Config):
        """Configures GKERunnerJob.

        Attributes:
            inner: GKE job configuration.
            output_dir: Output directory for artifacts (e.g. XLA dumps).
            namespace: K8s namespace propagated to inner.
            cluster: GKE cluster.
            env_vars: Optional env vars to set.
            status_interval_seconds: Interval to poll status.
            vertexai_tb_uploader: Optional VertexAI Tensorboard Uploader.
        """

        inner: Required[GKEJob.Config] = REQUIRED
        output_dir: Required[str] = REQUIRED
        namespace: str = "default"
        cluster: Required[str] = REQUIRED
        env_vars: dict[str, str] = {}
        status_interval_seconds: float = 30
        vertexai_tb_uploader: Optional[VertexAITensorboardUploader.Config] = None

    @classmethod
    def validate_inner(cls):
        if cls.inner is None:
            raise ValueError(f"A GKERunnerJob should subclass {cls} and define `inner`.")

    @classmethod
    def define_flags(cls, fv: flags.FlagValues = FLAGS):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )
        flags.DEFINE_string("namespace", "default", "K8s namespace.", **common_kwargs)
        flags.DEFINE_string("cluster", None, "GKE cluster name.", **common_kwargs)
        flags.DEFINE_multi_string("env", [], "Env var in the format key:value.", **common_kwargs)
        # Allow inner to be unspecified for help/list/stop.
        if hasattr(cls, "inner"):
            cls.inner.define_flags(fv)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cls.validate_inner()
        cfg: GKERunnerJob.Config = super().from_flags(fv, **kwargs)
        cfg.max_tries = cfg.max_tries or 10
        cfg.name = cfg.name or generate_job_name()
        cfg.env_vars = {**cfg.env_vars, **parse_kv_flags(fv.env)}
        cfg.output_dir = (
            cfg.output_dir or f"gs://{gcp_settings('ttl_bucket', fv=fv)}/axlearn/jobs/{cfg.name}"
        )
        cfg.retry_interval = 60  # Unused.
        # Construct wrapped job config. Don't propagate any configs here. Instead, propagate them in
        # __init__, since the caller may set(...) configs between now and __init__.
        cfg.inner = cast(GKEJob, cls.inner).from_flags(fv, **kwargs)
        # Configure bundler at the job level, which will be propagated to inner in __init__. This
        # ensures that callers do not need to know about the `inner` implementation when configuring
        # bundler.
        cfg.bundler = get_bundler_config(
            bundler_type=fv.bundler_type or ArtifactRegistryBundler.TYPE,
            spec=fv.bundler_spec,
            fv=fv,
        )
        cfg.cluster = cfg.cluster or gcp_settings("gke_cluster", required=False, fv=fv)
        # NOTE: service account uses workload identity to grant permissions to resources.
        cfg.service_account = cfg.service_account or gcp_settings(
            "k8s_service_account", default="default", fv=fv
        )
        return cfg

    @property
    def bundler(self):
        return self._inner.bundler

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        # Instantiate inner job impl.
        self._inner: GKEJob = cfg.inner.set(
            name=cfg.name,
            bundler=cfg.bundler,
            namespace=cfg.namespace,
            env_vars=cfg.env_vars,
            command=cfg.command,
            service_account=cfg.service_account,
            max_tries=cfg.max_tries,
            retry_interval=cfg.retry_interval,
        ).instantiate()
        # Use the bundler configured on inner.
        self._bundler = None
        # Log sync process.
        self._tb_uploader = None
        if cfg.vertexai_tb_uploader:
            self._tb_uploader: VertexAITensorboardUploader = cfg.vertexai_tb_uploader.set(
                summary_dir=cfg.output_dir
            ).instantiate()

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
            SUCCEEDED: JobSet succeeded (all Jobs suceeded). Typically also manifests as COMPLETED.
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
        RESCHEDULED = "RESCHEDULED"

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
            # If tier does not match reservation, we need to restart the jobset.
            if (str(tier) == "0") != (reservation is not None):
                logging.info(
                    "Bastion tier is %s but reservation is %s. Jobset will be recreated.",
                    tier,
                    reservation,
                )
                return GKERunnerJob.Status.RESCHEDULED

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

            # If any slice fails, consider PENDING (waiting on jobset to retry).
            # If jobset fails overall, it'll show up in the "conditions" above.
            if statuses.get("failed", 0):
                logging.info("One or more child jobs failed, waiting for jobset to retry.")
                return GKERunnerJob.Status.PENDING

            # No statuses to report yet.
            if all(v == 0 for v in statuses.values()):
                return GKERunnerJob.Status.PENDING

            # Return status if all replicas agree.
            for status, count in statuses.items():
                if count == cfg.inner.accelerator.num_replicas:
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

    def _execute(self):
        cfg: GKERunnerJob.Config = self.config

        while True:
            status = self._get_status()

            # Don't retry if FAILED, since we ask GKE to handle retries.
            # Note that job remains ACTIVE until all retries are exhausted.
            if status in {
                GKERunnerJob.Status.FAILED,
                GKERunnerJob.Status.SUCCEEDED,
                GKERunnerJob.Status.COMPLETED,
            }:
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return
            elif status == GKERunnerJob.Status.RESCHEDULED:
                logging.info("Jobset does not match scheduling tier. Deleting the jobset...")
                self._delete()
            elif status == GKERunnerJob.Status.NOT_STARTED:
                logging.info("Task does not exist. Submitting it now...")
                # Only bundle on first start, not if we're resuming monitoring.
                # If running from bastion VM, bundling should have happened on the user's machine.
                if not running_from_vm():
                    self._inner.bundler.bundle(cfg.name)
                self._inner.execute()
            else:
                # Ensure VertexAI Tensorboard Uploader is running.
                if self._tb_uploader:
                    self._tb_uploader.upload()
                logging.info("Task %s has status: %s", cfg.name, status)
            time.sleep(cfg.status_interval_seconds)


class TPUGKERunnerJob(GKERunnerJob):
    """A GKERunnerJob that uses TPUGKEJob."""

    inner = TPUGKEJob

    @classmethod
    def define_flags(cls, fv: flags.FlagValues = FLAGS):
        super().define_flags(fv)
        # TODO(markblee): Remove these, which are for backwards compat with old client.
        flags.DEFINE_alias("tpu_type", "instance_type", flag_values=fv)
        flags.DEFINE_alias("num_slices", "num_replicas", flag_values=fv)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg = super().from_flags(fv, **kwargs)
        cfg = with_tpu_training_defaults(cfg, flag_values=fv)
        return cfg


def _get_runner_or_exit(instance_type: str):
    if instance_type.startswith("tpu"):
        return TPUGKERunnerJob
    else:
        raise app.UsageError(f"Unknown instance_type {instance_type}")


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(argv, options=["start", "list", "stop"])
    load_kube_config(
        project=gcp_settings("project", fv=flag_values),
        zone=gcp_settings("zone", fv=flag_values),
        cluster=flag_values.cluster or gcp_settings("gke_cluster", fv=flag_values, required=False),
    )

    if action == "start":
        command = " ".join(argv[2:])
        if not command:
            raise app.UsageError("Command is required.")

        runner = _get_runner_or_exit(flag_values.instance_type)
        job: GKERunnerJob = runner.from_flags(flag_values, command=command).instantiate()
        job.execute()
    elif action == "list":
        print(k8s_jobset_table(list_k8s_jobsets(namespace=flag_values.namespace)))
    elif action == "stop":
        if not flag_values.name:
            raise app.UsageError("--name is required.")

        delete_k8s_jobset(flag_values.name, namespace=flag_values.namespace)
    else:
        # Unreachable -- `parse_action` will handle validation.
        raise app.UsageError(f"Unknown action {action}")


def _private_flags():
    flags.DEFINE_string("instance_type", None, "Instance type to launch.")
    FLAGS(sys.argv, known_only=True)
    # At minimum define the base GKE runner flags.
    GKERunnerJob.define_flags(FLAGS)
    # Allow instance_type to be None when running --help without any flags.
    # Otherwise, if provided, attempt to define additional per-runner flags.
    if FLAGS.instance_type:
        _get_runner_or_exit(FLAGS.instance_type).define_flags(FLAGS)


if __name__ == "__main__":
    configure_logging(logging.INFO)
    _private_flags()
    app.run(main)

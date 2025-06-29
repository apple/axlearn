"""Runs and monitors Dataflow Jobs.
"""

import enum
import platform
import shlex
import signal
import subprocess
import time
from typing import Optional

from absl import flags, logging

from axlearn.cloud.common.bastion import JobLifecycleEvent, JobLifecycleState
from axlearn.cloud.common.bundler import BaseDockerBundler, Bundler, get_bundler_config
from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.event_queue import BaseQueueClient
from axlearn.cloud.common.utils import generate_job_name, handle_popen, send_signal
from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.event_queue import event_queue_from_config
from axlearn.cloud.gcp.jobs.dataflow import (
    _docker_bundler_to_flags,
    _get_dataflow_jobs,
    _stop_dataflow_job,
)
from axlearn.cloud.gcp.runners.base import BaseRunnerJob
from axlearn.common.config import REQUIRED, Required, config_class, maybe_instantiate

FLAGS = flags.FLAGS


class DataflowRunnerJob(BaseRunnerJob):
    """Launches and monitors a Dataflow Job."""

    @config_class
    class Config(BaseRunnerJob.Config):
        """Configures DataflowRunnerJob.

        Attributes:
            name: The name of the jobset.
            output_dir: Output directory for artifacts (e.g. XLA dumps).
            status_interval_seconds: Interval to poll status.
            event_publisher: Optional event publisher configuration.
            bundler: Bundler config.
        """

        name: Required[str] = REQUIRED
        # Worker VM type.
        vm_type: Required[str] = REQUIRED
        # Dataflow command.
        command: Required[str] = REQUIRED
        # Setup command. This is used to prepare the local machine for running `cfg.command`,
        # including installing docker (if not already present) and building the worker image.
        # `cfg.command` will then be run within the worker image, to ensure a consistent build +
        # execute environment.
        setup_command: Required[str] = REQUIRED
        # Dataflow service account.
        service_account: Required[str] = REQUIRED
        # Bundler config.
        bundler: Required[Bundler.Config] = REQUIRED
        output_dir: Required[str] = REQUIRED
        status_interval_seconds: float = 30
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
        flags.DEFINE_string("vm_type", "n2-standard-2", "Worker VM type.", **common_kwargs)
        flags.DEFINE_multi_string(
            "dataflow_spec",
            [],
            "Bundler spec provided as key=value.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "service_account", None, "The dataflow service account.", **common_kwargs
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        # Don't override `name` if already specified, since the default is non-deterministic.
        # NOTE: Accessing fv.name directly reads any values or default values set on either --name
        # or its aliases. On the other hand, accessing fv["name"].default ignores any values or
        # default values set by aliases.
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default(
            "output_dir", f"gs://{gcp_settings('ttl_bucket', fv=fv)}/axlearn/jobs/{fv.name}"
        )
        fv.set_default("service_account", gcp_settings("service_account_email", fv=fv))

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: DataflowRunnerJob.Config = super().from_flags(fv, **kwargs)
        cfg.event_publisher = event_queue_from_config(flag_values=fv)
        cfg.name = cfg.name or generate_job_name()
        cfg.max_tries = cfg.max_tries or 1
        cfg.retry_interval = cfg.retry_interval or 60
        cfg.service_account = cfg.service_account or gcp_settings("service_account_email", fv=fv)

        # Construct bundler.
        cfg.bundler = get_bundler_config(
            bundler_type=fv.bundler_type or ArtifactRegistryBundler.TYPE,
            spec=fv.bundler_spec,
            fv=fv,
        )
        if not issubclass(cfg.bundler.klass, BaseDockerBundler):
            raise NotImplementedError("Expected a DockerBundler.")
        cfg.bundler.image = cfg.bundler.image or cfg.name

        # Construct bundle command.
        docker_setup_cmd = (
            # Install a docker version with buildkit support.
            # Buildkit is required for actual multi-stage '--target' (without it, docker will
            # attempt to build all stages up to the target).
            # https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script
            # We use apt-get update and wait for apt lock to release (often held on first boot).
            "if [[ ! -x $(which docker) ]]; then "
            "sudo apt-get -o DPkg::Lock::Timeout=60 update; "
            "curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh;"
            "fi"
        )
        docker_auth_cmd = (
            f"gcloud auth configure-docker {registry_from_repo(cfg.bundler.repo)} --quiet"
        )
        bundle_cmd = " ".join(
            [
                f"python3 -m {bundler.__name__} --name={cfg.name}",
                *_docker_bundler_to_flags(cfg.bundler, fv=fv),
            ]
        )

        # Construct dataflow command.
        dataflow_spec, multi_flags = cls._dataflow_spec_from_flags(cfg, fv)
        dataflow_flags = " ".join(
            sorted(flags.flag_dict_to_args(dataflow_spec, multi_flags=multi_flags))
        )
        cfg.setup_command = f"{docker_setup_cmd} && {docker_auth_cmd} && {bundle_cmd}"
        cfg.command = f"{cfg.command} {dataflow_flags}"
        return cfg

    def __init__(self, cfg: Config, *, bundler: Bundler):  # pylint: disable=redefined-outer-name
        cfg = cfg.clone(max_tries=cfg.inner.max_tries, retry_interval=cfg.inner.retry_interval)
        super().__init__(cfg, bundler=bundler)
        cfg = self.config
        self._inner: DataflowRunnerJob = cfg.inner.instantiate(bundler=self._bundler)
        self._event_publisher: BaseQueueClient = maybe_instantiate(cfg.event_publisher)

    def _start_dataflow_job(self):
        cfg: DataflowRunnerJob.Config = self.config
        # Run the setup command locally, but the launch command via docker.
        # This is to ensure that the launch environment matches the worker environment.
        processor = platform.processor().lower()
        if "arm" in processor:
            # Disable running from docker on Mac M1 chip due to qemu core dump bug.
            # https://github.com/docker/for-mac/issues/5342.
            logging.info(
                (
                    "%s processor detected. "
                    "Skipping docker launch and running from local environment instead."
                ),
                processor,
            )
            cmd = cfg.command
        else:
            cmd = (
                "docker run --rm "
                "--mount type=bind,src=$HOME/.config/gcloud,dst=/root/.config/gcloud "
                "--entrypoint /bin/bash "
                f"{self._bundler.id(cfg.name)} -c '{cfg.command}'"
            )
        cmd = f"{cfg.setup_command} && {cmd}"
        cmd = f"bash -c {shlex.quote(cmd)}"
        logging.info("Executing in subprocess: %s", cmd)
        with subprocess.Popen(cmd, shell=True, text=True) as proc:
            # Attempt to cleanup the process when exiting.
            def sig_handler(sig: int, _):
                send_signal(proc, sig=sig)

            # SIGTERM for kill, SIGINT for ctrl+c, and SIGHUP for screen quit.
            for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
                signal.signal(sig, sig_handler)

            handle_popen(proc)

    class Status(enum.Enum):
        """Dataflow Job status.

        See also:
        https://cloud.google.com/dataflow/docs/reference/rest/v1b3/projects.jobs#Job.JobState
        """

        JOB_STATE_UNKNOWN = "JOB_STATE_UNKNOWN"
        JOB_STATE_STOPPED = "JOB_STATE_STOPPED"
        JOB_STATE_RUNNING = "JOB_STATE_RUNNING"
        JOB_STATE_DONE = "JOB_STATE_DONE"
        JOB_STATE_FAILED = "JOB_STATE_FAILED"
        JOB_STATE_CANCELLED = "JOB_STATE_CANCELLED"
        JOB_STATE_UPDATED = "JOB_STATE_UPDATED"
        JOB_STATE_DRAINING = "JOB_STATE_DRAINING"
        JOB_STATE_DRAINED = "JOB_STATE_DRAINED"
        JOB_STATE_PENDING = "JOB_STATE_PENDING"
        JOB_STATE_CANCELLING = "JOB_STATE_CANCELLING"
        JOB_STATE_QUEUED = "JOB_STATE_QUEUED"
        JOB_STATE_RESOURCE_CLEANING_UP = "JOB_STATE_RESOURCE_CLEANING_UP"
        JOB_STATE_NOT_STARTED = "JOB_STATE_NOT_STARTED"

    def _get_status(self) -> Status:
        cfg: DataflowRunnerJob.Config = self.config
        jobs = _get_dataflow_jobs(project=cfg.project, zone=cfg.zone, job_name=cfg.name)
        if len(jobs) == 0:
            return DataflowRunnerJob.Status.JOB_STATE_NOT_STARTED
        elif len(jobs) == 1:
            # TODO: Define behavior when there are multiple jobs with the same name
            job = jobs[0]
            return DataflowRunnerJob.Status[job.get("currentState")]
        return DataflowRunnerJob.Status.JOB_STATE_UNKNOWN

    def _delete(self):
        self._inner._delete()  # pylint: disable=protected-access
        cfg: DataflowRunnerJob.Config = self.config
        _stop_dataflow_job(project=cfg.project, zone=cfg.zone, job_name=cfg.name)

    def _execute(self):
        cfg: DataflowRunnerJob.Config = self.config

        # Keep track of last status to prevent duplicate events.
        last_job_status = None

        while True:
            status = self._get_status()
            logging.info("Task %s has status: %s", cfg.name, status)
            if status == DataflowRunnerJob.Status.JOB_STATE_STOPPED:
                if status != last_job_status:
                    self._maybe_publish(
                        cfg.name, msg="Job not yet started to run", state=JobLifecycleState.STARTING
                    )
                    last_job_status = status
            elif status == DataflowRunnerJob.Status.JOB_STATE_DONE:
                self._maybe_publish(cfg.name, msg="Job succeeds", state=JobLifecycleState.SUCCEEDED)
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return
            elif status == DataflowRunnerJob.Status.JOB_STATE_FAILED:
                self._maybe_publish(
                    cfg.name, msg="Job failed with error", state=JobLifecycleState.FAILED
                )
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return
            elif status == DataflowRunnerJob.Status.JOB_STATE_CANCELLED:
                self._maybe_publish(
                    cfg.name, msg="Job cancelled", state=JobLifecycleState.SUCCEEDED
                )
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return
            elif status == DataflowRunnerJob.Status.JOB_STATE_UPDATED:
                # TODO: Define how to handle this case
                pass
            elif status == DataflowRunnerJob.Status.JOB_STATE_DRAINING:
                if status != last_job_status:
                    self._maybe_publish(
                        cfg.name, msg="Job draining", state=JobLifecycleState.CANCELLING
                    )
                    last_job_status = status
            elif status == DataflowRunnerJob.Status.JOB_STATE_DRAINED:
                self._maybe_publish(cfg.name, msg="Job drained", state=JobLifecycleState.SUCCEEDED)
                logging.info("Task %s exited with status: %s.", cfg.name, status)
                return
            elif status == DataflowRunnerJob.Status.JOB_STATE_PENDING:
                if status != last_job_status:
                    self._maybe_publish(
                        cfg.name, msg="Job pending", state=JobLifecycleState.STARTING
                    )
                    last_job_status = status
            elif status == DataflowRunnerJob.Status.JOB_STATE_CANCELLING:
                if status != last_job_status:
                    self._maybe_publish(
                        cfg.name, msg="Job cancelling", state=JobLifecycleState.CANCELLING
                    )
                    last_job_status = status
            elif status == DataflowRunnerJob.Status.JOB_STATE_QUEUED:
                if status != last_job_status:
                    self._maybe_publish(cfg.name, msg="Job queued", state=JobLifecycleState.QUEUED)
                    last_job_status = status
            elif status == DataflowRunnerJob.Status.JOB_STATE_NOT_STARTED:
                logging.info("Job %s is starting", cfg.name)
                self._start_dataflow_job()
            else:
                # Only emit events when status changes.
                if (
                    status == DataflowRunnerJob.Status.JOB_STATE_RUNNING
                    and status != last_job_status
                ):
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

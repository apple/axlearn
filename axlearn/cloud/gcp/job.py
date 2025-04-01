# Copyright © 2023 Apple Inc.

"""Utilities for executing commands on GCP.

See also ``On configuration`` in `axlearn/cloud/gcp/job.py`.
"""

import logging
import shlex
import subprocess
from collections.abc import Sequence
from typing import Any, Optional

import kubernetes as k8s
from absl import flags
from google.auth.credentials import Credentials

from axlearn.cloud.common.bundler import BaseDockerBundler
from axlearn.cloud.common.job import Job
from axlearn.cloud.common.utils import subprocess_run
from axlearn.cloud.gcp.config import default_env_id, default_project, default_zone, gcp_settings
from axlearn.cloud.gcp.jobset_utils import A3ReplicatedJob, BaseReplicatedJob, TPUReplicatedJob
from axlearn.cloud.gcp.utils import custom_jobset_kwargs, delete_k8s_jobset, get_credentials
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested


class GCPJob(Job):
    """Base GCP Job definition."""

    @config_class
    class Config(Job.Config):
        """Configures GCPJob."""

        # GCP project.
        project: Required[str] = REQUIRED
        # GCP zone.
        zone: Required[str] = REQUIRED
        # GCP env_id.
        env_id: Optional[str] = None
        # If not none, the current job will be executed as the service account.
        service_account: Optional[str] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("project", None, "The GCP project name.", **common_kwargs)
        flags.DEFINE_string("zone", None, "The GCP zone name.", **common_kwargs)
        flags.DEFINE_string(
            "env_id",
            None,
            "The env_id, used along with project to identify `gcp_settings`.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "service_account",
            None,
            "If specified, will run job as the service account. "
            "Otherwise will fallback to application-default credentials.",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        fv.set_default("project", default_project())
        fv.set_default("zone", default_zone())
        fv.set_default("env_id", default_env_id())

    # TODO(markblee): Remove this from GCPJob.
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


# TODO(markblee): Rename to GKEJobSet.
class GKEJob(GCPJob):
    """Base GKE JobSet interface."""

    # Flags defined in `builder.define_flags` will be considered part of this class' flag API.
    builder: type[BaseReplicatedJob]

    @config_class
    class Config(GCPJob.Config):
        """Configures GKEJob.

        Attributes:
            builder: A builder that returns one or more replicated job specs.
            namespace: The namespace to use within the k8s cluster.
                https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
            queue: The Kueue LocalQueue to use. If not set, no queue is used.
        """

        builder: Required[BaseReplicatedJob.Config] = REQUIRED
        namespace: str = "default"
        queue: Optional[str] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string(
            "queue",
            None,
            "The name of the Kueue LocalQueue to use. If not set, no queue is used.",
            **common_kwargs,
        )
        cls.builder.define_flags(fv)

    @classmethod
    def set_defaults(cls, fv):
        super().set_defaults(fv)
        fv.set_default("max_tries", fv.max_tries or 10)
        fv.set_default("retry_interval", fv.retry_interval or 60)
        fv.set_default(
            "service_account", gcp_settings("k8s_service_account", default="default", fv=fv)
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        cfg: GKEJob.Config = super().from_flags(fv, **kwargs)
        # The command is not used at the jobset, but at the builder(s).
        cfg.command = None
        cfg.builder = cls.builder.from_flags(fv, **kwargs)
        return cfg

    def __init__(self, cfg: Config, *, bundler: BaseDockerBundler):
        super().__init__(cfg)
        cfg: GKEJob.Config = self.config
        if cfg.bundler is not None:
            raise ValueError("Pass instantiated bundler directly.")
        self._bundler = bundler
        # This instantiatees a builder for constructing replicated job specs, which will be managed
        # together under the jobset represented by this class.
        # Note the distinction from bundlers, which are responsible for bundling any code assets
        # required to run the job.
        self._builder: BaseReplicatedJob = cfg.builder.instantiate(bundler=bundler)

    def _delete(self):
        cfg: GKEJob.Config = self.config
        # Issues a delete request for the JobSet and proactively delete its descendants. This is not
        # fully blocking; after the call returns there can be a delay before everything is deleted.
        delete_k8s_jobset(cfg.name, namespace=cfg.namespace)

    def _build_jobset(self) -> Nested[Any]:
        """Builds a config for a JobSet, which is a set of Jobs.

        https://github.com/kubernetes-sigs/jobset/blob/d49514bee57da8ac9aec2fcea06c3a13c21afeae/docs/concepts/README.md

        Returns:
            A nested dict corresponding to a k8s JobSet config.
        """
        cfg: GKEJob.Config = self.config
        annotations = {}
        if cfg.queue:
            annotations["kueue.x-k8s.io/queue-name"] = cfg.queue
        return dict(
            metadata=dict(name=cfg.name, annotations=annotations),
            spec=dict(
                failurePolicy=dict(maxRestarts=cfg.max_tries - 1),
                replicatedJobs=self._builder(),
            ),
        )

    def _execute(self) -> Any:
        """Submits a JobSet to the cluster."""
        cfg: GKEJob.Config = self.config
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


class TPUGKEJob(GKEJob):
    """A TPU job represented as a k8s JobSet.

    See also `gke_runner` as an example.
    """

    builder = TPUReplicatedJob
    Config = GKEJob.Config

    def _build_jobset(self):
        jobset = super()._build_jobset()
        jobset["metadata"]["annotations"].update(
            {
                # The exclusive topology annotation will ensure that all Pods will have affinity
                # rules added that will ensure that they are fully scheduled on the same
                # pod-slice node-pools.
                "alpha.jobset.sigs.k8s.io/exclusive-topology": "cloud.google.com/gke-nodepool",
            }
        )
        return jobset


class GPUGKEJob(GKEJob):
    """A GPU job represented as a k8s JobSet.

    See also `gke_runner` as an example.
    """

    builder = A3ReplicatedJob
    Config = GKEJob.Config


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

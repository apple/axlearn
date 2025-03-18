# Copyright © 2023 Apple Inc.

"""Launch commands on Google Cloud.

The launch command provides a few benefits:
1. Provides a uniform API and entrypoint for interacting with different instance types.
2. Provides utilities for registering custom launcher implementations.

The launch flow depends on the launcher being used. Each launcher must define a "matcher" function
that decides, for a given CLI action (e.g. 'start') and instance type (e.g. 'tpu-v4-8'), whether the
launcher can be used. See `_LAUNCHERS` for a full list, and `BastionManagedTPUJob` for an example.

Possible actions: [start|update|stop|list]

    Start: submits a job to the queue.
    Update: updates a job without resubmission.
    Stop: stops the job or removes a job from the queue.
    List: lists jobs and their statuses.

Additional flags may be supplied based on --instance_type. Run with --help to see all flags.
Commands are parsed from the positional arguments (so anything after a trailing `--`).

Examples:

    # View all launch flags for the given instance type.
    axlearn gcp launch --instance_type=tpu-v4-8 --help

    # Simple TPU launch. Internally submits `axlearn gcp tpu ...` to bastion to run, instead of
    # running locally.
    axlearn gcp launch --instance_type=tpu-v4-32 -- python3 my_script.py

    # Launch with extra dependencies.
    axlearn gcp launch --instance_type=tpu-v4-32 --bundler_spec=extras=dev -- python3 my_script.py

    # Dry-run: prints the job config without launching.
    axlearn gcp launch --instance_type=tpu-v4-32 --dry_run -- python3 my_script.py

    # Launch with docker.
    axlearn gcp launch --instance_type=tpu-v4-8 \
        --bundler_type=artifactregistry
        --bundler_spec=repo=my-repo \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=build_arg1=my-build-arg ...

    # Update an existing job without resubmission.
    axlearn gcp launch update --instance_type=tpu-v4-32 ... -- python3 my_script2.py

    # To stop a job.
    axlearn gcp launch stop --name=... --instance_type=tpu

More on the Update command:

    The update command allows updating bundles and job command of an existing job
    without resubmission. It currently only works with axlearn.cloud.gcp.jobs.gke_runner.

    Resource related flags including instance_type, num_replicas and enable_pre_provisioner
    are not allowed to change.

    When bundles are updated before the job update, job will run with new bundles.
    If bundle update is not desired, use `--bundler_spec=skip_bundle=True` flag
    to skip bundle update.

    To be able to update the job without re-provisioning the resources (e.g. TPU node pools),
    use `--enable_pre_provisioner` to submit the job.

"""
# pylint: disable=redefined-outer-name,protected-access

import functools
import os
import shlex
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple, Optional, Protocol, TextIO

from absl import app, flags, logging

from axlearn.cloud.common.bastion import BastionDirectory
from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import new_jobspec, serialize_jobspec
from axlearn.cloud.common.quota import QUOTA_CONFIG_PATH, get_user_projects
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.common.types import JobSpec, ResourceMap
from axlearn.cloud.common.utils import (
    configure_logging,
    generate_job_id,
    generate_job_name,
    infer_cli_name,
    parse_action,
)
from axlearn.cloud.gcp.bundler import CloudBuildBundler
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import Job
from axlearn.cloud.gcp.jobs import gke_runner, tpu_runner
from axlearn.cloud.gcp.jobs.bastion_vm import bastion_root_dir, shared_bastion_name
from axlearn.cloud.gcp.jobs.launch_utils import (
    JobsToTableFn,
    jobs_table,
    match_by_regex,
    project_usage_table,
    serialized_flags_for_job,
    user_usage_table,
    validate_resource_flags,
    with_k8s_jobset_state,
    with_qrm_tpu_state,
)
from axlearn.cloud.gcp.tpu import infer_tpu_resources, infer_tpu_type, infer_tpu_workers
from axlearn.cloud.gcp.utils import (
    GCPAPI,
    catch_auth,
    get_credentials,
    load_kube_config,
    validate_k8s_name,
    validate_resource_name,
)
from axlearn.cloud.gcp.vm import _compute_resource, get_vm_node
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)

FLAGS = flags.FLAGS


def _get_bastion_vm(bastion_name: str) -> Optional[dict[str, Any]]:
    return get_vm_node(bastion_name, _compute_resource(get_credentials()))


class _Matcher(Protocol):
    def __call__(self, *, action: str, instance_type: str, gcp_api: str) -> bool:
        pass


class Launcher(NamedTuple):
    """A job launcher.

    Consists of:
    * job_cls:
        A Job class which defines, at minimum, `_execute()`, `_list()` and `_delete()` methods,
        invoked for "start", "list", "stop" actions respectively. It can optionally define a
        `define_flags()` classmethod, which can be used for registering launch flags. These flags
        will automatically be printed when invoking with --help, and will automatically be provided
        to `Job.from_flags` when instantiating the Job.
    * matcher:
        A `_Matcher` (or a config instantiating thereof), used to decide whether the launcher is
        applicable for a given action and instance type.
    * description:
        Human-readable description of the launcher. Printed e.g. if no launchers are matched.
    """

    # We take a class here instead of a config, since a materialized config is often not available
    # at the time of registry. Instead, we often construct the config via `job_cls.from_flags`.
    job_cls: type[Job]
    # A config is usually more print-friendly, but not strictly required.
    matcher: ConfigOr[_Matcher]
    description: str


class BaseBastionManagedJob(Job):
    """A base job definition for jobs managed by a bastion.

    It provides functionality to submit, delete, and list bastion jobs, but is agnostic to specific
    resource types. At minimum, subclasses should override `runner` and `_resources()` to specify
    the implementation of the job executed by the bastion, as well as resources used by the job,
    which will be used for quota management and scheduling.

    See `BastionManagedTPUJob` as an example.
    """

    # Runner class, a subclass of Job that runs locally on the bastion.
    # Used to infer launch flags and launch command.
    runner: type[Job]

    @config_class
    class Config(Job.Config):
        """Configures BaseBastionManagedJob."""

        # Used along with project to identify `gcp_settings`.
        env_id: Optional[str] = None
        # Where to run the remote job.
        zone: Required[str] = REQUIRED
        # Instance type to launch.
        instance_type: Required[str] = REQUIRED
        # Bastion name.
        bastion_name: Required[str] = REQUIRED
        # Bastion dir.
        bastion_dir: BastionDirectory.Config = BastionDirectory.default_config()
        # User ID for bastion quota and scheduling.
        user_id: Required[str] = REQUIRED
        # Project ID for bastion quota and scheduling.
        project_id: Required[str] = REQUIRED
        # Job priority.
        priority: Required[int] = REQUIRED
        # Output directory for job logs.
        output_dir: Required[str] = REQUIRED
        # Runner executed by the bastion.
        runner: Required[Job.Config] = REQUIRED
        # One or more functions for displaying `list` information. Each is invoked with the bastion
        # jobs and is expected to return a printable table.
        output_tables: Sequence[ConfigOr[JobsToTableFn]] = [
            jobs_table,
            user_usage_table,
            project_usage_table,
        ]
        # Resources used by the job.
        resources: ConfigOr[ResourceMap[int]] = {}

    @classmethod
    def with_runner(cls, runner: type[Job]):
        return type(f"{cls.__name__}_{runner.__name__}", (cls,), {"runner": runner})

    @classmethod
    def validate_runner(cls):
        if cls.runner is None:
            raise ValueError(f"A BastionManagedJob should subclass {cls} and define `runner`.")

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines launch flags on the provided flag_values."""
        cls.validate_runner()
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("bastion", None, "Name of bastion VM to use.", **common_kwargs)
        flags.DEFINE_integer(
            "priority",
            5,
            "Job priority. Smaller means higher priority.",
            **common_kwargs,
        )
        # TODO(markblee): Support configuring an identity provider for --user_id.
        flags.DEFINE_string(
            "user_id",
            os.getenv("USER"),
            "User ID to use for resource attribution.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "project_id",
            None,
            "Quota project ID to use for scheduling.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )
        cls.runner.define_flags(fv)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, *, command: str, action: str, **kwargs) -> Config:
        """Constructs config from flags defined by `define_flags()`.

        Args:
            fv: Flag values (e.g., FLAGS).
            command: The user-supplied command, i.e. everything after `--` as a string.
            action: Action requested by the user.
            kwargs: Optional key/values to set on the config.

        Returns:
            The job config.
        """
        cls.validate_runner()
        cfg: BaseBastionManagedJob.Config = super().from_flags(fv, **kwargs)
        if not cfg.bastion_name:
            cfg.bastion_name = fv.bastion or shared_bastion_name(fv, gcp_api=_gcp_api(fv))
        cfg.bastion_dir.root_dir = bastion_root_dir(cfg.bastion_name, fv=fv)
        # Default output_dir depends on the final value of --name.
        if not cfg.output_dir:
            cfg.output_dir = f"gs://{gcp_settings('ttl_bucket', fv=fv)}/axlearn/jobs/{fv.name}"
        # We use the bundler defined by the runner impl, ensuring that bundling is consistent
        # between local and bastion.
        cfg.bundler = None
        # Construct runner only for start and update.
        if action in ("start", "update"):
            cfg.runner = cls.runner.from_flags(fv, command=command)
            runner_flags = " ".join(serialized_flags_for_job(fv, cls.runner))
            cfg.command = f"python3 -m {cls.runner.__module__} {action} {runner_flags} -- {command}"
            if cfg.runner.bundler and fv.bundler_exclude:
                cfg.runner.bundler.set(exclude=fv.bundler_exclude)
        else:
            cfg.runner = None
            cfg.command = None
        return cfg

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        if not (cfg.instance_type and cfg.output_dir):
            raise ValueError("instance_type, output_dir cannot be empty")
        self._bastion_dir: BastionDirectory = cfg.bastion_dir.instantiate()
        self._runner: Optional[Job] = (
            cfg.runner.set(name="runner").instantiate() if cfg.runner else None
        )
        self._output_tables = maybe_instantiate(cfg.output_tables)

    def _delete(self):
        """Submits a delete request to bastion."""
        self._bastion_dir.cancel_job(self.config.name)

    def _list(self, output_file: Optional[TextIO] = None) -> dict[str, BastionJob]:
        """Lists running jobs and optionally prints them in tabular format.

        Args:
            output_file: Output file. If None, prints to stdout, otherwise writes to the file.
                Subclasses can send outputs to /dev/null if they intend to reformat the output.

        Returns:
            A mapping from job name to bastion job.
        """
        jobs = self._bastion_dir.list_jobs()
        for fn in self._output_tables:
            print(fn(jobs), file=output_file)
        return jobs

    def _execute(self) -> JobSpec:
        """Submits the command to bastion."""
        cfg: BaseBastionManagedJob.Config = self.config

        # Check for group membership if --project_id is provided.
        # TODO(markblee): Make this check at the bastion level.
        if cfg.project_id:
            quota_file = (
                f"gs://{gcp_settings('private_bucket')}/{cfg.bastion_name}/{QUOTA_CONFIG_PATH}"
            )
            user_projects = get_user_projects(quota_file, user_id=cfg.user_id)
            if cfg.project_id.lower() not in user_projects:
                raise ValueError(
                    f"User '{cfg.user_id}' is not a member of the project '{cfg.project_id}'. "
                    f"Instead, user '{cfg.user_id}' is a member of: {user_projects}"
                )

        if self._runner and self._runner.bundler:
            self._runner.bundler.bundle(cfg.name)

        logging.info("Starting run for job name %s", cfg.name)
        logging.info("Command: %s", cfg.command)
        with tempfile.NamedTemporaryFile("w") as f:
            job_id = generate_job_id()
            metadata = JobMetadata(
                user_id=cfg.user_id,
                project_id=cfg.project_id or "none",
                creation_time=datetime.now(timezone.utc),
                resources=maybe_instantiate(cfg.resources),
                priority=cfg.priority,
                job_id=job_id,
            )
            jobspec = new_jobspec(name=cfg.name, command=cfg.command, metadata=metadata)
            serialize_jobspec(
                jobspec,
                f,
            )
            self._bastion_dir.submit_job(cfg.name, job_spec_file=f.name)
        gcp_api = "gke" if "gke" in cfg.bastion_name else "qrm"
        print(
            "\nView bastion outputs with: (if not found, check job and project history)\n"
            f"gsutil cat {os.path.join(self._bastion_dir.logs_dir, cfg.name)}\n"
            f"\nStop/cancel the job with:\n"
            f"{infer_cli_name()} gcp launch stop "
            f"--name={cfg.name} --bastion={cfg.bastion_name} --instance_type={cfg.instance_type} "
            f"--env_id={cfg.env_id} --gcp_api={gcp_api}\n"
            "\nCheck job history with:\n"
            f"{infer_cli_name()} gcp bastion history "
            f"--name={cfg.bastion_name} --env_id={cfg.env_id} "
            f"--job_name={cfg.name}"
            "\nCheck project history with:\n"
            f"{infer_cli_name()} gcp bastion history "
            f"--name={cfg.bastion_name} --env_id={cfg.env_id} "
            f"{cfg.project_id or ''}"
        )
        return jobspec

    def _update(self) -> JobSpec:
        """Update an existing job without resubmission.

        This will fetch the existing job from Bastion, change
        the trainer command, increment the version in metadata, and then update the job on Bastion.

        The resource related flags including instance_type, num_replicas and enable_pre_provisioner
        are not allowed to change.
        """
        cfg: BaseBastionManagedJob.Config = self.config

        # Get current job spec.
        job_spec = self._bastion_dir.get_job(job_name=cfg.name)

        if self._runner and self._runner.bundler:
            self._runner.bundler.bundle(cfg.name)

        logging.info("Starting update for job name %s", cfg.name)
        logging.info("Command: %s", cfg.command)

        # Update the job version.
        job_version = job_spec.metadata.version or 0
        job_spec.metadata.version = job_version + 1

        # The resource related flags are not allowed to change.
        validate_resource_flags(job_spec.command, cfg.command)

        job_spec.command = cfg.command

        logging.info("Updated jobspec: %s", job_spec)

        jobspec = self._bastion_dir.update_job(cfg.name, job_spec=job_spec)

        return jobspec


# TODO(markblee): Add a BastionManagedCPUJob.
class BastionManagedTPUJob(BaseBastionManagedJob):
    """Launches a TPU job via bastion."""

    runner = tpu_runner.TPURunnerJob

    @config_class
    class Config(BaseBastionManagedJob.Config):
        """Configures BastionManagedTPUJob.

        Attributes:
            num_replicas: Number of TPU slices.
        """

        num_replicas: int = 1

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines launch flags using tpu_runner."""
        super().define_flags(fv)
        fv.set_default("name", generate_job_name())

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, *, command: str, action: str, **kwargs) -> Config:
        cfg = super().from_flags(fv, command=command, action=action, **kwargs)
        cfg.output_tables = [with_qrm_tpu_state(jobs_table), user_usage_table, project_usage_table]
        cfg.resources = config_for_function(infer_tpu_resources).set(
            instance_type=fv.instance_type,
            num_replicas=fv.num_replicas,
        )
        return cfg

    def _execute(self) -> JobSpec:
        """Submits the command to bastion.

        In addition to logic defined in `BaseBastionManagedJob._execute()`, also emits the output
        logs for each TPU worker.
        """
        cfg: BastionManagedTPUJob.Config = self.config

        bastion_node = _get_bastion_vm(cfg.bastion_name)
        if bastion_node is None or bastion_node.get("status", None) != "RUNNING":
            logging.warning(
                "Bastion %s does not appear to be running yet. "
                "It will need to be running before jobs will execute.",
                cfg.bastion_name,
            )

        # Job name has a suffix "-{slice_index}" for multi-slice.
        validate_resource_name(
            cfg.name if cfg.num_replicas == 1 else f"{cfg.name}-{cfg.num_replicas}"
        )

        job_spec = super()._execute()
        num_workers = infer_tpu_workers(infer_tpu_type(cfg.instance_type))
        worker_log = f'gsutil cat "{cfg.output_dir}/output/*-0/run.log"'
        print(
            "\nNote that the job may take a few minutes to start. "
            f"Once started, view TPU log outputs with:\n{worker_log}\n"
            f"Replace `*-0` with `*-{{idx}}` where idx is between [0, {num_workers}).\n"
        )
        return job_spec


class BastionManagedGKEJob(BaseBastionManagedJob):
    """A GKE job managed by bastion."""

    @config_class
    class Config(BaseBastionManagedJob.Config):
        """Configures BastionManagedGKEJob.

        Attributes:
            namespace: K8s namespace.
            project: Used for load_kube_config.
            zone: Used to infer total quota.
            cluster: K8s cluster.
            num_replicas: Number of replicas.
        """

        namespace: str = "default"
        project: Required[str] = REQUIRED
        zone: Required[str] = REQUIRED
        cluster: Required[str] = REQUIRED
        num_replicas: int = 1

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines launch flags using tpu_runner."""
        super().define_flags(fv)
        fv.set_default("name", generate_job_name())
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("namespace", "default", "K8s namespace.", **common_kwargs)
        flags.DEFINE_string("cluster", None, "K8s cluster.", **common_kwargs)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, *, command: str, action: str, **kwargs) -> Config:
        # Set default docker flags. These will automatically propagate to the runner on the bastion.
        if action in ("start", "update"):
            fv.set_default("bundler_type", CloudBuildBundler.TYPE)
        cfg: BastionManagedGKEJob.Config = super().from_flags(
            fv, command=command, action=action, **kwargs
        )
        cfg.cluster = cfg.cluster or gcp_settings("gke_cluster", required=False, fv=fv)
        cfg.output_tables = [
            with_k8s_jobset_state(jobs_table, namespace=cfg.namespace),
            user_usage_table,
            project_usage_table,
        ]
        # Use config_for_function to delay instantiation unless needed, e.g., it may not be
        # necessary for list action.
        cfg.resources = config_for_function(infer_tpu_resources).set(
            instance_type=fv.instance_type,
            num_replicas=fv.num_replicas,
        )
        return cfg

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Ensure that the user has installed GKE plugin.
        load_kube_config(project=cfg.project, zone=cfg.zone, cluster=cfg.cluster)

    def _execute(self) -> JobSpec:
        """Submits the command to bastion."""
        cfg: BastionManagedGKEJob.Config = self.config
        try:
            num_workers = infer_tpu_workers(infer_tpu_type(cfg.instance_type))
        except ValueError:
            logging.warning(
                "Failed to infer number of workers for instance_type: %s.", cfg.instance_type
            )
            num_workers = None
        if num_workers is not None:
            validate_k8s_name(cfg.name, num_workers=num_workers, num_replicas=cfg.num_replicas)
            # TODO(markblee): add the logs command.
            worker_log = f"{infer_cli_name()} gcp logs --name={cfg.name} --worker=0"
            print(
                f"\nOnce started, view TPU log outputs with:\n{worker_log}\n"
                "Replace `--worker=0` with `--worker={idx}` "
                f"where idx is between [0, {num_workers}).\n"
            )
        job_spec = super()._execute()
        print(
            "\nView running pods with:\nkubectl get pods\n"
            "\nNote that the job may take a few minutes to start."
        )
        return job_spec


# Launchers specified here will be tried (in the given order) when launching a given instance type.
_LAUNCHERS = [
    # TPU QRM launcher.
    Launcher(
        job_cls=BastionManagedTPUJob,
        matcher=config_for_function(match_by_regex).set(
            match_regex=dict(start=r"tpu-v.+-(\d)+", list=r"tpu.*", stop=r"tpu.*"),
            gcp_api=GCPAPI.QRM.value,
        ),
        description=(
            "Supports launching TPU jobs via QRM. "
            "For 'start', provide --gcp_api=qrm, as well as the full TPU version, "
            "e.g. --instance_type=tpu-v4-8. "
            "For 'list' or 'stop', provide --gcp_api=qrm, as well as the accelerator type, "
            "e.g. --instance_type=tpu."
        ),
    ),
    # TPU GKE launcher.
    Launcher(
        job_cls=BastionManagedGKEJob.with_runner(gke_runner.TPUGKERunnerJob),
        matcher=config_for_function(match_by_regex).set(
            match_regex=dict(start=r"tpu-v.+-(\d)+", update=r"tpu.*", list=r"tpu.*", stop=r"tpu.*"),
            gcp_api=GCPAPI.GKE.value,
        ),
        description=(
            "Supports launching TPU jobs via GKE. "
            "For 'start' or 'update', provide --gcp_api=gke, as well as the full instance type, "
            "e.g. --instance_type=tpu-v4-8. "
            "For 'list' or 'stop', provide --gcp_api=gke as well as the accelerator type, "
            "e.g. --instance_type=tpu."
        ),
    ),
]


def _get_launcher_or_exit(*, action: str, instance_type: str, gcp_api: str) -> Launcher:
    """Retrieves launcher by matching instance_type and gcp_api.

    If there are multiple matches, the first one in the registry is returned.
    """
    # Identify launcher from instance type.
    for launcher in _LAUNCHERS:
        m = maybe_instantiate(launcher.matcher)
        if m(action=action, instance_type=instance_type, gcp_api=gcp_api):
            return launcher

    launchers = "\n".join(
        [
            f"Job: {launcher.job_cls.__name__}\n"
            f"Description: {launcher.description}\n"
            f"Matcher: {launcher.matcher}\n"
            for launcher in _LAUNCHERS
        ]
    )
    raise app.UsageError(
        f"Don't know how to launch {instance_type=} for {action=} and {gcp_api=}.\n"
        f"The registered launchers are:\n\n{launchers}"
    )


def _gcp_api(fv: flags.FlagValues = FLAGS) -> str:
    if getattr(fv, "gcp_api", None) is not None:
        return fv.gcp_api.lower()
    # The return value depends on --zone, so cannot be set as the default value of fv.gcp_api.
    return gcp_settings(
        "launch_gcp_api", default=GCPAPI.QRM.lower(), required=False, fv=fv
    )  # pytype: disable=bad-return-type


@catch_auth
def main(_):
    if FLAGS.instance_type is None:
        raise app.UsageError("--instance_type is required.")

    action = parse_action(sys.argv, options=["start", "stop", "update", "list"], default="start")
    launcher = _get_launcher_or_exit(
        action=action,
        instance_type=FLAGS.instance_type,
        gcp_api=_gcp_api(),
    )

    # Parse the command from argv. Note that argv may or may not contain action, so we explicitly
    # look for '--' and extract all args after it. Use sys.argv instead of argv from params, since
    # the param argv has '--' stripped.
    command = ""
    for i, arg in enumerate(sys.argv):
        if arg.strip() == "--":
            command = shlex.join(sys.argv[i + 1 :])
            break
    cfg = launcher.job_cls.from_flags(FLAGS, command=command, action=action)
    logging.info("Launcher config:\n%s", cfg)
    job: BaseBastionManagedJob = cfg.instantiate()
    if FLAGS.dry_run:
        print(f"Action: {action}\nJob config:\n{job.config}")
        return

    if action == "start":
        job._execute()
    elif action == "list":
        job._list()
    elif action == "stop":
        job._delete()
    elif action == "update":
        job._update()
    else:
        raise app.UsageError(f"Unsupported action {action}")


def _prelaunch_flags(fv: flags.FlagValues = FLAGS):
    """Flags necessary for `_get_launcher_or_exit`."""
    flags.DEFINE_string("instance_type", None, "Instance type to launch.", flag_values=fv)
    # pytype: disable=missing-parameter
    flags.DEFINE_enum(
        "gcp_api",
        None,
        [v for gcp_api in GCPAPI for v in [gcp_api.upper(), gcp_api.lower()]],
        help="GCP API.",
        flag_values=fv,
    )
    # pytype: enable=missing-parameter
    flags.DEFINE_bool(
        "dry_run", False, "Output job config and exit without running.", flag_values=fv
    )


def _private_flags():
    """Defines all launch flags, and amends `app.usage` with additional launch help info."""

    _prelaunch_flags()
    FLAGS(sys.argv, known_only=True)

    launch_help = None
    # Allow instance_type to be None when running --help without any flags. On the other hand, if
    # instance_type is provided when running --help, we show additional help info.
    if FLAGS.instance_type:
        action = parse_action(
            sys.argv, options=["start", "update", "stop", "list"], default="start"
        )
        launcher = _get_launcher_or_exit(
            action=action,
            instance_type=FLAGS.instance_type,
            gcp_api=_gcp_api(),
        )
        orig_flags = FLAGS.flag_values_dict()
        launcher.job_cls.define_flags(FLAGS)
        output_lines = []
        FLAGS._render_flag_list(
            [FLAGS[name] for name in FLAGS if name not in orig_flags], output_lines
        )
        launch_help = "\n".join(output_lines)

    app.usage = functools.partial(_wrapped_usage, usage=app.usage, launch_help=launch_help)


def _wrapped_usage(
    *,
    usage: Callable,
    launch_help: Optional[str],
    writeto_stdout: bool = False,
    **kwargs,
):
    """Wraps original usage by printing additional launch-specific help."""
    usage(writeto_stdout=writeto_stdout, **kwargs)
    of = sys.stdout if writeto_stdout else sys.stderr
    if launch_help:
        of.write(
            f"\nFlags for the provided --instance_type={FLAGS.instance_type}:\n{launch_help}\n"
        )
    else:
        of.write("\nPass --instance_type to see additional flags.\n")


if __name__ == "__main__":
    configure_logging(logging.INFO)
    _private_flags()
    app.run(main)

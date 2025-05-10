# Copyright © 2023 Apple Inc.

"""Launch commands on Google Cloud.

The launch command provides a few benefits:
1. Provides a uniform API and entrypoint for interacting with different instance types.
2. Provides utilities for registering custom launcher implementations.

The launch flow depends on the launcher being used. Each launcher must define a "matcher" function
that decides, for a given CLI action (e.g. 'start'), whether the launcher can be used. Run with
`--help` to display the list of launchers.

Further, the launch command supports specifying the `runner` implementation to use for deploying and
monitoring the job on the bastion. The API follows a similar API as training, where
`--runner_module` points to a module exposing `named_runner_configs`, and `--runner_name` specifies
a specific runner to be used. Run with `--help` to display a list of runners, or with
`--runner_name` and `--help` to see all possible flags for a specific runner.

Some additional background on runners can be found in the infrastructure documentation:
https://github.com/apple/axlearn/blob/main/docs/04-infrastructure.md#bastion-runners

Possible actions: [start|update|stop|list|run]

    Start: submits a job to the queue.
    Update: updates a job without resubmission.
    Stop: stops the job or removes a job from the queue.
    List: lists jobs and their statuses.
    Run: run the job locally for debugging.

Additional flags may be supplied depending on the launcher. Run with --help to see all flags.
Commands are either parsed from the positional arguments (so anything after a trailing `--`), or
from flags, depending on the runner being used.

Examples:

    # View possible runners.
    axlearn gcp launch --help

    # View all launch flags for the given instance type.
    # This internally attempts to infer the --runner_name automatically.
    axlearn gcp launch --instance_type=tpu-v4-8 --help

    # View all launch flags for a specific runner.
    axlearn gcp launch --runner_name=gke_tpu_single --help

    # Simple TPU launch, running locally.
    axlearn gcp launch run --instance_type=tpu-v4-8 \
        --bundler_spec=image=tpu --bundler_spec=dockerfile=Dockerfile -- python3 my_script.py

    # Simple TPU launch, submitting to bastion to run instead of locally.
    axlearn gcp launch --instance_type=tpu-v4-32 \
        --bundler_spec=image=tpu --bundler_spec=dockerfile=Dockerfile -- python3 my_script.py

    # Simple TPU launch, targeting a specific runner.
    # By default, this users `axlearn.cloud.gcp.runners` as the runner module.
    axlearn gcp launch --runner_name=gke_tpu_single --instance_type=tpu-v4-32 \
        --bundler_spec=image=tpu --bundler_spec=dockerfile=Dockerfile -- python3 my_script.py

    # Simple TPU launch, targeting a specific runner in a custom module.
    # In this case, the available flags depend on the custom runner implementation.
    axlearn gcp launch --runner_module=my.custom.module --runner_name=my_custom_runner \
        --instance_type=my-custom-instance ... -- python3 my_script.py

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

    # To list running jobs.
    axlearn gcp launch list

    # To stop a job.
    axlearn gcp launch stop --name=...


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

import enum
import functools
import os
import shlex
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import Callable, NamedTuple, Optional, TextIO

from absl import app, flags, logging

from axlearn.cloud.common.bastion import BastionDirectory
from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import new_jobspec, serialize_jobspec
from axlearn.cloud.common.bundler import Bundler, bundler_flags, get_bundler_config
from axlearn.cloud.common.quota import QUOTA_CONFIG_DIR, QUOTA_CONFIG_FILE, get_user_projects
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.common.types import JobSpec, ResourceMap
from axlearn.cloud.common.utils import (
    FlagConfigurable,
    configure_logging,
    define_flags,
    from_flags,
    generate_job_id,
    generate_job_name,
    infer_cli_name,
    infer_resources,
    parse_action,
)
from axlearn.cloud.gcp import runners
from axlearn.cloud.gcp.bundler import CloudBuildBundler
from axlearn.cloud.gcp.config import default_env_id, default_project, default_zone, gcp_settings
from axlearn.cloud.gcp.jobs.bastion_vm import bastion_root_dir, infer_bastion_name
from axlearn.cloud.gcp.jobs.launch_utils import (
    JobsToTableFn,
    Matcher,
    infer_module_qualname,
    jobs_table,
    match_gcp_api,
    project_usage_table,
    user_usage_table,
    validate_resource_flags,
    with_k8s_jobset_state,
)
from axlearn.cloud.gcp.runners.base import BaseRunnerJob
from axlearn.cloud.gcp.utils import GCPAPI, catch_auth, load_kube_config, running_from_vm
from axlearn.common.config import (
    REQUIRED,
    ConfigBase,
    ConfigOr,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)

FLAGS = flags.FLAGS
_ACTIONS = ["start", "stop", "update", "list", "run"]
_RUNNER_ACTIONS = ["start", "update", "run"]  # Actions which require a runner.


class _RegistryMember(NamedTuple):
    """A member of a class registry.

    Consists of:
    * cfg: A `FlagConfigurable` config.
    * matcher: A `_Matcher` (or a config instantiating thereof), used to decide whether the class
        is applicable for a given action and flag values.
    * description: Human-readable description of the job. Printed e.g. if no classes are matched.
    """

    # A partially defined config.
    config: FlagConfigurable.Config
    # A config is usually more print-friendly, but not strictly required.
    matcher: ConfigOr[Matcher]
    description: str


# A sequence of registry members, to be matched in the given order. The first match is used.
_Registry = Sequence[_RegistryMember]


# TODO(markblee): Remove GCP-specific logic and move to a common launch file.
class BaseBastionManagedJob(FlagConfigurable):
    """A base job definition for jobs managed by a bastion.

    It provides functionality to submit, delete, and list bastion jobs, but is agnostic to specific
    resource types. At minimum, subclasses should override `runner` configure `resources` used by
    the job, which will be used for quota management and scheduling.

    See `BastionManagedGKEJob` as an example.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures BaseBastionManagedJob."""

        # Name of the job. It's used for bastion job management as well as the name of the runner.
        name: Required[str] = REQUIRED
        # Command to submit to the bastion.
        command: Optional[str] = None
        # Used along with project to identify `gcp_settings`.
        # TODO(markblee): Move to where project/zone are defined and use `common_flags`.
        env_id: Optional[str] = None
        # Where to run the remote job.
        zone: Required[str] = REQUIRED
        # Bastion name. TODO(markblee): Move to bastion_dir and infer the root dir.
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
        # Bundler config.
        bundler: Optional[Bundler.Config] = None
        # Runner executed by the bastion.
        runner: Optional[BaseRunnerJob.Config] = None
        # One or more functions for displaying `list` information. Each is invoked with the bastion
        # jobs and is expected to return a printable table.
        output_tables: Sequence[ConfigOr[JobsToTableFn]] = [
            jobs_table,
            user_usage_table,
            project_usage_table,
        ]
        # Resources used by the job.
        resources: Callable[[ConfigBase], ResourceMap[int]] = infer_resources
        # If True, wait for a job to stop when cancelling.
        # Default to None for backwards compatibility.
        wait_for_stop: Optional[bool] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines launch flags on the provided flag_values."""
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        bundler_flags(required=False, **common_kwargs)
        flags.DEFINE_string("name", None, "Name of the job.", **common_kwargs)
        flags.DEFINE_string(
            "env_id",
            None,
            "The env_id, used along with project to identify `gcp_settings`.",
            **common_kwargs,
        )
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
        flags.DEFINE_bool(
            "wait_for_stop",
            None,
            "If True, wait for a job to stop when cancelling.",
            **common_kwargs,
        )

    @classmethod
    def set_defaults(cls, fv):
        super().set_defaults(fv)
        # Don't override `name` if already specified, since the default is non-deterministic.
        fv.set_default("name", fv.name or generate_job_name())
        fv.set_default("env_id", default_env_id())

    @classmethod
    def from_flags(
        cls, fv: flags.FlagValues, *, action: str, command: Optional[str] = None, **kwargs
    ) -> Config:
        """Constructs config from flags defined by `define_flags()`.

        Args:
            fv: Flag values (e.g., FLAGS).
            action: Action requested by the user.
            command: The user-supplied command, i.e. everything after `--` as a string.
                If None, command should be specified via flags (e.g., in the case of multiple
                subcommands).
            kwargs: Optional key/values to set on the config.

        Returns:
            The job config.
        """
        cfg: BaseBastionManagedJob.Config = super().from_flags(fv, **kwargs)
        if not cfg.bastion_name:
            cfg.bastion_name = fv.bastion or infer_bastion_name(fv)
        cfg.bastion_dir.root_dir = bastion_root_dir(cfg.bastion_name, fv=fv)
        # Default output_dir depends on the final value of --name.
        if not cfg.output_dir:
            cfg.output_dir = f"gs://{gcp_settings('ttl_bucket', fv=fv)}/axlearn/jobs/{fv.name}"
        # Construct runner only for start and update.
        if action in _RUNNER_ACTIONS:
            # We construct a bundler and propagate to runner during instantiate, ensuring the
            # bundling is consistent between local and bastion.
            cfg.bundler = get_bundler_config(
                bundler_type=fv.bundler_type, spec=fv.bundler_spec, fv=fv
            )
            # Build launch command. We take the same flags provided to this module and run the
            # command again, this time on the bastion with action="run".
            # For backwards compatibility with legacy behavior where command is specified with argv
            # after `--`, we take any command supplied via `kwargs` and use argv.
            # TODO(markblee): Handle quoting for flag commands that contain spaces.
            own_flags = " ".join(fv.flags_into_string().split("\n"))
            cfg.command = f"python3 -m {infer_module_qualname(cls)} run {own_flags}"
            if command is not None:
                cfg.command = f"{cfg.command} -- {command}"
        else:
            cfg.command = None
        return cfg

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        if not cfg.output_dir:
            raise ValueError("output_dir cannot be empty")
        self._bastion_dir: BastionDirectory = cfg.bastion_dir.instantiate()
        self._bundler = self._runner = None
        if cfg.command is not None:
            self._bundler: Bundler = cfg.bundler.instantiate()
            self._runner: BaseRunnerJob = cfg.runner.instantiate(bundler=self._bundler)
        self._output_tables = maybe_instantiate(cfg.output_tables)

    def cancel(self):
        """Submits a cancel request to bastion."""
        cfg = self.config
        # Default wait_for_stop to True, unless explicitly set to False.
        self._bastion_dir.cancel_job(cfg.name, wait_for_stop=cfg.wait_for_stop is not False)

    def list(self, output_file: Optional[TextIO] = None) -> dict[str, BastionJob]:
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

    def run(self):
        """Runs the command locally.

        Note that 'locally' can also mean locally on the bastion: `submit()` takes the runner
        command, ships it to the bastion, and the bastion executes it with `run()`.
        """
        cfg: BaseBastionManagedJob.Config = self.config
        from_vm = running_from_vm()

        if not from_vm:
            os.environ["BASTION_TIER"] = "disabled"
            self._bundler.bundle(cfg.name)

        # Check for group membership if --project_id is provided.
        if cfg.project_id:
            if not from_vm:
                logging.warning("Supplying --project_id has no purpose if not running in bastion.")
            quota_file = os.path.join(
                "gs://",
                gcp_settings("private_bucket"),
                cfg.bastion_name,
                QUOTA_CONFIG_DIR,
                QUOTA_CONFIG_FILE,
            )
            user_projects = get_user_projects(quota_file, user_id=cfg.user_id)
            if cfg.project_id.lower() not in user_projects:
                raise ValueError(
                    f"User '{cfg.user_id}' is not a member of the project '{cfg.project_id}'. "
                    f"Instead, user '{cfg.user_id}' is a member of: {user_projects}"
                )
        self._runner.execute()

    def submit(self) -> JobSpec:
        """Submits the command to bastion."""
        cfg: BaseBastionManagedJob.Config = self.config
        self._bundler.bundle(cfg.name)

        logging.info("Starting run for job name %s", cfg.name)
        logging.info("Command: %s", cfg.command)
        with tempfile.NamedTemporaryFile("w") as f:
            job_id = generate_job_id()
            metadata = JobMetadata(
                user_id=cfg.user_id,
                project_id=cfg.project_id or "none",
                creation_time=datetime.now(timezone.utc),
                resources=cfg.resources(cfg),
                priority=cfg.priority,
                job_id=job_id,
            )
            jobspec = new_jobspec(name=cfg.name, command=cfg.command, metadata=metadata)
            serialize_jobspec(jobspec, f)
            self._bastion_dir.submit_job(cfg.name, job_spec_file=f.name)
        print(
            "\nView bastion outputs with: (if not found, check job and project history)\n"
            f"gsutil cat {os.path.join(self._bastion_dir.logs_dir, cfg.name)}\n"
            f"\nStop/cancel the job with:\n"
            f"{infer_cli_name()} gcp launch stop "
            f"--name={cfg.name} --bastion={cfg.bastion_name} --env_id={cfg.env_id}\n"
            "\nCheck job history with:\n"
            f"{infer_cli_name()} gcp bastion history "
            f"--name={cfg.bastion_name} --env_id={cfg.env_id} --job_name={cfg.name}\n"
            "\nCheck project history with:\n"
            f"{infer_cli_name()} gcp bastion history "
            f"--name={cfg.bastion_name} --env_id={cfg.env_id} {cfg.project_id or ''}"
        )
        return jobspec

    def update(self) -> JobSpec:
        """Update an existing job without resubmission.

        This will fetch the existing job from Bastion, change
        the trainer command, increment the version in metadata, and then update the job on Bastion.

        The resource related flags including instance_type, num_replicas and enable_pre_provisioner
        are not allowed to change.
        """
        cfg: BaseBastionManagedJob.Config = self.config

        # Get current job spec.
        job_spec = self._bastion_dir.get_job(job_name=cfg.name)
        # Update the bundle.
        self._bundler.bundle(cfg.name)

        logging.info("Starting update for job name %s", cfg.name)
        logging.info("Command: %s", cfg.command)

        # Update the job version.
        job_version = job_spec.metadata.version or 0
        job_spec.metadata.version = job_version + 1

        # The resource related flags are not allowed to change.
        validate_resource_flags(job_spec.command, cfg.command)

        job_spec.command = cfg.command

        logging.info("Updated jobspec: %s", job_spec)

        return self._bastion_dir.update_job(cfg.name, job_spec=job_spec)


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
        """

        namespace: str = "default"
        project: Required[str] = REQUIRED
        zone: Required[str] = REQUIRED
        cluster: Required[str] = REQUIRED

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines launch flags."""
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_string("project", None, "The GCP project name.", **common_kwargs)
        flags.DEFINE_string("zone", None, "The GCP zone name.", **common_kwargs)
        flags.DEFINE_string("namespace", None, "K8s namespace.", **common_kwargs)
        flags.DEFINE_string("cluster", None, "K8s cluster.", **common_kwargs)

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        super().set_defaults(fv)
        fv.set_default("project", default_project())
        fv.set_default("zone", default_zone())
        fv.set_default("namespace", "default")
        # While bundler is only instantiated for certain actions, it doesn't hurt to specify a
        # default even if it's not used.
        fv.set_default("bundler_type", CloudBuildBundler.TYPE)

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, *, action: str, **kwargs) -> Config:
        cfg: BastionManagedGKEJob.Config = super().from_flags(fv, action=action, **kwargs)
        cfg.cluster = cfg.cluster or gcp_settings("gke_cluster", required=False, fv=fv)
        cfg.output_tables = [
            with_k8s_jobset_state(jobs_table, namespace=cfg.namespace),
            user_usage_table,
            project_usage_table,
        ]
        return cfg

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Ensure that the user has installed GKE plugin.
        load_kube_config(project=cfg.project, zone=cfg.zone, cluster=cfg.cluster)

    def submit(self) -> JobSpec:
        """Submits the command to bastion."""
        cfg: BastionManagedGKEJob.Config = self.config
        worker_log = f"{infer_cli_name()} gcp logs --name={cfg.name} --replica=0 --worker=0"
        print(f"\nOnce started, view log outputs with:\n{worker_log}\n")
        job_spec = super().submit()
        print(
            "\nView running pods with:\nkubectl get pods\n"
            "\nNote that the job may take a few minutes to start."
        )
        return job_spec


# Launchers specified here will be tried (in the given order) when launching a given instance type.
_LAUNCHERS: _Registry = [
    # GKE launcher.
    _RegistryMember(
        config=BastionManagedGKEJob.default_config(),
        matcher=config_for_function(match_gcp_api).set(gcp_api=GCPAPI.GKE.value),
        description="Supports launching jobs to GKE.",
    )
]


def _registry_help_string(registry: Sequence[_RegistryMember]) -> str:
    return "\n".join(
        [
            f"Job: {member.config.klass.__name__}\n"
            f"Description: {member.description}\n"
            f"Matcher: {member.matcher}\n"
            for member in registry
        ]
    )


def _get_runner_or_exit(fv: flags.FlagValues = FLAGS) -> BaseRunnerJob.Config:
    """Infers the runner config from runner_module and runner_name."""
    module = import_module(fv.runner_module)
    return module.named_runner_configs(getattr(fv, "runner_name", None))


def _get_launcher_or_exit(
    *, action: str, require_runner: bool = True, flag_values: flags.FlagValues = FLAGS
) -> BaseBastionManagedJob.Config:
    """Retrieves launcher by matching `action` and flags.

    If there are multiple matches, the first one in the registry is returned.
    """
    for launcher in _LAUNCHERS:
        m = maybe_instantiate(launcher.matcher)
        if m(action=action, flag_values=flag_values):
            break
    else:
        raise app.UsageError(
            f"Failed to find a match in the registry for {action=}. "
            f"The provided flags are:\n\n{flag_values.flag_values_dict()}.\n\n"
            f"The registered options are:\n\n{_registry_help_string(_LAUNCHERS)}"
        )
    # Make a copy of the config to avoid modifying global launcher.
    launch_cfg: BaseBastionManagedJob.Config = launcher.config.clone()
    try:
        flag_values.runner_name = _infer_runner_name(flag_values)
        launch_cfg.runner = _get_runner_or_exit(flag_values)
    except Exception:  # pylint: disable=broad-except
        if require_runner:
            raise
        logging.warning(
            "Failed to infer runner name. Proceeding since require_runner=%s", require_runner
        )
    return launch_cfg


def _parse_command_from_argv(argv: Sequence[str], *, action: str) -> str:
    # Parse the command from argv. Note that argv may or may not contain action, so we explicitly
    # look for '--' and extract all args after it.
    command = ""
    for i, arg in enumerate(argv):
        if arg.strip() == "--":
            # For "run", we don't do additional quoting as the joined command is supplied directly
            # to the runner (e.g. k8s manifest), without going through the shell.
            # For "submit", we quote since the command will be shipped to the bastion, at which
            # point it goes through the shell before "run".
            # TODO(markblee): Improve/simplify this behavior.
            command = " ".join(argv[i + 1 :]) if action == "run" else shlex.join(argv[i + 1 :])
            break
    return command


@catch_auth
def main(_, fv: flags.FlagValues = FLAGS):
    action = parse_action(sys.argv, options=_ACTIONS, default="start")
    cfg = _get_launcher_or_exit(
        action=action, require_runner=action in _RUNNER_ACTIONS, flag_values=fv
    )

    # Use sys.argv instead of argv from params, since the param argv has '--' stripped.
    command = _parse_command_from_argv(sys.argv, action=action)

    # If command is supplied, treat as a single --command flag.
    # In all other cases, we expect one or more commands specified as flags.
    from_flags_kwargs = dict(action=action)
    if command:
        from_flags_kwargs["command"] = command

    cfg = from_flags(cfg, fv, **from_flags_kwargs)
    logging.info("Action: %s, Launcher config:\n%s", action, cfg)
    job: BaseBastionManagedJob = cfg.instantiate()
    if fv.dry_run:
        return

    if action == "start":
        job.submit()
    elif action == "list":
        job.list()
    elif action == "stop":
        job.cancel()
    elif action == "update":
        job.update()
    elif action == "run":
        try:
            job.run()
        except KeyboardInterrupt:
            # This is mostly for local debugging, if executing the runner locally.
            logging.info("Got keyboard interrupt, deleting resources...")
            job._runner._delete()  # pytype: disable=attribute-error
    else:
        raise app.UsageError(f"Unsupported action {action}")


class _JobType(enum.Enum):
    """Represents possible values for `--job_type`.

    This is used for selecting the type of runner to use, in the cases where the same instance
    type can map to multiple possible runners.
    """

    # NOTE: do not add more types here. This will be removed in the future.
    DEFAULT = "default"
    FLINK = "flink"


def _prelaunch_flags(fv: flags.FlagValues = FLAGS):
    """Flags necessary for `_get_launcher_or_exit`."""
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
    flags.DEFINE_string("runner_module", runners.__name__, "The runner module.", flag_values=fv)
    flags.DEFINE_string("runner_name", None, "The runner module.", flag_values=fv)

    # TODO(markblee): These flags are for backwards compatibility.
    flags.DEFINE_string(
        "instance_type",
        None,
        "(Deprecated) Instance type to launch. "
        "Please use --runner_module and --runner_name instead.",
        flag_values=fv,
    )
    flags.DEFINE_enum(
        "job_type",
        _JobType.DEFAULT.value,
        [v.value for v in _JobType],
        "(Deprecated) Job type. Please use --runner_module and --runner_name instead.",
        flag_values=fv,
    )


def _infer_runner_name(fv: flags.FlagValues = FLAGS) -> str:
    """Infers the --runner_name from legacy flags --instance_type and --job_type."""
    if fv.runner_name:
        return fv.runner_name

    logging.warning("--runner_name was not provided. Will attempt to infer automatically.")
    runner_name = None
    if getattr(fv, "job_type", None) == _JobType.FLINK.value:
        runner_name = "gke_tpu_flink"
    elif getattr(fv, "instance_type", None):
        if fv.instance_type.startswith("tpu"):
            runner_name = "gke_tpu_single"
        elif fv.instance_type.startswith("gpu-a3-high"):
            runner_name = "gke_gpu_a3_high_single"
        elif fv.instance_type.startswith("gpu-a3-mega"):
            runner_name = "gke_gpu_a3_mega_single"
        elif fv.instance_type.startswith("gpu-a3-ultra"):
            runner_name = "gke_gpu_a3_ultra_single"
        elif fv.instance_type.startswith("gpu-a4-high"):
            runner_name = "gke_gpu_a4_high_single"
        else:
            raise app.UsageError(
                f"Unable to infer --runner_name from --instance_type={fv.instance_type}; "
                "Please specify --runner_name explicitly, or use a different --instance_type."
            )
    else:
        raise app.UsageError(
            "Unable to infer --runner_name. "
            "Please specify --runner_name explicitly, or specify a valid --instance_type."
        )

    assert runner_name is not None
    logging.info("Inferred runner name: %s", runner_name)
    return runner_name


def _private_flags(
    fv: flags.FlagValues = FLAGS,
    prelaunch_fn: Callable[[flags.FlagValues], None] = _prelaunch_flags,
):
    """Defines all launch flags, and amends `app.usage` with additional launch help info."""

    prelaunch_fn(fv)
    fv(sys.argv, known_only=True)

    action = parse_action(sys.argv, options=_ACTIONS, default="start")
    cfg = _get_launcher_or_exit(action=action, require_runner=False, flag_values=fv)
    orig_flags = fv.flag_values_dict()
    define_flags(cfg, fv)

    # If we are able to infer runner (e.g., user specified `--instance_type`), also amend the
    # helpstring with more flags (which would be printed if user also specified `--help`).
    if fv.runner_name:
        output_lines = [f"\nRunner flags [{fv.runner_name}]:\n"]
        fv._render_flag_list([fv[name] for name in fv if name not in orig_flags], output_lines)
        launch_help = "\n".join(output_lines)
    else:
        runner_list = list(_get_runner_or_exit(fv).keys())
        launch_help = (
            f"\nPossible launchers are:\n\n{_registry_help_string(_LAUNCHERS)}\n"
            f"Possible runners are:\n\n{runner_list}\n\n"
            f"Specify --runner_module to use different runners.\n"
            f"Specify --runner_name to see additional flags."
        )

    app.usage = functools.partial(_wrapped_usage, usage=app.usage, launch_help=launch_help)


def _wrapped_usage(
    *,
    usage: Callable,
    launch_help: str,
    writeto_stdout: bool = False,
    **kwargs,
):
    """Wraps original usage by printing additional launch-specific help."""
    usage(writeto_stdout=writeto_stdout, **kwargs)
    of = sys.stdout if writeto_stdout else sys.stderr
    of.write(launch_help)


if __name__ == "__main__":
    configure_logging(logging.INFO)
    _private_flags()
    app.run(main)

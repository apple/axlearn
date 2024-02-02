# Copyright © 2023 Apple Inc.

"""Launch commands on Google Cloud.

The launch command provides a few benefits:
1. Provides a uniform API and entrypoint for interacting with different instance types.
2. Provides utilities for registering custom launcher implementations.

The launch flow depends on the launcher being used. Each launcher must define a "matcher" function
that decides, for a given CLI action (e.g. 'start') and instance type (e.g. 'tpu-v4-8'), whether the
launcher can be used. See `_LAUNCHERS` for a full list, and `LaunchTPUJob` for an example.

Possible actions: [start|stop|list]

    Start: submits a job to the queue.
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

"""
# pylint: disable=redefined-outer-name,protected-access

import functools
import os
import shlex
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, NamedTuple, Optional, TextIO, Tuple, Type, cast

import regex as re
from absl import app, flags, logging

from axlearn.cloud.common.bastion import Job as BastionJob
from axlearn.cloud.common.bastion import (
    JobState,
    download_job_batch,
    new_jobspec,
    serialize_jobspec,
)
from axlearn.cloud.common.bundler import bundler_flags, get_bundler_config
from axlearn.cloud.common.quota import QUOTA_CONFIG_PATH, get_user_projects
from axlearn.cloud.common.scheduler import JobMetadata
from axlearn.cloud.common.types import ResourceMap, ResourceType
from axlearn.cloud.common.utils import (
    configure_logging,
    format_table,
    generate_job_name,
    infer_cli_name,
    parse_action,
)
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.job import Job
from axlearn.cloud.gcp.jobs import tpu_runner
from axlearn.cloud.gcp.jobs.bastion_vm import SubmitBastionJob, output_dir, shared_bastion_name
from axlearn.cloud.gcp.tpu import (
    infer_tpu_cores,
    infer_tpu_version,
    infer_tpu_workers,
    list_tpu_info,
)
from axlearn.cloud.gcp.utils import catch_auth, common_flags, get_credentials
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)

FLAGS = flags.FLAGS


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
        A callable `(action, instance_type) -> bool` (or a config instantiating thereof), used to
        decide whether the launcher is applicable for a given action and instance type.
    * description:
        Human-readable description of the launcher. Printed e.g. if no launchers are matched.
    """

    # We take a class here instead of a config, since a materialized config is often not available
    # at the time of registry. Instead, we often construct the config via `job_cls.from_flags`.
    job_cls: Type[Job]
    # A config is usually more print-friendly, but not strictly required.
    matcher: ConfigOr[Callable[[str, str], bool]]
    description: str


class BaseBastionLaunchJob(Job):
    """A base job definition that launches commands via bastion.

    It provides functionality to submit, delete, and list bastion jobs, but is agnostic to specific
    resource types. At minimum, subclasses should override `_resources()` to specify resources used
    by the job, which will be used for quota management and scheduling.

    See `LaunchTPUJob` as an example.
    """

    @config_class
    class Config(Job.Config):
        """Configures BaseBastionLaunchJob."""

        # Instance type to launch.
        instance_type: Required[str] = REQUIRED
        # Bastion submit config.
        bastion: Required[SubmitBastionJob.Config] = REQUIRED
        # User ID for bastion quota and scheduling.
        user_id: Required[str] = REQUIRED
        # Project ID for bastion quota and scheduling.
        project_id: Required[str] = REQUIRED
        # Job priority.
        priority: Required[int] = REQUIRED
        # Output directory for job logs.
        output_dir: Required[str] = REQUIRED

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines launch flags on the provided flag_values."""
        # We specify allow_override=True so that subclasses may redefine common flags without
        # conflict.
        common_kwargs = dict(flag_values=fv, allow_override=True)
        common_flags(**common_kwargs)
        bundler_flags(**common_kwargs)
        # TODO(markblee): Support configuring an identity provider for --user_id.
        flags.DEFINE_string("name", generate_job_name(), "Job name.", **common_kwargs)
        flags.DEFINE_string("bastion", None, "Name of bastion VM to use.", **common_kwargs)
        flags.DEFINE_integer(
            "priority",
            5,
            "Job priority. Smaller means higher priority.",
            **common_kwargs,
        )
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

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        """Constructs config from flags defined by `define_flags()`.

        Args:
            fv: Flag values (e.g., FLAGS).
            kwargs: Optional key/values to set on the config.

        Returns:
            The job config.
        """
        # Default bastion name depends on the final value of --zone.
        fv.set_default("bastion", shared_bastion_name(fv.zone))
        # Default output_dir depends on the final value of --name.
        fv.set_default("output_dir", f"gs://{gcp_settings('ttl_bucket')}/axlearn/jobs/{fv.name}")
        cfg = super().from_flags(fv, **kwargs)
        # Construct bundler config.
        cfg.bundler = get_bundler_config(bundler_type=fv.bundler_type, spec=fv.bundler_spec)
        # Construct bastion/job config. Note that the output_dir should match the bastion dir.
        cfg.bastion = SubmitBastionJob.from_flags(fv, name=fv.bastion, job_name=cfg.name).set(
            job_spec_file="", bastion_dir=output_dir(fv.bastion)
        )
        return cfg

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        if not (cfg.instance_type and cfg.output_dir):
            raise ValueError("instance_type, output_dir cannot be empty")

    def _delete(self):
        """Submits a delete request to bastion."""
        bastion: SubmitBastionJob = self.config.bastion.instantiate()
        bastion._delete()

    def _list(self, output_file: Optional[TextIO] = None) -> Dict[str, Any]:
        """Lists running jobs and optionally prints them in tabular format.

        Args:
            output_file: Output file. If None, prints to stdout, otherwise writes to the file.
                Subclasses can send outputs to /dev/null if they intend to reformat the output.

        Returns:
            A dict containing:
            * jobs: A list of all bastion-managed jobs sorted by name.
            * usage_by_user: A dict mapping user_id to (total usage, number of jobs), sorted
                descending by total usage.
            * usage_by_project: A dict mapping project_id to (total usage, number of jobs), sorted
                descending by total usage.
        """
        cfg = self.config
        bastion: SubmitBastionJob = cfg.bastion.instantiate()
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = bastion.bastion_dir
            jobs, _ = download_job_batch(
                spec_dir=f"{base_dir}/jobs/active",
                state_dir=f"{base_dir}/jobs/states",
                user_state_dir=f"{base_dir}/jobs/user_states",
                local_spec_dir=tmpdir,
            )
            jobs: Dict[str, BastionJob] = dict(sorted(jobs.items(), key=lambda kv: kv[0]))

            # Maps user_id -> resource_type -> (total_usage, count).
            usage_by_user = defaultdict(lambda: defaultdict(lambda: [0, 0]))
            usage_by_project = defaultdict(lambda: defaultdict(lambda: [0, 0]))
            for job in jobs.values():
                if job.state == JobState.PENDING:
                    continue
                user_id = job.spec.metadata.user_id
                project_id = job.spec.metadata.project_id
                resources = job.spec.metadata.resources
                for resource_type, usage in resources.items():
                    usage_by_user[user_id][resource_type][0] += usage
                    usage_by_user[user_id][resource_type][1] += 1
                    usage_by_project[project_id][resource_type][0] += usage
                    usage_by_project[project_id][resource_type][1] += 1

            print(format_table(**self._jobs_table(jobs)), file=output_file)
            print(format_table(**self._usage_table(usage_by_project)), file=output_file)
            print(format_table(**self._usage_table(usage_by_user)), file=output_file)
            return dict(
                jobs=jobs,
                usage_by_user=usage_by_user,
                usage_by_project=usage_by_project,
            )

    def _resources(self) -> ResourceMap:
        """Infers resources from instance_type. Can be overridden by subclasses.

        Should return a ResourceMap, where keys are resource types and values are resource amounts.
        """
        return {}

    def _execute(self):
        """Submits the command to bastion."""
        cfg = self.config

        # Check for group membership if --project_id is provided.
        # TODO(markblee): Make this check at the bastion level.
        if cfg.project_id:
            quota_file = (
                f"gs://{gcp_settings('private_bucket')}/{cfg.bastion.name}/{QUOTA_CONFIG_PATH}"
            )
            user_projects = get_user_projects(quota_file, user_id=cfg.user_id)
            if cfg.project_id.lower() not in user_projects:
                raise ValueError(
                    f"User '{cfg.user_id}' is not a member of the project '{cfg.project_id}'. "
                    f"Instead, user '{cfg.user_id}' is a member of: {user_projects}"
                )
        if cfg.bundler:
            self._bundler.bundle(cfg.name)
        logging.info("Starting run for job name %s", cfg.name)
        with tempfile.NamedTemporaryFile("w") as f:
            logging.info("Command: %s", cfg.command)
            metadata = JobMetadata(
                user_id=cfg.user_id,
                project_id=cfg.project_id or "none",
                creation_time=datetime.now(),
                resources=self._resources(),
                priority=cfg.priority,
            )
            serialize_jobspec(new_jobspec(name=cfg.name, command=cfg.command, metadata=metadata), f)
            bastion: SubmitBastionJob = cfg.bastion.set(job_spec_file=f.name).instantiate()
            bastion._execute()
        print(
            f"\nStop/cancel the job with:\n"
            f"{infer_cli_name()} gcp launch stop "
            f"--name={cfg.name} --bastion={cfg.bastion.name} --instance_type={cfg.instance_type}"
        )

    def _jobs_table(self, jobs: Dict[str, BastionJob]) -> Dict:
        """Construct tabular jobs info.

        Args:
            jobs: A mapping from job name to job info.

        Returns:
            A table which can be passed to `format_table`.
        """
        return dict(
            headings=["NAME", "USER_ID", "JOB_STATE", "PROJECT_ID", "RESOURCES", "PRIORITY"],
            rows=[
                [
                    job.spec.name,
                    job.spec.metadata.user_id,
                    job.state,
                    job.spec.metadata.project_id,
                    str(job.spec.metadata.resources),
                    str(job.spec.metadata.priority),
                ]
                for job in jobs.values()
            ],
        )

    def _usage_table(self, usage_info: Dict[str, Dict[ResourceType, Tuple[float, int]]]):
        """Construct tabular usage info.

        Args:
            usage_info: A mapping from principal to resource type to
                (total usage, total number of jobs).

        Returns:
            A table which can be passed to `format_table`.
        """
        table = dict(
            headings=["PRINCIPAL", "RESOURCE", "USAGE", "COUNT"],
            rows=[
                [
                    principal or "unknown",
                    resource_type,
                    usage[0],
                    usage[1],
                ]
                for principal, resource_usage in usage_info.items()
                for resource_type, usage in resource_usage.items()
            ],
        )
        # Sort by usage descending.
        table["rows"] = sorted(table["rows"], key=lambda v: (v[1], v[2]), reverse=True)
        return table


# TODO(markblee): Add a LaunchCPUJob.
class LaunchTPUJob(BaseBastionLaunchJob):
    """Launches a TPU job via bastion."""

    @config_class
    class Config(BaseBastionLaunchJob.Config):
        """Configures LaunchTPUJob."""

        # Number of TPU slices.
        num_slices: int = 1

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Defines launch flags using tpu_runner."""
        super().define_flags(fv)
        tpu_runner.launch_flags(flag_values=fv)
        fv.set_default("name", generate_job_name())
        fv.set_default("tpu_type", cls._tpu_type(fv.instance_type))
        fv["tpu_type"].help += " (Note: inherited from --instance_type)."

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, *, command: str, **kwargs) -> Config:
        """Constructs config from flags defined by `define_flags()`.

        In addition to logic defined in `BaseBastionLaunchJob.from_flags()`:
        * Ensures "tpu" extras are bundled.
        * Automatically includes flags used by tpu_runner as part of the command.

        Args:
            fv: Flag values (e.g., FLAGS).
            command: The user-supplied command, i.e. everything after `--` as a string.
            kwargs: Optional key/values to set on the config.

        Returns:
            The job config.
        """
        cfg = super().from_flags(fv, **kwargs)

        # Construct bundler config.
        cfg.bundler = tpu_runner.with_tpu_extras(cfg.bundler)

        # Save flags values corresponding to our launch flags.
        launch_fv = flags.FlagValues()
        tpu_runner.launch_flags(flag_values=launch_fv)

        # Convert the user-supplied flags into a space-separated string, which is forwarded to the
        # command executed by bastion. Only flags which are used by the tpu_runner are forwarded.
        filtered = []
        for _, module_flags in fv.flags_by_module_dict().items():
            for flag in module_flags:
                if flag.name in launch_fv and flag.value is not None:
                    # Multi-flags get serialized with newlines.
                    filtered.extend(flag.serialize().split("\n"))
        launch_flags = " ".join(filtered)
        # Note: command is only used for start.
        cfg.command = f"python3 -m {tpu_runner.__name__} start {launch_flags} -- {command}"
        return cfg

    @classmethod
    def _tpu_type(cls, instance_type: str) -> str:
        # Infers tpu type from instance type.
        return instance_type.replace("tpu-", "")

    def _list(self, output_file: Optional[TextIO] = None) -> Dict[str, Any]:
        """Lists running jobs and their associated TPU resources.

        In addition to outputs provided by `BaseBastionLaunchJob._list()`, also returns a dict with:
        * running_tpu_infos: The currently-running TPUs.
        * running_tpu_to_job_name: A mapping from TPU name to job name.
        """
        with open(os.devnull, "w", encoding="utf-8") as f:
            list_info = super()._list(output_file=f)
        running_tpu_infos = {
            tpu_info.name: tpu_info for tpu_info in list_tpu_info(get_credentials())
        }
        running_tpu_to_job_name = {}

        # Append TPU state information to the jobs table.
        jobs_table = self._jobs_table(list_info["jobs"])
        jobs_table["headings"].append("TPU_STATE")
        for i, job in enumerate(list_info["jobs"].values()):
            job = cast(BastionJob, job)

            # In the multislice case, job_tpu_names come from job_name-<slice>.
            # TODO(markblee): Don't rely on parsing flags.
            if matches := re.search(r"--num_slices[= ](\d+)", job.spec.command):
                num_slices = int(matches[1])
            else:
                num_slices = 1

            if num_slices > 1:
                job_tpu_names = [f"{job.spec.name}-{slice_idx}" for slice_idx in range(num_slices)]
            else:
                job_tpu_names = [job.spec.name]

            # Gather unique TPU states for the given job.
            tpu_states = set()
            for tpu_name in job_tpu_names:
                if tpu_name in running_tpu_infos:
                    tpu_states.add(running_tpu_infos[tpu_name].state or "UNKNOWN")
                    running_tpu_to_job_name[tpu_name] = job.spec.name
                else:
                    tpu_states.add("PENDING")

            jobs_table["rows"][i].append(",".join(tpu_states))

        print(format_table(**jobs_table), file=output_file)
        print("Usage by project:", file=output_file)
        print(format_table(**self._usage_table(list_info["usage_by_project"])), file=output_file)
        print("Usage by user:", file=output_file)
        print(format_table(**self._usage_table(list_info["usage_by_user"])), file=output_file)

        return dict(
            running_tpu_infos=running_tpu_infos,
            running_tpu_to_job_name=running_tpu_to_job_name,
            **list_info,
        )

    def _resources(self) -> ResourceMap:
        """Defines TPU resources used by the job."""
        cfg = self.config
        tpu_type = self._tpu_type(cfg.instance_type)
        return {infer_tpu_version(tpu_type): infer_tpu_cores(tpu_type) * cfg.num_slices}

    def _execute(self):
        """Submits the command to bastion.

        In addition to logic defined in `BaseBastionLaunchJob._execute()`, also emits the output
        logs for each TPU worker.
        """
        cfg = self.config
        super()._execute()
        all_logs = "\n".join(
            [
                f'gsutil cat "{cfg.output_dir}/output/*-{i}/run.log"'
                for i in range(infer_tpu_workers(self._tpu_type(cfg.instance_type)))
            ]
        )
        print(
            "\nNote that the job may take a few minutes to start. "
            f"Once started, view TPU log outputs with:\n{all_logs}\n"
        )


def _match_by_regex(match_regex: Dict[str, str]):
    """Matches action and instance type by regex.

    For example:

        match_regex={'start': 'pat1', 'list': 'pat2'}

    ... means that the launcher will be used if action is 'start' and --instance_type regex matches
    'pat1', or if action is 'list' and --instance_type regex matches 'pat2'. The launcher will not
    be invoked for any other action.
    """

    def fn(action: str, instance_type: str) -> bool:
        """Returns True iff the launcher supports the given action and instance_type."""
        return action in match_regex and bool(re.match(match_regex[action], instance_type))

    return fn


# Launchers specified here will be tried (in the given order) when launching a given instance type.
_LAUNCHERS = [
    Launcher(
        job_cls=LaunchTPUJob,
        matcher=config_for_function(_match_by_regex).set(
            match_regex=dict(start=r"tpu-v.+-(\d)+", list=r"tpu.*", stop=r"tpu.*"),
        ),
        description=(
            "Supports launching TPU jobs. "
            "For 'start', provide the full TPU version, e.g. --instance_type=tpu-v4-8. "
            "For 'list' or 'stop', provide the full version or simply --instance_type=tpu."
        ),
    )
]


def _get_launcher_or_exit(*, action: str, instance_type: str) -> Launcher:
    """Retrieves launcher by matching instance_type.

    If there are multiple matches, the first one in the registry is returned.
    """
    # Idenfity launcher from instance type.
    for launcher in _LAUNCHERS:
        m = maybe_instantiate(launcher.matcher)
        if m(action, instance_type):
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
        f"Don't know how to launch instance type '{instance_type}' for action '{action}'.\n"
        f"The registered launchers are:\n\n{launchers}"
    )


@catch_auth
def main(_):
    if FLAGS.instance_type is None:
        raise app.UsageError("--instance_type is required.")

    action = parse_action(sys.argv, options=["start", "stop", "list"], default="start")
    launcher = _get_launcher_or_exit(action=action, instance_type=FLAGS.instance_type)

    # Parse the command from argv. Note that argv may or may not contain action, so we explicitly
    # look for '--' and extract all args after it. Use sys.argv instead of argv from params, since
    # the param argv has '--' stripped.
    command = ""
    for i, arg in enumerate(sys.argv):
        if arg.strip() == "--":
            command = shlex.join(sys.argv[i + 1 :])
            break
    cfg = launcher.job_cls.from_flags(FLAGS, command=command)
    job: BaseBastionLaunchJob = cfg.instantiate()
    if FLAGS.dry_run:
        print(f"Action: {action}\nJob config:\n{job.config}")
        return

    if action == "start":
        job._execute()
    elif action == "list":
        job._list()
    elif action == "stop":
        job._delete()
    else:
        raise app.UsageError(f"Unsupported action {action}")


def _private_flags():
    """Defines all launch flags, and amends `app.usage` with additional launch help info."""

    flags.DEFINE_string("instance_type", None, "Instance type to launch.")
    flags.DEFINE_bool("dry_run", False, "Output job config and exit without running.")
    FLAGS(sys.argv, known_only=True)

    launch_help = None
    # Allow instance_type to be None when running --help without any flags. On the other hand, if
    # instance_type is provided when running --help, we show additional help info.
    if FLAGS.instance_type:
        action = parse_action(sys.argv, options=["start", "stop", "list"], default="start")
        launcher = _get_launcher_or_exit(action=action, instance_type=FLAGS.instance_type)
        orig_flags = FLAGS.flag_values_dict()
        if hasattr(launcher.job_cls, "define_flags"):
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

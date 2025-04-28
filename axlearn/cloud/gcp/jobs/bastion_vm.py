# Copyright © 2023 Apple Inc.

"""Launches a bastion VM on Google Cloud Platform (GCP).

See `axlearn/cloud/common/bastion.py` for bastion details.

To submit and cancel bastion jobs, please see `axlearn/cloud/gcp/jobs/launch.py`.

Possible actions: [history]

    History: prints history of a job or a project id.

Examples:

    # Check the job status/history.
    axlearn gcp bastion history --name=shared-bastion --job_name=my-job

    # If it is not running, check the project history to see the limit and the queue.
    axlearn gcp bastion history --name=shared-bastion <project_id>

"""

# pylint: disable=consider-using-with,too-many-branches,too-many-instance-attributes,too-many-lines
import os
import re
from collections.abc import Sequence
from typing import Optional

from absl import app, flags, logging

from axlearn.cloud.common.bastion import bastion_job_flags
from axlearn.cloud.common.quota import QUOTA_CONFIG_DIR, QUOTA_CONFIG_FILE, get_resource_limits
from axlearn.cloud.common.utils import configure_logging, parse_action
from axlearn.cloud.gcp.config import default_env_id, default_project, default_zone, gcp_settings
from axlearn.cloud.gcp.utils import catch_auth, common_flags
from axlearn.common.file_system import exists, glob
from axlearn.common.file_system import open as fs_open
from axlearn.common.file_system import readfile

FLAGS = flags.FLAGS


def _private_flags(flag_values: flags.FlagValues = FLAGS):
    common_flags(flag_values=flag_values)
    bastion_job_flags(flag_values=flag_values)
    flag_values.set_default("project", default_project())
    flag_values.set_default("zone", default_zone())
    flag_values.set_default("env_id", default_env_id())

    def _validate_name(name: str):
        # Must be a valid GCP VM name, as well as a valid docker tag name. For simplicity, check
        # that it's some letters followed by "-bastion", and that it's not too long (VM names are
        # capped at 63 chars).
        return len(name) < 64 and re.match("[a-z][a-z0-9-]*-bastion", name)

    flags.register_validator(
        "name",
        _validate_name,
        message="Must be < 64 chars and match <name>-bastion.",
        flag_values=flag_values,
    )


def infer_bastion_name(fv: Optional[flags.FlagValues]) -> Optional[str]:
    # The env_id-namespacing is necessary because of quirks with compute API. Specifically, even if
    # creating VMs within a specific zone, names are global.
    env_id = gcp_settings("env_id", fv=fv)
    return gcp_settings(  # pytype: disable=bad-return-type
        "bastion_name",
        default=f"{env_id}-gke-bastion",
        fv=fv,
    )


def bastion_root_dir(bastion: str, *, fv: Optional[flags.FlagValues]) -> str:
    """Directory in gs where jobs are recorded."""
    return os.path.join("gs://", gcp_settings("permanent_bucket", fv=fv), bastion)


def _job_history(*, job_name: str, root_dir: str) -> str:
    result = ""
    spec_path_pattern = os.path.join(root_dir, "jobs", "*", job_name)
    spec_paths = glob(spec_path_pattern)
    if not spec_paths:
        raise FileNotFoundError(f"Job spec not found in {spec_path_pattern}")
    for spec_path in spec_paths:
        result += f"<spec path={spec_path}>\n{readfile(spec_path)}\n</spec>\n"
    history_path = os.path.join(root_dir, "history", "jobs", job_name)
    result += f"<history path={history_path}>\n{readfile(history_path)}</history>\n"
    return result


def _project_history(*, root_dir: str, project_id: str) -> str:
    project_dir = os.path.join(
        root_dir,
        "history",
        "projects",
        project_id,
    )
    if not exists(project_dir):
        raise FileNotFoundError(f"Project {project_id} not found at {project_dir}")
    paths = sorted(glob(os.path.join(project_dir, "*")))
    entries = []
    for path in paths[-2:]:
        with fs_open(path, mode="r") as f:
            entry = None
            for line in f:
                if re.search("^[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}", line):
                    # Timestamp line.
                    if entry:
                        entries.append(entry)
                    entry = line
                else:
                    entry += line
            if entry:
                entries.append(entry)
    if len(entries) > 3:
        # Only keep the last three entries.
        entries = entries[-3:]
    lines = "".join(entries)
    return f"<history project_id={project_id}>\n{lines}</history project_id={project_id}>"


def quota_file(flag_values: flags.FlagValues) -> str:
    return os.path.join(
        "gs://",
        gcp_settings("private_bucket", fv=flag_values),
        flag_values.name,
        QUOTA_CONFIG_DIR,
        QUOTA_CONFIG_FILE,
    )


@catch_auth
def main(argv: Sequence[str], *, flag_values: flags.FlagValues = FLAGS):
    action = parse_action(argv, options=["history"])

    def root_dir():
        return bastion_root_dir(flag_values.name, fv=flag_values)

    if action == "history":
        if flag_values.job_name:
            # Print job history.
            history = _job_history(root_dir=root_dir(), job_name=flag_values.job_name)
        else:
            # Print project history.
            if len(argv) > 2:
                project_id = argv[2]
            else:
                project_id = "none"
            try:
                history = _project_history(root_dir=root_dir(), project_id=project_id)
            except FileNotFoundError as e:
                limits = get_resource_limits(quota_file(flag_values))
                raise FileNotFoundError(
                    f"Available projects are {list(limits.project_resources.keys()) + ['none']}"
                ) from e
        print(history)
    else:
        raise ValueError(f"Unknown action {action}")


if __name__ == "__main__":
    _private_flags()
    configure_logging(logging.INFO)
    app.run(main)

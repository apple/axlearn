# Copyright © 2024 Apple Inc.

"""Deprecated. Please see `axlearn/cloud/gcp/jobs/launch.py`."""
# pylint: disable=protected-access

import sys
from collections.abc import Sequence

from absl import app, flags, logging

from axlearn.cloud.common.utils import configure_logging, define_flags, from_flags, parse_action
from axlearn.cloud.gcp.jobs import launch
from axlearn.cloud.gcp.utils import catch_auth

FLAGS = flags.FLAGS


# This is purely for backwards compatibility with old clients.
# TODO(markblee): Remove this.
@catch_auth
def main(argv: Sequence[str]):
    # Parse to sanity check action="start".
    parse_action(argv, options=["start"])
    # Treat as "run" action.
    action = "run"

    cfg = launch._get_launcher_or_exit(action=action, require_runner=True, flag_values=FLAGS)
    # Use sys.argv instead of argv from params, since the param argv has '--' stripped.
    command = launch._parse_command_from_argv(sys.argv, action=action)

    # Retain flags that are used by the launch.
    fv = flags.FlagValues()
    define_flags(cfg, fv)
    fv(sys.argv, known_only=True)

    # Command is always specified as args in the legacy codepath.
    cfg = from_flags(cfg, fv, action=action, command=command)
    job: launch.BaseBastionManagedJob = cfg.instantiate()
    job.run()


if __name__ == "__main__":
    configure_logging(logging.INFO)
    launch._private_flags()
    flags.DEFINE_alias("tpu_type", "instance_type")
    flags.DEFINE_alias("num_slices", "num_replicas")
    app.run(main)

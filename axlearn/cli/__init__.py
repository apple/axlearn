# Copyright © 2023 Apple Inc.

"""AXLearn CLI, backed by absl.

We construct the command-line syntax tree with the library absl, using non-leaf nodes of type
`CommandGroup` and leaf nodes of type `Command`. Each command is a full path from root to leaf, e.g:

    axlearn cmd_group cmd [--flag1] [--flag2] [arg1] [arg2]

Initially, the entire `argv` supplied by the user is fed into the root `CommandGroup`. Each child
`CommandGroup` is responsible for dropping its name from the positional args, to ensure that the
positional args supplied via the CLI and via invoking the leaf `Command` directly are the same.
This is handled automatically via the methods `group.add_cmd*`. From a developer's perspective,
adding a command group and/or commands can be done simply:

    group = CommandGroup("my_cmd_group", parent=root)
    group.add_flag("--common_flag", ...)
    group.add_cmd_from_module("my_cmd", ...)

Which can be invoked like:

    axlearn my_cmd_group --common_flag=123 my_cmd [arg1] [arg2]

The order of the flags doesn't matter, so the following is equivalent:

    axlearn my_cmd_group my_cmd --common_flag=123 [arg1] [arg2]

This is because all flags supplied at runtime (minus the dropped positional args) are ultimately
forwarded to the leaf `Command`. By default, absl parses flags strictly in the sense that a flag
defined at a `CommandGroup` must be supported by all leaf `Commands` rooted at that group, or else
it will throw an error. If you need to relax this requirement, you can do so by passing
`undefok=True` when adding the flag, i.e. `group.add_flag(..., undefok=True)`.

The executed `Command` ultimately gets processed at `absl_main`. Depending on which
`Command.add_cmd*` method was used to add the command, the `args` passed to `absl_main` may
contain a `module`, (shell) `command`, or something else. This is used to determine how to process
the command, and can be easily extended to support additional command types.
"""
# pylint: disable=import-outside-toplevel
from axlearn.cli.utils import main as base_main
from axlearn.cli.utils import register_root_command_group


def main(**kwargs):
    # Register default command groups, depending on which extras the user has installed.
    # Do this in __main__ to avoid registering on import.
    try:
        from axlearn.cli.gcp import add_cmd_group

        register_root_command_group(add_cmd_group, name="gcp")
    except (ImportError, ModuleNotFoundError):
        # GCP extras not installed.
        pass

    base_main(**kwargs)


if __name__ == "__main__":
    # Note: don't use `app.run(main)`, as we invoke `app.run` within `main` itself.
    # This lets us use `main` as an entrypoint for setuptools, flit, and the like.
    main()

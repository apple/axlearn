# Copyright © 2023 Apple Inc.

"""Helper utilities."""

import functools
import os
import re
import shlex
import signal
import subprocess
from enum import Enum
from typing import Any, Optional, Protocol

from absl import app
from absl.flags import argparse_flags

# Note: argparse doesn't have great typing support.
# pylint: disable-next=protected-access
Command = argparse_flags.argparse._ActionsContainer
Flag = argparse_flags.argparse.Action


class CommandType(Enum):
    HELP = 0
    MODULE = 1
    SHELL = 2


def absl_main(args: argparse_flags.argparse.Namespace, **kwargs) -> int:
    """The main function that processes commands.

    Implements the logic that handles different types of commands registered via
    `CommandGroup.add_cmd*` variants.

    Args:
        args: Parsed args returned from root `CommandGroup.parse_known_args`.
        kwargs: Default kwargs forwarded to each command.

    Returns:
        Return code.
    """
    # pylint: disable=consider-using-with
    procs: list[subprocess.Popen] = []

    def _sig_handler(sig: int, _):
        for proc in procs:
            proc.send_signal(sig)

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    returncode = 0
    try:
        # In some cases, e.g. invoking a MODULE type command, argv[0] can potentially change. In
        # these cases, we make it available via env variable.
        env = kwargs.get("env", os.environ.copy())
        env["AXLEARN_CLI_NAME"] = args.argv[0]
        kwargs["env"] = env
        kwargs.setdefault("text", True)  # Receive inputs/outputs as text.

        # Invoke a module.
        if args.command_type == CommandType.MODULE:
            proc = subprocess.Popen(
                ["python3", "-m", args.module] + args.argv[1:], start_new_session=True, **kwargs
            )
            procs.append(proc)
            returncode = proc.wait()

        # Invoke a subprocess.
        elif args.command_type == CommandType.SHELL:
            cmd = args.command
            if len(args.argv) > 1:
                cmd += " " + shlex.join(args.argv[1:])
            proc = subprocess.Popen(cmd, shell=True, **kwargs)
            procs.append(proc)
            returncode = proc.wait()

        # Invoke original command with --help (flag can be specified multiple times).
        # TODO(markblee): See if we can get the invoked subparser here and print_help.
        else:  # args.command_type == CommandType.HELP
            proc = subprocess.Popen(_insert_flags(args.argv, ["--help"]), **kwargs)
            procs.append(proc)
            returncode = proc.wait()
    except KeyboardInterrupt as _:
        # Return a non-zero returncode, e.g. if chaining subsequent commands with &&.
        returncode = 1

    return returncode


class CommandGroup:
    """A CommandGroup groups multiple commands under a common namespace."""

    def __init__(
        self,
        name: str,
        *,
        parent: Optional["CommandGroup"] = None,
        argv: Optional[list[str]] = None,
        **kwargs,
    ):
        """Instantiates a Command Group.

        Args:
            name: Name of the group.
            parent: Optional parent CommandGroup. If None, constructs root CommandGroup.
            argv: Optional argv. Must not be None at root, and None at non-root groups.

        Raises:
            ValueError: If argv not present at root.
        """
        if (parent is None) != (argv is not None):
            raise ValueError("argv must be present iff we're at root.")

        if parent is None:
            self.parser = argparse_flags.ArgumentParser(**kwargs)
            self.namespace = argparse_flags.argparse.Namespace(
                root_parser=self.parser,
                defaults={},
                argv=argv,
                filter_argv=None,
                command_type=CommandType.HELP,
            )
        else:
            self.parser: argparse_flags.ArgumentParser = parent.subparsers.add_parser(
                name, **kwargs
            )
            self.namespace = parent.namespace  # Copy reference.
            argv = _drop_one(parent.argv, name)  # Copy value.

        self.argv = argv
        self.subparsers = self.parser.add_subparsers()

    def add_cmd(self, name: str, filter_argv: Optional[str] = None, **kwargs) -> Command:
        """Add a generic command.

        Args:
            name: Command name.
            filter_argv: Optional regex pattern to filter argv. Only the args that match this
                pattern will be forwarded to the command. This is useful when invoking a command
                that accepts a fixed, known set of flags, but is not capable of handling optional
                flags (such as some bash commands, or non-absl modules that don't accept undefok).
            kwargs: Default kwargs forwarded to `subparsers.add_parser`.

        Returns:
            The command.
        """
        cmd = self.subparsers.add_parser(name, **kwargs)
        argv = _drop_one(self.argv, name)
        cmd.set_defaults(argv=argv, parser=cmd, filter_argv=filter_argv)
        return cmd

    def add_cmd_from_module(self, name: str, *, module: str, **kwargs) -> Command:
        """Adds a command that invokes an existing absl script.

        Args:
            name: Command name.
            module: A string of the format `module_name`.
            kwargs: Default kwargs forwarded to `subparsers.add_parser`.

        Returns:
            The command.
        """
        # Note: always disable help here, since we want to leverage absl help.
        cmd = self.add_cmd(name, add_help=False, **kwargs)
        # Handle any added flags marked with `undefok=True`.
        if hasattr(self.namespace, "undefok"):
            argv = cmd.get_default("argv") or []
            argv = _insert_flags(argv, [f"--undefok={self.namespace.undefok}"])
            cmd.set_defaults(argv=argv)
        cmd.set_defaults(command_type=CommandType.MODULE, module=module)
        return cmd

    def add_cmd_from_bash(self, name: str, *, command: str, **kwargs) -> Command:
        """Adds a command that invokes a bash command.

        Args:
            name: Command name.
            command: Bash command to execute.
            kwargs: Default kwargs forwarded to `subparsers.add_parser`.

        Returns:
            The command.
        """
        cmd = self.add_cmd(name, **kwargs)
        cmd.set_defaults(command_type=CommandType.SHELL, command=command)
        return cmd

    def add_flag(
        self, name: str, undefok: bool = False, default: Optional[Any] = None, **kwargs
    ) -> Flag:
        """Adds a flag to the command group.

        Args:
            name: Name of the flag. Should include "--".
            undefok: If True, the flag will be ignored if the invoked command does not support it,
                rather than throwing an error.
            default: Default flag value.
            kwargs: Default kwargs forwarded to `parser.add_argument`.

        Returns:
            The flag.
        """
        if undefok:
            self.namespace.undefok = _add_to_csv(
                getattr(self.namespace, "undefok", ""), name.lstrip("--")
            )
        if default is not None:
            self.namespace.defaults[name] = default
        return self.parser.add_argument(name, default=default, **kwargs)

    def parse_known_args(self) -> argparse_flags.argparse.Namespace:
        """Parse argv.

        Returns:
            A Namespace with parsed flags.
        """
        args, _ = self.parser.parse_known_args(self.argv[1:], self.namespace)

        # Defaults for which no flags were provided are added as flags.
        for arg in _flags(self.argv)[1:]:
            # Remove an arg from `args.defaults` if its value is given explicitly.
            # Note that args after `--` are not considered, as they are not returned by `_flags`.
            args.defaults.pop(_argname(arg), None)
        # Insert --<flag>=<default_value> to `args.argv`.
        args.argv = _insert_flags(args.argv, ["=".join(arg) for arg in args.defaults.items()])

        # Filter args if requested.
        if args.filter_argv:
            pat = re.compile(args.filter_argv)
            args.argv = [args.argv[0]] + [
                arg for arg in _flags(args.argv)[1:] if pat.match(_argname(arg))
            ]

        return args


def _drop_one(items: list[str], drop: str) -> list[str]:
    """Returns a copy of `items`, dropping at most one instance of `drop`."""
    try:
        i = items.index(drop)
        items = items[:i] + items[i + 1 :]
    except ValueError:
        items = items.copy()  # Nothing to drop.
    return items


def _add_to_csv(csv: str, item: str) -> str:
    """Adds a string to a comma-separated string of items."""
    items = csv.split(",") if csv else []
    items.append(item)
    return ",".join(items)


def _argname(arg: str) -> str:
    """Parses arg name from arg.
    Args are either key=value or key (in the case where value is space separated).
    """
    return arg.split("=")[0]


def _flags(argv: list[str]):
    """Returns flags prior to `--`."""
    args = []
    for arg in argv:
        if arg.strip() == "--":
            break
        args.append(arg)
    return args


def _insert_flags(argv: list[str], flags: list[str]) -> list[str]:
    """Insert a sequence of --flag=value into argv, ensuring they are inserted before `--`."""
    assert len(argv) > 0 and not argv[0].startswith("--"), "argv[0] should contain the program name"
    i = len(_flags(argv))
    return argv[0:i] + flags + argv[i:]


def get_path(d: dict[str, Any], k: str, default: Optional[Any] = None) -> Optional[Any]:
    """Nested dict access with default."""
    parts = k.split(".")
    for part in parts[:-1]:
        d = d.get(part, {})
    return d.get(parts[-1], default)


_command_group_registry = {}


class CommandGroupFn(Protocol):
    """Adds flags and subcommands to the given CommandGroup."""

    def __call__(self, *, parent: CommandGroup):
        pass


def clear_root_command_groups():
    _command_group_registry.clear()


def register_root_command_group(fn: CommandGroupFn, name: Optional[str] = None):
    """Registers a root command group."""
    name = name or fn.__name__
    if name in _command_group_registry:
        raise ValueError(f"Command group {name} already registered.")
    _command_group_registry[name] = fn
    return fn


def parse_flags(argv: list[str]) -> argparse_flags.argparse.Namespace:
    root = CommandGroup(
        "root", argv=argv, description="AXLearn: An Extensible Deep Learning Library."
    )
    # TODO(markblee): Add top level arguments here, like verbosity.
    for command_group in _command_group_registry.values():
        command_group(parent=root)
    return root.parse_known_args()


def main(**kwargs):
    app.run(functools.partial(absl_main, **kwargs), flags_parser=parse_flags)

# Copyright © 2023 Apple Inc.

"""Tests CLI utilities."""

import shutil
import subprocess
import tempfile
from io import StringIO
from unittest import mock

import pytest
from absl.flags import FlagValues, argparse_flags
from absl.testing import parameterized

from axlearn.cli.utils import CommandGroup, CommandType, _insert_flags, absl_main


def _parse(argv: list[str]) -> argparse_flags.argparse.Namespace:
    kwargs = dict(inherited_absl_flags=FlagValues())
    root = CommandGroup("root", argv=argv, **kwargs)
    root.add_flag("--root", undefok=True, action="store_true")
    root.add_flag("--root_default", undefok=True, default="some_value")

    child1 = CommandGroup("child1", parent=root, **kwargs)
    child1.add_flag("--child1", undefok=True, action="store_true")
    grandchild1 = CommandGroup("grandchild1", parent=child1, **kwargs)
    grandchild1.add_cmd_from_module("command1", module="axlearn.cli.testdata.dummy", **kwargs)
    grandchild1.add_cmd_from_module(
        "command2", module="axlearn.cli.testdata.dummy", filter_argv="undefok|.*default", **kwargs
    )

    child2 = CommandGroup("child2", parent=root, **kwargs)
    # All leaf commands must support --child2.
    child2.add_flag("--child2", undefok=False, action="store_true", required=True)
    child2.add_cmd_from_module("command2", module="axlearn.cli.testdata.dummy", **kwargs)
    return root.parse_known_args()


def _run(args: argparse_flags.argparse.Namespace) -> tuple[int, str, str]:
    """Wraps absl_main by returning (returncode, stdout, stderr)."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        returncode = absl_main(args, stdout=stdout, stderr=stderr)
        stdout.seek(0)
        stderr.seek(0)
        stdout = stdout.read().strip().decode("utf-8")
        stderr = stderr.read().strip().decode("utf-8")
    return returncode, stdout, stderr


class TestUtils(parameterized.TestCase):
    """Tests utils."""

    @property
    def root_module(self):
        """Subclasses can override to test a different CLI entrypoint."""
        return "axlearn"

    def setUp(self):
        if not shutil.which(self.root_module):
            pytest.skip(f"CLI {self.root_module} has not been installed -- skipping tests.")

    def test_basic(self):
        # Test that invoking an invalid command group or command exits with error.
        with mock.patch("sys.stderr", new=StringIO()) as stderr:
            with self.assertRaises(SystemExit) as e:
                _parse([self.root_module, "unknown"])
            self.assertEqual(e.exception.code, 2)
            self.assertIn(
                "invalid choice: 'unknown' (choose from 'child1', 'child2')", stderr.getvalue()
            )

        # Test that invoking CLI by itself prints the help message.
        returncode, stdout, stderr = _run(_parse([self.root_module]))
        self.assertRegex(stdout, r"usage: .*")
        self.assertEqual(returncode, 0, msg=stderr)

        # Test that invoking subcommand by itself injects --help.
        args = _parse([self.root_module, "child1"])
        self.assertEqual(args.command_type, CommandType.HELP)

    def test_help(self):
        # Test that passing --help exits without error.
        with mock.patch("sys.stdout", new=StringIO()) as stdout:
            with self.assertRaises(SystemExit) as e:
                _parse([self.root_module, "child1", "--help"])
            self.assertEqual(e.exception.code, 0)

            helpstring = stdout.getvalue().replace("\n", "  ")
            self.assertRegex(
                helpstring, r"usage: .* child1 \[-h\] \[--helpfull\] \[--child1\] +{grandchild1}"
            )

        # Test that passing --help exits without error.
        with mock.patch("sys.stdout", new=StringIO()) as stdout:
            with self.assertRaises(SystemExit) as e:
                _parse([self.root_module, "child1", "grandchild1", "--help"])
            self.assertEqual(e.exception.code, 0)

            helpstring = stdout.getvalue().replace("\n", "  ")
            self.assertRegex(
                helpstring,
                r"usage: .* child1 grandchild1 \[-h\] \[--helpfull\] +{command1,command2}",
            )

    def test_invoke_basic(self):
        # Test invoking a command.
        args = _parse(
            [
                self.root_module,
                "child1",
                "grandchild1",
                "command1",
                "--required=test1",
                "--optional=test2",
            ]
        )
        self.assertEqual(
            args.argv,
            [
                self.root_module,
                "--required=test1",
                "--optional=test2",
                "--undefok=root,root_default,child1",
                "--root_default=some_value",
            ],
        )
        returncode, stdout, _ = _run(args)
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout, "required: test1, optional: test2, root_default: some_value")

    def test_invoke_required(self):
        # Test invoking a command without a required flag.
        args = _parse([self.root_module, "child1", "grandchild1", "command1"])
        self.assertEqual(
            args.argv,
            [self.root_module, "--undefok=root,root_default,child1", "--root_default=some_value"],
        )
        returncode, _, stderr = _run(args)
        self.assertEqual(returncode, 1)
        self.assertIn("Flag --required must have a value other than None", stderr)

        # Test invoking a command without a required command group flag (--child2).
        with mock.patch("sys.stderr", new=StringIO()) as stderr:
            with self.assertRaises(SystemExit) as e:
                _parse([self.root_module, "child2", "command2", "--required=test"])
            self.assertEqual(e.exception.code, 2)
            self.assertIn("the following arguments are required: --child2", stderr.getvalue())

    def test_invoke_unknown(self):
        # Test invoking a command with an unknown flag.
        args = _parse(
            [
                self.root_module,
                "child1",
                "grandchild1",
                "command1",
                "--required=test",
                "--unknown",
            ]
        )
        self.assertEqual(
            args.argv,
            [
                self.root_module,
                "--required=test",
                "--unknown",
                "--undefok=root,root_default,child1",
                "--root_default=some_value",
            ],
        )
        returncode, _, stderr = _run(args)
        self.assertEqual(returncode, 1)
        self.assertIn("Flags parsing error: Unknown command line flag 'unknown'", stderr)

    def test_invoke_undefok(self):
        # Test invoking a command with --child2, which has undefok=False.
        # Since the leaf command does not support this flag, it should fail.
        args = _parse([self.root_module, "child2", "--child2", "command2", "--required=test"])
        self.assertEqual(
            args.argv,
            [
                self.root_module,
                "--child2",
                "--required=test",
                "--undefok=root,root_default,child1",
                "--root_default=some_value",
            ],
        )
        returncode, _, stderr = _run(args)
        self.assertEqual(returncode, 1)
        self.assertIn("Flags parsing error: Unknown command line flag 'child2'", stderr)

        # Test invoking a command with --child1, which has undefok=True.
        # This means that even though the command doesn't support the flag, it shouldn't fail.
        args = _parse(
            [
                self.root_module,
                "child1",
                "--child1",
                "grandchild1",
                "command1",
                "--required=test",
            ]
        )
        self.assertEqual(
            args.argv,
            [
                self.root_module,
                "--child1",
                "--required=test",
                "--undefok=root,root_default,child1",
                "--root_default=some_value",
            ],
        )
        returncode, stdout, _ = _run(args)
        self.assertEqual(returncode, 0)
        self.assertEqual("required: test, optional: None, root_default: some_value", stdout)

        # Test invoking a command with --child1, which has undefok=True, in addition to --.
        args = _parse(
            [
                self.root_module,
                "child1",
                "--child1",
                "grandchild1",
                "command1",
                "--",
                "--required=test",
            ]
        )
        self.assertEqual(
            args.argv,
            [
                self.root_module,
                "--child1",
                "--undefok=root,root_default,child1",  # Should be inserted before --.
                "--root_default=some_value",  # Should be inserted before --.
                "--",
                "--required=test",
            ],
        )

    @parameterized.parameters(
        (["--root_default=override"],),
        (["--root_default", "override"],),
    )
    def test_invoke_default(self, flags: list[str]):
        # Test invoking with a flag for which a default already exists.
        args = _parse(
            [
                self.root_module,
                "child1",
                "grandchild1",
                "command1",
                "--required=test1",
            ]
            + flags
        )
        self.assertEqual(
            args.argv,
            [self.root_module, "--required=test1"] + flags + ["--undefok=root,root_default,child1"],
        )
        returncode, stdout, _ = _run(args)
        self.assertEqual(returncode, 0)
        self.assertEqual(stdout, "required: test1, optional: None, root_default: override")

        # If the flags appear after "--", they are ignored, so the default flags should be kept.
        args = _parse(
            [
                self.root_module,
                "child1",
                "grandchild1",
                "command1",
                "--required=test1",
                "--",
            ]
            + flags
        )
        self.assertEqual(
            args.argv,
            [
                self.root_module,
                "--required=test1",
                "--undefok=root,root_default,child1",
                "--root_default=some_value",  # Keep the default value.
                "--",
            ]
            + flags,  # Overrides come after "--".
        )

    def test_filter_argv(self):
        # Test invoking a command which specifies a flag filter.
        args = _parse(
            [
                self.root_module,
                "child1",
                "grandchild1",
                "command2",
                "--child1",
                # Tests that the regex does not match against flag value.
                "--required=root_default",
            ]
        )
        self.assertEqual(args.argv, [self.root_module, "--root_default=some_value"])

    def test_insert_flags(self):
        cases = [
            (
                [self.root_module],
                [self.root_module, "--i", "100"],
            ),
            (
                [self.root_module, "--a=1", "--b", "2"],
                [self.root_module, "--a=1", "--b", "2", "--i", "100"],
            ),
            (
                [self.root_module, "--a=1", "--b", "2", "--", self.root_module, "--c", "3"],
                [
                    self.root_module,
                    "--a=1",
                    "--b",
                    "2",
                    "--i",
                    "100",
                    "--",
                    self.root_module,
                    "--c",
                    "3",
                ],
            ),
            (
                [self.root_module, "--", self.root_module, "--", self.root_module],
                [
                    self.root_module,
                    "--i",
                    "100",
                    "--",
                    self.root_module,
                    "--",
                    self.root_module,
                ],
            ),
            (
                [self.root_module, "a", "b", " -- ", self.root_module],
                [self.root_module, "a", "b", "--i", "100", " -- ", self.root_module],
            ),
        ]
        for argv, expected in cases:
            result = _insert_flags(argv, ["--i", "100"])
            self.assertSequenceEqual(result, expected)

    @parameterized.parameters([dict(argv=[]), dict(argv=["--flag", "value"])])
    def test_insert_flags_argv0(self, argv: list[str]):
        with self.assertRaises(AssertionError):
            _insert_flags(argv, ["--i", "100"])

    def test_subprocess_argv(self):
        argv = [self.root_module, "--i", '"hello && world"', "--", "test", "command"]
        mock_args = mock.MagicMock(
            command_type=CommandType.MODULE,
            module=self.root_module,
            argv=argv,
        )
        expected = ["python3", "-m", mock_args.module] + argv[1:]

        # In the module case, test that argv retains quotes, e.g. "--i 'hello && world'".
        with mock.patch.object(subprocess, "Popen", autospec=True) as mock_popen:
            absl_main(mock_args)
            self.assertEqual(1, len(mock_popen.call_args_list))
            self.assertEqual((expected,), mock_popen.call_args[0])
            self.assertTrue({"text": True}.items() <= mock_popen.call_args[1].items())
            self.assertEqual(self.root_module, mock_popen.call_args[1]["env"]["AXLEARN_CLI_NAME"])

        shell_cases = [
            # Test a basic shell case.
            dict(
                command=f"{self.root_module} --i 'hello && world' -- test command",
                argv=[self.root_module],
                expected=f"{self.root_module} --i 'hello && world' -- test command",
            ),
            # Command should include supplied argv.
            dict(
                command=self.root_module,
                argv=[self.root_module, "--i", "hello && world", "--", "test", "command"],
                expected=f"{self.root_module} --i 'hello && world' -- test command",
            ),
        ]
        for test_case in shell_cases:
            mock_args = mock.MagicMock(
                command_type=CommandType.SHELL,
                command=test_case["command"],
                module=self.root_module,
                argv=test_case["argv"],
            )
            expected = test_case["expected"]
            with mock.patch.object(subprocess, "Popen", autospec=True) as mock_popen:
                absl_main(mock_args)
                self.assertEqual(1, len(mock_popen.call_args_list))
                self.assertEqual((expected,), mock_popen.call_args[0])
                self.assertTrue(
                    {"text": True, "shell": True}.items() <= mock_popen.call_args[1].items()
                )
                self.assertEqual(
                    self.root_module, mock_popen.call_args[1]["env"]["AXLEARN_CLI_NAME"]
                )

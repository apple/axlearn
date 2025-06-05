# Copyright © 2023 Apple Inc.

"""Tests general utils."""

import contextlib
import os
import pathlib
import shlex
import signal
import subprocess
import tempfile
import time
from collections.abc import Sequence
from typing import Optional, Union
from unittest import mock

import psutil
import pytest
from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud import ROOT_MODULE
from axlearn.cloud.common import utils
from axlearn.common.config import REQUIRED, ConfigBase, Configurable, Required, config_class
from axlearn.common.test_utils import TestWithTemporaryCWD


@contextlib.contextmanager
def _fake_module_root(module_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Make a top level module directory.
        temp_module = os.path.join(temp_dir, module_name)
        os.makedirs(temp_module)
        yield temp_module


class UtilsTest(TestWithTemporaryCWD):
    """Tests utils."""

    def test_get_package_root(self):
        with _fake_module_root("test_module") as temp_module:
            with self.assertRaisesRegex(ValueError, "Not running within"):
                utils.get_package_root(temp_module)

        self.assertEqual(str(ROOT_MODULE), utils.get_package_root())

    @parameterized.parameters(
        dict(
            kv_flags=["key1:value1", "key2:value2", "key1:value3"],
            expected=dict(
                key1="value3",
                key2="value2",
            ),
        ),
        dict(
            kv_flags=["key1=value1", "key2=value2", "key1=value3"],
            expected=dict(
                key1="value3",
                key2="value2",
            ),
            delimiter="=",
        ),
        dict(kv_flags=[], expected={}),
        dict(kv_flags=["malformatted"], expected=ValueError()),
        dict(kv_flags=["a:b"], delimiter="=", expected=ValueError()),
        dict(kv_flags=["a=b=c"], delimiter="=", expected=dict(a="b=c")),
    )
    def test_parse_kv_flags(
        self, kv_flags: Sequence[str], expected: Union[dict, Exception], delimiter: str = ":"
    ):
        if issubclass(type(expected), ValueError):
            with self.assertRaises(type(expected)):
                utils.parse_kv_flags(kv_flags, delimiter=delimiter)
        else:
            self.assertEqual(expected, utils.parse_kv_flags(kv_flags, delimiter=delimiter))

    def test_format_table(self):
        headings = ["COLUMN1", "LONG_COLUMN2", "COL3"]
        rows = [
            ["long_value1", 123, {"a": "dict"}],
            ["short", 12345678, {}],
        ]
        expected = (
            "\n"
            "COLUMN1          LONG_COLUMN2      COL3               \n"
            "long_value1      123               {'a': 'dict'}      \n"
            "short            12345678          {}                 "
            "\n"
        )
        self.assertEqual(expected, utils.format_table(headings=headings, rows=rows))

    @parameterized.parameters(
        dict(argv=["cli", "activate"], expected="activate"),
        dict(argv=["cli", "cleanup"], expected="cleanup"),
        dict(argv=["cli", "list", "something"], expected="list"),
        dict(argv=["cli", "--flag1", "activate"], expected="activate"),
        dict(argv=["cli"], default="list", expected="list"),
        dict(argv=["cli", "invalid"], default="list", expected="list"),
        dict(argv=["cli", "invalid", "activate"], default="list", expected="list"),
        # Test failure case.
        dict(argv=[], expected=app.UsageError("")),
        dict(argv=["cli", "invalid"], expected=app.UsageError("")),
    )
    def test_parse_action(self, argv, expected, default=None):
        options = ["activate", "list", "cleanup"]
        if isinstance(expected, BaseException):
            with self.assertRaises(type(expected)):
                utils.parse_action(argv, options=options, default=default)
        else:
            self.assertEqual(expected, utils.parse_action(argv, options=options, default=default))

    # TODO(markblee): Understand and fix flakiness on CI.
    @pytest.mark.skip(reason="Intended to be run manually, can be flaky in CI.")
    def test_send_signal(self):
        """Tests send_signal by starting a subprocess which has child subprocesses.

        Unlike p.kill(), send_signal(p, sig=signal.SIGKILL) should recursively kill the children,
        which does not leave orphan processes running. This test will fail by replacing
        send_signal(p, sig=signal.SIGKILL) with p.kill().
        """
        test_script = os.path.join(os.path.dirname(__file__), "testdata/counter.py")
        with tempfile.NamedTemporaryFile("r+") as f:

            def _read_count():
                f.seek(0, 0)
                return int(f.read())

            # pylint: disable-next=consider-using-with
            p = subprocess.Popen(
                shlex.split(f"python3 {test_script} {f.name} parent"), start_new_session=True
            )
            time.sleep(1)
            # Check that the count has incremented.
            self.assertGreater(_read_count(), 0)
            # Kill the subprocess.
            utils.send_signal(p, sig=signal.SIGKILL)
            # Get the count again, after kill has finished.
            count = _read_count()
            self.assertGreater(count, 0)
            # Wait for some ticks.
            time.sleep(1)
            # Ensure that the count is still the same.
            self.assertEqual(_read_count(), count)

    @mock.patch("psutil.Process")
    def test_send_signal_with_children_no_such_process(self, mock_process_class):
        # Create a mock parent process
        mock_process = mock.Mock()
        mock_process.pid = 1234

        # Create three mock child processes
        mock_child1 = mock.Mock()
        mock_child2 = mock.Mock()
        mock_child3 = mock.Mock()

        mock_child1.send_signal.side_effect = psutil.NoSuchProcess(pid=mock_child1.pid)
        mock_child2.send_signal.return_value = None
        mock_child3.send_signal.return_value = None

        mock_process.children.return_value = [mock_child1, mock_child2, mock_child3]

        # Ensure the parent process has the expected PID
        mock_process.pid = 1234
        mock_process_class.return_value = mock_process

        mock_popen = mock.Mock()
        mock_popen.pid = 1234

        utils.send_signal(mock_popen, signal.SIGKILL)

        mock_child1.send_signal.assert_called_once_with(signal.SIGKILL)
        mock_child2.send_signal.assert_called_once_with(signal.SIGKILL)
        mock_child3.send_signal.assert_called_once_with(signal.SIGKILL)

        mock_popen.send_signal.assert_called_once_with(signal.SIGKILL)

        try:
            utils.send_signal(mock_popen, signal.SIGKILL)
        except psutil.NoSuchProcess:
            self.fail("send_signal() raised NoSuchProcess unexpectedly!")

    # TODO(tom_gunter,markblee): Understand & fix flakiness on CI.
    @pytest.mark.skip(reason="Passes locally & in docker but fails on CI, to be fixed.")
    def test_copy_blobs(self):
        with tempfile.TemporaryDirectory() as read_dir:
            read_dir_path = pathlib.Path(read_dir)
            file_a = read_dir_path / "file.txt"
            file_a.touch()
            sub_directory = read_dir_path / "subdir"
            sub_directory.mkdir()
            file_b = sub_directory / "subdirfile.txt"
            file_b.touch()
            with tempfile.TemporaryDirectory() as write_dir:
                utils.copy_blobs("file://" + read_dir, to_prefix=write_dir)
                write_dir_path = pathlib.Path(write_dir)
                # Check that both files have been copied.
                copied_file_a = write_dir_path / file_a.relative_to(read_dir_path)
                self.assertTrue(copied_file_a.exists())
                copied_file_b = write_dir_path / file_b.relative_to(read_dir_path)
                self.assertTrue(copied_file_b.exists())

    @parameterized.parameters(
        dict(v_seq=[], v_str="", v_list=[], delimiter=","),
        dict(
            v_seq=["a", " b ", "c,d"], v_str="a,b,c,d", v_list=["a", "b", "c", "d"], delimiter=","
        ),
        dict(v_seq=["a", " b ", "c,d"], v_str="a.b.c,d", v_list=["a", "b", "c,d"], delimiter="."),
    )
    def test_canonicalize(self, v_seq: Sequence[str], v_str: str, v_list: str, delimiter: str):
        self.assertEqual(v_str, utils.canonicalize_to_string(v_str, delimiter=delimiter))
        self.assertEqual(v_str, utils.canonicalize_to_string(v_seq, delimiter=delimiter))
        self.assertEqual(v_list, utils.canonicalize_to_list(v_str, delimiter=delimiter))
        self.assertEqual(v_list, utils.canonicalize_to_list(v_list, delimiter=delimiter))

    @parameterized.parameters(
        dict(
            base={"a": {"d": None}, "c": [3, 4]},
            overrides={"a": {"b": 123}, "c": [1, 2]},
            expected={"a": {"b": 123, "d": None}, "c": [1, 2]},
        ),
    )
    def test_merge(self, base, overrides, expected):
        self.assertEqual(expected, utils.merge(base, overrides))

    def test_infer_resources(self):
        @config_class
        class DummyConfig(ConfigBase):
            name: Required[str] = REQUIRED
            inner: Optional[Sequence[ConfigBase]] = None
            resources: Optional[utils.AcceleratorConfig] = None

        cfg = DummyConfig(
            inner=[
                DummyConfig(
                    resources=utils.AcceleratorConfig(instance_type="tpu-v4-8", num_replicas=2),
                ),
                DummyConfig(
                    inner=[
                        DummyConfig(),
                        DummyConfig(
                            resources=utils.AcceleratorConfig(
                                instance_type="tpu-v5litepod-16", num_replicas=1
                            )
                        ),
                    ],
                ),
            ],
        )
        self.assertEqual({"v4": 16, "v5litepod": 16}, utils.infer_resources(cfg))


class TestTable(parameterized.TestCase):
    """Tests table utils."""

    @parameterized.parameters(
        dict(headings="a", rows=[1, 2, 3], expected=ValueError("sequence")),
        dict(headings=["a", "b", "c"], rows=1, expected=ValueError("sequence")),
        dict(headings=["a", "b", "c"], rows=[1, 2, 3], expected=ValueError("sequence")),
        dict(headings=["a", "b", "c"], rows=[[1, 2]], expected=ValueError("3 columns")),
        dict(headings=["a", "b", "c"], rows=[[1, 2, 3]], expected=None),  # OK.
    )
    def test_init(self, headings, rows, expected):
        if isinstance(expected, Exception):
            with self.assertRaisesRegex(type(expected), str(expected)):
                utils.Table(headings=headings, rows=rows)
        else:
            utils.Table(headings=headings, rows=rows)

    def test_add_row(self):
        table = utils.Table(headings=["a", "b", "c"], rows=[])
        table.add_row([1, 2, 3])
        # fmt: off
        self.assertEqual(
            (
                "\n"
                "A      B      C      \n"
                "1      2      3      \n"
            ),
            str(table),
        )
        # fmt: on
        with self.assertRaisesRegex(ValueError, "3 columns"):
            table.add_row([1, 2])

    def test_add_col(self):
        table = utils.Table(headings=["a", "b", "c"], rows=[[1, 2, 3]])
        table.add_col("d", [4])
        # fmt: off
        self.assertEqual(
            (
                "\n"
                "A      B      C      D      \n"
                "1      2      3      4      \n"
            ),
            str(table),
        )
        # fmt: on
        with self.assertRaisesRegex(ValueError, "1 row"):
            table.add_col("e", [5, 6])

    def test_get_col(self):
        table = utils.Table(headings=["a", "b"], rows=[[1, 2], [3, 4]])
        self.assertEqual([[1], [3]], table.get_col("a"))
        self.assertEqual([[1, 2], [3, 4]], table.get_col("a", "b"))
        with self.assertRaisesRegex(ValueError, ""):
            table.get_col("c")

    def test_sort(self):
        table = utils.Table(headings=["a", "b"], rows=[[1, 4], [3, 2]])
        table.sort(key=lambda row: row[-1])
        self.assertEqual(
            utils.Table(headings=["a", "b"], rows=[[3, 2], [1, 4]]),
            table,
        )
        table.sort(key=lambda row: row[-1], reverse=True)
        self.assertEqual(
            utils.Table(headings=["a", "b"], rows=[[1, 4], [3, 2]]),
            table,
        )


class FlagConfigurableTest(parameterized.TestCase):
    """Tests FlagConfigurable."""

    def test_set_defaults(self):
        class Parent(utils.FlagConfigurable):
            """A parent class."""

            @classmethod
            def define_flags(cls, fv):
                flags.DEFINE_string("shared", None, "", flag_values=fv, allow_override=True)
                flags.DEFINE_string("parent_only", None, "", flag_values=fv, allow_override=True)

            @classmethod
            def set_defaults(cls, fv: flags.FlagValues):
                super().set_defaults(fv)
                fv.set_default("shared", "parent-default")
                fv.set_default("parent_only", "parent-default")
                fv.set_default("shared_override", "parent-default")

        class Child(Parent):
            """A child class."""

            @classmethod
            def define_flags(cls, fv):
                super().define_flags(fv)
                flags.DEFINE_string("shared", None, "", flag_values=fv, allow_override=True)
                flags.DEFINE_string("child_only", None, "", flag_values=fv, allow_override=True)
                flags.DEFINE_string(
                    "shared_override", None, "", flag_values=fv, allow_override=True
                )

            @classmethod
            def set_defaults(cls, fv: flags.FlagValues):
                super().set_defaults(fv)
                fv.set_default("shared_override", "child-default")
                fv.set_default("child_only", "child-default")

        fv = flags.FlagValues()
        cfg = Child.default_config()
        utils.define_flags(cfg, fv)
        fv.mark_as_parsed()
        utils.from_flags(cfg, fv)

        # "parent-only" and "child-only" should follow original defaults.
        self.assertEqual(fv.parent_only, "parent-default")
        self.assertEqual(fv.child_only, "child-default")

        # "shared" should follow parent default, because it is not overridden.
        # Note that this is the case even though child defines the same flag.
        self.assertEqual(fv.shared, "parent-default")

        # "shared_override" should follow child default, because it is overridden.
        self.assertEqual(fv.shared_override, "child-default")

    def test_flag_utils(self):
        """Tests define_flags and from_flags."""

        class Inner(utils.FlagConfigurable):
            """An inner config."""

            @config_class
            class Config(utils.FlagConfigurable.Config):
                common_value: Required[str] = REQUIRED
                inner_value: Required[str] = REQUIRED

            @classmethod
            def define_flags(cls, fv):
                super().define_flags(fv)
                common_kwargs = dict(flag_values=fv, allow_override=True)
                flags.DEFINE_string("common_value", None, "", **common_kwargs)
                flags.DEFINE_string("inner_value", None, "", **common_kwargs)

            @classmethod
            def set_defaults(cls, fv):
                super().set_defaults(fv)
                fv.set_default("common_value", "child-default")

        # Test that it traverses non FlagConfigurables.
        class RegularConfigurable(Configurable):
            """A dummy container config to test traversal."""

            @config_class
            class Config(Configurable.Config):
                # Test that it traverses other non-config containers.
                inner: list[Inner.Config] = [Inner.default_config()]

        class Outer(utils.FlagConfigurable):
            """An outer config."""

            @config_class
            class Config(utils.FlagConfigurable.Config):
                common_value: Required[str] = REQUIRED
                outer_value: Required[str] = REQUIRED
                inner_enabled: Optional[bool] = None
                inner: Optional[Configurable.Config] = RegularConfigurable.default_config()

            @classmethod
            def define_flags(cls, fv):
                super().define_flags(fv)
                common_kwargs = dict(flag_values=fv, allow_override=True)
                flags.DEFINE_string("common_value", None, "", **common_kwargs)
                flags.DEFINE_string("outer_value", None, "", **common_kwargs)
                flags.DEFINE_bool("inner_enabled", None, "", **common_kwargs)

            @classmethod
            def set_defaults(cls, fv):
                super().set_defaults(fv)
                fv.set_default("common_value", "parent-default")

            @classmethod
            def from_flags(cls, fv, **kwargs):
                cfg = super().from_flags(fv, **kwargs)
                if not fv.inner_enabled:
                    cfg.inner = None
                return cfg

        cfg = Outer.default_config()
        fv = flags.FlagValues()
        utils.define_flags(cfg, fv)
        fv.mark_as_parsed()

        # Check that both outer and inner are defined.
        self.assertIn("inner_value", fv)
        self.assertIn("outer_value", fv)

        fv.outer_value = "outer-value"
        fv.inner_value = "inner-value"
        fv.inner_enabled = True
        cfg_with_inner = utils.from_flags(cfg.clone(), fv)

        # Check that from_flags respects set_default override.
        self.assertEqual("child-default", cfg_with_inner.common_value)
        self.assertEqual("child-default", cfg_with_inner.inner.inner[0].common_value)
        # Check that flag values are set.
        self.assertEqual("outer-value", cfg_with_inner.outer_value)
        self.assertEqual("inner-value", cfg_with_inner.inner.inner[0].inner_value)

        # Check that cfg can be modified in from_flags.
        fv.inner_enabled = False
        cfg_no_inner = utils.from_flags(cfg.clone(), fv)
        self.assertIsNone(cfg_no_inner.inner)

    def test_namespaced(self):
        # pylint: disable=missing-class-docstring
        fv = flags.FlagValues()

        class GrandChild(utils.FlagConfigurable):
            @config_class
            class Config(utils.FlagConfigurable.Config):
                grandchild: Optional[str] = None
                child_default: Optional[str] = None
                parent_default: Optional[str] = None

            @classmethod
            def define_flags(cls, fv: flags.FlagValues):
                flags.DEFINE_string("grandchild", None, "", flag_values=fv, allow_override=True)
                flags.DEFINE_string("child_default", None, "", flag_values=fv, allow_override=True)
                flags.DEFINE_string("parent_default", None, "", flag_values=fv, allow_override=True)

        @utils.namespaced("inner")
        class Child(utils.FlagConfigurable):
            @config_class
            class Config(utils.FlagConfigurable.Config):
                inner: Required[dict[str, ConfigBase]] = REQUIRED
                child: Optional[str] = None
                child_default: Optional[str] = None
                parent_default: Optional[str] = None

            @classmethod
            def default_config(cls) -> Config:
                return super().default_config().set(inner={"c": GrandChild.default_config()})

            @classmethod
            def define_flags(cls, fv: flags.FlagValues):
                flags.DEFINE_string("child", None, "", flag_values=fv, allow_override=True)
                flags.DEFINE_string(
                    "child_default", "from_child", "", flag_values=fv, allow_override=True
                )

        @utils.namespaced("inner")
        class Parent(utils.FlagConfigurable):
            @config_class
            class Config(utils.FlagConfigurable.Config):
                inner: Required[dict[str, ConfigBase]] = REQUIRED
                parent: Optional[str] = None
                parent_default: Optional[str] = None

            @classmethod
            def default_config(cls):
                return (
                    super()
                    .default_config()
                    .set(inner={"a": Child.default_config(), "b": Child.default_config()})
                )

            @classmethod
            def define_flags(cls, fv: flags.FlagValues):
                flags.DEFINE_string("parent", None, "", flag_values=fv, allow_override=True)
                flags.DEFINE_string(
                    "parent_default", "from_parent", "", flag_values=fv, allow_override=True
                )

        # Test defining flags.
        cfg = Parent.default_config()
        utils.define_flags(cfg, fv)

        # 1. Inner flags should be namespaced.
        # 2. Defaults should be defined.
        self.assertSameElements(
            {
                # Parent flags.
                "parent": None,
                "parent_default": "from_parent",
                # Child 'a' flags.
                "a.child": None,
                "a.child_default": "from_inner",
                # GrandChild 'a.c' flags.
                "a.c.grandchild": None,
                "a.c.child_default": None,
                "a.c.parent_default": None,
                # Child 'b' flags.
                "b.child": None,
                "b.child_default": "from_inner",
                # GrandChild 'b.c' flags.
                "b.c.grandchild": None,
                "b.c.child_default": None,
                "b.c.parent_default": None,
            },
            fv.flag_values_dict(),
        )

        # Set some flags.
        setattr(fv, "parent", "set_parent")
        setattr(fv, "a.child", "set_a_child")
        setattr(fv, "b.child_default", "set_b_default")
        setattr(fv, "a.c.child_default", "set_a_c_default")
        setattr(fv, "b.c.grandchild", "set_b_c_grandchild")

        utils.from_flags(cfg, fv)

        # 1. Parent flags.
        self.assertEqual(cfg.parent, "set_parent")
        self.assertEqual(cfg.parent_default, "from_parent")
        # 2. Child 'a' flags.
        self.assertEqual(cfg.inner["a"].child, "set_a_child")  # set.
        self.assertEqual(cfg.inner["a"].child_default, "from_child")  # default.
        self.assertEqual(cfg.inner["a"].parent_default, "from_parent")  # inherited.
        # 3. Child 'b' flags.
        self.assertEqual(cfg.inner["b"].child, None)  # not set.
        self.assertEqual(cfg.inner["b"].child_default, "set_b_default")  # overridden.
        self.assertEqual(cfg.inner["b"].parent_default, "from_parent")  # inherited.
        # 4. GrandChild 'a.c' flags.
        self.assertEqual(cfg.inner["a"].inner["c"].grandchild, None)  # not set.
        self.assertEqual(cfg.inner["a"].inner["c"].child_default, "set_a_c_default")  # overridden.
        self.assertEqual(cfg.inner["a"].inner["c"].parent_default, "from_parent")  # inherited.
        # 5. GrandChild 'b.c' flags.
        self.assertEqual(cfg.inner["b"].inner["c"].grandchild, "set_b_c_grandchild")  # set.
        self.assertEqual(cfg.inner["b"].inner["c"].child_default, "set_b_default")  # inherited.
        self.assertEqual(cfg.inner["b"].inner["c"].parent_default, "from_parent")  # inherited.

    def test_invalid_namespace(self):
        # pylint: disable=missing-class-docstring
        fv = flags.FlagValues()

        @utils.namespaced("nonexistent")
        class NonExistent(utils.FlagConfigurable):
            pass

        cfg = NonExistent.default_config()
        with self.assertRaisesRegex(ValueError, "mapping at nonexistent"):
            utils.define_flags(cfg, fv)

        @utils.namespaced("inner")
        class NotMapping(utils.FlagConfigurable):
            @config_class
            class Config(utils.FlagConfigurable.Config):
                inner: Required[str] = REQUIRED

        # Test when inner is not a mapping.
        cfg = NotMapping.default_config()
        with self.assertRaisesRegex(ValueError, "mapping at inner"):
            utils.define_flags(cfg, fv)

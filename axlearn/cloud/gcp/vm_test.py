# Copyright © 2023 Apple Inc.

"""Tests VM utilities."""

from absl.testing import parameterized

from axlearn.cloud.gcp.vm import VmInfo, format_vm_info, get_vm_node_status


class VmUtilsTest(parameterized.TestCase):
    """Tests VM utils."""

    @parameterized.parameters(
        dict(node={}, expected="UNKNOWN"),
        dict(node={"status": "RUNNING"}, expected="RUNNING"),
        dict(node={"status": "RUNNING", "labels": {"boot_status": ""}}, expected="RUNNING"),
        dict(node={"status": "RUNNING", "labels": {"boot_status": "done"}}, expected="BOOTED"),
        dict(
            node={"status": "RUNNING", "labels": {"boot_status": "failed"}}, expected="BOOT_FAILED"
        ),
    )
    def test_get_vm_node_status(self, node, expected):
        self.assertEqual(expected, get_vm_node_status(node))

    def test_format_vm_info(self):
        vms = [
            VmInfo(name="test2", metadata={"a": 123}),
            VmInfo(name="test1", metadata={"a": 123, "b": 234}),
        ]
        # List without metadata.
        self.assertEqual(
            ("\n" "NAME       \n" "test1      \n" "test2      " "\n"),
            format_vm_info(vms, metadata=None),
        )
        # List with metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       METADATA        \n"
                "test1      {'a': 123}      \n"
                "test2      {'a': 123}      "
                "\n"
            ),
            format_vm_info(vms, metadata=["a"]),
        )
        # List with metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       METADATA         \n"
                "test1      {'b': 234}       \n"
                "test2      {'b': None}      "
                "\n"
            ),
            format_vm_info(vms, metadata=["b"]),
        )

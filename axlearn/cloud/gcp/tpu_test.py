# Copyright © 2023 Apple Inc.

"""Tests TPU utilities."""

from absl.testing import parameterized

from axlearn.cloud.gcp.tpu import (
    QueuedResourceInfo,
    TpuInfo,
    format_queued_resource_info,
    format_tpu_info,
    infer_tpu_cores,
    infer_tpu_version,
    infer_tpu_workers,
)


class TpuUtilsTest(parameterized.TestCase):
    """Tests TPU utils."""

    @parameterized.parameters(
        dict(tpu_type="v4-128", version="v4", cores=128, workers=16),
        dict(tpu_type="v5litepod-16", version="v5litepod", cores=16, workers=4),
        dict(tpu_type="v3-64", version="v3", cores=64, workers=8),
    )
    def test_infer_utils(self, tpu_type: str, version: str, cores: int, workers: int):
        self.assertEqual(version, infer_tpu_version(tpu_type))
        self.assertEqual(cores, infer_tpu_cores(tpu_type))
        self.assertEqual(workers, infer_tpu_workers(tpu_type))

    def test_unknown_tpu_version(self):
        with self.assertRaisesRegex(ValueError, "Unknown TPU version"):
            infer_tpu_version("v5lite-16")

    def test_format_tpu_info(self):
        tpus = [
            TpuInfo(
                name="test2",
                accelerator_type="v4-32",
                state="READY",
                metadata={"a": 123},
            ),
            TpuInfo(
                name="test1",
                accelerator_type="v4-8",
                state="PENDING",
                metadata={"a": 123, "b": 234},
            ),
        ]
        # List without metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       ACCELERATOR_TYPE      STATE        \n"
                "test1      v4-8                  PENDING      \n"
                "test2      v4-32                 READY        "
                "\n"
            ),
            format_tpu_info(tpus, metadata=None),
        )
        # List with metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       ACCELERATOR_TYPE      STATE        METADATA        \n"
                "test1      v4-8                  PENDING      {'a': 123}      \n"
                "test2      v4-32                 READY        {'a': 123}      "
                "\n"
            ),
            format_tpu_info(tpus, metadata=["a"]),
        )
        # List with metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       ACCELERATOR_TYPE      STATE        METADATA         \n"
                "test1      v4-8                  PENDING      {'b': 234}       \n"
                "test2      v4-32                 READY        {'b': None}      "
                "\n"
            ),
            format_tpu_info(tpus, metadata=["b"]),
        )

    def test_format_queued_resource_info(self):
        tpus = [
            QueuedResourceInfo(
                name="test2",
                accelerator_type="v4-32",
                state="READY",
                metadata={"a": 123},
                num_slices=1,
                reserved=False,
            ),
            QueuedResourceInfo(
                name="test1",
                accelerator_type="v4-8",
                state="PENDING",
                metadata={"a": 123, "b": 234},
                num_slices=1,
                reserved=True,
            ),
        ]
        # pylint: disable=line-too-long
        # List without metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       ACCELERATOR_TYPE      STATE        NUM_SLICES      RESERVED      \n"
                "test1      v4-8                  PENDING      1               True          \n"
                "test2      v4-32                 READY        1               False         "
                "\n"
            ),
            format_queued_resource_info(tpus, metadata=None),
        )
        # List with metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       ACCELERATOR_TYPE      STATE        METADATA        NUM_SLICES      RESERVED      \n"
                "test1      v4-8                  PENDING      {'a': 123}      1               True          \n"
                "test2      v4-32                 READY        {'a': 123}      1               False         "
                "\n"
            ),
            format_queued_resource_info(tpus, metadata=["a"]),
        )
        # List with metadata.
        self.assertEqual(
            (
                "\n"
                "NAME       ACCELERATOR_TYPE      STATE        METADATA         NUM_SLICES      RESERVED      \n"
                "test1      v4-8                  PENDING      {'b': 234}       1               True          \n"
                "test2      v4-32                 READY        {'b': None}      1               False         "
                "\n"
            ),
            format_queued_resource_info(tpus, metadata=["b"]),
        )
        # pylint: enable=line-too-long

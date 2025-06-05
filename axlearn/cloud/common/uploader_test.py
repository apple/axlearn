# Copyright © 2023 Apple Inc.

"""Tests artifact uploader."""
# pylint: disable=protected-access

from unittest import mock

from absl.testing import parameterized

from axlearn.cloud.common.uploader import Uploader
from axlearn.common.config import config_for_function


class UploaderTest(parameterized.TestCase):
    """Tests Uploader."""

    def test_upload(self):
        mock_upload_fn = mock.Mock()
        mock_proc = mock.Mock()
        mock_proc.is_alive.side_effect = [True, False]
        with mock.patch("multiprocessing.Process", return_value=mock_proc) as mock_multiproc:
            cfg = Uploader.default_config().set(
                src_dir="test-src",
                dst_dir="test-dst",
                upload_fn=config_for_function(lambda: mock_upload_fn),
            )
            up = cfg.instantiate()

            # First call should start the process.
            up()
            self.assertEqual(mock_proc.start.call_count, 1)
            self.assertIsNotNone(up._upload_proc)
            self.assertEqual(
                dict(src_dir="test-src", dst_dir="test-dst"),
                mock_multiproc.call_args.kwargs["kwargs"],
            )

            # Second call should do nothing, since process is still alive.
            up()
            self.assertEqual(mock_proc.start.call_count, 1)
            self.assertIsNotNone(up._upload_proc)

            # Third call should restart the process.
            up()
            self.assertEqual(mock_proc.start.call_count, 2)
            self.assertIsNotNone(up._upload_proc)

            # Cleanup the upload_proc
            up.cleanup()
            self.assertEqual(mock_proc.terminate.call_count, 1)
            self.assertEqual(mock_proc.join.call_count, 4)
            self.assertIsNone(up._upload_proc)

# Copyright © 2023 Apple Inc.

"""Tests GCS utilities.

Example usage:

    python3 -m axlearn.cloud.gcp.storage_test --project=my-project

"""

import os
import subprocess
import uuid

import pytest
from absl.testing import absltest, parameterized

from axlearn.cloud.gcp.storage import blob_exists, delete_blob, list_blobs, upload_blob
from axlearn.cloud.gcp.utils import common_flags


# pylint: disable=no-self-use
@pytest.mark.gs_login
class GsUtilsTest(parameterized.TestCase):
    """Tests GCS utils."""

    def test_list_blobs_exact_match(self):
        # Check for full match.
        url = "gs://axlearn-public/testdata/gcp_test/test.txt"
        blobs = list_blobs(url)
        assert len(blobs) == 1
        assert blobs.pop() == url

    def test_blob_exists(self):
        url = "gs://axlearn-public/testdata/gcp_test/test"
        assert not blob_exists(url)
        url = "gs://axlearn-public/testdata/gcp_test/test.txt"
        assert blob_exists(url)

    def test_list_blobs_non_exact_match(self):
        # Check for partial match.
        url = "gs://axlearn-public/testdata/gcp_test"
        blobs = list_blobs(url)
        assert len(blobs) > 0
        assert os.path.join(url, "test.txt") in blobs

    def test_delete_blob(self):
        filename = uuid.uuid4().hex.lower()[:6]
        local_path = f"/tmp/{filename}"
        subprocess.check_call(["touch", local_path])
        url = f"gs://axlearn-public/testdata/gcp_test/tmp/{filename}"
        upload_blob(local_path, url=url)
        delete_blob(url)
        url = f"gs://axlearn-public/testdata/gcp_test/tmp/{filename}"
        assert not list_blobs(url)

    def test_upload_blob(self):
        filename = uuid.uuid4().hex.lower()[:6]
        local_path = f"/tmp/{filename}"
        subprocess.check_call(["touch", local_path])
        url = f"gs://axlearn-public/testdata/gcp_test/tmp/{filename}"
        # Delete any existing blob.
        if blob_exists(url):
            delete_blob(url)
        assert not list_blobs(url)
        upload_blob(local_path, url=url)
        assert len(list_blobs(url)) == 1


if __name__ == "__main__":
    common_flags()
    absltest.main()

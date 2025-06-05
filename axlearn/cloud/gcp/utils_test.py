# Copyright © 2023 Apple Inc.

"""Tests general GCP utils."""

import contextlib
import os
import tempfile
from unittest import mock

from absl import app
from absl.testing import parameterized

from axlearn.cloud.gcp import utils


class UtilsTest(parameterized.TestCase):
    """Tests utils."""

    @parameterized.parameters(
        dict(name="test--01-exp123", should_raise=False),
        dict(name="123-test", should_raise=True),  # Must begin with letter.
        dict(name="test+123", should_raise=True),  # No other special characters allowed.
        dict(name="a" * 64, should_raise=True),  # Too long.
    )
    def test_validate_resource_name(self, name: str, should_raise: bool):
        if should_raise:
            with self.assertRaises(ValueError):
                utils.validate_resource_name(name)
        else:
            utils.validate_resource_name(name)

    @parameterized.parameters(
        # OK.
        dict(name="test-job0123", num_workers=1, num_replicas=1, ok=True),
        # Too long.
        dict(name="a" * 64, num_workers=1, num_replicas=1, ok=False),
        dict(name="a" * 48, num_workers=10, num_replicas=10, ok=False),
        dict(name="a" * 48, num_workers=100, num_replicas=1, ok=False),
        # Invalid chars.
        dict(name="a_b", num_workers=1, num_replicas=1, ok=False),
        dict(name="a/b", num_workers=1, num_replicas=1, ok=False),
        dict(name="-ab", num_workers=1, num_replicas=1, ok=False),
    )
    def test_validate_jobset_name(self, name, num_workers, num_replicas, ok):
        if ok:
            ctx = contextlib.nullcontext()
        else:
            ctx = self.assertRaisesRegex(ValueError, "Job name")
        with ctx:
            utils.validate_jobset_name(name, num_workers=num_workers, num_replicas=num_replicas)

    @parameterized.product(
        running_from_gcp=[False, True],
        raise_config_exc=[False, True],
    )
    def test_load_kube_config(self, running_from_gcp: bool, raise_config_exc: bool):
        # pylint: disable-next=import-error,import-outside-toplevel
        import kubernetes as k8s  # pytype: disable=import-error

        if raise_config_exc:
            side_effect = [k8s.config.config_exception.ConfigException, None]
        else:
            side_effect = [None, None]

        with tempfile.TemporaryDirectory() as d:
            patch_env = mock.patch.multiple(
                utils.__name__,
                running_from_k8s=mock.Mock(return_value=running_from_gcp),
                running_from_vm=mock.Mock(return_value=running_from_gcp),
                subprocess_run=mock.DEFAULT,
            )
            patch_expanduser = mock.patch(
                "os.path.expanduser", return_value=os.path.join(d, "user")
            )
            patch_kube_config = mock.patch(
                "kubernetes.config.load_kube_config", side_effect=side_effect
            )
            with patch_expanduser, patch_env as mock_env, patch_kube_config as mock_kube_config:
                if raise_config_exc and not running_from_gcp:
                    ctx = self.assertRaisesRegex(app.UsageError, "kube-config")
                else:
                    ctx = contextlib.nullcontext()

                with ctx:
                    utils.load_kube_config(
                        project="my-project", zone="us-test1-a", cluster="my-cluster"
                    )
                self.assertTrue(mock_kube_config.called)
                self.assertEqual(
                    "gke_my-project_us-test1_my-cluster", mock_kube_config.call_args[1]["context"]
                )
                if running_from_gcp:
                    self.assertEqual(raise_config_exc, mock_env["subprocess_run"].called)

    @parameterized.parameters(
        dict(project=None, zone="test", cluster="test"),
        dict(project="test", zone=None, cluster="test"),
        dict(project="test", zone="test", cluster=None),
    )
    def test_load_kube_config_none(self, **kwargs):
        with self.assertRaisesRegex(app.UsageError, "must all be specified"):
            utils.load_kube_config(**kwargs)

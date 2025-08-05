"""Tests k8s service module."""

from unittest import mock

from absl import flags

from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp import k8s_service
from axlearn.common.test_utils import TestCase

FLAGS = flags.FLAGS


class GKELWSService(TestCase):
    """Tests GKEService with LWS(TPU)."""

    def _service_config(
        self,
        *,
        command: str,
        **kwargs,
    ) -> tuple[k8s_service.LWSService.Config]:
        fv = flags.FlagValues()
        cfg = k8s_service.LWSService.default_config().set()

        define_flags(cfg, fv)
        for key, value in kwargs.items():
            if value is not None:
                # Use setattr rather than set_default to set flags.
                setattr(fv, key, value)
        fv.name = "fake-name"
        fv.project = "fake-project"
        fv.mark_as_parsed()
        from_flags(cfg, fv, command=command)
        # Test that retries are configured on fv by default.
        self.assertIsNotNone(fv["name"])
        return cfg

    def test_instantiate(
        self,
    ):
        cfg = self._service_config(
            command="test-command",
        )
        self.assertIsInstance(cfg, k8s_service.LWSService.Config)
        self.assertEqual(cfg.project, "fake-project")
        gke_lws_service = cfg.set().instantiate()
        self.assertEqual(cfg.name + "-service", gke_lws_service.name)
        self.assertEqual(cfg.port, gke_lws_service.port)

    def test_delete(self):
        patch_delete = mock.patch(f"{k8s_service.__name__}.delete_k8s_service")
        with patch_delete as mock_delete:
            cfg = self._service_config(command="test-command")
            gke_service = cfg.set().instantiate()
            print(gke_service.name)
            gke_service._delete()  # pylint: disable=protected-access
            mock_delete.assert_called()

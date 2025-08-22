# Copyright © 2023 Apple Inc.

"""Tests VertexAI Tensorboard tools."""

from unittest import mock

from absl import flags
from absl.testing import absltest
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.tensorboard import uploader, uploader_utils

from axlearn.cloud.gcp import config as gcp_config
from axlearn.cloud.gcp import test_utils
from axlearn.cloud.gcp.vertexai_tensorboard import (
    VertexAITensorboardUploader,
    _vertexai_experiment_name_from_output_dir,
)


def fake_process(target, *args, kwargs, **rest):
    del args, rest

    class FakeProcess:
        def start(self):  # pylint: disable=no-self-use
            return target(**kwargs)

    return FakeProcess()


class VertexAITensorboardUploaderTest(absltest.TestCase):
    """Tests VertexAITensorboardUploader."""

    @mock.patch("resource.setrlimit")
    @mock.patch("multiprocessing.get_context")
    @mock.patch(f"{uploader.TensorBoardUploader.__module__}.TensorBoardUploader", autospec=True)
    @mock.patch(
        f"{initializer.global_config.__module__}.global_config.create_client",
        autospec=True,
    )
    @mock.patch(
        (
            f"{uploader_utils.get_blob_storage_bucket_and_folder.__module__}"
            ".get_blob_storage_bucket_and_folder"
        ),
        autospec=True,
        return_value=("fake_bucket", "fake_folder"),
    )
    def test_uploader_calls(
        self,
        bucket_folder_fn,
        create_client_fn,
        tb_uploader_class,
        mock_get_context,
        mock_set_resource_limit,
    ):  # pylint: disable=no-self-use
        mock_context = mock.MagicMock()
        mock_context.Process.side_effect = fake_process
        mock_get_context.return_value = mock_context

        mock_settings = {
            "vertexai_tensorboard": "fake_tb_instance",
            "vertexai_region": "us-west4",
            "project": "fake_project_id",
        }
        with test_utils.mock_gcp_settings(gcp_config.__name__, settings=mock_settings):
            fv = flags.FlagValues()
            cfg = VertexAITensorboardUploader.from_flags(fv).set(
                summary_dir="gs://fake/summary_dir"
            )
        tb_uploader = cfg.instantiate()
        tb_uploader.upload()
        mock_set_resource_limit.assert_called_once()
        mock_get_context.assert_called_once_with("spawn")
        create_client_fn.assert_called_once()
        bucket_folder_fn.assert_called_once()
        tb_uploader_class.return_value.create_experiment.assert_called_once()
        tb_uploader_class.return_value.start_uploading.assert_called_once()


class ExperimentNameTest(absltest.TestCase):
    def test_exp_name_len(self):
        with self.assertRaises(ValueError):
            _vertexai_experiment_name_from_output_dir("gs://abc/" + "a" * 128)

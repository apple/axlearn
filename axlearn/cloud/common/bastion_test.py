# Copyright © 2023 Apple Inc.

"""Tests bastion orchestrator."""

# pylint: disable=no-self-use,protected-access
# pytype: disable=wrong-arg-types
import contextlib
import copy
import itertools
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence
from unittest import mock

from absl.testing import absltest, parameterized

from axlearn.cloud.common import bastion
from axlearn.cloud.common.bastion import (
    _JOB_DIR,
    _LOG_DIR,
    Bastion,
    BastionDirectory,
    Job,
    JobState,
    JobStatus,
    ValidationError,
    _download_job_state,
    _load_runtime_options,
    _PipedProcess,
    _upload_job_state,
    _validate_jobspec,
    deserialize_jobspec,
    download_job_batch,
    is_valid_job_name,
    new_jobspec,
    serialize_jobspec,
    set_runtime_options,
)
from axlearn.cloud.common.cleaner import Cleaner
from axlearn.cloud.common.quota import UserQuotaInfo
from axlearn.cloud.common.scheduler import JobScheduler
from axlearn.cloud.common.scheduler_test import mock_quota_config
from axlearn.cloud.common.types import JobMetadata, JobSpec, ResourceMap
from axlearn.cloud.common.uploader import Uploader
from axlearn.common.config import config_for_function


class TestDownloadJobBatch(parameterized.TestCase):
    """Tests download utils."""

    def test_download_job_batch(self):
        spec_dir = "gs://test_spec_dir"
        state_dir = "gs://test_state_dir"
        user_state_dir = "gs://user_state_dir"

        user_states = {
            "job_test1": JobState(
                status=JobStatus.CANCELLING,
                metadata={"ignored": 123},
            ),
            "job_test2": JobState(status=JobStatus.ACTIVE, metadata={"ignored": 123}),
            "job_test0": JobState(status=JobStatus.CANCELLING),
            "job_test3": JobState(status=JobStatus.CANCELLING),
        }
        states = {
            "job_test1": JobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
            "job_test0": JobState(status=JobStatus.CLEANING, metadata={"tier": 0}),
            "job_test3": JobState(status=JobStatus.COMPLETED, metadata={"tier": 1}),
            "job_test4": JobState(status=JobStatus.PENDING),
        }
        jobspecs = {
            "job_test2": mock.Mock(),
            "job_test1": mock.Mock(),
            "job_test0": mock.Mock(),
            "job_test3": mock.Mock(),
            "job_test4": mock.Mock(),
        }
        expected = {
            # User state is invalid and is ignored. Job state defaults to PENDING, since it's
            # missing a state.
            "job_test2": JobState(status=JobStatus.PENDING),
            # User state should take effect. Note that we do not read metadata from user states.
            "job_test1": JobState(status=JobStatus.CANCELLING, metadata={"tier": 0}),
            # User state should not affect CLEANING/COMPLETED.
            "job_test0": JobState(status=JobStatus.CLEANING, metadata={"tier": 0}),
            "job_test3": JobState(status=JobStatus.COMPLETED, metadata={"tier": 1}),
            # Has no user state.
            "job_test4": JobState(status=JobStatus.PENDING),
        }

        def mock_listdir(path):
            if path == spec_dir:
                return list(jobspecs.keys())
            if path == state_dir:
                return list(states.keys())
            if path == user_state_dir:
                return list(user_states.keys())
            assert False  # Should not be reached.

        def mock_download_jobspec(job_name, **kwargs):
            del kwargs
            return jobspecs[job_name]

        def mock_download_job_state(job_name, *, remote_dir, **kwargs):
            del kwargs
            if remote_dir == state_dir:
                # Job state may be initially missing, thus defaults to PENDING.
                return states.get(job_name, JobState(status=JobStatus.PENDING))
            if remote_dir == user_state_dir:
                # We should only query user states if one exists, so don't use get().
                return user_states[job_name]
            assert False  # Should not be reached.

        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _download_jobspec=mock.Mock(side_effect=mock_download_jobspec),
            _download_job_state=mock.Mock(side_effect=mock_download_job_state),
            _listdir=mock.Mock(side_effect=mock_listdir),
            _remove=mock.DEFAULT,
        )
        # Ensure that results are in the right order and pairing.
        with patch_fns as mock_fns, tempfile.TemporaryDirectory() as tmpdir:
            jobs, jobs_with_user_states = download_job_batch(
                spec_dir=spec_dir,
                state_dir=state_dir,
                user_state_dir=user_state_dir,
                local_spec_dir=tmpdir,
                remove_invalid_job_specs=True,
            )
            self.assertSameElements(expected.keys(), jobs.keys())
            # "job_test1" is the only valid user state, but we still cleanup the others.
            self.assertSameElements(jobs_with_user_states, user_states.keys())
            for job_name, job in jobs.items():
                self.assertEqual(job.state, expected[job_name])
                self.assertEqual(job.spec, jobspecs[job_name])
            # Make sure we do not remove any valid jobspecs.
            self.assertFalse(mock_fns["_remove"].called)

    @parameterized.parameters(
        dict(name="", valid=False),
        dict(name="test", valid=True),
        dict(name=".", valid=False),
        dict(name="..", valid=False),
        dict(name="test/dir", valid=False),
        dict(name="..test", valid=True),  # This is a valid file name.
        dict(name="test.job..", valid=True),  # This is a valid file name.
    )
    def test_is_valid_job_name(self, name, valid):
        self.assertEqual(valid, is_valid_job_name(name))

    def test_invalid_names(self):
        # Test that we drop invalid names.
        jobspecs = ["test-job", "test0123_job0123", "test/invalid"]
        user_states = ["test/invalid_user_state"]
        valid_jobspecs = [job_name for job_name in jobspecs if is_valid_job_name(job_name)]

        def mock_listdir(d):
            if d == "FAKE_SPECS":
                return jobspecs
            elif d == "FAKE_USER_STATES":
                return user_states
            return []

        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _listdir=mock.Mock(side_effect=mock_listdir),
            _download_jobspec=mock.DEFAULT,
            _download_job_state=mock.DEFAULT,
            _remove=mock.DEFAULT,
        )
        with patch_fns as mock_fns:
            download_job_batch(
                spec_dir="FAKE_SPECS",
                state_dir="FAKE_STATES",
                user_state_dir="FAKE_USER_STATES",
                local_spec_dir="FAKE",
                remove_invalid_job_specs=True,
            )
            downloaded_jobs = []
            for call_args in mock_fns["_download_jobspec"].call_args_list:
                # call_args[0] is the positional args.
                downloaded_jobs.append(call_args[0][0])
            self.assertSequenceEqual(valid_jobspecs, downloaded_jobs)
            self.assertContainsSubset(
                ["FAKE_SPECS/test/invalid", "FAKE_USER_STATES/test/invalid_user_state"],
                [call_args[0][0] for call_args in mock_fns["_remove"].call_args_list],
            )

    def test_invalid_membership(self):
        # Test that we drop jobs where the user is not a member of the specified quota project id.
        user_ids = ["a", "b", "c"]
        project_ids = ["proj1", "proj2", "proj3", "non_existent_proj"]
        jobspecs = {
            str(i): JobSpec(
                version=0,
                name=str(i),
                command="",
                cleanup_command=None,
                env_vars=None,
                metadata=JobMetadata(
                    user_id=user_id, project_id=project_id, creation_time=None, resources={}
                ),
            )
            for i, (user_id, project_id) in enumerate(itertools.product(user_ids, project_ids))
        }
        user_states = []
        project_membership = dict(proj1=["a", "b"], proj2=["a"], proj3=[".*"])
        valid_jobspecs = [
            name
            for name, job_spec in jobspecs.items()
            if job_spec.metadata.project_id == "proj3"
            or job_spec.metadata.project_id != "non_existent_proj"
            and job_spec.metadata.user_id in project_membership[job_spec.metadata.project_id]
        ]

        def mock_listdir(d):
            if d == "FAKE_SPECS":
                return list(jobspecs.keys())
            elif d == "FAKE_USER_STATES":
                return user_states
            return []

        def mock_download_jobspec(job_name, **kwargs):
            del kwargs
            return jobspecs[job_name]

        mocked_download = mock.Mock(side_effect=mock_download_jobspec)

        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _listdir=mock.Mock(side_effect=mock_listdir),
            _download_jobspec=mocked_download,
            _download_job_state=mock.DEFAULT,
            _remove=mock.DEFAULT,
        )
        with patch_fns as mock_fns:
            returned_jobs, _ = download_job_batch(
                spec_dir="FAKE_SPECS",
                state_dir="FAKE_STATES",
                user_state_dir="FAKE_USER_STATES",
                local_spec_dir="FAKE",
                remove_invalid_job_specs=True,
                quota=lambda: UserQuotaInfo(
                    total_resources=None,
                    project_resources=None,
                    project_membership=project_membership,
                ),
            )
            self.assertSequenceEqual(valid_jobspecs, list(returned_jobs.keys()))
            self.assertContainsSubset(
                [f"FAKE_SPECS/{i}" for i in jobspecs if i not in valid_jobspecs],
                [call_args[0][0] for call_args in mock_fns["_remove"].call_args_list],
            )


class TestJobSpec(parameterized.TestCase):
    """Tests job specs."""

    @parameterized.parameters(
        [
            {"env_vars": None},
            {"env_vars": {"TEST_ENV": "TEST_VAL", "TEST_ENV2": 'VAL_WITH_SPECIAL_:,"-{}'}},
        ],
    )
    def test_serialization_job_spec(self, env_vars):
        test_spec = new_jobspec(
            name="test_job",
            command="test command",
            env_vars=env_vars,
            metadata=JobMetadata(
                user_id="test_id",
                project_id="test_project",
                # Make sure str timestamp isn't truncated even when some numbers are 0.
                creation_time=datetime(1900, 1, 1, 0, 0, 0, 0),
                resources={"test": 8},
                priority=1,
            ),
        )
        with tempfile.NamedTemporaryFile("w+b") as f:
            serialize_jobspec(test_spec, f.name)
            deserialized_jobspec = deserialize_jobspec(f=f.name)
            for key in test_spec.__dataclass_fields__:
                self.assertIn(key, deserialized_jobspec.__dict__)
                self.assertEqual(deserialized_jobspec.__dict__[key], test_spec.__dict__[key])

    @parameterized.parameters(
        # Test with Nones in place of Optional.
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command=None,
                env_vars=None,
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources={},
                ),
            ),
            expected=None,
        ),
        # Test with values in place of Optional.
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1="value1"),
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1=123),
                ),
            ),
            expected=None,
        ),
        # Test mismatch on child (wrong type on name).
        dict(
            x=JobSpec(
                version=1,
                name=123,
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1="value1"),
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1=123),
                ),
            ),
            expected=ValidationError("jobspec.name=123 to be a string"),
        ),
        # Test mismatch on child (name is None).
        dict(
            x=JobSpec(
                version=1,
                name=None,
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1="value1"),
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1=123),
                ),
            ),
            expected=ValidationError("jobspec.name=None to be a string"),
        ),
        # Test mismatch on grandchild (invalid user_id type).
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1="value1"),
                metadata=JobMetadata(
                    user_id=123,
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1=123),
                ),
            ),
            expected=ValidationError("metadata.user_id=123 to be a string"),
        ),
        # Test mismatch on grandchild (user_id is None).
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1="value1"),
                metadata=JobMetadata(
                    user_id=None,
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1=123),
                ),
            ),
            expected=ValidationError("metadata.user_id=None to be a string"),
        ),
        # Invalid type of env var keys.
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command="test_cleanup",
                env_vars={123: "value1"},
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1=123),
                ),
            ),
            expected=ValidationError("string keys and values"),
        ),
        # Invalid type of env var values.
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1=123),
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1=123),
                ),
            ),
            expected=ValidationError("string keys and values"),
        ),
        # Invalid type of resources keys.
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1="value1"),
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources={123: 123},
                ),
            ),
            expected=ValidationError("string keys and int values"),
        ),
        # Invalid type of resources values.
        dict(
            x=JobSpec(
                version=1,
                name="test",
                command="test",
                cleanup_command="test_cleanup",
                env_vars=dict(env1="value1"),
                metadata=JobMetadata(
                    user_id="user",
                    project_id="project",
                    creation_time=datetime.now(),
                    resources=dict(resource1="test"),
                ),
            ),
            expected=ValidationError("string keys and int values"),
        ),
    )
    def test_validate_jobspec(self, x, expected):
        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            _validate_jobspec(x)


class TestJobState(parameterized.TestCase):
    """Tests job state utils."""

    @parameterized.parameters(
        dict(state=JobState(status=JobStatus.PENDING)),
        dict(state=JobState(status=JobStatus.ACTIVE, metadata={"tier": 0})),
    )
    def test_upload_download(self, state):
        with tempfile.TemporaryDirectory() as temp_dir:
            job_name = "test_job"
            _upload_job_state(job_name, state, remote_dir=temp_dir)
            downloaded = _download_job_state(job_name, remote_dir=temp_dir)
            self.assertEqual(state, downloaded)

    def test_download_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                JobState(status=JobStatus.PENDING),
                _download_job_state("unknown", remote_dir=temp_dir),
            )

    def test_download_compat(self):
        # Test backwards compat with string statuses.
        with tempfile.TemporaryDirectory() as temp_dir:
            job_name = "test_job"
            with open(os.path.join(temp_dir, job_name), "w", encoding="utf-8") as f:
                f.write("active\n")
            self.assertEqual(
                JobState(status=JobStatus.ACTIVE),
                _download_job_state(job_name, remote_dir=temp_dir),
            )


class TestRuntimeOptions(parameterized.TestCase):
    """Tests runtime options."""

    def test_load_and_set_runtime_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initially empty.
            self.assertEqual({}, _load_runtime_options(temp_dir))

            # Set some values.
            set_runtime_options(temp_dir, a="1", b={"c": "2"})
            self.assertEqual({"a": "1", "b": {"c": "2"}}, _load_runtime_options(temp_dir))

            # Update.
            set_runtime_options(temp_dir, a="2", b={"c": "3"})
            self.assertEqual({"a": "2", "b": {"c": "3"}}, _load_runtime_options(temp_dir))


# Returns a new mock Popen for each subprocess.Popen call.
def _mock_popen_fn(mock_spec: Dict[str, Dict]):
    """Returns a callable that outputs mocked Popens for predetermined commands.

    For example:
        Input:
            {'my_command': {'terminate.side_effect': ValueError}}
        Result:
            mock = subprocess.Popen('my_command')
            mock.terminate()  # Raises ValueError.
    """

    def popen(cmd, env: Optional[Dict] = None, **kwargs):
        del kwargs
        if cmd not in mock_spec:
            raise ValueError(f"Don't know how to mock: {cmd}")
        m = mock.MagicMock()
        m.configure_mock(**mock_spec[cmd])
        m.env = env
        return m

    return popen


# Returns a new mock _PipedProcess.
def _mock_piped_popen_fn(mock_spec: Dict[str, Dict]):
    """See `_mock_popen_fn`."""
    mock_popen_fn = _mock_popen_fn(mock_spec)

    def piped_popen(cmd, f, env_vars=None):
        mock_fd = mock.MagicMock()
        mock_fd.name = f
        return _PipedProcess(popen=mock_popen_fn(cmd, env=env_vars), fd=mock_fd)

    return piped_popen


class BastionTest(parameterized.TestCase):
    """Tests Bastion."""

    def test_piped_popen_env(self):
        # Ensures that _piped_popen converts env values to strings.
        # This avoids potential TypeError when invoking subprocess.Popen.
        patch_popen = mock.patch.object(subprocess, "Popen", autospec=True)
        with tempfile.NamedTemporaryFile("w") as f, patch_popen as mock_popen:
            # Check that by default we call with os.environ.
            bastion._piped_popen("test command", f.name, env_vars=None)
            self.assertEqual(os.environ, mock_popen.call_args[1]["env"])

            # Check that tier is converted to string.
            bastion._piped_popen("test command", f.name, env_vars={"tier": 123})
            self.assertEqual("123", mock_popen.call_args[1]["env"]["tier"])

    @contextlib.contextmanager
    def _patch_bastion(self, mock_popen_spec: Optional[Dict] = None):
        mocks = []
        module_name = bastion.__name__

        if mock_popen_spec:
            mock_popen = mock.patch.object(subprocess, "Popen", autospec=True)
            mock_popen.side_effect = _mock_popen_fn(mock_popen_spec)
            mocks.extend(
                [
                    mock_popen,
                    mock.patch(
                        f"{module_name}._piped_popen",
                        side_effect=_mock_piped_popen_fn(mock_popen_spec),
                    ),
                ]
            )

        class NoOpCleaner(Cleaner):
            def sweep(self, jobs):
                del jobs

        def noop_upload_fn(*args, **kwargs):
            del args, kwargs

        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory() as tmpdir:
            # Boilerplate to register multiple mocks at once.
            for m in mocks:
                stack.enter_context(m)

            cfg = Bastion.default_config().set(
                scheduler=JobScheduler.default_config(),
                cleaner=NoOpCleaner.default_config(),
                uploader=Uploader.default_config().set(
                    upload_fn=config_for_function(lambda: noop_upload_fn)
                ),
                output_dir=tmpdir,
                quota=config_for_function(mock_quota_config),
            )
            yield cfg.instantiate()

    def test_sync_jobs(self):
        """Tests downloading jobspecs."""

        with self._patch_bastion() as mock_bastion:
            os.makedirs(mock_bastion._active_dir, exist_ok=True)
            os.makedirs(_JOB_DIR, exist_ok=True)
            # Create some jobspecs to download.
            specs = [
                new_jobspec(
                    name="job1",
                    command="",
                    metadata=JobMetadata(
                        user_id="user1",
                        project_id="project1",
                        creation_time=datetime(1900, 1, 1, 0, 0, 0, 0),
                        resources={"test": 8},
                    ),
                ),
                new_jobspec(
                    name="job2",
                    command="",
                    metadata=JobMetadata(
                        user_id="user2",
                        project_id="project1",
                        creation_time=datetime(1900, 1, 1, 0, 0, 0, 1),
                        resources={"test": 8},
                    ),
                ),
            ]
            # Write them to the Bastion submission directory.
            for spec in specs:
                with tempfile.NamedTemporaryFile("w") as f:
                    serialize_jobspec(spec, f)
                    bastion_dir = (
                        BastionDirectory.default_config()
                        .set(root_dir=mock_bastion._output_dir)
                        .instantiate()
                    )
                    bastion_dir.submit_job(spec.name, job_spec_file=f.name)
            # Download the jobspecs.
            mock_bastion._sync_jobs()
            # Confirm expected jobs were downloaded.
            self.assertSequenceEqual(list(mock_bastion._active_jobs), ["job1"])

    @parameterized.product(
        [
            dict(
                # Command has not terminated -- expect kill() to be called.
                # We should not need to consult terminate() or poll().
                popen_spec={
                    "command": {
                        "wait.return_value": None,
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                    # cleanup should have no effect here, so we just raise if it's ever used.
                    "cleanup": {
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                },
            ),
            dict(
                # Command has already terminated. Expect state to transition to PENDING and
                # command_proc to be None.
                popen_spec={
                    "cleanup": {"poll.return_value": 0, "terminate.side_effect": ValueError},
                },
            ),
        ],
        user_state_exists=[False, True],
    )
    def test_pending(self, popen_spec, user_state_exists):
        """Test PENDING state transitions.

        1. If command_proc is still running, it should be terminated (killed).
        2. The state should remain PENDING, command_proc must be None, and log file should be
            uploaded.
        """
        mock_proc = _mock_piped_popen_fn(popen_spec)
        job = Job(
            spec=new_jobspec(
                name="test_job",
                command="command",
                cleanup_command="cleanup",
                metadata=JobMetadata(
                    user_id="test_user",
                    project_id="test_project",
                    creation_time=datetime.now(),
                    resources={"v4": 8},
                ),
            ),
            state=JobState(status=JobStatus.PENDING),
            command_proc=mock_proc("command", "test_command") if "command" in popen_spec else None,
            cleanup_proc=mock_proc("cleanup", "test_cleanup") if "cleanup" in popen_spec else None,
        )
        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
            send_signal=mock.DEFAULT,
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.Mock(return_value=user_state_exists),
            copy=mock.DEFAULT,
            remove=mock.DEFAULT,
        )
        with (
            self._patch_bastion(popen_spec) as mock_bastion,
            patch_fns as mock_fns,
            patch_tfio as mock_tfio,
        ):
            # Run a couple updates to test transition to PENDING and staying in PENDING.
            for _ in range(2):
                orig_command_proc = job.command_proc
                updated_job = mock_bastion._update_single_job(job)
                # Job should now be in pending.
                self.assertEqual(updated_job.state, JobState(status=JobStatus.PENDING))
                # Command should be None.
                self.assertIsNone(updated_job.command_proc)

                if orig_command_proc is not None:
                    # Kill should have been called, and fd should have been closed.
                    mock_fns["send_signal"].assert_called()
                    self.assertTrue(
                        orig_command_proc.fd.close.called  # pytype: disable=attribute-error
                    )

                    # Log should be uploaded if command was initially running.
                    upload_call = mock.call(
                        orig_command_proc.fd.name,
                        os.path.join(
                            mock_bastion._log_dir, os.path.basename(orig_command_proc.fd.name)
                        ),
                        overwrite=True,
                    )
                    mock_tfio["copy"].assert_has_calls([upload_call], any_order=False)

                # Cleanup command should not be involved.
                updated_job.cleanup_proc.popen.poll.assert_not_called()
                updated_job.cleanup_proc.popen.terminate.assert_not_called()

                updated_job = job

    @parameterized.product(
        [
            dict(
                popen_spec={
                    # Runs for one update step and then completes.
                    # terminate() raises, since we don't expect it to be called.
                    "command": {
                        "poll.side_effect": [None, 0],
                        "terminate.side_effect": ValueError,
                    },
                    # cleanup should have no effect here, so we just raise if it's ever used.
                    "cleanup": {
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                },
                expect_poll_calls=2,
            ),
            dict(
                popen_spec={
                    # Command terminates instantly.
                    "command": {
                        "poll.return_value": 1,
                        "terminate.side_effect": ValueError,
                    },
                    # cleanup should have no effect here, so we just raise if it's ever used.
                    "cleanup": {
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                },
                expect_poll_calls=1,
            ),
        ],
        logfile_exists=[False, True],
    )
    def test_active(self, popen_spec, expect_poll_calls, logfile_exists):
        """Test ACTIVE state transitions.

        1. If command_proc is not running, it should be started. If a log file exists remotely, it
            should be downloaded.
        2. If command_proc is already running, stay in ACTIVE.
        3. If command_proc is completed, move to CLEANING.
        """
        mock_proc = _mock_piped_popen_fn(popen_spec)
        job = Job(
            spec=new_jobspec(
                name="test_job",
                command="command",
                cleanup_command="cleanup",
                metadata=JobMetadata(
                    user_id="test_user",
                    project_id="test_job",
                    creation_time=datetime.now(),
                    resources={"v4": 8},
                ),
            ),
            state=JobState(status=JobStatus.ACTIVE, metadata={"tier": 1}),
            command_proc=None,  # Initially, command is None.
            cleanup_proc=mock_proc("cleanup", "test_cleanup"),
        )

        def mock_tfio_exists(f):
            if "logs" in f and os.path.basename(f) == "test_job":
                return logfile_exists
            return False

        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.MagicMock(side_effect=mock_tfio_exists),
            copy=mock.DEFAULT,
        )
        with patch_fns, self._patch_bastion(popen_spec) as mock_bastion, patch_tfio as mock_tfio:
            # Initially, job should have no command.
            self.assertIsNone(job.command_proc)

            # Run single update step to start the job.
            updated_job = mock_bastion._update_single_job(job)

            # Command should be started on the first update.
            self.assertIsNotNone(updated_job.command_proc)
            # Scheduling metadata should be set.
            self.assertEqual({"BASTION_TIER": 1}, updated_job.command_proc.popen.env)

            # Log should be downloaded if it exists.
            download_call = mock.call(
                os.path.join(mock_bastion._log_dir, job.spec.name),
                os.path.join(_LOG_DIR, job.spec.name),
                overwrite=True,
            )
            mock_tfio["copy"].assert_has_calls([download_call], any_order=False)

            # Run until expected job completion.
            for _ in range(expect_poll_calls - 1):
                self.assertEqual(
                    updated_job.state, JobState(status=JobStatus.ACTIVE, metadata={"tier": 1})
                )
                updated_job = mock_bastion._update_single_job(updated_job)

            # Job state should be CLEANING.
            self.assertEqual(
                updated_job.state, JobState(status=JobStatus.CLEANING, metadata={"tier": 1})
            )

    # pylint: disable-next=too-many-branches
    def test_update_jobs(self):
        """Tests the global update step."""

        def popen_spec(command_poll=2, cleanup_poll=2):
            return {
                # Constructs a command_proc that "completes" after `command_poll` updates.
                "command": {
                    "wait.return_value": None,
                    "poll.side_effect": [None] * (command_poll - 1) + [0],
                    "terminate.side_effect": None,
                },
                # Constructs a cleanup_proc that completes after `cleanup_poll` updates.
                "cleanup": {
                    "poll.side_effect": [None] * (cleanup_poll - 1) + [0],
                    "terminate.side_effect": ValueError,
                },
            }

        def mock_proc(cmd, **kwargs):
            fn = _mock_piped_popen_fn(popen_spec(**kwargs))
            return fn(cmd, "test_file")

        yesterday = datetime.now() - timedelta(days=1)

        # Test state transitions w/ interactions between jobs (scheduling).
        # See also `mock_quota_config` for mock project quotas and limits.
        active_jobs = {
            # This job will stay PENDING, since user "b" has higher priority.
            "pending": Job(
                spec=new_jobspec(
                    name="pending",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="a",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=3),
                        resources={"v4": 12},  # Doesn't fit if "resume" job is scheduled.
                    ),
                ),
                state=JobState(status=JobStatus.PENDING),
                command_proc=None,  # No command proc for PENDING jobs.
                cleanup_proc=None,
            ),
            # This job will go from PENDING to ACTIVE.
            "resume": Job(
                spec=new_jobspec(
                    name="resume",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="b",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v4": 5},  # Fits within v4 budget in project2.
                    ),
                ),
                state=JobState(status=JobStatus.PENDING),
                command_proc=None,  # No command proc for PENDING jobs.
                cleanup_proc=None,
            ),
            # This job will stay in ACTIVE, since it takes 2 updates to complete.
            "active": Job(
                spec=new_jobspec(
                    name="active",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="c",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v3": 2},  # Fits within the v3 budget in project2.
                    ),
                ),
                state=JobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
                command_proc=mock_proc("command"),
                cleanup_proc=None,  # No cleanup_proc for ACTIVE jobs.
            ),
            # This job will go from ACTIVE to PENDING, since it's using part of project2's v4
            # quota, and "b" is requesting project2's v4 quota.
            # Even though poll()+terminate() typically takes a few steps, we instead go through
            # kill() to forcefully terminate within one step.
            "preempt": Job(
                spec=new_jobspec(
                    name="preempt",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="d",
                        project_id="project1",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v4": 12},  # Uses part of project2 budget.
                    ),
                ),
                state=JobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
                command_proc=mock_proc("command"),
                cleanup_proc=None,  # No cleanup_proc for ACTIVE.
            ),
            # This job will go from ACTIVE to CLEANING.
            "cleaning": Job(
                spec=new_jobspec(
                    name="cleaning",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="f",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v3": 2},  # Fits within the v3 budget in project2.
                    ),
                ),
                state=JobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
                command_proc=mock_proc("command", command_poll=1),
                cleanup_proc=None,
            ),
            # This job will go from CANCELLING to CLEANING.
            # Note that CANCELLING jobs will not be "pre-empted" by scheduler; even though this job
            # is out-of-budget, it will go to CLEANING instead of SUSPENDING.
            "cleaning_cancel": Job(
                spec=new_jobspec(
                    name="cleaning_cancel",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="g",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=4),
                        resources={"v4": 100},  # Does not fit into v4 budget.
                    ),
                ),
                state=JobState(status=JobStatus.CANCELLING),
                command_proc=mock_proc("command", command_poll=1),
                cleanup_proc=None,
            ),
            # This job will go from CLEANING to COMPLETED.
            "completed": Job(
                spec=new_jobspec(
                    name="completed",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="e",
                        project_id="project3",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v5": 2},
                    ),
                ),
                state=JobState(status=JobStatus.CLEANING),
                command_proc=None,
                cleanup_proc=mock_proc("cleanup", cleanup_poll=1),  # Should have cleanup_proc.
            ),
        }
        # Copy original jobs, since updates happen in-place.
        orig_jobs = copy.deepcopy(active_jobs)
        # Pretend that only 'cleaning_cancel' came from a user state.
        jobs_with_user_states = {"cleaning_cancel"}

        # Patch all network calls and utils.
        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
            send_signal=mock.DEFAULT,
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.DEFAULT,
            copy=mock.DEFAULT,
            remove=mock.DEFAULT,
        )
        with (
            self._patch_bastion(popen_spec()) as mock_bastion,
            patch_fns as mock_fns,
            patch_tfio as mock_tfio,
        ):
            mock_bastion._active_jobs = active_jobs
            mock_bastion._jobs_with_user_states = jobs_with_user_states
            mock_bastion._update_jobs()

            # Ensure _active_jobs membership stays same.
            self.assertEqual(mock_bastion._active_jobs.keys(), orig_jobs.keys())

            # Note that scheduling metadata is also part of the state.
            expected_states = {
                "pending": JobState(status=JobStatus.PENDING),
                "resume": JobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
                "active": JobState(status=JobStatus.ACTIVE, metadata={"tier": 0}),
                "preempt": JobState(status=JobStatus.PENDING),
                "cleaning": JobState(status=JobStatus.CLEANING, metadata={"tier": 0}),
                "cleaning_cancel": JobState(status=JobStatus.CLEANING),
                "completed": JobState(status=JobStatus.COMPLETED),
            }
            for job_name in active_jobs:
                self.assertEqual(
                    mock_bastion._active_jobs[job_name].state,
                    expected_states[job_name],
                    msg=job_name,
                )

            for job in mock_bastion._active_jobs.values():
                # For jobs that are ACTIVE, expect command_proc to be non-None.
                if job.state.status == JobStatus.ACTIVE:
                    self.assertIsNotNone(job.command_proc)
                    self.assertIsNone(job.cleanup_proc)
                # For jobs that are COMPLETED, expect both procs to be None.
                elif job.state.status == JobStatus.COMPLETED:
                    self.assertIsNone(job.command_proc)
                    self.assertIsNone(job.cleanup_proc)

                    # Remote jobspec should not be deleted until gc.
                    for delete_call in mock_tfio["remove"].mock_calls:
                        self.assertNotIn(
                            os.path.join(_JOB_DIR, job.spec.name),
                            delete_call.args,
                        )

                # User states should only be deleted if the job's state was read from
                # user_state_dir.
                self.assertEqual(
                    any(
                        os.path.join(mock_bastion._user_state_dir, job.spec.name)
                        in delete_call.args
                        for delete_call in mock_tfio["remove"].mock_calls
                    ),
                    job.spec.name in mock_bastion._jobs_with_user_states,
                )

                # For jobs that went from ACTIVE to PENDING, expect kill() to have been called.
                if (
                    orig_jobs[job.spec.name].state.status == JobStatus.ACTIVE
                    and job.state.status == JobStatus.PENDING
                ):
                    mock_fns["send_signal"].assert_called()
                    self.assertFalse(
                        orig_jobs[
                            job.spec.name
                        ].command_proc.popen.terminate.called  # pytype: disable=attribute-error
                    )

                # For jobs that went from PENDING to ACTIVE, expect command to have been invoked
                # with "tier" in the env.
                if (
                    orig_jobs[job.spec.name].state.status == JobStatus.PENDING
                    and job.state.status == JobStatus.ACTIVE
                ):
                    self.assertIn("BASTION_TIER", job.command_proc.popen.env)

            for job_name in active_jobs:
                history_file = os.path.join(mock_bastion._job_history_dir, job_name)
                if job_name in ("active", "pending"):
                    # The 'active'/'pending' jobs do not generate hisotry.
                    self.assertFalse(os.path.exists(history_file), msg=history_file)
                else:
                    self.assertTrue(os.path.exists(history_file), msg=history_file)
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = f.read()
                        expected_msg = {
                            "resume": "ACTIVE: start process command",
                            "preempt": "PENDING: pre-empting",
                            "cleaning": "CLEANING: process finished",
                            "cleaning_cancel": "CLEANING: process terminated",
                            "completed": "COMPLETED: cleanup finished",
                        }
                        self.assertIn(expected_msg[job_name], history)

            all_history_files = []
            for project_id in [f"project{i}" for i in range(1, 3)]:
                project_history_dir = os.path.join(mock_bastion._project_history_dir, project_id)
                project_history_files = list(os.scandir(project_history_dir))
                for history_file in project_history_files:
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = f.read()
                        print(f"[{project_id}] {history}")
                all_history_files.extend(project_history_files)
            # "project1" and "project2".
            self.assertLen(all_history_files, 2)

    def test_gc_jobs(self):
        """Tests GC mechanism.

        1. Only PENDING/COMPLETED jobs are cleaned.
        2. COMPLETED jobs that finish gc'ing should remove jobspecs.
        """
        # Note: command_proc and cleanup_proc shouldn't matter for GC. We only look at state +
        # resources.
        active_jobs: Dict[str, Job] = {}
        init_job_states = {
            "pending": JobState(status=JobStatus.PENDING),
            "active": JobState(status=JobStatus.ACTIVE),
            "cleaning": JobState(status=JobStatus.CLEANING),
            "completed": JobState(status=JobStatus.COMPLETED),
            "completed_gced": JobState(status=JobStatus.COMPLETED),
            "rescheduled": JobState(status=JobStatus.PENDING, metadata={"tier": "0"}),
        }
        for job_name, job_state in init_job_states.items():
            active_jobs[job_name] = Job(
                spec=new_jobspec(
                    name=job_name,
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id=f"{job_name}_user",
                        project_id="project1",
                        creation_time=datetime.now() - timedelta(days=1),
                        resources={"v4": 1},
                    ),
                ),
                state=job_state,
                command_proc=None,
                cleanup_proc=None,
            )
        # We pretend that only some jobs are "fully gc'ed".
        fully_gced = ["completed_gced"]
        rescheduled = ["rescheduled"]

        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            remove=mock.DEFAULT,
        )
        with self._patch_bastion() as mock_bastion, patch_tfio as mock_tfio:

            def mock_clean(jobs: Dict[str, ResourceMap]) -> Sequence[str]:
                self.assertTrue(
                    all(
                        active_jobs[job_name].state.status == JobStatus.COMPLETED
                        or (
                            active_jobs[job_name].state.status == JobStatus.PENDING
                            and active_jobs[job_name].state.metadata.get("tier") is None
                        )
                        for job_name in jobs
                    )
                )
                for job_spec in jobs.values():
                    self.assertIsInstance(job_spec, JobSpec)
                return fully_gced

            with mock.patch.object(mock_bastion, "_cleaner") as mock_cleaner:
                mock_cleaner.configure_mock(**{"sweep.side_effect": mock_clean})
                mock_bastion._active_jobs = active_jobs
                mock_bastion._gc_jobs()

            # Ensure that each fully GC'ed COMPLETED job deletes jobspec and state.
            for job_name in fully_gced:
                deleted_state = any(
                    os.path.join(mock_bastion._state_dir, job_name) in delete_call.args
                    for delete_call in mock_tfio["remove"].mock_calls
                )
                deleted_jobspec = any(
                    os.path.join(mock_bastion._active_dir, job_name) in delete_call.args
                    for delete_call in mock_tfio["remove"].mock_calls
                )
                self.assertEqual(
                    active_jobs[job_name].state == JobState(status=JobStatus.COMPLETED),
                    deleted_state and deleted_jobspec,
                )

            # Ensure that rescheduled jobs do not get deleted.
            for job_name in rescheduled:
                self.assertEqual(init_job_states[job_name], active_jobs[job_name].state)
                for delete_call in mock_tfio["remove"].mock_calls:
                    self.assertNotIn(
                        os.path.join(mock_bastion._state_dir, job_name), delete_call.call_args
                    )

    @parameterized.parameters(
        dict(
            initial_jobs={
                "pending": JobState(status=JobStatus.PENDING),
                "active": JobState(status=JobStatus.ACTIVE),
                "cancelling": JobState(status=JobStatus.CANCELLING),
                "completed": JobState(status=JobStatus.COMPLETED),
            },
            runtime_options={},
            expect_schedulable=["pending", "active"],
        ),
        # Test runtime options.
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"dry_run": True, "verbosity": 1}},
            expect_schedulable=[],
            expect_dry_run=True,
            expect_verbosity=1,
        ),
        # Test invalid runtime options.
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"dry_run": "hello", "verbosity": None}},
            expect_schedulable=[],
        ),
        # Test invalid runtime options schema.
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"verbosity": None}},
            expect_schedulable=[],
        ),
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"unknown": 123}},
            expect_schedulable=[],
        ),
        dict(
            initial_jobs={},
            runtime_options={"scheduler": [], "unknown": 123},
            expect_schedulable=[],
        ),
        dict(
            initial_jobs={},
            runtime_options={"unknown": 123},
            expect_schedulable=[],
        ),
    )
    def test_update_scheduler(
        self,
        *,
        initial_jobs: Dict[str, JobState],
        runtime_options: Optional[Dict[str, Any]],
        expect_schedulable: Sequence[str],
        expect_dry_run: bool = False,
        expect_verbosity: int = 0,
    ):
        with self._patch_bastion() as mock_bastion:
            patch_update = mock.patch.object(mock_bastion, "_update_single_job")
            patch_history = mock.patch.object(mock_bastion, "_append_to_project_history")
            patch_scheduler = mock.patch.object(mock_bastion, "_scheduler")

            with patch_update, patch_history, patch_scheduler as mock_scheduler:
                mock_bastion._active_jobs = {
                    job_name: Job(
                        spec=mock.Mock(), state=state, command_proc=None, cleanup_proc=None
                    )
                    for job_name, state in initial_jobs.items()
                }
                mock_bastion._runtime_options = runtime_options
                mock_bastion._update_jobs()
                args, kwargs = mock_scheduler.schedule.call_args
                self.assertSameElements(expect_schedulable, args[0].keys())
                self.assertEqual({"dry_run": expect_dry_run, "verbosity": expect_verbosity}, kwargs)

    def test_exception(self):
        patch_signal = mock.patch(f"{bastion.__name__}.send_signal")
        with patch_signal, self._patch_bastion() as mock_bastion:
            mock_job = Job(
                spec=mock.Mock(),
                state=mock.Mock(),
                command_proc=mock.Mock(),
                cleanup_proc=mock.Mock(),
            )

            def mock_execute():
                mock_bastion._active_jobs["test"] = mock_job
                raise ValueError("Mock error")

            with mock.patch.multiple(
                mock_bastion,
                _wait_and_close_proc=mock.DEFAULT,
                _execute=mock.Mock(side_effect=mock_execute),
            ) as mock_methods:
                with self.assertRaisesRegex(ValueError, "Mock error"):
                    mock_bastion.execute()
                self.assertIn(
                    mock_job.command_proc, mock_methods["_wait_and_close_proc"].call_args_list[0][0]
                )
                self.assertIn(
                    mock_job.cleanup_proc, mock_methods["_wait_and_close_proc"].call_args_list[1][0]
                )

    def test_execute_with_exception_and_job_failure(self):
        job_1 = Job(
            spec=mock.Mock(),
            state=mock.Mock(),
            command_proc=mock.Mock(),
            cleanup_proc=mock.Mock(),
        )
        job_2 = Job(
            spec=mock.Mock(),
            state=mock.Mock(),
            command_proc=mock.Mock(),
            cleanup_proc=mock.Mock(),
        )
        active_jobs = {
            "job1": job_1,
            "job2": job_2,
        }

        with self._patch_bastion() as mock_bastion:
            mock_bastion._execute = mock.Mock(side_effect=Exception("Execution failed"))
            mock_bastion._kill_job = mock.Mock(side_effect=[Exception("Cannot kill job"), None])

            mock_bastion._active_jobs = active_jobs

            with self.assertRaises(Exception):
                mock_bastion.execute()

            self.assertEqual(mock_bastion._kill_job.call_count, 2)
            expected_calls = [mock.call(job_1), mock.call(job_2)]
            self.assertEqual(mock_bastion._kill_job.call_args_list, expected_calls)


class BastionDirectoryTest(parameterized.TestCase):
    """Tests BastionDirectory."""

    @parameterized.product(
        job_name=[
            "test-job",
            "test0123_job0123",
            "test/invalid",
        ],
        spec_exists=[True, False],
    )
    def test_submit_job(self, job_name, spec_exists):
        job_name = "test-job"
        job_spec_file = "spec"
        bastion_dir = (
            bastion.BastionDirectory.default_config().set(root_dir="test-dir").instantiate()
        )
        self.assertEqual("test-dir", str(bastion_dir))
        self.assertEqual("test-dir/logs", bastion_dir.logs_dir)
        self.assertEqual("test-dir/jobs/active", bastion_dir.active_job_dir)
        self.assertEqual("test-dir/jobs/complete", bastion_dir.complete_job_dir)
        self.assertEqual("test-dir/jobs/states", bastion_dir.job_states_dir)
        self.assertEqual("test-dir/jobs/user_states", bastion_dir.user_states_dir)
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.MagicMock(return_value=spec_exists),
            copy=mock.DEFAULT,
        )
        if is_valid_job_name(job_name):
            ctx = contextlib.nullcontext()
        else:
            ctx = self.assertRaisesRegex(ValueError, "not a valid job name")
        with ctx, patch_tfio as mock_tfio:
            bastion_dir.submit_job(job_name, job_spec_file=job_spec_file)
            if not spec_exists:
                mock_tfio["copy"].assert_called_with(
                    job_spec_file,
                    os.path.join(bastion_dir.active_job_dir, job_name),
                )
            else:
                mock_tfio["copy"].assert_not_called()

    @parameterized.parameters(True, False)
    def test_delete(self, spec_exists):
        job_name = "test-job"
        bastion_dir = (
            bastion.BastionDirectory.default_config().set(root_dir="test-dir").instantiate()
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.MagicMock(side_effect=[spec_exists, False]),
            copy=mock.DEFAULT,
        )
        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
        )
        with patch_tfio, patch_fns as mock_fns:
            bastion_dir.cancel_job(job_name)
            if not spec_exists:
                mock_fns["_upload_job_state"].assert_not_called()
            else:
                mock_fns["_upload_job_state"].assert_called_with(
                    job_name,
                    JobState(status=JobStatus.CANCELLING),
                    remote_dir=bastion_dir.user_states_dir,
                )


if __name__ == "__main__":
    absltest.main()

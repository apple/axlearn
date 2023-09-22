# Copyright © 2023 Apple Inc.

"""Tests bastion VM."""
# pylint: disable=no-self-use,protected-access
import contextlib
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence
from unittest import mock

from absl.testing import parameterized

from axlearn.cloud.common.scheduler import JobMetadata, JobScheduler
from axlearn.cloud.common.scheduler_test import mock_scheduler
from axlearn.cloud.common.types import ResourceMap
from axlearn.cloud.gcp.jobs import bastion_vm
from axlearn.cloud.gcp.jobs.bastion_vm import (
    _JOB_DIR,
    BastionJob,
    Job,
    JobState,
    _PipedProcess,
    deserialize_jobspec,
    download_job_batch,
    new_jobspec,
    serialize_jobspec,
)
from axlearn.cloud.gcp.tpu_cleaner import TPUCleaner


class TestDownloadJobBatch(parameterized.TestCase):
    """Tests download utils."""

    def test_download_job_batch(self):
        spec_dir = "gs://test_spec_dir"
        state_dir = "gs://test_state_dir"
        user_state_dir = "gs://user_state_dir"

        user_states = {
            f"{user_state_dir}/job_test1": JobState.CANCELLING,
        }
        states = {
            f"{state_dir}/job_test2": JobState.CLEANING,
            f"{state_dir}/job_test1": JobState.ACTIVE,
            f"{state_dir}/job_test0": JobState.ACTIVE,
        }
        jobspecs = {
            f"{spec_dir}/job_test2": mock.Mock(),
            f"{spec_dir}/job_test1": mock.Mock(),
            f"{spec_dir}/job_test0": mock.Mock(),
        }

        def mock_list_blobs(blob_dir):
            if blob_dir == user_state_dir:
                return list(user_states.keys())
            if blob_dir == spec_dir:
                return list(jobspecs.keys())
            assert False  # Should not be reached.

        def mock_download_jobspec(job_name, *, remote_dir, **kwargs):
            del kwargs
            return jobspecs[os.path.join(remote_dir, job_name)]

        def mock_download_job_state(job_name, *, remote_dir, **kwargs):
            del kwargs
            key = os.path.join(remote_dir, job_name)
            if key in user_states:
                return user_states[key]
            return states[key]

        patch_fns = mock.patch.multiple(
            bastion_vm.__name__,
            list_blobs=mock.Mock(side_effect=mock_list_blobs),
            _download_jobspec=mock.Mock(side_effect=mock_download_jobspec),
            _download_job_state=mock.Mock(side_effect=mock_download_job_state),
        )

        # Ensure that results are in the right order and pairing.
        with patch_fns, tempfile.TemporaryDirectory() as tmpdir:
            jobs = download_job_batch(
                spec_dir=spec_dir,
                state_dir=state_dir,
                user_state_dir=user_state_dir,
                local_spec_dir=tmpdir,
            )
            self.assertSameElements(["job_test0", "job_test1", "job_test2"], jobs.keys())
            for job_name, job in jobs.items():
                # Make sure the states are expected. User states take precedence.
                expect_state = user_states.get(
                    os.path.join(user_state_dir, job_name),
                    states[os.path.join(state_dir, job_name)],
                )
                self.assertEqual(job.state, expect_state)
                self.assertEqual(job.spec, jobspecs[os.path.join(spec_dir, job_name)])


class TestJobSpec(parameterized.TestCase):
    """Tests job specs."""

    def test_serialization_job_spec(self):
        test_spec = new_jobspec(
            name="test_job",
            command="test command",
            metadata=JobMetadata(
                user_id="test_id",
                project_id="test_project",
                creation_time=datetime.now(),
                resources={"test": 8.0},
                priority=1,
            ),
        )
        with tempfile.NamedTemporaryFile("w+b") as f:
            serialize_jobspec(test_spec, f.name)
            deserialized_jobspec = deserialize_jobspec(f=f.name)
            for key in test_spec.__dataclass_fields__:
                self.assertIn(key, deserialized_jobspec.__dict__)
                self.assertEqual(deserialized_jobspec.__dict__[key], test_spec.__dict__[key])


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

    def popen(cmd, **kwargs):
        del kwargs
        if cmd not in mock_spec:
            raise ValueError(f"Don't know how to mock: {cmd}")
        m = mock.MagicMock()
        m.configure_mock(**mock_spec[cmd])
        return m

    return popen


# Returns a new mock _PipedProcess.
def _mock_piped_popen_fn(mock_spec: Dict[str, Dict]):
    """See `_mock_popen_fn`."""
    mock_popen_fn = _mock_popen_fn(mock_spec)

    def piped_popen(cmd, f):
        mock_fd = mock.MagicMock()
        mock_fd.name = f
        return _PipedProcess(popen=mock_popen_fn(cmd), fd=mock_fd)

    return piped_popen


class BastionJobTest(parameterized.TestCase):
    """Tests BastionJob."""

    @contextlib.contextmanager
    def _patch_bastion(self, mock_popen_spec: Optional[Dict] = None):
        mocks = [mock_scheduler()]
        module_name = bastion_vm.__name__

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

        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory() as tmpdir:
            # Boilerplate to register multiple mocks at once.
            for m in mocks:
                stack.enter_context(m)

            with mock.patch(f"{module_name}._bastion_dir", return_value=tmpdir):
                cfg = BastionJob.default_config().set(
                    scheduler=JobScheduler.default_config().set(project_quota_file="test"),
                    cleaner=TPUCleaner.default_config(),
                    max_tries=1,
                )
                bastion = cfg.set(
                    name="test", project="test", zone="test", retry_interval=30, command=""
                ).instantiate()

                yield bastion

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
            state=JobState.PENDING,
            command_proc=mock_proc("command", "test_command") if "command" in popen_spec else None,
            cleanup_proc=mock_proc("cleanup", "test_cleanup") if "cleanup" in popen_spec else None,
        )
        patch_fns = mock.patch.multiple(
            bastion_vm.__name__,
            blob_exists=mock.Mock(return_value=user_state_exists),
            _upload_job_state=mock.DEFAULT,
            upload_blob=mock.DEFAULT,
            delete_blob=mock.DEFAULT,
            send_signal=mock.DEFAULT,
        )
        with self._patch_bastion(popen_spec) as bastion, patch_fns as mock_fns:
            # Run a couple updates to test transition to PENDING and staying in PENDING.
            for _ in range(2):
                orig_command_proc = job.command_proc
                updated_job = bastion._update_single_job(job)
                # Job should now be in pending.
                self.assertEqual(updated_job.state, JobState.PENDING)
                # Command should be None.
                self.assertIsNone(updated_job.command_proc)

                if orig_command_proc is not None:
                    # Kill should have been called, and fd should have been closed.
                    mock_fns["send_signal"].assert_called()
                    self.assertTrue(
                        orig_command_proc.fd.close.called  # pytype: disable=attribute-error
                    )

                    # Log should be uploaded if command was initially running.
                    mock_fns["upload_blob"].assert_called()

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
            state=JobState.ACTIVE,
            command_proc=None,  # Initially, command is None.
            cleanup_proc=mock_proc("cleanup", "test_cleanup"),
        )

        def mock_blob_exists(f):
            if "logs" in f and os.path.basename(f) == "test_job":
                return logfile_exists
            return False

        patch_network = mock.patch.multiple(
            bastion_vm.__name__,
            blob_exists=mock.MagicMock(side_effect=mock_blob_exists),
            download_blob=mock.DEFAULT,
            _upload_job_state=mock.DEFAULT,
        )
        with self._patch_bastion(popen_spec) as bastion, patch_network as mock_network:
            # Initially, job should have no command.
            self.assertIsNone(job.command_proc)

            # Run single update step to start the job.
            updated_job = bastion._update_single_job(job)

            # Command should be started on the first update.
            self.assertIsNotNone(updated_job.command_proc)
            # Log should be downloaded if it exists.
            self.assertEqual(mock_network["download_blob"].called, logfile_exists)

            # Run until expected job completion.
            for _ in range(expect_poll_calls - 1):
                self.assertEqual(updated_job.state, JobState.ACTIVE)
                updated_job = bastion._update_single_job(updated_job)

            # Job state should be CLEANING.
            self.assertEqual(updated_job.state, JobState.CLEANING)

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
        # See also `mock_scheduler` for mock project quotas and limits.
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
                state=JobState.PENDING,
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
                state=JobState.PENDING,
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
                state=JobState.PENDING,
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
                state=JobState.ACTIVE,
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
                state=JobState.ACTIVE,
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
                state=JobState.CANCELLING,
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
                        resources={"v5": 2.5},
                    ),
                ),
                state=JobState.CLEANING,
                command_proc=None,
                cleanup_proc=mock_proc("cleanup", cleanup_poll=1),  # Should have cleanup_proc.
            ),
        }

        # Patch all network calls and utils.
        patch_fns = mock.patch.multiple(
            bastion_vm.__name__,
            _upload_job_state=mock.DEFAULT,
            blob_exists=mock.DEFAULT,
            download_blob=mock.DEFAULT,
            upload_blob=mock.DEFAULT,
            delete_blob=mock.DEFAULT,
            send_signal=mock.DEFAULT,
        )
        with self._patch_bastion(popen_spec()) as bastion, patch_fns as mock_fns:
            bastion._active_jobs = active_jobs
            bastion._update_jobs()

            # Ensure _active_jobs membership stays same.
            self.assertEqual(bastion._active_jobs.keys(), active_jobs.keys())

            expected_states = {
                "pending": JobState.PENDING,
                "resume": JobState.ACTIVE,
                "active": JobState.ACTIVE,
                "preempt": JobState.PENDING,
                "cleaning": JobState.CLEANING,
                "cleaning_cancel": JobState.CLEANING,
                "completed": JobState.COMPLETED,
            }
            for job_name in active_jobs:
                self.assertEqual(bastion._active_jobs[job_name].state, expected_states[job_name])

            for job in bastion._active_jobs.values():
                # For jobs that are ACTIVE, expect command_proc to be non-None.
                if job.state == JobState.ACTIVE:
                    self.assertIsNotNone(job.command_proc)
                    self.assertIsNone(job.cleanup_proc)
                # For jobs that are COMPLETED, expect both procs to be None.
                elif job.state == JobState.COMPLETED:
                    self.assertIsNone(job.command_proc)
                    self.assertIsNone(job.cleanup_proc)

                    # Remote jobspec should not be deleted until gc.
                    for delete_call in mock_fns["delete_blob"].mock_calls:
                        self.assertNotIn(
                            os.path.join(_JOB_DIR, job.spec.name),
                            delete_call.args,
                        )

                # User states should be deleted.
                self.assertTrue(
                    any(
                        os.path.join(bastion._user_state_dir, job.spec.name) in delete_call.args
                        for delete_call in mock_fns["delete_blob"].mock_calls
                    )
                )

                # For jobs that went from ACTIVE to PENDING, expect kill() to have been called.
                if active_jobs[job.spec.name] == JobState.ACTIVE and job.state == JobState.PENDING:
                    mock_fns["send_signal"].assert_called()
                    self.assertFalse(
                        active_jobs[
                            job.spec.name
                        ].command_proc.popen.terminate.called  # pytype: disable=attribute-error
                    )

            for job_name in active_jobs:
                history_file = os.path.join(bastion._job_history_dir, job_name)
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
                project_history_dir = os.path.join(bastion._project_history_dir, project_id)
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
        active_jobs = {}
        init_job_states = {
            "pending": JobState.PENDING,
            "active": JobState.ACTIVE,
            "cleaning": JobState.CLEANING,
            "completed": JobState.COMPLETED,
            "completed_gced": JobState.COMPLETED,
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

        patch_network = mock.patch.multiple(bastion_vm.__name__, delete_blob=mock.DEFAULT)
        with self._patch_bastion() as bastion, patch_network as mock_network:

            def mock_clean(jobs: Dict[str, ResourceMap]) -> Sequence[str]:
                self.assertTrue(
                    all(
                        active_jobs[job_name].state in {JobState.PENDING, JobState.COMPLETED}
                        for job_name in jobs
                    )
                )
                return fully_gced

            with mock.patch.object(bastion, "_cleaner") as mock_cleaner:
                mock_cleaner.configure_mock(**{"sweep.side_effect": mock_clean})
                bastion._active_jobs = active_jobs
                bastion._gc_jobs()

            # Ensure that each fully GC'ed COMPLETED job deletes jobspec and state.
            for job_name in fully_gced:
                deleted_state = any(
                    os.path.join(bastion._state_dir, job_name) in delete_call.args
                    for delete_call in mock_network["delete_blob"].mock_calls
                )
                deleted_jobspec = any(
                    os.path.join(bastion._active_dir, job_name) in delete_call.args
                    for delete_call in mock_network["delete_blob"].mock_calls
                )
                self.assertEqual(
                    active_jobs[job_name].state == JobState.COMPLETED,
                    deleted_state and deleted_jobspec,
                )

    # TODO(markblee): Implement the following checks in sync_jobs.
    # def test_sync_jobs(self):
    #     # Only support user state of CANCELLING.
    #     # CANCELLING/CLEANING should not go back to CANCELLING.

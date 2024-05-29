# Copyright © 2023 Apple Inc.

"""Tests job utilities."""

from absl import flags

from axlearn.cloud.common.job import Job, _with_retry
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.test_utils import TestCase


def _mock_flag_values() -> flags.FlagValues:
    mock_flags = {
        "name": "test_name",
        "command": "test_command",
        "max_tries": 1,
        "retry_interval": 2,
    }
    fv = flags.FlagValues()
    for k, v in mock_flags.items():
        if isinstance(v, int):
            flags.DEFINE_integer(k, v, help=k, flag_values=fv)
        else:
            flags.DEFINE_string(k, v, help=k, flag_values=fv)
    fv.mark_as_parsed()
    return fv


class JobTest(TestCase):
    """Tests job utils."""

    def test_from_flags(self):
        fv = _mock_flag_values()
        cfg = Job.from_flags(fv)
        for k, v in cfg.items():
            if k in fv:
                self.assertEqual(v, fv.get_flag_value(k, None))

    def test_execute(self):
        class FailNJob(Job):
            """A dummy job that fails execute a fixed number of times."""

            @config_class
            class Config(Job.Config):
                n: Required[int] = REQUIRED

            def __init__(self, cfg):
                super().__init__(cfg)
                self.i = 0
                self.delete_called = False

            def _delete(self):
                self.delete_called = True

            def _execute(self):
                if self.i < self.config.n:
                    self.i += 1
                    raise ValueError(f"Deliberate failure {self.i}")

        fv = _mock_flag_values()
        cfg = FailNJob.from_flags(fv)
        cfg.set(max_tries=3, retry_interval=0.1)

        # Test max tries.
        job = cfg.set(n=cfg.max_tries).instantiate()
        with self.assertRaisesRegex(ValueError, "Failed to execute"):
            job.execute()
        self.assertEqual(cfg.max_tries, job.i)
        self.assertTrue(job.delete_called)

        # Test success.
        job = cfg.set(n=cfg.max_tries - 1).instantiate()
        job.execute()
        self.assertEqual(cfg.n, job.i)
        self.assertFalse(job.delete_called)


class UtilTest(TestCase):
    """Tests util functions."""

    def test_with_retry(self):
        class FailN:
            """A dummy callable that fails a fixed number of times."""

            def __init__(self, n):
                self.n = n

            def __call__(self):
                if self.n == 0:
                    return True
                self.n -= 1
                raise ValueError("Failed.")

        self.assertTrue(_with_retry(FailN(2), max_tries=3))

        with self.assertRaisesRegex(ValueError, "Failed"):
            _with_retry(FailN(3), max_tries=2)

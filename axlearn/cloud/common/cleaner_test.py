# Copyright © 2024 Apple Inc.

"""Tests for cleaner.py."""
import datetime
from typing import Dict, Sequence

from absl.testing import parameterized

from axlearn.cloud.common.bastion import new_jobspec
from axlearn.cloud.common.cleaner import Cleaner, CompositeCleaner, UnschedulableCleaner
from axlearn.cloud.common.quota import QuotaInfo
from axlearn.cloud.common.scheduler import JobMetadata, JobScheduler
from axlearn.cloud.common.types import JobSpec
from axlearn.common.config import REQUIRED, Required, config_class, config_for_function


class MockCleaner(Cleaner):
    @config_class
    class Config(Cleaner.Config):
        # Sweep returns job names that are in `names`.
        names: Required[Sequence[str]] = REQUIRED

    def sweep(self, jobs: Dict[str, JobSpec]) -> Sequence[str]:
        cfg = self.config
        return list(set(cfg.names).intersection(jobs.keys()))


class CompositeCleanerTest(parameterized.TestCase):
    def test_sweep(self):
        cleaner_a = MockCleaner.default_config().set(names=["a"])
        cleaner_b = MockCleaner.default_config().set(names=["b"])
        cfg = CompositeCleaner.default_config().set(cleaners=[cleaner_a, cleaner_b])
        cleaner = cfg.instantiate()

        jobs = {"a": None, "b": None, "c": None}
        result = cleaner.sweep(jobs)
        self.assertSequenceEqual(sorted(result), ["a", "b"])


class UnschedulableCleanerTest(parameterized.TestCase):
    def test_sweep(self):
        def mock_jobs(*, prefix: str, resource_count: int):
            return {
                f"{prefix}_a": new_jobspec(
                    name="a",
                    command="echo hello world",
                    metadata=JobMetadata(
                        user_id="user_1",
                        project_id="default",
                        creation_time=datetime.datetime(1900, 1, 1, 0, 0, 0),
                        resources={"aws_4:p5.48xlarge": resource_count},
                    ),
                ),
                f"{prefix}_b": new_jobspec(
                    name="b",
                    command="echo hello world",
                    metadata=JobMetadata(
                        user_id="user_1",
                        project_id="fm-proj-lm-pretrain",
                        creation_time=datetime.datetime(1900, 1, 1, 0, 0, 0),
                        resources={"aws_4:p5.48xlarge": resource_count},
                    ),
                ),
            }

        quota = QuotaInfo(
            total_resources={
                "gcp1:a2-ultragpu-8g": 148.000001,
                "aws_5:p5.48xlarge": 1e-06,
                "aws_2:g5.12xlarge": 5.000001,
                "aws_4:p5.48xlarge": 8.000001,
            },
            project_resources={
                "fm-proj-lm-pretrain": {
                    "gcp1:a2-ultragpu-8g": 400,
                    "aws_5:p5.48xlarge": 400,
                    "aws_2:g5.12xlarge": 400,
                    "aws_4:p5.48xlarge": 400,
                },
            },
        )

        schedulable_jobs = mock_jobs(prefix="schedulable", resource_count=8)
        unschedulable_jobs = mock_jobs(prefix="unschedulable", resource_count=9)
        jobs = schedulable_jobs | unschedulable_jobs

        cfg = UnschedulableCleaner.default_config().set(
            scheduler=JobScheduler.default_config().set(
                quota=config_for_function(lambda: lambda: quota)
            )
        )

        cleaner = cfg.instantiate()
        result = cleaner.sweep(jobs)
        self.assertSequenceEqual(result, list(unschedulable_jobs.keys()))

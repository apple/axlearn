# Copyright © 2024 Apple Inc.

"""A dummy recorder used for measurement tests."""

from axlearn.common import measurement


@measurement.register_recorder("dummy_recorder")
class DummyRecorder(measurement.Recorder):
    @classmethod
    def from_flags(cls, fv) -> measurement.Recorder:
        del fv
        return (
            cls.default_config()
            .set(name="dummy_recorder", upload_dir="/dummy/upload_dir", upload_interval=15)
            .instantiate()
        )

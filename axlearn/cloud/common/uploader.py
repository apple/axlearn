# Copyright © 2023 Apple Inc.

"""A simple artifact uploader."""

import multiprocessing
import time
from typing import Protocol

from absl import logging

from axlearn.common.config import REQUIRED, Configurable, InstantiableConfig, Required, config_class


class UploadFn(Protocol):
    def __call__(self, *, src_dir: str, dst_dir: str):
        """Uploads contents of `src_dir` to `dst_dir`."""


def with_interval(upload_fn: UploadFn, interval_seconds: int = 10) -> UploadFn:
    """Wraps `upload_fn` with a loop that uploads every `interval_seconds`."""

    def fn(*, src_dir: str, dst_dir: str):
        upload_s = 0
        while True:
            logging.log_every_n(
                logging.INFO,
                "Uploading outputs %s -> %s. Last duration: %s",
                6,
                src_dir,
                dst_dir,
                upload_s,
            )
            start = time.time()
            try:
                upload_fn(src_dir=src_dir, dst_dir=dst_dir)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Upload failed: %s", e)
            upload_s = time.time() - start
            if upload_s > interval_seconds:
                logging.warning(
                    "Uploading outputs exceeded interval: %s > %s",
                    upload_s,
                    interval_seconds,
                )
            time.sleep(max(0, interval_seconds - upload_s))

    return fn


class Uploader(Configurable):
    """A utility to periodically upload artifacts."""

    @config_class
    class Config(Configurable.Config):
        """Configures Uploader."""

        src_dir: Required[str] = REQUIRED
        dst_dir: Required[str] = REQUIRED
        upload_fn: Required[InstantiableConfig[UploadFn]] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._upload_proc = None
        self._upload_fn = cfg.upload_fn.instantiate()

    def __call__(self):
        """Uploads outputs from `self.config.src_dir` to `self.config.dst_dir`.

        Internally, this uses a separate process to periodically perform the upload without
        blocking. This can be invoked multiple times to poll the status of the upload process and
        restart it if it has terminated.

        The upload implementation depends on `self.config.upload_fn`.
        """
        cfg: Uploader.Config = self.config

        if self._upload_proc is not None:
            self._upload_proc.join(timeout=0)
            if not self._upload_proc.is_alive():
                logging.info("Upload process died, removing...")
                self._upload_proc.kill()
                self._upload_proc.join()
                self._upload_proc = None
                logging.info("Upload process removed. Will restart...")

        if self._upload_proc is None:
            logging.info("Starting upload process.")
            self._upload_proc = multiprocessing.Process(
                target=self._upload_fn,
                kwargs=dict(src_dir=cfg.src_dir, dst_dir=cfg.dst_dir),
                daemon=True,
            )
            self._upload_proc.start()
            logging.info("Upload process started.")
        else:
            logging.info("Upload process is still running.")

    def cleanup(self):
        """Terminate the upload process."""
        logging.info("Cleanup upload process.")
        if self._upload_proc is not None:
            logging.info("Terminating upload process.")
            self._upload_proc.terminate()
            self._upload_proc.join()  # make sure uploading finished
            self._upload_proc = None
            logging.info("Upload process terminated.")

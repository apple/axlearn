# Copyright © 2023 Apple Inc.

"""Utilities to interact with Google Cloud Storage (GCS)."""
# TODO(markblee): Deprecate and remove.

import json
import os
import random
import time
from typing import Any, Callable
from urllib.parse import urlparse

from absl import flags, logging
from google.auth import default
from google.cloud import storage  # type: ignore
from google.cloud.exceptions import BadRequest
from tqdm import tqdm

from axlearn.cloud.gcp.utils import get_credentials

_GS_CLIENT: storage.Client = None


def get_gs_client() -> storage.Client:
    """Creates or retrieves the GCS client."""
    global _GS_CLIENT  # pylint: disable=global-statement
    if _GS_CLIENT is None:
        _ = get_credentials()  # Checks if we have logged in/have a service account.
        _GS_CLIENT = storage.Client(project=flags.FLAGS.project, credentials=default()[0])
    return _GS_CLIENT


def list_blobs(url: str) -> list[str]:
    """List files on GS with <url> prefix.

    Args:
        url: gs:// prefixed url.

    Returns:
        List of full paths to objects with <url> prefix.
    """
    parse = urlparse(url)
    bucket = get_gs_client().bucket(parse.netloc)
    prefix = parse.path[1:]
    blobs = bucket.list_blobs(prefix=prefix)
    blob_names = [blob.name for blob in blobs if blob.name[-1] != "/"]
    # If this is actually a file, return it, otherwise return all matches
    if prefix in blob_names:
        return [f"gs://{parse.netloc}/{prefix}"]
    return [f"gs://{parse.netloc}/{name}" for name in blob_names if name[-1] != "/"]


def blob_exists(url: str) -> bool:
    """Returns whether the blob exists."""
    # list_blobs returns prefixes that match.
    for blob in list_blobs(url):
        if blob == url:
            return True
    return False


def download_blob(url: str, dst: str):
    """Download object on GS at <url>.

    Args:
        url: gs:// prefixed url.
        dst: Local filepath.

    Raises:
        ValueError: If url does not exactly point to single object.
    """
    blobs = list_blobs(url)
    if len(blobs) != 1:
        raise ValueError(f"url: {url} matched {len(blobs)} items.")
    if blobs[0] != url:
        raise ValueError(f"Requested to delete {url}, only found {blobs[0]}")
    parse = urlparse(url)
    blob = get_gs_client().bucket(parse.netloc).blob(parse.path[1:])
    blob.download_to_filename(dst)


def delete_blob(url: str):
    """Delete object on GS at <url>.

    Args:
        url: gs:// prefixed url.

    Raises:
        ValueError: If url does not exactly point to single object.
    """
    blobs = list_blobs(url)
    if len(blobs) != 1:
        raise ValueError(f"url: {url} matched {len(blobs)} items.")
    if blobs[0] != url:
        raise ValueError(f"Requested to delete {url}, only found {blobs[0]}")
    parse = urlparse(url)
    blob = get_gs_client().bucket(parse.netloc).blob(parse.path[1:])
    blob.delete()


def copy_blob(src: str, dst: str):
    """Copies an object on GS from <src> to <dst>.

    Args:
        src: gs:// prefixed url.
        dst: gs:// prefixed url.
    """
    src, dst = urlparse(src), urlparse(dst)
    client = get_gs_client()
    src_bucket = client.bucket(src.netloc)
    dst_bucket = src_bucket if dst.netloc == src.netloc else client.bucket(dst.netloc)
    src_blob = src_bucket.blob(src.path[1:])
    src_bucket.copy_blob(src_blob, destination_bucket=dst_bucket, new_name=dst.path[1:])


class CRC32MismatchError(IOError):
    pass


def upload_blob(local_path: str, *, url: str, verbose: bool = True):
    """Upload object at local path to url.

    Args:
        local_path: Local object to upload.
        url: Destination path in GCS.
        verbose: Whether to output upload progress.

    Raises:
        ValueError: If local_path doesn't exist, or points to a directory and not a file.
    """
    if not os.path.exists(local_path):
        raise ValueError(f"local_path: {local_path} doesn't exist.")
    if os.path.isdir(local_path):
        raise ValueError(f"local_path: {local_path} is a directory not a file.")
    parse = urlparse(url)

    def _upload(*, blob: storage.Blob, obj: Any, total_bytes: int):
        try:
            blob.upload_from_file(obj, content_type=None, size=total_bytes, timeout=3600)
        except BadRequest as err:
            # pylint: disable-next=protected-access
            reason = json.loads(err._response._content.decode("utf-8"))["error"]
            if (reason["code"] == 400) and (reason["message"] == "Invalid Value"):
                # This is what we get when the uploaded file doesn't have a matching crc32.
                raise CRC32MismatchError("Uploaded file did not have intact crc32c.") from err
            raise RuntimeError(f"Failed to upload file {local_path}.") from err

    def put():
        blob = get_gs_client().bucket(parse.netloc).blob(parse.path[1:])
        with open(local_path, "rb") as of:
            total_bytes = os.fstat(of.fileno()).st_size
            if verbose:
                with tqdm.wrapattr(
                    of, "read", total=total_bytes, miniters=1, desc=f"Uploading to {url}."
                ) as obj:
                    _upload(blob=blob, obj=obj, total_bytes=total_bytes)
            else:
                _upload(blob=blob, obj=of, total_bytes=total_bytes)
        blob.reload()

    _retry_crc32cmismatch(put)


def _retry_crc32cmismatch(fn: Callable):
    # Retry blob download if fails due to CRC32MismatchError.
    cause = None
    for attempt in range(10):
        try:
            return fn()
        except CRC32MismatchError as err:
            cause = err
            sleep_for = 2**attempt + random.random()
            logging.warning(
                "CRC32MismatchError: attempt %d, backing off %0.2f seconds",
                attempt,
                sleep_for,
            )
            time.sleep(sleep_for)
    # pylint: disable-next=broad-exception-raised
    raise Exception("Attempts exhausted.") from cause

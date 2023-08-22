# Copyright © 2023 Apple Inc.

"""TPU job cleaner, e.g. to be used with BastionJob."""

import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Sequence

from absl import logging
from oauth2client.client import GoogleCredentials

from axlearn.cloud.common.cleaner import Cleaner
from axlearn.cloud.common.types import ResourceMap
from axlearn.cloud.gcp.tpu import (
    TPUDeletionError,
    delete_tpu,
    infer_tpu_version,
    list_queued_resource_info,
    list_tpu_info,
)
from axlearn.cloud.gcp.utils import get_credentials


class TPUCleaner(Cleaner):
    """Cleans up unused TPUs."""

    def sweep(self, jobs: Dict[str, ResourceMap]) -> Sequence[str]:
        """Removes TPU resources in a non-blocking manner.

        Note: If jobs use multiple resources, we only remove the TPU resources.
        Jobs are currently responsible for cleaning other resources.

        Args:
            jobs: A mapping {job_name: resource_map} of jobs to delete.

        Returns:
            The list of job_names that are no longer associated with any TPUs. The bastion should
            only consider these jobs fully cleaned (which may be a subset of those provided in the
            `jobs` arg).
        """
        credentials = get_credentials()
        running_tpus = {
            tpu_info.name: tpu_info
            for tpu_info in list_tpu_info(credentials) + list_queued_resource_info(credentials)
        }
        already_terminated = []
        need_termination = []
        for job_name, resources in jobs.items():
            tpu_info = running_tpus.get(job_name)
            # The job has no associated TPU -- it's already terminated.
            if tpu_info is None:
                already_terminated.append(job_name)
            # We found a TPU with the same name as the job. We should terminate it.
            else:
                tpu_version = infer_tpu_version(tpu_info.accelerator_type)
                if tpu_version not in resources:
                    logging.warning(
                        "Found a TPU corresponding to the job %s. "
                        "However, the TPU version %s is not reflected in the job resources %s. "
                        "The TPU will be reclaimed as the job spec is considered invalid.",
                        job_name,
                        tpu_version,
                        resources,
                    )
                need_termination.append(job_name)

        logging.info("Need termination: %s", need_termination)
        logging.info("Already terminated: %s", already_terminated)
        self._delete_batch(need_termination, credentials=credentials)
        return already_terminated

    # pylint: disable-next=no-self-use
    def _delete_batch(self, tpu_names: Sequence[str], *, credentials: GoogleCredentials):
        """Deletes the TPU jobs with the given names.

        We handle deletes in a separate method for easier testing.

        Blocks until all deletion requests are sent, but does not block until deletions finish
        successfully. Hence once the function returns, no more deletion requests will be sent and
        therefore it's safe to re-create the TPU jobs if needed.

        Deletions of TPU jobs may get stuck or fail. Instead, each `sweep` we poll for TPUs that are
        still present and retry the delete if necessary.
        """

        def _delete_tpu_async(tpu_name: str, *, credentials: GoogleCredentials):
            try:
                logging.info("Attempting to delete %s", tpu_name)
                delete_tpu(tpu_name, credentials=credentials, wait=False)
            except TPUDeletionError as e:
                logging.warning("Failed to delete TPU %s: %s", tpu_name, e)

        with ThreadPoolExecutor() as pool:
            pool.map(
                functools.partial(_delete_tpu_async, credentials=credentials),
                tpu_names,
            )

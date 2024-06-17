# Copyright © 2024 Apple Inc.

"""Utils shared between runner implementations."""

from typing import Optional

from absl import logging


def should_recreate_job(
    tier: Optional[str], reservation: Optional[str], is_pending: bool = False
) -> bool:
    """Decides whether the job on `tier` using `reservation` should be recreated.

    Typically, if tier does not match reservation, we may need to recreate the job. To avoid
    frequently restarting jobs which bounce between tiers, jobs scheduled on tier=0 but not using
    the reservation are allowed to continue running until pre-emption, after which point they can be
    recreated on the reservation.

    Args:
        tier: Current scheduling tier.
        reservation: Current reservation status.
        is_pending: Whether the job can be recreated with minimal impact to uptime.

    Returns:
        A verdict of recreate or not.
    """
    if str(tier) != "0" and reservation is not None:
        logging.info(
            "Bastion tier is %s but reservation is %s. Job resources will be recreated.",
            tier,
            reservation,
        )
        return True
    elif str(tier) == "0" and reservation is None:
        if is_pending:
            logging.info(
                "Bastion tier is %s but reservation is %s. "
                "Since the job is pending, will take the opportunity to recreate the job.",
                tier,
                reservation,
            )
            return True
        else:
            logging.info(
                "Bastion tier is %s but reservation is %s. "
                "Since the job is active, will continue to run until pre-emption.",
                tier,
                reservation,
            )
    return False

# Copyright © 2023 Apple Inc.

"""Entrypoint function for launching the fault tolerant agent.

The agent spawns the trainer in a subprocess for fault tolerance and process isolation.
Agent flags and trainer flags are separated by ``--``. Everything after ``--`` is forwarded
verbatim to the trainer subprocess.

Example usage:
python3 -m axlearn.ft.agent \
  --max_restarts=5 \
  -- \
  --module=text.gpt.c4_trainer \
  --config=fuji-70B-v2-flash \
  --trainer_dir=gs://bucket/experiments/test \
  --data_dir=gs://bucket/tensorflow_datasets \
  --jax_backend=tpu
"""

import enum
import shlex
import signal
import subprocess
import sys

from absl import app, flags, logging

from axlearn.ft.monitor import StatusMonitor, StragglerMonitor
from axlearn.ft.utils import TrainerProcessController


class TerminationAction(enum.Enum):
    """Action to take after a termination request."""

    EXIT = "exit"  # Pod shutdown - exit gracefully
    RESTART = "restart"  # Coordinated restart - restart without counting against max_restarts


# Constant for pod shutdown signal reason prefix
_POD_SHUTDOWN_REASON_PREFIX = "Pod shutdown signal"

flags.DEFINE_integer(
    "max_restarts", 3, "Maximum number of times to restart the trainer subprocess on failure"
)
flags.DEFINE_string(
    "trainer_cmd",
    None,
    "The trainer command to run as a subprocess. "
    "If not set, defaults to 'python3 -m axlearn.common.launch_trainer_main'.",
)
flags.DEFINE_boolean(
    "straggler_detection",
    True,
    "Enable detection of workers that are slowing down overall training progress.",
)
flags.DEFINE_float(
    "straggler_worker_sensitivity",
    8.0,
    # pylint: disable=line-too-long
    "Number of standard deviations (for tensorcore utilization) from median to consider a worker a straggler. "
    "Lower values are more sensitive, higher values are more forgiving, 8.0 is a good medium for workloads.",
)
flags.DEFINE_integer(
    "straggler_worker_sustained_duration_seconds",
    300,
    "Seconds a worker must be continuously a straggler before taking action.",
)


def _handle_termination_request(
    process_controller: TrainerProcessController,
) -> TerminationAction | None:
    """Check if termination was requested and determine the appropriate action.

    Args:
        process_controller: The trainer process controller to check for termination requests.

    Returns:
        TerminationAction.EXIT if pod shutdown was requested (caller should return/exit
            gracefully),
        TerminationAction.RESTART if coordinated restart was requested (caller should continue
            without incrementing restart count),
        None if no termination was requested (caller should handle returncode normally).
    """
    termination_requested, reason = process_controller.check_termination_requested()
    if not termination_requested:
        return None

    # Distinguish between pod shutdown (exit) and coordinated restart (restart)
    if reason.startswith(_POD_SHUTDOWN_REASON_PREFIX):
        # SIGTERM received - pod is being killed, exit gracefully
        logging.info(
            "FT Agent: Pod shutdown signal received (%s), exiting gracefully",
            reason,
        )
        return TerminationAction.EXIT

    # Coordinated restart (JAX re-init) - restart trainer without counting against max_restarts
    logging.info(
        "FT Agent: Coordinated restart requested (%s), restarting trainer",
        reason,
    )
    return TerminationAction.RESTART


def _run_single_attempt(
    entrypoint_cmd: list[str],
    restart_count: int,
    max_restarts: int,
    process_controller: TrainerProcessController,
    monitor: "StatusMonitor",
) -> TerminationAction | None:
    """Run a single trainer attempt and return the action to take.

    Args:
        entrypoint_cmd: Command to launch the trainer subprocess.
        restart_count: Current restart attempt number.
        max_restarts: Maximum allowed restarts.
        process_controller: Controller for the trainer process.
        monitor: Status monitor for the trainer.

    Returns:
        TerminationAction.EXIT if the agent should exit gracefully,
        TerminationAction.RESTART if a coordinated restart was requested,
        None if the attempt failed and the caller should increment restart_count and retry.

    Raises:
        RuntimeError: When training fails and max restarts are exhausted.
    """
    with subprocess.Popen(
        entrypoint_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        process_controller.set_process(process)
        returncode = monitor.monitor_training_process(process)
        process_controller.clear_process()

        action = _handle_termination_request(process_controller)
        if action is not None:
            return action

        if returncode == 0:
            logging.info("FT Agent: Training completed successfully")
            return TerminationAction.EXIT

        if restart_count < max_restarts:
            logging.error("FT Agent: Training failed (code %d), restarting...", returncode)
        else:
            logging.error("FT Agent: Max restarts (%d) reached", max_restarts)
            raise RuntimeError(f"Max restarts reached, last exit code: {returncode}")

    return None


def run_ft_agent(trainer_argv: list[str]):
    """The agent launches trainer as a subprocess with fault tolerance.

    Args:
        trainer_argv: Arguments to forward to the trainer subprocess.
    """
    logging.info("Starting fault tolerant trainer agent...")

    # Initialize FT system
    process_controller = TrainerProcessController()
    straggler_monitor = StragglerMonitor(
        enabled=flags.FLAGS.straggler_detection,
        sensitivity=flags.FLAGS.straggler_worker_sensitivity,
        sustained_duration_seconds=flags.FLAGS.straggler_worker_sustained_duration_seconds,
    )
    monitor = StatusMonitor(
        process_controller=process_controller,
        straggler_monitor=straggler_monitor,
    )
    monitor.start()

    # Build trainer command
    if flags.FLAGS.trainer_cmd:
        entrypoint_cmd = shlex.split(flags.FLAGS.trainer_cmd)
    else:
        entrypoint_cmd = [sys.executable, "-m", "axlearn.common.launch_trainer_main"]
    entrypoint_cmd.extend(trainer_argv)

    # Set up signal handling
    def signal_handler(signum, *_):
        logging.warning(
            "FT Agent: Received signal %d, reporting to global manager",
            signum,
        )

        # Report pod shutdown to global manager for coordination
        try:
            monitor.manager.report_pod_shutdown(f"{_POD_SHUTDOWN_REASON_PREFIX} {signum}")
            logging.info("FT Agent: Pod shutdown reported to global manager")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("FT Agent: Failed to report pod shutdown: %s", e)

        # Terminate the trainer process
        logging.debug("FT Agent: Terminating trainer due to signal %d", signum)
        process_controller.terminate_training(f"{_POD_SHUTDOWN_REASON_PREFIX} {signum}")

        logging.warning("FT Agent: exit with non-zero code due to signal %d recieved.", signum)
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)

    max_restarts = flags.FLAGS.max_restarts
    restart_count = 0

    try:
        while restart_count <= max_restarts:
            if restart_count > 0:
                logging.info("FT Agent: Restart attempt %d/%d", restart_count, max_restarts)

            logging.info("FT Agent: Starting trainer: %s", " ".join(entrypoint_cmd))

            try:
                # Start trainer in its own process group (start_new_session=True)
                # This allows os.killpg() to terminate all threads cleanly
                action = _run_single_attempt(
                    entrypoint_cmd, restart_count, max_restarts, process_controller, monitor
                )
                if action == TerminationAction.EXIT:
                    return
                if action == TerminationAction.RESTART:
                    continue  # Coordinated restart - don't increment restart_count
                restart_count += 1

            except RuntimeError as e:
                logging.error("FT Agent: %s", e)
                sys.exit(1)
            except OSError as e:
                logging.error("FT Agent: Failed to start trainer: %s", e)
                process_controller.clear_process()
                if restart_count < max_restarts:
                    restart_count += 1
                else:
                    logging.error("FT Agent: Max restarts reached after exception")
                    sys.exit(1)
    finally:
        monitor.stop()


def main(argv):
    # argv[0] is the program name; argv[1:] is everything after "--".
    run_ft_agent(argv[1:])


if __name__ == "__main__":
    app.run(main)

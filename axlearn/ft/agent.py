# Copyright © 2023 Apple Inc.

"""Entrypoint function for launching the fault tolerant agent.

The agent spawns the trainer in a subprocess for fault tolerance and process isolation.

Example usage:
python3 -m axlearn.ft.agent \
  --module=text.gpt.c4_trainer \
  --config=fuji-70B-v2-flash \
  --trainer_dir=gs://bucket/experiments/test \
  --data_dir=gs://bucket/tensorflow_datasets \
  --jax_backend=tpu \
  --max_restarts=5
"""

import signal
import subprocess
import sys

from absl import app, flags, logging

# Import launch and launch_trainer to get the FLAGS definitions.
from axlearn.common import launch, launch_trainer, measurement  # pylint: disable=unused-import
from axlearn.ft.monitor import StatusMonitor
from axlearn.ft.utils import TrainerProcessController

flags.DEFINE_integer(
    "max_restarts", 3, "Maximum number of times to restart the trainer subprocess on failure"
)


def run_ft_agent():
    """The agent launches trainer as a subprocess with fault tolerance."""
    logging.info("Starting fault tolerant trainer agent...")

    # Initialize FT system
    process_controller = TrainerProcessController()
    monitor = StatusMonitor(process_controller=process_controller)
    monitor.start()

    # Build trainer command
    entrypoint_cmd = [sys.executable, "-m", "axlearn.common.launch_trainer_main"]
    entrypoint_cmd.extend(arg for arg in sys.argv[1:] if not arg.startswith("--max_restarts"))

    # Set up signal handling
    def signal_handler(signum, *_):
        logging.info("FT Agent: Received signal %d, terminating trainer", signum)
        process_controller.terminate_training(f"Signal {signum}")

    signal.signal(signal.SIGTERM, signal_handler)

    max_restarts = flags.FLAGS.max_restarts
    restart_count = 0

    try:
        while restart_count <= max_restarts:
            if restart_count > 0:
                logging.info("FT Agent: Restart attempt %d/%d", restart_count, max_restarts)

            logging.info("FT Agent: Starting trainer: %s", " ".join(entrypoint_cmd))

            try:
                with subprocess.Popen(
                    entrypoint_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                ) as process:
                    process_controller.set_process(process)
                    returncode = monitor.monitor_training_process(process)
                    process_controller.clear_process()

                    if returncode == 0:
                        logging.info("FT Agent: Training completed successfully")
                        return
                    elif restart_count < max_restarts:
                        restart_count += 1
                        logging.error(
                            "FT Agent: Training failed (code %d), restarting...", returncode
                        )
                    else:
                        logging.error("FT Agent: Max restarts (%d) reached", max_restarts)
                        sys.exit(returncode)

            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("FT Agent: Failed to start trainer: %s", e)
                process_controller.clear_process()
                if restart_count < max_restarts:
                    restart_count += 1
                else:
                    logging.error("FT Agent: Max restarts reached after exception")
                    sys.exit(1)
    finally:
        monitor.stop()


def main(_):
    run_ft_agent()


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)

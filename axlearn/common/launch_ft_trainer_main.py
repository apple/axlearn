# Copyright © 2023 Apple Inc.

"""Main function for launching the fault tolerant trainer.

This is a simplified wrapper that executes the trainer in a subprocess for
fault tolerance and process isolation.

Example usage:
python3 -m axlearn.common.launch_ft_trainer_main \
  --module=text.gpt.c4_trainer \
  --config=fuji-70B-v2-flash \
  --trainer_dir=gs://bucket/experiments/test \
  --data_dir=gs://bucket/tensorflow_datasets \
  --jax_backend=tpu \
  --max_restarts=5
"""

import os
import signal
import subprocess
import sys
from typing import Optional

from absl import app, flags, logging

# Import launch and launch_trainer to get the FLAGS definitions.
from axlearn.common import launch, launch_trainer, measurement  # pylint: disable=unused-import

flags.DEFINE_integer(
    "max_restarts", 3, "Maximum number of times to restart the trainer subprocess on failure"
)


def forward_signal_to_trainer(trainer_process: Optional[subprocess.Popen]):
    """Set up signal handler to forward signals to the trainer subprocess."""

    def signal_handler(signum, *_):
        """Forward signals to the trainer subprocess."""
        if trainer_process is not None:
            logging.info("Forwarding signal %d to trainer subprocess", signum)
            try:
                trainer_process.send_signal(signum)
            except ProcessLookupError:
                pass
        else:
            logging.info("Ft trainer received signal %d, no active trainer process", signum)
            sys.exit(1)

    # Only register the SIGTERM for the preemption.
    signal.signal(signal.SIGTERM, signal_handler)


def run_ft_trainer():
    """Run the trainer as a subprocess for fault tolerance."""
    logging.info("Starting fault tolerant trainer...")

    trainer_process: Optional[subprocess.Popen] = None

    # Build command to run the original trainer
    cmd = [sys.executable, "-m", "axlearn.common.launch_trainer_main"]

    # Pass through all other arguments except FT-specific ones
    for arg in sys.argv[1:]:
        if not arg.startswith("--max_restarts"):
            cmd.append(arg)

    max_restarts = flags.FLAGS.max_restarts
    restart_count = 0

    while restart_count <= max_restarts:
        if restart_count > 0:
            logging.info("Ft trainer restart attempt %d/%d", restart_count, max_restarts)

        logging.info("Ft trainer starting trainer subprocess: %s", " ".join(cmd))

        try:
            # Run trainer as subprocess
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                env=os.environ.copy(),
            ) as process:
                # Update global process reference and set up signal handler
                trainer_process = process
                forward_signal_to_trainer(trainer_process)
                # Stream output in real-time
                if process.stdout:
                    for line in iter(process.stdout.readline, ""):
                        if line:
                            # Use print instead of logging to avoid double formatting.
                            print(f"[FT_TRAINER] {line}", end="", flush=True)

                # Wait for completion
                returncode = process.wait()

                # Clear process reference
                trainer_process = None

                if returncode == 0:
                    logging.info("Trainer completed successfully in ft trainer")
                    return
                else:
                    logging.error("Trainer process exited with code %d", returncode)
                    if restart_count < max_restarts:
                        restart_count += 1
                        continue
                    else:
                        logging.error(
                            "Trainer process reachees max restarts (%d). Exiting...", max_restarts
                        )
                        sys.exit(returncode)

        except (subprocess.SubprocessError, OSError) as e:
            logging.error("Ft trainer failed to run trainer subprocess: %s", e)
            # Clear process reference on exception
            trainer_process = None
            if restart_count < max_restarts:
                restart_count += 1
                continue
            else:
                logging.error(
                    "Trainer process reaches max restarts (%d) after exception. Exiting...",
                    max_restarts,
                )
                sys.exit(1)


def main(_):
    run_ft_trainer()


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)

# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""
from absl import app, logging

from axlearn.common import launch, launch_trainer

from ml_goodput_measurement import goodput
import jax

def main(_):
    launch.setup()
    run_name='test'
    goodput_logger_name = f'goodput_{run_name}'
    # TODO: create Goodput Recorder object
    goodput_recorder = goodput.GoodputRecorder(job_name=run_name, logger_name=goodput_logger_name, logging_enabled=(jax.process_index() == 0))
    # TODO: record job's overall start time
    goodput_recorder.record_job_start_time()
    logging.info("===========>STARTED JOB")

    trainer_config = launch_trainer.get_trainer_config()
    launch_trainer.run_trainer(trainer_config)

    goodput_recorder.record_job_end_time()
    logging.info("===========>STOPPED JOB")


if __name__ == "__main__":
    app.run(main)

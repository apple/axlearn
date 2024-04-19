#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127
#SBATCH --exclusive
#SBATCH --nodes=4

srun  --kill-on-bad-exit=1  run_trainer.sh
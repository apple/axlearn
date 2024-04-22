#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127
#SBATCH --exclusive
#SBATCH --nodes=8
#SBATCH --exclude=compute1-st-kaena-training-0-[20,26,64-65,112,204,228,123,147,58,53,174,188,178,18]

srun  --kill-on-bad-exit=1  run_trainer.sh
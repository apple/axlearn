#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --exclusive
#SBATCH --nodes=1

#Usage : $1=training script $2=test_artifact_sub_directory
#Example : run_trn.sh run_trainer.sh my_test_dir
srun  --kill-on-bad-exit=1 "$@"

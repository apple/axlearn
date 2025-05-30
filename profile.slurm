#!/bin/bash
#SBATCH --job-name rh_profile
#SBATCH --nodes 1
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

# Example: sb -N 2 rh_profile.slurm 7519 2node_seqlen2k_dim4k
DATE=`date '+%Y%m%d-%H%M%S'`

export S3_PROFILE_BASE_PATH=s3://kaena-tempdata/huilgolr/fs-moe/profiles
export PROFILE_JOB_NAME=$2
export PROFILE_JOB_ID=$1
export AXLEARN_PROFILE_MODE=capture

srun -l setup_node.sh ../may-artifacts/
srun -l runner.sh

#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%x_%j.out

DATE=`date '+%Y%m%d-%H%M%S'`
srun -l setup_node.sh ../mar-artifacts/
srun -l runner.sh profile ${1} s3://kaena-tempdata/huilgolr/fs-moe/profiles/$DATE/${2}

#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

export AXLEARN_NUM_LAYERS=$((4 * SLURM_NNODES))
export AXLEARN_REMAT_LAYER=nonmatmul
export AXLEARN_MODEL_NAME="envy-Mistral-16x10B"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((16 * SLURM_NNODES))
export AXLEARN_MAX_SEQ_LEN=4096
#export AXLEARN_REPEATED=1
#export AXLEARN_PROFILE_MODE="tracerun"
#export CUSTOM_TAG_rsshift=1

if [ ${1:-0} = "1" ]; then
    srun -l setup_node.sh ../apr-artifacts/
else
    echo "Skip installing"
fi
srun -l runner.sh

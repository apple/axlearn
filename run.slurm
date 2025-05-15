#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

export AXLEARN_NUM_LAYERS=$((4 * SLURM_NNODES))
export AXLEARN_REMAT_LAYER=true
export AXLEARN_MODEL_NAME="envy-Mistral-16x10B"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((16 * SLURM_NNODES))
export AXLEARN_MAX_SEQ_LEN=4096
export AXLEARN_USE_BLOCKWISE=1
# export AXLEARN_USE_BLOCKWISE_MLP_KERNEL=0
# export AXLEARN_REPEATED=1
# export AXLEARN_PROFILE_MODE="tracerun"
# export AXLEARN_JAX_BACKEND=cpu

if [ ${1:-1} = "1" ]; then
    srun -l ./setup_node.sh ../apr-artifacts-shrihari/
else
    echo "Skip installing"
fi
srun -l runner.sh

#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

export AXLEARN_NUM_LAYERS=$((4 * SLURM_NNODES))
export AXLEARN_REMAT_LAYER=selkernel
export AXLEARN_MODEL_NAME="envy-Mistral-16x10B"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((8 * SLURM_NNODES))
export AXLEARN_MAX_SEQ_LEN=8192
export AXLEARN_USE_BLOCKWISE=1

# set the env to use here
# it expects the env to be at ../$VENV_NAME
export VENV_NAME=jaxmoe

# Set to use repeated, make sure to use /fsx/huilgolr/may-artifacts/repeated/libneuronxla-2.2.20250521+7e624b6.dev-py3-none-linux_x86_64
# export AXLEARN_REPEATED=1

# When not using repeated, latest pjrt is /fsx/huilgolr/may-artifacts/stacked/libneuronxla-2.2.20250522+e95a622.dev-py3-none-linux_x86_64.whl

# Setting below two flags profiles on 3rd step and exits
# export AXLEARN_PROFILE_MODE="tracerun"
# export PROFILE_JOB_NAME="4l_150b_bs16_4k_selkernelremat_repeated_1n"

# Set to disable kernel, runs into compiler error for 150b at least
# export AXLEARN_USE_BLOCKWISE_MLP_KERNEL=0

# export AXLEARN_JAX_BACKEND=cpu

if [ ${1:-1} = "1" ]; then
    srun -l ./setup_node.sh ../may-artifacts/
else
    echo "Skip installing"
fi
srun -l runner.sh

#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

export AXLEARN_REMAT_LAYER=selective
export AXLEARN_MODEL_NAME="envy-Mistral-8x20B"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((16 * SLURM_NNODES))
export AXLEARN_USE_BLOCKWISE=1

# set the env to use here
# it expects the env to be at ../$VENV_NAME
export VENV_NAME=jaxmoe

# Set to use repeated, make sure to use /fsx/huilgolr/may-artifacts/repeated/libneuronxla-2.2.20250521+7e624b6.dev-py3-none-linux_x86_64
# export AXLEARN_REPEATED=1

# When not using repeated, latest pjrt is /fsx/csavsrin/artifacts_kernel_csavsrin/test_remat/libneuronxla-2.2.20250530+c4f06ac.dev-py3-none-linux_x86_64.whl

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

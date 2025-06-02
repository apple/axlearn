#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --job-name rh_16x10b_convergence_gpu
#SBATCH --output logs/%j_%x.out
#SBATCH --switches 1
#SBATCH --nodes 8
export AXLEARN_NUM_LAYERS=$((4 * SLURM_NNODES))
export AXLEARN_REMAT_LAYER=selective
export VENV_NAME=jaxmoe
export AXLEARN_MODEL_NAME="envy-Mistral-16x10B"
export JOB_ID="16x10b_convergence"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((16 * SLURM_NNODES))
export AXLEARN_CAP_FACTOR=2
export AXLEARN_USE_BLOCKWISE=1
export AXLEARN_REPEATED=1
export AXLEARN_NUM_KV_HEADS=16
export AXLEARN_SHUFFLE_FILES=0
export DATA_SEED=42

export JAX_COMPILATION_CACHE_DIR="cache/"
mkdir -p ${JAX_COMPILATION_CACHE_DIR}

export CUSTOM_TAG_experiment=convergence
export RENAME_JOB=true
export RENAME_JOB_PREFIX=rh

if [ ${1:-1} = "1" ]; then
    srun -l ./setup_node.sh ../may-artifacts/
else
    echo "Skip installing"
fi

srun -l runner.sh

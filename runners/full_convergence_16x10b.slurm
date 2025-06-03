#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --job-name rh_16x10b_convergence_gpu
#SBATCH --output logs/%j_%x.out
#SBATCH --nodes 8
#SBATCH --switches 1

export AXLEARN_REMAT_LAYER=selective
export VENV_NAME=jaxmoe
export AXLEARN_MODEL_NAME="envy-Mistral-16x10B"
export JOB_ID="16x10b_convergence_nofused"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((8 * SLURM_NNODES))
export AXLEARN_CAP_FACTOR=2
export AXLEARN_USE_BLOCKWISE=1
export AXLEARN_REPEATED=1
export AXLEARN_NUM_KV_HEADS=16
export AXLEARN_SHUFFLE_FILES=0
export DATA_SEED=42
export SAVE_EVERY_N_STEPS=500

#export NEURON_ALL_REDUCE_UPCASTER=1
#export NEURON_PROMOTE_TP_REDUCE=1

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

#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out
#SBATCH --time 3:00:00

# #SBATCH --nodes 8
# #SBATCH --switches 1

export AXLEARN_NUM_LAYERS=4
export AXLEARN_REMAT_LAYER=selective
export VENV_NAME=jaxmoe
export AXLEARN_MODEL_NAME="envy-Mistral-8x20B"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((8 * SLURM_NNODES))
export AXLEARN_USE_BLOCKWISE=1
export AXLEARN_REPEATED=1

#export AXLEARN_PROFILE_MODE="tracerun" 
#export PROFILE_JOB_NAME="4l_8x20b_bs8_8k_selectiveremat_repeated_8n"

export RENAME_JOB=true
export RENAME_JOB_PREFIX=rh
#export NEURON_ALL_REDUCE_UPCASTER=1
#export NEURON_PROMOTE_TP_REDUCE=1


if [ ${1:-1} = "1" ]; then
    srun -l ./setup_node.sh ../may-artifacts/
else
    echo "Skip installing"
fi

srun -l runner.sh

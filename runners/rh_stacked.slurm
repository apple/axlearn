#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

export AXLEARN_NUM_LAYERS=$((4 * SLURM_NNODES))
export AXLEARN_REMAT_LAYER=selective
export VENV_NAME=jaxmoestacked
export AXLEARN_MODEL_NAME="envy-Mistral-8x20B"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((8 * SLURM_NNODES))

export AXLEARN_USE_BLOCKWISE=1
# export AXLEARN_USE_BLOCKWISE_MLP_KERNEL=0
export AXLEARN_REPEATED=0

#export NEURON_ALL_REDUCE_UPCASTER=1
#export NEURON_PROMOTE_TP_REDUCE=1

# export AXLEARN_JAX_BACKEND=cpu
#export CUSTOM_TAG_experiment=skipdma
#export CUSTOM_TAG_experiment=fsdpoverlap
#export CUSTOM_TAG_experiment=noembedshard

# export CUSTOM_TAG_experiment=pjrtdebug
# export CUSTOM_TAG_pjrt=main_agsplitfix

export RENAME_JOB=true
export RENAME_JOB_PREFIX=rh

#export AXLEARN_PROFILE_MODE="tracerun"
#export PROFILE_JOB_NAME="actual4l_16x10b_topk4_cf4_bs8_8k_selectiveremat_stacked_8n_tuned"

if [ ${1:-1} = "1" ]; then
    srun -l ./setup_node.sh ../may-artifacts/
else
    echo "Skip installing"
fi

srun -l runner.sh

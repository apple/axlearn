#!/bin/bash
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out
#SBATCH --time 1:00:00

export AXLEARN_NUM_LAYERS=$((4 * SLURM_NNODES))
export AXLEARN_REMAT_LAYER=true
export AXLEARN_MODEL_NAME="envy-Mistral-toy"
export AXLEARN_TP_DEGREE=16
export AXLEARN_TRAIN_BATCH_SIZE=$((16 * SLURM_NNODES))
export AXLEARN_MAX_SEQ_LEN=256
export AXLEARN_USE_BLOCKWISE=1
# export AXLEARN_REPEATED=1
# export AXLEARN_PROFILE_MODE="tracerun"
export AXLEARN_JAX_BACKEND=cpu

export CUSTOM_TAG_experiment=kernel

# export CUSTOM_TAG_experiment=pjrtdebug
export CUSTOM_TAG_pjrt=moe

export RENAME_JOB=true
export RENAME_JOB_PREFIX=rh

if [ ${1:-1} = "1" ]; then
    srun -l setup_node.sh ../apr-artifacts-shrihari/
else
    echo "Skip installing"
fi

source ../jaxmoe/bin/activate
# if [ "$CUSTOM_TAG_pjrt" = "nodownprojfix" ]; then
#    pip install --no-deps --force-reinstall ../apr-artifacts/repeatedmoe_8e99724/libneuronxla-2.2.20250417-py3-none-linux_x86_64.whl
#elif [ "$CUSTOM_TAG_pjrt" = "repmoemain" ]; then
#    pip install --no-deps --force-reinstall ../apr-artifacts/libneuronxla-2.2.20250425.dev0+571f075-py3-none-linux_x86_64.whl
#else
    # working repeated_moe
#    pip install --no-deps --force-reinstall ../apr-artifacts/libneuronxla-2.2.20250425.dev0+398a057-py3-none-linux_x86_64.whl
#fi

srun -l runner.sh

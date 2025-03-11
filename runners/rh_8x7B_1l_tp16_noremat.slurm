#!/bin/bash
#SBATCH --nodes 2
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

export AXLEARN_MIXTRAL_MOE=1
export AXLEARN_NUM_LAYERS=1
export AXLEARN_REMAT_LAYER=false
export AXLEARN_MODEL_NAME="envy-Mistral-8x7B"
export AXLEARN_TP_DEGREE=16
export AXLEARN_JAX_BACKEND=neuron
export FOR_PROFILE=0

if [ ${1:-1} = "1" ]; then
    srun -l setup_node.sh ../mar-artifacts/
else
    echo "Skip installing"
fi

srun -l runner.sh
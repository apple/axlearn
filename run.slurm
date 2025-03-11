#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%j_%x.out

export AXLEARN_MIXTRAL_MOE=${1:-1}
export AXLEARN_NUM_LAYERS=${2:-4}
export AXLEARN_REMAT_LAYER=${3:-true}
export AXLEARN_MODEL_NAME=${4:-"envy-Mistral-toy"}
export AXLEARN_TP_DEGREE=${5:-4}
export AXLEARN_JAX_BACKEND=${6:-neuron}
srun -l setup_node.sh ../mar-artifacts/
srun -l runner.sh
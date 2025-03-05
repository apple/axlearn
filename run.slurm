#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 127
#SBATCH --output logs/%x_%j.out

srun -l setup_node.sh ../mar-artifacts/
srun -l runner.sh ${1:-1} ${2:-4} ${3:-neuron}
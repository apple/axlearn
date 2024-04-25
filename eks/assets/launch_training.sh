#!/bin/bash
# Editable paths
OUTPUT_DIR="/shared/axlearn_out"
# NEURON_DUMP_PATH=${PWD}/neuron_dump
# HLO_DUMP_PATH=${PWD}/hlo_dump

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --num-concat-graphs=8'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
#export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export CCOM_SOCKET_IFNAME=eth0

# Neuron env vars for distributed training
nodes=`/neuron/scripts/nodelist_helper.py`
devices_per_node=32
export COORDINATOR_ADDRESS=$(echo "$nodes" | head -n 1):64272
export NEURON_RT_ROOT_COMM_ID=$(echo "$nodes" | head -n 1):5552
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $OMPI_COMM_WORLD_SIZE | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$OMPI_COMM_WORLD_RANK
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1

unset OMPI_MCA_orte_hnp_uri
set

mkdir -p $OUTPUT_DIR

# Run the training script
DATA_DIR=gs://axlearn-public/tensorflow_datasets
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn1n.32xlarge-64 \
    --distributed_coordinator=$COORDINATOR_ADDRESS \
    --num_processes=$OMPI_COMM_WORLD_SIZE \
    --process_id=$OMPI_COMM_WORLD_RANK 2>&1 | tee ${OUTPUT_DIR}/${PMIX_HOSTNAME}.log

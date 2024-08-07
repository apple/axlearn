#!/usr/bin/env bash
sudo dpkg -i /home/ptoulme/compiler_5th/aws-neuronx-runtime-lib-2.x.x.x-b0ad3b9d9.deb
sudo dpkg -i /home/ptoulme/compiler_5th/aws-neuronx-collectives-2.x.x.x-99c2ee88a.deb

PY_VENV_PATH="/home/ptoulme/axlearn_venv/bin/activate"
source ${PY_VENV_PATH}

cd /axlearn

ARTIFACTS_PATH="/home/ptoulme/artifacts"
TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${TIMESTAMP}"
mkdir -p "$TEST_ARTIFACTS_PATH"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

vnc_size=2

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=${vnc_size} --internal-hlo2tensorizer-options='--verify-hlo --fuse-dot-logistic=false'"
export NEURON_RT_VIRTUAL_CORE_SIZE=${vnc_size}
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
#--internal-compiler-debug-mode=all
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --distribution-strategy=llm-training"
# export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --num-concat-graphs=8'" # Set indside fuji.py with gradient_accumulation size
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --disable-internal-dge"

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
#export NEURON_TRANSFORMER_SHARDING=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1

# Neuron env vars for distributed training based on SLURM
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# num_nodes=$(echo "$nodes" | wc -l)
# process_idx=$(echo "$nodes" | grep -n "$SLURMD_NODENAME" | cut -d: -f1)
# devices_per_node=32
# MASTER_ADDR=$(echo "$nodes" | head -n 1)
# MASTER_PORT=41000
# JAX_COORDINATOR_PORT=41001
# export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
# export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
# export NEURON_PJRT_PROCESS_INDEX=$((process_idx - 1))
# export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
# export FI_LOG_LEVEL="warn"
# export FI_EFA_USE_DEVICE_RDMA="1"
# export FI_PROVIDER="efa"
# export FI_EFA_FORK_SAFE=1

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Run the training script
python -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v1 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=trn2 


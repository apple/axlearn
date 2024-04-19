#!/usr/bin/env bash
set -o pipefail
set -e

ulimit -n 65535
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
sudo sysctl -w net.ipv4.ip_local_reserved_ports=41000
if which lctl >/dev/null 2>&1; then
    sudo lctl set_param 'osc.*.max_dirty_mb=64' # Cap max space each connection to FSx reserves so we avoid OODs
fi
IPS=""
for h in $(scontrol show hostname); do
    IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
done
HOSTS=(${IPS//\ / })
NODEID=$SLURM_NODEID
NTASKS=$SLURM_NTASKS
export PROCESSES_PER_NODE=1
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"

export NEURON_RT_EXEC_TIMEOUT=100
export DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

export PJRT_DEVICE="NEURON"
export NEURON_RT_NUM_CORES=32
export NEURON_PJRT_PROCESS_INDEX=$NODEID
export RANK=$NODEID
export PJRT_LOCAL_PROCESS_COUNT=1
export WORLD_SIZE=$((NTASKS * 32))
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '32,'%.0s $(seq 1 $NTASKS) | sed 's/,$//')
export XLA_FLAGS="--xla_force_host_platform_device_count=32 --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=./jax_dump --xla_dump_hlo_pass_re='.*'"
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1 # for init and small graphs

# export NEURON_CC_FLAGS="--dump=./compiler_dump --framework=XLA --model-type transformer --no-internal-hlo-remat --distribution-strategy=llm-training --enable-mixed-precision-accumulation -O1"
export NEURON_CC_FLAGS="--dump=./compiler_dump --framework=XLA --model-type transformer --no-internal-hlo-remat --distribution-strategy=llm-training --internal-hlo2tensorizer-options='--verify-hlo --num-concat-graphs=8' --enable-mixed-precision-accumulation -O1"
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=5

# export TF_CPP_MIN_LOG_LEVEL=0 # Enable SPMD verbose logging - 0 means most verbose
# export TF_CPP_MAX_VLOG_LEVEL=2 # Needs above flag for logging but goes in reverse. 0 means no log
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MAX_VLOG_LEVEL=2


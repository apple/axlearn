#!/usr/bin/env bash

set -e
# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ -z "$SLURM_JOB_NODELIST" ]; then
	nodes="localhost"
	SLURM_NODEID=0
fi

num_nodes=$(echo "$nodes" | wc -l)
devices_per_node=64
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$SLURM_NODEID

# Print nodenames for debug
hostname

JOB_ID=${SLURM_JOB_ID}
ARTIFACTS_PATH="artifacts/"
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${JOB_ID}"
if [ "$1" -ne "profile" ]; then
	mkdir -p "$TEST_ARTIFACTS_PATH"
fi
NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
PROFILE_DUMP_PATH=${TEST_ARTIFACTS_PATH}/profiles

export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
# export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_snapshots"
export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_proto"

# PJRT Flags 
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2
export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1
export NEURON_FSDP=1
export NEURON_FSDP_NUM_LAYER_COALESCE=-1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export NEURON_HLO_ANALYZER=1
export NEURON_DISABLE_BOUNDARY_MARKER=1

# Neuron runtime flags
export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1

# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope --verify-hlo'"
if [ "$FOR_PROFILE" = "1" ]; then
	export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-compiler-debug-mode=penguin"
	export XLA_IR_DEBUG=1
	export XLA_HLO_DEBUG=1
fi
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --auto-cast=none"

# use to add debug logging at module level in xla
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_VMODULE="gather_scatter_handler=5"

# JAX Cache
# export JAX_COMPILATION_CACHE_DIR="cache/"
# mkdir -p ${JAX_COMPILATION_CACHE_DIR}

deactivate || true
source ../jaxmoe/bin/activate

echo "Listing apt dependencies"
apt list --installed | grep neuron
echo "Listing pip dependencies"
pip list | grep neuron
echo "Done listing dependencies"
printenv | grep NEURON
printenv | grep XLA
printenv | grep AXLEARN || true
which python

# TC MALLOC HACK
LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
 
if [ -n "$LIBTCMALLOC" ]; then
	# Create a symbolic link to the found libtcmalloc version
	sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
	echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"
		     
		       # Export LD_PRELOAD
	export LD_PRELOAD=/usr/lib/libtcmalloc.so
	echo "LD_PRELOAD set to: $LD_PRELOAD"
else
	echo "Error: libtcmalloc.so not found"
	exit 1
fi

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# fuji-7B-v2-flash

if [ "$AXLEARN_JAX_BACKEND" == "cpu" ]; then
	export XLA_FLAGS="${XLA_FLAGS} --xla_force_host_platform_device_count=64 "
	export JAX_PLATFORMS="cpu"
	jax_backend="cpu"
else
	jax_backend="neuron"
fi

upload_profile() {
	neff_path=$1
	ntff_path=$2
	profile_name=$3
	set -x
	curl -X POST -m 1800 -F "neff=@${neff_path}" -F "ntff=@${ntff_path}" -F "name=${profile_name}" http://localhost:8050/api/upload > /dev/null || true
	set +x
}

profile() {
	set -ex
	neff_path=$1
	profile_dir=$2
	s3_profile_path=$3
	profile_id=$4
	NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 NEURON_RT_PROFILE_BUF_DMA_MB=256 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1 NEURON_RT_VIRTUAL_CORE_SIZE=2 neuron-profile capture -r 64 --num-exec 3 \
		--collectives-worker-count $((64* $SLURM_JOB_NUM_NODES)) \
		--collectives-worker-start-id $((64 * $SLURM_PROCID)) \
		-i 0 \
		-n $neff_path \
		-s $profile_dir/profile.ntff

	echo "Done profiling"
	
	upload_dir=$(realpath $profile_dir/to_upload)
	mkdir -p $upload_dir
	cp $profile_dir/profile_rank_0_exec_3.ntff $upload_dir
	cd $(dirname $neff_path)
	cp file.neff $upload_dir
	set +e
	tar -cvf penguin-text.tar penguin-sg*
	cp penguin-text.tar $upload_dir
	set -e
	if [ $SLURM_PROCID -eq 0 ]; then
		aws s3 sync $upload_dir $s3_profile_path
		echo "Profile uploaded to $s3_profile_path"
		echo "profile-upload -F \"s3=$s3_profile_path\" -F name=$profile_id -F \"profiler-opts='--enable-memory-tracker'\""
	fi
	
}

if [ "$1" = "profile" ]; then
	job_dir=artifacts/$2
	s3_profile_path=$3
	profile_id=$4
	NEFF_PATH=$(ls ${job_dir}/neuron_dump/**/file.neff | tail -n1)
	echo "Using $NEFF_PATH for profiling"
	PROFILE_DIR=${job_dir}/profiles
	mkdir -p $PROFILE_DIR
	profile $NEFF_PATH $PROFILE_DIR $s3_profile_path $profile_id
else
	# MIXTRAL_MOE being
	# 0 adds dense MLP layers
	# 1 adds all sparse MLP layers
	# 2 adds alternating sparse and dense layers
	# export MIXTRAL_MOE=$1
	# export NUM_LAYERS=$2
	# envy-Mistral-${AXLEARN_MODEL_NAME}
	python -m axlearn.common.launch_trainer_main \
		--module=text.gpt.c4_trainer --config=$AXLEARN_MODEL_NAME \
		--trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
		--jax_backend=$jax_backend --mesh_selector=neuron-trn2.48xlarge-64 \
		--distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
		--process_id=$NEURON_PJRT_PROCESS_INDEX
fi
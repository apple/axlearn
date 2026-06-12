# Replication Guide: GKE TPU Benchmarking (Image Streaming vs. Standard Pull)

This guide provides the exact Dockerfile configurations, CLI commands, and JAX model configurations required to replicate the GKE Image Streaming (GCFS) vs. Standard Pull benchmarks on TPU v6e-16 slices.

---

## 📦 1. Dockerfile & Stress-Test Image Construction

To isolate **physical image size** from **code execution paths**, we inflate JAX training images using a passive, highly compressible zero-data layer. This forces GKE `containerd` to decompress and write physical gigabytes to the node's local SSD during cold pulls, while ensuring JAX executes the exact same code on startup.

### The Dockerfile Setup:
Create a `Dockerfile` using AXLearn's base TPU target, and append a zero-dummy file of the desired size:

```dockerfile
# Use the official AXLearn TPU base image (already contains JAX, TF, and PyGrain)
FROM us-docker.pkg.dev/kuiyue-gke-dev/axlearn/tpu:base

# Append a passive zero-dummy layer to inflate the physical image size
# Example: Replace <SIZE_MB> with 8100 for a 10GB total image, or 48100 for a 50GB image.
RUN dd if=/dev/zero of=/root/dummy_layer bs=1M count=<SIZE_MB>
```

---

## ⚙️ 2. GKE Node Pool Provisioning (True Cold Starts)

To ensure scientific isolation, **always delete any existing TPU node pool first to completely wipe GKE's local containerd cache!**

### Test Case A: GKE Image Streaming Disabled (Standard Pull)
```bash
# Create a fresh TPU v6e node pool without image streaming
gcloud container node-pools create tpu-benchmark-pool \
  --cluster=kuiyue-axlearn \
  --region=us-central1 \
  --node-locations=us-central1-b \
  --machine-type=ct6e-standard-4t \
  --tpu-topology=4x4 \
  --num-nodes=4 \
  --spot \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --quiet

# Update taints to allow AXLearn scheduling
gcloud container node-pools update tpu-benchmark-pool \
  --cluster=kuiyue-axlearn \
  --region=us-central1 \
  --node-taints=google.com/tpu=present:NoSchedule --quiet
```

### Test Case B: GKE Image Streaming Enabled (GCFS)
```bash
# Create a fresh TPU v6e node pool WITH image streaming enabled
gcloud container node-pools create tpu-benchmark-pool \
  --cluster=kuiyue-axlearn \
  --region=us-central1 \
  --node-locations=us-central1-b \
  --machine-type=ct6e-standard-4t \
  --tpu-topology=4x4 \
  --num-nodes=4 \
  --spot \
  --enable-image-streaming \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --quiet

# Update taints to allow AXLearn scheduling
gcloud container node-pools update tpu-benchmark-pool \
  --cluster=kuiyue-axlearn \
  --region=us-central1 \
  --node-taints=google.com/tpu=present:NoSchedule --quiet
```

---

## 🚀 3. The Exact AXLearn/JAX Launch Command

Submit the training job using the `axlearn gcp launch run` command. We use the `localcode` bundler type and point `image_id` to our pre-built tag to bypass the local bundling phase.

```bash
# 1. Disable the bastion rescheduling daemon so it does not interrupt during long pulls
export BASTION_TIER=disabled

# 2. Launch the JAX job on GKE
axlearn gcp launch run \
  --instance_type=tpu-v6e-16 \
  --name=bench-fuji-run \
  --bundler_type=localcode \
  --bundler_spec=image_id=us-docker.pkg.dev/kuiyue-gke-dev/axlearn/tpu:<IMAGE_TAG> \
  -- \
  python3 -m axlearn.common.launch_trainer_main \
  --module=text.gpt.c4_trainer \
  --config=fuji-golden-run-test-v1 \
  --trainer_dir=gs://kuiyue-gke-dev-axlearn/benchmarks/bench-fuji-run \
  --data_dir=FAKE \
  --mesh_selector=tpu-v6e-16 \
  --jax_backend=tpu
```

---

## 🧬 4. The JAX Code & Model Configuration

The training run executes a LLaMA-style GPT model defined in the AXLearn codebase:
*   **The Entry Point:** `python3 -m axlearn.common.launch_trainer_main` (initializes the distributed JAX/PJRT runtime across all 4 GKE hosts).
*   **The JAX Model Code:** The model architecture, layers, and data pipeline are defined under the [`fuji-golden-run-test-v1`](axlearn/experiments/text/gpt/fuji.py#L1193-L1208) configuration class in **[axlearn/experiments/text/gpt/fuji.py](file:///usr/local/google/home/kuiyue/src/axlearn/axlearn/experiments/text/gpt/fuji.py)**.
*   **Execution:** JAX compiles the computational graph using XLA and executes the training steps on the 16 TPU v6e chips. The job terminates automatically after **5 steps** (defined in the golden run config) to keep benchmarking runs fast.

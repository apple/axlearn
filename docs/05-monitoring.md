# GCP Monitoring

This document describes how to enable GCP workload monitoring.

---

## Overview

Google Cloud offers a **workload performance monitoring feature** to track workload metrics and ensure performance thresholds are met. If thresholds are not met, the Google Cloud on-call team is alerted.

To enable monitoring:
1. Publish workload performance indicators and heartbeat metrics to Cloud Monitoring.
2. Collaborate with your Customer Engineer (CE) to define alert thresholds.

---

## Pre-requisites

- GCP account with a GKE cluster.
- Service account with access to Storage Bucket, Artifact Repository, and GKE.
- At least 2 node pools with **16x16 TPU v6e machines**.
- Workload Identity enabled on the cluster.

---

## Setup Instructions

### Step 1: Edit Configuration File
Copy and edit the default configuration file:
```bash
cp .axlearn/axlearn.default.config .axlearn/.axlearn.config
```

Ensure the following fields are configured:
```bash
project="my-gcp-project"
zone="my-zone"
labels="my-unique-label"
enable_gcp_workload_monitoring=true
workload_id="my_workload_id" # Optional
replica_id="0" # Optional
```

### Configuration Details
- **`labels`**: Unique identifier for the configuration.
- **`enable_gcp_workload_monitoring`**: Set to `true` to enable monitoring.
- **`workload_id`**: Unique ID for the workload. Defaults:
  1. From config file (highest precedence).
  2. From `JOBSET_NAME` or `JOB_NAME` environment variables.
  3. `"unknown"` (fallback).
- **`replica_id`**: Replica index (e.g., `"0"`, `"1"`). Default: `"0"`.

---

## Supported Environments

### **TPU GKE**
Set the environment variable:
```bash
--name=<jobset-name> --env=JOBSET_NAME:<jobset-name>
```

---

## Installing AXLearn

1. Install AXLearn:
    ```bash
    export BASTION_TIER=0 # Use 0 for reserved, 1 for spot
    pip install -e '.[gcp]'
    ```

2. Launch jobs using AXLearn:
    ```bash
    axlearn gcp config activate --label=<my-unique-label>

    axlearn gcp gke start \
        --instance_type=<instance-type> \
        --num_replicas=<num-replicas> \
        --cluster=<cluster-name> \
        --bundler_type=artifactregistry \
        --bundler_spec=image=tpu \
        --reservation=<reservation-name> \
        --name=<jobset-name> \
        --env=JOBSET_NAME:<jobset_name> \
        -- python3 -m axlearn.common.launch_trainer_main \
            --module=text.gpt.c4_trainer \
            --config=fuji-7B-v2-flash \
            --trainer_dir=gs://<your-bucket-name>/<train-dir>-gke-v6e-7b/ \
            --data_dir=gs://axlearn-public/tensorflow_datasets \
            --jax_backend=tpu \
            --mesh_selector=<instance-type> \
            --trace_at_steps=16
    ```

---

## Notes
- Ensure proper permissions for GCP resources.
- Collaborate with Google Cloud teams for alert configurations.

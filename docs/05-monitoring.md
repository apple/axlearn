# GCP Monitoring

**This doc is still under construction.**

This document describes how to enable GCP workload monitoring.

## Overview

With GCP monitoring, you can currently send performance and heartbeat of your accelerators to Cloud monitoring to setup alerts and notifications.

The following sections guide you through setting up and launching a job via Bastion.

<br>

### Pre-requisites

We assume you have:
1. GCP account and GKE cluster setup.
2. Service account with access to Storage Bucket, Artifact Repository and GKE.

<br>

### Instructions

```bash
#Copy and edit the default config file. 
cp .axlearn/axlearn.default.config .axlearn/.axlearn.config
```
For monitoring, make sure the following are present.

```bash
project='my-gcp-project'
zone='us-east5'
enable_gcp_workload_monitoring = true
workload_id = "my_workload_id"
replica_id = "0"
```

Build a Docker image using the `Dockerfile.tpu` in the repo:
```bash
docker build -f Dockerfile.tpu --target tpu -t <path>/tpu:latest .
docker push <path>/tpu:latest
```

Next, we create a GKE Job yaml file, which defines the cluster, nodepools, topology etc:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: axlearn-headless-svc
  namespace: default
spec:
  clusterIP: None
  selector:
    job-name: axlearn-workload-job
---
apiVersion: batch/v1
kind: Job
metadata:
  name: axlearn-workload-job
  namespace: default
spec:
  completionMode: Indexed  # Required for TPU workloads
  backoffLimit: 0
  completions: 4
  parallelism: 4
  template:
    metadata:
      labels:
        job-name: axlearn-workload-job
    spec:
      restartPolicy: Never
      subdomain: axlearn-headless-svc
      tolerations:
      - key: "google.com/tpu"
        operator: "Exists"
        effect: "NoSchedule"
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
        cloud.google.com/gke-tpu-topology: 4x4
      dnsPolicy: ClusterFirstWithHostNet  # Ensure proper name resolution for TPU pods
      containers:
      - name: test-container
        image: <path>/tpu:latest
        ports:
        - containerPort: 8471  # Default TPU communication port
        - containerPort: 9431  # TPU metrics port for monitoring
        command:
        - /bin/bash
        - -c
        - |
          echo "Job starting!";
          axlearn gcp config activate --label=axlearn-exp-trill;
          python3 -m axlearn.common.launch_trainer_main --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash --trainer_dir=gs://<train-dir>-axlearn/<train-dir>-gke-v6e-7b/ --data_dir=gs://axlearn-public/tensorflow_datasets --jax_backend=tpu --mesh_selector=tpu-v6e-16 --trace_at_steps=16 
          echo "Job completed!";

        env:
        - name: JAX_PLATFORMS
          value: "tpu"  # Let JAX auto-detect TPU
        - name: JAX_USE_PJRT_C_API_ON_TPU
          value: "1"
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/secrets/key.json"  # Path to the mounted service account key
        volumeMounts:
        - name: gcs-key
          mountPath: "/secrets"
          readOnly: true
        resources:
          requests:
            google.com/tpu: "4"  # Adjust based on TPU topology
          limits:
            google.com/tpu: "4"
      volumes:
      - name: gcs-key
        secret:
          secretName: gcs-key
```

Please modify the yaml file as needed.

Finally, launch the GKE service:
```bash
kubectl apply -f <job-name>.yaml
```

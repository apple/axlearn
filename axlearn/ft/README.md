# AXLearnFT

## Background

### Starting a training job

When we start an AXLearn job, Bastion asks GKE to:

1. provision TPU VMs,
2. schedule Kubernetes pods on those VMs, and
3. start a trainer process inside each pod.
   - Inside each pod, Kuberentes starts a main container, along with other sidecar containers
   - The trainer process runs in each main container.

After that, all trainer processes call `jax.distributed.initialize` so they can discover one another and collectively form a device mesh across all TPUs.

### Fault recovery

If any trainer process fails, the error propagates and causes the other trainers to fail as well. Each failing trainer ends its pod. After all pods exit, GKE is configured to restart the AXLearn job and repeat the three steps above. The restarted job then loads the latest checkpoint and continues training.

## The problem

In many cases, recovery only requires the last step: restarting the trainer processes. The first two steps take minutes, while the last step takes less than a second. It would be ideal to try restarting only the trainer processes when a job fails. If that does not recover the job, we can fall back to letting GKE restart the pods.

## The solution

The key is to keep the pod running even when its trainer process fails. Since a pod ends when its main process ends, we design AXLearn FT to run an agent process as the pod’s main process instead of the trainer. The agent spawns the trainer, monitors it, and restarts it if it fails.

The agents must also propagate failures so that all trainers restart together. This allows the restarted trainers to call `jax.distributed.initialize` again and rebuild the device mesh.

## Design concerns

### How agents monitor trainers

1. Monitor crashes.
   The agent detects trainer termination by monitoring its subprocess.
2. Monitor hangs.
   The agent parses step numbers from trainer log output. If no step progress occurs for a configurable period (default 600 seconds), the agent assumes the trainer is hung.
3. Diagnose failures.
   The agent monitors trainer logs and captures output to aid fast failover and debugging.

### How agents propagate a failure

Agents could publish incidents to etcd and have peers subscribe, but that may overload a cluster-wide service. Instead, the system uses a 3-tier hierarchy with gRPC communication:

1. Each worker reports status to its replica manager (worker 0 in each replica).
2. Each replica manager aggregates worker status and reports to the global manager (replica 0, worker 0).
3. When the global manager detects a failure, it sends restart requests to all replica managers, which forward the requests to their workers.

This hierarchical approach scales better than broadcasting and avoids overloading centralized services.

### When agents fall back to GKE

Sometimes, restarting trainers is not enough. For example, if a TPU has failed, repeatedly restarting the trainer on that VM will continue to fail. In such cases the agent should stop after a configured number of retries and allow GKE to reprovision and restart the pods.

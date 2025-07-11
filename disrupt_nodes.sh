#!/bin/bash
# --- !!! WARNING: This script performs destructive operations continuously. Use with extreme caution. !!! ---
# USAGE: ./disrupt_nodes.sh <pod-prefix>
#
# Continuously finds a pod starting with <pod-prefix> and deletes its GCE node.
# WARNING: This script runs in a destructive loop. Use with extreme caution.
#
# Example: ./disupt_nodes.sh my-job-worker-
# ---

# Check if a pod prefix was passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <pod-prefix>"
    echo "Please provide the jobset name as the pod prefix to identify the target pods."
    exit 1
fi

# The prefix for the pods you want to target (passed as the first argument)
POD_PREFIX="$1"

# Set your GCP Project ID and Zone here
# GCP_PROJECT_ID="cloud-tpu-best-effort-colo"
# GCP_ZONE="us-east5-a"

GCP_PROJECT_ID="tpu-prod-env-one-vm"
GCP_ZONE="southamerica-west1-a"

# Log file path
LOG_FILE="./disrupt_nodes.log"

# Sleep duration in seconds (e.g., 3600 for 1 hour)
SLEEP_SECONDS=7200

# --- Script Starts ---

# Function for logging with timestamps
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_FILE}"
}

# Ensure gcloud/kubectl are in PATH
# export PATH="/snap/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

log "--- Script initialized. Target pod prefix: '${POD_PREFIX}'. Loop starts now. ---"

# Main loop
while true; do
    log "--- New iteration started ---"

    # 1. Get the name of the first pod that matches the specified prefix.
    #    Using awk is slightly more robust than `grep | head | awk`.
    #    It finds the first line where the first column starts with the prefix, prints the name, and exits.
    POD_NAME=$(kubectl get pods --no-headers=true | awk -v prefix="${POD_PREFIX}" '$1 ~ "^"prefix {print $1; exit}')

    if [ -z "${POD_NAME}" ]; then
        log "No running pod found with prefix '${POD_PREFIX}'. Skipping deletion for this cycle."
    else
        log "Identified target pod: ${POD_NAME}"

        # 2. Get the node name (VM instance name) where the pod is running.
        NODE_NAME=$(kubectl get pod "${POD_NAME}" -o=jsonpath='{.spec.nodeName}' 2>/dev/null)

        if [ -z "${NODE_NAME}" ]; then
            log "Could not determine node for pod ${POD_NAME}. It might be terminating. Skipping."
        else
            log "Pod ${POD_NAME} is on node: ${NODE_NAME}"

            # 3. Delete the underlying Compute Engine VM instance.
            log "Attempting to delete GCE instance: ${NODE_NAME} in zone ${GCP_ZONE}"

            # The --quiet flag suppresses the interactive confirmation prompt.
            kubectl exec -it ${POD_NAME} -c ${POD_PREFIX} -- sh -c "kill -s SIGILL 1" 2>&1 | tee -a "${LOG_FILE}"

            # Check the exit code of the gcloud command
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                log "Successfully initiated deletion for node: ${NODE_NAME}"
            else
                log "ERROR: Failed to delete node: ${NODE_NAME}. See gcloud output above."
            fi
        fi
    fi

    log "--- Iteration finished. Sleeping for ${SLEEP_SECONDS} seconds... ---"
    sleep "${SLEEP_SECONDS}"
done

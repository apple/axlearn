#!/bin/bash
#
# This script finds pods that are stuck in the 'Terminating' state for more
# than a specified duration and forcibly deletes them.

# --- Configuration ---
# The maximum duration (in seconds) a pod is allowed to be in the Terminating state.
STUCK_DURATION_SECONDS=300

# How often (in seconds) the script should check for stuck pods.
CHECK_INTERVAL_SECONDS=60
# --- End Configuration ---

echo "Starting pod termination monitor..."
echo "Stuck Threshold: ${STUCK_DURATION_SECONDS}s | Check Interval: ${CHECK_INTERVAL_SECONDS}s"

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking for stuck pods..."

    # Get all pods with a deletion timestamp in JSON format.
    # The 'deletionTimestamp' field is only present for pods that are being terminated.
    stuck_pods_json=$(kubectl get pods -o json | jq -c '.items[] | select(.metadata.deletionTimestamp)')

    if [ -z "$stuck_pods_json" ]; then
        echo "No pods are currently in a terminating state."
    else
        echo "$stuck_pods_json" | while read -r pod_json; do
            # Extract details from the pod's JSON data
            pod_name=$(echo "$pod_json" | jq -r '.metadata.name')
            pod_namespace=$(echo "$pod_json" | jq -r '.metadata.namespace')
            deletion_timestamp_str=$(echo "$pod_json" | jq -r '.metadata.deletionTimestamp')

            # Sanitize the timestamp for macOS `date` by removing fractional seconds and the 'Z' suffix.
            # This handles formats like "2024-01-01T12:34:56.123456Z" -> "2024-01-01T12:34:56"
            sanitized_timestamp_str=$(echo "$deletion_timestamp_str" | sed -e 's/\.[0-9]*Z$/Z/' -e 's/Z$//')

            # Convert the RFC3339 timestamp to a Unix epoch timestamp
            # Works on both GNU and BSD (macOS) date commands.
            if date --version >/dev/null 2>&1; then # GNU date
                deletion_ts=$(date -d "$sanitized_timestamp_str" +%s)
            else # BSD date
                # On macOS, use -u to interpret the time as UTC.
                deletion_ts=$(date -u -jf "%Y-%m-%dT%H:%M:%S" "$sanitized_timestamp_str" +%s)
            fi

            # Get the current time as a Unix epoch timestamp
            now_ts=$(date +%s)

            # Calculate how long the pod has been terminating
            duration=$((now_ts - deletion_ts))

            echo "  - Checking pod '$pod_name' in namespace '$pod_namespace' (terminating for ${duration}s)"

            if [ "$duration" -gt "$STUCK_DURATION_SECONDS" ]; then
                echo "    -> STUCK! Pod '$pod_name' has been terminating for ${duration}s. Forcing deletion."
                # Force delete the pod. The --grace-period=0 is crucial.
                kubectl delete pod "$pod_name" -n "$pod_namespace" --force --grace-period=0
            fi
        done
    fi

    echo "Check complete. Sleeping for ${CHECK_INTERVAL_SECONDS} seconds..."
    sleep "$CHECK_INTERVAL_SECONDS"
done

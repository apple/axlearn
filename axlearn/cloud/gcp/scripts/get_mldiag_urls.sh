#!/usr/bin/env bash
#
# Extracts profiler session TensorBoard URLs and Cloud Console URL
# for a given ML run as JSON.
#
# Usage:
#   ./axlearn/cloud/gcp/scripts/get_mldiag_urls.sh <RUN_NAME> <LOCATION> <PROJECT>

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <RUN_NAME> <LOCATION> <PROJECT>" >&2
  echo "Example: $0 rapatchi-fuji-golden-mldiag-run us-central1 rapatchiexperiment-b6dk7n" >&2
  exit 1
fi

RUN="$1"
LOCATION="$2"
PROJECT="$3"

# Resolve display name to resource ID if needed
RUN_ID=$(gcloud alpha mldiagnostics machine-learning-run list \
  --location="${LOCATION}" \
  --project="${PROJECT}" \
  --filter="displayName:${RUN} OR name:${RUN} OR workloadDetails.gke.id:${RUN}" \
  --format="value(name)" \
  --limit=1 \
  --quiet)

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="${RUN}"
fi

HEX_ID="${RUN_ID##*/}"
CONSOLE_URL="https://console.cloud.google.com/cluster-director/diagnostics/details/${LOCATION}/${HEX_ID}?project=${PROJECT}"

# Fetch profiler sessions as JSON and format with jq
gcloud alpha mldiagnostics profiler-session list \
  --machine-learning-run="${RUN_ID}" \
  --location="${LOCATION}" \
  --project="${PROJECT}" \
  --filter="dashboardUri:*" \
  --format=json \
  --quiet | jq \
    --arg run "${RUN}" \
    --arg console "${CONSOLE_URL}" \
    '{
      ($run): {
        console_url: $console,
        profiler_sessions: [ .[] | { ((.name | split("/") | last)): .dashboardUri } ]
      }
    }'

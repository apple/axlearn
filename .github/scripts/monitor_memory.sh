#!/bin/bash
# Simple memory monitor that prints to console periodically.
# Designed for GitHub Actions monitoring.

INFO="\033[90m"
RESET="\033[0m"

echo -e "${INFO}====== Starting memory monitor at $(date) ======${RESET}"

while true; do
  # Get current timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  echo ""
  echo -e "${INFO}=== Memory Check: $timestamp ===${RESET}"

  # Print simplified memory stats.
  free -h | grep "Mem:" | awk "{printf \"${INFO}Memory: %s used, %s free, %s total\n${RESET}\", \$3, \$4, \$2}"

  # Print memory usage percentage.
  mem_total=$(free | grep Mem | awk '{print $2}')
  mem_used=$(free | grep Mem | awk '{print $3}')
  mem_percent=$(awk "BEGIN {printf \"%.1f\", $mem_used/$mem_total*100}")
  echo -e "${INFO}Memory usage: $mem_percent%${RESET}"

  # Sleep for 30 seconds
  sleep 30
done

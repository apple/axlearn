#!/bin/bash

base_path="/sys/devices/virtual/neuron_device/neuron0"

# Sum across core0 and core1
sum_files() {
    awk '{sum += $1} END {print sum}' "$1" "$2"
}

for subdir in "$base_path/neuron_core0/stats/memory_usage/device_mem"/*; do
    if [ -d "$subdir" ]; then
        dirname=$(basename "$subdir")
        core0_file="$base_path/neuron_core0/stats/memory_usage/device_mem/$dirname/peak"
        core1_file="$base_path/neuron_core1/stats/memory_usage/device_mem/$dirname/peak"
        if [ -f "$core0_file" ] && [ -f "$core1_file" ]; then
            echo -n "$dirname: "
            sum_files "$core0_file" "$core1_file"
        fi
    fi
done

echo "Peak:"
sum_files "$base_path/neuron_core0/stats/memory_usage/device_mem/peak" "$base_path/neuron_core1/stats/memory_usage/device_mem/peak"

echo -e "\nTotal:"
sum_files "$base_path/neuron_core0/stats/memory_usage/device_mem/total" "$base_path/neuron_core1/stats/memory_usage/device_mem/total"
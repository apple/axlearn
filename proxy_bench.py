"""A script to benchmark JAX device_put throughput."""

import asyncio
import os
import time

import jax
import numpy as np
import pathwaysutils
from jax.sharding import SingleDeviceSharding


async def benchmark_host_to_device_throughput(
    device_put_buffer_mb: int = 512, num_transfers: int = 10
):
    """Benchmarks JAX device_put throughput from CPU host to a v5e-32 TPU slice."""
    print(f"JAX version: {jax.__version__}")
    devices = jax.devices() if os.environ.get("JAX_PLATFORMS") else jax.local_devices()
    num_devices = len(devices)
    print(f"Available devices: {num_devices}")

    data_bytes_per_device = int(device_put_buffer_mb * 1024 * 1024)
    dtype = np.float32
    num_elements = data_bytes_per_device // np.dtype(dtype).itemsize
    data_gb_per_device = num_elements * np.dtype(dtype).itemsize / (1024**3)

    print(
        f"Creating {num_devices} NumPy arrays of shape ({num_elements},) type {dtype}, size"
        f" {data_gb_per_device:.2f} GiB each"
    )
    host_arrays = [np.arange(num_elements, dtype=dtype) for _ in range(num_devices)]
    shardings = [SingleDeviceSharding(device) for device in devices]

    loop = asyncio.get_running_loop()
    transfer_times = []

    print(f"Starting benchmark ({num_transfers} transfers)...")
    for i in range(num_transfers):
        if i == 0:
            trace_dir = "gs://cloud-tpu-multipod-dev-axlearn/stoelinga-proxy-benchmark"
            jax.profiler.start_trace(f"{trace_dir}/{device_put_buffer_mb}mb")

        start_time = time.perf_counter()

        # Issue device_put calls in parallel.
        device_put_futures = [
            loop.run_in_executor(None, jax.device_put, host_arrays[j], shardings[j])
            for j in range(num_devices)
        ]
        device_arrays = await asyncio.gather(*device_put_futures)

        # Block until all transfers are complete.
        for device_array in device_arrays:
            device_array.block_until_ready()

        end_time = time.perf_counter()

        duration = end_time - start_time
        transfer_times.append(duration)
        print(f"Transfer {i+1}/{num_transfers}: {duration:.4f} seconds")

        if i == 0:
            jax.profiler.stop_trace()

        # Optional: hint for early deletion.
        del device_arrays

    avg_time = np.mean(transfer_times)
    print(f"\nAverage time per parallel device_put batch: {avg_time:.4f} seconds")

    total_data_moved_gb = data_gb_per_device * num_devices
    throughput_gb_s = total_data_moved_gb / avg_time

    print(f"Data per device: {data_gb_per_device:.2f} GiB")
    print(f"Total data transferred from host per operation: {total_data_moved_gb:.2f} GiB")
    print(f"Aggregated Host -> Devices Throughput: {throughput_gb_s:.2f} GiB/s")
    print(f"Aggregated Host -> Devices Throughput: {throughput_gb_s * 8:.2f} Gbps/s")


if __name__ == "__main__":
    if os.environ.get("JAX_PLATFORMS") == "proxy":
        pathwaysutils.initialize()
    else:
        jax.distributed.initialize()
    scenarios_mb = [1, 128, 1024, 2048]
    for scenario in scenarios_mb:
        print(f"Running scenario {scenario}MB")
        asyncio.run(benchmark_host_to_device_throughput(scenario))

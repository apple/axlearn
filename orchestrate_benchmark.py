import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("benchmark_orchestrator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
CLUSTER = "kuiyue-axlearn"
REGION = "us-central1"
ZONE = "us-central1-b"
PROJECT = "kuiyue-gke-dev"
POOL_NAME = "tpu-benchmark-pool"
REGISTRY = "us-docker.pkg.dev/kuiyue-gke-dev/axlearn/tpu"
INSTANCE_TYPE = "tpu-v6e-16"

# Image sizes and their dummy sizes in MB (Base is ~1.9GB)
IMAGE_CONFIGS = {
    "base": 0,          # Base image (~1.9GB)
    "5gb": 3100,        # ~5GB total
    "8gb": 6100,        # ~8GB total
    "15gb": 13100,      # ~15GB total
    "20gb": 18100,      # ~20GB total
    "35gb": 33100,      # ~35GB total
    "50gb": 48100       # ~50GB total
}

def run_cmd(cmd, check=True, timeout=None):
    logging.info(f"Running command: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout
    )
    if check and result.returncode != 0:
        logging.error(f"Command failed with return code {result.returncode}")
        logging.error(f"Stdout: {result.stdout}")
        logging.error(f"Stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result

def build_and_push_images():
    logging.info("Starting image building phase...")
    # Base image is already built as 'base'
    for tag, dummy_size_mb in IMAGE_CONFIGS.items():
        if tag == "base":
            continue
        logging.info(f"Building bloated image for tag: {tag} ({dummy_size_mb}MB dummy)...")
        dockerfile_content = f"""
FROM {REGISTRY}:base
RUN dd if=/dev/zero of=/root/dummy_file bs=1M count={dummy_size_mb}
"""
        dockerfile_path = f"Dockerfile.{tag}"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        try:
            # Build
            run_cmd(f"docker build -f {dockerfile_path} -t {REGISTRY}:{tag} .")
            # Push
            run_cmd(f"docker push {REGISTRY}:{tag}")
        finally:
            if os.path.exists(dockerfile_path):
                os.remove(dockerfile_path)
    logging.info("All images built and pushed successfully.")

def delete_node_pool():
    logging.info(f"Deleting node pool {POOL_NAME} (if exists)...")
    cmd = (
        f"gcloud container node-pools delete {POOL_NAME} "
        f"--cluster={CLUSTER} --region={REGION} --quiet"
    )
    try:
        run_cmd(cmd, check=True)
        logging.info(f"Node pool {POOL_NAME} deleted successfully.")
    except Exception as e:
        # Ignore if it doesn't exist
        logging.info(f"Node pool deletion failed (it might not exist): {e}")

def create_node_pool(streaming_enabled):
    logging.info(f"Creating node pool {POOL_NAME} (Streaming={streaming_enabled})...")
    streaming_flag = "--enable-image-streaming" if streaming_enabled else ""
    cmd = (
        f"gcloud container node-pools create {POOL_NAME} "
        f"--cluster={CLUSTER} --region={REGION} "
        f"--node-locations={ZONE} --machine-type=ct6e-standard-4t "
        f"--tpu-topology=4x4 --num-nodes=4 --spot {streaming_flag} "
        f"--scopes=https://www.googleapis.com/auth/cloud-platform --quiet"
    )
    run_cmd(cmd, check=True)
    logging.info(f"Node pool {POOL_NAME} created successfully.")
    
    # Remove spot taint / replace with TPU taint
    logging.info("Updating node pool taints...")
    taint_cmd = (
        f"gcloud container node-pools update {POOL_NAME} "
        f"--cluster={CLUSTER} --region={REGION} "
        f"--node-taints=google.com/tpu=present:NoSchedule --quiet"
    )
    run_cmd(taint_cmd, check=True)
    logging.info("Node pool taints updated.")

def launch_job(job_name, tag):
    logging.info(f"Launching Fuji job {job_name} with image tag {tag}...")
    # We use localcode bundler and point image_id to our specific tag
    image_id = f"{REGISTRY}:{tag}"
    cmd = (
        f"export CLOUDSDK_CORE_DISABLE_PROMPTS=1 && "
        f"source .venv/bin/activate && "
        f"axlearn gcp launch run --instance_type={INSTANCE_TYPE} --name={job_name} "
        f"--bundler_type=localcode --bundler_spec=image_id={image_id} -- "
        f"python3 -m axlearn.common.launch_trainer_main "
        f"--module=text.gpt.c4_trainer --config=fuji-golden-run-test-v1 "
        f"--trainer_dir=gs://kuiyue-gke-dev-axlearn/benchmarks/{job_name} "
        f"--data_dir=FAKE --mesh_selector={INSTANCE_TYPE} --jax_backend=tpu"
    )
    logging.info("Starting launch command in background...")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable="/bin/bash"
    )
    logging.info(f"Launch process started with PID {proc.pid}.")
    return proc

def monitor_job_and_extract_metrics(job_name):
    logging.info(f"Monitoring job {job_name}...")
    config.load_kube_config()
    v1 = client.CoreV1Api()
    
    timeout = 900 # 15 minutes timeout to allow for slow scheduling/pulls
    start_time = time.time()
    
    t_pull = None
    t_import = None
    t_compile = None
    
    container_started_time = None
    pod_scheduled_time = None
    first_log_time = None
    step_1_time = None
    
    while time.time() - start_time < timeout:
        try:
            # Always list pods dynamically to handle recreations/rescheduling
            pods = v1.list_namespaced_pod(namespace="default", label_selector=f"jobset.sigs.k8s.io/jobset-name={job_name}")
            if not pods.items:
                logging.info("Waiting for pod to be created...")
                time.sleep(5)
                continue
                
            # Use the latest active pod
            pod = pods.items[0]
            pod_name = pod.metadata.name
            status = pod.status
            
            # 1. Capture Pod Scheduled Time
            if not pod_scheduled_time:
                for condition in status.conditions:
                    if condition.type == "PodScheduled" and condition.status == "True":
                        pod_scheduled_time = condition.last_transition_time
                        logging.info(f"Pod {pod_name} scheduled at: {pod_scheduled_time}")
                        break
            
            # 2. Capture Container Started Time
            if status.container_statuses and not container_started_time:
                c_status = status.container_statuses[0]
                if c_status.state.running:
                    container_started_time = c_status.state.running.started_at
                    logging.info(f"Container {pod_name} started at: {container_started_time}")
                    # Calculate t_pull immediately once we have both
                    if pod_scheduled_time:
                        t_pull = (container_started_time - pod_scheduled_time).total_seconds()
                        logging.info(f"Measured Container Provisioning (t_pull): {t_pull}s")
                elif c_status.state.terminated:
                    logging.error(f"Container {pod_name} terminated prematurely!")
                    raise RuntimeError("Container terminated prematurely")
            
            # 3. Read logs to extract t_import and step_1_time
            if container_started_time and not step_1_time:
                try:
                    # Read logs with timestamps
                    logs = v1.read_namespaced_pod_log(name=pod_name, namespace="default", timestamps=True)
                    lines = logs.strip().split("\n")
                    
                    for line in lines:
                        if not line:
                            continue
                        parts = line.split(" ", 1)
                        if len(parts) < 2:
                            continue
                        ts_str, content = parts
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        
                        if not first_log_time and "config.py:" in content:
                            first_log_time = ts
                            logging.info(f"First JAX/AXLearn log at: {first_log_time}")
                            if container_started_time:
                                t_import = (first_log_time - container_started_time).total_seconds()
                                logging.info(f"Measured JAX Init & Import (t_import): {t_import}s")
                            
                        if not step_1_time and "step: 1" in content:
                            step_1_time = ts
                            logging.info(f"Step 1 completed at: {step_1_time}")
                            if first_log_time:
                                t_compile = (step_1_time - first_log_time).total_seconds()
                                logging.info(f"Measured XLA Compile & Step 1 (t_compile): {t_compile}s")
                            break
                except ApiException:
                    # Logs might not be ready yet, ignore and retry
                    pass
            
            # Check if we have all metrics
            if t_pull is not None and t_import is not None and t_compile is not None:
                logging.info("All metrics successfully extracted!")
                break
                
        except ApiException as e:
            logging.warning(f"API exception in monitor loop: {e}")
            
        time.sleep(10)
        
    # Clean up the JobSet
    logging.info(f"Deleting jobset {job_name}...")
    run_cmd(f"kubectl delete jobset {job_name}", check=False)
    
    if t_pull is None or t_import is None or t_compile is None:
        raise RuntimeError("Failed to extract all metrics within timeout.")
        
    return t_pull, t_import, t_compile


def run_benchmark():
    # Build and push images first
    build_and_push_images()
    
    results = []
    
    # We will loop through each configuration
    try:
        for tag in IMAGE_CONFIGS.keys():
            for streaming_enabled in [True, False]:
                mode_str = "enabled" if streaming_enabled else "disabled"
                job_name = f"bench-{tag}-{mode_str}"
                
                logging.info(f"==================================================")
                logging.info(f"Starting Test Case: Image={tag}, Streaming={mode_str}")
                logging.info(f"==================================================")
                
                # 1. Ensure COLD START by deleting and recreating the node pool
                delete_node_pool()
                create_node_pool(streaming_enabled)
                
                # 2. Launch the job
                t_pull, t_import, t_compile = None, None, None
                proc = None
                try:
                    proc = launch_job(job_name, tag)
                    # 3. Monitor and measure
                    t_pull, t_import, t_compile = monitor_job_and_extract_metrics(job_name)
                    status = "SUCCESS"
                except Exception as e:
                    logging.error(f"Test case failed: {e}")
                    status = f"FAILED: {str(e)}"
                finally:
                    if proc:
                        logging.info(f"Terminating launch process PID {proc.pid}...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logging.warning("Process did not terminate, killing...")
                            proc.kill()
                
                results.append({
                    "tag": tag,
                    "size_gb": tag_to_size_gb(tag),
                    "streaming": streaming_enabled,
                    "status": status,
                    "t_pull": t_pull,
                    "t_import": t_import,
                    "t_compile": t_compile,
                    "t_e2e": (t_pull + t_import + t_compile) if t_pull and t_import and t_compile else None
                })
                
                # Write intermediate results to a file
                with open("benchmark_results_partial.json", "w") as f:
                    json.dump(results, f, indent=2)
                    
    finally:
        # ALWAYS delete the node pool at the end to save costs!
        logging.info("Final cleanup: deleting benchmark node pool...")
        delete_node_pool()
        
    # Write final results
    write_final_report(results)

def tag_to_size_gb(tag):
    if tag == "base": return 1.9
    if tag == "5gb": return 5.0
    if tag == "8gb": return 8.0
    if tag == "15gb": return 15.0
    if tag == "20gb": return 20.0
    if tag == "35gb": return 35.0
    if tag == "50gb": return 50.0
    return 0.0

def write_final_report(results):
    logging.info("Writing final report...")
    # Generate Markdown table
    report = """# Benchmark Results: GKE Image Streaming vs. Standard Pull (Real Workload)

Generated on: {date}

## 📊 End-to-End Comparison Table

| Image Size (GB) | Streaming Mode | Container Provisioning (s) | JAX Init & Import (s) | Compile & Step 1 (s) | Total E2E TTFS (s) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for r in results:
        mode_str = "Enabled (GCFS)" if r["streaming"] else "Disabled (Standard)"
        if r["status"] == "SUCCESS":
            report += f"| **{r['size_gb']:.1f} GB** | {mode_str} | {r['t_pull']:.2f}s | {r['t_import']:.2f}s | {r['t_compile']:.2f}s | **{r['t_e2e']:.2f}s** | 🏆 Success |\n"
        else:
            report += f"| **{r['size_gb']:.1f} GB** | {mode_str} | N/A | N/A | N/A | N/A | ❌ {r['status']} |\n"
            
    with open("benchmarking_results_real.md", "w") as f:
        f.write(report)
    logging.info("Final report written to benchmarking_results_real.md")

if __name__ == "__main__":
    run_benchmark()

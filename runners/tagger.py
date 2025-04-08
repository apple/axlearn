import os
import glob
import subprocess
import argparse
from collections import OrderedDict
import logging
import shutil
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

INDEX_DIR=os.path.join("logs", "index")
LOGS_DIR="logs"
ARTIFACTS_DIR="artifacts"
PREFIX = 'rh'

os.makedirs(INDEX_DIR, exist_ok=True)

# def write_keys(key_to_search, f, tags=None):
#     for k,v in os.environ.items():
#         if key_to_search in k:
#             f.write(f"{k}\t{v}\n")
#             if tags is not None:
#                 tags[k] = v
# def write_all_keys(job_id, tags=None):
#     with open(os.path.join(INDEX_DIR, job_id), 'w') as f:
#         write_keys('AXLEARN', f, tags=tags)
#         write_keys('NEURON', f, tags=tags)
#         write_keys('XLA', f, tags=tags)
#         write_keys('CUSTOM_TAG', f, tags=tags)

def load_tags(job_id):
    tags = OrderedDict()
    index_file = get_index_path(job_id)
    with open(index_file, 'r') as f:
        line = f.readline()
        while line:
            line = line.rstrip()
            k, v = line.split('=', 1)
            tags[k] = v
            line = f.readline()
    return tags

def get_custom_tags(tags):
    return [k for k in tags.keys() if 'CUSTOM_TAG' in k]

def log_tags(tags, tag_types="all", prefix=None):
    logging.info(f"{prefix}tags:")
    for k, v in tags.items():
        if (tag_types == "axlearn" and "AXLEARN" in k) or (tag_types == "all"):
            logging.info(f"{prefix}  {k}: {v}")

def build_name_with_main_attrs(tags):
    # can be duplicated across jobs
    logging.debug(f"Loaded tags {tags}")
    
    job_name_parts = []
    job_name_parts.append(tags['AXLEARN_MODEL_NAME'].split('-')[-1])
    
    num_nodes=int(tags.get('SLURM_JOB_NUM_NODES', 1))
    if num_nodes > 1:
        job_name_parts.append(f"{num_nodes}n")
    
    num_layers=tags.get("AXLEARN_NUM_LAYERS", None)
    if num_layers:
        job_name_parts.append(f"{num_layers}l")
    
    tp_degree = tags.get("AXLEARN_TP_DEGREE", None)
    if tp_degree:
        job_name_parts.append(f"tp{tp_degree}")
    
    batch_size = tags.get("AXLEARN_TRAIN_BATCH_SIZE", None)
    if batch_size:
        bs = int(batch_size)//num_nodes
        job_name_parts.append(f"bsnode{bs}")

    remat=tags.get("AXLEARN_REMAT_LAYER", "true")
    if remat == "true":
        remat_str = "remat"
    elif remat == "nonmatmul":
        remat_str = "selremat"
    else:
        remat_str = "noremat"
    job_name_parts.append(remat_str)

    for k in get_custom_tags(tags):
        custom_tag_key = k.split('CUSTOM_TAG_')[1]
        v = tags[k]
        job_name_parts.append(f"{custom_tag_key}={v}")
    job_name='_'.join(job_name_parts)
    return job_name

def get_index_path(job_id):
    return os.path.join(INDEX_DIR, job_id)


def describe_job(job_id, tag_types="none", filter=None, log_versions=False):
    loaded_tags = load_tags(job_id)
    if filter:
        for k, v in filter.items():
            if loaded_tags.get(k) is None or loaded_tags[k].lower() != v.lower():
                return  # Skip this job if it doesn't match the filter

    log_file = f"{os.path.join(LOGS_DIR, job_id)}.out"
    job_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, job_id)
    # Find the first match of neuron log file using regex pattern
    neuron_log_pattern = os.path.join(job_ARTIFACTS_DIR, "neuron_dump", "pid*-program0", "log-neuron-cc.txt")
    neuron_log_matches = glob.glob(neuron_log_pattern)
    neuron_log_file = neuron_log_matches[0] if neuron_log_matches else None
    logging.info(f"{job_id}")
    # logging.info(f"  slurm_job_id: {job_id}")
    logging.info(f"  name: {build_name_with_main_attrs(loaded_tags)}")
    logging.info(f"  log: {log_file}")
    logging.info(f"  artifacts_dir: {job_ARTIFACTS_DIR}")
    if neuron_log_file:
        logging.info(f"  neuron_log: {neuron_log_file}")
    if tag_types != "none":
        log_tags(loaded_tags, tag_types=tag_types, prefix='  ')
    if log_versions:
        logging.info("  versions:")
        with open(f"{job_ARTIFACTS_DIR}/packages.txt", 'r') as f:
            line = f.readline().strip()
            while line:
                logging.info(f"    {line}")
                line = f.readline().strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process job mode.")

    parser.add_argument("--submit", "-s", action="store_true")
    parser.add_argument("--list", "-l", action="store_true", help="list n jobs")
    parser.add_argument("-n", type=int, default=10, help="Num recent jobs to list")
    parser.add_argument("-a", type=str, choices=["all", "axlearn", "none"], default="none", help="Log attributes for jobs listed")
    parser.add_argument("-v", action="store_true", help="Log package versions for jobs listed")
    parser.add_argument("-f", "--filter", action="store_true", help="filter with --keys")
    parser.add_argument("-k", "--keys", nargs='+', help="Keys to search for (can provide multiple)")
    
    parser.add_argument("--describe", "-d", action="store_true")
    parser.add_argument("-j", type=str, help="job id to describe")
    args = parser.parse_args()

    if args.submit:
        job_id=os.environ.get('SLURM_JOB_ID')
        shutil.copyfile(
            os.path.join(ARTIFACTS_DIR, job_id, 'env.txt'),
            os.path.join(INDEX_DIR, job_id)
        )
        tags = load_tags(job_id)
        name = build_name_with_main_attrs(tags)
        name = f"{PREFIX}_{name}"
        subprocess.run(
            f"scontrol update job {job_id} name={name}", shell=True
        )
        logging.info(f"Changed slurm job name to {name}")

    elif args.list:
        files = sorted(
            os.listdir(INDEX_DIR), 
            key=lambda x: os.path.getmtime(os.path.join(INDEX_DIR, x)), 
            reverse=True
        )
        search_keys_and_vals = {}
        if args.filter:
            assert args.keys, "Pass keys as k=v when trying to filter jobs"
            for kv in args.keys:
                k, v = kv.split('=')
                search_keys_and_vals[k] = v
        for job_id in files[:args.n]:
            describe_job(job_id, tag_types=args.a, log_versions=args.v, filter=search_keys_and_vals)

    elif args.describe is not None:
        assert args.j is not None, "Pass -j if --describe is set"
        describe_job(args.j, log_versions=args.v, tag_types="all")
        
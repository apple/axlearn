import os

import re
import glob
import subprocess
import argparse
from collections import OrderedDict
import logging
import shutil
from analyze import sort_live_range_report
import sys
from prettytable import PrettyTable
        
logging.basicConfig(
    level=logging.WARN,
    format='%(message)s'
)

INDEX_DIR=os.path.join("logs", "index")
LOGS_DIR="logs"
ARTIFACTS_DIR="artifacts"

# try:
#     PREFIX = re.match("(/fsx/)[^/]+(?=/*)", os.getcwd())[0].split('/')[2]
# except RuntimeError:
#     PREFIX = ''
# SLURM_SCRIPT_NAME="runs.slurm"
# if PREFIX:
#     SLURM_SCRIPT_NAME = PREFIX + "_" + SLURM_SCRIPT_NAME

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
    job_name_parts.append(tags.get('AXLEARN_MODEL_NAME', '-').split('-')[-1])
    
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

    repeated=str(tags.get("AXLEARN_REPEATED", '0'))
    job_name_parts.append(f"repeated{repeated}")

    for k in get_custom_tags(tags):
        custom_tag_key = k.split('CUSTOM_TAG_')[1]
        v = tags[k]
        job_name_parts.append(f"{custom_tag_key}={v}")
    job_name='_'.join(job_name_parts)
    return job_name

def get_index_path(job_id):
    return os.path.join(INDEX_DIR, job_id)

def get_log_file_path(job_id):
    log_file = f"{os.path.join(LOGS_DIR, job_id)}"
    logs = glob.glob(f"{log_file}_*")
    assert len(logs) == 1, "No log file found"
    return logs[0]

def describe_job(job_id, tag_types="none", filter=None, log_versions=False, t=None):
    loaded_tags = load_tags(job_id)
    if filter:
        for k, v in filter.items():
            if loaded_tags.get(k) is None or loaded_tags[k].lower() != v.lower():
                return  # Skip this job if it doesn't match the filter

    log_file = get_log_file_path(job_id)

    job_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, job_id)
    # Find the first match of neuron log file using regex pattern
    neuron_log_pattern = os.path.join(job_ARTIFACTS_DIR, "neuron_dump", "pid*-program*", "log-neuron-cc.txt")
    neuron_log_matches = sorted(glob.glob(neuron_log_pattern), key=os.path.getsize)
    row = []
    row.append(job_id)
    logging.info(f"{job_id}: ")
    # logging.info(f"  slurm_job_id: {job_id}")
    name = build_name_with_main_attrs(loaded_tags)
    logging.info(f"  name: {name}")
    row.append(name)
    logging.info(f"  log: {log_file}")
    row.append(log_file)
    logging.info(f"  artifacts_dir: {job_ARTIFACTS_DIR}")
    row.append(job_ARTIFACTS_DIR)
    if neuron_log_matches:
        neuron_log_matches=neuron_log_matches[-1]
        neuron_dir = os.path.dirname(neuron_log_matches)
        live_range_log = os.path.join(neuron_dir, 'LiveRangeReport_PostHloPart.txt')
        logging.info(f"  neuron_log: {neuron_log_matches}")
        logging.info(f"  live_range: {live_range_log}")
        row.append((neuron_log_matches, live_range_log))
    else:
        row.append("")
    # not printed in table
    if tag_types != "none":
        log_tags(loaded_tags, tag_types=tag_types, prefix='  ')
    if log_versions:
        logging.info("  versions:")
        with open(f"{job_ARTIFACTS_DIR}/packages.txt", 'r') as f:
            line = f.readline().strip()
            while line:
                logging.info(f"    {line}")
                line = f.readline().strip()
    if t:
        t.add_row(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process job mode.")

    parser.add_argument("--submit", "-s", action="store_true")

    parser.add_argument("--clean_job", nargs='+')
    
    parser.add_argument("--tail", "-t", action="store_true", help="tail log")
    parser.add_argument("--tail_compiler", "-tc", action="store_true", help="tail log")

    parser.add_argument("--list", "-l", action="store_true", help="list n jobs")
    parser.add_argument("-n", type=int, default=10, help="Num recent jobs to list")
    parser.add_argument("-a", type=str, choices=["all", "axlearn", "none"], default="none", help="Log attributes for jobs listed")
    parser.add_argument("-v", action="store_true", help="Log package versions for jobs listed")
    parser.add_argument("-f", "--filter", action="store_true", help="filter with --keys")
    parser.add_argument("-k", "--keys", help="Keys to search for k=v(can provide multiple as comma separated)")
    
    parser.add_argument("--describe", "-d", action="store_true")
    parser.add_argument("-j", type=str, help="job id to describe")

    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.submit:
        job_id=os.environ.get('SLURM_JOB_ID')
        shutil.copyfile(
            os.path.join(ARTIFACTS_DIR, job_id, 'env.txt'),
            os.path.join(INDEX_DIR, job_id)
        )
        tags = load_tags(job_id)
        job_name = os.environ.get('SLURM_JOB_NAME')
        if os.getenv('RENAME_JOB', 'false') == 'true':
            PREFIX = os.getenv('RENAME_JOB_PREFIX', '.')
            name = build_name_with_main_attrs(tags)
            name = f"{PREFIX}_{name}"
            subprocess.run(
                f"scontrol update job {job_id} name={name}", shell=True
            )
            logging.info(f"Changed slurm job name to {name}")
        full_artifacts_dir = os.path.realpath(ARTIFACTS_DIR)
        subprocess.run(f'cd {LOGS_DIR}; ln -s {os.path.basename(job_id)} {full_artifacts_dir}/{job_id}/log', shell=True)
    elif args.list:
        files = sorted(
            os.listdir(INDEX_DIR), 
            key=lambda x: os.path.getmtime(os.path.join(INDEX_DIR, x)), 
            reverse=True
        )
        search_keys_and_vals = {}
        if args.filter:
            assert args.keys, "Pass keys as k=v when trying to filter jobs"
            kvs = args.keys.split(',')
            for kv in kvs:
                k, v = kv.split('=')
                search_keys_and_vals[k] = v
        
        t = PrettyTable(['Jobid', 'Name', 'Log', 'Artifacts', 'Others'])
        for job_id in files[:args.n]:
            describe_job(job_id, tag_types=args.a, log_versions=args.v, filter=search_keys_and_vals, t=t)
        print(t)
    elif args.describe:
        assert args.j is not None, "Pass -j if --describe is set"
        describe_job(args.j, log_versions=args.v, tag_types="all")
    elif args.analyze:
        assert args.j is not None, "Pass -j if --analyze is set"
        job_id = int(args.j)
        for fpath in glob.glob(f"artifacts/{job_id}/neuron_dump/pid*-program*/LiveRangeReport_PostHloPart.txt"):
            print(fpath)
            sort_live_range_report(fpath)
    elif args.tail:
        assert args.j is not None, "Pass -j if --tail is set"
        log = get_log_file_path(args.j)
        subprocess.run(f"tail -f {log}", shell=True)
    elif args.tail_compiler:
        assert args.j is not None, "Pass -j if --tail is set"
        job_id = int(args.j)
        matches = sorted(glob.glob(f"artifacts/{job_id}/neuron_dump/*-program*/log-neuron-cc.txt"), key=os.path.getsize)
        assert len(matches) == 1, "No neuron log file found"
        if matches:
            log_file = matches[-1]
            print(log_file)
            subprocess.run(f"tail -f {log_file}", shell=True)
    elif args.clean_job:
        for j in args.clean_job:
            try:
                shutil.rmtree(f"{ARTIFACTS_DIR}/{j}")    
            except FileNotFoundError:
                pass
            try:
                log = get_log_file_path(j)
                os.remove(f"{log}")
            except FileNotFoundError:
                pass
            try:
                os.remove(f"{INDEX_DIR}/{j}")
            except FileNotFoundError:
                pass


import itertools
import numpy as np
import json
import copy
import random
import os
import subprocess
from absl import logging, flags, app
from multiprocessing import Queue, Manager
from gin import config
from pathos import multiprocessing
import traceback
import time
import sys
import pprint
from filelock import FileLock
which_gpus = [0, 1, 2, 3]
max_worker_num = len(which_gpus) * 2
#alg_type = sys.argv[1]
#environment = sys.argv[2]
#repeat = int(sys.argv[3])
COMMANDS = []
desired_changes= [
    "suite_karel_env.load.perception_noise_prob=0.0",
    "suite_karel_env.load.perception_noise_prob=0.25",
    "suite_karel_env.load.perception_noise_prob=0.5",
    "suite_karel_env.load.perception_noise_prob=0.75",
    "suite_karel_env.load.perception_noise_prob=1.0",
]
environments = [
    #"harvester_0.25",
    #"harvester_0.50",
    #"harvester_0.75",
    "topOff",
    "cleanHouse",
    "stairClimber",
    "randomMaze",
    "harvester",
    "fourCorners",
    
]
#for alg_type in ["global", "recurrent"]:
for alg_type in ["recurrent"]:
    for environment in environments:
        for desired_change in desired_changes:
            record_string = f"{alg_type}: {environment} {desired_change}"
            for repeat in range(5):
                root_dir = f"/home/jesse/karel_rl_logs/best_param_{alg_type}_{environment}/{environment}_{repeat}"
                command = f"xvfb-run -a python -m alf.bin.play --num_episodes 50 --norender --root_dir {root_dir}"
                config_file=f"{root_dir}/configured.gin"
                COMMANDS.append((command, record_string, config_file, desired_change))
                print(COMMANDS[-1])
#finalized_outputs = {}
#for alg_type in ["global", "recurrent"]:
#    #for environment in environments:
##    for desired_change in desired_changes:
##        for repeat in range(5):
#    for environment in environments:
#        for desired_change in desired_changes:
#            record_string = f"{alg_type}: {environment} {desired_change}"
#            finalized_outputs[record_string] = []
#            for repeat in range(5):
#                root_dir = f"/home/jesse/karel_rl_logs/best_param_{alg_type}_{environment}/{environment}_{repeat}"
#                COMMANDS.append(f"xvfb-run -a python -m alf.bin.play --num_episodes 50 --norender --root_dir {root_dir}")
#                config_file=f"{root_dir}/configured.gin"
#                lock = FileLock(config_file + ".lock")
#                with lock:
#                    with open(config_file, 'r') as f:
#                        lines = f.readlines()
#                    found_change = False
#                    old_lines = copy.deepcopy(lines)
#                    modified_lines = []
#                    for line in lines:
#                        if not found_change and desired_change.split("=")[0] in line:
#                            found_change = True
#                            old_value = line.split("=")[1].strip()
#                            line.replace(old_value, desired_change.split("=")[-1])
#                        modified_lines.append(line)
#                    if not found_change:
#                        modified_lines.append("\n" + desired_change + "\n")
#                    with open(config_file, 'w') as f:
#                        f.writelines(modified_lines)
#                    cmd_output = os.popen("CUDA_VISIBLE_DEVICES=%d " % which_gpus[0] + COMMANDS[-1])
#                    with open(config_file, 'w') as f:
#                        f.writelines(old_lines)
#                    for line in cmd_output:
#                        if "AverageReturn" in line:
#                            reward = float(line.strip().split(" ")[-1])
#                    finalized_outputs[record_string].append(reward)
#            finalized_outputs[record_string] = (np.mean(finalized_outputs[record_string]), np.std(finalized_outputs[record_string]))
#            print(finalized_outputs, finalized_outputs[record_string])
#with open('custom_eval_data.json', 'w') as fp:
#    data = json.dump(finalized_outputs, fp, sort_keys=True, indent=4)

def _init_device_queue(max_worker_num):
    m = Manager()
    device_queue = m.Queue()
    for i in range(max_worker_num):
        idx = i % len(which_gpus)
        gpu = which_gpus[idx]
        device_queue.put(gpu)
    return device_queue

def run():
    """Run trainings with all possible parameter combinations in
    the configured space.
    """
    finalized_outputs = {}
    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(max_worker_num)
    results = []
    for command, record_string, config_file, desired_change in COMMANDS:
        reward = process_pool.apply_async(
            func=_worker,
            args=[command, device_queue, config_file, desired_change],
            error_callback=lambda e: logging.error(e))
        results.append((record_string, reward))
    process_pool.close()
    process_pool.join()
    for record_string, reward in results:
        if record_string not in finalized_outputs:
            finalized_outputs[record_string] = []
        finalized_outputs[record_string].append(reward.get())
    for key, list_of_results in finalized_outputs.items():
        finalized_outputs[key] = f"{np.mean(list_of_results)} ({np.std(list_of_results)})"
    with open('../../custom_eval_data.json', 'w') as fp:
        json.dump(finalized_outputs, fp, sort_keys=True, indent=4)


def _worker(command, device_queue, config_file, desired_change):
    # sleep for random seconds to avoid crowded launching
    try:

        time.sleep(random.uniform(0, 3))

        device = device_queue.get()

        logging.set_verbosity(logging.INFO)

        logging.info("command %s" % command)

        lock = FileLock(config_file + ".lock")
        with lock:
            with open(config_file, 'r') as f:
                lines = f.readlines()
            found_change = False
            old_lines = copy.deepcopy(lines)
            modified_lines = []
            for line in lines:
                if not found_change and desired_change.split("=")[0] in line:
                    found_change = True
                    old_value = line.split("=")[1].strip()
                    line.replace(old_value, desired_change.split("=")[-1])
                modified_lines.append(line)
            if not found_change:
                modified_lines.append("\n" + desired_change + "\n")
            with open(config_file, 'w') as f:
                f.writelines(modified_lines)
            cmd_output = os.popen("CUDA_VISIBLE_DEVICES=%d " % which_gpus[0] + command).readlines()
            with open(config_file, 'w') as f:
                f.writelines(old_lines)
            for line in cmd_output:
                if "AverageReturn" in line:
                    reward = float(line.strip().split(" ")[-1])
        device_queue.put(device)
        return reward
    except Exception as e:
        logging.info(traceback.format_exc())
        with open(config_file, 'w') as f:
            f.writelines(old_lines)
        raise e
run()


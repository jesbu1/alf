import itertools
import random
import subprocess
import os
from absl import logging, flags, app
from multiprocessing import Queue, Manager
from pathos import multiprocessing
import traceback
import time
import sys
which_gpus = [0, 1, 2, 3]
max_worker_num = len(which_gpus) * 3

BASE_COMMAND = "python -m alf.bin.train "
COMMANDS = []
environments = [
    "harvester_0.25",
    "harvester_0.50",
    "harvester_0.75",
    'harvester_0.05',
    'harvester_0.1',
    "topOff_0.05",
    "topOff_0.1",
    "topOff_0.25",
    "topOff_0.50",
    "topOff_0.75",
    "harvester",
    "topOff",
    "cleanHouse",
    "fourCorners",
    "stairClimber",
    "randomMaze",
]
for alg_type in ["global", "recurrent"]:
#for alg_type in ["recurrent"]:
#for alg_type in ["global"]:
    for environment in environments:
        for repeat in range(5):
                COMMANDS.append(f"python -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_{alg_type}_{environment}.gin --root_dir ~/karel_rl_logs/best_param_{alg_type}_{environment}/{environment}_{repeat}")
                #COMMANDS.append(f"python -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_{alg_type}_{environment}.gin --root_dir ~/karel_rl_logs/best_param_transpose_default_ppo_param_{alg_type}_{environment}/{environment}_{repeat}")
                print(COMMANDS[-1])
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

    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(max_worker_num)

    for command in COMMANDS:
        process_pool.apply_async(
            func=_worker,
            args=[command, device_queue],
            error_callback=lambda e: logging.error(e))
    process_pool.close()
    process_pool.join()

def _worker(command, device_queue):
    # sleep for random seconds to avoid crowded launching
    try:
        time.sleep(random.uniform(0, 3))

        device = device_queue.get()

        logging.set_verbosity(logging.INFO)

        logging.info("command %s" % command)
        os.system("CUDA_VISIBLE_DEVICES=%d " % device + command)

        device_queue.put(device)
    except Exception as e:
        logging.info(traceback.format_exc())
        raise e
run()


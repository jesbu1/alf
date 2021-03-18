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
which_gpus = [0]
max_worker_num = len(which_gpus) * 2
COMMANDS = [
    #"python -m alf.bin.grid_search --search_config ~/alf/alf/examples/ppo_karel_grid_search_8x8.json --gin_file ~/alf/alf/examples/ppo_karel_recurrent_8x8.gin --root_dir ~/karel_rl_logs/grid_search_local_recurrent",
    #"python -m alf.bin.grid_search --search_config ~/alf/alf/examples/ppo_karel_grid_search_12x12.json --gin_file ~/alf/alf/examples/ppo_karel_recurrent_12x12.gin --root_dir ~/karel_rl_logs/grid_search_local_recurrent",
    #"python -m alf.bin.grid_search --search_config ~/alf/alf/examples/ppo_karel_grid_search_cleanHouse.json --gin_file ~/alf/alf/examples/ppo_karel_recurrent_clean_house.gin --root_dir ~/karel_rl_logs/grid_search_local_recurrent",
    "python -m alf.bin.grid_search --search_config ~/alf/alf/examples/ppo_karel_grid_search.json --gin_file ~/alf/alf/examples/ppo_karel_recurrent_12x12.gin --root_dir ~/karel_rl_logs/grid_search_placeSetter/recurrent",
    "python -m alf.bin.grid_search --search_config ~/alf/alf/examples/ppo_karel_grid_search.json --gin_file ~/alf/alf/examples/ppo_karel_global.gin --root_dir ~/karel_rl_logs/grid_search_placeSetter/global",
]
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


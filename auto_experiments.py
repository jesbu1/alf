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
max_worker_num = len(which_gpus) * 2
COMMANDS = [
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel.gin --root_dir ~/karel_rl_logs/0000r0+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=1",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel.gin --root_dir ~/karel_rl_logs/0000r1+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=2",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel.gin --root_dir ~/karel_rl_logs/0000r2+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=3",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel.gin --root_dir ~/karel_rl_logs/0000r3+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=4",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel.gin --root_dir ~/karel_rl_logs/0000r4+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=5",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_large_maze.gin --root_dir ~/karel_rl_logs/0000r0+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=1",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_large_maze.gin --root_dir ~/karel_rl_logs/0000r1+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=2",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_large_maze.gin --root_dir ~/karel_rl_logs/0000r2+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=3",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_large_maze.gin --root_dir ~/karel_rl_logs/0000r3+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=4",
    "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_large_maze.gin --root_dir ~/karel_rl_logs/0000r4+suite_karel_env.load.env_task='randomMaze'+local_maze_transfer=5",
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


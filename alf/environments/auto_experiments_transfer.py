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
which_gpus = [0, 1]
num_seeds = 5
max_worker_num = len(which_gpus) * 6
environments = ['chainSmoker',
                'cleanHouse',
                'fourCorners',
                'harvester',
                'randomMaze',
                'stairClim_sparse',
                'topOff',
                'shelfStocker',
                ]
extra_args = [
        "--gin_param 'suite_karel_env.load.width=12' --gin_param 'suite_karel_env.load.height=12' --gin_param 'suite_karel_env.load.max_episode_steps=100'",
        "--gin_param 'suite_karel_env.load.width=22' --gin_param 'suite_karel_env.load.height=14' --gin_param 'suite_karel_env.load.max_episode_steps=300'",
        "--gin_param 'suite_karel_env.load.width=12' --gin_param 'suite_karel_env.load.height=12' --gin_param 'suite_karel_env.load.max_episode_steps=100'",
        "--gin_param 'suite_karel_env.load.width=8' --gin_param 'suite_karel_env.load.height=8' --gin_param 'suite_karel_env.load.max_episode_steps=100'",
        "--gin_param 'suite_karel_env.load.width=8' --gin_param 'suite_karel_env.load.height=8' --gin_param 'suite_karel_env.load.max_episode_steps=100'",
        "--gin_param 'suite_karel_env.load.width=12' --gin_param 'suite_karel_env.load.height=12' --gin_param 'suite_karel_env.load.max_episode_steps=100'",
        "--gin_param 'suite_karel_env.load.width=12' --gin_param 'suite_karel_env.load.height=12' --gin_param 'suite_karel_env.load.max_episode_steps=100'",
        "--gin_param 'suite_karel_env.load.width=12' --gin_param 'suite_karel_env.load.height=12' --gin_param 'suite_karel_env.load.max_episode_steps=100'",
        ]
default_string = "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/"
COMMANDS = []
for i, environment_from in enumerate(environments):
    for j, environment_to in enumerate(environments):
        if environment_from == environment_to:
            continue
        for count in range(num_seeds):
            modified_string = f"ppo_karel_recurrent.gin --root_dir ~/karel_rl_logs/recurrent_transfer_exps/recurrent/0000r{count}+suite_karel_env.load.env_task={environment_to}+transfer_from_{environment_from} {extra_args[i]} --gin_param 'suite_karel_env.load.env_task=\"{environment_from}\"'"
            command = default_string + modified_string
            COMMANDS.append(command)
for i, environment_from in enumerate(environments):
    for j, environment_to in enumerate(environments):
        if environment_from == environment_to:
            continue
        for count in range(num_seeds):
            modified_string = f"ppo_karel_recurrent.gin --root_dir ~/karel_rl_logs/recurrent_transfer_exps/recurrent/0000r{count}+suite_karel_env.load.env_task={environment_to}+transfer_from_{environment_from} {extra_args[j]} --gin_param 'suite_karel_env.load.env_task=\"{environment_to}\"' --gin_param 'TrainerConfig.num_env_steps=2000000'"
            command = default_string + modified_string
            COMMANDS.append(command)
print(f"NUMBER OF EXPERIMENTS TO RUN: {len(COMMANDS)}")
    #"python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_stair.gin --root_dir ~/karel_rl_logs/stairClimber_sparse_transfer/recurrent/0000r0+suite_karel_env.load.env_task='\'stairClim_sparse'\'+local_stairClim_transfer=1",
    #"python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_stair.gin --root_dir ~/karel_rl_logs/stairClimber_sparse_transfer/recurrent/0000r1+suite_karel_env.load.env_task='\'stairClim_sparse'\'+local_stairClim_transfer=2",
    #"python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_stair.gin --root_dir ~/karel_rl_logs/stairClimber_sparse_transfer/recurrent/0000r2+suite_karel_env.load.env_task='\'stairClim_sparse'\'+local_stairClim_transfer=3",
    #"python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_stair.gin --root_dir ~/karel_rl_logs/stairClimber_sparse_transfer/recurrent/0000r3+suite_karel_env.load.env_task='\'stairClim_sparse'\'+local_stairClim_transfer=4",
    #"python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_stair.gin --root_dir ~/karel_rl_logs/stairClimber_sparse_transfer/recurrent/0000r4+suite_karel_env.load.env_task='\'stairClim_sparse'\'+local_stairClim_transfer=5",
 #   "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_maze.gin --root_dir ~/karel_rl_logs/randomMaze_transfer/recurrent/0000r0+suite_karel_env.load.env_task=\'randomMaze\'+local_maze_transfer",
 #   "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_maze.gin --root_dir ~/karel_rl_logs/randomMaze_transfer/recurrent/0000r1+suite_karel_env.load.env_task=\'randomMaze\'+local_maze_transfer",
 #   "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_maze.gin --root_dir ~/karel_rl_logs/randomMaze_transfer/recurrent/0000r2+suite_karel_env.load.env_task=\'randomMaze\'+local_maze_transfer",
 #   "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_maze.gin --root_dir ~/karel_rl_logs/randomMaze_transfer/recurrent/0000r3+suite_karel_env.load.env_task=\'randomMaze\'+local_maze_transfer",
 #   "python3 -m alf.bin.train --gin_file ~/alf/alf/examples/ppo_karel_recurrent_maze.gin --root_dir ~/karel_rl_logs/randomMaze_transfer/recurrent/0000r4+suite_karel_env.load.env_task=\'randomMaze\'+local_maze_transfer",
#
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

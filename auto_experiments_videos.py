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
max_worker_num = len(which_gpus) * 1 + 1
#tasks = [
#    #"cleanHouse", 
#    #"fourCorner", 
#    #"harvester", 
#    #"randomMaze", 
#    "vi_env_ass_sce_dea_cor.cfg", 
#    #"topOff"
#    ]
tasks = os.listdir("/data/jesse/vizdoom_rl_logs")
base_command = "python3 -m alf.bin.train --gin_file /data/jesse/alf/alf/examples/"
vis_command = "xvfb-run -a python3 -m alf.bin.play --num_episodes 5 --root_dir /data/jesse/vizdoom_rl_logs/"
COMMANDS = [
]

for task in tasks:
    COMMANDS.append(f'{vis_command}"{task}" --record_file /data/jesse/vizdoom_visualizations/{task}.mp4')
    #COMMANDS.append(f"{base_command}ppo_karel_global_{task}.gin --root_dir ~/karel_rl_logs/global_vids6/{task}")
#COMMANDS.append("git rm pytorch_a2c_ppo_acktr_gail/karel_env/generator.py")
#COMMANDS.append("git mv pytorch_a2c_ppo_acktr_gail/karel_env/temporary_fixed_generator.py pytorch_a2c_ppo_acktr_gail/karel_env/generator.py")
#for task in tasks:
#    for i in range(1):
#        COMMANDS.append(f"{vis_command}{task} --record_file ~/karel_rl_logs/global_vids6/{task}_{i}.mp4")
#for task in tasks:
#    for i in range(1):
#        COMMANDS.append(f"ffmpeg -i ~/karel_rl_logs/global_vids6/{task}_{i}.mp4 -vf 'scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' ~/karel_rl_logs/global_vids6/{task}_{i}.gif")
for command in COMMANDS:
    print(command)
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


import os
import sys
import errno
sys.path.insert(0, '.')
sys.path.insert(0, 'karel_env')
import argparse

import numpy as np
from PIL import Image

from karel_env import karel
from karel_env.dsl import get_DSL
from karel_env.dsl.dsl_parse import parse
from karel_env.generator import KarelStateGenerator
from pretrain.misc_utils import create_directory
from prl_gym.exec_env import ExecEnv1, ExecEnv2

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='pretrain/image_dir/')
#parser.add_argument('--use_simplified_dsl', action='store_true', help='use simplified DSL or not')
parser.add_argument('--width', default=8, type=int)
parser.add_argument('--height', default=8, type=int)
parser.add_argument('--wall_prob', default=0.5)
parser.add_argument('--task_definition', default='program')
args = parser.parse_args()

def create_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
create_directory(args.save_dir)


#for env_task in ['maze']:
for env_task in ['stairClimber', 'shelfStocker', 'placeSetter', 'topOff', 'chainSmoker', 'fourCorners']:
    print(env_task)

    dsl = get_DSL(dsl_type='prob', seed=123, environment='karel')
    s_gen = KarelStateGenerator(seed=123)
    _world = karel.Karel_world(make_error=False, env_task=env_task,
            task_definition=args.task_definition, reward_diff=True)

    w = args.width if env_task != 'maze' else 13
    h = args.height if env_task != 'maze' else 13
    wall_prob = args.wall_prob

    if env_task == 'program':
        s, _, _, _, _ = s_gen.generate_single_state(h, w, wall_prob)
    elif env_task == 'maze' or env_task == 'maze_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_find_marker(h, w, wall_prob)
    elif env_task == 'stairClimber' or env_task == 'stairClimber_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_stair_climber(h, w, wall_prob)
    elif env_task == 'placeSetter' or env_task == 'placeSetter_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_place_setter(h, w, wall_prob)
    elif env_task == 'shelfStocker' or env_task == 'shelfStocker_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_shelf_stocker(h, w, wall_prob)
    elif env_task == 'chainSmoker' or env_task == 'chainSmoker_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_chain_smoker(h, w, wall_prob)
    elif env_task == 'topOff' or env_task == 'topOff_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_chain_smoker(h, w, wall_prob)
    elif env_task == 'fourCorners' or env_task == 'fourCorners_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_four_corners(h, w, wall_prob)
    else:
        raise NotImplementedError('{} task not implemented yet'.format(env_task))


    args.seed = 123
    args.num_demo_per_program = 2
    args.min_demo_length = 1
    args.max_demo_length = 200
    args.reward_type = 'sparse'
    args.reward_validity = False
    args.max_program_len = 40
    args.env_name = 'karel'
    args.env_task = env_task
    args.task_definition = 'program' if 'maze' not in env_task else 'custom_reward'
    args.reward_diff = True
    args.execution_guided = False
    args.experiment = 'egps'
    args.task_file = os.path.join('tasks', env_task+"1.txt")
    args.final_reward_scale = False
    args.cover_all_branches_in_demos = False

    gt_program_seq = dsl.str2intseq(open(os.path.join('tasks', env_task+"1.txt"), "r").readlines()[0].strip())
    pred_program_seq = gt_program_seq
    if args.task_definition == 'program':
        env = ExecEnv1(args, gt_program_seq)
    else:
        env = ExecEnv2(args)
        env2_pred = env.execute_pred_program(gt_program_seq)

    #reward, exec_data = env.reward(pred_program_seq)
    if args.task_definition == 'program':
        init_img = env.gt_program['s_h'][0][0]
        final_img = env.gt_program['s_h'][0][env.gt_program['s_h_len'][0]-1]
    else:
        init_img = env._world.s_h[0]
        final_img = env2_pred['s_h'][0][env2_pred['s_h_len'][0]-1]

    img = _world.state2image(s=init_img, grid_size=100)
    img = img.astype('uint8')[:,:,0]
    img = Image.fromarray(img)

    demo_name = 'init_{}_h_{}_w_{}.png'.format(env_task, h, w)
    img.save(os.path.join(args.save_dir, demo_name), 'PNG')

    img = _world.state2image(s=final_img, grid_size=100)
    img = img.astype('uint8')[:,:,0]
    img = Image.fromarray(img)

    demo_name = 'final_{}_h_{}_w_{}.png'.format(env_task, h, w)
    img.save(os.path.join(args.save_dir, demo_name), 'PNG')


programs = {
        "gt_placeSetter": "DEF run m( WHILE c( frontIsClear c) w( IF c( not c( leftIsClear c) c) i( putMarker i) move w) m)",
#        "gt_shelfStocker": "DEF run m( WHILE c( frontIsClear c) w( pickMarker turnLeft move putMarker turnLeft turnLeft move turnLeft move IF c( frontIsClear c) i( move i) w) m)",
#        "VAEC_shelfStocker": "DEF run m( WHILE c( frontIsClear c) w( pickMarker move w) turnRight m)",
#        "gt_test4_program": "DEF run m( IF c( frontIsClear c) i( move i) turnLeft turnRight turnLeft move move move m)",
#        "VAE_test4_program": "DEF run m( turnLeft turnRight turnLeft move move move m)",
}

for env_task, env_task_program in programs.items():
    print(env_task)

    dsl = get_DSL(dsl_type='prob', seed=123, environment='karel')
    s_gen = KarelStateGenerator(seed=123)
    _world = karel.Karel_world(make_error=False, env_task='program',
            task_definition=args.task_definition, reward_diff=True)

    w = args.width if env_task != 'maze' else 13
    h = args.height if env_task != 'maze' else 13
    wall_prob = args.wall_prob

    if 'program' in env_task:
        s, _, _, _, _ = s_gen.generate_single_state(h, w, wall_prob)
    elif 'maze' in env_task or 'maze_sparse' in env_task:
        s, _, _, _, _ = s_gen.generate_single_state_find_marker(h, w, wall_prob)
    elif 'stairClimber' in env_task or 'stairClimber_sparse' in env_task:
        s, _, _, _, _ = s_gen.generate_single_state_stair_climber(h, w, wall_prob)
    elif 'placeSetter' in env_task or 'placeSetter_sparse' in env_task:
        s, _, _, _, _ = s_gen.generate_single_state_place_setter(h, w, wall_prob)
    elif 'shelfStocker' in env_task or 'shelfStocker_sparse' in env_task:
        s, _, _, _, _ = s_gen.generate_single_state_shelf_stocker(h, w, wall_prob)
    elif 'chainSmoker' in env_task or 'chainSmoker_sparse' in env_task:
        s, _, _, _, _ = s_gen.generate_single_state_chain_smoker(h, w, wall_prob)
    elif 'topOff' in env_task or 'topOff_sparse' in env_task:
        s, _, _, _, _ = s_gen.generate_single_state_chain_smoker(h, w, wall_prob)
    elif env_task == 'fourCorners' or env_task == 'fourCorners_sparse':
        s, _, _, _, _ = s_gen.generate_single_state_four_corners(h, w, wall_prob)
    else:
        raise NotImplementedError('{} task not implemented yet'.format(env_task))


    args.seed = 59
    args.num_demo_per_program = 2
    args.min_demo_length = 1
    args.max_demo_length = 200
    args.reward_type = 'sparse'
    args.reward_validity = False
    args.max_program_len = 40
    args.env_name = 'karel'
    args.env_task = env_task.split('_')[-1]
    args.task_definition = 'program' if 'maze' not in env_task else 'custom_reward'
    args.reward_diff = True
    args.execution_guided = False
    args.experiment = 'egps'
    args.wall_prob = 0.1 if 'placeSetter' not in env_task else 0.5
    args.final_reward_scale = False
    args.cover_all_branches_in_demos = False

    gt_program_seq = dsl.str2intseq(env_task_program)
    pred_program_seq = gt_program_seq
    env = ExecEnv1(args, gt_program_seq)
    #reward, exec_data = env.reward(pred_program_seq)
    init_img = env.gt_program['s_h'][0][0]
    final_img = env.gt_program['s_h'][0][env.gt_program['s_h_len'][0]-1]


    images_list = []
    for state in env.gt_program['s_h'][1][:env.gt_program['s_h_len'][1]]:
        img = _world.state2image(s=state, grid_size=100)
        img = img.astype('uint8').squeeze()
        img = Image.fromarray(img)
        images_list.append(img)

    images_list[0].save(os.path.join(args.save_dir, '{}_{}_{}.gif'.format(env_task, h, w)),
            save_all=True, append_images=images_list[1:], duration=1000, loop=0)

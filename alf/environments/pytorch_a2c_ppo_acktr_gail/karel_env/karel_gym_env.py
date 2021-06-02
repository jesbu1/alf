import os
import sys
import numpy as np

import gym
import gym.spaces as spaces
sys.path.insert(0, ".")
sys.path.insert(0, "karel_env")
from karel_env import generator
from karel_env.karel import Karel_world, action_table
import cv2

import argparse
import time
from PIL import Image


class KarelGymEnv(gym.Env):
    """Environment that will follow gym interface"""

    def __init__(self, config):
        #super(KarelGymEnv, self).__init__()
        self.config = config
        self.metadata = {'render.modes': ['rgb_array', 'state']}

        self.s_gen = generator.KarelStateGenerator(seed=config.seed)
        self._world = Karel_world(s=None, make_error=False, env_task=config.env_task,
                                   task_definition=config.task_definition, reward_diff=False,
                                   final_reward_scale=False,
                                   incorrect_marker_penalty=self.config.incorrect_marker_penalty,
                                   perception_noise_prob=self.config.perception_noise_prob)
        self._world.set_task_metadata(config.env_task, config.env_task_metadata)
        new_state, metadata = self._generate_state()
        self._world.set_new_state(new_state, metadata)

        if config.obv_type == 'local':
            self.observation_space = spaces.Box(low=0, high=1, shape=self._world.get_perception_vector().shape, dtype=np.float32)
            self.initial_obv = self._world.get_perception_vector()
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=self._world.s.shape, dtype=np.float32)
            self.initial_obv = self._world.s

        self.num_actions = len(action_table)
        self.action_space = spaces.Discrete(self.num_actions)

        self._elapsed_steps = 0
        self.state = self.initial_obv

    def step(self, action):
        self._elapsed_steps += 1

        # FIXME: remove this if check if no-op action not required
        #if action == self.num_actions - 1:
        if action == self.num_actions:
            made_error = False
            a_idx = np.argmax(action)
            loc = self._world.get_location()
            self._world.add_to_history(a_idx, loc, made_error)
        else:
            one_hot_action = np.zeros(self.num_actions, dtype=np.uint8)
            one_hot_action[action] = 1
            self._world.state_transition(one_hot_action)

        self.state = self._world.p_v_h[-1] if self.config.obv_type == 'local' else self._world.s
        reward = self._world.r_h[-1]
        done = self._world.d_h[-1]
        info = {}

        # FIXME: need to shift this code under envs.TimeLimitMask
        done, info = self._set_bad_transition(done, info)
        if self.config.delayed_reward:
            reward = reward if done else 0

        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self._elapsed_steps = 0
        new_state, metadata = self._generate_state()
        self._world.set_new_state(new_state, metadata)
        self.initial_obv = self._world.get_perception_vector() if self.config.obv_type == 'local' else self._world.s
        self.state = self.initial_obv
        return self.state

    def render(self, mode='state'):
        """render environment"""
        if mode == 'rgb_array':
            image = self._world.state2image(s=self._world.s).astype(np.float32)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.uint8)
            return rgb_image
        elif mode == 'state':
            return self._world.get_perception_vector() if self.config.obv_type == 'local' else self._world.s
        else:
            raise NotImplementedError('render mode not found')

    def _set_bad_transition(self, done, info):
        # FIXME: need to shift this code under envs.TimeLimitMask
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            info['bad_transition'] = done
            done = True
        return done, info

    def _generate_state(self):
        h = self.config.height
        w = self.config.width
        wall_prob = self.config.wall_prob
        env_task_metadata = self.config.env_task_metadata
        if self.config.env_task == 'program':
            s, _, _, _, metadata = self.s_gen.generate_single_state(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'cleanHouse' or self.config.env_task == 'cleanHouse_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_clean_house(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'harvester' or self.config.env_task == 'harvester_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_harvester(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'fourCorners' or self.config.env_task == 'fourCorners_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_four_corners(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'maze' or self.config.env_task == 'maze_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_find_marker(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'randomMaze' or self.config.env_task == 'randomMaze_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_random_maze(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'stairClimber' or self.config.env_task == 'stairClimber_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_stair_climber(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'placeSetter' or self.config.env_task == 'placeSetter_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_place_setter(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'shelfStocker' or self.config.env_task == 'shelfStocker_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_shelf_stocker(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'chainSmoker' or self.config.env_task == 'chainSmoker_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_chain_smoker(h, w, wall_prob, env_task_metadata)
        elif self.config.env_task == 'topOff' or self.config.env_task == 'topOff_sparse':
            s, _, _, _, metadata = self.s_gen.generate_single_state_chain_smoker(h, w, wall_prob, env_task_metadata, is_top_off=True)
        else:
            raise NotImplementedError('{} task not implemented yet'.format(self.config.env_task))

        return s, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')
    # karel environment specific arguments
    parser.add_argument(
        '--seed', type=int, default=123, help='random seed (default: 1)')
    parser.add_argument('--task_definition',
                        default='custom_reward',
                        choices=['program', 'custom_reward'])
    parser.add_argument('--env_task',
                        default='fourCorners')
    parser.add_argument('--max_episode_steps',
                        type=int,
                        default=100,
                        help='set done=True for environment after max_episode_steps (to reset environment)')
    parser.add_argument('--obv_type',
                        default='global',
                        choices=['local', 'global'])
    parser.add_argument('--wall_prob',
                        type=float,
                        default=0.25,
                        help='wall probability')
    parser.add_argument('--height',
                        type=int,
                        default=6,
                        help='height of karel maze')
    parser.add_argument('--width',
                        type=int,
                        default=6,
                        help='width of karel maze')
    parser.add_argument('--incorrect_marker_penalty',
                        type=bool,
                        default=True,
                        help='penalize incorrect markers')
    parser.add_argument('--delayed_reward',
                        type=bool,
                        default=True,
                        help='whether to delay reward')
    parser.add_argument('--env_task_metadata',
                        type=dict,
                        default={},
                        help='metadata dict for karel generator and karel environment')
    parser.add_argument('--perception_noise_prob',
                        type=float,
                        default=0.0,
                        help='noise probability in perception vector')

    args = parser.parse_args()

    env = KarelGymEnv(args)
    env._max_episode_steps = args.max_episode_steps
    img = env.render()

    images_list = []
    for i in range(args.max_episode_steps):
        action = np.random.randint(6)
        print(action)
        obs, reward, done, info = env.step(action)
        img = env.render(mode='rgb_array')
        state = env.render(mode='state')
        print(20*'*' + 'state {}'.format(i) + 20*'*')
        env._world.print_state(state)
        #img = img.astype('uint8').squeeze()
        img = Image.fromarray(img)
        images_list.append(img)
        time.sleep(1)

    images_list[0].save(os.path.join('./', 'test_{}_{}_{}.gif'.format(args.env_task, args.height, args.width)),
                        save_all=True, append_images=images_list[1:], duration=1000, loop=0)






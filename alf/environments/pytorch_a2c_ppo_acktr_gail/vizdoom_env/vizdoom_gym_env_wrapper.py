import os
import sys
import numpy as np

import gym
import gym.spaces as spaces
sys.path.insert(0, ".")
sys.path.insert(0, "karel_env")
from vizdoom_env import generator
from vizdoom_env.vizdoom_env_wrapper import Vizdoom_env
import cv2

import argparse
import time
from PIL import Image


class VizDoomGymEnv(gym.Env):
    """Environment that will follow gym interface"""

    def __init__(self, config):
        #super(KarelGymEnv, self).__init__()
        self.config = config
        self.metadata = {'render.modes': ['rgb_array', 'state', 'perception_vector']}

        self.s_gen = generator.DoomStateGenerator(seed=config.seed)
        #self._world = Karel_world(s=None, make_error=False, env_task=config.env_task,
        #                           task_definition=config.task_definition, reward_diff=False,
        #                           final_reward_scale=False,
        #                           incorrect_marker_penalty=self.config.incorrect_marker_penalty)
        self._world = Vizdoom_env(config=config.vizdoom_config_file, perception_type='simple', env_task=config.env_task)
        self._world.init_game()
        new_state, _ = self._generate_state()
        self._world.new_episode(new_state)

        if config.obv_type == 'local':
            self.observation_space = spaces.Box(low=0, high=1, shape=self._world.get_perception_vector().shape, dtype=np.float32)
            self.initial_obv = self._world.get_perception_vector()
        else:
            assert 0

        self.num_actions = len(self._world.get_action_list())+1
        self.action_space = spaces.Discrete(self.num_actions)

        self._elapsed_steps = 0
        self.state = self.initial_obv

    def step(self, action):
        self._elapsed_steps += 1
        if action == self.num_actions - 1:
            action_string = 'NONE'
        else:
            action_string = self._world.action_strings[action]
        self._world.state_transition(action_string)

        self.state = self._world.p_v_h[-1] if self.config.obv_type == 'local' else self._world.s_h[-1]
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
        self._world.new_episode(new_state)
        self.initial_obv = self._world.get_perception_vector() if self.config.obv_type == 'local' else self._world.s_h[-1]
        self.state = self.initial_obv
        return self.state

    def render(self, mode='state'):
        """render environment"""
        if mode == 'rgb_array':
            return self._world.s_h[-1]
        elif mode == 'state':
            return self._world.get_perception_vector() if self.config.obv_type == 'local' else self._world.s_h[-1]
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
        if self.config.env_task == 'survive':
            s = self.s_gen.generate_initial_state()
        elif self.config.env_task == 'preloaded':
            #s = self.s_gen.generate_initial_state()
            s = None
        else:
            raise NotImplementedError('{} task not implemented yet'.format(self.config.env_task))

        return s, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')
    # karel environment specific arguments
    parser.add_argument(
        '--seed', type=int, default=123, help='random seed (default: 1)')
    parser.add_argument('--env_task',
                        default='preloaded')
    parser.add_argument('--vizdoom_config_file',
                        default='vizdoom_env/asset/default.cfg')
    parser.add_argument('--max_episode_steps',
                        type=int,
                        default=100,
                        help='set done=True for environment after max_episode_steps (to reset environment)')
    parser.add_argument('--obv_type',
                        default='local',
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
    parser.add_argument('--delayed_reward',
                        type=bool,
                        default=False,
                        help='whether to delay reward')

    args = parser.parse_args()

    env = VizDoomGymEnv(args)
    env._max_episode_steps = args.max_episode_steps
    img = env.render()

    images_list = []
    np.random.seed(1)
    for i in range(args.max_episode_steps):
        action = np.random.randint(len(env._world.action_strings))
        obs, reward, done, info = env.step(action)
        state = env._world.game.get_state()
        labels = state.labels if state is not None else None
        print(action, env._world.action_strings[action], reward, done, labels)
        if done: break
        img = env.render(mode='rgb_array')
        state = env.render(mode='state')
        print(20*'*' + 'state {}'.format(i) + 20*'*')
        #env._world.print_state()
        #img = img.astype('uint8').squeeze()
        img = Image.fromarray(img)
        images_list.append(img)

    images_list[0].save(os.path.join('./', 'test_{}_{}_{}.gif'.format(args.env_task, args.height, args.width)),
                        save_all=True, append_images=images_list[1:], duration=1000, loop=0)






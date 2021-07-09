import functools
import random
from alf.environments.utils import UnwrappedEnvChecker
from alf.environments import suite_gym, alf_wrappers, process_environment
_unwrapped_env_checker_ = UnwrappedEnvChecker()
#from alf.environments.pytorch_a2c_ppo_acktr_gail.karel_env.karel_gym_env import KarelGymEnv
from alf.environments.karel_env.karel_gym_env import KarelGymEnv
from spirl.models.prl_bc_mdl import BCMdl
import gin
import numpy as np
import gym
from gym.spaces import Box
import torch
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class KarelEnvWrapper(gym.Wrapper):
    def __init__(self, env=None, op=[2, 0, 1], model=None):
        """
        Transpose observation space for images
        """
        gym.Wrapper.__init__(self, env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 3:
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0], [
                    obs_shape[self.op[0]], obs_shape[self.op[1]],
                    obs_shape[self.op[2]]
                ],
                dtype=self.observation_space.dtype)
        self.model = model
        self.action_space = Box(low=np.array([-2] * 10), high = np.array([2] * 10))

    def step(self, action):
        #action_plan = self.generate_action_plan(np.expand_dims(action, 0))
        #done = False
        #while len(action_plan) > 0 and not done:
        #    env_action = action_plan.pop(0)
        #    ob, reward, done, info = self.env.step(env_action)
        #    accumulated_reward += reward
        #return self.observation(ob.astype(np.float32)), float(reward), done, {}
        action = action.astype(np.int)
        accumulated_reward = 0
        for i, a_t in enumerate(action):
            if i == 0 and a_t == 5:
                a_t = 0
            if a_t != 5: # skip no-op actions 
                ob, reward, done, _ = self.env.step(a_t)
                accumulated_reward += reward
                if done:
                    break
            else:
                break
        return self.observation(ob.astype(np.float32)), np.float(accumulated_reward), done, {}

    def generate_action_plan(self, z):
        if isinstance(z, np.ndarray) and np.all(z == 0):
            return [0]
        with torch.no_grad():
            z = torch.tensor(z)
            print(z.shape)
            action_plan = self.model.decode(z, z, self.model.n_rollout_steps)[0]
            action_plan = action_plan.cpu().detach().tolist()
        return action_plan

    def reset(self):
        ob = self.observation(np.array(self.env.reset(), dtype=np.float32))
        return ob

    def observation(self, ob):
        if len(self.observation_space.shape) == 3:
            return np.transpose(ob, (self.op[0], self.op[1], self.op[2]))
        return ob


@gin.configurable
def load(env_name,
         model,
         env_id=None,
         discount=1.0,
         max_episode_steps=100,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         task_definition='custom_reward',
         width=12,
         height=12,
         env_task='fourCorners',
         obv_type='local',
         wall_prob=0.25,
         incorrect_marker_penalty=True,
         delayed_reward=True,
         wrap_with_process=False):
    """Loads the selected environment and wraps it with the specified wrappers.
    Note that by default a ``TimeLimit`` wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.
    Args:
        env_name: Ignored, but required for create_environment in utils.py
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no ``timestep_limit`` set in the environment's spec.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.
    Returns:
        An AlfEnvironment instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)
    #parser.add_argument(
    #    '--seed', type=int, default=1, help='random seed (default: 1)')
    #parser.add_argument('--task_definition',
    #                    default='custom_reward',
    #                    choices=['program', 'custom_reward'])
    #parser.add_argument('--env_task',
    #                    default='fourCorners',
    #                    choices=['fourCorners', 'fourCorners_sparse', 'maze', 'maze_sparse', 'randomMaze',
    #                             'randomMaze_sparse'])
    #parser.add_argument('--max_episode_steps',
    #                    type=int,
    #                    default=100,
    #                    help='set done=True for environment after max_episode_steps (to reset environment)')
    #parser.add_argument('--obv_type',
    #                    default='global',
    #                    choices=['local', 'global'])
    #parser.add_argument('--wall_prob',
    #                    type=float,
    #                    default=0.25,
    #                    help='wall probability')
    #parser.add_argument('--height',
    #                    type=int,
    #                    default=6,
    #                    help='height of karel maze')
    #parser.add_argument('--width',
    #                    type=int,
    #                    default=6,
    #                    help='width of karel maze')



    def env_ctor(env_id=None):
        return suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers,
            normalize_action=False,
            clip_action=False,)
    args = dict(task_definition=task_definition,
                  env_task=env_task,
                  max_episode_steps=max_episode_steps,
                  obv_type=obv_type,
                  wall_prob=wall_prob,
                  height=height,
                  width=width,
                  incorrect_marker_penalty=incorrect_marker_penalty,
                  delayed_reward=delayed_reward,
                  seed=random.randint(0, 100000000))
    config = AttrDict()
    config.update(args) 
    env = KarelGymEnv(config)
    env._max_episode_steps = config.max_episode_steps
    env = KarelEnvWrapper(env, model=model)
    print(f'ENVIRONMENT: {env_task}')
    #env.reset()
    #env = ActionScalingWrapper(env)

    if wrap_with_process:
        process_env = process_environment.ProcessEnvironment(
            functools.partial(env_ctor))
        process_env.start()
        torch_env = alf_wrappers.AlfEnvironmentBaseWrapper(process_env)
    else:
        torch_env = env_ctor(env_id=env_id)

    return torch_env
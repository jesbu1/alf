import functools
import random
from alf.environments.utils import UnwrappedEnvChecker
from alf.environments import suite_gym, alf_wrappers, process_environment
_unwrapped_env_checker_ = UnwrappedEnvChecker()
#from alf.environments.pytorch_a2c_ppo_acktr_gail.karel_env.karel_gym_env import KarelGymEnv
from alf.environments.vizdoom_env.vizdoom_gym_env_wrapper import VizDoomGymEnv
from spirl.models.prl_bc_mdl import BCMdl
import gin
import numpy as np
import gym
from gym.spaces import Box
import torch
import cv2 
import collections
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class StackFrames(gym.ObservationWrapper):
  #init the new obs space (gym.spaces.Box) low & high bounds as repeat of n_steps. These should have been defined for vizdooom
  
  #Create a return a stack of observations
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box( env.observation_space.low.repeat(repeat, axis=0),
                              env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)
    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        return  np.array(self.stack).reshape(self.observation_space.low.shape)
    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

class VizDoomEnvWrapper(gym.Wrapper):
    def __init__(self, env=None, shape=[64, 48, 1], model=None):
        """
        Transpose observation space for images
        """
        gym.Wrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        #self.shape = (shape[2], shape[0], shape[1])
        self.shape = (shape[0], shape[1], shape[2])
        if len(obs_shape) == 3:
            self.observation_space = Box(low=0.0, high=1.0,
                                        shape=self.shape, dtype=np.float32)
        self.accumulated_reward = 0
        self.action_space = Box(low=np.array([-2] * 10), high = np.array([2] * 10))

    def step(self, action):
        action = action.astype(np.int)
        accumulated_reward = 0
        check_out_of_bounds = lambda x: x >= self.env.action_space.n
        for i, a_t in enumerate(action):
            
            if i == 0 and check_out_of_bounds(a_t):
                a_t = 0
            if not check_out_of_bounds(a_t): # skip no-op actions 
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
        self.accumulated_reward = 0
        return ob

    def observation(self, obs):
        if len(self.observation_space.shape) == 3:
            #set observation space to new shape using gym.spaces.Box (0 to 1.0)
            new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

            resized_screen = cv2.resize(new_frame, self.shape[0:2],
            #resized_screen = cv2.resize(new_frame, self.shape[1:],
                                        interpolation=cv2.INTER_AREA)
            new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
            new_obs = new_obs / 255.0
        else:
            new_obs = obs
        return new_obs

@gin.configurable
def load(env_name,
         model,
         env_id=None,
         discount=1.0,
         max_episode_steps=100,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         task_definition='custom_reward',
         env_task='survive',
         obv_type='local',
         vizdoom_config_file='vizdoom_env/asset/default.cfg',
         delayed_reward=False,
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
                  vizdoom_config_file=vizdoom_config_file,
                  obv_type=obv_type,
                  delayed_reward=delayed_reward,
                  seed=random.randint(0, 100000000))
    config = AttrDict()
    config.update(args) 
    env = VizDoomGymEnv(config)
    env._max_episode_steps = config.max_episode_steps
    env = VizDoomEnvWrapper(env)
    #if obv_type == 'global':
    #    env = StackFrames(env, 4)
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

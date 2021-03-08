# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PPO algorithm."""

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.networks.value_networks import ValueNetwork
from alf.networks.actor_distribution_networks import ActorDistributionNetwork
import numpy as np
from alf.algorithms.config import TrainerConfig
from collections import namedtuple
import gin
import torch

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm, ActorCriticInfo, ActorCriticState
from alf.algorithms.ppo_loss import PPOLoss
from alf.data_structures import AlgStep, Experience, TimeStep
from alf.utils import common, dist_utils, value_ops

from spirl.models.prl_bc_mdl import BCMdl
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

PPOInfo = namedtuple("PPOInfo",
                     ["action_distribution", "returns", "advantages"])


@gin.configurable
class PPOAlgorithm(ActorCriticAlgorithm):
    """PPO Algorithm.
    Implement the simplified surrogate loss in equation (9) of "Proximal
    Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347

    It works with ``ppo_loss.PPOLoss``. It should have same behavior as
    `baselines.ppo2`.
    """
    def __init__(self,
                 observation_spec,
                 action_spec,
                 actor_network_ctor=ActorDistributionNetwork,
                 value_network_ctor=ValueNetwork,
                 env=None,
                 config: TrainerConfig = None,
                 loss=None,
                 loss_class=ActorCriticLoss,
                 optimizer=None,
                 debug_summaries=False,
                 name="ActorCriticAlgorithm",
                 ll_model_path=None,
                 hierarchical=False,
                 n_rollout_steps=20):
        super().__init__(observation_spec, action_spec, actor_network_ctor=actor_network_ctor, value_network_ctor=value_network_ctor, env=env, config=config, loss=loss, loss_class=loss_class, optimizer=optimizer, debug_summaries=debug_summaries, name=name)
        if hierarchical:
            self._action_plan = []
            self.n_rollout_steps=n_rollout_steps
            self._ll_policy = self._load_model(ll_model_path)
        self._hierarchical = hierarchical

    def is_on_policy(self):
        return False

    def preprocess_experience(self, exp: Experience):
        """Compute advantages and put it into exp.rollout_info."""
        advantages = value_ops.generalized_advantage_estimation(
            rewards=exp.reward,
            values=exp.rollout_info.value,
            step_types=exp.step_type,
            discounts=exp.discount * self._loss._gamma,
            td_lambda=self._loss._lambda,
            time_major=False)
        advantages = torch.cat([
            advantages,
            torch.zeros(*advantages.shape[:-1], 1, dtype=advantages.dtype)
        ],
                               dim=-1)
        returns = exp.rollout_info.value + advantages
        return exp._replace(
            rollout_info=PPOInfo(exp.rollout_info.action_distribution, returns,
                                 advantages))

    def generate_action_plan(self, z):
        with torch.no_grad():
            z = torch.from_numpy(z)
            action_plan = self.model.decode(z, z, self.model.n_rollout_steps)[0]
            action_plan = action_plan.cpu().detach().tolist()
        return action_plan

    def rollout_step(self, time_step: TimeStep, state: ActorCriticState):
        """Rollout for one step."""
        #if time_step.is_first():
        #    self.action_dist, self.actor_state, self.action = None, None, None
        #    self.value, self.value_state = None, None
        value, value_state = self._value_network(
            time_step.observation, state=state.value)

        # We detach exp.observation here so that in the case that exp.observation
        # is calculated by some other trainable module, the training of that
        # module will not be affected by the gradient back-propagated from the
        # actor. However, the gradient from critic will still affect the training
        # of that module.
        action_distribution, actor_state = self._actor_network(
            common.detach(time_step.observation), state=state.actor)

        action = dist_utils.sample_action_distribution(action_distribution)
        if self._hierarchical:
            ll_action, use_new_action = self._gen_ll_action(action)
            if use_new_action or self.action_dist is None:
                self.action_dist, self.actor_state, self.action = action_distribution, actor_state, action
                self.value, self.value_state = value, value_state

        return AlgStep(
            output=action if not self._hierarchical else ll_action,
            state=ActorCriticState(actor=actor_state if not self._hierarchical else self.actor_state, value=value_state if not self._hierarchical else self.value_state),
            info=ActorCriticInfo(
                value=value if not self._hierarchical else self.value, action_distribution=action_distribution if not self._hierarchical else self.action_dist))

    def predict_step(self, time_step: TimeStep, state: ActorCriticState,
                     epsilon_greedy):
        """Predict for one step."""
        #if time_step.is_first():
        #    self.action_dist, self.actor_state, self.action = None, None, None
        action_dist, actor_state = self._actor_network(
            time_step.observation, state=state.actor)

        action = dist_utils.epsilon_greedy_sample(action_dist, epsilon_greedy)
        if self._hierarchical:
            ll_action, use_new_action = self._gen_ll_action(action)
            if use_new_action or self.action_dist is None:
                self.action_dist, self.actor_state, self.action = action_dist, actor_state, action

        return AlgStep(
            output=action,
            state=ActorCriticState(actor=actor_state if not self._hierarchical else self.actor_state),
            info=ActorCriticInfo(action_distribution=action_dist if not self._hierarchical else self.action_dist))

    def _gen_ll_action(self, action):
        use_new_action = False
        if len(self._action_plan) == 0:
            self._action_plan = self.generate_action_plan(np.expand_dims(action, 0))
            use_new_action = True
        action = self._action_plan.pop(0)
        return action, use_new_action
                    
    def _load_model(self, model_path):
        ll_model_params = AttrDict(
            state_dim=5,
            action_dim=5,
            kl_div_weight=1.0,
            batch_size=128,
            nz_enc=128,
            nz_mid=128,
            input_res=16,
            #n_processing_layers=5,
            nz_vae=5,
            n_rollout_steps=self.n_rollout_steps,
            device='cpu'
        )
        model = BCMdl(ll_model_params)
        model.load_state_dict(torch.load(model_path))
        model.cpu()
        model.eval()
        return model
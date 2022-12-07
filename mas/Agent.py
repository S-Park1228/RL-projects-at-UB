from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam



class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = 64, non_linear = nn.ReLU()):
        super(Network, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain = gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        
        return self.net(x)

class Agent:
    
    """
    Each "Agent" is managed by the MADDPG.
    It directly manages its own networks and target networks.
    """

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.actor = Network(obs_dim, act_dim)
        self.critic = Network(global_obs_dim, 1) # The critic network receives all the observations and actions.
                                                 # ex. 3 agents case -> Inputs for critic are (obs1, obs2, obs3, act1, act2, act3).
        self.actor_optimizer = Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    @staticmethod
    def gumbel_softmax(logits, tau = 1.0, eps = 1e-20):
        # PyTorch built-in function: torch.nn.functional.gumbel_softmax, which may be removed in the future
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        
        return F.softmax(logits / tau, dim = -1)

    def action(self, obs, model_out = False):
        # Actions are derived by the following two cases:
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard = True)
        if model_out: # actions for updating the actor-network, where inputs (obs) are sampled from replay buffer with size:
                      # torch.Size([batch_size, state_dim])
            
            return action, logits
        
        # when an agent interacts with the environment..
        return action

    def target_action(self, obs):
        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard = True)
        
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward(retain_graph = True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
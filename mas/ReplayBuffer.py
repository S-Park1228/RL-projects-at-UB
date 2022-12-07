import numpy as np
import torch


class ReplayBuffer:
    
    """
    Replay Buffer for each agent
    A dictionary for replay buffers is created in a given multiagent reinforcement learning algorithm.
    NOTE that the experience is a dict with agent name as its key.
    Please refer to the main.py.
    """

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype = bool)
        self.capacity = capacity

        self._index = 0
        self._size = 0

        self.device = device

    def push(self, obs, action, reward, next_obs, done):
        
        """
        Push an experience into the memory.
        experience: observations, action, reward, next observations, done
        """
        
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        # Increase the counter by 1 for the next "push".
        # If the counter reaches the capacity of this replay buffer,
        # fill up the replay buffer from the beginning.
        self._index = (self._index + 1) % self.capacity
        
        # Update the length of the replay buffer.
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # Retrieve data.
        # Note that the data stored have ndarray datatype.
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # NOTE that `obs`, `action` and `next_obs` will be passed to networks (nn.Module).
        # Thus, the first dimension should be `batch_size`!
        obs = torch.from_numpy(obs).float().to(self.device)    # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)    # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(reward).float().to(self.device)    # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)    # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)    # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity    = capacity
        self.obs_buf     = np.zeros((capacity, obs_dim),     dtype=np.float32)
        self.act_buf     = np.zeros((capacity, action_dim),  dtype=np.float32)
        self.rew_buf     = np.zeros((capacity, 1),           dtype=np.float32)
        self.next_obs_buf= np.zeros((capacity, obs_dim),     dtype=np.float32)
        self.done_buf    = np.zeros((capacity, 1),           dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, o, a, r, o2, d):
        self.obs_buf[self.ptr]      = o
        self.act_buf[self.ptr]      = a
        self.rew_buf[self.ptr]      = r
        self.next_obs_buf[self.ptr] = o2
        self.done_buf[self.ptr]     = d
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return dict(
            obs      = torch.FloatTensor(self.obs_buf[idx]),
            action   = torch.FloatTensor(self.act_buf[idx]),
            reward   = torch.FloatTensor(self.rew_buf[idx]),
            next_obs = torch.FloatTensor(self.next_obs_buf[idx]),
            done     = torch.FloatTensor(self.done_buf[idx])
        )


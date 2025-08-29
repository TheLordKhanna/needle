
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean    = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)

    def forward(self, obs):
        x       = self.net(obs)
        mean    = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mean, std)
        z       = dist.rsample()
        action  = torch.tanh(z)
        logp    = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        logp    = logp.sum(-1, keepdim=True)
        return action, logp, torch.tanh(mean)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


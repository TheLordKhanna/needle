
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#initialise weights. we are using xavier uniform initialisation whihc is a common method
#initialise bias as zero as is convention. 
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)


#actor policy initialised. 2 hidden layers, 256 units, ReLU actiation 
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        #important- output is the mean and log standard deviation 
        self.mean    = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)



    #forward prop for actor 
    def forward(self, obs):

       
        x       = self.net(obs)

        #gte mean and log std
        mean    = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)

        #take exponential of log std to make it positive
        std     = log_std.exp()

        #creat a gauss/normal distribution from the mean and std
        dist    = torch.distributions.Normal(mean, std)

        #now sample a value from the disribution with reparametrisation and make this a 
        #differentiable solution
        z       = dist.rsample()

        #squash action 
        action  = torch.tanh(z)

        #obtain log probability, used for back-prop. refer to agent.py
        logp    = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        logp    = logp.sum(-1, keepdim=True)
        return action, logp, torch.tanh(mean)


#critic forward prop
class Critic(nn.Module):

    #critic architecture same as actor, but use two critics 
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
        
    #given state and action, get the q values from critic 1 and 2
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)



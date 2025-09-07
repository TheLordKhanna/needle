
#please refer to the explaination given in the dissertation for a detailed breakdown of each operation in this file
#forward prop is held in network.py

import numpy as np
import torch
import torch.nn.functional as F
from network import Actor, Critic


class SACAgent:

    #define the parameters, device (use gpu if it is there)
    def __init__(self, obs_dim, act_dim, a_low, a_high,
                 device="cpu",
                 gamma=0.995, tau=0.005, lr=3e-4, clip_grad=5.0):

        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau

        #clip gradients
        self.clip_grad = clip_grad


        #initialise actor, critic and critic target and send to device
        self.actor  = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_tgt = Critic(obs_dim, act_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())


        #define optimisers for actor, critic. We use Adam as a default, but other optimisers such as RMSprop might be useful
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        #log alpha is the temperature of entropy, essentially a entropy scaling factor
        #log is used to ensure positive values
        self.log_alpha  = torch.tensor(np.log(0.2), requires_grad=True,
                                       device=self.device)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=lr)

        #target entropy is -3
        self.target_entropy = -float(act_dim)


        #scale and bias to convert the action 'u' from [-1,1] to the env action range
        self.a_low  = torch.tensor(a_low,  device=self.device)
        self.a_high = torch.tensor(a_high, device=self.device)
        self.scale  = (self.a_high - self.a_low) / 2.0
        self.bias   = (self.a_high + self.a_low) / 2.0
        self.alpha  = self.log_alpha.exp()


    def _to_env(self, a_raw):
        """[-1,1]  → env range"""
        return a_raw * self.scale + self.bias


    @torch.no_grad()
    def select_action(self, obs, eval_mode=False):
        obs = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        a_raw, _, mean = self.actor(obs)
        a_raw = mean if eval_mode else a_raw
        return self._to_env(a_raw).cpu().numpy()[0]


    #update function, which will use the replace buffer of size 256 (256 steps from previous memory)
    #and updates the actor, cirtic and alpha policies
    def update(self, buf, batch_size=256):

        #extract the batch from the buffer- for each step, current state, action, reward, new state and termination condition
        batch = buf.sample(batch_size)
        o  = batch["obs"].to(self.device)
        a  = batch["action"].to(self.device)
        r  = batch["reward"].to(self.device)
        o2 = batch["next_obs"].to(self.device)
        d  = batch["done"].to(self.device)


        #critic backprop
        with torch.no_grad():
            a2_raw, logp2, _ = self.actor(o2)
            a2 = self._to_env(a2_raw)

            #obtain q1 and q2 critic values from the slow updating target critic applied to the new observation and action
            q1_t, q2_t = self.critic_tgt(o2, a2)

            #obtain the minimum q value - conservative Q learning. subtract the Q by the log prob scaled by alpha. 
            #log prob is for the squashed action u, measures how typical or expected (low surprisal) the sampled action is for the current state
            #it was defined in network.py during forward propogration
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * logp2

            #online critic weights updated with this bellman equation. 
            backup  = r + (1 - d) * self.gamma * min_q_t


        #get current q values from the online critic for the current observation and action taken
        q1, q2 = self.critic(o, a)

        #critic loss, using MSE, then backprop, clip gradients and complete step
        loss_c = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.critic_opt.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_opt.step()


        # actor backprop

        #sample new action from the current actor policy given state 'o', and then get the log prob 
        a_new_raw, logp_new, _ = self.actor(o)

        #scale to env
        a_new = self._to_env(a_new_raw)

        #get the q values from online critics for current state and new action
        q1_pi, q2_pi = self.critic(o, a_new)


        min_q_pi = torch.min(q1_pi, q2_pi)

        #actor loss. ideally maximise q and entropy. the actor tries to minimise loss by choosing actions
        #that maximise q (actions preferred by critics) and max entropy (high log prob) to explore the env 
        loss_a = (self.alpha * logp_new - min_q_pi).mean()
        self.actor_opt.zero_grad()
        loss_a.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_opt.step()


        # alpha backprop
        loss_alpha = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.critic_tgt.parameters()):
                tp.mul_(1 - self.tau).add_(p, alpha=self.tau)




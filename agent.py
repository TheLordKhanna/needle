
import numpy as np
import torch
import torch.nn.functional as F
from network import Actor, Critic


class SACAgent:
    def __init__(self, obs_dim, act_dim, a_low, a_high,
                 device="cpu",
                 gamma=0.995, tau=0.005, lr=3e-4, clip_grad=5.0):

        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        self.clip_grad = clip_grad

        self.actor  = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.critic_tgt = Critic(obs_dim, act_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha  = torch.tensor(np.log(0.2), requires_grad=True,
                                       device=self.device)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)

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

    def update(self, buf, batch_size=256):
        batch = buf.sample(batch_size)
        o  = batch["obs"].to(self.device)
        a  = batch["action"].to(self.device)
        r  = batch["reward"].to(self.device)
        o2 = batch["next_obs"].to(self.device)
        d  = batch["done"].to(self.device)

        with torch.no_grad():
            a2_raw, logp2, _ = self.actor(o2)
            a2 = self._to_env(a2_raw)
            q1_t, q2_t = self.critic_tgt(o2, a2)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            backup  = r + (1 - d) * self.gamma * min_q_t

        q1, q2 = self.critic(o, a)
        loss_c = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.critic_opt.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_opt.step()

        a_new_raw, logp_new, _ = self.actor(o)
        a_new = self._to_env(a_new_raw)
        q1_pi, q2_pi = self.critic(o, a_new)
        min_q_pi = torch.min(q1_pi, q2_pi)
        loss_a = (self.alpha * logp_new - min_q_pi).mean()
        self.actor_opt.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_opt.step()

        loss_alpha = -(self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(),
                             self.critic_tgt.parameters()):
                tp.mul_(1 - self.tau).add_(p, alpha=self.tau)


import torch
from needle_steering_env import NeedleSteeringEnv
from agent import SACAgent
from buffer import ReplayBuffer


env = NeedleSteeringEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
agent   = SACAgent(obs_dim, act_dim,
                   env.action_space.low,
                   env.action_space.high)

buffer  = ReplayBuffer(200_000, obs_dim, act_dim)
updates_per_step = 1
max_eps   = 2000
for ep in range(1, max_eps + 1):
    obs, _ = env.reset()
    ep_r = 0.0
    for _ in range(env.max_steps):
        act = agent.select_action(obs)
        obs2, r, done, _, _ = env.step(act)
        buffer.add(obs, act, r, obs2, float(done))
        obs, ep_r = obs2, ep_r + r

        if buffer.size > 2_000:
            for _ in range(updates_per_step):
                agent.update(buffer)

        if done:
            break

    print(f"Episode {ep:5d}  |  Return = {ep_r:7.2f}  |  Buffer = {buffer.size}")

    if ep % 500 == 0:
        torch.save(agent.actor.state_dict(),  f"sac_actor_ep{ep}.pth")
        torch.save(agent.critic.state_dict(), f"sac_critic_ep{ep}.pth")

torch.save(agent.actor.state_dict(),  "sac_actor.pth")
torch.save(agent.critic.state_dict(), "sac_critic.pth")
env.close()

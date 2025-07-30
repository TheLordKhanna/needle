# test_sac.py  ──────────────────────────────────────────────────────────────
#
# Roll out a single trajectory with a trained SAC policy in the
# fixed-curvature NeedleSteeringEnv.
#
# Requirements:
#   • needle_steering_env.py   (updated version with steer_flag)
#   • agent.py                 (your SAC implementation)
#   • sac_actor.pth            (weights trained with 3-D action space)
#   • PyTorch, Gymnasium, NumPy, Matplotlib
#
# ───────────────────────────────────────────────────────────────────────────

import time
import torch
from agent import SACAgent
from needle_steering_env import NeedleSteeringEnv


# ─────────────────────────  environment  ────────────────────────────────
env = NeedleSteeringEnv(render_mode="human")
obs, _ = env.reset()

obs_dim = env.observation_space.shape[0]     # 9
act_dim = env.action_space.shape[0]          # 3  (len, steer_flag, plane)

# ─────────────────────────  SAC actor  ──────────────────────────────────
agent = SACAgent(obs_dim, act_dim,
                 env.action_space.low,
                 env.action_space.high)

agent.actor.load_state_dict(
    torch.load("sac_actor.pth", map_location="cpu")
)
agent.actor.eval()

# ─────────────────────────  rollout  ────────────────────────────────────
total_reward = 0.0
terminated    = False

print("\nstep | mode      | len   steer φ(rad) | reward")
print("─────┼───────────┼─────────────────────┼────────")

while not terminated:
    # select deterministic action
    action = agent.select_action(obs, eval_mode=True)   # → np.ndarray, shape (3,)

    # unpack three components
    length, steer_flag, plane = action
    mode = "Curved " if steer_flag >= 0.5 else "Straight"

    obs, reward, terminated, _, _ = env.step(action)
    total_reward += reward

    print(f"{env.steps:4d} | {mode:<9} | "
          f"{length:5.3f}   {steer_flag:.0f}   {plane:6.2f} | "
          f"{reward:6.2f}")
    env.render()
    

print("\nTOTAL REWARD:", f"{total_reward:.2f}")

input("Press Enter to close the window…")
env.close()

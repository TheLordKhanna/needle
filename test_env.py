import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from needle_steering_env import NeedleSteeringEnv

def test_random_actions():
    env = NeedleSteeringEnv(render_mode='human')
    
    obs, _ = env.reset()

    done = False
    while not done:
      
        action = np.array([np.random.choice([0, 1, 2]), np.random.uniform(0, 0.1), np.random.uniform(-np.pi, np.pi)])

   
        obs, reward, done, truncated, info = env.step(action)

        print(f"Step: {env.steps}, Position: {obs[:3]}, Orientation: {obs[3:6]}, Reward: {reward}")

        env.render()

    env.close()

if __name__ == "__main__":
    test_random_actions()

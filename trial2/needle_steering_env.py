import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class NeedleSteeringEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        *,
        curvature_fraction: float = 1.0,
        alpha: float = 3.0,
        eta: float = 0.0005,
        zeta: float = 0.0005,
        distance_weight: float = 150.0,
        length_cost: float = 0.0,
        switch_dist: float = 0.005,
        inside_scale = 1,
        evaluation_mode: bool = False,
        render_mode=None,
        seed=None,
    ):
        super().__init__()
        self.np_random = np.random.RandomState(seed)

        self.max_curvature = 10.44
        self.max_step_length = 0.005
        self.collision_rad = 0.005
        
        self.kappa_star = float(np.clip(curvature_fraction, 0, 1)) * self.max_curvature
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.zeta = float(zeta)
        self.dist_w = float(distance_weight)
        self.length_cost = float(length_cost)
        self.switch_dist = float(switch_dist)
        self.target_radius = 0.003
        self.time_penalty = 0.0
        self.straight_bonus = 10.0
        self.render_mode = render_mode
        self.inside_scale = float(inside_scale)

        self.evaluation_mode = evaluation_mode
        self.episode_counter = 0
        self.curriculum_schedule = [
            (500, 0.005),  
            (1500, 0.004), 
        ]

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi], dtype=np.float32),
            high=np.array([self.max_step_length, 1.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        
        self.workspace_bounds = {
            'low': np.array([-0.05, 0.0, 0.0], dtype=np.float32),
            'high': np.array([0.2, 0.15, 0.08], dtype=np.float32)
        }
        obs_low = np.concatenate([self.workspace_bounds['low'], [-1,-1,-1], self.workspace_bounds['low']])
        obs_high = np.concatenate([self.workspace_bounds['high'], [1,1,1], self.workspace_bounds['high']])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random.seed(seed)

        if not self.evaluation_mode:
            self.episode_counter += 1
            
            if self.episode_counter < self.curriculum_schedule[0][0]:
                self.target_radius = self.curriculum_schedule[0][1]
            elif self.episode_counter < self.curriculum_schedule[1][0]:
                self.target_radius = self.curriculum_schedule[1][1]
            else:
                self.target_radius = 0.003

            if self.episode_counter == self.curriculum_schedule[0][0]:
                print(f"\n--- Episode {self.episode_counter}: Curriculum Change -> Target Radius set to {self.target_radius*1000:.0f}mm ---\n")
            if self.episode_counter == self.curriculum_schedule[1][0]:
                print(f"\n--- Episode {self.episode_counter}: Curriculum Change -> Target Radius set to {self.target_radius*1000:.0f}mm ---\n")
        else:
            self.target_radius = 0.003

        self.pos = np.array([0.06, 0.06, 0.06], dtype=np.float32)
        self.orientation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.orientation /= np.linalg.norm(self.orientation)
        self.target = np.array([0.18, 0.09, 0.03], dtype=np.float32)

        self.steps = 0
        self.max_steps = 200
        self.path_length = 0.0
        self.trajectory = [self.pos.copy()]
        self.prev_plane = 0.0

        self.obstacles = [
            np.array([0.13, 0.06, 0.044], dtype=np.float32),
            np.array([0.13, 0.065, 0.044], dtype=np.float32),
            np.array([0.13, 0.07, 0.044], dtype=np.float32),
            np.array([0.13, 0.075, 0.044], dtype=np.float32),
            np.array([0.13, 0.08, 0.044], dtype=np.float32),
            np.array([0.13, 0.085, 0.044], dtype=np.float32),
            np.array([0.13, 0.09, 0.044], dtype=np.float32),
            np.array([0.13, 0.095, 0.044], dtype=np.float32),
            np.array([0.13, 0.06, 0.0465], dtype=np.float32),
            np.array([0.13, 0.065, 0.049], dtype=np.float32),
            np.array([0.13, 0.07, 0.0515], dtype=np.float32),
            np.array([0.13, 0.075, 0.054], dtype=np.float32),
        ]

        self.entered_switch_sphere = False
        
        return self.get_obs(), {}

    def step(self, action):
        length, sf, plane = action
        length = float(np.clip(length, 1e-5, self.max_step_length))
        plane = float(np.clip(plane, -np.pi, np.pi))
        steer_flag = 1.0 if sf >= 0.5 else 0.0

        self.path_length += length

        points, new_ori = self.move_segment(length, plane,
            self.kappa_star if steer_flag else 0.0)
        
        prev_pt = self.pos.copy()
        prev_dist = np.linalg.norm(prev_pt - self.target)
        success = False
        reward = 0.0

        for pt in points:
            dist = np.linalg.norm(pt - self.target)
            reward += self.dist_w * (prev_dist - dist)

            if prev_dist <= self.switch_dist:
                to_tgt = (self.target - prev_pt)
                to_tgt /= np.linalg.norm(to_tgt) + 1e-8
                proj = np.dot(pt - prev_pt, to_tgt)
                reward += (self.alpha * self.inside_scale) * proj
                
                if steer_flag == 0.0 and length > 1e-5:
                    reward += (self.straight_bonus * self.inside_scale) / len(points)
            
            if not self.entered_switch_sphere and dist < self.switch_dist:
                reward += 100.0 
                self.entered_switch_sphere = True

            if dist < self.target_radius:
                success = True
            
            prev_pt, prev_dist = pt.copy(), dist

            if success:
                break

        reward -= self.eta * steer_flag
        reward -= self.zeta * abs(plane - self.prev_plane)
        self.prev_plane = plane
        reward -= self.time_penalty

        collided = self.is_collision(prev_pt)
        if collided:
            reward -= 50.0

        if success:
            reward += 20000.0
            reward -= self.length_cost * self.path_length
            rem = max(self.max_steps - (self.steps + 1), 0)
            reward += 100.0 * (rem / self.max_steps)

        self.pos, self.orientation = prev_pt, new_ori
        self.trajectory.extend(points)
        self.steps += 1

        done = success or collided or (self.steps >= self.max_steps)
        return self.get_obs(), float(reward), done, self.steps >= self.max_steps, {}

    def render(self):
        if self.render_mode != "human": return
        if not hasattr(self, "_fig"):
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection="3d")
            plt.ion()
            plt.show()

        ax = self._ax
        ax.clear()
        
        ax.set_xlim(self.workspace_bounds['low'][0], self.workspace_bounds['high'][0])
        ax.set_ylim(self.workspace_bounds['low'][1], self.workspace_bounds['high'][1])
        ax.set_zlim(self.workspace_bounds['low'][2], self.workspace_bounds['high'][2])
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        
        ax.quiver(*self.pos, *self.orientation, length=0.02, color='blue')
        ax.scatter(*self.target, color="green", s=100, label="Target")
        for o in self.obstacles:
            ax.scatter(*o, color="red", s=60)
        tr = np.array(self.trajectory)
        ax.plot(tr[:,0], tr[:,1], tr[:,2], alpha=0.7, label="Needle Path")
        ax.legend()
        plt.pause(0.001)

    def close(self):
        if hasattr(self, "_fig"):
            plt.close(self._fig)
            del self._fig

    def get_obs(self):
        return np.concatenate([self.pos, self.orientation, self.target]).astype(np.float32)

    def is_collision(self, point):
        return any(np.linalg.norm(point - o) < self.collision_rad for o in self.obstacles)

    def rotate_about_local_z(self, angle):
        z = self.orientation / (np.linalg.norm(self.orientation) + 1e-8)
        Kz = np.array([[0, -z[2], z[1]],
                           [z[2], 0, -z[0]],
                           [-z[1], z[0], 0]])
        return np.eye(3) + np.sin(angle) * Kz + (1 - np.cos(angle)) * (Kz @ Kz)

    def move_segment(self, length, plane_angle, kappa):
        if kappa < 1e-4:
            end = self.pos + self.orientation * length
            return [end], self.orientation.copy()

        perp = np.cross(self.orientation, [0, 0, 1])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(self.orientation, [0, 1, 0])
        perp /= np.linalg.norm(perp)
        perp = self.rotate_about_local_z(plane_angle) @ perp

        R = 1.0 / kappa
        theta = length / R
        axis = np.cross(self.orientation, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        centre = self.pos + perp * R

        thetas = np.linspace(0, theta, 16)[1:]
        points = []
        for t in thetas:
            Rbend = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
            pt = centre + Rbend @ (self.pos - centre)
            points.append(pt)

        Rbend_f = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        new_ori = Rbend_f @ self.orientation
        new_ori /= np.linalg.norm(new_ori)

        return points, new_ori

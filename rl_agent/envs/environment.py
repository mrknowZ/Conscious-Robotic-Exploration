import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ToyGridworldEnv(gym.Env):
    """
    Toy Gridworld RL Environment for Mobile Robot Navigation.
    Simulates a 2D grid space with battery drain dynamics.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=5, initial_battery=100.0):
        super(ToyGridworldEnv, self).__init__()
        
        self.grid_size = grid_size
        self.initial_battery = initial_battery
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4: Wait/Idle
        self.action_space = spaces.Discrete(5)
        
        # State: [x, y, battery_level]
        low = np.array([0, 0, 0.0], dtype=np.float32)
        high = np.array([grid_size - 1, grid_size - 1, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Target Goal Position
        self.target_pos = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)
        
        self.agent_pos = None
        self.battery = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start robot at position (0, 0)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.battery = float(self.initial_battery)
        
        observation = np.array([self.agent_pos[0], self.agent_pos[1], self.battery], dtype=np.float32)
        info = {}
        
        return observation, info

    def step(self, action):
        # Action Movement Logic
        if action == 0 and self.agent_pos[1] < self.grid_size - 1:  # Up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:                # Down
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:                # Left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # Right
            self.agent_pos[0] += 1
        # Action 4 is Wait/Idle (no position change)

        # Battery Consumption Logic (2% per step)
        self.battery -= 2.0
        
        # Goal Check
        reached_goal = np.array_equal(self.agent_pos, self.target_pos)
        
        # Rewards Logic
        if reached_goal:
            reward = 100.0
            terminated = True
        elif self.battery <= 0:
            reward = -50.0
            terminated = True
        else:
            reward = -1.0  # Step penalty to encourage efficiency
            terminated = False

        truncated = False
        observation = np.array([self.agent_pos[0], self.agent_pos[1], self.battery], dtype=np.float32)
        info = {"battery_remaining": self.battery}

        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    env = ToyGridworldEnv()
    obs, info = env.reset()
    print("Initial State [x, y, battery]:", obs)
    
    next_obs, reward, terminated, truncated, info = env.step(3)
    print("State after moving Right:", next_obs)
    print("Reward received:", reward)
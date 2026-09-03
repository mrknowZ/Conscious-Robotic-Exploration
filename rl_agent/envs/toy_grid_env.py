import gymnasium as gym
from gymnasium import spaces
import numpy as np

from energy_model.ina219_energy_model import INA219EnergyModel


class ToyGridEnv(gym.Env):
    """
    Energy-Aware NxN Grid World Environment (Action-Masking Ready).

    This is the exact v5 configuration that achieved:
        - 100% coverage (49/49 cells) on 7x7
        - 0 wall collisions
        - 924.00 J total energy (77 steps x 12J flat rate)

    Actions:
        0 = Up
        1 = Down
        2 = Left
        3 = Right

    Observation (for grid_size=10 -> 108 values):
        - Robot position (normalized):      2
        - Boundary info [U,D,L,R]:          4
        - Visited grid (flattened):         grid_size * grid_size
        - Coverage fraction so far:         1
        - Steps remaining (normalized):     1
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size=10):

        super().__init__()

        self.grid_size = grid_size
        self.total_cells = grid_size * grid_size

        # =====================================================
        # ACTION SPACE
        # =====================================================

        self.action_space = spaces.Discrete(4)

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================

        observation_size = 2 + 4 + self.total_cells + 1 + 1

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32
        )

        # =====================================================
        # ENERGY MODEL
        # =====================================================

        self.energy_sensor = INA219EnergyModel(
            voltage_nom=12.0
        )

        # =====================================================
        # EPISODE SETTINGS
        # =====================================================
        # Generous buffer (2.2x the theoretical minimum) so the
        # agent has real room to explore before being cut off.

        self.max_steps = int((self.total_cells - 1) * 2.2)

        # =====================================================
        # STATE VARIABLES
        # =====================================================

        self.robot_pos = None
        self.grid_matrix = None

        self.visited_count = 0
        self.steps_taken = 0

        self.wall_collisions = 0
        self.total_energy = 0.0

        self.previous_positions = []

    # =========================================================
    # ACTION MASK (used by sb3-contrib MaskablePPO)
    # =========================================================

    def action_masks(self):
        """
        Returns a boolean array of shape (4,).
        True  = action is legal (won't hit a wall)
        False = action is illegal (would hit a wall)
        """

        x, y = self.robot_pos

        return np.array(
            [
                y < self.grid_size - 1,  # UP
                y > 0,                   # DOWN
                x > 0,                   # LEFT
                x < self.grid_size - 1,  # RIGHT
            ],
            dtype=bool
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_obs(self):

        if self.grid_size > 1:
            pos_norm = (
                self.robot_pos.astype(np.float32)
                / float(self.grid_size - 1)
            )
        else:
            pos_norm = np.zeros(2, dtype=np.float32)

        x, y = self.robot_pos

        boundary_info = np.array(
            [
                1.0 if y >= self.grid_size - 1 else 0.0,
                1.0 if y <= 0 else 0.0,
                1.0 if x <= 0 else 0.0,
                1.0 if x >= self.grid_size - 1 else 0.0
            ],
            dtype=np.float32
        )

        grid_flat = self.grid_matrix.flatten()

        coverage = np.array(
            [self.visited_count / self.total_cells],
            dtype=np.float32
        )

        steps_remaining = np.array(
            [max(0.0, 1.0 - (self.steps_taken / self.max_steps))],
            dtype=np.float32
        )

        observation = np.concatenate(
            [pos_norm, boundary_info, grid_flat, coverage, steps_remaining]
        ).astype(np.float32)

        return observation

    # =========================================================
    # GET NEAREST UNVISITED DISTANCE
    # =========================================================

    def _nearest_unvisited_distance(self):

        unvisited = np.argwhere(self.grid_matrix == 0)

        if len(unvisited) == 0:
            return 0

        x, y = self.robot_pos

        distances = (
            np.abs(unvisited[:, 0] - x) + np.abs(unvisited[:, 1] - y)
        )

        return int(np.min(distances))

    # =========================================================
    # ACTION TARGET
    # =========================================================

    def _action_target(self, action):

        x, y = self.robot_pos

        if action == 0:
            y += 1
        elif action == 1:
            y -= 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1

        return x, y

    # =========================================================
    # RESET
    # =========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.robot_pos = np.array([0, 0], dtype=np.int32)

        self.grid_matrix = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.float32
        )

        self.grid_matrix[0, 0] = 1.0

        self.visited_count = 1
        self.steps_taken = 0

        self.wall_collisions = 0
        self.total_energy = 0.0

        self.previous_positions = [(0, 0)]

        return self._get_obs(), {}

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        action = int(action)

        self.steps_taken += 1

        old_distance = self._nearest_unvisited_distance()

        hit_wall = False
        new_cell = False

        # =====================================================
        # MOVEMENT
        # =====================================================

        target_x, target_y = self._action_target(action)

        if (
            target_x < 0 or target_x >= self.grid_size
            or target_y < 0 or target_y >= self.grid_size
        ):
            # Should basically never happen once MaskablePPO is
            # used for training/inference -- kept as a safety net.
            hit_wall = True
        else:
            self.robot_pos[0] = target_x
            self.robot_pos[1] = target_y

        # =====================================================
        # ENERGY (flat rate every step -- v5 behavior)
        # =====================================================

        energy_spent = self.energy_sensor.get_energy_consumption(
            action_type="flat",
            step_duration_sec=1.0
        )

        self.total_energy += energy_spent

        energy_penalty = 0.02 * energy_spent

        # =====================================================
        # REWARD
        # =====================================================

        reward = 0.0

        if hit_wall:

            self.wall_collisions += 1
            reward = -8.0

        else:

            x, y = self.robot_pos

            if self.grid_matrix[x, y] == 0.0:

                self.grid_matrix[x, y] = 1.0
                self.visited_count += 1
                new_cell = True

                reward = 10.0
                reward += 2.0 * (self.visited_count / self.total_cells)

                if self.visited_count == self.total_cells:
                    reward += 50.0

            else:

                revisit_penalty = min(
                    1.0 + (self.steps_taken * 0.02), 3.0
                )
                reward = -revisit_penalty

        # =====================================================
        # DISTANCE SHAPING
        # =====================================================

        new_distance = self._nearest_unvisited_distance()

        if not hit_wall:
            if new_distance < old_distance:
                reward += 0.60
            elif new_distance > old_distance:
                reward -= 0.40

        # =====================================================
        # LOOP DETECTION
        # =====================================================

        current_position = (int(self.robot_pos[0]), int(self.robot_pos[1]))

        if len(self.previous_positions) >= 4:
            if current_position in self.previous_positions[-4:]:
                reward -= 2.0

        self.previous_positions.append(current_position)

        if len(self.previous_positions) > 10:
            self.previous_positions.pop(0)

        # =====================================================
        # ENERGY PENALTY (applied exactly once, every step)
        # =====================================================

        reward -= energy_penalty

        # =====================================================
        # TERMINATION
        # =====================================================

        terminated = self.visited_count >= self.total_cells
        truncated = self.steps_taken >= self.max_steps and not terminated

        # =====================================================
        # INFO
        # =====================================================

        info = {
            "energy_spent": float(energy_spent),
            "total_energy": float(self.total_energy),
            "visited_cells": int(self.visited_count),
            "coverage": float(self.visited_count / self.total_cells),
            "steps": int(self.steps_taken),
            "wall_collision": bool(hit_wall),
            "wall_collisions": int(self.wall_collisions),
            "new_cell": bool(new_cell),
            "position": (int(self.robot_pos[0]), int(self.robot_pos[1])),
        }

        return self._get_obs(), float(reward), terminated, truncated, info
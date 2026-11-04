"""Custom 6x6 warehouse delivery environment (Gymnasium-compatible)."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WarehouseEnv(gym.Env):
    """Deliver one package while handling battery and temporary blocked cells.

    Coordinates are (row, column). Actions: 0 up, 1 down, 2 left,
    3 right, 4 recharge. The observation contains position, battery,
    package flag, package position, and two temporary-obstacle flags.
    """
    metadata = {"render_modes": ["ansi"]}
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ("up", "down", "left", "right", "recharge")

    def __init__(self, size=6, max_battery=12, max_steps=100,
                 obstacle_probability=0.15, randomize_package=False,
                 render_mode=None):
        super().__init__()
        if size != 6:
            raise ValueError("This milestone uses a fixed 6x6 layout")
        if max_battery < 1 or max_steps < 1:
            raise ValueError("max_battery and max_steps must be positive")
        if not 0 <= obstacle_probability <= 1:
            raise ValueError("obstacle_probability must be in [0, 1]")
        self.size, self.max_battery, self.max_steps = size, max_battery, max_steps
        self.obstacle_probability = obstacle_probability
        self.randomize_package = randomize_package
        self.render_mode = render_mode
        self.start = (5, 0)
        self.charger = (5, 2)
        self.delivery = (0, 5)
        self.default_package = (2, 1)
        self.walls = frozenset({(1, 1), (1, 2), (3, 3), (4, 3)})
        self.dynamic_cells = ((2, 3), (3, 4))
        self.action_space = spaces.Discrete(5)
        # r, c, battery, carrying, package r, package c, obstacle flags
        self.observation_space = spaces.MultiDiscrete(
            [6, 6, max_battery + 1, 2, 6, 6, 2, 2]
        )
        self.position = self.start
        self.battery = max_battery
        self.carrying = False
        self.package = self.default_package
        self.obstacles = set()
        self.steps = 0
        self.done = False

    def _observation(self):
        return np.array([*self.position, self.battery, int(self.carrying),
                         *self.package, *(int(c in self.obstacles) for c in self.dynamic_cells)],
                        dtype=np.int64)

    def _info(self, success=False, reason="running"):
        return {"success": success, "reason": reason, "steps": self.steps,
                "battery": self.battery, "carrying": self.carrying}

    def _sample_obstacles(self):
        # Independently resample temporary blocks each time step.
        # Never block the robot's current cell or any special cell.
        protected = {self.position, self.package, self.start, self.charger, self.delivery}
        self.obstacles = {cell for cell in self.dynamic_cells
                          if cell not in protected and self.np_random.random() < self.obstacle_probability}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start
        self.battery = self.max_battery
        self.carrying = False
        self.steps = 0
        self.done = False
        self.package = self.default_package
        if self.randomize_package:
            excluded = self.walls | {self.start, self.charger, self.delivery} | set(self.dynamic_cells)
            candidates = [(r, c) for r in range(6) for c in range(6) if (r, c) not in excluded]
            self.package = candidates[int(self.np_random.integers(len(candidates)))]
        self._sample_obstacles()
        return self._observation(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.steps += 1
        reward = -1.0  # discourage wasting time, including pointless recharging
        if action == 4:
            if self.position == self.charger:
                self.battery = self.max_battery
            else:
                reward -= 2.0
        else:
            # Every movement attempt consumes one battery unit, including blocked moves.
            self.battery -= 1
            dr, dc = self.ACTIONS[int(action)]
            nxt = (self.position[0] + dr, self.position[1] + dc)
            if (not 0 <= nxt[0] < 6 or not 0 <= nxt[1] < 6
                    or nxt in self.walls or nxt in self.obstacles):
                reward -= 2.0
            else:
                self.position = nxt
        if self.position == self.package and not self.carrying:
            self.carrying = True  # automatic pickup
        success = self.carrying and self.position == self.delivery
        terminated = bool(success or self.battery <= 0)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        reason = "running"
        if success:
            reward += 30.0
            reason = "delivered"
        elif self.battery <= 0:
            reward -= 5.0
            reason = "battery_empty"
        elif truncated:
            reason = "time_limit"
        self.done = terminated or truncated
        if not self.done:
            self._sample_obstacles()
        return self._observation(), float(reward), terminated, truncated, self._info(success, reason)

    def render(self):
        grid = [["." for _ in range(6)] for _ in range(6)]
        for r, c in self.walls:
            grid[r][c] = "#"
        for r, c in self.obstacles:
            grid[r][c] = "X"
        for point, char in ((self.charger, "C"), (self.delivery, "D")):
            grid[point[0]][point[1]] = char
        if not self.carrying:
            grid[self.package[0]][self.package[1]] = "P"
        grid[self.position[0]][self.position[1]] = "R"
        return "\n".join(" ".join(row) for row in grid) + (f"\nBattery: {self.battery}/{self.max_battery} | "
                f"Carrying: {self.carrying} | Step: {self.steps}")

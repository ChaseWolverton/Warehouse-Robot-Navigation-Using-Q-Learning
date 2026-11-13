"""Sparse tabular Q-learning agent for discrete warehouse observations."""
from collections import defaultdict
import numpy as np

class QLearningAgent:
    def __init__(self, n_actions=5, alpha=0.15, gamma=0.97, seed=7):
        self.n_actions, self.alpha, self.gamma = n_actions, alpha, gamma
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def key(observation):
        return tuple(int(x) for x in observation)

    def act(self, observation, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q[self.key(observation)]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))  # random tie-breaking

    def update(self, observation, action, reward, next_observation, terminated):
        state = self.key(observation)
        next_state = self.key(next_observation)
        target = reward if terminated else reward + self.gamma * np.max(self.q[next_state])
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def save(self, path):
        keys = np.asarray(list(self.q), dtype=np.int64).reshape(-1, 8)
        values = np.asarray(list(self.q.values()), dtype=np.float64).reshape(-1, self.n_actions)
        np.savez_compressed(path, keys=keys, values=values)

    def load(self, path):
        data = np.load(path)
        self.q.clear()
        for key, value in zip(data['keys'], data['values']):
            self.q[tuple(int(x) for x in key)] = value.copy()

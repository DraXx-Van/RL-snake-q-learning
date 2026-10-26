"""The Q-learning agent: a table of 2048 states x 3 actions."""
import os
import random
from typing import NamedTuple

import numpy as np

import config


class LearnStep(NamedTuple):
    """Everything about one Q-update, so the UI can show the maths."""
    state: int
    action: int
    reward: float
    done: bool
    old: float        # Q[s][a] before the update
    future: float     # max Q[s'] (0 if the game ended)
    target: float     # r + gamma * max Q[s']
    new: float        # Q[s][a] after the update


class QAgent:
    def __init__(self, seed=None):
        self.q = np.zeros((config.N_STATES, config.N_ACTIONS))
        self.visits = np.zeros(config.N_STATES, dtype=np.int64)
        self.epsilon = config.EPS_START
        self.episodes = 0
        self.rng = random.Random(seed)

    # ----------------------------------------------------------- acting
    def best_action(self, s):
        """Highest-valued action; ties (e.g. an all-zero row) broken randomly."""
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best.tolist()))

    def choose_action(self, s, greedy=False):
        """Epsilon-greedy. Returns (action, explored)."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(config.N_ACTIONS), True
        return self.best_action(s), False

    # --------------------------------------------------------- learning
    def learn(self, s, a, r, s2, done):
        """Q[s][a] += alpha * (r + gamma * max Q[s'] - Q[s][a])"""
        old = float(self.q[s, a])
        future = 0.0 if done else float(self.q[s2].max())
        target = r + config.GAMMA * future
        new = old + config.ALPHA * (target - old)
        self.q[s, a] = new
        self.visits[s] += 1
        return LearnStep(s, a, r, done, old, future, target, new)

    def end_episode(self):
        self.episodes += 1
        self.epsilon = max(config.EPS_MIN, self.epsilon * config.EPS_DECAY)

    # -------------------------------------------------------- save/load
    def save(self, path=config.Q_TABLE_PATH):
        np.savez(path, q=self.q, visits=self.visits,
                 epsilon=self.epsilon, episodes=self.episodes)

    @classmethod
    def load(cls, path=config.Q_TABLE_PATH, seed=None):
        data = np.load(path)
        agent = cls(seed)
        agent.q = data["q"]
        agent.visits = data["visits"]
        agent.epsilon = float(data["epsilon"])
        agent.episodes = int(data["episodes"])
        return agent

    @staticmethod
    def exists(path=config.Q_TABLE_PATH):
        return os.path.exists(path)

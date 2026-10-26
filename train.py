"""Headless training (no window): fast, used for the real training run."""
import time

import numpy as np

import config
from agent import QAgent
from game import SnakeEnv


def train(episodes=config.EPISODES, seed=config.SEED, log_every=100,
          save_path=config.Q_TABLE_PATH, verbose=True):
    env = SnakeEnv(seed=seed)
    agent = QAgent(seed=seed)
    history = {"scores": [], "epsilon": [], "steps": []}
    best = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        while not done:
            a, _ = agent.choose_action(s)
            r, done, score = env.step(a)
            s2 = env.get_state()
            agent.learn(s, a, r, s2, done)
            s = s2
        history["scores"].append(env.score)
        history["epsilon"].append(agent.epsilon)
        history["steps"].append(env.steps)
        best = max(best, env.score)
        agent.end_episode()

        if verbose and ep % log_every == 0:
            avg = np.mean(history["scores"][-log_every:])
            print(f"Episode {ep:5d} | avg score (last {log_every}): {avg:5.2f} "
                  f"| best: {best:3d} | epsilon: {history['epsilon'][-1]:.3f}")

    if save_path:
        agent.save(save_path)
        if verbose:
            print(f"Saved Q-table to {save_path}  ({time.time() - t0:.1f}s)")
    return agent, history


if __name__ == "__main__":
    train()

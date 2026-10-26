"""Evaluate an agent on fresh games with learning switched off."""
import random

import numpy as np

import config
from agent import QAgent
from game import SnakeEnv

EVAL_SEED = 12345   # different from the training seed -> games the agent never trained on


def evaluate(policy, games=100, seed=EVAL_SEED):
    """policy: function state -> action. Returns a dict of statistics."""
    env = SnakeEnv(seed=seed)
    scores, steps = [], []
    for _ in range(games):
        s = env.reset()
        done = False
        while not done:
            _, done, _ = env.step(policy(s))
            s = env.get_state()
        scores.append(env.score)
        steps.append(env.steps)
    return {
        "games": games,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "best_score": int(np.max(scores)),
        "std_score": float(np.std(scores)),
        "mean_steps": float(np.mean(steps)),
    }


def random_policy(seed=0):
    rng = random.Random(seed)
    return lambda s: rng.randrange(config.N_ACTIONS)


def trained_policy(agent):
    return agent.best_action      # pure greedy: epsilon = 0


def compare(path=config.Q_TABLE_PATH, games=100):
    agent = QAgent.load(path)
    results = {
        "Random snake": evaluate(random_policy(), games),
        "Trained Q-agent": evaluate(trained_policy(agent), games),
    }
    lines = [f"{'Agent':<18}{'Avg score':>10}{'Median':>8}{'Best':>6}{'Std':>7}{'Avg steps':>11}"]
    for name, r in results.items():
        lines.append(f"{name:<18}{r['mean_score']:>10.2f}{r['median_score']:>8.1f}"
                     f"{r['best_score']:>6d}{r['std_score']:>7.2f}{r['mean_steps']:>11.1f}")
    text = "\n".join(lines)
    print(f"\nEvaluation over {games} fresh games (no learning, epsilon = 0 for the trained agent)\n")
    print(text)
    return results, text


if __name__ == "__main__":
    compare()

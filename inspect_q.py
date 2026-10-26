"""Look inside the learned Q-table: is it sensible, not a black box?"""
import numpy as np

import config
from agent import QAgent
from game import ACTION_NAMES, describe_state, encode_state

# (name, danger[ahead,right,left], food[ahead,behind,left,right], sensible actions)
SCENARIOS = [
    ("Apple straight ahead, path clear",   (0, 0, 0), (1, 0, 0, 0), {0}),
    ("Wall ahead, apple on the left",      (1, 0, 0), (0, 0, 1, 0), {2}),
    ("Wall ahead, apple on the right",     (1, 0, 0), (0, 0, 0, 1), {1}),
    ("Danger on the right, apple right",   (0, 1, 0), (0, 0, 0, 1), {0, 2}),
    ("Wall ahead and right, apple ahead",  (1, 1, 0), (1, 0, 0, 0), {2}),
]


def scenario_state(danger, food):
    return encode_state(list(danger) + list(food))


def inspect(path=config.Q_TABLE_PATH, top=10):
    agent = QAgent.load(path)
    visited = int((agent.visits > 0).sum())
    print(f"Q-table: {config.N_STATES} situations x {config.N_ACTIONS} moves, "
          f"trained for {agent.episodes} episodes")
    print(f"Situations ever met: {visited} / {config.N_STATES} "
          f"({100 * visited / config.N_STATES:.0f}%)  (the rest are impossible, e.g. apple both ahead and behind)\n")

    print("Hand-picked situations (does the snake choose sensibly?)")
    for name, danger, food, ok in SCENARIOS:
        s = scenario_state(danger, food)
        row = agent.q[s]
        best = int(np.argmax(row))
        mark = "OK " if best in ok else "?? "
        q = "  ".join(f"{ACTION_NAMES[i]}={row[i]:+6.2f}" for i in range(3))
        print(f"  {mark}{name}\n       situation #{s}: {q}  ->  {ACTION_NAMES[best]}")

    print(f"\nTop {top} most-visited situations")
    for s in np.argsort(agent.visits)[::-1][:top]:
        if agent.visits[s] == 0:
            break
        dangers, food = describe_state(int(s))
        best = int(np.argmax(agent.q[s]))
        print(f"  {agent.visits[s]:>8,}x  danger: {', '.join(dangers) or 'none':<18}"
              f"| apple: {', '.join(food) or '-':<14} -> {ACTION_NAMES[best]} ({agent.q[s, best]:+.1f})")


if __name__ == "__main__":
    inspect()

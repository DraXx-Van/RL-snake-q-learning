import numpy as np
import pytest

import config
from agent import QAgent
from evaluate import evaluate, random_policy, trained_policy
from train import train


def test_q_table_shape_and_zero_init():
    a = QAgent(seed=0)
    assert a.q.shape == (config.N_STATES, config.N_ACTIONS) == (128, 3)
    assert not a.q.any()


def test_greedy_picks_argmax():
    a = QAgent(seed=0)
    a.q[10] = [1.0, 5.0, 2.0]
    a.epsilon = 0.0
    assert all(a.choose_action(10)[0] == 1 for _ in range(50))


def test_epsilon_one_explores_all_actions():
    a = QAgent(seed=0)
    a.q[10] = [1.0, 5.0, 2.0]
    a.epsilon = 1.0
    seen = {a.choose_action(10)[0] for _ in range(200)}
    assert seen == {0, 1, 2}


def test_ties_are_broken_randomly():
    a = QAgent(seed=0)           # all-zero row: every action ties
    assert {a.best_action(5) for _ in range(100)} == {0, 1, 2}


def test_update_rule_matches_hand_calculation():
    """Worked example from the guide: alpha=0.1, gamma=0.9, r=+10, maxQ(s')=2 -> 1.18"""
    a = QAgent(seed=0)
    a.q[7] = [0, 2.0, 0]          # best Q in next state is 2
    step = a.learn(s=3, a=0, r=10, s2=7, done=False)  # 3 and 7 are just row numbers
    assert step.target == pytest.approx(10 + config.GAMMA * 2)
    assert a.q[3, 0] == pytest.approx(1.18)
    assert step.old == 0 and step.new == pytest.approx(1.18)


def test_terminal_update_ignores_future():
    a = QAgent(seed=0)
    a.q[7] = [50, 50, 50]
    a.learn(s=3, a=1, r=-10, s2=7, done=True)
    assert a.q[3, 1] == pytest.approx(config.ALPHA * -10)


def test_update_moves_toward_target():
    a = QAgent(seed=0)
    a.q[3, 0] = 20.0
    a.learn(3, 0, 0, 4, done=False)           # target = 0 -> value must drop
    assert a.q[3, 0] < 20.0


def test_epsilon_decays_but_not_below_min():
    a = QAgent(seed=0)
    for _ in range(5000):
        a.end_episode()
    assert a.epsilon == pytest.approx(config.EPS_MIN)
    assert a.episodes == 5000


def test_save_load_roundtrip(tmp_path):
    a = QAgent(seed=0)
    a.q[:] = np.random.default_rng(0).normal(size=a.q.shape)
    a.visits[5] = 9
    a.end_episode()
    p = str(tmp_path / "q.npz")
    a.save(p)
    b = QAgent.load(p)
    assert np.array_equal(a.q, b.q) and np.array_equal(a.visits, b.visits)
    assert b.epsilon == a.epsilon and b.episodes == 1


def test_training_beats_random_baseline():
    """End-to-end: a short training run must clearly outperform a random snake."""
    agent, _ = train(episodes=1000, seed=42, save_path=None, verbose=False)
    trained = evaluate(trained_policy(agent), games=40, seed=999)["mean_score"]
    baseline = evaluate(random_policy(1), games=40, seed=999)["mean_score"]
    assert trained > baseline + 3

import pytest

import config
from game import (DOWN, LEFT, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, SnakeEnv,
                  decode_state, encode_state)


@pytest.fixture
def env():
    return SnakeEnv(seed=1)


def test_initial_snake(env):
    assert len(env.snake) == 3
    assert env.direction == RIGHT
    assert env.score == 0
    assert env.food not in env.snake


def test_straight_moves_one_cell(env):
    hx, hy = env.snake[0]
    env.food = (0, 0)
    env.step(STRAIGHT)
    assert env.snake[0] == (hx + 1, hy)
    assert len(env.snake) == 3


def test_turning_right_four_times_returns_to_start(env):
    env.food = (0, 0)
    start = env.direction
    for _ in range(4):
        env.direction = (env.direction + 1) % 4
    assert env.direction == start


def test_turn_right_and_left_change_heading(env):
    env.food = (0, 0)
    env.step(TURN_RIGHT)          # heading RIGHT -> DOWN
    assert env.direction == DOWN
    env.step(TURN_LEFT)           # DOWN -> RIGHT
    assert env.direction == RIGHT


def test_wall_collision_gives_death_reward(env):
    env.snake = [(config.GRID_W - 1, 5), (config.GRID_W - 2, 5), (config.GRID_W - 3, 5)]
    env.direction = RIGHT
    env.food = (0, 0)
    reward, done, _ = env.step(STRAIGHT)
    assert done and reward == config.R_DEATH


def test_self_collision_ends_game(env):
    # head at (5,5) moving up into its own body at (5,4)
    env.snake = [(5, 5), (6, 5), (6, 4), (5, 4), (4, 4)]
    env.direction = LEFT
    env.food = (0, 0)
    reward, done, _ = env.step(TURN_RIGHT)    # LEFT -> UP, into (5,4)
    assert done and reward == config.R_DEATH


def test_moving_into_vacating_tail_is_safe(env):
    # 2x2 loop: head chases its own tail cell, which moves away this step
    env.snake = [(5, 5), (5, 4), (4, 4), (4, 5)]
    env.direction = DOWN
    env.food = (0, 0)
    _, done, _ = env.step(TURN_RIGHT)         # DOWN -> LEFT, into (4,5) = tail
    assert not done


def test_eating_food_scores_and_grows(env):
    hx, hy = env.snake[0]
    env.food = (hx + 1, hy)
    reward, done, score = env.step(STRAIGHT)
    assert reward == config.R_FOOD and not done and score == 1
    assert len(env.snake) == 4
    assert env.food not in env.snake


def test_food_never_spawns_on_snake():
    env = SnakeEnv(seed=3)
    for _ in range(200):
        env._place_food()
        assert env.food not in env.snake


def test_timeout_ends_game(env):
    env.food = (0, 0)
    env.steps_since_food = config.TIMEOUT_FACTOR * len(env.snake)
    env.snake = [(7, 7), (6, 7), (5, 7)]
    reward, done, _ = env.step(STRAIGHT)
    assert done and env.timed_out and reward == config.R_DEATH


def test_state_always_in_range():
    env = SnakeEnv(seed=7)
    import random
    rng = random.Random(0)
    for _ in range(2000):
        s = env.get_state()
        assert 0 <= s < config.N_STATES
        _, done, _ = env.step(rng.randrange(3))
        if done:
            env.reset()


def test_encode_decode_roundtrip():
    for s in (0, 1, 45, 100, config.N_STATES - 1):
        assert encode_state(decode_state(s)) == s
    assert len(decode_state(0)) == config.N_BITS == 7


def test_danger_flags_at_left_wall():
    env = SnakeEnv(seed=1)
    env.snake = [(0, 5), (1, 5), (2, 5)]
    env.direction = LEFT
    env.food = (10, 10)
    ahead, right, left = env.get_state_bits()[:3]
    assert ahead == 1          # wall straight ahead
    assert right == 0          # heading LEFT, right of that is UP -> (0,4) free
    assert left == 0           # DOWN -> (0,6) free


def test_danger_flags_corner():
    env = SnakeEnv(seed=1)
    env.snake = [(0, 0), (1, 0), (2, 0)]
    env.direction = LEFT
    env.food = (10, 10)
    ahead, right, left = env.get_state_bits()[:3]
    assert ahead == 1 and right == 1 and left == 0     # wall ahead, wall above (UP), free below


def _food_bits(direction, food):
    env = SnakeEnv(seed=1)
    env.snake = [(7, 7), (6, 7), (5, 7)]
    env.direction = direction
    env.food = food
    return [int(b) for b in env.get_state_bits()[3:7]]      # [ahead, behind, left, right]


def test_food_bits_are_relative_to_heading():
    # snake at (7,7) facing RIGHT: ahead = +x, snake's right = +y (down on screen)
    assert _food_bits(RIGHT, (12, 7)) == [1, 0, 0, 0]       # straight ahead
    assert _food_bits(RIGHT, (2, 7)) == [0, 1, 0, 0]        # behind
    assert _food_bits(RIGHT, (7, 2)) == [0, 0, 1, 0]        # screen-up = snake's left
    assert _food_bits(RIGHT, (7, 12)) == [0, 0, 0, 1]       # screen-down = snake's right
    assert _food_bits(RIGHT, (10, 3)) == [1, 0, 1, 0]       # ahead and left


def test_same_food_looks_different_when_heading_changes():
    # the apple is at the same place, but the snake's point of view changes
    assert _food_bits(RIGHT, (7, 2)) == [0, 0, 1, 0]        # facing right: apple on left
    assert _food_bits(LEFT, (7, 2)) == [0, 0, 0, 1]         # facing left: apple on right
    assert _food_bits(UP, (7, 2)) == [1, 0, 0, 0]           # facing up: apple ahead
    assert _food_bits(DOWN, (7, 2)) == [0, 1, 0, 0]         # facing down: apple behind

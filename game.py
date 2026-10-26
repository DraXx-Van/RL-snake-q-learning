"""Snake environment: pure game logic, no drawing.

Gym-style API:
    state = env.reset()
    reward, done, score = env.step(action)
    state = env.get_state()

Actions are *relative* to the snake's heading:
    0 = straight, 1 = turn right, 2 = turn left
"""
import random

import config

# Clockwise order, so "turn right" is +1 and "turn left" is -1. y grows downward.
RIGHT, DOWN, LEFT, UP = range(4)
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIR_NAMES = ["RIGHT", "DOWN", "LEFT", "UP"]

STRAIGHT, TURN_RIGHT, TURN_LEFT = range(3)
ACTION_NAMES = ["Straight", "Turn right", "Turn left"]

# Order of the 7 yes/no facts that make up a state. Everything is relative to
# the way the snake is facing, so "left" means the snake's left.
BIT_LABELS = [
    "danger ahead", "danger right", "danger left",
    "food ahead", "food behind", "food left", "food right",
]
DANGER_NAMES = ["ahead", "right", "left"]
FOOD_NAMES = ["ahead", "behind", "left", "right"]


def encode_state(bits):
    """7 yes/no facts -> one integer 0..127 (the Q-table row)."""
    s = 0
    for b in bits:
        s = s * 2 + int(b)
    return s


def decode_state(s):
    """Integer 0..127 -> list of the 7 yes/no facts."""
    n = config.N_BITS
    return [(s >> (n - 1 - i)) & 1 for i in range(n)]


def describe_state(s):
    """Plain-English summary of a state: ([dangers], [food directions])."""
    bits = decode_state(s)
    dangers = [n for n, b in zip(DANGER_NAMES, bits[0:3]) if b]
    food = [n for n, b in zip(FOOD_NAMES, bits[3:7]) if b]
    return dangers, food


class SnakeEnv:
    def __init__(self, w=config.GRID_W, h=config.GRID_H, seed=None):
        self.w, self.h = w, h
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.direction = RIGHT
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]   # head first
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.ate_food = False
        self.timed_out = False
        self.won = False
        self.done = False
        self.food = None
        self._place_food()
        return self.get_state()

    def _place_food(self):
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in occupied]
        self.food = self.rng.choice(free) if free else None

    # ----------------------------------------------------------------- step
    def step(self, action):
        """Advance one move. Returns (reward, done, score)."""
        self.ate_food = False
        if action == TURN_RIGHT:
            self.direction = (self.direction + 1) % 4
        elif action == TURN_LEFT:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRS[self.direction]
        hx, hy = self.snake[0]
        head = (hx + dx, hy + dy)
        self.steps += 1
        self.steps_since_food += 1

        eating = head == self.food
        # The tail cell is freed this move unless we are growing.
        body = self.snake if eating else self.snake[:-1]
        if self._hits_wall(head) or head in body:
            self.done = True
            return config.R_DEATH, True, self.score

        self.snake.insert(0, head)
        if eating:
            self.score += 1
            self.steps_since_food = 0
            self.ate_food = True
            reward = config.R_FOOD
            if len(self.snake) == self.w * self.h:     # board completely full
                self.won = True
                self.done = True
                self.food = None
                return reward, True, self.score
            self._place_food()
        else:
            self.snake.pop()
            reward = config.R_STEP

        if self.steps_since_food > config.TIMEOUT_FACTOR * len(self.snake):
            self.timed_out = True
            self.done = True
            return config.R_DEATH, True, self.score
        return reward, False, self.score

    # ---------------------------------------------------------------- state
    def _hits_wall(self, p):
        return not (0 <= p[0] < self.w and 0 <= p[1] < self.h)

    def _danger(self, rel):
        """Would moving straight (0) / right (1) / left (2) kill us next step?"""
        d = (self.direction + (0, 1, -1)[rel]) % 4
        dx, dy = DIRS[d]
        hx, hy = self.snake[0]
        p = (hx + dx, hy + dy)
        return self._hits_wall(p) or p in self.snake[:-1]

    def get_state_bits(self):
        """3 danger facts + 4 food facts, all relative to the snake's heading."""
        hx, hy = self.snake[0]
        fx, fy = self.food if self.food else (hx, hy)
        dx, dy = fx - hx, fy - hy
        fwd_x, fwd_y = DIRS[self.direction]
        rgt_x, rgt_y = DIRS[(self.direction + 1) % 4]
        ahead = dx * fwd_x + dy * fwd_y       # >0: food is in front, <0: behind
        side = dx * rgt_x + dy * rgt_y        # >0: food is to the right, <0: to the left
        return [
            self._danger(0), self._danger(1), self._danger(2),
            ahead > 0, ahead < 0, side < 0, side > 0,
        ]

    def get_state(self):
        return encode_state(self.get_state_bits())

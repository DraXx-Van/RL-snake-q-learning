"""One play session: TRAIN (watch it learn), WATCH (trained agent), or HUMAN."""
import os
import shutil
import time
from collections import deque

import numpy as np
import pygame

import config
from agent import QAgent
from game import STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, DOWN, LEFT, RIGHT, SnakeEnv
from ui import panels, sound, theme
from ui.renderer import Renderer
from ui.theme import text

SPEEDS = [3, 5, 8, 12, 20, 35, 60, 120]     # game steps per second
SMOOTH_MAX_SPEED = 20                       # slide between cells only at watchable speeds
TURBO_BUDGET = 0.03                         # seconds of game steps per frame in turbo

KEY_DIRS = {
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
    pygame.K_UP: UP, pygame.K_w: UP,
}


class Session:
    def __init__(self, screen, mode, table_path=config.Q_TABLE_PATH):
        self.screen = screen
        self.mode = mode                        # 'train' | 'watch' | 'human'
        self.table_path = table_path
        self.renderer = Renderer()
        self.clock = pygame.time.Clock()
        self.env = SnakeEnv(seed=config.SEED if mode == "train" else None)

        self.agent = None
        self.toast = None
        self.random_agent = False
        if mode in ("train", "watch"):
            if QAgent.exists(table_path):
                self.agent = QAgent.load(table_path, seed=config.SEED if mode == "train" else None)
            else:
                self.agent = QAgent(seed=config.SEED if mode == "train" else None)
                if mode == "watch":
                    self.say("No trained table found - run Train first (or: python main.py train)", 6)
        self.loaded_episodes = self.agent.episodes if self.agent else 0

        self.speed_idx = {"train": 3, "watch": 3, "human": 2}[mode]
        self.turbo = False
        self.paused = False
        self.view = 0
        self.detail = False          # simple panel by default; D = detailed (for the viva)
        self.view_cache = {}

        self.scores = []
        self.best = 0
        self.brain = None
        self.last_update = None
        self.update_flash = 0.0
        self.state = self.env.reset()
        self.prev = None
        self.accum = 0.0
        self.dead = False
        self.over_timer = 0.0
        self.waiting = mode == "human"
        self.keyq = deque()

    # ------------------------------------------------------------ helpers
    def say(self, msg, seconds=2.5):
        self.toast = (msg, time.time() + seconds)

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]

    def save(self):
        if os.path.exists(self.table_path):
            shutil.copyfile(self.table_path, self.table_path.replace(".npz", ".prev.npz"))
        self.agent.save(self.table_path)
        self.loaded_episodes = self.agent.episodes
        self.say(f"Saved {self.table_path} ({self.agent.episodes} episodes)")

    def _new_episode(self):
        self.state = self.env.reset()
        self.prev = None
        self.dead = False
        self.over_timer = 0.0
        self.accum = 0.0
        self.keyq.clear()
        if self.mode == "human":
            self.waiting = True

    # --------------------------------------------------------------- game
    def _human_action(self):
        cur = self.env.direction
        while self.keyq:
            want = self.keyq.popleft()
            if want in (cur, (cur + 2) % 4):
                continue
            return TURN_RIGHT if want == (cur + 1) % 4 else TURN_LEFT
        return STRAIGHT

    def _tick(self):
        env = self.env
        explored = False
        if self.mode == "human":
            a = self._human_action()
        elif self.random_agent:
            a, explored = self.agent.rng.randrange(config.N_ACTIONS), True
        else:
            a, explored = self.agent.choose_action(self.state, greedy=self.mode != "train")

        self.prev = list(env.snake)
        r, done, _ = env.step(a)
        s2 = env.get_state()
        if self.mode == "train":
            self.last_update = self.agent.learn(self.state, a, r, s2, done)
            self.update_flash = 1.0
        if self.mode != "human":
            self.brain = (self.state, a, explored)
        self.state = s2
        if not self.turbo:
            if env.ate_food:
                sound.play("eat")
            if done and not env.won:
                sound.play("crash")
        if env.ate_food and not self.turbo:
            self.renderer.add_pop(env.snake[0])
        if done:
            self._episode_over()

    def _episode_over(self):
        self.scores.append(self.env.score)
        self.best = max(self.best, self.env.score)
        if self.mode == "train":
            self.agent.end_episode()
        if self.mode == "human":
            self.dead = True
        elif self.turbo:
            self._new_episode()
        else:
            self.dead = True
            self.over_timer = 0.9

    def update(self, dt):
        self.update_flash = max(0.0, self.update_flash - dt * 3)
        if self.paused or (self.waiting and self.mode == "human"):
            return
        if self.dead:
            if self.mode != "human":
                self.over_timer -= dt
                if self.over_timer <= 0:
                    self._new_episode()
            return
        if self.turbo and self.mode != "human":
            end = time.perf_counter() + TURBO_BUDGET
            while time.perf_counter() < end:
                self._tick()
            return
        interval = 1.0 / self.speed
        self.accum += dt
        n = 0
        while self.accum >= interval and n < 10 and not self.dead:
            self.accum -= interval
            self._tick()
            n += 1
        if n == 10:
            self.accum = 0.0

    # ------------------------------------------------------------- events
    def handle_key(self, key):
        if key in KEY_DIRS and self.mode == "human":
            d = KEY_DIRS[key]
            self.waiting = False
            if not self.keyq or self.keyq[-1] != d:
                if len(self.keyq) < 2:
                    self.keyq.append(d)
        elif key in (pygame.K_SPACE, pygame.K_RETURN):
            if self.mode == "human" and self.dead:
                self._new_episode()
            else:
                self.paused = not self.paused
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.speed_idx = min(len(SPEEDS) - 1, self.speed_idx + 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_idx = max(0, self.speed_idx - 1)
        elif key == pygame.K_x:
            sound.muted = not sound.muted
            self.say("Sound off" if sound.muted else "Sound on")
        elif key == pygame.K_f and self.mode != "human":
            self.turbo = not self.turbo
            self.accum = 0.0
        elif key == pygame.K_d and self.mode != "human":
            self.detail = not self.detail
        elif key == pygame.K_t and self.detail:
            self.view = (self.view + 1) % len(panels.VIEW_NAMES)
        elif key == pygame.K_r:
            self._new_episode()
        elif key == pygame.K_s and self.mode == "train":
            self.save()
        elif key == pygame.K_n and self.mode == "train":
            self.agent = QAgent(seed=config.SEED)
            self.loaded_episodes = 0
            self.scores, self.best, self.brain, self.last_update = [], 0, None, None
            self.view_cache = {}
            self._new_episode()
            self.say("Fresh empty Q-table - watch it learn from zero!")
        elif key == pygame.K_m and self.mode == "watch":
            self.random_agent = not self.random_agent
            self.scores, self.best, self.view_cache = [], 0, {}
            self._new_episode()
            self.say("Random snake (no brain)" if self.random_agent else "Trained agent")

    # --------------------------------------------------------------- draw
    def _chip(self):
        if self.mode == "train":
            return "TRAINING", theme.GOLD
        if self.mode == "human":
            return "HUMAN", theme.LIGHT
        return ("RANDOM SNAKE", theme.BAD) if self.random_agent else ("TRAINED AGENT", theme.GOOD)

    def draw(self):
        scr = self.screen
        scr.fill(theme.BG)

        if self.mode == "train":
            avg = np.mean(self.scores[-100:]) if self.scores else 0.0
            line1 = f"Episode {self.agent.episodes:,}   |   epsilon {self.agent.epsilon:.3f}"
            line2 = f"average score (last 100): {avg:.1f}"
        elif self.mode == "watch":
            avg = np.mean(self.scores) if self.scores else 0.0
            line1 = f"Games watched: {len(self.scores)}   |   average {avg:.1f}"
            line2 = "epsilon 0 - always picks the best known move" if not self.random_agent \
                else "every move is random"
        else:
            line1, line2 = f"Games played: {len(self.scores)}", "arrow keys / WASD to steer"
        chip, chip_col = self._chip()
        right = "TURBO" if self.turbo else (f"speed {self.speed}/s" + ("  (paused)" if self.paused else ""))
        self.renderer.draw_header(scr, self.env.score, self.best, line1, line2, chip, chip_col, right)

        smooth = self.speed <= SMOOTH_MAX_SPEED and not self.turbo and not self.paused and not self.dead
        t = min(1.0, self.accum * self.speed) if smooth else 1.0
        tint = min(1.0, self.over_timer / 0.9) if (self.dead and self.mode != "human") else \
            (1.0 if self.dead else 0.0)
        self.renderer.draw_board(scr, self.env, self.prev, t, tint)

        if self.waiting and self.mode == "human":
            self.renderer.draw_overlay(scr, [("Press an arrow key", 26, theme.TEXT),
                                             ("to start", 20, theme.MUTED)])
        elif self.dead and self.mode == "human":
            self.renderer.draw_overlay(scr, [("Game over", 30, theme.BAD),
                                             (f"Score {self.env.score}", 24, theme.TEXT),
                                             ("Space = play again", 18, theme.MUTED)])
        elif self.dead:
            why = "timed out" if self.env.timed_out else "crashed"
            self.renderer.draw_overlay(scr, [(f"Snake {why}", 24, theme.BAD),
                                             (f"Score {self.env.score}", 20, theme.TEXT)])

        px, pw = theme.PANEL_X, theme.PANEL_W
        if self.mode == "human":
            panels.draw_human_panel(scr, pygame.Rect(px, theme.PANEL_Y, pw, theme.BOARD_H), self.best)
        elif not self.detail:
            panels.draw_simple(scr, pygame.Rect(px, theme.PANEL_Y, pw, theme.BOARD_H), self.agent, self.brain,
                               self.last_update, self.mode == "train", self.update_flash, self.scores,
                               self.view_cache)
        else:
            panels.draw_brain(scr, pygame.Rect(px, theme.PANEL_Y, pw, 230), self.agent, self.brain)
            panels.draw_update(scr, pygame.Rect(px, theme.PANEL_Y + 240, pw, 102), self.last_update,
                               self.mode == "train", self.update_flash)
            panels.draw_view(scr, pygame.Rect(px, theme.PANEL_Y + 352, pw, theme.BOARD_H - 352),
                             self.view, self.scores, self.agent, self.state, self.view_cache)

        hints = {
            "train": "Space pause  |  +/- speed  |  F turbo  |  D detailed  |  X sound  |  S save  |  N new table  |  Esc menu",
            "watch": "Space pause  |  +/- speed  |  F turbo  |  D detailed  |  X sound  |  M random snake  |  Esc menu",
            "human": "Arrow keys / WASD steer  |  Space pause / restart  |  +/- speed  |  X sound  |  Esc menu",
        }[self.mode]
        text(scr, hints, (theme.WIN_W / 2, theme.WIN_H - 14), 13, theme.MUTED, anchor="center")
        if self.toast and time.time() < self.toast[1]:
            img = theme.font(16, True).render(self.toast[0], True, (25, 40, 20))
            rect = img.get_rect(center=(theme.BOARD_X + theme.BOARD_W / 2, theme.BOARD_Y + theme.BOARD_H - 28))
            pygame.draw.rect(scr, theme.GOLD, rect.inflate(28, 14), border_radius=14)
            scr.blit(img, rect)

    # ---------------------------------------------------------------- run
    def run(self):
        """Returns 'menu' or 'quit'."""
        while True:
            dt = min(0.1, self.clock.tick(60) / 1000.0)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._finish()
                    return "quit"
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self._finish()
                        return "menu"
                    self.handle_key(ev.key)
            self.update(dt)
            self.draw()
            pygame.display.flip()

    def _finish(self):
        if self.mode == "train" and self.agent.episodes > self.loaded_episodes:
            self.save()

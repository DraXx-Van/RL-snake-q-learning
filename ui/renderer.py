"""Draws the Google-style board: checkerboard, rounded blue snake, apple."""
import math
import time

import pygame

import config
from ui import theme
from ui.theme import BLOCK, BOARD_H, BOARD_W, BOARD_X, BOARD_Y, mix, text


class Renderer:
    def __init__(self):
        self.checker = self._make_checker()
        self.apple = self._make_apple(BLOCK)
        self.apple_small = self._make_apple(34)
        self.pops = []   # (cell, start_time) for the "ate an apple" ring

    # --------------------------------------------------------- static art
    def _make_checker(self):
        surf = pygame.Surface((BOARD_W, BOARD_H))
        for gy in range(config.GRID_H):
            for gx in range(config.GRID_W):
                color = theme.LIGHT if (gx + gy) % 2 == 0 else theme.DARK
                pygame.draw.rect(surf, color, (gx * BLOCK, gy * BLOCK, BLOCK, BLOCK))
        return surf

    @staticmethod
    def _make_apple(size):
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size / 2
        r = size * 0.34
        pygame.draw.ellipse(surf, (0, 0, 0, 55), (c - r, c + r * 0.55, r * 2, r * 0.7))   # shadow
        pygame.draw.circle(surf, theme.APPLE, (c, c + size * 0.03), r)
        pygame.draw.circle(surf, (255, 150, 120), (c - r * 0.4, c - r * 0.25), r * 0.22)    # shine
        pygame.draw.line(surf, (110, 70, 30), (c, c - r * 0.8), (c + size * 0.04, c - r * 1.25), 2)
        pygame.draw.ellipse(surf, theme.LEAF, (c + size * 0.02, c - r * 1.45, r * 0.9, r * 0.5))
        return surf

    def add_pop(self, cell):
        self.pops.append((cell, time.time()))

    # -------------------------------------------------------------- board
    def draw_board(self, surf, env, prev, t, tint=0.0):
        frame = pygame.Rect(BOARD_X, BOARD_Y, BOARD_W, BOARD_H).inflate(16, 16)
        pygame.draw.rect(surf, theme.FRAME, frame, border_radius=12)
        surf.blit(self.checker, (BOARD_X, BOARD_Y))

        if env.food is not None:
            fx, fy = env.food
            bob = math.sin(time.time() * 4) * 1.5
            surf.blit(self.apple, (BOARD_X + fx * BLOCK, BOARD_Y + fy * BLOCK + bob))
        self._draw_pops(surf)
        self._draw_snake(surf, env, prev, t, tint)

    def _draw_pops(self, surf):
        now = time.time()
        self.pops = [p for p in self.pops if now - p[1] < 0.3]
        for (cx, cy), t0 in self.pops:
            age = (now - t0) / 0.3
            size = BLOCK * 3
            layer = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(layer, (255, 255, 255, int(200 * (1 - age))),
                               (size / 2, size / 2), BLOCK * (0.3 + age * 1.0), 3)
            surf.blit(layer, (BOARD_X + (cx + 0.5) * BLOCK - size / 2,
                              BOARD_Y + (cy + 0.5) * BLOCK - size / 2))

    def _draw_snake(self, surf, env, prev, t, tint):
        snake = env.snake
        # While a move animates only the head slides forward and the tail slides in; every other body
        # point stays exactly on its cell centre, so a turn keeps a clean square corner (no diagonal cut).
        if prev and t < 1.0 and len(prev) > 0:
            def lerp(a, b):
                return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            cells = [lerp(prev[0], snake[0])] + list(snake[1:]) + [lerp(prev[-1], snake[-1])]
        else:
            cells = list(snake)
        pts = [(BOARD_X + (gx + 0.5) * BLOCK, BOARD_Y + (gy + 0.5) * BLOCK) for gx, gy in cells]

        body = mix(theme.SNAKE, (214, 48, 49), tint)
        head_col = mix(theme.SNAKE_DARK, (190, 30, 30), tint)
        width = int(BLOCK * 0.74)
        radius = width // 2

        for i in range(len(pts) - 1, 0, -1):
            pygame.draw.line(surf, body, pts[i], pts[i - 1], width)
        for p in pts[1:]:
            pygame.draw.circle(surf, body, p, radius)
        hx, hy = pts[0]
        pygame.draw.circle(surf, head_col, (hx, hy), radius * 1.12)

        from game import DIRS
        dx, dy = DIRS[env.direction]
        px, py = -dy, dx
        for side in (-1, 1):
            ex = hx + dx * BLOCK * 0.12 + px * side * BLOCK * 0.17
            ey = hy + dy * BLOCK * 0.12 + py * side * BLOCK * 0.17
            pygame.draw.circle(surf, (255, 255, 255), (ex, ey), BLOCK * 0.13)
            pygame.draw.circle(surf, (20, 20, 30), (ex + dx * BLOCK * 0.04, ey + dy * BLOCK * 0.04),
                               BLOCK * 0.07)

    # ------------------------------------------------------------- header
    def draw_header(self, surf, score, best, line1, line2, chip, chip_color, right_text):
        pygame.draw.rect(surf, theme.HEADER, (0, 0, theme.WIN_W, theme.HEADER_H))
        surf.blit(self.apple_small, (22, 14))
        text(surf, score, (66, 32), 28, bold=True, anchor="midleft")
        text(surf, f"Best  {best}", (140, 32), 20, theme.GOLD, True, anchor="midleft")
        text(surf, line1, (290, 20), 16, anchor="midleft")
        text(surf, line2, (290, 44), 14, theme.MUTED, anchor="midleft")

        chip_font = theme.font(15, True)
        img = chip_font.render(chip, True, (25, 40, 20))
        rect = img.get_rect(midright=(theme.WIN_W - 36, 22))
        pygame.draw.rect(surf, chip_color, rect.inflate(24, 10), border_radius=14)
        surf.blit(img, rect)
        text(surf, right_text, (theme.WIN_W - 24, 46), 14, theme.TEXT, anchor="midright")

    def draw_overlay(self, surf, lines):
        """Centered translucent box on the board. lines: [(text, size, color)]"""
        h = sum(s + 12 for _, s, _ in lines) + 28
        w = 360
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(box, (20, 40, 16, 215), (0, 0, w, h), border_radius=16)
        y = 18
        for s, size, color in lines:
            text(box, s, (w / 2, y), size, color, True, anchor="midtop")
            y += size + 12
        surf.blit(box, box.get_rect(center=(BOARD_X + BOARD_W / 2, BOARD_Y + BOARD_H / 2)))

"""Colours, sizes and font helpers (Google-Snake inspired palette)."""
import pygame

import config

# ---- layout ------------------------------------------------------------
BLOCK = 40
WIN_W, WIN_H = 1200, 720
HEADER_H = 64
BOARD_X, BOARD_Y = 24, 88
BOARD_W = config.GRID_W * BLOCK
BOARD_H = config.GRID_H * BLOCK
PANEL_X = BOARD_X + BOARD_W + 32
PANEL_W = WIN_W - PANEL_X - 24
PANEL_Y = BOARD_Y

# ---- colours -----------------------------------------------------------
BG = (36, 62, 30)
HEADER = (74, 117, 44)
FRAME = (87, 138, 52)
LIGHT = (170, 215, 81)
DARK = (162, 209, 73)
SNAKE = (78, 124, 246)
SNAKE_DARK = (58, 100, 226)
APPLE = (231, 71, 29)
LEAF = (60, 140, 40)

CARD = (48, 80, 38)
CARD_EDGE = (66, 104, 50)
TEXT = (240, 246, 232)
MUTED = (160, 190, 140)
GOOD = (120, 220, 90)
BAD = (240, 100, 80)
GOLD = (255, 210, 80)
CHIP_OFF = (40, 66, 32)
CHIP_ON = (255, 210, 80)
HEAT_ZERO = (58, 84, 52)


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont("segoeui,arial,helvetica", size, bold=bold)
    return _fonts[key]


def text(surf, s, pos, size=16, color=TEXT, bold=False, anchor="topleft"):
    img = font(size, bold).render(str(s), True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect

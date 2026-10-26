"""Start menu."""
import numpy as np
import pygame

import config
from agent import QAgent
from ui import theme
from ui.renderer import Renderer
from ui.theme import text

BUTTONS = [
    ("train", "1   Train the agent  (watch it learn)"),
    ("watch", "2   Watch the trained agent"),
    ("human", "3   Play it yourself"),
    ("quit", "Q   Quit"),
]


def run_menu(screen):
    """Returns 'train' | 'watch' | 'human' | 'quit'."""
    renderer = Renderer()
    clock = pygame.time.Clock()
    if QAgent.exists():
        a = QAgent.load()
        status = f"Saved Q-table found: {a.episodes} episodes, {int((a.visits > 0).sum())} situations learned"
        status_col = theme.GOOD
    else:
        status = "No saved Q-table yet - choose Train first"
        status_col = theme.GOLD

    # decorative checkerboard backdrop
    back = pygame.Surface((theme.WIN_W, theme.WIN_H))
    size = 60
    for gy in range(theme.WIN_H // size + 1):
        for gx in range(theme.WIN_W // size + 1):
            col = theme.mix(theme.LIGHT if (gx + gy) % 2 == 0 else theme.DARK, theme.BG, 0.72)
            pygame.draw.rect(back, col, (gx * size, gy * size, size, size))

    card_rect = pygame.Rect(0, 0, 620, 520)
    card_rect.center = (theme.WIN_W // 2, theme.WIN_H // 2)
    rects = []
    for i in range(len(BUTTONS)):
        r = pygame.Rect(0, 0, 480, 56)
        r.midtop = (theme.WIN_W // 2, card_rect.y + 190 + i * 70)
        rects.append(r)

    while True:
        clock.tick(60)
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.KEYDOWN:
                keymap = {pygame.K_1: "train", pygame.K_2: "watch", pygame.K_3: "human",
                          pygame.K_q: "quit", pygame.K_ESCAPE: "quit"}
                if ev.key in keymap:
                    return keymap[ev.key]
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for (key, _), r in zip(BUTTONS, rects):
                    if r.collidepoint(ev.pos):
                        return key

        screen.blit(back, (0, 0))
        pygame.draw.rect(screen, theme.CARD, card_rect, border_radius=24)
        pygame.draw.rect(screen, theme.CARD_EDGE, card_rect, 3, border_radius=24)
        screen.blit(renderer.apple, (card_rect.x + 36, card_rect.y + 30))
        text(screen, "Snake  x  Q-Learning", (card_rect.centerx, card_rect.y + 56), 40, theme.TEXT, True,
             anchor="center")
        text(screen, "A snake that teaches itself to play", (card_rect.centerx, card_rect.y + 100), 18,
             theme.MUTED, anchor="center")
        text(screen, status, (card_rect.centerx, card_rect.y + 148), 15, status_col, anchor="center")

        for (key, label), r in zip(BUTTONS, rects):
            hover = r.collidepoint(mouse)
            col = theme.SNAKE if hover else theme.CARD_EDGE
            if key == "quit" and not hover:
                col = theme.CHIP_OFF
            pygame.draw.rect(screen, col, r, border_radius=16)
            text(screen, label, (r.x + 28, r.centery), 22, theme.TEXT, True, anchor="midleft")
        text(screen, f"Grid {config.GRID_W}x{config.GRID_H}  |  {config.N_STATES} situations x 3 moves  |  alpha {config.ALPHA}"
                     f"  gamma {config.GAMMA}", (card_rect.centerx, card_rect.bottom - 26), 14, theme.MUTED,
             anchor="center")
        pygame.display.flip()

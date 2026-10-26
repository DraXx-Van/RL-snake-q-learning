"""Right-hand dashboard: agent brain, live Q-update, and the chart/heatmap/rules views."""
import numpy as np
import pygame

import config
from game import ACTION_NAMES, decode_state, describe_state
from ui import theme
from ui.theme import text

VIEW_NAMES = ["Score chart", "Q-table heatmap", "Learned rules"]


def card(surf, rect, title=None, glow=0.0):
    pygame.draw.rect(surf, theme.CARD, rect, border_radius=14)
    edge = theme.mix(theme.CARD_EDGE, theme.GOLD, glow)
    pygame.draw.rect(surf, edge, rect, width=2, border_radius=14)
    if title:
        text(surf, title, (rect.x + 16, rect.y + 10), 15, theme.MUTED, True)


def _chip(surf, x, y, w, h, label, lit):
    color = theme.CHIP_ON if lit else theme.CHIP_OFF
    pygame.draw.rect(surf, color, (x, y, w, h), border_radius=8)
    text(surf, label, (x + w / 2, y + h / 2), 13, (40, 30, 5) if lit else theme.MUTED,
         lit, anchor="center")


# ------------------------------------------------------------------ brain
def draw_brain(surf, rect, agent, brain):
    """brain = (state, action, explored) or None."""
    card(surf, rect, "AGENT BRAIN")
    if brain is None:
        text(surf, "waiting for the first move...", (rect.x + 16, rect.y + 44), 15, theme.MUTED)
        return
    s, action, explored = brain
    bits = decode_state(s)
    text(surf, f"State #{s}", (rect.x + 140, rect.y + 10), 15, theme.TEXT, True)
    tag, tag_col = (("EXPLORING (random move)", theme.GOLD) if explored
                    else ("EXPLOITING (best known)", theme.GOOD))
    text(surf, tag, (rect.right - 16, rect.y + 10), 14, tag_col, True, anchor="topright")

    rows = [
        ("Danger", ["Ahead", "Right", "Left"], bits[0:3], 84),
        ("Apple", ["Ahead", "Behind", "Left", "Right"], bits[3:7], 70),
    ]
    y = rect.y + 40
    for name, labels, vals, w in rows:
        text(surf, name, (rect.x + 16, y + 12), 14, theme.TEXT, anchor="midleft")
        for i, (lab, v) in enumerate(zip(labels, vals)):
            _chip(surf, rect.x + 100 + i * (w + 8), y, w, 24, lab, v)
        y += 32

    text(surf, "Q-values for this state", (rect.x + 16, y + 6), 14, theme.MUTED, True)
    y += 30
    row = agent.q[s]
    scale = max(5.0, float(np.abs(row).max()))
    bar_x0 = rect.x + 120
    bar_w = rect.w - 120 - 90
    zero_x = bar_x0 + bar_w / 2
    for i in range(config.N_ACTIONS):
        yy = y + i * 30
        if i == action:
            pygame.draw.rect(surf, theme.GOLD, (rect.x + 8, yy - 3, rect.w - 16, 28), 2, border_radius=8)
        text(surf, ACTION_NAMES[i], (rect.x + 18, yy + 11), 14, theme.TEXT, anchor="midleft")
        length = abs(row[i]) / scale * (bar_w / 2)
        color = theme.GOOD if row[i] >= 0 else theme.BAD
        if row[i] >= 0:
            pygame.draw.rect(surf, color, (zero_x, yy + 2, max(2, length), 18), border_radius=4)
        else:
            pygame.draw.rect(surf, color, (zero_x - length, yy + 2, max(2, length), 18), border_radius=4)
        text(surf, f"{row[i]:+.2f}", (rect.right - 16, yy + 11), 14, theme.TEXT, True, anchor="midright")
    pygame.draw.line(surf, theme.MUTED, (zero_x, y), (zero_x, y + 88), 1)


# ----------------------------------------------------------------- update
def draw_update(surf, rect, info, learning_on, flash):
    card(surf, rect, "LAST Q-TABLE UPDATE", glow=flash)
    x = rect.x + 16
    if not learning_on:
        text(surf, "Learning is OFF - the agent only uses what it already learned.",
             (x, rect.y + 40), 14, theme.MUTED)
        return
    if info is None:
        text(surf, "no update yet", (x, rect.y + 40), 14, theme.MUTED)
        return
    delta = info.new - info.old
    col = theme.GOOD if delta >= 0 else theme.BAD
    text(surf, f"Q[{info.state}][{ACTION_NAMES[info.action]}]   {info.old:+.2f}  ->  {info.new:+.2f}"
               f"   ({delta:+.2f})", (x, rect.y + 32), 16, col, True)
    if info.done:
        tgt = f"target = r = {info.reward:+.0f}   (game over, no future)"
    else:
        tgt = (f"target = r + {config.GAMMA}*maxQ(s') = {info.reward:+.0f} + {config.GAMMA} * "
               f"{info.future:.2f} = {info.target:.2f}")
    text(surf, tgt, (x, rect.y + 54), 13, theme.TEXT)
    text(surf, f"new = old + {config.ALPHA} * (target - old) = {info.old:.2f} + {config.ALPHA} * "
               f"({info.target:.2f} - {info.old:.2f})", (x, rect.y + 72), 13, theme.MUTED)


# ------------------------------------------------------------------- views
def draw_view(surf, rect, view, scores, agent, current_state, cache):
    card(surf, rect)
    x = rect.x + 16
    for i, name in enumerate(VIEW_NAMES):
        on = i == view
        w = theme.font(14, on).size(name)[0]
        text(surf, name, (x, rect.y + 10), 14, theme.GOLD if on else theme.MUTED, on)
        x += w + 22
    text(surf, "[T] switch", (rect.right - 16, rect.y + 10), 13, theme.MUTED, anchor="topright")
    inner = pygame.Rect(rect.x + 14, rect.y + 38, rect.w - 28, rect.h - 50)
    if view == 0:
        _chart(surf, inner, scores, cache)
    elif view == 1:
        _heatmap(surf, inner, agent, current_state)
    else:
        _rules(surf, inner, agent)


def _bin(values, width):
    values = np.asarray(values, dtype=float)
    if len(values) <= width:
        return values
    return np.array([c.mean() for c in np.array_split(values, width)])


def _chart(surf, inner, scores, cache):
    if len(scores) < 2:
        text(surf, "Waiting for finished games...", inner.center, 15, theme.MUTED, anchor="center")
        return
    if cache.get("n") != len(scores):
        arr = np.asarray(scores, dtype=float)
        win = min(100, len(arr))
        csum = np.cumsum(np.insert(arr, 0, 0.0))
        ma = np.concatenate([np.cumsum(arr[:win - 1]) / np.arange(1, win), (csum[win:] - csum[:-win]) / win])
        w = inner.w - 40
        cache.update(n=len(scores), raw=_bin(arr, w), ma=_bin(ma, w), top=max(5.0, float(arr.max())))
    plot = pygame.Rect(inner.x + 40, inner.y + 6, inner.w - 40, inner.h - 30)
    top = cache["top"]
    for frac in (0, 0.5, 1):
        yy = plot.bottom - frac * plot.h
        pygame.draw.line(surf, theme.CARD_EDGE, (plot.x, yy), (plot.right, yy), 1)
        text(surf, f"{top * frac:.0f}", (plot.x - 6, yy), 12, theme.MUTED, anchor="midright")

    def pts(series):
        n = len(series)
        return [(plot.x + (i / max(1, n - 1)) * plot.w, plot.bottom - (v / top) * plot.h)
                for i, v in enumerate(series)]

    if len(cache["raw"]) > 1:
        pygame.draw.lines(surf, (110, 150, 200), False, pts(cache["raw"]), 1)
        pygame.draw.lines(surf, theme.GOLD, False, pts(cache["ma"]), 3)
    text(surf, f"{len(scores)} games   |   thin = score per game, gold = average of last 100",
         (plot.x, plot.bottom + 6), 12, theme.MUTED)


HEAT_COLS = 16
HEAT_ROWS = config.N_STATES // HEAT_COLS


def _heat_surface(col, scale):
    v = np.clip(col / scale, -1, 1).reshape(HEAT_ROWS, HEAT_COLS).T      # [x, y]
    pos = np.clip(v, 0, 1)[..., None]
    neg = np.clip(-v, 0, 1)[..., None]
    base = np.array(theme.HEAT_ZERO, dtype=float)
    rgb = base * (1 - pos - neg) + np.array(theme.GOOD) * pos + np.array(theme.BAD) * neg
    return pygame.surfarray.make_surface(rgb.astype(np.uint8))


def _heatmap(surf, inner, agent, current_state):
    scale = max(5.0, float(np.abs(agent.q).max()))
    gw, gh, gap = 150, 96, 22
    x0 = inner.x + (inner.w - (3 * gw + 2 * gap)) // 2
    y0 = inner.y + 22
    for a in range(config.N_ACTIONS):
        gx = x0 + a * (gw + gap)
        text(surf, ACTION_NAMES[a], (gx + gw / 2, y0 - 6), 13, theme.TEXT, True, anchor="midbottom")
        img = pygame.transform.scale(_heat_surface(agent.q[:, a], scale), (gw, gh))
        surf.blit(img, (gx, y0))
        pygame.draw.rect(surf, theme.CARD_EDGE, (gx, y0, gw, gh), 1)
        if current_state is not None:
            cx = gx + (current_state % HEAT_COLS) * gw / HEAT_COLS
            cy = y0 + (current_state // HEAT_COLS) * gh / HEAT_ROWS
            pygame.draw.rect(surf, (255, 255, 255), (cx - 1, cy - 1, gw / HEAT_COLS + 2, gh / HEAT_ROWS + 2), 2)
    seen = int((agent.visits > 0).sum())
    text(surf, f"each cell = one (state, action) Q-value  |  green = good, red = bad, grey = unlearned "
               f"(scale +/-{scale:.0f})", (inner.x, inner.bottom - 32), 12, theme.MUTED)
    text(surf, f"states visited so far: {seen} / {config.N_STATES}   |   white box = current state",
         (inner.x, inner.bottom - 16), 12, theme.MUTED)


def _rules(surf, inner, agent):
    order = np.argsort(agent.visits)[::-1][:6]
    y = inner.y + 2
    if agent.visits[order[0]] == 0:
        text(surf, "Nothing learned yet.", (inner.x, y), 15, theme.MUTED)
        return
    text(surf, "Most-visited situations and what the snake does there", (inner.x, y), 12, theme.MUTED)
    y += 22
    for s in order:
        if agent.visits[s] == 0:
            break
        dangers, food = describe_state(int(s))
        best = int(np.argmax(agent.q[s]))
        desc = f"danger: {', '.join(dangers) or 'none'}  |  apple: {', '.join(food) or '-'}"
        text(surf, desc, (inner.x, y), 13, theme.TEXT)
        text(surf, f"{ACTION_NAMES[best]} ({agent.q[s, best]:+.1f})", (inner.right, y), 13,
             theme.GOOD if agent.q[s, best] >= 0 else theme.BAD, True, anchor="topright")
        y += 24


# ------------------------------------------------------------ simple view
# Same column order as the college slide: Left | Right | Forward  (action index in brackets)
SIMPLE_COLS = [("Left", 2), ("Right", 1), ("Forward", 0)]
# (label, danger[ahead,right,left], apple[ahead,behind,left,right])
SIMPLE_ROWS = [
    ("Apple ahead, path clear", (0, 0, 0), (1, 0, 0, 0)),
    ("Wall ahead, apple on left", (1, 0, 0), (0, 0, 1, 0)),
    ("Wall ahead, apple on right", (1, 0, 0), (0, 0, 0, 1)),
    ("Danger on right, apple right", (0, 1, 0), (0, 0, 0, 1)),
]
# the live app shows a few more situations than the 4 used in the explainer video
APP_ROWS = SIMPLE_ROWS + [
    ("Apple on left, path clear", (0, 0, 0), (0, 0, 1, 0)),
    ("Apple on right, path clear", (0, 0, 0), (0, 0, 0, 1)),
    ("Wall ahead and right, apple ahead", (1, 1, 0), (1, 0, 0, 0)),
]


def _simple_rows():
    from inspect_q import scenario_state
    return [(label, scenario_state(d, f)) for label, d, f in APP_ROWS]


def draw_simple(surf, rect, agent, brain, info, learning_on, flash, scores, cache):
    x, w = rect.x, rect.w

    # ---- 1. what the snake sees (plain English)
    a = pygame.Rect(x, rect.y, w, 124)
    card(surf, a, "WHAT THE SNAKE SEES")
    if brain is None:
        text(surf, "waiting for the first move...", (a.x + 16, a.y + 40), 15, theme.MUTED)
    else:
        s, action, explored = brain
        dangers, food = describe_state(s)
        rows = [("Danger", ", ".join(dangers) + " blocked" if dangers else "none nearby",
                 theme.BAD if dangers else theme.GOOD),
                ("Apple is", ", ".join(food) if food else "-", theme.GOLD)]
        for i, (lab, val, col) in enumerate(rows):
            text(surf, lab, (a.x + 16, a.y + 36 + i * 26), 16, theme.MUTED)
            text(surf, val, (a.x + 110, a.y + 36 + i * 26), 17, col, True)
        how = "random try" if explored else "best known move"
        text(surf, f"Decision:  {ACTION_NAMES[action]}   ({how})", (a.x + 16, a.y + 94), 16,
             theme.GOLD if explored else theme.GOOD, True)

    # ---- 2. the Q-table (same look as the slide)
    b = pygame.Rect(x, rect.y + 134, w, 337)
    card(surf, b, "Q-TABLE  -  higher number = better move")
    label_w, col_w = 226, 78
    hx = b.x + 16 + label_w
    text(surf, "Situation", (b.x + 16, b.y + 44), 14, theme.MUTED, True, anchor="midleft")
    for i, (name, _) in enumerate(SIMPLE_COLS):
        text(surf, name, (hx + i * col_w + col_w / 2, b.y + 44), 14, theme.MUTED, True, anchor="center")

    table = [("RIGHT NOW", brain[0] if brain else None)] + _simple_rows()
    shown = [agent.q[s] for _, s in table if s is not None]
    scale = max(8.0, float(np.abs(shown).max()) if shown else 8.0)
    for r, (label, s) in enumerate(table):
        y = b.y + 62 + r * 34
        now = r == 0
        if now:
            pygame.draw.rect(surf, theme.CARD_EDGE, (b.x + 8, y - 2, b.w - 16, 29), border_radius=8)
        text(surf, label, (b.x + 16, y + 12), 14 if not now else 15,
             theme.GOLD if now else theme.TEXT, now, anchor="midleft")
        if s is None:
            continue
        row = agent.q[s]
        best = int(np.argmax(row))
        for i, (_, act) in enumerate(SIMPLE_COLS):
            cell = pygame.Rect(hx + i * col_w + 3, y, col_w - 6, 25)
            v = row[act]
            tint = theme.GOOD if v >= 0 else theme.BAD
            pygame.draw.rect(surf, theme.mix(theme.CARD, tint, min(1.0, abs(v) / scale) * 0.75),
                             cell, border_radius=6)
            is_best = act == best and row.max() > row.min()      # no "best" while all scores are tied
            if is_best:
                pygame.draw.rect(surf, theme.GOLD, cell, 2, border_radius=6)
            text(surf, f"{v:+.1f}", cell.center, 14, theme.TEXT, is_best, anchor="center")

    # ---- 3. learning in one sentence
    c = pygame.Rect(x, rect.y + 481, w, 90)
    card(surf, c, "LEARNING", glow=flash)
    if not learning_on:
        text(surf, "Watching only - the snake uses what it already learned.", (c.x + 16, c.y + 40), 14,
             theme.MUTED)
    elif info is None:
        text(surf, "no update yet", (c.x + 16, c.y + 40), 14, theme.MUTED)
    else:
        if info.done:
            what = "Crashed (-10)"
        elif info.reward > 0:
            what = "Ate an apple (+10)"
        else:
            what = "Normal move"
        delta = info.new - info.old
        col = theme.TEXT if abs(delta) < 0.05 else (theme.GOOD if delta > 0 else theme.BAD)
        text(surf, f"{what}: score of '{ACTION_NAMES[info.action]}' in that situation", (c.x + 16, c.y + 36),
             14, theme.TEXT)
        text(surf, f"{info.old:+.1f}  ->  {info.new:+.1f}", (c.x + 16, c.y + 56), 17, col, True)
        text(surf, f"Random tries: {agent.epsilon * 100:.0f}% of moves", (c.right - 16, c.y + 60), 14,
             theme.GOLD, True, anchor="midright")



# ------------------------------------------------------------- human mode
def draw_human_panel(surf, rect, best):
    card(surf, rect, "YOU ARE THE AGENT")
    lines = [
        ("Steer with the arrow keys or WASD.", theme.TEXT),
        ("Eat apples, avoid walls and your own tail.", theme.TEXT),
        ("", theme.TEXT),
        ("The AI sees only 7 yes/no facts:", theme.MUTED),
        ("  danger ahead / right / left", theme.MUTED),
        ("  apple ahead / behind / left / right", theme.MUTED),
        ("  (all from the snake's own point of view)", theme.MUTED),
        ("", theme.TEXT),
        ("Think you can beat it? Watch it from the menu,", theme.TEXT),
        ("then come back and try!", theme.TEXT),
        ("", theme.TEXT),
        (f"Your best this session: {best}", theme.GOLD),
    ]
    y = rect.y + 44
    for line, color in lines:
        text(surf, line, (rect.x + 20, y), 16, color)
        y += 24

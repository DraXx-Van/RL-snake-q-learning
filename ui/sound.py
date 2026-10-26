"""Tiny game sounds, synthesized in code (no audio files): apple eaten and crash.

If there is no audio device, everything silently does nothing.
"""
import time

import numpy as np
import pygame

muted = False
_sounds = {}
_ready = None            # None = not tried yet
_last = {}


def _eat(sr):
    n = int(sr * 0.11)
    t = np.arange(n) / sr
    f = 300 + 900 * (t / t[-1])                               # quick rising "pip"
    wave = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-10 * t) * np.minimum(1, t / 0.004)
    click = np.sin(2 * np.pi * 2000 * t) * np.exp(-120 * t) * 0.2
    return wave * 0.8 + click


def _crash(sr):
    n = int(sr * 0.3)
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    x = rng.normal(size=n)
    cutoff = 2500 * (1 - t / t[-1]) + 200                     # low-passed noise that gets duller
    a = 1 - np.exp(-2 * np.pi * cutoff / sr)
    y, acc = np.zeros(n), 0.0
    for i in range(n):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    y = y / (np.abs(y).max() + 1e-9) * np.exp(-9 * t)
    thud = np.sin(2 * np.pi * 90 * t) * np.exp(-12 * t)
    return 0.6 * y + 0.7 * thud


def _to_sound(wave, ch, volume):
    pcm = (np.clip(wave / (np.abs(wave).max() + 1e-9), -1, 1) * 32767 * volume).astype(np.int16)
    if ch == 2:
        pcm = np.repeat(pcm, 2)                               # duplicate to interleaved stereo
    return pygame.mixer.Sound(buffer=pcm.tobytes())


def init():
    """Prepare the sounds once. Returns True if sound is available."""
    global _ready
    if _ready is not None:
        return _ready
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(44100, -16, 2)
        sr, _, ch = pygame.mixer.get_init()
        _sounds["eat"] = _to_sound(_eat(sr), ch, 0.55)
        _sounds["crash"] = _to_sound(_crash(sr), ch, 0.6)
        _ready = True
    except Exception:
        _ready = False
    return _ready


def play(name, min_gap=0.06):
    if muted or not init():
        return
    now = time.time()
    if now - _last.get(name, 0.0) < min_gap:                  # never machine-gun at very high speeds
        return
    _last[name] = now
    _sounds[name].play()

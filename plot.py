"""Training graphs saved as PNG (for the report / slides)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        window = max(1, len(values))
    csum = np.cumsum(np.insert(values, 0, 0.0))
    ma = (csum[window:] - csum[:-window]) / window
    return np.concatenate([np.full(window - 1, np.nan), ma])


def save_training_plots(history, outdir=config.RESULTS_DIR, tag=""):
    os.makedirs(outdir, exist_ok=True)
    scores = history["scores"]
    x = np.arange(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, scores, color="#9ec5fe", linewidth=0.6, label="score per episode")
    ax.plot(x, moving_average(scores), color="#1a56db", linewidth=2.2, label="moving average (100)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (apples eaten)")
    ax.set_title("Q-learning Snake: learning curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(outdir, f"training_curve{tag}.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x, history["epsilon"], color="#e8710a", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration rate (epsilon) over training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(outdir, f"epsilon{tag}.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2

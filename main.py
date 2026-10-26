"""Entry point.

    python main.py                 open the app (menu)
    python main.py train           headless training (fast) + graphs
    python main.py evaluate        trained agent vs random snake
    python main.py inspect         look inside the Q-table
    python main.py watch|human     jump straight into a mode
"""
import argparse
import os

import config


def run_ui(mode=None):
    import pygame
    from ui import theme
    from ui.menu import run_menu
    from ui.session import Session

    pygame.init()
    pygame.display.set_caption("Snake x Q-Learning")
    screen = pygame.display.set_mode((theme.WIN_W, theme.WIN_H))
    while True:
        choice = mode or run_menu(screen)
        if choice == "quit":
            break
        if Session(screen, choice).run() == "quit" or mode:
            break
    pygame.quit()


def cmd_train(args):
    from plot import save_training_plots
    from train import train
    _, history = train(episodes=args.episodes, seed=args.seed)
    p1, p2 = save_training_plots(history)
    print(f"Graphs saved: {p1}, {p2}")


def cmd_evaluate(args):
    from evaluate import compare
    _, text = compare(games=args.games)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "eval_stats.txt"), "w") as f:
        f.write(text + "\n")


def main():
    p = argparse.ArgumentParser(description="Snake x Q-Learning")
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="headless training")
    t.add_argument("--episodes", type=int, default=config.EPISODES)
    t.add_argument("--seed", type=int, default=config.SEED)
    e = sub.add_parser("evaluate", help="trained vs random")
    e.add_argument("--games", type=int, default=100)
    sub.add_parser("inspect", help="inspect the Q-table")
    sub.add_parser("watch", help="watch the trained agent")
    sub.add_parser("human", help="play yourself")
    args = p.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "inspect":
        from inspect_q import inspect
        inspect()
    elif args.cmd in ("watch", "human"):
        run_ui(args.cmd)
    else:
        run_ui()


if __name__ == "__main__":
    main()

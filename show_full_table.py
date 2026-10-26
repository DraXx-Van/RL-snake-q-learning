"""Writes the FULL Q-table (all 128 situations) as an HTML page and a CSV file.

    python show_full_table.py
Uses the deterministic 3000-episode training run (seed 42), the same table the video shows,
so it does not depend on / overwrite q_table.npz.
"""
import csv
import os

import numpy as np

import config
from game import describe_state, decode_state
from train import train

OUT = config.RESULTS_DIR
SCALE = 13.0


def cell_style(v):
    q = max(-1.0, min(1.0, v / SCALE))
    base, tgt = (58, 84, 52), ((120, 220, 90) if q >= 0 else (240, 100, 80))
    c = [round(b + (t - b) * abs(q)) for b, t in zip(base, tgt)]
    return f"background:rgb({c[0]},{c[1]},{c[2]})"


def main():
    agent, _ = train(episodes=config.EPISODES, seed=config.SEED, save_path=None, verbose=False)
    os.makedirs(OUT, exist_ok=True)
    cols = ["Left", "Right", "Forward"]
    idx = [2, 1, 0]                                   # table columns -> action index (same order as the slide)
    rows = []
    for s in range(config.N_STATES):
        dangers, food = describe_state(s)
        bits = "".join(str(b) for b in decode_state(s))
        vals = [float(agent.q[s, a]) for a in idx]
        seen = int(agent.visits[s])
        best = cols[int(np.argmax(vals))] if seen else "-"
        rows.append(dict(s=s, bits=bits, danger=", ".join(dangers) or "none", apple=", ".join(food) or "-",
                         vals=vals, seen=seen, best=best))

    with open(os.path.join(OUT, "full_q_table.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["situation", "answers(danger ahead,right,left | apple ahead,behind,left,right)",
                    "danger", "apple", "Left", "Right", "Forward", "best_move", "times_visited"])
        for r in rows:
            w.writerow([r["s"], r["bits"], r["danger"], r["apple"], *[f"{v:.2f}" for v in r["vals"]], r["best"], r["seen"]])

    body = []
    for r in rows:
        cls = "seen" if r["seen"] else "never"
        cells = "".join(f'<td class="num" style="{cell_style(v)}">{v:+.1f}</td>' if r["seen"] else '<td class="num">0.0</td>'
                        for v in r["vals"])
        best = f'<b>{r["best"]}</b>' if r["seen"] else "never happens"
        body.append(f'<tr class="{cls}"><td>{r["s"]}</td><td class="mono">{r["bits"]}</td><td>{r["danger"]}</td>'
                    f'<td>{r["apple"]}</td>{cells}<td>{best}</td><td class="num">{r["seen"]:,}</td></tr>')
    seen_n = sum(1 for r in rows if r["seen"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Full Q-table</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#243e1e;color:#f0f6e8;margin:24px}}
h1{{margin:0 0 4px}} p{{color:#a0be8c;margin:4px 0 14px}}
table{{border-collapse:collapse;font-size:14px}} th,td{{padding:6px 10px;border-bottom:1px solid #35592a;text-align:left}}
th{{position:sticky;top:0;background:#305026}} .num{{text-align:center;font-weight:600}} .mono{{font-family:Consolas,monospace;letter-spacing:2px}}
tr.never{{color:#6f8a62}} label{{cursor:pointer}}
</style></head><body>
<h1>Full Q-table: all {config.N_STATES} situations x 3 moves</h1>
<p>Trained for {config.EPISODES:,} games. Higher number = better move. {seen_n} situations actually happen; the other {config.N_STATES - seen_n}
are impossible combinations (they stay at zero).<br>
Answers are in this order: danger ahead, danger right, danger left | apple ahead, apple behind, apple left, apple right.</p>
<p><label><input type="checkbox" id="only" onchange="f()"> show only the {seen_n} situations that actually happen</label></p>
<table><thead><tr><th>#</th><th>answers</th><th>danger</th><th>apple</th><th>Left</th><th>Right</th><th>Forward</th><th>best move</th><th>visited</th></tr></thead>
<tbody id="b">{''.join(body)}</tbody></table>
<script>function f(){{var on=document.getElementById('only').checked;
document.querySelectorAll('tr.never').forEach(function(r){{r.style.display=on?'none':''}});}}</script>
</body></html>"""
    path = os.path.join(OUT, "full_q_table.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"{config.N_STATES} situations written ({seen_n} that actually happen)")
    print("  page :", os.path.abspath(path))
    print("  excel:", os.path.abspath(os.path.join(OUT, "full_q_table.csv")))


if __name__ == "__main__":
    main()

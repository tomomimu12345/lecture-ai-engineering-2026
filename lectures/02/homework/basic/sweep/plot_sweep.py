"""エポック掃引の集計と作図。引数: sweep_seed*.jsonl を置いたディレクトリ、出力ディレクトリ。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
recs = [json.loads(l) for p in sorted(src.glob("sweep_seed*.jsonl")) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
epochs = sorted({r["epochs"] for r in recs})
seeds = sorted({r["seed"] for r in recs})
N = 21

# 根拠あり（15 問）と根拠なし（6 問）に分けた正解率。failed から数える
for r in recs:
    fg = sum(1 for f in r["failed"] if not f["gold_needs_review"])
    fu = sum(1 for f in r["failed"] if f["gold_needs_review"])
    r["grounded_rate"] = (15 - fg) / 15
    r["ungrounded_rate"] = (6 - fu) / 6

rows = []
for ep in epochs:
    rs = [r for r in recs if r["epochs"] == ep]
    row = {"epochs": ep, "steps": rs[0]["steps"], "n_seeds": len(rs)}
    for k in ["format_rate", "content_rate", "grounded_rate", "ungrounded_rate", "final_loss", "eval_loss", "train_seconds"]:
        v = np.array([r[k] for r in rs])
        row[k] = {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                  "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
    rows.append(row)
(out / "sweep_summary.json").write_text(json.dumps({"seeds": seeds, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

# 学習時間 = a * steps + b の当てはめ
steps = np.array([r["steps"] for r in recs]); secs = np.array([r["train_seconds"] for r in recs])
a, b = np.polyfit(steps, secs, 1)
print(f"time fit: {a:.2f} s/step * steps + {b:.1f} s")
for row in rows:
    f, c, e = row["format_rate"], row["content_rate"], row["eval_loss"]
    g, u = row["grounded_rate"], row["ungrounded_rate"]
    print(f'{row["epochs"]:>3} ep {row["steps"]:>3} st | format {f["mean"]:.3f}±{f["sd"]:.3f} | content {c["mean"]:.3f}±{c["sd"]:.3f} [{c["min"]:.3f},{c["max"]:.3f}] '
          f'| grounded {g["mean"]:.3f}±{g["sd"]:.3f} | ungrounded {u["mean"]:.3f}±{u["sd"]:.3f} | final_loss {row["final_loss"]["mean"]:.3f} | eval_loss {e["mean"]:.3f}±{e["sd"]:.3f} | {row["train_seconds"]["mean"]:.0f}s')

from matplotlib import font_manager
font_manager.fontManager.addfont(str(Path.home() / ".fonts" / "meiryo.ttc"))
plt.rcParams["font.family"] = ["Meiryo", "DejaVu Sans"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
rng = np.random.default_rng(0)

def panel(ax, key, color, title, ylim=None, logy=False):
    for ep in epochs:
        v = [r[key] for r in recs if r["epochs"] == ep]
        x = ep + rng.uniform(-0.18, 0.18, len(v)) * (ep ** 0.5) * 0.4
        ax.scatter(x, v, s=18, color=color, alpha=0.45, linewidths=0, zorder=2)
    m = [np.mean([r[key] for r in recs if r["epochs"] == ep]) for ep in epochs]
    ax.plot(epochs, m, color=color, lw=2, marker="o", ms=6, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(epochs); ax.set_xticklabels(epochs)
    ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
        ax.minorticks_off()
    if ylim: ax.set_ylim(*ylim)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("エポック数", color=MUTED)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)

panel(axes[0], "grounded_rate", BLUE, "根拠あり 正解率（15 問）", ylim=(0, 1.05))
panel(axes[1], "ungrounded_rate", BLUE, "根拠なし 正解率（6 問）", ylim=(0, 1.05))
panel(axes[2], "eval_loss", ORANGE, "valid loss", logy=True)
fig.tight_layout()
fig.savefig(out / "sweep_epochs.png", dpi=160, facecolor="#fcfcfb")
print("saved")

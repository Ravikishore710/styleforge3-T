"""Training-curve plots from the JSONL log."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_log(log_path="outputs/logs/log.jsonl", out_dir="outputs/logs"):
    rows = [json.loads(l) for l in open(log_path)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, keys, title in [
        (axes[0], ["loss_G", "loss_D"], "adversarial losses"),
        (axes[1], ["D_real", "D_fake"], "D scores"),
        (axes[2], ["r1"], "R1 penalty"),
    ]:
        for k in keys:
            xs = [r["kimg"] for r in rows if r.get(k, -1) is not None and r.get(k, -1) >= 0]
            ys = [r[k] for r in rows if r.get(k, -1) is not None and r.get(k, -1) >= 0]
            if ys:
                ax.plot(xs, ys, label=k)
        ax.set_xlabel("kimg"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    import os
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/curves.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

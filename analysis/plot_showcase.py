"""Training curves of the show model: phase 1 (teacher, special tracks) and
phase 2 (GA fine-tuning) validation scores per run.

    python analysis/plot_showcase.py      -> figures/showcase_training.png
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(style.ROOT, "experiments", "results", "showcase")


def main():
    style.apply()
    runs = sorted(glob.glob(os.path.join(RES, "seed*")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, gridspec_kw={"width_ratios": [1, 2.2]})
    for k, d in enumerate(runs):
        color = style.SERIES[k]
        lab = os.path.basename(d)
        p1 = os.path.join(d, "phase1_bc.csv")
        if os.path.exists(p1):
            rows = list(csv.DictReader(open(p1)))
            it = [int(r["iter"]) for r in rows]
            axes[0].plot(it, [float(r["val_score"]) for r in rows], color=color, lw=1, alpha=0.45)
            axes[0].plot(it, [float(r["best"]) for r in rows], color=color, label=lab)
        p2 = os.path.join(d, "phase2_ga.csv")
        if os.path.exists(p2):
            rows = [r for r in csv.DictReader(open(p2)) if r["val_score"]]
            g = [int(r["gen"]) for r in rows]
            axes[1].plot(g, [float(r["val_score"]) for r in rows], color=color, lw=1, alpha=0.45)
            axes[1].plot(g, [float(r["champion_score"]) for r in rows], color=color, label=lab)
    for ax in axes:
        ax.axhline(165.6, color=style.TEXT_2, lw=1, ls="--")
    axes[0].annotate("start: BC base model", (1, 165.6), xytext=(0, 5), textcoords="offset points", fontsize=8.5,
                     color=style.TEXT_2)
    axes[0].set_title("Phase 1: learning from the teacher on extreme tracks")
    axes[0].set_xlabel("DAgger iteration")
    axes[0].set_ylabel("Validation score (extreme 2.5 / 3.0 + level 2.0)")
    axes[1].set_title("Phase 2: genetic algorithm keeps training on its own")
    axes[1].set_xlabel("Generation")
    axes[1].legend(loc="lower right", title="thin: current, thick: best kept")
    fig.tight_layout()
    print(style.save(fig, "showcase_training.png"))


if __name__ == "__main__":
    main()

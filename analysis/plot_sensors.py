"""Experiment D: how many sensor dimensions does the driver need?
Raw ray fans (3 / 5 / 9 / 17) vs PCA of the 17-ray fan (k = 2 / 4 / 8 / 17)
vs random orthonormal projections (k = 4 / 8). Behavior cloning, L 2x32.

    python analysis/plot_sensors.py

Writes figures/sens_*.png and data/sensors_summary.json.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = style.ROOT
RES = os.path.join(ROOT, "experiments", "results", "sensors")
FAMILIES = [("raw", "Raw rays (fewer rays)", [3, 5, 9, 17], style.SERIES[0]),
            ("pca", "PCA of 17 rays (eigenvectors)", [2, 4, 8, 17], style.SERIES[1]),
            ("rand", "Random projection of 17 rays", [4, 8], style.SERIES[4])]


def load(cond):
    rows = []
    for d in sorted(glob.glob(os.path.join(RES, cond, "seed*"))):
        p = os.path.join(d, "final_test.json")
        if not os.path.exists(p):
            continue
        ft = json.load(open(p))
        dif, f = np.array(ft["difficulty"]), np.array(ft["finished"], bool)
        sp = np.array(ft["speed"])
        rows.append({"all": f.mean() * 100, "hard": f[dif >= 1.5].mean() * 100,
                     "speed": sp[f].mean() if f.any() else np.nan, "score": float(np.mean(ft["score"])),
                     "params": json.load(open(os.path.join(d, "config.json")))["params"]})
    return rows


def main():
    style.apply()
    data = {f"{fam}{k}": load(f"{fam}{k}") for fam, _, ks, _ in FAMILIES for k in ks}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric, title in ((axes[0], "all", "All 7 levels"), (axes[1], "hard", "Hard tracks (1.5–2.0)")):
        for fam, lab, ks, color in FAMILIES:
            xs = [k for k in ks if data[f"{fam}{k}"]]
            if not xs:
                continue
            for k in xs:
                v = [r[metric] for r in data[f"{fam}{k}"]]
                ax.scatter(np.full(len(v), k), v, s=12, color=color, alpha=0.45, lw=0)
            ax.plot(xs, [np.median([r[metric] for r in data[f"{fam}{k}"]]) for k in xs], "-o", color=color,
                    label=lab, ms=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 3, 4, 5, 8, 9, 17], ["2", "3", "4", "5", "8", "9", "17"])
        ax.set_xlabel("Sensor dimensions fed to the network")
        ax.set_title(title)
    axes[0].set_ylabel("Laps finished, final model (%)")
    axes[0].set_ylim(0, 105)
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("How many sensor dimensions are needed? (L 2×32, behavior cloning, median of 4 seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.0), ncol=3)
    print(style.save(fig, "sens_dimensions.png"))
    summ = {c: {"n": len(r), **{m: (np.round([x[m] for x in r], 2).tolist()) for m in ("all", "hard", "speed", "score")},
                "params": r[0]["params"] if r else None} for c, r in data.items()}
    with open(os.path.join(ROOT, "data", "sensors_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    for c, r in data.items():
        if r:
            print(f"{c:7s} n{len(r)} params {r[0]['params']:5d}  all {np.median([x['all'] for x in r]):5.1f}  "
                  f"hard {np.median([x['hard'] for x in r]):5.1f}  speed {np.nanmedian([x['speed'] for x in r]):4.1f}  "
                  f"seeds {[round(x['all']) for x in r]}")


if __name__ == "__main__":
    main()

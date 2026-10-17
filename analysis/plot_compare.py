"""GA vs behavior cloning + DAgger, same sizes, same stations, same test.

    python analysis/plot_compare.py

Writes figures/cmp_*.png and data/compare_summary.json.
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
RES = os.path.join(ROOT, "experiments", "results")
SIZES = [("S9_XS_linear", "XS Linear", 56), ("S9_S_2x8", "S 2×8", 220),
         ("S9_L_2x32", "L 2×32", 1636), ("S9_XL_2x64", "XL 2×64", 5316)]
METHODS = [("stage3", "GA (trial and error)", style.SERIES[0]),
           ("bc", "Behavior cloning + DAgger (gradient)", style.SERIES[1])]


def collect(method, key):
    out = {"after_finish": [], "during_finish": [], "after_speed": [], "after_score": [], "hours": []}
    for d in sorted(glob.glob(os.path.join(RES, method, key, "seed*"))):
        if not os.path.exists(os.path.join(d, "final_test.json")):
            continue
        ft = json.load(open(os.path.join(d, "final_test.json")))
        st = json.load(open(os.path.join(d, "station_tests.json")))
        f = np.array(ft["finished"], bool)
        out["after_finish"].append(f.mean() * 100)
        out["during_finish"].append(np.mean([np.mean(x["finished"]) for x in st]) * 100)
        out["after_speed"].append(float(np.array(ft["speed"])[f].mean()) if f.any() else np.nan)
        out["after_score"].append(float(np.mean(ft["score"])))
        out["hours"].append(json.load(open(os.path.join(d, "status.json")))["wall_seconds"] / 3600)
    return out


def fig_metric(data, metric, ylabel, title, name, ylim=None):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    w = 0.36
    for j, (m, lab, color) in enumerate(METHODS):
        xs, meds = [], []
        for i, (key, _, _) in enumerate(SIZES):
            v = np.array(data[m][key][metric], float)
            if not len(v):
                continue
            x = i + (j - 0.5) * w
            med = np.nanmedian(v)
            ax.bar(x, med, w * 0.9, color=color, label=lab if i == 0 else None)
            ax.scatter(np.full(len(v), x), v, s=16, color=style.TEXT, zorder=5, lw=0)
            ax.annotate(f"{med:.0f}", (x, 0), xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(range(len(SIZES)), [f"{lab}\n{p:,} params" for _, lab, p in SIZES])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), ncol=2)
    fig.tight_layout()
    return style.save(fig, name)


def main():
    style.apply()
    data = {m: {key: collect(m, key) for key, _, _ in SIZES} for m, _, _ in METHODS}
    print(fig_metric(data, "after_finish", "Laps finished on test T (%)",
                     "Final model on all 7 levels: GA vs gradient (behavior cloning)", "cmp_finish.png", (0, 105)))
    print(fig_metric(data, "after_speed", "Lap speed of finished laps (m/s)",
                     "Speed of finished laps: GA vs gradient", "cmp_speed.png"))
    print(fig_metric(data, "hours", "Training time (hours, wall clock)",
                     "Training cost: GA vs gradient", "cmp_hours.png"))
    summ = {m: {key: {k: np.round(v, 2).tolist() for k, v in d.items()} for key, d in dm.items()}
            for m, dm in data.items()}
    with open(os.path.join(ROOT, "data", "compare_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    for m, _, _ in METHODS:
        for key, lab, _ in SIZES:
            d = data[m][key]
            print(f"{m:7s} {lab:10s} n {len(d['after_finish'])}  during {np.median(d['during_finish']) if d['during_finish'] else float('nan'):5.1f}"
                  f"  after {np.median(d['after_finish']) if d['after_finish'] else float('nan'):5.1f}"
                  f"  speed {np.nanmedian(d['after_speed']) if d['after_speed'] else float('nan'):5.1f}"
                  f"  hours {np.median(d['hours']) if d['hours'] else float('nan'):5.2f}")


if __name__ == "__main__":
    main()

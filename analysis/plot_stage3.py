"""Figures of the model-size experiment (stage 3, v2.6): 4 sizes x 4 seeds,
seven curriculum stations, a test at each station and a final re-test.

    python analysis/plot_stage3.py

Writes figures/s3_*.png and data/stage3_summary.json.
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

ROOT = style.ROOT
RES = os.path.join(ROOT, "experiments", "results", "stage3")
PREFIX, METHOD, XLABEL = "s3", "GA", "Generation"   # switched by --bc
LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SIZES = [("S9_XS_linear", "XS  Linear", 56), ("S9_S_2x8", "S  2×8", 220),
         ("S9_L_2x32", "L  2×32", 1636), ("S9_XL_2x64", "XL  2×64", 5316)]
COLOR = {k: style.SERIES[i] for i, (k, _, _) in enumerate(SIZES)}
DURING, AFTER = style.SERIES[0], style.SERIES[1]


def per_level(t, key, finished_only=False):
    d = np.array(t["difficulty"])
    v = np.array(t[key], dtype=float)
    f = np.array(t["finished"], dtype=bool)
    out = []
    for lvl in LEVELS:
        m = d == lvl
        if finished_only:
            m = m & f
        out.append(v[m].mean() if m.any() else np.nan)
    return np.array(out)


def load():
    runs = {}
    for key, _, _ in SIZES:
        runs[key] = []
        for d in sorted(glob.glob(os.path.join(RES, key, "seed*"))):
            if not os.path.exists(os.path.join(d, "final_test.json")):
                continue
            st = json.load(open(os.path.join(d, "station_tests.json")))
            ft = json.load(open(os.path.join(d, "final_test.json")))
            status = json.load(open(os.path.join(d, "status.json")))
            by_lvl = {round(x["level"], 2): x for x in st}
            during = {k: np.array([per_level(by_lvl[l], k, k == "speed")[LEVELS.index(l)] if l in by_lvl else np.nan
                                   for l in LEVELS]) for k in ("finished", "score", "speed")}
            after = {k: per_level(ft, k, k == "speed") for k in ("finished", "score", "speed")}
            with open(os.path.join(d, "eval_v1.csv")) as f:
                v1 = [(int(r["gen"]), float(r["v1_score"]), float(r["difficulty"])) for r in csv.DictReader(f)]
            runs[key].append({"dir": d, "seed": int(d[-1]), "during": during, "after": after,
                              "status": status, "v1": np.array(v1)})
    return runs


def size_label(key):
    return next(f"{lab}  ({p:,} params)" for k, lab, p in SIZES if k == key)


# ------------------------------------------------------------------ figures
def fig_station_lines(runs, metric="finished", scale=100, ylabel="Laps finished (%)", name="station_finish.png"):
    """Skill per station: test taken right after each station (during
    training) vs the final model re-taking every level (after training)."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    for ax, (key, lab, p) in zip(axes, SIZES):
        rs = runs[key]
        for which, color, text in (("during", DURING, "During training (station test)"),
                                   ("after", AFTER, "After training (final re-test)")):
            M = np.array([r[which][metric] for r in rs]) * scale
            for row in M:
                ax.plot(LEVELS, row, color=color, lw=0.8, alpha=0.25)
            ax.plot(LEVELS, np.nanmedian(M, 0), "-o", color=color, ms=5, label=text, zorder=4)
        ax.set_title(f"{lab}", loc="left")
        ax.text(0, 1.0, f"{p:,} params · {len(rs)} seeds", transform=ax.transAxes, fontsize=8.5,
                color=style.TEXT_2, va="bottom")
        ax.set_xticks(LEVELS, [f"{x:g}" for x in LEVELS], fontsize=8.5)
        ax.set_xlabel("Difficulty level")
    axes[0].set_ylabel(ylabel)
    if metric == "finished":
        axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle(f"[{METHOD}] Skill at every station: during vs after training (median of seeds, thin lines = single seeds)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    return style.save(fig, f"{PREFIX}_{name}")


def fig_forgetting(runs):
    """After - during, per level: negative = forgotten, positive = improved later."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0, color=style.TEXT_2, lw=1)
    off = np.linspace(-0.045, 0.045, len(SIZES))
    for o, (key, lab, _) in zip(off, SIZES):
        D = np.array([(r["after"]["finished"] - r["during"]["finished"]) * 100 for r in runs[key]])
        med = np.nanmedian(D, 0)
        ax.plot(np.array(LEVELS) + o, med, "-o", color=COLOR[key], label=lab, ms=6)
    ax.set_xticks(LEVELS, [f"{x:g}" for x in LEVELS])
    ax.set_xlabel("Difficulty level")
    ax.set_ylabel("Change in laps finished (points)")
    ax.set_title(f"[{METHOD}] Forgetting: final model minus station test (below 0 = forgot)")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    return style.save(fig, f"{PREFIX}_forgetting.png")


def fig_overall(runs):
    """Mean over the seven levels, per seed (dots) and median (bar tick)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, (key, lab, _) in enumerate(SIZES):
        for j, (which, color) in enumerate((("during", DURING), ("after", AFTER))):
            vals = np.array([np.nanmean(r[which]["finished"]) * 100 for r in runs[key]])
            x = i + (j - 0.5) * 0.32
            ax.bar(x, np.median(vals), width=0.28, color=color, alpha=0.9,
                   label=("During training (station tests)" if which == "during" else "After training (final re-test)")
                   if i == 0 else None)
            ax.scatter(np.full(len(vals), x), vals, s=22, color=style.TEXT, zorder=5, lw=0)
            ax.annotate(f"{np.median(vals):.0f}", (x, 2), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(range(len(SIZES)), [lab for _, lab, _ in SIZES])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Laps finished, mean of 7 levels (%)")
    ax.set_title(f"[{METHOD}] Overall skill, during vs after training")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    fig.tight_layout()
    return style.save(fig, f"{PREFIX}_overall.png")


def fig_learning(runs):
    """Skill per training round: champion's V1 score (12 fixed tracks at
    1.0/1.5/2.0, never trained on) every 25 generations, and the curriculum."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for key, lab, _ in SIZES:
        for k, r in enumerate(runs[key]):
            g, v, d = r["v1"].T
            sm = np.convolve(v, np.ones(4) / 4, mode="valid")   # trailing mean, no edge padding
            axes[0].plot(g[3:], sm, color=COLOR[key], lw=1.1, alpha=0.75, label=lab if k == 0 else None)
            axes[1].plot(g, d, color=COLOR[key], lw=1.1, alpha=0.75)
    axes[0].set_ylabel("V1 score (smoothed)")
    axes[0].set_title(f"[{METHOD}] Skill per training round (V1 score, one line per seed)")
    axes[0].legend(ncol=4, loc="lower right")
    axes[1].set_ylabel("Difficulty")
    axes[1].set_xlabel(XLABEL)
    axes[1].set_yticks(LEVELS[::2] + [0])
    fig.tight_layout()
    return style.save(fig, f"{PREFIX}_learning_curve.png")


def fig_speed(runs):
    """Average speed of finished laps (distance / time) after training."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for key, lab, _ in SIZES:
        M = np.array([r["after"]["speed"] for r in runs[key]])
        ax.plot(LEVELS, np.nanmedian(M, 0), "-o", color=COLOR[key], label=lab)
    ax.set_xticks(LEVELS, [f"{x:g}" for x in LEVELS])
    ax.set_xlabel("Difficulty level")
    ax.set_ylabel("Lap speed of finished laps (m/s)")
    ax.set_title(f"[{METHOD}] Speed (distance / time) of the final models")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    return style.save(fig, f"{PREFIX}_speed.png")


def summary(runs):
    out = {}
    for key, lab, p in SIZES:
        rs = runs[key]
        out[key] = {
            "label": lab, "params": p, "seeds": [r["seed"] for r in rs],
            "generations": [r["status"]["generations"] for r in rs],
            "hours": [round(r["status"]["wall_seconds"] / 3600, 2) for r in rs],
            "outcomes": [s["outcome"] for r in rs for s in r["status"]["stations"]],
        }
        for which in ("during", "after"):
            for m, sc in (("finished", 100), ("score", 1), ("speed", 1)):
                M = np.array([r[which][m] for r in rs]) * sc
                out[key][f"{which}_{m}_median_by_level"] = np.round(np.nanmedian(M, 0), 2).tolist()
                out[key][f"{which}_{m}_mean_per_seed"] = np.round(np.nanmean(M, 1), 2).tolist()
    with open(os.path.join(ROOT, "data", f"{PREFIX}_summary.json" if PREFIX != "s3" else "stage3_summary.json"), "w") as f:
        json.dump({"levels": LEVELS, "sizes": out}, f, indent=1)
    return out


def main():
    global RES, PREFIX, METHOD, XLABEL
    if "--bc" in sys.argv:
        RES = os.path.join(ROOT, "experiments", "results", "bc")
        PREFIX, METHOD, XLABEL = "bc", "Behavior cloning + DAgger", "DAgger iteration"
    style.apply()
    runs = load()
    print({k: len(v) for k, v in runs.items()})
    for p in (fig_station_lines(runs),
              fig_station_lines(runs, "score", 1, "Race score", "station_score.png"),
              fig_forgetting(runs), fig_overall(runs), fig_learning(runs), fig_speed(runs)):
        print(p)
    s = summary(runs)
    for key, v in s.items():
        print(key, "during", np.round(np.mean(v["during_finished_mean_per_seed"])), "after",
              np.round(np.mean(v["after_finished_mean_per_seed"])), v["hours"])


if __name__ == "__main__":
    main()

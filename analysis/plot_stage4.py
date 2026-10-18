"""Figures of the compression experiment (stage 4).

    python analysis/plot_stage4.py

Writes figures/s4_*.png and data/stage4_summary.json.
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
TRAINED = [("S9_XS_linear", "XS Linear"), ("S9_S_2x8", "S 2×8"), ("S9_L_2x32", "L 2×32"), ("S9_XL_2x64", "XL 2×64")]
SRC_LABEL = {"S9_XL_2x64": "XL 2×64", "S9_L_2x32": "L 2×32"}
COMP, TRAIN, PREFIX, TAG = "stage4", "stage3", "s4", "GA"   # switched by --bc
METHODS = [("prune_ls", "Prune + least squares", style.SERIES[0]),
           ("prune_importance", "Prune by importance", style.SERIES[1]),
           ("prune_random", "Prune at random (control)", style.SERIES[4]),
           ("quantise", "Quantise (fewer bits)", style.SERIES[2]),
           ("prune_ls+quantise", "Prune LS + 4-bit", style.SERIES[6]),
           ("low_rank", "Low rank (truncated SVD)", style.SERIES[5])]


def load_compressed(src):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, COMP, src, "seed*.json"))):
        for v in json.load(open(f))["variants"]:
            out.setdefault(v["name"], []).append(v)
    return out


def load_trained():
    out = {}
    for key, lab in TRAINED:
        vals, params = [], None
        for d in sorted(glob.glob(os.path.join(RES, TRAIN, key, "seed*"))):
            p = os.path.join(d, "final_test.json")
            if os.path.exists(p):
                vals.append(np.mean(json.load(open(p))["finished"]) * 100)
                params = json.load(open(os.path.join(d, "config.json")))["params"]
        out[key] = {"label": lab, "finish": vals, "params": params, "kbit": params * 32 / 1000}
    return out


def finish(vs):
    return np.array([np.mean(v["finished"]) * 100 for v in vs])


def fig_pareto(src, comp, trained):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for m, lab, color in METHODS:
        names = sorted({n for n, vs in comp.items() if vs[0]["method"] == m},
                       key=lambda n: comp[n][0]["size_bits"])
        if not names:
            continue
        x = [comp[n][0]["size_bits"] / 1000 for n in names]
        y = [np.median(finish(comp[n])) for n in names]
        ax.plot(x, y, "-o", color=color, label=lab, ms=6, zorder=3)
        if m == "quantise":
            for n, xi, yi in zip(names, x, y):
                if yi >= 5:   # points on the zero line stay unlabeled (they would pile up)
                    ax.annotate(f"{comp[n][0]['bits']} bit", (xi, yi), xytext=(-7, 0), textcoords="offset points",
                                fontsize=8, color=style.TEXT_2, ha="right", va="center")
        elif m == "low_rank":
            for n, xi, yi in zip(names, x, y):
                if yi >= 5:
                    ax.annotate(f"rank {n[7:]}", (xi, yi), xytext=(5, 5), textcoords="offset points",
                                fontsize=8, color=style.TEXT_2)
        elif m == "prune_ls":
            for n, xi, yi in zip(names, x, y):
                if yi >= 5:
                    ax.annotate(f"width {comp[n][0]['width']}", (xi, yi), xytext=(5, 5), textcoords="offset points",
                                fontsize=8, color=style.TEXT_2)
    o = comp["original"]
    ax.scatter([o[0]["size_bits"] / 1000], [np.median(finish(o))], marker="D", s=70, color=style.TEXT, zorder=5,
               label=f"Original {SRC_LABEL[src]} (32-bit)")
    tx = [t["kbit"] for t in trained.values() if t["finish"]]
    ty = [np.median(t["finish"]) for t in trained.values() if t["finish"]]
    ax.scatter(tx, ty, marker="*", s=230, color=style.SERIES[3], edgecolor=style.TEXT, lw=0.6, zorder=6,
               label="Trained from scratch (32-bit)")
    for t in trained.values():
        if t["finish"]:
            ax.annotate(t["label"], (t["kbit"], np.median(t["finish"])), xytext=(0, -13), textcoords="offset points",
                        ha="center", va="top", fontsize=8.5, color=style.TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (kilobits, log scale)")
    ax.set_ylabel("Laps finished on test T (%)")
    ax.set_ylim(-3, 105)
    ax.set_title(f"[{TAG}] Compressing {SRC_LABEL[src]}: size vs skill (median of seeds)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.tight_layout()
    return style.save(fig, f"{PREFIX}_pareto_{SRC_LABEL[src].split()[0]}.png")


def fig_matched(comps, trained):
    """Same shape: big model pruned down vs small model trained at that shape."""
    pairs = [("S9_XL_2x64", 32, "S9_L_2x32"), ("S9_XL_2x64", 8, "S9_S_2x8"), ("S9_L_2x32", 8, "S9_S_2x8")]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    width = 0.2
    for i, (src, w, tgt) in enumerate(pairs):
        comp = comps[src]
        bars = [(f"prune_ls_{w}", "Prune + least squares", style.SERIES[0]),
                (f"prune_importance_{w}", "Prune by importance", style.SERIES[1]),
                (f"prune_random_{w}", "Prune at random", style.SERIES[4])]
        for j, (n, lab, color) in enumerate(bars):
            v = finish(comp[n])
            x = i + (j - 1.5) * width
            ax.bar(x, np.median(v), width * 0.92, color=color, label=lab if i == 0 else None)
            ax.scatter(np.full(len(v), x), v, s=14, color=style.TEXT, zorder=5, lw=0)
        t = trained[tgt]["finish"]
        x = i + 1.5 * width
        ax.bar(x, np.median(t), width * 0.92, color=style.SERIES[3], label="Trained at that size" if i == 0 else None)
        ax.scatter(np.full(len(t), x), t, s=14, color=style.TEXT, zorder=5, lw=0)
    ax.set_xticks(range(len(pairs)),
                  [f"{SRC_LABEL[s]} → width {w}\nvs trained {trained[t]['label']}" for s, w, t in pairs])
    ax.set_ylabel("Laps finished on test T (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"[{TAG}] Same shape: compressed big model vs small model trained from scratch")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    fig.tight_layout()
    return style.save(fig, f"{PREFIX}_matched_shape.png")


def main():
    global COMP, TRAIN, PREFIX, TAG
    if "--bc" in sys.argv:
        COMP, TRAIN, PREFIX, TAG = "stage4_bc", "bc_t75_long", "s4bc", "BC teacher 75%"
    style.apply()
    trained = load_trained()
    comps = {src: load_compressed(src) for src in SRC_LABEL}
    for src, comp in comps.items():
        print(fig_pareto(src, comp, trained))
    print(fig_matched(comps, trained))
    summary = {"trained": {k: {"label": v["label"], "params": v["params"], "kbit": v["kbit"],
                               "finish_per_seed": np.round(v["finish"], 1).tolist()} for k, v in trained.items()},
               "compressed": {src: {n: {"method": vs[0]["method"], "width": vs[0]["width"], "bits": vs[0]["bits"],
                                        "kbit": vs[0]["size_bits"] / 1000, "params": vs[0]["params"],
                                        "finish_per_seed": np.round(finish(vs), 1).tolist(),
                                        "open_loop_rms_per_seed": [round(v["open_loop_rms"], 4) for v in vs]}
                                    for n, vs in comp.items()} for src, comp in comps.items()}}
    with open(os.path.join(ROOT, "data", f"{PREFIX}_summary.json" if PREFIX != "s4" else "stage4_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for src, comp in comps.items():
        for n in sorted(comp, key=lambda n: -comp[n][0]["size_bits"]):
            print(f"{src:12s} {n:24s} {comp[n][0]['size_bits'] / 1000:6.1f} kbit  finish median "
                  f"{np.median(finish(comp[n])):5.1f}  seeds {np.round(finish(comp[n])).astype(int).tolist()}")


if __name__ == "__main__":
    main()

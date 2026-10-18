"""Summary of the friends' demo tests (data/demo_results.csv, written by app/demo.py).

    python analysis/demo_summary.py

Prints a table per tester and per model and writes figures/demo_results.png
and data/demo_summary.json - material for the presentation's demo slide.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

PATH = os.path.join(style.ROOT, "data", "demo_results.csv")


def main():
    if not os.path.exists(PATH):
        print("no demo results yet:", PATH)
        return
    rows = list(csv.DictReader(open(PATH)))
    testers = sorted({r["tester"] for r in rows})
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} tests by {len(testers)} testers")
    out = {"tests": len(rows), "testers": {}, "models": {}}
    for t in testers:
        rs = [r for r in rows if r["tester"] == t]
        out["testers"][t] = len(rs)
        print(f"  {t:20s} {len(rs)} tests")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        fin = [r["finished"] in ("1", "True", "true") for r in rs]
        prog = [float(r["progress_pct"]) for r in rs]
        spd = [float(r["avg_speed_kmh"]) for r, f in zip(rs, fin) if f and r["avg_speed_kmh"]]
        out["models"][m] = {"tests": len(rs), "finished": int(sum(fin)), "mean_progress_pct": float(np.mean(prog)),
                            "mean_speed_kmh_finished": float(np.mean(spd)) if spd else None}
        print(f"  {m:45s} {sum(fin)}/{len(rs)} finished, progress {np.mean(prog):5.1f} %, "
              f"speed {np.mean(spd) if spd else float('nan'):5.1f} km/h")
    with open(os.path.join(style.ROOT, "data", "demo_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    style.apply()
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(models) + 1.5))
    y = np.arange(len(models))
    prog = [out["models"][m]["mean_progress_pct"] for m in models]
    ax.barh(y, prog, color=style.SERIES[0], height=0.6)
    for yi, m, p in zip(y, models, prog):
        d = out["models"][m]
        ax.text(p + 1, yi, f"{d['finished']}/{d['tests']} finished", va="center", fontsize=9, color=style.TEXT)
    ax.set_yticks(y, models)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Mean lap progress in friends' tests (%)")
    ax.set_title(f"Demo: {len(rows)} tests by {len(testers)} testers")
    ax.invert_yaxis()
    fig.tight_layout()
    print(style.save(fig, "demo_results.png"))


if __name__ == "__main__":
    main()

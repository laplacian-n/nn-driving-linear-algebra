"""Final evaluation of every finished run: the representative (champion of
the best-V1 snapshot) drives V2 and T; per-track results go to
<run>/eval_final.json. Runs in parallel, skips runs already evaluated.

    python experiments/evaluate.py stage1 stage2 --jobs 10
    python experiments/evaluate.py stage3 --budget 400      # also the fixed-budget representative
"""
import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def representative(run_dir, max_gen=None):
    with open(os.path.join(run_dir, "eval_v1.csv")) as f:
        rows = [r for r in csv.DictReader(f) if max_gen is None or int(r["gen"]) <= max_gen]
    best = max(rows, key=lambda r: float(r["v1_score"]))
    return int(best["gen"]), float(best["v1_score"])


def evaluate_run(run_dir, max_gen=None, tag="final", snapshot=None):
    from racing.inputs import Calibration, InputSpec
    from racing.network import Population
    from racing.sets import load as load_sets
    from racing.sim import evaluate
    out_path = os.path.join(run_dir, f"eval_{tag}.json")
    if os.path.exists(out_path):
        return run_dir, "skip"
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    if snapshot is None:
        gen, v1 = representative(run_dir, max_gen)
        pop, meta = Population.load(os.path.join(run_dir, "snapshots", f"gen_{gen:05d}.npz"))
    else:  # a station model: the champion saved when that station saturated
        pop, meta = Population.load(snapshot)
        gen, v1 = meta["gen"], None
    champ = pop.subset([meta["champion"]])
    calib = Calibration.load(os.path.join(ROOT, "data", "calibration.json"))
    spec = InputSpec(cfg["rays"], calib, reduction=cfg.get("reduction"), k=cfg.get("k"), seed=cfg["seed"])
    sets = load_sets()
    result = {"run": os.path.relpath(run_dir, ROOT), "gen": gen, "v1": v1, "params": cfg["params"],
              "preprocessing_params": cfg["preprocessing_params"]}
    for name in ("V2", "T"):
        r = evaluate(champ, spec, sets[name], rng_seed=777)
        result[name] = {"seed": [it["seed"] for it in sets[name]],
                        "difficulty": [it["difficulty"] for it in sets[name]],
                        "score": r["score"][:, 0].tolist(), "frac": r["frac"][:, 0].tolist(),
                        "finished": r["finished"][:, 0].astype(int).tolist(),
                        "lap_time": np.nan_to_num(r["lap_time"][:, 0], nan=-1).tolist()}
    with open(out_path, "w") as f:
        json.dump(result, f)
    return run_dir, f"V2 {np.mean(result['V2']['score']):.1f}  T {np.mean(result['T']['score']):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--budget", type=int, default=None, help="also evaluate the best snapshot with gen <= budget")
    a = ap.parse_args()
    runs = []
    for st in a.stages:
        for status in sorted(glob.glob(os.path.join(ROOT, "experiments", "results", st, "*", "seed*", "status.json"))):
            runs.append(os.path.dirname(status))
    jobs = [(r, None, "final", None) for r in runs]
    # station models: each one sits the whole test (all 7 levels) - the
    # cross-test matrix that also shows forgetting
    for r in runs:
        for st in sorted(glob.glob(os.path.join(r, "station_*.npz"))):
            lvl = os.path.basename(st)[len("station_"):-4]
            jobs.append((r, None, f"station{lvl}", st))
    if a.budget:
        jobs += [(r, a.budget, f"budget{a.budget}", None) for r in runs]
    print(f"{len(jobs)} evaluations", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for run, msg in ex.map(evaluate_run, *zip(*jobs)):
            print(os.path.relpath(run, ROOT), msg, flush=True)


if __name__ == "__main__":
    main()

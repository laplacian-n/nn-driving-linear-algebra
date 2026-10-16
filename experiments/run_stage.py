"""Run all runs of an experiment stage in parallel (one process per run).

    python experiments/run_stage.py stage0 --jobs 10
    python experiments/run_stage.py stage1 stage2 --jobs 10 --gens 400
    python experiments/run_stage.py stage3 --jobs 10 --rays 17 --reduction pca --k 8

A run whose status.json already exists is skipped, so a stage can be
restarted after an interruption and only the unfinished runs are redone.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
SEEDS5 = [0, 1, 2, 3, 4]
STATIONS = [1.0, 1.5, 2.0]


def stage_runs(stage, a):
    G = a.gens
    if stage == "stage0":
        return [dict(name=f"stage0/{n}", rays=17, depth=d, width=w, seed=s, gens=600)
                for n, d, w in [("linear", 0, 0), ("2x16", 2, 16), ("3x32", 3, 32)] for s in [0, 1, 2]]
    if stage == "stage1":
        matched = [(3, 21), (5, 20), (9, 18), (17, 16)]
        runs = [dict(name=f"stage1/S{r}_2x{w}", rays=r, depth=2, width=w, seed=s, gens=G)
                for r, w in matched for s in SEEDS5]
        runs += [dict(name=f"stage1/S{r}_2x16", rays=r, depth=2, width=16, seed=s, gens=G)
                 for r in (3, 5, 9) for s in SEEDS5]
        return runs
    if stage == "stage2":
        cfgs = [("pca", 2, 21), ("pca", 4, 20), ("pca", 8, 19), ("pca", 17, 16), ("random", 4, 20), ("random", 8, 19)]
        return [dict(name=f"stage2/{red}{k}_2x{w}", rays=17, reduction=red, k=k, depth=2, width=w, seed=s, gens=G)
                for red, k, w in cfgs for s in SEEDS5]
    seeds = a.seeds or SEEDS5
    if stage == "stage3":
        # five sizes, each ~3x the previous (all 2 hidden layers except Linear,
        # so halving the width of one model gives exactly the next smaller one)
        inp = dict(rays=a.rays, reduction=a.reduction, k=a.k)
        tag = f"{a.reduction}{a.k}" if a.reduction else f"S{a.rays}"
        archs = [("XS_linear", 0, 0), ("S_2x8", 2, 8), ("M_2x16", 2, 16), ("L_2x32", 2, 32), ("XL_2x64", 2, 64)]
        return [dict(name=f"stage3/{tag}_{n}", depth=d, width=w, seed=s, gens=G, saturate=True,
                     max_gens=a.max_gens, stations=STATIONS, **inp)
                for n, d, w in archs for s in seeds]
    if stage == "stage3b":
        return [dict(name=f"stage3b/S17_2x{w}", rays=17, depth=2, width=w, seed=s, gens=G, saturate=True,
                     max_gens=a.max_gens, stations=STATIONS) for w in (8, 32) for s in SEEDS5]
    raise ValueError(stage)


def command(run):
    cmd = [sys.executable, "-m", "racing.train", "--name", run["name"], "--rays", str(run["rays"]),
           "--depth", str(run["depth"]), "--width", str(run["width"]), "--seed", str(run["seed"]),
           "--gens", str(run["gens"])]
    if run.get("reduction"):
        cmd += ["--reduction", run["reduction"], "--k", str(run["k"])]
    if run.get("saturate"):
        cmd += ["--saturate", "--max-gens", str(run["max_gens"])]
    if run.get("stations"):
        cmd += ["--stations", ",".join(str(x) for x in run["stations"])]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--max-gens", type=int, default=3000)
    ap.add_argument("--rays", type=int, default=17)
    ap.add_argument("--reduction", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seeds", type=lambda x: [int(v) for v in x.split(",")], default=None,
                    help="comma-separated seeds (default 0-4)")
    ap.add_argument("--largest-first", action="store_true",
                    help="queue the biggest networks first so the long runs don't trail at the end")
    a = ap.parse_args()

    runs = [r for st in a.stages for r in stage_runs(st, a)]
    if a.largest_first:
        def size(r):
            d, w = r["depth"], r["width"]
            n_in = (r.get("k") or r["rays"]) + 4
            return (n_in + 1) * (w or 4) + max(d - 1, 0) * (w + 1) * w + (w + 1) * 4 if d else (n_in + 1) * 4
        runs.sort(key=size, reverse=True)
    todo = [r for r in runs if not os.path.exists(os.path.join(RESULTS, r["name"], f"seed{r['seed']}", "status.json"))]
    print(f"{len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to run, {a.jobs} in parallel",
          flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    running = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < a.jobs:
            r = todo.pop(0)
            out = os.path.join(RESULTS, r["name"], f"seed{r['seed']}")
            os.makedirs(out, exist_ok=True)
            log = open(os.path.join(out, "stdout.log"), "a")
            p = subprocess.Popen(command(r), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, r, log))
        time.sleep(5)
        for item in list(running):
            p, r, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                state = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
                print(f"[{time.time() - t0:7.0f}s] {r['name']} seed{r['seed']} {state}; "
                      f"{len(todo)} queued, {len(running)} running", flush=True)


if __name__ == "__main__":
    main()

"""Gate between the pilot (stage 0) and stages 1+2.

Waits for stage 0 to finish, evaluates it, checks the pass criteria, and only
if they hold freezes the configuration and launches stages 1 and 2.

Pass criteria (EXPERIMENT_PLAN 6.0, v2.2):
  * 2x16 reaches difficulty 1.0 in at least 2 of 3 seeds
  * forgetting is gone: median finish rate over all 9 pilot runs on the
    T tracks at levels 0.5 / 0.75 / 1.0 is at least 40 %
Budget for stages 1-2: G = 1.25 x median generations 2x16 needed to reach
difficulty 2.0, clipped to [300, 800] (2.0 if never reached -> 800).

    nohup python experiments/after_pilot.py > experiments/results/after_pilot.log 2>&1 &
"""
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
PY = sys.executable


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    while len(glob.glob(os.path.join(RES, "stage0", "*", "seed*", "status.json"))) < 9:
        time.sleep(60)
    log("pilot finished, evaluating")
    subprocess.run([PY, os.path.join(ROOT, "experiments", "evaluate.py"), "stage0", "--jobs", "9"], check=True)

    import numpy as np
    reached1, reached2 = [], []
    for d in sorted(glob.glob(os.path.join(RES, "stage0", "2x16", "seed*"))):
        st = json.load(open(os.path.join(d, "status.json")))
        reached1.append(st["reached_full_difficulty_at"])
        g2 = None
        for line in open(os.path.join(d, "history.csv")).read().splitlines()[1:]:
            gen, diff = line.split(",")[:2]
            if float(diff) >= 2.0:
                g2 = int(gen)
                break
        reached2.append(g2)
    fin = []
    for f in sorted(glob.glob(os.path.join(RES, "stage0", "*", "seed*", "eval_final.json"))):
        T = json.load(open(f))["T"]
        d = np.array(T["difficulty"])
        fin.append(float(np.array(T["finished"])[d <= 1.0].mean()))
    ok1 = sum(r is not None for r in reached1) >= 2
    med_fin = float(np.median(fin))
    ok2 = med_fin >= 0.40
    g2 = [g for g in reached2 if g is not None]
    G = int(np.clip(1.25 * np.median(g2), 300, 800)) if g2 else 800
    G = int(round(G / 25) * 25)
    report = {"2x16_reached_1.0_at": reached1, "2x16_reached_2.0_at": reached2,
              "finish_rate_levels_le_1.0_per_run": fin, "median_finish_rate_le_1.0": med_fin,
              "pass_curriculum": ok1, "pass_forgetting": ok2, "G": G}
    log(json.dumps(report))
    if not (ok1 and ok2):
        log("PILOT GATE FAILED - stopping, nothing launched")
        json.dump(report, open(os.path.join(RES, "pilot_gate.json"), "w"), indent=1)
        return
    frozen = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "pilot_gate": report, "G_stage1_2": G,
              "stations_stage3": [1.0, 1.5, 2.0], "seeds": [0, 1, 2, 3, 4],
              "source_hashes": json.load(open(glob.glob(os.path.join(RES, "stage0", "2x16", "seed0",
                                                                     "config.json"))[0]))["source_hashes"]}
    json.dump(frozen, open(os.path.join(ROOT, "experiments", "FROZEN_CONFIG.json"), "w"), indent=1)
    log(f"gate passed, config frozen, launching stage1 + stage2 with G = {G}")
    with open(os.path.join(RES, "stage12_runner.log"), "w") as out:
        subprocess.run([PY, os.path.join(ROOT, "experiments", "run_stage.py"), "stage1", "stage2",
                        "--jobs", "10", "--gens", str(G)], stdout=out, stderr=subprocess.STDOUT)
    log("stage1 + stage2 finished, evaluating")
    subprocess.run([PY, os.path.join(ROOT, "experiments", "evaluate.py"), "stage1", "stage2", "--jobs", "10"])
    log("done")


if __name__ == "__main__":
    main()

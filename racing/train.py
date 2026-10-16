"""One training run (config + seed), as specified in docs/EXPERIMENT_PLAN.md.

    python -m racing.train --name pilot --rays 17 --depth 2 --width 16 --seed 0 --gens 600
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np

from .ga import evolve
from .inputs import Calibration, InputSpec
from .network import Population, layer_sizes
from .sets import load as load_sets
from .sim import evaluate, fitness_of, get_track, rollout
from .track import MAX_DIFFICULTY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION = os.path.join(ROOT, "data", "calibration.json")

POP = 100
TRACKS_PER_GEN = 4
REHEARSAL = 2      # of the 4 tracks, this many are drawn uniformly from [0, current level]
DIFF_STEP = 0.025
ADVANCE_STREAK = 3
SNAPSHOT_EVERY = 25
SNAPSHOT_KEEP = 8          # elites stored per snapshot
SAT_EPS = 1.0
SAT_PATIENCE = 8           # snapshot evaluations (= 200 generations)


def source_hashes():
    h = {}
    for f in sorted(os.listdir(os.path.join(ROOT, "racing"))):
        if f.endswith(".py"):
            with open(os.path.join(ROOT, "racing", f), "rb") as fh:
                h[f] = hashlib.sha1(fh.read()).hexdigest()[:12]
    return h


def make_spec(cfg, calib):
    return InputSpec(cfg["rays"], calib, reduction=cfg.get("reduction"), k=cfg.get("k"), seed=cfg["seed"])


def run(cfg, out_dir):
    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    calib = Calibration.load(CALIBRATION)
    V1 = load_sets()["V1"]
    spec = make_spec(cfg, calib)
    sizes = layer_sizes(spec.input_dim, cfg["depth"], cfg["width"])
    ss = np.random.SeedSequence(cfg["seed"])
    rng_ga, rng_tracks, rng_sensor = [np.random.default_rng(s) for s in ss.spawn(3)]

    cfg = dict(cfg, layer_sizes=sizes, params=int(sum((i + 1) * o for i, o in zip(sizes[:-1], sizes[1:]))),
               preprocessing_params=spec.preprocessing_params, source_hashes=source_hashes())
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    pop = Population(POP, sizes, rng=rng_ga)
    difficulty, streak, last_advance = 0.0, 0, 0
    reached_full_at = None
    best_v1, since_best = -np.inf, 0
    sim_seconds = 0.0
    hist_f = open(os.path.join(out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["gen", "difficulty", "best_fitness", "mean_fitness", "median_fitness",
                   "finish_rate", "best_finished_all", "sim_seconds", "wall_seconds"])
    eval_f = open(os.path.join(out_dir, "eval_v1.csv"), "w", newline="")
    ev = csv.writer(eval_f)
    ev.writerow(["gen", "difficulty", "champion_rank", "v1_score", "v1_finish_rate", "best_v1_so_far"])
    t_start = time.time()
    gen = 0
    stop_reason = "budget"
    while True:
        seeds = rng_tracks.integers(0, 2**31, size=TRACKS_PER_GEN)
        # rehearsal: keep practising easier (wider) roads so the population does
        # not forget them once the curriculum has moved on (pilot v1 finding)
        levels = [difficulty] * (TRACKS_PER_GEN - REHEARSAL) + \
            [float(np.round(rng_tracks.uniform(0, difficulty) / DIFF_STEP) * DIFF_STEP) for _ in range(REHEARSAL)]
        fits, finished = [], []
        for s, lvl in zip(seeds, levels):
            res = rollout(pop, spec, get_track(int(s), lvl), rng_sensor)
            fits.append(fitness_of(res))
            finished.append(res["finished"])
            sim_seconds += res["sim_time"]
        fit = np.mean(fits, axis=0)
        finished = np.array(finished)
        best = int(np.argmax(fit))
        all3 = bool(finished[:, best].all())
        streak = streak + 1 if all3 else 0
        hist.writerow([gen, round(difficulty, 3), f"{fit.max():.1f}", f"{fit.mean():.1f}", f"{np.median(fit):.1f}",
                       f"{finished.mean():.3f}", int(all3), f"{sim_seconds:.0f}", f"{time.time() - t_start:.0f}"])
        hist_f.flush()

        if streak >= ADVANCE_STREAK and difficulty < MAX_DIFFICULTY:
            difficulty = round(min(MAX_DIFFICULTY, difficulty + DIFF_STEP), 4)
            streak = 0
            last_advance = gen
            if difficulty >= 1.0 and reached_full_at is None:
                reached_full_at = gen + 1

        order = np.argsort(fit)[::-1]
        gen += 1
        if gen % SNAPSHOT_EVERY == 0:
            elites = pop.subset(order[:SNAPSHOT_KEEP])
            r = evaluate(elites, spec, V1, rng_seed=cfg["seed"])
            score = r["score"].mean(axis=0)
            champ = int(np.argmax(score))
            elites.save(os.path.join(out_dir, "snapshots", f"gen_{gen:05d}.npz"),
                        meta={"gen": gen, "difficulty": difficulty, "champion": champ,
                              "v1_scores": score.tolist()})
            # saturation: only counted once the F1 level (1.0) is reached, and
            # a curriculum advance also counts as improvement
            if difficulty >= 1.0:
                if score[champ] > best_v1 + SAT_EPS or gen - last_advance < SNAPSHOT_EVERY:
                    best_v1, since_best = max(best_v1, score[champ]), 0
                else:
                    since_best += 1
            ev.writerow([gen, difficulty, champ, f"{score[champ]:.2f}",
                         f"{r['finished'][:, champ].mean():.3f}", f"{best_v1:.2f}"])
            eval_f.flush()
            print(f"[{cfg['name']} s{cfg['seed']}] gen {gen} diff {difficulty:.2f} "
                  f"V1 {score[champ]:.1f} finish {r['finished'][:, champ].mean():.0%} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
            if cfg.get("saturate") and since_best >= SAT_PATIENCE:
                stop_reason = "saturated"
                break
        if cfg.get("saturate"):
            if gen >= cfg.get("max_gens", 3000):
                stop_reason = "censored"
                break
        elif gen >= cfg["gens"]:
            break
        pop, _, _ = evolve(pop, fit, rng_ga)

    hist_f.close()
    eval_f.close()
    with open(os.path.join(out_dir, "status.json"), "w") as f:
        json.dump({"stop_reason": stop_reason, "generations": gen, "reached_full_difficulty_at": reached_full_at,
                   "final_difficulty": difficulty, "best_v1": best_v1 if np.isfinite(best_v1) else None,
                   "wall_seconds": time.time() - t_start, "sim_seconds": sim_seconds}, f, indent=1)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rays", type=int, default=17)
    ap.add_argument("--reduction", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--saturate", action="store_true")
    ap.add_argument("--max-gens", type=int, default=3000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = {"name": a.name, "rays": a.rays, "reduction": a.reduction, "k": a.k, "depth": a.depth,
           "width": a.width, "seed": a.seed, "gens": a.gens, "saturate": a.saturate, "max_gens": a.max_gens}
    out = a.out or os.path.join(ROOT, "experiments", "results", a.name, f"seed{a.seed}")
    run(cfg, out)


if __name__ == "__main__":
    main()

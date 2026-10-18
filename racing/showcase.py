"""The show model: the best recipe found by the experiments, trained further
on specially selected extreme tracks, with its own detailed log.

  phase 0  special tracks: generate many extreme tracks (difficulty 2.5 and
           3.0, beyond every experiment) and keep the ones the reference
           controller has to drive slowest (tightest, most technical); separate
           pools for training / validation / show test
  phase 1  base = behavior cloning + DAgger, XL 2x64, teacher at 75 % grip,
           2x training (experiment C winner); continue DAgger on the special
           tracks (+ rehearsal over all levels) with all data kept
  phase 2  GA fine-tuning from the BC model: the population starts as small
           mutations of it, fitness rewards finishing fast (no teacher any
           more), champion chosen on validation tracks so it never gets worse
  final    BC base, BC-special and BC+GA all sit test T and the show tracks

    python -m racing.showcase --seed 0 --deadline <unix time>
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from .bc import MLP, collect
from .ga import evolve
from .inputs import Calibration, InputSpec
from .network import Population
from .sets import load as load_sets
from .sim import evaluate, fitness_of, get_track, reference_speed, rollout
from .train import CALIBRATION, ROOT, source_hashes

OUT = os.path.join(ROOT, "experiments", "results", "showcase")
SPECIAL = os.path.join(ROOT, "data", "special_tracks.json")
BASE = os.path.join(ROOT, "experiments", "results", "bc_t75_long", "S9_XL_2x64", "seed0", "station_2.00.npz")
POP = 100
SIGMAS = (0.003, 0.01, 0.03)       # small steps: fine-tuning a good model, not searching from scratch
EVAL_EVERY = 20


def build_special(per_level=200, levels=(2.5, 3.0), keep=(60, 8), show=20, first_seed=700000):
    """Hardest extreme tracks by reference lap speed (lower = harder): for each
    level the slowest `keep` tracks of `per_level` candidates go to training /
    validation; the show test is the slowest `show` tracks at the top level,
    from a separate seed range."""
    if os.path.exists(SPECIAL):
        return json.load(open(SPECIAL))
    from concurrent.futures import ProcessPoolExecutor

    def hardest(lvl, seeds, n):
        with ProcessPoolExecutor(3) as ex:
            v = list(ex.map(reference_speed, seeds, [lvl] * len(seeds)))
        ok = sorted([(s, x) for s, x in zip(seeds, v) if np.isfinite(x)], key=lambda t: t[1])
        return [{"seed": s, "difficulty": lvl, "v_ref": float(x)} for s, x in ok[:n]], len(ok), \
            float(np.median([x for _, x in ok]))

    out = {"train": [], "val": [], "stats": {}}
    rng = np.random.default_rng(0)
    for i, lvl in enumerate(levels):
        items, n_ok, med = hardest(lvl, list(range(first_seed + 1000 * i, first_seed + 1000 * i + per_level)), sum(keep))
        rng.shuffle(items)
        out["train"] += items[:keep[0]]
        out["val"] += items[keep[0]:]
        out["stats"][str(lvl)] = {"candidates": per_level, "drivable": n_ok, "v_ref_median_all": med,
                                  "v_ref_median_kept": float(np.median([x["v_ref"] for x in items]))}
    top = levels[-1]
    out["show"], n_ok, med = hardest(top, list(range(first_seed + 50000, first_seed + 50000 + per_level)), show)
    out["stats"]["show"] = {"level": top, "candidates": per_level, "drivable": n_ok, "v_ref_median_all": med}
    with open(SPECIAL, "w") as f:
        json.dump(out, f, indent=1)
    return out


def score_on(pop, spec, items, seed):
    r = evaluate(pop, spec, items, rng_seed=seed)
    return r["score"].mean(axis=0), r["finished"].mean(axis=0), r


def summarize(r, items):
    f = r["finished"][:, 0].astype(bool)
    sp = r["speed"][:, 0]
    return {"finish": float(f.mean()), "score": float(r["score"][:, 0].mean()),
            "lap_speed": float(sp[f].mean()) if f.any() else None,
            "per_track": [{"seed": it["seed"], "difficulty": it["difficulty"], "finished": int(fi),
                           "speed": round(float(s), 2), "score": round(float(sc), 1)}
                          for it, fi, s, sc in zip(items, f, sp, r["score"][:, 0])]}


def run(seed, deadline, bc_iters=25, base_path=BASE, out_root=OUT):
    out = os.path.join(out_root, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    special = build_special()
    sets = load_sets()
    calib = Calibration.load(CALIBRATION)
    bcfg = json.load(open(os.path.join(os.path.dirname(base_path), "config.json")))   # sensors of the base model
    spec = InputSpec(bcfg["rays"], calib, reduction=bcfg.get("reduction"), k=bcfg.get("k"), seed=seed)
    val = special["val"] + [x for x in sets["VS7"] if x["difficulty"] == 2.0]   # extreme 2.5 / 3.0 + normal 2.0
    rng = np.random.default_rng(seed)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"seed": seed, "base": os.path.relpath(base_path, ROOT),
                   "rays": bcfg["rays"], "reduction": bcfg.get("reduction"), "k": bcfg.get("k"), "pop": POP, "sigmas": SIGMAS,
                   "bc_iters": bc_iters, "eval_every": EVAL_EVERY, "deadline": deadline,
                   "special_tracks": special["stats"],
                   "source_hashes": source_hashes()}, f, indent=1)

    # ---------------------------------------------------------- phase 1: BC
    base, meta = Population.load(base_path)
    w, b = base.get(meta["champion"])
    net = MLP(base.sizes, rng)
    net.load_state((w, b))
    base1 = net.population()
    s0, f0, _ = score_on(base1, spec, val, seed)
    say(f"phase 1 start: BC base on validation  score {s0[0]:.1f}  finish {f0[0]:.0%}")
    with open(os.path.join(out, "phase1_bc.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter", "samples", "train_loss", "val_score", "val_finish", "best"])
        X_all, Y_all = [], []
        best, best_state = s0[0], net.state()

        def special_track():
            item = special["train"][rng.integers(len(special["train"]))]
            return get_track(item["seed"], item["difficulty"])

        def rehearsal_track():
            return get_track(int(rng.integers(900000, 990000)), float(np.round(rng.uniform(0, 3) / 0.025) * 0.025))

        # fine-tuning a trained network on a little new data overwrites it (first
        # attempt: validation 166 -> 8 after one iteration), so start from a broad
        # teacher-driven dataset over all levels and use a 10x smaller learning rate
        for k in range(16):
            X, Y = collect(net, spec, special_track() if k < 8 else rehearsal_track(), rng, expert_drives=True,
                           grip_margin=0.75, brake_margin=0.62)
            X_all.append(X)
            Y_all.append(Y)
        for it in range(1, bc_iters + 1):
            for k in range(6):   # 4 special training tracks + 2 rehearsal tracks anywhere in [0, 3], network drives
                X, Y = collect(net, spec, special_track() if k < 4 else rehearsal_track(), rng, expert_drives=False,
                               grip_margin=0.75, brake_margin=0.62)
                X_all.append(X)
                Y_all.append(Y)
            loss = net.fit(np.concatenate(X_all), np.concatenate(Y_all), 2000, rng, lr=3e-4)
            s, f, _ = score_on(net.population(), spec, val, seed)
            if s[0] > best:
                best, best_state = s[0], net.state()
            wr.writerow([it, sum(len(x) for x in X_all), f"{loss:.4f}", f"{s[0]:.2f}", f"{f[0]:.3f}", f"{best:.2f}"])
            fh.flush()
            say(f"phase 1 BC it {it}: loss {loss:.4f}  val score {s[0]:.1f}  finish {f[0]:.0%}  best {best:.1f}")
    net.load_state(best_state)
    bc_model = net.population()
    bc_model.save(os.path.join(out, "bc_special.npz"), meta={"gen": 0, "champion": 0, "val_score": best})

    # ---------------------------------------------------------- phase 2: GA
    pop = Population.from_individuals([bc_model.get(0)] * POP, bc_model.sizes)
    pop, _, _ = evolve(pop, np.zeros(POP), rng, sigmas=SIGMAS)   # elite copies + small mutations
    champ_score, gen = best, 0
    champ = bc_model
    hist = open(os.path.join(out, "phase2_ga.csv"), "w", newline="")
    hw = csv.writer(hist)
    hw.writerow(["gen", "best_fitness", "mean_fitness", "finish_rate", "best_finished_all", "best_lap_speed",
                 "best_sigma", "val_score", "val_finish", "champion_score", "wall_seconds"])
    t0 = time.time()
    sigma = np.zeros(POP)
    say(f"phase 2 start: GA fine-tuning, population {POP}, sigmas {SIGMAS}")
    while time.time() < deadline:
        picks = [special["train"][rng.integers(len(special["train"]))] for _ in range(3)]
        tracks = [get_track(it["seed"], it["difficulty"]) for it in picks]
        tracks.append(get_track(int(rng.integers(900000, 990000)), float(np.round(rng.uniform(0, 3) / 0.025) * 0.025)))
        fits, fin, spd = [], [], []
        for tr in tracks:
            res = rollout(pop, spec, tr, rng)
            fits.append(fitness_of(res))
            fin.append(res["finished"])
            spd.append(np.where(res["finished"], res["speed"], np.nan))
        fit = np.mean(fits, axis=0)
        fin = np.array(fin)
        bi = int(np.argmax(fit))
        gen += 1
        vs, vf = "", ""
        if gen % EVAL_EVERY == 0:
            order = np.argsort(fit)[::-1][:8]
            elites = pop.subset(order)
            s, f, _ = score_on(elites, spec, val, seed)
            j = int(np.argmax(s))
            vs, vf = f"{s[j]:.2f}", f"{f[j]:.3f}"
            if s[j] > champ_score:
                champ_score, champ = s[j], elites.subset([j])
                champ.save(os.path.join(out, "bc_ga.npz"), meta={"gen": gen, "champion": 0, "val_score": float(s[j])})
            say(f"phase 2 gen {gen}: best fitness {fit[bi]:.0f}  finish {fin.mean():.0%}  "
                f"val {s[j]:.1f} (finish {f[j]:.0%})  champion {champ_score:.1f}")
        hw.writerow([gen, f"{fit[bi]:.1f}", f"{fit.mean():.1f}", f"{fin.mean():.3f}", int(fin[:, bi].all()),
                     f"{np.nanmean(np.array(spd)[:, bi]) if fin[:, bi].any() else float('nan'):.2f}",
                     f"{sigma[bi]:.3f}", vs, vf, f"{champ_score:.2f}", f"{time.time() - t0:.0f}"])
        hist.flush()
        pop, _, sigma = evolve(pop, fit, rng, sigmas=SIGMAS)
    hist.close()
    if not os.path.exists(os.path.join(out, "bc_ga.npz")):
        champ.save(os.path.join(out, "bc_ga.npz"), meta={"gen": 0, "champion": 0, "val_score": float(champ_score)})

    # ---------------------------------------------------------- final tests
    say("final tests")
    results = {}
    for name, model in (("bc_base", base1), ("bc_special", bc_model), ("bc_ga", champ)):
        results[name] = {}
        for set_name, items in (("T", sets["T"]), ("show", special["show"])):
            r = evaluate(model, spec, items, rng_seed=777 + seed)
            results[name][set_name] = summarize(r, items)
        say(f"{name:10s}  T finish {results[name]['T']['finish']:.0%}  "
            f"show finish {results[name]['show']['finish']:.0%}  "
            f"show lap speed {results[name]['show']['lap_speed'] or 0:.1f} m/s")
    results["teacher_show_v_ref"] = [it["v_ref"] for it in special["show"]]
    results["ga_generations"] = gen
    with open(os.path.join(out, "final.json"), "w") as f:
        json.dump(results, f)
    say("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--bc-iters", type=int, default=25)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--base", default=BASE, help="BC model to start from (its config.json sets the sensors)")
    ap.add_argument("--out", default=OUT, help="results folder")
    a = ap.parse_args()
    if a.build_only:
        s = build_special()
        print(json.dumps(s["stats"], indent=1))
        return
    run(a.seed, a.deadline, a.bc_iters, a.base, a.out)


if __name__ == "__main__":
    main()

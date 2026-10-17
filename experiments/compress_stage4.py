"""Stage 4: compress the final models of stage 3 and let every compressed
model sit the same test (T, 7 levels x 30 tracks) as the trained ones.

Sources: final model (station 2.0 champion) of every XL 2x64 and L 2x32 seed.
Variants:
  prune to width w (every hidden layer) by importance / least squares / random
  quantise weights to b bits (per-row scale)
  prune (least squares) + quantise
No fine-tuning (no time); the comparison is "big model compressed after
training" vs "small model trained from scratch" at the same shape / size.

    python experiments/compress_stage4.py --jobs 8
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "stage4")
SOURCES = {"S9_XL_2x64": [32, 16, 8, 4], "S9_L_2x32": [16, 8, 4]}
BITS = [8, 6, 4, 3, 2]
COMBO = [(16, 4), (8, 4)]          # (width, bits) for prune(ls) + quantise


def calibration_inputs(pop1, spec, sets):
    """Inputs the original network sees while driving the calibration tracks
    (C at 1.0 + three VS7 tracks at 0.5 / 1.5 / 2.0; never the test set T)."""
    from racing.sim import get_track, rollout
    items = sets["C"] + [it for lvl in (0.5, 1.5, 2.0)
                         for it in [x for x in sets["VS7"] if x["difficulty"] == lvl][:3]]
    X = []
    for it in items:
        r = rollout(pop1, spec, get_track(it["seed"], it["difficulty"]), np.random.default_rng(it["seed"]),
                    record_activations=True)
        X.append(r["activations"][0])
    X = np.concatenate(X).astype(np.float64)
    return X[np.random.default_rng(0).choice(len(X), min(len(X), 20000), replace=False)]


def run_source(run_dir):
    from racing import compress as C
    from racing.inputs import Calibration, InputSpec
    from racing.network import Population
    from racing.sets import load as load_sets
    from racing.sim import evaluate
    size_key = os.path.basename(os.path.dirname(run_dir))
    seed = int(run_dir[-1])
    out_path = os.path.join(OUT, size_key, f"seed{seed}.json")
    if os.path.exists(out_path):
        return out_path, "skip"
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    calib = Calibration.load(os.path.join(ROOT, "data", "calibration.json"))
    spec = InputSpec(cfg["rays"], calib, seed=cfg["seed"])
    sets = load_sets()
    pop, meta = Population.load(os.path.join(run_dir, "station_2.00.npz"))
    src = pop.subset([meta["champion"]])
    net = src.get(0)
    X = calibration_inputs(src, spec, sets)

    variants = [{"name": "original", "method": "original", "width": src.sizes[1], "bits": 32,
                 "net": C.copy_net(net), "size_bits": C.size_bits(net)}]
    for b in BITS:
        q, _ = C.quantise(net, b)
        variants.append({"name": f"quant{b}", "method": "quantise", "width": src.sizes[1], "bits": b,
                         "net": q, "size_bits": C.quantised_size(net, b)})
    rng = np.random.default_rng(seed)
    for w in SOURCES[size_key]:
        for m in ("importance", "ls", "random"):
            p = C.prune(net, w, X, method=m, rng=rng)
            variants.append({"name": f"prune_{m}_{w}", "method": f"prune_{m}", "width": w, "bits": 32,
                             "net": p, "size_bits": C.size_bits(p)})
    for w, b in COMBO:
        if w in SOURCES[size_key]:
            p = C.prune(net, w, X, method="ls")
            q, _ = C.quantise(p, b)
            variants.append({"name": f"prune_ls_{w}+quant{b}", "method": "prune_ls+quantise", "width": w,
                             "bits": b, "net": q, "size_bits": C.quantised_size(p, b)})
    for v in variants:
        v["open_loop_rms"] = C.open_loop_error(net, v["net"], X)

    # every variant of the same shape drives together as one population
    T = sets["T"]
    results = []
    for shape in sorted({tuple(x.shape for x in v["net"][0]) for v in variants}):
        group = [v for v in variants if tuple(x.shape for x in v["net"][0]) == shape]
        sizes = [shape[0][1]] + [s[0] for s in shape]
        gp = Population.from_individuals([([w.astype(np.float32) for w in v["net"][0]],
                                           [b.astype(np.float32) for b in v["net"][1]]) for v in group], sizes)
        r = evaluate(gp, spec, T, rng_seed=777 + seed)
        for i, v in enumerate(group):
            results.append({k: v[k] for k in ("name", "method", "width", "bits", "size_bits", "open_loop_rms")} | {
                "params": int(sum(x.size for x in v["net"][0]) + sum(x.size for x in v["net"][1])),
                "difficulty": [it["difficulty"] for it in T],
                "finished": r["finished"][:, i].astype(int).tolist(),
                "score": r["score"][:, i].round(2).tolist(),
                "speed": r["speed"][:, i].round(3).tolist()})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"source": os.path.relpath(run_dir, ROOT), "seed": seed, "variants": results}, f)
    orig = next(x for x in results if x["name"] == "original")
    return out_path, f"{len(results)} variants, original finish {np.mean(orig['finished']):.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    runs = [d for k in SOURCES for d in sorted(glob.glob(os.path.join(RES, "stage3", k, "seed*")))
            if os.path.exists(os.path.join(d, "final_test.json"))]
    print(len(runs), "source models", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for path, msg in ex.map(run_source, runs):
            print(os.path.relpath(path, ROOT), msg, flush=True)


if __name__ == "__main__":
    main()

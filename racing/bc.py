"""Behavior cloning + DAgger: the same networks, trained by gradient descent
to copy the reference controller instead of by the genetic algorithm.

Protocol mirrors the GA runs (racing/train.py) so results are comparable:
  * same inputs (9 raw rays + 4 car-state values), same layer sizes, same
    tanh network stored as a Population of one -> same test code
  * the same seven stations 0.5 ... 2.0; at each station the network trains
    until its score on the station's validation tracks (VS7) stops improving
    by more than 1 point for 4 iterations, then sits the station test (T at
    that level); after the last station the final model re-takes all of T
  * every iteration drives 4 tracks: 2 at the station level + 2 rehearsal
    tracks uniform in [0, station]  (as in the GA)

One DAgger iteration
  1. drive the 4 tracks (the reference drives in the first iteration of a
     station, the network itself afterwards) and at every control step
     record the input x and the reference controller's command for that
     exact state (the "teacher's label", even where the network went astray)
  2. add them to the station's dataset (aggregation)
  3. train with Adam on the whole station dataset

Loss (outputs y in [-1, 1], see Population.controls):
  steer      (y0 - s)^2
  throttle   (y1 - (2 t - 1))^2
  brake      (y2 - b)^2 if b > 0 else max(y2, 0)^2   (any y2 <= 0 means no brake)
  handbrake  max(y3, 0)^2                            (the reference never uses it)

    python -m racing.bc --size 2 32 --seed 0
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from .inputs import Calibration, InputSpec
from .network import Population, layer_sizes
from .physics import CarParams, CarSwarm
from .reference import ReferenceDriver
from .sets import load as load_sets
from .sim import evaluate, get_track
from .train import CALIBRATION, ROOT, source_hashes, summary, test_model

TRACKS_PER_ITER = 4
REHEARSAL = 2
STEPS_PER_ITER = 1500
BATCH = 256
LR = 3e-3
SAT_EPS = 1.0
PATIENCE = 4
CAP = 60                    # iterations per station
TRACK_SEED0 = 600000        # training tracks: a seed range used by no evaluation set


class MLP:
    """tanh network with plain numpy backprop + Adam."""

    def __init__(self, sizes, rng):
        self.sizes = sizes
        self.W = [rng.normal(0, 1 / np.sqrt(i), (o, i)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [rng.normal(0, 0.1, o) for o in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        acts = [X]
        for W, b in zip(self.W, self.b):
            acts.append(np.tanh(acts[-1] @ W.T + b))
        return acts

    def loss_grad(self, X, Y):
        acts = self.forward(X)
        y = acts[-1]
        s, thr, brk = Y[:, 0], Y[:, 1], Y[:, 2]
        d = np.zeros_like(y)                      # dLoss / dy
        d[:, 0] = y[:, 0] - s
        d[:, 1] = y[:, 1] - (2 * thr - 1)
        d[:, 2] = np.where(brk > 0, y[:, 2] - brk, np.maximum(y[:, 2], 0))
        d[:, 3] = np.maximum(y[:, 3], 0)
        loss = float(np.mean(np.sum(d * d, axis=1)))
        n = len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        delta = 2 * d / n * (1 - y * y)           # back through tanh
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = delta.T @ acts[l]             # (out, in) outer products summed over the batch
            gb[l] = delta.sum(axis=0)
            if l:
                delta = (delta @ self.W[l]) * (1 - acts[l] * acts[l])
        return loss, gW + gb

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self.W + self.b
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)

    def fit(self, X, Y, steps, rng):
        losses = []
        for _ in range(steps):
            idx = rng.integers(0, len(X), min(BATCH, len(X)))
            loss, g = self.loss_grad(X[idx], Y[idx])
            self.adam(g, LR)
            losses.append(loss)
        return float(np.mean(losses[-100:]))

    def population(self):
        pop = Population.from_individuals([([W.astype(np.float32) for W in self.W],
                                            [b.astype(np.float32) for b in self.b])], self.sizes)
        return pop

    def state(self):
        return [W.copy() for W in self.W], [b.copy() for b in self.b]

    def load_state(self, st):
        self.W = [W.copy() for W in st[0]]
        self.b = [b.copy() for b in st[1]]


def collect(net, spec, track, rng, expert_drives, grip_margin=0.85, brake_margin=0.7):
    """One car on one track. Returns inputs and the teacher's labels."""
    cars = CarSwarm(1, track, num_sensors=spec.num_rays, sensor_noise=0.02, dropout=0.01, rng=rng)
    teacher = ReferenceDriver(track, CarParams(), grip_margin=grip_margin, brake_margin=brake_margin)
    X, Y = [], []
    while cars.active.any():
        x = spec.build(cars.sensor_readings(), cars)
        label = np.stack(teacher.act(cars), axis=1)
        X.append(x[0])
        Y.append(label[0])
        if expert_drives:
            cars.step(*[label[:, k] for k in range(4)])
        else:
            y = net.forward(x)[-1]
            cars.step(*Population.controls(y))
    return np.array(X), np.array(Y)


def run(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    calib = Calibration.load(CALIBRATION)
    sets = load_sets()
    spec = InputSpec(cfg["rays"], calib, seed=cfg["seed"])
    sizes = layer_sizes(spec.input_dim, cfg["depth"], cfg["width"])
    ss = np.random.SeedSequence(cfg["seed"])
    rng_init, rng_tracks, rng_sensor, rng_batch = [np.random.default_rng(s) for s in ss.spawn(4)]
    net = MLP(sizes, rng_init)
    cfg = dict(cfg, method="behavior_cloning_dagger", layer_sizes=sizes,
               params=int(sum((i + 1) * o for i, o in zip(sizes[:-1], sizes[1:]))),
               preprocessing_params=spec.preprocessing_params, source_hashes=source_hashes(),
               hyper={"tracks_per_iter": TRACKS_PER_ITER, "rehearsal": REHEARSAL, "steps_per_iter": STEPS_PER_ITER,
                      "batch": BATCH, "lr": LR, "patience": PATIENCE, "cap": CAP})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    hist_f = open(os.path.join(out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["iter", "station", "station_iter", "expert_drove", "samples", "train_loss", "vs_score",
                   "vs_finish", "best_vs", "wall_seconds"])
    ev_f = open(os.path.join(out_dir, "eval_v1.csv"), "w", newline="")
    ev = csv.writer(ev_f)
    ev.writerow(["gen", "difficulty", "champion_rank", "v1_score", "v1_finish_rate", "best_v1_so_far"])
    t0 = time.time()
    it = 0
    station_log, station_tests = [], []
    track_seed = TRACK_SEED0 + 10000 * cfg["seed"]
    X_all, Y_all = [], []           # every station's data, kept when aggregate == "all"
    for level in cfg["stations"]:
        vs = [x for x in sets["VS7"] if abs(x["difficulty"] - level) < 1e-6]
        X_st, Y_st = [], []
        best, best_state, since, k, best_it = -np.inf, None, 0, 0, None
        entered = it
        while True:
            expert = k == 0
            levels = [level] * (TRACKS_PER_ITER - REHEARSAL) + \
                [float(np.round(rng_tracks.uniform(0, level) / 0.025) * 0.025) for _ in range(REHEARSAL)]
            for lvl in levels:
                X, Y = collect(net, spec, get_track(track_seed, lvl), rng_sensor, expert,
                               cfg.get("teacher_grip", 0.85), cfg.get("teacher_brake", 0.7))
                track_seed += 1
                X_st.append(X)
                Y_st.append(Y)
            Xa, Ya = np.concatenate(X_all + X_st), np.concatenate(Y_all + Y_st)
            loss = net.fit(Xa, Ya, cfg.get("steps", STEPS_PER_ITER), rng_batch)
            pop = net.population()
            r = evaluate(pop, spec, vs, rng_seed=cfg["seed"])
            score = float(r["score"].mean())
            v1 = evaluate(pop, spec, sets["V1"], rng_seed=cfg["seed"])
            it += 1
            k += 1
            if score > best + SAT_EPS:
                best, best_state, since, best_it = score, net.state(), 0, it
                pop.save(os.path.join(out_dir, f"station_{level:.2f}.npz"),
                         meta={"gen": it, "difficulty": level, "champion": 0, "station": level, "vs_scores": [score]})
            else:
                since += 1
            hist.writerow([it, level, k, int(expert), len(Xa), f"{loss:.4f}", f"{score:.2f}",
                           f"{r['finished'].mean():.3f}", f"{best:.2f}", f"{time.time() - t0:.0f}"])
            hist_f.flush()
            ev.writerow([it, level, 0, f"{v1['score'].mean():.2f}", f"{v1['finished'].mean():.3f}", ""])
            ev_f.flush()
            print(f"[bc {cfg['name']} s{cfg['seed']}] station {level:.2f} it {k} samples {len(Xa)} loss {loss:.4f} "
                  f"VS {score:.1f} best {best:.1f} since {since} ({time.time() - t0:.0f}s)", flush=True)
            if since >= cfg.get("patience", PATIENCE) or k >= CAP:
                break
        # continue from the best model of this station (as the GA keeps its champion)
        net.load_state(best_state)
        if cfg.get("aggregate") == "all":   # standard DAgger: the dataset keeps growing over all stations
            X_all += X_st
            Y_all += Y_st
        station_log.append({"level": level, "entered_gen": entered, "best_gen": best_it, "left_gen": it,
                            "best_vs": best, "outcome": "saturated" if since >= cfg.get("patience", PATIENCE)
                            else "censored"})
        path = os.path.join(out_dir, f"station_{level:.2f}.npz")
        t = test_model(path, spec, [x for x in sets["T"] if abs(x["difficulty"] - level) < 1e-6], cfg["seed"])
        station_tests.append(dict(level=level, left_gen=it, **t))
        with open(os.path.join(out_dir, "station_tests.json"), "w") as f:
            json.dump(station_tests, f)
        print(f"[bc {cfg['name']} s{cfg['seed']}] TEST station {level:.2f}: {summary(t)}", flush=True)
    t = test_model(path, spec, sets["T"], cfg["seed"])
    with open(os.path.join(out_dir, "final_test.json"), "w") as f:
        json.dump(t, f)
    hist_f.close()
    ev_f.close()
    with open(os.path.join(out_dir, "status.json"), "w") as f:
        json.dump({"stop_reason": "stations_done", "generations": it, "wall_seconds": time.time() - t0,
                   "stations": station_log}, f, indent=1)
    print(f"[bc {cfg['name']} s{cfg['seed']}] done, final {summary(t)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rays", type=int, default=9)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stations", default="0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    ap.add_argument("--teacher-grip", type=float, default=0.85, help="share of tyre grip the teacher uses")
    ap.add_argument("--teacher-brake", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=STEPS_PER_ITER)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--aggregate", default="station", choices=["station", "all"],
                    help="train on this station's data only (like the GA) or on all data so far (standard DAgger)")
    a = ap.parse_args()
    cfg = {"name": a.name, "rays": a.rays, "depth": a.depth, "width": a.width, "seed": a.seed,
           "stations": [float(x) for x in a.stations.split(",")], "aggregate": a.aggregate,
           "teacher_grip": a.teacher_grip, "teacher_brake": a.teacher_brake, "steps": a.steps, "patience": a.patience}
    run(cfg, os.path.join(ROOT, "experiments", "results", a.name, f"seed{a.seed}"))


if __name__ == "__main__":
    main()

"""Loading trained drivers from experiment results."""
import csv
import glob
import json
import os

import numpy as np

from racing.inputs import Calibration, InputSpec
from racing.network import Population

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION = os.path.join(ROOT, "data", "calibration.json")


def best_snapshot(run_dir):
    """Snapshot with the highest V1 score (falls back to the latest)."""
    path = os.path.join(run_dir, "eval_v1.csv")
    snaps = sorted(glob.glob(os.path.join(run_dir, "snapshots", "gen_*.npz")))
    if not snaps:
        raise FileNotFoundError(f"no snapshots in {run_dir}")
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = max(rows, key=lambda r: float(r["v1_score"]))
            cand = os.path.join(run_dir, "snapshots", f"gen_{int(best['gen']):05d}.npz")
            if os.path.exists(cand):
                return cand
    return snaps[-1]


class Driver:
    """A trained network plus the input pipeline it was trained with."""

    def __init__(self, run_dir, snapshot=None, which="champion"):
        with open(os.path.join(run_dir, "config.json")) as f:
            self.config = json.load(f)
        self.snapshot = snapshot or best_snapshot(run_dir)
        elites, meta = Population.load(self.snapshot)
        self.meta = meta
        if which == "champion":
            self.pop = elites.subset([meta.get("champion", 0)])
        else:
            self.pop = elites
        calib = Calibration.load(CALIBRATION)
        c = self.config
        # show-model runs store a shorter config: 9 raw rays, no name
        self.spec = InputSpec(c.get("rays", 9), calib, reduction=c.get("reduction"), k=c.get("k"), seed=c["seed"])
        name = c.get("name") or os.path.relpath(run_dir, os.path.join(ROOT, "experiments", "results"))
        self.name = f"{name} (seed {c['seed']}, gen {meta.get('gen', '?')})"

    @property
    def input_labels(self):
        n = self.spec.sensor_dim
        prefix = "PC" if self.spec.proj is not None else "R"
        return [f"{prefix}{i + 1}" for i in range(n)] + ["vx", "vy", "r", "δ"]


def list_runs(stage_glob="*/*"):
    out = []
    for cfg in sorted(glob.glob(os.path.join(ROOT, "experiments", "results", stage_glob, "seed*", "config.json"))):
        d = os.path.dirname(cfg)
        if glob.glob(os.path.join(d, "snapshots", "*.npz")):
            out.append(d)
    return out

"""Evaluation track sets (EXPERIMENT_PLAN section 3).

Every evaluation track is checked with the reference controller; a track it
cannot finish is replaced by the next seed. The verified lists are cached in
data/eval_sets.json so every run uses exactly the same tracks.

    python -m racing.sets        # build / verify the cache
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "eval_sets.json")

TEST_LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SPEC = {
    # name: [(difficulty, first_seed, count), ...]
    "V1": [(1.0, 100000, 4), (1.5, 110000, 4), (2.0, 120000, 4)],
    "V2": [(1.0, 200000, 8), (1.5, 210000, 8), (2.0, 220000, 8)],
    "C": [(1.0, 300000, 10)],
    # station validation: decides saturation at each curriculum station
    "VS": [(1.0, 130000, 8), (1.5, 140000, 8), (2.0, 150000, 8)],
    "T": [(lvl, 400000 + 1000 * i, 30) for i, lvl in enumerate(TEST_LEVELS)],
}


def build():
    """Verify and cache every set; sets already in the cache are kept as is."""
    from .sim import reference_speed
    import numpy as np
    sets = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            sets = json.load(f)
    for name, parts in SPEC.items():
        if name in sets:
            continue
        items = []
        for diff, first, count in parts:
            seed, got, rejected = first, 0, 0
            while got < count:
                v = reference_speed(seed, diff)
                if np.isfinite(v):
                    items.append({"seed": seed, "difficulty": diff, "v_ref": float(v)})
                    got += 1
                else:
                    rejected += 1
                seed += 1
            print(f"{name} d={diff}: {count} tracks, {rejected} rejected", flush=True)
        sets[name] = items
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(sets, f, indent=1)
    return sets


def load():
    if not os.path.exists(CACHE):
        return build()
    with open(CACHE) as f:
        sets = json.load(f)
    if any(name not in sets for name in SPEC):
        return build()
    return sets


if __name__ == "__main__":
    build()

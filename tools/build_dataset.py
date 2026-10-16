"""Data pipeline: collect -> statistics -> clean -> before/after -> calibration.

1. COLLECT  The reference controller drives the calibration tracks (set C)
   three times each with random disturbances (steering/speed upsets) so the
   data also covers off-line and recovery states. Every control step (30 Hz)
   is one row: 17 raw ray ranges, car state, controller command, context.
   Sensors use the fault model of a real range sensor: 2 % noise, max-range
   readings when nothing is hit, and 1 % dropouts (a dropout reads 0).
   The logger keeps recording for 1 s after a crash, like a real data logger.
2. STATISTICS  count / mean / SD / min / max per variable, share of each
   data type (road segment, speed band, action, ray reading type).
3. CLEAN  complete / fixable (repaired) / incomplete (dropped).
4. BEFORE/AFTER  distributions and the PCA eigenvalues before and after.
5. CALIBRATION  per-ray mean/std + PCA basis used by every experiment.

    python tools/build_dataset.py
"""
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from racing.inputs import Calibration, FULL_RAYS, SENSOR_RANGE, encode
from racing.physics import CarParams, CarSwarm
from racing.reference import ReferenceDriver
from racing.track import generate_track

C_SEEDS = range(300000, 300010)
LAPS_PER_TRACK = 3
DROPOUT = 0.01
NOISE = 0.02
OUT = os.path.join(ROOT, "data")
RAY_COLS = [f"ray{i:02d}" for i in range(FULL_RAYS)]
COLS = (["episode", "track_seed", "t", "progress_m", "lap_frac"] + RAY_COLS
        + ["vx", "vy", "yaw_rate", "steer_angle", "cmd_steer", "cmd_throttle", "cmd_brake", "cmd_handbrake",
           "lateral_offset", "curvature_radius", "status"])


def collect():
    rows = []
    params = CarParams()
    episode = 0
    for seed in C_SEEDS:
        track = generate_track(seed=seed, difficulty=1.0)
        for lap in range(LAPS_PER_TRACK):
            rng = np.random.default_rng(seed * 10 + lap)
            car = CarSwarm(1, track, params=params, num_sensors=FULL_RAYS, sensor_noise=0, rng=rng)
            drv = ReferenceDriver(track, params)
            upset_until, upset_steer, upset_speed = -1.0, 0.0, 1.0
            next_upset = rng.uniform(1.5, 4.0)
            post_crash = 0
            while True:
                if car.active[0]:
                    steer, thr, brk, hb = drv.act(car)
                    if car.time >= next_upset:
                        upset_until = car.time + rng.uniform(0.3, 1.0)
                        upset_steer = rng.uniform(-0.25, 0.25)
                        upset_speed = rng.uniform(1.0, 1.15)
                        next_upset = upset_until + rng.uniform(2.0, 5.0)
                    if car.time < upset_until:
                        steer = np.clip(steer + upset_steer, -1, 1)
                        thr = np.clip(thr * upset_speed + (upset_speed - 1) * 3, 0, 1)
                    cmd = (steer[0], thr[0], brk[0], hb[0])
                else:
                    cmd = (0.0, 0.0, 0.0, 0.0)
                # raw sensor with the fault model, no repair (cleaning happens later)
                d = car.sensor_distances()[0]
                d = np.clip(d * (1 + rng.normal(0, NOISE, d.shape)), 0, SENSOR_RANGE)
                d = np.where(rng.random(d.shape) < DROPOUT, 0.0, d)
                j = int(np.floor(car.idx[0])) % len(track["centerline"])
                status = "ok" if car.active[0] else ("crashed" if car.crashed[0] else "ended")
                rows.append([episode, seed, round(car.time, 4), car.progress[0], car.progress[0] / track["length"],
                             *d, car.vx[0], car.vy[0], car.r[0], car.delta[0], *cmd,
                             car.lateral[0], min(track["curvature_radius"][j], 1e4), status])
                if car.active[0]:
                    car.step(np.array([cmd[0]]), np.array([cmd[1]]), np.array([cmd[2]]), np.array([cmd[3]]))
                    car.time = round(car.time, 6)
                else:
                    post_crash += 1
                    car.time += car.dt
                    if status != "crashed" or post_crash >= 30:
                        break
            episode += 1
    return rows


def to_arrays(rows):
    data = {c: np.array([r[i] for r in rows]) for i, c in enumerate(COLS)}
    return data


def segment_type(radius):
    return np.select([radius >= 300, radius >= 100, radius >= 40], ["straight", "fast corner", "medium corner"],
                     "slow corner / hairpin")


def describe(data, mask=None):
    mask = np.ones(len(data["t"]), bool) if mask is None else mask
    out = {}
    for c in RAY_COLS + ["vx", "vy", "yaw_rate", "steer_angle", "cmd_throttle", "cmd_brake", "lateral_offset"]:
        v = data[c][mask].astype(float)
        out[c] = {"count": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
                  "min": float(v.min()), "max": float(v.max())}
    return out


def shares(labels):
    u, c = np.unique(labels, return_counts=True)
    return {str(k): {"count": int(n), "percent": round(100 * n / len(labels), 2)} for k, n in zip(u, c)}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = collect()
    data = to_arrays(rows)
    n = len(rows)
    rays = np.stack([data[c] for c in RAY_COLS], axis=1).astype(float)

    with gzip.open(os.path.join(OUT, "driving_dataset_raw.csv.gz"), "wt") as f:
        f.write(",".join(COLS) + "\n")
        for r in rows:
            f.write(",".join(f"{x:.5g}" if isinstance(x, (float, np.floating)) else str(x) for x in r) + "\n")

    # ---------------------------------------------------- classify rows
    status = data["status"]
    vx = data["vx"].astype(float)
    t = data["t"].astype(float)
    numeric = np.stack([data[c].astype(float) for c in COLS if c not in ("status",)], axis=1)
    non_finite = ~np.all(np.isfinite(numeric), axis=1)
    key = np.stack([data["episode"].astype(float), t], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    duplicate = np.ones(n, bool)
    duplicate[first] = False
    after_crash = status != "ok"
    low_speed = vx < 5.0
    warmup = t < 0.5
    drop = after_crash | low_speed | non_finite | duplicate | warmup

    dropout = rays == 0.0
    no_return = rays >= SENSOR_RANGE * 0.999
    fixable = ~drop & (dropout.any(axis=1) | no_return.any(axis=1))
    complete = ~drop & ~fixable

    reasons = {"after crash / off track": int(after_crash.sum()), "speed < 5 m/s": int((low_speed & ~after_crash).sum()),
               "first 0.5 s (warm-up)": int((warmup & ~after_crash & ~low_speed).sum()),
               "NaN / inf": int(non_finite.sum()), "duplicate timestamp": int(duplicate.sum())}

    # ---------------------------------------------------- repair
    clean_rays = rays.copy()
    ep = data["episode"].astype(int)
    for i in np.where(~drop & dropout.any(axis=1))[0]:
        for k in np.where(dropout[i])[0]:
            # hold the ray's last valid value in the same episode (what the
            # car's own sensor pipeline does), else average of neighbours
            prev = i - 1
            while prev >= 0 and ep[prev] == ep[i] and clean_rays[prev, k] == 0.0:
                prev -= 1
            if prev >= 0 and ep[prev] == ep[i]:
                clean_rays[i, k] = clean_rays[prev, k]
            else:
                nb = [clean_rays[i, j] for j in (k - 1, k + 1) if 0 <= j < FULL_RAYS and clean_rays[i, j] > 0]
                clean_rays[i, k] = np.mean(nb) if nb else SENSOR_RANGE
    keep = ~drop

    # ---------------------------------------------------- statistics
    curv = data["curvature_radius"].astype(float)
    speed_kmh = vx * 3.6
    speed_band = np.select([speed_kmh < 80, speed_kmh < 160, speed_kmh < 240], ["<80", "80-160", "160-240"], ">=240")
    thr, brk = data["cmd_throttle"].astype(float), data["cmd_brake"].astype(float)
    action = np.select([brk > 0.05, thr > 0.05], ["braking", "accelerating"], "coasting")
    reading_type = np.where(dropout, "dropout (0)", np.where(no_return, "no return (max range)", "valid")).ravel()

    summary = {
        "rows_total": n, "episodes": int(ep.max() + 1), "tracks": len(C_SEEDS),
        "columns": COLS,
        "row_quality": {
            "complete": {"count": int(complete.sum()), "percent": round(100 * complete.mean(), 2)},
            "fixable (repaired)": {"count": int(fixable.sum()), "percent": round(100 * fixable.mean(), 2)},
            "incomplete (dropped)": {"count": int(drop.sum()), "percent": round(100 * drop.mean(), 2)},
        },
        "drop_reasons": reasons,
        "ray_reading_types_before": shares(reading_type),
        "segment_type": shares(segment_type(curv[keep])),
        "speed_band_kmh": shares(speed_band[keep]),
        "action": shares(action[keep]),
        "stats_before": describe(data),
        "stats_after": describe({**data, **{c: clean_rays[:, i] for i, c in enumerate(RAY_COLS)}}, keep),
    }

    # ---------------------------------------------------- PCA before / after
    calib_before = Calibration.fit(encode(rays))
    calib_after = Calibration.fit(encode(clean_rays[keep]))
    summary["pca_eigenvalues_before"] = calib_before.eigvals.tolist()
    summary["pca_eigenvalues_after"] = calib_after.eigvals.tolist()
    summary["pca_explained_after"] = (np.cumsum(calib_after.eigvals) / calib_after.eigvals.sum()).tolist()
    calib_after.save(os.path.join(OUT, "calibration.json"))
    np.savez_compressed(os.path.join(OUT, "dataset_arrays.npz"), rays_raw=rays, rays_clean=clean_rays,
                        keep=keep, complete=complete, fixable=fixable, drop=drop, vx=vx,
                        vy=data["vy"].astype(float), yaw_rate=data["yaw_rate"].astype(float),
                        steer=data["cmd_steer"].astype(float), throttle=thr, brake=brk, curvature=curv,
                        episode=ep)
    with open(os.path.join(OUT, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    q = summary["row_quality"]
    print(f"rows {n}  complete {q['complete']['percent']}%  fixable {q['fixable (repaired)']['percent']}%  "
          f"dropped {q['incomplete (dropped)']['percent']}%")
    print("drop reasons", reasons)
    print("eigenvalues before", np.round(calib_before.eigvals[:6], 3))
    print("eigenvalues after ", np.round(calib_after.eigvals[:6], 3))
    print("explained (after)", np.round(summary["pca_explained_after"][:8], 3))


if __name__ == "__main__":
    main()

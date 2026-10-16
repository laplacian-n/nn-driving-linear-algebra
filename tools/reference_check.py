"""Drive the reference controller (no NN) over a set of tracks and report
completion and lap times.   python tools/reference_check.py [first_seed] [count] [difficulty]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from racing.physics import CarParams, CarSwarm
from racing.reference import ReferenceDriver
from racing.track import generate_track


def drive(track, params=CarParams()):
    car = CarSwarm(1, track, params=params, sensor_noise=0)
    drv = ReferenceDriver(track, params)
    while car.active.any():
        car.step(*drv.act(car))
    return bool(car.finished[0]), car.progress[0] / track["length"], car.lap_time[0]


if __name__ == "__main__":
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    diff = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    rows = []
    for seed in range(first, first + count):
        tr = generate_track(seed=seed, difficulty=diff)
        fin, prog, lap = drive(tr)
        rows.append((seed, fin, prog, lap, tr["length"] / lap if fin else np.nan))
        if not fin:
            print(f"  seed {seed}: crashed at {prog:.0%}", flush=True)
    fin = np.array([r[1] for r in rows])
    v = np.array([r[4] for r in rows])
    lap = np.array([r[3] for r in rows])
    print(f"finished {fin.sum()}/{len(rows)}  mean lap {np.nanmean(lap):.1f} s  "
          f"mean speed {np.nanmean(v):.1f} m/s ({np.nanmean(v) * 3.6:.0f} km/h)")

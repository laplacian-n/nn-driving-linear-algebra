"""Running networks on tracks: rollouts, fitness, race score."""
import numpy as np

from .network import Population
from .physics import CarParams, CarSwarm
from .reference import ReferenceDriver
from .track import generate_track

TIME_LIMIT = 120.0
FINISH_BONUS = 30.0  # fitness metres per second saved on a finished lap

_track_cache = {}
_ref_cache = {}


def get_track(seed, difficulty=1.0):
    key = (int(seed), round(float(difficulty), 4))
    if key not in _track_cache:
        if len(_track_cache) > 400:
            _track_cache.clear()
        _track_cache[key] = generate_track(seed=int(seed), difficulty=difficulty)
    return _track_cache[key]


def reference_speed(seed, difficulty=1.0):
    """Average speed of the reference controller on this track (m/s)."""
    key = (int(seed), round(float(difficulty), 4))
    if key not in _ref_cache:
        track = get_track(seed, difficulty)
        car = CarSwarm(1, track, sensor_noise=0)
        drv = ReferenceDriver(track, CarParams())
        while car.active.any():
            car.step(*drv.act(car))
        _ref_cache[key] = track["length"] / car.lap_time[0] if car.finished[0] else np.nan
    return _ref_cache[key]


def rollout(pop, spec, track, rng, noise=0.02, dropout=0.01, time_limit=TIME_LIMIT,
            record_activations=False):
    cars = CarSwarm(pop.n, track, num_sensors=spec.num_rays, sensor_noise=noise, dropout=dropout,
                    time_limit=time_limit, rng=rng)
    acts = []
    while cars.active.any():
        x = spec.build(cars.sensor_readings(), cars)
        if record_activations:
            y, a = pop.forward(x, return_activations=True)
            live = cars.active
            acts.append([layer[live] for layer in a])
        else:
            y = pop.forward(x)
        cars.step(*Population.controls(y))
    out = {
        "progress": np.minimum(cars.progress, track["length"]),
        "frac": np.minimum(cars.progress / track["length"], 1.0),
        "finished": cars.finished.copy(),
        "crashed": cars.crashed.copy(),
        "lap_time": cars.lap_time.copy(),
        "sim_time": cars.time,
        "length": track["length"],
    }
    if record_activations:
        out["activations"] = [np.concatenate([step[l] for step in acts]) for l in range(len(acts[0]))]
    return out


def fitness_of(result):
    lap = np.where(result["finished"], result["lap_time"], TIME_LIMIT)
    return result["progress"] + FINISH_BONUS * np.where(result["finished"], TIME_LIMIT - lap, 0.0)


def race_score(result, v_ref):
    """100 x lap fraction, + 100 x (average speed / reference speed) if the
    lap was finished. Finishing always beats not finishing; no ceiling."""
    v = np.where(result["finished"], result["length"] / np.where(result["finished"], result["lap_time"], 1), 0)
    return 100 * result["frac"] + np.where(result["finished"], 100 * v / v_ref, 0.0)


def evaluate(pop, spec, seeds, difficulty=1.0, rng_seed=0, noise=0.02, dropout=0.01):
    """Every individual of pop on every track in seeds. Returns per-track
    arrays of shape (n_tracks, pop.n)."""
    rows = {"score": [], "frac": [], "finished": [], "lap_time": [], "v_ref": []}
    for i, s in enumerate(seeds):
        track = get_track(s, difficulty)
        res = rollout(pop, spec, track, np.random.default_rng([rng_seed, int(s)]), noise=noise, dropout=dropout)
        vref = reference_speed(s, difficulty)
        rows["score"].append(race_score(res, vref))
        rows["frac"].append(res["frac"])
        rows["finished"].append(res["finished"])
        rows["lap_time"].append(res["lap_time"])
        rows["v_ref"].append(np.full(pop.n, vref))
    return {k: np.array(v) for k, v in rows.items()}

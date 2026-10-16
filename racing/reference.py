"""Reference controller (no neural network): pure-pursuit steering along the
centerline + a curvature-based speed profile with braking points.

Used to (1) check that every evaluation track is drivable, (2) generate the
on-distribution calibration dataset, (3) give a reference lap time.
"""
import numpy as np

from .physics import G


def speed_profile(track, params, grip_margin=0.85, brake_margin=0.7):
    """Max cornering speed at every centerline sample, then a backward pass so
    the car can actually brake down to it in time.

    Cornering limit with downforce:  v^2 / R = mu_eff (g + c v^2),
    c = rho ClA / (2 m)  ->  v = sqrt(mu_eff g / (1/R - mu_eff c))."""
    p = params
    mu = p.mu * grip_margin
    c = 0.5 * p.rho * p.cla / p.mass
    curv = 1.0 / np.maximum(track["curvature_radius"], 1e-3)
    # smooth curvature a little so single-sample spikes don't dominate
    k = 5
    curv = np.convolve(np.concatenate([curv[-k:], curv, curv[:k]]), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    denom = curv - mu * c
    v = np.where(denom > 1e-9, np.sqrt(mu * G / np.maximum(denom, 1e-9)), 100.0)
    v = np.minimum(v, 95.0)
    ds = track["ds"]
    a_brake = brake_margin * p.mu * G
    n = len(v)
    for _ in range(2):  # circular track: two backward passes settle the seam
        for i in range(n - 1, -1, -1):
            nxt = v[(i + 1) % n]
            v[i] = min(v[i], np.sqrt(nxt * nxt + 2 * a_brake * ds))
    return v


class ReferenceDriver:
    def __init__(self, track, params, grip_margin=0.85, brake_margin=0.7):
        self.track = track
        self.p = params
        self.v_target = speed_profile(track, params, grip_margin, brake_margin)

    def act(self, cars):
        t = self.track
        n_pts, ds = len(t["centerline"]), t["ds"]
        v = cars.vx
        look = np.clip(0.35 * v + 6.0, 8.0, 40.0)
        tgt_idx = (np.floor(cars.idx + look / ds).astype(int)) % n_pts
        tgt = t["centerline"][tgt_idx]
        rel = tgt - cars.pos
        ang = np.arctan2(rel[:, 1], rel[:, 0]) - cars.psi
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        ld = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1.0)
        delta = np.arctan(2 * self.p.wheelbase * np.sin(ang) / ld)
        steer = np.clip(delta / self.p.max_steer, -1, 1)

        ahead = (np.floor(cars.idx).astype(int) + 3) % n_pts
        vt = self.v_target[ahead]
        err = vt - v
        throttle = np.clip(0.25 * err, 0, 1)
        brake = np.clip(-0.25 * err, 0, 1)

        # throttle control by friction circle (a driver's right foot): only
        # ask the rear tyres for the drive force their remaining grip allows
        p = self.p
        vv = np.maximum(v, 1.0)
        downforce = 0.5 * p.rho * p.cla * vv * vv
        grip_r = p.mu * p.rear_grip * (p.mass * G * p.a / p.wheelbase + downforce * (1 - p.aero_front))
        fy_r = p.mass * np.abs(v * cars.r) * p.a / p.wheelbase
        fx_room = np.sqrt(np.maximum((0.8 * grip_r) ** 2 - fy_r ** 2, 0.0))
        throttle = np.minimum(throttle, fx_room * vv / p.power)
        return steer, throttle, brake, np.zeros_like(steer)

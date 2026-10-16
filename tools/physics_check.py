"""Scripted (no neural network) physics sanity checks against F1 reference
values. Runs on a synthetic straight 'test pad' far wider than any car needs.

    python tools/physics_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from racing.physics import CarSwarm, G


def test_pad(length=20000.0, width=4000.0, ds=3.0):
    n = int(length / ds)
    x = np.arange(n) * ds
    cl = np.stack([x, np.zeros(n)], axis=1)
    tang = np.tile([1.0, 0.0], (n, 1))
    norm = np.tile([0.0, 1.0], (n, 1))
    hw = np.full(n, width / 2)
    return {"centerline": cl, "tangents": tang, "normals": norm, "half_width": hw,
            "left": cl + norm * hw[:, None], "right": cl - norm * hw[:, None],
            "ds": ds, "length": length * 10, "start_pos": cl[0].copy(), "start_angle": 0.0}


def make(n=1, v0=0.0):
    car = CarSwarm(n, test_pad(), start_speed=v0, stall_timeout=1e9, time_limit=1e9, sensor_noise=0)
    return car


def run(car, seconds, steer=0.0, throttle=0.0, brake=0.0, hb=0.0, until=None):
    log = []
    for _ in range(int(seconds * 30)):
        car.step(np.full(car.n, steer), np.full(car.n, throttle), np.full(car.n, brake), np.full(car.n, hb))
        log.append((car.time, car.vx.copy(), car.vy.copy(), car.r.copy()))
        if until is not None and until(car):
            break
    return log


def main():
    ok = True

    def report(name, value, lo, hi, unit):
        nonlocal ok
        good = lo <= value <= hi
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:<38} {value:8.2f} {unit:<6} (expected {lo}-{hi})")

    # top speed
    car = make()
    run(car, 60, throttle=1)
    report("top speed", car.vx[0] * 3.6, 330, 360, "km/h")

    # 0-100 km/h
    car = make()
    run(car, 10, throttle=1, until=lambda c: c.vx[0] >= 100 / 3.6)
    report("0-100 km/h", car.time, 2.3, 3.6, "s")

    # 0-200 km/h
    car = make()
    run(car, 20, throttle=1, until=lambda c: c.vx[0] >= 200 / 3.6)
    report("0-200 km/h", car.time, 4.0, 7.5, "s")

    # braking from 300 km/h: peak deceleration and distance to 100 km/h
    car = make(v0=300 / 3.6)
    x0 = car.pos[0, 0]
    v_prev, peak = car.vx[0], 0.0
    while car.vx[0] > 100 / 3.6:
        car.step(np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1))
        peak = max(peak, (v_prev - car.vx[0]) / car.dt / G)
        v_prev = car.vx[0]
    report("peak braking from 300 km/h", peak, 4.5, 6.5, "g")
    report("braking distance 300->100 km/h", car.pos[0, 0] - x0, 60, 110, "m")

    # steady-state lateral acceleration at 250 km/h: sweep steering, take the max
    best = 0.0
    for st in np.linspace(0.02, 0.4, 20):
        car = make(v0=250 / 3.6)
        log = run(car, 3, steer=st, throttle=0.35)
        _, vx, vy, r = log[-1]
        best = max(best, abs(vx[0] * r[0]) / G)
    report("max lateral accel @250 km/h (steady)", best, 3.5, 5.5, "g")

    # low-speed lateral grip (no downforce to speak of)
    best = 0.0
    for st in np.linspace(0.05, 1.0, 20):
        car = make(v0=70 / 3.6)
        log = run(car, 4, steer=st, throttle=0.12)
        _, vx, vy, r = log[-1]
        best = max(best, abs(vx[0] * r[0]) / G)
    report("max lateral accel @70 km/h (steady)", best, 1.4, 2.4, "g")

    # snap oversteer: full lock at speed -> big body slip angle
    car = make(v0=200 / 3.6)
    log = run(car, 1.5, steer=1.0, throttle=1.0)
    beta = max(abs(np.degrees(np.arctan2(vy[0], max(vx[0], 1)))) for _, vx, vy, _ in log)
    report("slip angle, full lock + full throttle @200", beta, 8, 180, "deg")

    # handbrake in a moderate corner -> tail steps out much more than without
    def corner_slip(hb):
        car = make(v0=100 / 3.6)
        run(car, 1.5, steer=0.12, throttle=0.25)
        log = run(car, 1.0, steer=0.12, throttle=0.0, hb=hb)
        return max(abs(np.degrees(np.arctan2(vy[0], max(vx[0], 1)))) for _, vx, vy, _ in log)
    no_hb, with_hb = corner_slip(0.0), corner_slip(1.0)
    report("slip angle without handbrake", no_hb, 0, 6, "deg")
    report("slip angle with handbrake", with_hb, 10, 180, "deg")

    # full lock at high speed while lifting: lift-off oversteer, but no full spin
    car = make(v0=250 / 3.6)
    log = run(car, 1.5, steer=1.0, throttle=0.0)
    beta = max(abs(np.degrees(np.arctan2(vy[0], max(vx[0], 1)))) for _, vx, vy, _ in log)
    report("lift-off oversteer, full lock @250 (no spin)", beta, 0, 45, "deg")

    # spin to a stop must stay finite
    car = make(v0=150 / 3.6)
    run(car, 8, steer=1.0, throttle=0.0, hb=1.0)
    finite = np.all(np.isfinite([car.vx[0], car.vy[0], car.r[0], car.pos[0, 0], car.pos[0, 1]]))
    report("spin to stop: finite state (1 = yes)", float(finite), 1, 1, "")

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()

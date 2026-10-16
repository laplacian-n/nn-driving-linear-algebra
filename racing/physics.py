"""Vectorised F1-like car physics for a whole population at once.

Every array has the population on its first axis, so one call advances every
car - no Python loop over cars.

Model (body frame: x forward, y left, yaw rate r counter-clockwise):
  * dynamic bicycle model, 3 DOF (vx, vy, r) + steered front wheel angle
  * Pacejka "magic formula" lateral tyre force per axle: grip rises with slip
    angle to a peak and then FALLS - past the peak at the rear axle the car
    snaps into oversteer, which is what makes a fast car's tail flick out
  * friction circle: longitudinal force used for drive/brake reduces the
    lateral force still available on that axle
  * longitudinal load transfer (braking loads the front, accelerating the rear)
  * aerodynamic drag and downforce, both ~ v^2 (grip grows with speed)
  * power-limited rear-wheel drive, traction-limited (wheelspin possible)
  * handbrake locks the rear wheels: the rear tyre only gives sliding friction
    opposite to its own velocity, so the tail lets go
  * below ~5 m/s blend into a kinematic bicycle model to avoid the slip-angle
    singularity at vx -> 0
"""
from dataclasses import dataclass

import numpy as np

G = 9.81
MIN_GAIN = 15.0  # m of track progress required per stall window

try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _raycast_kernel(pos, psi, offsets, left, right, base, w0, w1, max_range, active):
        """For every (car, ray): smallest t > 0 with  p + t d = a + u e,
        0 <= u <= 1, over the nearby wall segments of both walls.
        det = -dx*ey + dy*ex ;  t = (-wx*ey + wy*ex)/det ;  u = (dx*wy - dy*wx)/det"""
        n, s = pos.shape[0], offsets.shape[0]
        m = left.shape[0]
        out = np.full((n, s), max_range)
        for i in range(n):
            if not active[i]:
                continue
            px, py = pos[i, 0], pos[i, 1]
            for k in range(s):
                ang = psi[i] + offsets[k]
                dx, dy = np.cos(ang), np.sin(ang)
                best = max_range
                for off in range(w0, w1):
                    j0 = (base[i] + off) % m
                    j1 = (j0 + 1) % m
                    for wall in range(2):
                        if wall == 0:
                            ax, ay, bx, by = left[j0, 0], left[j0, 1], left[j1, 0], left[j1, 1]
                        else:
                            ax, ay, bx, by = right[j0, 0], right[j0, 1], right[j1, 0], right[j1, 1]
                        ex, ey = bx - ax, by - ay
                        det = -dx * ey + dy * ex
                        if abs(det) < 1e-9:
                            continue
                        wx, wy = ax - px, ay - py
                        t = (-wx * ey + wy * ex) / det
                        if t <= 0.0 or t >= best:
                            continue
                        u = (dx * wy - dy * wx) / det
                        if 0.0 <= u <= 1.0:
                            best = t
                out[i, k] = best
        return out
    @njit(cache=True, fastmath=True)
    def _physics_kernel(pos, psi, vx_a, vy_a, r_a, delta_a, ax_a, active, steer_target, throttle, brake,
                        handbrake, h, substeps, pv):
        """Same equations as CarSwarm._substep, one car at a time, skipping
        cars that are no longer driving. pv = _param_vector(params)."""
        (m, iz, a, b, hcg, mu, rear_grip, mu_slide, pb, pc, pe, rho, cda, cla, af, power, bf, roll,
         steer_rate, blo, bhi) = (pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], pv[7], pv[8], pv[9],
                                  pv[10], pv[11], pv[12], pv[13], pv[14], pv[15], pv[16], pv[17], pv[18],
                                  pv[19], pv[20])
        L = a + b
        for i in range(pos.shape[0]):
            if not active[i]:
                continue
            vx, vy, r, delta, ax_prev, ps = vx_a[i], vy_a[i], r_a[i], delta_a[i], ax_a[i], psi[i]
            px, py = pos[i, 0], pos[i, 1]
            thr, brk, hb, st = throttle[i], brake[i], handbrake[i], steer_target[i]
            for _ in range(substeps):
                dmax = steer_rate * h
                dd = st - delta
                if dd > dmax:
                    dd = dmax
                elif dd < -dmax:
                    dd = -dmax
                delta += dd
                v = vx if vx > 0.0 else 0.0
                q = 0.5 * rho * v * v
                down, drag = q * cla, q * cda
                transfer = m * ax_prev * hcg / L
                fz_f = max(m * G * b / L + down * af - transfer, 0.0)
                fz_r = max(m * G * a / L + down * (1 - af) + transfer, 0.0)
                grip_f, grip_r = mu * fz_f, mu * rear_grip * fz_r
                drive = thr * power / max(v, 1.0)
                bforce = brk * (grip_f + grip_r)
                fx_f = min(max(-bf * bforce, -grip_f), grip_f)
                fx_r = min(max(drive - (1 - bf) * bforce, -grip_r), grip_r)
                moving = min(max(v / 0.5, 0.0), 1.0)
                fx_f *= moving
                if fx_r < 0:
                    fx_r *= moving
                vxe = max(v, 0.5)
                alpha_f = np.arctan2(vy + a * r, vxe) - delta
                alpha_r = np.arctan2(vy - b * r, vxe)
                cap_f = np.sqrt(max(grip_f * grip_f - fx_f * fx_f, 0.0))
                cap_r = np.sqrt(max(grip_r * grip_r - fx_r * fx_r, 0.0))
                baf, bar = pb * alpha_f, pb * alpha_r
                fy_f = -cap_f * np.sin(pc * np.arctan(baf - pe * (baf - np.arctan(baf))))
                fy_r = -cap_r * np.sin(pc * np.arctan(bar - pe * (bar - np.arctan(bar))))
                vrx, vry = v, vy - b * r
                vr = np.sqrt(vrx * vrx + vry * vry) + 1e-6
                slide = mu_slide * fz_r * min(max(vr / 0.5, 0.0), 1.0)
                fx_r = (1 - hb) * fx_r - hb * slide * vrx / vr
                fy_r = (1 - hb) * fy_r - hb * slide * vry / vr
                rolling = roll * (fz_f + fz_r) * moving
                cd, sd = np.cos(delta), np.sin(delta)
                ax = (fx_r + fx_f * cd - fy_f * sd - drag - rolling) / m
                ay = (fy_r + fy_f * cd + fx_f * sd) / m
                r_dot = (a * (fy_f * cd + fx_f * sd) - b * fy_r) / iz
                nvx = max(vx + (ax + vy * r) * h, 0.0)
                nvy = vy + (ay - v * r) * h
                nr = r + r_dot * h
                w = min(max((nvx - blo) / (bhi - blo), 0.0), 1.0)
                r_kin = nvx * np.tan(delta) / L
                nr = w * nr + (1 - w) * r_kin
                nvy = w * nvy + (1 - w) * r_kin * b
                vx, vy, r, ax_prev = nvx, nvy, nr, ax
                ps += r * h
                c, s = np.cos(ps), np.sin(ps)
                px += (vx * c - vy * s) * h
                py += (vx * s + vy * c) * h
            vx_a[i], vy_a[i], r_a[i], delta_a[i], ax_a[i], psi[i] = vx, vy, r, delta, ax_prev, ps
            pos[i, 0], pos[i, 1] = px, py
except ImportError:  # pragma: no cover - NumPy fallback is used instead
    _raycast_kernel = None
    _physics_kernel = None


def _param_vector(p):
    return np.array([p.mass, p.yaw_inertia, p.a, p.b, p.cg_height, p.mu, p.rear_grip, p.mu_slide, p.pac_b,
                     p.pac_c, p.pac_e, p.rho, p.cda, p.cla, p.aero_front, p.power, p.brake_front, p.rolling,
                     p.steer_rate, p.blend_lo, p.blend_hi])


@dataclass(frozen=True)
class CarParams:
    mass: float = 798.0           # kg, incl. driver
    yaw_inertia: float = 1200.0   # kg m^2
    a: float = 1.95               # CG -> front axle, m
    b: float = 1.65               # CG -> rear axle, m
    cg_height: float = 0.30       # m
    length: float = 5.6           # body, m
    width: float = 2.0            # body, m
    mu: float = 1.6               # peak tyre friction coefficient (front)
    rear_grip: float = 1.10       # rear tyres are wider: rear mu = mu * rear_grip -> stable,
                                  # mildly understeering at the limit like a real setup
    mu_slide: float = 1.28        # locked-wheel sliding friction (0.8 mu)
    pac_b: float = 10.0
    pac_c: float = 1.9
    pac_e: float = 0.97
    rho: float = 1.225            # air density
    cda: float = 1.3              # drag coefficient x frontal area, m^2
    cla: float = 4.8              # downforce coefficient x area, m^2
    aero_front: float = 0.45      # share of downforce on the front axle
    power: float = 760e3          # W
    brake_front: float = 0.6      # brake balance
    rolling: float = 0.012        # rolling resistance coefficient
    max_steer: float = 0.35       # rad at the road wheels
    steer_rate: float = 1.2       # rad/s
    blend_lo: float = 2.0         # m/s, fully kinematic below
    blend_hi: float = 5.0         # m/s, fully dynamic above

    @property
    def wheelbase(self):
        return self.a + self.b


def pacejka(alpha, p):
    """Normalised lateral force (-1..1) for slip angle alpha (rad)."""
    ba = p.pac_b * alpha
    return np.sin(p.pac_c * np.arctan(ba - p.pac_e * (ba - np.arctan(ba))))


class CarSwarm:
    def __init__(self, n, track, params=CarParams(), num_sensors=17, sensor_fov_deg=200.0,
                 sensor_range=200.0, control_hz=30, substeps=4, start_speed=30.0,
                 stall_timeout=3.0, time_limit=120.0, sensor_noise=0.02, dropout=0.0,
                 rng=None):
        self.n = n
        self.p = params
        self.num_sensors = num_sensors
        self.sensor_range = sensor_range
        self.sensor_offsets = (np.radians(np.linspace(-sensor_fov_deg / 2, sensor_fov_deg / 2, num_sensors))
                               if num_sensors > 1 else np.zeros(1))
        self.dt = 1.0 / control_hz
        self.substeps = substeps
        self.start_speed = start_speed
        self.stall_timeout = stall_timeout
        self.time_limit = time_limit
        self.sensor_noise = sensor_noise
        self.dropout = dropout
        self.rng = np.random.default_rng() if rng is None else rng
        self.use_kernel = True
        self.reset(track)

    # ------------------------------------------------------------------ state
    def reset(self, track):
        self.track = track
        n = self.n
        self.pos = np.tile(track["start_pos"], (n, 1)).astype(float)
        self.psi = np.full(n, track["start_angle"], dtype=float)
        self.vx = np.full(n, float(self.start_speed))
        self.vy = np.zeros(n)
        self.r = np.zeros(n)
        self.delta = np.zeros(n)
        self.ax = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.finished = np.zeros(n, dtype=bool)
        self.crashed = np.zeros(n, dtype=bool)
        self.idx = np.zeros(n)              # continuous (unwrapped) centerline index
        self.progress = np.zeros(n)         # metres along the centerline, best so far
        self.lateral = np.zeros(n)          # signed offset from the centerline, m
        self.stall = np.zeros(n)
        self.stall_ref = np.zeros(n)       # progress at the last time the stall timer was reset
        self.time = 0.0
        self.lap_time = np.full(n, np.nan)
        self.last_reading = np.full((n, self.num_sensors), self.sensor_range)
        self.throttle = np.zeros(n)
        self.brake = np.zeros(n)
        self.handbrake = np.zeros(n)

    @property
    def active(self):
        return self.alive & ~self.finished

    @property
    def speed(self):
        return np.hypot(self.vx, self.vy)

    @property
    def slip_angle(self):
        """Body slip angle beta = atan(vy/vx): how sideways the car is moving."""
        return np.arctan2(self.vy, np.maximum(self.vx, 1.0))

    # --------------------------------------------------------------- controls
    def step(self, steer_target, throttle, brake, handbrake):
        """One control step (1/control_hz s). steer_target in [-1, 1] is the
        commanded wheel angle as a fraction of max_steer; pedals in [0, 1]."""
        p = self.p
        act = self.active
        if not np.any(act):
            return
        steer_target = np.clip(steer_target, -1, 1) * p.max_steer
        self.throttle = np.clip(throttle, 0, 1)
        self.brake = np.clip(brake, 0, 1)
        self.handbrake = np.clip(handbrake, 0, 1)
        h = self.dt / self.substeps
        m, L = p.mass, p.wheelbase
        if _physics_kernel is not None and self.use_kernel:
            _physics_kernel(self.pos, self.psi, self.vx, self.vy, self.r, self.delta, self.ax, act,
                            steer_target.astype(float), self.throttle.astype(float), self.brake.astype(float),
                            self.handbrake.astype(float), h, self.substeps, _param_vector(p))
        else:
            for _ in range(self.substeps):
                self._substep(steer_target, h, act, m, L)
        self.time += self.dt
        self._update_track_state(act)

    def _substep(self, steer_target, h, act, m, L):
        p = self.p
        # steering wheel moves at a limited rate
        self.delta += np.clip(steer_target - self.delta, -p.steer_rate * h, p.steer_rate * h) * act
        vx = np.maximum(self.vx, 0.0)
        delta = self.delta

        # aero + normal loads (longitudinal load transfer from last substep's ax)
        q = 0.5 * p.rho * vx * vx
        downforce = q * p.cla
        drag = q * p.cda
        transfer = m * self.ax * p.cg_height / L
        fz_f = np.maximum(m * G * p.b / L + downforce * p.aero_front - transfer, 0.0)
        fz_r = np.maximum(m * G * p.a / L + downforce * (1 - p.aero_front) + transfer, 0.0)
        grip_f, grip_r = p.mu * fz_f, p.mu * p.rear_grip * fz_r

        # longitudinal: power-limited drive (rear), brakes on both axles
        drive = self.throttle * p.power / np.maximum(vx, 1.0)
        brake_force = self.brake * (grip_f + grip_r)
        fx_f = np.clip(-p.brake_front * brake_force, -grip_f, grip_f)
        fx_r = np.clip(drive - (1 - p.brake_front) * brake_force, -grip_r, grip_r)
        moving = np.clip(vx / 0.5, 0.0, 1.0)           # no braking force once stopped
        fx_f *= moving
        fx_r = np.where(fx_r < 0, fx_r * moving, fx_r)

        # lateral: Pacejka inside the friction circle
        vxe = np.maximum(vx, 0.5)
        alpha_f = np.arctan2(self.vy + p.a * self.r, vxe) - delta
        alpha_r = np.arctan2(self.vy - p.b * self.r, vxe)
        cap_f = np.sqrt(np.maximum(grip_f ** 2 - fx_f ** 2, 0.0))
        cap_r = np.sqrt(np.maximum(grip_r ** 2 - fx_r ** 2, 0.0))
        fy_f = -cap_f * pacejka(alpha_f, p)
        fy_r = -cap_r * pacejka(alpha_r, p)

        # handbrake: locked rear wheel -> sliding friction against the rear
        # contact patch's own velocity (blended by how hard it is pulled)
        hb = self.handbrake
        vrx, vry = vx, self.vy - p.b * self.r
        vr = np.hypot(vrx, vry) + 1e-6
        slide = p.mu_slide * fz_r * np.clip(vr / 0.5, 0.0, 1.0)
        fx_r = (1 - hb) * fx_r - hb * slide * vrx / vr
        fy_r = (1 - hb) * fy_r - hb * slide * vry / vr

        rolling = p.rolling * (fz_f + fz_r) * moving
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        fx_total = fx_r + fx_f * cos_d - fy_f * sin_d - drag - rolling
        fy_total = fy_r + fy_f * cos_d + fx_f * sin_d
        ax = fx_total / m
        ay = fy_total / m
        r_dot = (p.a * (fy_f * cos_d + fx_f * sin_d) - p.b * fy_r) / p.yaw_inertia

        new_vx = np.maximum(self.vx + (ax + self.vy * self.r) * h, 0.0)
        new_vy = self.vy + (ay - vx * self.r) * h
        new_r = self.r + r_dot * h

        # low-speed: blend toward the kinematic bicycle model
        w = np.clip((new_vx - p.blend_lo) / (p.blend_hi - p.blend_lo), 0.0, 1.0)
        r_kin = new_vx * np.tan(delta) / L
        new_r = w * new_r + (1 - w) * r_kin
        new_vy = w * new_vy + (1 - w) * r_kin * p.b

        self.vx = np.where(act, new_vx, self.vx)
        self.vy = np.where(act, new_vy, self.vy)
        self.r = np.where(act, new_r, self.r)
        self.ax = np.where(act, ax, self.ax)
        self.psi += self.r * h * act
        c, s = np.cos(self.psi), np.sin(self.psi)
        self.pos[:, 0] += (self.vx * c - self.vy * s) * h * act
        self.pos[:, 1] += (self.vx * s + self.vy * c) * h * act

    # ------------------------------------------------- progress / collisions
    def _update_track_state(self, act):
        t = self.track
        cl, tang, norm, hw = t["centerline"], t["tangents"], t["normals"], t["half_width"]
        n_pts, ds = len(cl), t["ds"]
        base = np.floor(self.idx).astype(int)
        offs = np.arange(-4, 9)
        cand = base[:, None] + offs[None, :]                     # unwrapped
        cw = cand % n_pts
        d2 = np.sum((self.pos[:, None, :] - cl[cw]) ** 2, axis=2)
        k = np.argmin(d2, axis=1)
        j = cand[np.arange(self.n), k]
        jw = j % n_pts
        rel = self.pos - cl[jw]
        along = np.sum(rel * tang[jw], axis=1)
        lateral = np.sum(rel * norm[jw], axis=1)
        new_idx = j + np.clip(along / ds, -0.5, 0.5)
        self.idx = np.where(act, new_idx, self.idx)
        self.lateral = np.where(act, lateral, self.lateral)
        s_now = self.idx * ds
        # a car must gain at least MIN_GAIN metres of track every
        # stall_timeout seconds (5 m/s average) or it is retired - stops
        # crawling/circling cars from soaking up simulation time
        improved = s_now >= self.stall_ref + MIN_GAIN
        self.progress = np.where(act, np.maximum(self.progress, s_now), self.progress)

        # body rectangle projected on the road normal must stay inside the road
        c, s = np.cos(self.psi), np.sin(self.psi)
        nx, ny = norm[jw, 0], norm[jw, 1]
        extent = (np.abs(self.p.length / 2 * (c * nx + s * ny))
                  + np.abs(self.p.width / 2 * (-s * nx + c * ny)))
        off_road = np.abs(lateral) + extent > hw[jw]

        self.stall = np.where(improved, 0.0, self.stall + self.dt)
        self.stall_ref = np.where(improved, s_now, self.stall_ref)
        dead = act & (off_road | (self.stall > self.stall_timeout) | (self.time >= self.time_limit))
        self.crashed |= act & off_road
        self.alive &= ~dead

        done = act & ~dead & (self.progress >= t["length"])
        self.finished |= done
        self.lap_time = np.where(done, self.time, self.lap_time)

    # ---------------------------------------------------------------- sensors
    def sensor_distances(self, window=(-6, 80)):
        """Raw range readings (m) of every ray of every car. Ray vs wall
        segment = a 2x2 linear system per pair, solved by Cramer's rule.
        Only walls within a window of centerline samples around each car are
        tested. Uses the compiled kernel when numba is available, otherwise
        the equivalent batched NumPy version."""
        if _raycast_kernel is not None:
            t = self.track
            return _raycast_kernel(self.pos, self.psi, self.sensor_offsets, t["left"], t["right"],
                                   np.floor(self.idx).astype(np.int64), window[0], window[1],
                                   float(self.sensor_range), self.active)
        return self._sensor_distances_numpy(window)

    def _sensor_distances_numpy(self, window=(-6, 80)):
        t = self.track
        left, right = t["left"], t["right"]
        n_pts = len(left)
        base = np.floor(self.idx).astype(int)
        offs = np.arange(window[0], window[1])
        i0 = (base[:, None] + offs[None, :]) % n_pts                 # (n, w)
        i1 = (i0 + 1) % n_pts
        seg_a = np.concatenate([left[i0], right[i0]], axis=1)        # (n, 2w, 2)
        seg_b = np.concatenate([left[i1], right[i1]], axis=1)
        e = seg_b - seg_a                                            # segment direction
        ang = self.psi[:, None] + self.sensor_offsets[None, :]       # (n, s)
        dx, dy = np.cos(ang)[:, :, None], np.sin(ang)[:, :, None]    # (n, s, 1)
        ex, ey = e[:, None, :, 0], e[:, None, :, 1]                  # (n, 1, m)
        wx = seg_a[:, None, :, 0] - self.pos[:, None, None, 0]
        wy = seg_a[:, None, :, 1] - self.pos[:, None, None, 1]
        # solve  t*d - u*e = w  for (t, u):  det = -dx*ey + dy*ex
        det = -dx * ey + dy * ex
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        t_hit = (-wx * ey + wy * ex) * inv
        u_hit = (dx * wy - dy * wx) * inv
        hit = ok & (t_hit > 0) & (u_hit >= 0) & (u_hit <= 1)
        dist = np.where(hit, t_hit, np.inf).min(axis=2)
        return np.minimum(dist, self.sensor_range)

    def sensor_readings(self):
        """What the car's controller actually receives: noisy ranges with
        occasional dropouts (a dropout reads exactly 0), repaired by holding
        the ray's last valid value - the same cleaning step used for the
        dataset in the data pipeline."""
        d = self.sensor_distances()
        if self.sensor_noise > 0:
            d = d * (1 + self.rng.normal(0, self.sensor_noise, d.shape))
        d = np.clip(d, 0.0, self.sensor_range)
        if self.dropout > 0:
            fault = self.rng.random(d.shape) < self.dropout
            d = np.where(fault, 0.0, d)
            d = np.where(fault, self.last_reading, d)      # cleaning: hold last valid
        self.last_reading = d
        return d

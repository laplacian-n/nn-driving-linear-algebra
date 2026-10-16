import numpy as np


class CarSwarm:
    """State for the whole population, stored as arrays so every physics update
    is one vectorized operation instead of a per-car loop.

    Physics is an arcade *drift* model: velocity is a real 2D vector that is
    NOT locked to the heading. The heading (yaw) rotates with steering, but the
    velocity only realigns toward it as fast as tire grip allows - so when the
    heading swings faster than grip can pull the velocity around, the car
    slides. That single mechanism produces understeer (running wide at speed),
    oversteer, and (with the handbrake cutting grip) sustained drift, instead
    of the old point-mass model where velocity was always exactly along the
    nose and the car could not slide at all."""

    def __init__(self, n, track, num_sensors=16, sensor_fov_deg=200, sensor_max_dist=300,
                 stall_timeout=3.5):
        self.n = n
        self.num_sensors = num_sensors
        self.sensor_max_dist = sensor_max_dist
        self.sensor_offsets = np.radians(np.linspace(-sensor_fov_deg / 2, sensor_fov_deg / 2, num_sensors))

        # Real F1 body (world units ~= meters): ~5.6 m long, ~2.0 m wide.
        self.car_length = 5.6
        self.car_width = 2.0
        self.contact_margin = 0.15  # a body corner this close to a wall counts as a hit

        # Roughly F1-scaled: ~350 km/h top speed, ~1.4 g drive, ~5.6 g braking
        # (brakes far outmatch the engine, as in a real car), ~5 g lateral grip.
        self.max_speed = 97.0
        self.accel_rate = 14.0
        self.brake_rate = 55.0
        self.handbrake_decel = 18.0   # longitudinal drag from a locked rear axle
        self.turn_rate = 2.2          # steering-lock yaw rate at parking speed, rad/s
        self.wheel_turn_speed = 2.5   # how fast the wheel itself turns, per second
        self.drag = 26.0              # coasting / rolling resistance
        self.max_lateral_accel = 50.0  # tire+downforce grip that fights sideways slide, m/s^2
        self.handbrake_grip_loss = 0.85  # fraction of lateral grip the handbrake kills -> drift
        self.slide_scrub = 0.25       # forward speed bled off while sliding sideways (tire scrub)
        self.stall_timeout = stall_timeout

        self.reset(track)

    def reset(self, track):
        self.pos = np.tile(track["start_pos"], (self.n, 1)).astype(float)
        self.angle = np.full(self.n, track["start_angle"], dtype=float)
        self.vel = np.zeros((self.n, 2))
        self.speed = np.zeros(self.n)   # signed forward speed (vel . heading)
        self.slip = np.zeros(self.n)    # signed lateral speed (vel . left-normal) = how sideways it is
        self.wheel_angle = np.zeros(self.n)
        self.alive = np.ones(self.n, dtype=bool)
        self.finished = np.zeros(self.n, dtype=bool)
        self.progress = np.zeros(self.n)
        self.stall_timer = np.zeros(self.n)

    def freeze_finished(self, mask):
        """Cars that complete the lap stop being simulated (parked at the finish
        line) instead of driving on into a second lap. Without this a finisher's
        single-lap-bounded progress stops improving the moment it crosses the
        line and the stall-timeout - meant for idling/circling cars - kills it a
        few seconds later even though it just won."""
        self.finished |= mask

    def sensor_readings(self, track):
        # Ray-vs-segment hit test = a 2x2 linear system per segment via Cramer's
        # rule, batched over every (car, sensor, segment) triple at once with
        # broadcasting instead of a Python loop per car and per sensor.
        seg_a, seg_b = track["seg_a"], track["seg_b"]  # (m, 2) each
        ab = seg_b - seg_a
        ab_x, ab_y = ab[:, 0], ab[:, 1]  # (m,)

        ray_angles = self.angle[:, None] + self.sensor_offsets[None, :]  # (n, s)
        dir_x = np.cos(ray_angles)
        dir_y = np.sin(ray_angles)

        denom = (dir_x[:, :, None] * (-ab_y)[None, None, :]
                 - (-ab_x)[None, None, :] * dir_y[:, :, None])  # (n, s, m)
        valid = np.abs(denom) > 1e-9
        safe_denom = np.where(valid, denom, 1.0)

        rhs_x = seg_a[None, None, :, 0] - self.pos[:, None, None, 0]  # (n, 1, m)
        rhs_y = seg_a[None, None, :, 1] - self.pos[:, None, None, 1]

        t = (rhs_x * (-ab_y)[None, None, :] - (-ab_x)[None, None, :] * rhs_y) / safe_denom
        s_param = (dir_x[:, :, None] * rhs_y - dir_y[:, :, None] * rhs_x) / safe_denom

        hit = valid & (t >= 0) & (t <= self.sensor_max_dist) & (s_param >= 0) & (s_param <= 1)
        readings = np.where(hit, t, self.sensor_max_dist).min(axis=2)  # (n, s)
        return readings / self.sensor_max_dist

    def step(self, steer_rate, throttle, brake, handbrake, dt, track):
        """steer_rate in [-1, 1] turns the steering wheel (a rate - the wheel has
        its own turn speed). throttle/brake/handbrake in [0, 1] are independent
        pedals."""
        steer_rate = np.clip(steer_rate, -1, 1)
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        handbrake = np.clip(handbrake, 0, 1)
        active = (self.alive & ~self.finished).astype(float)

        self.wheel_angle += steer_rate * self.wheel_turn_speed * dt * active
        self.wheel_angle = np.clip(self.wheel_angle, -1, 1)

        # --- yaw: rotate the heading from steering -----------------------------
        # Yaw rate is capped by grip at speed (max_lateral / speed) so a fast car
        # physically cannot snap its nose around - that cap is exactly what makes
        # it run wide (understeer) unless it slows or drifts. A gentle low-speed
        # ramp keeps a near-parked car from spinning on the spot for free.
        fwd_speed = np.abs(self.speed)
        crawl_speed = self.max_speed * 0.06
        low_speed_ramp = np.clip(fwd_speed / crawl_speed, 0.1, 1.0)
        grip_limited_rate = self.max_lateral_accel / np.maximum(fwd_speed, 1.0)
        max_yaw_rate = np.minimum(self.turn_rate, grip_limited_rate) * low_speed_ramp
        self.angle += self.wheel_angle * max_yaw_rate * dt * active

        # --- decompose velocity onto the (now-rotated) heading -----------------
        # The velocity vector did NOT rotate with the heading; the mismatch is
        # the slip. Forward and lateral components are handled by different
        # forces below.
        heading = np.stack([np.cos(self.angle), np.sin(self.angle)], axis=1)     # (n, 2)
        left = np.stack([-np.sin(self.angle), np.cos(self.angle)], axis=1)       # (n, 2)
        fwd = np.sum(self.vel * heading, axis=1)   # signed forward speed
        lat = np.sum(self.vel * left, axis=1)      # signed lateral speed (slip)

        # --- longitudinal: engine / brakes / drag along the heading -----------
        drive = throttle * self.accel_rate - brake * self.brake_rate - handbrake * self.handbrake_decel
        fwd += drive * dt * active
        coasting = (throttle < 0.05) & (brake < 0.05) & (handbrake < 0.05)
        fwd -= self.drag * dt * coasting * active
        # sliding sideways scrubs off forward speed too (tires burning energy)
        fwd -= self.slide_scrub * np.abs(lat) * dt * active
        fwd = np.clip(fwd, 0.0, self.max_speed)

        # --- lateral: tire grip fights the sideways slide ---------------------
        # Grip removes up to (max_lateral_accel * dt) of lateral speed each step.
        # The handbrake kills most of that grip, so the slide persists = drift.
        grip = self.max_lateral_accel * (1.0 - self.handbrake_grip_loss * handbrake) * dt
        lat = np.sign(lat) * np.maximum(np.abs(lat) - grip, 0.0)

        # --- reassemble velocity, clamp total speed, move ---------------------
        self.vel = heading * fwd[:, None] + left * lat[:, None]
        speed_mag = np.linalg.norm(self.vel, axis=1)
        over = speed_mag > self.max_speed
        self.vel[over] *= (self.max_speed / speed_mag[over])[:, None]
        self.pos += self.vel * dt * active[:, None]

        self.speed = fwd
        self.slip = lat

        self._update_progress(track, dt)
        self._check_collisions(track)

    def _update_progress(self, track, dt):
        # Nearest-point lookup is ambiguous wherever the track runs close to a
        # different, arc-length-distant part of itself (every hairpin/chicane by
        # design), so the search is restricted to a window of indices around
        # where the car's own progress already says it is - only arc-length-near
        # candidates are considered, which also covers the start/finish seam.
        centerline = track["centerline"]
        cumdist = track["cumdist"]
        n = len(centerline)
        window = 40
        approx_idx = np.clip(np.searchsorted(cumdist, self.progress), 0, n - 1)
        offsets = np.arange(-window, window + 1)
        candidate_idx = (approx_idx[:, None] + offsets[None, :]) % n
        d = np.linalg.norm(self.pos[:, None, :] - centerline[candidate_idx], axis=2)
        nearest_idx = candidate_idx[np.arange(self.n), np.argmin(d, axis=1)]
        candidate = cumdist[nearest_idx]
        # The plausibility margin must clear the centerline's own sample spacing,
        # not just top-speed-per-frame; otherwise crossing into the next sample
        # can look like an implausible jump and get rejected, and once one real
        # update is rejected progress latches forever and reads as a false stall.
        max_step = max(self.max_speed * dt * 3, float(np.max(np.diff(cumdist))) * 2)
        plausible = candidate <= self.progress + max_step
        improved = plausible & (candidate > self.progress + 0.5)
        self.progress = np.where(plausible, np.maximum(self.progress, candidate), self.progress)

        # A car that circles or idles never improves progress, racks up stall
        # time, and gets killed instead of squatting as a "safe but useless"
        # strategy. Finished cars are exempt - they're frozen, not stalled.
        self.stall_timer = np.where(improved, 0.0, self.stall_timer + dt)
        self.alive &= self.finished | (self.stall_timer < self.stall_timeout)

    def _body_corners(self):
        """The four corners of each car's rectangular body, in world space."""
        hl, hw = self.car_length / 2, self.car_width / 2
        heading = np.stack([np.cos(self.angle), np.sin(self.angle)], axis=1)   # (n, 2)
        left = np.stack([-np.sin(self.angle), np.cos(self.angle)], axis=1)     # (n, 2)
        # (n, 4, 2): front-left, front-right, rear-left, rear-right
        signs = np.array([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])          # (4, 2)
        corners = (self.pos[:, None, :]
                   + signs[None, :, 0, None] * heading[:, None, :]
                   + signs[None, :, 1, None] * left[:, None, :])
        return corners  # (n, 4, 2)

    def _check_collisions(self, track):
        # Rectangular body: a car crashes when any of its four body corners
        # reaches a wall (point-to-segment distance under a small contact
        # margin), instead of a single centre point vs one collision radius -
        # so a long car clips an apex and a wide car brushes a wall on a
        # straight, both correctly.
        seg_a, seg_b = track["seg_a"], track["seg_b"]
        ab = seg_b - seg_a
        ab_len2 = np.sum(ab ** 2, axis=1) + 1e-9
        corners = self._body_corners().reshape(self.n * 4, 2)          # (4n, 2)
        ap = corners[:, None, :] - seg_a[None, :, :]                   # (4n, m, 2)
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab_len2[None, :], 0, 1)
        closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]   # (4n, m, 2)
        corner_min = np.min(np.linalg.norm(corners[:, None, :] - closest, axis=2), axis=1)  # (4n,)
        min_dist = corner_min.reshape(self.n, 4).min(axis=1)          # (n,)
        self.alive &= self.finished | (min_dist >= self.contact_margin)

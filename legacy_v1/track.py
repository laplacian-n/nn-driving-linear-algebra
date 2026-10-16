import numpy as np

# Cubic Bezier basis matrix: point(t) = [1 t t^2 t^3] . M . P
BEZIER_MATRIX = np.array([
    [1, 0, 0, 0],
    [-3, 3, 0, 0],
    [3, -6, 3, 0],
    [-1, 3, -3, 1],
], dtype=float)


def bezier_segment(p0, p1, p2, p3, num_samples):
    t = np.linspace(0, 1, num_samples, endpoint=False)
    t_powers = np.stack([np.ones_like(t), t, t ** 2, t ** 3], axis=1)
    control = np.array([p0, p1, p2, p3])
    return t_powers @ BEZIER_MATRIX @ control


def centripetal_catmull_rom_to_bezier(anchors, alpha=0.5):
    # Converts each (p1, p2) span of the closed anchor loop into cubic Bezier
    # control points using *centripetal* (sqrt-distance) knot spacing rather than
    # uniform spacing. With uniformly-spaced knots, a segment between unevenly
    # spaced anchors can overshoot into a small self-intersecting loop (a known
    # failure mode of "uniform" Catmull-Rom). Centripetal parameterization is
    # proven to never produce that loop/cusp, for any anchor placement.
    n = len(anchors)
    segments = []
    for i in range(n):
        p0 = anchors[(i - 1) % n]
        p1 = anchors[i]
        p2 = anchors[(i + 1) % n]
        p3 = anchors[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        m1 = (t2 - t1) * ((p1 - p0) / (t1 - t0) - (p2 - p0) / (t2 - t0) + (p2 - p1) / (t2 - t1))
        m2 = (t2 - t1) * ((p2 - p1) / (t2 - t1) - (p3 - p1) / (t3 - t1) + (p3 - p2) / (t3 - t2))
        segments.append((p1, p1 + m1 / 3, p2 - m2 / 3, p2))
    return segments


def _curvature_radius(centerline):
    # Discrete curvature via the circumradius of each point with its neighbors:
    # R = (|p1-p0| |p2-p1| |p2-p0|) / (2 |cross(p1-p0, p2-p1)|). A tight turn with
    # small R is where a *constant-width* offset would fold the boundary over
    # itself - handled below by narrowing the road locally instead of banning
    # tight turns outright.
    p0 = np.roll(centerline, 1, axis=0)
    p1 = centerline
    p2 = np.roll(centerline, -1, axis=0)
    v1, v2 = p1 - p0, p2 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    d1 = np.linalg.norm(v1, axis=1)
    d2 = np.linalg.norm(v2, axis=1)
    d02 = np.linalg.norm(p2 - p0, axis=1)
    return d1 * d2 * d02 / (2 * np.abs(cross) + 1e-9)


def _polyline_self_intersects(poly, min_index_gap=3):
    """True if a *closed* polyline crosses itself anywhere. The per-point
    curvature/width relationship only guards against a single bend folding
    its own two edges over each other locally - it says nothing about two
    unrelated stretches of the loop (e.g. either side of a tight hairpin,
    or just an S-curve wobbling too far) coming back around and crossing
    somewhere else entirely, which is what actually produces a boundary
    line that spikes into the road. Checked with plain segment-vs-segment
    intersection, vectorized over every pair of edges at once."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    n = len(poly)
    d1 = b - a
    denom = d1[:, None, 0] * d1[None, :, 1] - d1[:, None, 1] * d1[None, :, 0]
    valid = np.abs(denom) > 1e-9
    safe_denom = np.where(valid, denom, 1.0)
    rhs0 = a[None, :, 0] - a[:, None, 0]
    rhs1 = a[None, :, 1] - a[:, None, 1]
    t = (rhs0 * d1[None, :, 1] - rhs1 * d1[None, :, 0]) / safe_denom
    s = (rhs0 * d1[:, None, 1] - rhs1 * d1[:, None, 0]) / safe_denom
    hit = valid & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, n - gap)
    hit &= gap > min_index_gap
    return bool(np.any(hit))


def _carve_challenge_features(radii, rng, num_hairpins=1, num_uturns=1, num_scurves=1):
    """Deliberately carve hairpins, near-180 U-turns, and S-curves into the
    loop (on top of the base organic jitter), so those specific shapes are
    guaranteed every track instead of left to chance. Counts scale with
    track difficulty. Only the radius of a handful of anchors is touched -
    the adaptive road width and the curvature safety check downstream still
    apply, so these can't self-intersect either.
    """
    n = len(radii)
    radii = radii.copy()
    needed = num_hairpins + num_uturns + num_scurves
    picks = rng.choice(n, size=min(needed, n), replace=False)
    idx = 0
    for _ in range(num_hairpins):
        radii[picks[idx % len(picks)]] *= 0.65    # sharp elbow turn
        idx += 1
    for _ in range(num_uturns):
        radii[picks[idx % len(picks)]] *= 0.5     # pulled in tight enough to double back on itself
        idx += 1
    for _ in range(num_scurves):
        s_i = picks[idx % len(picks)]
        s_j = (s_i + 1) % n
        radii[s_i] *= 0.78                        # S-curve: two adjacent anchors kink
        radii[s_j] *= 1.28                        # opposite ways, so the curve bends, then reverses
        idx += 1
    return radii


def _build_track(num_anchors, base_radius, radius_jitter, angle_jitter_frac,
                  center, samples_per_segment, width, min_width, curvature_margin, seed,
                  num_hairpins=1, num_uturns=1, num_scurves=1, num_narrow=1):
    rng = np.random.default_rng(seed)
    spacing = 2 * np.pi / num_anchors
    # Anchors start evenly spaced around the loop and only wobble within a
    # fraction of that spacing - this is what stops two anchors from randomly
    # landing almost on top of each other (which produces a near-degenerate,
    # extremely tight-curvature mini-segment between them).
    base_angles = np.linspace(0, 2 * np.pi, num_anchors, endpoint=False)
    angles = base_angles + rng.uniform(-angle_jitter_frac, angle_jitter_frac, num_anchors) * spacing
    radii = base_radius + rng.uniform(-radius_jitter, radius_jitter, num_anchors)
    radii = _carve_challenge_features(radii, rng, num_hairpins, num_uturns, num_scurves)
    cx, cy = center
    anchors = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)

    segments = centripetal_catmull_rom_to_bezier(anchors)
    centerline = np.concatenate(
        [bezier_segment(*seg, samples_per_segment) for seg in segments], axis=0
    )

    tangents = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # Road width narrows automatically wherever the curve turns tight, instead
    # of banning tight corners from the track altogether. That's what lets the
    # track have real hairpins/chicanes without the boundary folding over
    # itself: width is capped at (curvature radius / margin), floored so the
    # road never gets absurdly thin.
    radius = _curvature_radius(centerline)
    half_w = np.clip(radius / curvature_margin, min_width / 2, width / 2)

    # Deliberately narrow stretches, independent of curvature - force the car
    # to actually thread a gap on what would otherwise be an easy straight,
    # rather than every narrow spot being explained by a sharp corner. Tapered
    # in/out with a smooth (Hann) window rather than snapping the width down
    # and back up, which would show up as a sharp step/notch in the boundary.
    n = len(centerline)
    narrow_len = max(16, n // 12)
    for _ in range(num_narrow):
        narrow_start = rng.integers(0, n)
        narrow_span = (narrow_start + np.arange(narrow_len)) % n
        taper = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(narrow_len) / narrow_len)
        narrow_target = min_width * 0.5
        half_w[narrow_span] = half_w[narrow_span] * (1 - taper) + narrow_target * taper

    left = centerline + normals * half_w[:, None]
    right = centerline - normals * half_w[:, None]
    self_intersects = _polyline_self_intersects(left) or _polyline_self_intersects(right)

    seg_a = np.concatenate([left, right], axis=0)
    seg_b = np.concatenate([np.roll(left, -1, axis=0), np.roll(right, -1, axis=0)], axis=0)

    diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    cumdist = np.concatenate([[0.0], np.cumsum(diffs)])

    start_dir = centerline[1] - centerline[0]
    start_angle = np.arctan2(start_dir[1], start_dir[0])

    return {
        "centerline": centerline,
        "left": left,
        "right": right,
        "seg_a": seg_a,
        "seg_b": seg_b,
        "cumdist": cumdist,
        "half_width": half_w,
        "width": width,
        "start_pos": centerline[0].copy(),
        "start_angle": start_angle,
        "min_curvature_radius": float(np.min(radius)),
        "self_intersects": self_intersects,
    }


def generate_track(num_anchors=13, base_radius=300, radius_jitter=60, angle_jitter_frac=0.25,
                    center=(500, 400), samples_per_segment=30, seed=None,
                    max_attempts=150, curvature_margin=1.15, difficulty=0.0):
    """Generate a closed-loop track. Road width narrows at tight corners so sharp
    turns are allowed; we still retry with a new anchor layout whenever a
    candidate either pinches tighter than min_width can safely support, or
    folds back and crosses itself anywhere in the loop (the "fang" spiking
    into the road that a purely local curvature check can't catch, since it
    comes from two unrelated stretches of the loop passing close to each
    other, not from any single bend being too sharp on its own).

    difficulty in [0, 1] scales up how many hairpins/U-turns/S-curves/narrow
    stretches get carved in AND narrows the whole road, for a training
    curriculum that starts wide and forgiving (a fresh random population can
    actually bootstrap on it) and ramps up to an F1-tight ribbon - a ~2 m car
    threading a road only a handful of car-widths across - once earlier tracks
    are mastered.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    num_hairpins = 1 + round(difficulty * 2)
    num_uturns = 1 + round(difficulty * 1)
    num_scurves = 1 + round(difficulty * 2)
    num_narrow = 1 + round(difficulty * 2)

    # Road width shrinks with difficulty: roomy at 0 so random cars survive
    # long enough to learn anything, F1-narrow at 1 (a 2 m car on an ~18 m road).
    width = 72.0 * (1 - difficulty) + 20.0 * difficulty
    min_width = 44.0 * (1 - difficulty) + 13.0 * difficulty

    base_seed = int(np.random.default_rng(seed).integers(0, 2 ** 31)) if seed is not None else None
    min_radius = min_width / 2 * curvature_margin

    best_track, best_radius = None, -np.inf
    fallback_track, fallback_radius = None, -np.inf
    for attempt in range(max_attempts):
        trial_seed = None if base_seed is None else base_seed + attempt
        track = _build_track(num_anchors, base_radius, radius_jitter, angle_jitter_frac,
                              center, samples_per_segment, width, min_width, curvature_margin,
                              trial_seed, num_hairpins, num_uturns, num_scurves, num_narrow)
        if not track["self_intersects"] and track["min_curvature_radius"] >= min_radius:
            return track
        if track["min_curvature_radius"] > fallback_radius:
            fallback_track, fallback_radius = track, track["min_curvature_radius"]
        # Self-intersecting candidates are never an acceptable fallback (that's
        # the actual visible defect) - only rank among the ones that don't.
        if not track["self_intersects"] and track["min_curvature_radius"] > best_radius:
            best_track, best_radius = track, track["min_curvature_radius"]

    # Every attempt turned out too tight or self-crossing somewhere (extremely
    # unlikely with the default parameters) - fall back to the least-bad
    # non-intersecting candidate seen, or the overall least-bad if every one
    # of the 80 attempts self-intersected.
    return best_track if best_track is not None else fallback_track

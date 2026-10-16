"""F1-style circuit generator.

A real circuit is mostly straights joined by corners of very different
severity - long flat-out straights, fast sweepers, tight hairpins, chicanes -
on a road of (almost) constant width. The old generator wobbled a smooth loop
around a circle, so every part of the lap looked and drove the same.

Construction:
  1. random points -> convex hull (gives the long straights)
  2. displace edge midpoints in/out (adds corners, chicanes, hairpins)
  3. scale to a target lap length
  4. at every vertex, cut the corner with a cubic Bezier that approximates a
     circular arc of a chosen radius (straight - corner - straight)
  5. resample uniformly by arc length, offset by the normals into two walls
Candidates that pinch, fold over themselves or run too close to another part
of the circuit are rejected and regenerated.
"""
import numpy as np

# Cubic Bezier basis matrix: point(t) = [1 t t^2 t^3] . M . P
BEZIER_MATRIX = np.array([
    [1, 0, 0, 0],
    [-3, 3, 0, 0],
    [3, -6, 3, 0],
    [-1, 3, -3, 1],
], dtype=float)

SAMPLE_SPACING = 3.0  # metres between centerline samples after resampling


def bezier_segment(p0, p1, p2, p3, num_samples):
    t = np.linspace(0, 1, num_samples, endpoint=False)
    t_powers = np.stack([np.ones_like(t), t, t ** 2, t ** 3], axis=1)
    control = np.array([p0, p1, p2, p3])
    return t_powers @ BEZIER_MATRIX @ control


def _convex_hull(points):
    """Andrew's monotone chain; returns hull vertices counter-clockwise."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _turn_angles(poly):
    """Signed exterior angle at every vertex of a closed polygon (radians)."""
    d_in = poly - np.roll(poly, 1, axis=0)
    d_out = np.roll(poly, -1, axis=0) - poly
    a_in = np.arctan2(d_in[:, 1], d_in[:, 0])
    a_out = np.arctan2(d_out[:, 1], d_out[:, 0])
    return (a_out - a_in + np.pi) % (2 * np.pi) - np.pi


def _polyline_self_intersects(poly, min_index_gap=3):
    """True if a closed polyline crosses itself anywhere (segment-vs-segment
    intersection, each test a 2x2 linear system, vectorised over all pairs)."""
    a = poly
    d1 = np.roll(poly, -1, axis=0) - a
    n = len(poly)
    denom = d1[:, None, 0] * d1[None, :, 1] - d1[:, None, 1] * d1[None, :, 0]
    valid = np.abs(denom) > 1e-9
    safe = np.where(valid, denom, 1.0)
    rhs0 = a[None, :, 0] - a[:, None, 0]
    rhs1 = a[None, :, 1] - a[:, None, 1]
    t = (rhs0 * d1[None, :, 1] - rhs1 * d1[None, :, 0]) / safe
    s = (rhs0 * d1[:, None, 1] - rhs1 * d1[:, None, 0]) / safe
    hit = valid & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, n - gap)
    return bool(np.any(hit & (gap > min_index_gap)))


def _resample(points, spacing):
    """Closed polyline -> evenly spaced samples (by arc length)."""
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(16, int(round(total / spacing)))
    s_new = np.arange(n) * (total / n)
    x = np.interp(s_new, s, closed[:, 0])
    y = np.interp(s_new, s, closed[:, 1])
    return np.stack([x, y], axis=1), total


def _curvature_radius(c):
    p0, p1, p2 = np.roll(c, 1, axis=0), c, np.roll(c, -1, axis=0)
    v1, v2 = p1 - p0, p2 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            * np.linalg.norm(p2 - p0, axis=1) / (2 * np.abs(cross) + 1e-9))


def _insert_features(poly, rng, difficulty, width, k=1.0):
    """Splice classic circuit features into the long straights of a (metre-
    scaled, counter-clockwise) polygon: hairpins, chicanes and esses. k scales
    the feature sizes with the lap length (1.0 = a ~3 km lap)."""
    out = []
    n = len(poly)
    p_feature = 0.45 + 0.4 * difficulty
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        out.append(a)
        edge = b - a
        length = np.linalg.norm(edge)
        if length < 260 * k or rng.random() > p_feature:
            continue
        u = edge / length
        left = np.array([-u[1], u[0]])          # points into the infield (CCW)
        side = 1.0 if rng.random() < 0.7 else -1.0
        p_hair = 0.10 + 0.30 * difficulty
        kind = rng.choice(["hairpin", "chicane", "esses"], p=[p_hair, (1 - p_hair) / 2, (1 - p_hair) / 2])
        c = a + u * length * rng.uniform(0.35, 0.65)
        if kind == "hairpin":
            # deep V into the infield: ~160 deg at the tip, arms ~ 2 * base apart
            base = rng.uniform(28, 42)
            depth = min(rng.uniform(140, 260) * k, 0.9 * length)
            out += [c - u * base, c + left * side * depth, c + u * base]
        elif kind == "chicane":
            s, d = max(35, rng.uniform(45, 80) * k), max(15, rng.uniform(18, 32) * k) * side
            out += [c - u * s, c - u * s / 3 + left * d, c + u * s / 3 + left * d, c + u * s]
        else:  # esses: alternating sweepers
            n_ess = int(rng.integers(3, 5))
            span = min(length * 0.7, n_ess * rng.uniform(70, 110) * k)
            offs = rng.uniform(25, 55, size=n_ess) * k * np.where(np.arange(n_ess) % 2 == 0, 1, -1) * side
            ts = np.linspace(-span / 2, span / 2, n_ess)
            out += [c + u * t + left * o for t, o in zip(ts, offs)]
    return np.array(out)


def _fit_corners(poly, rng, difficulty, width):
    """Radius per vertex, then shrink pairs of radii until both corners of
    every edge fit on it. Returns (turn angles, radius, tangent length)."""
    turns = _turn_angles(poly)
    theta = np.abs(turns)
    half_w = width / 2
    # easy tracks only get open corners; hard ones go down to F1 hairpins
    r_min = max(half_w * 1.4, 12.0) + (1 - difficulty) * 25
    edge_len = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    # sharp turns tend to get tight radii (hairpins), shallow ones fast sweepers
    r_lo = r_min + (1 - difficulty) * 15
    radius = rng.uniform(r_lo, r_lo + 30 + 180 * (1 - theta / np.pi) ** 2)
    tangent = radius * np.tan(theta / 2)
    n = len(poly)
    for _ in range(60):
        changed = False
        for i in range(n):
            j = (i + 1) % n
            need = tangent[i] + tangent[j]
            if need > 0.92 * edge_len[i]:
                f = 0.92 * edge_len[i] / need
                tangent[i] *= f
                tangent[j] *= f
                changed = True
        if not changed:
            break
    radius = tangent / np.maximum(np.tan(theta / 2), 1e-9)
    ok = not np.any(radius[theta > np.radians(4)] < r_min)
    return turns, theta, radius, tangent, ok


def _build(rng, difficulty, target_length, width):
    # 1. hull of random points -> long straights
    n_pts = int(rng.integers(9, 15))
    pts = rng.uniform([0, 0], [1.7, 1.0], size=(n_pts, 2))
    poly = _convex_hull(pts)

    # 2. midpoint displacement, biased inward -> concave sections
    amp = 0.20 + 0.25 * difficulty
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        out.append(a)
        edge = b - a
        length = np.linalg.norm(edge)
        if length > 0.25 and rng.random() < 0.55 + 0.3 * difficulty:
            normal = np.array([-edge[1], edge[0]]) / length
            out.append((a + b) / 2 + normal * rng.uniform(-0.4 * amp, amp) * length)
    poly = np.array(out)

    # 3. scale to ~80% of the target length; features add the rest
    perim = np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1))
    poly = poly * (0.8 * target_length / perim)
    k = target_length / 3000.0
    poly = _insert_features(poly, rng, difficulty, width, k)
    merge = max(15.0, 25.0 * k)

    # drop vertices that sit too close together (degenerate mini-edges)
    keep = [0]
    for i in range(1, len(poly)):
        if np.linalg.norm(poly[i] - poly[keep[-1]]) > merge:
            keep.append(i)
    poly = poly[keep]
    if np.linalg.norm(poly[0] - poly[-1]) < merge:
        poly = poly[:-1]
    if len(poly) < 5 or np.any(np.abs(_turn_angles(poly)) > np.radians(172)):
        return None

    # rescale so the finished lap (corners cut by the fillets) lands near the target
    perim = np.sum(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1))
    poly = poly * (1.06 * target_length / perim)

    # 4. corners: radius per vertex, tangent length t = R tan(theta/2)
    turns, theta, radius, tangent, ok = _fit_corners(poly, rng, difficulty, width)
    if not ok:
        return None

    pieces = []
    n_v = len(poly)
    for i in range(n_v):
        v = poly[i]
        u_in = v - poly[i - 1]
        u_in /= np.linalg.norm(u_in)
        u_out = poly[(i + 1) % n_v] - v
        u_out /= np.linalg.norm(u_out)
        t_in, t_out = v - tangent[i] * u_in, v + tangent[i] * u_out
        # cubic Bezier approximating a circular arc: handle = 4/3 R tan(theta/4)
        h = 4.0 / 3.0 * radius[i] * np.tan(theta[i] / 4)
        arc_len = radius[i] * theta[i]
        pieces.append(bezier_segment(t_in, t_in + h * u_in, t_out - h * u_out, t_out,
                                     max(4, int(arc_len / 1.0))))
        # straight from this corner's exit to the next corner's entry
        nxt = poly[(i + 1) % n_v]
        u_next = poly[(i + 2) % n_v] - nxt
        u_next /= np.linalg.norm(u_next)
        entry_next = nxt - tangent[(i + 1) % n_v] * u_out
        seg_len = np.linalg.norm(entry_next - t_out)
        k = max(2, int(seg_len / 1.0))
        tt = np.linspace(0, 1, k, endpoint=False)[:, None]
        pieces.append(t_out + tt * (entry_next - t_out))
    dense = np.concatenate(pieces, axis=0)

    # 5. uniform resampling, start on the longest straight
    centerline, length = _resample(dense, SAMPLE_SPACING)
    radius_c = _curvature_radius(centerline)
    straight = radius_c > 400
    # find the longest run of "straight" samples (circular)
    best_len, best_start, run, start = 0, 0, 0, 0
    for i in range(2 * len(centerline)):
        if straight[i % len(centerline)]:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run = 0
    best_len = min(best_len, len(centerline))
    start_idx = (best_start + max(1, best_len // 5)) % len(centerline)
    centerline = np.roll(centerline, -start_idx, axis=0)
    return centerline, length


def _finish(centerline, length, width, difficulty):
    tangents = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    half_w = np.full(len(centerline), width / 2)
    left = centerline + normals * half_w[:, None]
    right = centerline - normals * half_w[:, None]
    n = len(centerline)
    ds = length / n
    return {
        "centerline": centerline,
        "tangents": tangents,
        "normals": normals,
        "left": left,
        "right": right,
        # wall segments: left walls first (i -> i+1), then right walls
        "seg_a": np.concatenate([left, right], axis=0),
        "seg_b": np.concatenate([np.roll(left, -1, axis=0), np.roll(right, -1, axis=0)], axis=0),
        "cumdist": np.arange(n) * ds,
        "ds": ds,
        "length": length,
        "half_width": half_w,
        "width": width,
        "start_pos": centerline[0].copy(),
        "start_angle": float(np.arctan2(tangents[0, 1], tangents[0, 0])),
        "curvature_radius": _curvature_radius(centerline),
        "difficulty": difficulty,
    }


def _clearance_ok(centerline, width, length):
    """Two parts of the circuit that are far apart along the lap must also be
    far apart in space - otherwise their walls touch or overlap."""
    n = len(centerline)
    sub = centerline[::3]
    m = len(sub)
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    idx = np.arange(m) * 3
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, n - gap) * (length / n)
    far_along = gap > 4 * width
    return bool(np.all(d[far_along] > width + 6.0))


def track_width(difficulty):
    # 30 m forgiving road at difficulty 0 down to a real F1 width (15 m) at 1
    return 30.0 - 15.0 * float(np.clip(difficulty, 0, 1))


def generate_track(seed=None, difficulty=1.0, length_range=(1500.0, 1800.0), max_attempts=600, **_):
    """Closed F1-style circuit. difficulty in [0, 1] narrows the road from
    30 m to 15 m and adds more / sharper corners."""
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    width = track_width(difficulty)
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        target_length = rng.uniform(*length_range)
        built = _build(rng, difficulty, target_length, width)
        if built is None:
            continue
        centerline, length = built
        if not (length_range[0] <= length <= length_range[1]):
            continue
        track = _finish(centerline, length, width, difficulty)
        if np.min(track["curvature_radius"]) < width / 2 * 1.2:
            continue
        if _polyline_self_intersects(track["left"]) or _polyline_self_intersects(track["right"]):
            continue
        if not _clearance_ok(centerline, width, length):
            continue
        return track
    raise RuntimeError(f"could not generate a valid track (seed={seed})")

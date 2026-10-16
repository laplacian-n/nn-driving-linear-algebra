import numpy as np
import pygame

# Deliberately raw/flat: white page, black borders and text, no gray, no
# rounded corners, no translucency. The one named exception is the track's
# center line (explicitly asked to stay gray) and the NN diagram, which keeps
# the green/positive - red/negative convention from the original reference.
PALETTE = {
    "page": (255, 255, 255),
    "panel": (255, 255, 255),
    "border": (0, 0, 0),
    "text": (0, 0, 0),
    "wall": (0, 0, 0),
    "center_line": (150, 150, 150),
    "road": (225, 225, 228),
    "car": (0, 0, 0),
    "car_best": (0, 150, 0),
    "car_drift": (220, 130, 0),
    "weight_pos": (0, 150, 0),
    "weight_neg": (200, 0, 0),
}


def init_fonts():
    return {
        "label": pygame.font.SysFont("consolas", 15),
        "value": pygame.font.SysFont("consolas", 15, bold=True),
        "title": pygame.font.SysFont("consolas", 15, bold=True),
        "node_label": pygame.font.SysFont("consolas", 12),
    }


def _frame(surface, rect):
    x, y, w, h = rect
    pygame.draw.rect(surface, PALETTE["panel"], (x, y, w, h))
    pygame.draw.rect(surface, PALETTE["border"], (x, y, w, h), width=2)


def draw_background(surface):
    surface.fill(PALETTE["page"])


def compute_view_transform(track, viewport, margin=0.9):
    """World (simulation) coordinates -> screen coordinates that fit the track
    inside `viewport`, so the track panel is a real bounded box instead of the
    whole window - the simulation's own coordinate space is untouched."""
    vx, vy, vw, vh = viewport
    pts = np.concatenate([track["left"], track["right"]], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = np.maximum(hi - lo, 1e-6)
    scale = min(vw / size[0], vh / size[1]) * margin
    world_center = (lo + hi) / 2
    screen_center = np.array([vx + vw / 2, vy + vh / 2])

    def to_screen(world_pts):
        return (np.asarray(world_pts) - world_center) * scale + screen_center

    return to_screen, scale


def draw_track(surface, rect, track, to_screen):
    _frame(surface, rect)
    surface.blit(pygame.font.SysFont("consolas", 15, bold=True).render("Track", True, PALETTE["text"]),
                 (rect[0] + 10, rect[1] + 8))

    clip = surface.get_clip()
    surface.set_clip(pygame.Rect(*rect))

    # Fill the tarmac between the two boundaries as one closed band (left edge
    # forward, right edge back) so the road reads as a real asphalt ribbon
    # instead of two thin outlines - much clearer on the now-narrow F1 tracks.
    left = to_screen(track["left"])
    right = to_screen(track["right"])
    road_poly = np.concatenate([left, right[::-1]], axis=0).tolist()
    pygame.draw.polygon(surface, PALETTE["road"], road_poly)
    pygame.draw.aalines(surface, PALETTE["wall"], True, left.tolist())
    pygame.draw.aalines(surface, PALETTE["wall"], True, right.tolist())

    centerline = to_screen(track["centerline"])
    n = len(centerline)
    dash = max(1, n // 60)
    for i in range(0, n, dash * 2):
        j = min(i + dash, n - 1)
        pygame.draw.line(surface, PALETTE["center_line"], centerline[i].tolist(), centerline[j].tolist(), 2)

    surface.set_clip(clip)


def _car_corners(pos, angle, length, width):
    d = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-d[1], d[0]])
    hl, hw = length / 2, width / 2
    return [pos + hl * d + hw * side, pos + hl * d - hw * side,
            pos - hl * d - hw * side, pos - hl * d + hw * side]


def draw_cars(surface, rect, cars, to_screen, scale, best_idx=None):
    clip = surface.get_clip()
    surface.set_clip(pygame.Rect(*rect))

    # A real F1 car is only ~2 m wide, which at these zoom levels can shrink to
    # a sub-pixel sliver, so the body is drawn at its true proportions but never
    # smaller than a few pixels - it stays visible without misrepresenting how
    # much room it actually has on the road.
    disp_len = max(6.0, cars.car_length * scale)
    disp_wid = max(3.0, cars.car_width * scale)
    for i in range(cars.n):
        if not cars.alive[i] and i != best_idx:
            continue
        is_best = i == best_idx
        world_corners = _car_corners(cars.pos[i], cars.angle[i], disp_len / scale, disp_wid / scale)
        pts = [to_screen(c).tolist() for c in world_corners]
        # A car that's sliding sideways (large slip) is tinted so a drift is
        # visible at a glance; the current best car is filled solid, the rest
        # are hollow so the pack doesn't turn into a blob.
        drifting = abs(cars.slip[i]) > 0.15 * cars.max_speed
        if is_best:
            color = PALETTE["car_drift"] if drifting else PALETTE["car_best"]
            pygame.draw.polygon(surface, color, pts)
        else:
            color = PALETTE["car_drift"] if drifting else PALETTE["car"]
            pygame.draw.polygon(surface, color, pts, width=1)

    surface.set_clip(clip)


def draw_hud(surface, fonts, rect, generation, alive, total, best_progress, best_ever,
             track_length, laps_finished=0):
    _frame(surface, rect)
    x, y, w, h = rect
    pad = 10
    surface.blit(fonts["title"].render("Stats", True, PALETTE["text"]), (x + pad, y + pad))
    rows = [
        ("Generation", f"{generation}"),
        ("Alive", f"{alive} / {total}"),
        ("This gen", f"{best_progress:.0f} / {track_length:.0f}"),
        ("Best ever", f"{best_ever:.0f} / {track_length:.0f}"),
        ("Laps", f"{laps_finished} / {total}"),
    ]
    ty = y + pad + 24
    for label, value in rows:
        surface.blit(fonts["label"].render(label, True, PALETTE["text"]), (x + pad, ty))
        value_surf = fonts["value"].render(value, True, PALETTE["text"])
        surface.blit(value_surf, (x + w - pad - value_surf.get_width(), ty))
        ty += 20


def _line_chart(surface, fonts, rect, title, values, y_label_fmt="{:.0f}"):
    _frame(surface, rect)
    x, y, w, h = rect
    pad = 10
    surface.blit(fonts["title"].render(title, True, PALETTE["text"]), (x + pad, y + pad))

    plot_rect = (x + pad, y + pad + 22, w - 2 * pad, h - pad * 2 - 22)
    px, py, pw, ph = plot_rect
    pygame.draw.line(surface, PALETTE["text"], (px, py + ph), (px + pw, py + ph), 1)

    if len(values) < 2:
        return
    arr = np.array(values[-200:], dtype=float)
    hi = arr.max()
    if hi <= 0:
        # Nothing real to show yet (e.g. no lap finished at all so far) - drawing
        # a flat line against a fake "1.0" fallback scale would read as if
        # something had actually happened.
        return
    xs = np.linspace(px, px + pw, len(arr))
    ys = py + ph - (arr / hi) * ph
    points = list(zip(xs.tolist(), ys.tolist()))
    pygame.draw.aalines(surface, PALETTE["text"], False, points)

    hi_label = fonts["node_label"].render(y_label_fmt.format(hi), True, PALETTE["text"])
    surface.blit(hi_label, (px + pw - hi_label.get_width(), py))


def draw_sparkline(surface, fonts, rect, history):
    _line_chart(surface, fonts, rect, "Best progress / gen", history)


def draw_speed_chart(surface, fonts, rect, speed_history):
    _line_chart(surface, fonts, rect, "Best lap speed / gen", speed_history, "{:.0f} px/s")


def draw_network(surface, fonts, rect, weights, activations, input_labels, output_labels):
    _frame(surface, rect)
    x0, y0, w, h = rect
    surface.blit(fonts["title"].render("Neural Network (best car)", True, PALETTE["text"]), (x0 + 14, y0 + 10))

    top_pad = 36
    layer_sizes = [weights[0].shape[1]] + [wl.shape[0] for wl in weights]
    xs = np.linspace(x0 + 90, x0 + w - 110, len(layer_sizes))

    positions = []
    for li, size in enumerate(layer_sizes):
        ys = np.linspace(y0 + top_pad + 10, y0 + h - 14, size)
        positions.append([(xs[li], y) for y in ys])

    for li, w_matrix in enumerate(weights):
        for j, (x2, y2) in enumerate(positions[li + 1]):
            for k, (x1, y1) in enumerate(positions[li]):
                weight = w_matrix[j, k]
                color = PALETTE["weight_pos"] if weight > 0 else PALETTE["weight_neg"]
                thickness = max(1, min(3, int(abs(weight) * 2)))
                pygame.draw.aaline(surface, color, (x1, y1), (x2, y2))
                if thickness > 1:
                    pygame.draw.line(surface, color, (x1, y1), (x2, y2), thickness)

    for li, layer_pos in enumerate(positions):
        act = activations[li] if li < len(activations) else None
        for i, (x, y) in enumerate(layer_pos):
            value = act[i] if act is not None else 0.0
            color = PALETTE["weight_pos"] if value > 0 else PALETTE["weight_neg"]
            pygame.draw.circle(surface, color, (int(x), int(y)), 5)
            pygame.draw.circle(surface, PALETTE["text"], (int(x), int(y)), 5, 1)

    for i, (x, y) in enumerate(positions[0]):
        if i < len(input_labels):
            label = fonts["node_label"].render(input_labels[i], True, PALETTE["text"])
            surface.blit(label, (x - 88, y - 7))

    for i, (x, y) in enumerate(positions[-1]):
        if i < len(output_labels):
            label = fonts["node_label"].render(output_labels[i], True, PALETTE["text"])
            surface.blit(label, (x + 12, y - 7))

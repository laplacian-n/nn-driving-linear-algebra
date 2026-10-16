"""Shared drawing code for the viewer and the demo app.

Layout (1600 x 900): a chase-camera track view on the left and a sidebar on
the right with a minimap, telemetry and a session panel. Every text field is
drawn into a fixed rectangle and is truncated if it would not fit, so nothing
ever spills out of its box.
"""
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
import pygame

W, H = 1600, 900
SIDEBAR_W = 420
VIEW = pygame.Rect(0, 0, W - SIDEBAR_W, H)
PAD = 16

BG = (14, 16, 20)
PANEL = (24, 27, 33)
BORDER = (46, 51, 61)
TEXT = (232, 234, 238)
MUTED = (139, 146, 158)
ACCENT = (255, 199, 44)
GREEN = (72, 201, 122)
RED = (235, 77, 75)
BLUE = (88, 160, 255)
GROUND = (31, 36, 33)
ASPHALT = (62, 64, 70)
EDGE = (215, 217, 222)
KERB_RED = (200, 55, 55)


def speed_color(v, vmax=95.0):
    """Slow = blue, fast = red, through yellow."""
    x = float(np.clip(v / vmax, 0, 1))
    stops = [(0.0, (70, 120, 255)), (0.5, (255, 214, 60)), (1.0, (240, 60, 50))]
    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return stops[-1][1]


class Fonts:
    def __init__(self):
        sans = pygame.font.match_font("ubuntusans,ubuntu,notosans,dejavusans")
        mono = pygame.font.match_font("ubuntusansmono,ubuntumono,dejavusansmono")
        self.small = pygame.font.Font(sans, 15)
        self.label = pygame.font.Font(sans, 17)
        self.title = pygame.font.Font(sans, 19)
        self.title.set_bold(True)
        self.num = pygame.font.Font(mono, 20)
        self.big = pygame.font.Font(mono, 64)
        self.big.set_bold(True)


def text(surf, s, font, color, rect, align="left"):
    """Draw s inside rect (vertically centred); truncate with an ellipsis if
    it is wider than the rect."""
    s = str(s)
    img = font.render(s, True, color)
    if img.get_width() > rect.width:
        while s and font.size(s + "…")[0] > rect.width:
            s = s[:-1]
        img = font.render(s + "…", True, color)
    if align == "right":
        x = rect.right - img.get_width()
    elif align == "center":
        x = rect.centerx - img.get_width() // 2
    else:
        x = rect.x
    surf.blit(img, (x, rect.centery - img.get_height() // 2))


def panel(surf, rect, fonts, title=None):
    pygame.draw.rect(surf, PANEL, rect, border_radius=8)
    pygame.draw.rect(surf, BORDER, rect, width=1, border_radius=8)
    if title:
        text(surf, title.upper(), fonts.small, MUTED, pygame.Rect(rect.x + 14, rect.y + 8, rect.width - 28, 20))
    return pygame.Rect(rect.x + 14, rect.y + (34 if title else 12), rect.width - 28,
                       rect.height - (46 if title else 24))


def row(surf, fonts, rect, y, label, value, color=TEXT):
    r = pygame.Rect(rect.x, y, rect.width, 24)
    text(surf, label, fonts.label, MUTED, pygame.Rect(r.x, r.y, r.width * 0.55, r.height))
    text(surf, value, fonts.num, color, pygame.Rect(r.x + r.width * 0.45, r.y, r.width * 0.55, r.height), "right")


class TrackRenderer:
    """Pre-renders the whole circuit once at the camera scale; each frame only
    blits the visible window, so drawing cost does not grow with track size."""

    def __init__(self, track, scale=4.0, margin=60.0):
        self.track = track
        self.scale = scale
        pts = np.concatenate([track["left"], track["right"]])
        self.origin = pts.min(axis=0) - margin
        size = (pts.max(axis=0) + margin - self.origin) * scale
        self.surface = pygame.Surface((int(size[0]), int(size[1])))
        self.surface.fill(GROUND)
        L, R = self.world_to_map(track["left"]), self.world_to_map(track["right"])
        n = len(L)
        for i in range(n):  # asphalt as quads between the two walls
            j = (i + 1) % n
            pygame.draw.polygon(self.surface, ASPHALT, [L[i], L[j], R[j], R[i]])
        tight = track["curvature_radius"] < 80
        for wall in (L, R):
            for i in range(n):
                j = (i + 1) % n
                if tight[i]:
                    col = KERB_RED if (i // 2) % 2 else EDGE
                    pygame.draw.line(self.surface, col, wall[i], wall[j], max(3, int(scale * 0.9)))
                else:
                    pygame.draw.line(self.surface, EDGE, wall[i], wall[j], 2)
        c0, n0 = track["centerline"][0], track["normals"][0]
        hw = track["half_width"][0]
        a, b = self.world_to_map(np.array([c0 + n0 * hw, c0 - n0 * hw]))
        for k in range(8):  # chequered start/finish line
            p = a + (b - a) * k / 8
            q = a + (b - a) * (k + 1) / 8
            pygame.draw.line(self.surface, (245, 245, 245) if k % 2 else (20, 20, 20), p, q, 6)

    def world_to_map(self, pts):
        return (np.asarray(pts) - self.origin) * self.scale

    def draw(self, surf, view, center_world):
        c = self.world_to_map(center_world)
        src = pygame.Rect(int(c[0] - view.width / 2), int(c[1] - view.height / 2), view.width, view.height)
        surf.fill(GROUND, view)
        surf.blit(self.surface, view.topleft, src)
        offset = np.array(view.topleft) - np.array(src.topleft)
        return lambda pts: self.world_to_map(pts) + offset


def car_polygon(pos, psi, length, width):
    c, s = np.cos(psi), np.sin(psi)
    f, l = np.array([c, s]), np.array([-s, c])
    hl, hw = length / 2, width / 2
    return [pos + f * hl + l * hw * 0.75, pos + f * (hl + 0.4), pos + f * hl - l * hw * 0.75,
            pos - f * hl - l * hw, pos - f * hl + l * hw]


def draw_car(surf, to_screen, pos, psi, params, color, outline=None):
    pts = to_screen(np.array(car_polygon(pos, psi, params.length, params.width)))
    pygame.draw.polygon(surf, color, pts.tolist())
    if outline:
        pygame.draw.polygon(surf, outline, pts.tolist(), 2)


class Minimap:
    def __init__(self, track, rect):
        self.rect = rect
        pts = np.concatenate([track["left"], track["right"]])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.s = min((rect.width - 24) / (hi - lo)[0], (rect.height - 24) / (hi - lo)[1])
        self.off = np.array(rect.center) - (lo + hi) / 2 * self.s
        self.line = self.to_screen(track["centerline"]).tolist()

    def to_screen(self, pts):
        return np.asarray(pts) * self.s + self.off

    def draw(self, surf, positions, alive, focus=None, trail=None):
        pygame.draw.lines(surf, ASPHALT, True, self.line, 6)
        pygame.draw.lines(surf, (90, 94, 104), True, self.line, 1)
        if trail is not None and len(trail) > 1:
            pts, speeds = trail
            sp = self.to_screen(pts)
            for i in range(0, len(sp) - 1, 2):
                pygame.draw.line(surf, speed_color(speeds[i]), sp[i], sp[i + 1], 3)
        for i in np.where(alive)[0]:
            if i != focus:
                pygame.draw.circle(surf, MUTED, self.to_screen(positions[i]), 2)
        if focus is not None:
            pygame.draw.circle(surf, ACCENT, self.to_screen(positions[focus]), 5)


def bar(surf, rect, value, color, vertical=True):
    pygame.draw.rect(surf, (40, 44, 53), rect, border_radius=4)
    v = float(np.clip(value, 0, 1))
    if vertical:
        h = int(rect.height * v)
        fill = pygame.Rect(rect.x, rect.bottom - h, rect.width, h)
    else:
        fill = pygame.Rect(rect.x, rect.y, int(rect.width * v), rect.height)
    if fill.width > 0 and fill.height > 0:
        pygame.draw.rect(surf, color, fill, border_radius=4)


def steering_bar(surf, rect, value):
    pygame.draw.rect(surf, (40, 44, 53), rect, border_radius=4)
    cx = rect.centerx
    x = cx + int(np.clip(value, -1, 1) * rect.width / 2)
    pygame.draw.rect(surf, BLUE, pygame.Rect(min(cx, x), rect.y, abs(x - cx) + 1, rect.height), border_radius=4)
    pygame.draw.line(surf, MUTED, (cx, rect.y - 3), (cx, rect.bottom + 3), 1)


def telemetry(surf, fonts, rect, cars, i, lap_time_best=None):
    inner = panel(surf, rect, fonts, "Telemetry")
    kmh = cars.vx[i] * 3.6
    text(surf, f"{kmh:3.0f}", fonts.big, TEXT, pygame.Rect(inner.x, inner.y, 170, 70), "right")
    text(surf, "km/h", fonts.label, MUTED, pygame.Rect(inner.x + 178, inner.y + 30, 60, 30))
    bx = inner.right - 110
    for k, (lab, val, col) in enumerate([("THR", cars.throttle[i], GREEN), ("BRK", cars.brake[i], RED),
                                          ("HB", cars.handbrake[i], ACCENT)]):
        r = pygame.Rect(bx + k * 38, inner.y, 22, 62)
        bar(surf, r, val, col)
        text(surf, lab, fonts.small, MUTED, pygame.Rect(r.x - 6, r.bottom + 2, 34, 18), "center")
    y = inner.y + 92
    text(surf, "Steering", fonts.label, MUTED, pygame.Rect(inner.x, y, 100, 22))
    steering_bar(surf, pygame.Rect(inner.x + 110, y + 6, inner.width - 110, 10), cars.delta[i] / cars.p.max_steer)
    y += 30
    beta = np.degrees(cars.slip_angle[i])
    row(surf, fonts, inner, y, "Slip angle", f"{beta:+5.1f}°", ACCENT if abs(beta) > 6 else TEXT)
    y += 26
    lap = cars.progress[i] / cars.track["length"]
    row(surf, fonts, inner, y, "Lap", f"{min(lap, 1) * 100:5.1f} %")
    y += 26
    t = cars.lap_time[i] if cars.finished[i] else cars.time
    row(surf, fonts, inner, y, "Lap time", f"{t:6.2f} s")
    y += 26
    row(surf, fonts, inner, y, "Best lap", "–" if lap_time_best is None else f"{lap_time_best:6.2f} s")


def draw_rays(surf, to_screen, cars, i, ranges):
    if ranges is None:
        return
    for k, off in enumerate(cars.sensor_offsets):
        ang = cars.psi[i] + off
        end = cars.pos[i] + ranges[k] * np.array([np.cos(ang), np.sin(ang)])
        near = ranges[k] < 40
        col = (255, 120, 90) if near else (120, 170, 255)
        a, b = to_screen(np.array([cars.pos[i], end]))
        pygame.draw.aaline(surf, col, a, b)
        pygame.draw.circle(surf, col, b, 3)


def network_panel(surf, fonts, rect, weights, activations, in_labels, out_labels):
    """Compact network view: nodes coloured by activation, only the
    strongest connections drawn so the diagram stays readable."""
    inner = panel(surf, rect, fonts, "Network")
    sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    xs = np.linspace(inner.x + 54, inner.right - 74, len(sizes))
    pos = [[(xs[l], y) for y in np.linspace(inner.y + 6, inner.bottom - 6, n)] for l, n in enumerate(sizes)]
    # input column: sensor group on top, the 4 car-state inputs below it with
    # a fixed 18 px spacing so their labels can never overlap
    n_state = 4
    n_sens = sizes[0] - n_state
    state_top = inner.bottom - 6 - (n_state - 1) * 18
    sens_y = np.linspace(inner.y + 6, state_top - 22, max(n_sens, 1)) if n_sens > 1 else [inner.y + 6]
    pos[0] = [(xs[0], y) for y in sens_y][:n_sens] + [(xs[0], state_top + 18 * k) for k in range(n_state)]
    label_every = 1
    if n_sens > 1:
        spacing = sens_y[1] - sens_y[0]
        label_every = int(np.ceil(18 / max(spacing, 1e-6)))
    for l, w in enumerate(weights):
        thr = np.quantile(np.abs(w), 0.85) if w.size > 12 else 0
        for j, (x2, y2) in enumerate(pos[l + 1]):
            for k, (x1, y1) in enumerate(pos[l]):
                if abs(w[j, k]) >= thr:
                    col = (60, 150, 100) if w[j, k] > 0 else (170, 70, 70)
                    pygame.draw.aaline(surf, col, (x1, y1), (x2, y2))
    for l, layer in enumerate(pos):
        a = activations[l] if l < len(activations) else np.zeros(len(layer))
        r = 4 if len(layer) <= 24 else 3
        for k, (x, y) in enumerate(layer):
            v = float(np.clip(a[k], -1, 1))
            col = tuple(int(c) for c in (np.array(GREEN) * max(v, 0) + np.array(RED) * max(-v, 0)
                                          + np.array((60, 64, 74)) * (1 - abs(v))))
            pygame.draw.circle(surf, col, (int(x), int(y)), r)
    for k, (x, y) in enumerate(pos[0]):
        if k < len(in_labels) and (k >= n_sens or k % label_every == 0):
            text(surf, in_labels[k], fonts.small, MUTED, pygame.Rect(inner.x - 4, y - 9, 50, 18), "right")
    for k, (x, y) in enumerate(pos[-1]):
        text(surf, out_labels[k], fonts.small, MUTED, pygame.Rect(x + 10, y - 9, 64, 18))

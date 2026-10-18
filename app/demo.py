"""Demo app for testers: pick a model and a track (random, by seed, or draw
your own), watch the network drive one lap, and every result is logged to
data/demo_results.csv for the presentation.

    python -m app.demo
"""
import csv
import datetime
import json
import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
import pygame

from app import render as R
from app.models import Driver, ROOT, list_runs
from app.viewer import Session
from racing.track import track_from_points, track_width

RESULTS = os.path.join(ROOT, "data", "demo_results.csv")
PRESETS = os.path.join(ROOT, "app", "demo_models.json")
FIELDS = ["timestamp", "tester", "model", "track_source", "track_seed", "difficulty", "length_m",
          "finished", "progress_pct", "lap_time_s", "avg_speed_kmh"]


def load_models():
    """Display name -> run dir. Uses app/demo_models.json if present,
    otherwise every run that has snapshots."""
    if os.path.exists(PRESETS):
        with open(PRESETS) as f:
            return [(d["name"], os.path.join(ROOT, d["run"]), d.get("snapshot")) for d in json.load(f)]
    out = []
    for run in list_runs():
        rel = os.path.relpath(run, os.path.join(ROOT, "experiments", "results"))
        out.append((rel.replace(os.sep, " / "), run, None))
    return out


def log_result(row):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def tests_by(tester):
    if not os.path.exists(RESULTS):
        return 0
    with open(RESULTS) as f:
        return sum(1 for r in csv.DictReader(f) if r["tester"] == tester)


# ------------------------------------------------------------------ widgets
class Button:
    def __init__(self, rect, label, primary=False):
        self.rect, self.label, self.primary = pygame.Rect(rect), label, primary

    def draw(self, surf, fonts, selected=False, enabled=True):
        hover = self.rect.collidepoint(pygame.mouse.get_pos()) and enabled
        if self.primary:
            bg = R.ACCENT if enabled else (90, 80, 40)
            fg = (20, 20, 20)
        else:
            bg = (52, 58, 70) if (hover or selected) else (36, 40, 49)
            fg = R.TEXT if enabled else R.MUTED
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        if selected and not self.primary:
            pygame.draw.rect(surf, R.ACCENT, self.rect, width=2, border_radius=6)
        R.text(surf, self.label, fonts.label, fg, self.rect.inflate(-16, 0), "center")

    def hit(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)


class TextInput:
    def __init__(self, rect, value="", numeric=False, placeholder=""):
        self.rect, self.value, self.numeric, self.placeholder = pygame.Rect(rect), value, numeric, placeholder
        self.active = False

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(e.pos)
        elif e.type == pygame.KEYDOWN and self.active:
            if e.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
            elif e.key in (pygame.K_RETURN, pygame.K_TAB):
                self.active = False
            elif e.unicode and len(self.value) < 24 and (not self.numeric or e.unicode.isdigit()):
                self.value += e.unicode

    def draw(self, surf, fonts):
        pygame.draw.rect(surf, (30, 34, 42), self.rect, border_radius=6)
        pygame.draw.rect(surf, R.ACCENT if self.active else R.BORDER, self.rect, width=2, border_radius=6)
        shown = self.value + ("|" if self.active else "")
        R.text(surf, shown or self.placeholder, fonts.label, R.TEXT if self.value else R.MUTED,
               self.rect.inflate(-20, 0))


class Slider:
    def __init__(self, rect, value=1.0, vmax=1.0, step=0.05):
        self.rect, self.value, self.drag = pygame.Rect(rect), value, False
        self.vmax, self.step = vmax, step

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and self.rect.inflate(0, 20).collidepoint(e.pos):
            self.drag = True
        if e.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        if (e.type == pygame.MOUSEMOTION and self.drag) or (e.type == pygame.MOUSEBUTTONDOWN and self.drag):
            frac = float(np.clip((e.pos[0] - self.rect.x) / self.rect.width, 0, 1))
            self.value = round(frac * self.vmax / self.step) * self.step

    def draw(self, surf, fonts):
        pygame.draw.rect(surf, (40, 44, 53), self.rect, border_radius=4)
        fill = self.rect.copy()
        fill.width = int(self.rect.width * self.value / self.vmax)
        pygame.draw.rect(surf, R.ACCENT, fill, border_radius=4)
        pygame.draw.circle(surf, R.TEXT, (self.rect.x + fill.width, self.rect.centery), 10)


# -------------------------------------------------------------------- app
class DemoApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((R.W, R.H))
        pygame.display.set_caption("Neural-network driver - demo")
        self.fonts = R.Fonts()
        self.clock = pygame.time.Clock()
        self.models = load_models()
        self.model_idx = 0
        self.model_scroll = 0
        self.name = TextInput((120, 150, 420, 44), placeholder="type your name")
        self.seed = TextInput((120, 520, 200, 44), numeric=True, placeholder="e.g. 1234")
        self.difficulty = Slider((120, 640, 420, 10), 1.0, vmax=3.0, step=0.25)   # > 2: extreme show tracks
        self.track_mode = "random"
        self.mode_buttons = {m: Button((120 + i * 142, 450, 132, 44), lbl) for i, (m, lbl) in
                             enumerate([("random", "Random"), ("seed", "By seed"), ("draw", "Draw")])}
        self.start = Button((120, 740, 420, 56), "Start", primary=True)
        self.state = "menu"
        self.message = ""
        self.stroke = []
        self.session = None
        self.result = None
        self.rng = np.random.default_rng()
        self.driver_cache = {}

    # ............................................................ menu
    def model_rows(self):
        top, h = 150, 46
        visible = 13
        rows = []
        for k in range(self.model_scroll, min(len(self.models), self.model_scroll + visible)):
            rows.append((k, pygame.Rect(700, top + (k - self.model_scroll) * (h + 6), 840, h)))
        return rows

    def draw_menu(self):
        s, f = self.screen, self.fonts
        s.fill(R.BG)
        R.text(s, "Neural-network driver — test it yourself", f.title, R.TEXT, pygame.Rect(120, 50, 1300, 40))
        tests = tests_by(self.name.value.strip()) if self.name.value.strip() else 0
        R.text(s, "1  Your name", f.label, R.MUTED, pygame.Rect(120, 112, 420, 30))
        self.name.draw(s, f)
        if self.name.value.strip():
            R.text(s, f"tests so far: {tests}", f.small, R.MUTED, pygame.Rect(120, 198, 420, 24))
        R.text(s, "2  Track", f.label, R.MUTED, pygame.Rect(120, 412, 420, 30))
        for m, b in self.mode_buttons.items():
            b.draw(s, f, selected=self.track_mode == m)
        if self.track_mode == "seed":
            self.seed.draw(s, f)
        elif self.track_mode == "draw":
            R.text(s, "You will draw the loop on the next screen", f.small, R.MUTED, pygame.Rect(120, 520, 420, 44))
        else:
            R.text(s, "A new circuit is generated for you", f.small, R.MUTED, pygame.Rect(120, 520, 420, 44))
        R.text(s, f"3  Difficulty  {self.difficulty.value:.2f}   (road {track_width(self.difficulty.value):.1f} m wide)",
               f.label, R.MUTED, pygame.Rect(120, 600, 420, 30))
        self.difficulty.draw(s, f)
        R.text(s, "4  Model", f.label, R.MUTED, pygame.Rect(700, 112, 840, 30))
        if not self.models:
            R.text(s, "No trained models yet", f.label, R.RED, pygame.Rect(700, 150, 840, 40))
        for k, rect in self.model_rows():
            sel = k == self.model_idx
            pygame.draw.rect(s, (52, 58, 70) if sel else (30, 34, 42), rect, border_radius=6)
            if sel:
                pygame.draw.rect(s, R.ACCENT, rect, width=2, border_radius=6)
            R.text(s, self.models[k][0], f.label, R.TEXT, rect.inflate(-24, 0))
        ok = bool(self.name.value.strip()) and bool(self.models) and (self.track_mode != "seed" or self.seed.value)
        self.start.draw(s, f, enabled=ok)
        if self.message:
            R.text(s, self.message, f.label, R.RED, pygame.Rect(120, 810, 1300, 30))
        return ok

    def menu_event(self, e, ok):
        self.name.handle(e)
        if self.track_mode == "seed":
            self.seed.handle(e)
        self.difficulty.handle(e)
        for m, b in self.mode_buttons.items():
            if b.hit(e):
                self.track_mode = m
        for k, rect in self.model_rows():
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and rect.collidepoint(e.pos):
                self.model_idx = k
        if e.type == pygame.MOUSEWHEEL:
            self.model_scroll = int(np.clip(self.model_scroll - e.y, 0, max(0, len(self.models) - 13)))
        if ok and self.start.hit(e):
            self.message = ""
            if self.track_mode == "draw":
                self.stroke = []
                self.state = "draw"
            else:
                seed = int(self.seed.value) if self.track_mode == "seed" else int(self.rng.integers(0, 2**31))
                self.begin_race(seed=seed)

    # ............................................................ draw
    def draw_canvas(self):
        s, f = self.screen, self.fonts
        s.fill(R.BG)
        area = pygame.Rect(60, 90, R.W - 120, R.H - 170)
        pygame.draw.rect(s, (24, 27, 33), area, border_radius=10)
        R.text(s, "Hold the left mouse button and draw one closed loop.  Release to build the track.  "
                  "ESC = back", f.label, R.MUTED, pygame.Rect(60, 40, R.W - 120, 30))
        if len(self.stroke) > 1:
            pygame.draw.lines(s, R.ACCENT, False, self.stroke, 4)
        if self.message:
            R.text(s, self.message, f.label, R.RED, pygame.Rect(60, R.H - 66, R.W - 120, 30))
        return area

    def canvas_event(self, e, area):
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.state = "menu"
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and area.collidepoint(e.pos):
            self.stroke = [e.pos]
            self.message = ""
        elif e.type == pygame.MOUSEMOTION and e.buttons[0] and self.stroke:
            if np.hypot(e.pos[0] - self.stroke[-1][0], e.pos[1] - self.stroke[-1][1]) > 4:
                self.stroke.append(e.pos)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and len(self.stroke) > 1:
            track, why = track_from_points(self.stroke, difficulty=self.difficulty.value)
            if track is None:
                self.message = f"Can't use this drawing: {why}"
                self.stroke = []
            else:
                self.begin_race(track=track)

    # ............................................................ race
    def driver(self):
        name, run, snap = self.models[self.model_idx]
        if run not in self.driver_cache:
            self.driver_cache[run] = Driver(run, snapshot=os.path.join(ROOT, snap) if snap else None)
        return name, self.driver_cache[run]

    def begin_race(self, seed=None, track=None):
        name, drv = self.driver()
        label = "Your track" if track is not None else f"Track {seed}"
        try:
            self.session = Session(drv, seed if seed is not None else -1, self.difficulty.value, track=track,
                                   label=label, loop=False)
        except RuntimeError as err:
            self.message = str(err)
            self.state = "menu"
            return
        self.race_info = {"model": name, "source": "drawn" if track is not None else self.track_mode,
                          "seed": "drawn" if track is not None else seed}
        self.result = None
        self.state = "race"

    def finish_race(self):
        c = self.session.cars
        fin = bool(c.finished[0])
        lap = float(c.lap_time[0]) if fin else float("nan")
        length = self.session.track["length"]
        self.result = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "tester": self.name.value.strip(), "model": self.race_info["model"],
            "track_source": self.race_info["source"], "track_seed": self.race_info["seed"],
            "difficulty": f"{self.session.difficulty:.2f}", "length_m": f"{length:.0f}",
            "finished": int(fin), "progress_pct": f"{min(c.progress[0] / length, 1) * 100:.1f}",
            "lap_time_s": f"{lap:.2f}" if fin else "",
            "avg_speed_kmh": f"{length / lap * 3.6:.1f}" if fin else "",
        }
        log_result(self.result)

    def draw_result(self):
        s, f = self.screen, self.fonts
        box = pygame.Rect(R.VIEW.centerx - 300, R.VIEW.centery - 170, 600, 340)
        shade = pygame.Surface(R.VIEW.size, pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        s.blit(shade, (0, 0))
        inner = R.panel(s, box, f, "Result (saved)")
        r = self.result
        head = "LAP COMPLETE" if r["finished"] else "CRASHED"
        R.text(s, head, f.title, R.GREEN if r["finished"] else R.RED, pygame.Rect(inner.x, inner.y, inner.width, 34))
        y = inner.y + 50
        for lab, val in [("Model", r["model"]), ("Track", f"{r['track_seed']} · {r['length_m']} m"),
                         ("Lap completed", f"{r['progress_pct']} %"),
                         ("Lap time", f"{r['lap_time_s']} s" if r["finished"] else "–"),
                         ("Average speed", f"{r['avg_speed_kmh']} km/h" if r["finished"] else "–")]:
            R.row(s, f, inner, y, lab, val)
            y += 30
        self.again = Button((inner.x, inner.bottom - 50, 180, 46), "Same track again")
        self.menu_btn = Button((inner.right - 180, inner.bottom - 50, 180, 46), "New test", primary=True)
        self.again.draw(s, f)
        self.menu_btn.draw(s, f)

    def run(self):
        running = True
        while running:
            if self.state == "menu":
                ok = self.draw_menu()
            elif self.state == "draw":
                area = self.draw_canvas()
            elif self.state == "race":
                self.session.step()
                hdr = f"{self.race_info['model']}   ·   tester {self.name.value.strip()}"
                self.session.draw(self.screen, self.fonts, hdr)
                if self.session.done and self.result is None:
                    self.finish_race()
                if self.result is not None:
                    self.draw_result()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif self.state == "menu":
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE and not self.name.active:
                        running = False
                    self.menu_event(e, ok)
                elif self.state == "draw":
                    self.canvas_event(e, area)
                elif self.state == "race":
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        self.state = "menu"
                    elif self.result is not None and self.again.hit(e):
                        tr = self.session.track
                        drawn = self.race_info["source"] == "drawn"
                        self.begin_race(seed=None if drawn else self.race_info["seed"], track=tr if drawn else None)
                    elif self.result is not None and self.menu_btn.hit(e):
                        self.state = "menu"
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    DemoApp().run()

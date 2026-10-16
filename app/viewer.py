"""Watch a trained driver.

    python -m app.viewer                                   # latest run with snapshots
    python -m app.viewer --run experiments/results/stage0/2x16/seed0 --track 400003
    python -m app.viewer --pack                           # all 8 elites of the snapshot

Keys: ESC quit · N new random track · R restart · TAB follow next car ·
      1/2/3 zoom · P screenshot (figures/screens/)
"""
import argparse
import os
import time

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
import pygame

from app import render as R
from app.models import Driver, ROOT, list_runs
from racing.network import Population
from racing.physics import CarSwarm
from racing.sim import get_track

OUT_LABELS = ["Steer", "Throttle", "Brake", "Handbr."]


class Session:
    def __init__(self, driver, seed, difficulty, zoom=6.0, track=None, label=None, loop=True):
        self.driver = driver
        self.seed, self.difficulty = seed, difficulty
        self.track = track if track is not None else get_track(seed, difficulty)
        self.label = label or f"Track {seed}"
        self.loop = loop
        self.done = False
        self.zooms = {}
        self.set_zoom(zoom)
        self.minimap = R.Minimap(self.track, pygame.Rect(R.VIEW.right + R.PAD + 1, R.PAD + 34,
                                                          R.SIDEBAR_W - 2 * R.PAD - 2, 230))
        self.best_lap = None
        self.laps = 0
        self.reset()

    def set_zoom(self, z):
        if z not in self.zooms:
            self.zooms[z] = R.TrackRenderer(self.track, scale=z)
        self.renderer = self.zooms[z]

    def reset(self):
        self.cars = CarSwarm(self.driver.pop.n, self.track, num_sensors=self.driver.spec.num_rays,
                             sensor_noise=0.02, dropout=0.01, rng=np.random.default_rng())
        self.focus = 0
        self.trail_pts, self.trail_v = [], []
        self.ranges = None
        self.acts = None

    def step(self):
        if self.done:
            return
        c = self.cars
        self.ranges = c.sensor_readings()
        x = self.driver.spec.build(self.ranges, c)
        y, self.acts = self.driver.pop.forward(x, return_activations=True)
        c.step(*Population.controls(y))
        if not c.active[self.focus] and c.active.any():
            self.focus = int(np.argmax(np.where(c.active, c.progress, -1)))
        self.trail_pts.append(c.pos[self.focus].copy())
        self.trail_v.append(c.vx[self.focus])
        self.trail_pts, self.trail_v = self.trail_pts[-4000:], self.trail_v[-4000:]
        if c.finished.any():
            lap = np.nanmin(c.lap_time)
            self.best_lap = lap if self.best_lap is None else min(self.best_lap, lap)
        if not c.active.any():
            self.laps += 1
            if self.loop:
                self.reset()
            else:
                self.done = True

    def draw(self, screen, fonts, header):
        c, i = self.cars, self.focus
        screen.fill(R.BG)
        to_screen = self.renderer.draw(screen, R.VIEW, c.pos[i])
        if len(self.trail_pts) > 1:
            pts = to_screen(np.array(self.trail_pts))
            for k in range(len(pts) - 1):
                pygame.draw.line(screen, R.speed_color(self.trail_v[k]), pts[k], pts[k + 1], 3)
        R.draw_rays(screen, to_screen, c, i, self.ranges[i] if self.ranges is not None else None)
        for k in range(c.n):
            if c.alive[k] and k != i:
                R.draw_car(screen, to_screen, c.pos[k], c.psi[k], c.p, (190, 194, 204))
        drift = abs(np.degrees(c.slip_angle[i])) > 6
        R.draw_car(screen, to_screen, c.pos[i], c.psi[i], c.p, R.ACCENT, (255, 120, 40) if drift else None)

        # header strip inside the view
        strip = pygame.Rect(R.PAD, R.PAD, R.VIEW.width - 2 * R.PAD, 34)
        s = pygame.Surface(strip.size, pygame.SRCALPHA)
        s.fill((14, 16, 20, 190))
        screen.blit(s, strip.topleft)
        R.text(screen, header, fonts.title, R.TEXT, strip.inflate(-24, 0))

        side = pygame.Rect(R.VIEW.right + R.PAD, R.PAD, R.SIDEBAR_W - 2 * R.PAD, 0)
        mm = pygame.Rect(side.x, side.y, side.width, 276)
        R.panel(screen, mm, fonts, f"{self.label} · {self.track['length'] / 1000:.2f} km")
        trail = (np.array(self.trail_pts), self.trail_v) if len(self.trail_pts) > 1 else None
        self.minimap.draw(screen, c.pos, c.alive & ~c.finished, focus=i, trail=trail)
        tel = pygame.Rect(side.x, mm.bottom + 12, side.width, 268)
        R.telemetry(screen, fonts, tel, c, i, self.best_lap)
        net = pygame.Rect(side.x, tel.bottom + 12, side.width, R.H - tel.bottom - 12 - R.PAD)
        if self.acts is not None:
            w, _ = self.driver.pop.get(i)
            R.network_panel(screen, fonts, net, w, [a[i] for a in self.acts], self.driver.input_labels, OUT_LABELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--track", type=int, default=400003)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--pack", action="store_true", help="drive all stored elites together")
    ap.add_argument("--frames", type=int, default=0, help="headless: save N frames to figures/frames and exit")
    ap.add_argument("--frame-every", type=int, default=1)
    a = ap.parse_args()
    if a.frames:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    run = a.run or list_runs()[-1]
    driver = Driver(run, which="pack" if a.pack else "champion")

    pygame.init()
    screen = pygame.display.set_mode((R.W, R.H))
    pygame.display.set_caption("Neural-network driver")
    fonts = R.Fonts()
    clock = pygame.time.Clock()
    sess = Session(driver, a.track, a.difficulty)
    zoom_levels = {pygame.K_1: 3.0, pygame.K_2: 6.0, pygame.K_3: 9.0}
    shot_dir = os.path.join(ROOT, "figures", "screens")
    frame_dir = os.path.join(ROOT, "figures", "frames")
    rng = np.random.default_rng()
    frame = 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_n:
                    sess = Session(driver, int(rng.integers(0, 2**31)), a.difficulty)
                elif e.key == pygame.K_r:
                    sess.reset()
                elif e.key == pygame.K_TAB:
                    alive = np.where(sess.cars.active)[0]
                    if len(alive):
                        sess.focus = int(alive[(np.searchsorted(alive, sess.focus, side="right")) % len(alive)])
                elif e.key in zoom_levels:
                    sess.set_zoom(zoom_levels[e.key])
                elif e.key == pygame.K_p:
                    os.makedirs(shot_dir, exist_ok=True)
                    pygame.image.save(screen, os.path.join(shot_dir, f"screen_{time.strftime('%H%M%S')}.png"))
        sess.step()
        header = f"{driver.name}   ·   difficulty {a.difficulty:.2f}   ·   laps {sess.laps}"
        sess.draw(screen, fonts, header)
        pygame.display.flip()
        if a.frames:
            if frame % a.frame_every == 0:
                os.makedirs(frame_dir, exist_ok=True)
                pygame.image.save(screen, os.path.join(frame_dir, f"f{frame // a.frame_every:05d}.png"))
            frame += 1
            if frame >= a.frames * a.frame_every:
                running = False
        else:
            clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()

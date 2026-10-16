"""Watch the single best-trained car drive, on its own, in a window.

This reuses main.py wholesale - the same track generation, the same PCA fit
(reproduced deterministically from the seeds, exactly as training does), the
same network forward pass and the same visualizer - so what you see here is
the identical simulation the population trains in, just with one car instead of
the whole swarm. It loads a checkpoint, takes the champion genome (the top
elite, stored at index 0 right after each evolve), and drives it in a loop:
finish a lap or crash, reset, go again, so you can watch it repeatedly.

    python showcase.py                         # champion of checkpoints_fast, its current track
    python showcase.py --checkpoint-dir checkpoints
    python showcase.py --track-index 3         # watch it on an earlier/other track
    python showcase.py --seed 42 --track-seed 7  # must match how it was trained (defaults do)
"""
import argparse
import os

import numpy as np

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from car import CarSwarm
from neural_network import Population
from pca import PCA
import visualizer as viz
import main as M


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default="checkpoints_fast",
                   help="which checkpoint to watch (default: checkpoints_fast, where fast_train writes)")
    p.add_argument("--track-index", type=int, default=None,
                   help="track to drive on (default: the checkpoint's current track)")
    p.add_argument("--seed", type=int, default=42, help="must match the training --seed (for the PCA fit)")
    p.add_argument("--track-seed", type=int, default=None,
                   help="base track seed (default: the checkpoint's own)")
    p.add_argument("--best-index", type=int, default=0,
                   help="which individual to drive (0 = champion / top elite)")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint = M.load_checkpoint(args.checkpoint_dir, np.random.default_rng(args.seed))
    if checkpoint is None:
        raise SystemExit(f"no checkpoint found in {args.checkpoint_dir!r}")
    population, meta = checkpoint
    base_track_seed = args.track_seed if args.track_seed is not None else meta["track_seed"]
    track_index = args.track_index if args.track_index is not None else meta.get("track_index", 0)
    generation = meta.get("next_generation", 0)

    # Reproduce the exact PCA the network was trained against: fit once on the
    # very first track from the same seeds, identical to main.main().
    first_track, _ = M.track_for_index(base_track_seed, 0)
    calibration_raw = M.collect_calibration_data(first_track, np.random.default_rng(args.seed))
    pca = PCA(M.PCA_COMPONENTS).fit(calibration_raw)

    # Pull the champion genome into a one-car population so forward() drives just it.
    champ = Population(1, population.layer_sizes, rng=np.random.default_rng(0))
    w, b = population.get_individual(args.best_index)
    champ.set_individual(0, w, b)

    track, track_seed = M.track_for_index(base_track_seed, track_index)
    track_length = track["cumdist"][-1]

    pygame.init()
    screen = pygame.display.set_mode((M.SCREEN_W, M.SCREEN_H))
    pygame.display.set_caption(f"Best car - {args.checkpoint_dir} gen {generation} - track {track_index}")
    clock = pygame.time.Clock()
    fonts = viz.init_fonts()
    to_screen, scale = viz.compute_view_transform(track, M.TRACK_PANEL)

    cars = CarSwarm(1, track, num_sensors=M.NUM_SENSORS_RAW)
    lap_count = 0
    step_in_lap = 0
    lap_times = []          # seconds per completed lap
    best_lap = None
    speed_trace = []        # live speed history for the speed panel

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        raw = cars.sensor_readings(track)
        x = M.build_input(raw, cars.speed, cars.wheel_angle, cars.slip, pca, cars.max_speed)
        out = champ.forward(x)
        cars.step(out[:, 0], (out[:, 1] + 1) / 2,
                  np.clip(out[:, 2], 0, 1), np.clip(out[:, 3], 0, 1), M.DT, track)
        step_in_lap += 1
        speed_trace.append(float(cars.speed[0]))

        finished = cars.progress[0] >= track_length - 1.0
        crashed = not cars.alive[0]
        if finished:
            lap_time = step_in_lap * M.DT
            lap_times.append(lap_time)
            best_lap = lap_time if best_lap is None else min(best_lap, lap_time)
            lap_count += 1
            cars.reset(track)
            step_in_lap = 0
        elif crashed:
            cars.reset(track)
            step_in_lap = 0

        # --- draw (same panels/visualizer as training) ---
        viz.draw_background(screen)
        viz.draw_track(screen, M.TRACK_PANEL, track, to_screen)
        viz.draw_cars(screen, M.TRACK_PANEL, cars, to_screen, scale, best_idx=0)

        a = x[0]
        activations = [a]
        for wl, bl in zip(w, b):
            a = np.tanh(wl @ a + bl[:, 0])
            activations.append(a)
        viz.draw_network(screen, fonts, M.NN_PANEL, w, activations, M.INPUT_LABELS, M.OUTPUT_LABELS)

        _draw_showcase_hud(screen, fonts, M.STATS_PANEL, generation, track_index,
                           cars, track_length, lap_count, best_lap)
        viz.draw_speed_chart(screen, fonts, M.SPEED_PANEL, speed_trace[-200:])

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _draw_showcase_hud(surface, fonts, rect, generation, track_index, cars, track_length,
                       lap_count, best_lap):
    viz._frame(surface, rect)
    x, y, w, h = rect
    pad = 10
    surface.blit(fonts["title"].render("Best car", True, viz.PALETTE["text"]), (x + pad, y + pad))
    speed_kmh = cars.speed[0] * 3.6  # world units ~= metres, so *3.6 gives a km/h-ish read
    rows = [
        ("Generation", f"{generation}"),
        ("Track", f"{track_index}"),
        ("Lap", f"{cars.progress[0] / track_length * 100:4.0f} %"),
        ("Speed", f"{speed_kmh:4.0f} km/h"),
        ("Slip", f"{cars.slip[0]:+5.1f}"),
        ("Laps done", f"{lap_count}"),
        ("Best lap", "-" if best_lap is None else f"{best_lap:4.1f} s"),
    ]
    ty = y + pad + 24
    for label, value in rows:
        surface.blit(fonts["label"].render(label, True, viz.PALETTE["text"]), (x + pad, ty))
        vs = fonts["value"].render(value, True, viz.PALETTE["text"])
        surface.blit(vs, (x + w - pad - vs.get_width(), ty))
        ty += 20


if __name__ == "__main__":
    main()

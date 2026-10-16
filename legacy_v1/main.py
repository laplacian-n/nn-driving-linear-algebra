import argparse
import csv
import datetime
import json
import logging
import os
import time

import numpy as np

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from track import generate_track
from car import CarSwarm
from neural_network import Population
from genetic_algorithm import evolve, mutate
from pca import PCA
import visualizer as viz

SCREEN_W, SCREEN_H = 1300, 800
NUM_SENSORS_RAW = 18
PCA_COMPONENTS = 8  # more sensor detail retained - a narrow F1-tight track needs a
                    # finer read of the road ahead to place a precise racing line than
                    # the 6 components the wider old tracks got by on
HIDDEN_LAYERS = [32, 28, 24, 20, 16, 10]  # deeper/wider than before: the drift physics
                                          # (slip, understeer, when to sacrifice grip)
                                          # is a much richer control problem than the old
                                          # point-mass model, so the net gets more capacity
OUTPUT_DIM = 4  # steering, throttle, brake, handbrake
MAX_STEPS = 4500  # a genuinely lap-capable driver still only averages ~30-35% of top
                   # speed (F1-scaled physics, technical track) - 3000 measured too
                   # optimistic in practice and was cutting a working elite off ~6s
                   # short of the finish line every single generation, so nothing ever
                   # earned the completion time-bonus that's supposed to reward speed
DT = 1 / 60
TIME_BONUS_WEIGHT = 0.6   # fitness points per simulation step saved on a finished lap -
                          # raised so finishing *fast* clearly out-scores finishing at all,
                          # pushing the population toward carrying speed instead of settling
                          # for a safe slow lap
AVG_SPEED_BONUS_WEIGHT = 3.0  # fitness points per m/s of average speed made good over
                               # the run - without this, a car that limps through a
                               # corner and one that carries speed through it score the
                               # same as long as they reach the same distance, since
                               # nothing else prices in *how long it took*
LAPS_PER_TRACK = 2000  # HARD CEILING only - a track normally rotates on demonstrated
                        # mastery (MIN_LAPS_BEFORE_ROTATE + MASTERY_STAGNATION below), and
                        # this cap just stops a track that somehow never plateaus from
                        # trapping the run on one layout forever. Set high so mastery, not
                        # a blind lap count, is what actually decides when to move on.
TRACK_DIFFICULTY_RAMP_TRACKS = 40  # track rotations to reach MAX_DIFFICULTY - stretched
                                    # from 20 so the road narrows/complexifies gradually
                                    # enough that the population can actually keep mastering
                                    # each step instead of being dragged onto tracks it can
                                    # only just survive (which capped lap speed and stalled
                                    # all further learning)
MAX_DIFFICULTY = 0.6  # hardest track the curriculum will ever reach for now. Full F1-narrow
                       # (1.0) proved unlearnable for a GA-trained net of this size - it
                       # could survive but never build speed there. Capping here keeps every
                       # track hard-but-masterable so lap speed can actually keep climbing;
                       # raise it later once the population is fast at 0.6.
STAGNATION_PATIENCE = 25  # generations without beating the CURRENT track's best before
                          # reheating the GA (reset on every track rotation - see below)
STAGNATION_BOOST = 3.0    # mutation-strength/immigrant-fraction multiplier while reheating
MIN_LAPS_BEFORE_ROTATE = 60  # don't even consider rotating tracks until this many laps have
                              # been finished on it - a floor so a track that's briefly
                              # stagnant early (before anything really works) isn't abandoned
MASTERY_STAGNATION = 15  # ...then rotate once best-on-this-track hasn't improved in this many
                          # generations, i.e. the population has plateaued and squeezed out
                          # what it can - "master, THEN move on" instead of a blind lap count
INPUT_LABELS = [f"PC{i + 1}" for i in range(PCA_COMPONENTS)] + ["Speed", "Wheel", "Slip"]
OUTPUT_LABELS = ["Steering", "Throttle", "Brake", "Handbrake"]
TRACK_CENTER = (0, 0)  # world-space only; the view transform maps it into TRACK_PANEL

# Layout: track boxed top-left, NN filling the whole right side, three stat/
# graph panels in a row along the bottom-left - each its own bordered frame,
# none overlapping.
TRACK_PANEL = (20, 20, 620, 500)
NN_PANEL = (660, 20, 620, 760)
STATS_PANEL = (20, 540, 196, 240)
PROGRESS_PANEL = (231, 540, 196, 240)
SPEED_PANEL = (442, 540, 196, 240)


def parse_args():
    p = argparse.ArgumentParser(description="Neural-network driving trained with a genetic algorithm.")
    p.add_argument("--headless", action="store_true", help="run without a window, as fast as possible")
    p.add_argument("--generations", type=int, default=None, help="stop after N generations (default: run forever)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the population/GA")
    p.add_argument("--track-seed", type=int, default=7, help="RNG seed for the first track (ignored on resume)")
    p.add_argument("--num-cars", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS, help="max simulated steps per generation")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--fresh", action="store_true", help="ignore any saved checkpoint and start over")
    return p.parse_args()


def setup_run_logging(log_dir):
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("nn_driving")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(run_dir, "run.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    csv_file = open(os.path.join(run_dir, "history.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["generation", "elapsed_sec", "alive", "best_distance", "best_fitness",
                          "mean_fitness", "median_fitness", "best_fitness_ever", "laps_finished",
                          "best_lap_speed", "track_length", "gens_since_improvement"])
    return logger, csv_writer, csv_file, run_dir


def track_for_index(base_track_seed, track_index):
    seed = base_track_seed + track_index
    difficulty = min(MAX_DIFFICULTY, track_index / TRACK_DIFFICULTY_RAMP_TRACKS)
    return generate_track(center=TRACK_CENTER, seed=seed, difficulty=difficulty), seed


def resize_population(old_pop, new_n, rng, clone_mutation=0.03):
    """Return a new Population of new_n individuals that reuses old_pop's trained
    genomes. If growing, the first old_pop.n slots are exact copies and the rest
    are lightly-mutated clones of random existing individuals (fresh diversity
    without discarding what was learned). If shrinking, the first new_n are kept.
    """
    new_pop = Population(new_n, old_pop.layer_sizes, rng=rng)
    for slot in range(new_n):
        src = slot if slot < old_pop.n else int(rng.integers(0, old_pop.n))
        w, b = old_pop.get_individual(src)
        if slot >= old_pop.n:
            w, b = mutate(w, b, rate=0.5, strength=clone_mutation, rng=rng)
        new_pop.set_individual(slot, w, b)
    return new_pop


def checkpoint_paths(checkpoint_dir):
    return os.path.join(checkpoint_dir, "population.npz"), os.path.join(checkpoint_dir, "meta.json")


def load_checkpoint(checkpoint_dir, rng):
    pop_path, meta_path = checkpoint_paths(checkpoint_dir)
    if not (os.path.exists(pop_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    population = Population.load(pop_path, rng=rng)
    return population, meta


def save_checkpoint(checkpoint_dir, population, next_generation, base_track_seed,
                     track_index, laps_since_track_start, best_ever, history, speed_history,
                     gens_since_improvement, best_this_track):
    os.makedirs(checkpoint_dir, exist_ok=True)
    pop_path, meta_path = checkpoint_paths(checkpoint_dir)
    population.save(pop_path)
    meta = {
        "next_generation": next_generation,
        "track_seed": base_track_seed,
        "track_index": track_index,
        "laps_since_track_start": laps_since_track_start,
        "best_ever": best_ever,
        "history": history[-500:],
        "speed_history": speed_history[-500:],
        "gens_since_improvement": gens_since_improvement,
        "best_this_track": best_this_track if np.isfinite(best_this_track) else None,
    }
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_meta, meta_path)


def collect_calibration_data(track, rng, num_samples=1500):
    """Sample random positions/headings around the track so PCA can be fit on
    realistic sensor data before any car has actually driven."""
    centerline = track["centerline"]
    idx = rng.integers(0, len(centerline), num_samples)
    probe = CarSwarm(num_samples, track, num_sensors=NUM_SENSORS_RAW)
    probe.pos = centerline[idx] + rng.normal(0, 5, (num_samples, 2))
    probe.angle = rng.uniform(-np.pi, np.pi, num_samples)
    return probe.sensor_readings(track)


def build_input(raw_sensors, speed, wheel_angle, slip, pca, max_speed):
    # Slip (sideways velocity) is fed in explicitly: with the drift physics the
    # car can be pointing one way while moving another, and a controller that
    # can't sense that mismatch is driving blind through every slide - it can
    # neither catch a slide nor deliberately hold a drift.
    reduced = pca.transform(raw_sensors)
    return np.concatenate([reduced, speed[:, None] / max_speed,
                           wheel_angle[:, None], slip[:, None] / max_speed], axis=1)


def run_generation(track, population, pca, num_cars, max_steps, generation, headless,
                    best_ever_so_far, history, speed_history, to_screen=None, scale=None,
                    render_ctx=None):
    cars = CarSwarm(num_cars, track, num_sensors=NUM_SENSORS_RAW)
    track_length = track["cumdist"][-1]
    finish_step = np.full(num_cars, -1)
    steps_survived = np.zeros(num_cars, dtype=int)
    steps_run = 0

    for step in range(max_steps):
        steps_run = step + 1
        active_before = cars.alive & ~cars.finished
        raw = cars.sensor_readings(track)
        x = build_input(raw, cars.speed, cars.wheel_angle, cars.slip, pca, cars.max_speed)
        out = population.forward(x)
        steer = out[:, 0]
        # Throttle uses the tanh output's full range (so an untrained/near-zero
        # network still idles forward at ~half throttle by default). Brake and
        # handbrake are clipped instead of rescaled, so their default at
        # near-zero weights is "not pressed" - rescaling them the same way as
        # throttle would make an untrained network start with brake and
        # handbrake both half-engaged, which alone out-decelerates the throttle
        # and the car never leaves the start line.
        throttle = (out[:, 1] + 1) / 2
        brake = np.clip(out[:, 2], 0, 1)
        handbrake = np.clip(out[:, 3], 0, 1)
        cars.step(steer, throttle, brake, handbrake, DT, track)
        steps_survived[active_before] = step + 1

        # A car that completes the lap stops accumulating distance (progress is
        # capped at track_length), so without this every finisher would tie on
        # fitness - completing sooner is what should separate them.
        just_finished = (cars.progress >= track_length - 1.0) & (finish_step < 0)
        finish_step[just_finished] = step
        cars.freeze_finished(just_finished)

        # Stop once nothing can change anymore: every car is either dead or
        # frozen-finished. A finished car stays "alive" forever now, so the old
        # "no one's alive" check alone would run every remaining step all the
        # way to max_steps even after every other car has already crashed.
        if np.all(cars.finished | ~cars.alive):
            break

        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            screen, clock, fonts = render_ctx
            viz.draw_background(screen)
            viz.draw_track(screen, TRACK_PANEL, track, to_screen)
            best_idx = int(np.argmax(cars.progress))
            viz.draw_cars(screen, TRACK_PANEL, cars, to_screen, scale, best_idx=best_idx)

            w, b = population.get_individual(best_idx)
            a = x[best_idx]
            activations = [a]
            for wl, bl in zip(w, b):
                a = np.tanh(wl @ a + bl[:, 0])
                activations.append(a)
            viz.draw_network(screen, fonts, NN_PANEL, w, activations, INPUT_LABELS, OUTPUT_LABELS)
            viz.draw_hud(screen, fonts, STATS_PANEL, generation, int(cars.alive.sum()),
                         num_cars, cars.progress.max(), max(best_ever_so_far, cars.progress.max()),
                         track_length, laps_finished=int((finish_step >= 0).sum()))
            viz.draw_sparkline(screen, fonts, PROGRESS_PANEL, history)
            viz.draw_speed_chart(screen, fonts, SPEED_PANEL, speed_history)

            pygame.display.flip()
            clock.tick(60)

    time_bonus = np.where(finish_step >= 0, (max_steps - finish_step) * TIME_BONUS_WEIGHT, 0.0)
    # Average speed made good over however long the car was actually active
    # (until it crashed, finished, or the episode ended) - see
    # AVG_SPEED_BONUS_WEIGHT above for why this term exists.
    avg_speed = cars.progress / np.maximum(steps_survived * DT, DT)
    fitness = cars.progress + time_bonus + AVG_SPEED_BONUS_WEIGHT * avg_speed
    finishers = finish_step[finish_step >= 0]
    best_lap_speed = float(track_length / (finishers.min() * DT)) if len(finishers) else 0.0
    laps_finished = int(len(finishers))
    return fitness, cars.progress, steps_run, int(cars.alive.sum()), laps_finished, best_lap_speed


def main():
    args = parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    logger, csv_writer, csv_file, run_dir = setup_run_logging(args.log_dir)
    rng = np.random.default_rng(args.seed)

    input_dim = PCA_COMPONENTS + 3  # + normalized speed + wheel angle + slip
    layer_sizes = [input_dim] + HIDDEN_LAYERS + [OUTPUT_DIM]

    checkpoint = None if args.fresh else load_checkpoint(args.checkpoint_dir, rng)
    if checkpoint:
        population, meta = checkpoint
        base_track_seed = meta["track_seed"]
        start_generation = meta["next_generation"]
        track_index = meta.get("track_index", 0)
        laps_since_track_start = meta.get("laps_since_track_start", 0)
        best_ever = meta.get("best_ever", 0.0)
        history = meta.get("history", [])
        speed_history = meta.get("speed_history", [])
        gens_since_improvement = meta.get("gens_since_improvement", 0)
        best_this_track = meta.get("best_this_track")
        best_this_track = -np.inf if best_this_track is None else best_this_track
        if list(population.layer_sizes) != layer_sizes:
            logger.warning("checkpoint architecture doesn't match current network - starting fresh instead")
            checkpoint = None
        elif population.n != args.num_cars:
            # Same brain shape, different population size: resize in place rather
            # than throwing the trained genomes away, so growing the population
            # (D) keeps everything learned so far and just adds room for wider
            # search - the extra slots start as lightly-mutated clones of the
            # existing individuals.
            logger.info("resizing checkpoint population %d -> %d (keeping trained genomes)",
                        population.n, args.num_cars)
            population = resize_population(population, args.num_cars, rng)

    if not checkpoint:
        population = Population(args.num_cars, layer_sizes, rng=rng)
        base_track_seed = args.track_seed
        start_generation = 0
        track_index = 0
        laps_since_track_start = 0
        best_ever = 0.0
        history, speed_history = [], []
        gens_since_improvement = 0
        best_this_track = -np.inf
        logger.info("starting fresh run")
    else:
        logger.info("resumed from checkpoint at generation=%d (track_index=%d, laps_since_track_start=%d)",
                     start_generation, track_index, laps_since_track_start)

    # PCA is fixed for the whole run (including across track rotations), fit
    # once from the very first track. It's recomputed deterministically from
    # base_track_seed + args.seed rather than saved, so it's reproducible
    # across resumes without needing to serialize it separately.
    first_track, _ = track_for_index(base_track_seed, 0)
    calibration_raw = collect_calibration_data(first_track, np.random.default_rng(args.seed))
    pca = PCA(PCA_COMPONENTS).fit(calibration_raw)

    config = {
        "num_cars": args.num_cars, "num_sensors_raw": NUM_SENSORS_RAW,
        "pca_components": PCA_COMPONENTS, "hidden_layers": HIDDEN_LAYERS,
        "layer_sizes": layer_sizes, "max_steps": args.max_steps,
        "seed": args.seed, "base_track_seed": base_track_seed,
        "laps_per_track": LAPS_PER_TRACK,
        "pca_explained_variance": pca.explained_variance.tolist(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info("run_dir=%s pca_explained_variance=%s", run_dir, np.round(pca.explained_variance, 3).tolist())

    render_ctx = None
    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Neural Network Driving - Linear Algebra Project")
        clock = pygame.time.Clock()
        fonts = viz.init_fonts()
        render_ctx = (screen, clock, fonts)

    generation = start_generation
    track, current_track_index = track_for_index(base_track_seed, track_index)
    to_screen, scale = viz.compute_view_transform(track, TRACK_PANEL) if not args.headless else (None, None)
    logger.info("track index=%d seed=%d (length=%.0f), laps_since_track_start=%d",
                track_index, current_track_index, track["cumdist"][-1], laps_since_track_start)

    try:
        while args.generations is None or generation - start_generation < args.generations:
            t0 = time.time()
            fitness, distance, steps_run, alive_at_end, laps_finished, best_lap_speed = run_generation(
                track, population, pca, args.num_cars, args.max_steps, generation, args.headless,
                best_ever, history, speed_history, to_screen=to_screen, scale=scale, render_ctx=render_ctx)
            elapsed = time.time() - t0
            gen_best = float(fitness.max())
            best_ever = max(best_ever, gen_best)
            # Stagnation is judged against the CURRENT track's own best, not a
            # global best_ever. Tracks differ in length so their raw fitness
            # scales differ; a global best set on the longest track seen can be
            # literally unbeatable on every shorter one, which used to leave the
            # reheat permanently latched on and the population perpetually
            # disrupted. best_this_track resets on every rotation (below).
            if gen_best > best_this_track + 1e-6:
                best_this_track = gen_best
                gens_since_improvement = 0
            else:
                gens_since_improvement += 1
            history.append(gen_best)
            speed_history.append(best_lap_speed)
            laps_since_track_start += laps_finished

            diversity_boost = STAGNATION_BOOST if gens_since_improvement >= STAGNATION_PATIENCE else 1.0
            if gens_since_improvement == STAGNATION_PATIENCE:
                logger.info("no improvement in %d generations - reheating mutation/immigrants x%.1f",
                            STAGNATION_PATIENCE, STAGNATION_BOOST)

            logger.info(
                "gen=%-4d steps=%-5d elapsed=%5.1fs alive_at_end=%2d/%d laps=%2d (track_total=%3d/%d) "
                "best_dist=%6.0f best_fit=%6.0f mean_fit=%6.0f median_fit=%6.0f best_ever=%6.0f best_lap_speed=%5.0f "
                "stagnant=%3d",
                generation, steps_run, elapsed, alive_at_end, args.num_cars, laps_finished,
                laps_since_track_start, LAPS_PER_TRACK,
                distance.max(), fitness.max(), fitness.mean(), np.median(fitness), best_ever, best_lap_speed,
                gens_since_improvement,
            )
            csv_writer.writerow([generation, f"{elapsed:.2f}", alive_at_end, f"{distance.max():.1f}",
                                  f"{fitness.max():.1f}", f"{fitness.mean():.1f}",
                                  f"{np.median(fitness):.1f}", f"{best_ever:.1f}", laps_finished,
                                  f"{best_lap_speed:.1f}", f"{track['cumdist'][-1]:.1f}", gens_since_improvement])
            csv_file.flush()

            population = evolve(population, fitness, generation=generation,
                                 diversity_boost=diversity_boost, rng=rng)
            generation += 1

            # Rotate on demonstrated mastery, not a blind lap count: once enough
            # laps are in AND the best-on-this-track has plateaued (nothing
            # better found in MASTERY_STAGNATION gens), the population has
            # squeezed out what it can here, so move on. LAPS_PER_TRACK is kept
            # only as a hard ceiling so a track that never plateaus can't trap
            # the run on one layout forever (which would just overfit to it).
            mastered = (laps_since_track_start >= MIN_LAPS_BEFORE_ROTATE
                        and gens_since_improvement >= MASTERY_STAGNATION)
            if mastered or laps_since_track_start >= LAPS_PER_TRACK:
                track_index += 1
                laps_since_track_start = 0
                track, current_track_index = track_for_index(base_track_seed, track_index)
                # New track = new fitness scale, so the stagnation baseline
                # starts over; otherwise the first few gens on a shorter/harder
                # track would falsely read as "no improvement" and reheat.
                best_this_track = -np.inf
                gens_since_improvement = 0
                if not args.headless:
                    to_screen, scale = viz.compute_view_transform(track, TRACK_PANEL)
                logger.info("track rotated -> index=%d seed=%d (length=%.0f)",
                            track_index, current_track_index, track["cumdist"][-1])

            save_checkpoint(args.checkpoint_dir, population, generation, base_track_seed,
                             track_index, laps_since_track_start, best_ever, history, speed_history,
                             gens_since_improvement, best_this_track)
    except SystemExit:
        logger.info("window closed, stopping")
    finally:
        csv_file.close()
        logger.info("finished after generation %d, best_ever=%.0f, logs at %s, checkpoint at %s",
                    generation, best_ever, run_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()

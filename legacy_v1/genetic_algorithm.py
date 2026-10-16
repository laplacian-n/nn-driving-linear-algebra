import numpy as np

from neural_network import Population


def tournament_select(fitness, k=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    contenders = rng.integers(0, len(fitness), size=k)
    return contenders[np.argmax(fitness[contenders])]


def crossover(weights_a, biases_a, weights_b, biases_b, rng):
    # Per-LAYER, not per-weight: a layer's weights only work as a coordinated
    # group (they jointly compute one transformation), so splicing individual
    # weight VALUES between two unrelated networks scrambles that coordination
    # in both parents at once - a well-known reason naive elementwise
    # crossover tends to actively destroy neural-net behavior rather than
    # blend it. Keeping each layer intact from whichever parent it came from
    # preserves whatever internal coherence that parent had within it.
    child_w, child_b = [], []
    for wa, wb, ba, bb in zip(weights_a, weights_b, biases_a, biases_b):
        take_a = rng.random() < 0.5
        child_w.append(wa.copy() if take_a else wb.copy())
        child_b.append(ba.copy() if take_a else bb.copy())
    return child_w, child_b


def mutate(weights, biases, rate, strength, rng):
    for arr_list in (weights, biases):
        for arr in arr_list:
            mask = rng.random(arr.shape) < rate
            arr[mask] += rng.normal(0, strength, size=arr.shape)[mask]
    return weights, biases


def evolve(population, fitness, generation=0, elite_frac=0.08, refine_frac=0.2,
           mutation_rate=0.15, mutation_strength=0.35,
           refine_mutation_rate=0.5, refine_mutation_strength=0.03,
           random_immigrant_frac=0.08, diversity_boost=1.0, rng=None):
    """Build the next generation from elites (kept exact), fine-mutated
    "refinement" clones of the best genomes, tournament-selected
    crossover+mutation children, and a few fresh random individuals.

    Refinement exists because elitism + coarse mutation alone can only ever
    copy the best genome unchanged or break it: a policy that has to stay
    correct for a multi-thousand-step lap is fragile, so a mutation sized for
    exploring from scratch (mutation_strength) usually wrecks it rather than
    nudging it, and crossing it with an unrelated (usually much weaker)
    tournament winner splices in incompatible weights. Once something actually
    works, what's needed is hill-climbing around it - touch most weights but
    by a tiny amount (refine_mutation_strength), so a slightly-faster or
    slightly-more-precise variant has a real chance of surviving instead of
    every descendant of the champion being either identical or crashed.

    Random immigrants keep some genetic diversity in the pool every generation,
    which matters because with only crossover+mutation the whole population can
    prematurely converge on one mediocre strategy (e.g. "hug the inside wall and
    stop") and never escape it. Mutation strength decays with generation so early
    exploration is wide and later refinement is fine-grained.

    diversity_boost temporarily scales up mutation strength (both the explorer
    and refinement kind) and the immigrant fraction (call site raises it when
    best fitness has stalled for many generations in a row).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = population.n
    order = np.argsort(fitness)[::-1]
    num_elite = max(1, int(n * elite_frac))
    num_refine = int(n * refine_frac)
    num_immigrants = int(n * min(0.4, random_immigrant_frac * diversity_boost))
    elites = order[:num_elite]
    decayed_strength = mutation_strength * max(0.3, 1.0 - generation * 0.015) * diversity_boost
    decayed_refine_strength = refine_mutation_strength * diversity_boost

    new_pop = Population(n, population.layer_sizes, rng=rng)

    for slot, idx in enumerate(elites):
        w, b = population.get_individual(idx)
        new_pop.set_individual(slot, w, b)

    refine_pool = order[:max(num_elite, 3)]
    for slot in range(num_elite, num_elite + num_refine):
        base_idx = refine_pool[rng.integers(0, len(refine_pool))]
        w, b = population.get_individual(base_idx)
        w, b = mutate(w, b, refine_mutation_rate, decayed_refine_strength, rng)
        new_pop.set_individual(slot, w, b)

    num_bred = n - num_elite - num_refine - num_immigrants
    for slot in range(num_elite + num_refine, num_elite + num_refine + num_bred):
        pa = tournament_select(fitness, rng=rng)
        pb = tournament_select(fitness, rng=rng)
        wa, ba = population.get_individual(pa)
        wb, bb = population.get_individual(pb)
        cw, cb = crossover(wa, ba, wb, bb, rng)
        cw, cb = mutate(cw, cb, mutation_rate, decayed_strength, rng)
        new_pop.set_individual(slot, cw, cb)

    # slots [num_elite + num_refine + num_bred, n) are left as the fresh
    # random weights Population() already initialized them with.

    return new_pop

"""Genetic algorithm (Deep-GA style, Such et al. 2017): truncation selection,
elitism and Gaussian mutation only - no crossover, no random immigrants - so
the algorithm is identical for every network depth.

Mutation is scaled by each layer's fan-in (same scale as the initialisation),
so one mutation perturbs a neuron's pre-activation by ~sigma_rel * |input|
regardless of how wide the layer is. Each child draws its own sigma_rel from a
small set, mixing fine refinement and wide exploration in every generation.
"""
import numpy as np

from .network import Population


def evolve(pop, fitness, rng, elite=5, parents=20, sigmas=(0.02, 0.1, 0.3)):
    order = np.argsort(fitness)[::-1]
    elite_idx = order[:elite]
    parent_idx = order[:parents]
    n = pop.n
    new = Population.__new__(Population)
    new.n, new.sizes = n, list(pop.sizes)
    new.weights = [np.empty_like(w) for w in pop.weights]
    new.biases = [np.empty_like(b) for b in pop.biases]

    src = np.concatenate([elite_idx, rng.choice(parent_idx, size=n - elite)])
    sigma = np.concatenate([np.zeros(elite), rng.choice(sigmas, size=n - elite)])
    for l, (w, b) in enumerate(zip(pop.weights, pop.biases)):
        fan_in = w.shape[2]
        noise_w = rng.standard_normal((n,) + w.shape[1:]).astype(w.dtype)
        noise_b = rng.standard_normal((n,) + b.shape[1:]).astype(b.dtype)
        s = sigma.astype(w.dtype)
        new.weights[l] = w[src] + noise_w * (s / np.sqrt(fan_in))[:, None, None]
        new.biases[l] = b[src] + noise_b * (s * 0.1)[:, None]
    return new, src, sigma

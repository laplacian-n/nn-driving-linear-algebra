"""A whole population of small feed-forward tanh networks, evaluated together.

Layer l of the population is a stack of n weight matrices, shape (n, out, in),
so one batched matmul evaluates every individual at once:
    a_{l+1} = tanh(W_l a_l + b_l)        for all n individuals simultaneously
A network with no hidden layer (D = 0) is a single matrix: the "Linear" driver.
"""
import json

import numpy as np

OUTPUT_DIM = 4  # steering target, throttle, brake, handbrake


def layer_sizes(input_dim, depth, width, output_dim=OUTPUT_DIM):
    return [input_dim] + [width] * depth + [output_dim]


def param_count(sizes):
    return int(sum((i + 1) * o for i, o in zip(sizes[:-1], sizes[1:])))


class Population:
    def __init__(self, n, sizes, rng=None, dtype=np.float32):
        rng = np.random.default_rng() if rng is None else rng
        self.n = n
        self.sizes = list(sizes)
        self.weights, self.biases = [], []
        for fan_in, out in zip(self.sizes[:-1], self.sizes[1:]):
            self.weights.append(rng.normal(0, 1 / np.sqrt(fan_in), (n, out, fan_in)).astype(dtype))
            self.biases.append(rng.normal(0, 0.1, (n, out)).astype(dtype))

    # ------------------------------------------------------------- compute
    def forward(self, x, return_activations=False):
        a = x.astype(self.weights[0].dtype)
        acts = [a]
        for w, b in zip(self.weights, self.biases):
            a = np.tanh(np.matmul(w, a[:, :, None])[:, :, 0] + b)
            acts.append(a)
        return (a, acts) if return_activations else a

    @staticmethod
    def controls(y):
        """Network outputs in [-1, 1] -> (steer target, throttle, brake, handbrake).
        Brake and handbrake use max(0, y) so a near-zero network does not
        drive with the brakes on (the dead half of the range is deliberate)."""
        return y[:, 0], (y[:, 1] + 1) / 2, np.maximum(y[:, 2], 0), np.maximum(y[:, 3], 0)

    # --------------------------------------------------------- individuals
    def get(self, i):
        return [w[i].copy() for w in self.weights], [b[i].copy() for b in self.biases]

    def set(self, i, weights, biases):
        for l, (w, b) in enumerate(zip(weights, biases)):
            self.weights[l][i] = w
            self.biases[l][i] = b

    def subset(self, idx):
        out = Population.__new__(Population)
        out.n = len(idx)
        out.sizes = list(self.sizes)
        out.weights = [w[idx].copy() for w in self.weights]
        out.biases = [b[idx].copy() for b in self.biases]
        return out

    @classmethod
    def from_individuals(cls, individuals, sizes):
        pop = cls.__new__(cls)
        pop.n = len(individuals)
        pop.sizes = list(sizes)
        pop.weights = [np.stack([ind[0][l] for ind in individuals]) for l in range(len(sizes) - 1)]
        pop.biases = [np.stack([ind[1][l] for ind in individuals]) for l in range(len(sizes) - 1)]
        return pop

    @property
    def num_params(self):
        return param_count(self.sizes)

    # ------------------------------------------------------------- storage
    def save(self, path, meta=None):
        arrays = {"sizes": np.array(self.sizes)}
        for l, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{l}"] = w
            arrays[f"b{l}"] = b
        arrays["meta"] = np.array(json.dumps(meta or {}))
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        import os
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        sizes = d["sizes"].tolist()
        pop = cls.__new__(cls)
        pop.sizes = sizes
        pop.weights = [d[f"w{l}"] for l in range(len(sizes) - 1)]
        pop.biases = [d[f"b{l}"] for l in range(len(sizes) - 1)]
        pop.n = pop.weights[0].shape[0]
        meta = json.loads(str(d["meta"])) if "meta" in d else {}
        return pop, meta

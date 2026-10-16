import os

import numpy as np


class Population:
    """N individuals, each its own small feedforward net, evaluated in one batched matmul.

    Per layer, weights has shape (n, out_dim, in_dim): a stack of n independent
    weight matrices. forward() runs W_l @ a_l for every individual simultaneously
    via np.matmul's batching over the leading axis, instead of a Python loop.
    """

    def __init__(self, n, layer_sizes, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.n = n
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier-ish scale: keeps pre-activations near zero at init so tanh
            # doesn't saturate immediately (saturated init -> every random network
            # starts out steering hard-left/hard-right with no graded response,
            # which is exactly the "spins in a circle" failure mode).
            scale = 1.0 / np.sqrt(in_dim)
            self.weights.append(rng.normal(0, scale, size=(n, out_dim, in_dim)))
            self.biases.append(rng.normal(0, 0.1, size=(n, out_dim, 1)))

    def forward(self, x):
        # x: (n, input_dim) -> column vectors (n, input_dim, 1)
        a = x[:, :, None]
        for w, b in zip(self.weights, self.biases):
            a = np.tanh(np.matmul(w, a) + b)
        return a[:, :, 0]

    def get_individual(self, i):
        return [w[i].copy() for w in self.weights], [b[i].copy() for b in self.biases]

    def set_individual(self, i, weights, biases):
        for layer in range(len(self.weights)):
            self.weights[layer][i] = weights[layer]
            self.biases[layer][i] = biases[layer]

    def save(self, path):
        arrays = {"layer_sizes": np.array(self.layer_sizes)}
        for i, w in enumerate(self.weights):
            arrays[f"w{i}"] = w
        for i, b in enumerate(self.biases):
            arrays[f"b{i}"] = b
        # Write to a temp file and rename over the real path, so a save that
        # gets interrupted (window closed, Ctrl+C) can't leave a half-written,
        # unreadable checkpoint behind.
        tmp_path = str(path) + ".tmp.npz"
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path, rng=None):
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()
        n = int(data["w0"].shape[0])
        pop = cls(n, layer_sizes, rng=rng)
        for layer in range(len(pop.weights)):
            pop.weights[layer] = data[f"w{layer}"]
            pop.biases[layer] = data[f"b{layer}"]
        return pop

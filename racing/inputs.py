"""Turning raw sensor ranges + car state into the network's input vector.

    ranges (n, S) --log encode--> e --standardise--> z --project (optional)--> z P
    input = [ z P | vx/100, vy/20, r/2, delta/max_steer ]          shape (n, d)

S rays are a nested subset of the 17-ray fan (3 c 5 c 9 c 17), so every
configuration shares one calibration computed from the 17-ray dataset.
"""
import json

import numpy as np

FULL_RAYS = 17
SENSOR_RANGE = 200.0
STATE_DIM = 4


def ray_subset(num_rays):
    """Indices of an S-ray fan inside the 17-ray fan (S = 2^k + 1)."""
    if num_rays == 1:
        return np.array([FULL_RAYS // 2])
    idx = np.linspace(0, FULL_RAYS - 1, num_rays)
    if not np.allclose(idx, np.round(idx)):
        raise ValueError(f"{num_rays} rays is not a nested subset of {FULL_RAYS}")
    return np.round(idx).astype(int)


def encode(ranges):
    """Log encoding: a 3 m change next to a wall matters far more than a 3 m
    change 150 m away, and log1p spreads the near range out accordingly."""
    return np.log1p(ranges) / np.log1p(SENSOR_RANGE)


class Calibration:
    """Per-ray mean/std of the encoded 17-ray readings, plus the PCA basis
    (eigenvectors of their covariance matrix, sorted by eigenvalue)."""

    def __init__(self, mean, std, eigvals, eigvecs):
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        self.eigvals, self.eigvecs = np.asarray(eigvals), np.asarray(eigvecs)

    @classmethod
    def fit(cls, encoded17):
        mean = encoded17.mean(axis=0)
        std = encoded17.std(axis=0) + 1e-6
        z = (encoded17 - mean) / std
        cov = z.T @ z / (len(z) - 1)                  # covariance matrix (17 x 17)
        vals, vecs = np.linalg.eigh(cov)              # symmetric -> real eigen-decomposition
        order = np.argsort(vals)[::-1]
        return cls(mean, std, vals[order], vecs[:, order])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(),
                       "eigvals": self.eigvals.tolist(), "eigvecs": self.eigvecs.tolist()}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["std"], d["eigvals"], d["eigvecs"])


class InputSpec:
    """reduction: None (raw rays), 'pca' or 'random' (orthonormal random
    projection, seeded) with k output dimensions; rays must be 17 for those."""

    def __init__(self, num_rays, calibration, reduction=None, k=None, seed=0, max_steer=0.35):
        self.num_rays = num_rays
        self.reduction = reduction
        self.k = k
        self.max_steer = max_steer
        self.rays = ray_subset(num_rays)
        self.mean = calibration.mean[self.rays]
        self.std = calibration.std[self.rays]
        if reduction is None:
            self.proj = None
        else:
            if num_rays != FULL_RAYS:
                raise ValueError("reductions are applied to the full 17-ray fan")
            if reduction == "pca":
                self.proj = calibration.eigvecs[:, :k]
            elif reduction == "random":
                q, _ = np.linalg.qr(np.random.default_rng(10_000 + seed).standard_normal((FULL_RAYS, k)))
                self.proj = q
            else:
                raise ValueError(reduction)

    @property
    def sensor_dim(self):
        return self.num_rays if self.proj is None else self.proj.shape[1]

    @property
    def input_dim(self):
        return self.sensor_dim + STATE_DIM

    @property
    def preprocessing_params(self):
        """Numbers needed at run time besides the network itself."""
        n = 2 * self.num_rays
        return n + (0 if self.proj is None else self.proj.size)

    def build(self, ranges, cars):
        z = (encode(ranges) - self.mean) / self.std
        if self.proj is not None:
            z = z @ self.proj
        state = np.stack([cars.vx / 100.0, cars.vy / 20.0, cars.r / 2.0, cars.delta / self.max_steer], axis=1)
        return np.concatenate([z, state], axis=1)

    def describe(self):
        if self.proj is None:
            return f"raw{self.num_rays}"
        return f"{self.reduction}{self.k}"

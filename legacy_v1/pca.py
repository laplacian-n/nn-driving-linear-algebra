import numpy as np


class PCA:
    """Reduce raw sensor readings to their top-k principal components:
    eigenvectors of the covariance matrix, kept in order of decreasing eigenvalue."""

    def __init__(self, num_components):
        self.k = num_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, x):
        self.mean = x.mean(axis=0)
        centered = x - self.mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        self.components = eigvecs[:, order][:, :self.k]
        self.explained_variance = eigvals[order][:self.k]
        return self

    def transform(self, x):
        return (x - self.mean) @ self.components

    def fit_transform(self, x):
        return self.fit(x).transform(x)

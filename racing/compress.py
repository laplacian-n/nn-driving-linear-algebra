"""Post-training compression of a single network (EXPERIMENT_PLAN 6.4).

A network is a pair (weights, biases): weights[l] has shape (out, in).
Hidden layer l is the output of weights[l] (l < last).

Pruning methods (cut every hidden layer to the same new width):
  importance  keep the nodes with the largest ||W_next[:, j]|| * std(a_j);
              fold the mean of each removed node into the next bias
  ls          same selection, then re-fit the next layer by least squares so
              the next layer's pre-activations change as little as possible
              (normal equations with a small ridge term)
  random      random selection with the same folding (control group)
Quantisation: symmetric uniform, one scale per output row, clipping ratio
chosen by minimum squared error; biases kept at 16 bits.
Low rank:     truncated SVD of each weight matrix.
"""
import numpy as np

BIAS_BITS = 16
SCALE_BITS = 32
FLOAT_BITS = 32


def copy_net(net):
    w, b = net
    return [x.astype(np.float64).copy() for x in w], [x.astype(np.float64).copy() for x in b]


def forward_all(net, X):
    """Activations of every layer for a batch of inputs X (N, d)."""
    w, b = net
    acts = [X]
    a = X
    for wl, bl in zip(w, b):
        a = np.tanh(a @ wl.T + bl)
        acts.append(a)
    return acts


def param_count(net):
    w, b = net
    return int(sum(x.size for x in w) + sum(x.size for x in b))


def size_bits(net, weight_bits=FLOAT_BITS, bias_bits=None, scales=0, preprocessing=0):
    """Storage size in bits. weight_bits applies to every weight; scales is the
    number of stored quantisation scale factors; preprocessing = number of
    float32 input-pipeline parameters (standardisation + projection)."""
    w, b = net
    bias_bits = weight_bits if bias_bits is None else bias_bits
    return (sum(x.size for x in w) * weight_bits + sum(x.size for x in b) * bias_bits
            + scales * SCALE_BITS + preprocessing * FLOAT_BITS)


# ------------------------------------------------------------------ pruning
def prune(net, new_width, X, method="importance", rng=None, ridge=1e-3):
    """Cut every hidden layer to new_width nodes. X: calibration inputs (N, d)
    recorded while the original network drove on the calibration tracks."""
    w, b = copy_net(net)
    rng = np.random.default_rng() if rng is None else rng
    n_hidden = len(w) - 1
    for l in range(n_hidden):
        width = w[l].shape[0]
        if new_width >= width:
            continue
        acts = forward_all((w, b), X)
        A = acts[l + 1]                                   # (N, width) hidden layer l
        if method == "random":
            keep = np.sort(rng.choice(width, new_width, replace=False))
        else:
            importance = np.linalg.norm(w[l + 1], axis=0) * A.std(axis=0)
            keep = np.sort(np.argsort(importance)[::-1][:new_width])
        drop = np.setdiff1d(np.arange(width), keep)
        if method == "ls":
            # target: next layer's pre-activations with the full layer
            Z = A @ w[l + 1].T + b[l + 1]                 # (N, out)
            Ak = np.hstack([A[:, keep], np.ones((len(A), 1))])
            G = Ak.T @ Ak + ridge * len(A) * np.eye(Ak.shape[1])
            sol = np.linalg.solve(G, Ak.T @ Z)            # (k+1, out)  normal equations
            w_next, b_next = sol[:-1].T, sol[-1]
        else:
            b_next = b[l + 1] + w[l + 1][:, drop] @ A[:, drop].mean(axis=0)
            w_next = w[l + 1][:, keep]
        w[l], b[l] = w[l][keep], b[l][keep]
        w[l + 1], b[l + 1] = w_next, b_next
    return w, b


# ------------------------------------------------------------- quantisation
def _quantise_rows(W, bits, clips=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    levels = 2 ** (bits - 1) - 1
    out = np.empty_like(W)
    for i, row in enumerate(W):
        m = np.abs(row).max()
        if m == 0:
            out[i] = 0
            continue
        best, best_err = None, np.inf
        for c in clips:
            s = c * m / levels
            q = np.clip(np.round(row / s), -levels, levels) * s
            err = np.sum((q - row) ** 2)
            if err < best_err:
                best, best_err = q, err
        out[i] = best
    return out


def quantise(net, bits):
    w, b = copy_net(net)
    w = [_quantise_rows(x, bits) for x in w]
    b = [_quantise_rows(x[None, :], BIAS_BITS)[0] for x in b]
    scales = sum(x.shape[0] for x in w) + len(b)
    return (w, b), scales


def quantised_size(net, bits, preprocessing=0):
    w, _ = net
    scales = sum(x.shape[0] for x in w) + len(net[1])
    return size_bits(net, weight_bits=bits, bias_bits=BIAS_BITS, scales=scales, preprocessing=preprocessing)


# ---------------------------------------------------------------- low rank
def low_rank(net, rank):
    """Replace each W (m x n) by U_r S_r V_r^T; stored as r(m + n) numbers."""
    w, b = copy_net(net)
    stored = 0
    for l, W in enumerate(w):
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        r = min(rank, len(S))
        if r * (W.shape[0] + W.shape[1]) < W.size:
            w[l] = (U[:, :r] * S[:r]) @ Vt[:r]
            stored += r * (W.shape[0] + W.shape[1])
        else:
            stored += W.size
    stored += sum(x.size for x in b)
    return (w, b), stored * FLOAT_BITS


# ---------------------------------------------------------------- analysis
def effective_rank(W, energy=0.9):
    s = np.linalg.svd(W, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    k90 = int(np.searchsorted(e, energy) + 1)
    p = s / s.sum()
    entropy_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return {"rank90": k90, "entropy_rank": entropy_rank, "max_rank": len(s), "singular_values": s.tolist()}


def open_loop_error(net_a, net_b, X):
    """RMS difference of the 4 outputs on the same inputs."""
    ya = forward_all(net_a, X)[-1]
    yb = forward_all(net_b, X)[-1]
    return float(np.sqrt(np.mean((ya - yb) ** 2)))

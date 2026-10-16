"""Statistics used in the report (EXPERIMENT_PLAN 5.2).

All functions take per-seed x per-track score matrices: scores[s, t].
"""
import numpy as np

B = 10_000


def hier_bootstrap_mean(scores, rng=None, b=B):
    """Mean and 95% CI by resampling seeds, then tracks within each seed."""
    rng = np.random.default_rng(0) if rng is None else rng
    s, t = scores.shape
    si = rng.integers(0, s, size=(b, s))
    ti = rng.integers(0, t, size=(b, s, t))
    boot = scores[si[:, :, None], ti].mean(axis=(1, 2))
    return float(scores.mean()), np.percentile(boot, [2.5, 97.5]), boot


def paired_diff_bootstrap(a, b_, rng=None, b=B):
    """Difference of means (a - b) between two configs evaluated on the same
    tracks; seeds are independent between configs, tracks are paired."""
    rng = np.random.default_rng(1) if rng is None else rng
    sa, t = a.shape
    sb, _ = b_.shape
    ia = rng.integers(0, sa, size=(b, sa))
    ib = rng.integers(0, sb, size=(b, sb))
    ti = rng.integers(0, t, size=(b, t))
    ma = a[ia][:, :, :].mean(axis=1)                 # (b, t)
    mb = b_[ib].mean(axis=1)
    diff = (np.take_along_axis(ma, ti, 1) - np.take_along_axis(mb, ti, 1)).mean(axis=1)
    return float(a.mean() - b_.mean()), np.percentile(diff, [2.5, 97.5]), diff


def tost_equivalent(diff_boot, delta, alpha=0.05):
    """Two one-sided tests via the bootstrap distribution: equivalent if the
    (1 - 2 alpha) interval of the difference lies inside (-delta, +delta)."""
    lo, hi = np.percentile(diff_boot, [100 * alpha, 100 * (1 - alpha)])
    return bool(lo > -delta and hi < delta), (float(lo), float(hi))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_p(diff_boot):
    """Two-sided p-value for 'difference = 0' from a bootstrap distribution."""
    d = np.asarray(diff_boot)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(p, 1 / len(d)))


def kaplan_meier(fracs, finished):
    """Survival curve of 'distance before crashing' (lap fraction), censored
    at the finish line. Returns x (lap fraction) and S(x)."""
    x = np.asarray(fracs, float)
    event = ~np.asarray(finished, bool)
    order = np.argsort(x)
    x, event = x[order], event[order]
    n = len(x)
    surv, xs, ss = 1.0, [0.0], [1.0]
    at_risk = n
    for xi, ev in zip(x, event):
        if ev:
            surv *= (at_risk - 1) / at_risk
            xs.append(xi)
            ss.append(surv)
        at_risk -= 1
    xs.append(1.0)
    ss.append(surv)
    return np.array(xs), np.array(ss)

"""Figures for the data section of the report (data/dataset_summary.json,
data/dataset_arrays.npz).   python analysis/plot_dataset.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from analysis import style as S
from racing.inputs import encode

ROOT = S.ROOT
S.apply()
summ = json.load(open(os.path.join(ROOT, "data", "dataset_summary.json")))
D = np.load(os.path.join(ROOT, "data", "dataset_arrays.npz"))
raw, clean, keep = D["rays_raw"], D["rays_clean"], D["keep"]

# 1. row quality: one horizontal 100% bar, three parts, labelled
q = summ["row_quality"]
parts = [("Complete", q["complete"]), ("Fixable (repaired)", q["fixable (repaired)"]),
         ("Incomplete (dropped)", q["incomplete (dropped)"])]
fig, ax = plt.subplots(figsize=(7.5, 2.0))
left = 0
for (name, v), col in zip(parts, [S.SERIES[0], S.SERIES[3], S.SERIES[7]]):
    ax.barh(0, v["percent"], left=left, color=col, height=0.5, edgecolor=S.SURFACE, linewidth=2,
            label=f"{name}: {v['count']:,} rows ({v['percent']:.1f}%)")
    left += v["percent"]
ax.set_xlim(0, 100)
ax.set_ylim(-0.45, 0.45)
ax.set_yticks([])
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("% of all rows")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, handlelength=1.2)
ax.set_title(f"Row quality of the driving dataset ({summ['rows_total']:,} vectors)")
S.save(fig, "data_row_quality.png")

# 2. drop reasons
reasons = {k: v for k, v in summ["drop_reasons"].items()}
fig, ax = plt.subplots(figsize=(6.5, 2.6))
names = list(reasons)[::-1]
vals = [reasons[k] for k in names]
ax.barh(names, vals, color=S.SERIES[0], height=0.55)
for i, v in enumerate(vals):
    ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=9)
ax.set_xlabel("rows")
ax.set_title("Why rows were dropped")
ax.grid(axis="y", visible=False)
S.save(fig, "data_drop_reasons.png")

# 3. histogram of ray readings before / after cleaning (small multiples)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
bins = np.linspace(0, 200, 51)
for ax, data, title in [(axes[0], raw.ravel(), "Before cleaning (all rows)"),
                        (axes[1], clean[keep].ravel(), "After cleaning (kept rows)")]:
    ax.hist(data, bins=bins, color=S.SERIES[0], edgecolor=S.SURFACE, linewidth=0.6)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("ray range (m)")
axes[0].set_ylabel("readings (log scale)")
n0 = int((raw == 0).sum())
axes[0].annotate(f"dropouts: {n0:,} readings at 0 m", (2, n0), xytext=(40, n0 * 3), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=S.TEXT_2))
nmax = int((raw >= 199.8).sum())
axes[0].annotate(f"no return: {nmax:,} at max range", (198, nmax), xytext=(80, nmax * 4), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=S.TEXT_2))
S.save(fig, "data_ray_histogram_before_after.png")

# 4. data-type shares (segment type, speed band, action)
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
for ax, key, title in [(axes[0], "segment_type", "Road segment"), (axes[1], "speed_band_kmh", "Speed band (km/h)"),
                       (axes[2], "action", "Controller action")]:
    d = summ[key]
    order = ["<80", "80-160", "160-240", ">=240"] if key == "speed_band_kmh" else sorted(d, key=lambda k: -d[k]["percent"])
    order = [k for k in order if k in d]
    vals = [d[k]["percent"] for k in order]
    ax.barh(order[::-1], vals[::-1], color=S.SERIES[0], height=0.55)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of kept rows")
S.save(fig, "data_shares.png")

# 5. scree before / after cleaning
ev_b = np.array(summ["pca_eigenvalues_before"])
ev_a = np.array(summ["pca_eigenvalues_after"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
k = np.arange(1, 18)
ax = axes[0]
ax.plot(k, ev_b, "-o", color=S.SERIES[1], label="before cleaning")
ax.plot(k, ev_a, "-o", color=S.SERIES[0], label="after cleaning")
S.label_line_end(ax, 1, ev_a[0], f"  λ₁ = {ev_a[0]:.2f}", S.SERIES[0])
S.label_line_end(ax, 1, ev_b[0], f"  λ₁ = {ev_b[0]:.2f}", S.SERIES[1])
ax.set_xlabel("component")
ax.set_ylabel("eigenvalue λ")
ax.set_title("Eigenvalues of the covariance matrix")
ax.legend()
ax = axes[1]
for ev, col, lab in [(ev_b, S.SERIES[1], "before"), (ev_a, S.SERIES[0], "after")]:
    c = np.cumsum(ev) / ev.sum() * 100
    ax.plot(k, c, "-o", color=col, label=lab)
c = np.cumsum(ev_a) / ev_a.sum() * 100
for kk in (2, 4, 8):
    ax.plot([kk], [c[kk - 1]], "o", ms=10, mfc="none", mec=S.TEXT, mew=1.2)
    ax.annotate(f"k={kk}: {c[kk - 1]:.0f}%", (kk, c[kk - 1]), xytext=(9, -6), textcoords="offset points",
                fontsize=9, ha="left", va="top")
for a_ in axes:
    a_.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17])
ax.set_ylim(0, 102)
ax.set_xlabel("number of components k")
ax.set_ylabel("variance explained (%)")
ax.set_title("Cumulative variance explained")
ax.legend(loc="lower right")
S.save(fig, "data_pca_scree.png")

# 6. eigen-sensors: first four eigenvectors drawn on the 17-ray fan
calib = json.load(open(os.path.join(ROOT, "data", "calibration.json")))
V = np.array(calib["eigvecs"])
ang = np.radians(np.linspace(-100, 100, 17))
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), subplot_kw={"aspect": "equal"})
norm = plt.Normalize(-0.45, 0.45)
for i, ax in enumerate(axes):
    v = V[:, i]
    for a, val in zip(ang, v):
        # car faces up the page: angle 0 = straight ahead
        x, y = -np.sin(a), np.cos(a)
        ax.plot([0, x], [0, y], color=S.DIVERGING(norm(val)), lw=2 + 10 * abs(val), solid_capstyle="round")
    ax.add_patch(plt.Rectangle((-0.06, -0.14), 0.12, 0.28, color=S.TEXT))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.4, 1.15)
    ax.axis("off")
    ax.set_title(f"PC{i + 1}  (λ = {calib['eigvals'][i]:.2f})", loc="center")
sm = plt.cm.ScalarMappable(cmap=S.DIVERGING, norm=norm)
cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.02, aspect=50)
cb.set_label("weight of each ray in the eigenvector (red = +, blue = −)")
S.save(fig, "data_eigen_sensors.png")
print("ok")

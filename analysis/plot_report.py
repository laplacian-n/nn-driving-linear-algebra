"""Figures made for the report (Thai labels).

    python analysis/plot_report.py
      -> figures/rep_method_size.png   GA vs learning from a teacher, 4 model sizes
      -> figures/rep_showcase.png      show models on the extreme show tracks
      -> figures/rep_geometry.png      ray / wall intersection (Cramer) and projections on the road frame
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

RES = os.path.join(style.ROOT, "experiments", "results")
SIZES = [("S9_XS_linear", "XS\n56 พารามิเตอร์"), ("S9_S_2x8", "S 2×8\n220"), ("S9_L_2x32", "L 2×32\n1,636"),
         ("S9_XL_2x64", "XL 2×64\n5,316")]


def thai():
    from matplotlib import font_manager
    for f in glob.glob(os.path.expanduser("~/.local/share/fonts/Sarabun-*.ttf")):
        font_manager.fontManager.addfont(f)
    style.apply()
    plt.rcParams["font.family"] = ["Sarabun", "Noto Sans Thai", "DejaVu Sans"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12.5


def finish_rates(stage, size):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, stage, size, "seed*", "final_test.json"))):
        out.append(100 * float(np.mean(json.load(open(f))["finished"])))
    return out


def method_size():
    thai()
    methods = [("stage3", "Genetic Algorithm (เรียนเองจากคะแนน)"),
               ("bc_long", "เรียนจากครู: ครูใช้ยาง 85%"),
               ("bc_t75_long", "เรียนจากครู: ครูใช้ยาง 75%")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(SIZES))
    bw = 0.26
    for k, (stage, label) in enumerate(methods):
        vals = [finish_rates(stage, s) for s, _ in SIZES]
        med = [np.median(v) for v in vals]
        xs = x + (k - 1) * (bw + 0.02)
        ax.bar(xs, med, bw, color=style.SERIES[k], label=label, zorder=2)
        for xi, v, m in zip(xs, vals, med):
            ax.scatter(np.full(len(v), xi), v, s=12, color=style.TEXT, zorder=3, linewidths=0)
            ax.text(xi, 2, f"{m:.0f}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold", zorder=4)
    ax.set_xticks(x, [lab for _, lab in SIZES])
    ax.set_ylim(0, 105)
    ax.set_ylabel("จบรอบบนชุดสอบ T (%)")
    ax.set_title("ขนาดโมเดลที่ดีที่สุดขึ้นกับวิธีฝึก (ค่ามัธยฐาน 4 seed, จุด = แต่ละ seed)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.grid(axis="x", visible=False)
    print(style.save(fig, "rep_method_size.png"))


def showcase():
    thai()
    stages = [("bc_base", "ก่อนฝึกต่อ"), ("bc_special", "+ ครูบนสนามพิเศษ"), ("bc_ga", "+ Genetic Algorithm")]
    variants = [("showcase", "รังสี 9 เส้น"), ("showcase_pca8", "รังสี 17 เส้น + PCA 8 มิติ")]
    data = {}
    for v, _ in variants:
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, v, "seed*", "final.json")))]
        data[v] = runs
    teacher = float(np.mean(data["showcase"][0]["teacher_show_v_ref"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(stages))
    bw = 0.36
    for k, (v, label) in enumerate(variants):
        fin = [np.mean([100 * r[s]["show"]["finish"] for r in data[v]]) for s, _ in stages]
        spd = [np.mean([r[s]["show"]["lap_speed"] for r in data[v]]) for s, _ in stages]
        xs = x + (k - 0.5) * (bw + 0.03)
        axes[0].bar(xs, fin, bw, color=style.SERIES[k], label=label, zorder=2)
        axes[1].bar(xs, spd, bw, color=style.SERIES[k], label=label, zorder=2)
        for xi, a, b in zip(xs, fin, spd):
            axes[0].text(xi, a + 1.5, f"{a:.0f}", ha="center", fontsize=10)
            axes[1].text(xi, b + 0.08, f"{b:.1f}", ha="center", fontsize=10)
    axes[1].axhline(teacher, color=style.TEXT, lw=1.2, ls="--")
    axes[1].text(-0.62, teacher + 0.1, f"ครู (ตัวควบคุมอ้างอิง) {teacher:.1f}", ha="left", fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(19.5, 25)
    axes[0].set_title("จบรอบสนามโชว์ระดับ 3.0 (%)")
    axes[1].set_title("ความเร็วเฉลี่ยของรอบที่จบ (ม./วิ.)")
    for ax in axes:
        ax.set_xticks(x, [lab for _, lab in stages])
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    print(style.save(fig, "rep_showcase.png"))


def arrow(ax, a, b, color, lw=2, label=None, off=(0, 0), fs=12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, zorder=4))
    if label:
        m = (np.array(a) + np.array(b)) / 2
        ax.text(m[0] + off[0], m[1] + off[1], label, color=color, fontsize=fs, ha="center", va="center")


def geometry():
    thai()
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) ray p + t d meets wall a + u (b - a)
    ax = axes[0]
    p, d = np.array([0.0, 0.0]), np.array([np.cos(0.55), np.sin(0.55)])
    a, b = np.array([3.2, -0.6]), np.array([2.6, 3.6])
    A = np.column_stack([d, -(b - a)])
    t, u = np.linalg.solve(A, a - p)
    hit = p + t * d
    ax.plot(*np.column_stack([a, b]), color=style.TEXT_2, lw=4, solid_capstyle="round")
    ax.text(*(a + [0.25, -0.05]), r"$\mathbf{a}$", fontsize=14)
    ax.text(*(b + [0.15, 0]), r"$\mathbf{b}$", fontsize=14)
    ax.plot(*np.column_stack([p, p + 4.6 * d]), color=style.SERIES[0], lw=1, ls=":")
    arrow(ax, p, p + 1.3 * d, style.SERIES[0], label=r"$\mathbf{d}$", off=(-0.15, 0.35), fs=14)
    ax.plot(*np.column_stack([p, hit]), color=style.SERIES[0], lw=2, alpha=0.35)
    ax.scatter(*hit, s=60, color=style.SERIES[1], zorder=5)
    ax.annotate(r"$\mathbf{p}+t\,\mathbf{d}=\mathbf{a}+u(\mathbf{b}-\mathbf{a})$", hit, xytext=(-140, 45),
                textcoords="offset points", fontsize=12, arrowprops=dict(arrowstyle="-", color=style.TEXT_2, lw=0.8))
    ax.add_patch(Polygon([[-0.35, -0.2], [0.35, -0.2], [0.35, 0.2], [-0.35, 0.2]], closed=True,
                         color=style.SERIES[2], zorder=3))
    ax.text(-0.1, -0.55, r"$\mathbf{p}$", fontsize=14)
    ax.text(-0.5, 3.6, rf"$t={t:.2f},\;u={u:.2f}$" + "\n" + r"$0\leq u\leq 1,\;t>0\Rightarrow$ ชนกำแพง",
            fontsize=11, va="top")
    ax.set_title("(ก) จุดตัดของรังสีกับขอบถนน")

    # (b) projection on the road frame
    ax = axes[1]
    c = np.array([0.0, 0.0])
    tvec = np.array([np.cos(0.25), np.sin(0.25)])
    nvec = np.array([-tvec[1], tvec[0]])
    w = 2.4
    for side in (1, -1):
        e0, e1 = c - 2.2 * tvec + side * w * nvec, c + 3.8 * tvec + side * w * nvec
        ax.plot(*np.column_stack([e0, e1]), color=style.TEXT_2, lw=3)
    ax.plot(*np.column_stack([c - 2.2 * tvec, c + 3.8 * tvec]), color=style.TEXT_2, lw=1, ls="--")
    p = c + 2.3 * tvec + 1.15 * nvec
    s, e = (p - c) @ tvec, (p - c) @ nvec
    arrow(ax, c, c + 1.0 * tvec, style.SERIES[0], label=r"$\mathbf{t}_j$", off=(0.1, -0.35), fs=14)
    arrow(ax, c, c + 1.0 * nvec, style.SERIES[0], label=r"$\mathbf{n}_j$", off=(-0.35, 0.1), fs=14)
    arrow(ax, c, p, style.SERIES[1], label=r"$\mathbf{p}-\mathbf{c}_j$", off=(-0.55, 0.35), fs=13)
    ax.plot(*np.column_stack([c + s * tvec, p]), color=style.TEXT_2, lw=1, ls=":")
    ax.text(*(c + s * tvec + [0.15, -0.45]), rf"$s=(\mathbf{{p}}-\mathbf{{c}}_j)\cdot\mathbf{{t}}_j$", fontsize=12)
    ax.text(*(p + [0.35, -0.1]), rf"$e=(\mathbf{{p}}-\mathbf{{c}}_j)\cdot\mathbf{{n}}_j$", fontsize=12)
    ax.scatter(*c, s=40, color=style.TEXT, zorder=5)
    ax.text(*(c + [-0.2, -0.45]), r"$\mathbf{c}_j$", fontsize=14)
    ax.text(*(c + 3.9 * tvec + w * nvec + [0, 0.2]), r"$+w/2$", fontsize=11)
    ax.text(*(c + 3.9 * tvec - w * nvec + [0, -0.35]), r"$-w/2$", fontsize=11)
    ax.set_title("(ข) ระยะตามถนน s และระยะเยื้อง e จากการฉาย")
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    print(style.save(fig, "rep_geometry.png"))


def sensors():
    thai()
    groups = [("รังสีจริง (ลดจำนวนเส้น)", [("raw3", 3), ("raw5", 5), ("raw9", 9), ("raw17", 17)], style.SERIES[0]),
              ("PCA จาก 17 รังสี", [("pca2", 2), ("pca4", 4), ("pca8", 8), ("pca17", 17)], style.SERIES[1]),
              ("ฉายแบบสุ่มจาก 17 รังสี", [("rand4", 4), ("rand8", 8)], style.TEXT_2)]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for label, conds, color in groups:
        xs, ys = [], []
        for c, k in conds:
            v = finish_rates("sensors", c)
            xs.append(k)
            ys.append(np.median(v))
        ax.plot(xs, ys, "o-", color=color, label=label, lw=2.2, ms=7, ls="--" if color == style.TEXT_2 else "-")
    ax.annotate("PCA 4 มิติ = 93%\nเกือบเท่ารังสีจริง 17 เส้น", (4, 93), xytext=(10, 62), fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=style.TEXT_2, lw=0.8))
    ax.annotate("มิติเท่ากัน แต่ทิศสุ่มได้ 50%", (4, 50), xytext=(5.6, 18), fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=style.TEXT_2, lw=0.8))
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 5, 8, 9, 17], ["2", "3", "4", "5", "8", "9", "17"])
    ax.set_xlabel("จำนวนมิติของข้อมูลเซนเซอร์ที่ป้อนเข้าโมเดล")
    ax.set_ylabel("จบรอบบนชุดสอบ T (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("ทิศของเวกเตอร์ไอเกนเก็บข้อมูลที่ใช้ขับได้เกือบทั้งหมด (L 2×32, เรียนจากครู)")
    ax.legend(loc="lower right")
    print(style.save(fig, "rep_sensors.png"))


def compression():
    thai()
    by = {}
    for f in sorted(glob.glob(os.path.join(RES, "stage4_bc", "S9_XL_2x64", "seed*.json"))):
        for v in json.load(open(f))["variants"]:
            by.setdefault(v["name"], []).append((v["method"], v["size_bits"] / 1000, 100 * np.mean(v["finished"])))
    med = {k: (v[0][0], v[0][1], float(np.median([x[2] for x in v]))) for k, v in by.items()}
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    lines = [("quantise", "ลดจำนวนบิต (D·Q)", style.SERIES[1]), ("prune_ls", "ตัดโหนด + กำลังสองน้อยสุด", style.SERIES[0]),
             ("low_rank", "low-rank (SVD)", style.SERIES[4])]
    for m, label, color in lines:
        pts = sorted((x[1], x[2]) for x in med.values() if x[0] == m)
        ax.plot(*zip(*pts), "o-", color=color, label=label, lw=2.2, ms=6)
    o = med["original"]
    ax.scatter([o[1]], [o[2]], marker="D", s=70, color=style.TEXT, zorder=5)
    ax.annotate("XL 32 bit", (o[1], o[2]), xytext=(-10, 8), textcoords="offset points", ha="right", fontsize=10)
    for size, kb in [("S9_S_2x8", 7.0), ("S9_L_2x32", 52.4)]:
        v = np.median(finish_rates("bc_t75_long", size))
        ax.scatter([kb], [v], marker="*", s=200, color=style.SERIES[3], edgecolor=style.TEXT, linewidth=0.6, zorder=5)
        ax.annotate(size.split("_")[1] + " ฝึกเอง", (kb, v), xytext=(8, -14), textcoords="offset points", fontsize=10)
    q8 = med["quant8"]
    ax.annotate("XL 8 bit: 48 kbit, 92%\nเล็กกว่าและเก่งกว่า L ที่ฝึกเอง", (q8[1], q8[2]), xytext=(9, 55), fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=style.TEXT_2, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("ขนาดโมเดล (กิโลบิต, สเกลลอการิทึม)")
    ax.set_ylabel("จบรอบบนชุดสอบ T (%)")
    ax.set_ylim(-3, 103)
    ax.set_title("บีบอัด XL 2×64: ลดบิตปลอดภัย ส่วนการลดแรงก์ทำให้ระบบวงปิดล้ม")
    ax.legend(loc="center left")
    print(style.save(fig, "rep_compression.png"))


def forgetting():
    thai()
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    labels = [lab.split("\n")[0] for _, lab in SIZES]
    x = np.arange(4)
    for k, (stage, label) in enumerate([("stage3", "Genetic Algorithm"), ("bc_t75_long", "เรียนจากครู (ครู 75%)")]):
        diff = []
        for size, _ in SIZES:
            v = []
            for run in sorted(glob.glob(os.path.join(RES, stage, size, "seed*"))):
                st, ft = os.path.join(run, "station_tests.json"), os.path.join(run, "final_test.json")
                if os.path.exists(st) and os.path.exists(ft):
                    v.append(100 * np.mean(json.load(open(ft))["finished"])
                             - 100 * np.mean([np.mean(t["finished"]) for t in json.load(open(st))]))
            diff.append(np.median(v))
        xs = x + (k - 0.5) * 0.36
        color = style.SERIES[0] if k == 0 else style.SERIES[2]
        ax.bar(xs, diff, 0.34, color=color, label=label, zorder=2)
        for xi, d in zip(xs, diff):
            ax.text(xi, d + (1 if d >= 0 else -1), f"{d:+.0f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=10.5)
    ax.axhline(0, color=style.TEXT, lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylim(-20, 12)
    ax.set_ylabel("สอบหลังฝึก − สอบระหว่างฝึก (จุด %)")
    ax.set_title("การลืม: ค่าติดลบ = ลืมสิ่งที่เคยทำได้ (ค่ามัธยฐาน 4 seed)")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    print(style.save(fig, "rep_forgetting.png"))


if __name__ == "__main__":
    method_size()
    showcase()
    geometry()
    sensors()
    compression()
    forgetting()

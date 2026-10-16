"""Build and execute notebooks/linear_algebra_calculations.ipynb - the
"calculation file" deliverable: every vector / matrix step of the project
worked through with real numbers from the simulator and the dataset.

    python tools/build_notebook.py
"""
import os

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "notebooks")

cells = []


def md(s):
    cells.append(nbformat.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbformat.v4.new_code_cell(s.strip()))


md("""
# การคำนวณทฤษฎีเวกเตอร์และเมทริกซ์ในโครงงาน
**2D map self driving car neural network**

ไฟล์นี้แสดงการคำนวณทุกขั้นที่ใช้ทฤษฎีเวกเตอร์และเมทริกซ์ในโครงงาน ด้วยตัวเลขจริงจากตัวจำลองและชุดข้อมูล
ทุกเซลล์เรียกโค้ดตัวเดียวกับที่โปรแกรมใช้ (`racing/`) แล้วตรวจผลกับการคำนวณด้วยมือ

| ส่วน | ทฤษฎี |
|---|---|
| 1 | เวกเตอร์สถานะ, การบวกเวกเตอร์, dot product, การแยกองค์ประกอบ, เมทริกซ์หมุน |
| 2 | cross product และรัศมีความโค้ง |
| 3 | เส้นโค้ง Bezier ในรูปผลคูณเมทริกซ์ |
| 4 | ระบบสมการเชิงเส้น 2×2 และกฎของคราเมอร์ (raycast เซนเซอร์) |
| 5 | เมทริกซ์ข้อมูล, standardization, covariance, eigen-decomposition (PCA) |
| 6 | neural network เป็นการคูณเมทริกซ์, tensor ของประชากร |
| 7 | GA: การกลายพันธุ์เป็นการบวกเวกเตอร์ |
| 8 | การบีบอัด: least squares (สมการปกติ), quantization, SVD |
""")

code("""
import os, sys, json
ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=110)
from racing.track import generate_track, bezier_segment, BEZIER_MATRIX
from racing.physics import CarSwarm, CarParams, pacejka
from racing.inputs import Calibration, InputSpec, encode, ray_subset
from racing.network import Population, layer_sizes
""")

# 1 ------------------------------------------------------------------
md("""
## 1. เวกเตอร์ของรถ

สถานะของรถแต่ละคันคือเวกเตอร์ ตำแหน่ง $\\mathbf{p}\\in\\mathbb{R}^2$ ความเร็วในกรอบโลก $\\mathbf{v}\\in\\mathbb{R}^2$
และทิศหน้ารถ $\\psi$ เวกเตอร์หน่วยของทิศหน้ารถและทิศซ้ายคือ

$$\\hat{\\mathbf{h}} = (\\cos\\psi, \\sin\\psi),\\qquad \\hat{\\mathbf{n}} = (-\\sin\\psi, \\cos\\psi)$$

ความเร็วไปข้างหน้าและความเร็วด้านข้าง (slip) หาด้วย **dot product** (การฉายเวกเตอร์ลงแกน):

$$v_x = \\mathbf{v}\\cdot\\hat{\\mathbf{h}},\\qquad v_y = \\mathbf{v}\\cdot\\hat{\\mathbf{n}}$$
""")
code("""
psi = np.radians(30)                      # รถหันไป 30 องศา
v_world = np.array([40.0, 30.0])          # ความเร็วในกรอบโลก (m/s)
h = np.array([np.cos(psi), np.sin(psi)])  # ทิศหน้ารถ
n = np.array([-np.sin(psi), np.cos(psi)]) # ทิศซ้าย
vx, vy = v_world @ h, v_world @ n
print("h =", h, "  n =", n, "  h.n =", round(h @ n, 12), "(ตั้งฉากกัน)")
print(f"vx = v.h = {vx:.3f} m/s   vy = v.n = {vy:.3f} m/s")
print(f"|v| = {np.linalg.norm(v_world):.3f} = sqrt(vx^2+vy^2) = {np.hypot(vx, vy):.3f}")
print(f"มุมลื่นไถล beta = atan(vy/vx) = {np.degrees(np.arctan2(vy, vx)):.2f} องศา")
""")
md("""
กลับจากกรอบรถไปกรอบโลกใช้ **เมทริกซ์หมุน** $R(\\psi)$ และการเคลื่อนที่ในแต่ละก้าวคือการบวกเวกเตอร์
$\\mathbf{p}_{t+1} = \\mathbf{p}_t + \\Delta t\\,R(\\psi)\\,[v_x, v_y]^T$
""")
code("""
R = np.array([[np.cos(psi), -np.sin(psi)],
              [np.sin(psi),  np.cos(psi)]])
print("R =\\n", R)
print("R [vx, vy] =", R @ np.array([vx, vy]), "  (ได้ v_world กลับมา)")
print("R^T R =\\n", R.T @ R, "\\n det R =", round(np.linalg.det(R), 12), "-> เมทริกซ์ตั้งฉาก ไม่เปลี่ยนความยาว")
p = np.array([100.0, 50.0]); dt = 1 / 240
print("p_next = p + dt * R v_body =", p + dt * (R @ np.array([vx, vy])))
""")
md("""
ตัวอย่างจากตัวจำลองจริง: ขับรถเลี้ยวขวาเต็มที่ที่ 200 km/h แล้ววัด slip ด้วยสูตรเดียวกัน
แรงยางด้านข้างมาจากสูตร Pacejka ซึ่งขึ้นกับมุม slip ของยางที่คำนวณจากองค์ประกอบของความเร็วเช่นกัน
""")
code("""
from tools.physics_check import test_pad
car = CarSwarm(1, test_pad(), start_speed=200/3.6, stall_timeout=1e9, time_limit=1e9, sensor_noise=0)
for _ in range(20):
    car.step(np.array([-1.0]), np.array([0.6]), np.array([0.0]), np.array([0.0]))
h = np.array([np.cos(car.psi[0]), np.sin(car.psi[0])]); n = np.array([-np.sin(car.psi[0]), np.cos(car.psi[0])])
v_world = car.vx[0] * h + car.vy[0] * n
print(f"หลัง {car.time:.2f} s: vx = {car.vx[0]:.2f}, vy = {car.vy[0]:.2f} m/s, beta = {np.degrees(car.slip_angle[0]):.2f} deg")
print("ตรวจ: v_world.h =", round(v_world @ h, 6), " v_world.n =", round(v_world @ n, 6))
""")

# 2 ------------------------------------------------------------------
md("""
## 2. Cross product และความโค้งของถนน

รัศมีความโค้งที่จุด $\\mathbf{p}_1$ จากสามจุดติดกัน คือรัศมีวงกลมที่ผ่านทั้งสามจุด:

$$R = \\frac{\\|\\mathbf{p}_1-\\mathbf{p}_0\\|\\,\\|\\mathbf{p}_2-\\mathbf{p}_1\\|\\,\\|\\mathbf{p}_2-\\mathbf{p}_0\\|}{2\\,|(\\mathbf{p}_1-\\mathbf{p}_0)\\times(\\mathbf{p}_2-\\mathbf{p}_1)|}$$

โดย cross product ใน 2 มิติ $\\mathbf{a}\\times\\mathbf{b} = a_x b_y - a_y b_x$ (เท่ากับสองเท่าของพื้นที่สามเหลี่ยม)
ใช้กำหนดว่าโค้งไหนแคบเกินความกว้างถนน และใช้หาความเร็วสูงสุดในโค้ง
""")
code("""
r_true = 50.0
th = np.radians([0, 4, 8])
p0, p1, p2 = [np.array([r_true*np.cos(t), r_true*np.sin(t)]) for t in th]
a, b = p1 - p0, p2 - p1
cross = a[0]*b[1] - a[1]*b[0]
R_est = np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(p2-p0) / (2*abs(cross))
print(f"cross = {cross:.4f}  -> R = {R_est:.4f} m (วงกลมจริง {r_true} m)")
track = generate_track(seed=400003, difficulty=1.0)
print(f"สนามจริง: รัศมีแคบสุด {track['curvature_radius'].min():.1f} m, ความยาว {track['length']:.0f} m, "
      f"{len(track['centerline'])} จุด")
""")

# 3 ------------------------------------------------------------------
md("""
## 3. เส้นโค้ง Bezier เป็นผลคูณเมทริกซ์

ทุกโค้งของสนามคือ cubic Bezier ที่เขียนเป็นผลคูณเมทริกซ์ได้

$$\\mathbf{p}(t) = \\begin{bmatrix}1 & t & t^2 & t^3\\end{bmatrix}
\\underbrace{\\begin{bmatrix}1&0&0&0\\\\-3&3&0&0\\\\3&-6&3&0\\\\-1&3&-3&1\\end{bmatrix}}_{M}
\\begin{bmatrix}\\mathbf{P}_0\\\\\\mathbf{P}_1\\\\\\mathbf{P}_2\\\\\\mathbf{P}_3\\end{bmatrix}$$

ประเมินหลายค่า $t$ พร้อมกันได้ด้วยการซ้อนแถว $[1\\ t\\ t^2\\ t^3]$ เป็นเมทริกซ์ $T$: $\\;\\mathbf{P}(t) = T M P$
""")
code("""
P = np.array([[0, 0], [30, 0], [50, 20], [50, 50]], dtype=float)   # จุดควบคุมของโค้ง 90 องศา
t = 0.5
T_row = np.array([1, t, t**2, t**3])
print("M =\\n", BEZIER_MATRIX)
print("[1 t t^2 t^3] M =", T_row @ BEZIER_MATRIX, " (น้ำหนักของจุดควบคุม รวม =", (T_row @ BEZIER_MATRIX).sum(), ")")
print("p(0.5) =", T_row @ BEZIER_MATRIX @ P)
print("de Casteljau ตรวจ:", ((1-t)**3*P[0] + 3*(1-t)**2*t*P[1] + 3*(1-t)*t**2*P[2] + t**3*P[3]))
pts = bezier_segment(*P, 5)
print("หลายค่า t พร้อมกัน (T M P):\\n", pts)
""")

# 4 ------------------------------------------------------------------
md("""
## 4. เซนเซอร์: ระบบสมการเชิงเส้น 2×2 และกฎของคราเมอร์

รังสี $\\mathbf{p} + t\\mathbf{d}$ ชนกำแพง $\\mathbf{a} + u\\mathbf{e}$ เมื่อ

$$t\\,\\mathbf{d} - u\\,\\mathbf{e} = \\mathbf{a}-\\mathbf{p}\\quad\\Longleftrightarrow\\quad
\\begin{bmatrix} d_x & -e_x\\\\ d_y & -e_y\\end{bmatrix}\\begin{bmatrix}t\\\\u\\end{bmatrix} = \\begin{bmatrix}w_x\\\\w_y\\end{bmatrix}$$

กฎของคราเมอร์: $t = \\det A_t / \\det A$, $u = \\det A_u / \\det A$ จะชนจริงเมื่อ $t>0$ และ $0\\le u\\le 1$
ถ้า $\\det A = 0$ แปลว่ารังสีขนานกับกำแพง โปรแกรมแก้ระบบนี้ประมาณ 3,000 ครั้งต่อรถหนึ่งคันต่อก้าว
""")
code("""
p = np.array([0.0, 0.0]); ang = np.radians(20); d = np.array([np.cos(ang), np.sin(ang)])
a = np.array([40.0, -10.0]); b = np.array([40.0, 40.0]); e = b - a; w = a - p
A = np.column_stack([d, -e])
det = np.linalg.det(A)
At = np.column_stack([w, -e]); Au = np.column_stack([d, w])
t_hit, u_hit = np.linalg.det(At)/det, np.linalg.det(Au)/det
print("A =\\n", A, "\\ndet A =", round(det, 4))
print(f"Cramer: t = {t_hit:.4f} m, u = {u_hit:.4f}  -> ชนที่ {p + t_hit*d}")
print("np.linalg.solve:", np.linalg.solve(A, w))
""")
code("""
cars = CarSwarm(1, track, num_sensors=17, sensor_noise=0)
fast = cars.sensor_distances()[0]
slow = cars._sensor_distances_numpy()[0]
print("ระยะรังสี 17 เส้นที่จุดสตาร์ท (m):\\n", fast)
print("kernel ที่คอมไพล์ vs NumPy ต่างกันสูงสุด", np.abs(fast - slow).max())
""")

# 5 ------------------------------------------------------------------
md("""
## 5. เมทริกซ์ข้อมูลและ PCA

ชุดข้อมูลการขับ (`data/`) มีแถวละหนึ่งเวกเตอร์ เมื่อนำระยะรังสี 17 ค่ามาเรียงเป็นแถวจะได้เมทริกซ์ข้อมูล
$X\\in\\mathbb{R}^{N\\times 17}$ ขั้นตอน:
1. encode: $e = \\ln(1+d)/\\ln(201)$
2. standardize ต่อคอลัมน์: $Z = (E-\\boldsymbol{\\mu})\\,\\mathrm{diag}(\\boldsymbol\\sigma)^{-1}$
3. covariance: $C = \\frac{1}{N-1}Z^TZ$ (สมมาตร 17×17)
4. eigen-decomposition: $C\\mathbf{v}_i = \\lambda_i\\mathbf{v}_i$
5. ฉายลง $k$ eigenvector แรก: $Z_k = Z V_k$
""")
code("""
D = np.load(os.path.join(ROOT, "data", "dataset_arrays.npz"))
X_raw, X_clean, keep = D["rays_raw"], D["rays_clean"], D["keep"]
print("เมทริกซ์ข้อมูลดิบ X:", X_raw.shape, "  หลัง clean:", X_clean[keep].shape)
E = encode(X_clean[keep]); mu, sd = E.mean(0), E.std(0)
Z = (E - mu) / sd
C = Z.T @ Z / (len(Z) - 1)
print("C สมมาตร:", np.allclose(C, C.T), "  ขนาด", C.shape)
lam, V = np.linalg.eigh(C); order = np.argsort(lam)[::-1]; lam, V = lam[order], V[:, order]
print("eigenvalues:", lam.round(3))
i = 0
print("ตรวจ C v1 = lambda1 v1:", np.allclose(C @ V[:, i], lam[i] * V[:, i]))
print("V^T V = I:", np.allclose(V.T @ V, np.eye(17)))
print("ผลรวม eigenvalue = trace(C) =", lam.sum().round(4), "=", np.trace(C).round(4))
cum = np.cumsum(lam) / lam.sum()
for k in (2, 4, 8): print(f"k = {k}: อธิบายความแปรปรวนได้ {cum[k-1]*100:.1f}%")
""")
code("""
k = 4
Zk = Z @ V[:, :k]                       # ฉายลง 4 มิติ
Z_rec = Zk @ V[:, :k].T                 # สร้างกลับ
err = np.mean((Z - Z_rec)**2) / np.mean(Z**2)
print("ฉาย 17 -> 4 มิติ:", Z.shape, "->", Zk.shape)
print(f"ความคลาดเคลื่อนจากการสร้างกลับ {err*100:.1f}% = 1 - (ร้อยละที่อธิบายได้) = {(1-cum[k-1])*100:.1f}%")
print("eigenvector 1 (น้ำหนักของแต่ละรังสี):\\n", V[:, 0].round(3))
print("eigenvector 2:\\n", V[:, 1].round(3))
""")
code("""
calib_file = Calibration.load(os.path.join(ROOT, "data", "calibration.json"))
print("eigenvalue ที่โปรแกรมใช้จริง (calibration.json):", np.round(calib_file.eigvals[:5], 3))
Eb = encode(X_raw); Zb = (Eb - Eb.mean(0)) / Eb.std(0); lb = np.sort(np.linalg.eigvalsh(Zb.T @ Zb / (len(Zb)-1)))[::-1]
print("ก่อน clean: ", lb[:5].round(3))
print("หลัง clean: ", lam[:5].round(3))
print("dropout (ค่า 0 ผิดพลาด) เป็น noise ที่ไม่สัมพันธ์กับอะไร จึงกระจายความแปรปรวนออกไปทุกทิศ และกด eigenvalue แรก")
""")

# 6 ------------------------------------------------------------------
md("""
## 6. Neural network คือการคูณเมทริกซ์

แต่ละชั้น: $\\mathbf{a}^{(l+1)} = \\tanh\\left(W^{(l)}\\mathbf{a}^{(l)} + \\mathbf{b}^{(l)}\\right)$
ประชากร 100 ตัวเก็บเป็น tensor $W^{(l)}\\in\\mathbb{R}^{100\\times m\\times n}$ แล้วคูณครั้งเดียวด้วย batched matmul
""")
code("""
spec = InputSpec(17, calib_file)
cars = CarSwarm(100, track, num_sensors=17, sensor_noise=0)
x = spec.build(cars.sensor_distances(), cars)
print("เวกเตอร์ input ของรถคันที่ 0 (d = 21):\\n", x[0].round(3))
pop = Population(100, layer_sizes(21, 2, 16), rng=np.random.default_rng(0))
print("ขนาด tensor น้ำหนัก:", [w.shape for w in pop.weights], " จำนวนพารามิเตอร์/คัน:", pop.num_params)
y = pop.forward(x)
w, b = pop.get(0)
a = x[0]
for wl, bl in zip(w, b):
    a = np.tanh(wl @ a + bl)
print("คำนวณมือสำหรับคันที่ 0:", a.round(4))
print("batched matmul คันที่ 0 :", y[0].round(4), "  เท่ากัน:", np.allclose(a, y[0], atol=1e-5))
""")

# 7 ------------------------------------------------------------------
md("""
## 7. Genetic algorithm: การกลายพันธุ์เป็นการบวกเวกเตอร์

น้ำหนักทั้งหมดของรถหนึ่งคันคือเวกเตอร์ $\\mathbf{w}\\in\\mathbb{R}^{P}$ (P = 692 สำหรับ 2×16)
ลูกถูกสร้างจากพ่อแม่ด้วย $\\mathbf{w}' = \\mathbf{w} + \\sigma\\,\\boldsymbol\\varepsilon$
โดยขนาดของ noise ของแต่ละชั้นหารด้วย $\\sqrt{n_{in}}$ ทำให้ขนาดการรบกวนของ pre-activation ไม่ขึ้นกับความกว้างของชั้น
""")
code("""
from racing.ga import evolve
fit = np.random.default_rng(1).random(100)
new, src, sigma = evolve(pop, fit, np.random.default_rng(2))
flat = lambda p, i: np.concatenate([w[i].ravel() for w in p.weights] + [b[i].ravel() for b in p.biases])
for child in (0, 5, 40):
    d = flat(new, child) - flat(pop, src[child])
    print(f"ลูก {child}: พ่อแม่ #{src[child]}, sigma_rel = {sigma[child]:.2f}, ||w' - w|| = {np.linalg.norm(d):.3f}")
for wl in pop.weights:
    n_in = wl.shape[2]
    a_in = np.random.default_rng(0).normal(size=n_in)
    dz = (np.random.default_rng(1).normal(size=(2000, n_in)) * 0.1 / np.sqrt(n_in)) @ a_in
    print(f"ชั้น n_in = {n_in:2d}: SD ของการเปลี่ยน pre-activation = {dz.std():.3f} ~ 0.1 * |a|/sqrt(n) * sqrt(n)")
""")

# 8 ------------------------------------------------------------------
md("""
## 8. การบีบอัด: least squares, quantization, SVD

**Pruning + least squares:** หลังตัดโหนดออก หาน้ำหนักชั้นถัดไปใหม่ $W'$ ที่ทำให้
$\\|A W^T + \\mathbf{1}\\mathbf{b}^T - [A_k\\ \\mathbf{1}]\\,\\Theta\\|^2$ ต่ำสุด
คำตอบจาก **สมการปกติ** $(\\tilde A^T\\tilde A + \\lambda I)\\Theta = \\tilde A^T Z$

**Quantization:** $W_q = s\\cdot\\mathrm{round}(W/s)$ แยกสเกล $s$ ต่อแถว

**SVD:** $W = U\\Sigma V^T$, เก็บแค่ $r$ ค่าแรก: $W_r = U_r\\Sigma_rV_r^T$ ซึ่งเป็นค่าประมาณอันดับ $r$ ที่ดีที่สุด (Eckart–Young)
""")
code("""
import glob
from racing.compress import prune, quantise, low_rank, open_loop_error, effective_rank, forward_all
snaps = sorted(glob.glob(os.path.join(ROOT, "experiments", "results", "*", "*", "seed*", "snapshots", "*.npz")))
net_src = None
for s in snaps[::-1]:
    p, meta = Population.load(s)
    if p.sizes[1:-1] and max(p.sizes[1:-1]) >= 16 and p.sizes[0] == 21:
        net_src = s; break
print("โมเดลที่ใช้:", os.path.relpath(net_src, ROOT) if net_src else "สุ่ม")
p, meta = Population.load(net_src) if net_src else (Population(1, layer_sizes(21, 2, 16)), {})
wts, bs = p.get(meta.get("champion", 0))
net = ([w.astype(float) for w in wts], [b.astype(float) for b in bs])
Xc = x  # input จริงของรถ 100 คันบนสนาม
Xc = np.vstack([Xc + np.random.default_rng(i).normal(0, 0.05, Xc.shape) for i in range(20)])
print("ขนาดโครงข่าย:", [w.shape for w in net[0]])
""")
code("""
width = net[0][0].shape[0]
for m in ("random", "importance", "ls"):
    pr = prune(net, width // 2, Xc, method=m, rng=np.random.default_rng(0))
    print(f"ตัดเหลือ {width//2} โหนด แบบ {m:10s}: open-loop RMSE = {open_loop_error(net, pr, Xc):.4f}")
for bits in (8, 4, 2):
    q, _ = quantise(net, bits)
    print(f"quantize {bits} บิต: RMSE = {open_loop_error(net, q, Xc):.4f}")
W = net[0][1] if len(net[0]) > 2 else net[0][0]
U, S, Vt = np.linalg.svd(W)
print("singular values ของชั้นซ่อน:", S.round(3))
er = effective_rank(W)
print(f"effective rank (พลังงาน 90%) = {er['rank90']} จาก {er['max_rank']},  entropy rank = {er['entropy_rank']:.2f}")
for r in (2, 4, 8):
    Wr = (U[:, :r] * S[:r]) @ Vt[:r]
    print(f"rank {r}: ||W - W_r||_F = {np.linalg.norm(W - Wr):.4f} = sqrt(sum sigma_i^2, i>r) = {np.sqrt((S[r:]**2).sum()):.4f}")
""")

nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, "linear_algebra_calculations.ipynb")
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": OUT_DIR}})
nbformat.write(nb, path)
html, _ = HTMLExporter().from_notebook_node(nb)
with open(path.replace(".ipynb", ".html"), "w") as f:
    f.write(html)
print("wrote", path)

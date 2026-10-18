# สคริปต์วิดีโออธิบายการทำงาน (~6–8 นาที)

อัดหน้าจอพร้อมพูด (OBS หรือ Win+G) เปิด VS Code/terminal และโปรแกรมไว้ล่วงหน้า

| เวลา | ภาพบนจอ | สิ่งที่พูด |
|---|---|---|
| 0:00–0:30 | `python -m app.demo` เมนู | แนะนำโครงงาน: รถแข่ง 2D ที่ขับด้วย neural network บนสนามสุ่มแบบ F1 |
| 0:30–1:30 | `figures/tracks_d10.png`, `tracks_d30.png`, `racing/track.py` | การสร้างสนาม: convex hull → ท่าโค้ง → Bezier (เมทริกซ์พื้นฐาน 4×4) → ขอบถนนจากเวกเตอร์ตั้งฉาก; ความยาก 0–3 |
| 1:30–2:30 | viewer ขับ 1 รอบ ชี้รังสีและแผง telemetry | ฟิสิกส์ (ยาง, แรงกด, ท้ายปัด) · เซนเซอร์ = ระบบสมการ 2×2 แก้ด้วยกฎของคราเมอร์ · การชนใช้การฉายด้วยผลคูณจุด |
| 2:30–3:30 | แผง NETWORK ขวาล่าง + `racing/network.py` forward | อินพุต 13 มิติ (log + standardize) → tanh(Wx+b) ทีละชั้น → 4 คำสั่ง; ประชากร 100 ตัวคูณเมทริกซ์ครั้งเดียว |
| 3:30–4:30 | `figures/data_*.png`, notebook หัวข้อ 5 | ข้อมูล 22,819 แถว → ทำความสะอาด → PCA: ค่าไอเกนของเมทริกซ์ความแปรปรวนร่วม, แกนมีความหมาย |
| 4:30–5:30 | `figures/s3_station_finish.png`, `cmp_variants_by_level.png` | ฝึก 2 แบบ: GA (การกลายพันธุ์ = บวกเวกเตอร์) vs เรียนจากครู (backprop = ผลคูณเมทริกซ์ทรานสโพส); ผล: ขนาดเหมาะขึ้นกับวิธีฝึก |
| 5:30–6:15 | `figures/sens_dimensions.png`, `s4bc_pareto_XL.png` | เซนเซอร์/PCA และการบีบอัด (สมการปกติ, quantization, SVD) |
| 6:15–7:30 | demo: สนามวาดเอง + SHOW (17 rays + PCA 8) บนระดับ 3.0, `figures/showcase_pca8_training.png` | ตัวโชว์: เรียนจากครู → สนามโหด → GA ฝึกต่อ จนเร็วกว่าครู (24.3 vs 23.3 ม./วิ.) และสรุป |

คำสั่งสำหรับอัดภาพนิ่ง/เฟรมแบบไม่เปิดหน้าต่าง: `python -m app.viewer --run <run> --snapshot <file> --track <seed> --difficulty <d> --frames 600 --frame-every 2`
(เฟรมอยู่ใน `figures/frames/` นำไปต่อเป็นวิดีโอด้วย `ffmpeg -framerate 30 -i figures/frames/f%05d.png -pix_fmt yuv420p clip.mp4`)

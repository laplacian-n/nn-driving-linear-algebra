// Numbers of the experiments that finish late (GA sensor check, show model),
// read straight from their result files so the report is rebuilt, not retyped.
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "experiments", "results");

const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const readJSON = (p) => JSON.parse(fs.readFileSync(p, "utf8"));

function gaSensors() {
  const conds = [["raw5", "รังสีดิบ 5 เส้น"], ["raw9", "รังสีดิบ 9 เส้น"], ["raw17", "รังสีดิบ 17 เส้น"], ["pca4", "PCA 4 มิติจาก 17 รังสี"]];
  const parts = [];
  let any = false;
  for (const [c, label] of conds) {
    const dir = path.join(RES, "ga_sensors", c);
    if (!fs.existsSync(dir)) continue;
    const vals = [], during = [], reached = [];
    for (const s of fs.readdirSync(dir).filter((d) => d.startsWith("seed"))) {
      const f = path.join(dir, s, "final_test.json");
      const st = path.join(dir, s, "station_tests.json");
      if (!fs.existsSync(f)) continue;
      const ft = readJSON(f);
      vals.push(100 * mean(ft.finished));
      if (fs.existsSync(st)) {
        const t = readJSON(st);
        during.push(100 * mean(t.map((x) => mean(x.finished))));
        reached.push(t[t.length - 1].level);
      }
    }
    if (!vals.length) continue;
    any = true;
    parts.push(`${label} ${median(vals).toFixed(0)}% (${vals.length} seed, ไปถึงสถานี ${Math.min(...reached).toFixed(2)}–${Math.max(...reached).toFixed(2)})`);
  }
  if (!any) return "[ผลการตรวจด้วย Genetic Algorithm ยังไม่เสร็จ]";
  return "เพื่อดูแนวโน้มของ Genetic Algorithm ประกอบ ผู้จัดทำได้ฝึกโมเดล S 2×8 ด้วย Genetic Algorithm กับอินพุต 3 แบบ แบบละ 3 seed โดยใช้หลักสูตรเดียวกับการทดลอง A แต่จำกัดเวลาฝึกไว้ประมาณ 6 ชั่วโมง run ที่ยังไม่จบเมื่อหมดเวลาจะถูกสอบด้วยโมเดลของสถานีล่าสุด ส่วนรังสีดิบ 9 เส้นใช้ผลของการทดลอง A ซึ่งฝึกจนจบโดยไม่จำกัดเวลา (9–11 ชั่วโมง) ผลอัตราการจบรอบของโมเดลสุดท้ายบนชุด T ทุกระดับ (ค่ามัธยฐาน) คือ "
    + parts.join(", ")
    + " ภายในเวลาเท่ากัน อินพุตยิ่งมีมิติมาก Genetic Algorithm ยิ่งเลื่อนผ่านหลักสูตรได้ช้า สอดคล้องกับงานของ Park และ Kim [18] ที่พบว่าการค้นหาด้วยวิวัฒนาการยากขึ้นตามมิติของปัญหา แต่ PCA 4 มิติกลับไม่ได้ช่วย Genetic Algorithm แม้จะทำได้ดีเมื่อฝึกด้วยการเรียนจากครู คำอธิบายที่เป็นไปได้คือในโมเดล S จำนวนพารามิเตอร์ของชั้นแรกต่างกันเพียงไม่กี่สิบค่า พื้นที่ค้นหาจึงแทบไม่เล็กลง ขณะที่รังสีดิบแต่ละเส้นบอกระยะในทิศหนึ่งโดยตรง การกลายพันธุ์ของน้ำหนักเพียงตัวเดียวจึงเปลี่ยนพฤติกรรมได้อย่างมีความหมาย ส่วนองค์ประกอบหลักแต่ละตัวเป็นผลรวมของทุกรังสี การปรับแบบสุ่มจึงหาทิศทางที่ดีได้ยากกว่า ผลชุดนี้มี seed น้อยและหลาย run ฝึกไม่ครบทุกสถานี จึงใช้ได้เพียงเป็นแนวโน้มประกอบเท่านั้น";
}

function showcase() {
  const pct = (x) => `${(100 * x).toFixed(0)}%`;
  const f1 = (x) => x.toFixed(1);
  const load = (name) => {
    const dir = path.join(RES, name);
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir).filter((d) => fs.existsSync(path.join(dir, d, "final.json"))).sort()
      .map((d) => ({ s: d, r: readJSON(path.join(dir, d, "final.json")) }));
  };
  const v9 = load("showcase"), vp = load("showcase_pca8");
  const intro = "โมเดลตัวโชว์รวมวิธีที่ดีที่สุดที่พบจากการทดลอง คือ XL 2×64 ที่เรียนจากครูที่ใช้ยาง 75% และฝึกนาน 2 เท่า (ผลดีที่สุดของการทดลอง C) แล้วฝึกต่อในสองระยะ ระยะแรกเรียนจากครูต่อด้วย DAgger บนสนามพิเศษ ซึ่งคัดจากสนามระดับ 2.5 และ 3.0 (ถนนกว้าง 8.5 และ 7 เมตร เกินขอบเขตของทุกการทดลอง) จำนวนระดับละ 200 สนาม โดยเลือกสนามที่ตัวควบคุมอ้างอิงต้องขับช้าที่สุด คือสนามที่มีโค้งแคบและโค้งต่อเนื่องมากที่สุด ระยะที่สองปล่อยให้ Genetic Algorithm ฝึกต่อเองโดยไม่มีครู เริ่มจากประชากรที่กลายพันธุ์เล็กน้อยจากโมเดลระยะแรก (σ = 0.003–0.03) และให้คะแนนตามความเร็วที่จบรอบ เพื่อดูว่ารถจะเก่งเกินครูได้หรือไม่ โมเดลที่เก็บไว้ในแต่ละระยะถูกเลือกด้วยสนาม validation ซึ่งประกอบด้วยสนามพิเศษระดับ 2.5 และ 3.0 กับสนามระดับ 2.0 ผู้จัดทำฝึกตัวโชว์ 2 แบบ แบบละ 3 seed คือแบบที่ใช้รังสี 9 เส้น และแบบที่ใช้รังสี 17 เส้นลดมิติด้วย PCA เหลือ 8 มิติ ตามผลของการทดลอง D ทั้งสองแบบใช้วิธีฝึกและเวลาใกล้เคียงกัน แล้วสอบบนสนามโชว์ระดับ 3.0 จำนวน 20 สนามที่ไม่เคยใช้ฝึก (คัดสนามที่ยากที่สุดจาก 200 สนามเช่นกัน) และบนชุดสอบ T";
  if (!v9.length && !vp.length) return { paragraphs: [intro, "[ผลของโมเดลตัวโชว์ยังไม่เสร็จ]"] };
  const avg = (runs, m, set, key) => mean(runs.map((x) => x.r[m][set][key] || 0));
  const rows = [];
  for (const [lab, runs] of [["9 รังสี", v9], ["17 รังสี + PCA 8", vp]]) {
    if (!runs.length) continue;
    for (const [stage, m] of [["โมเดลเริ่มต้น", "bc_base"], ["+ ครูบนสนามพิเศษ", "bc_special"], ["+ Genetic Algorithm", "bc_ga"]])
      rows.push([`${lab}: ${stage}`, pct(avg(runs, m, "show", "finish")), f1(avg(runs, m, "show", "lap_speed")), pct(avg(runs, m, "T", "finish"))]);
  }
  const vref = mean((v9[0] || vp[0]).r.teacher_show_v_ref);
  const paras = [intro];
  if (v9.length && vp.length) {
    const gaSpd = [...v9, ...vp].map((x) => x.r.bc_ga.show.lap_speed);
    paras.push(`ตารางที่ 13 แสดงค่าเฉลี่ยของ 3 seed ในแต่ละแบบ ผลที่สำคัญมี 4 ประการ ประการแรก การฝึกต่อบนสนามพิเศษได้ผลชัดเจน โมเดลเริ่มต้นจบรอบสนามโชว์ได้เพียง ${pct(avg(v9, "bc_base", "show", "finish"))} (9 รังสี) และ ${pct(avg(vp, "bc_base", "show", "finish"))} (PCA) แต่หลังเรียนจากครูบนสนามพิเศษเพิ่มเป็น ${pct(avg(v9, "bc_special", "show", "finish"))} และ ${pct(avg(vp, "bc_special", "show", "finish"))} ประการที่สอง Genetic Algorithm ทำให้รถเร็วขึ้นในทุก run โดยความเร็วเฉลี่ยของรอบที่จบบนสนามโชว์อยู่ระหว่าง ${f1(Math.min(...gaSpd))}–${f1(Math.max(...gaSpd))} เมตรต่อวินาที สูงกว่าตัวควบคุมอ้างอิงที่ใช้สร้างข้อมูลสอน (${f1(vref)} เมตรต่อวินาทีบนสนามชุดเดียวกัน) ทั้ง 6 run เพราะ Genetic Algorithm ให้คะแนนตามความเร็วโดยตรง ไม่ได้เลียนแบบครู แต่แลกกับอัตราการจบรอบที่ลดลงเล็กน้อย ประการที่สาม แบบ PCA ดีกว่าแบบ 9 รังสีในทุกระยะหลังการฝึกต่อ ทั้งอัตราการจบรอบสนามโชว์ ความเร็ว และการรักษาความสามารถบนชุด T ซึ่งสอดคล้องกับผลการทดลอง D ว่ารังสีที่มากขึ้นช่วยให้เห็นโค้งแคบได้ดีขึ้น และ PCA ลดมิติลงได้โดยไม่เสียข้อมูลเมื่อฝึกด้วย gradient ประการที่สี่ การฝึกต่อทำให้โมเดลลืมสนามปกติบางส่วน อัตราการจบรอบบนชุด T ลดจากเกือบ 100% เหลือ ${pct(avg(v9, "bc_ga", "T", "finish"))} (9 รังสี) และ ${pct(avg(vp, "bc_ga", "T", "finish"))} (PCA) เนื่องจากสนาม validation ไม่มีสนามง่ายระดับต่ำกว่า 2.0 การเลือกโมเดลจึงไม่ได้ป้องกันการลืมสนามเหล่านั้น`);
    const best = vp.slice().sort((a, b) => b.r.bc_ga.show.finish - a.r.bc_ga.show.finish || b.r.bc_ga.T.finish - a.r.bc_ga.T.finish)[0];
    paras.push(`ข้อสังเกตเพิ่มเติมคือผลระหว่าง seed ต่างกันได้ 10–20 จุด และสนามโชว์มีเพียง 20 สนาม (1 สนามเท่ากับ 5%) ความต่างเล็กน้อยระหว่างสองแบบจึงควรตีความอย่างระมัดระวัง สำหรับโปรแกรมสาธิตเลือกใช้แบบ PCA ${best.s} ซึ่งจบรอบสนามโชว์ ${pct(best.r.bc_ga.show.finish)} ที่ความเร็ว ${f1(best.r.bc_ga.show.lap_speed)} เมตรต่อวินาที และจบรอบชุด T ${pct(best.r.bc_ga.T.finish)} พร้อมโมเดลระยะแรกของ run เดียวกันเป็นตัวเลือกที่ขับปลอดภัยกว่า (สนามโชว์ ${pct(best.r.bc_special.show.finish)}, ชุด T ${pct(best.r.bc_special.T.finish)})`);
  }
  return {
    paragraphs: paras,
    table: ["ผลของโมเดลตัวโชว์แต่ละระยะการฝึก (ค่าเฉลี่ย 3 seed)", ["โมเดล", "สนามโชว์ 3.0 จบรอบ", "สนามโชว์ ความเร็ว (ม./วิ.)", "ชุด T จบรอบ"], rows, [3800, 1800, 1800, 1600]],
  };
}

const sc = showcase();
module.exports = {
  gaSensorsParagraph: gaSensors(),
  showcaseParagraphs: sc.paragraphs,
  showcaseTable: sc.table,
  showcaseFigure: fs.existsSync(path.join(ROOT, "figures", "showcase_pca8_training.png"))
    ? ["figures/showcase_pca8_training.png", "การฝึกโมเดลตัวโชว์แบบ 17 รังสี + PCA 8: คะแนน validation ระหว่างเรียนจากครูบนสนามพิเศษ และระหว่าง Genetic Algorithm ฝึกต่อ (แบบ 9 รังสีมีแนวโน้มเดียวกัน)", 15.5] : null,
};

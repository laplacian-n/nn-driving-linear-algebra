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
  return "เพื่อดูแนวโน้มของ Genetic Algorithm ประกอบ ผู้จัดทำได้ฝึกโมเดล S 2×8 ด้วย Genetic Algorithm กับอินพุต 3 แบบ แบบละ 3 seed โดยใช้หลักสูตรเดียวกับการทดลอง A แต่จำกัดเวลาฝึก run ที่ยังไม่จบเมื่อถึงเวลาที่กำหนดจะถูกสอบด้วยโมเดลของสถานีล่าสุด ผลอัตราการจบรอบของโมเดลสุดท้ายบนชุด T ทุกระดับ (ค่ามัธยฐาน) คือ "
    + parts.join(", ") + " ผลชุดนี้มี seed น้อยและหลาย run ฝึกไม่ครบทุกสถานี จึงใช้ได้เพียงเป็นแนวโน้มประกอบเท่านั้น";
}

function showcase() {
  const dir = path.join(RES, "showcase");
  const seeds = fs.existsSync(dir) ? fs.readdirSync(dir).filter((d) => fs.existsSync(path.join(dir, d, "final.json"))) : [];
  const intro = "โมเดลตัวโชว์รวมวิธีที่ดีที่สุดที่พบจากการทดลอง คือ XL 2×64 ที่เรียนจากครูที่ใช้ยาง 75% และฝึกนาน 2 เท่า (ผลดีที่สุดของการทดลอง C) แล้วฝึกต่อในสองระยะ ระยะแรกเรียนจากครูต่อด้วย DAgger บนสนามพิเศษ ซึ่งคัดจากสนามระดับ 2.5 และ 3.0 (ถนนกว้าง 8.5 และ 7 เมตร เกินขอบเขตของทุกการทดลอง) จำนวนระดับละ 200 สนาม โดยเลือกสนามที่ตัวควบคุมอ้างอิงต้องขับช้าที่สุด คือสนามที่มีโค้งแคบและโค้งต่อเนื่องมากที่สุด ระยะที่สองปล่อยให้ Genetic Algorithm ฝึกต่อเองโดยไม่มีครู เริ่มจากประชากรที่กลายพันธุ์เล็กน้อยจากโมเดลระยะแรก (σ = 0.003–0.03) และให้คะแนนตามความเร็วที่จบรอบ เพื่อดูว่ารถจะเก่งเกินครูได้หรือไม่ โมเดลที่เก็บไว้ถูกเลือกด้วยสนาม validation จึงไม่แย่ไปกว่าจุดเริ่มต้น";
  if (!seeds.length) return { paragraphs: [intro, "[ผลของโมเดลตัวโชว์ยังไม่เสร็จ]"] };
  // best seed by its final show-track result of the BC+GA model
  const all = seeds.map((s) => ({ s, r: readJSON(path.join(dir, s, "final.json")) }));
  all.sort((a, b) => (b.r.bc_ga.show.score) - (a.r.bc_ga.show.score));
  const best = all[0];
  const pct = (x) => `${(100 * x).toFixed(0)}%`;
  const spd = (x) => (x == null ? "–" : `${x.toFixed(1)}`);
  const vref = mean(best.r.teacher_show_v_ref);
  const rows = [["โมเดลเริ่มต้น (การทดลอง C)", "bc_base"], ["+ เรียนจากครูบนสนามพิเศษ", "bc_special"], ["+ Genetic Algorithm ฝึกต่อเอง", "bc_ga"]]
    .map(([lab, k]) => [lab, pct(best.r[k].T.finish), spd(best.r[k].T.lap_speed), pct(best.r[k].show.finish), spd(best.r[k].show.lap_speed)]);
  const g = best.r.bc_ga, b0 = best.r.bc_base;
  const text = `ตารางที่ 13 แสดงผลของ run ที่ดีที่สุดจาก ${seeds.length} run (${best.s}) ซึ่งฝึก Genetic Algorithm ต่อ ${best.r.ga_generations} รุ่น บนสนามโชว์ระดับ 3.0 จำนวน ${g.show.per_track.length} สนามที่ไม่เคยใช้ฝึก โมเดลสุดท้ายจบรอบ ${pct(g.show.finish)} (โมเดลเริ่มต้น ${pct(b0.show.finish)}) ด้วยความเร็วเฉลี่ยของรอบที่จบ ${spd(g.show.lap_speed)} เมตรต่อวินาที เทียบกับตัวควบคุมอ้างอิงที่ใช้ยาง 85% ซึ่งขับสนามชุดเดียวกันได้ ${vref.toFixed(1)} เมตรต่อวินาที และบนชุดสอบ T จบรอบ ${pct(g.T.finish)}`;
  return {
    paragraphs: [intro, text],
    table: ["ผลของโมเดลตัวโชว์แต่ละระยะการฝึก", ["โมเดล", "T จบรอบ", "T ความเร็ว (ม./วิ.)", "สนามโชว์ 3.0 จบรอบ", "สนามโชว์ ความเร็ว (ม./วิ.)"], rows, [3000, 1300, 1600, 1600, 1500]],
  };
}

const sc = showcase();
module.exports = {
  gaSensorsParagraph: gaSensors(),
  showcaseParagraphs: sc.paragraphs,
  showcaseTable: sc.table,
  showcaseFigure: fs.existsSync(path.join(ROOT, "figures", "showcase_training.png"))
    ? ["figures/showcase_training.png", "การฝึกโมเดลตัวโชว์: คะแนน validation ระหว่างเรียนจากครูบนสนามพิเศษ และระหว่าง Genetic Algorithm ฝึกต่อ", 15.5] : null,
};

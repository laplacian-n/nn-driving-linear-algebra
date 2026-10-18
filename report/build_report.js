// Builds report/report.docx (Thai, TH Sarabun New).   node report/build_report.js
// Layout: KMITL thesis guide (see lib.js). Front matter numbered I, II, ... at the bottom centre,
// chapters numbered 1, 2, ... at the top right, no number on the first page of a chapter.
const fs = require("fs");
const path = require("path");
const { D, P, FONT, BODY, sz, PAGE, TEXT_W, numberingConfig, resetNumbering } = require("./lib");
const part1 = require("./part1");
const part2 = require("./part2");

const center = (text, size, bold = false, after = 120) => new D.Paragraph({
  children: [new D.TextRun({ text, font: FONT, size: sz(size), bold })],
  alignment: D.AlignmentType.CENTER, spacing: { after },
});

const cover = [
  center("2D Map Self-Driving Car Neural Network", 40, true, 120),
  center("รถแข่งขับเคลื่อนอัตโนมัติในแผนที่สองมิติด้วยโครงข่ายประสาทเทียม", 36, true, 2400),
  center("ชื่อ นามสกุล ____________________  รหัสนักศึกษา ____________", 32),
  center("ชื่อ นามสกุล ____________________  รหัสนักศึกษา ____________", 32, false, 2400),
  center("รายงานนี้เป็นส่วนหนึ่งของการศึกษารายวิชาพีชคณิตเชิงเส้น (Linear Algebra)", 32),
  center("อาจารย์ผู้สอน ____________________", 32),
  center("คณะ ____________________", 32),
  center("สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง", 32),
  center("ภาคเรียนที่ 1 ปีการศึกษา 2569", 32),
];

const title = (text, newPage = true) => new D.Paragraph({
  children: [new D.TextRun({ text, font: FONT, size: sz(48), bold: true })],
  alignment: D.AlignmentType.CENTER, spacing: { after: 360 }, pageBreakBefore: newPage,
});

const abstract = [
  title("บทคัดย่อ", false),
  P("โครงงานนี้พัฒนาระบบจำลองรถแข่งสองมิติที่ขับด้วยโครงข่ายประสาทเทียมบนสนามแบบ F1 ซึ่งสร้างขึ้นใหม่แบบสุ่มทุกครั้ง รถรับรู้สนามผ่านเซนเซอร์วัดระยะเท่านั้น ระบบทั้งหมดเขียนด้วย NumPy และอาศัยพีชคณิตเชิงเส้นในทุกขั้นตอน ได้แก่ การหาจุดตัดของรังสีด้วยกฎของคราเมอร์ การหมุนและการฉายเวกเตอร์ในแบบจำลองฟิสิกส์ การคูณเมทริกซ์ใน neural network การลดมิติข้อมูลด้วยค่าไอเกน (PCA) การหา gradient ด้วยเมทริกซ์ทรานสโพส และการบีบอัดโมเดลด้วยสมการปกติและการแยกค่าเอกฐาน"),
  P("ผลการทดลองกับโมเดล 4 ขนาดบนสนามทดสอบ 210 สนามที่ไม่เคยใช้ฝึก พบว่าขนาดโมเดลที่เหมาะสมขึ้นกับวิธีฝึก เมื่อฝึกด้วย Genetic Algorithm โมเดลขนาดเล็กดีที่สุด (จบรอบร้อยละ 81) แต่เมื่อเรียนจากครูด้วย gradient โมเดลขนาดใหญ่ดีที่สุด (ร้อยละ 94) การใช้เซนเซอร์ 17 รังสีแล้วลดมิติด้วย PCA เหลือ 4–8 มิติรักษาความสามารถไว้ได้เกือบทั้งหมด และการลดจำนวนบิตเหลือ 8 bit ลดขนาดโมเดลได้ 3.6 เท่าโดยแทบไม่เสียความสามารถ ส่วนการตัดโหนดและ low-rank ทำให้การขับแบบวงปิดล้มเหลว โมเดลตัวโชว์ที่เรียนจากครูแล้วให้ Genetic Algorithm ฝึกต่อบนสนามที่ถนนกว้างเพียง 7 เมตร จบรอบสนามที่ยากที่สุดได้ร้อยละ 80 และขับเร็วกว่าตัวควบคุมที่ใช้สอนมัน"),
  P("คำสำคัญ: พีชคณิตเชิงเส้น, โครงข่ายประสาทเทียม, รถขับเคลื่อนอัตโนมัติ, PCA, Genetic Algorithm, behavior cloning", { noIndent: true }),
];

const pageWord = new D.Paragraph({
  children: [new D.TextRun({ text: "หน้า", font: FONT, size: BODY, bold: true })],
  alignment: D.AlignmentType.RIGHT, spacing: { after: 120 },
});
const listHead = (label) => new D.Paragraph({
  tabStops: [{ type: D.TabStopType.RIGHT, position: TEXT_W }],
  children: [new D.TextRun({ text: `${label}\tหน้า`, font: FONT, size: BODY, bold: true })],
  spacing: { after: 120 },
});

const contents = [
  title("สารบัญ"), pageWord,
  new D.TableOfContents("สารบัญ", { hyperlink: true, headingStyleRange: "1-2" }),
  title("สารบัญตาราง"), listHead("ตารางที่"),
  new D.TableOfContents("สารบัญตาราง", { hyperlink: true, stylesWithLevels: [new D.StyleLevel("TableCaption", 1)] }),
  title("สารบัญรูป"), listHead("รูปที่"),
  new D.TableOfContents("สารบัญรูป", { hyperlink: true, stylesWithLevels: [new D.StyleLevel("FigureCaption", 1)] }),
];

// pass 1 only collects the per-chapter numbers of figures and tables
part1(); part2();
resetNumbering(true);
const body = [...part1(), ...part2()];

// split the body into one section per chapter
const chapters = [];
for (const item of body) {
  if (item && item.chapterStart) chapters.push([...item.paragraphs]);
  else chapters[chapters.length - 1].push(item);
}

const margin = { top: PAGE.top, right: PAGE.right, bottom: PAGE.bottom, left: PAGE.left, header: PAGE.header, footer: PAGE.header };
const pageNo = (alignment) => new D.Paragraph({ alignment,
  children: [new D.TextRun({ children: [D.PageNumber.CURRENT], font: FONT, size: BODY })] });
const empty = () => new D.Header({ children: [new D.Paragraph({ children: [] })] });

const sections = [
  { properties: { page: { size: { width: PAGE.width, height: PAGE.height }, margin } }, children: cover },
  { properties: { page: { size: { width: PAGE.width, height: PAGE.height }, margin,
                          pageNumbers: { start: 1, formatType: D.NumberFormat.UPPER_ROMAN } } },
    footers: { default: new D.Footer({ children: [pageNo(D.AlignmentType.CENTER)] }) },
    children: [...abstract, ...contents] },
  ...chapters.map((children, i) => ({
    properties: { titlePage: true,
                  page: { size: { width: PAGE.width, height: PAGE.height }, margin,
                          pageNumbers: i === 0 ? { start: 1, formatType: D.NumberFormat.DECIMAL } : { formatType: D.NumberFormat.DECIMAL } } },
    headers: { default: new D.Header({ children: [pageNo(D.AlignmentType.RIGHT)] }), first: empty() },
    footers: { default: new D.Footer({ children: [new D.Paragraph({ children: [] })] }) },
    children,
  })),
];

const doc = new D.Document({
  creator: "laplacian-n",
  title: "รายงานโครงงาน 2D Map Self-Driving Car Neural Network",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: sz(48), bold: true }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: sz(36), bold: true }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, bold: true }, paragraph: { outlineLevel: 2 } },
      { id: "FigureCaption", name: "FigureCaption", basedOn: "Normal", next: "Normal", run: { font: FONT, size: BODY } },
      { id: "TableCaption", name: "TableCaption", basedOn: "Normal", next: "Normal", run: { font: FONT, size: BODY } },
    ],
  },
  numbering: numberingConfig,
  sections,
});

D.Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, process.env.PDF_MODE ? "report_pdf.docx" : "report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${chapters.length} chapters)`);
});

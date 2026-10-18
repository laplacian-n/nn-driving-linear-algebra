// Builds report/report.docx (Thai, TH Sarabun New).   node report/build_report.js
const fs = require("fs");
const path = require("path");
const { D, P, FONT, sz, numberingConfig } = require("./lib");
const part1 = require("./part1");
const part2 = require("./part2");

const center = (text, size, bold = false, after = 120) => new D.Paragraph({
  children: [new D.TextRun({ text, font: FONT, size: sz(size), bold })],
  alignment: D.AlignmentType.CENTER, spacing: { after },
});

const cover = [
  center("รายงานโครงงาน", 44, true, 240),
  center("รายวิชา Linear Algebra", 36, false, 600),
  center("2D Map Self-Driving Car Neural Network", 40, true, 120),
  center("รถแข่งขับเคลื่อนอัตโนมัติในแผนที่สองมิติด้วยโครงข่ายประสาทเทียม", 36, true, 1200),
  center("ผู้จัดทำ", 32, true),
  center("ชื่อ–นามสกุล ____________________________   รหัสนักศึกษา ____________________", 32),
  center("ชื่อ–นามสกุล ____________________________   รหัสนักศึกษา ____________________", 32, false, 600),
  center("อาจารย์ผู้สอน ____________________________", 32, false, 1200),
  center("สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง", 32),
  center("ภาคเรียนที่ 1 ปีการศึกษา 2569", 32),
];

const toc = [
  new D.Paragraph({ children: [new D.TextRun({ text: "สารบัญ", font: FONT, size: sz(40), bold: true })],
                    alignment: D.AlignmentType.CENTER, pageBreakBefore: true, spacing: { after: 240 } }),
  new D.TableOfContents("สารบัญ", { hyperlink: true, headingStyleRange: "1-2" }),
];

const doc = new D.Document({
  creator: "laplacian-n",
  title: "รายงานโครงงาน 2D Map Self-Driving Car Neural Network",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: sz(32) } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: sz(40), bold: true }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: sz(36), bold: true }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: sz(32), bold: true }, paragraph: { outlineLevel: 2 } },
    ],
  },
  numbering: numberingConfig,
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1700, right: 1300 } } },
    footers: { default: new D.Footer({ children: [new D.Paragraph({ alignment: D.AlignmentType.CENTER,
      children: [new D.TextRun({ children: [D.PageNumber.CURRENT], font: FONT, size: sz(28) })] })] }) },
    children: [...cover, ...toc, ...part1(), ...part2()],
  }],
});

D.Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, process.env.PDF_MODE ? "report_pdf.docx" : "report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});

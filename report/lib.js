// Small helpers on top of docx-js so the report content reads like a document.
const fs = require("fs");
const path = require("path");
const D = require("docx");

// PDF_MODE: LibreOffice has no TH Sarabun New here, so the PDF build uses the
// open Sarabun font at the size that matches TH Sarabun New 16 pt visually
const PDF_MODE = !!process.env.PDF_MODE;
const FONT = PDF_MODE ? "Sarabun" : "TH Sarabun New";
const SCALE = PDF_MODE ? 0.78 : 1;
const sz = (n) => Math.round(n * SCALE);
const MATH = "Cambria Math";
const BODY = sz(32);      // half-points: 16 pt, the usual size for Thai academic text
const ROOT = path.resolve(__dirname, "..");

let figN = 0;
let tabN = 0;

function runs(text, base = {}) {
  // **bold** and `code` inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new D.TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new D.TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: BODY, ...base }));
    else out.push(new D.TextRun({ text: t.slice(1, -1), font: "Consolas", size: BODY - 8, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new D.TextRun({ text: text.slice(last), font: FONT, size: BODY, ...base }));
  return out;
}

const P = (text, opt = {}) => new D.Paragraph({
  children: runs(text),
  alignment: opt.align || D.AlignmentType.THAI_DISTRIBUTE,
  indent: opt.noIndent ? undefined : { firstLine: 720 },
  spacing: { after: 120, line: 276 },
  keepNext: opt.keepNext,
});

const H = (level, text) => new D.Paragraph({
  heading: [D.HeadingLevel.HEADING_1, D.HeadingLevel.HEADING_2, D.HeadingLevel.HEADING_3][level - 1],
  children: [new D.TextRun({ text, font: FONT, bold: true, size: sz([40, 36, 32][level - 1]) })],
  spacing: { before: level === 1 ? 360 : 240, after: 120 },
  keepNext: true,
  pageBreakBefore: level === 1,
});

const Eq = (text, label) => new D.Paragraph({
  children: [new D.TextRun({ text, font: MATH, size: sz(26) }),
             ...(label ? [new D.TextRun({ text: `\t(${label})`, font: FONT, size: BODY })] : [])],
  alignment: D.AlignmentType.CENTER,
  spacing: { before: 60, after: 120 },
});

const Bullets = (items, ref = "bullets") => items.map((t) => new D.Paragraph({
  children: runs(t),
  numbering: { reference: ref, level: 0 },
  spacing: { after: 60, line: 276 },
  alignment: D.AlignmentType.THAI_DISTRIBUTE,
}));

const Numbered = (items, ref) => Bullets(items, ref);

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function Fig(rel, caption, widthCm = 15.5) {
  figN += 1;
  const file = path.join(ROOT, rel);
  const { w, h } = pngSize(file);
  const wpx = widthCm / 2.54 * 96;
  return [
    new D.Paragraph({
      children: [new D.ImageRun({ type: "png", data: fs.readFileSync(file),
                                  transformation: { width: wpx, height: wpx * h / w } })],
      alignment: D.AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    }),
    new D.Paragraph({
      children: [new D.TextRun({ text: `รูปที่ ${figN} `, bold: true, font: FONT, size: sz(28) }),
                 ...runs(caption, { size: sz(28) })],
      alignment: D.AlignmentType.CENTER, spacing: { after: 200 },
    }),
  ];
}

const border = { style: D.BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function Table(caption, header, rows, widths, opt = {}) {
  tabN += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const size = sz(opt.size || 28);
  const cell = (t, i, head) => new D.TableCell({
    borders,
    width: { size: widths[i], type: D.WidthType.DXA },
    shading: head ? { fill: "E8EEF4", type: D.ShadingType.CLEAR } : undefined,
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: [new D.Paragraph({
      children: runs(String(t), { size, bold: head || undefined }),
      alignment: i === 0 && !opt.centerFirst ? D.AlignmentType.LEFT : D.AlignmentType.CENTER,
    })],
  });
  return [
    new D.Paragraph({
      children: [new D.TextRun({ text: `ตารางที่ ${tabN} `, bold: true, font: FONT, size: sz(28) }),
                 ...runs(caption, { size: sz(28) })],
      keepNext: true, spacing: { before: 160, after: 60 },
    }),
    new D.Table({
      width: { size: total, type: D.WidthType.DXA },
      columnWidths: widths,
      rows: [new D.TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
             ...rows.map((r) => new D.TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))],
    }),
    new D.Paragraph({ children: [], spacing: { after: 120 } }),
  ];
}

const Code = (lines) => lines.map((l) => new D.Paragraph({
  children: [new D.TextRun({ text: l || " ", font: "Consolas", size: sz(18) })],
  spacing: { after: 0, line: 240 },
  shading: { fill: "F4F4F4", type: D.ShadingType.CLEAR },
}));

const numberingConfig = {
  config: [
    { reference: "bullets", levels: [{ level: 0, format: D.LevelFormat.BULLET, text: "•", alignment: D.AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ...Array.from({ length: 12 }, (_, i) => ({ reference: `num${i}`, levels: [{ level: 0, format: D.LevelFormat.DECIMAL,
      text: "%1.", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] })),
  ],
};

module.exports = { sz, D, P, H, Eq, Bullets, Numbered, Fig, Table, Code, FONT, BODY, ROOT, numberingConfig };

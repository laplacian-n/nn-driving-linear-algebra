// Small helpers on top of docx-js so the report content reads like a document.
// Layout follows the KMITL thesis guide (แบบฟอร์มการจัดรูปเล่มวิทยานิพนธ์ภาษาไทย, คณะวิศวกรรมศาสตร์):
// TH Sarabun New 16 pt, chapter "บทที่ n" 20 pt + chapter title 24 pt centred, section 18 pt,
// sub-section 16 pt indented, tables / figures numbered per chapter (ตารางที่ 2.1, รูปที่ 2.1),
// equations in Cambria Math 11 pt with the number at the right margin.
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
const BODY = sz(32);      // half-points: 16 pt
const ROOT = path.resolve(__dirname, "..");

// page: A4, margins top / right / bottom 1 inch, left 1.5 inch
const PAGE = { width: 11906, height: 16838, top: 1440, right: 1440, bottom: 1440, left: 2160, header: 720 };
const TEXT_W = PAGE.width - PAGE.left - PAGE.right;
const INDENT = 720;       // 7 characters of TH Sarabun New 16 pt

// numbering state. The text refers to figures and tables by their running
// number (รูปที่ 7); the report is built twice and the second pass rewrites
// those references to the per-chapter numbers (รูปที่ 2.5).
const S = { chapter: 0, fig: 0, tab: 0, figAll: 0, tabAll: 0, figMap: {}, tabMap: {} };
function resetNumbering(keepMaps) {
  Object.assign(S, { chapter: 0, fig: 0, tab: 0, figAll: 0, tabAll: 0 });
  if (!keepMaps) Object.assign(S, { figMap: {}, tabMap: {} });
}

function renumber(text) {
  const swap = (map) => (m, word, a, b) => `${word} ${map[a] || a}${b ? ` และ ${map[b] || b}` : ""}`;
  return text
    .replace(/(รูปที่) (\d+)(?![.\d])(?: และ (\d+)(?![.\d]))?/g, swap(S.figMap))
    .replace(/(ตารางที่) (\d+)(?![.\d])(?: และ (\d+)(?![.\d]))?/g, swap(S.tabMap));
}

function runs(text, base = {}) {
  // **bold** and `code` inline markup
  text = renumber(text);
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
  alignment: opt.align || D.AlignmentType.LEFT,
  indent: opt.noIndent ? undefined : { firstLine: INDENT },
  spacing: { after: 120, line: 276 },
  keepNext: opt.keepNext,
});

// H(1, "2. ชื่อบท") starts a chapter: build_report.js turns the marker into a new
// section (new page, no page number on the first page)
function H(level, text) {
  if (level === 1) {
    const m = text.match(/^(\d+)\.\s*(.*)$/);
    S.chapter = m ? Number(m[1]) : S.chapter + 1;
    S.fig = 0;
    S.tab = 0;
    return {
      chapterStart: S.chapter,
      paragraphs: [new D.Paragraph({
        heading: D.HeadingLevel.HEADING_1,
        alignment: D.AlignmentType.CENTER,
        spacing: { before: 0, after: 360 },
        keepNext: true,
        children: [new D.TextRun({ text: `บทที่ ${S.chapter}`, font: FONT, bold: true, size: sz(40) }),
                   new D.TextRun({ text: m ? m[2] : text, font: FONT, bold: true, size: sz(48), break: 1 })],
      })],
    };
  }
  return new D.Paragraph({
    heading: level === 2 ? D.HeadingLevel.HEADING_2 : D.HeadingLevel.HEADING_3,
    indent: level === 3 ? { left: INDENT } : undefined,
    children: [new D.TextRun({ text: text.replace(/^(\d+(?:\.\d+)+)\s+/, "$1  "), font: FONT, bold: true,
                               size: sz(level === 2 ? 36 : 32) })],
    spacing: { before: level === 2 ? 320 : 160, after: 120 },
    keepNext: true,
  });
}

const toOmml = require("./math").makeConverter({ size: 26, thaiFont: FONT, thaiSize: BODY });
const NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
  + 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"';
const oMath = (tex) => D.ImportedXmlComponent.fromXmlString(`<m:oMath ${NS}>${toOmml(tex)}</m:oMath>`);

// display equation written in LaTeX, centred, with its number at the right margin
const Eq = (tex, label) => new D.Paragraph({
  tabStops: [{ type: D.TabStopType.CENTER, position: Math.round(TEXT_W / 2) },
             { type: D.TabStopType.RIGHT, position: TEXT_W }],
  children: [new D.TextRun({ text: "\t", font: FONT, size: BODY }), oMath(tex),
             ...(label ? [new D.TextRun({ text: `\t(${label})`, font: FONT, size: BODY })] : [])],
  spacing: { before: 160, after: 160 },
});

const Bullets = (items, ref = "bullets") => items.map((t) => new D.Paragraph({
  children: runs(t),
  numbering: { reference: ref, level: 0 },
  spacing: { after: 60, line: 276 },
  alignment: D.AlignmentType.LEFT,
}));

const Numbered = (items, ref) => Bullets(items, ref);

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const MAX_FIG_H_CM = 9.5;   // taller figures leave half-empty pages when they do not fit under the text

function Fig(rel, caption, widthCm = 15) {
  S.fig += 1;
  S.figAll += 1;
  const num = `${S.chapter}.${S.fig}`;
  S.figMap[S.figAll] = num;
  const file = path.join(ROOT, rel);
  const { w, h } = pngSize(file);
  const wpx = Math.min(widthCm, 15, MAX_FIG_H_CM * w / h) / 2.54 * 96;
  const long = caption.length > 75;   // the guide: one-line captions centred, longer ones from the left margin
  return [
    new D.Paragraph({
      children: [new D.ImageRun({ type: "png", data: fs.readFileSync(file),
                                  transformation: { width: wpx, height: wpx * h / w } })],
      alignment: D.AlignmentType.CENTER, keepNext: true, spacing: { before: 240, after: 80 },
    }),
    new D.Paragraph({
      style: "FigureCaption",
      children: [new D.TextRun({ text: `รูปที่ ${num}  `, bold: true, font: FONT, size: BODY }),
                 ...runs(caption)],
      alignment: long ? D.AlignmentType.LEFT : D.AlignmentType.CENTER, spacing: { after: 240 },
    }),
  ];
}

const border = { style: D.BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };

function Table(caption, header, rows, widths, opt = {}) {
  S.tab += 1;
  S.tabAll += 1;
  const num = `${S.chapter}.${S.tab}`;
  S.tabMap[S.tabAll] = num;
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum > TEXT_W) widths = widths.map((x) => Math.floor(x * TEXT_W / sum));
  const total = widths.reduce((a, b) => a + b, 0);
  const size = sz(opt.size || 28);
  const cell = (t, i, head, last) => new D.TableCell({
    borders,
    width: { size: widths[i], type: D.WidthType.DXA },
    shading: head ? { fill: "EDEDED", type: D.ShadingType.CLEAR } : undefined,
    margins: { top: 40, bottom: 40, left: 90, right: 90 },
    children: [new D.Paragraph({
      children: runs(String(t), { size, bold: head || undefined }),
      alignment: i === 0 && !opt.centerFirst && !head ? D.AlignmentType.LEFT : D.AlignmentType.CENTER,
      keepNext: !last,       // keep the whole table on one page
    })],
  });
  return [
    new D.Paragraph({
      style: "TableCaption",
      children: [new D.TextRun({ text: `ตารางที่ ${num}  `, bold: true, font: FONT, size: BODY }),
                 ...runs(caption)],
      keepNext: true, spacing: { before: 240, after: 80 },
    }),
    new D.Table({
      width: { size: total, type: D.WidthType.DXA },
      columnWidths: widths,
      rows: [new D.TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
             ...rows.map((r, k) => new D.TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, k === rows.length - 1)) }))],
    }),
    new D.Paragraph({ children: [], spacing: { after: 160 } }),
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
      style: { paragraph: { indent: { left: INDENT + 360, hanging: 360 } } } }] },
    ...Array.from({ length: 12 }, (_, i) => ({ reference: `num${i}`, levels: [{ level: 0, format: D.LevelFormat.DECIMAL,
      text: "%1)", alignment: D.AlignmentType.LEFT, style: { paragraph: { indent: { left: INDENT + 360, hanging: 360 } } } }] })),
  ],
};

module.exports = { oMath, sz, D, P, H, Eq, Bullets, Numbered, Fig, Table, Code, FONT, BODY, ROOT, PAGE, TEXT_W, INDENT,
                   numberingConfig, resetNumbering, S };

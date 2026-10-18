// A small LaTeX subset -> Office Math (OMML) converter, so the report's equations are
// native Word equations (fractions, scripts, sums, matrices, brackets) instead of plain text.
// Supported: { } ^ _ ' \frac \sqrt \sum \prod \mathbf \mathrm \text \hat \bar \tilde
// \left( \right) \begin{bmatrix|pmatrix|vmatrix|Bmatrix|matrix}, Greek letters and common symbols.

const SYM = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", eta: "η", theta: "θ", lambda: "λ",
  mu: "μ", sigma: "σ", tau: "τ", phi: "φ", psi: "ψ", omega: "ω", pi: "π", rho: "ρ", xi: "ξ", kappa: "κ",
  Delta: "Δ", Sigma: "Σ", Lambda: "Λ", Phi: "Φ", Psi: "Ψ", Omega: "Ω", Gamma: "Γ",
  cdot: "·", times: "×", top: "⊤", leq: "≤", geq: "≥", le: "≤", ge: "≥", neq: "≠", approx: "≈", in: "∈",
  to: "→", rightarrow: "→", leftarrow: "←", Rightarrow: "⇒", gets: "←", odot: "⊙", otimes: "⊗", partial: "∂",
  nabla: "∇", infty: "∞", pm: "±", sim: "∼", ldots: "…", cdots: "⋯", vdots: "⋮", ddots: "⋱", lVert: "‖", rVert: "‖",
  "|": "‖", mid: "|", perp: "⊥", forall: "∀", setminus: "∖", circ: "∘", star: "⋆", prime: "′", lfloor: "⌊",
  rfloor: "⌋", lceil: "⌈", rceil: "⌉", langle: "⟨", rangle: "⟩", "{": "{", "}": "}", ",": " ", ";": " ",
  quad: "  ", qquad: "    ", " ": " ", "!": "", sum: "∑", prod: "∏",
};
const FUNCS = ["tanh", "sin", "cos", "tan", "ln", "log", "exp", "max", "min", "det", "diag", "arg", "argmin", "argmax",
  "rank", "tr", "sign", "clip", "round", "softmax", "arctan"];
const BB = { R: "ℝ", N: "ℕ", Z: "ℤ", E: "𝔼", 1: "𝟙" };
const CAL = { N: "𝒩", L: "ℒ", D: "𝒟" };

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function tokenize(src) {
  const out = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === "\\") {
      const m = src.slice(i + 1).match(/^([A-Za-z]+|.)/);
      out.push({ cmd: m[1] });
      i += 1 + m[1].length;
    } else if (/\s/.test(c)) {
      i++;
    } else if (/[0-9.]/.test(c)) {
      const m = src.slice(i).match(/^[0-9]+(\.[0-9]+)?|^\./);
      out.push({ ch: m[0], num: true });
      i += m[0].length;
    } else {
      out.push({ ch: c });
      i++;
    }
  }
  return out;
}

function makeConverter(opt) {
  const rpr = `<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="${opt.thaiFont}"/>`
    + `<w:sz w:val="${opt.size}"/><w:szCs w:val="${opt.thaiSize}"/></w:rPr>`;
  const run = (text, sty) => `<m:r>${sty ? `<m:rPr><m:sty m:val="${sty}"/></m:rPr>` : ""}${rpr}`
    + `<m:t xml:space="preserve">${esc(text)}</m:t></m:r>`;

  return function convert(src) {
    const tk = tokenize(src);
    let pos = 0;
    const peek = () => tk[pos];
    const next = () => tk[pos++];

    function readRawGroup() {          // text of {...} without parsing (for \text, \begin)
      const t = next();
      if (!t || t.ch !== "{") throw new Error(`expected { in ${src}`);
      let depth = 1, s = "";
      while (pos < tk.length) {
        const u = next();
        if (u.ch === "{") depth++;
        if (u.ch === "}" && --depth === 0) break;
        s += u.cmd ? (u.cmd.length === 1 ? u.cmd : `\\${u.cmd} `) : u.ch;
      }
      return s;
    }

    function styled(inner, sty) {      // re-run the group with another run style
      return makeConverter({ ...opt, sty })(inner).replace(/<m:r>(?!<m:rPr>)/g, `<m:r><m:rPr><m:sty m:val="${sty}"/></m:rPr>`);
    }

    function seq(stop) {
      let out = "";
      while (pos < tk.length) {
        const t = peek();
        if (stop && stop(t)) break;
        out += scripted();
      }
      return out;
    }

    function group() {                 // one atom used as an argument
      const t = peek();
      if (t && t.ch === "{") {
        next();
        const s = seq((u) => u.ch === "}");
        next();
        return s;
      }
      return atom();
    }

    function scripted() {
      let base = atom();
      let sub = null, sup = null;
      for (;;) {
        const t = peek();
        if (!t) break;
        if (t.ch === "^") { next(); sup = group(); }
        else if (t.ch === "_") { next(); sub = group(); }
        else if (t.ch === "'") { next(); sup = (sup || "") + run("′"); }
        else break;
      }
      if (sub && sup) return `<m:sSubSup><m:e>${base}</m:e><m:sub>${sub}</m:sub><m:sup>${sup}</m:sup></m:sSubSup>`;
      if (sub) return `<m:sSub><m:e>${base}</m:e><m:sub>${sub}</m:sub></m:sSub>`;
      if (sup) return `<m:sSup><m:e>${base}</m:e><m:sup>${sup}</m:sup></m:sSup>`;
      return base;
    }

    function delim(beg, end, inner) {
      return `<m:d><m:dPr><m:begChr m:val="${esc(beg)}"/><m:endChr m:val="${esc(end)}"/></m:dPr><m:e>${inner}</m:e></m:d>`;
    }

    function atom() {
      const t = next();
      if (!t) return "";
      if (t.ch === "{") { const s = seq((u) => u.ch === "}"); next(); return s; }
      if (t.ch !== undefined) {
        if (t.num) return run(t.ch);
        if (/[A-Za-z]/.test(t.ch)) return run(t.ch, opt.sty);
        return run(t.ch === "-" ? "−" : t.ch, "p");
      }
      const c = t.cmd;
      if (c === "frac") { const a = group(), b = group(); return `<m:f><m:num>${a}</m:num><m:den>${b}</m:den></m:f>`; }
      if (c === "sqrt") return `<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>${group()}</m:e></m:rad>`;
      if (c === "sum" || c === "prod") {
        let sub = "", sup = "";
        for (;;) {
          const u = peek();
          if (u && u.ch === "_") { next(); sub = group(); } else if (u && u.ch === "^") { next(); sup = group(); } else break;
        }
        const e = group();
        return `<m:nary><m:naryPr><m:chr m:val="${SYM[c]}"/><m:limLoc m:val="undOvr"/>${sub ? "" : '<m:subHide m:val="1"/>'}`
          + `${sup ? "" : '<m:supHide m:val="1"/>'}</m:naryPr><m:sub>${sub}</m:sub><m:sup>${sup}</m:sup><m:e>${e}</m:e></m:nary>`;
      }
      if (c === "mathbf" || c === "boldsymbol") return styled(readRawGroup(), "b");
      if (c === "mathrm" || c === "operatorname") return run(readRawGroup(), "p");
      if (c === "text") return run(readRawGroup(), "p");
      if (c === "mathcal") return run(readRawGroup().split("").map((x) => CAL[x] || x).join(""), "p");
      if (c === "mathbb") return run(readRawGroup().split("").map((x) => BB[x] || x).join(""), "p");
      if (c === "hat" || c === "bar" || c === "tilde" || c === "dot") {
        const chr = { hat: "̂", bar: "̅", tilde: "̃", dot: "̇" }[c];
        return `<m:acc><m:accPr><m:chr m:val="${chr}"/></m:accPr><m:e>${group()}</m:e></m:acc>`;
      }
      if (c === "left") {
        const open = next();
        const beg = open.ch || SYM[open.cmd] || "";
        const inner = seq((u) => u.cmd === "right");
        next();
        const close = next();
        return delim(beg === "." ? "" : beg, close.ch === "." ? "" : (close.ch || SYM[close.cmd] || ""), inner);
      }
      if (c === "begin") {
        const env = readRawGroup();
        const rows = [[""]];
        while (pos < tk.length) {
          const u = peek();
          if (u.cmd === "end") { next(); readRawGroup(); break; }
          if (u.ch === "&") { next(); rows[rows.length - 1].push(""); continue; }
          if (u.cmd === "\\") { next(); rows.push([""]); continue; }
          const cells = rows[rows.length - 1];
          cells[cells.length - 1] += scripted();
        }
        const ncol = Math.max(...rows.map((r) => r.length));
        const m = `<m:m><m:mPr><m:mcs><m:mc><m:mcPr><m:count m:val="${ncol}"/><m:mcJc m:val="center"/></m:mcPr></m:mc></m:mcs></m:mPr>`
          + rows.filter((r) => r.some((x) => x)).map((r) => `<m:mr>${r.map((x) => `<m:e>${x}</m:e>`).join("")}</m:mr>`).join("")
          + "</m:m>";
        const br = { bmatrix: ["[", "]"], pmatrix: ["(", ")"], vmatrix: ["|", "|"], Bmatrix: ["{", "}"], matrix: null }[env];
        return br ? delim(br[0], br[1], m) : m;
      }
      if (FUNCS.includes(c)) return run(c, "p");
      if (c in SYM) return run(SYM[c], "p");
      return run(c, "p");
    }

    return seq();
  };
}

module.exports = { makeConverter };

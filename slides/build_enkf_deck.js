// Builds slides/enkf_optimisation.pptx: how the EnKF was optimised.
// Numbers on the result slide are read from methods/enkf/check_outputs/original_vs_final/
// comparison.json, not typed in. Run from slides/:  node build_enkf_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const P = (...p) => path.join(ROOT, ...p);

// ---------- palette and type ----------
const INK = "1D2B36", MUTED = "5B6773", LIGHT = "F1F4F6", RULE = "D5DCE2";
const RED = "C0392B", TEAL = "1B7F72", AMBER = "B9770E", NAVY = "2E5A87", SLATE = "44546A";
const HEAD = "Cambria", BODY = "Calibri", MONO = "Courier New";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";          // 10 x 5.625 in
pres.title = "Optimising the EnKF";

// ---------- helpers ----------
function tag(slide, text, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 0.3, w: 0.16 + 0.095 * text.length, h: 0.27, rectRadius: 0.13,
    fill: { color }, line: { color } });
  slide.addText(text, { x: 0.5, y: 0.3, w: 0.16 + 0.095 * text.length, h: 0.27,
    fontFace: BODY, fontSize: 9, bold: true, color: "FFFFFF", align: "center",
    valign: "middle", charSpacing: 2, margin: 0, isTextBox: true });
}
function title(slide, text) {
  slide.addText(text, { x: 0.5, y: 0.62, w: 9.0, h: 0.62, fontFace: HEAD, fontSize: 22,
    bold: true, color: INK, valign: "top", margin: 0, isTextBox: true });
}
function foot(slide, text, n) {
  slide.addText(text, { x: 0.5, y: 5.2, w: 8.4, h: 0.3, fontFace: BODY, fontSize: 8.5,
    color: MUTED, italic: true, valign: "top", margin: 0, isTextBox: true });
  slide.addText(String(n), { x: 9.1, y: 5.2, w: 0.4, h: 0.3, fontFace: BODY, fontSize: 9,
    color: MUTED, align: "right", valign: "top", margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, fill = LIGHT) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { color: fill } });
}
function bullets(slide, items, opt) {
  const runs = items.map((t, i) => {
    const base = { bullet: { indent: 12 }, breakLine: i < items.length - 1, paraSpaceAfter: 5 };
    if (Array.isArray(t)) return t.map((r, j) => ({ text: r.text,
      options: { ...base, bullet: j === 0 ? base.bullet : false, breakLine: j === t.length - 1 && base.breakLine, bold: !!r.bold, color: r.color || INK } }));
    return [{ text: t, options: base }];
  }).flat();
  slide.addText(runs, { fontFace: BODY, fontSize: 12, color: INK, valign: "top", margin: 0,
    isTextBox: true, ...opt });
}
function numCircle(slide, x, y, n, color) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.36, h: 0.36, fill: { color }, line: { color } });
  slide.addText(String(n), { x, y, w: 0.36, h: 0.36, fontFace: BODY, fontSize: 13, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function table(slide, rows, opt, highlightRow = -1, hiColor = "E3F1EE", leftAll = false) {
  const t = rows.map((r, i) => r.map((c, j) => ({
    text: String(c),
    options: {
      bold: i === 0 || (i === highlightRow && j === 0), fontFace: BODY,
      fontSize: i === 0 ? 10.5 : 11.5, color: i === 0 ? "FFFFFF" : INK,
      fill: { color: i === 0 ? SLATE : (i === highlightRow ? hiColor : (i % 2 ? "FFFFFF" : "F7F9FA")) },
      align: (j === 0 || leftAll) ? "left" : "center", valign: "middle",
    } })));
  slide.addTable(t, { border: { type: "solid", pt: 0.5, color: RULE }, margin: 0.05, ...opt });
}
function readCsv(file) {
  const [head, ...lines] = fs.readFileSync(file, "utf8").trim().split("\n");
  const keys = head.split(",");
  return lines.map((l) => { const v = l.split(","); return Object.fromEntries(keys.map((k, i) => [k, v[i]])); });
}
const fmt = (v, d = 3) => (v === undefined || v === null || Number.isNaN(v)) ? "pending" : Number(v).toFixed(d);
// "x_i" / "x_{ab}" -> subscript runs, for equations
function math(str, options = {}) {
  return str.split(/(_\{[^}]*\}|_\w)/).filter((p) => p).map((p) => p.startsWith("_")
    ? { text: p.replace(/^_\{?|\}$/g, ""), options: { ...options, subscript: true } }
    : { text: p, options });
}
const pct = (v, d = 1) => (v === undefined || v === null) ? "pending" : (100 * v).toFixed(d) + "%";

// ---------- data read from result files ----------
const cmpFile = P("methods/enkf/check_outputs/original_vs_final/comparison.json");
const cmp = fs.existsSync(cmpFile) ? JSON.parse(fs.readFileSync(cmpFile, "utf8")) : null;
function msPerFrame(dir) {
  let s = 0, n = 0;
  for (const f of (fs.existsSync(dir) ? fs.readdirSync(dir) : []).filter((f) => f.endsWith(".json"))) {
    const d = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
    if (d.seconds && d.frames_used) { s += d.seconds; n += d.frames_used; }
  }
  return n ? 1000 * s / n : null;
}
const msOrig = msPerFrame(P("methods/enkf/check_outputs/original_vs_final/runs"));
const msFinal = msPerFrame(P("supervisor_evaluation/outputs/full/raw/enkf"));
const acc = readCsv(P("supervisor_evaluation/outputs/full/accuracy.csv"));
const unc = readCsv(P("supervisor_evaluation/outputs/full/uncertainty.csv"));
const byCh = readCsv(P("supervisor_evaluation/outputs/full/uncertainty_by_channel.csv"));
const SHORT = { "Senseiver-A": "Senseiver-A", "Senseiver-G Temporal (ours)": "Senseiver-G (ours)",
  "DINCAE": "DINCAE", "4DVarNet MSE": "4DVarNet", "4DVarNet aughead_obs (single)": "4DVarNet (aug. head)",
  "Localized EnKF (100 members)": "EnKF" };

let n = 0;
const note = (s, text, y, color = INK, opt = {}) => s.addText(text, { x: 0.5, y, w: 9.0, h: 0.4,
  fontFace: BODY, fontSize: 12, color, valign: "top", margin: 0, isTextBox: true, ...opt });

// =====================================================================
// 1. The problem
// =====================================================================
{
  const s = pres.addSlide(); n++;
  tag(s, "PROBLEM", RED);
  title(s, "The problem: the EnKF's uncertainty had collapsed");
  card(s, 0.5, 1.45, 4.1, 3.3);
  s.addText("Ensemble spread (σ̂)", { x: 0.75, y: 1.65, w: 3.6, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, isTextBox: true });
  s.addText("0.0025", { x: 0.75, y: 1.95, w: 3.6, h: 0.7, fontFace: HEAD, fontSize: 36, bold: true, color: RED, margin: 0, isTextBox: true });
  s.addText("Actual error (RMSE)", { x: 0.75, y: 2.8, w: 3.6, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, isTextBox: true });
  s.addText("0.24", { x: 0.75, y: 3.1, w: 3.6, h: 0.7, fontFace: HEAD, fontSize: 36, bold: true, color: INK, margin: 0, isTextBox: true });
  bullets(s, [
    [{ text: "σ̂ is the spread of the 100 members. ", bold: true }, { text: "It was about 1% of the real error: the filter was a hundred times too confident." }],
    [{ text: "So the robots were ignored. ", bold: true }, { text: "The Kalman update moved the estimate only 0.2% of the way towards the observations." }],
  ], { x: 4.95, y: 1.6, w: 4.55, h: 3.0, fontSize: 14, paraSpaceAfter: 14 });
  foot(s, "Original configuration, test days.", n);
  s.addNotes("The EnKF's uncertainty is the spread of its 100 members. In the original filter the members almost agreed, so the spread was about 1% of the real error. Because the Kalman update weighs forecast against observations by their uncertainty, a filter that thinks its forecast is perfect ignores the robots: the update moved the estimate by only 0.2% of the gap.");
}

// =====================================================================
// 2. The cause
// =====================================================================
{
  const s = pres.addSlide(); n++;
  tag(s, "CAUSE", AMBER);
  title(s, "The cause: only 1% of the process noise was added");
  card(s, 0.5, 1.45, 4.3, 0.78, "1F2A33");
  s.addText([{ text: "X_f += rng.normal(0, ", options: { color: "E6EDF3" } },
             { text: "0.01", options: { color: "FF8A80", bold: true } },
             { text: " * proc_noise_vec)", options: { color: "E6EDF3" } }],
    { x: 0.68, y: 1.55, w: 4.05, h: 0.32, fontFace: MONO, fontSize: 10.5, margin: 0, isTextBox: true });
  s.addText("original forecast step", { x: 0.68, y: 1.9, w: 4.0, h: 0.25,
    fontFace: BODY, fontSize: 9, italic: true, color: "AAB6C1", margin: 0, isTextBox: true });
  s.addText("The forecast model pulls the members together every step; the spread settles at the size of the noise that is added back. At 0.01 × that is 1% of the error.",
    { x: 0.5, y: 2.5, w: 4.3, h: 1.4, fontFace: BODY, fontSize: 13, color: INK, valign: "top", margin: 0, isTextBox: true });
  table(s, [
    ["Noise multiplier", "Spread / RMSE", "CRPS ↓"],
    ["0.01 (original)", "0.013", "0.122"],
    ["1 (full strength)", "0.956", "0.097"],
  ], { x: 5.1, y: 1.45, w: 4.4, colW: [1.8, 1.3, 1.3], rowH: 0.42 }, 2);
  s.addText("Changing only this constant already fixes the size of the uncertainty. But with independent noise per cell, σ̂ was nearly flat in space and could not show where the errors are.",
    { x: 5.1, y: 2.95, w: 4.4, h: 1.6, fontFace: BODY, fontSize: 12, color: INK, valign: "top", margin: 0, isTextBox: true });
  foot(s, "Original filter, only the multiplier changed; one test day.", n);
  s.addNotes("The original code multiplied the process noise by 0.01. The forecast model damps the differences between members, so the spread settles at the size of the noise added each step: 1% of the error. Changing only that constant brings spread/RMSE to 0.96 and CRPS down by 20%. But the noise was independent per cell with one size everywhere, so the uncertainty was flat in space and did not point at the risky cells, especially for velocity. That is what the fix addresses.");
}

// =====================================================================
// 3. The fix: a bank of real forecast errors
// =====================================================================
{
  const s = pres.addSlide(); n++;
  tag(s, "FIX", TEAL);
  title(s, "The fix: add real forecast errors as the noise");
  const steps = [
    ["Record the model's one-step errors", "r = x(t+1) − f( x(t−4), …, x(t) )", "32 training days → 8,192 whole error maps, all four channels"],
    ["Remove their average", "r_k ← r_k − r̄", "keeps only the random part; the bias correction handles the systematic part"],
    ["Each step, each member adds one map", "drawn at random from the bank", ""],
  ];
  steps.forEach(([head, eq, sub], i) => {
    const y = 1.45 + i * 1.05;
    numCircle(s, 0.5, y + 0.02, i + 1, TEAL);
    s.addText(head, { x: 1.0, y, w: 4.6, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(math(eq), { x: 1.0, y: y + 0.33, w: 4.6, h: 0.3, fontFace: HEAD, fontSize: 13, italic: true, color: TEAL, margin: 0, isTextBox: true });
    if (sub) s.addText(sub, { x: 1.0, y: y + 0.64, w: 4.6, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, margin: 0, isTextBox: true });
  });
  card(s, 5.9, 1.45, 3.6, 1.35, LIGHT);
  s.addText("A real error map is large on the walkways, zero at the walls, and density and velocity err together. The ensemble spread inherits that shape.",
    { x: 6.05, y: 1.58, w: 3.3, h: 1.5, fontFace: BODY, fontSize: 12, color: INK, valign: "top", margin: 0, isTextBox: true });
  card(s, 5.9, 2.95, 3.6, 1.1, "E3F1EE");
  s.addText("−17.6%", { x: 5.95, y: 3.01, w: 3.5, h: 0.55, fontFace: HEAD, fontSize: 24, bold: true, color: TEAL, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText("CRPS vs. Gaussian noise of the same size", { x: 5.95, y: 3.57, w: 3.5, h: 0.35, fontFace: BODY, fontSize: 10.5, color: INK, align: "center", margin: 0, isTextBox: true });
  foot(s, "Seven test days; better on all seven.", n);
  s.addNotes("Instead of inventing noise, we use the forecast model's real mistakes. On training days we feed in the true state, predict one step, and subtract the prediction from the true next frame. 8,192 such error maps make the bank. We subtract their average, so only the random part remains; the systematic part is the bias correction's job. In filtering each member adds one map drawn at random. Real error maps have the right shape, so the spread does too. Against Gaussian noise of the same size, CRPS drops 17.6%.");
}

// =====================================================================
// 4. The noise step
// =====================================================================
{
  const s = pres.addSlide(); n++;
  tag(s, "FIX", TEAL);
  title(s, "The noise step, for each member i");
  const eqs = [
    "ε_i  ~  bank",
    "η_i(t) = ρ · η_i(t−1) + √(1−ρ²) · ε_i",
    "q_i = 1.5 · b ⊙ η_i",
    "q_i ← q_i − mean_j q_j",
    "x_i ← f(x_i) − bias + q_i",
  ];
  card(s, 0.5, 1.45, 4.55, 3.55, "1F2A33");
  eqs.forEach((e, i) => {
    const y = 1.66 + i * 0.66;
    s.addText(String(i + 1), { x: 0.68, y, w: 0.3, h: 0.36, fontFace: BODY, fontSize: 11, bold: true, color: "7FB8AE", valign: "middle", margin: 0, isTextBox: true });
    s.addText(math(e), { x: 1.0, y, w: 3.95, h: 0.36, fontFace: HEAD, fontSize: 14, italic: true, color: "E6EDF3", valign: "middle", margin: 0, isTextBox: true });
  });
  const symOpt = { fontFace: HEAD, italic: true, bold: true, fontSize: 11.5, color: TEAL };
  const legend = [
    ["ε_i", "a random error map from the bank"],
    ["η_i", "member i's noise, with memory over time"],
    ["ρ = 0.5", "share of last frame's noise that is kept"],
    ["√(1−ρ²)", "keeps the noise size unchanged"],
    ["q_i", "the noise actually added to member i"],
    ["1.5", "overall size factor"],
    ["b ⊙", "per-cell factor: 1 where a robot sees; on unseen cells 1, 1.25, 1.4, 0.93 for density, vx, vy, variance"],
    ["mean_j", "average over the 100 members, so the noise spreads them without shifting them"],
    ["f, bias", "forecast model; learned systematic error"],
  ].map(([k, v]) => [
    { text: math(k, symOpt), options: { valign: "middle" } },
    { text: v, options: { fontFace: BODY, fontSize: 10, color: INK, valign: "middle" } }]);
  s.addTable(legend, { x: 5.3, y: 1.45, w: 4.2, colW: [0.85, 3.35], margin: 0.04,
    border: { type: "solid", pt: 0.5, color: RULE } });
  foot(s, "Bank used at the model's own measured error size. ρ, 1.5 and b chosen on validation days, checked once on test days.", n);
  s.addNotes("This is the whole noise step. Draw one error map. Give it memory over time: half of last step's noise carries over, and the square-root factor keeps the size unchanged. Scale by 1.5, and more on cells no robot currently sees; this uses the robots' positions, not the truth. Subtract the mean over the members, so the noise spreads the ensemble without moving its mean. Add it to the bias-corrected forecast. Density is perturbed in log(1+density) so it stays non-negative. One last detail: the bank is used at its own measured size. An earlier version rescaled it to a size borrowed from the original project, which for velocity variance was several times too large and, after clipping at zero, pushed the variance estimate up.");
}

// =====================================================================
// 5. What changed
// =====================================================================
{
  const s = pres.addSlide(); n++;
  tag(s, "SUMMARY", NAVY);
  title(s, "What changed from the original filter");
  const rows = [
    ["", "Original", "Final"],
    ["Forecast model", "original model, fed 1 frame", "our retrained model, 5 frames"],
    ["Process noise", "independent per cell, ×0.01", "real error maps, measured size ×1.5"],
    ["", "", "persist over time, more on unobserved cells"],
    ["Kalman update", "radius 7, inflation 1.02", "same + cross-channel weights"],
    ["Bias correction", "yes", "same"],
  ];
  const t = rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
    bold: i === 0 || j === 0, fontFace: BODY, fontSize: i === 0 ? 11 : 12.5,
    color: i === 0 ? "FFFFFF" : (j === 2 && i >= 1 && i <= 4 ? TEAL : INK),
    fill: { color: i === 0 ? SLATE : "FFFFFF" }, align: "left", valign: "middle" } })));
  s.addTable(t, { x: 0.5, y: 1.45, w: 9.0, colW: [2.2, 3.1, 3.7], rowH: 0.45, margin: 0.08,
    border: { type: "solid", pt: 0.5, color: RULE } });
  note(s, "Teal: changed. Everything else is identical, so the results on the next slide come from these changes.", 4.45, MUTED, { italic: true, fontSize: 11 });
  foot(s, "", n);
  s.addNotes("What changed: the forecast model, now our retrained one seeing five frames; the process noise; and cross-channel weights in the Kalman update. Localisation radius, inflation and the bias correction are the original ones. Both filters are run through the same GPU code, which also corrects the original localisation so velocity observations are assimilated; that correction applies to both, so the comparison on the next slide isolates these changes.");
}

// =====================================================================
// 6. Result
// =====================================================================
{
  const s = pres.addSlide(); n++;
  tag(s, "RESULT", NAVY);
  title(s, "Result: original vs. final, scored identically");
  const o = cmp ? cmp.original.all : {}, f = cmp ? cmp.final.all : {};
  table(s, [
    ["Unobserved cells, 4 channels", "Original", "Final", "Ideal"],
    ["RMSE (lower is better)", fmt(o.rmse), fmt(f.rmse), "—"],
    ["Spread / RMSE", fmt(o.spread_rmse), fmt(f.spread_rmse), "1"],
    ["CRPS skill (higher is better)", fmt(o.crps_skill), fmt(f.crps_skill), "> 0"],
  ], { x: 0.5, y: 1.45, w: 9.0, colW: [3.6, 1.8, 1.8, 1.8], rowH: 0.48 }, 2, "E8EEF5");
  const d = (a, b) => Number((100 * (b - a) / a).toFixed(0));
  bullets(s, [
    [{ text: "Uncertainty: unusable → usable. ", bold: true, color: TEAL }, { text: `Spread from 1% of the error to ${fmt(f.spread_rmse, 2)}×.` }],
    [{ text: "Accuracy kept. ", bold: true, color: TEAL }, { text: `RMSE ${f.rmse <= o.rmse ? "slightly lower" : "slightly higher"}; velocity variance is still ${cmp ? d(cmp.original.variance.rmse, cmp.final.variance.rmse) : "?"}% worse than the original.` }],
  ], { x: 0.5, y: 3.65, w: 9.0, h: 1.2, fontSize: 13 });
  foot(s, "Seven test days, unobserved walkable cells, four channels; same code, observations and scoring for both.", n);
  s.addNotes("Both filters run through the same code on the same observations and are scored exactly as in the six-method comparison. The uncertainty goes from unusable to usable: spread from 1% of the error to about 1.0 times it, and CRPS skill from negative, worse than a constant sigma, to positive. Pooled RMSE is slightly lower than the original; velocity variance is still somewhat worse.");
}

// =====================================================================
// 7. The final filter, step by step
// =====================================================================
{
  const s = pres.addSlide(); n++;
  tag(s, "SUMMARY", NAVY);
  title(s, "The final filter, every frame");
  const steps = [
    ["Forecast", "Each of the 100 members runs PedPred3 on its last 5 frames; bias subtracted.", [["retrained model", TEAL]]],
    ["Noise", "Each member adds one real error map (previous slides).", [["new", TEAL]]],
    ["Kalman update", "Members pulled towards the robots' observations, localised.", [["cross-channel weights", TEAL], ["radius, inflation", SLATE]]],
    ["Bias update", "Slow running estimate of the systematic error.", [["original", SLATE]]],
    ["Output", "Mean = reconstruction; spread = σ̂.", []],
  ];
  const w = 1.66, gap = 0.18, y = 1.45, h = 2.7;
  steps.forEach(([head, body, tags], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, y, w, h, LIGHT);
    numCircle(s, x + 0.12, y + 0.14, i + 1, NAVY);
    s.addText(head, { x: x + 0.56, y: y + 0.12, w: w - 0.62, h: 0.42, fontFace: BODY, fontSize: 12.5, bold: true, color: INK, valign: "middle", margin: 0, isTextBox: true });
    s.addText(body, { x: x + 0.12, y: y + 0.66, w: w - 0.24, h: 1.4, fontFace: BODY, fontSize: 11, color: INK, valign: "top", margin: 0, isTextBox: true });
    tags.forEach(([t, c], j) => {
      const ty = y + h - 0.42 - (tags.length - 1 - j) * 0.36;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.12, y: ty, w: w - 0.24, h: 0.28, rectRadius: 0.1, fill: { color: c }, line: { color: c } });
      s.addText(t, { x: x + 0.12, y: ty, w: w - 0.24, h: 0.28, fontFace: BODY, fontSize: 9, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
    });
    if (i < steps.length - 1) s.addText("→", { x: x + w - 0.02, y: y + h / 2 - 0.2, w: gap + 0.04, h: 0.4, fontFace: BODY, fontSize: 16, bold: true, color: MUTED, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  note(s, "↻  step 5 feeds step 1 of the next frame (one second later)", 4.35, MUTED, { italic: true, fontSize: 11 });
  foot(s, "", n);
  s.addNotes("Every second the filter forecasts each member with the retrained model, adds a real error map as noise, pulls the members towards the robots' observations, updates the slow bias estimate, and outputs the mean as the reconstruction and the spread as the uncertainty. The forecast model, the noise and the cross-channel weights were changed; the localisation radius, inflation and the bias correction are the original ones. We checked the bias correction: switching it off makes RMSE 25% worse.");
}

const out = path.join(__dirname, "enkf_optimisation.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("[deck] " + out + (cmp ? "" : "  (slide 8 pending: comparison.json not found)")));

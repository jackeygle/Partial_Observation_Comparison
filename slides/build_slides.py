"""
build_slides.py — one CONTENT definition, two renderers: .pptx and .pdf

Why a script rather than hand-assembly: the figures get regenerated and the numbers move when a
run is repeated. Everything here points at products under check_outputs, so after a rerun this
is executed again and the deck is current.

Two renderers, one source:
  * .pptx via python-pptx  — editable text boxes and real tables, what you actually present from
  * .pdf  via matplotlib   — a fixed rendering that can be looked at without PowerPoint, which is
                             also the only way to check the layout on a machine with no Office

Slide text is English. That is not a style choice: Triton has no CJK font at all
(`fc-list :lang=zh` is empty), so the pdf path would render Chinese as tofu boxes, and the two
renderers would then disagree about what the deck says.

    python3 -m slides.build_slides                 # both formats, every deck
    python3 -m slides.build_slides --only pptx
"""
from __future__ import annotations
import argparse
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "methods/enkf/check_outputs/eval/figs")

W_IN, H_IN = 13.333, 7.5                 # 16:9
INK, MUTED, ACCENT = "#1A1A1A", "#5A5A5A", "#2E7D6E"

#: A slide = title + optional image + optional bullets/table/after + optional footnote.
#: With an image the text goes to the right of it, otherwise it spans the page.
DECKS = {
    "enkf_uncertainty": {
        "title": "The EnKF as shipped: what the analysis found",
        "subtitle": "two separate problems  —  the Kalman update barely acts on the forecast, "
                    "and sigma carries no information",
        "slides": [
            dict(
                title="Overview: the EnKF's numbers",
                # Same numbers and cells as the README (7 test days).
                table=[["EnKF, 7 test days", "blind walkable cells", "all walkable cells", "ideal"],
                       ["RMSE", "0.263", "0.275", "-"],
                       ["spread / skill  (mean sigma / RMSE)", "0.010", "0.009", "1.0"],
                       ["coverage of the nominal 90% interval", "1.6%", "1.3%", "90%"]],
                after=[
                    "For reference, the three learned methods score RMSE 0.222-0.230 on blind walkable cells.",
                    "",
                    "Two separate problems behind these numbers:",
                    "  A. point estimate - the Kalman update barely acts on the forecast",
                    "  B. uncertainty - sigma carries no information",
                    "",
                    "Everything that follows describes the configuration as shipped.",
                ],
            ),
            dict(
                title="A1. The Kalman gain ignores vx, vy and var: a localisation bug",
                bullets=[
                    "update() is called without the observed cell positions - by the original project's own",
                    "evaluation code as well as by ours - so it rebuilds them from the observation matrix:",
                    "",
                    "    observed_cells = [(i // W, i % W) for i in C.nonzero()[1]]",
                    "",
                    "The state stacks the four channels one block after another. For a vx, vy or var observation",
                    "the column index i is offset by 1, 2 or 3 x H*W, so i // W is a row 36, 72 or 108 rows below",
                    "the grid. Its distance to its own cell is at least 36 > localisation radius 7: weight 0.",
                ],
                table=[["channel", "density", "vx", "vy", "var"],
                       ["mean self-gain", "9.5e-5", "0", "0", "0"]],
                after=["vx, vy and var observations never reach the state through the Kalman gain;",
                       "they still enter through the bias correction (A3)."],
                footnote="gain census: day atc-20130811, 400 frames, ensemble 100, radius 7, as shipped. Code: "
                         "enkf_lab/pedpred/ENKF.py, LocalizedEnsembleKalmanFilter.update(); the original's own "
                         "calls are enkf.step(C_joint, obs_vector, model=model)",
            ),
            dict(
                title="A2. Even for density the gain is ~1e-4",
                bullets=[
                    "K = P_xy @ pinv(P_yy),    P_yy = ensemble term + R (observation noise) + 1e-3 (hard-coded)",
                    "",
                    "The ensemble term is a tiny share of that denominator:",
                ],
                table=[["channel", "ensemble term", "R", "regularisation", "ensemble share"],
                       ["density", "4.3e-7", "3.2e-3", "1e-3", "1.0e-4"],
                       ["vx", "1.3e-5", "9.9e-2", "1e-3", "1.3e-4"],
                       ["vy", "2.8e-6", "7.4e-3", "1e-3", "3.3e-4"],
                       ["var", "1.8e-5", "4.2e-5", "1e-3", "1.7e-2"]],
                after=[
                    "The ensemble has collapsed (part B), so the filter trusts its forecast and all but",
                    "ignores the observations. For var the hard-coded 1e-3 is larger than the observation",
                    "noise itself. (vx, vy, var are listed for the denominator; their gain is already 0, A1.)",
                ],
                footnote="gain census: day atc-20130811, 400 frames, ensemble 100, radius 7, as shipped; "
                         "each term averaged over analysis steps",
            ),
            dict(
                title="A3. What does reduce the error: an undocumented bias correction",
                bullets=[
                    "Besides the Kalman gain, forecast() subtracts a moving average of (ensemble mean - observation),",
                    "at full strength. Not in the paper; the code comment reads \"estimatin NN model bias\".",
                    "Switching the two correction paths on and off separately:",
                ],
                table=[["arm", "Kalman gain", "bias correction", "RMSE"],
                       ["A", "off", "off", "0.255"],
                       ["B", "off", "on", "0.175"],
                       ["C", "on", "off", "0.255"],
                       ["D  (as shipped)", "on", "on", "0.175"]],
                after=[
                    "C = A and D = B: the Kalman gain adds nothing; the ~32% error reduction is the bias correction.",
                    "Per channel, B vs A: density -62%, vx -6%, vy -21%, var -48% - it corrects all four channels.",
                    "Its size is ~1100x the Kalman increment.",
                ],
                footnote="arms: day atc-20130811, frames 500-2000, blind cells of the full grid, ensemble 100, radius 7, "
                         "paired (one random stream); per-channel line: all cells. Bias / increment ratio: gain census, 400 frames",
            ),
            dict(
                title="B1. Symptom: sigma never moves",
                image="A_symptom.png",
                bullets=[
                    "grey  = actual density error, 0.03-0.22 all day",
                    "green = the sigma the EnKF reports: flat on the axis",
                    "",
                    "It does not respond to anything, and it is",
                    "about 100x too small.",
                    "",
                    "All four channels, spread / skill over 7 days:",
                    "  density 0.004  vx 0.010  vy 0.010  var 0.022",
                    "",
                    "sigma = spread of the 100 members at a cell. It",
                    "should be small where a robot just looked, large",
                    "where nobody has looked for a while.",
                ],
                footnote="figure: density, day atc-20130811. Per-channel spread / skill: 7 test days, all grid cells",
            ),
            dict(
                title="B2. Cause: the ensemble collapses within ~5 frames",
                image="B_cause.png",
                bullets=[
                    "y-axis = sigma / noise injected per step",
                    "",
                    "Each step does two things:",
                    "  the forecast model damps member differences by 65%",
                    "  a little noise is added, np.full(H*W, s)",
                    "     - one number for the entire grid",
                    "",
                    "After a few frames the ensemble holds only the",
                    "noise just added; everything else is gone.",
                    "",
                    "The injection is a hard-coded 0.01 x Q.",
                    "Mean sigma / injected noise, 7 days:",
                    "  density 2.0  vx 1.2  vy 1.3  var 1.0",
                ],
                footnote="damping: methods/varnet/check_outputs/eval/enkf_spread_growth.json (verdict 'contractive')",
            ),
            dict(
                title="Summary: two separate problems",
                bullets=[
                    "A. The Kalman update barely acts on the forecast",
                    "    A1  vx, vy and var observations get zero Kalman weight - a localisation bug",
                    "    A2  the density gain is ~1e-4: the collapsed ensemble trusts its forecast",
                    "    A3  the error reduction the filter does achieve is an undocumented bias correction",
                    "",
                    "B. sigma carries no information",
                    "    B1  flat all day and ~100x too small, in all four channels",
                    "    B2  the ensemble collapses within ~5 frames onto the injected noise",
                    "",
                    "Link: the collapse in B is also why the gain in A2 is so small.",
                    "Open: whether any of this was intended by the original authors cannot be told from the code.",
                ],
            ),
        ],
    },
}


# ---------------------------------------------------------------- pptx ----
def build_pptx(spec, path):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Emu, Inches, Pt
    from PIL import Image

    rgb = lambda h: RGBColor.from_string(h.lstrip("#"))

    def put(shapes, lines, x, y, w, size, color):
        tb = shapes.add_textbox(x, y, w, Inches(0.36 * len(lines) + 0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(5)
            p.level = min((len(line) - len(line.lstrip(" "))) // 2, 4)
            r = p.add_run()
            r.text = line.strip()
            r.font.size, r.font.color.rgb = Pt(size), rgb(color)
        return Inches(0.38 * len(lines) + 0.15)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W_IN), Inches(H_IN)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    put(s.shapes, [spec["title"]], Inches(1.0), Inches(2.6), Inches(W_IN - 2), 40, INK)
    put(s.shapes, [spec["subtitle"]], Inches(1.0), Inches(3.9), Inches(W_IN - 2), 16, MUTED)
    bar = s.shapes.add_textbox(Inches(1.0), Inches(3.68), Inches(3.2), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(ACCENT); bar.line.fill.background()

    for sl in spec["slides"]:
        s = prs.slides.add_slide(blank)
        put(s.shapes, [sl["title"]], Inches(0.7), Inches(0.45), Inches(W_IN - 1.4), 26, INK)

        img = sl.get("image")
        has_img = bool(img) and os.path.exists(os.path.join(FIGS, img))
        if img and not has_img:
            print(f"  ! missing figure {img}; slide falls back to text only")
        if has_img:
            box_w, box_h = Inches(7.2), Inches(4.9)
            iw, ih = Image.open(os.path.join(FIGS, img)).size
            k = min(box_w / iw, box_h / ih)
            s.shapes.add_picture(os.path.join(FIGS, img), Inches(0.55),
                                 Inches(1.5) + Emu(int((box_h - ih * k) / 2)),
                                 width=Emu(int(iw * k)), height=Emu(int(ih * k)))
            tx, tw = Inches(8.1), Inches(W_IN - 8.8)
        else:
            tx, tw = Inches(1.1), Inches(W_IN - 2.2)

        y, fs = Inches(1.5), 14 if has_img else 17
        if sl.get("bullets"):
            y += put(s.shapes, sl["bullets"], tx, y, tw, fs, INK)
        if sl.get("table"):
            rows = sl["table"]
            h = Inches(0.4 * len(rows))
            t = s.shapes.add_table(len(rows), len(rows[0]), tx, y, tw, h).table
            for ri, row in enumerate(rows):
                for ci, val in enumerate(row):
                    t.cell(ri, ci).text = val
                    p = t.cell(ri, ci).text_frame.paragraphs[0]
                    if p.runs:
                        p.runs[0].font.size = Pt(fs - 1)
                        p.runs[0].font.bold = (ri == 0) or (ci == 0)
                        p.runs[0].font.color.rgb = rgb(INK)
            y += h + Inches(0.28)
        if sl.get("after"):
            put(s.shapes, sl["after"], tx, y, tw, fs, INK)
        if sl.get("footnote"):
            put(s.shapes, [sl["footnote"]], Inches(0.7), Inches(H_IN - 0.7),
                Inches(W_IN - 1.4), 10, MUTED)

    prs.save(path)
    print(f"[pptx] {path}")


# ----------------------------------------------------------------- pdf ----
def build_pdf(spec, path, png_dir=None):
    """Same content through matplotlib. Exists so the layout can be inspected without Office,
    and so there is a handout that opens anywhere.

    `png_dir` additionally writes one PNG per page. That is the only way to actually look at the
    layout on this machine -- there is no Office and no poppler, so neither the pptx nor the pdf
    can be rendered here."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    n = [0]

    def page(pdf, draw):
        fig = plt.figure(figsize=(W_IN, H_IN))
        draw(fig)
        pdf.savefig(fig)
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
            fig.savefig(os.path.join(png_dir, f"p{n[0]}.png"), dpi=110)
        n[0] += 1
        plt.close(fig)

    with PdfPages(path) as pdf:
        def cover(fig):
            fig.text(0.075, 0.55, spec["title"], fontsize=34, color=INK, va="bottom")
            fig.add_artist(plt.Line2D([0.075, 0.32], [0.515, 0.515], lw=3, color=ACCENT))
            fig.text(0.075, 0.44, "\n".join(textwrap.wrap(spec["subtitle"], 78)),
                     fontsize=13, color=MUTED, va="top")
        page(pdf, cover)

        for sl in spec["slides"]:
            def draw(fig, sl=sl):
                fig.text(0.055, 0.915, sl["title"], fontsize=22, color=INK, va="center")
                img = sl.get("image")
                has_img = bool(img) and os.path.exists(os.path.join(FIGS, img))
                if has_img:
                    ax = fig.add_axes([0.04, 0.13, 0.53, 0.70])
                    ax.imshow(mpimg.imread(os.path.join(FIGS, img)))
                    ax.axis("off")
                    tx, wrap, fs = 0.60, 52, 11.5
                else:
                    tx, wrap, fs = 0.075, 96, 14
                y = 0.80
                for line in sl.get("bullets", []):
                    ind = (len(line) - len(line.lstrip(" "))) * 0.004
                    for seg in (textwrap.wrap(line.strip(), wrap) or [""]):
                        fig.text(tx + ind, y, seg, fontsize=fs, color=INK, va="top")
                        y -= 0.042
                if sl.get("table"):
                    rows = sl["table"]
                    widths = [max(len(r[c]) for r in rows) + 3 for c in range(len(rows[0]))]
                    y -= 0.02
                    for ri, row in enumerate(rows):
                        line = "".join(v.ljust(widths[ci]) for ci, v in enumerate(row))
                        fig.text(tx, y, line, fontsize=fs - 0.5, color=INK, va="top",
                                 family="monospace",
                                 fontweight="bold" if ri == 0 else "normal")
                        y -= 0.045
                        if ri == 0:
                            # rule goes BELOW the header row -- drawing it before advancing y
                            # put it straight through the text (va="top" means the glyphs hang
                            # below y, not above it).
                            fig.add_artist(plt.Line2D([tx, tx + 0.0082 * sum(widths)],
                                                      [y + 0.008, y + 0.008],
                                                      lw=0.8, color="#BBBBBB"))
                            y -= 0.008
                    y -= 0.02
                for line in sl.get("after", []):
                    for seg in (textwrap.wrap(line.strip(), wrap) or [""]):
                        fig.text(tx, y, seg, fontsize=fs, color=INK, va="top")
                        y -= 0.042
                if sl.get("footnote"):
                    fig.text(0.055, 0.045, "\n".join(textwrap.wrap(sl["footnote"], 120)),
                             fontsize=9, color=MUTED, va="bottom")
            page(pdf, draw)
    print(f"[pdf]  {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="")
    ap.add_argument("--only", choices=["pptx", "pdf"], default="")
    ap.add_argument("--outdir", default=os.path.join(HERE, "out"))
    ap.add_argument("--png", action="store_true",
                    help="also write one PNG per page, for checking the layout")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for name in ([a.deck] if a.deck else list(DECKS)):
        spec = DECKS[name]
        if a.only != "pdf":
            build_pptx(spec, os.path.join(a.outdir, f"{name}.pptx"))
        if a.only != "pptx":
            build_pdf(spec, os.path.join(a.outdir, f"{name}.pdf"),
                      png_dir=os.path.join(a.outdir, f"{name}_png") if a.png else None)


if __name__ == "__main__":
    main()

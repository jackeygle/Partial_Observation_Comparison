"""
build_slides.py  —  slide rendering helpers (pptx / pdf / speaker-notes)
=======================================================================

Shared, deck-agnostic rendering library. Slides are DEFINED elsewhere (see
slides/build_meeting_deck.py) as a list of dict "specs"; the three renderers here
turn that list into a .pptx, a .pdf, and a markdown speaker-notes file.

A spec is a dict with any of:
    kind="title" + title/subtitle/author         -> a title slide
    title                                         -> section title at the top of the slide
    bullets  = [(x, y, w, h, size, [(text, level), ...]), ...]
    images   = [(png_path, (box_x, box_y, box_w, box_h)), ...]
    captions = [(text, x, y, w), ...]
    notes    = "speaker notes for this slide"
All coordinates are in inches on a 16:9 (13.333 x 7.5) page.
"""

from __future__ import annotations

import os
from crowdcore import paths
import sys
import textwrap


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

# This script used to live under 4dvarnet_enkf/, where ROOT always pointed at that
# directory (runs/ and check_outputs/ hang off it). After the 2026-09-03 refactor it
# moved to the top level, where dirname(dirname(__file__)) becomes the repo root and
# every os.path.join(ROOT, ...) would silently point somewhere wrong -- hence the
# explicit binding.
ROOT = paths.method(paths.VARNET)
OUTPUTS = os.path.join(ROOT, "check_outputs")             # output root of all check/plot scripts

PAGE_W, PAGE_H = 13.333, 7.5                              # 16:9, inches
DARK = (0x20 / 255, 0x33 / 255, 0x4d / 255)
GRAY = (0x59 / 255, 0x66 / 255, 0x77 / 255)


def _fit(img_path, box_x, box_y, box_w, box_h):
    """Return (x, y, w, h) in inches placing the image inside the box, centred."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    scale = min(box_w / iw, box_h / ih)
    w, h = iw * scale, ih * scale
    return box_x + (box_w - w) / 2, box_y + (box_h - h) / 2, w, h


# --------------------------------------------------------------------------- #
# renderer 1: PowerPoint (.pptx) — download and present
# --------------------------------------------------------------------------- #
def render_pptx(slides, outpath):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(PAGE_W), Inches(PAGE_H)
    blank = prs.slide_layouts[6]
    dark = RGBColor(0x20, 0x33, 0x4d)
    gray = RGBColor(0x59, 0x66, 0x77)

    for spec in slides:
        s = prs.slides.add_slide(blank)
        if spec.get("notes"):
            s.notes_slide.notes_text_frame.text = spec["notes"]

        if spec.get("kind") == "title":
            tb = s.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(PAGE_W - 2.0), Inches(1.8))
            for k, (text, size, bold, col) in enumerate([
                    (spec["title"], 34, True, dark),
                    (spec["subtitle"], 20, False, gray),
                    (spec["author"], 16, False, gray)]):
                p = tb.text_frame.paragraphs[0] if k == 0 else tb.text_frame.add_paragraph()
                p.text = text
                p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, col
            continue

        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(PAGE_W - 1.0), Inches(0.8))
        p = tb.text_frame.paragraphs[0]
        p.text = spec["title"]
        p.font.size, p.font.bold, p.font.color.rgb = Pt(28), True, dark

        for (x, y, w, h, size, items) in spec.get("bullets", []):
            tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
            tf = tb.text_frame
            tf.word_wrap = True
            # An empty ("", lvl) item means "leave a gap". It must NOT become its own
            # paragraph: PowerPoint draws a bullet glyph for an empty bulleted paragraph,
            # which shows up as a stray dot. Carry the gap into the next real paragraph's
            # space_before instead.
            first, pending_gap = True, 0
            for text, lvl in items:
                if not text.strip():
                    pending_gap += 1
                    continue
                p = tf.paragraphs[0] if first else tf.add_paragraph()
                first = False
                p.text = text
                # lvl < 0 means "no bullet glyph": used for table cells and plain lines,
                # where a dot in front of every number reads as a list, not a table.
                # python-pptx has no API for it, hence the buNone element.
                if lvl < 0:
                    from pptx.oxml.ns import qn
                    pPr = p._pPr if p._pPr is not None else p._p.get_or_add_pPr()
                    pPr.append(pPr.makeelement(qn("a:buNone"), {}))
                else:
                    p.level = lvl
                p.font.size = Pt(size if lvl <= 0 else size - 2)
                p.font.color.rgb = dark if lvl <= 0 else gray
                p.space_after = Pt(6)
                if pending_gap:
                    p.space_before = Pt(9 * pending_gap)
                    pending_gap = 0

        for (path, box) in spec.get("images", []):
            x, y, w, h = _fit(path, *box)
            s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))

        for (text, x, y, w) in spec.get("captions", []):
            tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.35))
            p = tb.text_frame.paragraphs[0]
            p.text = text
            p.font.size, p.font.italic, p.font.color.rgb = Pt(13), True, gray

    prs.save(outpath)
    return len(slides)


# --------------------------------------------------------------------------- #
# renderer 2: PDF (previewable/presentable on the cluster)
# --------------------------------------------------------------------------- #
def _pdf_text(fig, x_in, y_in, text, size, color, bold=False, italic=False, ha="left"):
    """Place text at inches from top-left (matplotlib fig coords are bottom-left)."""
    fig.text(x_in / PAGE_W, 1 - y_in / PAGE_H, text, fontsize=size, color=color,
             fontweight="bold" if bold else "normal",
             fontstyle="italic" if italic else "normal", ha=ha, va="top")


def render_pdf(slides, outpath):
    with PdfPages(outpath) as pdf:
        for spec in slides:
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))

            if spec.get("kind") == "title":
                _pdf_text(fig, PAGE_W / 2, 3.0, spec["title"], 26, DARK, bold=True, ha="center")
                _pdf_text(fig, PAGE_W / 2, 3.7, spec["subtitle"], 16, GRAY, ha="center")
                _pdf_text(fig, PAGE_W / 2, 4.2, spec["author"], 13, GRAY, ha="center")
                pdf.savefig(fig)
                plt.close(fig)
                continue

            _pdf_text(fig, 0.5, 0.35, spec["title"], 22, DARK, bold=True)

            for (x, y, w, h, size, items) in spec.get("bullets", []):
                cur_y = y + 0.1
                for text, lvl in items:
                    fs = size if lvl <= 0 else size - 2
                    if not text.strip():                 # gap only — no bullet glyph
                        cur_y += fs * 0.75 / 72
                        continue
                    indent = 0.0 if lvl <= 0 else 0.35
                    prefix = "" if lvl < 0 else ("•  " if lvl == 0 else "–  ")
                    # estimate chars per line: average char width ~ 0.5*fontsize(pt) -> inches
                    chars = max(20, int((w - indent) / (0.5 * fs / 72)))
                    wrapped = textwrap.fill(prefix + text, width=chars,
                                            subsequent_indent="   ")
                    _pdf_text(fig, x + indent, cur_y, wrapped, fs,
                              DARK if lvl <= 0 else GRAY)
                    n_lines = wrapped.count("\n") + 1
                    cur_y += n_lines * (fs * 1.45 / 72) + 0.09

            for (path, box) in spec.get("images", []):
                x, y, w, h = _fit(path, *box)
                ax = fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H])
                ax.imshow(plt.imread(path))
                ax.axis("off")

            for (text, x, y, w) in spec.get("captions", []):
                _pdf_text(fig, x, y + 0.05, text, 12, GRAY, italic=True)

            pdf.savefig(fig)
            plt.close(fig)
    return len(slides)


# --------------------------------------------------------------------------- #
# renderer 3: speaker notes as markdown
# --------------------------------------------------------------------------- #
def render_notes(slides, outpath):
    lines = ["# Speaker notes\n"]
    for k, spec in enumerate(slides, 1):
        title = spec.get("title", "(title slide)")
        lines.append(f"## Slide {k}: {title}\n")
        lines.append((spec.get("notes") or "(no notes)") + "\n")
    with open(outpath, "w") as f:
        f.write("\n".join(lines))

"""Python twin of the dataviz skill's validate_palette.js — the cluster has no node.

Same thresholds and the same Machado-Oliveira-Fernandes (2009) severity-1.0 CVD transforms,
so the verdicts are the JS tool's verdicts, not an approximation of them.
"""
import sys, math, itertools

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR, CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = 0.10, 8.0, 6.0, 15.0, 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
MACHADO = {
 "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
 "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
 "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}

hex2srgb = lambda h: [int(h.strip().lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)]
s2lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin2oklab(r, g, b):
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return (0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s)


def oklab(hexs, sim=None):
    r, g, b = (s2lin(c) for c in hex2srgb(hexs))
    if sim:
        M = MACHADO[sim]
        r, g, b = (M[i][0]*r + M[i][1]*g + M[i][2]*b for i in range(3))
        r, g, b = (max(0.0, min(1.0, v)) for v in (r, g, b))
    return lin2oklab(r, g, b)


def oklch(h):
    L, a, b = oklab(h)
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def dE(h1, h2, sim=None):
    p, q = oklab(h1, sim), oklab(h2, sim)
    return 100 * math.dist(p, q)


def lum(h):
    r, g, b = (s2lin(c) for c in hex2srgb(h))
    return 0.2126*r + 0.7152*g + 0.0722*b


def contrast(h, surf):
    a, b = sorted((lum(h) + 0.05, lum(surf) + 0.05))
    return b / a


def _cli():
    pal = [c for c in sys.argv[1].split(",") if c.strip()]
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    surf = SURFACE[mode]
    lo, hi = BAND[mode]
    rows, ok = [], True

    bad = [c for c in pal if not (lo <= oklch(c)[0] <= hi)]
    rows.append(("Lightness band", not bad,
                 f"{lo}-{hi}: " + (str(bad) if bad else f"all {len(pal)} inside")))
    lowc = [c for c in pal if oklch(c)[1] < CHROMA_FLOOR]
    rows.append(("Chroma floor", not lowc,
                 str(lowc) if lowc else f"all {len(pal)} >= {CHROMA_FLOOR}"))

    adj = list(zip(range(len(pal) - 1), range(1, len(pal))))
    worst = {k: min((dE(pal[i], pal[j], k), pal[i], pal[j]) for i, j in adj)
             for k in ("protan", "deutan", "tritan")}
    wd = min(worst["protan"][0], worst["deutan"][0])
    state = "PASS" if wd >= CVD_TARGET else ("WARN" if wd >= CVD_FLOOR else "FAIL")
    rows.append(("CVD separation (adjacent)", state,
                 f"min(protan,deutan) dE={wd:.1f}  target {CVD_TARGET}, floor {CVD_FLOOR}"
                 f"   [tritan {worst['tritan'][0]:.1f}]"))
    nd, ni, nj = min((dE(pal[i], pal[j]), pal[i], pal[j]) for i, j in adj)
    rows.append(("Normal-vision floor", nd >= NORMAL_FLOOR,
                 f"worst adjacent dE={nd:.1f} ({ni} vs {nj})  floor {NORMAL_FLOOR}"))
    low = [(c, round(contrast(c, surf), 2)) for c in pal if contrast(c, surf) < CONTRAST_MIN]
    rows.append(("Contrast vs surface", not low,
                 str(low) if low else f"all {len(pal)} >= {CONTRAST_MIN}:1"))

    print(f"\n  palette: {', '.join(pal)}   mode={mode}  surface={surf}\n")
    for name, st, detail in rows:
        g = st if isinstance(st, str) else ("PASS" if st else "FAIL")
        if g == "FAIL":
            ok = False
        print(f"  {g:5s}  {name:28s} {detail}")
    print(f"\n  -> {'ALL CHECKS PASS' if ok else 'FAILED - fix the marked checks'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_cli())

"""
MAKE_LOGO: BRAND ASSET GENERATOR
=================================

Regenerate every `images/logo_dyco3_*` file: the Step mark, the dyco wordmark,
and the two locked up together.

Geometry is defined once here and emitted twice - as SVG text, and as PNG drawn
directly with Pillow. There is no SVG rasterizer in the dependency set, so the
two output formats are produced by independent code paths; both read the same
constants, which is what keeps them from drifting apart. After changing any
geometry, check an SVG against its PNG rather than trusting one of them.

The mark is a filled tile carrying four bars that step to the right, one per
chunk. Because the tile is its own background it reads the same on paper and on
ink, so unlike the disc mark it replaced there is no per-ground version and no
per-ground amber: `logo_dyco3_mark_dark` would have been a byte-for-byte copy of
`logo_dyco3_mark` and is not emitted. The lockup still has one, since the
wordmark beside it does change with the ground. A one-colour cut is emitted for
print and for anywhere amber cannot go.

Not imported by the package; `pyproject.toml` ships only `dyco`. Pillow arrives
as a matplotlib dependency, so no extra install is needed:

    uv run python tools/make_logo.py

Part of the dyco package: https://github.com/holukas/dyco
"""

import math
from itertools import pairwise
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "images"
# The documentation's copy of the mark. It lives in the package rather than in
# `images/`, which is excluded from the sdist, because the docs have to build
# from an sdist too. Emitted here so it cannot drift from the one in `images/`.
DOCS_LOGO = _ROOT / "dyco" / "assets" / "logo.svg"
SS = 4  # supersample factor, downsampled with LANCZOS

TILE = "#2b4c6f"
BARS = ("#6b93b5", "#93b5d0", "#bcd4e6")  # the three that step toward the accent
AMBER = "#f0a500"
INK = "#15171C"                           # the wordmark, and the one-colour cut
PAPER = "#EDECE6"


def rgba(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


# ---------------------------------------------------------------- geometry
# The Step mark, authored in a 64 x 64 square and scaled into whatever box it is
# emitted to. A rounded tile, and four bars each starting 8 further right than the
# one above it. The offset is a third of the 24-unit bar; below about a fifth the
# eye stops reading it as a step once this is a 16 px favicon.
MARK_UNIT = 64
MARK_TILE = (2, 2, 60, 60, 13)        # x, y, w, h, corner radius
MARK_BAR = (24, 8, 4)                 # w, h, corner radius
MARK_BAR_XY = ((8, 8.5), (16, 21.5), (24, 34.5), (32, 47.5))

# The wordmark, in a 240 x 100 box. One stroke weight throughout; d, c and o are
# the same circle - closed, opened, and cut. Ink spans x 2..228, y 2..92.
SW = 12
D_BOWL = (30, 48, 22)
D_STEM = [(52, 8), (52, 70)]
Y_LEFT = [(68, 26), (83, 62)]
Y_RIGHT_LINE = [(98, 26), (76, 78)]
Y_TAIL_CTRL = (72, 88)
Y_TAIL_END = (62, 86)
C_CENTER = (130, 48, 22)
C_ARC = (42, 318)            # degrees, clockwise on screen; the gap faces right
O_CENTER = (181, 48, 22)
O_SLIP_DX = 8
O_CUT = (44, 51)

# Ink bounds of the word, stroke included. The y right arm passes x = 88.7 at the
# x-height midline, so the c cannot sit any further left without colliding.
WORD_INK_L = D_BOWL[0] - D_BOWL[2] - SW / 2
WORD_INK_R = O_CENTER[0] + O_SLIP_DX + O_CENTER[2] + SW / 2
WORD_INK_T = D_STEM[0][1] - SW / 2
WORD_INK_B = 92.0

WORD_W, WORD_H = 240, 100
MARK_W, MARK_H = 100, 100
MARK_VIEW_SCALE = MARK_W / MARK_UNIT

# Lockup: the mark scaled to 74 tall and optically centred on the x-height axis
# (y = 48), tile flush left, then 26 units of air before the word. The tile is
# 60 units tall in the authoring square and square, so it is also 74 wide.
LOCK_MARK_SCALE = 74.0 / MARK_TILE[3]
LOCK_MARK_DX = -MARK_TILE[0]                      # user units, pre-scale
LOCK_MARK_DY = 48.0 / LOCK_MARK_SCALE - MARK_UNIT / 2
LOCK_WORD_DX = 74.0 + 26.0 - WORD_INK_L
LOCK_WORD_DY = -WORD_INK_T
LOCK_W = round(LOCK_WORD_DX + WORD_INK_R)
LOCK_H = round(WORD_INK_B - WORD_INK_T)


def quad_points(p0, ctrl, p1, n=20):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((
            u * u * p0[0] + 2 * u * t * ctrl[0] + t * t * p1[0],
            u * u * p0[1] + 2 * u * t * ctrl[1] + t * t * p1[1],
        ))
    return out


# ------------------------------------------------------------------- raster
class Pen:
    """Draws user-space geometry onto a supersampled RGBA layer."""

    def __init__(self, size_px, scale, dx=0.0, dy=0.0):
        self.img = Image.new("RGBA", (size_px[0] * SS, size_px[1] * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.k = scale * SS
        self.dx, self.dy = dx, dy

    def p(self, x, y):
        return ((x + self.dx) * self.k, (y + self.dy) * self.k)

    def dot(self, x, y, r, colour):
        cx, cy = self.p(x, y)
        rr = r * self.k
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=colour)

    def rrect(self, x, y, w, h, r, colour):
        x0, y0 = self.p(x, y)
        x1, y1 = self.p(x + w, y + h)
        self.d.rounded_rectangle([x0, y0, x1, y1], radius=r * self.k, fill=colour)

    def ring(self, cx, cy, r, w, colour):
        # Pillow strokes inward from the bbox, so push the bbox out by half the width
        x, y = self.p(cx, cy)
        rr = (r + w / 2) * self.k
        self.d.ellipse([x - rr, y - rr, x + rr, y + rr],
                       outline=colour, width=round(w * self.k))

    def arc(self, cx, cy, r, a0, a1, w, colour):
        x, y = self.p(cx, cy)
        rr = (r + w / 2) * self.k
        self.d.arc([x - rr, y - rr, x + rr, y + rr], a0, a1,
                   fill=colour, width=round(w * self.k))
        for a in (a0, a1):  # round caps
            self.dot(cx + r * math.cos(math.radians(a)),
                     cy + r * math.sin(math.radians(a)), w / 2, colour)

    def stroke(self, pts, w, colour):
        # Segment by segment, with a disc at every vertex for the joins and caps.
        # Pillow's joint="curve" throws a fan of spikes when the vertices are as
        # dense as the sampled bezier of the y descender.
        wpx = round(w * self.k)
        dev = [self.p(*q) for q in pts]
        for a, b in pairwise(dev):
            self.d.line([a, b], fill=colour, width=wpx)
        for q in pts:
            self.dot(q[0], q[1], w / 2, colour)

    def band(self, y0, y1):
        """Keep only the rows between y0 and y1 (user units)."""
        mask = Image.new("L", self.img.size, 0)
        ImageDraw.Draw(mask).rectangle(
            [0, (y0 + self.dy) * self.k, self.img.size[0], (y1 + self.dy) * self.k - 1],
            fill=255)
        self.img.putalpha(ImageChops.multiply(self.img.getchannel("A"), mask))

    def onto(self, base):
        base.alpha_composite(self.img)


def draw_mark(base, size_px, scale, dx, dy, tile, bars):
    pen = Pen(size_px, scale, dx, dy)
    pen.rrect(*MARK_TILE[:4], MARK_TILE[4], tile)
    bw, bh, br = MARK_BAR
    for (bx, by), col in zip(MARK_BAR_XY, bars):
        pen.rrect(bx, by, bw, bh, br, col)
    pen.onto(base)


def draw_word(base, size_px, scale, dx, dy, colour, accent):
    pen = Pen(size_px, scale, dx, dy)
    pen.ring(*D_BOWL, SW, colour)
    pen.stroke(D_STEM, SW, colour)
    pen.stroke(Y_LEFT, SW, colour)
    pen.stroke(Y_RIGHT_LINE + quad_points(Y_RIGHT_LINE[1], Y_TAIL_CTRL, Y_TAIL_END)[1:],
               SW, colour)
    pen.arc(*C_CENTER, C_ARC[0], C_ARC[1], SW, colour)
    pen.onto(base)

    cx, cy, r = O_CENTER
    for ox, (y0, y1), col in ((O_SLIP_DX, (-50, O_CUT[0]), accent),
                              (0, (O_CUT[1], 200), colour)):
        pen = Pen(size_px, scale, dx, dy)
        pen.ring(cx + ox, cy, r, SW, col)
        pen.band(y0, y1)
        pen.onto(base)


def render(path, size_px, build):
    base = Image.new("RGBA", (size_px[0] * SS, size_px[1] * SS), (0, 0, 0, 0))
    build(base)
    base.resize(size_px, Image.LANCZOS).save(path)
    print("wrote", path.name, size_px)


def mark_colours(dark, mono, conv=lambda c: c):
    """(tile, four bar colours) for the requested ground and colour treatment.

    Only `mono` reads `dark`. In full colour the tile supplies its own ground, so
    the mark is the same on paper as on ink.
    """
    if not mono:
        return conv(TILE), [conv(c) for c in (*BARS, AMBER)]
    base, on = (PAPER, INK) if dark else (INK, PAPER)
    return conv(base), [conv(on)] * 4


def word_colours(dark, mono, conv=lambda c: c):
    """(stroke, accent) for the requested ground and colour treatment."""
    col = PAPER if dark else INK
    return conv(col), conv(col if mono else AMBER)


def mark_png(path, px, dark=False, mono=False):
    tile, bars = mark_colours(dark, mono, rgba)
    render(path, (px, px),
           lambda b: draw_mark(b, (px, px), px / MARK_UNIT, 0, 0, tile, bars))


def lockup_png(path, px_w, dark=False, mono=False):
    tile, bars = mark_colours(dark, mono, rgba)
    col, acc = word_colours(dark, mono, rgba)
    px_h = round(px_w * LOCK_H / LOCK_W)
    s = px_w / LOCK_W
    size = (px_w, px_h)

    def build(b):
        draw_mark(b, size, s * LOCK_MARK_SCALE, LOCK_MARK_DX, LOCK_MARK_DY, tile, bars)
        draw_word(b, size, s, LOCK_WORD_DX, LOCK_WORD_DY, col, acc)

    render(path, size, build)


# ---------------------------------------------------------------------- svg
def _mark_body(tile, bars, pad="  "):
    x, y, w, h, r = MARK_TILE
    bw, bh, br = MARK_BAR
    out = [f'{pad}<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{tile}"/>']
    for (bx, by), col in zip(MARK_BAR_XY, bars):
        out.append(f'{pad}<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                   f'rx="{br}" fill="{col}"/>')
    return "\n".join(out)


def _word_body(col, acc):
    r = C_CENTER[2]
    a = math.radians(C_ARC[0])
    x1 = C_CENTER[0] + r * math.cos(a)
    ylo = C_CENTER[1] - r * math.sin(a)
    yhi = C_CENTER[1] + r * math.sin(a)
    return f'''  <g fill="none" stroke="{col}" stroke-width="{SW}" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="{D_BOWL[0]}" cy="{D_BOWL[1]}" r="{D_BOWL[2]}"/>
    <path d="M{D_STEM[0][0]} {D_STEM[0][1]} V{D_STEM[1][1]}"/>
    <path d="M{Y_LEFT[0][0]} {Y_LEFT[0][1]} L{Y_LEFT[1][0]} {Y_LEFT[1][1]}"/>
    <path d="M{Y_RIGHT_LINE[0][0]} {Y_RIGHT_LINE[0][1]} L{Y_RIGHT_LINE[1][0]} {Y_RIGHT_LINE[1][1]} Q{Y_TAIL_CTRL[0]} {Y_TAIL_CTRL[1]} {Y_TAIL_END[0]} {Y_TAIL_END[1]}"/>
    <path d="M{x1:.2f} {ylo:.2f} A{r} {r} 0 1 0 {x1:.2f} {yhi:.2f}"/>
  </g>
  <g fill="none" stroke-width="{SW}" stroke-linecap="round">
    <g clip-path="url(#dycoOT)"><circle cx="{O_CENTER[0] + O_SLIP_DX}" cy="{O_CENTER[1]}" r="{O_CENTER[2]}" stroke="{acc}"/></g>
    <g clip-path="url(#dycoOB)"><circle cx="{O_CENTER[0]}" cy="{O_CENTER[1]}" r="{O_CENTER[2]}" stroke="{col}"/></g>
  </g>'''


# The clip ids are fixed, so inlining two wordmarks or two lockups into one HTML
# document would make the second set of definitions collide with the first.
# Reference the files with <img> or <picture>, which keeps each one its own
# document. The mark is plain rectangles and carries no clips, so it is safe to
# inline as many times as you like.
_CLIPS_WORD = f'''    <clipPath id="dycoOT"><rect x="0" y="0" width="{WORD_W}" height="{O_CUT[0]}"/></clipPath>
    <clipPath id="dycoOB"><rect x="0" y="{O_CUT[1]}" width="{WORD_W}" height="{100 - O_CUT[1]}"/></clipPath>'''


def svg_mark(dark=False, mono=False):
    tile, bars = mark_colours(dark, mono)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {MARK_W} {MARK_H}" width="{MARK_W}" height="{MARK_H}" role="img" aria-label="dyco">
  <title>dyco</title>
  <g transform="scale({MARK_VIEW_SCALE})">
{_mark_body(tile, bars, pad="    ")}
  </g>
</svg>
'''


def svg_word(dark, mono=False):
    col, acc = word_colours(dark, mono)
    w = round(WORD_INK_R - WORD_INK_L)
    h = round(WORD_INK_B - WORD_INK_T)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="dyco">
  <title>dyco</title>
  <defs>
{_CLIPS_WORD}
  </defs>
  <g transform="translate({-WORD_INK_L} {-WORD_INK_T})">
{_word_body(col, acc)}
  </g>
</svg>
'''


def svg_lockup(dark, mono=False):
    tile, bars = mark_colours(dark, mono)
    col, acc = word_colours(dark, mono)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {LOCK_W} {LOCK_H}" width="{LOCK_W}" height="{LOCK_H}" role="img" aria-label="dyco">
  <title>dyco</title>
  <defs>
{_CLIPS_WORD}
  </defs>
  <g transform="translate({LOCK_MARK_DX * LOCK_MARK_SCALE:.3f} {LOCK_MARK_DY * LOCK_MARK_SCALE:.3f}) scale({LOCK_MARK_SCALE:.5f})">
{_mark_body(tile, bars, pad="    ")}
  </g>
  <g transform="translate({LOCK_WORD_DX:.3f} {LOCK_WORD_DY})">
{_word_body(col, acc)}
  </g>
</svg>
'''


def main():
    for name, text in {
        # primary: the stepping bars, amber on the last
        "logo_dyco3_mark.svg": svg_mark(),
        "logo_dyco3_wordmark.svg": svg_word(False),
        "logo_dyco3_wordmark_dark.svg": svg_word(True),
        "logo_dyco3_lockup.svg": svg_lockup(False),
        "logo_dyco3_lockup_dark.svg": svg_lockup(True),
        # one colour: print, favicons, anywhere amber cannot go
        "logo_dyco3_mark_mono.svg": svg_mark(mono=True),
        "logo_dyco3_mark_mono_dark.svg": svg_mark(True, mono=True),
        "logo_dyco3_lockup_mono.svg": svg_lockup(False, mono=True),
        "logo_dyco3_lockup_mono_dark.svg": svg_lockup(True, mono=True),
    }.items():
        (OUT / name).write_text(text, encoding="utf-8")
        print("wrote", name)

    DOCS_LOGO.parent.mkdir(parents=True, exist_ok=True)
    DOCS_LOGO.write_text(svg_mark(), encoding="utf-8")
    print("wrote", DOCS_LOGO.relative_to(_ROOT).as_posix())

    mark_png(OUT / "logo_dyco3_mark_1024px.png", 1024)
    mark_png(OUT / "logo_dyco3_mark_256px.png", 256)
    mark_png(OUT / "logo_dyco3_mark_mono_256px.png", 256, mono=True)
    lockup_png(OUT / "logo_dyco3_lockup_1024px.png", 1024)
    lockup_png(OUT / "logo_dyco3_lockup_dark_1024px.png", 1024, dark=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Figure 1 - study design, portrait flowchart.

Journal density: tight boxes, bold heads over regular body text, hairline
rules with heavy rules reserved for the three leakage controls, grey fills
on the quantities the argument turns on, black terminal.

    python pipeline_flow.py            # black and white (default)
    python pipeline_flow.py --color    # restrained colour

Every box's y is derived from its own line count and a fixed gap, and every
label's rendered width is measured against its box on each run, so neither
a collision nor an overflow can appear silently.
"""
import argparse
import os
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures")
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch

ap = argparse.ArgumentParser()
ap.add_argument("--color", action="store_true")
ap.add_argument("--fw", type=float, default=6.0)
ap.add_argument("--out", default=None)
A = ap.parse_args()

plt.rcParams.update({
    "font.family": "Liberation Sans",      # Arial metrics
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "svg.fonttype": "path"})

if A.color:
    INK, LINE = "#101820", "#44505a"
    PLAIN_F, PLAIN_E, PLAIN_W = "#ffffff", "#44505a", 1.0
    CTRL_F, CTRL_E, CTRL_W = "#ffffff", "#8c2f2f", 2.0
    KEYF, KEY_E, KEY_W = "#e4ecf4", "#44505a", 1.0
    TERM_F, TERM_E, TERM_T = "#17303f", "#17303f", "#ffffff"
else:
    INK, LINE = "#000000", "#414141"
    PLAIN_F, PLAIN_E, PLAIN_W = "#ffffff", "#414141", 1.1
    CTRL_F, CTRL_E, CTRL_W = "#ffffff", "#000000", 2.0
    KEYF, KEY_E, KEY_W = "#d9d9d9", "#414141", 1.1
    TERM_F, TERM_E, TERM_T = "#1c1c1c", "#1c1c1c", "#ffffff"

W, HSPAN = 109.0, 176.0
fig, ax = plt.subplots(figsize=(A.fw, A.fw * HSPAN / W))
ax.set_xlim(-10, 102)
ax.set_ylim(-30, 146)
ax.axis("off")
ax.set_position([0, 0, 1, 1])

HS, BS = 8.0, 7.3          # head size, body size
LH, HG, PAD = 2.35, 0.55, 3.0  # line height, head gap, vertical padding
GAP = 2.1                  # clearance between stacked boxes
B = {}


def height(n):
    return PAD + HG + LH * (1 + n)


TOP = {}      # key -> (top edge, bottom edge), filled by place()


def place(key, n, top=None, below=None, gap=GAP, align=None):
    """Return the centre y for a box, derived from its own line count.

    Pass `below` (one key or several) to hang it under what is already
    placed, or `align` to share another box's centre. Nothing is positioned
    by hand, so changing a label's line count cannot silently close a gap.
    """
    h = height(n)
    if align is not None:
        c = sum(TOP[k][0] + TOP[k][1] for k in np.atleast_1d(align)) / (
            2 * len(np.atleast_1d(align)))
        top = c + h / 2
    elif top is None:
        ks = [below] if isinstance(below, str) else list(below)
        top = min(TOP[k][1] for k in ks) - gap
    TOP[key] = (top, top - h)
    return top - h / 2


def box(key, x, y, w, head, body=(), kind="plain", bs=BS, round_=False,
        bold_body=False):
    fc, ec, lw = {"plain": (PLAIN_F, PLAIN_E, PLAIN_W),
                  "ctrl":  (CTRL_F, CTRL_E, CTRL_W),
                  "key":   (KEYF, KEY_E, KEY_W)}[kind]
    h = height(len(body))
    style = ("round,pad=0,rounding_size=%.2f" % (h / 2) if round_
             else "round,pad=0,rounding_size=0.9")
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=3, mutation_aspect=1))
    top = y + h / 2 - PAD / 2 - LH / 2
    ax.text(x, top, head, ha="center", va="center", fontsize=HS,
            fontweight="bold", color=INK, zorder=4)
    for i, t in enumerate(body):
        ax.text(x, top - HG - (i + 1) * LH, t, ha="center", va="center",
                fontsize=bs, color=INK, zorder=4,
                fontweight="bold" if bold_body else "normal")
    B[key] = (x, y, w, h)


def diamond(key, x, y, w, h, l1, l2):
    ax.add_patch(Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2],
                          [x - w / 2, y]], closed=True, facecolor=PLAIN_F,
                         edgecolor=INK, linewidth=1.3, zorder=3))
    ax.text(x, y + 1.3, l1, ha="center", va="center", fontsize=BS,
            fontweight="bold", color=INK, zorder=4)
    ax.text(x, y - 1.5, l2, ha="center", va="center", fontsize=BS,
            fontweight="bold", color=INK, zorder=4)
    B[key] = (x, y, w, h)


def side(k, s):
    x, y, w, h = B[k]
    return {"t": (x, y + h / 2), "b": (x, y - h / 2),
            "l": (x - w / 2, y), "r": (x + w / 2, y)}[s]


AH_L, AH_W, LW = 1.15, 0.30, 0.95   # head length, half-width, stroke
TIP_GAP = 0.45                      # clear air between tip and box border


SEGS = []


def link(p0, p1, waypoints=(), head=True):
    """Orthogonal connector; elbows are vertical-then-horizontal.

    The head is a filled triangle and the stroke stops at its base, so the
    line is never drawn twice over itself at the tip - which is what makes
    a matplotlib arrow look blunt and smudged next to a drawn one.
    """
    pts = [p0, *waypoints, p1]
    verts, codes = [pts[0]], [Path.MOVETO]
    for a, b in zip(pts, pts[1:]):
        if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:
            verts.append((a[0], b[1])); codes.append(Path.LINETO)
        verts.append(b); codes.append(Path.LINETO)
    if head:
        (x0, y0), (x1, y1) = verts[-2], verts[-1]
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        if L > 1e-6:
            ux, uy = dx / L, dy / L
            if L > TIP_GAP + AH_L:
                x1, y1 = x1 - ux * TIP_GAP, y1 - uy * TIP_GAP
                L -= TIP_GAP
                verts[-1] = (x1, y1)
            a = min(AH_L, L * 0.9)
            bx, by = x1 - ux * a, y1 - uy * a
            px, py = -uy * AH_W, ux * AH_W
            ax.add_patch(Polygon([(x1, y1), (bx + px, by + py),
                                  (bx - px, by - py)], closed=True,
                                 facecolor=LINE, edgecolor=LINE,
                                 linewidth=0.35, joinstyle="miter",
                                 zorder=2.5))
            verts[-1] = (bx, by)
    for _a, _b in zip(verts, verts[1:]):
        SEGS.append((_a, _b))
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none",
                           edgecolor=LINE, linewidth=LW, zorder=2,
                           joinstyle="miter", capstyle="butt"))


def dot(x, y):
    ax.add_patch(Circle((x, y), 0.6, facecolor=LINE, edgecolor="none",
                        zorder=5))


# ---------------------------------------------------------------- sources
box("ZF", 12, place("ZF", 2, top=133), 34, "Zebrafish",
    ["embryo · larva · morphology · lipid", "48 tasks"])
box("TX", 12, place("TX", 2, below="ZF"), 34, "Tox21 · MUV · PCBA",
    ["in vitro panels", "67 tasks"])
box("CH", 12, place("CH", 2, below="TX"), 34, "ChEMBL sparse",
    ["0.21% label density", "180 tasks"])

for k in ("ZF", "TX", "CH"):
    link(side(k, "r"), (34, B[k][1]), head=False)
link((34, B["ZF"][1]), (34, B["CH"][1]), head=False)

box("AS", 50, place("AS", 1, align=["ZF", "CH"]), 24,
    "295 assays", ["339,249 compounds"], "key")
dot(34, B["AS"][1])
link((34, B["AS"][1]), side("AS", "l"))

# ---------------------------------------------------------------- control 1
box("C1", 84, place("C1", 5, top=135), 32, "CONTROL 1",
    ["Global scaffold assignment", "Bemis–Murcko scaffolds",
     "taken over the union", "of all compounds",
     "never within a single task"], "ctrl")
link(side("AS", "t"), side("C1", "l"))

box("SP", 84, place("SP", 1, below="C1"), 32,
    "Scaffold-disjoint", ["train / test split"], "key", bold_body=True)
link(side("C1", "b"), side("SP", "t"))

# ------------------------------------------------- two arms of the measure
y_rail = TOP["SP"][1] - GAP
box("SM", 50, place("SM", 2, below="SP", gap=2 * GAP), 28,
    "Source model", ["random forest · ECFP6", "1024-bit"])
box("NF", 84, place("NF", 2, align="SM"), 30, "Noise floor",
    ["295 tasks × 20 seeds", "seed varied, nothing else"], "key")
link(side("SP", "b"), (84, y_rail), head=False)
link((84, y_rail), (50, y_rail), head=False)
dot(84, y_rail)
link((50, y_rail), side("SM", "t"))
link((84, y_rail), side("NF", "t"))

box("C2", 50, place("C2", 4, below="SM"), 28, "CONTROL 2",
    ["Out-of-fold predictions", "source predicts its own",
     "training compounds", "out of fold"], "ctrl")
link(side("SM", "b"), side("C2", "t"))

box("TH", 84, place("TH", 1, align="C2"), 36, "Per-task 2σ threshold",
    ["fixed before any pair examined"], "key", round_=True)
link(side("NF", "b"), side("TH", "t"))

box("TF", 50, place("TF", 3, below="C2"), 28, r"Target features $\oplus$",
    ["source column", "ΔAUPRC =", "with source − alone"])
link(side("C2", "b"), side("TF", "t"))

# ---------------------------------------------------------------- decision
y_join = TOP["TF"][1] - GAP
QH = 10.4
qc = TOP["TF"][1] - 2 * GAP - QH / 2
TOP["Q"] = (qc + QH / 2, qc - QH / 2)
diamond("Q", 50, qc, 39, QH, "ΔAUPRC exceeds", "2σ of this target?")
link(side("TF", "b"), (50, y_join), head=False)
link(side("TH", "b"), (50, y_join), [(84, y_join)], head=False)
dot(50, y_join)
link((50, y_join), side("Q", "t"))

y_br = TOP["Q"][1] - 1.0
box("KY", 40, place("KY", 1, top=y_br - 2.2), 34, "1,930 pairs · 7.7%",
    ["1,036 helped · 894 hurt"], "key")
box("NO", 84, place("NO", 2, top=y_br - 2.2), 32, "23,070 pairs · 92.3%",
    ["indistinguishable", "from seed noise"])
link((30.5, qc), side("KY", "t"), [(30.5, y_br), (40, y_br)])
link((69.5, qc), side("NO", "t"), [(69.5, y_br), (84, y_br)])
ax.text(31.7, qc - 3.4, "yes", fontsize=BS, ha="left", va="center", color=INK)
ax.text(68.3, qc - 3.4, "no", fontsize=BS, ha="right", va="center", color=INK)

box("C3", 40, place("C3", 4, below=("KY", "NO")), 34, "CONTROL 3",
    ["Direction prediction", "folds grouped by target",
     "0.557 ± 0.124", "chance 0.500"], "ctrl")
box("OV", 84, place("OV", 3, top=TOP["C3"][0]), 32, "Overlap threshold",
    ["within target p = 0.73", r"pooled p = $2.7{\times}10^{-9}$ was",
     "Simpson's paradox"])
link(side("KY", "b"), side("C3", "t"))
link(side("NO", "b"), side("OV", "t"))

# ------------------------------------------------- ancillary arms, left bus
box("ST", 12, place("ST", 3, below="CH"), 34, "Stacking sources",
    ["k = 1 → 50, nested pools", "spread ×2",
     "mean negative at every k"])
box("SC", 12, place("SC", 3, below="ST"), 34, "Screening",
    ["vs. all and vs. random-k", "−0.0040",
     "no arm beats single-task"])
box("VC", 12, place("VC", 4, below="SC"), 34, "Validity check",
    ["zebrafish benchmark", "reproduced",
     "AUROC 0.633 vs 0.638", "published"])

link(side("TF", "l"), (34, B["TF"][1]), head=False)
link((34, B["ST"][1]), (34, B["VC"][1]), head=False)
dot(34, B["TF"][1])
for k in ("ST", "SC", "VC"):
    link((34, B[k][1]), side(k, "r"))
    link(side(k, "l"), (-7, B[k][1]), head=False)
    dot(-7, B[k][1])

# ---------------------------------------------------------------- terminal
box("END", 50, place("END", 1, below="C3", gap=2.6 * GAP), 62,
    "Transfer effects are smaller",
    ["than the noise in estimating them"], bs=HS, bold_body=True)
e_x, e_y, e_w, e_h = B["END"]
ax.add_patch(FancyBboxPatch((e_x - e_w / 2, e_y - e_h / 2), e_w, e_h,
                            boxstyle="round,pad=0,rounding_size=0.9",
                            facecolor=TERM_F, edgecolor=TERM_E,
                            linewidth=1.0, zorder=3.5))
for t in ax.texts:
    if t.get_text() in ("Transfer effects are smaller",
                        "than the noise in estimating them"):
        t.set_color(TERM_T); t.set_zorder(4.5); t.set_fontweight("bold")


link(side("C3", "b"), (40, TOP["END"][0]))
link(side("OV", "b"), side("END", "r"), [(84, e_y)])
link((-7, B["ST"][1]), (-7, e_y), head=False)
link((-7, e_y), side("END", "l"))

# ---------------------------------------- fit the canvas to what was drawn
_xs = [-10, 102]
_ys = [v for t in TOP.values() for v in t] + [B["END"][1] - B["END"][3] / 2]
_y0, _y1 = min(_ys) - 3.0, max(_ys) + 3.0
ax.set_xlim(*_xs)
ax.set_ylim(_y0, _y1)
fig.set_size_inches(A.fw, A.fw * (_y1 - _y0) / (_xs[1] - _xs[0]))

# ------------------------------------------- collision and overflow audits
bad = []
for k1, (x1, y1, w1, h1) in B.items():
    for k2, (x2, y2, w2, h2) in B.items():
        if k1 >= k2 or abs(x1 - x2) >= (w1 + w2) / 2:
            continue
        g = abs(y1 - y2) - (h1 + h2) / 2
        if g < GAP - 0.05:
            bad.append(("clearance", k1, k2, round(g, 2)))

fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()
for t in ax.texts:
    bb = t.get_window_extent(renderer=rend)
    x0, _ = inv.transform((bb.x0, bb.y0))
    x1, _ = inv.transform((bb.x1, bb.y1))
    tw, (tx, ty) = abs(x1 - x0), t.get_position()
    for k, (bx, by, bw, bh) in B.items():
        if abs(tx - bx) < 1.0 and by - bh / 2 - 1 <= ty <= by + bh / 2 + 1:
            if tw > bw - 2.2:
                bad.append(("overflow", k, round(tw, 1), t.get_text()[:34]))
            break

# connectors must not cut through a box, nor grind along its edge
def _seg_box(a, b, r):
    x0, x1 = sorted((a[0], b[0])); y0, y1 = sorted((a[1], b[1]))
    bx, by, bw, bh = r
    ix = min(x1, bx + bw / 2) - max(x0, bx - bw / 2)
    iy = min(y1, by + bh / 2) - max(y0, by - bh / 2)
    return ix, iy

for (a, b) in SEGS:
    for k, r in B.items():
        rr = (r[0], r[1], r[2] * 0.62, r[3]) if k == "Q" else r
        ix, iy = _seg_box(a, b, rr)
        if ix > 0.6 and iy > 0.6:
            bad.append(("crosses", k, tuple(round(v, 1) for v in a),
                        tuple(round(v, 1) for v in b)))
        elif (k != "Q" and -1.4 < ix <= 0.6 and iy > 1.5
              and abs(a[0] - b[0]) < 1e-6):
            bad.append(("grazes", k, round(ix, 2),
                        tuple(round(v, 1) for v in a)))

print(*bad, sep="\n") if bad else print("audit: clean")
if os.environ.get("AUDIT_ONLY"):
    sys.exit(1 if bad else 0)

os.makedirs(_OUT, exist_ok=True)
out = (A.out or os.path.join(_OUT, "figure1_pipeline")) + ("_color" if A.color else "")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
matplotlib.rcParams["svg.fonttype"] = "path"
fig.savefig(f"{out}.svg", bbox_inches="tight", facecolor="white")
matplotlib.rcParams["svg.fonttype"] = "none"
fig.savefig(f"{out}_editable.svg", bbox_inches="tight", facecolor="white")
print("wrote", out + ".png / .pdf / .svg / _editable.svg")

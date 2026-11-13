#!/usr/bin/env python
"""Figure 1 - landscape. Same content and style as pipeline_flow.py.

Sixteen boxes in a single left-to-right row would run to roughly 5:1, wider
than any page, so the flow is laid out in two bands: corpus and split along
the top, measurement through conclusion along the bottom, with the three
ancillary arms filling the top-right. A single return rail carries the split
down to the second band.

    python pipeline_land.py           # black and white (default)
    python pipeline_land.py --color   # restrained colour
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch

ap = argparse.ArgumentParser()
ap.add_argument("--color", action="store_true")
ap.add_argument("--fw", type=float, default=13.0)
ap.add_argument("--out", default="figures/figure1_land")
A = ap.parse_args()

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
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

HS, BS = 7.3, 6.7
LH, HG, PAD = 2.45, 0.85, 3.8
GAP = 3.2
AH_L, AH_W, LW = 1.45, 0.56, 1.05
B, TOP = {}, {}

fig, ax = plt.subplots(figsize=(A.fw, A.fw * 0.36))
ax.axis("off")
ax.set_position([0, 0, 1, 1])


def height(n):
    return PAD + HG + LH * (1 + n)


def box(key, x, y, w, head, body=(), kind="plain", round_=False):
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
                fontsize=BS, color=INK, zorder=4)
    B[key] = (x, y, w, h)
    TOP[key] = (y + h / 2, y - h / 2)


def diamond(key, x, y, w, h, l1, l2):
    ax.add_patch(Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2],
                          [x - w / 2, y]], closed=True, facecolor=PLAIN_F,
                         edgecolor=INK, linewidth=1.3, zorder=3))
    ax.text(x, y + 1.5, l1, ha="center", va="center", fontsize=BS,
            fontweight="bold", color=INK, zorder=4)
    ax.text(x, y - 1.6, l2, ha="center", va="center", fontsize=BS,
            fontweight="bold", color=INK, zorder=4)
    B[key] = (x, y, w * 0.52, h)
    TOP[key] = (y + h / 2, y - h / 2)


def side(k, s):
    x, y, w, h = B[k]
    return {"t": (x, y + h / 2), "b": (x, y - h / 2),
            "l": (x - w / 2, y), "r": (x + w / 2, y)}[s]


def link(p0, p1, waypoints=(), head=True, vfirst=True):
    pts = [p0, *waypoints, p1]
    verts, codes = [pts[0]], [Path.MOVETO]
    for a, b in zip(pts, pts[1:]):
        if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:
            elbow = (a[0], b[1]) if vfirst else (b[0], a[1])
            verts.append(elbow); codes.append(Path.LINETO)
        verts.append(b); codes.append(Path.LINETO)
    if head:
        (x0, y0), (x1, y1) = verts[-2], verts[-1]
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        if L > 1e-6:
            ux, uy = dx / L, dy / L
            a = min(AH_L, L * 0.9)
            bx, by = x1 - ux * a, y1 - uy * a
            px, py = -uy * AH_W, ux * AH_W
            ax.add_patch(Polygon([(x1, y1), (bx + px, by + py),
                                  (bx - px, by - py)], closed=True,
                                 facecolor=LINE, edgecolor="none", zorder=2.5))
            verts[-1] = (bx, by)
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none",
                           edgecolor=LINE, linewidth=LW, zorder=2,
                           joinstyle="miter", capstyle="butt"))


def dot(x, y):
    ax.add_patch(Circle((x, y), 0.6, facecolor=LINE, edgecolor="none",
                        zorder=5))


# ======================================================== band 1 : corpus
box("ZF", 30, 130, 50, "Zebrafish",
    ["embryo · larva · morphology · lipid", "48 tasks"])
box("TX", 30, 112, 50, "Tox21 · MUV · PCBA", ["in vitro panels", "69 tasks"])
box("CH", 30, 94, 50, "ChEMBL sparse", ["0.21% label density", "180 tasks"])
for k in ("ZF", "TX", "CH"):
    link(side(k, "r"), (62, B[k][1]), head=False)
link((62, 130), (62, 94), head=False)
dot(62, 112)

box("AS", 90, 112, 34, "297 assays", ["339,249 compounds"], "key")
link((62, 112), side("AS", "l"))

box("C1", 148, 112, 62, "CONTROL 1",
    ["Global scaffold assignment · Bemis–Murcko",
     "over the union of all compounds"], "ctrl")
link(side("AS", "r"), side("C1", "l"))

box("SP", 208, 112, 40, "Scaffold-disjoint", ["train / test split"], "key")
link(side("C1", "r"), side("SP", "l"))

# ancillary arms occupy the top right
box("ST", 274, 128, 54, "Stacking sources",
    ["k = 1 → 50, nested pools", "spread ×2 · mean negative at every k"])
box("SC", 274, 110, 54, "Screening",
    ["vs. all and vs. random-k · −0.0040", "no arm beats single-task"])
box("VC", 274, 92, 54, "Validity check",
    ["zebrafish benchmark reproduced", "AUROC 0.633 vs 0.638 published"])

# ==================================================== band 2 : measurement
box("SM", 34, 60, 48, "Source model", ["random forest · ECFP6 1024-bit"])
box("NF", 34, 28, 44, "Noise floor",
    ["295 of 297 tasks × 20 seeds", "seed varied, nothing else"], "key")
# return rail from the split down into band 2
link(side("SP", "b"), (208, 78), head=False)
link((208, 78), (34, 78), head=False)
dot(34, 78)
link((34, 78), side("SM", "t"))
link((34, 78), side("NF", "t"), [(6, 78), (6, 28)], head=False)
link((6, 28), side("NF", "l"))

box("C2", 100, 60, 58, "CONTROL 2",
    ["Out-of-fold predictions · source predicts",
     "its own training compounds out of fold"], "ctrl")
link(side("SM", "r"), side("C2", "l"))

box("TH", 96, 28, 44, "Per-task 2σ threshold",
    ["fixed before any pair examined"], "key", round_=True)
link(side("NF", "r"), side("TH", "l"))

box("TF", 166, 60, 56, r"Target features $\oplus$ source column",
    ["ΔAUPRC = with source − alone"])
link(side("C2", "r"), side("TF", "l"))

diamond("Q", 166, 28, 60, 17, "ΔAUPRC exceeds", "2σ of this target?")
link(side("TF", "b"), side("Q", "t"))
link(side("TH", "r"), side("Q", "l"))

box("KY", 228, 50, 44, "1,930 pairs · 7.7%", ["1,036 helped · 894 hurt"], "key")
box("NO", 228, 14, 50, "23,070 pairs · 92.3%",
    ["indistinguishable from seed noise"])
link(side("Q", "r"), side("KY", "l"), [(200, 28), (200, 50)], vfirst=False)
link(side("Q", "r"), side("NO", "l"), [(200, 28), (200, 14)], vfirst=False)
ax.text(196, 44, "yes", fontsize=BS, ha="right", va="center", color=INK)
ax.text(196, 20, "no", fontsize=BS, ha="right", va="center", color=INK)

box("RP", 302, 50, 64, "PAIRED REPLICATION",
    ["200 pairs × 10 seeds, both arms · 1.0% clear",
     "0 of 200 survive BH-FDR · macro Δ = +0.0001"], "key")
link(side("KY", "r"), side("RP", "l"))

box("OV", 302, 14, 54, "Overlap threshold",
    ["within target p = 0.73 · pooled",
     r"p = $2.7{\times}10^{-9}$ was Simpson's paradox"])
link(side("NO", "r"), side("OV", "l"))

box("C3", 374, 50, 58, "CONTROL 3",
    ["Direction prediction · folds grouped",
     "by target · 0.560 ± 0.123 (chance 0.500)"], "ctrl")
link(side("RP", "r"), side("C3", "l"))

# ancillary arms draw from the measurement and return to the conclusion
link(side("TF", "t"), (166, 82), head=False)
link((166, 82), (244, 82), head=False)
for k in ("ST", "SC", "VC"):
    link((244, 82), side(k, "l"), [(244, B[k][1])], vfirst=True)
link(side("ST", "r"), (308, 128), head=False)
link(side("SC", "r"), (308, 110), head=False)
link(side("VC", "r"), (308, 92), head=False)
link((308, 128), (308, 92), head=False)
for yy in (128, 110, 92):
    dot(308, yy)

# ================================================================ terminal
box("END", 374, 14, 62, "sd(ΔAUPRC) = 1.34 × single-task σ",
    ["the 2σ screening rule is a 1.5σ bound",
     "a +0.002 effect needs 190 seeds to detect"])
e_x, e_y, e_w, e_h = B["END"]
ax.add_patch(FancyBboxPatch((e_x - e_w / 2, e_y - e_h / 2), e_w, e_h,
                            boxstyle="round,pad=0,rounding_size=0.9",
                            facecolor=TERM_F, edgecolor=TERM_E,
                            linewidth=1.2, zorder=3.5))
for t in ax.texts:
    if t.get_text().startswith(("sd(ΔAUPRC)", "the 2σ screening",
                                "a +0.002 effect")):
        t.set_color(TERM_T); t.set_zorder(4.5); t.set_fontweight("bold")

link(side("C3", "b"), side("END", "t"))
link(side("OV", "r"), (344, 14), head=False)
link((308, 92), (344, 92), head=False)
link((344, 92), (344, 24), head=False)
link((344, 24), side("END", "l"))

# ---------------------------------------- fit the canvas to what was drawn
_xs = [0, 410]
_ys = [v for t in TOP.values() for v in t]
_y0, _y1 = min(_ys) - 5.0, max(_ys) + 5.0
ax.set_xlim(*_xs)
ax.set_ylim(_y0, _y1)
fig.set_size_inches(A.fw, A.fw * (_y1 - _y0) / (_xs[1] - _xs[0]))

# ------------------------------------------- collision and overflow audits
bad = []
for k1, (x1, y1, w1, h1) in B.items():
    for k2, (x2, y2, w2, h2) in B.items():
        if k1 >= k2:
            continue
        if abs(x1 - x2) < (w1 + w2) / 2 and abs(y1 - y2) < (h1 + h2) / 2:
            bad.append(("overlap", k1, k2))

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
            if tw > bw - 2.6:
                bad.append(("overflow", k, round(tw, 1), t.get_text()[:38]))
            break

print(*bad, sep="\n") if bad else print("audit: clean")
if os.environ.get("AUDIT_ONLY"):
    sys.exit(1 if bad else 0)

os.makedirs("figures", exist_ok=True)
out = A.out + ("_color" if A.color else "")
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=400, bbox_inches="tight", facecolor="white")
matplotlib.rcParams["svg.fonttype"] = "path"
fig.savefig(f"{out}.svg", bbox_inches="tight", facecolor="white")
matplotlib.rcParams["svg.fonttype"] = "none"
fig.savefig(f"{out}_editable.svg", bbox_inches="tight", facecolor="white")
print("wrote", out + ".png / .pdf / .svg / _editable.svg")

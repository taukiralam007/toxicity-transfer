#!/usr/bin/env python
"""Figure 1 - the measurement protocol.

A methods figure: what was done, in order. No results, no conclusion. Numbers
belong in the caption and in Figures 2-6; a box that carries a finding turns a
diagram into a slide.

    python pipeline_clean.py           # black and white (default)
    python pipeline_clean.py --color   # restrained colour

Box positions derive from line counts and a fixed clearance, and every label's
rendered width is measured against its box, so neither a collision nor an
overflow can appear silently.
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
ap.add_argument("--fw", type=float, default=6.6)
ap.add_argument("--out", default="figures/figure1")
A = ap.parse_args()

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "svg.fonttype": "path"})

if A.color:
    INK, LINE = "#101820", "#47535d"
    PLAIN_F, PLAIN_E, PLAIN_W = "#ffffff", "#47535d", 0.9
    CTRL_F, CTRL_E, CTRL_W = "#ffffff", "#8c3030", 1.9
    KEYF, KEY_E, KEY_W = "#eef2f6", "#47535d", 0.9
else:
    INK, LINE = "#000000", "#4a4a4a"
    PLAIN_F, PLAIN_E, PLAIN_W = "#ffffff", "#4a4a4a", 0.9
    CTRL_F, CTRL_E, CTRL_W = "#ffffff", "#000000", 1.8
    KEYF, KEY_E, KEY_W = "#eeeeee", "#4a4a4a", 0.9

W, HSPAN = 112.0, 143.0
fig, ax = plt.subplots(figsize=(A.fw, A.fw * HSPAN / W))
ax.set_xlim(-4, 108)
ax.set_ylim(-22, 121)
ax.axis("off")
ax.set_position([0, 0, 1, 1])

HS, BS = 7.6, 7.0          # head, body
LH, HG, PAD = 2.5, 0.9, 4.6   # line height, head gap, vertical padding
GAP = 4.6                  # clearance between stacked boxes - generous
AH_L, AH_W, LW = 1.3, 0.52, 1.0
B, TOP = {}, {}


def height(n):
    return PAD + (HG + LH * n if n else 0) + LH


def place(key, n, top=None, below=None, gap=GAP, align=None):
    h = height(n)
    if align is not None:
        ks = [align] if isinstance(align, str) else list(align)
        c = sum(TOP[k][0] + TOP[k][1] for k in ks) / (2 * len(ks))
        top = c + h / 2
    elif top is None:
        ks = [below] if isinstance(below, str) else list(below)
        top = min(TOP[k][1] for k in ks) - gap
    TOP[key] = (top, top - h)
    return top - h / 2


def box(key, x, y, w, head, body=(), kind="plain"):
    fc, ec, lw = {"plain": (PLAIN_F, PLAIN_E, PLAIN_W),
                  "ctrl":  (CTRL_F, CTRL_E, CTRL_W),
                  "key":   (KEYF, KEY_E, KEY_W)}[kind]
    h = height(len(body))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=3, mutation_aspect=1))
    top = y + h / 2 - PAD / 2 - LH / 2
    ax.text(x, top, head, ha="center", va="center", fontsize=HS,
            fontweight="bold", color=INK, zorder=4)
    for i, t in enumerate(body):
        ax.text(x, top - HG - (i + 1) * LH, t, ha="center", va="center",
                fontsize=BS, color=INK, zorder=4)
    B[key] = (x, y, w, h)


def diamond(key, x, y, w, h, l1, l2):
    ax.add_patch(Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2],
                          [x - w / 2, y]], closed=True, facecolor=PLAIN_F,
                         edgecolor=INK, linewidth=1.1, zorder=3))
    ax.text(x, y + 1.5, l1, ha="center", va="center", fontsize=BS,
            fontweight="bold", color=INK, zorder=4)
    ax.text(x, y - 1.6, l2, ha="center", va="center", fontsize=BS,
            fontweight="bold", color=INK, zorder=4)
    B[key] = (x, y, w * 0.5, h)
    TOP[key] = (y + h / 2, y - h / 2)


def side(k, s):
    x, y, w, h = B[k]
    return {"t": (x, y + h / 2), "b": (x, y - h / 2),
            "l": (x - w / 2, y), "r": (x + w / 2, y)}[s]


def link(p0, p1, waypoints=(), head=True):
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
    ax.add_patch(Circle((x, y), 0.55, facecolor=LINE, edgecolor="none",
                        zorder=5))


# ------------------------------------------------------------------ corpus
box("D", 52, place("D", 1, top=117), 52, "Eight public collections",
    ["297 assays · 339,249 compounds"], "key")

box("C1", 52, place("C1", 1, below="D"), 58, "Control 1 · Scaffold assignment",
    ["Bemis–Murcko scaffolds over the union of all tasks"], "ctrl")
link(side("D", "b"), side("C1", "t"))

box("SP", 52, place("SP", 0, below="C1"), 44, "Scaffold-disjoint split")
link(side("C1", "b"), side("SP", "t"))

# --------------------------------------------------- the two arms, measured
y_rail = TOP["SP"][1] - GAP / 2
box("NF", 24, place("NF", 1, below="SP"), 42, "Single-task baseline",
    ["20 seeds; nothing else varied"])
box("SM", 80, place("SM", 1, align="NF"), 42, "Source model",
    ["random forest · ECFP6"])
link(side("SP", "b"), (52, y_rail), head=False)
link((24, y_rail), (80, y_rail), head=False)
dot(52, y_rail)
link((24, y_rail), side("NF", "t"))
link((80, y_rail), side("SM", "t"))

box("TH", 24, place("TH", 1, below="NF"), 42, "Per-task noise floor",
    ["2σ, fixed before any pair"], "key")
link(side("NF", "b"), side("TH", "t"))

box("C2", 80, place("C2", 1, align="TH"), 44,
    "Control 2 · Out-of-fold prediction",
    ["source never predicts its own labels in sample"], "ctrl")
link(side("SM", "b"), side("C2", "t"))

box("TF", 80, place("TF", 1, below="C2"), 44, "Transfer measurement",
    ["ΔAUPRC = with source − alone"])
link(side("C2", "b"), side("TF", "t"))

# ------------------------------------------------------------------ test
box("RP", 52, place("RP", 1, below=("TH", "TF")), 50, "Paired replication",
    ["both arms across the same seeds"])
link(side("TH", "b"), side("RP", "t"), [(24, TOP["RP"][0] + GAP / 2),
                                        (52, TOP["RP"][0] + GAP / 2)])
link(side("TF", "b"), side("RP", "t"), [(80, TOP["RP"][0] + GAP / 2),
                                        (52, TOP["RP"][0] + GAP / 2)])

box("ST", 52, place("ST", 1, below="RP"), 56, "Wilcoxon per pair",
    ["Benjamini–Hochberg across pairs · bootstrap CI per target"])
link(side("RP", "b"), side("ST", "t"))

box("C3", 52, place("C3", 1, below="ST"), 54, "Control 3 · Direction prediction",
    ["cross-validation folds grouped by target"], "ctrl")
link(side("ST", "b"), side("C3", "t"))

# ---------------------------------------------------------- legend, bottom
lx, ly = -2, TOP["C3"][1] - 5.0
ax.add_patch(FancyBboxPatch((lx, ly - 1.1), 3.0, 2.2,
                            boxstyle="round,pad=0,rounding_size=0.5",
                            facecolor=CTRL_F, edgecolor=CTRL_E,
                            linewidth=CTRL_W, zorder=4))
ax.text(lx + 4.2, ly, "leakage control", fontsize=6.6, va="center", color=INK)
ax.add_patch(FancyBboxPatch((lx + 31, ly - 1.1), 3.0, 2.2,
                            boxstyle="round,pad=0,rounding_size=0.5",
                            facecolor=KEYF, edgecolor=KEY_E,
                            linewidth=KEY_W, zorder=4))
ax.text(lx + 35.2, ly, "quantity the analysis turns on",
        fontsize=6.6, va="center", color=INK)

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
            if tw > bw - 3.0:
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

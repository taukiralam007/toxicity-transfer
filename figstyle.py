"""Shared figure style, so every figure in the paper reads as one set.

Import at the top of a plotting script, before any figure is created:

    from figstyle import INK, LINE, FILL, POS, NEG, ACC, save

The values here are the ones measured off the pipeline figure, not chosen by
eye: strokes are #414141 rather than black, so that the few genuinely black
marks (the leakage controls, the terminal) carry the emphasis; fills are
#d9d9d9. Liberation Sans is metric-compatible with Arial, which is what most
chemistry journals set their figures in.

Greyscale-first on purpose. Figures must survive a black-and-white print, and
a palette that only works in colour hides that it is doing the work.
"""
import os

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
INK = "#000000"     # text, and the few marks that must dominate
LINE = "#414141"    # rules, axes, connectors - never pure black
FILL = "#d9d9d9"    # neutral fill: histogram bars, bands
POS = "#9a9a9a"     # "helped" - lighter grey, reads as background
NEG = "#2b2b2b"     # "hurt" - near-black, reads as foreground
ACC = "#555555"     # single-series emphasis

FONT = "Liberation Sans"

plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",

    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.facecolor": "white",

    "font.size": 7.6,
    "axes.titlesize": 8.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 7.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,

    "axes.edgecolor": LINE,
    "axes.linewidth": 0.9,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": LINE,
    "ytick.color": LINE,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,
    "patch.linewidth": 0.8,
    "figure.autolayout": True,
    "svg.fonttype": "path",
})


def save(fig, name, out="figures", exts=("png", "pdf", "svg")):
    """Write a figure in every format the submission needs."""
    os.makedirs(out, exist_ok=True)
    for ext in exts:
        if ext == "svg":
            matplotlib.rcParams["svg.fonttype"] = "path"
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}/{name}." + "/".join(exts))

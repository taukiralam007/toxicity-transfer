import os
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures")
from math import ceil
from statistics import NormalDist
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, RED, GREY = "#3D6FA8", "#E08A2E", "#C14840", "#9A9A9A"
INK, AX = "#111111", "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 7.6, "axes.labelsize": 8.0, "text.color": INK,
    "axes.labelcolor": INK, "axes.labelpad": 3.5,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "axes.edgecolor": AX, "axes.linewidth": 0.8,
    "xtick.color": AX, "ytick.color": AX,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.2,
    "legend.handlelength": 1.4, "legend.handletextpad": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

SIG_D, POWER = 0.00611, 0.80          # median sd(dAUPRC), 200 pairs x 10 seeds
z = NormalDist().inv_cdf
def seeds(d, alpha):
    return ((z(1 - alpha / 2) + z(POWER)) * SIG_D / d) ** 2

fig, ax = plt.subplots(figsize=(5.0, 3.1))
dd = np.logspace(np.log10(6e-4), np.log10(0.06), 400)
s1 = np.array([seeds(d, 0.05) for d in dd])
s2 = np.array([seeds(d, 0.05 / 200) for d in dd])

# the band between a single pair and a 200-pair screen
ax.fill_between(dd, s1, s2, color=BLUE, alpha=0.13, lw=0, zorder=2)
ax.plot(dd, s1, "-", color=BLUE, lw=2.0, zorder=4, label="single pair")
ax.plot(dd, s2, "--", color=ORANGE, lw=2.0, zorder=4,
        label="corrected for 200 pairs (Bonferroni)")

# the region no study will ever run
ax.axhspan(100, 3e3, color=RED, alpha=0.06, lw=0, zorder=1)
ax.text(5.3e-2, 1.25e2, "beyond any realistic budget", ha="right", va="bottom",
        fontsize=6.9, color=RED)

ax.axhline(1, color=GREY, lw=0.9, ls=":", zorder=3)
ax.text(6.4e-4, 1.12, "1 seed — current practice", ha="left", va="bottom",
        fontsize=6.9, color=AX)

ax.axvline(0.0101, color=GREY, lw=0.9, ls=":", zorder=3)
ax.scatter([0.0101], [seeds(0.0101, 0.05)], s=42, color=RED, zorder=6,
           edgecolor="white", lw=0.9)
ax.annotate("+0.0101\npublished stacking gain\n(3 seeds, or 8 when screened)",
            xy=(0.0101, seeds(0.0101, 0.05)), xytext=(0.0122, 42),
            fontsize=6.9, color=INK, va="top", ha="left",
            arrowprops=dict(arrowstyle="-", color=AX, lw=0.7))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.75, 3e3)
ax.set_xlabel("true effect size (ΔAUPRC)")
ax.set_ylabel("seeds per arm, 80% power")
ax.set_xlim(5e-4, 7e-2)
ax.legend(fontsize=7.2, loc="upper right", bbox_to_anchor=(1.0, 1.0),
          facecolor="white", framealpha=0.9, edgecolor="none", frameon=True)
ax.grid(axis="y", color="#DEDEDE", lw=0.55, zorder=0); ax.set_axisbelow(True)

fig.tight_layout()
os.makedirs(_OUT, exist_ok=True)
for e in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(_OUT, f"fig4_seedbudget.{e}"), dpi=600,
                bbox_inches="tight", facecolor="white")
print("wrote", os.path.join(_OUT, "fig4_seedbudget.png"))
for d in (0.02, 0.0101, 0.005, 0.002, 0.001, 0.0001):
    print(f"  {d:.4f}: {ceil(seeds(d,0.05)):>7}  {ceil(seeds(d,0.05/200)):>7}")

#!/usr/bin/env python3
"""Figure 3 - the screening rule is mis-calibrated.

Run:  python fig3_calibration.py [results_dir]
Writes figures/fig3_calibration.(png|pdf|svg) and prints every plotted number.
"""
from style import *   # noqa: F401,F403

# ======================================== FIG 3  the screening rule
fig = plt.figure(figsize=(7.0, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.80, 1.0], wspace=0.52)

# a — effect distribution against the floor
ax = fig.add_subplot(gs[0])
d = at.delta_auprc.values
lim = np.percentile(np.abs(d), 99.3)
ax.hist(d, bins=np.linspace(-lim, lim, 90), color=BLUE, alpha=0.85,
        edgecolor="white", lw=0.25, zorder=3)
mnf = nf.noise2s.median()
ax.axvspan(-mnf, mnf, color=ORANGE, alpha=0.17, zorder=2)
ax.axvline(0, color=INK, lw=0.9, zorder=4)
ax.set_xlabel("ΔAUPRC  (source added − target alone)")
ax.set_ylabel("source–target pairs")
ax.set_xlim(-lim, lim)
ax.text(mnf * 1.08, ax.get_ylim()[1] * 0.93,
        f"median per-task\nnoise floor  ±{mnf:.3f}", fontsize=6.5, color="#9a5a12")
ax.text(-lim * 0.97, ax.get_ylim()[1] * 0.93,
        f"n = {len(at):,} pairs\nmean = {d.mean():+.4f}", fontsize=6.1,
        color=INK, va="top")
tidy(ax); panel(ax, "a", dx=-0.16)

# b — where the mis-calibration comes from
ax = fig.add_subplot(gs[1])
vals = [SD_SINGLE, SD_DIFF]
bars = ax.bar([0, 1], vals, 0.56, color=[GREY, RED],
              edgecolor="white", lw=0.7, zorder=3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["σ\nsingle-task", "sd(ΔAUPRC)\nthresholded"])
ax.set_ylabel("standard deviation (AUPRC)")
ax.set_ylim(0, max(vals) * 1.42)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.035, f"{v:.5f}",
            ha="center", fontsize=6.2, color=INK)
y = max(vals) * 1.22
ax.annotate("", xy=(1, y), xytext=(0, y),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
ax.text(0.5, y * 1.02, "× 1.34", ha="center", fontsize=8.2,
        fontweight="bold", color=RED)
tidy(ax); panel(ax, "b", dx=-0.30)

# c — what survives each stage
ax = fig.add_subplot(gs[2])
stage = ["2σ rule\nsingle run", "2σ rule\nseed-\naveraged", "BH-FDR\nq < 0.10"]
pct = [100 * at.cleared.mean(), PCT_SEED_AVG, PCT_BH]
cols = [BLUE, ORANGE, RED]
bars = ax.bar(range(3), pct, 0.56, color=cols, edgecolor="white", lw=0.7, zorder=3)
ax.axhline(100 * 2 * (1 - stats.norm.cdf(2 / (SD_DIFF / SD_SINGLE))), color=INK, lw=0.9, ls=(0, (4, 2.5)), zorder=4)
ax.text(2.46, 14.6, "expected by chance under\na 1.5σ bound  (14%)",
        fontsize=6.5, color=INK, ha="right", va="bottom")
ax.set_xticks(range(3)); ax.set_xticklabels(stage)
ax.set_ylabel("pairs declared transferring (%)")
ax.set_ylim(0, 18)
for b, v in zip(bars, pct):
    ax.text(b.get_x() + b.get_width()/2, v + 0.45,
            f"{v:.2f}%", ha="center", fontsize=6.9, color=INK)
tidy(ax); panel(ax, "c", dx=-0.26)

save(fig, "fig3_calibration")

# ----------------------------------------------------------- numbers drawn
print("\nFIGURE 3 - values plotted")
print(f"  a  pairs = {len(at):,}   mean dAUPRC = {at.delta_auprc.mean():+.6f}")
print(f"  a  shaded band = median per-task 2sigma = +/-{nf.noise2s.median():.4f}")
print(f"  b  single-task sigma = {SD_SINGLE:.5f}   sd(dAUPRC) = {SD_DIFF:.5f}"
      f"   ratio = {SD_DIFF/SD_SINGLE:.4f}        [EXTERNAL]")
_k = 2 / (SD_DIFF / SD_SINGLE)
print(f"  c  2 sigma_single = {_k:.3f} sd(delta); chance tail = "
      f"{100*2*(1-stats.norm.cdf(_k)):.2f}%")
print(f"  c  single run    = {100*at.cleared.mean():.2f}%  "
      f"({at.cleared.sum():,} of {len(at):,})")
print(f"  c  seed-averaged = {PCT_SEED_AVG:.2f}%                      [EXTERNAL]")
print(f"  c  BH q<0.10     = {PCT_BH:.2f}%                      [EXTERNAL]")

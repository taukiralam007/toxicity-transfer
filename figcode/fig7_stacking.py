#!/usr/bin/env python3
"""Figure 7 - stacking and screening.

Run:  python fig7_stacking.py [results_dir]
Writes figures/fig7_stacking.(png|pdf|svg) and prints every plotted number.
"""
from style import *   # noqa: F401,F403

# ====================================== FIG 7  stacking and screening
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                         gridspec_kw=dict(wspace=0.30))

# a — stacking k
ax = axes[0]
ks = sorted(ms.k.unique())
med = [ms[ms.k == k].delta_auprc.abs().median() for k in ks]
mean = [ms[ms.k == k].delta_auprc.mean() for k in ks]
pc = [100 * (ms[ms.k == k].delta_auprc.abs() > ms[ms.k == k].noise2s).mean()
      for k in ks]
x = np.arange(len(ks))
bars = ax.bar(x, med, 0.56, color=PURPLE, edgecolor="white", lw=0.7,
              zorder=3, label="median |ΔAUPRC|  (spread)")
ax.plot(x, mean, "o-", color=RED, lw=1.5, ms=4.2, zorder=5,
        label="mean ΔAUPRC  (centre)")
ax.axhline(0, color=INK, lw=0.9, zorder=4)
for b, v, p in zip(bars, med, pc):
    ax.text(b.get_x() + b.get_width()/2, v + 0.0005, f"{v:.4f}",
            ha="center", fontsize=6.5, color=INK)
    ax.text(b.get_x() + b.get_width()/2, v * 0.10, f"{p:.0f}%\ncleared",
            ha="center", fontsize=6.2, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"k = {k}" for k in ks])
ax.set_xlabel("simultaneous sources")
ax.set_ylabel("ΔAUPRC")
ax.set_ylim(min(mean) * 1.35, max(med) * 1.22)
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          fontsize=6.9, columnspacing=1.4)
tidy(ax); panel(ax, "a", dx=-0.17)

# b — screening arms
ax = axes[1]
arms = [("gain_all", "all sources"), ("gain_screened", "screened sources"),
        ("gain_random_k", "random k\n(feature control)")]
vals = [sc[k].mean() for k, _ in arms]
cols = [RED if v < 0 else BLUE for v in vals]
bars = ax.bar(range(3), vals, 0.56, color=cols, edgecolor="white", lw=0.7,
              zorder=3)
for a, (k, _) in enumerate(arms):
    v = sc[k].values
    ax.scatter(np.random.default_rng(a).normal(a, 0.085, len(v)), v,
               s=7, color=INK, alpha=0.42, lw=0, zorder=5)
ax.axhline(0, color=INK, lw=1.0, zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in arms])
ax.set_ylabel("ΔAUPRC vs single-task")
for b, v in zip(bars, vals):
    ax.annotate(f"{v:+.4f}",
                xy=(b.get_x() + b.get_width()/2, v),
                xytext=(b.get_x() + b.get_width()/2 + 0.34,
                        v + (0.016 if v >= 0 else -0.016)),
                ha="center", va="center", fontsize=6.9, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.4),
                arrowprops=dict(arrowstyle="-", color=AX, lw=0.7))
ax.text(0.02, 0.03, f"n = {len(sc)} targets · no paired test significant",
        transform=ax.transAxes, fontsize=6.1, color=AX)
tidy(ax); panel(ax, "b", dx=-0.19)

save(fig, "fig7_stacking")

# ----------------------------------------------------------- numbers drawn
print("\nFIGURE 7 - values plotted")
print("  a  stacking:")
for _k, _m, _mu, _p in zip(ks, med, mean, pc):
    print(f"       k={_k:<3} n={len(ms[ms.k==_k]):<4} median|d|={_m:.4f}  "
          f"mean d={_mu:+.4f}  cleared={_p:.2f}%")
print("  b  screening (n = %d targets):" % len(sc))
for (_k, _l), _v in zip(arms, vals):
    _w = stats.wilcoxon(sc[_k]).pvalue
    print(f"       {_k:<15} mean={_v:+.4f}   Wilcoxon vs 0  p={_w:.3f}")

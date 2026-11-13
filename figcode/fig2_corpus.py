#!/usr/bin/env python3
"""Figure 2 - corpus and per-task noise floor.

Run:  python fig2_corpus.py [results_dir]
Writes figures/fig2_corpus.(png|pdf|svg) and prints every plotted number.
"""
from style import *   # noqa: F401,F403

# ======================================================= FIG 2  corpus
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))

# a — tasks per collection
ax = axes[0]
n = nf.collection.value_counts().reindex(order)
bars = ax.barh([NICE[c] for c in order], n.values,
               color=[COLL[c] for c in order], edgecolor="white", lw=0.7, zorder=3)
for b, v in zip(bars, n.values):
    ax.text(v + 3, b.get_y() + b.get_height() / 2, str(v),
            va="center", fontsize=6.2, color=INK)
ax.set_xlabel("tasks"); ax.set_xlim(0, n.max() * 1.22)
ax.invert_yaxis()
tidy(ax, "x"); panel(ax, "a", dx=-0.34)

# b — baseline performance
ax = axes[1]
g = nf.groupby("collection")[["auroc", "auprc"]].mean().reindex(order)
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, g.auroc, w, color=BLUE, edgecolor="white", lw=0.6,
       label="ROC-AUC", zorder=3)
ax.bar(x + w/2, g.auprc, w, color=ORANGE, edgecolor="white", lw=0.6,
       label="AUPRC", zorder=3)
ax.axhline(0.5, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=4,
           label="chance (ROC-AUC)")
ax.set_xticks(x); ax.set_xticklabels([NICE[c] for c in order],
                                     rotation=38, ha="right")
ax.set_ylabel("mean score"); ax.set_ylim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.58, 1.0),
          fontsize=6.4, columnspacing=1.0, handlelength=1.2)
tidy(ax); panel(ax, "b", dx=-0.30)

# c — the noise floor itself
ax = axes[2]
data = [nf[nf.collection == c].noise2s.values for c in order]
bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color=INK, lw=1.3),
                whiskerprops=dict(color=AX, lw=0.8),
                capprops=dict(color=AX, lw=0.8),
                boxprops=dict(lw=0.7, edgecolor="white"))
for b, c in zip(bp["boxes"], order):
    b.set_facecolor(COLL[c]); b.set_alpha(0.82)
for i, c in enumerate(order):
    v = nf[nf.collection == c].noise2s.values
    ax.scatter(np.random.default_rng(i).normal(i + 1, 0.075, len(v)), v,
               s=2.8, color=INK, alpha=0.38, lw=0, zorder=4)
ax.set_xticklabels([NICE[c] for c in order], rotation=38, ha="right")
ax.set_ylabel("noise floor (2σ AUPRC)")
tidy(ax); panel(ax, "c", dx=-0.27)

fig.tight_layout(w_pad=1.3)
save(fig, "fig2_corpus")

# ----------------------------------------------------------- numbers drawn
print("\nFIGURE 2 - values plotted")
print("  panel a/b/c, by collection (ordered as drawn):")
_g = nf.groupby("collection").agg(tasks=("task", "size"), auroc=("auroc", "mean"),
                                  auprc=("auprc", "mean"),
                                  n2s_med=("noise2s", "median"),
                                  n2s_mean=("noise2s", "mean")).reindex(order)
print(_g.round(4).to_string())
print(f"  total tasks = {len(nf)}   median 2sigma overall = {nf.noise2s.median():.4f}")
print(f"  widest/narrowest median 2sigma = "
      f"{_g.n2s_med.max()/_g.n2s_med.min():.1f}x")

#!/usr/bin/env python3
"""Figure 6 - overlap relationship is an artefact of pooling.

Run:  python fig6_overlap.py [results_dir]
Writes figures/fig6_overlap.(png|pdf|svg) and prints every plotted number.
"""
from style import *   # noqa: F401,F403

# ================================= FIG 6  overlap and Simpson's paradox
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9),
                         gridspec_kw=dict(wspace=0.42))

CUT = 0.02
cl = cl.copy()
cl["hi"] = cl.eff_overlap >= CUT
cl["pos"] = cl.delta_auprc > 0

# a — pooled: looks like a real dose-response
ax = axes[0]
bo = pd.read_csv(f"{R}/boundary_by_overlap.csv")
bo = bo[bo.n_cleared >= 20]
xs = np.arange(len(bo))
bars = ax.bar(xs, bo.pct_positive, 0.66,
              color=[BLUE if v < 55 else ORANGE for v in bo.pct_positive],
              edgecolor="white", lw=0.7, zorder=3)
ax.axhline(50, color=INK, lw=1.0, ls=(0, (4, 2.5)), zorder=4)
ax.set_xticks(xs)
ax.set_xticklabels([t.replace("-", "–") for t in bo.overlap],
                   rotation=45, ha="right")
ax.set_xlabel("effective compound overlap")
ax.set_ylabel("positive among cleared (%)")
ax.set_ylim(0, 90)
for b, v, nn in zip(bars, bo.pct_positive, bo.n_cleared):
    ax.text(b.get_x() + b.get_width()/2, v + 1.8, f"{v:.0f}",
            ha="center", fontsize=6.7, color=INK)
    ax.text(b.get_x() + b.get_width()/2, 2.2, f"n = {nn}", ha="center",
            va="bottom", rotation=90, fontsize=5.9, color="white",
            fontweight="bold")
ax.text(0.02, 0.97, "pooled across collections\n" r"$p = 2.7\times10^{-9}$",
        transform=ax.transAxes, fontsize=6.2, va="top", color=INK)
tidy(ax); panel(ax, "a", dx=-0.21)

# b — disaggregated: the effect is not there
ax = axes[1]
keep = [c for c in order
        if ((cl.target_collection.map(lambda t: True)) & True).any()]
sub = []
for c, g in cl.groupby("target_collection"):
    lo, hi = g[~g.hi], g[g.hi]
    if len(lo) >= 25 and len(hi) >= 25:
        sub.append((c, 100*lo.pos.mean(), 100*hi.pos.mean(), len(lo), len(hi)))
sub.sort(key=lambda r: r[1])
ends = sorted((b for _, _, b, _, _ in sub))
place, last = {}, -1e9
span = (max(ends) - min(ends)) if len(ends) > 1 else 1.0
for e in ends:
    y = e if e - last > span * 0.055 else last + span * 0.055
    place[e] = y; last = y
for i, (c, a, b, nl, nh) in enumerate(sub):
    col = COLL.get(c, GREY)
    ax.plot([0, 1], [a, b], "-", color=col, lw=1.3, zorder=3, alpha=0.9)
    ax.scatter([0, 1], [a, b], s=24, color=col, zorder=4,
               edgecolor="white", lw=0.7)
    ax.text(1.08, place[b], f"{NICE.get(c, c)}  ({nl}/{nh})",
            fontsize=6.5, va="center", color=col,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.2))
ax.axhline(50, color=INK, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
ax.set_xlim(-0.14, 2.30); ax.set_xticks([0, 1])
ax.set_xticklabels([f"< {CUT}", f"≥ {CUT}"])
ax.set_xlabel("effective compound overlap")
ax.set_ylabel("positive among cleared (%)")
ax.text(0.03, 0.46, "within collection\nnone significant",
        transform=ax.transAxes, fontsize=6.2, va="top", color=INK)
tidy(ax); panel(ax, "b", dx=-0.21)

# c — within target: the decisive test
ax = axes[2]
pair = []
for t, g in cl.groupby("target"):
    lo, hi = g[~g.hi], g[g.hi]
    if len(lo) >= 5 and len(hi) >= 5:
        pair.append((lo.pos.mean(), hi.pos.mean()))
pair = np.array(pair)
w = stats.wilcoxon(pair[:, 0], pair[:, 1]) if len(pair) else None
nrise = int((pair[:, 1] > pair[:, 0]).sum()) if len(pair) else 0
nfall = int((pair[:, 1] < pair[:, 0]).sum()) if len(pair) else 0
pair = pair * 100
if len(pair):
    up = pair[:, 1] > pair[:, 0]
    for (a, b), u in zip(pair, up):
        ax.plot([0, 1], [a, b], "-", color=(ORANGE if u else BLUE),
                lw=0.9, alpha=0.5, zorder=3)
    ax.scatter(np.zeros(len(pair)), pair[:, 0], s=11, color=BLUE,
               zorder=4, lw=0, alpha=0.75)
    ax.scatter(np.ones(len(pair)), pair[:, 1], s=11, color=ORANGE,
               zorder=4, lw=0, alpha=0.75)
    ax.plot([0, 1], [pair[:, 0].mean(), pair[:, 1].mean()], "-",
            color=INK, lw=2.0, zorder=6)
    ax.scatter([0, 1], [pair[:, 0].mean(), pair[:, 1].mean()], s=34,
               color=INK, zorder=7, marker="D")
    ax.text(0.02, 0.985,
            f"within target  ·  {len(pair)} targets\n"
            f"{nrise} rise, {nfall} fall, {len(pair)-nrise-nfall} tied"
            f"\np = {w.pvalue:.2f}",
            transform=ax.transAxes, fontsize=6.7, va="top", color=INK,
            bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.6))
    ax.text(0.5, pair[:, 0].mean() + 9.0, f"mean of {len(pair)} targets", ha="center",
            fontsize=6.4, color=INK, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
ax.axhline(50, color=AX, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
ax.set_xlim(-0.18, 1.18); ax.set_xticks([0, 1])
ax.set_xticklabels([f"< {CUT}", f"≥ {CUT}"])
ax.set_xlabel("effective compound overlap")
ax.set_ylabel("positive rate within target (%)")
ax.set_ylim(-8, 152); ax.set_yticks([0, 20, 40, 60, 80, 100])
tidy(ax); panel(ax, "c", dx=-0.21)

save(fig, "fig6_overlap")

# ----------------------------------------------------------- numbers drawn
print("\nFIGURE 6 - values plotted")
print("  a  pooled, by overlap stratum:")
for _, _r in bo.iterrows():
    print(f"       {_r.overlap}   n={int(_r.n_cleared):>5}   "
          f"positive={_r.pct_positive:.2f}%")
_hi = cl[cl.hi]; _lo = cl[~cl.hi]
_bt = stats.binomtest(int(_hi.pos.sum()), len(_hi), 0.5)
print(f"       high stratum {100*_hi.pos.mean():.2f}% (n={len(_hi)})  "
      f"vs chance p={_bt.pvalue:.3g}")
print(f"       low  stratum {100*_lo.pos.mean():.2f}% (n={len(_lo)})")
print("  b  within collection (low% -> high%, n_low/n_high):")
for _c, _a, _b, _nl, _nh in sub:
    print(f"       {NICE.get(_c,_c):<11} {_a:.2f} -> {_b:.2f}   {_nl}/{_nh}")
print(f"  c  within target: {len(pair)} targets, "
      f"{nrise} rise, {nfall} fall, {len(pair)-nrise-nfall} tied")
print(f"       mean low={pair[:,0].mean():.2f}%  high={pair[:,1].mean():.2f}%  "
      f"Wilcoxon p={w.pvalue:.4f}")

#!/usr/bin/env python3
"""Figure 5 - dataset geometry does not predict direction.

Run:  python fig5_direction.py [results_dir]
Writes figures/fig5_direction.(png|pdf|svg) and prints every plotted number.
"""
from style import *   # noqa: F401,F403

BA_UNGROUPED, BA_GROUPED, BA_SD, BA_LOCO = direction_accuracies()
print(f"  ungrouped {BA_UNGROUPED:.3f} | grouped {BA_GROUPED:.3f}"
      f" +/- {BA_SD:.3f} | leave-one-collection-out {BA_LOCO:.3f}")

# ================================================ FIG 5  direction
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                         gridspec_kw=dict(width_ratios=[1.3, 1.0], wspace=0.36))

# a — descriptor correlations, signed bars
ax = axes[0]
desc = [("eff_overlap", "effective overlap"),
        ("exact_overlap", "exact overlap"),
        ("bridge_kappa", "bridging concordance κ"),
        ("mean_max_sim", "mean max similarity"),
        ("prevalence_shift", "prevalence shift"),
        ("target_imbalance", "target class imbalance"),
        ("log_n_target", "log target label count")]
rows = []
for k, lab in desc:
    s = at[[k, "delta_auprc"]].dropna()
    rho, p = stats.spearmanr(s[k], s.delta_auprc)
    rows.append((lab, rho, p, len(s)))
rows.sort(key=lambda r: r[1])
labs = [r[0] for r in rows]; rho = np.array([r[1] for r in rows])
cols = [BLUE if v > 0 else RED for v in rho]
ax.barh(range(len(rows)), rho, 0.62, color=cols, edgecolor="white",
        lw=0.6, zorder=3)
ax.axvline(0, color=INK, lw=0.9, zorder=4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labs)
ax.set_xlabel("Spearman ρ with ΔAUPRC")
m = max(abs(rho)) * 2.35
ax.set_xlim(-max(abs(rho)) * 1.25, m)
for i, (lab, r, p, nn) in enumerate(rows):
    off = m * 0.035
    ax.text(max(r, 0) + off, i,
            (f"ρ = {r:+.3f},  " + (r"$p = %s\times10^{%d}$" % tuple(f"{p:.1e}".split("e")[:1] + [int(f"{p:.1e}".split("e")[1])])
             if p < 1e-3 else f"p = {p:.2f}")), va="center",
            ha="left", fontsize=6.4, color=INK)
tidy(ax, "x"); panel(ax, "a", dx=-0.52)

# b — what grouping does to the estimate
ax = axes[1]
names = ["ungrouped\nfolds", "grouped\nby target", "leave-one-\ncollection-out"]
vals = [BA_UNGROUPED, BA_GROUPED, BA_LOCO]
errs = [0.0, BA_SD, 0.0]
cols = [RED, BLUE, GREEN]
bars = ax.bar(range(3), vals, 0.56, color=cols, edgecolor="white", lw=0.7,
              zorder=3)
ax.errorbar([1], [vals[1]], yerr=[errs[1]], fmt="none", ecolor=INK,
            lw=1.0, capsize=4, capthick=1.0, zorder=5)
ax.axhline(0.5, color=INK, lw=1.0, ls=(0, (4, 2.5)), zorder=4)
ax.text(2.97, 0.508, "chance", fontsize=6.4, color=INK, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels(names)
ax.set_ylabel("balanced accuracy (sign of Δ)")
ax.set_ylim(0.40, 1.0); ax.set_xlim(-0.55, 3.0)
for b, v, e in zip(bars, vals, errs):
    ax.text(b.get_x() + b.get_width()/2, v + e + 0.022, f"{v:.3f}",
            ha="center", fontsize=6.9, color=INK)
ax.text(0, 0.69, "target memorisation", ha="center", va="center", rotation=90,
        fontsize=6.5, color="white",
        fontweight="bold")
tidy(ax); panel(ax, "b", dx=-0.24)

save(fig, "fig5_direction")

# ----------------------------------------------------------- numbers drawn
print("\nFIGURE 5 - values plotted")
print("  a  Spearman rho vs dAUPRC:")
for _l, _r, _p, _n in rows:
    print(f"       {_l:<26} rho={_r:+.6f}  p={_p:.4g}  n={_n}")
print(f"  b  ungrouped folds        = {BA_UNGROUPED:.3f}")
print(f"  b  grouped by target      = {BA_GROUPED:.3f} +/- {BA_SD:.3f}")
print(f"  b  leave-one-collection   = {BA_LOCO:.3f}")
print(f"     (chance = 0.500; n cleared pairs = {len(cl):,}, "
      f"positives = {(cl.delta_auprc>0).mean():.3f})")

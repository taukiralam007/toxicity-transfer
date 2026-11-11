#!/usr/bin/env python3
"""Two additions that turn a null result into a measured one. Both run on
results/analysis_table.csv -- no retraining, seconds not hours."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" if (ROOT / "results").exists() else ROOT

FEATS = ["eff_overlap", "exact_overlap", "bridge_kappa", "mean_max_sim",
         "prevalence_shift", "target_imbalance", "log_n_target"]


def cv_balanced_accuracy(d, feats, min_n=60):
    """Cross-validated, never in-sample. Returns nan when underpowered."""
    sub = d[d.transferred == 1].copy()
    sub["sign"] = (sub.delta_auprc > 0).astype(int)
    use = [f for f in feats if f in sub and sub[f].notna().sum() > .8 * len(sub)]
    sub = sub.dropna(subset=use + ["sign"])
    if len(sub) < min_n or sub.sign.nunique() < 2:
        return np.nan
    if min(sub.sign.value_counts()) < 15:
        return np.nan
    X, y = sub[use].to_numpy(float), sub.sign.to_numpy()
    groups = sub["target"].to_numpy()
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                         learning_rate=.06, random_state=0)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    try:
        return float(cross_val_score(clf, X, y, cv=cv, groups=groups,
                                     scoring="balanced_accuracy").mean())
    except Exception:
        return np.nan


def part1_boundary(df):
    print("=" * 72)
    print("1. WHERE DOES TRANSFER BECOME PREDICTABLE?")
    print("=" * 72)
    edges = [0, .02, .05, .1, .2, .35, .5, .7, .9, 1.001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = df[(df.eff_overlap >= lo) & (df.eff_overlap < hi)]
        if len(d) < 50:
            continue
        res = d[d.transferred == 1]
        rows.append(dict(
            overlap=f"{lo:.2f}-{hi:.2f}", n_pairs=len(d), n_cleared=len(res),
            pct_cleared=100 * len(res) / len(d),
            pct_positive=100 * (res.delta_auprc > 0).mean() if len(res) else np.nan,
            mean_delta=d.delta_auprc.mean(),
            cv_bal_acc=cv_balanced_accuracy(d, FEATS)))
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    ok = out.dropna(subset=["cv_bal_acc"])
    good = ok[ok.cv_bal_acc >= .60]
    print()
    if good.empty:
        print("  No overlap stratum reaches 0.60 cross-validated balanced")
        print("  accuracy. The boundary, if it exists, lies above the range")
        print("  this corpus covers -- which is the range toxicology data")
        print("  occupies. That is the reportable finding.")
    else:
        print(f"  Predictability appears from overlap {good.overlap.iloc[0]} "
              f"upward (CV balanced accuracy {good.cv_bal_acc.iloc[0]:.3f}).")
        print("  That is the measured boundary -- report it against the")
        print("  30-40% figure derived synthetically in the 2026 analysis.")
    return out


def part2_imbalance(df):
    print("\n" + "=" * 72)
    print("2. THE ONE DESCRIPTOR THAT WAS NOT FLAT: target_imbalance")
    print("=" * 72)
    s = df[["target_imbalance", "delta_auprc"]].dropna()
    r = spearmanr(s.target_imbalance, s.delta_auprc)
    print(f"  overall: rho = {r.statistic:+.4f}  p = {r.pvalue:.3g}  n = {len(s)}")
    print("  (target_imbalance = min(positive rate, 1 - positive rate);")
    print("   SMALL means a rare-positive, hard, data-poor target)\n")
    q = pd.qcut(df.target_imbalance, 5, duplicates="drop")
    g = df.groupby(q, observed=True).agg(
        n=("delta_auprc", "size"),
        median_pos_rate=("target_imbalance", "median"),
        mean_delta=("delta_auprc", "mean"),
        pct_cleared=("transferred", lambda v: 100 * v.mean()))
    print(g.round(4).to_string())
    med = df.target_imbalance.median()
    rare = df[df.target_imbalance <= med].delta_auprc.dropna()
    bal = df[df.target_imbalance > med].delta_auprc.dropna()
    if len(rare) > 50 and len(bal) > 50:
        u = mannwhitneyu(rare, bal, alternative="two-sided")
        print(f"\n  rare-positive targets : mean delta {rare.mean():+.4f}  n={len(rare)}")
        print(f"  balanced targets      : mean delta {bal.mean():+.4f}  n={len(bal)}")
        print(f"  Mann-Whitney p = {u.pvalue:.3g}")
        if u.pvalue < .05:
            who = "rare-positive" if rare.mean() > bal.mean() else "balanced"
            print(f"\n  -> transfer favours {who} targets. This is an actionable")
            print("     rule: it tells practitioners when multitask learning is")
            print("     worth attempting at all.")
        else:
            print("\n  -> no directional difference; the correlation is real but")
            print("     too weak to act on. Report it and say so.")
    if "log_n_target" in df:
        small = df[df.log_n_target <= df.log_n_target.median()]
        large = df[df.log_n_target > df.log_n_target.median()]
        print(f"\n  by target size:")
        print(f"    small targets: mean delta {small.delta_auprc.mean():+.4f}  "
              f"cleared {100*small.transferred.mean():.1f}%  n={len(small)}")
        print(f"    large targets: mean delta {large.delta_auprc.mean():+.4f}  "
              f"cleared {100*large.transferred.mean():.1f}%  n={len(large)}")
    return g


def part3_sanity(df):
    print("\n" + "=" * 72)
    print("3. ABLATION, CROSS-VALIDATED (replaces the in-sample figure)")
    print("=" * 72)
    variants = {
        "all descriptors": FEATS,
        "effective overlap only": ["eff_overlap"],
        "exact overlap only": ["exact_overlap"],
        "bridging concordance only": ["bridge_kappa"],
        "target imbalance only": ["target_imbalance"],
    }
    for name, fs in variants.items():
        acc = cv_balanced_accuracy(df, fs)
        print(f"  {name:28s} {acc:.3f}" if np.isfinite(acc) else f"  {name:28s}   n/a")
    print("\n  0.50 is chance. Yesterday's 'full 0.953' was an in-sample fit")
    print("  and must not appear in the manuscript.")


def main():
    at = RES / "analysis_table.csv"
    if not at.exists():
        sys.exit(f"missing {at}")
    df = pd.read_csv(at)
    print(f"loaded {len(df):,} pairs, {df.transferred.sum():,} cleared the noise floor\n")
    b = part1_boundary(df)
    g = part2_imbalance(df)
    part3_sanity(df)
    b.to_csv(RES / "boundary_by_overlap.csv", index=False)
    g.to_csv(RES / "imbalance_strata.csv")
    print(f"\nwrote {RES/'boundary_by_overlap.csv'} and {RES/'imbalance_strata.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Empirical null for transfer effects: both arms replicated across seeds.

The pairwise sweep measures ONE transfer run against a 20-seed baseline mean,
then thresholds at 2 sigma of the baseline. That is a screening rule with an
unknown false-positive rate: the delta's own sampling distribution was never
measured, and it turns out to be far from normal (kurtosis ~467).

This script measures it. For a stratified subset of pairs it runs BOTH arms
across the same seeds, which gives

  * a paired per-pair test over seeds, with BH-FDR across pairs;
  * a target-level macro-average with a bootstrap CI, matching the unit of
    inference used by Wang et al. (Environ Sci Technol 2026), who report
    stacking at +0.0101 ROC-AUC (Holm-adjusted Wilcoxon p = 0.006);
  * the empirical clear-rate of the 2-sigma rule under matched seeds, i.e.
    what the 7.7% headline actually means.

Usage
    python seed_null.py --pairs 1000 --seeds 20
    python seed_null.py --pairs 200 --seeds 10     # quick look first
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import transfer_law as T           # reuse the published pipeline verbatim

RES = Path("results")


# ───────────────────────────────────────────────────────── one pair, one seed
def pair_once(src, tgt, fps, ncomp, seed):
    """Both arms at the same seed. Returns (single-task, transfer) AUPRC/AUROC."""
    base_clf = RandomForestClassifier(random_state=seed, **T.RF).fit(
        fps[tgt.tr_i], tgt.tr_y)
    pb = base_clf.predict_proba(fps[tgt.te_i])[:, 1]

    feat = T.source_feature(src, fps, ncomp, seed)
    Xtr = np.hstack([fps[tgt.tr_i], feat[tgt.tr_i, None]]).astype(np.float32)
    Xte = np.hstack([fps[tgt.te_i], feat[tgt.te_i, None]]).astype(np.float32)
    pt = RandomForestClassifier(random_state=seed, **T.RF).fit(
        Xtr, tgt.tr_y).predict_proba(Xte)[:, 1]

    return (average_precision_score(tgt.te_y, pb), roc_auc_score(tgt.te_y, pb),
            average_precision_score(tgt.te_y, pt), roc_auc_score(tgt.te_y, pt))


def pair_block(sname, tnames, idx, fps, ncomp, seeds):
    """All seeds for one source, reusing nothing across seeds (source feature
    depends on the seed, so it is rebuilt — that is the point)."""
    src = idx[sname]
    rows = []
    for tn in tnames:
        tgt = idx[tn]
        try:
            vals = [pair_once(src, tgt, fps, ncomp, s) for s in range(seeds)]
        except Exception as e:                       # a task can be degenerate
            print(f"  skip {sname}->{tn}: {type(e).__name__}", file=sys.stderr)
            continue
        v = np.asarray(vals)                          # (seeds, 4)
        rows.append(dict(source=sname, target=tn,
                         source_collection=src.coll, target_collection=tgt.coll,
                         base_ap=v[:, 0], base_auc=v[:, 1],
                         tr_ap=v[:, 2], tr_auc=v[:, 3]))
    return rows


# ───────────────────────────────────────────────────────────────── statistics
def analyse(rows, seeds, nf):
    recs = []
    for r in rows:
        d_ap = r["tr_ap"] - r["base_ap"]             # paired across seeds
        d_auc = r["tr_auc"] - r["base_auc"]
        # paired test over seeds; Wilcoxon is safer than t given the heavy tails
        if np.allclose(d_ap, 0):
            p = 1.0
        else:
            try:
                p = stats.wilcoxon(d_ap, zero_method="pratt").pvalue
            except ValueError:
                p = 1.0
        recs.append(dict(
            source=r["source"], target=r["target"],
            target_collection=r["target_collection"],
            mean_delta_ap=d_ap.mean(), sd_delta_ap=d_ap.std(ddof=1),
            mean_delta_auc=d_auc.mean(),
            base_ap=r["base_ap"].mean(), base_sd=r["base_ap"].std(ddof=1),
            p_pair=p))
    df = pd.DataFrame(recs)

    # BH-FDR across pairs
    order = np.argsort(df.p_pair.values)
    ranked = df.p_pair.values[order]
    m = len(ranked)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    df["q_pair"] = np.nan
    df.loc[df.index[order], "q_pair"] = np.clip(q, 0, 1)

    # what the original 2-sigma rule does under MATCHED seeds
    df = df.merge(nf[["task", "noise2s"]].rename(columns={"task": "target"}),
                  on="target", how="left")
    df["cleared_2s"] = (df.mean_delta_ap.abs() > df.noise2s).astype(int)
    return df


def macro(df, col="mean_delta_ap", B=2000, rng=0):
    """Target-level macro-average + paired Wilcoxon + bootstrap CI.
    This is Wang et al.'s unit of inference: one number per target."""
    per_t = df.groupby("target")[col].mean()
    obs = per_t.mean()
    r = np.random.default_rng(rng)
    v = per_t.to_numpy()
    boot = np.array([r.choice(v, len(v), replace=True).mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = stats.wilcoxon(v, zero_method="pratt").pvalue
    except ValueError:
        p = 1.0
    return obs, lo, hi, p, len(v), int((v > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()

    comp, tasks, fps = T.build_corpus()
    idx = T.index_tasks(tasks)
    nf = pd.read_csv(RES / "noise_floor.csv")
    at = pd.read_csv(RES / "analysis_table.csv")

    # stratify the subset by target collection and overlap decile so the
    # subset is not quietly dominated by ChEMBL
    at = at[at.source.isin(idx) & at.target.isin(idx)].copy()
    at["ov_bin"] = pd.qcut(at.eff_overlap, 5, labels=False, duplicates="drop")
    frac = a.pairs / len(at)
    sub = (at.groupby(["target_collection", "ov_bin"], group_keys=False)
             .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))),
                                       random_state=0)))
    sub = sub.sample(min(a.pairs, len(sub)), random_state=0)
    print(f"{len(sub)} pairs x {a.seeds} seeds x 2 arms")
    print(sub.target_collection.value_counts().to_string(), "\n")

    by_src = {}
    for s, t in zip(sub.source, sub.target):
        by_src.setdefault(s, []).append(t)

    out = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(pair_block)(s, t, idx, fps, len(comp), a.seeds)
        for s, t in by_src.items())
    rows = [r for blk in out for r in blk]
    print(f"\ncompleted {len(rows)} pairs")

    df = analyse(rows, a.seeds, nf)
    df.to_csv(RES / "seed_null.csv", index=False)

    print("\n" + "=" * 66)
    print("PER-PAIR, paired across seeds")
    print("=" * 66)
    for q in (0.05, 0.10):
        n = int((df.q_pair < q).sum())
        pos = int(((df.q_pair < q) & (df.mean_delta_ap > 0)).sum())
        print(f"  BH-FDR q<{q:<5} {n:5d}/{len(df)} pairs ({100*n/len(df):5.2f}%)"
              f"   {pos} positive, {n-pos} negative")
    print(f"\n  the original 2-sigma rule, under matched seeds: "
          f"{100*df.cleared_2s.mean():.2f}% of this subset")
    print(f"  (it was {100*at.transferred.mean():.2f}% on the full single-seed sweep)")

    print("\n  per-pair sd of delta vs per-task sigma of the baseline:")
    print(f"    median sd(delta over seeds) = {df.sd_delta_ap.median():.5f}")
    print(f"    median sigma(baseline)      = {df.base_sd.median():.5f}")
    print(f"    ratio                       = "
          f"{df.sd_delta_ap.median()/max(df.base_sd.median(),1e-9):.2f}"
          "   <- 2-sigma assumes this is ~1.0")

    print("\n" + "=" * 66)
    print("TARGET-LEVEL MACRO (Wang et al.'s unit of inference)")
    print("=" * 66)
    for name, col in (("AUPRC", "mean_delta_ap"), ("ROC-AUC", "mean_delta_auc")):
        obs, lo, hi, p, n, npos = macro(df, col)
        print(f"  macro delta {name:<8} {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"   {npos}/{n} targets improved   Wilcoxon p = {p:.4f}")
    print("\n  Wang et al. 2026, stacking, ROC-AUC: +0.0101 "
          "[+0.0053, +0.0148], 12/13 endpoints, p = 0.006")

    print(f"\nwrote {RES/'seed_null.csv'}")


if __name__ == "__main__":
    main()

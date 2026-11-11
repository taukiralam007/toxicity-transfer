#!/usr/bin/env python3
"""Closes the main hole in the pairwise result: we measured ONE source at a
time, while real multitask systems stack many at once."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

import transfer_law as T

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
K_VALUES = (1, 5, 20, 50)


def multi_source_feature(srcs, fps, ncomp, seed=0):
    return np.column_stack([T.source_feature(s, fps, ncomp, seed) for s in srcs])


def one_target(tname, idx, fps, ncomp, base, rng_seed, k_values, seed=0):
    tgt = idx[tname]
    others = [n for n in idx if n != tname]
    rng = np.random.default_rng(rng_seed)
    pool = list(rng.permutation(others))
    rows, feats, built = [], None, 0
    for k in sorted(k_values):
        if k > len(pool):
            continue
        new = [idx[n] for n in pool[built:k]]
        block = multi_source_feature(new, fps, ncomp, seed)
        feats = block if feats is None else np.hstack([feats, block])
        built = k
        Xtr = np.hstack([fps[tgt.tr_i], feats[tgt.tr_i]]).astype(np.float32)
        Xte = np.hstack([fps[tgt.te_i], feats[tgt.te_i]]).astype(np.float32)
        p = RandomForestClassifier(random_state=seed, **T.RF).fit(Xtr, tgt.tr_y).predict_proba(Xte)[:, 1]
        ap = float(average_precision_score(tgt.te_y, p))
        rows.append(dict(target=tname, collection=tgt.coll, k=k, auprc=ap,
                         baseline=base[tname], delta_auprc=ap - base[tname]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()

    comp, tasks, fps = T.build_corpus()
    idx = T.index_tasks(tasks)
    print(f"modelling {len(idx)} tasks")

    nf_path = RES / "noise_floor.csv"
    if nf_path.exists():
        nf = pd.read_csv(nf_path); print(f"reusing noise floor from {nf_path}")
    else:
        print("no noise_floor.csv -- computing (slow)")
        nf = T.noise_floor(idx, fps, a.seeds)
        RES.mkdir(parents=True, exist_ok=True); nf.to_csv(nf_path, index=False)

    base = dict(zip(nf.task, nf.auprc))
    floor = dict(zip(nf.task, nf.noise2s))
    usable = [t for t in idx if t in base]

    byc = {}
    for t in usable:
        byc.setdefault(idx[t].coll, []).append(t)
    rng = np.random.default_rng(0)
    per = max(1, a.targets // len(byc))
    targets = []
    for c, lst in byc.items():
        pick = rng.choice(len(lst), min(per, len(lst)), replace=False)
        targets += [lst[i] for i in pick]
    targets = targets[:a.targets]
    print(f"targets: {len(targets)} across {len(byc)} collections | k = {K_VALUES}")

    out = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(one_target)(t, idx, fps, len(comp), base, 1000 + i, K_VALUES)
        for i, t in enumerate(targets))
    df = pd.DataFrame([r for chunk in out for r in chunk])
    df["noise2s"] = df["target"].map(floor)
    df["cleared"] = (df.delta_auprc.abs() > df.noise2s).astype(int)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "multisource.csv", index=False)

    print("\n=== effect size vs number of sources ===")
    g = df.groupby("k").agg(
        n=("delta_auprc", "size"), mean_delta=("delta_auprc", "mean"),
        median_abs=("delta_auprc", lambda v: v.abs().median()),
        pct_cleared=("cleared", "mean"),
        pct_positive=("delta_auprc", lambda v: (v > 0).mean()))
    g["pct_cleared"] = (g["pct_cleared"] * 100).round(1)
    g["pct_positive"] = (g["pct_positive"] * 100).round(1)
    print(g.round(4).to_string())

    print("\n=== per collection, mean delta by k ===")
    print(df.pivot_table(index="collection", columns="k",
                         values="delta_auprc", aggfunc="mean").round(4).to_string())

    k_lo, k_hi = min(K_VALUES), max(K_VALUES)
    lo = df[df.k == k_lo].set_index("target").delta_auprc
    hi = df[df.k == k_hi].set_index("target").delta_auprc
    common = lo.index.intersection(hi.index)
    if len(common) >= 10:
        from scipy.stats import wilcoxon
        d_lo, d_hi = lo[common].abs(), hi[common].abs()
        st = wilcoxon(d_hi, d_lo)
        print(f"\npaired |delta| at k={k_hi} vs k={k_lo} over {len(common)} targets:")
        print(f"  median |delta| k={k_lo}: {d_lo.median():.4f}")
        print(f"  median |delta| k={k_hi}: {d_hi.median():.4f}")
        print(f"  Wilcoxon p = {st.pvalue:.4g}")
        grew = d_hi.median() > d_lo.median() and st.pvalue < 0.05
        print("\nVERDICT:", "effect GROWS with more sources -> pairwise result understates transfer"
              if grew else "effect does NOT grow -> negative result holds")
    print(f"\nwrote {RES/'multisource.csv'}")


if __name__ == "__main__":
    main()

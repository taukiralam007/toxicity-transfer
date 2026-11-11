#!/usr/bin/env python3
"""If which sources help cannot be predicted from dataset geometry, the
rational response is to stop predicting and start measuring -- cheaply."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import transfer_law as T

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def _fit_eval(X_tr, y_tr, X_te, y_te, seed=0):
    if len(np.unique(y_tr)) < 2:
        return float("nan")
    m = RandomForestClassifier(random_state=seed, **T.RF).fit(X_tr, y_tr)
    return float(average_precision_score(y_te, m.predict_proba(X_te)[:, 1]))


def screen_target(tname, idx, fps, ncomp, cand_names, seed=0, inner_frac=0.3):
    tgt = idx[tname]
    y = tgt.tr_y
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 6:
        return None
    try:
        itr, iva = train_test_split(np.arange(len(y)), test_size=inner_frac,
                                    stratify=y, random_state=seed)
    except ValueError:
        return None
    if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2:
        return None

    tr_i = tgt.tr_i
    base_inner = _fit_eval(fps[tr_i[itr]], y[itr], fps[tr_i[iva]], y[iva], seed)
    if not np.isfinite(base_inner):
        return None

    feats, deltas = {}, {}
    for cn in cand_names:
        f = T.source_feature(idx[cn], fps, ncomp, seed)
        feats[cn] = f
        Xtr = np.hstack([fps[tr_i[itr]], f[tr_i[itr], None]]).astype(np.float32)
        Xva = np.hstack([fps[tr_i[iva]], f[tr_i[iva], None]]).astype(np.float32)
        s = _fit_eval(Xtr, y[itr], Xva, y[iva], seed)
        deltas[cn] = (s - base_inner) if np.isfinite(s) else np.nan

    d = pd.Series(deltas).dropna()
    kept = list(d[d > 0].index)
    rng = np.random.default_rng(seed + 7)
    rnd = list(rng.choice(cand_names, size=min(len(kept), len(cand_names)),
                          replace=False)) if kept else []

    def arm(names):
        if not names:
            return _fit_eval(fps[tgt.tr_i], y, fps[tgt.te_i], tgt.te_y, seed)
        M = np.column_stack([feats[n] for n in names])
        Xtr = np.hstack([fps[tgt.tr_i], M[tgt.tr_i]]).astype(np.float32)
        Xte = np.hstack([fps[tgt.te_i], M[tgt.te_i]]).astype(np.float32)
        return _fit_eval(Xtr, y, Xte, tgt.te_y, seed)

    out = dict(target=tname, collection=tgt.coll,
               n_candidates=len(cand_names), n_kept=len(kept),
               single=arm([]), all=arm(cand_names),
               screened=arm(kept), random_k=arm(rnd),
               screen_mean_delta=float(d.mean()),
               screen_frac_positive=float((d > 0).mean()))
    for k in ("all", "screened", "random_k"):
        out[f"gain_{k}"] = out[k] - out["single"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=30)
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    comp, tasks, fps = T.build_corpus()
    idx = T.index_tasks(tasks)
    names = list(idx)
    print(f"modelling {len(names)} tasks")

    byc = {}
    for t in names:
        byc.setdefault(idx[t].coll, []).append(t)
    rng = np.random.default_rng(a.seed)
    per = max(1, a.targets // len(byc))
    targets = []
    for c, lst in byc.items():
        pick = rng.choice(len(lst), min(per, len(lst)), replace=False)
        targets += [lst[i] for i in pick]
    targets = targets[:a.targets]
    print(f"targets: {len(targets)} across {len(byc)} collections")
    print(f"candidates per target: {a.candidates}")

    jobs = []
    for i, t in enumerate(targets):
        pool = [n for n in names if n != t]
        r2 = np.random.default_rng(100 + i)
        cand = [pool[j] for j in r2.choice(len(pool),
                min(a.candidates, len(pool)), replace=False)]
        jobs.append((t, cand))

    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(screen_target)(t, idx, fps, len(comp), cand, a.seed)
        for t, cand in jobs)
    df = pd.DataFrame([r for r in rows if r])
    if df.empty:
        sys.exit("no usable targets")
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / "screening.csv", index=False)

    print("\n=== mean AUPRC by arm ===")
    print(df[["single", "all", "screened", "random_k"]].mean().round(4).to_string())
    print("\n=== gain over single-task ===")
    g = df[["gain_all", "gain_screened", "gain_random_k"]]
    print(g.mean().round(4).to_string())
    print("\nwin rate vs single-task:")
    print((g > 0).mean().round(3).to_string())

    from scipy.stats import wilcoxon
    print("\n=== paired tests (n = %d targets) ===" % len(df))
    for lab, x, y in (("screened vs all", df.screened, df["all"]),
                      ("screened vs random-k", df.screened, df.random_k),
                      ("screened vs single", df.screened, df.single),
                      ("all vs single", df["all"], df.single)):
        d = (x - y).dropna()
        if len(d) >= 8 and d.abs().sum() > 0:
            p = wilcoxon(x[d.index], y[d.index]).pvalue
            print(f"  {lab:24s} median diff {d.median():+.4f}   p = {p:.4g}")

    print("\n=== by collection: mean gain ===")
    print(df.groupby("collection")[["gain_all", "gain_screened"]].mean().round(4).to_string())
    print(f"\nsources kept: median {df.n_kept.median():.0f} of {df.n_candidates.iloc[0]}")

    scr, alla, rnd = df.gain_screened.mean(), df.gain_all.mean(), df.gain_random_k.mean()
    print("\nVERDICT:", end=" ")
    if scr > alla and scr > rnd and scr > 0:
        print("screening BEATS naive all-sources and random-k -> the method works")
    elif scr > alla and scr > 0:
        print("screening beats all-sources but not random-k -> gain may be feature count")
    else:
        print("screening does NOT beat the baselines -> further negative result")
    print(f"\nwrote {RES/'screening.csv'}")


if __name__ == "__main__":
    main()

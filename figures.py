#!/usr/bin/env python
"""Figures 1-4 for the transfer-law manuscript.

Reads whatever result CSVs exist under ./ and ./results, identifies them by
their columns rather than their filenames, and writes PNG + PDF to ./figures.
Any figure whose inputs are missing is skipped with a message, so a partial
run still produces what it can.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Style lives in figstyle.py so every figure in the paper matches the
# pipeline figure: same typeface, same greys, same line weights, greyscale
# first. Importing it sets rcParams.
from figstyle import INK, LINE, FILL, POS, NEG, ACC   # noqa: E402

MID = FILL


# ---------------------------------------------------------------- discovery

def load_all():
    paths = sorted(set(glob.glob("*.csv") + glob.glob("results/*.csv")
                       + glob.glob("results/**/*.csv", recursive=True)))
    frames = {}
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        if len(df):
            frames[p] = df
    return frames


def col(df, *names):
    """First column present whose lowercase name matches any candidate."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def pick(frames, required, forbidden=(), prefer=()):
    """Largest frame containing all `required` columns and none `forbidden`."""
    best, best_n = None, -1
    for path, df in frames.items():
        if any(col(df, r) is None for r in required):
            continue
        if any(col(df, f) is not None for f in forbidden):
            continue
        score = len(df) + (10 ** 9 if any(k in path.lower() for k in prefer) else 0)
        if score > best_n:
            best, best_n = (path, df), score
    return best


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.png")


# ---------------------------------------------------------------- figure 1

def figure1(frames):
    hit = pick(frames, ["delta_auprc"], prefer=("pair", "transfer", "sweep"))
    if hit is None:
        print("Figure 1: no frame with delta_auprc - skipped")
        return
    path, pairs = hit
    d = col(pairs, "delta_auprc")
    s2 = col(pairs, "noise_2sigma", "noise2sigma", "sigma2", "noise_floor")
    print(f"Figure 1: using {path} ({len(pairs)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # 1a - per-task noise floor
    ax = axes[0]
    if s2 is not None:
        tcol = col(pairs, "target", "target_task", "task")
        floors = (pairs.groupby(tcol)[s2].first() if tcol else pairs[s2]).dropna()
        ax.hist(floors, bins=40, color=MID, edgecolor="white", linewidth=0.4)
        med = float(np.median(floors))
        ax.axvline(med, color=INK, lw=1.2)
        ax.annotate(f"median 2σ = {med:.4f}", xy=(med, ax.get_ylim()[1] * 0.88),
                    xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8)
        ax.set_xlabel("per-task noise floor (2σ of AUPRC over 20 seeds)")
        ax.set_ylabel("tasks")
        ax.set_title("a   How much the metric moves on seed alone", loc="left")
    else:
        ax.axis("off")
        ax.text(.5, .5, "no noise-floor column found", ha="center", color=MID)

    # 1b - effect distribution, cleared highlighted
    ax = axes[1]
    dv = pairs[d].dropna()
    lo, hi = np.percentile(dv, [0.5, 99.5])
    bins = np.linspace(lo, hi, 70)
    ax.hist(dv, bins=bins, color=MID, edgecolor="white", linewidth=0.3,
            label="all pairs")
    if s2 is not None:
        cleared = pairs[pairs[d].abs() > pairs[s2]]
        ax.hist(cleared[cleared[d] > 0][d], bins=bins, color=POS,
                label="cleared, positive")
        ax.hist(cleared[cleared[d] < 0][d], bins=bins, color=NEG,
                label="cleared, negative")
        npos = int((cleared[d] > 0).sum())
        nneg = int((cleared[d] < 0).sum())
        pct = 100 * len(cleared) / len(pairs)
        ax.set_title(f"b   {pct:.1f}% of {len(pairs):,} pairs clear their own floor "
                     f"({npos:,} +, {nneg:,} −)", loc="left")
    else:
        ax.set_title("b   Effect distribution", loc="left")
    ax.axvline(0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("ΔAUPRC (target with source − target alone)")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=7.5, loc="upper right")

    save(fig, "figure1_noise_and_effects")


# ---------------------------------------------------------------- figure 2

def figure2(frames):
    """Descriptor importance for predicting transfer direction, grouped by target."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    except ImportError:
        print("Figure 2: scikit-learn unavailable - skipped")
        return

    hit = pick(frames, ["delta_auprc"], prefer=("pair", "transfer", "sweep"))
    if hit is None:
        print("Figure 2: no pairs frame - skipped")
        return
    path, pairs = hit
    d = col(pairs, "delta_auprc")
    s2 = col(pairs, "noise_2sigma", "noise2sigma", "sigma2", "noise_floor")
    tcol = col(pairs, "target", "target_task", "task")
    if tcol is None:
        print("Figure 2: no target column - skipped")
        return

    sub = pairs[pairs[d].abs() > pairs[s2]].copy() if s2 else pairs.copy()
    if len(sub) < 60:
        print(f"Figure 2: only {len(sub)} cleared pairs - skipped")
        return

    drop = {d, tcol}
    if s2:
        drop.add(s2)
    feats = [c for c in sub.columns
             if c not in drop
             and pd.api.types.is_numeric_dtype(sub[c])
             and sub[c].notna().mean() > 0.9
             and sub[c].nunique() > 2]
    if not feats:
        print("Figure 2: no usable descriptors - skipped")
        return

    X = sub[feats].fillna(sub[feats].median())
    y = (sub[d] > 0).astype(int)
    g = sub[tcol].astype(str)
    print(f"Figure 2: {len(sub)} pairs, {len(feats)} descriptors, "
          f"{g.nunique()} distinct targets")

    rf = RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                min_samples_leaf=5, n_jobs=-1, random_state=0)
    n_splits = int(min(5, g.nunique(), y.value_counts().min()))
    if n_splits < 2:
        print("Figure 2: too few groups/classes to cross-validate - skipped")
        return
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(rf, X, y, groups=g, cv=cv,
                             scoring="balanced_accuracy", n_jobs=1)
    acc, sd = scores.mean(), scores.std()
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()[-12:]

    fig, ax = plt.subplots(figsize=(5.4, 0.28 * len(imp) + 1.5))
    ax.barh(range(len(imp)), imp.values, color=ACC, height=0.7)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([s.replace("_", " ") for s in imp.index])
    ax.set_xlabel("Gini importance")
    ax.set_title("Dataset-geometry descriptors do not predict transfer direction\n"
                 f"grouped-CV balanced accuracy {acc:.3f} ± {sd:.3f} "
                 "(chance = 0.500)", loc="left")
    save(fig, "figure2_direction_importance")
    print(f"  grouped-CV balanced accuracy {acc:.3f} +/- {sd:.3f}")


# ---------------------------------------------------------------- figure 3

def figure3(frames):
    hit = pick(frames, ["delta_auprc"], prefer=("pair", "transfer", "sweep"))
    if hit is None:
        print("Figure 3: no pairs frame - skipped")
        return
    _, pairs = hit
    d = col(pairs, "delta_auprc")
    ov = col(pairs, "eff_overlap", "effective_overlap", "overlap_eff", "overlap")
    if ov is None:
        print("Figure 3: no overlap column - skipped")
        return

    sub = pairs[[ov, d]].dropna()
    sub["decile"] = pd.qcut(sub[ov], 10, labels=False, duplicates="drop")
    g = sub.groupby("decile")
    rate = g[d].apply(lambda x: (x > 0).mean())
    n = g.size()
    mid = g[ov].median()
    se = np.sqrt(rate * (1 - rate) / n)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.errorbar(mid, rate, yerr=1.96 * se, fmt="o-", color=ACC,
                ecolor=MID, capsize=3, lw=1.3, ms=5)
    ax.axhline(0.5, color=INK, ls=":", lw=0.9)
    ax.annotate("no net direction", xy=(mid.iloc[0], 0.5), xytext=(0, 5),
                textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_ylim(0, 1)
    ax.set_xlabel("effective compound overlap (Tanimoto τ = 0.35), decile median")
    ax.set_ylabel("fraction of pairs with positive ΔAUPRC")
    ax.set_title("No overlap threshold separates positive from negative transfer",
                 loc="left")
    for x, y_, nn in zip(mid, rate, n):
        ax.annotate(f"n={nn:,}", xy=(x, y_), xytext=(0, -13),
                    textcoords="offset points", ha="center",
                    fontsize=6.5, color=MID)
    save(fig, "figure3_overlap_boundary")


# ---------------------------------------------------------------- figure 4

def figure4(frames):
    hit = pick(frames, ["k", "delta_auprc"], prefer=("multi", "source"))
    if hit is None:
        print("Figure 4: no frame with k + delta_auprc - skipped")
        return
    path, ms = hit
    d = col(ms, "delta_auprc")
    k = col(ms, "k")
    print(f"Figure 4: using {path} ({len(ms)} rows)")

    ks = sorted(ms[k].dropna().unique())
    data = [ms.loc[ms[k] == kk, d].dropna().values for kk in ks]

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    parts = ax.violinplot(data, positions=range(len(ks)), showextrema=False,
                          widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(MID)
        b.set_alpha(0.35)
        b.set_edgecolor("none")
    for i, vals in enumerate(data):
        jit = np.random.default_rng(0).normal(0, 0.045, len(vals))
        ax.scatter(i + jit, vals, s=7, color=INK, alpha=0.45, linewidths=0)
        m = vals.mean()
        ax.hlines(m, i - 0.3, i + 0.3, color=NEG, lw=2.2, zorder=5)
        ax.annotate(f"mean {m:+.4f}", xy=(i, m), xytext=(0, -14),
                    textcoords="offset points", ha="center",
                    fontsize=7, color=NEG)
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([int(x) for x in ks])
    ax.set_xlabel("number of simultaneous source tasks (k)")
    ax.set_ylabel("ΔAUPRC vs. single-task")
    ax.set_title("Stacking sources widens the spread without moving the mean",
                 loc="left")
    save(fig, "figure4_multisource")

    print("  k   n   median|Δ|   meanΔ")
    for kk, vals in zip(ks, data):
        print(f"  {int(kk):<3} {len(vals):<3} {np.median(np.abs(vals)):.4f}"
              f"      {vals.mean():+.4f}")


# ---------------------------------------------------------------- main

def main():
    print("Scanning for result CSVs...")
    frames = load_all()
    if not frames:
        print("No CSVs found in . or ./results - nothing to plot.")
        sys.exit(1)
    for p, df in frames.items():
        print(f"  {p}: {len(df):,} rows, {len(df.columns)} cols")
    print()
    for fn in (figure1, figure2, figure3, figure4):
        try:
            fn(frames)
        except Exception as e:
            print(f"{fn.__name__} failed: {type(e).__name__}: {e}")
        print()
    print(f"Done. Figures in ./{OUT}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Seed budget: how many seeds does a transfer claim of size delta need?

This is the instrument the paper hands the field. Everything else measures
that single-source transfer effects sit inside the measurement error; this
says what it would take to measure one properly.

The argument, in three steps:

1. Each task has a single-task seed sd, sigma, estimated from the noise floor
   (sigma = noise2s / 2).
2. The paired difference between two arms is noisier than one arm, by a factor
   we measure rather than assume: r = sd(dAUPRC) / sigma. The single-run
   estimate was r = 1.34, which turns a nominal 2 sigma rule into a 1.5 sigma
   bound. This script recomputes r from whatever paired file it is given.
3. For S seeds the standard error of the mean difference is sigma_d / sqrt(S),
   so detecting a true effect delta at level alpha with power 1 - beta needs

       S >= ( (z_{1-alpha/2} + z_{1-beta}) * sigma_d / delta )^2

   With P pairs tested at once, alpha is replaced by the level a BH procedure
   effectively applies. Bonferroni (alpha/P) is the conservative version and is
   reported alongside, because a referee will ask.

    python power.py
    python power.py --results results --pairs 200 --power 0.8

Writes results/seed_budget.csv and figures/figure6_power.(png|pdf|svg).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--floor", default="noise_floor.csv")
ap.add_argument("--paired", default="seed_null.csv",
                help="paired per-seed file from seed_null.py")
ap.add_argument("--pairs", type=int, default=200,
                help="number of pairs tested simultaneously (for correction)")
ap.add_argument("--power", type=float, default=0.80)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--out", default="figures/figure6_power")
A = ap.parse_args()

R = lambda n: os.path.join(A.results, n)


# ---------------------------------------------------------------- normal quantile
def z(p):
    """Inverse standard normal CDF (Acklam), so scipy is not required."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------- inputs
if not os.path.exists(R(A.floor)):
    sys.exit(f"missing {R(A.floor)} - run transfer_law.py first")
nf = pd.read_csv(R(A.floor))
nf["sigma"] = nf.noise2s / 2.0

ratio, ratio_src = None, None
if os.path.exists(R(A.paired)):
    pr = pd.read_csv(R(A.paired))
    cols = {c.lower(): c for c in pr.columns}
    ab = cols.get("ap_base") or cols.get("auprc_base")
    at = cols.get("ap_tr") or cols.get("auprc_tr") or cols.get("ap_transfer")
    if ab and at:
        pr["d"] = pr[at] - pr[ab]
        key = [c for c in ("source", "target") if c in pr.columns]
        sd_d = pr.groupby(key).d.std(ddof=1).median() if key else pr.d.std(ddof=1)
        ratio = float(sd_d / nf.sigma.median())
        ratio_src = f"measured from {A.paired}"
if ratio is None:
    ratio, ratio_src = 1.34, "single-run estimate (paired file not found)"

sigma_med = float(nf.sigma.median())
sigma_d = ratio * sigma_med
print(f"median single-task sigma : {sigma_med:.5f}")
print(f"paired inflation ratio r : {ratio:.3f}  ({ratio_src})")
print(f"=> sd of paired difference: {sigma_d:.5f}")
print(f"   a nominal 2-sigma rule is a {2/ratio:.2f}-sigma bound\n")


def seeds_for(delta, alpha, power):
    """Seeds per arm to detect a true effect `delta` at `alpha`, `power`."""
    if delta <= 0:
        return np.inf
    k = (z(1 - alpha / 2) + z(power)) * sigma_d / delta
    return float(np.ceil(k * k))


# ---------------------------------------------------------------- table
DELTAS = [0.001, 0.002, 0.005, 0.0101, 0.02, 0.05, 0.10]
rows = []
for d in DELTAS:
    rows.append(dict(
        delta_auprc=d,
        seeds_uncorrected=seeds_for(d, A.alpha, A.power),
        seeds_bonferroni=seeds_for(d, A.alpha / A.pairs, A.power),
    ))
tab = pd.DataFrame(rows)
os.makedirs(A.results, exist_ok=True)
tab.to_csv(R("seed_budget.csv"), index=False)

print(f"seeds per arm for {int(A.power*100)}% power, alpha={A.alpha}, "
      f"{A.pairs} pairs tested\n")
print(f"{'effect (dAUPRC)':>16} {'1 pair':>10} {'Bonferroni':>12}")
for _, r_ in tab.iterrows():
    f = lambda v: "-" if not np.isfinite(v) else f"{int(v):,}"
    print(f"{r_.delta_auprc:>16.4f} {f(r_.seeds_uncorrected):>10} "
          f"{f(r_.seeds_bonferroni):>12}")

print("\nReference points:")
print(f"  Wang et al. 2026 stacking gain, +0.0101 ROC-AUC -> "
      f"{int(seeds_for(0.0101, A.alpha, A.power)):,} seeds (1 pair), "
      f"{int(seeds_for(0.0101, A.alpha/A.pairs, A.power)):,} corrected")
print(f"  Our macro effect, +0.0001 AUPRC -> "
      f"{seeds_for(0.0001, A.alpha, A.power):,.0f} seeds "
      f"(i.e. not reachable)")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Liberation Sans", "svg.fonttype": "path"})
fig, ax = plt.subplots(figsize=(5.4, 3.9))
dd = np.logspace(np.log10(5e-4), np.log10(0.15), 300)
for alpha, lab, ls in ((A.alpha, "single pair", "-"),
                       (A.alpha / A.pairs, f"corrected for {A.pairs} pairs", "--")):
    ax.plot(dd, [seeds_for(d, alpha, A.power) for d in dd],
            ls, color="#1a1a1a", lw=1.6, label=lab)
ax.axhline(20, color="#888", lw=1.0, ls=":")
ax.text(0.145, 22, "20 seeds (this study)", ha="right", va="bottom",
        fontsize=7.4, color="#555")
ax.axvline(0.0101, color="#888", lw=1.0, ls=":")
ax.text(0.0101, ax.get_ylim()[1], " +0.0101\n published\n stacking gain",
        ha="left", va="top", fontsize=7.4, color="#555")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("true effect size (ΔAUPRC)", fontsize=9)
ax.set_ylabel(f"seeds per arm for {int(A.power*100)}% power", fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7.8, frameon=False, loc="upper right")
ax.set_title("Seed budget required to detect a transfer effect",
             fontsize=9.6, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{A.out}.{ext}", dpi=400, bbox_inches="tight", facecolor="white")
print(f"\nwrote {R('seed_budget.csv')} and {A.out}.png/.pdf/.svg")

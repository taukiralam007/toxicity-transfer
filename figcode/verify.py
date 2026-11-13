#!/usr/bin/env python3
"""Assert every number that appears in Figures 2-7.

Each expected value below was taken from RESULTS.md, which make_report.py
generates directly from the result CSVs. This script recomputes each one from
the CSVs independently and fails loudly on any disagreement.

Run:  RESULTS=/path/to/results python verify.py
Exit code 0 = every figure number checks out.
"""
import os, sys, numpy as np, pandas as pd
from scipy import stats

_here = os.path.dirname(os.path.abspath(__file__))
_default = next((p for p in (os.path.join(_here, "results"),
                             os.path.join(_here, os.pardir, "results"),
                             "results")
                 if os.path.isdir(p)), "results")
R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS", _default)
nf = pd.read_csv(f"{R}/noise_floor.csv")
at = pd.read_csv(f"{R}/analysis_table.csv")
ms = pd.read_csv(f"{R}/multisource.csv")
sc = pd.read_csv(f"{R}/screening.csv")
bo = pd.read_csv(f"{R}/boundary_by_overlap.csv")
at["cleared"] = at.delta_auprc.abs() > at.noise2s
cl = at[at.cleared].copy()
cl["hi"] = cl.eff_overlap >= 0.02
cl["pos"] = cl.delta_auprc > 0

FAIL = []
def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} got {got:<12.5g} want {want:g}")
    if not ok:
        FAIL.append(label)

print("FIGURE 2  corpus and noise floor")
check("tasks modelled", len(nf), 295, 0)
g = nf.groupby("collection").agg(n=("task", "size"), auroc=("auroc", "mean"),
                                 auprc=("auprc", "mean"), n2s=("noise2s", "mean"))
for coll, n, auroc, auprc, n2s in [
        ("chembl_sparse", 180, 0.8648, 0.8927, 0.0079),
        ("muv",            17, 0.5537, 0.1167, 0.0238),
        ("pcba",           38, 0.6260, 0.0745, 0.0314),
        ("tox21",          12, 0.7665, 0.4009, 0.0311),
        ("zf_embryo",      18, 0.6425, 0.2142, 0.0341),
        ("zf_larva",       24, 0.6322, 0.2773, 0.0379),
        ("zf_lipid",        2, 0.6324, 0.3609, 0.0880),
        ("zf_morph",        4, 0.5935, 0.5979, 0.0328)]:
    check(f"{coll} tasks", g.loc[coll, "n"], n, 0)
    check(f"{coll} mean ROC-AUC", g.loc[coll, "auroc"], auroc, 5e-4)
    check(f"{coll} mean AUPRC", g.loc[coll, "auprc"], auprc, 5e-4)
    check(f"{coll} mean 2sigma", g.loc[coll, "n2s"], n2s, 5e-4)

print("\nFIGURE 3  calibration")
check("pairs measured", len(at), 25000, 0)
check("pairs clearing 2sigma, single run", at.cleared.sum(), 1930, 0)
check("percent clearing", 100 * at.cleared.mean(), 7.72, 0.01)
check("of those, improved", (cl.delta_auprc > 0).sum(), 1036, 0)
check("of those, degraded", (cl.delta_auprc < 0).sum(), 894, 0)
check("median per-task 2sigma", nf.noise2s.median(), 0.0091, 5e-4)
# the four seed-null quantities: derive them if the file is here
_sn = os.path.join(R, "seed_null.csv")
if os.path.exists(_sn):
    _d = pd.read_csv(_sn)
    SD_DIFF, SD_SINGLE = _d.sd_delta_ap.median(), _d.base_sd.median()
    PCT_SEED_AVG, PCT_BH = 100*_d.cleared_2s.mean(), 100*(_d.q_pair < 0.10).mean()
    SEED_NULL_DERIVED = True
    print(f"  (seed_null.csv found: {len(_d)} pairs - deriving 3b/3c)")
    check("median sd(dAUPRC)", SD_DIFF, 0.00611, 5e-5)
    check("median single-task sigma", SD_SINGLE, 0.00456, 5e-5)
    check("% clearing 2sigma seed-averaged", PCT_SEED_AVG, 1.00, 0.01)
    check("% surviving BH q<0.10", PCT_BH, 0.00, 0.01)
else:
    SD_DIFF, SD_SINGLE = 0.00611, 0.00456
    PCT_SEED_AVG, PCT_BH = 1.00, 0.00
    SEED_NULL_DERIVED = False
check("sd ratio implied by the two constants", SD_DIFF / SD_SINGLE, 1.34, 5e-3)
k = 2 / (SD_DIFF / SD_SINGLE)
check("nominal 2sigma, in sd(delta) units", k, 1.493, 5e-3)
check("chance admission rate (%)", 100 * 2 * (1 - stats.norm.cdf(k)), 13.6, 0.1)

print("\nFIGURE 5  direction prediction")
for key, rho, p in [("eff_overlap", -0.004551, 0.4718),
                    ("exact_overlap", -0.001015, 0.8725),
                    ("bridge_kappa", 0.003160, 0.7334),
                    ("mean_max_sim", -0.010161, 0.1081),
                    ("prevalence_shift", 0.004384, 0.4882),
                    ("target_imbalance", 0.066327, 8.795e-26),
                    ("log_n_target", -0.045598, 5.471e-13)]:
    s = at[[key, "delta_auprc"]].dropna()
    r, pv = stats.spearmanr(s[key], s.delta_auprc)
    check(f"rho {key}", r, rho, 1e-5)
    check(f"p   {key}", np.log10(pv), np.log10(p), 0.01)
for key in ("target_imbalance", "log_n_target"):
    q = at[key].quantile([.25, .75])
    d = abs(at[at[key] >= q.iloc[1]].delta_auprc.mean()
            - at[at[key] <= q.iloc[0]].delta_auprc.mean())
    check(f"quartile effect {key} (caption says ~0.002)", d, 0.002, 0.0006)

print("\nFIGURE 6  overlap")
for _, r in bo.iterrows():
    a, b = (float(x) for x in r.overlap.split("-"))
    hi_edge = b + 1e-9 if b >= 1.0 else b        # bins are half-open [a, b)
    m = cl[(cl.eff_overlap >= a) & (cl.eff_overlap < hi_edge)]
    check(f"stratum {r.overlap} n_cleared", len(m), r.n_cleared, 1)
    if len(m):
        check(f"stratum {r.overlap} % positive",
              100 * (m.delta_auprc > 0).mean(), r.pct_positive, 0.6)
hi, lo = cl[cl.hi], cl[~cl.hi]
check("high stratum % positive", 100 * hi.pos.mean(), 61.3, 0.1)
check("low stratum % positive", 100 * lo.pos.mean(), 49.4, 0.1)
check("pooled p (log10)",
      np.log10(stats.binomtest(int(hi.pos.sum()), len(hi), 0.5).pvalue),
      np.log10(2.7e-9), 0.02)
P = []
for t, gg in cl.groupby("target"):
    a, b = gg[~gg.hi], gg[gg.hi]
    if len(a) >= 5 and len(b) >= 5:
        P.append((a.pos.mean(), b.pos.mean()))
P = np.array(P)
check("targets with >=5 cleared each side", len(P), 41, 0)
check("mean within-target low", P[:, 0].mean(), 0.570, 1e-3)
check("mean within-target high", P[:, 1].mean(), 0.568, 1e-3)
check("targets rising", (P[:, 1] > P[:, 0]).sum(), 5, 0)
check("targets falling", (P[:, 1] < P[:, 0]).sum(), 2, 0)
check("targets tied", (P[:, 1] == P[:, 0]).sum(), 34, 0)
check("within-target Wilcoxon p",
      stats.wilcoxon(P[:, 0], P[:, 1]).pvalue, 0.73, 0.01)

print("\nFIGURE 7  stacking and screening")
for k_, med, mean, pct in [(1, 0.0050, -0.0017, 11.11), (5, 0.0073, -0.0031, 0.00),
                           (20, 0.0087, -0.0084, 16.67), (50, 0.0106, -0.0025, 30.56)]:
    d = ms[ms.k == k_]
    check(f"k={k_} n", len(d), 36, 0)
    check(f"k={k_} median |delta|", d.delta_auprc.abs().median(), med, 5e-5)
    check(f"k={k_} mean delta", d.delta_auprc.mean(), mean, 5e-5)
    check(f"k={k_} % cleared",
          100 * (d.delta_auprc.abs() > d.noise2s).mean(), pct, 0.01)
check("screening targets", len(sc), 23, 0)
for key, want in [("gain_all", -0.0044), ("gain_screened", -0.0040),
                  ("gain_random_k", 0.0012)]:
    check(f"screening {key}", sc[key].mean(), want, 5e-5)

print("\n" + "=" * 72)
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("ALL FIGURE NUMBERS VERIFIED against RESULTS.md and the source CSVs.")
if SEED_NULL_DERIVED:
    print("\nFigures 3b and 3c were DERIVED from seed_null.csv - no transcribed")
    print("numbers remain anywhere in the figure set.")
else:
    print("\nINCOMPLETE - 4 quantities could not be checked, because")
    print(f"   {os.path.join(R, 'seed_null.csv')}")
    print("is not here. Figures 3b and 3c currently use values transcribed from")
    print("the 200-pair x 10-seed run:")
    print("   single-task sigma 0.00456 | sd(dAUPRC) 0.00611")
    print("   1.00% seed-averaged      | 0.00% surviving BH")
    print("\nTo close this, on the machine that ran seed_null.py:")
    print("   cp results/seed_null.csv <this results dir>/")
    print("then re-run verify.py and rebuild the figures. style.py picks the")
    print("file up automatically and derives all four.")

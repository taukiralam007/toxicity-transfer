#!/usr/bin/env python3
"""Shared style, palette, data loading and external constants.

Imported by fig2..fig7. Edit the palette or the external constants here
and every figure picks the change up.
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

_here = os.path.dirname(os.path.abspath(__file__))
_default = next((p for p in (os.path.join(_here, "results"),
                             os.path.join(_here, os.pardir, "results"),
                             "results")
                 if os.path.isdir(p)), "results")
R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS", _default)
OUT = os.path.join(_here, os.pardir, "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- style
BLUE, ORANGE, GREEN, RED = "#3D6FA8", "#E08A2E", "#4E9A51", "#C14840"
PURPLE, TEAL, GREY, BROWN = "#7E62A3", "#3C9CA6", "#9A9A9A", "#946044"
INK, AX = "#111111", "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 7.6,
    "text.color": INK,
    "axes.linewidth": 0.7, "axes.edgecolor": AX, "axes.labelcolor": INK,
    "axes.titlesize": 8.0, "axes.labelsize": 8.0, "axes.labelpad": 3.5,
    "xtick.color": AX, "ytick.color": AX,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.pad": 2.2, "ytick.major.pad": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "legend.fontsize": 7.2,
    "legend.handlelength": 1.4, "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.1, "legend.labelspacing": 0.35,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 110,
})

COLL = {
    "chembl_sparse": BLUE, "pcba": ORANGE, "tox21": GREEN, "muv": RED,
    "zf_embryo": PURPLE, "zf_larva": TEAL, "zf_morph": BROWN, "zf_lipid": GREY,
}
NICE = {
    "chembl_sparse": "ChEMBL", "pcba": "PCBA", "tox21": "Tox21", "muv": "MUV",
    "zf_embryo": "ZF embryo", "zf_larva": "ZF larva",
    "zf_morph": "ZF morph", "zf_lipid": "ZF lipid",
}

def tidy(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#DEDEDE", lw=0.55, zorder=0)
        ax.set_axisbelow(True)

def panel(ax, letter, dx=-0.14, dy=1.13):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=INK)

def vlab(ax, xs, ys, fmt="{:.2f}", dy=0.012, size=6.5, colors=None):
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, (x, y) in enumerate(zip(xs, ys)):
        c = colors[i] if colors else INK
        ax.text(x, y + (dy * span if y >= 0 else -dy * span - 0.012 * span),
                fmt.format(y), ha="center",
                va="bottom" if y >= 0 else "top", fontsize=size, color=c)

def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print("  ", name)

# ----------------------------------------------------------------- data
nf = pd.read_csv(f"{R}/noise_floor.csv")
at = pd.read_csv(f"{R}/analysis_table.csv")
ms = pd.read_csv(f"{R}/multisource.csv")
sc = pd.read_csv(f"{R}/screening.csv")

at["cleared"] = at.delta_auprc.abs() > at.noise2s
cl = at[at.cleared]
order = nf.groupby("collection").noise2s.median().sort_values().index.tolist()

# --- the four seed-null quantities -----------------------------------------
# All four are derived from results/seed_null.csv, which seed_null.py writes
# (columns: sd_delta_ap, base_sd, cleared_2s, q_pair). If that file is absent
# the values recorded from the 200-pair x 10-seed run are used instead and a
# warning is printed - they are never substituted silently.

SEED_NULL_RECORDED = dict(SD_DIFF=0.00611, SD_SINGLE=0.00456,
                          PCT_SEED_AVG=1.00, PCT_BH=0.00)

def _seed_null(path=None):
    f = path or os.path.join(R, "seed_null.csv")
    if not os.path.exists(f):
        sys.stderr.write(
            "\n  WARNING  %s not found.\n"
            "           Figure 3b/3c fall back to values transcribed from the\n"
            "           200-pair x 10-seed run. Copy seed_null.csv into %s\n"
            "           to derive them instead.\n\n" % (f, R))
        d = dict(SEED_NULL_RECORDED); d["source"] = "recorded"; d["n_pairs"] = None
        return d
    df = pd.read_csv(f)
    d = dict(SD_DIFF=float(df.sd_delta_ap.median()),
             SD_SINGLE=float(df.base_sd.median()),
             PCT_SEED_AVG=float(100 * df.cleared_2s.mean()),
             PCT_BH=float(100 * (df.q_pair < 0.10).mean()),
             source=f, n_pairs=len(df))
    return d

SEED_NULL = _seed_null()
SD_DIFF      = SEED_NULL["SD_DIFF"]
SD_SINGLE    = SEED_NULL["SD_SINGLE"]
PCT_SEED_AVG = SEED_NULL["PCT_SEED_AVG"]
PCT_BH       = SEED_NULL["PCT_BH"]
# ---------------------------------------------------------------------------

# direction prediction, recomputed here rather than quoted
def direction_accuracies():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    F = ["eff_overlap", "exact_overlap", "bridge_kappa", "mean_max_sim",
         "prevalence_shift", "target_imbalance", "log_n_target"]
    X = cl[F].to_numpy(); yy = (cl.delta_auprc > 0).astype(int).to_numpy()
    grp = cl.target.to_numpy(); coll = cl.target_collection.to_numpy()
    mk = lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                learning_rate=0.06, random_state=0)
    def run(sp):
        return np.array([balanced_accuracy_score(
            yy[te], mk().fit(X[tr], yy[tr]).predict(X[te])) for tr, te in sp])
    g = run(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, yy, grp))
    u = run(StratifiedKFold(5, shuffle=True, random_state=0).split(X, yy))
    lo = []
    for c in np.unique(coll):
        te = np.where(coll == c)[0]; tr = np.where(coll != c)[0]
        if len(te) < 30 or len(np.unique(yy[te])) < 2: continue
        lo.append(balanced_accuracy_score(yy[te], mk().fit(X[tr], yy[tr]).predict(X[te])))
    return u.mean(), g.mean(), g.std(ddof=1), float(np.mean(lo))

print("corpus :", len(nf), "tasks |", len(at), "pairs |",
      f"{100*at.cleared.mean():.2f}% cleared |",
      f"{100*(cl.delta_auprc > 0).mean():.1f}% of those positive")


#!/usr/bin/env python3
"""Does effective chemical overlap predict multitask transfer, where exact
overlap does not?  Self-contained pipeline."""
from __future__ import annotations
import argparse, itertools, subprocess, urllib.request
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
DATA, RES = ROOT / "data", ROOT / "results"
RF = dict(n_estimators=500, max_features="log2", n_jobs=1)
TAU = 0.35
MAX_PER_TASK = 4000     # cap labels per task: keeps forests fast, tasks comparable
SIM_CAP = 1500          # cap compounds used in the Tanimoto step per side

ZF_GROUPS = {
    "zf_lipid": ["MIC", "METAB"],
    "zf_larva": ["MO24","DP24","SM24","NC24","MORT","YSE","AXIS","EYE","SNOU","JAW",
                 "OTIC","PE","BRAI","SOMI","PFIN","CFIN","PIG","CIRC","TRUN","SWIM",
                 "NC","TR","MOV21","AUC21"],
    "zf_embryo": ["LEL_MORT","LEL_YSE","LEL_AXIS","LEL_EYE","LEL_SNOU","LEL_JAW",
                  "LEL_OTIC","LEL_PE","LEL_BRAI","LEL_SOMI","LEL_PFIN","LEL_CFIN",
                  "LEL_PIG","LEL_CIRC","LEL_TRUN","LEL_SWIM","LEL_NC","LEL_TR"],
    "zf_morph": ["18_END_LEC","MOR_LEC","SUBLETH_17_END_LEC","TOX_SCO"],
}
T2G = {t: g for g, ts in ZF_GROUPS.items() for t in ts}


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / "zebrafish.csv").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/m946107011/MTForestNet.git",
                        "/tmp/mtfn"], check=True)
        (DATA / "zebrafish.csv").write_bytes(Path("/tmp/mtfn/Zebrafish.csv").read_bytes())
    import gzip
    base = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/"
    for name in ("tox21", "muv", "pcba", "chembl_sparse"):
        if (DATA / f"{name}.csv").exists():
            continue
        print("fetching", name, "...", flush=True)
        raw = urllib.request.urlopen(base + name + ".csv.gz").read()
        (DATA / f"{name}.csv").write_bytes(gzip.decompress(raw))
    print("datasets ready in", DATA)


def standardize(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(fr) > 1:
        m = max(fr, key=lambda x: x.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    if m.GetNumHeavyAtoms() < 2 or not any(a.GetSymbol() == "C" for a in m.GetAtoms()):
        return None
    return Chem.MolToSmiles(m, canonical=True)


def load_long():
    out = []
    z = pd.read_csv(DATA / "zebrafish.csv", index_col=0)
    tasks = [c for c in z.columns if c != "SMILES"]
    g = z.melt("SMILES", tasks, var_name="task", value_name="raw").dropna(subset=["raw"])
    g["label"] = (g["raw"] == "ACT").astype(np.int8)
    g["collection"] = g["task"].map(T2G).fillna("zf_other")
    out.append(g.rename(columns={"SMILES": "smiles_raw"}))

    if (DATA / "tox21.csv").exists():
        t = pd.read_csv(DATA / "tox21.csv")
        tt = [c for c in t.columns if c not in ("mol_id", "smiles")]
        g = t.melt("smiles", tt, var_name="task", value_name="raw").dropna(subset=["raw"])
        g["label"] = (g["raw"] == 1.0).astype(np.int8)
        g["collection"] = "tox21"
        g["task"] = "tox21:" + g["task"].astype(str)
        out.append(g.rename(columns={"smiles": "smiles_raw"}))
    else:
        print("WARNING: no tox21.csv -- run with --download.")

    # Wide collections. What each contributes is its LABEL DENSITY, which sets
    # how much compound overlap its tasks share:
    #   pcba   ~61%  dense   -> high overlap, like tox21
    #   muv    ~16%  medium
    #   chembl ~0.2% sparse  -> near-zero overlap. THIS is the regime the whole
    #                           question is about, and why it must be included.
    # thresh=None -> already 0/1.  thresh=x -> continuous, active if value >= x
    # (ChEMBL ships pChEMBL values, roughly 4-11; 6.5 is the usual activity cut)
    for name, max_tasks, min_lab, thresh in (("pcba", 40, 300, None),
                                             ("muv", 17, 300, None),
                                             ("chembl_sparse", 180, 150, 6.5)):
        p = DATA / f"{name}.csv"
        if not p.exists():
            print(f"  (skipping {name}: not downloaded)")
            continue
        t = pd.read_csv(p)
        cols = [c for c in t.columns if c not in ("mol_id", "smiles")]
        dens = t[cols].notna().sum().sort_values(ascending=False)
        chosen = []
        for c in dens.index:
            if dens[c] < min_lab:
                break
            v = t[c].dropna()
            pos = (v >= thresh).sum() if thresh is not None else (v == 1).sum()
            if v.nunique() >= 2 and pos >= 15 and (len(v) - pos) >= 15:
                chosen.append(c)
            if len(chosen) >= max_tasks:
                break
        if not chosen:
            print(f"  (skipping {name}: no tasks met the label criteria)")
            continue
        # per-column rather than one melt: a melt over 180 sparse columns
        # materialises tens of millions of mostly-null rows before dropna
        n_lab = 0
        for c in chosen:
            v = t[["smiles", c]].dropna(subset=[c])
            lab = ((v[c] >= thresh) if thresh is not None
                   else (v[c] == 1.0)).astype(np.int8)
            out.append(pd.DataFrame({
                "collection": name,
                "task": f"{name}:{c}",
                "smiles_raw": v["smiles"].to_numpy(),
                "label": lab.to_numpy()}))
            n_lab += len(v)
        print(f"  {name}: {len(chosen)} tasks, {n_lab} labels")

    long = pd.concat(out, ignore_index=True)[
        ["collection", "task", "smiles_raw", "label"]]

    # Cap each task, stratified so the positive rate is preserved. An
    # unstratified cap would silently shift class balance, which is one of the
    # descriptors we are trying to measure.
    sizes = long.groupby("task")["label"].size()
    keep = []
    for (task, _lab), g in long.groupby(["task", "label"], sort=False):
        n = sizes[task]
        if n <= MAX_PER_TASK:
            keep.append(g.index.to_numpy())
        else:
            k = max(15, int(round(len(g) * MAX_PER_TASK / n)))
            keep.append(g.sample(min(k, len(g)), random_state=0).index.to_numpy())
    long = long.loc[np.concatenate(keep)].reset_index(drop=True)
    print(f"  after per-task cap ({MAX_PER_TASK}): {len(long)} rows, "
          f"{long['task'].nunique()} tasks")
    return long


def build_corpus(seed=8):
    long = load_long()
    uniq = pd.Series(long.smiles_raw.unique(), name="smiles_raw")
    std = uniq.map(standardize)
    smap = pd.DataFrame({"smiles_raw": uniq[std.notna()], "smiles": std[std.notna()]})
    long = long.merge(smap, on="smiles_raw")
    agg = long.groupby(["task", "smiles"])["label"].agg(["mean"]).reset_index()
    agg = agg[(agg["mean"] == 0) | (agg["mean"] == 1)]
    agg["label"] = agg["mean"].astype(np.int8)
    tasks = agg.merge(long[["task", "collection"]].drop_duplicates(), on="task")
    comp = pd.DataFrame({"smiles": sorted(tasks.smiles.unique())})
    comp["cid"] = np.arange(len(comp), dtype=np.int32)
    comp["scaffold"] = [
        (MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False) or "__ACY__")
        for s in comp.smiles]
    groups = sorted(comp.groupby("scaffold").groups.items(),
                    key=lambda kv: (-len(kv[1]), str(kv[0])))
    rng = np.random.default_rng(seed)
    big = [k for k in groups if len(k[1]) >= 5]
    small = [k for k in groups if len(k[1]) < 5]
    rng.shuffle(small)
    ntr, nva = int(.7 * len(comp)), int(.1 * len(comp))
    split = pd.Series(index=comp.index, dtype=object)
    ctr = cva = 0
    for _, idx in big + small:
        if ctr + len(idx) <= ntr:
            split.loc[idx] = "train"; ctr += len(idx)
        elif cva + len(idx) <= nva:
            split.loc[idx] = "valid"; cva += len(idx)
        else:
            split.loc[idx] = "test"
    comp["split"] = split.values
    gen = GetMorganGenerator(radius=3, fpSize=1024)
    fps = np.zeros((len(comp), 1024), dtype=np.uint8)
    for i, s in enumerate(comp.smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps[i] = gen.GetFingerprintAsNumPy(m).astype(np.uint8)
    tasks = tasks.merge(comp[["smiles", "cid", "split"]], on="smiles")
    stat = tasks.groupby("task")["label"].agg(["size", "sum"])
    keep = stat[(stat["size"] >= 40) & (stat["sum"] >= 5)].index
    tasks = tasks[tasks.task.isin(keep)]
    print(f"corpus: {len(comp)} compounds, {tasks.task.nunique()} tasks, "
          f"{tasks.collection.nunique()} collections, "
          f"split {comp.split.value_counts().to_dict()}")
    return comp, tasks, fps


class TD:
    def __init__(self, name, coll, tr, te):
        self.name, self.coll = name, coll
        self.tr_i, self.tr_y = tr
        self.te_i, self.te_y = te


def index_tasks(tasks):
    out = {}
    for t, g in tasks.groupby("task", sort=False):
        a = g[g.split.isin(["train", "valid"])]; b = g[g.split == "test"]
        if len(a) < 20 or len(b) < 10 or a.label.nunique() < 2 or b.label.nunique() < 2:
            continue
        out[t] = TD(t, g.collection.iloc[0],
                    (a.cid.to_numpy(np.int32), a.label.to_numpy(np.int8)),
                    (b.cid.to_numpy(np.int32), b.label.to_numpy(np.int8)))
    return out


def st_score(td, fps, seed=0):
    c = RandomForestClassifier(random_state=seed, **RF).fit(fps[td.tr_i], td.tr_y)
    p = c.predict_proba(fps[td.te_i])[:, 1]
    return average_precision_score(td.te_y, p), roc_auc_score(td.te_y, p)


def noise_floor(idx, fps, seeds):
    def one(td):
        v = np.array([st_score(td, fps, s)[0] for s in range(seeds)])
        _, au = st_score(td, fps, 0)
        return dict(task=td.name, collection=td.coll, auprc=v.mean(), auroc=au,
                    noise2s=2 * v.std(ddof=1) if seeds > 1 else .02)
    return pd.DataFrame(Parallel(n_jobs=-1, verbose=1)(
        delayed(one)(td) for td in idx.values()))


def source_feature(src, fps, ncomp, seed=0):
    """OUT-OF-FOLD for the source's own training compounds -- without this,
    transfer is manufactured from leakage."""
    f = np.full(ncomp, float(src.tr_y.mean()), np.float32)
    k = min(5, int(np.bincount(src.tr_y).min()))
    if k >= 2:
        X, y = fps[src.tr_i], src.tr_y
        oof = np.zeros(len(y), np.float32)
        for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
            oof[va] = RandomForestClassifier(random_state=seed, **RF).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        f[src.tr_i] = oof
    full = RandomForestClassifier(random_state=seed, **RF).fit(fps[src.tr_i], src.tr_y)
    m = np.ones(ncomp, bool); m[src.tr_i] = False
    i = np.flatnonzero(m)
    if len(i):
        f[i] = full.predict_proba(fps[i])[:, 1]
    return f


def tanimoto_max(tfp, sfp, chunk=2048):
    best = np.zeros(len(tfp), np.float32); arg = np.zeros(len(tfp), np.int32)
    a_all = tfp.astype(np.float32); b = sfp.astype(np.float32)
    nb = b.sum(1)[None, :]
    for s in range(0, len(tfp), chunk):
        a = a_all[s:s + chunk]
        inter = a @ b.T
        union = a.sum(1, keepdims=True) + nb - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 0).astype(np.float32)
        arg[s:s + chunk] = sim.argmax(1); best[s:s + chunk] = sim.max(1)
    return best, arg


def kappa(a, b):
    if len(a) == 0:
        return np.nan
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def _sub(idx, y, cap, seed=0):
    if len(idx) <= cap:
        return idx, y
    sel = np.random.default_rng(seed).choice(len(idx), cap, replace=False)
    return idx[sel], y[sel]


def descriptors(tgt_i, tgt_y, src_i, src_y, fps):
    # subsample both sides: the Tanimoto step is O(n_t * n_s) and would
    # otherwise dominate the whole sweep on the large collections
    tgt_i, tgt_y = _sub(tgt_i, tgt_y, SIM_CAP)
    src_i, src_y = _sub(src_i, src_y, SIM_CAP)
    best, arg = tanimoto_max(fps[tgt_i], fps[src_i])
    sset = set(src_i.tolist())
    exact = np.fromiter((int(i) in sset for i in tgt_i), bool, len(tgt_i))
    br = (best >= TAU) & (~exact)
    return dict(eff_overlap=float((best >= TAU).mean()),
                exact_overlap=float(exact.mean()),
                mean_max_sim=float(best.mean()),
                n_bridged=int(br.sum()),
                bridge_kappa=kappa(tgt_y[br], src_y[arg[br]]) if br.sum() >= 10 else np.nan,
                prevalence_shift=abs(float(tgt_y.mean()) - float(src_y.mean())),
                target_imbalance=min(float(tgt_y.mean()), 1 - float(tgt_y.mean())),
                log_n_target=float(np.log2(max(len(tgt_i), 2))))


def sweep(idx, fps, base, ncomp, pairs, seed=0):
    by_src = {}
    for s, t in pairs:
        by_src.setdefault(s, []).append(t)

    def one(sname, tnames):
        src = idx[sname]
        feat = source_feature(src, fps, ncomp, seed)
        rows = []
        for tn in tnames:
            tgt = idx[tn]
            Xtr = np.hstack([fps[tgt.tr_i], feat[tgt.tr_i, None]]).astype(np.float32)
            Xte = np.hstack([fps[tgt.te_i], feat[tgt.te_i, None]]).astype(np.float32)
            p = RandomForestClassifier(random_state=seed, **RF).fit(Xtr, tgt.tr_y).predict_proba(Xte)[:, 1]
            ap = average_precision_score(tgt.te_y, p)
            d = descriptors(tgt.tr_i, tgt.tr_y, src.tr_i, src.tr_y, fps)
            d.update(source=sname, target=tn, delta_auprc=ap - base[tn],
                     source_collection=src.coll, target_collection=tgt.coll)
            rows.append(d)
        return rows

    out = Parallel(n_jobs=-1, verbose=2)(delayed(one)(s, t) for s, t in by_src.items())
    return pd.DataFrame([r for c in out for r in c])


FEATS = ["eff_overlap", "bridge_kappa", "prevalence_shift", "target_imbalance",
         "log_n_target", "mean_max_sim"]


def report(df):
    df["sign"] = (df.delta_auprc > 0).astype(int)
    res = df[df.transferred == 1]
    print("\n=== transfer ===")
    print(f"pairs {len(df)}   cleared noise floor {len(res)} ({len(res)/len(df):.1%})")
    print(f"  positive {int((df.delta_auprc>df.noise2s).sum())}   negative {int((df.delta_auprc<-df.noise2s).sum())}")
    print(f"GATE A: {'PASS' if len(res)/len(df)>=.10 else 'FAIL -> negative paper'}")
    print("\n=== does the independent variable have range? ===")
    for lo, hi in [(0,.1),(.1,.3),(.3,.6),(.6,.9),(.9,1.01)]:
        m = (df.eff_overlap >= lo) & (df.eff_overlap < hi)
        print(f"  eff_overlap [{lo},{hi}): {m.sum():5d} ({m.mean():5.1%})")
    r = df.eff_overlap.corr(df.exact_overlap)
    print(f"  corr(effective, exact) = {r:.3f}" + ("   <-- TOO HIGH: add a collection" if r > .9 else ""))
    print("\n=== univariate vs delta AUPRC ===")
    for c in ["eff_overlap", "exact_overlap", "bridge_kappa", "mean_max_sim",
              "prevalence_shift", "target_imbalance"]:
        s = df[[c, "delta_auprc"]].dropna()
        if len(s) > 20 and s[c].nunique() > 3:
            sr = spearmanr(s[c], s.delta_auprc)
            print(f"  {c:18s} rho={sr.statistic:+.3f}  p={sr.pvalue:.4f}  n={len(s)}")
    print("\n=== ablation (balanced accuracy on transfer sign) ===")
    for nm, fs in [("full", FEATS), ("effective only", ["eff_overlap"]), ("exact only", ["exact_overlap"])]:
        sub = res.dropna(subset=["delta_auprc"])
        if len(sub) < 40 or sub.sign.nunique() < 2:
            print(f"  {nm:16s} n/a"); continue
        X = sub[fs].to_numpy(float); y = sub.sign.to_numpy()
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, random_state=0).fit(X, y)
        print(f"  {nm:16s} {balanced_accuracy_score(y, m.predict(X)):.3f}   n={len(sub)}")
    print("\n=== leave-one-collection-out ===")
    accs = []
    for held in sorted(df.target_collection.unique()):
        tr = res[res.target_collection != held]; te = res[res.target_collection == held]
        if len(tr) < 30 or len(te) < 15 or tr.sign.nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, random_state=0).fit(tr[FEATS].to_numpy(float), tr.sign)
        a = balanced_accuracy_score(te.sign, m.predict(te[FEATS].to_numpy(float)))
        accs.append(a); print(f"  {held:22s} balanced acc {a:.3f}  (n={len(te)})")
    if accs:
        mean = float(np.mean(accs))
        v = ("Gate C PASS -> aim Nat Mach Intell / Nat Commun" if mean >= .80
             else "Gate C PARTIAL -> Brief Bioinform / J Cheminform" if mean >= .70
             else "Gate C FAIL -> J Cheminform, negative framing")
        print(f"\n  MEAN {mean:.3f}  ->  {v}")
        if r > .9:
            print("  (but corr(eff,exact) too high -- benchmark limitation, NOT a failed hypothesis)")


def choose_pairs(idx, max_pairs, seed=0):
    """Every within-collection pair, plus a cross-collection sample.

    Cross-collection pairs are where effective and exact overlap come apart,
    so they are kept preferentially rather than being swamped by the much more
    numerous within-collection pairs.
    """
    allp = list(itertools.permutations(idx, 2))
    within = [(s, t) for s, t in allp if idx[s].coll == idx[t].coll]
    cross = [(s, t) for s, t in allp if idx[s].coll != idx[t].coll]
    rng = np.random.default_rng(seed)
    if not max_pairs:
        return allp
    n_cross = min(len(cross), max(max_pairs // 2, max_pairs - len(within)))
    n_within = max(0, min(len(within), max_pairs - n_cross))
    pick = lambda lst, n: [lst[i] for i in rng.choice(len(lst), n, replace=False)] if n < len(lst) else lst
    out = pick(cross, n_cross) + pick(within, n_within)
    print(f"  pairs: {len(out)} ({n_cross} cross-collection, {n_within} within)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    if a.download:
        download(); return
    RES.mkdir(parents=True, exist_ok=True)
    comp, tasks, fps = build_corpus()
    idx = index_tasks(tasks)
    print(f"modelling {len(idx)} tasks")
    print("\n[1/2] noise floor + single-task baseline ...")
    nf = noise_floor(idx, fps, a.seeds)
    nf.to_csv(RES / "noise_floor.csv", index=False)
    print(f"  mean single-task AUROC {nf.auroc.mean():.3f}  (MTForestNet layer 1: 0.638; >0.70 means the split leaks)")
    pairs = choose_pairs(idx, a.max_pairs)
    print(f"\n[2/2] transfer sweep over {len(pairs)} pairs ...")
    base = dict(zip(nf.task, nf.auprc))
    df = sweep(idx, fps, base, len(comp), pairs)
    df = df.merge(nf[["task", "noise2s"]].rename(columns={"task": "target"}), on="target")
    df["transferred"] = (df.delta_auprc.abs() > df.noise2s).astype(int)
    df.to_csv(RES / "analysis_table.csv", index=False)
    report(df)
    print(f"\nwrote {RES/'analysis_table.csv'}")


if __name__ == "__main__":
    main()
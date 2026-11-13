#!/usr/bin/env python
"""The seed-null experiment, run with a graph neural network instead of RF.

Why this exists
---------------
The RF result invites one obvious objection: maybe single-source transfer looks
like noise because random forests on ECFP6 cannot represent what is being
transferred, and a learned representation would do better. This script answers
that with the same protocol, so the two are comparable line for line.

It measures, for each (source, target) pair:

    base arm      target-only D-MPNN
    transfer arm  the same model with the source signal added

across the same seeds, then compares the paired difference against the GNN's
*own* noise floor - which is not the RF noise floor, and is expected to be
larger. That is the point: if the measurement error grows faster than the
effect, a representation upgrade does not rescue the claim.

Two transfer arms, because a referee will ask about both:

    --arm concat      the source model's out-of-fold prediction is appended to
                      the pooled molecule vector before the readout MLP. Strict
                      analogue of the RF arm (one extra feature column).
    --arm multitask   one shared encoder, two output heads, trained jointly.
                      What the field actually does, and what MTForestNet and
                      the massively-multitask line of work claim gains from.

Usage
-----
    # 1. check it wires up against transfer_law.py at all (minutes, CPU ok)
    python gnn_null.py --smoke

    # 2. noise floor only - run this FIRST and look at it before anything else
    python gnn_null.py --stage floor --tasks 60 --seeds 10 --gpu 0

    # 3. paired pairs
    python gnn_null.py --stage pairs --arm concat    --pairs 200 --seeds 10 --gpu 0
    python gnn_null.py --stage pairs --arm multitask --pairs 200 --seeds 10 --gpu 1

    # 4. analysis (no GPU)
    python gnn_null.py --stage analyse

Outputs gnn_floor.csv, gnn_pairs_<arm>.csv, gnn_report.txt in the working dir.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# transfer_law.py owns the data: the task list, the scaffold-disjoint split and
# the out-of-fold source feature. We import rather than re-derive so the GNN
# arm sees byte-identical splits to the RF arm. If any of these names differ in
# your copy, fix the four lines in bind() below - do not re-implement them here,
# because a second implementation of the split is a second chance to leak.
# --------------------------------------------------------------------------
import transfer_law as T


CACHE = os.path.join("results", "corpus_cache.pkl")


def bind(use_cache=True):
    """Build (or reload) the corpus and return tasks, SMILES-by-cid, fps.

    transfer_law.build_corpus() is expensive - RDKit standardisation, Murcko
    scaffolds and fingerprints over every compound - and the GNN stages each
    need it. It is cached to results/corpus_cache.pkl on first use. The cache
    is keyed to nothing: delete the file if transfer_law's corpus code changes.

    cid is the row index into fps, so comp.smiles in cid order is the SMILES
    array aligned with every index the TD objects carry.
    """
    import pickle
    if use_cache and os.path.exists(CACHE):
        print(f"loading corpus from {CACHE}", flush=True)
        with open(CACHE, "rb") as fh:
            comp, tasks_df, fps = pickle.load(fh)
    else:
        print("building corpus (slow on first run) ...", flush=True)
        comp, tasks_df, fps = T.build_corpus()
        os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
        with open(CACHE, "wb") as fh:
            pickle.dump((comp, tasks_df, fps), fh, protocol=4)
        print(f"cached to {CACHE}", flush=True)

    idx = T.index_tasks(tasks_df)
    smiles = comp.sort_values("cid").smiles.to_numpy(dtype=object)
    print(f"{len(idx)} tasks, {len(smiles)} compounds", flush=True)
    return list(idx.values()), smiles, fps


# --------------------------------------------------------------------------
# Featurisation - a compact D-MPNN in the Chemprop idiom, written out so there
# is no dependency on a chemprop release whose API may move under you.
# --------------------------------------------------------------------------
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import torch
import torch.nn as nn
import torch.nn.functional as F

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B",
                "Se", "Na", "K", "Li", "Ca", "Zn", "other"]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
ATOM_FDIM = len(ATOM_SYMBOLS) + 6 + 5 + 5 + 1 + 1
BOND_FDIM = len(BOND_TYPES) + 2


def _onehot(x, choices):
    v = [0.0] * len(choices)
    v[choices.index(x) if x in choices else len(choices) - 1] = 1.0
    return v


def atom_features(a):
    return (_onehot(a.GetSymbol(), ATOM_SYMBOLS)
            + _onehot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
            + _onehot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
            + _onehot(int(a.GetTotalNumHs()), [0, 1, 2, 3, 4])
            + [1.0 if a.GetIsAromatic() else 0.0]
            + [a.GetMass() * 0.01])


def bond_features(b):
    return (_onehot(b.GetBondType(), BOND_TYPES)
            + [1.0 if b.GetIsConjugated() else 0.0]
            + [1.0 if b.IsInRing() else 0.0])


class MolGraph:
    """Directed bond representation: messages live on bonds, not atoms."""

    __slots__ = ("n_atoms", "f_atoms", "f_bonds", "a2b", "b2a", "b2rev")

    def __init__(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles("C")          # keep indices aligned
        self.n_atoms = mol.GetNumAtoms()
        self.f_atoms = [atom_features(a) for a in mol.GetAtoms()]
        self.f_bonds, self.b2a, self.b2rev = [], [], []
        self.a2b = [[] for _ in range(self.n_atoms)]
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            fb = bond_features(bond)
            for src, dst in ((u, v), (v, u)):
                self.f_bonds.append(self.f_atoms[src] + fb)
                self.a2b[dst].append(len(self.b2a))
                self.b2a.append(src)
            n = len(self.b2a)
            self.b2rev += [n - 1, n - 2]


def featurise(smiles, cids):
    """MolGraph for just the compounds in play, keyed by cid.

    The corpus runs to hundreds of thousands of compounds; a stage only ever
    touches those belonging to its tasks, so building all of them wastes
    minutes and gigabytes.
    """
    want = sorted(set(int(c) for c in cids))
    t0 = time.time()
    g = {c: MolGraph(smiles[c]) for c in want}
    print(f"featurised {len(g)} of {len(smiles)} compounds "
          f"in {time.time()-t0:.0f}s", flush=True)
    return g


def batch_graphs(graphs, device):
    f_atoms, f_bonds = [[0.0] * ATOM_FDIM], [[0.0] * (ATOM_FDIM + BOND_FDIM)]
    a2b, b2a, b2rev, scope = [[]], [0], [0], []
    for g in graphs:
        a_off, b_off = len(f_atoms), len(f_bonds)
        f_atoms += g.f_atoms
        f_bonds += g.f_bonds
        a2b += [[b + b_off for b in bs] for bs in g.a2b]
        b2a += [a + a_off for a in g.b2a]
        b2rev += [b + b_off for b in g.b2rev]
        scope.append((a_off, g.n_atoms))
    maxb = max(1, max(len(bs) for bs in a2b))
    a2b = [bs + [0] * (maxb - len(bs)) for bs in a2b]
    t = lambda x, d: torch.tensor(x, dtype=d, device=device)
    return (t(f_atoms, torch.float), t(f_bonds, torch.float),
            t(a2b, torch.long), t(b2a, torch.long), t(b2rev, torch.long), scope)


class DMPNN(nn.Module):
    def __init__(self, hidden=300, depth=3, dropout=0.0, n_extra=0, n_out=1):
        super().__init__()
        self.depth = depth
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden + n_extra, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, batch, extra=None):
        f_atoms, f_bonds, a2b, b2a, b2rev, scope = batch
        msg = F.relu(self.W_i(f_bonds))
        for _ in range(self.depth):
            nei = msg[a2b].sum(dim=1)              # messages into each atom
            msg = F.relu(self.W_i(f_bonds) + self.W_h(nei[b2a] - msg[b2rev]))
            msg = self.drop(msg)
        nei = msg[a2b].sum(dim=1)
        hidden_atoms = self.drop(F.relu(self.W_o(
            torch.cat([f_atoms, nei], dim=1))))
        mols = torch.stack([hidden_atoms[a:a + n].mean(dim=0)
                            for a, n in scope])
        if extra is not None:
            mols = torch.cat([mols, extra], dim=1)
        return self.ffn(mols)


# --------------------------------------------------------------------------
# One fit
# --------------------------------------------------------------------------
from sklearn.metrics import average_precision_score, roc_auc_score


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def fit_predict(graphs, tr_i, tr_y, te_i, seed, dev, extra=None,
                aux=None, epochs=30, bs=50, lr=1e-3, hidden=300, depth=3):
    """Train on tr_i, return P(positive) on te_i.

    extra : (n_total, 1) float array appended to the molecule vector, or None.
    aux   : (n_total,) auxiliary labels for a second head, or None. Rows whose
            aux label is NaN contribute nothing to the auxiliary loss, which is
            how a sparse source task is handled without dropping target rows.
    """
    set_seed(seed)
    n_extra = 0 if extra is None else 1
    n_out = 1 if aux is None else 2
    model = DMPNN(hidden, depth, 0.0, n_extra, n_out).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * math.ceil(len(tr_i) / bs)))

    pos = float(np.sum(tr_y)); neg = float(len(tr_y) - pos)
    w = torch.tensor([neg / max(pos, 1.0)], device=dev, dtype=torch.float)
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)

    tr_i = np.asarray(tr_i); tr_y = np.asarray(tr_y, dtype=np.float32)
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(tr_i))
        for k in range(0, len(order), bs):
            idx = tr_i[order[k:k + bs]]
            batch = batch_graphs([graphs[int(i)] for i in idx], dev)
            ex = (None if extra is None else
                  torch.tensor(extra[idx], dtype=torch.float, device=dev))
            out = model(batch, ex)
            y = torch.tensor(tr_y[order[k:k + bs]], device=dev).unsqueeze(1)
            loss = lossf(out[:, :1], y)
            if aux is not None:
                a = torch.tensor(aux[idx], dtype=torch.float, device=dev)
                m = ~torch.isnan(a)
                if m.any():
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out[:, 1][m], a[m])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    model.eval(); out = []
    te_i = np.asarray(te_i)
    with torch.no_grad():
        for k in range(0, len(te_i), 256):
            idx = te_i[k:k + 256]
            batch = batch_graphs([graphs[int(i)] for i in idx], dev)
            ex = (None if extra is None else
                  torch.tensor(extra[idx], dtype=torch.float, device=dev))
            out.append(torch.sigmoid(model(batch, ex)[:, 0]).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_floor(tasks, smiles, A, dev):
    """The GNN's own per-task 2-sigma threshold. Run and read this first."""
    sel = tasks[:A.tasks]
    graphs = featurise(smiles, np.concatenate(
        [np.concatenate([t.tr_i, t.te_i]) for t in sel]))
    rows = []
    for ti, tgt in enumerate(sel):
        ap = []
        for s in range(A.seeds):
            p = fit_predict(graphs, tgt.tr_i, tgt.tr_y, tgt.te_i, s, dev,
                            epochs=A.epochs)
            ap.append(average_precision_score(tgt.te_y, p))
        rows.append(dict(task=tgt.name, collection=tgt.coll, auprc=float(np.mean(ap)),
                         sd=float(np.std(ap, ddof=1)),
                         noise2s=float(2 * np.std(ap, ddof=1))))
        print(f"[floor {ti+1}/{len(sel)}] {rows[-1]['task']} "
              f"AUPRC {rows[-1]['auprc']:.4f} 2s {rows[-1]['noise2s']:.4f}",
              flush=True)
    pd.DataFrame(rows).to_csv("gnn_floor.csv", index=False)
    print("wrote gnn_floor.csv")


def stage_pairs(tasks, smiles, fps, A, dev):
    floor = pd.read_csv("gnn_floor.csv").set_index("task")
    rng = np.random.default_rng(0)
    names = [t.name for t in tasks]
    cand = [(i, j) for i in range(len(tasks)) for j in range(len(tasks))
            if i != j and names[j] in floor.index]
    pairs = [cand[k] for k in rng.choice(len(cand), size=min(A.pairs, len(cand)),
                                         replace=False)]
    ncomp = len(smiles)
    graphs = featurise(smiles, np.concatenate(
        [np.concatenate([tasks[i].tr_i, tasks[i].te_i, tasks[j].tr_i])
         for i, j in pairs]))
    rows = []
    for n, (si, tj) in enumerate(pairs):
        src, tgt = tasks[si], tasks[tj]
        for s in range(A.seeds):
            pb = fit_predict(graphs, tgt.tr_i, tgt.tr_y, tgt.te_i, s, dev,
                             epochs=A.epochs)
            if A.arm == "concat":
                # Deliberately the SAME out-of-fold source feature the RF arm
                # used: an RF source model on ECFP6. That makes this arm ask a
                # clean question - can a stronger TARGET model exploit a signal
                # the RF target model could not? - at the cost of leaving the
                # source model unchanged. The multitask arm has no RF anywhere
                # in it, so between them the two arms cover both readings.
                feat = T.source_feature(src, fps, ncomp, s).astype(np.float32)
                pt = fit_predict(graphs, tgt.tr_i, tgt.tr_y, tgt.te_i, s, dev,
                                 extra=feat[:, None], epochs=A.epochs)
            else:
                aux = np.full(ncomp, np.nan, dtype=np.float32)
                aux[src.tr_i] = np.asarray(src.tr_y, dtype=np.float32)
                pt = fit_predict(graphs, tgt.tr_i, tgt.tr_y, tgt.te_i, s, dev,
                                 aux=aux, epochs=A.epochs)
            rows.append(dict(
                source=names[si], target=names[tj], seed=s,
                ap_base=average_precision_score(tgt.te_y, pb),
                ap_tr=average_precision_score(tgt.te_y, pt),
                auc_base=roc_auc_score(tgt.te_y, pb),
                auc_tr=roc_auc_score(tgt.te_y, pt)))
        d = np.mean([r["ap_tr"] - r["ap_base"] for r in rows[-A.seeds:]])
        print(f"[pair {n+1}/{len(pairs)}] {names[si]} -> {names[tj]} "
              f"mean dAUPRC {d:+.4f}", flush=True)
        pd.DataFrame(rows).to_csv(f"gnn_pairs_{A.arm}.csv", index=False)
    print(f"wrote gnn_pairs_{A.arm}.csv")


def stage_analyse():
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    floor = pd.read_csv("gnn_floor.csv")
    out = [f"GNN noise floor: {len(floor)} tasks, "
           f"median 2s = {floor.noise2s.median():.4f}, "
           f"median AUPRC = {floor.auprc.median():.4f}", ""]
    for arm in ("concat", "multitask"):
        f = f"gnn_pairs_{arm}.csv"
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df["d"] = df.ap_tr - df.ap_base
        per, ps = [], []
        for (s, t), g in df.groupby(["source", "target"]):
            try:
                p = wilcoxon(g.ap_tr, g.ap_base, zero_method="pratt").pvalue
            except ValueError:
                p = 1.0
            per.append(dict(source=s, target=t, d=g.d.mean(), p=p)); ps.append(p)
        per = pd.DataFrame(per)
        per["q"] = multipletests(ps, method="fdr_bh")[1]
        sd_pair = df.groupby(["source", "target"]).d.std(ddof=1).median()
        ratio = sd_pair / floor.sd.median()
        boot = [np.mean(per.d.sample(len(per), replace=True))
                for _ in range(2000)]
        out += [f"--- arm: {arm} ---",
                f"pairs: {len(per)}   seeds/arm: {df.seed.nunique()}",
                f"sd(dAUPRC) / single-task sd = {ratio:.2f}  "
                f"(a nominal 2s rule is really a {2/max(ratio,1e-9):.2f}s bound)",
                f"seed-averaged |d| > 2s of target: "
                f"{100*np.mean(np.abs(per.d) > 2*floor.sd.median()):.1f}%",
                f"BH-FDR q < 0.10: {int((per.q < 0.10).sum())} of {len(per)}",
                f"macro d = {per.d.mean():+.4f} AUPRC "
                f"(95% CI {np.percentile(boot,2.5):+.4f} to "
                f"{np.percentile(boot,97.5):+.4f})", ""]
    txt = "\n".join(out)
    open("gnn_report.txt", "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["floor", "pairs", "analyse"],
                    default="floor")
    ap.add_argument("--arm", choices=["concat", "multitask"], default="concat")
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()

    if A.stage == "analyse":
        return stage_analyse()

    if A.smoke:
        A.tasks, A.pairs, A.seeds, A.epochs = 3, 2, 2, 3

    dev = torch.device(f"cuda:{A.gpu}" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)

    tasks, smiles, fps = bind()
    if A.stage == "floor":
        stage_floor(tasks, smiles, A, dev)
    else:
        stage_pairs(tasks, smiles, fps, A, dev)


if __name__ == "__main__":
    main()

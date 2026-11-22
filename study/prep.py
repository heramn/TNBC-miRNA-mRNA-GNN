"""Outer/inner folds and fold-local preprocessing (study-final).

Everything label- or data-dependent is fit on the OUTER-TRAINING samples of a fold only:
pydeseq2 DE (3 pairwise contrasts, union), log2(x+1) z-scoring, the TCGA-mappable node subset,
tissue co-expression edges, and the miRDB / TCGA edge subsets. Test examples are scored from their
own single modality.
"""
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C
import edges as E
import geo_data
import mappings

warnings.filterwarnings('ignore')
CACHE = Path(os.environ.get('STUDY_CACHE', Path.home() / '.capy/work/study_cache'))
CACHE.mkdir(parents=True, exist_ok=True)


def samples(scheme):
    t = geo_data.load()['samples']
    if scheme == 'subset230':
        t = t[t.in_subset230]
    return t


def group_col(scheme):
    return 'group_conservative' if scheme == 'conservative' else 'group'


def outer_folds(scheme, repeat):
    """List of test-id arrays (one per fold). Main: per-modality StratifiedGroupKFold, fold k = union."""
    t = samples(scheme)
    if scheme == 'asplit':
        a = pd.read_csv(C_ASPLIT)
        return [list(a[a.part == 'test'].id)]
    gc = group_col(scheme)
    rs = C.FOLD_SEED_BASE + repeat
    if scheme == 'conservative':
        sg = StratifiedGroupKFold(C.N_FOLDS, shuffle=True, random_state=rs)
        return [list(t.index[te]) for _, te in sg.split(t, t.y * 2 + t.modality, t[gc])]
    out = [[] for _ in range(C.N_FOLDS)]
    for m in (0, 1):
        tm = t[t.modality == m]
        sg = StratifiedGroupKFold(C.N_FOLDS, shuffle=True, random_state=rs)
        for k, (_, te) in enumerate(sg.split(tm, tm.y, tm[gc])):
            out[k] += list(tm.index[te])
    return out


C_ASPLIT = Path(__file__).resolve().parent / 'frozen/a_split_ids.csv'


def inner_split(train_tab, scheme, repeat, fold):
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=C.INNER_SEED_BASE + 10 * repeat + fold)
    fit, val = next(sg.split(train_tab, train_tab.y * 2 + train_tab.modality, train_tab[group_col(scheme)]))
    return list(train_tab.index[fit]), list(train_tab.index[val])


def de_features(counts, y, fc):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    sig = set()
    for a, b in ((0, 1), (1, 2), (0, 2)):
        ids = y.index[y.isin([a, b])]
        cnt = counts.loc[ids].fillna(0)
        cnt = cnt.loc[:, cnt.sum(0) > 0].round().astype(int)
        meta = pd.DataFrame({'condition': np.where(y[ids] == a, 'g1', 'g2')}, index=ids)
        dds = DeseqDataSet(counts=cnt, metadata=meta, design_factors='condition', ref_level=['condition', 'g1'], quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=['condition', 'g2', 'g1'], quiet=True)
        st.summary()
        r = st.results_df
        sig |= set(r.index[(r.padj < C.DE_PADJ) & (r.log2FoldChange.abs() > fc)])
    return [c for c in counts.columns if c in sig]


def zscore(df, train_ids):
    lg = np.log2(df.clip(lower=0) + 1)
    mu, sd = lg.loc[train_ids].mean(), lg.loc[train_ids].std().replace(0, 1.0)
    return (lg - mu) / sd


def prep(scheme, repeat, fold):
    f = CACHE / f'{scheme}_r{repeat}_f{fold}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    d = geo_data.load()
    t = samples(scheme)
    test = outer_folds(scheme, repeat)[fold]
    train = [i for i in t.index if i not in set(test)]
    if scheme == 'asplit':
        a = pd.read_csv(C_ASPLIT)
        train = list(a[a.part == 'train'].id)
    ttr = t.loc[train]
    fit, val = inner_split(ttr, scheme, repeat, fold)
    Z, de, ctrl = {}, {}, {}
    for m in (0, 1):
        ids_m = list(t.index[t.modality == m])
        tr_m = [i for i in train if t.at[i, 'modality'] == m]
        cnt = d['counts'][m].loc[ids_m]
        de[m] = de_features(cnt.loc[tr_m], t.loc[tr_m, 'y'], C.MIRNA_FC if m == 0 else C.MRNA_FC)
        Z[m] = zscore(cnt, tr_m).astype(np.float32)
        ctrl[m] = zscore(d['controls'][m].loc[ids_m], tr_m).astype(np.float32)
    tmi, tmr = E.tcga_mappable()
    nodes = {'full': (de[0], de[1]),
             'subset': ([g for g in de[0] if g in tmi], [g for g in de[1] if g in tmr])}
    out = dict(scheme=scheme, repeat=repeat, fold=fold, train=train, test=test, fit=fit, val=val,
               samples=t, Z=Z, ctrl=ctrl, de=de, nodes=nodes)
    pickle.dump(out, open(f, 'wb'))
    return out


def coexpr(Zm, ids, nodes, offset, top_n=C.TOP_N_COEXPR):
    X = Zm.loc[ids, nodes].values.astype(np.float64)
    r = np.nan_to_num(np.corrcoef(X.T))
    np.fill_diagonal(r, 0)
    iu = np.triu_indices_from(r, k=1)
    v = np.abs(r[iu])
    keep = np.argsort(-v, kind='stable')[:top_n]
    return [(int(iu[0][k] + offset), int(iu[1][k] + offset), float(v[k]), 0.0) for k in keep]


_MIRDB = None


def mirdb():
    global _MIRDB
    if _MIRDB is None:
        _MIRDB = mappings.mirdb_pairs(C.MIRDB_MIN_SCORE)
    return _MIRDB


def cross_pairs(fd, node_set, cross, rng=None):
    mi, mr = fd['nodes'][node_set]
    md = list(E.pairs_in(mirdb(), mi, mr)[['probe', 'gene']].itertuples(index=False, name=None))
    if cross == 'none':
        return []
    if cross == 'mirdb':
        return md
    if cross == 'rewired':
        return E.rewire_bipartite(md, rng)
    kind = {'tcga': 'tumor', 'tcga_normal': 'normal', 'tcga_estimate': 'estimate'}[cross]
    return E.select_tcga(mi, mr, len(md), kind)


def build_graph(fd, node_set, cross, rng=None, labels=None, Z=None):
    """Returns nodes, node_types, {tissue: [(a, b, w, is_cross)]}, cross pair list.
    labels / Z override the fold's labels / values (positive control only)."""
    mi, mr = fd['nodes'][node_set]
    nodes = list(mi) + list(mr)
    idx = {g: i for i, g in enumerate(mi)} | {g: len(mi) + i for i, g in enumerate(mr)}
    t = fd['samples']
    y = t.y if labels is None else labels
    Z = Z or fd['Z']
    pairs = cross_pairs(fd, node_set, cross, rng)
    ce = [(idx[a], idx[b], 1.0, 1.0) for a, b in pairs]
    nets = {}
    tr = set(fd['train'])
    for c in range(3):
        mi_ids = [i for i in t.index[(t.modality == 0) & (y == c)] if i in tr]
        mr_ids = [i for i in t.index[(t.modality == 1) & (y == c)] if i in tr]
        nets[c] = coexpr(Z[0], mi_ids, list(mi), 0) + coexpr(Z[1], mr_ids, list(mr), len(mi)) + ce
    return nodes, np.array([0] * len(mi) + [1] * len(mr)), nets, pairs


def examples(fd, ids, nodes, n_mi, Z=None):
    """X (values, unobserved = 0) and M (observed mask) for single-modality examples."""
    Z = Z or fd['Z']
    t = fd['samples']
    mi_nodes, mr_nodes = nodes[:n_mi], nodes[n_mi:]
    N = len(nodes)
    X = np.zeros((len(ids), N), np.float32)
    M = np.zeros((len(ids), N), np.float32)
    mods = t.loc[ids, 'modality'].values
    for m, cols, sl in ((0, mi_nodes, slice(0, n_mi)), (1, mr_nodes, slice(n_mi, N))):
        sel = np.where(mods == m)[0]
        if len(sel) and len(cols):
            X[sel, sl] = Z[m].loc[[ids[i] for i in sel], cols].values
            M[sel, sl] = 1
    return X, M

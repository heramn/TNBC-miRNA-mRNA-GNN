#!/usr/bin/env python3
"""Does correlation structure learned on TCGA true pairs replicate in GEO GSE45498?
Usage: 03_edge_replication.py <tcga_cohort_dir> <out_dir>

For every pair of nodes (GEO DE features mappable to TCGA), compare the Pearson r in TCGA
(true same-specimen pairs) with the Pearson r in each GEO tissue (corrected build, all 230
samples). Within-modality pairs test network replication across platforms; miRNA-mRNA pairs
additionally test whether GEO's tissue-level pairing carries patient-level cross-modal signal
(random within-tissue pairing would give r_GEO ~ 0 and no agreement with TCGA). Also reports the
overlap of top-150 edge sets against the hypergeometric expectation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_common import REPO, _corr, build_network, map_geo_mirna_to_tcga, mimat_to_name

COH, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
P = REPO / 'data/processed'
meta = pd.read_csv(COH / 'tcga_tnbc_meta.csv', index_col=0)
tmir = pd.read_csv(COH / 'tcga_tnbc_mirna.csv.gz', index_col=0).rename(
    columns=mimat_to_name(str(Path.home() / '.capy/work/mirbase/mature.fa')))
tmir = tmir.loc[:, ~tmir.columns.duplicated()]
tmr = pd.read_csv(COH / 'tcga_tnbc_mrna.csv.gz', index_col=0)
gmir = pd.concat([pd.read_csv(P / 'train_mirna.csv', index_col=0), pd.read_csv(P / 'test_mirna.csv', index_col=0)])
gmr = pd.concat([pd.read_csv(P / 'train_mrna.csv', index_col=0), pd.read_csv(P / 'test_mrna.csv', index_col=0)])
labels = pd.concat([pd.read_csv(P / 'train_labels.csv', index_col=0), pd.read_csv(P / 'test_labels.csv', index_col=0)]).tissue_group
mmap = map_geo_mirna_to_tcga(list(gmir.columns), tmir)
mmap = {g: t for g, t in mmap.items() if list(mmap.values()).count(t) == 1}
mi = [g for g in gmir.columns if g in mmap]
mr = [g for g in gmr.columns if g in tmr.columns]
T = pd.concat([tmir[[mmap[g] for g in mi]].set_axis(mi, axis=1), tmr[mr]], axis=1)
G = pd.concat([gmir[mi], gmr[mr]], axis=1)

def pair_r(df):
    a, b = df[mi].values, df[mr].values
    out = {}
    for kind, x, z in (('miRNA-miRNA', a, a), ('mRNA-mRNA', b, b), ('miRNA-mRNA', a, b)):
        r = _corr(x, z)
        out[kind] = r[np.triu_indices(r.shape[0], 1)] if kind != 'miRNA-mRNA' else r.ravel()
    return out

def top_set(df, kind):
    e, _ = build_network(df[mi], df[mr])
    e = e[e.edge_type == kind]
    return {tuple(sorted((s, t))) if kind != 'miRNA-mRNA' else (s, t) for s, t in zip(e.source, e.target)}

rows = []
tcga_sets = {'tumor': meta.index[meta.tissue == 'tumor'], 'normal': meta.index[meta.tissue == 'normal']}
pairs = [('tumor', 'primary'), ('tumor', 'metastatic'), ('normal', 'normal'), ('tumor', 'normal'), ('normal', 'primary')]
tr = {k: pair_r(T.loc[v]) for k, v in tcga_sets.items()}
gr = {c: pair_r(G.loc[labels.index[labels == c]]) for c in ['normal', 'primary', 'metastatic']}
for tk, gk in pairs:
    for kind in ('miRNA-miRNA', 'mRNA-mRNA', 'miRNA-mRNA'):
        a, b = tr[tk][kind], gr[gk][kind]
        ok = np.isfinite(a) & np.isfinite(b)
        rho, p = stats.spearmanr(a[ok], b[ok])
        ts, gs = top_set(T.loc[tcga_sets[tk]], kind), top_set(G.loc[labels.index[labels == gk]], kind)
        n_pairs = int(ok.sum())
        exp = len(ts) * len(gs) / n_pairs if n_pairs else np.nan
        ov = len(ts & gs)
        hp = stats.hypergeom.sf(ov - 1, n_pairs, len(ts), len(gs)) if len(ts) and len(gs) else np.nan
        rows.append({'tcga': tk, 'geo': gk, 'edge_type': kind, 'n_pairs': n_pairs,
                     'spearman_r_vs_r': rho, 'p': p, 'top_tcga': len(ts), 'top_geo': len(gs),
                     'top_overlap': ov, 'overlap_expected': exp, 'hypergeom_p': hp,
                     'n_tcga': len(tcga_sets[tk]), 'n_geo': int((labels == gk).sum())})
res = pd.DataFrame(rows)
# Null for the cross-modal rows: shuffle which miRNA profile goes with which mRNA profile within
# the GEO tissue (what random within-tissue pairing would look like). Within-modality pairs are
# untouched by pairing, so they get no such null.
rng = np.random.default_rng(0)
N_NULL = 200
null_p, null_mean, null_sd = [], [], []
for _, row in res.iterrows():
    if row.edge_type != 'miRNA-mRNA':
        null_p.append(np.nan); null_mean.append(np.nan); null_sd.append(np.nan)
        continue
    g = G.loc[labels.index[labels == row.geo]]
    a = tr[row.tcga]['miRNA-mRNA']
    nulls = []
    for _ in range(N_NULL):
        gm = g[mi].values[rng.permutation(len(g))]
        b = _corr(gm, g[mr].values).ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        nulls.append(stats.spearmanr(a[ok], b[ok])[0])
    nulls = np.array(nulls)
    null_mean.append(nulls.mean()); null_sd.append(nulls.std())
    null_p.append((1 + (nulls >= row.spearman_r_vs_r).sum()) / (1 + N_NULL))
res['pairing_null_mean'], res['pairing_null_sd'], res['pairing_null_p_one_sided'] = null_mean, null_sd, null_p
res['pairing_null_z'] = (res.spearman_r_vs_r - res.pairing_null_mean) / res.pairing_null_sd
res.to_csv(OUT / 'edge_replication_tcga_vs_geo.csv', index=False)
print(res.round(4).to_string())

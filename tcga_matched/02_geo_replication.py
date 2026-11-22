#!/usr/bin/env python3
"""GEO GSE45498 (corrected build) as the replication cohort for TCGA true-pair networks.
Usage: 02_geo_replication.py <tcga_cohort_dir> <out_dir> [n_seeds] [n_rewire]
Needs data/processed from notebooks/01_preprocessing_DE.py (corrected 190/40 split).

A. Same GEO task, split and nodes as the paper (3-class normal/primary/metastatic, GEO DE
   features that also exist in TCGA). Only the graph changes:
   1 geo_all        : GEO-derived networks (03 recipe). Cross-modal edges rest on GEO's
                      tissue-level (not patient-verified) pairing.
   2 hybrid         : GEO within-modality edges + cross-modal edges learned on TCGA TRUE pairs
                      (normal <- TCGA normal; primary, metastatic <- TCGA TNBC tumor).
   3 tcga_all       : all edges from TCGA true-pair networks (normal, TNBC tumor).
   4 no_edges       : edges removed.
   5 hybrid_rewired : degree-preserving rewiring of the hybrid networks (null).
   + XGBoost on the same nodes.
B. Cross-cohort transfer: train on TCGA (TNBC tumor vs normal, all samples), test on all GEO
   samples (normal vs primary+metastatic); each cohort z-scored on its own data.
C. miRDB support of TCGA true-pair cross-modal edges (all GEO-panel miRNA x mRNA pairs, 102
   TNBC tumors): FDR-significant negative pairs vs the rest, Fisher exact test.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_common import (REPO, _corr, _pvals, build_network, cluster_bootstrap_ci, edges_to_tensor,
                         map_geo_mirna_to_tcga, metrics, mimat_to_name, rewire, train_gat)

torch.set_num_threads(1)
COH, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N_REWIRE = int(sys.argv[4]) if len(sys.argv) > 4 else 30
OUT.mkdir(parents=True, exist_ok=True)
P = REPO / 'data/processed'
CLASSES = ['normal', 'primary', 'metastatic']

meta = pd.read_csv(COH / 'tcga_tnbc_meta.csv', index_col=0)
tmir = pd.read_csv(COH / 'tcga_tnbc_mirna.csv.gz', index_col=0).rename(
    columns=mimat_to_name(str(Path.home() / '.capy/work/mirbase/mature.fa')))
tmir = tmir.loc[:, ~tmir.columns.duplicated()]
tmr = pd.read_csv(COH / 'tcga_tnbc_mrna.csv.gz', index_col=0)

gmir = pd.concat([pd.read_csv(P / 'train_mirna.csv', index_col=0), pd.read_csv(P / 'test_mirna.csv', index_col=0)])
gmr = pd.concat([pd.read_csv(P / 'train_mrna.csv', index_col=0), pd.read_csv(P / 'test_mrna.csv', index_col=0)])
labels = pd.concat([pd.read_csv(P / 'train_labels.csv', index_col=0), pd.read_csv(P / 'test_labels.csv', index_col=0)]).tissue_group
pids = pd.read_csv(P / 'patient_ids.csv', index_col=0).patient_id
split = json.load(open(P / 'split_indices.json'))
tr_ids, te_ids = split['train_sample_ids'], split['test_sample_ids']

mmap = map_geo_mirna_to_tcga(list(gmir.columns), tmir)
mmap = {g: t for g, t in mmap.items() if list(mmap.values()).count(t) == 1}
mir_nodes_geo = [g for g in gmir.columns if g in mmap]
mr_nodes = [g for g in gmr.columns if g in tmr.columns]
mir_nodes_tcga = [mmap[g] for g in mir_nodes_geo]
nodes = mir_nodes_geo + mr_nodes
node_types = torch.tensor([0] * len(mir_nodes_geo) + [1] * len(mr_nodes))
to_geo = {t: g for g, t in mmap.items()}

G = pd.concat([gmir[mir_nodes_geo], gmr[mr_nodes]], axis=1)
mu, sd = G.loc[tr_ids].mean(), G.loc[tr_ids].std().replace(0, 1)
GZ = (G - mu) / sd
y = labels.map({c: i for i, c in enumerate(CLASSES)})

T = pd.concat([tmir[mir_nodes_tcga].rename(columns=to_geo), tmr[mr_nodes]], axis=1)
tn_ids = meta.index[meta.tissue == 'normal']
tt_ids = meta.index[meta.tissue == 'tumor']

def net(df):
    e, s = build_network(df[mir_nodes_geo], df[mr_nodes])
    return e, s

geo_nets, stats_out = {}, {}
for c in CLASSES:
    rows = [s for s in tr_ids if labels[s] == c]
    geo_nets[c], stats_out[f'geo_{c}'] = net(G.loc[rows])
tcga_nets = {}
for name, rows in (('tcga_normal', tn_ids), ('tcga_tumor', tt_ids)):
    tcga_nets[name], stats_out[name] = net(T.loc[rows])
    tcga_nets[name].to_csv(OUT / f'{name}_edges_on_geo_nodes.csv', index=False)
cross_src = {'normal': 'tcga_normal', 'primary': 'tcga_tumor', 'metastatic': 'tcga_tumor'}
hybrid = {c: pd.concat([geo_nets[c][geo_nets[c].edge_type != 'miRNA-mRNA'],
                        tcga_nets[cross_src[c]][tcga_nets[cross_src[c]].edge_type == 'miRNA-mRNA']], ignore_index=True)
          for c in CLASSES}

def tens(d):
    return {k: edges_to_tensor(v, nodes) for k, v in d.items()}
variants = {'GAT geo_all (GEO-derived edges)': tens(geo_nets),
            'GAT hybrid (GEO within + TCGA true-pair cross-modal)': tens(hybrid),
            'GAT tcga_all (TCGA true-pair networks)': tens(tcga_nets),
            'GAT no_edges (edges removed)': {c: (torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))) for c in CLASSES}}

ytr = y[tr_ids].values
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fit_j, va_j = next(sgk.split(tr_ids, ytr, pids[tr_ids].values))
fit_ids, va_ids = [tr_ids[i] for i in fit_j], [tr_ids[i] for i in va_j]
Xf, yf = GZ.loc[fit_ids].values.astype(np.float32), y[fit_ids].values
Xv, yv = GZ.loc[va_ids].values.astype(np.float32), y[va_ids].values
Xt, yt = GZ.loc[te_ids].values.astype(np.float32), y[te_ids].values
gt = pids[te_ids].values

rows, probs = [], {}
for name, nets in variants.items():
    ps = []
    for seed in range(N_SEEDS):
        p = train_gat(Xf, yf, Xv, yv, Xt, nets, node_types, 3, seed)
        ps.append(p)
        rows.append({'model': name, 'seed': seed} | metrics(yt, p))
    probs[name] = np.mean(ps, 0)
    print(name, np.mean([r['macro_f1'] for r in rows if r['model'] == name]), flush=True)
xs = []
for seed in range(N_SEEDS):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      random_state=seed, eval_metric='mlogloss').fit(GZ.loc[tr_ids].values, ytr)
    p = m.predict_proba(Xt)
    xs.append(p)
    rows.append({'model': 'XGBoost (same nodes)', 'seed': seed} | metrics(yt, p))
probs['XGBoost (same nodes)'] = np.mean(xs, 0)
rng = np.random.default_rng(11)
rew = []
for i in range(N_REWIRE):
    p = train_gat(Xf, yf, Xv, yv, Xt, tens({c: rewire(e, rng) for c, e in hybrid.items()}), node_types, 3, i % N_SEEDS)
    rew.append(metrics(yt, p) | {'rewire': i})
rew = pd.DataFrame(rew)
rew.to_csv(OUT / 'geo_hybrid_rewired_null.csv', index=False)
per_seed = pd.DataFrame(rows)
per_seed.to_csv(OUT / 'geo_per_seed_metrics.csv', index=False)
summ = per_seed.groupby('model').agg(macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std'),
                                     auroc_mean=('auroc', 'mean'), acc_mean=('accuracy', 'mean'))
for m, p in probs.items():
    summ.loc[m, 'ensemble_macro_f1'] = metrics(yt, p)['macro_f1']
    summ.loc[m, 'f1_ci_lo'], summ.loc[m, 'f1_ci_hi'] = cluster_bootstrap_ci(yt, p, gt)
hyb_mean = per_seed[per_seed.model.str.startswith('GAT hybrid')].macro_f1.mean()
perm = {'hybrid_seed_mean_f1': hyb_mean, 'rewired_f1_mean': rew.macro_f1.mean(), 'rewired_f1_sd': rew.macro_f1.std(),
        'p_hybrid_vs_rewired': float((1 + (rew.macro_f1 >= hyb_mean).sum()) / (1 + len(rew))), 'n_rewire': len(rew)}
summ.to_csv(OUT / 'geo_replication_summary.csv')

# B. cross-cohort transfer TCGA -> GEO (binary normal vs tumor)
Tz_ids = list(tn_ids) + list(tt_ids)
TT = T.loc[Tz_ids]
TZ = ((TT - TT.mean()) / TT.std().replace(0, 1)).values.astype(np.float32)
ty = np.array([0] * len(tn_ids) + [1] * len(tt_ids))
GA = G.loc[labels.index]
GAZ = ((GA - GA.mean()) / GA.std().replace(0, 1)).values.astype(np.float32)
gy = (labels != 'normal').astype(int).values
tr_rows = []
for seed in range(N_SEEDS):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      random_state=seed, eval_metric='logloss').fit(TZ, ty)
    tr_rows.append({'model': 'XGBoost', 'seed': seed} | metrics(gy, m.predict_proba(GAZ)))
lr = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(TZ, ty)
tr_rows.append({'model': 'Logistic regression (L2)', 'seed': 0} | metrics(gy, lr.predict_proba(GAZ)))
tpat = meta.loc[Tz_ids, 'patient'].values
sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
a, b = next(sg.split(TZ, ty, tpat))
tnets = {k: edges_to_tensor(v, nodes) for k, v in tcga_nets.items()}
for seed in range(N_SEEDS):
    p = train_gat(TZ[a], ty[a], TZ[b], ty[b], GAZ, tnets, node_types, 2, seed)
    tr_rows.append({'model': 'GAT (TCGA true-pair networks)', 'seed': seed} | metrics(gy, p))
tr = pd.DataFrame(tr_rows)
tr.to_csv(OUT / 'transfer_tcga_to_geo_per_seed.csv', index=False)
tr_summ = tr.groupby('model')[['macro_f1', 'auroc', 'accuracy']].agg(['mean', 'std'])
tr_summ.to_csv(OUT / 'transfer_tcga_to_geo_summary.csv')

# C. miRDB support of true-pair cross-modal edges (whole GEO panel, TCGA TNBC tumors)
with gzip.open(REPO / 'data/raw/GSE45498_mRNA_non-normalized_data.txt.gz', 'rt') as fh:
    ref2sym = {}
    for l in fh:
        if l.startswith('Endogenous'):
            f = l.rstrip('\n').split('\t')
            ref2sym[f[2].split('.')[0]] = f[1]
md = pd.read_csv(REPO / 'data/raw/miRDB_v6.0_prediction_result.txt.gz', sep='\t', header=None, names=['mirna', 'ref', 'score'])
md = md[md.mirna.str.startswith('hsa-')]
md['gene'] = md.ref.str.split('.').str[0].map(ref2sym)
md = md.dropna(subset=['gene'])
targets = set(zip(md[md.score >= 80].mirna, md[md.score >= 80].gene))
all_mir = sorted(set(map_geo_mirna_to_tcga(
    [l.split('\t')[1] for l in gzip.open(REPO / 'data/raw/GSE45498_raw_data.txt.gz', 'rt') if l.startswith('Endogenous')], tmir).values()))
all_mr = sorted(set(ref2sym.values()) & set(tmr.columns))
mirdb_rows = []
for cohort, rows_ in (('TCGA TNBC tumors (true pairs)', tt_ids), ('TCGA normals (true pairs)', tn_ids)):
    A, B = tmir.loc[rows_, all_mir], tmr.loc[rows_, all_mr]
    A = A.loc[:, A.std() > 0]
    r = _corr(A.values, B.values)
    q = multipletests(_pvals(r.ravel(), len(rows_)), method='fdr_bh')[1].reshape(r.shape)
    tgt = np.array([[(m, g) in targets for g in B.columns] for m in A.columns])
    sig = (q < 0.05) & (r < 0)
    tab = [[int((sig & tgt).sum()), int((sig & ~tgt).sum())], [int((~sig & tgt).sum()), int((~sig & ~tgt).sum())]]
    orr, pv = fisher_exact(tab, alternative='greater')
    sigpos = (q < 0.05) & (r > 0)
    tabp = [[int((sigpos & tgt).sum()), int((sigpos & ~tgt).sum())], [int((~sigpos & tgt).sum()), int((~sigpos & ~tgt).sum())]]
    orp, pvp = fisher_exact(tabp, alternative='greater')
    mirdb_rows.append({'cohort': cohort, 'n_samples': len(rows_), 'pairs_tested': int(r.size),
                       'miRDB_targets_ge80_in_panel': int(tgt.sum()),
                       'fdr_sig_negative': int(sig.sum()), 'fdr_sig_negative_miRDB': tab[0][0],
                       'frac_miRDB_sig_negative': tab[0][0] / max(1, sig.sum()),
                       'frac_miRDB_background': tab[1][0] / max(1, tab[1][0] + tab[1][1]),
                       'odds_ratio_negative': orr, 'fisher_p_negative': pv,
                       'fdr_sig_positive': int(sigpos.sum()), 'odds_ratio_positive_control': orp, 'fisher_p_positive_control': pvp})
mirdb = pd.DataFrame(mirdb_rows)
mirdb.to_csv(OUT / 'mirdb_support_true_pair_edges.csv', index=False)

info = {'geo_nodes': {'miRNA': len(mir_nodes_geo), 'mRNA': len(mr_nodes)},
        'geo_miRNA_dropped_unmappable': [g for g in gmir.columns if g not in mmap],
        'geo_mRNA_dropped_unmappable': [g for g in gmr.columns if g not in tmr.columns],
        'geo_n': {'train': len(tr_ids), 'fit': len(fit_ids), 'val': len(va_ids), 'test': len(te_ids)},
        'networks': stats_out, 'permutation': perm}
json.dump(info, open(OUT / 'geo_replication_info.json', 'w'), indent=2, default=float)
print(json.dumps(info, indent=1, default=float))
print(summ.round(3).to_string())
print(tr_summ.round(3).to_string())
print(mirdb.T.to_string())

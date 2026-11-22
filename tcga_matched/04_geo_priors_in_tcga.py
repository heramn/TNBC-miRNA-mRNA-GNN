#!/usr/bin/env python3
"""GEO (primary cohort) -> TCGA (true same-patient pairs) validation of the paper's own networks.
Usage: 04_geo_priors_in_tcga.py <tcga_cohort_dir> <out_dir> <part> [n_seeds] [n_rewire]
part = edges | gat_real | gat_noedge | gat_rewired | xgb | summarize

edges : do the committed GEO networks (results/networks/*_edges_top150.csv, corrected build)
        hold on TCGA true pairs? Each GEO edge gets its Pearson r in the matching TCGA tissue
        (normal -> TCGA normals; primary and metastatic -> TCGA TNBC primary tumors, the only
        TCGA TNBC tissue with n > 1). Compared against random node pairs of the same type
        (10,000 size-matched random edge sets).
gat_* / xgb : the paper's GEO model trained on the GEO 190-sample training split with the
        committed GEO networks (3-class), applied unchanged to every TCGA sample. External
        score = 1 - P(normal); TCGA truth = tumor vs normal. Each cohort is z-scored on its own
        samples (label-free). Early stopping uses a patient-grouped 20% of the GEO training split.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_common import (REPO, _corr, _pvals, edges_to_tensor, map_geo_mirna_to_tcga, mimat_to_name,
                         rewire, train_gat)

torch.set_num_threads(1)
COH, OUT, PART = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
N_SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
N_REWIRE = int(sys.argv[5]) if len(sys.argv) > 5 else 20
OUT.mkdir(parents=True, exist_ok=True)
P = REPO / 'data/processed'
CLASSES = ['normal', 'primary', 'metastatic']
TCGA_TISSUE = {'normal': 'normal', 'primary': 'tumor', 'metastatic': 'tumor'}

meta = pd.read_csv(COH / 'tcga_tnbc_meta.csv', index_col=0)
meta = meta[meta.tissue.isin(['tumor', 'normal'])]
tmir = pd.read_csv(COH / 'tcga_tnbc_mirna.csv.gz', index_col=0).rename(
    columns=mimat_to_name(str(Path.home() / '.capy/work/mirbase/mature.fa')))
tmir = tmir.loc[:, ~tmir.columns.duplicated()]
tmr = pd.read_csv(COH / 'tcga_tnbc_mrna.csv.gz', index_col=0)
gmir = pd.concat([pd.read_csv(P / 'train_mirna.csv', index_col=0), pd.read_csv(P / 'test_mirna.csv', index_col=0)])
gmr = pd.concat([pd.read_csv(P / 'train_mrna.csv', index_col=0), pd.read_csv(P / 'test_mrna.csv', index_col=0)])
labels = pd.concat([pd.read_csv(P / 'train_labels.csv', index_col=0), pd.read_csv(P / 'test_labels.csv', index_col=0)]).tissue_group
pids = pd.read_csv(P / 'patient_ids.csv', index_col=0).patient_id
tr_ids = json.load(open(P / 'split_indices.json'))['train_sample_ids']

mmap = map_geo_mirna_to_tcga(list(gmir.columns), tmir)
mmap = {g: t for g, t in mmap.items() if list(mmap.values()).count(t) == 1}
mi = [g for g in gmir.columns if g in mmap]
mr = [g for g in gmr.columns if g in tmr.columns]
nodes = mi + mr
node_types = torch.tensor([0] * len(mi) + [1] * len(mr))
T = pd.concat([tmir[[mmap[g] for g in mi]].set_axis(mi, axis=1), tmr[mr]], axis=1).loc[meta.index]
G = pd.concat([gmir[mi], gmr[mr]], axis=1)
geo_nets = {c: pd.read_csv(REPO / f'results/networks/{c}_edges_top150.csv').rename(columns={'pearson_r': 'r'})
            for c in CLASSES}
for c in CLASSES:
    e = geo_nets[c]
    geo_nets[c] = e[e.source.isin(nodes) & e.target.isin(nodes)][['source', 'target', 'edge_type', 'r']].reset_index(drop=True)

if PART == 'edges':
    rng = np.random.default_rng(0)
    rows = []
    for c in CLASSES:
        sub = T[meta.tissue == TCGA_TISSUE[c]]
        n = len(sub)
        idx = {v: i for i, v in enumerate(nodes)}
        R = _corr(sub[nodes].values, sub[nodes].values)
        for kind in ('miRNA-miRNA', 'mRNA-mRNA', 'miRNA-mRNA'):
            e = geo_nets[c][geo_nets[c].edge_type == kind]
            a_nodes, b_nodes = (mi, mi) if kind == 'miRNA-miRNA' else (mr, mr) if kind == 'mRNA-mRNA' else (mi, mr)
            ai, bi = np.array([idx[x] for x in a_nodes]), np.array([idx[x] for x in b_nodes])
            sub_r = R[np.ix_(ai, bi)]
            if kind != 'miRNA-mRNA':
                iu = np.triu_indices(len(ai), 1)
                bg = sub_r[iu]
            else:
                bg = sub_r.ravel()
            bg = bg[np.isfinite(bg)]
            q_bg = multipletests(_pvals(bg, n), method='fdr_bh')[1]
            sign = -1 if kind == 'miRNA-mRNA' else 1
            thr = np.max(np.abs(bg[q_bg < 0.05])) if False else None
            er = np.array([R[idx[s], idx[t]] for s, t in zip(e.source, e.target)])
            p_e = _pvals(er, n)
            # FDR-significance of each GEO edge within the full family of TCGA pairs of that type
            allp = np.concatenate([_pvals(bg, n), p_e])
            q_all = multipletests(allp, method='fdr_bh')[1][len(bg):]
            same = np.sign(er) == np.sign(e.r.values)
            sig_same = same & (q_all < 0.05)
            bg_sig_dir = float(((np.sign(bg) == sign) & (q_bg < 0.05)).mean())
            null_med = np.array([np.median(sign * rng.choice(bg, len(e), replace=False)) for _ in range(10000)])
            obs_med = float(np.median(sign * er))
            rows.append({'geo_network': c, 'tcga_tissue': TCGA_TISSUE[c], 'tcga_n': n, 'edge_type': kind,
                         'geo_edges_mappable': len(e), 'geo_edges_fdr_sig_in_geo': None,
                         'tcga_median_r_signed': obs_med, 'random_pairs_median_r_signed': float(np.median(sign * bg)),
                         'frac_same_sign_in_tcga': float(same.mean()),
                         'frac_fdr_sig_same_sign_in_tcga': float(sig_same.mean()),
                         'random_pairs_frac_fdr_sig_expected_sign': bg_sig_dir,
                         'empirical_p_vs_random_pairs': float((1 + (null_med >= obs_med).sum()) / 10001)})
    res = pd.DataFrame(rows).drop(columns=['geo_edges_fdr_sig_in_geo'])
    res.to_csv(OUT / 'geo_edges_on_tcga_true_pairs.csv', index=False)
    print(res.round(4).to_string())
    sys.exit(0)

ytr = labels[tr_ids].map({c: i for i, c in enumerate(CLASSES)}).values
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fj, vj = next(sgk.split(tr_ids, ytr, pids[tr_ids].values))
Gtr = G.loc[tr_ids]
GZ = ((Gtr - Gtr.mean()) / Gtr.std().replace(0, 1)).values.astype(np.float32)
TZ = ((T - T.mean()) / T.std().replace(0, 1)).values.astype(np.float32)
ty = (meta.tissue == 'tumor').astype(int).values

def score(prob):
    s = 1 - prob[:, 0]
    return {'auroc': roc_auc_score(ty, s), 'macro_f1': f1_score(ty, (s > 0.5).astype(int), average='macro'),
            'accuracy': float(((s > 0.5).astype(int) == ty).mean())}

def nets(d):
    return {k: edges_to_tensor(v, nodes) for k, v in d.items()}

rows, probs = [], []
if PART == 'gat_real':
    for seed in range(N_SEEDS):
        p = train_gat(GZ[fj], ytr[fj], GZ[vj], ytr[vj], TZ, nets(geo_nets), node_types, 3, seed)
        rows.append({'model': 'GAT (GEO networks)', 'seed': seed} | score(p)); probs.append(p)
elif PART == 'gat_noedge':
    empty = {c: (torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))) for c in CLASSES}
    for seed in range(N_SEEDS):
        p = train_gat(GZ[fj], ytr[fj], GZ[vj], ytr[vj], TZ, empty, node_types, 3, seed)
        rows.append({'model': 'GAT (edges removed)', 'seed': seed} | score(p)); probs.append(p)
elif PART == 'gat_rewired':
    rng = np.random.default_rng(5)
    for i in range(N_REWIRE):
        p = train_gat(GZ[fj], ytr[fj], GZ[vj], ytr[vj], TZ, nets({c: rewire(e, rng) for c, e in geo_nets.items()}),
                      node_types, 3, i % N_SEEDS)
        rows.append({'model': 'GAT (GEO networks rewired)', 'seed': i} | score(p)); probs.append(p)
elif PART == 'xgb':
    for seed in range(N_SEEDS):
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          random_state=seed, eval_metric='mlogloss').fit(GZ, ytr)
        p = m.predict_proba(TZ)
        rows.append({'model': 'XGBoost', 'seed': seed} | score(p)); probs.append(p)
elif PART == 'summarize':
    from tcga_common import cluster_bootstrap_ci
    per = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('geo2tcga_*_per_seed.csv'))])
    summ = per.groupby('model').agg(auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'),
                                    macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std'),
                                    acc_mean=('accuracy', 'mean'), n=('seed', 'size'))
    groups = meta.patient.values
    for f in sorted(OUT.glob('geo2tcga_*_probs.npy')):
        p = np.load(f).mean(0)
        model = per[per['part'] == f.name.split('geo2tcga_')[1].replace('_probs.npy', '')].model.iloc[0]
        two = np.stack([p[:, 0], 1 - p[:, 0]], 1)
        summ.loc[model, 'ensemble_auroc'] = roc_auc_score(ty, two[:, 1])
        summ.loc[model, 'auroc_ci_lo'], summ.loc[model, 'auroc_ci_hi'] = cluster_bootstrap_ci(ty, two, groups, key='auroc')
        summ.loc[model, 'f1_ci_lo'], summ.loc[model, 'f1_ci_hi'] = cluster_bootstrap_ci(ty, two, groups)
    real = per[per.model == 'GAT (GEO networks)']
    rew = per[per.model == 'GAT (GEO networks rewired)']
    perm = {m: float((1 + (rew[m] >= real[m].mean()).sum()) / (1 + len(rew))) for m in ('auroc', 'macro_f1')}
    summ.to_csv(OUT / 'geo2tcga_summary.csv')
    json.dump({'p_real_vs_rewired': perm, 'tcga_n': {'tumor': int(ty.sum()), 'normal': int((1 - ty).sum())},
               'nodes': {'miRNA': len(mi), 'mRNA': len(mr)},
               'geo_edges_mappable': {c: int(len(e)) for c, e in geo_nets.items()}},
              open(OUT / 'geo2tcga_info.json', 'w'), indent=2)
    print(summ.round(3).to_string()); print(perm)
    sys.exit(0)
else:
    sys.exit(f'unknown part {PART}')
pd.DataFrame(rows).assign(part=PART).to_csv(OUT / f'geo2tcga_{PART}_per_seed.csv', index=False)
np.save(OUT / f'geo2tcga_{PART}_probs.npy', np.stack(probs))
print(pd.DataFrame(rows).describe().round(3).to_string())
